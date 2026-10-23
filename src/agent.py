# -*- coding: utf-8 -*-
"""
src/agent.py
------------
LangGraph agent around the hybrid retriever.

    route -> (greeting) -> smalltalk -> END
          -> rewrite -> retrieve -> grade -> generate -> END
                  ^                    |
                  +---- weak, retry ---+--> refuse -> END (attempts exhausted)

* route:    rule-based; greetings/thanks get a canned reply (no LLM call).
* rewrite:  skipped on the first try when there is no history. With history it
            makes a standalone query; on a retry it words the query differently.
* retrieve: HybridRetriever.search (BM25 + vector + RRF + cross-encoder).
* grade:    compares the best reranker score to RERANK_THRESHOLD. No LLM call.
* generate: grounded answer from the top chunks, with sources and pages.
* refuse:   the fixed refusal sentence once MAX_ATTEMPTS searches stay weak.

LLM budget per question: 1 call for a clear question, 2 for a follow-up, at most
about 4 in the worst retry case.
"""

import logging
import re
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src import config, llm

log = logging.getLogger("rag.agent")

REFUSAL = config.REFUSAL_SENTENCE

_GREETING_RE = re.compile(
    r"^\s*(hi|hii+|hello|hey|yo|good\s+(morning|afternoon|evening)|thanks|thank\s+you|"
    r"thx|ok(ay)?|bye|goodbye|see\s+you)\b[\s!.,?]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(r"\b(thanks|thank\s+you|thx)\b", re.IGNORECASE)
_BYE_RE = re.compile(r"\b(bye|goodbye|see\s+you)\b", re.IGNORECASE)

_STOPWORDS = frozenset(
    "a an the is are was were be been of in on at to for and or what which who how why when "
    "does do did can could would should explain describe tell me about give define".split()
)

REWRITE_PROMPT = """Rewrite the user's latest message as one standalone search query for a \
study-notes search engine. Resolve pronouns using the conversation. Output ONLY the query.

Conversation:
{history}

Latest message: {question}"""

RETRY_PROMPT = """The search query below did not find good results in the study notes. Write ONE \
different query for the same question, using synonyms and only the key technical terms. \
Output ONLY the query.

Question: {question}
Queries already tried: {tried}"""

ANSWER_PROMPT = """You answer questions strictly from the numbered excerpts below.

Rules:
- Use ONLY the excerpts. Never invent facts, numbers or definitions.
- Cite the source file and page, e.g. (file.pdf, p. 3).
- If the excerpts do not contain the answer, reply with exactly:
  "{refusal}"
- Be concise.

Excerpts:
{context}

Question: {question}"""


class AgentState(TypedDict, total=False):
    question: str
    history: list          # [{"role": "user"|"assistant", "content": str}]
    search_query: str
    tried_queries: list
    chunks: list
    best_score: float
    attempts: int
    llm_calls: int
    kind: str
    answer: str
    sources: list
    trace: list


def is_smalltalk(text: str) -> bool:
    return bool(_GREETING_RE.match(text))


def smalltalk_reply(text: str) -> str:
    if _THANKS_RE.search(text):
        return "You're welcome! Ask me anything about your study notes."
    if _BYE_RE.search(text):
        return "Goodbye! Good luck with your studying."
    return "Hi! Ask me a question about your study notes and I'll answer from them."


def keyword_query(text: str) -> str:
    """LLM-free fallback query: the question minus stopwords."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text) if w.lower() not in _STOPWORDS]
    return " ".join(words) or text


def clean_query(raw: str, fallback: str) -> str:
    q = (raw or "").strip().splitlines()[0].strip().strip('"\'`') if (raw or "").strip() else ""
    q = re.sub(r"^(search query|query)\s*:\s*", "", q, flags=re.IGNORECASE)
    return q or fallback


def _format_history(history: list, turns: int) -> str:
    recent = history[-turns:] if turns else []
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def format_context(chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Excerpt {i} -- {c['source']}, p. {c['page']}]\n{c['text'].strip()}"
        for i, c in enumerate(chunks, 1)
    )


def clean_answer(raw: str) -> str:
    answer = (raw or "").strip()
    if answer[:7].lower() == "answer:":
        answer = answer[7:].strip()
    return answer


def unique_sources(chunks: list) -> list:
    seen, out = set(), []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            out.append({"source": c["source"], "page": c["page"], "score": round(c["score"], 4)})
    return out


def build_agent(
    retriever,
    chat: Callable = llm.chat,
    threshold: Optional[float] = None,
    max_attempts: Optional[int] = None,
    history_turns: Optional[int] = None,
):
    """Compile the graph. `retriever.search(query)` and `chat(messages)` are injectable."""
    threshold = config.RERANK_THRESHOLD if threshold is None else threshold
    max_attempts = max_attempts or config.MAX_ATTEMPTS
    history_turns = config.HISTORY_TURNS if history_turns is None else history_turns

    def ask(prompt: str, state: AgentState, **kw) -> str:
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        return chat([{"role": "user", "content": prompt}], **kw)

    def node_route(state: AgentState) -> dict:
        smalltalk = is_smalltalk(state["question"])
        route = "smalltalk" if smalltalk else "search"
        return {
            "kind": route,
            "attempts": 0,
            "llm_calls": 0,
            "tried_queries": [],
            "trace": [{"step": "route", "detail": route}],
        }

    def node_smalltalk(state: AgentState) -> dict:
        return {"answer": smalltalk_reply(state["question"]), "sources": [], "chunks": [],
                "best_score": 0.0,
                "trace": state["trace"] + [{"step": "smalltalk", "detail": "direct reply, no search"}]}

    def node_rewrite(state: AgentState) -> dict:
        trace = list(state["trace"])
        question, history = state["question"], state.get("history") or []
        tried = list(state.get("tried_queries", []))
        scratch = {"llm_calls": state.get("llm_calls", 0)}

        if state["attempts"] == 0 and not history:
            query = question
            trace.append({"step": "rewrite", "detail": "skipped (first try, no history)"})
        else:
            if state["attempts"] == 0:
                prompt = REWRITE_PROMPT.format(
                    history=_format_history(history, history_turns), question=question)
                fallback = question
            else:
                prompt = RETRY_PROMPT.format(question=question, tried="; ".join(tried))
                fallback = keyword_query(question)
            try:
                query = clean_query(ask(prompt, scratch, max_tokens=64, temperature=0.3), fallback)
            except llm.LLMError as e:
                log.warning("rewrite failed (%s); using fallback query", e)
                query = fallback
                trace.append({"step": "rewrite", "detail": "LLM unavailable, used fallback"})
            if state["attempts"] > 0 and query in tried:
                query = keyword_query(question) if keyword_query(question) not in tried else question
            trace.append({"step": "rewrite", "detail": query})
        return {"search_query": query, "tried_queries": tried + [query],
                "llm_calls": scratch["llm_calls"], "trace": trace}

    def node_retrieve(state: AgentState) -> dict:
        chunks = retriever.search(state["search_query"])
        best = max((c["score"] for c in chunks), default=0.0)
        attempts = state["attempts"] + 1
        detail = f"query={state['search_query']!r} best_score={best:.3f} attempt={attempts}"
        return {"chunks": chunks, "best_score": best, "attempts": attempts,
                "trace": state["trace"] + [{"step": "retrieve", "detail": detail}]}

    def decide_after_grade(state: AgentState) -> str:
        if state["best_score"] >= threshold:
            return "generate"
        return "rewrite" if state["attempts"] < max_attempts else "refuse"

    def node_grade(state: AgentState) -> dict:
        decision = decide_after_grade(state)
        detail = f"best_score={state['best_score']:.3f} threshold={threshold} -> {decision}"
        return {"trace": state["trace"] + [{"step": "grade", "detail": detail}]}

    def node_generate(state: AgentState) -> dict:
        scratch = {"llm_calls": state.get("llm_calls", 0)}
        prompt = ANSWER_PROMPT.format(
            refusal=REFUSAL, context=format_context(state["chunks"]), question=state["question"])
        answer = clean_answer(ask(prompt, scratch))
        refused = REFUSAL.lower() in answer.lower()
        return {"answer": answer, "llm_calls": scratch["llm_calls"],
                "sources": [] if refused else unique_sources(state["chunks"]),
                "trace": state["trace"] + [
                    {"step": "generate", "detail": "model refused" if refused else "answered"}]}

    def node_refuse(state: AgentState) -> dict:
        return {"answer": REFUSAL, "sources": [],
                "trace": state["trace"] + [
                    {"step": "refuse", "detail": f"best_score stayed below {threshold} after "
                                                  f"{state['attempts']} search(es)"}]}

    g = StateGraph(AgentState)
    g.add_node("route", node_route)
    g.add_node("smalltalk", node_smalltalk)
    g.add_node("rewrite", node_rewrite)
    g.add_node("retrieve", node_retrieve)
    g.add_node("grade", node_grade)
    g.add_node("generate", node_generate)
    g.add_node("refuse", node_refuse)

    g.set_entry_point("route")
    g.add_conditional_edges("route", lambda s: s["kind"],
                            {"smalltalk": "smalltalk", "search": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide_after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("smalltalk", END)
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


def run_agent(agent, question: str, history: Optional[list] = None) -> AgentState:
    """Run one turn. Returns the final state (answer, sources, chunks, trace, llm_calls...)."""
    return agent.invoke({"question": question, "history": history or []})
