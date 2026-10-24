"""
app.py
------
Gradio chat page. Mounted inside FastAPI by src/api.py so one process serves the
chat page and the API on one port.

Run:
    python app.py        (starts uvicorn on 127.0.0.1:8000)
"""

import gradio as gr

EXAMPLES = [
    "What is a stack and what are its main operations?",
    "How does quicksort work and what is its time complexity?",
    "What is the difference between BFS and DFS?",
    "How does a hash table handle collisions?",
    "What is an AVL tree and why does it rotate?",
    "How do you convert an infix expression to postfix?",
]

INTRO = (
    "# Data Structures & Algorithms Assistant\n"
    "Ask any **Data Structures or Algorithms** question: arrays, stacks, queues, linked lists, "
    "trees, graphs, hashing, sorting, searching, recursion and complexity. "
    "Answers come only from the knowledge base (lecture notes plus Wikipedia articles), "
    "with sources shown. Questions on other topics are politely declined."
)


def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = []
    for s in sources:
        name = s["source"].replace("\\", "/").split("/")[-1]   # show the file name, not the path
        page = f" p. {s['page']}" if s.get("page") is not None else ""
        lines.append(f"- `{name}`{page}")
    return "\n\n**Sources**\n" + "\n".join(lines)


def format_trace(result: dict) -> str:
    rows = [f"{i}. **{t['step']}**: {t['detail']}" for i, t in enumerate(result["trace"], 1)]
    footer = (f"\n\nbest reranker score `{result['best_score']}` | LLM calls `{result['llm_calls']}` "
              f"| `{result['latency_s']}s`")
    return "\n".join(rows) + footer


def respond(message, history):
    """Generator: shows a progress line, then the answer. Second output is the trace panel."""
    from src import llm
    from src.api import app_state, run_chat

    history = history or []
    turn = history + [{"role": "user", "content": message}]
    yield turn + [{"role": "assistant", "content": "_Searching the knowledge base..._"}], ""

    if app_state["agent"] is None:
        reply, trace = "The knowledge base is still waking up. Please try again in a moment.", ""
    else:
        try:
            plain = [{"role": m["role"], "content": m["content"]} for m in history
                     if isinstance(m.get("content"), str)]
            result = run_chat(message, plain)
            reply, trace = result["answer"] + format_sources(result["sources"]), format_trace(result)
        except llm.LLMError:
            reply, trace = "The answer service is temporarily unavailable. Please try again later.", ""
        except Exception:
            reply, trace = "Something went wrong. Please try again.", ""
    yield turn + [{"role": "assistant", "content": reply}], trace


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Data Structures & Algorithms Assistant") as ui:
        gr.Markdown(INTRO)
        chatbot = gr.Chatbot(type="messages", height=460, label="Chat")
        box = gr.Textbox(placeholder="Ask a Data Structures question, e.g. what is a queue?",
                         show_label=False, max_lines=4)
        with gr.Accordion("Agent trace (rewritten query, scores, retries)", open=False):
            trace = gr.Markdown("_Ask something to see the agent's steps._")
        gr.Examples(EXAMPLES, inputs=box)
        clear = gr.Button("Clear chat")

        box.submit(respond, [box, chatbot], [chatbot, trace]).then(lambda: "", None, box)
        clear.click(lambda: ([], ""), None, [chatbot, trace])
    return ui


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000)
