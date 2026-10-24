| Variant | Hit@4 | MRR | Faithfulness (1-5) | Refusal correct | Wrong refusals | Errors | Avg latency (s) | Avg LLM calls |
|---|---|---|---|---|---|---|---|---|
| A: Naive (vector top 4) | 95% (n=21) | 0.81 | 4.81 (n=21) | 100% (n=9) | 0/21 | 0/30 | 8.60 | 1.00 |
| B: Hybrid + rerank | 100% (n=21) | 0.93 | 5.00 (n=19) | 100% (n=9) | 2/21 | 0/30 | 11.99 | 1.00 |
| C: Agentic | 100% (n=21) | 0.95 | 5.00 (n=21) | 100% (n=9) | 0/21 | 0/30 | 16.31 | 1.23 |
