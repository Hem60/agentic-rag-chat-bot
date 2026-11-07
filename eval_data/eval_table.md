| Variant | Hit@4 | MRR | Faithfulness (1-5) | Refusal correct | Wrong refusals | Errors | Avg latency (s) | Avg LLM calls |
|---|---|---|---|---|---|---|---|---|
| A: Naive (vector top 4) | 100% (n=33) | 0.95 | 4.63 (n=30) | 100% (n=9) | 3/33 | 0/42 | 7.44 | 1.00 |
| B: Hybrid + rerank | 100% (n=33) | 0.91 | 4.87 (n=31) | 100% (n=9) | 2/33 | 0/42 | 13.62 | 1.00 |
| C: Agentic | 100% (n=33) | 0.91 | 4.97 (n=32) | 100% (n=9) | 1/33 | 0/42 | 11.33 | 0.98 |
