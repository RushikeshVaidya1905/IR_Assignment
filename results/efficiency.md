| Method | Overlap@10 vs exhaustive | Docs scored / query | Latency (ms) | P@10 | NDCG@10 |
|---|---|---|---|---|---|
| exhaustive | 1.0000 | 2031.8000 | 6.2300 | 0.6500 | 0.5834 |
| index_elimination | 0.7750 | 226.9000 | 1.6900 | 0.6550 | 0.5837 |
| champion | 0.9650 | 178.1000 | 2.4800 | 0.6400 | 0.5692 |

Index elimination: idf threshold 1.0, docs must contain >= half of the remaining query terms. Champion lists: r = 50 (built once in 2.3 s). Corpus size N = 10788.
