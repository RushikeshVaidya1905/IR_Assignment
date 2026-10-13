| Method | Memory limit (postings) | Blocks | MB written | Peak memory (MB) | Time (s) |
|---|---|---|---|---|---|
| SPIMI | 25000 | 59 | 16.7600 | 5.5000 | 7.6800 |
| SPIMI | 50000 | 30 | 16.2400 | 8.0000 | 6.6000 |
| SPIMI | 100000 | 15 | 15.8100 | 12.5000 | 6.2200 |
| SPIMI | 200000 | 8 | 15.4900 | 21.1000 | 6.2000 |
| SPIMI | 400000 | 4 | 15.2400 | 37.4000 | 6.0700 |
| BSBI | 25000 | 59 | 16.7600 | 10.7000 | 8.5700 |
| BSBI | 50000 | 30 | 16.2400 | 15.0000 | 7.5300 |
| BSBI | 100000 | 15 | 15.8100 | 22.9000 | 7.2700 |
| BSBI | 200000 | 8 | 15.4900 | 38.6000 | 7.0400 |
| BSBI | 400000 | 4 | 15.2400 | 68.8000 | 6.9600 |

| Shards | Slowest worker (s) | Inversion wall time incl. process start (s) | Total incl. merge (s) |
|---|---|---|---|
| 1 | 3.5400 | 3.5800 | 6.9000 |
| 4 | 1.2400 | 2.8700 | 6.1500 |

Speedup with 4 shards: 2.85x (parallel inversion work), 1.25x (inversion wall time), 1.12x (total; the k-way merge is sequential)
