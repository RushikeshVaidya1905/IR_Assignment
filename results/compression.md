Doc ID postings of the exact index

| Terms | # | Uncompressed (32-bit docIDs, bytes) | Gaps (fixed width, bytes) | Variable byte | Gamma | Ratio VB | Ratio gamma |
|---|---|---|---|---|---|---|---|
| All terms | 30952 | 3289332 | 896883 | 982728 | 901746 | 3.35x | 3.65x |
| 100 most frequent | 100 | 1034908 | 186897 | 258737 | 108335 | 4.00x | 9.55x |
| Rarest 10,000 | 10000 | 40000 | 29726 | 19861 | 35730 | 2.01x | 1.12x |

| Dictionary | Bytes |
|---|---|
| Fixed-width array (28 B/term) | 866656 |
| Raw strings (sum of term lengths) | 209806 |
| Front coding + VB (dictionary.bin) | 206790 |
