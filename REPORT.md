# Lexis: Search Engine over the Reuters-21578 News Archive

CSL 436 Information Retrieval, Winter 2026, Programming Assignment

All numbers below come from `python run_experiments.py all` (raw output in `results/`). The corpus is the 10,788 articles in `data/corpus/`.

---

## 1. Design

### 1.1 Architecture

```
data/corpus/doc_XXXXX.txt
      │  streamed by 4 parallel workers (one shard = a contiguous range of files)
      ▼
  tokenizer + normalizer ──► SPIMI (or BSBI) blocks, flushed at the memory limit, sorted by term
      ▼
  one external k-way merge (heap over all blocks of all shards)
      ▼
  index_data/<index>/postings.bin    positional postings: VB(docID gap), VB(tf), VB(position gaps)
                     dictionary.bin  front-coded terms + VB(df) + VB(postings length)
                     docs.json       per document: title length, token count, zone vector norms
                     aux.json        auxiliary index + deleted documents (dynamic indexing)
```

Lexis builds two indexes with the same code. Only the normalizer settings differ.

| Index | Normalization | Used for |
|---|---|---|
| `index_data/exact` | case folding only | Boolean, phrase, proximity, wildcard, spelling correction |
| `index_data/ranked` | case folding, stop words removed, Porter stemming | ranked free-text retrieval |

**Why two indexes.** Exact queries must return the same results as a brute-force scan of the lowercased text, with stemming switched off. Stemming would merge terms such as *rates* and *rate*, and stop word removal would break phrases such as "bank of japan". Ranking, on the other hand, gains from stemming (see the ablations). Building both indexes takes about 9 s and 7 MB in total.

### 1.2 Tokenizer and normalizer (`src/parsing/preprocessor.py`)

1. **Case folding:** the whole text is lowercased.
2. **Tokenization:** a token is a maximal run of `[a-z0-9]`, and every other character is a separator. Before choosing this rule we checked candidate tokenizers against `boolean_tests.tsv` with a brute-force scan. This rule reproduces all 25 expected result sets. Alternatives such as letters only, or keeping apostrophes, do not.
3. **Optional stop word removal:** uses NLTK's English list. Removed words keep their positions as gaps, so the positions of the remaining words do not shift.
4. **Optional Porter stemming.**

The same object processes documents at index time and queries at query time, with the settings stored in the index (`docs.json → config`). Positions run continuously through the title and then the body. A position below `title_len` belongs to the title zone, so zone tf needs no separate index.

### 1.3 Index construction (`build_index.py`, `src/indexing/spimi.py`)

- **Streaming:** each worker reads its files one document at a time.
- **Memory limit:** we use a dummy limit on the number of postings (token positions) held in memory. When the limit is reached, the block is sorted by term and written to disk as a text file with one line per term.
- **SPIMI and BSBI** write the same block format.
  - SPIMI adds postings straight into a term → postings dictionary.
  - BSBI keeps a global term → termID map, collects (termID, docID, position) triples, and sorts each block by termID.
- **Parallelism:** a `multiprocessing.Pool` runs 4 shard workers. Every block from every shard then goes through one heap-based k-way merge into a single index. The merge holds only one term's postings in memory at a time.
- **Dynamic index (IIR 4.5):** `IndexReader.add_document()` puts new documents into an auxiliary index, and `delete_document()` adds the doc ID to an invalidation set. Both are saved in `aux.json`. Every lookup combines the main and auxiliary postings and filters out deleted documents. `merge_aux()` folds the auxiliary index into the main index files without re-reading the corpus.

### 1.4 Compression (`src/indexing/compress.py`, `src/indexing/index_format.py`)

- **Postings:** doc IDs are stored as gaps and VB-encoded. Each posting also stores tf and the gap-encoded positions, which are needed for phrase and proximity queries. Elias gamma is also implemented, and is compared with VB in §4.
- **Dictionary:** terms are stored sorted and front-coded (shared-prefix length, suffix length, suffix). df and the postings byte length are VB-encoded, and postings offsets are cumulative sums.
- **Why VB on disk:** it is byte-aligned, so it decodes quickly. Gamma gives a better ratio overall, mainly on very frequent terms, but needs bit-level decoding.

### 1.5 Exact queries (`src/querying/boolean_parser.py`, `positional.py`, `wildcard.py`, `spellcheck.py`)

- **Query parser:** a shunting-yard parser handles AND, OR, NOT (unary), parentheses, `"phrases"`, proximity `a /k b` and wildcards. Two adjacent operands without an operator are ANDed.
- **AND:** a postings intersection with skip pointers. The default spacing is √L.
- **Phrases:** consecutive positions in the positional postings.
- **Proximity `/k`:** the two terms are at most k positions apart, in either order. This matches the expected results.
- **Wildcards:** a permuterm index. A query `X*Y` is rotated to the prefix `Y$X`. For several `*`, the outer parts `X*…*Z` give the prefix `Z$X` (for example `pet*chem*` → `$pet`), and the candidates are post-filtered with a regular expression (IIR 3.2.1).
- **Spelling:** a term that is not in the vocabulary gets suggestions within edit distance 2, using Levenshtein distance plus adjacent transpositions (Damerau). Suggestions are ranked by distance, then by document frequency.

**Correctness checks** (`python run_tests.py`):

| Check | Result |
|---|---|
| `boolean_tests.tsv`: 10 Boolean, 7 phrase, 4 wildcard and 4 proximity queries | 25 / 25 identical to the expected sets |
| `misspellings.tsv`: top suggestion equals the intended word | 59 / 60 (`prokkects` → "prospects"; "projects" is ranked 2nd at the same distance) |
| Add document, delete document, reload, merge | OK |

### 1.6 Ranked retrieval (`src/querying/vector_space.py`)

- **Weighting:** lnc.ltc. The document weight is 1 + log₁₀ tf, cosine-normalized. The query weight is (1 + log₁₀ tf) · log₁₀(N/df).
- **Weighted zone scoring:** score(q, d) = g · cos(q, title_d) + (1 − g) · cos(q, body_d). g is tuned on the training queries only (§5).
- **Top 10 without scoring every document:**
  - Scoring is term-at-a-time over the postings of the query terms only, so documents that share no term with the query are never touched.
  - The top k is taken with a heap (`heapq.nlargest`) instead of a full sort.
- **Pruning (IIR 7.1), §6:**
  - *Index elimination:* query terms with idf < 1 are dropped, and only documents containing at least half of the remaining terms are scored.
  - *Champion lists:* the r = 50 highest-weighted documents per term are precomputed, and only their union is scored.

---

## 2. Vocabulary (`results/laws.md`, `results/zipfs_law.png`, `results/heaps_law.png`)

These figures use the exact index (case folding only): T = 1,468,154 tokens and M = 30,952 terms.

| Law | Lexis | Reuters-RCV1 (textbook) |
|---|---|---|
| Heaps k | 38.94 | 44 |
| Heaps b | 0.471 | 0.49 |
| Zipf exponent (least squares, log-log, all ranks) | 1.58 | 1 |

- **Heaps' law:** the fit, M = 38.9 · T^0.471, is close to the RCV1 values. Both collections are English newswire, so vocabulary grows at a similar rate. The slightly lower k and b suggest a more repetitive vocabulary, as expected for a smaller, financial-news collection.
- **Zipf's law:** the frequent terms follow cf ∝ 1/rank closely: fitted on the top 1,000 ranks only, the exponent is 0.95. The fitted exponent over all ranks is steeper than 1 because of the long flat tail of terms with cf = 1 or 2. 35% of the terms occur only once and 49% at most twice. These plateaus fill most of the rank axis and pull the least-squares line down.

---

## 3. Exact retrieval

### 3.1 Skip pointers (`results/exact_retrieval.md`)

The table counts doc ID comparisons for AND queries on the exact index.

| Query | \|p1\| | \|p2\| | Results | No skips | skip=4 | skip=16 | skip=64 | skip=√L |
|---|---|---|---|---|---|---|---|---|
| oil AND opec | 1038 | 126 | 120 | 1932 | 1015 (47% saved) | 1727 (11%) | 1945 (−1%) | 1958 (−1%) |
| wheat AND corn | 327 | 209 | 90 | 796 | 832 (−5%) | 820 (−3%) | 802 (−1%) | 817 (−3%) |
| gold AND silver | 229 | 62 | 49 | 435 | 390 (10%) | 449 (−3%) | 438 (−1%) | 453 (−4%) |
| bank AND yen | 1338 | 318 | 156 | 2842 | 1965 (31%) | 2641 (7%) | 2864 (−1%) | 2885 (−2%) |
| trade AND deficit | 1117 | 378 | 250 | 2236 | 2054 (8%) | 2300 (−3%) | 2253 (−1%) | 2268 (−1%) |
| interest AND rates | 839 | 563 | 289 | 1937 | 2020 (−4%) | 1992 (−3%) | 1952 (−1%) | 1966 (−1%) |
| coffee AND quota | 159 | 139 | 41 | 467 | 499 (−7%) | 479 (−3%) | 470 (−1%) | 487 (−4%) |
| said AND tanker | 6784 | 35 | 33 | 13263 | 2001 (85%) | 1430 (89%) | 3041 (77%) | 3708 (72%) |
| dollar AND yen | 590 | 318 | 194 | 1232 | 1247 (−1%) | 1264 (−3%) | 1241 (−1%) | 1259 (−2%) |
| the AND merger | 6951 | 375 | 338 | 13604 | 4864 (64%) | 7966 (41%) | 13333 (2%) | 13520 (1%) |

Skip pointers pay off when one list is much longer than the other: *said/tanker* saves up to 89% and *the/merger* up to 64%. Long skips are then usable on the long list. When the two lists have similar lengths and many matches (*wheat/corn*, *interest/rates*), skips are rarely taken, and the extra skip checks cost a few percent.

On spacing, short skips (4, 16) are tried more often but each jump is small. Long skips (64, √L) jump far but rarely land below the target. This is the trade-off described in IIR 2.3.

### 3.2 Positional vs plain index

| Index (exact, gap + VB) | Postings size |
|---|---|
| Plain: doc ID gaps only | 0.98 MB |
| Positional: doc ID gaps + tf + position gaps | 3.55 MB |

The positional index is 3.6× the size of the plain one. That is the cost of supporting phrase and proximity queries: every one of the 1.47 M tokens contributes one position gap.

---

## 4. Index construction (`results/construction.md`)

These runs use the exact index and a single shard, so that memory is measured cleanly. Peak memory is the peak Python heap measured with `tracemalloc`.

| Method | Memory limit (postings) | Blocks | MB written | Peak memory (MB) | Time (s) |
|---|---|---|---|---|---|
| SPIMI | 25,000 | 59 | 16.53 | 5.7 | 5.46 |
| SPIMI | 50,000 | 30 | 16.07 | 8.2 | 4.91 |
| SPIMI | 100,000 | 15 | 15.69 | 12.6 | 4.84 |
| SPIMI | 200,000 | 8 | 15.40 | 21.2 | 4.81 |
| SPIMI | 400,000 | 4 | 15.18 | 37.5 | 4.79 |
| BSBI | 25,000 | 59 | 16.53 | 11.2 | 5.40 |
| BSBI | 50,000 | 30 | 16.07 | 15.7 | 5.58 |
| BSBI | 100,000 | 15 | 15.69 | 24.0 | 5.34 |
| BSBI | 200,000 | 8 | 15.40 | 40.5 | 5.46 |
| BSBI | 400,000 | 4 | 15.18 | 72.3 | 5.30 |

- **Memory limit:** halving the limit doubles the number of blocks. Bytes written grow slightly, because terms are repeated across blocks. Peak memory follows the limit almost linearly.
- **SPIMI vs BSBI:**
  - At the same limit, BSBI needs about 2× the memory of SPIMI. It stores a (termID, docID, position) tuple for every posting and keeps a global term → termID table.
  - BSBI also has to sort every block, while SPIMI only sorts the term list. This makes BSBI slightly slower in most runs.

| Shards | Slowest worker (s) | Inversion wall time (s) | Total incl. merge (s) |
|---|---|---|---|
| 1 | 2.27 | 2.28 | 4.73 |
| 4 | 0.70 | 0.74 | 3.17 |

With 4 shards, parsing and inversion run **3.1× faster**, and the whole build is **1.5× faster**. The final k-way merge (about 2.4 s) runs in one process and limits the overall speedup, as Amdahl's law predicts.

---

## 5. Compression (`results/compression.md`)

The table covers the doc ID postings of the exact index.

| Terms | # terms | Uncompressed (32-bit doc IDs) | Gaps (fixed width per list) | Variable byte | Gamma | Ratio VB | Ratio gamma |
|---|---|---|---|---|---|---|---|
| All terms | 30,952 | 3,289,332 B | 896,883 B | 982,728 B | 901,746 B | 3.35× | 3.65× |
| 100 most frequent | 100 | 1,034,908 B | 186,897 B | 258,737 B | 108,335 B | 4.00× | 9.55× |
| Rarest 10,000 | 10,000 | 40,000 B | 29,722 B | 19,863 B | 35,741 B | 2.01× | 1.12× |

"Gaps (fixed width)" stores each list's gaps in the number of bits needed for its largest gap.

- **Frequent terms** have small gaps, mostly 1 or 2. Gamma codes these in 1–3 bits, while VB needs a full byte, so gamma reaches 9.6× against VB's 4.0×.
- **Rare terms** (df = 1) have a single "gap" equal to the doc ID itself, around 5,000 on average. Gamma codes it in about 25 bits and VB in 2 bytes, so VB wins on the rare terms (2.0× vs 1.1×).
- **Overall** gamma is about 8% smaller than VB. We still use VB on disk because it decodes byte by byte.

| Dictionary | Bytes |
|---|---|
| Fixed-width array (20-byte term + 4-byte df + 4-byte pointer, IIR 5.2.1) | 866,656 |
| Front coding + VB df + VB postings length (`dictionary.bin`) | 206,790 (4.2× smaller) |

The whole exact index (postings, dictionary and document table) is 4.1 MB. The earlier prototype stored positions and tf as uncompressed JSON and took about 41 MB (for the 21,578-article SGML collection).

---

## 6. Ranking (`results/ranking.md`, `results/pr_curve_test.png`)

We used three systems:
- **Boolean:** an AND of the query words after stop word removal, on the stemmed index, returned unranked (doc ID order).
- **Raw-tf baseline:** score = Σ tf(t, d) over the query terms, with no idf, no normalization and no zones.
- **Lexis best:** lnc.ltc with zone weighting.

The title weight g was tuned on the 40 training queries. MAP on the training queries for each g:

| g | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MAP | 0.5645 | **0.5750** | 0.5636 | 0.5449 | 0.5208 | 0.4993 | 0.4802 | 0.4651 | 0.4490 | 0.4319 | 0.3948 |

**Final results.** The test set was evaluated once, after tuning:

| Queries | System | P@10 | MAP | NDCG@10 (graded) |
|---|---|---|---|---|
| train | Boolean (AND) | 0.3750 | 0.0795 | 0.4187 |
| train | Raw-tf baseline | 0.5225 | 0.3438 | 0.4991 |
| train | Lexis best (g = 0.1) | 0.7425 | 0.5750 | 0.6616 |
| **test** | Boolean (AND) | 0.3225 | 0.0875 | 0.3620 |
| **test** | Raw-tf baseline | 0.3125 | 0.2392 | 0.3102 |
| **test** | **Lexis best (g = 0.1)** | **0.6500** | **0.5337** | **0.5834** |

11-point interpolated precision on the test queries, averaged over queries:

| System | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Boolean | 0.624 | 0.403 | 0.156 | 0.087 | 0.010 | 0 | 0 | 0 | 0 | 0 | 0 |
| Raw-tf | 0.570 | 0.435 | 0.345 | 0.302 | 0.263 | 0.244 | 0.210 | 0.180 | 0.154 | 0.119 | 0.054 |
| Lexis best | 0.871 | 0.784 | 0.746 | 0.704 | 0.630 | 0.593 | 0.528 | 0.428 | 0.353 | 0.230 | 0.086 |

- **Boolean AND** is precise at the top but returns few documents, and returns them unordered, so its recall and MAP are very low.
- **Raw tf** favours long documents and common words.
- **lnc.ltc** more than doubles MAP over raw tf on the test set (0.534 vs 0.239).

---

## 7. Efficiency (`results/efficiency.md`)

These figures use the 40 test queries and the top 10 results. Postings are read from disk for every query.

| Method | Overlap@10 vs exhaustive | Docs scored / query | Latency (ms) | P@10 | NDCG@10 |
|---|---|---|---|---|---|
| Exhaustive (all docs in the query terms' postings) | 1.000 | 2031.8 | 4.08 | 0.650 | 0.583 |
| Index elimination (idf ≥ 1, ≥ half the terms) | 0.775 | 226.9 | 1.30 | 0.655 | 0.584 |
| Champion lists (r = 50) | 0.965 | 178.1 | 1.80 | 0.640 | 0.569 |

- **Both techniques** score about 10× fewer documents and are 2–3× faster.
- **Champion lists** keep 96.5% of the exact top 10. They need a precomputation step that takes 1.5 s once.
- **Index elimination** changes about a quarter of the top 10, yet keeps the same P@10 and NDCG@10. Documents that contain most of the rare query terms are usually the relevant ones.

---

## 8. Ablations (`results/ablations.md`)

Each feature was turned off one at a time from the best system and evaluated on the test queries. p-values come from a two-sided paired t-test on per-query AP (n = 40).

| Variant | MAP | Change | p-value |
|---|---|---|---|
| Full system | 0.5337 | | |
| No stemming | 0.5051 | −0.0286 | 0.038 |
| No stop word removal | 0.5298 | −0.0039 | 0.153 |
| No idf | 0.4288 | −0.1049 | < 0.001 |
| No length normalization | 0.5354 | +0.0017 | 0.955 |
| No zone weighting | 0.5415 | +0.0078 | 0.094 |

- **idf** is by far the most important component. Removing it costs 0.105 MAP, which is highly significant.
- **Stemming** helps significantly (p < 0.05): it matches *subsidies/subsidy*, *exports/export*, and so on.
- **Stop word removal, length normalization and zone weighting** change MAP by less than 0.01, and none of these changes is significant.
- **Zone weighting** with g = 0.1 helped on the training queries (0.5750 vs 0.5645 for body only) but not on the test queries. Its effect is within noise for this collection, where titles are short and mostly repeat words from the body.

---

## 9. How to run

```
pip install -r requirements.txt
python build_index.py                 # builds index_data/exact and index_data/ranked (about 10 s)
python run_tests.py                   # boolean_tests.tsv, misspellings.tsv, dynamic indexing
python search.py "oil AND opec"       # or: python search.py   (interactive)
python run_experiments.py all         # every table and plot in this report -> results/
```
