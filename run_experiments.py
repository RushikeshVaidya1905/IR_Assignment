"""Produces every number and plot for the report into results/.

Usage:  python run_experiments.py [laws|exact|construction|compression|ranking|efficiency|ablations|all]
(Build the indexes first with:  python build_index.py)
"""
import os
import sys
import json
import math
import time
import shutil
from src.querying.index_reader import IndexReader
from src.querying.boolean_parser import BooleanSearchEngine, intersect_with_skips
from src.querying.vector_space import VectorSpaceModel
from src.indexing.compress import VariableByteEncoder as VB, GammaEncoder
from src.parsing.corpus_parser import doc_id_to_name
from build_index import build_index

OUT = "results"
EXACT, RANKED = "index_data/exact", "index_data/ranked"
os.makedirs(OUT, exist_ok=True)
# numpy / scipy / matplotlib are imported inside the functions that need them, so the 4 index-building
# worker processes (which re-import this script) start quickly.


def save(name, data, table_md=None):
    with open(os.path.join(OUT, name + ".json"), 'w') as f:
        json.dump(data, f, indent=2)
    if table_md:
        with open(os.path.join(OUT, name + ".md"), 'w') as f:
            f.write(table_md + "\n")
        print(table_md)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- 1. Vocabulary
def laws():
    from src.evaluation.laws import LawFitter
    fitter = LawFitter(EXACT, OUT)
    data = {**fitter.plot_zipfs_law(), **fitter.plot_heaps_law()}
    save("laws", data, md_table(["Law", "Lexis", "Reuters-RCV1 (textbook)"], [
        ["Heaps k", round(data["heaps_k"], 2), data["rcv1_k"]],
        ["Heaps b", round(data["heaps_b"], 3), data["rcv1_b"]],
        ["Zipf exponent", round(data["zipf_s"], 3), data["rcv1_s"]],
    ]))


# ---------------------------------------------------------------- 2. Exact retrieval
SKIP_PAIRS = [("oil", "opec"), ("wheat", "corn"), ("gold", "silver"), ("bank", "yen"), ("trade", "deficit"),
              ("interest", "rates"), ("coffee", "quota"), ("said", "tanker"), ("dollar", "yen"), ("the", "merger")]
SKIP_SPACINGS = [4, 16, 64, None]  # None = sqrt(list length)

def exact():
    reader = IndexReader(EXACT)
    rows, data = [], []
    for t1, t2 in SKIP_PAIRS:
        p1, p2 = reader.get_postings(t1), reader.get_postings(t2)
        result, base = intersect_with_skips(p1, p2, 0, 0)
        row = [f"{t1} AND {t2}", len(p1), len(p2), len(result), base]
        for s in SKIP_SPACINGS:
            res, comps = intersect_with_skips(p1, p2, s, s)
            assert res == result
            row.append(f"{comps} ({100 * (base - comps) / base:.0f}% saved)")
        rows.append(row)
        data.append(row)
    table = md_table(["Query", "|p1|", "|p2|", "Results", "No skips"] + [f"skip={s or 'sqrt(L)'}" for s in SKIP_SPACINGS], rows)

    # Positional index vs plain (doc IDs only) index, both gap + VB encoded
    positional = os.path.getsize(os.path.join(EXACT, "postings.bin"))
    plain = 0
    for term in reader.vocabulary():
        docs = reader.get_postings(term)
        plain += len(VB.encode_list(VB.gap_encode(docs)))
        reader._cache.clear()
    size_table = md_table(["Index", "Postings size (MB)"], [
        ["Plain (doc ID gaps, VB)", round(plain / 1e6, 3)],
        ["Positional (doc ID gaps + tf + position gaps, VB)", round(positional / 1e6, 3)],
    ])
    size_table += f"\n\nPositional / plain = {positional / plain:.2f}x"
    save("exact_retrieval", {"skip_pointers": data, "positional_bytes": positional, "plain_bytes": plain},
         table + "\n\n" + size_table)


# ---------------------------------------------------------------- 3. Index construction
MEMORY_LIMITS = [25_000, 50_000, 100_000, 200_000, 400_000]  # postings held in memory (dummy constraint)

def construction():
    tmp = "index_data/tmp_build"
    rows, data = [], []
    for method in ("spimi", "bsbi"):
        for limit in MEMORY_LIMITS:
            mem = build_index(tmp, False, False, num_shards=1, memory_limit=limit, method=method,
                              measure_memory=True, verbose=False)
            timed = build_index(tmp, False, False, num_shards=1, memory_limit=limit, method=method, verbose=False)
            row = [method.upper(), limit, timed["blocks"], round(timed["bytes_written"] / 1e6, 2),
                   round(mem["peak_memory"] / 1e6, 1), round(timed["total_time"], 2)]
            rows.append(row)
            data.append(dict(zip(["method", "memory_limit", "blocks", "mb_written", "peak_mb", "time_s"], row)))
            print(row)
    table = md_table(["Method", "Memory limit (postings)", "Blocks", "MB written", "Peak memory (MB)", "Time (s)"], rows)

    # best of 3 runs (the first parallel run also pays the one-time cost of starting worker processes)
    one = min((build_index(tmp, False, False, num_shards=1, verbose=False) for _ in range(3)), key=lambda s: s["total_time"])
    four = min((build_index(tmp, False, False, num_shards=4, verbose=False) for _ in range(3)), key=lambda s: s["total_time"])
    shutil.rmtree(tmp)
    speed = md_table(["Shards", "Slowest worker (s)", "Inversion wall time incl. process start (s)", "Total incl. merge (s)"], [
        [1, round(one["worker_time"], 2), round(one["inversion_time"], 2), round(one["total_time"], 2)],
        [4, round(four["worker_time"], 2), round(four["inversion_time"], 2), round(four["total_time"], 2)],
    ])
    speed += (f"\n\nSpeedup with 4 shards: {one['worker_time'] / four['worker_time']:.2f}x (parallel inversion work), "
              f"{one['inversion_time'] / four['inversion_time']:.2f}x (inversion wall time), "
              f"{one['total_time'] / four['total_time']:.2f}x (total; the k-way merge is sequential)")
    save("construction", {"runs": data, "one_shard": one, "four_shards": four}, table + "\n\n" + speed)


# ---------------------------------------------------------------- 4. Compression
def fixed_width_gap_bytes(gaps):
    """'Gaps' column: each list stores its gaps in a fixed number of bits = bits of its largest gap."""
    width = max(gaps).bit_length()
    return math.ceil(len(gaps) * width / 8) + 1  # + 1 byte to store the width

def compression():
    reader = IndexReader(EXACT)
    terms = sorted(reader.vocabulary(), key=lambda t: -reader.df(t))
    groups = {"All terms": terms, "100 most frequent": terms[:100], "Rarest 10,000": terms[-10000:]}
    sizes = {}
    for term in terms:
        docs = reader.get_postings(term)
        reader._cache.clear()
        gaps = VB.gap_encode(docs)
        sizes[term] = (4 * len(docs), fixed_width_gap_bytes(gaps), len(VB.encode_list(gaps)),
                       len(GammaEncoder.encode_list(gaps)))
    assert GammaEncoder.decode(GammaEncoder.encode_list([1, 5, 300]), 3) == [1, 5, 300]
    rows = []
    for name, group in groups.items():
        u, g, v, y = (sum(sizes[t][i] for t in group) for i in range(4))
        rows.append([name, len(group), u, g, v, y, f"{u / v:.2f}x", f"{u / y:.2f}x"])
    table = md_table(["Terms", "#", "Uncompressed (32-bit docIDs, bytes)", "Gaps (fixed width, bytes)",
                      "Variable byte", "Gamma", "Ratio VB", "Ratio gamma"], rows)

    # Dictionary: fixed-width array (20-byte term + 4-byte df + 4-byte pointer, IIR 5.2.1) vs front coding
    fixed = 28 * len(terms)
    front = os.path.getsize(os.path.join(EXACT, "dictionary.bin"))
    dict_table = md_table(["Dictionary", "Bytes"], [
        ["Fixed-width array (28 B/term)", fixed], ["Raw strings (sum of term lengths)", sum(map(len, terms))],
        ["Front coding + VB (dictionary.bin)", front]])
    save("compression", {"rows": rows, "dictionary_fixed": fixed, "dictionary_front_coded": front},
         "Doc ID postings of the exact index\n\n" + table + "\n\n" + dict_table)


# ---------------------------------------------------------------- 5. Ranking
QRELS = None
def qrels():
    from src.evaluation.metrics import load_qrels
    global QRELS
    if QRELS is None:
        QRELS = load_qrels("data/qrels.tsv")
    return QRELS

def evaluate_run(run, qrels):
    from src.evaluation.metrics import evaluate_run
    return evaluate_run(run, qrels)

def load_queries(path):
    from src.evaluation.metrics import load_queries
    return load_queries(path)

def ranked_run(vsm, queries, method="exhaustive", top_k=1000):
    return {q: [doc_id_to_name(d) for d, _ in vsm.search(text, top_k=top_k, method=method)]
            for q, text in queries.items()}

def boolean_run(queries):
    """Boolean system: AND of the (non stop word) query words, unranked (doc ID order)."""
    engine = BooleanSearchEngine(RANKED)
    pre = engine.preprocessor
    run = {}
    for q, text in queries.items():
        words = [w for w in pre.tokenize(text) if w not in pre.stop_words]
        run[q] = [doc_id_to_name(d) for d in engine.search(" AND ".join(words))]
    return run

def best_title_weight():
    path = os.path.join(OUT, "ranking.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["best_title_weight"]
    return 0.1

def ranking():
    from src.evaluation.metrics import Evaluator
    train, test = load_queries("data/queries_train.tsv"), load_queries("data/queries_test.tsv")
    # Tune the title zone weight g on the training queries only
    tuning = []
    for g in [i / 10 for i in range(11)]:
        res = evaluate_run(ranked_run(VectorSpaceModel(title_weight=g), train), qrels())
        tuning.append([g, res["MAP"], res["P@10"], res["NDCG@10"]])
        print(f"g = {g:.1f}: train MAP {res['MAP']:.4f}")
    best_g = max(tuning, key=lambda r: r[1])[0]
    tune_table = md_table(["Title weight g", "Train MAP", "Train P@10", "Train NDCG@10"], tuning)

    systems = {
        "Boolean (AND)": lambda qs: boolean_run(qs),
        "Raw-tf baseline": lambda qs: ranked_run(VectorSpaceModel(raw_tf=True, use_idf=False, use_length_norm=False, use_zones=False), qs),
        f"Lexis best (lnc.ltc, zones g={best_g})": lambda qs: ranked_run(VectorSpaceModel(title_weight=best_g), qs),
    }
    rows, curves, data = [], {}, {}
    for split, queries in (("train", train), ("test", test)):  # test is run once, after tuning
        for name, system in systems.items():
            res = evaluate_run(system(queries), qrels())
            rows.append([split, name, res["P@10"], res["MAP"], res["NDCG@10"]])
            data[f"{split}/{name}"] = {k: v for k, v in res.items() if k != "per_query"}
            if split == "test":
                curves[name] = res["11pt"]
    Evaluator.plot_11_point_pr_curve(None, curves=curves, output_file=os.path.join(OUT, "pr_curve_test.png"))
    table = md_table(["Queries", "System", "P@10", "MAP", "NDCG@10"], rows)
    pr = md_table(["System"] + [f"R={r / 10:.1f}" for r in range(11)], [[n] + [round(p, 3) for p in c] for n, c in curves.items()])
    save("ranking", {"best_title_weight": best_g, "tuning": tuning, "results": data},
         tune_table + f"\n\nBest g on train = {best_g}\n\n" + table + "\n\n11-point interpolated precision (test)\n\n" + pr)


# ---------------------------------------------------------------- 6. Efficiency
def efficiency():
    test = load_queries("data/queries_test.tsv")
    vsm = VectorSpaceModel(title_weight=best_title_weight())
    t0 = time.time()
    vsm.build_champion_lists()
    champion_build = time.time() - t0
    exhaustive = {}
    rows = []
    for method in ("exhaustive", "index_elimination", "champion"):
        overlap, scored, latency = [], [], []
        run = {}
        for q, text in test.items():
            vsm.reader._cache.clear()  # every query reads its postings from disk
            t0 = time.time()
            top = vsm.search(text, top_k=10, method=method)
            latency.append(time.time() - t0)
            scored.append(vsm.last_docs_scored)
            docs = [d for d, _ in top]
            if method == "exhaustive":
                exhaustive[q] = docs
            overlap.append(len(set(docs) & set(exhaustive[q])) / max(1, len(exhaustive[q])))
            run[q] = [doc_id_to_name(d) for d in docs]
        res = evaluate_run(run, qrels())
        rows.append([method, sum(overlap) / len(overlap), round(sum(scored) / len(scored), 1),
                     round(1000 * sum(latency) / len(latency), 2), res["P@10"], res["NDCG@10"]])
    table = md_table(["Method", "Overlap@10 vs exhaustive", "Docs scored / query", "Latency (ms)", "P@10", "NDCG@10"], rows)
    table += (f"\n\nIndex elimination: idf threshold {vsm.idf_threshold}, docs must contain >= half of the remaining "
              f"query terms. Champion lists: r = {vsm.champion_r} (built once in {champion_build:.1f} s). "
              f"Corpus size N = {vsm.N}.")
    save("efficiency", {"rows": rows}, table)


# ---------------------------------------------------------------- 7. Ablations
def ablations():
    from src.evaluation.ablation import AblationStudy
    test = load_queries("data/queries_test.tsv")
    g = best_title_weight()
    for path, stem, stop in (("index_data/ablation_nostem", False, True), ("index_data/ablation_nostop", True, False)):
        if not os.path.exists(path):
            build_index(path, stem, stop)
    variants = {
        "Full system": VectorSpaceModel(title_weight=g),
        "No stemming": VectorSpaceModel("index_data/ablation_nostem", title_weight=g),
        "No stop word removal": VectorSpaceModel("index_data/ablation_nostop", title_weight=g),
        "No idf": VectorSpaceModel(title_weight=g, use_idf=False),
        "No length normalization": VectorSpaceModel(title_weight=g, use_length_norm=False),
        "No zone weighting": VectorSpaceModel(use_zones=False),
    }
    base = evaluate_run(ranked_run(variants["Full system"], test), qrels())
    queries = sorted(base["per_query"]["ap"])
    rows = []
    for name, vsm in variants.items():
        res = evaluate_run(ranked_run(vsm, test), qrels())
        if name == "Full system":
            rows.append([name, res["MAP"], "", ""])
            continue
        p = AblationStudy.calculate_significance([base["per_query"]["ap"][q] for q in queries],
                                                 [res["per_query"]["ap"][q] for q in queries])
        rows.append([name, res["MAP"], f"{res['MAP'] - base['MAP']:+.4f}", f"{p:.4f}"])
    table = md_table(["Variant (test queries)", "MAP", "Change", "p-value (paired t-test)"], rows)
    save("ablations", {"rows": rows}, table)


SECTIONS = {"laws": laws, "exact": exact, "construction": construction, "compression": compression,
            "ranking": ranking, "efficiency": efficiency, "ablations": ablations}

if __name__ == "__main__":
    chosen = sys.argv[1:] or ["all"]
    for name, fn in SECTIONS.items():
        if "all" in chosen or name in chosen:
            print(f"\n===== {name} =====")
            fn()
