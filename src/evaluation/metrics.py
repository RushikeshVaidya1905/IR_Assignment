import numpy as np
import matplotlib.pyplot as plt

class Evaluator:
    @staticmethod
    def precision_at_k(retrieved: list[int], relevant: set[int], k: int = 10) -> float:
        """Calculates Precision@K."""
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0
        relevant_retrieved = sum(1 for doc in retrieved_k if doc in relevant)
        return relevant_retrieved / k

    @staticmethod
    def average_precision(retrieved: list[int], relevant: set[int]) -> float:
        """Calculates Average Precision (AP) for a single query."""
        if not relevant:
            return 0.0
            
        relevant_hits = 0
        sum_precisions = 0.0
        
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_hits += 1
                sum_precisions += relevant_hits / (i + 1)
                
        return sum_precisions / len(relevant)

    @staticmethod
    def ndcg_at_k(retrieved: list, relevant, k: int = 10) -> float:
        """Calculates NDCG@K. `relevant` may be a set (binary gain) or a dict doc -> grade (graded gain)."""
        grades = relevant if isinstance(relevant, dict) else {d: 1 for d in relevant}
        if not retrieved or not any(g > 0 for g in grades.values()):
            return 0.0
            
        # Calculate actual DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            gain = grades.get(doc, 0)
            if gain > 0:
                dcg += gain / np.log2(i + 2) # i+2 because rank is 1-indexed (i+1) and formula requires log2(rank+1)
                
        # Calculate Ideal DCG (all judged docs sorted by grade, highest first)
        ideal_gains = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
            
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def eleven_point_precision(retrieved: list, relevant: set) -> list[float]:
        """Interpolated precision at recall levels 0.0, 0.1, ..., 1.0 for a single query."""
        if not relevant:
            return [0.0] * 11
            
        precisions = []
        recalls = []
        relevant_hits = 0
        
        # Calculate P and R at each rank
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_hits += 1
            precisions.append(relevant_hits / (i + 1))
            recalls.append(relevant_hits / len(relevant))
            
        # Maximum precision observed at any recall level >= r
        interpolated_precisions = []
        for r in np.linspace(0.0, 1.0, 11):
            valid_precisions = [p for p, rec in zip(precisions, recalls) if rec >= r - 1e-9]
            interpolated_precisions.append(max(valid_precisions) if valid_precisions else 0.0)
        return interpolated_precisions

    @staticmethod
    def plot_11_point_pr_curve(retrieved, relevant=None, output_file="pr_curve.png", curves: dict = None):
        """Plots 11-point interpolated PR curve(s).
        Either pass (retrieved, relevant) for one query, or curves={label: [11 averaged precisions]}."""
        if curves is None:
            if not relevant:
                return
            curves = {"System": Evaluator.eleven_point_precision(retrieved, relevant)}
            
        recall_levels = np.linspace(0.0, 1.0, 11)
        plt.figure(figsize=(8, 5))
        for label, interpolated_precisions in curves.items():
            plt.plot(recall_levels, interpolated_precisions, marker='o', linestyle='-', linewidth=2, label=label)
        plt.title('11-Point Interpolated Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Interpolated Precision')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(recall_levels)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.legend()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()


def load_qrels(qrels_path: str) -> dict[str, dict[str, int]]:
    """query_id -> {doc_name: grade} (grade 0 entries are kept but count as non-relevant)."""
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f) # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                q_id, doc_id, grade = parts[0], parts[1], int(parts[2])
                qrels.setdefault(q_id, {})[doc_id] = grade
    return qrels

def load_queries(queries_path: str) -> dict[str, str]:
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        next(f) # header
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries

def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict:
    """run: query_id -> ranked list of doc names ('doc_XXXXX'). Returns mean and per-query metrics."""
    per_query = {"p@10": {}, "ap": {}, "ndcg@10": {}}
    curves = []
    for q_id, retrieved in run.items():
        grades = qrels.get(q_id, {})
        relevant = {d for d, g in grades.items() if g > 0}
        if not relevant:
            continue
        per_query["p@10"][q_id] = Evaluator.precision_at_k(retrieved, relevant, k=10)
        per_query["ap"][q_id] = Evaluator.average_precision(retrieved, relevant)
        per_query["ndcg@10"][q_id] = Evaluator.ndcg_at_k(retrieved, grades, k=10)
        curves.append(Evaluator.eleven_point_precision(retrieved, relevant))
    n = len(per_query["ap"]) or 1
    return {
        "P@10": sum(per_query["p@10"].values()) / n,
        "MAP": sum(per_query["ap"].values()) / n,
        "NDCG@10": sum(per_query["ndcg@10"].values()) / n,
        "11pt": list(np.mean(curves, axis=0)) if curves else [0.0] * 11,
        "per_query": per_query,
    }


if __name__ == "__main__":
    import os
    import sys
    from src.querying.vector_space import VectorSpaceModel
    from src.parsing.corpus_parser import doc_id_to_name
    
    # Usage: python -m src.evaluation.metrics [train|test]   (tune on train, run test once)
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    queries_path = f"data/queries_{split}.tsv"
    qrels_path = "data/qrels.tsv"
    
    if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
        print(f"Error: {queries_path} or {qrels_path} not found in data/ folder.")
        exit(1)
        
    qrels = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} {split} queries and qrels for {len(qrels)} queries.")
    
    vsm = VectorSpaceModel()
    
    run = {}
    for q_id, q_text in queries.items():
        # vsm.search returns [(int_doc_id, score)]; qrels use 'doc_XXXXX' names
        results = vsm.search(q_text, top_k=1000)
        run[q_id] = [doc_id_to_name(doc_id) for doc_id, _ in results]
        
    res = evaluate_run(run, qrels)
    print(f"Evaluated {len(res['per_query']['ap'])} queries")
    print(f"P@10:    {res['P@10']:.4f}")
    print(f"MAP:     {res['MAP']:.4f}")
    print(f"NDCG@10: {res['NDCG@10']:.4f}")
    print("11-pt interpolated precision:", " ".join(f"{p:.3f}" for p in res["11pt"]))
    
    Evaluator.plot_11_point_pr_curve(None, curves={f"VSM ({split})": res["11pt"]}, output_file="pr_curve.png")
    print("Saved 11-point interpolated PR curve to pr_curve.png")
