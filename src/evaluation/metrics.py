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
    def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int = 10) -> float:
        """Calculates Normalized Discounted Cumulative Gain at K using binary relevance."""
        if not retrieved or not relevant:
            return 0.0
            
        # Calculate actual DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                dcg += 1.0 / np.log2(i + 2) # i+2 because rank is 1-indexed (i+1) and formula requires log2(rank+1)
                
        # Calculate Ideal DCG (if all relevant docs were ranked at the top)
        ideal_retrieved = list(relevant)[:k]
        idcg = 0.0
        for i in range(len(ideal_retrieved)):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def plot_11_point_pr_curve(retrieved: list[int], relevant: set[int], output_file="pr_curve.png"):
        """Generates an 11-point interpolated Precision-Recall curve."""
        if not relevant:
            return
            
        precisions = []
        recalls = []
        relevant_hits = 0
        
        # Calculate P and R at each rank
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_hits += 1
            precisions.append(relevant_hits / (i + 1))
            recalls.append(relevant_hits / len(relevant))
            
        # Interpolate precision at 11 standard recall levels (0.0 to 1.0)
        recall_levels = np.arange(0.0, 1.1, 0.1)
        interpolated_precisions = []
        
        for r in recall_levels:
            # Maximum precision observed at any recall level >= r
            valid_precisions = [p for p, rec in zip(precisions, recalls) if rec >= r]
            p_max = max(valid_precisions) if valid_precisions else 0.0
            interpolated_precisions.append(p_max)
            
        plt.figure(figsize=(8, 5))
        plt.plot(recall_levels, interpolated_precisions, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title('11-Point Interpolated Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Interpolated Precision')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(recall_levels)
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')


# Append this to the bottom of src/evaluation/metrics.py

if __name__ == "__main__":
    import os
    
    # Paths to the dataset files placed in data/
    queries_path = "data/queries_test.tsv"  # or queries_train.tsv for tuning
    qrels_path = "data/qrels.tsv"
    
    if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
        print("Error: Test query or qrels files not found in data/ folder.")
        exit(1)
        
    # 1. Load qrels: mapping query_id -> set of relevant doc_ids
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                q_id, doc_id, grade = parts[0], parts[1], int(parts[2])
                if grade > 0: # Relevant or highly relevant
                    if q_id not in qrels:
                        qrels[q_id] = set()
                    qrels[q_id].add(doc_id)
                    
    # 2. Load queries
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                q_id, q_text = parts[0], parts[1]
                queries[q_id] = q_text

    print(f"Loaded {len(queries)} queries and {len(qrels)} qrel entries.")
    
    # 3. Simulate or invoke your search engine retriever here
    # Example placeholder loop integrating with your index and vector space model:
    # from src.querying.vector_space import VectorSpaceModel
    # vsm = VectorSpaceModel()
    from src.querying.vector_space import VectorSpaceModel  # Adjust the import path if your file name is different
    vsm = VectorSpaceModel()
    
    aps = []
    ndcgs = []
    all_retrieved_for_curve = []
    all_relevant_for_curve = set()
    
    for q_id, q_text in queries.items():
        if q_id not in qrels:
            continue
            
        relevant_docs = qrels[q_id]
        
        # TODO: Replace with your actual search engine retrieval call:
        # retrieved_docs = vsm.search(q_text, top_k=100)
        retrieved_docs = vsm.search(q_text, top_k=100)

        # print(f"--- Debug for Query ID: {q_id} ---")
        # print(f"Query Text: {q_text}")
        # print(f"Retrieved Docs (first 5): {retrieved_docs[:5]}")
        # print(f"Relevant Docs in Qrels: {list(relevant_docs)[:5]}")
        
        ap = Evaluator.average_precision(retrieved_docs, relevant_docs)
        ndcg = Evaluator.ndcg_at_k(retrieved_docs, relevant_docs, k=10)
        
        aps.append(ap)
        ndcgs.append(ndcg)
        
        all_retrieved_for_curve.extend(retrieved_docs)
        all_relevant_for_curve.update(relevant_docs)
        
    mean_ap = sum(aps) / len(aps) if aps else 0.0
    mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    
    print(f"Mean Average Precision (MAP): {mean_ap:.4f}")
    print(f"Mean NDCG@10: {mean_ndcg:.4f}")
    
    # Generate and save PR curve
    Evaluator.plot_11_point_pr_curve(all_retrieved_for_curve, all_relevant_for_curve, output_file="pr_curve.png")
    print("Saved 11-point interpolated PR curve to pr_curve.png")