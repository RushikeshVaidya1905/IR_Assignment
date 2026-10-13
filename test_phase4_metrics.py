from src.querying.vector_space import VectorSpaceModel
from src.evaluation.metrics import Evaluator
from src.evaluation.ablation import AblationStudy

def main():
    print("--- Phase 4: IR Metrics & Ablation ---")
    vsm_engine = VectorSpaceModel()
    evaluator = Evaluator()
    
    # 1. Define a benchmark query and its assumed "Ground Truth" (QREL) relevant documents
    query = "market inflation rates"
    ground_truth_relevant = {18001, 19001, 20001, 1} # Simulated relevant set
    
    # 2. Retrieve ranked results
    print(f"\nQuerying: '{query}'")
    vsm_results = vsm_engine.search(query, top_k=10)
    retrieved_docs = [doc_id for doc_id, score in vsm_results]
    print(f"System Retrieved Docs: {retrieved_docs}")
    print(f"Ground Truth Relevant Docs: {list(ground_truth_relevant)}")
    
    # 3. Calculate IR Metrics
    p_at_3 = evaluator.precision_at_k(retrieved_docs, ground_truth_relevant, k=3)
    ap = evaluator.average_precision(retrieved_docs, ground_truth_relevant)
    ndcg = evaluator.ndcg_at_k(retrieved_docs, ground_truth_relevant, k=3)
    
    print("\n--- System Metrics ---")
    print(f"Precision@3: {p_at_3:.3f}")
    print(f"Average Precision (AP): {ap:.3f}")
    print(f"NDCG@3: {ndcg:.3f}")
    
    # 4. Generate PR Curve
    print("\nGenerating 11-point PR Curve...")
    evaluator.plot_11_point_pr_curve(retrieved_docs, ground_truth_relevant, output_file="pr_curve.png")
    print("-> PR Curve saved to 'pr_curve.png'")
    
    # 5. Simulate Ablation Study (Paired t-test)
    print("\n--- Simulated Ablation Study (Removing Stemmer) ---")
    # Simulated MAP scores across 5 different queries for baseline vs. ablated system
    baseline_map = [0.85, 0.70, 0.90, 0.65, 0.80]
    no_stemmer_map = [0.60, 0.65, 0.75, 0.50, 0.60] 
    
    p_val = AblationStudy.calculate_significance(baseline_map, no_stemmer_map)
    print(f"Baseline MAP distribution: {baseline_map}")
    print(f"Ablated MAP distribution: {no_stemmer_map}")
    print(f"Calculated p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Removing the stemmer caused a statistically significant drop in performance.")
    else:
        print("Conclusion: The performance drop was not statistically significant.")

if __name__ == "__main__":
    main()