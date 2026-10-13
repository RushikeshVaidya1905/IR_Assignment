from src.querying.positional import PositionalSearchEngine
from src.querying.vector_space import VectorSpaceModel

def main():
    print("--- Initializing Search Engines ---")
    pos_engine = PositionalSearchEngine()
    vsm_engine = VectorSpaceModel()
    
    print("\n--- Testing Positional / Proximity Queries ---")
    # Using terms we know are in Document 1 from your previous test (e.g., 'bahia' and 'cocoa')
    t1, t2 = "bahia", "cocoa"
    print(f"Query: '{t1}' within 2 words of '{t2}'")
    pos_results = pos_engine.search_proximity(t1, t2, k=2)
    print(f"Results (Doc IDs): {pos_results}")
    
    print(f"\nQuery: '{t1}' within 5 words of '{t2}'")
    pos_results_wide = pos_engine.search_proximity(t1, t2, k=5)
    print(f"Results (Doc IDs): {pos_results_wide}")

    print("\n--- Testing Ranked Free-Text Retrieval (Vector Space Model) ---")
    queries = [
        "cocoa drought in bahia",
        "market inflation rates",
        "agricultural review"
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        vsm_results = vsm_engine.search(q, top_k=3)
        if not vsm_results:
            print("No matching documents found.")
        else:
            for rank, (doc_id, score) in enumerate(vsm_results, 1):
                print(f"  Rank {rank}: Doc ID {doc_id} (Score: {score:.4f})")

if __name__ == "__main__":
    main()