from src.querying.boolean_parser import BooleanSearchEngine

def main():
    print("--- Loading Index Shards ---")
    engine = BooleanSearchEngine(shards_dir="index_data", num_shards=4)
    print(f"Index loaded. Total documents in corpus: {len(engine.universal_set)}")
    
    # Define some test queries covering AND, OR, NOT, and parentheses
    queries = [
        "cocoa",
        "cocoa AND drought",
        "bahia OR (cocoa AND drought)",
        "cocoa NOT drought"
    ]
    
    print("\n--- Running Boolean Queries ---")
    for q in queries:
        results = engine.search(q)
        print(f"Query: '{q}'")
        print(f"Results (Doc IDs): {results}\n")

if __name__ == "__main__":
    main()