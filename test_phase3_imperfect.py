from src.querying.wildcard import PermutermIndex
from src.querying.spellcheck import SpellChecker

def main():
    print("--- Initializing Imperfect Query Engines ---")
    wildcard_engine = PermutermIndex()
    spell_checker = SpellChecker()
    
    print("\n--- Testing Wildcard Queries (Permuterm Index) ---")
    wildcard_queries = [
        "coco*",   # Prefix match
        "*ght",    # Suffix match (e.g., drought)
        "b*ia"     # Interior wildcard (e.g., bahia)
    ]
    
    for wq in wildcard_queries:
        matches = wildcard_engine.search(wq)
        print(f"Wildcard: '{wq}' -> Matches: {matches}")

    print("\n--- Testing Spelling Correction (Levenshtein Distance) ---")
    misspelled_queries = [
        "cacoa",      # Should suggest 'cocoa'
        "droght",     # Should suggest 'drought'
        "inflaton"    # Should suggest 'inflation'
    ]
    
    for mq in misspelled_queries:
        suggestions = spell_checker.suggest(mq)
        print(f"Misspelled: '{mq}' -> Suggestions: {suggestions}")

if __name__ == "__main__":
    main()