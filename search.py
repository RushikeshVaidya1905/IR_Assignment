"""Lexis command-line search.

  python search.py                       interactive prompt
  python search.py "oil AND opec"        one query

Queries with AND / OR / NOT, parentheses, "phrases", proximity (a /k b) or wildcards (*) are answered
exactly from the unstemmed index; any other query is ranked (tf-idf cosine, top 10).
Query words that are not in the vocabulary get a spelling suggestion.
"""
import re
import sys
from src.querying.boolean_parser import BooleanSearchEngine
from src.querying.vector_space import VectorSpaceModel
from src.querying.spellcheck import SpellChecker
from src.parsing.corpus_parser import doc_id_to_name
from src.parsing.preprocessor import TextPreprocessor

def is_exact_query(query: str) -> bool:
    return bool(re.search(r'\b(AND|OR|NOT)\b|["()*]|/\d+', query))

def main():
    boolean = BooleanSearchEngine("index_data/exact")
    ranked = VectorSpaceModel("index_data/ranked")  # title weight g = 0.1, tuned on the training queries
    speller = SpellChecker(boolean.reader)
    tokenizer = TextPreprocessor(use_stemming=False, remove_stopwords=False)

    queries = sys.argv[1:] or iter(lambda: input("\nlexis> "), "")
    for query in queries:
        words = [w for w in tokenizer.tokenize(re.sub(r'\b(AND|OR|NOT)\b|\S*\*\S*|/\d+', ' ', query))]
        for w in words:
            if w not in speller.vocabulary:
                suggestion = speller.suggest(w)
                print(f"  '{w}' is not in the vocabulary." + (f" Did you mean '{suggestion[0]}'?" if suggestion else ""))

        if is_exact_query(query):
            results = boolean.search(query)
            print(f"{len(results)} documents: " + ", ".join(doc_id_to_name(d) for d in results[:20])
                  + (" ..." if len(results) > 20 else ""))
        else:
            for rank, (doc_id, score) in enumerate(ranked.search(query, top_k=10), 1):
                with open(f"data/corpus/{doc_id_to_name(doc_id)}.txt", encoding="utf-8", errors="ignore") as f:
                    title = f.readline().strip()
                print(f"{rank:2}. {doc_id_to_name(doc_id)}  {score:.4f}  {title}")

if __name__ == "__main__":
    main()
