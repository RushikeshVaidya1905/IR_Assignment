"""Checks Lexis against the assignment's test files (build the indexes first: python build_index.py).
  1. data/boolean_tests.tsv : Boolean / phrase / wildcard / proximity results must equal the expected sets
  2. data/misspellings.tsv  : top spelling suggestion vs the intended word
  3. dynamic indexing       : add + delete documents without rebuilding
"""
import shutil
from src.querying.boolean_parser import BooleanSearchEngine
from src.querying.spellcheck import SpellChecker
from src.parsing.corpus_parser import doc_id_to_name

def test_boolean(engine):
    passed = 0
    lines = list(open("data/boolean_tests.tsv", encoding="utf-8"))[1:]
    for line in lines:
        q_id, query, q_type, _, expected = line.rstrip("\n").split("\t")
        got = {doc_id_to_name(d) for d in engine.search(query)}
        expected = set(expected.split(",")) if expected else set()
        passed += got == expected
        print(f"  {q_id} {q_type:9} {query:40} {'PASS' if got == expected else 'FAIL'} ({len(got)}/{len(expected)})")
    print(f"Boolean tests: {passed}/{len(lines)} identical to the expected results\n")

def test_spelling(engine):
    checker = SpellChecker(engine.reader)
    correct = 0
    lines = list(open("data/misspellings.tsv", encoding="utf-8"))[1:]
    for line in lines:
        word, accepted = line.rstrip("\n").split("\t")
        suggestions = checker.suggest(word)
        ok = bool(suggestions) and suggestions[0] in accepted.split(",")
        correct += ok
        if not ok:
            print(f"  {word}: expected {accepted}, suggested {suggestions}")
    print(f"Spelling: {correct}/{len(lines)} top-1 suggestions correct\n")

def test_dynamic():
    shutil.copytree("index_data/exact", "index_data/dynamic_test", dirs_exist_ok=True)
    try:
        engine = BooleanSearchEngine("index_data/dynamic_test")
        before = len(engine.search("oil AND opec"))
        engine.reader.add_document(99999, "Zzyzx oil find", "OPEC said the zzyzx field holds crude oil.")
        engine.reader.delete_document(34)  # doc_00034 contains oil and opec
        engine = BooleanSearchEngine("index_data/dynamic_test")  # reload from disk
        assert engine.search("zzyzx") == [99999] and engine.search('"crude oil" AND zzyzx') == [99999]
        assert 34 not in engine.search("oil") and len(engine.search("oil AND opec")) == before
        engine.reader.merge_aux()
        engine = BooleanSearchEngine("index_data/dynamic_test")
        assert engine.search("zzyzx") == [99999] and 34 not in engine.search("oil")
        print("Dynamic indexing: add, delete, reload and merge OK\n")
    finally:
        shutil.rmtree("index_data/dynamic_test")

if __name__ == "__main__":
    engine = BooleanSearchEngine("index_data/exact")
    test_boolean(engine)
    test_spelling(engine)
    test_dynamic()
