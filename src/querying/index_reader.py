import os
import json
from src.indexing.index_format import read_dictionary, decode_postings, IndexWriter
from src.indexing.spimi import analyze_document
from src.parsing.preprocessor import TextPreprocessor
from src.parsing.reuters_parser import Document

class IndexReader:
    """Reads one Lexis index (see src/indexing/index_format.py).
    Dynamic indexing (IIR 4.5): new documents go into a small auxiliary index and deleted documents into an
    invalidation list (both saved in aux.json). Every lookup combines main + auxiliary and filters deletions,
    and merge_aux() folds them into the main index without re-reading the corpus."""
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.postings_path = os.path.join(index_dir, "postings.bin")
        self.dictionary = read_dictionary(os.path.join(index_dir, "dictionary.bin"))  # term -> (df, offset, size)

        with open(os.path.join(index_dir, "docs.json"), 'r') as f:
            data = json.load(f)
        self.config = data["config"]
        self.docs = {int(d): meta for d, meta in data["docs"].items()}
        self.preprocessor = TextPreprocessor(**self.config)

        # Auxiliary index + deletions
        self.aux_path = os.path.join(index_dir, "aux.json")
        self.aux_postings = {}  # term -> {doc_id: [positions]}
        self.deleted = set()
        if os.path.exists(self.aux_path):
            with open(self.aux_path, 'r') as f:
                aux = json.load(f)
            self.aux_postings = {t: {int(d): p for d, p in docs.items()} for t, docs in aux["postings"].items()}
            self.docs.update({int(d): meta for d, meta in aux["docs"].items()})
            self.deleted = set(aux["deleted"])
        self._cache = {}

    def get_positions(self, term: str) -> dict[int, list[int]]:
        """Reads and decodes the positional postings of a term: doc_id -> sorted positions."""
        if term in self._cache:
            return self._cache[term]
        positions = {}
        if term in self.dictionary:
            _, offset, size = self.dictionary[term]
            with open(self.postings_path, 'rb') as f:
                f.seek(offset)
                positions = decode_postings(f.read(size))
        if term in self.aux_postings:
            positions.update(self.aux_postings[term])
        if self.deleted:
            positions = {d: p for d, p in positions.items() if d not in self.deleted}
        self._cache[term] = positions
        return positions

    def get_postings(self, term: str) -> list[int]:
        """Sorted list of doc IDs containing the term."""
        return sorted(self.get_positions(term))

    def df(self, term: str) -> int:
        if not self.deleted and term not in self.aux_postings:
            return self.dictionary[term][0] if term in self.dictionary else 0
        return len(self.get_positions(term))

    def vocabulary(self) -> set[str]:
        return set(self.dictionary) | set(self.aux_postings)

    def get_all_doc_ids(self) -> set[int]:
        """All live doc IDs (used for NOT queries)."""
        return set(self.docs) - self.deleted

    @property
    def N(self) -> int:
        return len(self.docs) - len(self.deleted)

    # ---------- dynamic indexing ----------
    def add_document(self, doc_id: int, title: str, body: str):
        tokens, meta = analyze_document(self.preprocessor, Document(doc_id, title, body))
        for term, pos in tokens:
            self.aux_postings.setdefault(term, {}).setdefault(doc_id, []).append(pos)
        self.docs[doc_id] = meta
        self.deleted.discard(doc_id)
        self._cache.clear()
        self._save_aux()

    def delete_document(self, doc_id: int):
        self.deleted.add(doc_id)
        self._cache.clear()
        self._save_aux()

    def _save_aux(self):
        aux_doc_ids = {d for docs in self.aux_postings.values() for d in docs}
        with open(self.aux_path, 'w') as f:
            json.dump({"postings": self.aux_postings,
                       "docs": {d: self.docs[d] for d in aux_doc_ids if d in self.docs},
                       "deleted": sorted(self.deleted)}, f)

    def merge_aux(self):
        """Merges the auxiliary index and deletions into the main index files."""
        tmp_dir = self.index_dir + "_tmp"
        writer = IndexWriter(tmp_dir)
        for term in sorted(self.vocabulary()):
            writer.add(term, self.get_positions(term))
        live_docs = {d: meta for d, meta in self.docs.items() if d not in self.deleted}
        writer.close(live_docs, self.config)
        for name in ("postings.bin", "dictionary.bin", "docs.json"):
            os.replace(os.path.join(tmp_dir, name), os.path.join(self.index_dir, name))
        os.rmdir(tmp_dir)
        if os.path.exists(self.aux_path):
            os.remove(self.aux_path)
        self.__init__(self.index_dir)
