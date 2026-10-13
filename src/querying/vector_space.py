import math
import heapq
from collections import defaultdict
from src.querying.index_reader import IndexReader

class VectorSpaceModel:
    """Ranked free-text retrieval: cosine similarity of tf-idf vectors (lnc.ltc, IIR 6.4),
    with weighted zone scoring (IIR 6.1):  score = g * cos(query, title) + (1 - g) * cos(query, body).

    Switches used for the baseline and the ablations:
      use_idf=False          -> query weight is 1 + log tf (no idf)
      use_length_norm=False  -> no cosine length normalization of documents
      use_zones=False        -> title and body are treated as one field
      raw_tf=True            -> raw term frequency instead of 1 + log tf (raw-tf baseline)

    Only documents in the postings of the query terms are scored (term-at-a-time) and the top k is taken
    with a heap. Two inexact top-k methods from IIR 7.1 can be used instead of exhaustive scoring:
      method="index_elimination" -> ignore low-idf query terms and only score docs containing most query terms
      method="champion"          -> only score docs in the champion lists (top r docs per term)"""
    def __init__(self, index_dir: str = "index_data/ranked", title_weight: float = 0.1,
                 use_idf: bool = True, use_length_norm: bool = True, use_zones: bool = True,
                 raw_tf: bool = False, champion_r: int = 50, idf_threshold: float = 1.0):
        self.reader = IndexReader(index_dir)
        self.preprocessor = self.reader.preprocessor
        self.N = self.reader.N
        self.title_weight = title_weight
        self.use_idf = use_idf
        self.use_length_norm = use_length_norm
        self.use_zones = use_zones
        self.raw_tf = raw_tf
        self.champion_r = champion_r
        self.idf_threshold = idf_threshold
        self.champion_lists = None
        self.last_docs_scored = 0

    def _tf_weight(self, tf: int) -> float:
        if tf == 0:
            return 0.0
        return tf if self.raw_tf else 1 + math.log10(tf)

    def doc_term_weight(self, doc_id: int, positions: list[int]) -> float:
        """Weight of a term in a document (document side of lnc), combining the zones."""
        title_len, _, norm_title, norm_body, norm_all = self.reader.docs[doc_id]
        if not self.use_zones:
            w = self._tf_weight(len(positions))
            return w / norm_all if self.use_length_norm and norm_all > 0 else w
        tf_title = sum(1 for p in positions if p < title_len)
        w_title = self._tf_weight(tf_title)
        w_body = self._tf_weight(len(positions) - tf_title)
        if self.use_length_norm:
            w_title = w_title / norm_title if norm_title > 0 else 0.0
            w_body = w_body / norm_body if norm_body > 0 else 0.0
        return self.title_weight * w_title + (1 - self.title_weight) * w_body

    def idf(self, term: str) -> float:
        df = self.reader.df(term)
        return math.log10(self.N / df) if df else 0.0

    def query_weights(self, query: str) -> dict[str, float]:
        """Query side of ltc: (1 + log tf) * idf for every query term that is in the vocabulary."""
        query_tf = defaultdict(int)
        for token in self.preprocessor.process(query):
            query_tf[token] += 1
        weights = {}
        for token, q_tf in query_tf.items():
            if self.reader.df(token) == 0:
                continue
            weights[token] = self._tf_weight(q_tf) * (self.idf(token) if self.use_idf else 1.0)
        return weights

    def build_champion_lists(self):
        """For every term, the r documents with the highest weight for that term (computed once, IIR 7.1.3)."""
        self.champion_lists = {}
        for term in self.reader.vocabulary():
            postings = self.reader.get_positions(term)
            best = heapq.nlargest(self.champion_r, postings.items(),
                                  key=lambda item: self.doc_term_weight(item[0], item[1]))
            self.champion_lists[term] = {doc_id for doc_id, _ in best}
        self.reader._cache.clear()

    def search(self, query: str, top_k: int = 10, method: str = "exhaustive") -> list[tuple[int, float]]:
        """Returns the top_k (doc_id, score) pairs, best first."""
        weights = self.query_weights(query)
        if not weights:
            self.last_docs_scored = 0
            return []

        candidates = None  # None = every doc in the postings of the query terms
        if method == "index_elimination":
            # 1) drop low-idf terms (keep at least the rarest one)
            kept = {t: w for t, w in weights.items() if self.idf(t) >= self.idf_threshold}
            if not kept:
                rarest = max(weights, key=self.idf)
                kept = {rarest: weights[rarest]}
            weights = kept
            # 2) only docs that contain at least half of the remaining query terms
            counts = defaultdict(int)
            for term in weights:
                for doc_id in self.reader.get_positions(term):
                    counts[doc_id] += 1
            needed = math.ceil(len(weights) / 2)
            candidates = {d for d, c in counts.items() if c >= needed}
        elif method == "champion":
            if self.champion_lists is None:
                self.build_champion_lists()
            candidates = set()
            for term in weights:
                candidates |= self.champion_lists.get(term, set())

        # Term-at-a-time accumulation
        scores = defaultdict(float)
        for term, w_q in weights.items():
            for doc_id, positions in self.reader.get_positions(term).items():
                if candidates is None or doc_id in candidates:
                    scores[doc_id] += w_q * self.doc_term_weight(doc_id, positions)

        self.last_docs_scored = len(scores)
        return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
