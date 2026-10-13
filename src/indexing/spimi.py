import os
import json
import math
import heapq
from collections import defaultdict, Counter

# Block files are text files with one line per term, sorted by term:  term \t {"doc_id": [positions], ...}
# Both SPIMI and BSBI write this format, so the same external k-way merge is used for both.


class SPIMIIndexer:
    """Single-pass in-memory indexing (IIR 4.3): term -> {doc_id: [positions]} is built directly in memory.
    Dummy memory constraint: at most `memory_limit` postings (token positions) may be held in memory;
    when it is reached the block is sorted by term and written to disk."""
    def __init__(self, output_dir: str, memory_limit: int = 100_000, block_prefix: str = "block"):
        self.output_dir = output_dir
        self.memory_limit = memory_limit
        self.block_prefix = block_prefix
        self.dictionary = defaultdict(lambda: defaultdict(list))
        self.in_memory = 0
        self.block_number = 0
        self.block_files = []
        self.bytes_written = 0
        os.makedirs(output_dir, exist_ok=True)

    def is_memory_full(self) -> bool:
        return self.in_memory >= self.memory_limit

    def add_token(self, term: str, doc_id: int, position: int):
        self.dictionary[term][doc_id].append(position)
        self.in_memory += 1

    def _sorted_block(self):
        for term in sorted(self.dictionary.keys()):
            yield term, self.dictionary[term]

    def _clear(self):
        self.dictionary.clear()

    def flush_to_disk(self):
        if self.in_memory == 0:
            return
        block_file = os.path.join(self.output_dir, f"{self.block_prefix}_{self.block_number}.txt")
        with open(block_file, 'w') as f:
            for term, docs in self._sorted_block():
                f.write(term + "\t" + json.dumps(docs) + "\n")
        self.bytes_written += os.path.getsize(block_file)
        self.block_files.append(block_file)
        self.block_number += 1
        self.in_memory = 0
        self._clear()


class BSBIIndexer(SPIMIIndexer):
    """Blocked sort-based indexing (IIR 4.2), for comparison with SPIMI: terms are mapped to termIDs
    (a global term -> termID table kept in memory), (termID, docID, position) triples are collected,
    and each full block is sorted by termID before being written."""
    def __init__(self, output_dir: str, memory_limit: int = 100_000, block_prefix: str = "block"):
        super().__init__(output_dir, memory_limit, block_prefix)
        self.term_ids = {}
        self.terms = []
        self.triples = []

    def add_token(self, term: str, doc_id: int, position: int):
        term_id = self.term_ids.get(term)
        if term_id is None:
            term_id = len(self.terms)
            self.term_ids[term] = term_id
            self.terms.append(term)
        self.triples.append((term_id, doc_id, position))
        self.in_memory += 1

    def _sorted_block(self):
        self.triples.sort()
        groups = {}
        for term_id, doc_id, pos in self.triples:
            groups.setdefault(term_id, {}).setdefault(doc_id, []).append(pos)
        # write in term order so BSBI blocks use the same merge as SPIMI blocks
        for term_id in sorted(groups, key=lambda t: self.terms[t]):
            yield self.terms[term_id], groups[term_id]

    def _clear(self):
        self.triples = []


def _read_block(path: str):
    with open(path, 'r') as f:
        for line in f:
            term, data = line.rstrip('\n').split('\t', 1)
            yield term, data


def merge_blocks(block_files: list[str], writer):
    """External k-way merge (heap over the block files). Only one term's postings are in memory at a time."""
    streams = [_read_block(p) for p in block_files]
    current_term = None
    postings = {}
    for term, data in heapq.merge(*streams, key=lambda x: x[0]):
        if term != current_term:
            if current_term is not None:
                writer.add(current_term, postings)
            current_term = term
            postings = {}
        for doc_id, positions in json.loads(data).items():
            postings.setdefault(int(doc_id), []).extend(positions)
    if current_term is not None:
        writer.add(current_term, postings)
    for path in block_files:
        os.remove(path)


def analyze_document(preprocessor, doc):
    """Turns a document into (term, position) pairs plus its per-document data
    [title_len, num_tokens, norm_title, norm_body, norm_all] (norms of the 1+log10(tf) vectors).
    Positions run continuously through title then body, like a scan of the lowercased text."""
    title_len = len(preprocessor.tokenize(doc.title))
    title_tokens = preprocessor.process_with_positions(doc.title)
    body_tokens = [(t, pos + title_len) for t, pos in preprocessor.process_with_positions(doc.body)]

    def norm(counter):
        return round(math.sqrt(sum((1 + math.log10(tf)) ** 2 for tf in counter.values())), 4)

    title_tf = Counter(t for t, _ in title_tokens)
    body_tf = Counter(t for t, _ in body_tokens)
    meta = [title_len, len(title_tokens) + len(body_tokens),
            norm(title_tf), norm(body_tf), norm(title_tf + body_tf)]
    return title_tokens + body_tokens, meta
