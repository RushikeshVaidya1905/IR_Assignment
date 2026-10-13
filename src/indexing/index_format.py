"""On-disk index format shared by the writer (indexing) and the reader (querying).

postings.bin   for each term (in dictionary order), for each doc containing it:
                   VB(docID gap), VB(tf), VB(position gaps ...)        -> positional postings, gap + VB encoded
dictionary.bin for each term in sorted order (front coding, IIR 5.2.2):
                   VB(length of prefix shared with previous term), VB(suffix length), suffix bytes,
                   VB(df), VB(byte length of the term's postings)       -> offsets are cumulative sums
docs.json      index config (stemming / stop words) and per-document data:
                   doc_id -> [title_len, num_tokens, norm_title, norm_body, norm_all]
               (title_len = number of title tokens; positions < title_len are in the title zone)
"""
import os
import json
from src.indexing.compress import VariableByteEncoder as VB


def encode_postings(postings: dict[int, list[int]]) -> bytes:
    numbers = []
    prev_doc = 0
    for doc_id in sorted(postings):
        positions = sorted(postings[doc_id])
        numbers.append(doc_id - prev_doc)
        numbers.append(len(positions))
        prev_pos = 0
        for pos in positions:
            numbers.append(pos - prev_pos)
            prev_pos = pos
        prev_doc = doc_id
    return VB.encode_list(numbers)


def decode_postings(stream: bytes) -> dict[int, list[int]]:
    numbers = VB.decode(stream)
    postings = {}
    i = 0
    doc_id = 0
    while i < len(numbers):
        doc_id += numbers[i]
        tf = numbers[i + 1]
        i += 2
        positions = []
        pos = 0
        for gap in numbers[i:i + tf]:
            pos += gap
            positions.append(pos)
        i += tf
        postings[doc_id] = positions
    return postings


class IndexWriter:
    """Writes postings term by term (terms must arrive in sorted order), then the dictionary."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.postings_file = open(os.path.join(output_dir, "postings.bin"), 'wb')
        self.entries = []  # (term, df, byte_size)

    def add(self, term: str, postings: dict[int, list[int]]):
        if not postings:
            return
        encoded = encode_postings(postings)
        self.postings_file.write(encoded)
        self.entries.append((term, len(postings), len(encoded)))

    def close(self, docs: dict, config: dict) -> int:
        """Writes dictionary.bin and docs.json. Returns total bytes of the final index."""
        self.postings_file.close()
        out = bytearray()
        prev = ""
        for term, df, size in self.entries:
            prefix = 0
            while prefix < min(len(prev), len(term)) and prev[prefix] == term[prefix]:
                prefix += 1
            suffix = term[prefix:].encode('utf-8')
            out += VB.encode_number(prefix) + VB.encode_number(len(suffix)) + suffix
            out += VB.encode_number(df) + VB.encode_number(size)
            prev = term
        with open(os.path.join(self.output_dir, "dictionary.bin"), 'wb') as f:
            f.write(out)
        with open(os.path.join(self.output_dir, "docs.json"), 'w') as f:
            json.dump({"config": config, "docs": docs}, f)
        return sum(os.path.getsize(os.path.join(self.output_dir, n))
                   for n in ("postings.bin", "dictionary.bin", "docs.json"))


def read_dictionary(path: str) -> dict[str, tuple[int, int, int]]:
    """Decodes dictionary.bin into term -> (df, offset, byte_size)."""
    with open(path, 'rb') as f:
        data = f.read()
    dictionary = {}
    i = 0
    prev = ""
    offset = 0

    def next_number():
        nonlocal i
        n = 0
        while True:
            byte = data[i]
            i += 1
            if byte < 128:
                n = 128 * n + byte
            else:
                return 128 * n + (byte - 128)

    while i < len(data):
        prefix = next_number()
        suffix_len = next_number()
        term = prev[:prefix] + data[i:i + suffix_len].decode('utf-8')
        i += suffix_len
        df = next_number()
        size = next_number()
        dictionary[term] = (df, offset, size)
        offset += size
        prev = term
    return dictionary
