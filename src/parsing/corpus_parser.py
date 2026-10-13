import os
import glob
from src.parsing.reuters_parser import Document

class CorpusParser:
    """Streams the official Lexis corpus: data/corpus/doc_XXXXX.txt (title line, blank line, body)."""
    def __init__(self, corpus_dir: str = os.path.join("data", "corpus")):
        self.corpus_dir = corpus_dir

    def list_files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.corpus_dir, "doc_*.txt")))

    def parse_file(self, file_path: str) -> Document:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            title = f.readline().strip()
            body = f.read().strip()
        return Document(doc_id=filename_to_doc_id(file_path), title=title, body=body)

    def parse_files(self, file_paths: list[str]):
        for file_path in file_paths:
            yield self.parse_file(file_path)

    def parse_all(self):
        yield from self.parse_files(self.list_files())


def filename_to_doc_id(file_path: str) -> int:
    """doc_09756.txt -> 9756"""
    return int(os.path.basename(file_path)[4:-4])

def doc_id_to_name(doc_id: int) -> str:
    """9756 -> 'doc_09756' (the format used in qrels.tsv and boolean_tests.tsv)"""
    return f"doc_{doc_id:05d}"

def doc_name_to_id(doc_name: str) -> int:
    """'doc_09756' -> 9756"""
    return int(doc_name.strip()[4:])
