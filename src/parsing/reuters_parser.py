import os
import glob
import re
from dataclasses import dataclass

@dataclass
class Document:
    doc_id: int
    title: str
    body: str

class ReutersParser:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir

    def parse_file(self, file_path: str):
        """Parses a single .sgm file using regex to bypass malformed SGML tags."""
        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
            content = f.read()

        # Extract all REUTERS blocks and their NEWID attributes
        articles = re.findall(r'<REUTERS.*?NEWID="(\d+)".*?>(.*?)</REUTERS>', content, re.DOTALL | re.IGNORECASE)

        for doc_id_str, article_content in articles:
            doc_id = int(doc_id_str)

            title_match = re.search(r'<TITLE>(.*?)</TITLE>', article_content, re.DOTALL | re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else ""

            body_match = re.search(r'<BODY>(.*?)</BODY>', article_content, re.DOTALL | re.IGNORECASE)
            body = body_match.group(1).strip() if body_match else ""
            
            # Clean up the trailing Reuters junk characters
            body = re.sub(r'Reuter\s*&#3;', '', body).strip()

            if title or body:
                yield Document(doc_id=doc_id, title=title, body=body)

    def parse_all(self):
        """Iterates over all .sgm files in the data/raw directory."""
        sgm_files = sorted(glob.glob(os.path.join(self.raw_data_dir, "reut2-*.sgm")))
        for file_path in sgm_files:
            yield from self.parse_file(file_path)