"""Step 1 - Document loader + text splitter. No model needed.

    python step1_load_split.py [path/to/file.pdf]
"""
import sys

from qabot import config
from qabot.ingest import split
from qabot.loader import PdfLoader

path = sys.argv[1] if len(sys.argv) > 1 else "sample_docs/acme_broadband_handbook.pdf"

# 1. LOAD: one Document per page, each with metadata saying where it came from.
pages = PdfLoader(path).load()
print(f"Loaded {len(pages)} pages from {path}")
print(f"Page 1 metadata: {pages[0].metadata}")
print(f"Page 1 starts:   {pages[0].page_content[:80]!r}\n")

# 2. SPLIT: cut pages into overlapping chunks small enough to embed and to fit in a prompt.
chunks = split(pages)
sizes = [len(c.page_content) for c in chunks]
print(f"Split into {len(chunks)} chunks (chunk_size={config.chunk_size()}, overlap={config.chunk_overlap()})")
print(f"Chunk sizes in characters: min {min(sizes)}, max {max(sizes)}\n")

for i in (0, 1):
    c = chunks[i]
    print(f"--- chunk {i} | page {c.metadata['page']} | starts at character {c.metadata['start_index']} ---")
    print(c.page_content, "\n")

print("Look at the end of chunk 0 and the start of chunk 1: the repeated text is the overlap.")
