"""Document loader: PDF file -> list of LangChain `Document`s, one per page.

Analogy: the mailroom clerk who opens every envelope and hands you plain pages,
each stamped with where it came from (file name + page number).

Why our own loader? Older tutorials import `PyPDFLoader` from `langchain_community`,
but that package is being sunset. A loader is only ~15 lines on top of `pypdf`,
and writing it shows there is no magic: a loader just returns `Document` objects.
"""
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError


class PdfLoader(BaseLoader):
    def __init__(self, path: str | Path, source_name: str | None = None):
        self.path = Path(path)
        self.source_name = source_name or self.path.name

    def lazy_load(self) -> Iterator[Document]:
        try:
            reader = PdfReader(self.path)
            if reader.is_encrypted:
                raise ValueError(f"{self.source_name}: the PDF is password-protected.")
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:  # skip blank pages and pages that are only images
                    yield Document(
                        page_content=text,
                        metadata={"source": self.source_name, "page": number},
                    )
        except (PyPdfError, OSError) as exc:
            raise ValueError(f"{self.source_name}: not a readable PDF ({exc}).") from exc
