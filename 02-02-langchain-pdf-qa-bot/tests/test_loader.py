"""Loader + splitter: no model involved."""
import pytest

from qabot.ingest import file_id, split
from qabot.loader import PdfLoader


def test_loader_returns_one_document_per_page(sample_pdf):
    pages = PdfLoader(sample_pdf).load()
    assert len(pages) == 6
    assert pages[0].metadata == {"source": "acme_broadband_handbook.pdf", "page": 1}
    assert "late payment fee of 7.50 dollars" in pages[2].page_content.replace("\n", " ")


def test_loader_rejects_a_file_that_is_not_a_pdf(tmp_path):
    fake = tmp_path / "notes.pdf"
    fake.write_text("just text pretending to be a PDF")
    with pytest.raises(ValueError, match="not a readable PDF"):
        PdfLoader(fake).load()


def test_split_respects_chunk_size_and_keeps_metadata(sample_pdf):
    chunks = split(PdfLoader(sample_pdf).load())
    assert len(chunks) > 6                                    # more chunks than pages
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all({"source", "page", "start_index"} <= set(c.metadata) for c in chunks)


def test_chunk_size_is_configurable(sample_pdf, monkeypatch):
    pages = PdfLoader(sample_pdf).load()
    default = len(split(pages))
    monkeypatch.setenv("QABOT_CHUNK_SIZE", "250")
    monkeypatch.setenv("QABOT_CHUNK_OVERLAP", "40")
    assert len(split(pages)) > default


def test_file_id_depends_on_content_not_name(sample_pdf, tmp_path):
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(sample_pdf.read_bytes())
    assert file_id(copy) == file_id(sample_pdf)
