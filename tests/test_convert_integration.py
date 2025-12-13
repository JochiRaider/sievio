import io

import pytest


def test_extract_pdf_records_real_bytes():
    try:
        from pypdf import PdfWriter
    except Exception:
        pytest.skip("pypdf not available")

    try:
        from sievio.sources.pdfio import extract_pdf_records
    except Exception as exc:  # pragma: no cover - import issues are env-specific
        pytest.skip(f"pdf handler not available: {exc}")

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    data = buf.getvalue()

    records = extract_pdf_records(data, rel_path="sample.pdf")

    assert len(records) == 1
    rec = records[0]
    assert rec["meta"]["rel_path"] == "sample.pdf"
    assert rec["meta"]["file_bytes"] == len(data)
