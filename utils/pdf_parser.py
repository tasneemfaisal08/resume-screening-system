import pdfplumber


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts all text from a PDF file and returns it as a single string.

    Improvements over the original:
    - Catches per-page errors so one bad page doesn't crash the whole file.
    - Returns an empty string (instead of crashing) if the file can't be opened.
    """
    text = ""

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_err:
                    # Skip the bad page and keep going
                    print(f"[pdf_parser] Warning: Could not read page {i + 1} — {page_err}")
                    continue

    except Exception as e:
        print(f"[pdf_parser] Error opening PDF: {e}")
        return ""

    return text.strip()
