import fitz  # this comes from PyMuPDF, not the broken 'fitz' module
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text
