from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf):
    pdfreader = PdfReader(pdf)
    text = ""
    for page in pdfreader.pages:
        text += page.extract_text()
    return text
