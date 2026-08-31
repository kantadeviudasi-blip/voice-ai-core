import io
from typing import Optional
import pypdf

class DocumentExtractor:
    """
    Extracts clean raw text from PDF files, plain text, or markdown documents.
    """
    @staticmethod
    def extract_from_pdf(pdf_bytes: bytes) -> str:
        text_content = []
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        except Exception as e:
            return f"[PDF Extraction Error: {str(e)}]"
        return "\n".join(text_content).strip()

    @staticmethod
    def extract_from_text(raw_text: str) -> str:
        return raw_text.strip()
