import fitz  # PyMuPDF
from docx import Document
import email
import os
import logging
import requests
import tempfile

logging.basicConfig(level=logging.INFO)

def download_file(url: str) -> str:
    """Download remote file into a temp path and return the local path."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        suffix = os.path.splitext(url)[-1].split("?")[0]  # handles ? in URL
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(response.content)
        tmp_file.close()

        logging.info(f"Downloaded file to temp: {tmp_file.name}")
        return tmp_file.name
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        raise

def clean_text(text: str) -> str:
    """Basic cleanup: strip blank lines; drop lines starting with 'page'."""
    lines = text.splitlines()
    filtered = [line.strip() for line in lines if line.strip() and not line.lower().startswith("page")]
    return "\n".join(filtered)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from PDF using PyMuPDF, concatenating page text."""
    try:
        texts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                texts.append(page.get_text("text"))
        return clean_text("\n".join(texts)).strip()
    except Exception as e:
        logging.error(f"Error extracting PDF text: {e}")
        raise

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return clean_text("\n".join([para.text for para in doc.paragraphs])).strip()
    except Exception as e:
        logging.error(f"Error extracting DOCX text: {e}")
        raise

def extract_text_from_email(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            msg = email.message_from_file(f)

        parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    parts.append(part.get_payload(decode=True).decode(errors="ignore"))
                except Exception:
                    parts.append(part.get_payload())

        return clean_text("\n".join(parts)).strip()
    except Exception as e:
        logging.error(f"Error extracting email text: {e}")
        raise

def extract_text(file_path_or_url: str) -> str:
    """
    Router: accepts a local path or URL; returns cleaned plain text.
    """
    # URL → download to temp
    if file_path_or_url.startswith(("http://", "https://")):
        file_path = download_file(file_path_or_url)
    else:
        file_path = file_path_or_url

    ext = os.path.splitext(file_path)[1].lower()
    logging.info(f"Extracting text from: {file_path} (type: {ext})")

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".eml":
        return extract_text_from_email(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


