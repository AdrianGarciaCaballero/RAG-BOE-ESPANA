import re
import secrets
from pathlib import Path


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
_SAFE_PDF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}\.pdf$", re.IGNORECASE)


def validate_pdf_filename(filename: str | None) -> str:
    """Return a safe PDF basename or raise ValueError."""
    if not filename or filename != Path(filename).name:
        raise ValueError("El nombre del archivo no es valido.")
    if not _SAFE_PDF_NAME.fullmatch(filename):
        raise ValueError("Solo se permiten nombres PDF simples de hasta 124 caracteres.")
    return filename


def document_path(base_dir: Path, filename: str | None) -> Path:
    safe_name = validate_pdf_filename(filename)
    base = base_dir.resolve()
    candidate = (base / safe_name).resolve()
    if candidate.parent != base:
        raise ValueError("La ruta del documento no es valida.")
    return candidate


def has_pdf_signature(header: bytes) -> bool:
    return header.startswith(PDF_MAGIC)


def token_matches(provided: str | None, expected: str | None) -> bool:
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided, expected)
