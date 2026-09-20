import tempfile
import unittest
from pathlib import Path

from src.api.security import document_path, has_pdf_signature, token_matches, validate_pdf_filename


class FileSecurityTests(unittest.TestCase):
    def test_accepts_simple_pdf_name(self):
        self.assertEqual(validate_pdf_filename("convenio-2026.pdf"), "convenio-2026.pdf")

    def test_rejects_path_traversal_and_wrong_extensions(self):
        for filename in ("../secret.pdf", "folder/file.pdf", "notes.txt", "", None):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    validate_pdf_filename(filename)

    def test_resolved_document_stays_in_base_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            self.assertEqual(document_path(base, "boe.pdf"), base.resolve() / "boe.pdf")

    def test_checks_pdf_signature(self):
        self.assertTrue(has_pdf_signature(b"%PDF-1.7\n"))
        self.assertFalse(has_pdf_signature(b"<html>not a pdf"))

    def test_compares_admin_tokens(self):
        self.assertTrue(token_matches("secret", "secret"))
        self.assertFalse(token_matches("wrong", "secret"))
        self.assertFalse(token_matches(None, "secret"))


if __name__ == "__main__":
    unittest.main()
