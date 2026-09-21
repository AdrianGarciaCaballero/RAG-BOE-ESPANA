import builtins
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, documents=None, chunks=()):
        self.documents = documents or []
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def json(self):
        return {"documents": self.documents}

    def raise_for_status(self):
        return None

    def iter_content(self, **_kwargs):
        return iter(self.chunks)


class FrontendAccessibilityTests(unittest.TestCase):
    def run_app(self):
        script_path = Path(__file__).resolve().parents[1] / "src/frontend/frontend.py"
        return AppTest.from_file(str(script_path)).run(timeout=20)

    def with_isolated_history(self):
        temporary_directory = tempfile.TemporaryDirectory()
        history_path = Path(temporary_directory.name) / "chat_history.json"
        real_open = builtins.open
        real_exists = os.path.exists

        def isolated_open(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and Path(file).name == "chat_history.json":
                file = history_path
            return real_open(file, *args, **kwargs)

        def isolated_exists(file):
            if isinstance(file, (str, os.PathLike)) and Path(file).name == "chat_history.json":
                return real_exists(history_path)
            return real_exists(file)

        patches = (
            patch("builtins.open", side_effect=isolated_open),
            patch("os.path.exists", side_effect=isolated_exists),
            patch.dict(os.environ, {"RAG_API_URL": "http://rag.test"}),
        )
        for mocked in patches:
            mocked.start()
        self.addCleanup(temporary_directory.cleanup)
        self.addCleanup(lambda: [mocked.stop() for mocked in reversed(patches)])

    def test_empty_conversation_and_document_collection_are_explained(self):
        self.with_isolated_history()
        with patch("requests.get", return_value=FakeResponse()):
            app = self.run_app()

        self.assertTrue(any("conversación está vacía" in item.value for item in app.info))
        self.assertTrue(any("No hay documentos indexados" in item.value for item in app.warning))
        self.assertIn("🗑️ Eliminar chat", [button.label for button in app.button])

    def test_api_connection_error_is_actionable_and_does_not_expose_details(self):
        self.with_isolated_history()
        with patch(
            "requests.get",
            side_effect=requests.ConnectionError("private-host:8000 secret detail"),
        ) as get_documents:
            app = self.run_app()
            self.assertEqual(get_documents.call_args.kwargs["timeout"], (5, 15))

        messages = " ".join(error.value for error in app.error)
        self.assertIn("Comprueba que la API esté en marcha", messages)
        self.assertNotIn("private-host", messages)
        self.assertNotIn("secret detail", messages)

    def test_chat_failure_can_be_retried_without_saving_an_error_as_an_answer(self):
        self.with_isolated_history()
        with (
            patch("requests.get", return_value=FakeResponse()),
            patch(
                "requests.post",
                side_effect=[
                    requests.ConnectionError("private-host:8000 secret detail"),
                    FakeResponse(chunks=("Respuesta recuperada",)),
                ],
            ) as post,
        ):
            app = self.run_app()
            app.chat_input[0].set_value("¿Qué dice el documento?").run(timeout=20)

            messages = app.session_state["sessions"][app.session_state["current_session"]]["messages"]
            self.assertEqual([message["role"] for message in messages], ["user"])
            self.assertTrue(any("Reintentar respuesta" == button.label for button in app.button))
            self.assertNotIn("secret detail", " ".join(error.value for error in app.error))

            next(button for button in app.button if button.label == "Reintentar respuesta").click().run(timeout=20)

        messages = app.session_state["sessions"][app.session_state["current_session"]]["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[-1]["content"], "Respuesta recuperada")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 120))

    def test_failed_prompt_retry_is_only_offered_in_its_session(self):
        self.with_isolated_history()
        question = "¿Qué dice el documento?"
        with (
            patch("requests.get", return_value=FakeResponse()),
            patch("requests.post", side_effect=requests.ConnectionError("offline")),
        ):
            app = self.run_app()
            app.chat_input[0].set_value(question).run(timeout=20)
            original_session = app.session_state["current_session"]
            original_title = app.session_state["sessions"][original_session]["title"]

            next(button for button in app.button if button.label == "+ Nuevo Chat").click().run(timeout=20)
            self.assertNotEqual(app.session_state["current_session"], original_session)
            self.assertNotIn("Reintentar respuesta", [button.label for button in app.button])

            next(button for button in app.button if original_title in button.label).click().run(timeout=20)
            self.assertEqual(app.session_state["current_session"], original_session)
            self.assertIn("Reintentar respuesta", [button.label for button in app.button])


if __name__ == "__main__":
    unittest.main()
