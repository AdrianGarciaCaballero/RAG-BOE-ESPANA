# Contributing

Thank you for helping improve RAG-BOE-ESPANA. Adrian Garcia Caballero
(`@AdrianGarciaCaballero`) is the primary maintainer.

## Before opening a change

1. Search existing issues and discussions.
2. Open an issue for substantial behavior, architecture, data, or security
   changes before starting implementation.
3. Do not commit secrets, real HR records, private documents, generated vector
   databases, model weights, or unlicensed content.
4. Keep pull requests focused and explain how the change was tested.

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Pull the local Ollama models documented in the README, ingest the sample data,
and start the API and frontend in separate terminals.

## Checks

Run the lightweight checks used by CI:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

Changes to retrieval or generation should also report the relevant Hit Rate,
MRR, faithfulness, or answer-relevancy results and describe the evaluation set.

## Pull requests

- Link the issue the change addresses.
- Add or update tests for behavior that can be exercised without model downloads.
- Update the README when commands, configuration, or architecture change.
- Confirm that sample data is synthetic and third-party content is attributed.
- Follow the Code of Conduct and report vulnerabilities through SECURITY.md.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
