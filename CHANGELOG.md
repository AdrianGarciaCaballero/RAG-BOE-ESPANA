# Changelog

All notable changes to this project will be documented in this file. The format
is based on Keep a Changelog, and the project follows Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-09-20

### Added

- Apache-2.0 licensing for the original project code and documentation.
- Explicit BOE document attribution and synthetic-data boundaries.
- Contributor, security, support, and community health documentation.
- Lightweight CI, Dependabot configuration, issue forms, and pull request checks.
- Unit tests for upload filename, path, signature, and token validation.
- Bearer authentication for document ingestion and deletion.

### Changed

- Bind the API to localhost by default.
- Stream uploads with a 25 MiB limit and reject invalid PDF headers.
- Return generic client errors while retaining server-side diagnostics.
- Make API endpoints and administrative credentials configurable through the
  environment.

### Removed

- Tracked macOS metadata files.
- Instructions that treated confidential employee data as disclosable.

[Unreleased]: https://github.com/AdrianGarciaCaballero/RAG-BOE-ESPANA/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AdrianGarciaCaballero/RAG-BOE-ESPANA/releases/tag/v0.1.0
