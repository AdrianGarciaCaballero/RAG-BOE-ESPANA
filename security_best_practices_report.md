# Security best-practices report

## Executive summary

The repository was reviewed as a local Python 3.11 application using FastAPI,
Streamlit, ChromaDB, Ollama, and multipart PDF uploads. Three high-impact file and
authorization issues were fixed on the `codex/oss-readiness` branch. The demo is
now materially safer for local use, but it is not a production service: network
deployment still requires complete user authentication, authorization, TLS,
rate limits, host/origin controls, monitoring, and a privacy review.

## High severity

### SEC-001: User-controlled paths in upload and deletion

- Rule ID: FASTAPI-FILES-001 / FASTAPI-UPLOAD-001
- Status: Fixed
- Location: `src/api/main.py`, `ingest_document` and `delete_document`;
  `src/api/security.py`, `validate_pdf_filename` and `document_path`
- Evidence: the previous implementation joined `file.filename` or the query
  parameter directly to `docs/`. The new implementation validates a simple PDF
  basename and confirms the resolved path remains inside the document directory.
- Impact: a crafted filename could write or delete files outside `docs/` with the
  permissions of the API process.
- Fix: allowlisted names and extensions, canonical path checks, exclusive file
  creation, and regression tests.
- Mitigation: run the API as an unprivileged user with a dedicated data volume.

### SEC-002: Administrative operations had no authentication

- Rule ID: FASTAPI-AUTH-001
- Status: Fixed for document upload and deletion
- Location: `src/api/main.py`, `require_admin_token`, `/ingest`, `/documents`
- Evidence: both state-changing routes now depend on an HTTP Bearer token read
  from `RAG_ADMIN_API_KEY` and fail closed when it is not configured.
- Impact: any network client could previously ingest arbitrary content or delete
  indexed documents.
- Fix: centralized dependency and constant-time token comparison.
- Mitigation: keep the service bound to localhost; use full identity and
  role-based authorization before any multi-user or network deployment.

### SEC-003: HR and health data workflow was unsafe for real records

- Rule ID: FASTAPI-AUTHZ-001 / FASTAPI-RESP-001
- Status: Demo scope corrected; production use remains unsupported
- Location: `src/api/main.py`, `SECURITY_DIRECTIVE`; `SECURITY.md`;
  `DATA_SOURCES.md`
- Evidence: the prior system prompt explicitly allowed disclosure of confidential
  payroll, absence, and health information when retrieved. Repository CSVs are
  now documented and treated as synthetic fixtures only.
- Impact: replacing fixtures with real records would expose special-category and
  employment data through unauthenticated chat routes.
- Fix: remove the disclosure instruction and state the synthetic-only boundary.
- Mitigation: do not load real records. A real deployment needs authentication,
  per-record authorization, encryption, audit logs, retention controls, and an
  independent privacy and legal review.

## Medium severity

### SEC-004: Uploads lacked content and size validation

- Rule ID: FASTAPI-LIMITS-001 / FASTAPI-UPLOAD-001
- Status: Fixed at application level
- Location: `src/api/main.py`, `ingest_document`; `src/api/security.py`
- Evidence: uploads are streamed in 64 KiB chunks, limited to 25 MiB, checked for
  a PDF signature, and removed after failed processing.
- Impact: unbounded or non-PDF uploads could consume disk, memory, and expensive
  model-processing resources.
- Fix: size, name, extension, and signature checks with generic errors.
- Mitigation: production still needs proxy-level body limits, rate limiting,
  malware scanning, and isolated document processing.

### SEC-005: Internal exception details were returned to clients

- Rule ID: FASTAPI-DEPLOY-002 / FASTAPI-RESP-001
- Status: Fixed in reviewed routes
- Location: `src/api/main.py`, document routes and streaming generator
- Evidence: operational details are now logged server-side while clients receive
  generic error messages.
- Impact: dependency, path, and runtime information could aid further attacks.
- Fix: `logger.exception` plus stable public responses.

## Low severity and deployment gaps

### SEC-006: Production perimeter controls are not implemented in-app

- Rule ID: FASTAPI-HOST-001 / FASTAPI-HEADERS-001 / FASTAPI-LIMITS-001
- Status: Accepted for a localhost research demo
- Location: `src/api/main.py`; `SECURITY.md`
- Evidence: the default host is now `127.0.0.1`, but there is no trusted-host
  middleware, global rate limiter, or production reverse-proxy configuration.
- Impact: exposing the demo directly would leave it vulnerable to abuse and
  resource exhaustion.
- Fix before deployment: add a reviewed production profile with complete auth,
  trusted hosts, strict CORS where needed, security headers, TLS termination,
  rate limits, body limits, timeouts, and monitoring.

### SEC-007: Large third-party document corpus increases supply-chain risk

- Rule ID: project licensing and data provenance
- Status: Documented; periodic review required
- Location: `DATA_SOURCES.md`, `NOTICE`, `README.md`
- Evidence: BOE documents retain their own terms and are expressly excluded from
  Apache-2.0. The repository cites the current AEBOE reuse notice.
- Impact: treating the whole repository as Apache-2.0 would misstate permissions
  and weaken attribution.
- Fix: separated code and data licensing with source attribution and disclaimers.
