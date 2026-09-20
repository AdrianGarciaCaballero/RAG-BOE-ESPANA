# Security policy

## Supported versions

Security fixes are applied to the latest commit on `main`. No older release line
is currently supported.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for this repository.
Do not disclose a vulnerability in a public issue before a fix is available.

Include the affected revision, reproduction steps, impact, and any suggested
mitigation. The maintainer will acknowledge a complete report within seven days
and provide a status update within fourteen days when possible.

## Security boundaries

- The bundled HR CSVs are synthetic fixtures. Never ingest real HR or medical
  data without authentication, authorization, encryption, retention controls,
  audit logging, and an independent privacy review.
- The demo binds to `127.0.0.1` by default. Exposing it to a network requires a
  production reverse proxy, TLS, strict host/origin configuration, rate limits,
  request-size limits, and authentication for every sensitive route.
- Document upload and deletion require `RAG_ADMIN_API_KEY`. Keep this value out
  of source control and send it only in the `Authorization: Bearer` header.
- Uploaded PDFs are limited to 25 MiB and receive basic filename and signature
  checks. Production deployments should add malware scanning and isolation.
- Model output is untrusted. Do not use it as legal advice or as the sole basis
  for employment, medical, financial, or legal decisions.

## Maintainer checklist

- Keep FastAPI, Starlette, Uvicorn, and `python-multipart` patched.
- Review Dependabot alerts and CI failures promptly.
- Rotate any credential suspected of exposure.
- Remove sensitive material from the working tree and Git history before making
  a repository public.
