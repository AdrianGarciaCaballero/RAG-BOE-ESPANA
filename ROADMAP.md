# Roadmap

This roadmap lists scoped contributions that can be opened as GitHub issues. It
is a planning document, not a commitment to a delivery date.

## Near term

1. **Reproducible dependency lock**
   Add a reviewed lock file or constraints workflow, document model versions,
   and verify installation on Linux and macOS.
2. **End-to-end API tests**
   Exercise authentication, PDF upload limits, duplicate handling, deletion,
   and streaming errors with dependency fakes.
3. **Document provenance manifest**
   Record source URL, retrieval date, checksum, document date, and reuse terms
   for each third-party document.
4. **Evaluation expansion**
   Grow the golden dataset, separate retrieval and generation regressions, and
   publish confidence intervals alongside aggregate metrics.
5. **Production deployment profile**
   Add a reverse-proxy example with TLS, trusted hosts, strict CORS, request
   limits, rate limits, timeouts, and structured logs.

## Later

6. **Incremental ingestion**
   Track content hashes and replace changed documents without rebuilding the
   complete index.
7. **Citation traceability**
   Return stable document identifiers, page numbers, and excerpts for every
   factual answer.
8. **Accessibility and Spanish UX review**
   Audit keyboard navigation, contrast, screen-reader labels, empty states, and
   error recovery in the Streamlit interface.

Contributors are welcome to propose an implementation in an issue before
starting larger changes.
