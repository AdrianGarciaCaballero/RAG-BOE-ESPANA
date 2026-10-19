# Accessibility and Error-Recovery Audit (Streamlit UI)

Scope: `src/frontend/frontend.py`

## Findings

1. **Bare exception handling** in `load_history()` could mask unexpected errors and
   made failures hard to diagnose.
2. **Chat history persistence** did not surface write failures to the user.
3. **Network timeouts were missing** for the streaming chat request and the admin
   document-list request, so a stalled backend could hang the UI indefinitely.
4. **Empty chat session state** had no first-run guidance, leaving an empty panel.
5. **Icon-only controls** (delete session) rely on tooltips, which is acceptable for
   sighted users but should keep `help` text up to date for assistive technology.

## Implemented fixes

- `load_history()` now catches the specific exceptions it can recover from.
- `save_history()` reports failures with `st.error` instead of failing silently.
- Added a timeout to the streaming chat request.
- Added a timeout to the admin document-list request.
- Added an informational empty state for chat sessions with no messages.

## Not yet implemented (follow-ups)

- Live-region announcements for dynamic updates.
- Keyboard focus order audit beyond Streamlit defaults.
- Retry buttons for failed network calls.
