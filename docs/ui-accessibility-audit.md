# Streamlit UI accessibility and recovery audit

Issue: [#15](https://github.com/AdrianGarciaCaballero/RAG-BOE-ESPANA/issues/15)

## Scope and method

Reviewed the Streamlit interface at a 1440 × 1000 viewport on Windows 11 with
Chrome 153 and Streamlit 1.64.0. The frontend was run against an unavailable
local API endpoint, then exercised with Streamlit `AppTest` and stubbed API
responses. No production documents or model data were used.

Keyboard focus was advanced through the first 14 interactive elements. The
sequence moved from sidebar controls into chat controls without a focus trap.
The Chat, tone selector, theme toggle, uploaders, and document actions have
visible labels. The session deletion button had only a trash icon and its
rendered button had an empty `aria-label`; the `help` text did not name it.

## Findings and changes

1. **Dark-mode sidebar text had insufficient contrast.** In Chrome, the tone
   options and PDF uploader label rendered as `rgb(49, 51, 63)` on the sidebar
   surface `rgb(38, 39, 48)`. Secondary buttons rendered nearly white text on
   a nearly white background. Dark-mode styles now set sidebar labels to a
   light foreground and secondary buttons to a dark surface.
2. **Session deletion was not clearly named.** The button now says “🗑️ Eliminar chat”
   so its visible and accessible name describes the action.
3. **The empty conversation looked unfinished.** An explicit empty-state message
   now invites the user to enter a question. The existing empty document warning
   remains visible in the management tab.
4. **API failures exposed connection details and chat failures looked like
   assistant answers.** Connection errors now show a short recovery instruction.
   A failed streamed response is not saved as an assistant message and can be
   retried without duplicating the user's question.
5. **Requests could block without a deadline and loading was unclear.** API
   calls now have connect/read timeouts; the document tab shows a loading
   spinner, and chat shows a status while a response is being generated.

## Verification and limits

Four `AppTest` cases cover empty states, safe connection errors, retrying a
failed chat request, and keeping retries within the originating chat. They use
a stub API and do not require Ollama, ChromaDB, model downloads, or real
documents. Browser inspection confirmed the contrast changes and that Chrome's
accessibility tree names the delete button “🗑️ Eliminar chat”. A physical screen
reader and a live Ollama/ChromaDB-backed session were not available, so those
combinations remain unverified.

Validation passed: `python -m compileall -q src tests` and
`python -m unittest discover -s tests -v` (9 tests).
