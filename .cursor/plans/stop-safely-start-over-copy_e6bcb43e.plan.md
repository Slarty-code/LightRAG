---
name: stop-safely-start-over-copy
overview: Align ingestion control terminology with actual behavior by replacing Pause/Resume phrasing with Stop Safely/Start Over and clarifying that restart reprocesses from the beginning.
todos:
  - id: update-ui-copy
    content: Update PipelineStatusDialog button and modal labels to Stop Safely / Start Over with explicit restart semantics
    status: completed
  - id: update-localization
    content: Revise related i18n keys in en.json and remove ambiguous pause/resume phrasing from user-visible strings
    status: completed
  - id: align-docs
    content: Update API docs to describe cooperative stop and restart-from-beginning behavior
    status: completed
  - id: verify-flow
    content: Run frontend lint/tests and perform manual UI flow check for control text and confirmations
    status: completed
isProject: false
---

# Plan: Stop Safely / Start Over terminology update

## Goal
Make ingestion controls unambiguous so users do not infer checkpoint resume behavior.

## Scope
- Update control labels and confirmation dialogs from pause/resume wording to `Stop Safely` / `Start Over`.
- Add helper/tooltip text stating resumed work reprocesses from the beginning.
- Ensure API/docs language matches UI semantics.

## Files to update
- [lightrag_webui/src/components/documents/PipelineStatusDialog.tsx](lightrag_webui/src/components/documents/PipelineStatusDialog.tsx)
- [lightrag_webui/src/locales/en.json](lightrag_webui/src/locales/en.json)
- [lightrag_webui/src/api/lightrag.ts](lightrag_webui/src/api/lightrag.ts) (only naming/comments/types if needed for clarity)
- [docs/LightRAG-API-Server.md](docs/LightRAG-API-Server.md)

## Planned changes
- Replace visible button copy:
  - Running state: `Stop Safely`
  - Post-stop state: `Start Over`
- Update modal copy:
  - Stop: cooperative stop after current in-flight step
  - Start Over: explicitly says processing restarts from beginning
- Add concise helper/tooltip text near Start Over:
  - `Start Over reprocesses documents from the beginning.`
- Keep backend endpoints unchanged (`/api/ingestion/pause/{job_id}`, `/api/ingestion/resume/{job_id}`), but adjust UI/docs wording to describe semantics rather than endpoint names.

## Validation
- Frontend lint/tests for updated strings and dialog flows.
- Manual smoke check:
  1. Active pipeline shows `Stop Safely`
  2. After stop, action shows `Start Over`
  3. Confirmation dialog and helper text clearly state restart-from-beginning behavior
- Docs reflect the same behavior language.

## Risks and mitigations
- Risk: wording drift between UI and docs.
- Mitigation: update both in same change and verify with one pass of text review for `pause/resume` user-facing strings.