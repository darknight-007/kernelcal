# Change Requests

This directory holds proposed and accepted change requests (CRs) for the
`kernelcal` package.  CRs are the canonical place to record substantial
multi-PR proposals, sequencing decisions, and out-of-scope deferrals.

## Conventions

* **Filename**: `YYYY-MM-DD-short-slug.md`.
* **Front matter**: every CR opens with a small metadata table (`Field`
  / `Value`) including `CR ID`, `Date`, `Author`, `Status`, `Target
  package`, `Estimated effort`, `Reviewers`.
* **Status lifecycle**: `Proposed` -> `Accepted (with revisions)` ->
  `In progress` -> `Implemented` (or `Withdrawn` / `Superseded by ...`).
* **Revisions to an accepted CR**: append a top-level addendum block
  explaining what changed and why; do not silently rewrite history.
* **Cross-references**: link to Field Notes, prior CRs, and shipped PRs
  in the body; the `Cross-references` section at the end is the
  canonical inventory.

## Index

| CR ID | Date | Status | Title |
|---|---|---|---|
| [2026-04-26-integration-spine-and-bookkeeping](./2026-04-26-integration-spine-and-bookkeeping.md) | 2026-04-26 | Accepted (with revisions) | Integration Spine and Bookkeeping |
