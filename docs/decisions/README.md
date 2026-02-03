# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records documenting significant design decisions for the AI Ethics in HR systematic review project.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision along with its context and consequences. ADRs help:
- Understand why decisions were made
- Onboard new contributors
- Avoid revisiting settled decisions
- Learn from past choices

## ADR Index

| ADR | Title | Date | Status |
|-----|-------|------|--------|
| [ADR-001](./ADR-001-initial-schema-design.md) | Initial Coding Schema Design (v1.0.0) | 2026-02-02 | Accepted |
| [ADR-002](./ADR-002-coding-schema-extension-v1.1.md) | Coding Schema Extension for Stance, Solutions, and Temporal Analysis | 2026-02-03 | Accepted |

## Creating a New ADR

1. Copy `template.md` to `ADR-XXX-title.md`
2. Fill in the sections
3. Update this README index
4. Commit with message: `docs(adr): Add ADR-XXX title`

## ADR Statuses

- **Proposed**: Under discussion
- **Accepted**: Decision made and implemented
- **Deprecated**: No longer relevant
- **Superseded**: Replaced by another ADR

## Template

See [template.md](./template.md) for the standard ADR format.

---

*Based on [MADR](https://adr.github.io/madr/) (Markdown Any Decision Records)*
