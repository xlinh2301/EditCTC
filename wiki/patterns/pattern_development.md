# Pattern: Spec-Driven Verification Workflow

- **Type**: Strategy / Workflow
- **Status**: Active

## 💡 Summary
Follow the Red-Green-Refactor cycle driven by Sub-Specs:
1. Scaffold spec: `mimi-sdlc spec new <slug>`
2. Define acceptance criteria and verification tests.
3. Implement minimal code changes.
4. Run validation: `pytest` and `mimi-sdlc sanitize --check`.
5. Document learnings into `wiki/logs.md`.
