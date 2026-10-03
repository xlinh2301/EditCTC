# 🤖 SDLC Guidelines & Agent Principles (AGENTS.md)

> [!IMPORTANT]
> **MANDATORY CONSTITUTION**:
> 1. All agents operating within this repository must read and adhere to `AGENTS.md`.
> 2. Every task instruction should reference these guidelines.

---

## 1. Core Principles

1. **Spec-Driven Execution (SDD)**:
   - Complex or multi-file changes must have a corresponding specification in `.ai/sub-specs/`.
   - Specifications should clearly define: Objectives, Architecture Decisions, Affected Symbols, Risk Levels, and Acceptance Criteria.
2. **Test-Driven Development (TDD)**:
   - Accompany all functional changes with automated unit and integration tests.
   - Run verification suites to ensure zero regression before committing.
3. **Capability Packs**:
   - Leverage installed capability packs (e.g. `abide`, `sandbox`, `pr-lens`, `token-saver`, `jira`, `compute-node`) for domain-specific automation and optimization.
4. **Data Hygiene & Security (Zero-Leak Policy)**:
   - Run `mimi-sdlc sanitize --check` prior to finalizing changes.
   - Never commit API keys, personal credentials, private tokens, or hardcoded machine paths.
5. **Git Discipline**:
   - Write clean, semantic Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).
   - Avoid AI attribution tags or unnecessary commit metadata.

## 📦 Enabled Capability Packs

Before starting project work, run `mimi-sdlc pack list`, identify the enabled packs, and use its reported `project_root` to read applicable instructions under `<project_root>/.agents/packs/<id>/rules/`. Follow those rules for the work. Do not install or enable packs, and do not change global runtime configuration as part of pack discovery.
