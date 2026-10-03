# Spec-Driven Development & Capability Pack Guidelines

You are operating under the **Spec-Driven Development (SDD)** framework powered by modular Capability Packs.

## 📦 Enabled Capability Packs

Before starting project work, run `mimi-sdlc pack list`, identify the enabled packs, and use its reported `project_root` to read applicable instructions under `<project_root>/.agents/packs/<id>/rules/`. Follow those rules for the work. Do not install or enable packs, and do not change global runtime configuration as part of pack discovery.


## 🎯 Core Operating Principles

1. **Spec-Driven Execution**:
   - For non-trivial features, ensure a Sub-Spec exists in `.ai/sub-specs/` describing requirements, design decisions, and acceptance criteria.
   - Use `mimi-sdlc spec new <slug>` to scaffold new specifications.
2. **Test-Driven Verification**:
   - Write tests alongside implementations to verify behavioral correctness.
   - Ensure local test suites pass before concluding tasks.
3. **Capability Packs**:
   - Utilize installed capability packs (e.g. `abide`, `sandbox`, `pr-lens`, `token-saver`, `jira`, `compute-node`) to execute specialized operations safely.
4. **Data Hygiene & Security**:
   - Run `mimi-sdlc sanitize --check` to prevent committing secrets, access tokens, or private workstation paths.
   - Follow standard Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).

Reference the AGENTS.md document for complete guidelines.
