# Pattern: Common Pitfalls & Workarounds

- **Type**: Failure Mode & Workaround
- **Status**: Active

## 💡 Identified Failure Modes
1. **Context Window Exhaustion**: Overloading prompts with full codebase dumps.
   - *Workaround*: Use progressive disclosure via Wiki summary links and targeted file reads.
2. **Secret Leakage**: Committing machine paths or API tokens.
   - *Workaround*: Enforce `mimi-sdlc sanitize --check` prior to commit.
3. **WSL2 Terminal Corruption with Chrome**:
   - *Workaround*: Use `--wsl` flag for browser auth (`nlm login --wsl`).
