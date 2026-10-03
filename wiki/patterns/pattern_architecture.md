# Pattern: Architectural Modularity & Clean Interfaces

- **Type**: Strategy / Convention
- **Status**: Active

## 💡 Summary
All functionality must be encapsulated into modular capability packs or distinct core modules with minimal inter-module coupling.

## 🛠 Recommended Practice
1. Keep core orchestrator lightweight and single-agent.
2. Delegate external integrations to dynamically discovered packs.
3. Enforce strict type annotations and docstrings.
