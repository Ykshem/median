# CLAUDE.md

All project instructions live in AGENTS.md so every coding agent shares one source of truth.

@AGENTS.md

## Claude Code specifics

- Start each task by reading `ARCHITECTURE.md` and the relevant section of `docs/ROADMAP.md`.
- For multi-file changes, propose a short plan first and wait for approval.
- Use `uv run` for every Python command; don't use a global `python` or `pip`.
- Don't run `uv run python -m pipeline.build` without `--sample` unless the user asks: the full
  build costs about $5 in API calls.
- If you change a prompt in `median/prompts/`, also update `docs/PROMPTS.md`.
