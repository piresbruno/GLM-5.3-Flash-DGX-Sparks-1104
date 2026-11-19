# Branch: refract-setup

## Business context

The recipe is worked on by coding agents on several machines (a devbox and rank 0's Spark), each starting from no
shared instructions. Refract gives every agent the same entry point (`AGENTS.md`), the same workflow skills and a
deterministic validator, so a session on any machine follows one process.

## What changed

- Refract v0.3.1 retrofitted with `--keep-existing` for Claude Code and omp: 58 new files plus `.refract/manifest.json`.
  No existing file was changed.
- `AGENTS.md` (with `CLAUDE.md` and `.omp/RULES.md` pointing to it) and 13 workflow skills in `.claude/skills/` and
  `.agents/skills/`.
- Process docs in `docs/` (`README.md`, `ARCHITECTURE.md`, `DEVELOPMENT_PROCESS.md`, `DEPLOYMENT.md`, `RELEASING.md`,
  `adr/`, `branches/` and others) and `tasks/task-template.md`.
- Opt-in features: hooks (git pre-commit and agent end-of-turn checks), GitHub docs-check workflow (pull requests and
  every Monday 09:00 UTC) with a PR template, MCP (context7) and a read-only architecture-reviewer subagent.
  Permission policies are off.

## Business rules and constraints

- Project facts: test `bash scripts/tests/run.sh`, run `./start.sh`, no build command (the start builds the image).
- The pre-commit hook runs Refract's validator, its docs check, gitleaks when installed, and the test command. It is
  pinned by hash: changing `.githooks/pre-commit`, `.refract/bin/*` or `.refract/config.json` blocks commits until
  `./setup-hooks.sh` is re-run.
- Generated files are owned by Refract (hashes in `.refract/manifest.json`); edit project-owned files instead, or
  re-run Refract.

## Edge cases handled

- The existing `docs/` and `.github/` content is untouched; Refract's files sit beside it.
- `docs/DEPLOYMENT.md` stays inactive (not `CONFIGURED`), so the deploy skills refuse to run until it is filled in.

## Integration impact

- Contributors who want the pre-commit hook run `./setup-hooks.sh` once per clone.
- The docs-check workflow runs on this fork's pull requests.
- `.mcp.json` makes Claude Code offer the context7 server (a third-party documentation service) in this project.

## Out of scope and follow-ups

- `docs/DEPLOYMENT.md` is not filled in; this recipe has no deploy pipeline beyond `./start.sh`.
- `AGENTS.md` lists languages and components as "Not detected" (Refract's inspector stopped at its file limit).
- `tp3-perf` picks this up when `main` is merged into it.

## Verification

- `refract retrofit --dry-run`: 58 CREATE plus the manifest, no conflicts, no warnings; post-apply validation passed.
- `.refract/bin/validate`: passed. `bash scripts/tests/run.sh`: passed on `main`'s tree.
- Scan of the new files for addresses, host names, users and credentials: none found.

## Changelog

- chore(refract): set up Refract v0.3.1 for Claude Code and omp, with hooks, GitHub checks, MCP and agents.
