---
name: gh-setup
description: One-time setup for GitHub access in this repo - install and authenticate the gh CLI, or connect the GitHub MCP server, and create the label taxonomy the PR skills rely on. Use when gh is missing or unauthenticated, when a GitHub MCP tool is unavailable, when "gh: command not found" appears, or when the pr-create / pr-review skills report they cannot reach GitHub.
---

# GitHub access setup

The `gh-pr-create` and `gh-pr-review` skills need a way to talk to
`github.com/jessiicamaru/habit-tracker`. There are two, and **either one is enough**.

## Invocation

`/gh-setup [flags]` — with no flags, run the diagnosis in *Check what works* below and
then walk only the steps that are actually missing.

| Flag | Effect |
| --- | --- |
| `--check` | Diagnose only. Report what works and what is missing; change nothing. |
| `--cli` | Set up Option A (`gh` CLI) only. |
| `--mcp` | Set up Option B (GitHub MCP server) only. |
| `--labels` | Skip access setup; only create the label taxonomy. |
| `--labels --dry-run` | Print the `gh label create` commands without running them. |
| `--repo <owner/name>` | Target a different repo than `jessiicamaru/habit-tracker`. |

Never run the label block without confirming first — it writes to the real repo.

Check what already works before changing anything:

```bash
gh auth status          # works -> Option A is done
```

If a `mcp__github__*` tool is listed in your available tools, Option B is already done.

---

## Option A - `gh` CLI (recommended, simplest)

The PR skills are written against `gh` because it needs no tokens in files and no extra
processes.

### Install

| OS | Command |
| --- | --- |
| Windows | `winget install --id GitHub.cli` |
| macOS | `brew install gh` |
| Debian/Ubuntu | `sudo apt install gh` |

This repo is developed on Windows with Git Bash. After `winget install`, **open a new
shell** — the existing one will not have `gh` on `PATH` yet. If `gh` still is not found,
it lives at `C:\Program Files\GitHub CLI\gh.exe`.

### Authenticate

```bash
gh auth login          # choose: GitHub.com -> HTTPS -> login with a web browser
gh auth status         # confirm
```

Ask for these scopes when prompted: `repo`, `read:org`. Adding a reviewer needs
`read:org` when the reviewer is a team rather than a person.

### Verify against this repo

```bash
gh repo view jessiicamaru/habit-tracker --json name,defaultBranchRef
gh pr list --limit 3
gh issue list --limit 3
```

All three must succeed before the PR skills will work.

---

## Option B - GitHub MCP server

Use this if you would rather not install a CLI, or you want GitHub tools available as
native tool calls.

### 1. Create a token

Create a fine-grained personal access token at
<https://github.com/settings/personal-access-tokens/new>, scoped to the
`habit-tracker` repository, with these **repository permissions**:

| Permission | Access | Needed for |
| --- | --- | --- |
| Contents | Read-only | reading the diff |
| Pull requests | Read and write | creating PRs, posting review comments |
| Issues | Read-only | linking issues (`read` is enough; use write only if you want the skill to close them) |
| Metadata | Read-only | always required |

Add `Members: Read-only` at the *organisation* level only if you need to request review
from a team.

### 2. Connect it

Project-scoped, so it is shared with anyone who clones the repo. Create `.mcp.json` in
the repo root:

```json
{
  "mcpServers": {
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "Bearer ${GITHUB_TOKEN}"
      }
    }
  }
}
```

Then export the token in your shell (or your OS keychain / environment):

```bash
export GITHUB_TOKEN=github_pat_...
```

**Never write the token into `.mcp.json` itself.** `${GITHUB_TOKEN}` is expanded at
load time, which keeps the secret out of git. If you must inline it for a quick test,
add `.mcp.json` to `.gitignore` first — this repo has already leaked credentials once
(see `docs/review-code-reports/`), so treat this as a hard rule.

Restart Claude Code, then confirm `mcp__github__*` tools appear.

### 3. If it will not connect

- `claude mcp list` shows configured servers and their status.
- A connection timeout usually means the token is missing or unexported — the header
  resolves to the literal `Bearer ${GITHUB_TOKEN}` and GitHub rejects it.
- A self-hosted alternative is the Docker image `ghcr.io/github/github-mcp-server`;
  use it only if the hosted endpoint is blocked on your network.

---

## Label taxonomy (run once per repo)

Both PR skills apply labels from a fixed set. Create it once — the command is
idempotent, so re-running it is safe:

```bash
# type - exactly one per PR, mirrors the Conventional Commit prefix
gh label create "type: feat"     --color 0E8A16 --description "New user-facing capability"      --force
gh label create "type: fix"      --color D73A4A --description "Bug fix"                          --force
gh label create "type: refactor" --color FBCA04 --description "Behaviour-preserving change"      --force
gh label create "type: docs"     --color 0075CA --description "Documentation only"               --force
gh label create "type: chore"    --color CFD3D7 --description "Tooling, CI, dependencies"        --force
gh label create "type: test"     --color BFD4F2 --description "Tests only"                       --force

# area - one or more, matches the commit scopes already used in this repo
gh label create "area: backend"  --color 5319E7 --description "server/ (.NET)"                   --force
gh label create "area: frontend" --color 1D76DB --description "apps/ (Flutter)"                  --force
gh label create "area: widget"   --color 006B75 --description "Android home screen widgets"      --force
gh label create "area: sync"     --color B60205 --description "Google Calendar sync path"        --force
gh label create "area: db"       --color 5319E7 --description "EF Core model or migrations"      --force

# risk - only when true; these are what a reviewer should look at first
gh label create "risk: migration" --color E99695 --description "Contains an EF migration"        --force
gh label create "risk: breaking"  --color B60205 --description "Breaking API or contract change" --force
gh label create "risk: security"  --color B60205 --description "Touches auth, authz or secrets"  --force

# size - set from the diff stat, see gh-pr-create
gh label create "size: XS" --color C2E0C6 --description "< 50 lines changed"    --force
gh label create "size: S"  --color C2E0C6 --description "50-200 lines"          --force
gh label create "size: M"  --color FEF2C0 --description "200-600 lines"         --force
gh label create "size: L"  --color F9D0C4 --description "600-1500 lines"        --force
gh label create "size: XL" --color E99695 --description "> 1500 lines - split it if you can" --force
```

Verify: `gh label list --limit 40`

---

## Which path a skill should take

`gh-pr-create` and `gh-pr-review` both express every step as a `gh` command. If only
the MCP server is available, use the equivalent `mcp__github__*` tool instead — the
mapping is one-to-one for everything the skills do (list/create/get PRs, list issues,
add labels, request reviewers, post review comments). If **neither** is available, stop
and point the user at this file rather than guessing.
