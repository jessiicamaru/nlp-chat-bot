---
name: gh-pr-create
description: Open a GitHub pull request for the current branch in this repo - work out the base branch, write the description from the project template, link the issue it closes, apply type/area/risk/size labels and badges, and request reviewers. Use when the user asks to create, open, raise or draft a PR, or says they are done with a branch and want it reviewed.
---

# Create a pull request

Repo: `jessiicamaru/habit-tracker`. Needs `gh` or the GitHub MCP server — if neither
answers, run `gh-setup` instead of guessing.

## Invocation

`/gh-pr-create [flags]` — with no flags: infer everything, show the assembled
description, and **ask once** before creating.

| Flag | Effect |
| --- | --- |
| `--draft` | Open as a draft. |
| `--wip` | `--draft` plus a `WIP: ` title prefix. |
| `--base <branch>` | Override the computed base branch. |
| `--title "<text>"` | Use this title verbatim instead of deriving one. |
| `--issue <n>` | Link this issue (repeatable). Implies `Closes #n`. |
| `--relates <n>` | Link without closing (`Refs #n`). |
| `--no-issue` | Do not search for or link any issue. |
| `--reviewer <user\|team>` | Request this reviewer (repeatable). |
| `--no-reviewer` | Skip the reviewer step. |
| `--label <label>` | Add this label (repeatable), on top of the inferred ones. |
| `--no-label` | Apply no labels at all. |
| `--no-badge` | Omit the shields.io badge row from the description. |
| `--template <feature\|bugfix\|refactor\|docs\|chore>` | Force a template variant instead of inferring it. |
| `--no-push` | Assume the branch is already pushed. |
| `--dry-run` | Print title, body, labels, reviewers and the exact `gh` commands. Create nothing. |
| `--yes` | Skip the confirmation prompt (for non-interactive runs). |
| `--web` | Open the PR in a browser once created. |

`--dry-run` is the safe way to preview. Prefer it whenever the user seems unsure.

## Steps

### 1. Preconditions

```bash
git branch --show-current          # must not be main
git status --porcelain             # must be clean
gh auth status
```

Refuse to continue on `main`, and refuse with uncommitted changes — offer to commit
first rather than silently including or excluding them. Never commit without asking:
this project's rules forbid unprompted git writes.

### 2. Base branch

Derive, then state what you picked and why:

- `--base` given → use it.
- Branch matches `<type>/<story>/<task>` → base is `<type>/<story>`, if that branch exists on the remote.
- Otherwise → the repo default branch (`gh repo view --json defaultBranchRef`), which is `main` here.

Confirm the base exists remotely before using it.

### 3. Scope

```bash
git fetch origin
git log --oneline origin/<base>..HEAD
git diff --stat origin/<base>...HEAD
```

If this is empty, there is nothing to open a PR for — say so and stop.

### 4. The issue it closes

Skip entirely on `--no-issue`.

```bash
gh issue list --state open --limit 30 --json number,title,labels,assignees
```

Match on: a number in the branch name (`feat/42-foo` → #42), then on title/keyword
overlap with the commits. Propose the best match and **let the user confirm** — do not
link an issue on a weak guess. `--issue N` skips the search.

Use `Closes #N` when the PR fully resolves it, `Refs #N` when it only relates.

### 5. Title

Conventional Commits, matching this repo's history
(`feat(backend):`, `fix(calendar):`, `refactor:`, `docs:`, `chore(agents):`):

- One commit → reuse its subject.
- Several → synthesise one line covering the whole change. Scope from the area touched
  (`backend`, `frontend`, `widget`, `calendar`, `sync`, `db`).
- Imperative mood, no trailing period, ≤ 72 chars.

### 6. Description

Read `templates/pr-description.md` and fill it in. Pick the variant from the branch
prefix unless `--template` says otherwise:

| Branch prefix | Template variant |
| --- | --- |
| `feat/`, `feature/` | feature |
| `fix/`, `bugfix/`, `hotfix/` | bugfix |
| `refactor/`, `perf/` | refactor |
| `docs/` | docs |
| `chore/`, `ci/`, `build/` | chore |

Fill every section from real evidence — the diff, the commit messages, the test output
you actually ran. **Delete sections that do not apply rather than writing "N/A".** Never
tick a checklist box you have not verified; leave it unticked and say so in the PR.

### 7. Labels and badge

Infer, then show the user the set before applying:

- **type** — exactly one, from the Conventional Commit prefix of the title.
- **area** — one or more: `server/` → `area: backend`; `apps/lib` → `area: frontend`;
  `apps/android/**/widget/` → `area: widget`; any Google-Calendar path → `area: sync`;
  `Migrations/` or `ApplicationDbContext` → `area: db`.
- **risk** — `risk: migration` if the diff adds a file under `server/src/Infrastructure/Migrations/`;
  `risk: security` if it touches auth, authorization, secrets or config;
  `risk: breaking` if a public endpoint, DTO or interface changed shape.
- **size** — from `git diff --shortstat origin/<base>...HEAD` (added+deleted):
  `XS` <50, `S` <200, `M` <600, `L` <1500, `XL` ≥1500.

If a label does not exist yet, run `gh-setup --labels` rather than creating ad-hoc
labels that fragment the taxonomy.

The badge row at the top of the description mirrors the labels — see the template.
Omit it with `--no-badge`.

### 8. Reviewers

Skip on `--no-reviewer`. With `--reviewer` given, use exactly those. Otherwise suggest
based on who last touched the changed files, and confirm before requesting:

```bash
git log --format="%an" -20 -- <changed paths> | sort | uniq -c | sort -rn
gh api repos/jessiicamaru/habit-tracker/collaborators --jq '.[].login'
```

Never request review from the PR author — GitHub rejects it and the whole command
fails. Drop the author from the list first.

### 9. Create

```bash
git push -u origin <branch>          # unless --no-push
gh pr create \
  --base <base> \
  --head <branch> \
  --title "<title>" \
  --body-file <tmp>.md \
  [--draft] \
  [--label "type: feat" --label "area: backend" ...] \
  [--reviewer <user> ...]
```

Write the body to a temp file — passing it inline mangles newlines and backticks on
Windows. Use the scratchpad directory, not the repo.

If `gh pr create` fails because a PR already exists, report the existing URL rather
than retrying.

### 10. Report

Give the user the URL, the applied labels, the requested reviewers, and the linked
issue. If you left any checklist box unticked, say which and why — that is the part a
reviewer most needs to know.

## Duplicate check

Before creating, look for an open PR covering the same ground:

```bash
gh pr list --state open --json number,title,headRefName,files
```

Warn on a same-base PR touching a majority of the same files. Do not block — say what
overlaps and let the user decide.
