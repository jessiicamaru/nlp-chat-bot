---
name: gh-pr-review
description: Fetch and review a GitHub pull request in this repo - pull the diff, review it through chosen lenses (security, bugs, quality, tests, contracts, docs), rank findings by confidence and impact, and optionally post inline comments or a verdict. Use when the user asks to review, look at, check or comment on a PR, gives a PR number or URL, or asks what changed in a PR.
---

# Review a pull request

Repo: `jessiicamaru/habit-tracker`. Needs `gh` or the GitHub MCP server — if neither
answers, run `gh-setup`.

## Invocation

`/gh-pr-review [target] [flags]` — `target` is a PR number, a PR URL, or nothing (use
the PR for the current branch). **Default behaviour is read-only**: report findings in
the terminal and post nothing.

| Flag | Effect |
| --- | --- |
| `--pr <n>` | Explicit PR number (same as passing it positionally). |
| `--lens <name>` | Restrict to one lens (repeatable). Default: all. See below. |
| `--severity <level>` | Only report at/above `low\|medium\|high`. Default `low`. |
| `--files <glob>` | Restrict to matching paths, e.g. `--files 'server/**'`. |
| `--since <ref>` | Review only commits after `<ref>` — for re-reviewing a pushed fix. |
| `--incremental` | Shorthand for `--since` the last commit you reviewed on this PR. |
| `--checklist <path>` | Also check the diff against a checklist file. |
| `--comment` | Post findings as inline review comments. **Writes to GitHub.** |
| `--summary` | Post one summary comment instead of inline ones. |
| `--approve` | Submit the review as APPROVE. |
| `--request-changes` | Submit the review as REQUEST_CHANGES. |
| `--dry-run` | With `--comment`, print exactly what would be posted, post nothing. |
| `--format <markdown\|table\|json>` | Output shape. Default markdown. |

Anything that writes to GitHub (`--comment`, `--summary`, `--approve`,
`--request-changes`) must be **confirmed with the user first**, every time. A review
posted under their name is public and hard to walk back. Never approve a PR the user
has not asked you to approve.

## Steps

### 1. Resolve the PR

```bash
gh pr view <n> --json number,title,body,author,baseRefName,headRefName,state,isDraft,labels,files,additions,deletions
gh pr view --json number   # no argument: the PR for the current branch
```

If there is no PR for the current branch, say so and offer `gh-pr-create`.

State up front: number, title, author, base ← head, size, draft status. If it is a
draft, say so — review expectations differ.

### 2. Get the diff

```bash
gh pr diff <n>                       # full patch
gh pr diff <n> --name-only           # file list first, to plan
gh pr view <n> --json commits        # commit-by-commit intent
```

For `--since` / `--incremental`, diff only the new commits — a re-review should look at
the fix as new code, not re-read everything:

```bash
git fetch origin && git diff <since>..<head> -- <paths>
```

Read by logical area (domain, endpoints, data, UI, tests, migrations), not top-to-bottom
by filename. Read the PR description and linked issue first so you review against the
stated intent.

### 3. Lenses

Run all unless `--lens` narrows it.

| Lens | Looks for |
| --- | --- |
| `security` | Missing `RequireAuthorization()`; a client-supplied id (userId, squadId) trusted without a membership/ownership check; a query not filtered by the caller; secrets or tokens in tracked files; anything weakening auth. |
| `bugs` | Logic that is wrong on a real input. Off-by-one, null paths, timezone/day-boundary handling, async gaps, silently swallowed exceptions. |
| `contracts` | Endpoint, DTO, entity or interface shape changes; whether the Flutter client still matches; EF migrations and what they do to existing rows. |
| `tests` | Whether new behaviour is actually covered, and whether a new test would fail without the fix. A test that passes before and after proves nothing. |
| `quality` | Duplication, magic numbers/strings, oversized widgets or handlers, project-rule violations (see `.agents/rules/project-rules.md`). |
| `docs` | Code/doc drift: does this change make a doc in `docs/`, the README or `CLAUDE.md` wrong? |

**Repo-specific things worth checking every time**, because they have all bitten before
(see `docs/review-code-reports/`):

- A handler writing several entities without `IUnitOfWork.ExecuteInTransactionAsync` —
  every repository method calls `SaveChangesAsync` itself, so N calls = N transactions.
- A read that loads a whole table or a user's entire all-time history to compute one value.
- Derived state (XP, streaks) recomputed at read time instead of recorded when earned.
- `BuildContext` used after an `await` without a guard, or guarded by an unrelated
  `mounted` check.
- A new hardcoded day-boundary offset instead of `StreakCalculator`.

### 4. Score before reporting

For each candidate, ask two questions and keep only what survives:

- **Confidence** — have I actually traced this, or am I pattern-matching? If you have
  not read the call chain, either trace it or label the finding UNVERIFIED.
- **Impact** — what breaks, for whom, when? A finding with no concrete failure case is
  a style opinion; mark it as such or drop it.

Rank most severe first. Prefer five real findings over twenty speculative ones. It is a
good outcome to report that a PR is clean.

### 5. Report

Per finding: **severity**, one-sentence defect, `path:line`, the code, the mechanism,
concrete impact, a specific fix. Include a short "checked and sound" list so the author
knows what you covered — a review that is only complaints reads as hostile and hides
what was verified.

### 6. Posting (only when asked)

```bash
# inline, on a specific line of the diff
gh api repos/jessiicamaru/habit-tracker/pulls/<n>/comments \
  -f body='<text>' -f commit_id='<sha>' -f path='<file>' -F line=<n> -f side=RIGHT

# one summary comment
gh pr comment <n> --body-file <tmp>.md

# a verdict
gh pr review <n> --approve
gh pr review <n> --request-changes --body-file <tmp>.md
```

Inline comments must land on a line the diff actually touches, or the API rejects them.
Show the user the exact text before posting. With `--dry-run`, print and stop.

## Reading a PR without reviewing it

If the user only wants to know what a PR does, answer from steps 1–2 and stop. Do not
volunteer a full audit they did not ask for.
