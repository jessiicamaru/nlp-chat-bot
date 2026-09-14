# PR description templates

One shared skeleton with five variants. Fill from real evidence; **delete any section
that does not apply** rather than writing "N/A" — empty headings make a PR harder to
read, not more complete.

Placeholders look like `{{this}}`.

---

## Badge row (top of every PR, omit with `--no-badge`)

Mirrors the labels. Colours match the label taxonomy in `gh-setup`.

```markdown
![type](https://img.shields.io/badge/type-{{type}}-{{type_color}})
![area](https://img.shields.io/badge/area-{{area}}-{{area_color}})
![size](https://img.shields.io/badge/size-{{size}}-{{size_color}})
{{risk_badges}}
```

| Field | Values → colour |
| --- | --- |
| `type` | feat→0E8A16, fix→D73A4A, refactor→FBCA04, docs→0075CA, chore→CFD3D7, test→BFD4F2 |
| `area` | backend→5319E7, frontend→1D76DB, widget→006B75, sync→B60205, db→5319E7. Join several with `%20%7C%20`. |
| `size` | XS/S→C2E0C6, M→FEF2C0, L→F9D0C4, XL→E99695 |
| `risk_badges` | Only when true. e.g. `![risk](https://img.shields.io/badge/risk-migration-E99695)` |

Spaces in a badge label must be `%20`; a literal `|` must be `%7C`.

---

## Base skeleton

```markdown
{{badge_row}}

## Summary

{{One paragraph: what this changes and why. Written for someone who has not read the
issue. Lead with the user-visible effect, not the implementation.}}

{{Closes #N   |   Refs #N   |   omit the line entirely if no issue}}

## What changed

{{Group by area, not by file. Each bullet says what now behaves differently.}}

- **Backend** — {{...}}
- **Frontend** — {{...}}

## Verification

| Check | Result |
| --- | --- |
| `dotnet build` | {{0 errors / n warnings}} |
| `dotnet test` | {{n passed / n failed}} |
| `flutter analyze` | {{n issues}} |
| `flutter test` | {{n passed}} |
| Manual / E2E | {{what you actually exercised, or "not run"}} |

{{Paste the real output for anything surprising. If a suite was not run, say so here —
do not leave it out.}}

## Review notes

{{Optional but valuable: where to start reading, a decision you are unsure about, an
alternative you rejected and why. Delete if you genuinely have nothing.}}

## Checklist

- [ ] Tests cover the new or changed behaviour (project rule 11)
- [ ] `dotnet test` and `flutter test` pass (project rule 14)
- [ ] `dotnet build` and `flutter analyze` are clean (project rule 16)
- [ ] No hardcoded strings or magic numbers (project rule 4)
- [ ] No secret, token or credential added to a tracked file
- [ ] UI uses `shadcn_ui` + `lucide_icons` only (project rule 7)
```

Only tick a box you have actually verified. An unticked box with a one-line reason is
useful; a ticked box that is not true is worse than no checklist.

---

## Variant: feature

Insert after **What changed**:

```markdown
## How to try it

{{Numbered steps a reviewer can follow to see the feature, starting from a clean
checkout. Name the seed data or account needed.}}

## Screenshots

| Before | After |
| --- | --- |
| {{img}} | {{img}} |
```

Screenshots are expected for anything under `apps/lib/features/**/presentation/`.

---

## Variant: bugfix

Replace **Summary** with:

```markdown
## The bug

{{What went wrong, from the user's point of view.}}

**Reproduce (before this PR):**
{{Exact steps or input, and the wrong output. Paste it.}}

## Root cause

{{The actual mechanism, with a `path/to/file.cs:line` citation. Not "fixed a typo" —
say what the code did and why that was wrong.}}

## The fix

{{What now happens instead, and why this addresses the cause rather than the symptom.}}

**After:**
{{Same steps, correct output. Paste it.}}
```

Add a regression-test line to the checklist:

```markdown
- [ ] A test now fails without this fix (negative control run)
```

A bugfix PR without that control is worth flagging — a test that passes both before and
after proves nothing.

---

## Variant: refactor

Insert after **Summary**:

```markdown
## Behaviour is unchanged because

{{How you know. Ideally: the same tests pass untouched. If a test had to change, say
which and why — that is the interesting part of a refactor.}}
```

---

## Variant: docs

Trim to **Summary**, **What changed**, and:

```markdown
## Accuracy

{{Docs drift. State what you verified against the code, and how — a command, a file
you read. Say plainly if any part is aspirational rather than descriptive.}}
```

Drop the build/test table; keep the secrets checklist line.

---

## Variant: chore

Trim to **Summary**, **What changed**, **Verification**. For dependency bumps, name the
version moved from → to and whether anything downstream changed.

---

## Migration warning (add whenever the diff adds a migration)

Put this directly under the badge row so it cannot be missed:

```markdown
> [!WARNING]
> **Contains an EF Core migration:** `{{migration_name}}`
> Reviewers must run:
> `dotnet ef database update --project src/Infrastructure --startup-project src/Web`
> {{If it backfills or rewrites existing rows, say exactly what it does to them and
> whether it is reversible.}}
```

---

## Breaking change (add whenever a contract changes shape)

```markdown
> [!CAUTION]
> **Breaking:** {{what breaks, and for whom — the Flutter client, an existing DB row,
> a stored token.}}
> **Migration path:** {{what a consumer has to do.}}
```
