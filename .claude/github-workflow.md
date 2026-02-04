# GitHub Workflow Automation Guide

**Purpose:** Rules for when Claude should proactively use `gh` commands during development sessions.

---

## Automatic Triggers

### 🐛 When to Create Issues

Claude should **proactively suggest** creating an issue when:

1. **Bug Discovery:**
   - Code produces incorrect results
   - Tests fail unexpectedly
   - Implementation doesn't match specification
   - Data format inconsistencies found

2. **Missing Features:**
   - Paper mentions feature not implemented
   - Documentation gap identified
   - Required functionality absent

3. **Technical Debt:**
   - Hardcoded values that should be configurable
   - Duplicated code needing refactoring
   - Performance bottlenecks identified

**Action:**
```bash
gh issue create --title "[Bug/Feature/Debt]: Title" \
  --body "Description with context" \
  --label "bug|enhancement|tech-debt"
```

---

### 🔀 When to Create Pull Requests

Claude should **proactively suggest** creating a PR when:

1. **Non-Trivial Changes:**
   - Modifying core algorithms (AR process, PACF, innovation)
   - Adding new model components
   - Changing training procedures
   - Updating data formats

2. **Multi-File Changes:**
   - Changes span > 3 files
   - Architecture modifications
   - API changes

3. **Experimental Features:**
   - Trying alternative approaches
   - A/B testing implementations
   - Performance optimizations

**Action:**
```bash
# Create feature branch
git checkout -b feature/descriptive-name

# After changes
gh pr create --title "feat: Description" \
  --body "## Changes\n- ...\n\n## Testing\n- ..." \
  [--draft if incomplete]
```

---

### ✅ When to Commit Directly to Main

**ONLY for:**
1. Documentation-only changes
2. Comment additions/fixes
3. README updates
4. Configuration tweaks (no logic changes)
5. Typo fixes

**Action:**
```bash
git add -A
git commit -m "docs: description"
git push
```

---

## CI/CD: Automatic Testing

### Test Requirements

**Every PR must:**
1. Pass all unit tests
2. Pass code quality checks
3. Generate validation report

**GitHub Actions will automatically:**
- Run `pytest pkd/tests/`
- Check code style
- Validate data format compatibility

---

## Session Workflow

### At Session Start

Claude checks:
```bash
gh issue list --label "bug" --limit 5
gh pr list --state open
```

Informs user of:
- Open critical bugs
- Pending PRs needing review

### During Session

**When Claude identifies a bug:**
> "🐛 Bug detected: [description]
>
> Should I create an issue to track this?"
>
> If yes → runs `gh issue create`

**When making non-trivial changes:**
> "🔀 These changes modify core algorithm.
>
> Should I create a PR for review?"
>
> If yes → creates branch + PR

### Before Commit

Claude checks:
1. Are changes trivial (docs/comments only)? → Direct commit
2. Do changes affect algorithm logic? → Suggest PR
3. Did we fix a bug? → Link to issue in commit message

### After Commit/PR

Claude suggests:
```bash
# If bug fix
"This fixes issue #X. Should I close it?"
gh issue close X --comment "Fixed in commit ABC"

# If PR merged
"PR merged. Should I delete the feature branch?"
git branch -d feature/name
```

---

## Labels System

**Claude uses these labels:**

| Label | When to Use |
|-------|-------------|
| `bug` | Incorrect behavior |
| `critical` | Blocks research/paper |
| `enhancement` | New feature request |
| `tech-debt` | Code quality issue |
| `documentation` | Docs improvement |
| `question` | Needs discussion |
| `paper-related` | Affects paper results |

**Priority labels:**
- `P0`: Must fix immediately
- `P1`: Fix before next milestone
- `P2`: Nice to have

---

## Example Scenarios

### Scenario 1: Bug Found During Code Review

**Claude detects:** MCS indexing is 1-10 instead of 0-9

**Claude says:**
> "🐛 **Bug Found:** MCS indexing incorrect
>
> **Issue:** Code assumes MCS 1-10, but data uses 0-9
> **Impact:** Wrong embeddings, incorrect predictions
> **Files:** encoder.py, example.py, tests/
>
> Should I:
> 1. Create issue to track this bug?
> 2. Create PR to fix it?
> 3. Fix directly on main (if you approve)?"

**User approves option 2:**

```bash
# Claude runs:
gh issue create --title "Bug: MCS indexing off by one" \
  --label "bug,critical,paper-related" \
  --body "See conversation for details"

git checkout -b fix/mcs-indexing

# Makes fixes...

gh pr create --title "fix: Correct MCS indexing to 0-9" \
  --body "Fixes #42\n\n## Changes\n- ..."
```

---

### Scenario 2: Paper Verification Finds Discrepancy

**Claude reads paper and finds:** Innovation should use flows, not Gaussian

**Claude says:**
> "📄 **Paper Discrepancy Found**
>
> **Paper says:** Use normalizing flows for innovation
> **Code does:** Uses Gaussian innovation
>
> Should I:
> 1. Create issue to discuss this?
> 2. Implement flow innovation in PR?
> 3. Confirm Gaussian is intentional?"

**User says "Gaussian is correct, paper is outdated":**

```bash
# Claude updates docs only (direct commit)
git add docs/PAPER_IMPLEMENTATION_COHERENCE.md
git commit -m "docs: Clarify Gaussian vs Flow innovation choice"
git push
```

---

### Scenario 3: Adding New Feature

**User asks:** "Add support for time-varying configurations"

**Claude plans:**
> "🔀 **Non-Trivial Change Detected**
>
> This will modify:
> - Core inference loop (infer.py)
> - Configuration caching (3 files)
> - Tests (new test cases)
>
> I recommend creating a feature branch and PR.
>
> Proceed with PR workflow?"

**User approves:**

```bash
git checkout -b feature/time-varying-config

# Implement feature...

gh pr create --draft \
  --title "feat: Support time-varying configurations" \
  --body "## Implementation\n...\n\n## TODO\n- [ ] Add tests\n- [ ] Update docs"
```

---

## Integration with Development

### User Can Say:

- **"Claude, check for open issues"** → `gh issue list`
- **"Claude, track this as a bug"** → Creates issue
- **"Claude, make a PR for this"** → Creates PR workflow
- **"Claude, link this commit to issue #X"** → Adds "Fixes #X" to commit

### Claude Will Proactively:

1. **Detect bugs** → Suggest issue creation
2. **Make algorithm changes** → Suggest PR
3. **Fix simple docs** → Direct commit
4. **Complete PR** → Suggest merge

---

## Benefits

✅ **Traceability:** Every bug/feature tracked
✅ **Review Process:** Non-trivial changes reviewed via PR
✅ **Automation:** CI/CD catches issues early
✅ **Documentation:** Issues = living TODO list
✅ **Reproducibility:** Can trace every change back to issue/PR

---

**Last Updated:** 2026-02-03
**Status:** Active - Claude follows these rules automatically
