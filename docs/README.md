# Documentation Index

This folder contains comprehensive project documentation for the Physical-Layer Knowledge Distillation (PKD) system.

## Quick Navigation

### For New Contributors
Start here to understand the project:
1. [../README.md](../README.md) - Project overview
2. [ARCHITECTURE.md](./ARCHITECTURE.md) - System design and key decisions
3. [../QUICK_START.md](../QUICK_START.md) - Getting started guide

### For Active Development
Day-to-day references:
- [COMMON_ISSUES.md](./COMMON_ISSUES.md) - **Read this first!** Known pitfalls and solutions
- [../CHANGELOG.md](../CHANGELOG.md) - Recent changes and fixes
- [../data/README.md](../data/README.md) - Data format specification

### For Bug Fixing
When something breaks:
1. Check [COMMON_ISSUES.md](./COMMON_ISSUES.md) - Is this a known issue?
2. Check [../CHANGELOG.md](../CHANGELOG.md) - Was this recently changed?
3. Check [ARCHITECTURE.md](./ARCHITECTURE.md) - What are the invariants?

---

## Documentation Structure

### `/docs/` (This Folder)

| File | Purpose | When to Read |
|------|---------|--------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System design, component overview, architectural decisions | Understanding how things work |
| [COMMON_ISSUES.md](./COMMON_ISSUES.md) | Known bugs, gotchas, pitfalls, prevention strategies | Before coding, when debugging |

### Root Directory

| File | Purpose | When to Read |
|------|---------|--------------|
| [../CHANGELOG.md](../CHANGELOG.md) | Chronological record of changes, fixes, and features | After git pull, when investigating bugs |
| [../README.md](../README.md) | Project overview, installation, basic usage | First time setup |
| [../QUICK_START.md](../QUICK_START.md) | Step-by-step getting started guide | Learning the workflow |

### Data Documentation

| File | Purpose | When to Read |
|------|---------|--------------|
| [../data/README.md](../data/README.md) | MAT file format, field definitions, indexing conventions | Loading data, understanding configs |

### Technical Deep-Dives

| File | Purpose | When to Read |
|------|---------|--------------|
| [../AR_STABILITY_FIXES.md](../AR_STABILITY_FIXES.md) | AR process stability analysis and PACF constraints | Modifying AR model |
| [../GAUSSIAN_INNOVATION.md](../GAUSSIAN_INNOVATION.md) | Innovation distribution design choice rationale | Changing innovation model |
| [../TRAINING_FIXES.md](../TRAINING_FIXES.md) | Training procedure improvements | Debugging training |

---

## Documentation Philosophy

### 1. Single Source of Truth
- **Data format**: `data/README.md` is canonical
- **Code behavior**: `ARCHITECTURE.md` is canonical
- **Changes**: `CHANGELOG.md` is canonical
- **Issues**: `COMMON_ISSUES.md` is canonical

**Never duplicate information** - link to the canonical source instead.

### 2. When to Update Documentation

**Update immediately when:**
- ✅ You fix a bug → Add to `CHANGELOG.md` and `COMMON_ISSUES.md`
- ✅ You change architecture → Update `ARCHITECTURE.md`
- ✅ You change data format → Update `data/README.md`
- ✅ You discover a gotcha → Add to `COMMON_ISSUES.md`

**Don't wait until "later"** - future you will forget the context!

### 3. Documentation for AI Assistants

**Why this structure helps Claude Code:**

- **Context Efficiency**: Claude can quickly find relevant info instead of re-reading all code
- **Consistency**: Single source of truth prevents conflicting information
- **Learning**: Documented mistakes prevent repetition
- **Onboarding**: New conversation = new Claude instance, docs provide continuity

**Best Practices for AI-Assisted Development:**

1. **Start every session** by asking Claude to read relevant docs:
   - "Read `docs/COMMON_ISSUES.md` before we start"
   - "Check `CHANGELOG.md` for recent MCS changes"

2. **When Claude fixes a bug**, ask it to update docs:
   - "Update `CHANGELOG.md` with this fix"
   - "Add this gotcha to `COMMON_ISSUES.md`"

3. **For architectural changes**, require documentation:
   - "Document this decision in `ARCHITECTURE.md`"
   - "Explain the trade-offs in the decision log"

4. **Project-specific knowledge ≠ Skills**:
   - **Don't store as Claude Skills** (Skills are for general reusable tools)
   - **Do store as project docs** (Version controlled, shareable)

---

## How to Use This Documentation

### Scenario 1: "I want to add a new feature"

1. Read [ARCHITECTURE.md](./ARCHITECTURE.md) → "Extension Points" section
2. Check [COMMON_ISSUES.md](./COMMON_ISSUES.md) → Related pitfalls
3. Review [../CHANGELOG.md](../CHANGELOG.md) → Recent changes to same area
4. Implement feature
5. **Update docs**: Add architectural decision, update changelog

### Scenario 2: "Tests are failing after git pull"

1. Check [../CHANGELOG.md](../CHANGELOG.md) → What changed recently?
2. Check [COMMON_ISSUES.md](./COMMON_ISSUES.md) → Known issues with this component?
3. Check [ARCHITECTURE.md](./ARCHITECTURE.md) → Invariants section
4. Debug and fix
5. **Update docs**: If it's a new issue, add to `COMMON_ISSUES.md`

### Scenario 3: "Generated sequences have NaN values"

1. Check [COMMON_ISSUES.md](./COMMON_ISSUES.md) → "Sequence Generation Validation"
2. Check [../AR_STABILITY_FIXES.md](../AR_STABILITY_FIXES.md) → PACF constraints
3. Verify `kappa_max` setting
4. Add validation checks
5. **Update docs**: If you found a new cause, document it

### Scenario 4: "Starting fresh development with Claude"

**Tell Claude:**
```
Please read the following before we start:
1. docs/COMMON_ISSUES.md - to avoid known pitfalls
2. CHANGELOG.md - to see recent changes
3. data/README.md - to understand data conventions
```

This gives Claude the project context efficiently!

---

## Contributing to Documentation

### Adding a New Document

1. Place in appropriate location:
   - High-level design → `/docs/`
   - Data-specific → `/data/`
   - Technical deep-dive → Root directory

2. Update this index (`docs/README.md`)

3. Cross-link from related documents

### Updating Existing Documents

1. **Add to bottom** (for chronological docs like CHANGELOG)
2. **Update in-place** (for reference docs like COMMON_ISSUES)
3. **Update "Last Updated" date** at bottom of file
4. **Commit with descriptive message**: "docs: Add MCS indexing fix to COMMON_ISSUES"

### Writing Style Guidelines

- **Be specific**: Include file paths, line numbers, code snippets
- **Explain "why"**: Not just "what" changed, but "why" it was wrong
- **Provide examples**: Good vs. bad code patterns
- **Link liberally**: Connect related documentation
- **Use checklists**: Actionable prevention strategies
- **Date everything**: Track when issues were discovered/fixed

---

## Documentation Maintenance

### Weekly Review
- [ ] Scan `CHANGELOG.md` for undocumented patterns
- [ ] Check if `COMMON_ISSUES.md` needs new sections
- [ ] Verify cross-links are not broken

### Before Major Release
- [ ] Review all documentation for accuracy
- [ ] Update "Last Updated" dates
- [ ] Ensure all architectural decisions are documented
- [ ] Validate code examples still work

### When Onboarding New Contributors
- [ ] Ask them to read docs in this order:
  1. README.md
  2. ARCHITECTURE.md
  3. COMMON_ISSUES.md
  4. QUICK_START.md
- [ ] Get feedback on what was unclear
- [ ] Update docs based on their questions

---

## Questions?

**Documentation unclear?** → This is a bug! Please:
1. Ask for clarification
2. Once clarified, update the docs yourself
3. Future readers will thank you

**Found a mistake?** → Fix it immediately:
1. Edit the relevant file
2. Commit with clear message
3. If it caused a bug, add to `COMMON_ISSUES.md`

**Want to add something?** → Just do it:
1. Follow the structure and style above
2. Update this index
3. Cross-link from related docs

---

**Maintained By:** All project contributors
**Last Updated:** 2026-02-03
**Philosophy:** Documentation is code - keep it DRY, tested, and up-to-date!
