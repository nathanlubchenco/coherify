# Documentation Workflow Guide
*Last Updated: 2025-01-28*

## Directory Structure

```
docs/
├── in_progress/     # Active work (max 1-2 docs)
├── completed/       # Finished documentation
├── backlog/         # Future work and ideas
├── paused/          # Work temporarily on hold
├── benchmarks/      # External benchmark READMEs (reference)
├── WORKFLOW.md      # This guide
└── DOCUMENTATION_STATUS.md  # Current categorization
```

## Workflow Rules

### 1. Starting New Documentation
- Create doc in `in_progress/` directory
- Maximum 2 documents in progress at once
- Use descriptive names: `FEATURE_NAME.md`
- Add header with creation date and status

### 2. Document Header Template
```markdown
# Document Title
*Created: YYYY-MM-DD*
*Last Updated: YYYY-MM-DD*
*Status: In Progress | Completed | Paused*
*Author: [Name/Handle]*

## Purpose
Brief description of what this document covers
```

### 3. Moving Between Categories

#### In Progress → Completed
- Document is fully written and reviewed
- All TODOs are resolved
- Add completion date to header
- Move with: `mv docs/in_progress/DOC.md docs/completed/`

#### In Progress → Paused
- Work needs to stop temporarily
- Add pause reason and expected resume date
- Move with: `mv docs/in_progress/DOC.md docs/paused/`

#### Backlog → In Progress
- Only when current in_progress count < 2
- Update header with start date
- Move with: `mv docs/backlog/DOC.md docs/in_progress/`

#### Paused → In Progress
- Resume work on paused document
- Update header with resume date
- Move with: `mv docs/paused/DOC.md docs/in_progress/`

### 4. Document Lifecycle

```
[Idea] → backlog/ → in_progress/ → completed/
                         ↓
                      paused/
```

### 5. Naming Conventions

#### Feature Documentation
- `FEATURE_NAME.md` - Implementation details
- `FEATURE_NAME_DESIGN.md` - Design decisions
- `FEATURE_NAME_RESULTS.md` - Performance results

#### Process Documentation
- `WORKFLOW_*.md` - Process guides
- `GUIDE_*.md` - How-to guides
- `TROUBLESHOOTING_*.md` - Problem solutions

#### Status Documentation
- `STATUS_*.md` - Current state snapshots
- `AUDIT_*.md` - Review/audit results
- `SUMMARY_*.md` - High-level summaries

### 6. Regular Maintenance

#### Weekly Review
- Check `in_progress/` isn't overcrowded
- Move completed work to `completed/`
- Review `paused/` for items to resume
- Update `DOCUMENTATION_STATUS.md`

#### Monthly Cleanup
- Archive old completed docs if needed
- Consolidate related documents
- Remove obsolete backlog items
- Update root-level status files

### 7. Root-Level Status Files

These remain in the root for visibility:
- `TODO.md` - Current tasks (mirror in in_progress/)
- `CURRENT_STATE.md` - Quick context (mirror in in_progress/)
- `PROJECT_SUMMARY.md` - High-level overview (mirror in in_progress/)
- `CLAUDE.md` - AI assistant instructions (don't move)
- `README.md` - Project readme (don't move)

### 8. Special Directories

#### benchmarks/
- Contains external benchmark documentation
- Read-only reference material
- Don't move or modify

### 9. Git Commit Messages

When moving docs between categories:
```bash
git add docs/
git commit -m "docs: move FEATURE.md from in_progress to completed

- Feature fully documented
- All sections complete
- Ready for reference"
```

### 10. Quality Checklist

Before marking as completed:
- [ ] All sections filled out
- [ ] Code examples tested
- [ ] Links verified
- [ ] Dates updated
- [ ] No remaining TODOs
- [ ] Reviewed for accuracy

## Examples

### Creating New Feature Doc
```bash
# Create in backlog first
echo "# New Feature Design" > docs/backlog/NEW_FEATURE_DESIGN.md

# When ready to work on it
mv docs/backlog/NEW_FEATURE_DESIGN.md docs/in_progress/

# After completion
mv docs/in_progress/NEW_FEATURE_DESIGN.md docs/completed/
```

### Pausing Work
```bash
# Add pause note to document
echo "\n## PAUSED: 2025-01-28\nReason: Waiting for API access\nExpected resume: 2025-02-01" >> docs/in_progress/API_INTEGRATION.md

# Move to paused
mv docs/in_progress/API_INTEGRATION.md docs/paused/
```

## Enforcement

- CI/CD can check `in_progress/` count
- Pre-commit hooks can validate headers
- Regular audits ensure compliance
