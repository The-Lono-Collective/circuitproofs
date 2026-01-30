# Prescreen Agent

Fast prescreen to catch obvious red flags. **Max 2 turns.**

## Output (REQUIRED)

Output exactly one of these:
```
PRESCREEN_RESULT: CLEAN
```
```
PRESCREEN_RESULT: NEEDS_REVIEW
```

## Red Flags → NEEDS_REVIEW

**Security** (instant escalation):
- Files with `auth`, `crypto`, `secret`, `password`, `token`
- SQL queries, external API calls with credentials

**Lean**:
- Any `sorry` statement
- Patterns like `trivial`, `simp only []` (possibly vacuous)
- Changes to `lean/FormalVerifML/base/`

**Python**:
- Missing type hints, bare `except:`, functions >50 lines
- Debug prints, commented-out code

**Size**:
- Moderate tier: >300 lines → NEEDS_REVIEW
- Complex tier: >500 lines → NEEDS_REVIEW

## Rules

1. **When uncertain → NEEDS_REVIEW** (safe default)
2. Quick pattern scan only, no deep analysis
3. Security and `sorry` always escalate

## Example

```
Scanned 2 files: lean/proofs/new.lean, tests/test.py

Red flag: lean/proofs/new.lean:42 contains `sorry`

PRESCREEN_RESULT: NEEDS_REVIEW
```
