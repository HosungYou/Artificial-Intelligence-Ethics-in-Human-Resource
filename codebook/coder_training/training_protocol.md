# Human Coder Training Protocol

## Overview

This protocol trains human coders for the Phase 3 verification sample of the AI Ethics in HR systematic review.

## Training Duration

- Total time: 4-6 hours
- Session 1: Codebook review (2 hours)
- Session 2: Practice coding (2 hours)
- Session 3: Calibration (1-2 hours)

## Training Steps

### Step 1: Codebook Review (2 hours)

1. Read the full AI_Ethics_HR_Codebook.md
2. Review all field definitions and coding rules
3. Study the provided examples
4. Note any questions or ambiguities

### Step 2: Practice Coding (2 hours)

Code the following 5 calibration papers:
1. `calibration_paper_1.pdf` - Focus on bias (expected: major_focus)
2. `calibration_paper_2.pdf` - Focus on privacy (expected: major_focus)
3. `calibration_paper_3.pdf` - Multiple ethics issues (expected: balanced)
4. `calibration_paper_4.pdf` - Theoretical paper (expected: framework focus)
5. `calibration_paper_5.pdf` - Empirical study (expected: specific findings)

Compare your coding to the gold standard answers.

### Step 3: Calibration Session (1-2 hours)

1. Review discrepancies between your coding and gold standard
2. Discuss challenging decisions with trainer
3. Recode any papers where you made errors
4. Achieve κ ≥ 0.80 agreement with gold standard

## Calibration Criteria

**Ready to code independently when:**
- Agreement with gold standard κ ≥ 0.80 on all categorical fields
- No systematic errors (e.g., consistently missing certain ethics principles)
- Can articulate reasoning for coding decisions
- Understands edge cases and how to handle them

## Edge Cases

### Multiple HR Functions
- If paper covers recruitment AND selection equally: code as `selection` (more downstream)
- If paper is truly cross-functional: code as `multiple`

### Implicit Ethics Discussion
- Only code as "mentioned" if ethics issue is explicitly named
- Implicit discussion (e.g., "algorithm may have issues") does NOT count

### Theoretical vs. Empirical
- Code theoretical framework even in empirical papers if explicitly applied
- Conceptual papers without named theory: theoretical_framework.applied = false

### Severity Ambiguity
- If unsure between levels, default to the lower severity
- "major_focus": Appears in title/abstract AND has dedicated section
- "discussed": At least one full paragraph
- "mentioned": Passing reference only

## Coding Tips

1. Always read abstract first for overview
2. Use Ctrl+F to search for ethics keywords
3. Code ethics principles in order (bias, transparency, etc.)
4. Double-check severity assignments
5. Leave notes for uncertain decisions

## Quality Assurance

### Intra-Rater Reliability
- Recode 20% of your sample after 2 weeks
- Target κ ≥ 0.90 for self-agreement
- Document any systematic shifts in interpretation

### Flagging Difficult Papers
- Use "needs_review" flag for genuinely ambiguous papers
- Document specific questions/concerns
- These will be reviewed in Phase 5

## Contact

For coding questions during the process:
- Document the question and your decision
- Continue coding (don't wait)
- Questions will be resolved in weekly meetings
