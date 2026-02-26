
# HiRasmus Billing Session Checker — Functional Specification

## 1. Overview
This application validates **1:1 BT Direct Service ABA sessions** for **Medicaid billing compliance** by cross-checking:
1. HiRasmus Session Export (Excel)
2. HiRasmus Session Notes (PDF)
3. Optional BT Contact File
4. Optional Aloha Billing Status File

The system enforces **strict audit-safe rules**, including a **strict 7-minute tolerance** (exactly 8 minutes fails).

---

## 2. App Structure
The app contains **three tabs**:
1. Tools – Extract External Sessions  
2. Session Checker  
3. Billed Checker  

Session state is preserved across tabs.

---

## 3. Global Rules & Constants

### Service Filters
- Status must equal: `Transferred to AlohaABA`
- Session type must equal: `1:1 BT Direct Service`

### Time & Duration Rules
| Rule | Value |
|----|----|
| Minimum session duration | 53 minutes |
| Maximum session duration | 360 minutes |
| Duration tolerance | < 8 minutes only |
| Daily max per BT | 480 minutes |
| Daily tolerance | < 8 minutes only |
| Parent signature early tolerance | < 8 minutes only |

**Exactly 8:00 minutes FAILS. 7:59 PASSES.**

---

## 4. Tab 1 — Tools: PDF → External Session List

### Input
- HiRasmus Session Notes PDF

### Output
- Parsed External Session List
- Stored for automatic use in Tab 2
- Downloadable as Excel

### PDF Parsing Rules
Each session note block starts with `Client:`. The parser extracts:
- Client Name
- Session Date
- Session Time (same line only)
- Session Location
- DOB
- Gender
- ICD-10 Diagnosis
- Primary Insurance
- Insurance ID

### Session Time Rules
- Must be on the same line as `Session Time:`
- Placeholder values (`-`, `—`, `–`) are treated as missing
- Must parse into a valid time range
- AM/PM inferred conservatively if missing

### Attendance Rules
The following must be present:
- Client
- BT/RBT
- Either Parent/Caregiver OR Sibling

### Provider Signature Rules
Valid signature must include:
- Alphabetic name
- BT or RBT identifier
- A valid date

### Compliance Enforcement
A note fails if any required field, attendance rule, or signature rule is violated.

---

## 5. Tab 2 — Session Checker

### Inputs
- HiRasmus Session Export (Excel)
- External Sessions from Tab 1
- Optional BT Contacts file

### Required Columns
- Status
- AlohaABA Appointment ID
- Client
- Duration
- Session
- Parent signature time
- User
- Start date

### Name Normalization
All names are normalized to `Last, First` format for matching.

### Session Matching
Sessions are matched using:
(Client Name, SessionIndex)

SessionIndex is based on original per-client order.

### Duration Validation
- < 53 min → FAIL
- 53–360 min → PASS
- Up to 7:59 over max → PASS
- Exactly 8:00+ → FAIL

### Parent Signature Timing
- Required
- Must be less than 8 minutes before session end
- No late limit
- Time-adjustment signature waives timing only

### Daily Hours Rule
- ≤ 487:59 → PASS
- ≥ 488:00 → FAIL

### Overall Pass
A session passes only if all checks succeed:
- Duration
- Signature
- External time
- Daily hours
- Attendance
- Note compliance

Failure reasons are human-readable.

---

## 6. Tab 3 — Billed Checker

### Input
- Aloha Billing Status file with:
  - Appointment ID
  - Date Billed

### Classification
- No Appointment ID → No Match
- Date Billed present → Billed
- Date Billed missing → Unbilled

Unbilled sessions are split into Clean vs Flagged.

---

## 7. Exports
The app produces Excel downloads for:
- All sessions
- Passed only
- Failed only
- Billed
- Unbilled (Clean)
- Unbilled (Flagged)
- No Match

All exports are Excel-safe sanitized.

---

## 8. Design Philosophy
- Audit defensible
- No rounding in favor of billing
- Seconds matter
- Deterministic matching
- Explainable failures

---
End of Specification
