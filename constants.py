# ===========================
# constants.py
# ===========================

STATUS_REQUIRED = {"Transferred to AlohaABA", "Completed"}
SESSION_REQUIRED = "1:1 BT Direct Service"

MIN_MINUTES = 53
MAX_MINUTES = 360
BILLING_TOL_DEFAULT = 8
DAILY_MAX_MINUTES = 480

TIME_ADJ_COL = "Adult Caregiver’s Signature Approval for Time Adjustment signature"

REQ_COLS = [
    "Status",
    "AlohaABA Appointment ID",
    "AlohaABA Appointment Type",
    "Client",
    "Duration",
    "Session",
    "Adult Caregiver signature time",
    "User",
    "Start date",
]

DATE_RE = (
    r"("
    r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}"          # ISO / compact-dot:  2026-3-14 | 2026.3.14
    r"|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}"          # US / EU compact:    3/14/2026 | 14.3.2026
    r"|\d{4}\.\s+\d{1,2}\.\s+\d{1,2}\.?"         # Korean spaced-dot:  2026. 3. 14.
    r")"
)

MIN_SESSION_GAP_MINUTES = 23