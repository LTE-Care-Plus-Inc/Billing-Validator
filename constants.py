# ===========================
# constants.py
# ===========================

STATUS_REQUIRED = "Transferred to AlohaABA"
SESSION_REQUIRED = "1:1 BT Direct Service"

MIN_MINUTES = 53
MAX_MINUTES = 360
BILLING_TOL_DEFAULT = 8
DAILY_MAX_MINUTES = 480

TIME_ADJ_COL = "Adult Caregiver’s Signature Approval for Time Adjustment signature"

REQ_COLS = [
    "Status",
    "AlohaABA Appointment ID",
    "Client",
    "Duration",
    "Session",
    "Adult Caregiver signature time",
    "User",
    "Start date",
]

DATE_RE = r"(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})"

# Minimum gap (in minutes) required between two sessions for the same client on the same day
# Set to 0 to disable
MIN_SESSION_GAP_MINUTES = 0