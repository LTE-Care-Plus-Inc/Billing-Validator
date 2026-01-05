import pytest
from datetime import date
import pandas as pd
from utils import (
    normalize_session_time,
    parse_session_time_range,
)

# =========================================================
# normalize_session_time tests
# =========================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        # Chinese AM/PM
        ("下午4:30 - 下午7:30", "4:30 PM - 7:30 PM"),

        # 24-hour time with spacing
        ("14:30  -  18:29", "2:30 PM - 6:29 PM"),

        # Standard AM/PM
        ("1:00 PM - 7:00 PM", "1:00 PM - 7:00 PM"),

        # Lowercase + punctuation
        ("9:55 a.m.  -  4:03 p.m.", "9:55 AM - 4:03 PM"),
    ],
)
def test_normalize_session_time(raw, expected):
    result = normalize_session_time(raw)
    assert result == expected


# =========================================================
# parse_session_time_range tests
# =========================================================

@pytest.mark.parametrize(
    "session_time, start_hour, start_min, end_hour, end_min",
    [
        ("4:30 PM - 7:30 PM", 16, 30, 19, 30),
        ("2:30 PM - 6:29 PM", 14, 30, 18, 29),
        ("1:00 PM - 7:00 PM", 13, 0, 19, 0),
        ("9:55 AM - 4:03 PM", 9, 55, 16, 3),
    ],
)
def test_parse_session_time_range(session_time, start_hour, start_min, end_hour, end_min):
    base_date = date(2025, 1, 15)

    start_dt, end_dt = parse_session_time_range(session_time, base_date)

    assert not pd.isna(start_dt)
    assert not pd.isna(end_dt)

    assert start_dt.hour == start_hour
    assert start_dt.minute == start_min

    assert end_dt.hour == end_hour
    assert end_dt.minute == end_min


# =========================================================
# End-to-end safety tests
# raw → normalized → parsed
# =========================================================

@pytest.mark.parametrize(
    "raw",
    [
        "下午4:30 - 下午7:30",
        "14:30  -  18:29",
        "1:00 PM - 7:00 PM",
        "9:55 a.m.  -  4:03 p.m.",
    ],
)
def test_end_to_end_time_parsing(raw):
    normalized = normalize_session_time(raw)

    start_dt, end_dt = parse_session_time_range(
        normalized,
        "2025-01-15"
    )

    assert normalized != ""
    assert not pd.isna(start_dt)
    assert not pd.isna(end_dt)
    assert end_dt > start_dt


# =========================================================
# Defensive / failure cases (optional but recommended)
# =========================================================

@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        "invalid time",
        "下午 - 下午",
        "25:99 - 30:00",
    ],
)
def test_invalid_times_do_not_crash(raw):
    normalized = normalize_session_time(raw)
    start_dt, end_dt = parse_session_time_range(normalized, "2025-01-15")

    # We don't require valid parsing, only that it fails safely
    assert start_dt is pd.NaT or pd.isna(start_dt)
    assert end_dt is pd.NaT or pd.isna(end_dt)
