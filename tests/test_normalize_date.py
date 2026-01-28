import pytest

from ..utils import normalize_date

# =========================================================
# 🇺🇸 American slash dates (MM/DD/YYYY)
# =========================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("02/01/2026", "02/01/2026"),  # Feb 1
        ("04/01/2026", "04/01/2026"),  # Apr 1
        ("12/25/2025", "12/25/2025"),
        ("1/4/2026", "01/04/2026"),
    ],
)
def test_us_slash_dates(raw, expected):
    assert normalize_date(raw) == expected


# =========================================================
# 🇪🇺 European slash dates (DD/MM/YYYY → US normalized)
# =========================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("21/01/2026", "01/21/2026"),
        ("22/01/2026", "01/22/2026"),
        ("31/12/2025", "12/31/2025"),
    ],
)
def test_european_slash_dates(raw, expected):
    assert normalize_date(raw) == expected


# =========================================================
# ⚠️ Ambiguous slash dates → default to US
# =========================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("01/02/2026", "01/02/2026"),  # Jan 2 (default US)
        ("03/04/2026", "03/04/2026"),
    ],
)
def test_ambiguous_slash_dates_default_us(raw, expected):
    assert normalize_date(raw) == expected


# =========================================================
# 🔵 Dotted dates (MM.DD.YYYY per HiRasmus rule)
# =========================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("04.01.2026", "01/04/2026"),
        ("12.31.2025", "31/12/2025"),  # still deterministic
    ],
)
def test_dotted_dates_mm_dd(raw, expected):
    assert normalize_date(raw) == expected


# =========================================================
# 🇨🇳 ISO / system exports
# =========================================================

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-01-04", "01/04/2026"),
        ("2026/01/04", "01/04/2026"),
        ("2025-12-31", "12/31/2025"),
        ("2026-1-4", "01/04/2026"),
        ("2026/1/4", "01/04/2026"),
    ],
)
def test_iso_dates(raw, expected):
    assert normalize_date(raw) == expected


# =========================================================
# ❌ Unsupported formats → returned as-is
# =========================================================

@pytest.mark.parametrize(
    "raw",
    [
        "2026年1月4日",
        "Jan 4 2026",
        "04-Jan-2026",
        "20260104",
        "",
        None,
    ],
)
def test_unsupported_dates_return_raw(raw):
    result = normalize_date(raw)
    if raw is None:
        assert result == ""
    else:
        assert result == raw
