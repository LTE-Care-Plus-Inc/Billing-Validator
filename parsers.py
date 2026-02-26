# ===========================
# parsers.py
# ===========================
import io
import re
import unicodedata

import fitz
import pandas as pd

from constants import DATE_RE
from utils import normalize_date, normalize_time_range, excel_sanitize_df


# =========================================================
# PDF → TEXT
# =========================================================
def pdf_bytes_to_text(pdf_bytes: bytes, preserve_layout: bool = True) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    all_text = []

    for page_num in range(total_pages):
        page = doc[page_num]
        if preserve_layout:
            text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES,
            )
        else:
            text = page.get_text("text")

        all_text.append(f"--- Page {page_num + 1} ---\n{text}\n\n")

    doc.close()
    return "".join(all_text)


# =========================================================
# Provider signature parsing + validation
# =========================================================
def _clean_pdf_line(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00a0", " ").replace("􀀀", " ")
    s = "".join(ch for ch in s if ch.isprintable())
    s = " ".join(s.split())
    return s.strip()


def extract_provider_signature_candidates(block_text: str) -> list[str]:
    label_re = re.compile(
        r"Provider\s+Signatures?\s*(?:/\s*)?Credentials\s+and\s+Date\s*:?\s*",
        re.I,
    )

    cands: list[str] = []
    for m in label_re.finditer(block_text):
        tail = block_text[m.end(): m.end() + 800]
        if not tail:
            continue

        lines = tail.splitlines()
        first_line = _clean_pdf_line(lines[0]) if len(lines) >= 1 else ""

        rest_lines = [_clean_pdf_line(x) for x in lines[1:10]]
        rest_lines = [x for x in rest_lines if x]

        pieces = []
        if first_line:
            pieces.append(first_line)
        if rest_lines:
            pieces.append(rest_lines[0])

        cand = _clean_pdf_line(" ".join(pieces))
        if cand:
            cands.append(cand)

    return cands


def any_signature_line_valid(block_text: str) -> tuple[bool, list[str]]:
    cands = extract_provider_signature_candidates(block_text)

    for cand in cands:
        if not re.search(r"R?BT", cand, re.I):
            continue
        if not re.search(r"\b\d{1,4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,4}\b", cand):
            continue
        if not any(ch.isalpha() for ch in cand):
            continue
        return True, cands

    return False, cands


# =========================================================
# NOTE PARSER
# =========================================================
def parse_notes(text: str):
    blocks = re.split(r"(?=Client\s*:)", text)
    results = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        client_match = re.search(r"Client\s*:\s*([^\n]+)", block)
        client_name = client_match.group(1).strip() if client_match else ""
        if not client_name:
            continue

        provider_match = re.search(r"Rendering Provider\s*:\s*([^\n]+)", block)
        provider = provider_match.group(1).strip() if provider_match else ""

        dob_match = re.search(rf"Date\s*of\s*Birth\s*[:\-]?\s*({DATE_RE})", block, re.I)
        dob = normalize_date(dob_match.group(1)) if dob_match else ""

        gender_match = re.search(r"Gender\s*:\s*([^\n\r]+)", block, re.I)
        gender_raw = gender_match.group(1).strip() if gender_match else ""

        g = gender_raw.strip().lower()
        if g in ("male", "m", "man", "boy", "男性", "男"):
            gender = "Male"
        elif g in ("female", "f", "woman", "girl", "女性", "女"):
            gender = "Female"
        else:
            gender = gender_raw

        diagnosis = ""
        dx_match = re.search(
            r"Diagnosis Code\s*\(\s*(?:ICD|IDC)\s*[-\s]*10\s*\)\s*:\s*([A-Za-z0-9\.\s]+)",
            block,
            re.I,
        )
        if dx_match:
            raw = dx_match.group(1)
            icd_match = re.search(r"[A-Za-z]\d{2}(?:\.\d+)?", raw, re.I)
            if icd_match:
                diagnosis = icd_match.group(0).upper()

        insurance_match = re.search(r"Primary Insurance\s*:\s*([^\n]+)", block)
        primary_insurance = insurance_match.group(1).strip() if insurance_match else ""

        ins_id_match = re.search(r"Insurance ID\s*:\s*([A-Z0-9]+)", block, re.I)
        insurance_id = ins_id_match.group(1).strip() if ins_id_match else ""

        date_match = re.search(rf"Session Date\s*:\s*{DATE_RE}", block, re.I)
        session_date = normalize_date(date_match.group(1)) if date_match else ""

        raw_session_time = ""
        m_time = re.search(r"(?im)^\s*Session Time\s*:\s*(.*)\s*$", block)
        if m_time:
            raw_session_time = m_time.group(1).strip()
            if raw_session_time in ("-", "–", "—"):
                raw_session_time = ""

        session_time = normalize_time_range(raw_session_time)

        location_match = re.search(r"Session Location\s*:\s*([^\n]+)", block)
        session_location = location_match.group(1).strip() if location_match else ""

        present_text = ""
        pos = block.lower().find("present at session")
        if pos != -1:
            present_text = block[pos: pos + 400]

        present_client = bool(re.search(r"\bClient\b", present_text, re.I))
        present_bt = bool(re.search(r"\b(BT/RBT|RBT/BT)\b", present_text, re.I))
        present_caregiver = bool(re.search(r"\bAdult Caregiver\b", present_text, re.I))
        present_sibling = bool(re.search(r"\bSibling(s)?\b", present_text, re.I))

        maladaptive_section = ""
        section_match = re.search(
            r"Maladaptive Status\s*:\s*(.*?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
            block,
            re.S,
        )
        if section_match:
            maladaptive_section = section_match.group(1).strip()

        maladaptive_behaviors = []
        if maladaptive_section:
            for line in maladaptive_section.splitlines():
                clean = line.strip()
                if not clean:
                    continue

                lower = clean.lower()
                if (
                    lower.endswith(":")
                    or "continues to display" in lower
                    or "in the following areas" in lower
                    or "maladaptive status" in lower
                    or "other maladaptive behaviors" in lower
                ):
                    continue

                clean = re.sub(r"[•▪◦\-–—\uf0b7\uf0a7]+", "", clean).strip()

                if len(clean.split()) > 6 and not clean.lower().startswith("other"):
                    continue

                maladaptive_behaviors.append(clean.lower())

        other_selected = any(b == "other" or b.startswith("other ") for b in maladaptive_behaviors)
        other_desc_match = re.search(r"Other maladaptive behaviors\s*:\s*(.+)", block, re.I)
        other_maladaptive_present = bool(other_desc_match and other_desc_match.group(1).strip())

        data_rows = re.findall(
            r"\n\s*([A-Za-z][A-Za-z0-9\s,.\-''()/]+?)\s+([0-9]{1,3})\s+([0-9]{1,3})\s*%?\s*(?=\n|$)",
            block
        )
        data_collected = len(data_rows) > 0

        summary_match = re.search(
            r"(Session Summary|Summary of Session)\s*:\s*(.+?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
            block,
            re.I | re.S,
        )
        session_summary_present = bool(summary_match and summary_match.group(2).strip())

        bt_attestation_present = bool(
            re.search(
                r"\battest\s+that\s+the\s+session\s+summary\s+is\s+accurate\s+and\s+correct\b",
                block,
                re.I,
            )
        )

        revision_attestation_present = bool(
            re.search(
                r"I\s+attest\s+the\s+revision/edit\s+made\s+to\s+this\s+note\s+as\s+signed\s+below\s+is\s+accurate\s+and\s+true",
                block,
                re.I,
            )
        )

        outcome_yes = bool(re.search(r"Outcome of Treatment.*?:\s*Yes", block, re.I | re.S))

        sig_valid, sig_cands = any_signature_line_valid(block)
        provider_signature_present = len(sig_cands) > 0
        provider_signature_valid = sig_valid

        compliance_errors = []

        if not dob:
            compliance_errors.append("Missing DOB")
        if not gender:
            compliance_errors.append("Missing Gender")
        if not diagnosis:
            compliance_errors.append("Missing ICD-10")
        if diagnosis and len(diagnosis) < 3:
            compliance_errors.append("Invalid ICD-10 code (truncated)")
        if not primary_insurance:
            compliance_errors.append("Missing Primary Insurance")
        if not insurance_id:
            compliance_errors.append("Missing Insurance ID")
        if not session_time:
            compliance_errors.append("Missing Session Time")
        if not session_location:
            compliance_errors.append("Missing Session Location")
        if not maladaptive_behaviors:
            compliance_errors.append("No maladaptive behaviors listed")
        if other_selected and not other_maladaptive_present:
            compliance_errors.append("Other maladaptive behavior selected but no description provided")
        if not data_collected:
            compliance_errors.append("Missing Session Data")
        if not session_summary_present:
            compliance_errors.append("Missing session summary narrative")
        if not outcome_yes:
            compliance_errors.append("Outcome of Treatment not Yes")
        if not bt_attestation_present:
            compliance_errors.append("Missing BT/RBT attestation statement")
        if not provider_signature_present:
            compliance_errors.append("Missing provider signature section")
        elif not provider_signature_valid:
            compliance_errors.append("Provider signature present but invalid format (must include Name, BT/RBT, and date)")
        if not present_client:
            compliance_errors.append("Attendance: Client not present")
        if not present_bt:
            compliance_errors.append("Attendance: BT/RBT not present")
        if not (present_caregiver or present_sibling):
            compliance_errors.append("Attendance: Parent/Caregiver or Sibling not present")

        results.append(
            {
                "Client": client_name,
                "Rendering Provider": provider,
                "Session Date": session_date,
                "Date of Birth": dob,
                "Gender": gender,
                "Diagnosis Code": diagnosis,
                "Primary Insurance": primary_insurance,
                "Insurance ID": insurance_id,
                "Session Time": session_time,
                "Session Location": session_location,
                "Present_Client": present_client,
                "Present_Adult_Caregiver": present_caregiver,
                "Present_Sibling": present_sibling,
                "Present_BT_RBT": present_bt,
                "Maladaptive Behaviors": maladaptive_behaviors,
                "Other Selected": other_selected,
                "Other Maladaptive Provided": other_maladaptive_present,
                "Outcome Yes": outcome_yes,
                "Data Collected": data_collected,
                "Session Summary Present": session_summary_present,
                "BT Attestation Present": bt_attestation_present,
                "Provider Signature Present": provider_signature_present,
                "Provider Signature Valid": provider_signature_valid,
                "Provider Signature Candidates": sig_cands,
                "Revision Attestation Present": revision_attestation_present,
                "Compliance Errors": compliance_errors,
                "PASS": len(compliance_errors) == 0,
            }
        )

    return results


def notes_to_excel_bytes(results, sheet_name="Notes") -> bytes:
    df = pd.DataFrame(results)
    df = excel_sanitize_df(df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()