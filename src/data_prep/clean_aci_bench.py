"""Stage 2: clean and normalize raw ACI-Bench CSVs into a unified per-encounter JSONL schema.

Raw format (verified by direct inspection of the downloaded CSVs, not assumed):
  - {split}.csv columns: dataset, encounter_id, dialogue, note
    - `dataset` here means capture workflow (aci / virtassist / virtscribe), not the
      challenge split -- the challenge split is which CSV file the row came from.
    - `dialogue` is newline-separated utterances, each line `[speaker] text`.
      Only three speaker tags occur in the corpus: doctor, patient, patient_guest.
      One single line in the entire corpus (D2N138) has a mid-turn ASR artifact
      `[ inaudible HH:MM:SS ]` that is not a speaker tag; lines that don't match a
      known speaker tag are appended to the previous turn rather than starting a new one.
    - `note` is plain text with ALL-CAPS section headers on their own line (some with
      a trailing colon, some abbreviated, e.g. "CC:" for "CHIEF COMPLAINT").
  - {split}_metadata.csv columns: dataset, encounter_id, id, doctor_name, patient_gender,
    patient_age, patient_firstname, patient_familyname, cc, 2nd_complaints

Known, intentionally-uncorrected upstream limitation (documented in the ACI-Bench README):
some subsets have doctor/patient speaker tags swapped by the ASR process. This is not
fixed here -- it would require a separate learned correction model -- but is carried
forward into the data card so downstream annotation (Stage 5) can account for it.
"""
import csv
import json
import re
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "aci-bench"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

SPLITS = [
    "train",
    "valid",
    "clinicalnlp_taskB_test1",
    "clinicalnlp_taskC_test2",
    "clef_taskC_test3",
]

KNOWN_SPEAKERS = {"doctor", "patient", "patient_guest"}
TURN_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")

# Raw header text (upper-cased, trailing colon stripped) -> canonical section key.
SECTION_KEY_MAP = {
    "CHIEF COMPLAINT": "chief_complaint",
    "CC": "chief_complaint",
    "HISTORY OF PRESENT ILLNESS": "history_of_present_illness",
    "HPI": "history_of_present_illness",
    "REVIEW OF SYSTEMS": "review_of_systems",
    "REVIEW OF SYMPTOMS": "review_of_systems",
    "MEDICAL HISTORY": "past_medical_history",
    "PAST MEDICAL HISTORY": "past_medical_history",
    "PAST HISTORY": "past_medical_history",
    "SURGICAL HISTORY": "surgical_history",
    "PAST SURGICAL HISTORY": "surgical_history",
    "SOCIAL HISTORY": "social_history",
    "FAMILY HISTORY": "family_history",
    "MEDICATIONS": "medications",
    "CURRENT MEDICATIONS": "medications",
    "ALLERGIES": "allergies",
    "VITALS": "vitals",
    "VITALS REVIEWED": "vitals",
    "PHYSICAL EXAM": "physical_exam",
    "PHYSICAL EXAMINATION": "physical_exam",
    "EXAM": "physical_exam",
    "RESULTS": "results",
    "ASSESSMENT": "assessment",
    "IMPRESSION": "assessment",
    "ASSESSMENT AND PLAN": "assessment_and_plan",
    "PLAN": "plan",
    "INSTRUCTIONS": "instructions",
    "PROCEDURE": "procedure",
    # Bare SOAP-bucket headers used in place of a specific section header
    # (observed once in the corpus, D2N123, in lieu of CHIEF COMPLAINT/HPI).
    "SUBJECTIVE": "subjective_narrative",
}

# canonical section key -> SOAP bucket
SOAP_BUCKET_MAP = {
    "chief_complaint": "subjective",
    "history_of_present_illness": "subjective",
    "review_of_systems": "subjective",
    "past_medical_history": "subjective",
    "surgical_history": "subjective",
    "social_history": "subjective",
    "family_history": "subjective",
    "medications": "subjective",
    "allergies": "subjective",
    "vitals": "objective",
    "physical_exam": "objective",
    "results": "objective",
    "assessment": "assessment",
    "assessment_and_plan": "assessment_and_plan",
    "plan": "plan",
    "instructions": "plan",
    "procedure": "plan",
    "subjective_narrative": "subjective",
}

HEADER_LINE_RE = re.compile(r"^([A-Z][A-Z0-9 /'&\-]{1,58}):?$")


def parse_dialogue(raw_dialogue: str) -> list[dict]:
    turns = []
    for line in raw_dialogue.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = TURN_RE.match(line)
        if m and m.group(1).strip().lower() in KNOWN_SPEAKERS:
            turns.append({
                "turn_id": len(turns),
                "speaker": m.group(1).strip().lower(),
                "text": m.group(2).strip(),
            })
        elif turns:
            # Not a recognized speaker tag (e.g. an inline ASR artifact like
            # "[ inaudible 00:09:25 ]") -- fold into the current turn's text.
            turns[-1]["text"] = (turns[-1]["text"] + " " + line).strip()
        else:
            # Malformed leading line with no prior turn to attach to; keep it
            # rather than silently dropping content.
            turns.append({"turn_id": 0, "speaker": "unknown", "text": line})
    return turns


def parse_note(raw_note: str) -> list[dict]:
    sections = []
    current_header_raw = None
    current_key = None
    current_bucket = None
    buffer: list[str] = []

    def flush():
        if current_header_raw is not None:
            sections.append({
                "order": len(sections),
                "header_raw": current_header_raw,
                "section_key": current_key,
                "soap_bucket": current_bucket,
                "text": "\n".join(buffer).strip(),
            })

    for line in raw_note.split("\n"):
        stripped = line.strip()
        m = HEADER_LINE_RE.match(stripped) if stripped else None
        header_candidate = stripped.rstrip(":") if m else None
        if header_candidate in SECTION_KEY_MAP or (m and header_candidate in SECTION_KEY_MAP):
            flush()
            current_header_raw = stripped
            current_key = SECTION_KEY_MAP[header_candidate]
            current_bucket = SOAP_BUCKET_MAP[current_key]
            buffer = []
        elif current_header_raw is None:
            # Preamble text before the first recognized header (rare) -- keep it
            # under an explicit "preamble" bucket instead of discarding it.
            if stripped:
                current_header_raw = "PREAMBLE"
                current_key = "preamble"
                current_bucket = "other"
                buffer = [stripped]
        else:
            buffer.append(line)
    flush()
    return sections


def to_int_or_none(value):
    if value is None or value == "":
        return None
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def load_split(split: str) -> list[dict]:
    data_path = RAW_DIR / f"{split}.csv"
    meta_path = RAW_DIR / f"{split}_metadata.csv"

    with open(meta_path, newline="", encoding="utf-8") as f:
        meta_by_id = {row["encounter_id"]: row for row in csv.DictReader(f)}

    encounters = []
    with open(data_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            enc_id = row["encounter_id"]
            meta = meta_by_id.get(enc_id, {})
            secondary = meta.get("2nd_complaints") or ""
            encounters.append({
                "encounter_id": enc_id,
                "challenge_split": split,
                "workflow_type": row["dataset"],
                "patient": {
                    "age": to_int_or_none(meta.get("patient_age")),
                    "gender": meta.get("patient_gender") or None,
                    "first_name": meta.get("patient_firstname") or None,
                    "family_name": meta.get("patient_familyname") or None,
                },
                "chief_complaint_meta": meta.get("cc") or None,
                "secondary_complaints": [s.strip() for s in secondary.split(";") if s.strip()],
                "dialogue": parse_dialogue(row["dialogue"]),
                "note_raw": row["note"],
                "note_sections": parse_note(row["note"]),
            })
    return encounters


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_encounters = []
    for split in SPLITS:
        encounters = load_split(split)
        out_path = OUT_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for enc in encounters:
                f.write(json.dumps(enc, ensure_ascii=False) + "\n")
        print(f"{split}: {len(encounters)} encounters -> {out_path}", file=sys.stderr)
        all_encounters.extend(encounters)

    combined_path = OUT_DIR / "aci_bench_clean.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for enc in all_encounters:
            f.write(json.dumps(enc, ensure_ascii=False) + "\n")
    print(f"combined: {len(all_encounters)} encounters -> {combined_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
