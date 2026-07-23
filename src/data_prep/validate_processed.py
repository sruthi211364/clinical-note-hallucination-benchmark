"""Stage 2: sanity-check the cleaned ACI-Bench output.

Not a general-purpose validation framework -- just the specific checks needed to trust
this dataset before it's used for note generation (Stage 3) and manual annotation
(Stage 5): every encounter parsed into turns and sections, no silently dropped note
content, and a coverage report on which raw section headers didn't map to a known key.
"""
import json
import sys
from collections import Counter
from pathlib import Path

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "aci_bench_clean.jsonl"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    encounters = load(PROCESSED)
    n = len(encounters)
    print(f"Loaded {n} encounters from {PROCESSED}")

    workflow_counts = Counter(e["workflow_type"] for e in encounters)
    split_counts = Counter(e["challenge_split"] for e in encounters)
    print(f"\nWorkflow type distribution: {dict(workflow_counts)}")
    print(f"Challenge split distribution: {dict(split_counts)}")

    zero_turn = [e["encounter_id"] for e in encounters if len(e["dialogue"]) == 0]
    zero_section = [e["encounter_id"] for e in encounters if len(e["note_sections"]) == 0]
    print(f"\nEncounters with zero dialogue turns: {len(zero_turn)} {zero_turn[:10]}")
    print(f"Encounters with zero note sections: {len(zero_section)} {zero_section[:10]}")

    unknown_speaker = [e["encounter_id"] for e in encounters
                       if any(t["speaker"] == "unknown" for t in e["dialogue"])]
    print(f"Encounters with an unrecognized leading speaker line: {len(unknown_speaker)} {unknown_speaker[:10]}")

    other_bucket = [e["encounter_id"] for e in encounters
                    if any(s["soap_bucket"] == "other" for s in e["note_sections"])]
    print(f"Encounters with an unmapped ('other') section (e.g. preamble text): {len(other_bucket)} {other_bucket[:10]}")

    # Reconstructed note length (header lines + body text) vs original note length, as a
    # coverage proxy: if section parsing silently dropped content, the reconstruction
    # will be shorter than note_raw. Headers are included since they're real note content
    # even though they're stored in a separate field.
    coverage_ratios = []
    for e in encounters:
        section_chars = sum(len(s["header_raw"]) + len(s["text"]) for s in e["note_sections"])
        raw_chars = len(e["note_raw"])
        coverage_ratios.append(section_chars / raw_chars if raw_chars else 1.0)
    low_coverage = [(e["encounter_id"], round(r, 2)) for e, r in zip(encounters, coverage_ratios) if r < 0.9]
    print(f"\nMean section-text/raw-note char coverage: {sum(coverage_ratios)/n:.3f}")
    print(f"Encounters with <90% coverage (possible parsing gaps): {len(low_coverage)}")
    for eid, ratio in low_coverage[:10]:
        print(f"  {eid}: {ratio}")

    section_key_counts = Counter(s["section_key"] for e in encounters for s in e["note_sections"])
    print(f"\nSection key frequency across corpus:")
    for key, cnt in section_key_counts.most_common():
        print(f"  {key}: {cnt}")

    turns_per_encounter = [len(e["dialogue"]) for e in encounters]
    print(f"\nDialogue turns per encounter: min={min(turns_per_encounter)} "
          f"max={max(turns_per_encounter)} mean={sum(turns_per_encounter)/n:.1f}")

    note_chars = [len(e["note_raw"]) for e in encounters]
    print(f"Note length (chars) per encounter: min={min(note_chars)} "
          f"max={max(note_chars)} mean={sum(note_chars)/n:.0f}")


if __name__ == "__main__":
    main()
