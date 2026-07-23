# ACI-Bench data notes (Stage 2)

Source: [wyim/aci-bench](https://github.com/wyim/aci-bench), CC BY 4.0. Downloaded verbatim
into `raw/aci-bench/`; cleaned into `processed/*.jsonl` by `src/data_prep/clean_aci_bench.py`.
Validated by `src/data_prep/validate_processed.py`.

## Schema

Each line of `processed/aci_bench_clean.jsonl` (and the per-split files) is one encounter:

- `encounter_id`, `challenge_split` (which CSV it came from: train/valid/test1/test2/test3),
  `workflow_type` (capture method: `aci` / `virtassist` / `virtscribe` — this is what the raw
  `dataset` column actually encodes; renamed to avoid confusion with `challenge_split`)
- `patient`: age, gender, first/family name (from the metadata CSV)
- `chief_complaint_meta`, `secondary_complaints`: from metadata, distinct from the note's own
  CHIEF COMPLAINT section text
- `dialogue`: list of `{turn_id, speaker, text}`, one entry per transcript line. `speaker` is
  one of `doctor`, `patient`, `patient_guest`, or (once, see below) `unknown`
- `note_raw`: the original note text, unmodified
- `note_sections`: list of `{order, header_raw, section_key, soap_bucket, text}`, parsed from
  `note_raw`'s ALL-CAPS section headers, normalized to a canonical key and coarse S/O/A/P bucket

## Known data-quality caveats (carry these into Stage 5 annotation)

1. **ASR speaker-tag swaps.** The upstream README states some subsets have doctor/patient
   tags swapped by the ASR pipeline, left uncorrected intentionally. Not fixed here — no
   reliable way to detect this without a separate trained model. Annotators in Stage 5 should
   read for speaker plausibility rather than trusting the `speaker` field blindly on
   inconsistent-sounding turns.
2. **One missing speaker tag.** `D2N131`'s first dialogue line has no `[doctor]`/`[patient]`
   tag in the source data. Kept as `speaker: "unknown"` rather than guessed — context (a
   greeting immediately followed by a `[patient]` reply) makes it very likely `doctor`, but
   the cleaning script does not infer it.
3. **One inline ASR artifact.** `D2N138` contains a mid-utterance `[ inaudible HH:MM:SS ]`
   marker; it's folded into the surrounding turn's text rather than treated as a speaker
   change.
4. **One structural outlier.** `D2N123` uses a bare `SUBJECTIVE` header instead of the usual
   `CHIEF COMPLAINT` / `HISTORY OF PRESENT ILLNESS` split; mapped to `section_key:
   subjective_narrative`, `soap_bucket: subjective`.
5. **Section coverage is ~99% by character count**, verified against `note_raw`; the residual
   ~1% is section header text itself plus incidental whitespace differences, not dropped
   clinical content.

## Corpus stats (207 encounters)

- Workflow type: `aci` 112, `virtassist` 55, `virtscribe` 40
- Dialogue turns per encounter: min 7, max 136, mean 55.2
- Note length: min 852 chars, max 5712 chars, mean 2687 chars
- Most common note sections: physical_exam (200), chief_complaint (192), results (158),
  review_of_systems (157), history_of_present_illness (153)
- Least common: procedure (2), subjective_narrative (1 — the D2N123 outlier)
