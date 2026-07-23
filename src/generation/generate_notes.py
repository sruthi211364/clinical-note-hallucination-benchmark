"""Stage 3: generate candidate clinical notes from ACI-Bench transcripts via the Gemini API.

Three prompting strategies, all producing a note from the transcript alone (the model
never sees the gold note):

  zero_shot       Minimal instruction: write a clinical note from this transcript.
  schema_guided   Same, but explicitly enumerates the SOAP section headers to use,
                  matching the canonical section_key taxonomy from Stage 2 cleaning
                  so downstream fact-alignment (Stage 5+) can key off known headers.
  few_shot        schema_guided plus one worked transcript->note example (from the
                  train split, so it never overlaps with what's being generated).

Uses the `gemini-flash-latest` alias rather than a pinned version string: pinned
IDs like gemini-2.5-flash returned 404 "no longer available to new users" for this
account even though they still appear in ListModels, while the *-latest alias
resolved successfully (to gemini-3.6-flash at the time this was written).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "generated"

MODEL = "gemini-flash-lite-latest"
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.3  # clinical documentation should be consistent, not creative

SECTION_HEADERS = [
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS",
    "PAST MEDICAL HISTORY", "SURGICAL HISTORY", "SOCIAL HISTORY", "FAMILY HISTORY",
    "MEDICATIONS", "ALLERGIES", "VITALS", "PHYSICAL EXAM", "RESULTS",
    "ASSESSMENT", "PLAN", "INSTRUCTIONS",
]

BASE_INSTRUCTION = (
    "You are an ambient clinical scribe. Below is a transcript of a doctor-patient "
    "office visit, with each line tagged by speaker. Write the clinical visit note "
    "a physician would produce from this conversation.\n\n"
    "Write only the note itself -- no preamble, no commentary, no markdown formatting. "
    "Base every statement strictly on what was said in the transcript; do not invent "
    "details, and do not omit clinically relevant information that was discussed."
)

SCHEMA_INSTRUCTION = (
    BASE_INSTRUCTION + "\n\n"
    "Organize the note under these section headers, in this order, using only the "
    "sections that are actually relevant to this visit (omit any that don't apply; "
    "you may combine ASSESSMENT and PLAN into one 'ASSESSMENT AND PLAN' section if "
    "that fits the visit better):\n" + "\n".join(SECTION_HEADERS)
)


def format_dialogue(turns: list[dict]) -> str:
    return "\n".join(f"[{t['speaker']}] {t['text']}" for t in turns)


def load_split(split: str) -> list[dict]:
    path = PROCESSED_DIR / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_few_shot_example() -> str:
    """One worked example from train (D2N001), never from the split being generated."""
    train = load_split("train")
    example = next(e for e in train if e["encounter_id"] == "D2N001")
    return (
        "Here is one example of a transcript and the note a physician wrote from it.\n\n"
        "--- EXAMPLE TRANSCRIPT ---\n" + format_dialogue(example["dialogue"]) + "\n\n"
        "--- EXAMPLE NOTE ---\n" + example["note_raw"] + "\n\n"
        "Now do the same for the following new transcript. Do not reuse any names, "
        "ages, or clinical details from the example above -- it is only a demonstration "
        "of style and structure.\n"
    )


def build_prompt(strategy: str, dialogue_text: str, few_shot_prefix: str | None) -> str:
    if strategy == "zero_shot":
        return f"{BASE_INSTRUCTION}\n\n--- TRANSCRIPT ---\n{dialogue_text}"
    if strategy == "schema_guided":
        return f"{SCHEMA_INSTRUCTION}\n\n--- TRANSCRIPT ---\n{dialogue_text}"
    if strategy == "few_shot":
        assert few_shot_prefix is not None
        return f"{SCHEMA_INSTRUCTION}\n\n{few_shot_prefix}\n--- TRANSCRIPT ---\n{dialogue_text}"
    raise ValueError(f"unknown strategy: {strategy}")


RETRYABLE_CODES = {429, 500, 503}


class DailyQuotaExhausted(Exception):
    """Free-tier per-day quota hit -- not worth backing off for, unlike per-minute limits."""


def generate_with_retry(client: genai.Client, prompt: str, max_retries: int = 6):
    config = types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    delay = 5.0
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt, config=config)
        except genai_errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                raise DailyQuotaExhausted(str(e)) from e
            if e.code in RETRYABLE_CODES and attempt < max_retries - 1:
                print(f"  {e.code} ({e.status}), backing off {delay:.0f}s...", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid")
    parser.add_argument("--strategies", nargs="+",
                         default=["zero_shot", "schema_guided", "few_shot"])
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N encounters (for piloting)")
    parser.add_argument("--sleep", type=float, default=3.0,
                         help="seconds to sleep between calls (free-tier rate limiting)")
    args = parser.parse_args()

    client = genai.Client()
    encounters = load_split(args.split)
    if args.limit:
        encounters = encounters[: args.limit]

    few_shot_prefix = build_few_shot_example() if "few_shot" in args.strategies else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for strategy in args.strategies:
        out_path = OUTPUT_DIR / f"{strategy}_{args.split}.jsonl"

        # Resume support: skip encounters already completed in a prior (possibly
        # crashed) run rather than re-spending calls on them.
        done_ids = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done_ids.add(json.loads(line)["encounter_id"])

        n_done = 0
        quota_hit = False
        with open(out_path, "a", encoding="utf-8") as out_f:
            for i, enc in enumerate(encounters):
                if enc["encounter_id"] in done_ids:
                    continue
                dialogue_text = format_dialogue(enc["dialogue"])
                prompt = build_prompt(strategy, dialogue_text, few_shot_prefix)
                print(f"[{strategy}] {i+1}/{len(encounters)} {enc['encounter_id']}", file=sys.stderr)
                try:
                    resp = generate_with_retry(client, prompt)
                except DailyQuotaExhausted:
                    print(f"  daily free-tier quota exhausted for {MODEL} -- "
                          f"stopping here, {n_done} new notes saved this run.", file=sys.stderr)
                    quota_hit = True
                    break
                usage = resp.usage_metadata
                record = {
                    "encounter_id": enc["encounter_id"],
                    "challenge_split": enc["challenge_split"],
                    "workflow_type": enc["workflow_type"],
                    "strategy": strategy,
                    "model": MODEL,
                    "model_version": resp.model_version,
                    "generated_note": resp.text,
                    "prompt_token_count": usage.prompt_token_count if usage else None,
                    "candidates_token_count": usage.candidates_token_count if usage else None,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                time.sleep(args.sleep)
        print(f"{strategy}: {len(done_ids) + n_done} total notes ({n_done} new) -> {out_path}",
              file=sys.stderr)
        if quota_hit:
            print(f"Stopping remaining strategies too -- re-run this same command "
                  f"tomorrow (or after the quota window resets) to resume.", file=sys.stderr)
            return


if __name__ == "__main__":
    main()
