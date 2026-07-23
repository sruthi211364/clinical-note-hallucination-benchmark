"""Stage 3: generate candidate clinical notes from ACI-Bench transcripts.

Supports two free-tier providers behind one interface:

  groq    llama-3.3-70b-versatile -- ~1,000 requests/day free tier, used for the full
          207-encounter corpus (fast enough to finish in one sitting).
  gemini  gemini-flash-lite-latest -- used for the initial 20-encounter valid-split
          pilot; its free daily quota (~20-50/day depending on the model version
          Google routes you to) made the full corpus impractical in one sitting.
          See data/generated/GENERATION_NOTES.md for what that pilot found.

Three prompting strategies, all producing a note from the transcript alone (the model
never sees the gold note):

  zero_shot       Minimal instruction: write a clinical note from this transcript.
  schema_guided   Same, but explicitly enumerates the SOAP section headers to use,
                  matching the canonical section_key taxonomy from Stage 2 cleaning
                  so downstream fact-alignment (Stage 5+) can key off known headers.
  few_shot        schema_guided plus one worked transcript->note example (D2N001,
                  from train). D2N001 itself is skipped as a *generation target* for
                  this strategy -- generating its own few-shot exemplar back at itself
                  would be trivial leakage, not a real test of the strategy.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "generated"

FEW_SHOT_EXAMPLE_ID = "D2N001"

PROVIDER_MODELS = {
    # llama-3.1-8b-instant has a much larger daily budget (500K TPD vs 100K) but only a
    # 6K-tokens-per-request ceiling, which our longer transcripts (up to 136 turns) blow
    # past outright (413, not a retryable 429). llama-3.3-70b-versatile's per-request
    # ceiling is 12K -- comfortably covers every transcript in the corpus -- so it's the
    # safer choice even though its daily budget (100K TPD) means ~35-40 calls/day.
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-flash-lite-latest",
}
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
    train = load_split("train")
    example = next(e for e in train if e["encounter_id"] == FEW_SHOT_EXAMPLE_ID)
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


@dataclass
class GenResult:
    text: str
    model_version: str
    prompt_tokens: int | None
    output_tokens: int | None


class DailyQuotaExhausted(Exception):
    """Free-tier per-day quota hit -- not worth backing off for, unlike per-minute limits."""


class RequestTooLarge(Exception):
    """Single request exceeds the model's per-request token ceiling -- not retryable,
    but shouldn't crash the whole run; skip this one encounter and move on."""


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _make_client(provider: str):
    if provider == "groq":
        from groq import Groq
        return Groq()
    if provider == "gemini":
        from google import genai
        return genai.Client()
    raise ValueError(f"unknown provider: {provider}")


def _call_groq(client, model: str, prompt: str) -> GenResult:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    usage = resp.usage
    return GenResult(
        text=resp.choices[0].message.content,
        model_version=resp.model,
        prompt_tokens=usage.prompt_tokens if usage else None,
        output_tokens=usage.completion_tokens if usage else None,
    )


def _call_gemini(client, model: str, prompt: str) -> GenResult:
    from google.genai import types
    config = types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE)
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    usage = resp.usage_metadata
    return GenResult(
        text=resp.text,
        model_version=resp.model_version,
        prompt_tokens=usage.prompt_token_count if usage else None,
        output_tokens=usage.candidates_token_count if usage else None,
    )


def generate_with_retry(provider: str, client, model: str, prompt: str, max_retries: int = 6) -> GenResult:
    delay = 5.0
    for attempt in range(max_retries):
        try:
            if provider == "groq":
                return _call_groq(client, model, prompt)
            return _call_gemini(client, model, prompt)
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            message = str(e)
            # Normalize away spaces/underscores/case so this matches both providers'
            # phrasing: Gemini's "...PerDayPerProject..." and Groq's "tokens per day (TPD)".
            normalized = message.lower().replace(" ", "").replace("_", "")
            if status == 429 and "perday" in normalized:
                raise DailyQuotaExhausted(message) from e
            if status == 413:
                raise RequestTooLarge(message) from e
            if status in RETRYABLE_STATUS and attempt < max_retries - 1:
                print(f"  {status}, backing off {delay:.0f}s...", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--split", default="aci_bench_clean",
                         help="processed split filename stem; 'aci_bench_clean' is the full 207-encounter corpus")
    parser.add_argument("--strategies", nargs="+",
                         default=["zero_shot", "schema_guided", "few_shot"])
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N encounters (for piloting)")
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="seconds to sleep between calls (rate limiting)")
    args = parser.parse_args()

    # Guard against two invocations targeting the same (provider, split) writing to the
    # same output files concurrently -- observed once as silent duplicate generations
    # with no clear trigger; this makes a second overlapping run fail fast instead.
    lock_path = OUTPUT_DIR / f".lock_{args.provider}_{args.split}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Another run is already writing to {args.provider}/{args.split} outputs "
              f"(lock file exists: {lock_path}). If no such run is actually active, "
              f"delete the lock file and retry.", file=sys.stderr)
        sys.exit(1)
    os.close(lock_fd)

    try:
        model = PROVIDER_MODELS[args.provider]
        client = _make_client(args.provider)
        encounters = load_split(args.split)
        if args.limit:
            encounters = encounters[: args.limit]
        _run(args, model, client, encounters)
    finally:
        lock_path.unlink(missing_ok=True)


def _run(args, model, client, encounters):
    few_shot_prefix = build_few_shot_example() if "few_shot" in args.strategies else None

    for strategy in args.strategies:
        out_path = OUTPUT_DIR / f"{args.provider}_{strategy}_{args.split}.jsonl"

        # Resume support: skip encounters already completed in a prior (possibly
        # crashed) run rather than re-spending calls on them. Only trust records that
        # match the model currently configured -- a prior run under a different model
        # (e.g. a mid-project model switch) must not be silently treated as "done" and
        # smuggle a mixed-model confound back into the corpus.
        done_ids = set()
        stale = 0
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("model") == model:
                        done_ids.add(rec["encounter_id"])
                    else:
                        stale += 1
        if stale:
            print(f"  WARNING: {stale} existing rows in {out_path} were generated with a "
                  f"different model and will NOT count as done -- resolve this file manually "
                  f"before continuing (delete stale rows or the whole file).", file=sys.stderr)
            continue

        targets = [e for e in encounters if e["encounter_id"] not in done_ids]
        if strategy == "few_shot":
            targets = [e for e in targets if e["encounter_id"] != FEW_SHOT_EXAMPLE_ID]

        n_done = 0
        quota_hit = False
        with open(out_path, "a", encoding="utf-8") as out_f:
            for i, enc in enumerate(targets):
                dialogue_text = format_dialogue(enc["dialogue"])
                prompt = build_prompt(strategy, dialogue_text, few_shot_prefix)
                print(f"[{args.provider}/{strategy}] {i+1}/{len(targets)} {enc['encounter_id']}", file=sys.stderr)
                try:
                    result = generate_with_retry(args.provider, client, model, prompt)
                except DailyQuotaExhausted:
                    print(f"  daily free-tier quota exhausted for {model} -- "
                          f"stopping here, {n_done} new notes saved this run.", file=sys.stderr)
                    quota_hit = True
                    break
                except RequestTooLarge:
                    print(f"  {enc['encounter_id']}: transcript too large for {model}'s "
                          f"per-request limit -- skipping this encounter.", file=sys.stderr)
                    continue
                record = {
                    "encounter_id": enc["encounter_id"],
                    "challenge_split": enc["challenge_split"],
                    "workflow_type": enc["workflow_type"],
                    "strategy": strategy,
                    "provider": args.provider,
                    "model": model,
                    "model_version": result.model_version,
                    "generated_note": result.text,
                    "prompt_token_count": result.prompt_tokens,
                    "candidates_token_count": result.output_tokens,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                time.sleep(args.sleep)
        print(f"{strategy}: {len(done_ids) + n_done} total notes ({n_done} new) -> {out_path}",
              file=sys.stderr)
        if quota_hit:
            print("Stopping remaining strategies too -- re-run this same command "
                  "later to resume.", file=sys.stderr)
            return


if __name__ == "__main__":
    main()
