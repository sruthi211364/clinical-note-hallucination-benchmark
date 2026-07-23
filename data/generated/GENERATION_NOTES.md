# Stage 3 generation notes (pilot run, `valid` split)

## Setup

- API: Google Gemini (free tier), via `google-genai`.
- Model: `gemini-flash-lite-latest` (currently resolves to `gemini-3.5-flash-lite`).
- Three prompting strategies per encounter: `zero_shot`, `schema_guided`, `few_shot`
  (see [generate_notes.py](../../src/generation/generate_notes.py) for exact prompts).
- Sampling: `temperature=0.3`, `max_output_tokens=4096`.

## Known issue: mixed model version in this pilot batch

The pilot started against the `gemini-flash-latest` alias, which resolved to
**`gemini-3.6-flash`**. That model's free-tier quota turned out to be a hard **20
requests/day**, which we hit partway through (see the `429 RESOURCE_EXHAUSTED` /
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` errors in the run log). The
script was switched mid-pilot to the `gemini-flash-lite-latest` alias
(-> `gemini-3.5-flash-lite`), which has a much higher usable daily quota and
completed the remaining calls in one sitting.

Net effect: **this pilot's 60 notes are not all from the same model version.**

| Strategy | gemini-3.6-flash | gemini-3.5-flash-lite |
|---|---|---|
| zero_shot | 4 | 16 |
| schema_guided | 3 | 17 |
| few_shot | 3 | 17 |

This is a genuine confound for comparing prompting strategies within this pilot
batch -- don't use it as-is for Stage 5 annotation. It's fine as a pipeline
validation run (confirms the API integration, retry/resume logic, and prompt
formats all work end-to-end and produce well-formed SOAP-structured notes).

## Plan for the real Stage 3 run (full 207-encounter corpus)

- Fix the model to `gemini-flash-lite-latest` for the **entire** run from the start
  (already the script default) -- no mid-run model switching.
- 207 encounters x 3 strategies = 621 calls. Given free-tier daily quotas, this will
  likely need to be paced across multiple days (`--limit` / resume-by-skipping
  already lets the script pick up where it left off across separate invocations).
- Every generated record carries `model_version` from the response, so if a quota
  switch ever happens again mid-run, it's caught in the data rather than assumed away.
