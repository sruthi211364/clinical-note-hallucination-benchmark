# Clinical Note Hallucination Benchmark and Detector

Measuring and detecting factual inconsistencies (fabrications, omissions, contradictions)
in LLM-generated clinical notes, relative to the doctor-patient conversation they were
generated from.

## Motivation

LLMs writing clinical notes from visit transcripts can produce text that is fluent but
unfaithful to the source conversation: inventing symptoms that were never mentioned,
dropping details that were, or reversing something like a dosage or a symptom's polarity.
Strong performance on medical knowledge benchmarks does not guarantee faithful note
generation from a real conversation. This project builds the machinery to measure that
gap: a labeled dataset of annotated errors, an automatic judge, and a lightweight
fine-tuned detector, evaluated against human annotation.

## Data source

[ACI-Bench](https://github.com/wyim/aci-bench) (Yim et al., 2023, *Scientific Data*),
CC BY 4.0. 207 real doctor-patient encounters (ASR-transcribed) paired with full SOAP
notes, spanning three capture workflows: ambient scribe, virtual assistant, and virtual
scribe.

> Yim, W.W., Fu, Y., Ben Abacha, A. et al. Aci-bench: a Novel Ambient Clinical
> Intelligence Dataset for Benchmarking Automatic Visit Note Generation.
> Sci Data 10, 586 (2023).

## Project stages

1. Source selection — done, see above.
2. Data cleaning — this stage. Normalizes transcript/note formatting into a unified schema.
3. Candidate note generation via LLM APIs (multiple prompting strategies).
4. Error taxonomy definition (fabrication / omission / contradiction).
5. Manual labeled dataset construction.
6. Automatic LLM-judge for error detection, benchmarked against human labels.
7. Classical NLP similarity metrics as a baseline comparison.
8. Lightweight LoRA-fine-tuned faithfulness detector.
9. Experiment tracking across prompting/judge/fine-tuning configurations.
10. Analysis and research write-up.
11. Demo app / API wrapping the detector.

## Repo layout

```
data/
  raw/aci-bench/        raw CSVs as published (not modified)
  processed/             cleaned, unified-schema JSONL output
src/
  data_prep/             cleaning + validation scripts (stage 2)
notebooks/                exploratory analysis
tests/                     unit tests for data_prep
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
