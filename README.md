# AccentRoute

An English accent classifier and the data pipeline behind it, built to be evaluated
honestly rather than to produce a flattering number.

Frozen `whisper-small` encoder + LoRA + masked mean-pooling + a linear head, trained on
speaker-disjoint splits and tested both in-domain and on a different corpus with a
different speaking style.

## Results

Four native varieties, `whisper-small` encoder frozen, LoRA r=16 on the attention q/v
projections, three seeds (17/42/1337), one B200, 26 minutes.

| | macro-F1 |
| --- | --- |
| **In-domain** — 468 clips, **257 held-out speakers** | **0.753 ± 0.015** |
| **Out-of-domain** — EdAcc, supported classes | **0.639 ± 0.028** |
| In-domain control on the same class subset | 0.769 |
| Majority-class baseline | 0.141 |

Per-class F1, mean over seeds: `en-IN` 0.804, `en-US` 0.769, `en-GB` 0.735, `en-AU` 0.706.
No class is degenerate, and the confusion that remains is mostly `en-GB` ↔ `en-US`.

**The out-of-domain number is the one that matters.** Training data comes from a single
corpus, so the confounding report flags all four classes as source-dominated — the obvious
suspicion is that the model learned GLOBE's recording fingerprint rather than accent. If
that were true, EdAcc would collapse toward chance: it is a different corpus, different
recording conditions, and spontaneous conversation instead of read speech. It drops 0.130
and stays far above baseline, which is evidence the learned signal survives a change of
corpus and of speaking style.

EdAcc covers three of the four classes (it contains no Australian speakers at all), so its
figure is a supported-class macro-F1 over 545 clips from 20 speakers, quoted against an
in-domain control computed on that same subset rather than against the full four-class
number.

Reproduce with `scripts/globe_prepare.sbatch` then `scripts/train_eval.sbatch`; raw
numbers in `runs/accent_v1/results.json`.

## Why the evaluation is set up this way

Accent classification is unusually easy to get wrong, and the failure is invisible if you
only look at the score.

**Speaker leakage.** Split clips at random and the same voice lands in train and test. The
model can then score highly by recognizing people, not accents — and it collapses on a new
speaker. Every split here is speaker-disjoint: a speaker appears in exactly one of
train/val/test, asserted in the pipeline and in the tests.

That is not a hypothetical. The first run of this pipeline, on EdAcc with a single test
speaker per class, scored macro-F1 0.461 — which on inspection was three of five test
speakers memorized and two never recognized at all. That run is written up in
[docs/smoke-test-edacc.md](docs/smoke-test-edacc.md), including why its number is not
evidence of anything.

**Source confounding.** When each class comes from one corpus, a model can read the
microphone, the room, or the clip length instead of the accent — and a speaker-disjoint
split does nothing about it. `accentroute report` emits a source × accent matrix and flags
classes dominated by a single source, plus classes whose clip-duration range does not
overlap any other class's (duration is a source fingerprint in disguise).

**Out-of-domain testing.** Training data is read speech; EdAcc is spontaneous
conversation from a different corpus. The gap between in-domain and EdAcc is what says
whether the model learned accent or learned read-speech accent. EdAcc covers only part of
the label set, so its figure is a supported-class macro-F1 reported next to an in-domain
control on that same subset — never against a full-label-set number.

**Wording.** Reported intervals cover test-set sampling, not training randomness, so no
result here is upgraded into a significance claim — the interval does not carry the
evidence for one. `eval.tables.check_wording` enforces the rule against report text and
against these docs in CI, and it has rejected two drafts of this README.

## Taxonomy

Native varieties `en-US` `en-GB` `en-AU` `en-IN`, plus L2 accents by the speaker's first
language `L1-Mandarin` `L1-Spanish` `L1-Korean` `L1-Arabic`. Anything outside the eight is
dropped and counted (`configs/taxonomy_v1.yaml`, a versioned whitelist).

The four L2 classes need L2-ARCTIC, whose access form is still pending, so the current
model covers the four native varieties. The pipeline and the training script are
label-set agnostic — adding the L2 data is a re-run, not a rewrite.

## Data

| Source | Role | License | Access |
| --- | --- | --- | --- |
| GLOBE (`MushanW/GLOBE_V2`) | training, four native varieties | CC0 | open on Hugging Face |
| EdAcc | out-of-domain test only | CC-BY-SA | open on Hugging Face |
| L2-ARCTIC | the four L2 classes | CC BY-NC 4.0 | request form, pending |

GLOBE is Common-Voice-derived and is the only reachable corpus that contains **en-AU** at
all — EdAcc has zero Australian speakers. It is streamed under a per-class quota and a
per-speaker cap: a class quota filled by a handful of voices would train a speaker
recognizer, which is the failure mode above.

Two biases it carries, both recorded here rather than buried: it is a TTS corpus curated
for clean audio, so it does not represent in-the-wild recordings; and V2 supersamples to
44.1 kHz, so bandwidth above Common Voice's original rate carries no information.

Audio is never redistributed from this repo. `data/` and credentials are gitignored.

## Running it

```bash
uv sync --group dev --extra audio --extra ml

python scripts/globe_pipeline.py fetch 10000 15   # stream a balanced subset
python scripts/globe_pipeline.py pipeline         # filter, dedup, speaker-disjoint split
python scripts/globe_pipeline.py report           # leakage audit + confounding matrices
python scripts/train_eval.py train 1500           # three seeds
python scripts/train_eval.py evaluate             # in-domain + out-of-domain
```

On Slurm, `scripts/globe_prepare.sbatch` (cpu partition, no GPU) and
`scripts/train_eval.sbatch` (one GPU) do the same.

The pipeline is seven stages — ingest → taxonomy → filter → dedup/split → weak-label →
augment → emit — each a pure function from one Parquet manifest to the next, validated
against a stage schema. Three invariants are machine-enforced in `schema.py`: rejected
rows carry a reason, weak labels never reach val/test, and accepted YouTube rows carry an
evidence level.

## Designed and tested, not yet run

Built and unit-tested but waiting on data or time. Listed here so the code is not mistaken
for a claim:

- **Budget-matched three-arm ablation** (`configs/arms/`) — measures what weakly labeled
  data is worth, using a control arm matched on optimizer steps so the gain cannot be
  attributed to a bigger training budget.
- **Weak labeling with Qwen2-Audio** (`weaklabel/`) — evidence levels, a consensus rule, a
  three-pool blind audit, a pinned model revision and prompt hash.
- **Speaker-stratified bootstrap** (`eval/bootstrap.py`) and the **Vox-Profile external
  axis** (`eval/external.py`).
- **Leakage demo** (`scripts/leakage_demo.py`) — trains the same model on the same clips
  under a speaker-disjoint and a deliberately leaky split to measure the inflation.

## Development

```bash
uv run pytest -q       # 219 tests
uv run ruff check .
```

CI installs core dependencies only and never downloads a model: every model call is
injected, tests run on synthetic fixtures, and torch-dependent tests skip automatically.

Design notes are in `docs/`.
