# AccentRoute

A multi-source data pipeline and weak-supervision ablation for 8-class English accent
recognition. The centre of gravity is the **data pipeline** — multi-source integration,
LLM weak labeling, quality control, and evaluation design. The model side is deliberately
standard (frozen whisper-small encoder + LoRA) and serves only to validate the data work.

**Status:** pipeline and evaluation machinery are complete and unit-tested, and the whole
chain has been run end to end on real EdAcc audio on a B200 — see
[the smoke test](docs/smoke-test-edacc.md), which also explains why its macro-F1 is not
evidence of accent recognition. The designed experiment still needs the Common Voice and
L2-ARCTIC data; see [Gates](#gates).

## Taxonomy (8 classes, locked)

Native varieties `en-US` `en-GB` `en-AU` `en-IN`; L2 accents by the speaker's first
language `L1-Mandarin` `L1-Spanish` `L1-Korean` `L1-Arabic`. Anything that does not map
into these eight is dropped and counted (`configs/taxonomy_v1.yaml`, a versioned
whitelist).

## Headline number: a budget-matched three-arm ablation

| Arm | Data | Training budget |
| --- | --- | --- |
| A `a_gold` | gold + self-reported | epoch-matched |
| B `b_gold_oversampled` | gold + self-reported | **identical optimizer-step count to C** |
| C `c_gold_weak` | gold + self-reported + accepted weak labels | epoch-matched (defines S_C) |

The headline is **C − B**, which isolates the value of the weak-labeled data itself.
A − B is reported separately to isolate the training-budget effect.

Arm B exists because the naive comparison is unfair: a gold+weak run has more samples,
more optimizer steps, and a different audio domain all at once, so a gain cannot be
attributed to label quality. All three arms share the sampler, augmentation, LR schedule
and checkpoint rule in `configs/train_common.yaml`; if an arm config tries to override any
shared field, the config loader raises rather than silently running an unfair comparison.

## Statistics and wording discipline

Three seeds per arm (17/42/1337). Two quantities are reported **separately and never
merged**:

1. A class-stratified **test-speaker bootstrap 95% CI** — speakers are resampled with
   replacement within each class, so a rare class can never vanish from a resample.
2. **Seed variability** — the per-seed paired deltas and their standard deviation.

The interval covers test-set sampling but *not* training randomness, so the only claim
this project makes is `test-speaker bootstrap CI excludes zero`. It never says
"statistically significant". That discipline is enforced in code, not by convention:
`AblationStats` has no `significant` field, and `eval.tables.check_wording` rejects any
report text containing the banned phrasing.

## Known validity limits (stated, not hidden)

- **Source-label confounding.** When a class correlates with a data source, a model can
  learn the microphone and the recording environment instead of the accent — and
  speaker-disjoint splitting does nothing to prevent that. Four defenses: the
  source × accent matrix and `confounded` flags from `accentroute report`; test sets drawn
  from ≥2 sources per class where possible; per-source per-class F1 in the results tables;
  and a leave-one-source-out diagnostic for the L2 classes. Classes dominated by a single
  source are flagged in the datasheet and their conclusions are qualified.
- **L2 speaker scarcity.** L2-ARCTIC provides only four gold speakers per L1, so after
  speaker-disjoint splitting the L2 test strata are thin and their intervals are wide.
  Reported as such.
- **Weak-label circularity.** Qwen2-Audio is both the weak-label source and a zero-shot
  baseline. Mitigations: a pinned model revision and prompt SHA-256; a blind human audit
  covering the accepted pool *and* the reject pool (so filter selection bias is visible);
  and a kill rule that drops any class whose audited precision falls below 0.80. The
  headline C − B comparison is unaffected because both arms are scored on the same
  gold test set.
- **Dedup scope.** In scope: speaker merging within the YouTube set and near-duplicate
  detection within Common Voice. Global cross-source speaker clustering is backlog; the
  residual risk is recorded in the datasheet.

## Pipeline

```
ingest → taxonomy → filter → dedup/split → weak-label → augment → emit
```

Every stage is a pure function of the form "read a Parquet manifest → transform →
validate against the stage schema → write a new Parquet". Three invariants are
machine-enforced by `src/accentroute/schema.py`:

- rejected rows always carry a `reject_reason`
- **`label_source == "weak"` never appears in val/test/ood_test**
- accepted YouTube rows must carry an E1/E2 evidence level

```bash
uv sync --group dev                             # core dependencies
uv sync --group dev --extra audio --extra ml    # add the audio and training stacks

uv run accentroute ingest common-voice
uv run accentroute filter data/manifests/raw_common_voice.parquet data/manifests/qc.parquet
uv run accentroute split data/manifests/qc.parquet data/manifests/split.parquet
uv run accentroute report data/manifests/split.parquet     # G1 confounding matrix
uv run accentroute emit data/manifests/split.parquet c_gold_weak data/datasets/c_gold_weak
```

GPU stages (Qwen2-Audio weak labeling, training) run on a single-GPU Slurm partition:

```bash
sbatch scripts/weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>
sbatch scripts/train.sbatch c_gold_weak 17
sbatch scripts/train.sbatch b_gold_oversampled 17 <S_C>   # B must match C's step count
bash scripts/watch_jobs.sh <jobid>...
```

`scripts/run_experiments.py` encodes that ordering as Slurm dependencies so the
budget-matching cannot be got wrong by hand.

## Data sources and licensing

| Source | Role | License | Access |
| --- | --- | --- | --- |
| Common Voice (English) | main training set, self-reported accents | CC0 | Mozilla Data Collective account + API key (**no longer distributed via Hugging Face** as of Oct 2025) |
| L2-ARCTIC | gold labels for the four L2 classes | CC BY-NC 4.0 | request form; **audio is not redistributed here** |
| EdAcc | **out-of-domain test only** | CC-BY-SA | open on Hugging Face (`edinburghcstr/edacc`) |
| YouTube interviews | weak-label expansion, train split only | — | **only URLs, timestamps and labels are published here**; audio is fetched locally |

VCTK and the Speech Accent Archive are backlog. `data/` and credentials (`.env`) are
gitignored.

### EdAcc out-of-domain coverage

Measured from the corpus metadata (122 speakers, 19,137 segments): EdAcc supports only
part of the taxonomy. `en-AU` has **no** speakers at all, and `L1-Korean` (1) and
`L1-Arabic` (2) fall below the five-speaker floor. Out-of-domain results are therefore
reported as **supported-class macro-F1** over the covered subset, paired with an
in-domain control computed on that same subset, and are never compared against a full
eight-class number.

## Gates

- **G1** — gold manifest with ≥200 clips per class; ≥20 speakers for native classes and
  ≥8 for L2 classes; leakage audit clean; confounding matrix reviewed; EdAcc
  supported-class set fixed.
- **G2** — arm A reproducible across three seeds and beating the majority-class baseline;
  weak-label pipeline running end to end with a measured acceptance rate.
- **G3** — three-arm ablation and out-of-domain numbers final. Only then does any
  release work start.

## Development

```bash
uv run pytest -q       # 176 tests
uv run ruff check .
```

CI installs core dependencies only and **never downloads a model**: every model call is
injected as a parameter, unit tests run against synthetic fixtures, and tests requiring
torch skip automatically.

The design spec and implementation plan live in `docs/superpowers/`.
