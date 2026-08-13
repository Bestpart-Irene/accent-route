# AccentRoute Implementation Plan v1.2

> **How this runs:** a plain task workflow — work the tasks in order, one at a time (five TDD steps + a commit per task), with no dependency on any session-specific skill. First step after approval: save this plan to `docs/superpowers/plans/2026-08-12-accentroute-implementation.md`.
> **What changed in v1.2 (response to the second review round):** (1) source-label confounding controls (confounding matrix, multi-source test set, per-source stratified reporting, LOSO, confounded flags); (2) the three-arm fair ablation A/B/C; (3) statistics reworked into a stratified bootstrap, with seed variation reported separately and strict wording rules; (4) staged schemas; (5) scope reduction: the core is CV + L2-ARCTIC + EdAcc + a small YouTube set, with VCTK/SAA/global speaker clustering/HF release/demo deferred.

**Goal:** Build a multi-source data pipeline for 8-class English accent recognition plus a weakly supervised three-arm ablation (headline number C−B), with whisper-small + LoRA as the verification model, delivered as a credible 3-week part-time project.

**Architecture:** A 7-stage pipeline (ingest → taxonomy → filter → dedup/split → weak-label → augment → emit) in which every stage is a pure function of the form "read a Parquet manifest → transform → validate against the stage schema → write a new Parquet". On the model side: frozen whisper-small encoder + LoRA (r=16, q/v) + masked mean-pooling (valid frame count derived from the processor's attention mask) + a linear head.

**Tech Stack:** Python 3.11 + uv, pandera + pyarrow, Silero VAD (pip package), faster-whisper (tiny, int8), fastText LID, SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`, via `speechbrain.inference`), `Qwen/Qwen2-Audio-7B-Instruct` (pinned revision sha), transformers + peft, typer, pytest, GitHub Actions.

**Compute split:** the data pipeline runs locally on the Mac CPU; everything that needs a GPU (Qwen2-Audio inference at ~17 GB BF16, 3 arms × 3 seeds = 9 training runs plus 1 LOSO run, and the baselines) goes to **AICR rtx-batch** (p2026_0038_neu). Cluster interaction is explicit `scripts/*.sbatch` files plus `scripts/watch_jobs.sh` (squeue/sacct polling with failure alerts) checked into the repo — no skill is assumed. yt-dlp runs locally only, so audio never leaves this machine.

## Context

The design spec was approved and then went through two rounds of review. The first round produced 9 items (data lineage / leakage prevention / weak-label auditing / statistical design) that are now locked-in decisions. The second round exposed two gaps in experimental validity — **the model may learn the data source rather than the accent** (when a class is tightly bound to a single source, a speaker-disjoint split still does not block microphone, room, or reading-material shortcuts) and **the ablation comparison is unfair** (gold+weak simultaneously adds sample count, optimizer steps and audio domain) — plus an over-optimistic CI methodology and an overloaded scope. v1.2 folds all of it in. The repo currently contains nothing but the spec; everything is built from scratch.

## Verified external facts (web research, 2026-08-12)

1. **EdAcc**: CC-BY-SA; available on HF as `edinburghcstr/edacc` or from Edinburgh DataShare (10283/8983). Per-speaker fields include `accent` (linguist-normalized), `raw_accent` and `l1` (first language), so the 8-class mapping is feasible; the speaker count per class (Korean and Arabic especially) gets measured for real in W1. Only dev/test exist, no train, which lines up with using it purely out-of-domain.
2. **Qwen2-Audio**: `Qwen/Qwen2-Audio-7B-Instruct` (Apache-2.0, ~17 GB in BF16) → must run on a cluster GPU. Integrated in mainline transformers (`Qwen2AudioForConditionalGeneration` + `AutoProcessor` + `apply_chat_template`). Pinning path: `HfApi().model_info(...).sha` → `configs/weaklabel.yaml` → `from_pretrained(revision=sha)`.
3. **L2-ARCTIC**: CC BY-NC 4.0, TAMU request form (automatic reply with the link, turnaround in hours). 4 speakers each for Arabic/Mandarin/Korean/Spanish → **only 4 gold speakers per class, the tightest statistical constraint in the whole project**.
4. **Common Voice**: as of 2025-10 distribution goes exclusively through the Mozilla Data Collective; **the primary path is HF `common_voice_17_0`** (gated but click-through, CC0, free-text `accents` field); MDC registration starts on day 1 of W1 as an optional upgrade. The accent fill rate gets measured during ingest.
5. **API status**: use the Silero VAD pip package; use `speechbrain.inference.classifiers.EncoderClassifier` for ECAPA (`speechbrain.pretrained` is deprecated).

## Design decisions (both review rounds merged, all final)

1. **Three-arm fair ablation (headline number = C−B)**:
   - **A** gold-only (epoch-matched: the same number of epochs as C); **B** gold-only oversampled (**exactly the same optimizer step count as C**); **C** gold + accepted weak.
   - All three arms share `configs/train_common.yaml`: same augmentation, same class-balanced sampler, same LR schedule, **a fixed step budget in place of early stopping**, and the same checkpoint selection rule (val macro-F1). C−B ≈ the value of the weakly labeled data itself; A−B isolates the effect of the training budget.
2. **Statistical methodology (wording discipline)**: 3 seeds per arm (17/42/1337). The main report carries two quantities, **presented separately and never merged**:
   - **A class-stratified test-speaker bootstrap 95% CI**: the true classes are the strata, speaker clusters are resampled with replacement within each stratum (so a rare class can never vanish from a resample), and a percentile CI is computed on the seed-averaged Δmacro-F1;
   - **Seed variation**: mean ± std of the per-seed paired Δ.
   - Wording rule: the only permitted phrasing is "test-speaker bootstrap CI excludes zero"; **a blanket claim of statistically significant is forbidden** (it would require a seed × speaker hierarchical bootstrap, which is out of scope this cycle).
3. **Source-confounding controls**:
   - Before G1, produce a **source × accent matrix** (n_speakers, hours, n_clips); any class where more than 90% of the training hours come from a single source is marked `confounded=True`, recorded in the datasheet, and every conclusion about it must be hedged (no claiming pure accent capability);
   - Test set composition: aim for ≥2 sources per class; the results tables **report per-class F1 stratified by source**;
   - **LOSO diagnostic** (W3, single seed): train the four L2 classes with L2-ARCTIC removed (CV self-reported L2 only) and test on held-out L2-ARCTIC speakers — this quantifies cross-source generalization versus a source shortcut. EdAcc itself is a cross-source out-of-domain test covering all classes.
4. **Weak-label circularity**: pin the Qwen revision sha and version the prompt (8-way choice + `unsure`, k=3 self-consistency votes, k configurable down to 1); acceptance rules live in the consensus spec; **the audit covers all three pools**: 25 accepted clips per class plus 50 clips drawn from the rejected/review pool stratified by reject_reason (to understand filter selection bias and false rejects); the datasheet reports per-class precision (Wilson interval, with the n=25 noise stated explicitly); kill rule: if precision in the accepted pool falls below 0.80 for a class, that class's weak labels are dropped wholesale. Weak labels only enter train (machine-enforced by the schema). Qwen zero-shot is only a secondary comparison; the headline C−B uses the same gold test set on both sides.
5. **YouTube evidence levels**: `lists/youtube_v1.csv` is curated by hand (**small and strictly curated: target 300–600 clips**) with an evidence_level tag (E1 self-statement/birthplace; E2 channel region + content cues; E3 model only). Only E1/E2 clips whose Qwen majority vote matches the prior are accepted, and they only enter train.
6. **Dedup (scoped down in v1.2)**: the split key is `speaker_key` (defaulting to `f"{source}:{speaker_id_raw}"`). Three things stay in the core scope: (1) ECAPA dedup within the YouTube set (the same interviewee showing up across videos and channels; ANN candidate edges + union-find, an architecture that scales from day one); (2) near-duplicate detection within CV (transcript Jaccard ≥0.8 and |Δduration| ≤0.5 s); (3) threshold calibration using known same-speaker pairs from L2-ARCTIC/CV as positives and **cross-source pairings (CV × YouTube) as hard negatives**. Global cross-source speaker clustering goes to the backlog; the residual risk — the same person appearing in both CV and YouTube without being caught — is stated in the datasheet.
7. **Staged schemas**: `RawManifestSchema` → (after filter/taxonomy) `QCManifestSchema` → (after dedup/split/weaklabel) `SplitManifestSchema`, built with inheritance plus `validate_manifest(df, stage)` for per-stage validation; each invariant hangs off the stage where it first becomes meaningful. Naming discipline: `snr_proxy_db` (a single-channel proxy, not true SNR), `consensus_score` (majority vote × evidence weight, not a calibrated confidence), the baseline is called a **frozen ECAPA embedding probe**, and the metric computed after the EdAcc exclusions is called **supported-class macro-F1** (always reported next to the in-domain model's score on that same supported-class subset, never compared against the full 8 classes).
8. **Masked mean-pooling**: the valid frame count is **derived from the WhisperFeatureExtractor attention mask** (`return_attention_mask=True`, mel mask → conv2 downsampling); the hand-derived formula exists only as a reference implementation for cross-checking in unit tests. Policy: anything over 30 s takes a center window; anything under 5 s has already been dropped by filter; one window, no sliding; normalization stays at the extractor default.
9. **Stop-loss ladder (the 8 classes are locked)**: (1) W1 slips → the EdAcc out-of-domain test shrinks to a supported-class report, and no sources are added; (2) W2 slips → k_votes 3→1 and the YouTube list shrinks to ~300 clips (**weak labeling is never postponed — it is the central claim**; the scale shrinks, the stage stays); (3) still not enough → cut LOSO and the proposal. Under no circumstances does the task definition change or the three-arm ablation get cut.
10. **Schedule gates**: the HF release and the routing demo are both pushed out of the 3 weeks (backlog); the datasheet is a core deliverable. G1/G2/G3 are in the task table.

## Global Constraints

- The 8 classes are locked: en-US, en-GB, en-AU, en-IN, L1-Mandarin, L1-Spanish, L1-Korean, L1-Arabic; anything that does not map is dropped and counted.
- Core data sources = Common Voice + L2-ARCTIC + EdAcc (out-of-domain) + YouTube (weak labels, train only). VCTK/SAA are backlog. `label_source="self_report"` may enter eval; `weak` may not.
- The `openai/whisper-small` encoder backbone is frozen; LoRA r=16 touches only `q_proj`/`v_proj`.
- The repo does not redistribute L2-ARCTIC or YouTube audio; for YouTube it publishes only URLs + timestamps + labels; `data/` is gitignored.
- **CI never downloads a model**: every model call is injected or monkeypatched, and CI runs on synthetic fixtures only.
- TDD on every task: write a failing test → confirm it fails → minimal implementation → make it pass → commit; nothing counts as done until CI is green.
- Wording discipline: no SOTA claims; no blanket claim of statistically significant (see decision 2); conclusions about confounded classes are hedged; weak supervision / data-centric narrative.

## Project structure

```
accent-route/
├── pyproject.toml                # py3.11+uv
├── .github/workflows/ci.yml     # ruff + pytest (synthetic fixtures)
├── configs/
│   ├── taxonomy_v1.yaml
│   ├── sources/{common_voice,l2_arctic,edacc,youtube}.yaml
│   ├── filter.yaml               # min_dur=5.0 max_dur=30.0 min_snr_proxy_db=10 min_vad_ratio=0.5 min_lang_prob=0.8
│   ├── dedup.yaml
│   ├── weaklabel.yaml            # model_id, PINned sha, prompt file + sha256, k_votes=3, kill_precision=0.80
│   ├── train_common.yaml         # shared by all three arms: steps, sampler, augment, ckpt rule, seeds [17,42,1337]
│   └── arms/{a_gold,b_gold_oversampled,c_gold_weak,loso_l2}.yaml
├── prompts/qwen2audio_accent_v1.txt
├── lists/youtube_v1.csv          # url,start_s,end_s,prior_label,evidence_level,evidence_note
├── data/                         # gitignored
├── scripts/                      # weaklabel_qwen.sbatch train.sbatch watch_jobs.sh run_experiments.py
├── src/accentroute/
│   ├── schema.py  taxonomy.py  audio.py  cli.py
│   ├── ingest/{base,common_voice,l2_arctic,edacc,youtube}.py
│   ├── filter.py  dedup.py  split.py
│   ├── weaklabel/{qwen,consensus,audit}.py
│   ├── augment.py  emit.py
│   ├── model/{pooling,whisper_lora}.py  train.py
│   ├── eval/{metrics,bootstrap,baselines,tables}.py
│   └── reports/{coverage_confounding,dataset_stats}.py
├── tests/
└── docs/ (datasheet.md, clipto-proposal.md)
```

## Core algorithm specs (normative)

### (a) Staged schemas (T1)

```python
# src/accentroute/schema.py
import pandera as pa
from pandera.typing import Series

ACCENTS = ["en-US", "en-GB", "en-AU", "en-IN",
           "L1-Mandarin", "L1-Spanish", "L1-Korean", "L1-Arabic"]
SOURCES = ["common_voice", "l2_arctic", "edacc", "youtube"]   # extend when vctk/saa come off the backlog
SPLITS = ["train", "val", "test", "ood_test", "unassigned"]

class RawManifestSchema(pa.DataFrameModel):        # produced by ingest
    clip_id: Series[str] = pa.Field(unique=True)
    source: Series[str] = pa.Field(isin=SOURCES)
    source_uri: Series[str];  orig_file: Series[str]
    offset_start_s: Series[float] = pa.Field(ge=0)
    offset_end_s: Series[float] = pa.Field(gt=0)
    sample_rate_orig: Series[int] = pa.Field(gt=0)
    duration_s: Series[float] = pa.Field(gt=0)
    license: Series[str];  speaker_id_raw: Series[str]
    accent_raw: Series[str] = pa.Field(nullable=True)

class QCManifestSchema(RawManifestSchema):         # after taxonomy + filter
    accent_label: Series[str] = pa.Field(isin=ACCENTS, nullable=True)
    taxonomy_version: Series[str]
    snr_proxy_db: Series[float] = pa.Field(nullable=True)      # single-channel proxy, not true SNR
    vad_speech_ratio: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    lang_prob: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    transcript: Series[str] = pa.Field(nullable=True)
    status: Series[str] = pa.Field(isin=["pending", "accepted", "rejected", "review"])
    reject_reason: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_check(name="rejected_has_reason")
    def _c1(cls, df): return ~((df["status"] == "rejected") & df["reject_reason"].isna())

class SplitManifestSchema(QCManifestSchema):       # after dedup + split + weaklabel
    speaker_key: Series[str]                       # defaults to f"{source}:{speaker_id_raw}", updated when dedup merges speakers
    split: Series[str] = pa.Field(isin=SPLITS)
    label_source: Series[str] = pa.Field(isin=["gold", "self_report", "weak"])
    consensus_score: Series[float] = pa.Field(ge=0, le=1, nullable=True)  # not a calibrated confidence
    evidence_level: Series[str] = pa.Field(isin=["E1", "E2", "E3"], nullable=True)

    @pa.dataframe_check(name="weak_never_in_eval")
    def _c2(cls, df): return ~((df["label_source"] == "weak")
                               & df["split"].isin(["val", "test", "ood_test"]))

    @pa.dataframe_check(name="youtube_requires_evidence")
    def _c3(cls, df): return ~((df["source"] == "youtube") & (df["status"] == "accepted")
                               & ~df["evidence_level"].isin(["E1", "E2"]))

STAGE_SCHEMAS = {"raw": RawManifestSchema, "qc": QCManifestSchema, "split": SplitManifestSchema}
def validate_manifest(df, stage: str):
    return STAGE_SCHEMAS[stage].validate(df, lazy=True)   # lazy → report every violation at once
```

### (b) Pooling: deriving the valid frame count from the attention mask (T9)

```python
# src/accentroute/model/pooling.py
import torch
N_ENC_MAX = 1500  # whisper's 30 s window

def valid_encoder_frames(mel_attention_mask: torch.Tensor) -> torch.Tensor:
    """mel_attention_mask: [B, 3000], from WhisperFeatureExtractor(..., return_attention_mask=True).
    conv1 preserves length; conv2 has k=3, s=2, p=1 → L_out = (L_in - 1)//2 + 1."""
    n_mel = mel_attention_mask.sum(-1)
    return ((n_mel - 1) // 2 + 1).clamp(max=N_ENC_MAX)

def num_valid_encoder_frames_ref(n_samples: int, hop: int = 160) -> int:
    """Reference implementation for unit tests only: cross-checked against the mask-derived path, never used in production."""
    return min(((n_samples // hop) - 1) // 2 + 1, N_ENC_MAX)

def masked_mean(hidden: torch.Tensor, n_valid: torch.Tensor) -> torch.Tensor:
    """hidden: [B,1500,D]; n_valid: [B] -> [B,D]"""
    idx = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    mask = (idx < n_valid[:, None]).to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
```

Unit-test anchors: on the real extractor, 30 s → 1500 and 5 s → 250, with both paths agreeing; `masked_mean([x; garbage padding])` == the plain mean of `x`.

### (c) Dedup: ANN candidate edges + union-find (T7, architecture built to scale)

```python
# src/accentroute/dedup.py
def candidate_edges(embs, k: int = 20, sim_threshold: float = 0.45) -> list[tuple[int, int]]:
    """embs: [n,192], L2-normalized. Top-k cosine neighbors generate the candidate edges, avoiding the O(n²) full matrix.
    Within the core scope n is only in the hundreds (inside the YouTube set), so sklearn NearestNeighbors is exact;
    the signature stays fixed — swap in faiss for the backlog's global clustering."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(embs)), metric="cosine").fit(embs)
    dist, idx = nn.kneighbors(embs)
    return [(i, int(j)) for i in range(len(embs))
            for j, d in zip(idx[i][1:], dist[i][1:]) if 1.0 - d >= sim_threshold]

def union_find_clusters(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    return [find(i) for i in range(n)]
```

Threshold calibration (a script inside T7): positives = known same-speaker clip pairs from L2-ARCTIC/CV; **negatives must include cross-source hard negatives (random CV × YouTube pairings)**; the script emits within-class and between-class similarity histograms and sets the threshold at a false-merge rate ≤1e-3 — 0.45 is only the pre-calibration default. Near-duplicates: embedding cosine ≥0.92 and |Δduration| ≤0.5 s makes a candidate → transcript Jaccard ≥0.8 confirms it → keep one, reject the rest (`reject_reason="near_duplicate"`).

### (d) Stratified bootstrap + seed variation (T12)

```python
# src/accentroute/eval/bootstrap.py
@dataclass(frozen=True)
class AblationStats:
    delta_mean: float          # bootstrap mean of the seed-averaged Δmacro-F1
    ci_low: float; ci_high: float; n_boot: int
    ci_excludes_zero: bool     # report wording may cite only this field
    seed_deltas: tuple[float, ...]   # per-seed paired Δ
    seed_delta_std: float

def stratified_cluster_bootstrap(y_true, preds_a, preds_b, speaker_keys, classes,
                                 n_boot: int = 10_000, seed: int = 0) -> AblationStats:
    """Stratified by true class: within each class, that class's speaker_keys are resampled
    with replacement (same count), so a rare class can never disappear from a resample. preds_a/b: [n_seeds, n]."""
    rng = np.random.default_rng(seed)
    idx_of = {k: np.flatnonzero(speaker_keys == k) for k in np.unique(speaker_keys)}
    strata = {c: np.unique(speaker_keys[classes == c]) for c in np.unique(classes)}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        chosen = np.concatenate([rng.choice(ks, size=len(ks)) for ks in strata.values()])
        idx = np.concatenate([idx_of[k] for k in chosen])
        fa = np.mean([macro_f1(y_true[idx], p[idx]) for p in preds_a])
        fb = np.mean([macro_f1(y_true[idx], p[idx]) for p in preds_b])
        deltas[b] = fa - fb
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    seed_d = tuple(float(macro_f1(y_true, a) - macro_f1(y_true, b))
                   for a, b in zip(preds_a, preds_b))
    return AblationStats(float(deltas.mean()), float(lo), float(hi), n_boot,
                         bool(lo > 0 or hi < 0), seed_d, float(np.std(seed_d)))
```

Unit tests: identical predictions → the CI straddles 0; inject a +5 point shift → it is detected; every resample contains at least one cluster per class; a fixed seed reproduces byte for byte. The report template is fixed: "Δmacro-F1 = X, test-speaker stratified bootstrap 95% CI [l, h] (excludes zero: yes/no); per-seed Δ = [...], std = Y" — the phrase "statistically significant" never appears.

### (e) Weak-label consensus (T13)

```python
# src/accentroute/weaklabel/consensus.py
@dataclass(frozen=True)
class WeakLabelDecision:
    accepted: bool; label: str | None; consensus_score: float; reason: str

EVIDENCE_WEIGHT = {"E1": 1.0, "E2": 0.85}

def consensus(evidence_level: str, prior_label: str,
              qwen_votes: list[str]) -> WeakLabelDecision:
    """consensus_score = majority-vote fraction × evidence weight; an engineering ranking score, not a calibrated confidence."""
    if evidence_level not in EVIDENCE_WEIGHT:
        return WeakLabelDecision(False, None, 0.0, "evidence_E3")
    top, n = Counter(qwen_votes).most_common(1)[0]
    if top != prior_label or n < 2:
        return WeakLabelDecision(False, None, 0.0, "qwen_disagrees")
    score = (n / len(qwen_votes)) * EVIDENCE_WEIGHT[evidence_level]
    return WeakLabelDecision(True, prior_label, score, "consensus")
```

Auditing (`weaklabel/audit.py`): `draw_audit_sample(df, accepted_per_class=25, reject_pool_n=50, seed=0)` — the accepted pool stratified by class plus the rejected/review pool stratified by reject_reason, merged into a single blind-listening CSV with no label column; `audit_report(annotated) -> AuditReport` (per-class precision + Wilson interval + false-reject rate in the reject pool). Kill rule: if precision for a class in the accepted pool falls below 0.80, that class's weak labels are dropped wholesale and the datasheet records it.

### (f) Three-arm training budget protocol (T10/T14)

```yaml
# configs/train_common.yaml — shared by the three arms and LOSO; no field may be overridden by an arm config
epochs_c: 15                 # anchored to arm C's data volume
batch_size: 32
sampler: class_balanced      # same implementation, same seed stream
augment: {speed: [0.9, 1.1], musan: true, rir: true}
lr_schedule: cosine, lr: 1.0e-4, warmup_ratio: 0.05
ckpt_select: best_val_macro_f1   # fixed step budget, no early stopping
seeds: [17, 42, 1337]
```

- **C**: gold+weak, trained for `epochs_c` → yields the total step count `S_C` (written into the run metadata).
- **B**: gold-only, **trained for exactly the same `S_C` steps** (looping over the data is the oversampling).
- **A**: gold-only, epoch-matched (`epochs_c` epochs, fewer steps than S_C).
- Headline = C−B (the value of the weakly labeled data itself); A−B isolates the budget effect and is reported alongside it.

## Task breakdown (16 tasks, 3 weeks, 3 gates; backlog listed separately)

Five TDD steps per task: (1) write a failing test (2) confirm it fails (3) minimal implementation (4) make it pass (5) commit.

### Week 1 — Gold pipeline (CV + L2-ARCTIC + EdAcc) → Gate G1

- [x] **T1 Scaffolding + staged schemas**: `pyproject.toml`, `ci.yml`, `schema.py` (spec (a)), `tests/test_schema.py`. Tests: a valid raw row passes raw validation but fails split validation (missing columns); weak-in-test rejected; rejected-without-reason rejected; CI green. **At the same time**: the 9 + 10 decisions are written back into the spec as "revision v1.2"; this plan lands in `docs/superpowers/plans/`. Depends on: nothing.
- [x] **T2 Taxonomy**: `taxonomy.py` + `configs/taxonomy_v1.yaml`. `load_taxonomy(path)->Taxonomy`; `Taxonomy.map(raw)->str|None`; `.version`. Tests: "united states english" → en-US; "scottish" → None and counted; robust to case and whitespace. Depends on T1.
- [x] **T3 Audio utilities + ingest base class**: `audio.py`, `ingest/base.py`. `to_wav16k_mono(src,dst)->AudioMeta`; `SourceIngestor.iter_records()`; `run_ingest(ing,out)->Path` (its output passes raw validation). Tests: a 44.1k stereo fixture → 16k mono; slice offsets are exact. Depends on T1.
- [x] **T4 The three source adapters**: `ingest/{common_voice,l2_arctic,edacc}.py` + `configs/sources/*.yaml`. Tests: tiny fixtures → correct row counts and license strings, plus a CV accent_raw fill-rate statistic. **Kicked off in parallel on day 1 of W1**: MDC registration, the L2-ARCTIC form, the HF CV17 gate. Depends on T2, T3.
- [x] **T5 Coverage and confounding report (input to G1)**: `reports/coverage_confounding.py`. `source_accent_matrix(df)->DataFrame` (source × accent: n_speakers/n_clips/hours); `flag_confounded(matrix, dominance=0.9)->DataFrame` (class-level confounded flags); `edacc_class_coverage(df)->DataFrame` (classes with fewer than 5 speakers marked excluded → the supported-class set). Tests: the matrix, the flags and the exclusions are all correct on fixtures. Depends on T2, T4.
- [x] **T6 Filter**: `filter.py` + `configs/filter.yaml`. `compute_vad_ratio`, `estimate_snr_proxy_db`, `transcribe_tiny`, `apply_filters(df,cfg)->df` (its output passes qc validation). Tests: silence is rejected with `low_vad`; error below 1 dB on synthetic audio with a known signal-to-noise ratio; every model monkeypatched. Depends on T3.
- [x] **T7 Scoped dedup**: `dedup.py` (spec (c)) + the calibration script + `configs/dedup.yaml`. `assign_speaker_keys(df)->df` (defaults to source:speaker_id_raw); `dedup_youtube_speakers(df, embs)->df` (union-find merges speaker_key); `find_near_duplicates(df)->df` (transcript + duration, within CV). Tests: synthetic embeddings merge for the same speaker and stay separate for different ones; near-duplicates rejected; the calibration script includes cross-source negatives and emits the histogram data. Depends on T6.
- [x] **T8 Speaker-disjoint split + multi-source test set**: `split.py` + `configs/split.yaml`. `assign_splits(df,ratios=(0.8,0.1,0.1),seed=17)->df` (its output passes split validation); `write_speaker_report(df,out)`. Tests: **no speaker_key crosses a split**; edacc lands only in ood_test; class stratification within tolerance; seed determinism; **≥2 sources per class in the test set, or a confounded record is triggered** (L2 classes = L2-ARCTIC holdout + CV self-report; the results tables break out label_source and source into separate columns). Depends on T7.

**Gate G1 (end of W1)**: the gold manifest has ≥200 clips per class; ≥20 speakers for the native classes and ≥8 for the L2 classes; the leakage check passes; **the confounding matrix and confounded flags have been reviewed**; the EdAcc supported-class set is settled. Failure → stop-loss ladder step (1).

### Week 2 — Model + evaluation infrastructure + weak labeling → Gate G2

- [x] **T9 Pooling + model**: `model/pooling.py` (spec (b)), `model/whisper_lora.py`. `WhisperEncoderClassifier.forward(input_features, n_valid)->Tensor[B,8]`; `build_model(cfg)` (peft LoRA r=16 on q/v, encoder frozen). Tests: the two frame-count paths agree; masked mean ignores padding exactly; only the LoRA parameters and the head have requires_grad. Depends on T1.
- [x] **T10 Training loop + budget protocol**: `train.py` + `configs/train_common.yaml` + `configs/arms/*.yaml` (spec (f)). `train(cfg:TrainConfig)->TrainResult(ckpt_path,val_macro_f1,seed,total_steps)`. Tests: a 16-clip synthetic batch can be overfit; **assert arm B's step count == arm C's**; fields shared across the three arms cannot be overridden (the config loader refuses); metrics json written to disk. Depends on T8, T9.
- [x] **T11 Metrics + baselines**: `eval/{metrics,baselines}.py`. `macro_f1`, `confusion`, `majority_baseline`, `ecapa_embedding_probe` (frozen embeddings + logistic regression — the name is the wording), `qwen_zero_shot(df,cfg)` (shares the pinned revision and prompt with T13). Tests: matches sklearn; majority class exact; Qwen output parsing handles `unsure`. Depends on T8.
- [x] **T12 Stratified bootstrap**: `eval/bootstrap.py` (spec (d)). Tests as listed in the spec. Depends on T11.
- [x] **T13 Weak-label pipeline**: `ingest/youtube.py`, `lists/youtube_v1.csv` (300–600 tightly curated clips), `weaklabel/{qwen,consensus,audit}.py`, `prompts/qwen2audio_accent_v1.txt`, `configs/weaklabel.yaml`, `scripts/weaklabel_qwen.sbatch` + `scripts/watch_jobs.sh` (squeue/sacct polling). `qwen_label_batch(manifest,cfg)->Path` (GPU); `consensus` (spec (e)); `draw_audit_sample`; `audit_report`. Tests: the consensus rules are table-driven (E3 → reject; disagreement → review; E1+3/3 → 1.0; E2+2/3 → 0.567); the audit sample includes the reject-pool strata; the schema blocks weak labels from reaching eval. Depends on T6, T8.

**Gate G2 (end of W2)**: arm A reproduces across 3 seeds, and its val macro-F1 beats the majority-class baseline with a CI that does not straddle 0 (against majority class); the weak-label pipeline runs end to end and produces an acceptance rate. Failure → stop-loss ladder step (2) (k_votes down to 1, shrink the list; weak labeling is not postponed).

### Week 3 — Three-arm experiments + stratified evaluation → Gate G3 → datasheet

- [x] **T14 Augmentation + the three dataset variants**: `augment.py`, `emit.py`. `augment_train(df,wav_dir,cfg)->df` (train rows only); `emit_dataset(df, arm: Literal["a_gold","b_gold_oversampled","c_gold_weak"], out_dir)->DatasetStats` (B's oversampling is driven by the step count on the training side; emit only records the arm and its data content). Tests: augmented rows appear only in train; the A and B variants contain zero weak rows; the statistics reconcile. Depends on T8, T13.
- [x] **T15 Experiment matrix + stratified evaluation**: `scripts/run_experiments.py`, `scripts/train.sbatch`, `eval/tables.py`. The matrix is 3 arms × 3 seeds (9 runs) plus the single-seed LOSO-L2 diagnostic (10 training runs in total, on AICR rtx, watched by watch_jobs.sh). `make_results_tables(runs_dir)->Path` produces: (1) the three-arm ablation table (headline C−B plus the A−B budget effect, with the bootstrap CI and the seed std in separate columns); (2) **per-class F1 stratified by source**; (3) EdAcc supported-class macro-F1 (with the in-domain model's score on that same supported-class subset alongside it); (4) confusion matrices focused on en-IN vs L2 and en-GB vs en-AU; (5) the LOSO table. Tests: every table generates from a fixture run directory; the CI columns match T12's output; the wording template is validated (bans "statistically significant"). Depends on T10–T12, T14.
- [ ] **T16 Datasheet + proposal + resume bullets (Gate G3: starts once the T15 numbers are final)**: `docs/datasheet.md`, `docs/clipto-proposal.md`. The datasheet must contain: the license table, drop statistics, the weak-label acceptance rate, results from all three audit pools (accepted-pool precision + Wilson intervals, false rejects in the reject pool), any kill-rule triggers, the taxonomy version, **the confounding matrix and the list of confounded classes**, a statement of residual dedup risk, and the composition of the L2 test set with an explanation of its wide CIs. The proposal (2–3 pages): the on-device distillation/quantization path, the integration point for routing ahead of ASR, and the MCP tool shape. Three resume bullets (canonical three-part form, no SOTA). Depends on T15.

### Backlog (outside the 3 weeks, ordered by value)

1. HF release (adapter + model card, push script with a dry-run mode); 2. VCTK/SAA adapters (shoring up en-GB/en-AU + updating the confounding matrix); 3. global cross-source speaker clustering (just swap faiss in for candidate_edges); 4. the accent-aware routing demo (`demo/route_asr.py`, WER comparison); 5. seed × speaker hierarchical bootstrap.

## Risks and contingencies (ordered by damage potential)

1. **Source-label confounding (the biggest validity risk)**: four lines of defense — the confounding matrix (T5), a multi-source test set (T8), per-source stratified reporting plus LOSO (T15), and hedged wording for confounded classes (T16); if some L2 class ends up with only L2-ARCTIC in its test set, it gets honestly marked confounded rather than padded out.
2. **Scarce L2 test speakers**: L2-ARCTIC has 4 speakers per class; CV self-report fills out the test set and is broken out by label_source; wide CIs are reported as they are.
3. **W2/W3 overload**: the training runs went from 6 to 10, but the scope reduction (3 sources, no HF release, no demo) offsets it; the contingency is k_votes 3→1, shrinking YouTube to 300 clips, and cutting LOSO and the proposal (the ladder in decision 9).
4. **Common Voice distribution changes**: the primary path (HF v17.0) is settled, so an MDC failure has no impact.
5. **Insufficient EdAcc Korean/Arabic coverage**: T5 produces the supported-class set in W1, so there are no surprises in W3.
6. **CI downloading models**: everything is injected or monkeypatched (Global Constraints).

## Verification (end to end)

1. **Per task**: `pytest tests/test_<module>.py -v` goes red → green; commit as soon as it is done.
2. **Pipeline smoke test**: the `accentroute` CLI runs the full chain (ingest → emit) on synthetic fixtures, with each stage's manifest passing its `validate_manifest(df, stage)`.
3. **Leakage audit (hard gate at G1)**: a script asserts that the `speaker_key` sets of train/val/test are pairwise disjoint, that EdAcc sits entirely in ood_test, that 100% of weak rows are in train, and it writes the per-class test source count to disk.
4. **Budget-matching audit (hard gate on three-arm fairness)**: the training logs assert that B and C have identical total steps and that the augment/sampler/ckpt-rule hashes are identical across the three arms.
5. **Reproducibility (hard gate at G2)**: two training runs with the same config and seed differ by less than 0.5 points of val macro-F1.
6. **Headline number (hard gate at G3)**: `stratified_cluster_bootstrap` emits the C−B CI and the seed std, which go into the results table using the fixed wording template.
7. **CI**: GitHub Actions ruff + pytest fully green, with no model downloads over the network.
