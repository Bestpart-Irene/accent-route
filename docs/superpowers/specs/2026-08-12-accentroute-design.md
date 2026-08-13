# AccentRoute — An English Accent Recognition Data Pipeline for Transcription Products (design spec)

Date: 2026-08-12 — Status: v1.2 (revisions from two review rounds merged in; implementation plan approved, see `../plans/2026-08-12-accentroute-implementation.md`)
Purpose: (1) fill the data-pipeline slot on my MLE resume (replaces LLM Serving once built); (2) a door-opener for the Clipto application
Location: `/Users/xxiellan/accent-route` (project root; this spec ships with the repo; GitHub: Bestpart-Irene)

## 1. Goal

Take a 5–30 second English speech clip and predict one of 8 accent labels. The center of gravity is the **data pipeline** (multi-source integration, LLM weak labeling, quality control); the modeling side is deliberately kept off-the-shelf (Whisper + LoRA), which maps onto the data pipeline + evaluation axis that defines this job family.

Success criteria (all quantifiable):
1. macro-F1 on a speaker-disjoint test set clearly ahead of both the ECAPA-TDNN baseline and Qwen2-Audio zero-shot;
2. **headline ablation number**: the macro-F1 gain from training on gold + weak labels versus gold only (this is what demonstrates the value of the data pipeline and of LLM labeling);
3. performance does not collapse on the EdAcc out-of-domain test (spontaneous conversation); the in-domain/out-of-domain gap is reported.

## 2. Accent taxonomy (8 classes, locked)

- Native varieties: `en-US`, `en-GB`, `en-AU`, `en-IN`
- L2 accents (by the speaker's first language): `L1-Mandarin`, `L1-Spanish`, `L1-Korean`, `L1-Arabic`

Clips that do not map into the 8 classes are dropped (Scottish, Filipino, and so on), with drop statistics recorded. en-IN is defined as Indian English, with no native/L2 distinction.

## 3. Data sources and licensing

| Source | Use | Licensing notes |
| --- | --- | --- |
| Common Voice (en) | Main training set (self-reported accent labels; free text normalized through a mapping table) | CC0, redistributable |
| L2-ARCTIC | Gold labels for the four L2 classes | Research license; the repo does not redistribute audio, a download script is provided instead |
| VCTK | Extra coverage for en-GB and the other native classes | CC BY 4.0 |
| Speech Accent Archive | Small supplement / audit set | CC BY-NC-SA, evaluation only |
| EdAcc | **Out-of-domain test set only, never trained on** | License verified in week 1 of implementation; if it does not check out, swap in a spontaneous Common Voice subset |
| YouTube interviews/podcasts | Weak-label expansion set | **Audio is not redistributed**; the repo publishes only a list of URLs + timestamps + labels; fetched locally with yt-dlp |

## 4. The data pipeline (7 stages, each an independent module with unit tests)

1. **ingest**: every source → 16 kHz mono WAV clips + a unified metadata schema (Parquet: source, speaker_id, accent_raw, duration, split)
2. **taxonomy**: accent_raw → the 8-class mapping table (versioned YAML; whitelist mapping for Common Voice free text, anything outside the map is dropped and counted)
3. **filter**: Silero VAD to strip silence, duration ≥5 s, an SNR floor, Whisper-tiny transcription + fastText LID to confirm the clip is English
4. **dedup & split**: speaker-level dedup; a **speaker-disjoint** train/val/test split (a speaker never crosses a split); the split script emits a reviewable speaker roster
5. **weak-label** (automatic labeling by a large hosted model): yt-dlp fetch → channel-region metadata prior + Qwen2-Audio zero-shot labeling → **a clip is kept only when both signals agree**; disagreements land in a manual spot-check pool; the acceptance rate is logged
6. **balance & augment**: per-class quota sampling; speed perturbation (0.9/1.1), added noise (MUSAN) and reverb (RIR), applied to the training split only
7. **emit**: two versions of the training set (gold-only / gold+weak, for the ablation) + a dataset statistics report (class distribution, duration distribution, speaker counts)

## 5. Model and training

- Backbone: `openai/whisper-small` encoder (MIT, frozen)
- Adaptation: LoRA (r=16, attention q/v projections) + mean-pooling + a linear classification head
- Training: cross-entropy with class weights; single GPU (AICR rtx tier / Explorer); a single run should take a few hours at most
- Baselines: majority class, ECAPA-TDNN (SpeechBrain, Apache-2.0), Qwen2-Audio zero-shot (Apache-2.0, run locally)
- Fully open source; the LoRA adapter and classification head are published to Hugging Face under my own account, with a model card

## 6. Evaluation

- Primary metric is macro-F1; the confusion-matrix analysis focuses on en-IN vs the L2 classes and en-GB vs en-AU
- EdAcc out-of-domain results are reported separately
- Ablation: gold-only vs gold+weak (the headline number)
- **Stretch (only if time allows)**: an accent-aware routing demo — set the Whisper decoding prompt from the predicted accent and compare WER against default decoding, which gives the Clipto proposal a business-facing number

## 7. Deliverables

1. A public GitHub repo (pipeline code + reproduction scripts + unit tests + CI)
2. A Hugging Face model page (adapter weights + model card)
3. A dataset datasheet (sources, licenses, drop statistics, weak-label acceptance rate)
4. A Clipto integration proposal (2–3 pages): the on-device distillation/quantization path (ONNX/CoreML), the integration point for routing ahead of ASR, and the MCP tool shape
5. Three MLE resume bullets (written once the project is built; replaces the LLM Serving slot; follows the canonical three-part form, avoids SOTA wording, and tells a weak supervision / data-centric story)

## 8. Schedule (3 weeks part-time)

- W1: ingest + taxonomy + filter + dedup/split running end to end, the gold dataset taking shape; EdAcc license check
- W2: LoRA training + baselines + in-domain evaluation; the weak-label pipeline running
- W3: ablation + EdAcc out-of-domain + HF release + datasheet; the routing demo if there is slack; write the Clipto proposal

## 9. Risks and honest boundaries

- Noise in the Common Voice self-reported labels → whitelist mapping + 50 manually spot-checked clips per class
- YouTube ToS → no audio redistribution, only a URL list; rate-limited fetching
- The speech stack is new territory for me → the schedule assumes 3 weeks; if ingest is not working by the end of W1, cut the number of L2 classes to stop the bleeding
- Resume/interview wording: LoRA fine-tuning of an open-source backbone (not training from scratch); the weak labels are weak supervision (not human gold); no SOTA claims

## 10. Revision log (v1.1 + v1.2, superseding anything above it conflicts with)

Design revisions locked in after two rounds of review; the full implementation is in the implementation plan. Where this section conflicts with §1–§9, this section wins.

**v1.1 (data lineage / leakage prevention / auditing / statistics — 9 items)**
1. Ablation statistics: 3 seeds per configuration (17/42/1337); Δmacro-F1 comes with a speaker-level stratified bootstrap 95% CI computed on the test set.
2. Breaking weak-label circularity: pin the Qwen2-Audio revision sha and version the prompt; blind human audit of accepted weak labels (25 clips per class); if precision for a class falls below 0.80, that class's weak labels are dropped wholesale; weak labels only ever enter train (machine-enforced by the schema).
3. YouTube evidence levels E1/E2/E3; only E1/E2 clips that also agree with Qwen are accepted; weakly labeled data only enters train.
4. Dedup goes beyond speaker_id: ECAPA speaker recognition + near-duplicate audio detection + transcript overlap checks.
5. The unified schema grows: clip_id, source file reference, license, sample rate, taxonomy version, quality metrics, label provenance, consensus_score, reject_reason.
6. EdAcc gets double verification in W1 (license + label mapping); classes with insufficient coverage are excluded from the out-of-domain macro-F1 (supported-class macro-F1).
7. Whisper masked mean-pooling, with the valid frame count derived from the attention mask; clips longer than 30 s use a center window.
8. The stop-loss ladder never cuts the 8 classes: trim supplementary sources first, then shrink the scale of weak labeling, and only then cut release-facing work.
9. The demo, the HF release and the proposal all queue behind the core experiments; weekly go/no-go gates.

**v1.2 (experimental validity — response to the second review round)**
10. **Source-confounding controls**: produce a source × accent matrix before G1; any class where a single source accounts for more than 90% is marked confounded and every conclusion about it is hedged accordingly; aim for ≥2 sources per class in the test set; report results stratified by source; add a LOSO diagnostic.
11. **Three-arm fair ablation**: A gold (epoch-matched) / B gold oversampled (same step count as C) / C gold+weak; shared augmentation, sampler, fixed step budget and checkpoint rule; **the headline number is C−B**, while A−B isolates the budget effect.
12. **Statistical wording**: report the test-speaker bootstrap CI and the seed variation separately; write only "bootstrap CI excludes zero", never a blanket claim of statistically significant.
13. Staged schemas (raw/qc/split); naming discipline: snr_proxy_db, consensus_score, frozen ECAPA embedding probe, supported-class macro-F1.
14. Scope reduction: the 3-week core is Common Voice + L2-ARCTIC + EdAcc + a small, tightly curated YouTube set (300–600 clips); VCTK, SAA, global speaker clustering, the HF release and the routing demo move to the backlog; **weak labeling is never postponed** (it is the central claim), only scaled down.
15. Implementation depends on no session-specific skill; cluster interaction is explicit sbatch files and polling scripts checked into the repo.
