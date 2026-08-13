# EdAcc pipeline smoke test

**What this is:** the first end-to-end run of the pipeline on real speech —
ingest → taxonomy → filter → speaker-disjoint split → LoRA training → evaluation, on
AICR (`cpu` partition for preparation, one B200 on `b200-devel` for training).

**What this is not:** the designed experiment. EdAcc is the out-of-domain test set and it
covers only part of the taxonomy, so nothing here is the headline C − B number. The point
was to prove the pipeline runs on real audio and to surface integration bugs before the
Common Voice and L2-ARCTIC data arrive.

## Data

| stage | count |
| --- | --- |
| clips ingested | 19,137 (122 speakers) |
| accepted by the filter | 1,168 |
| rejected: outside the taxonomy | 14,131 |
| rejected: shorter than 5 s | 3,333 |
| rejected: SNR proxy below floor | 271 |
| rejected: longer than 30 s | 147 |
| rejected: speech ratio below floor | 87 |

Largest dropped varieties, all correctly outside the locked 8 classes: Nigerian English
(1,356 clips), Irish English (1,317), Kenyan English (1,157), Romanian (1,134),
Vietnamese (1,046), Catalan (855), Jamaican English (753), Italian (733).

Five classes cleared the ≥4-speaker floor: `en-US`, `en-GB`, `en-IN`, `L1-Mandarin`,
`L1-Spanish`. Split: 772 train / 103 val / 128 test clips, over 24 / 5 / 5 speakers.
Leakage audit: **no speaker_key spans two splits**.

## Result

Training: loss 0.667 → 0.0003 over 600 steps. Best val macro-F1 0.593.

Test macro-F1 **0.461**, majority-class baseline **0.042**.

Confusion matrix (rows = true, columns = predicted):

| | L1-Mandarin | L1-Spanish | en-GB | en-IN | en-US |
| --- | --- | --- | --- | --- | --- |
| **L1-Mandarin** | 1 | 0 | 27 | 0 | 0 |
| **L1-Spanish** | 0 | 15 | 0 | 0 | 0 |
| **en-GB** | 0 | 0 | 30 | 0 | 0 |
| **en-IN** | 0 | 0 | 0 | 22 | 0 |
| **en-US** | 0 | 20 | 4 | 7 | 2 |

## Reading the result honestly

**The headline number is not evidence of accent recognition.** The test set has exactly
one speaker per class, and the per-class outcome is all-or-nothing: three speakers are
classified almost perfectly (L1-Spanish 15/15, en-GB 30/30, en-IN 22/22) and two are
almost never right (L1-Mandarin 1/28, en-US 2/33). Macro-F1 of 0.461 is, to a good
approximation, "3 of the 5 test speakers happened to land on the right side" — the metric
is degenerate at this sample size.

That pattern is the signature of speaker-level generalization rather than accent-level
generalization: the model appears to be deciding per voice, not per accent. Training loss
reaching 0.0003 on 772 clips from 24 speakers says the same thing from the other end.

This is a useful negative result rather than a disappointing one. It is a direct,
measured demonstration of the premise the project is built on: a speaker-disjoint split
is necessary but nowhere near sufficient, and a per-class test stratum of one speaker
cannot support any claim about accent. It is exactly why the real experiment requires
several test speakers per class, per-source stratified reporting, and a speaker-level
bootstrap interval — with one speaker per class there is nothing for a bootstrap to
resample.

No claim is made here about the model's accent-recognition ability. The number's only
role is to show the pipeline runs end to end and produces a metric.

## Bugs this surfaced

Three defects that the 176 unit tests did not catch, all found by touching real data:

1. **EdAcc audio arrives as encoded bytes.** The HF release stores the Audio feature as
   `{"bytes": <WAV file bytes>, "path"}`; only `datasets` decoding yields
   `{"array", "sampling_rate"}`. The test fixture had encoded the wrong assumption, so the
   suite passed while extraction died on the real corpus. The fixture now builds real
   encoded bytes and a round-trip test compares the extracted signal to the source.
2. **The training loop never moved batches to the model's device.** DataLoader always
   yields CPU tensors, so any model on mps or cuda failed inside whisper's first conv.
   This would have hit the cluster runs identically.
3. **The taxonomy was missing `mainstream us english`,** EdAcc's standardized label for
   en-US, which silently zeroed out the entire class in the coverage report.

## Reproducing

```bash
sbatch scripts/smoke_prepare.sbatch                              # cpu partition, no GPU
sbatch --dependency=afterok:<id> scripts/smoke_train.sbatch 600  # b200-devel, 1 GPU
```

Preparation took 4 min 21 s; training took 2 min 35 s on one B200.
