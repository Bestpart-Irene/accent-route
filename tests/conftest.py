"""Shared fixtures. `make_manifest` is used by both the speaker-disjoint split tests
and the leaky-split tests, which have to run on identical rows for the comparison between
them to mean anything."""

import pandas as pd

SR = 16000


def _row(clip_id, source, speaker, label, status="accepted", duration=6.0):
    return {
        "clip_id": clip_id,
        "source": source,
        "source_uri": "x://",
        "orig_file": f"{clip_id}.wav",
        "offset_start_s": 0.0,
        "offset_end_s": duration,
        "sample_rate_orig": SR,
        "duration_s": duration,
        "license": "CC0-1.0",
        "speaker_id_raw": speaker,
        "accent_raw": label,
        "accent_label": label if status == "accepted" else None,
        "taxonomy_version": "v1",
        "snr_proxy_db": 20.0,
        "vad_speech_ratio": 0.9,
        "lang_prob": 0.99,
        "transcript": f"transcript {clip_id}",
        "status": status,
        "reject_reason": None if status == "accepted" else "low_vad",
        "speaker_key": f"{source}:{speaker}",
    }


def make_manifest() -> pd.DataFrame:
    rows = []
    # CV: 2 classes × 10 speakers × 4 clips
    for label in ["en-US", "L1-Korean"]:
        for s in range(10):
            spk = f"{label}-cv{s}"
            for c in range(4):
                rows.append(_row(f"cv-{label}-{s}-{c}", "common_voice", spk, label))
    # L2-ARCTIC: L1-Korean, 4 speakers × 5 clips (mirrors the real constraint)
    for s, spk in enumerate(["HJK", "HKK", "YDCK", "YKWK"]):
        for c in range(5):
            rows.append(_row(f"l2-{s}-{c}", "l2_arctic", spk, "L1-Korean"))
    # EdAcc: goes to ood_test only
    for s in range(3):
        rows.append(_row(f"ed-{s}", "edacc", f"P{s}", "en-US"))
    # YouTube weak labels: train only
    rows.append(
        {**_row("yt-0", "youtube", "chan1:v1", "en-AU"), "evidence_level": "E1"}
    )
    # one rejected row: its split must come out "unassigned"
    rows.append(_row("cv-rej", "common_voice", "rejspk", "en-US", status="rejected"))
    return pd.DataFrame(rows)


