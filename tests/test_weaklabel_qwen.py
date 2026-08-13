"""T13 (cont.): Qwen config pinning and batch voting — the generation function is
injected, so CI downloads nothing.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from accentroute.weaklabel.qwen import WeakLabelConfig, qwen_label_batch

CONFIG = Path(__file__).parent.parent / "configs" / "weaklabel.yaml"


class TestConfigPinning:
    def test_loads_repo_config(self):
        cfg = WeakLabelConfig.from_yaml(CONFIG)
        assert cfg.model_id == "Qwen/Qwen2-Audio-7B-Instruct"
        assert cfg.k_votes == 3
        assert cfg.kill_precision == 0.80

    def test_prompt_sha_matches_repo_prompt(self):
        """The prompt file must match the sha256 pinned in the config."""
        WeakLabelConfig.from_yaml(CONFIG).load_prompt()

    def test_prompt_drift_rejected(self, tmp_path):
        prompt = tmp_path / "p.txt"
        prompt.write_text("changed prompt")
        cfg = WeakLabelConfig(
            model_id="m", revision="abc", prompt_file=prompt,
            prompt_sha256="0" * 64, k_votes=3, temperature=0.7, kill_precision=0.8,
        )
        with pytest.raises(ValueError, match="prompt file changed"):
            cfg.load_prompt()

    def test_repo_config_flags_unpinned_revision(self):
        """While revision is PIN_ME, the sbatch refuses to submit; this pins that contract."""
        cfg = WeakLabelConfig.from_yaml(CONFIG)
        assert cfg.revision == "PIN_ME", (
            "revision is now pinned — update this test and the datasheet pin-date record"
        )


class TestLabelBatch:
    def test_k_votes_and_metadata(self, tmp_path):
        manifest = tmp_path / "in.parquet"
        pd.DataFrame({"clip_id": ["y1", "y2"]}).to_parquet(manifest)
        prompt = tmp_path / "p.txt"
        prompt.write_text("pick one")
        cfg = WeakLabelConfig(
            model_id="Qwen/Qwen2-Audio-7B-Instruct", revision="deadbeef",
            prompt_file=prompt, prompt_sha256="", k_votes=3, temperature=0.7,
            kill_precision=0.8,
        )
        calls = []

        def fake_generate(wav_path: str, prompt_text: str) -> str:
            calls.append(wav_path)
            return "en-AU" if "y1" in wav_path else "not a label"

        out = qwen_label_batch(
            manifest, cfg, tmp_path / "out.parquet",
            generate_fn=fake_generate, wav_path_fn=lambda cid: f"/wav/{cid}.wav",
        )
        df = pd.read_parquet(out)
        assert list(df["qwen_votes"].iloc[0]) == ["en-AU"] * 3
        assert list(df["qwen_votes"].iloc[1]) == ["unsure"] * 3
        assert len(calls) == 6  # 2 clips × k=3 votes

        meta = json.loads(Path(str(out).replace(".parquet", ".meta.json")).read_text())
        assert meta["revision"] == "deadbeef"
        assert meta["k_votes"] == 3
        assert len(meta["prompt_sha256"]) == 64
