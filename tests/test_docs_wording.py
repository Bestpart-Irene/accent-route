"""The wording gate applies to the repo's own prose, not just to generated tables.

An earlier README draft quoted a banned phrase while explaining the rule that bans it.
Without this test the gate would only ever have run against text nobody reads.
"""

from pathlib import Path

import pytest

from accentroute.eval.tables import check_wording

ROOT = Path(__file__).parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_doc_passes_wording_gate(path: Path):
    check_wording(path.read_text())


def test_docs_are_english_only():
    """Public portfolio repo: no CJK anywhere in the prose."""
    cjk = [(0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF)]
    for path in DOCS:
        offending = [
            c for c in path.read_text() if any(lo <= ord(c) <= hi for lo, hi in cjk)
        ]
        assert not offending, f"{path.name} contains CJK characters: {offending[:5]}"
