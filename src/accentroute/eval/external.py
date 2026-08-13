"""External comparison axis: re-report results in a published benchmark's label set.

This project's 8-class taxonomy is custom, so its macro-F1 has no reference point a
reader can calibrate against. Vox-Profile (arXiv 2505.14648) aggregates 11 corpora with
speaker-disjoint splits, publishes a narrow regional label set, and ships models — so
reporting a second axis in its vocabulary buys an external comparison.

The two label sets do not nest. Ours is partly first-language based (L1-Mandarin,
L1-Korean); Vox-Profile's is regional (East Asia). Collapsing ours into theirs therefore
*hides* real errors: a Mandarin/Korean confusion disappears. The report always states
which groups were collapsed and how many errors that concealed, so the external number is
never quoted as if it were the primary result.
"""

import numpy as np
import pandas as pd

from accentroute.eval.metrics import macro_f1

# The narrow label set published with the Vox-Profile accent models.
VOX_PROFILE_NARROW = (
    "East Asia", "English", "Germanic", "Irish", "North America", "Northern Irish",
    "Oceania", "Other", "Romance", "Scottish", "Semitic", "Slavic", "South African",
    "Southeast Asia", "South Asia", "Welsh",
)

# Only unambiguous correspondences. en-GB maps to "English" (Vox-Profile keeps Scottish,
# Welsh, Irish and Northern Irish as separate labels, so "English" means England English,
# which is what en-GB denotes here).
_TO_VOX_PROFILE = {
    "en-US": "North America",
    "en-GB": "English",
    "en-AU": "Oceania",
    "en-IN": "South Asia",
    "L1-Mandarin": "East Asia",
    "L1-Korean": "East Asia",
    "L1-Spanish": "Romance",
    "L1-Arabic": "Semitic",
}


def to_vox_profile(accent_label: str) -> str:
    """Map one of our 8 classes onto the Vox-Profile narrow label set."""
    return _TO_VOX_PROFILE[accent_label]


def collapsed_groups() -> dict[str, list[str]]:
    """Vox-Profile labels that more than one of our classes maps onto."""
    groups: dict[str, list[str]] = {}
    for ours, theirs in _TO_VOX_PROFILE.items():
        groups.setdefault(theirs, []).append(ours)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def vox_profile_report(y_true, y_pred) -> dict:
    """Score predictions after collapsing both sides onto the Vox-Profile label set.

    n_errors_hidden_by_collapse counts clips that are wrong in our taxonomy but correct in
    theirs — the price of the external axis, reported rather than buried.
    """
    y_true = pd.Series(list(y_true))
    y_pred = pd.Series(list(y_pred))
    vp_true = y_true.map(to_vox_profile)
    vp_pred = y_pred.map(to_vox_profile)

    hidden = int(((y_true != y_pred) & (vp_true == vp_pred)).sum())
    labels = sorted(set(vp_true) | set(vp_pred))
    return {
        "classes": labels,
        "n_clips": len(y_true),
        "vox_profile_macro_f1": macro_f1(
            np.asarray(vp_true), np.asarray(vp_pred), labels=labels
        ),
        "collapsed_groups": collapsed_groups(),
        "n_errors_hidden_by_collapse": hidden,
    }
