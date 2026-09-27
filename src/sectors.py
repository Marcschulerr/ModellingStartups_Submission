"""
Sector controls from the Crunchbase industry tag list.

`cbcsv_industries` is a comma-separated multi-tag string ("Energy, Marketplace,
Renewable Energy, ..."), 97.9 % filled. We encode it as a multi-hot over the
top-20 most frequent tags (pinned in src/schema.py TOP_INDUSTRY_TAGS) plus a
tag count. Firms without tags get n_industries = <NA> and all dummies 0:
"no tags" is itself information, not a zero sector.
"""
from __future__ import annotations

import pandas as pd

from .schema import TOP_INDUSTRY_TAGS


def attach_sector_controls(firms: pd.DataFrame) -> pd.DataFrame:
    tags = firms["industries"].str.split(", ")
    firms["n_industries"] = tags.str.len().astype("Int16")
    tag_sets = tags.map(lambda t: set(t) if isinstance(t, list) else set())
    for tag, slug in TOP_INDUSTRY_TAGS.items():
        firms[slug] = tag_sets.map(lambda s, tag=tag: tag in s).astype("int8")
    return firms
