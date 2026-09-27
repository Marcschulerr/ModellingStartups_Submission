"""
Blind coding sample for the classifier validation (thesis data chapter).

Draws 200 analysis firms stratified by machine label (20 per circular sub-type,
so 100 circular, plus 100 non-circular, parse errors excluded) and pairs each with
the EXACT text the classifier saw (`openai_description` from panel_fat, the
GPT-4o-mini description built from Crunchbase + scraped website content).

Writes:
  outputs/tables/classifier_validation_blind.csv   for the blind coder: machine labels
      hidden, order shuffled; fill columns: circular (0/1), cluster (Design/
      Waste/Platform/Service/Nature, only if circular=1), hard_case (0/1), notes
  outputs/tables/classifier_validation_key.csv     code_id -> firm + machine
      label. DO NOT OPEN before coding is finished.

Codebook = the original classification prompt: circular if the business model
contributes to closing, slowing, narrowing, or regenerating resource loops;
conservative default to non-circular; exactly ONE dominant cluster.

Seed pinned. Run:  conda run -n ml python scripts/make_validation_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loading import open_panel_fat  # noqa: E402

OUT = ROOT / "outputs" / "tables"
SEED = 20260712
PER_SUBTYPE = 20
N_NONCIRC = 100


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "firms_analysis.parquet",
                         columns=["firm_id", "cb_company_name", "circ", "circ_subtype",
                                  "cbm_type", "cbm_confidence", "parse_error"])
    df = df[df["parse_error"] == 0]
    rng = np.random.default_rng(SEED)

    parts = []
    for sub in ("Design", "Platform", "Nature", "Service", "Waste"):
        pool = df[df["circ_subtype"] == sub]
        parts.append(pool.iloc[rng.choice(len(pool), PER_SUBTYPE, replace=False)])
    pool = df[df["circ"] == 0]
    parts.append(pool.iloc[rng.choice(len(pool), N_NONCIRC, replace=False)])
    sample = pd.concat(parts, ignore_index=True)
    ids = set(sample["firm_id"])

    print(f"sampled {len(sample)} firms; streaming descriptions from panel_fat ...")
    cols = ["crunchbase_url", "openai_description"]
    reader = open_panel_fat(columns=cols,
                            column_types={c: pa.string() for c in cols})
    desc: dict[str, str] = {}
    for batch in reader:
        d = batch.to_pandas()
        hit = d[d["crunchbase_url"].isin(ids) & ~d["crunchbase_url"].isin(desc)]
        for _, r in hit.drop_duplicates("crunchbase_url").iterrows():
            desc[r["crunchbase_url"]] = r["openai_description"]
        if len(desc) == len(ids):
            break
    missing = ids - set(desc)
    assert not missing, f"{len(missing)} firms without description"
    sample["description"] = sample["firm_id"].map(desc)

    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample["code_id"] = range(1, len(sample) + 1)

    blind = sample[["code_id", "cb_company_name", "description"]].copy()
    for c in ("circular", "cluster", "hard_case", "notes"):
        blind[c] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    blind.to_csv(OUT / "classifier_validation_blind.csv", index=False)
    sample[["code_id", "firm_id", "cb_company_name", "cbm_type",
            "cbm_confidence"]].to_csv(OUT / "classifier_validation_key.csv", index=False)
    print(f"wrote classifier_validation_blind.csv ({len(blind)} firms, seed {SEED}), "
          f"fill circular/cluster/hard_case")
    print("wrote classifier_validation_key.csv, DO NOT OPEN before coding")


if __name__ == "__main__":
    main()
