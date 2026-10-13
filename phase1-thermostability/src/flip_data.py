"""Parse FLIP meltome FASTA (biotrainer header format) -> tidy parquet."""
import re
from pathlib import Path
import pandas as pd

RAW, OUT = Path("data/raw"), Path("data/processed")
SPLITS = ["mixed_split", "human", "human_cell"]
KV = re.compile(r"(\w+)=(\S+)")


def read_fasta(path: Path) -> pd.DataFrame:
    rows, header, seq = [], None, []

    def flush():
        if header is not None:
            kv = {k.lower(): v for k, v in KV.findall(header)}
            rows.append({"id": header.split()[0], "seq": "".join(seq), **kv})

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                flush()
                header, seq = line[1:], []
            elif line:
                seq.append(line)
    flush()
    return pd.DataFrame(rows)


def tidy(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    missing = {"target", "set"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{split_name}: header keys missing {missing}; got {list(df.columns)}. "
            "Inspect with `head -n 4` and adjust KV parsing."
        )
    df = df.copy()
    df["target"] = df["target"].astype(float)
    df["subset"] = df["set"].str.lower()
    if "validation" in df.columns:
        is_val = df["validation"].str.lower().eq("true")
    else:
        is_val = pd.Series(False, index=df.index)
    df.loc[is_val & df["subset"].eq("train"), "subset"] = "val"
    df["length"] = df["seq"].str.len()
    df["split_name"] = split_name
    return df[["id", "seq", "target", "subset", "length", "split_name"]]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        df = tidy(read_fasta(RAW / f"{name}.fasta"), name)
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"\n== {name}: {len(df)} sequences")
        print(
            df.groupby("subset")
            .agg(n=("id", "size"), tm_mean=("target", "mean"), len_median=("length", "median"))
            .round(2)
            .to_string()
        )
