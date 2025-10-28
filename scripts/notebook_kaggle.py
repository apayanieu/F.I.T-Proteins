import os, sys
import pandas as pd
from collections import defaultdict

# -----------------------------
# Config
# -----------------------------
TARGET_PROTEIN = "BRD4"
TARGET_N = 50_000
TARGET_COL = "binds"
USECOLS = [
    "id","protein_name","molecule_smiles",
    "buildingblock1_smiles","buildingblock2_smiles","buildingblock3_smiles",
    TARGET_COL
]
OUTPUT = f"train_{TARGET_PROTEIN.lower()}_50k_stratified.parquet"
RANDOM_STATE = 42

# Use GLOBAL positive rate for the 50k sample
BALANCE_SCOPE = "global"

# Safety: don't download the full 10s-of-GB file unless you explicitly allow it.
ALLOW_FULL_DOWNLOAD = False

# Where to look (Kaggle + local-friendly)
CANDIDATE_DIRS = [
    r".",
    r".\data\leash-BELKA",
    r".\data",
    r"C:\kaggle\leash-belka",
    r"C:\kaggle\input\leash-belka",
    r"/kaggle/input/leash-belka",
]
CANDIDATE_FILES = ["train.parquet", "train.csv", "train.parquet.gz", "train.csv.gz"]


# -----------------------------
# Helpers: locate input
# -----------------------------
def find_input_path():
    env_p = os.getenv("LEASH_INPUT_PATH")
    if env_p and os.path.exists(env_p):
        print(f"Using LEASH_INPUT_PATH: {env_p}")
        return env_p

    for d in CANDIDATE_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in CANDIDATE_FILES:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                print(f"Found input at: {p}")
                return p

    try:
        root_base = "/kaggle/input"
        if os.path.isdir(root_base):
            for pref in CANDIDATE_FILES:
                for root, _, files in os.walk(root_base):
                    if pref in files:
                        p = os.path.join(root, pref)
                        print(f"Found input via scan: {p}")
                        return p
    except Exception as e:
        print("Scan of /kaggle/input failed:", e)

    print("Searched these directories for train.[csv|parquet]:")
    for d in CANDIDATE_DIRS:
        print(" -", os.path.abspath(d))
    print("Tip: In Kaggle, attach the competition in 'Add data'. "
          "Or set LEASH_INPUT_PATH to the exact file.")
    return None





# -----------------------------
# Stats: counts & global rate
# -----------------------------
def compute_counts_csv(path, usecols, protein, chunksize=1_000_000):
    counts = defaultdict(int); total = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        c = chunk[chunk["protein_name"] == protein]
        if c.empty:
            continue
        total += len(c)
        vc = c[TARGET_COL].value_counts()
        for k, v in vc.items():
            counts[int(k)] += int(v)
    return dict(counts), total


def compute_counts_parquet_stream(path, usecols, protein):
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except Exception:
        df = pd.read_parquet(path, columns=usecols)
        df = df[df["protein_name"] == protein]
        total = len(df)
        vc = df[TARGET_COL].value_counts()
        return {int(k): int(v) for k, v in vc.items()}, total

    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=usecols, filter=(ds.field("protein_name") == protein))

    counts = defaultdict(int); total = 0
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        df = pa.Table.from_batches([batch]).to_pandas()
        total += len(df)
        if not df.empty:
            vc = df[TARGET_COL].value_counts()
            for k, v in vc.items():
                counts[int(k)] += int(v)
    return dict(counts), total


def compute_global_pos_rate_csv(path, chunksize=1_000_000):
    pos = 0; tot = 0
    for chunk in pd.read_csv(path, usecols=[TARGET_COL], chunksize=chunksize):
        tot += len(chunk)
        pos += (chunk[TARGET_COL] == 1).sum()
    if tot == 0:
        raise ValueError("Global: no rows read.")
    return pos / tot


def compute_global_pos_rate_parquet_stream(path):
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(path, format="parquet")
        tot = 0; pos = 0
        for batch in dataset.scanner(columns=[TARGET_COL]).to_batches():
            col = batch.column(0).to_pandas()
            tot += len(col)
            pos += (col == 1).sum()
        if tot == 0:
            raise ValueError("Global: no rows read.")
        return pos / tot
    except Exception:
        df = pd.read_parquet(path, columns=[TARGET_COL])
        if len(df) == 0:
            raise ValueError("Global: no rows read.")
        return float((df[TARGET_COL] == 1).mean())


# -----------------------------
# Target counts
# -----------------------------
def compute_targets(counts, total_rows, target_n):
    if total_rows <= 0:
        raise ValueError(f"No rows for protein {TARGET_PROTEIN}.")
    if target_n > total_rows:
        raise ValueError(f"Requested {target_n} rows but only {total_rows} for {TARGET_PROTEIN}.")
    c1 = int(counts.get(1, 0))
    p1 = c1 / total_rows
    n1 = int(round(target_n * p1))
    n0 = target_n - n1
    return {0: n0, 1: n1}


def compute_targets_from_rate(p1, target_n):
    n1 = int(round(target_n * p1))
    n0 = target_n - n1
    return {0: n0, 1: n1}


# -----------------------------
# Sampling
# -----------------------------
def sample_from_csv_to_memory(path, usecols, protein, targets, chunksize=1_000_000):
    remaining = {int(k): int(v) for k, v in targets.items()}
    pieces = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        c = chunk[chunk["protein_name"] == protein]
        if c.empty:
            continue
        for y, g in c.groupby(TARGET_COL, observed=True, sort=False):
            y_int = int(y)
            need = remaining.get(y_int, 0)
            if need <= 0:
                continue
            take = min(need, len(g))
            if take > 0:
                pieces.append(g.sample(n=take, random_state=RANDOM_STATE))
                remaining[y_int] -= take
        if all(v <= 0 for v in remaining.values()):
            break
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols)


def sample_from_parquet_to_memory(path, usecols, protein, targets):
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        remaining = {int(k): int(v) for k, v in targets.items()}
        pieces = []

        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=usecols, filter=(ds.field("protein_name") == protein))

        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            df = pa.Table.from_batches([batch]).to_pandas()
            if df.empty:
                continue
            for y, g in df.groupby(TARGET_COL, observed=True, sort=False):
                y_int = int(y)
                need = remaining.get(y_int, 0)
                if need <= 0:
                    continue
                take = min(need, len(g))
                if take > 0:
                    pieces.append(g.sample(n=take, random_state=RANDOM_STATE))
                    remaining[y_int] -= take
            if all(v <= 0 for v in remaining.values()):
                break

        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols)
    except Exception as e:
        print("Arrow streaming unavailable, falling back to pandas (may use more RAM):", e)
        df = pd.read_parquet(path, columns=usecols)
        df = df[df["protein_name"] == protein]
        pieces = []
        for y, g in df.groupby(TARGET_COL, observed=True, sort=False):
            need = int(targets.get(int(y), 0))
            if need > 0 and len(g) > 0:
                pieces.append(g.sample(n=min(need, len(g)), random_state=RANDOM_STATE))
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols)


# -----------------------------
# Exact trimming + shuffle
# -----------------------------
def trim_to_exact_targets(df, targets, target_col=TARGET_COL, random_state=RANDOM_STATE):
    parts = []
    for y, want in targets.items():
        block = df[df[target_col] == y]
        if len(block) < want:
            raise ValueError(f"Need {want} rows for class {y}, have {len(block)}.")
        if len(block) > want:
            block = block.sample(n=want, random_state=random_state + int(y))
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


# -----------------------------
# Writers
# -----------------------------
def write_parquet_zstd(df, path):
    # Ensure minimal size
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].astype("uint8")

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        data_page_size=1 << 16,
        write_statistics=True
    )
    print(f"Wrote: {path} | size MB: {round(os.path.getsize(path)/1024/1024, 2)}")


def write_csv_gz(df, path):
    csv_out = path.replace(".parquet", ".csv.gz")
    df.to_csv(csv_out, index=False, compression="gzip")
    print(f"Wrote: {csv_out} | size MB: {round(os.path.getsize(csv_out)/1024/1024, 2)}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    inp = find_input_path()

    if inp is None:
        in_kaggle = os.path.exists("/kaggle/input")
        if inp is None and not in_kaggle and not ALLOW_FULL_DOWNLOAD:
            raise FileNotFoundError(
                "No train.[csv|parquet] found and ALLOW_FULL_DOWNLOAD is False.\n"
                "- Run this in Kaggle (data is pre-mounted) or\n"
                "- Place train.parquet/train.csv under one of:\n  " +
                "\n  ".join(os.path.abspath(d) for d in CANDIDATE_DIRS)
            )
        if inp is None and ALLOW_FULL_DOWNLOAD:
            print("Attempting FULL download via Kaggle API...")
            dest = r".\data\leash-BELKA"
            try:
                try_kaggle_download(dest)
                inp = find_input_path()
            except Exception as e:
                print("Kaggle download failed:", e)

    if inp is None:
        raise FileNotFoundError("Could not find or fetch train.[csv|parquet].")

    print("Using input file:", os.path.abspath(inp))
    is_csv = inp.lower().endswith(".csv") or inp.lower().endswith(".csv.gz")

    # PASS 1: stats for BRD4
    if is_csv:
        counts, total = compute_counts_csv(inp, USECOLS, TARGET_PROTEIN)
        p1_global = compute_global_pos_rate_csv(inp)
    else:
        counts, total = compute_counts_parquet_stream(inp, USECOLS, TARGET_PROTEIN)
        p1_global = compute_global_pos_rate_parquet_stream(inp)

    print(f"{TARGET_PROTEIN} total rows: {total} | class counts: {counts}")

    # Decide targets based on scope (global is your requirement)
    if BALANCE_SCOPE == "global":
        targets = compute_targets_from_rate(p1_global, TARGET_N)
        print(f"Using GLOBAL balance p1={p1_global:.6f} -> targets: {targets} | sum={sum(targets.values())}")
    else:
        targets = compute_targets(counts, total, TARGET_N)
        print(f"Using {TARGET_PROTEIN}-specific balance -> targets: {targets} | sum={sum(targets.values())}")

    # PASS 2: sample BRD4 rows
    if is_csv:
        sampled = sample_from_csv_to_memory(inp, USECOLS, TARGET_PROTEIN, targets)
    else:
        sampled = sample_from_parquet_to_memory(inp, USECOLS, TARGET_PROTEIN, targets)

    if sampled.empty:
        raise RuntimeError("Sampling returned no rows; check inputs and filters.")

    # Trim to exact target per class, keep only USECOLS, final shuffle
    sampled = trim_to_exact_targets(sampled, targets, target_col=TARGET_COL)
    sampled = sampled[USECOLS].reset_index(drop=True)
    assert len(sampled) == TARGET_N, f"Expected {TARGET_N}, got {len(sampled)}"

    # Overwrite any previous output explicitly
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    # Write compact Parquet and (optional) CSV.GZ
    write_parquet_zstd(sampled, OUTPUT)
    # Uncomment next line if you also want a CSV.GZ alongside Parquet:
    # write_csv_gz(sampled, OUTPUT)

    # Quick sanity check
    out = pd.read_parquet(OUTPUT, columns=[TARGET_COL])
    print("Rows:", len(out))
    print("binds proportion in 50k (global-matched):", out[TARGET_COL].mean())