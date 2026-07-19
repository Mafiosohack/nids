"""
CIC-IDS2017 loader — shared data layer for both retraining paths.

The dataset ships as 8 per-day CSVs (Mon–Fri) in two forms that we keep in
separate folders under data/cicids2017/:

  features/  — the "MachineLearningCSV" release: 78 flow-statistics features +
               Label, no identifiers. Used by the DEDICATED FLOW MODEL (Model A).
  labelled/  — the "GeneratedLabelledFlows" release: the same features plus
               Flow ID / Source-Dest IP / ports / Protocol / Timestamp. Used by
               the LIVE-SCHEMA RETRAIN (Model B), which needs Protocol + IPs to
               reconstruct the sensor's protocol_type / service / flag / land.

Both loaders return a cleaned multiclass label (per attack type) and a derived
binary label (BENIGN vs ATTACK). Known dataset quirks handled here:

  * Header whitespace — every column name has a leading space (" Flow Duration").
  * Non-ASCII label byte — the three "Web Attack" classes contain 0x96 (a
    Windows-1252 en dash); we read as latin-1 and normalise to ASCII hyphens.
  * Inf / NaN — Flow Bytes/s and Flow Packets/s divide by a zero-duration flow,
    producing Infinity; we map Inf -> NaN. HistGradientBoosting consumes NaN
    natively; the live-schema path fills remaining NaN with 0.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "cicids2017" / "features"
LABELLED_DIR = ROOT / "data" / "cicids2017" / "labelled"

# Identifier columns that exist only in the labelled/ set. They must never be
# fed to Model A (they would leak host identity / capture order into the model);
# Model B consumes a couple of them (IPs, Protocol) explicitly, not as features.
IDENTIFIER_COLS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP",
    "Protocol", "Timestamp",
]

LABEL_COL = "Label"


# ─────────────────────────────────────────────────────────────────────
#  LABEL CLEANING
# ─────────────────────────────────────────────────────────────────────
def clean_label(raw: str) -> str:
    """Normalise a raw CIC-IDS label to ASCII, stable spacing.

    "Web Attack \x96 Brute Force" -> "Web Attack - Brute Force".
    """
    s = str(raw).strip()
    # Replace the non-ASCII byte (0x96 en dash) in the Web Attack labels with a
    # plain hyphen; leave already-ASCII names (e.g. "FTP-Patator") untouched.
    s = re.sub(r"[^\x00-\x7f]+", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_and_dedup(cols) -> List[str]:
    """Strip header whitespace and make column names unique (defensive:
    CIC-IDS has a duplicated 'Fwd Header Length' that pandas already suffixes)."""
    out, seen = [], {}
    for c in cols:
        name = str(c).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


# Source columns the live-schema mapping (Model B) needs. Reading only these
# from the 85-column labelled/ set keeps memory ~6x lower than a full read.
LIVE_SOURCE_COLS = {
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "RST Flag Count",
    "Protocol", "Destination Port", "Source IP", "Destination IP", LABEL_COL,
}


def _list_files(directory: Path) -> List[str]:
    files = sorted(glob.glob(str(directory / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSVs found in {directory}. Expected the 8 CIC-IDS2017 day files."
        )
    return files


def _read_one(path: str, keep: set | None = None) -> pd.DataFrame:
    """Read one CSV with header whitespace stripped. If `keep` is given, only
    those (stripped) columns are loaded, to save memory."""
    usecols = None
    if keep is not None:
        usecols = lambda c: str(c).strip() in keep  # noqa: E731
    df = pd.read_csv(path, encoding="latin-1", low_memory=False, usecols=usecols)
    df.columns = _strip_and_dedup(df.columns)
    return df


def _clean_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows with a missing/blank Label (a known CIC-IDS quirk in the
    labelled/ files) and normalise the rest. Returns (df, n_dropped)."""
    before = len(df)
    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].map(clean_label)
    df = df[df[LABEL_COL].str.lower().ne("nan") & (df[LABEL_COL] != "")]
    return df, before - len(df)


def _coerce_numeric(df: pd.DataFrame, cols: List[str],
                    dtype: str = "float32") -> pd.DataFrame:
    """Coerce feature columns to numeric, mapping +/-Inf to NaN."""
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)
    return num.astype(dtype)


# ─────────────────────────────────────────────────────────────────────
#  MODEL A — full 78 flow features (features/ set)
# ─────────────────────────────────────────────────────────────────────
def load_flow_features(
    nrows: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Load the full flow-feature matrix for the dedicated model.

    Returns (X, y_multi, y_bin, feature_names) where X is float32 with NaN
    preserved (HistGradientBoosting handles NaN natively), y_multi is the
    cleaned per-attack-type label, and y_bin is 0 (BENIGN) / 1 (attack).
    """
    # Process each day file down to a compact float32 block, freeing the raw
    # float64 frame immediately — never hold all 8 raw frames at once.
    x_parts, y_parts, feature_names, total_dropped = [], [], None, 0
    for f in _list_files(FEATURES_DIR):
        df = _read_one(f)
        df, dropped = _clean_labels(df)
        total_dropped += dropped
        if feature_names is None:
            feature_names = [c for c in df.columns if c != LABEL_COL]
        x_parts.append(_coerce_numeric(df, feature_names))
        y_parts.append(df[LABEL_COL].reset_index(drop=True))
        del df
    if total_dropped:
        print(f"    [loader] dropped {total_dropped:,} rows with missing Label")

    X = pd.concat(x_parts, ignore_index=True)
    del x_parts
    y = pd.concat(y_parts, ignore_index=True)
    del y_parts

    if nrows and nrows < len(X):
        idx = X.sample(n=nrows, random_state=42).index
        X = X.loc[idx].reset_index(drop=True)
        y = y.loc[idx].reset_index(drop=True)

    y_multi = y.astype("category")
    y_bin = (y != "BENIGN").astype(int)
    return X, y_multi, y_bin, feature_names


# ─────────────────────────────────────────────────────────────────────
#  MODEL B — CIC-IDS mapped onto the live sensor's 10+3 feature schema
# ─────────────────────────────────────────────────────────────────────
#  The mapping below MIRRORS the runtime semantics in live_ids_v2.py so that a
#  model trained here can be served by the (future) live sensor without a
#  train/serve skew. Where CIC-IDS lacks a direct signal we approximate and say
#  so. The PORT_SERVICE_MAP and flag logic are intentional copies of
#  live_ids_v2.py's — that file is the source of truth; keep them in sync.

# NSL-KDD-style port -> service map (mirror of live_ids_v2.PORT_SERVICE_MAP).
PORT_SERVICE_MAP = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 67: "domain_u", 68: "domain_u", 80: "http", 110: "pop_3",
    111: "sunrpc", 113: "auth", 119: "nntp", 123: "ntp_u", 143: "imap4",
    179: "bgp", 194: "IRC", 389: "ldap", 443: "http_443", 445: "netbios_ssn",
    465: "smtp", 514: "shell", 515: "printer", 587: "smtp", 993: "imap4",
    995: "pop_3", 1433: "mssql", 1521: "sql_net", 1900: "other", 3306: "sql_net",
    3389: "other", 5432: "sql_net", 5900: "other", 6667: "IRC", 8080: "http_8001",
    8443: "http_443", 9200: "other", 11211: "other",
}


def _protocol_to_type(proto: pd.Series) -> pd.Series:
    """IP protocol number -> {tcp, udp, icmp}, clamped to the sensor's 3-value
    vocabulary (the live sensor only ever emits these three)."""
    p = pd.to_numeric(proto, errors="coerce")
    out = np.where(p == 6, "tcp", np.where(p == 17, "udp", "icmp"))
    return pd.Series(out, index=proto.index)


def _port_to_service(port: pd.Series, proto_type: pd.Series) -> pd.Series:
    p = pd.to_numeric(port, errors="coerce").fillna(-1).astype(int)
    svc = p.map(PORT_SERVICE_MAP).fillna("other")
    svc = svc.where(proto_type != "icmp", "eco_i")  # icmp -> eco_i, mirror sensor
    return svc


def _derive_flag(syn, syn_ack, fin, rst_dst) -> np.ndarray:
    """Vectorised mirror of live_ids_v2.derive_kdd_flag (rst_src is unobservable
    in CIC-IDS aggregates, so RSTO never fires; all RSTs treated as responder)."""
    conds = [
        fin & syn_ack,            # SF  — normal establish + terminate
        rst_dst & syn_ack,        # RSTR — established then RST from responder
        rst_dst & ~syn_ack,       # REJ — rejected
        syn & syn_ack,            # S1  — established, still open
        syn & ~syn_ack,           # S0  — SYN, no response
    ]
    choices = ["SF", "RSTR", "REJ", "S1", "S0"]
    return np.select(conds, choices, default="OTH")


LIVE_NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "land", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
]
LIVE_CATEGORICAL = ["protocol_type", "service", "flag"]


def _map_live_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map one cleaned labelled/ day-frame to the 13-column live schema."""
    # Flag-count booleans (CIC aggregates a whole flow, so these are presence
    # flags over the flow, the closest analogue to the sensor's per-flow state).
    syn = pd.to_numeric(df["SYN Flag Count"], errors="coerce").fillna(0) > 0
    ack = pd.to_numeric(df["ACK Flag Count"], errors="coerce").fillna(0) > 0
    fin = pd.to_numeric(df["FIN Flag Count"], errors="coerce").fillna(0) > 0
    rst = pd.to_numeric(df["RST Flag Count"], errors="coerce").fillna(0) > 0
    syn_ack = syn & ack                       # approx: no separate SYN-ACK count

    serror = (syn & ~syn_ack).astype("float32")   # SYN with no completion
    rerror = rst.astype("float32")                 # any RST -> REJ/RSTR family

    proto_type = _protocol_to_type(df["Protocol"])
    service = _port_to_service(df["Destination Port"], proto_type)
    flag = _derive_flag(syn.to_numpy(), syn_ack.to_numpy(),
                        fin.to_numpy(), rst.to_numpy())

    fwd = pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
    bwd = pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
    count = (fwd + bwd).astype("float32")

    land = (df["Source IP"].astype(str) == df["Destination IP"].astype(str)) \
        .astype("float32")

    out = pd.DataFrame({
        # Flow Duration is microseconds in CIC-IDS; the sensor's duration is
        # seconds — convert so the trained distribution matches serving.
        "duration": pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0)
                    .astype("float32") / 1e6,
        "src_bytes": pd.to_numeric(df["Total Length of Fwd Packets"],
                                   errors="coerce").fillna(0).astype("float32"),
        "dst_bytes": pd.to_numeric(df["Total Length of Bwd Packets"],
                                   errors="coerce").fillna(0).astype("float32"),
        "land": land.values,
        "count": count.values,
        "srv_count": count.values,          # sensor sets srv_count = count
        "serror_rate": serror.values,
        "srv_serror_rate": serror.values,
        "rerror_rate": rerror.values,
        "srv_rerror_rate": rerror.values,
        "protocol_type": proto_type.values,
        "service": service.values,
        "flag": flag,
    })
    return out


def load_live_mapped(
    nrows: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Map the labelled/ CIC-IDS set onto the sensor's live feature schema.

    Returns (X, y_multi, y_bin, numeric_features, categorical_features) where X
    has the 10 numeric + 3 categorical columns the live sensor emits, in the
    canonical order. Numeric NaN is filled with 0 (RandomForest / one-hot path
    has no native NaN support and the sensor never emits NaN). Only the ~14
    source columns the mapping needs are read from disk, per file.
    """
    x_parts, y_parts, total_dropped = [], [], 0
    for f in _list_files(LABELLED_DIR):
        df = _read_one(f, keep=LIVE_SOURCE_COLS)
        df, dropped = _clean_labels(df)
        total_dropped += dropped
        x_parts.append(_map_live_frame(df))
        y_parts.append(df[LABEL_COL].reset_index(drop=True))
        del df
    if total_dropped:
        print(f"    [loader] dropped {total_dropped:,} rows with missing Label")

    X = pd.concat(x_parts, ignore_index=True)
    del x_parts
    y = pd.concat(y_parts, ignore_index=True)
    del y_parts

    if nrows and nrows < len(X):
        idx = X.sample(n=nrows, random_state=42).index
        X = X.loc[idx].reset_index(drop=True)
        y = y.loc[idx].reset_index(drop=True)

    y_multi = y.astype("category")
    y_bin = (y != "BENIGN").astype(int)
    return X, y_multi, y_bin, LIVE_NUMERIC, LIVE_CATEGORICAL


if __name__ == "__main__":
    # Smoke test on a small sample.
    print("[*] flow features sample...")
    X, ym, yb, feats = load_flow_features(nrows=20000)
    print(f"    X={X.shape}  n_features={len(feats)}  classes={list(ym.cat.categories)}")
    print(f"    attack ratio={yb.mean():.3f}  NaN cells={int(X.isna().sum().sum())}")
    print("[*] live-mapped sample...")
    Xl, ym2, yb2, num, cat = load_live_mapped(nrows=20000)
    print(f"    X={Xl.shape}  numeric={num}\n    categorical={cat}")
    print(Xl.head(3).to_string())
    print(f"    flag values: {sorted(Xl['flag'].unique())}")
    print(f"    service values (top): {Xl['service'].value_counts().head(6).to_dict()}")
