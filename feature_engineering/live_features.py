"""
Live-computable feature set — single source of truth.

The NSL-KDD schema has 41 features, but the live sensor (live_ids_v2.py) can
only derive a subset of them from raw packet headers. The remaining ~28
features are application-layer (hot, logged_in, num_failed_logins, ...) or
require per-destination-host history (dst_host_*) and are zeroed at runtime.

Training on all 41 features and then serving a vector where most are zero is a
train/serve distribution mismatch — the model leans on columns that are always
zero in production. To keep training and inference aligned, BOTH the trainer
(train_random_forest_v2.py) and the sensor (live_ids_v2.py) import the feature
lists below and emit features in exactly this order.

Order convention: numeric features first, then the one-hot encoding of the
categorical features appended — i.e. hstack([numeric_vector, encoder.transform]).
"""

from typing import List

# Numeric features we can compute from packet headers / flow state.
#
# Caveats worth knowing (documented honestly rather than hidden):
#   - count / srv_count: NSL-KDD defines these over a 2s window of connections
#     to the same host/service. The live sensor approximates them with per-flow
#     packet intensity. The signal is correlated (both rise under scans/floods)
#     but not identical.
#   - serror_rate / rerror_rate: derived live from observed SYN-without-response
#     and RST-without-handshake counts.
LIVE_NUMERIC_FEATURES: List[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
]

# Categorical features, one-hot encoded. Derivable from packet headers:
#   protocol_type -> IP proto, service -> dst port mapping, flag -> TCP state.
LIVE_CATEGORICAL_FEATURES: List[str] = [
    "protocol_type",
    "service",
    "flag",
]

# Full ordered list (numeric then categorical), used for column selection.
LIVE_FEATURES: List[str] = LIVE_NUMERIC_FEATURES + LIVE_CATEGORICAL_FEATURES

# Artifact paths for the live-aligned model produced by train_random_forest_v2.py.
LIVE_MODEL_PATH = "models/rf_live.pkl"
LIVE_ENCODER_PATH = "models/rf_live_encoder.pkl"
LIVE_FEATURE_META_PATH = "models/rf_live_features.json"
