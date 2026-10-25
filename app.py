import json
import os
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pulsecheck_isolation_forest_tuned.joblib"
CONFIG_PATH = BASE_DIR / "pulsecheck_tuned_config.json"

SAMPLE_INTERVAL = 15
ROLLING_1M = 4
ROLLING_5M = 20

st.set_page_config(
    page_title="PulseCheck AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Load the trained model and configuration
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, config


if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.error(
        "Model files not found. Place the tuned model and configuration "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model, config = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


FEATURES = config["features"]
THRESHOLD = float(config["anomaly_threshold"])


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=240)

if "previous_sample" not in st.session_state:
    st.session_state.previous_sample = None

if "process_baseline" not in st.session_state:
    st.session_state.process_baseline = {}

if "monitoring" not in st.session_state:
    st.session_state.monitoring = True


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def build_features(history):
    """
    Build the exact feature names and rolling-window sizes
    expected by the trained model.

    Requires 20 historical readings to calculate the
    five-minute rolling statistics.
    """

    df = pd.DataFrame(history)

    if len(df) < ROLLING_5M:
        return None

    cpu = df["cpu_usage"]
    memory = df["memory_usage"]

    features = pd.DataFrame(index=df.index)

    features["cpu_usage"] = cpu
    features["memory_usage"] = memory

    features["cpu_usage_change"] = cpu.diff()
    features["memory_usage_change"] = memory.diff()

    features["cpu_usage_rolling_mean_1m"] = (
        cpu.rolling(window=ROLLING_1M, min_periods=ROLLING_1M)
        .mean()
    )

    features["memory_usage_rolling_mean_1m"] = (
        memory.rolling(window=ROLLING_1M, min_periods=ROLLING_1M)
        .mean()
    )

    features["cpu_usage_rolling_mean_5m"] = (
        cpu.rolling(window=ROLLING_5M, min_periods=ROLLING_5M)
        .mean()
    )

    features["memory_usage_rolling_mean_5m"] = (
        memory.rolling(window=ROLLING_5M, min_periods=ROLLING_5M)
        .mean()
    )

    features["cpu_usage_rolling_std_5m"] = (
        cpu.rolling(window=ROLLING_5M, min_periods=ROLLING_5M)
        .std()
    )

    features["memory_usage_rolling_std_5m"] = (
        memory.rolling(window=ROLLING_5M, min_periods=ROLLING_5M)
        .std()
    )

    # Preserve the feature order used during training.
    features = features[FEATURES]

    # Only the latest observation is needed for prediction.
    latest = features.iloc[[-1]]

    if latest.isna().any().any():
        return None

    return latest


# --------------------------------------------------
# System monitoring
# --------------------------------------------------

def collect_metrics():
    # Measure CPU utilization over a short interval.
    # This blocks for 1 second but gives a more meaningful reading.
    cpu_usage = psutil.cpu_percent(interval=1.0)

    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))

    return {
        "timestamp": pd.Timestamp.now(),
        "cpu_usage": float(cpu_usage),
        "memory_usage": float(memory.percent),
        "disk_usage": float(disk.percent),
        "memory_used_gb": round(
            memory.used / (1024 ** 3), 2
        ),
    }


def get_top_processes(limit=8):
    processes = []

    for proc in psutil.process_iter(
        ["pid", "name", "memory_percent"]
    ):
        try:
            # CPU usage is calculated between calls, so the first
            # reading for a process may be zero.
            info = proc.info

            processes.append({
                "PID": info["pid"],
                "Process": info["name"] or "Unknown",
                "RAM (%)": round(
                    info["memory_percent"] or 0, 2
                ),
            })

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue

    return (
        pd.DataFrame(processes)
        .sort_values("RAM (%)", ascending=False)
        .head(limit)
        if processes
        else pd.DataFrame(columns=["PID", "Process", "RAM (%)"])
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("🩺 PulseCheck AI")
    st.caption("Experimental system anomaly detection")

    st.divider()

    st.write("### Monitoring")

    st.session_state.monitoring = st.toggle(
        "Live monitoring",
        value=st.session_state.monitoring,
    )

    st.write(f"**Sampling interval:** {SAMPLE_INTERVAL} seconds")
    st.write(f"**Detection threshold:** {THRESHOLD:.4f}")

    st.divider()

    st.caption(
        "This tool identifies unusual resource patterns. "
        "It does not diagnose hardware failure or malware."
    )


# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("PulseCheck AI")
st.markdown(
    "### Understand what your laptop is doing."
)

st.write(
    "Monitor resource usage, detect unusual patterns, "
    "and investigate processes consuming your system resources."
)


# --------------------------------------------------
# Collect a new sample
# --------------------------------------------------

if st.session_state.monitoring:
    sample = collect_metrics()

    # Avoid duplicate samples if Streamlit reruns before the
    # next refresh interval.
    last_sample = (
        st.session_state.history[-1]
        if st.session_state.history
        else None
    )

    if (
        last_sample is None
        or (
            sample["timestamp"] - last_sample["timestamp"]
        ).total_seconds() >= SAMPLE_INTERVAL - 2
    ):
        st.session_state.history.append(sample)


# --------------------------------------------------
# Predict anomaly
# --------------------------------------------------

history = list(st.session_state.history)

if history:
    latest_sample = history[-1]
    feature_row = build_features(history)

    if feature_row is not None:
        try:
            anomaly_score = float(
                -model.decision_function(feature_row)[0]
            )

            is_anomaly = anomaly_score >= THRESHOLD

            latest_sample["anomaly_score"] = anomaly_score
            latest_sample["is_anomaly"] = bool(is_anomaly)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            latest_sample["anomaly_score"] = np.nan
            latest_sample["is_anomaly"] = False

    else:
        latest_sample["anomaly_score"] = np.nan
        latest_sample["is_anomaly"] = False


# --------------------------------------------------
# System overview
# --------------------------------------------------

if not history:
    st.info("Waiting for the first system measurement...")
    st.stop()

latest = history[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "CPU usage",
    f"{latest['cpu_usage']:.1f}%",
)

col2.metric(
    "Memory usage",
    f"{latest['memory_usage']:.1f}%",
)

col3.metric(
    "Disk usage",
    f"{latest['disk_usage']:.1f}%",
)

col4.metric(
    "Memory used",
    f"{latest['memory_used_gb']:.2f} GB",
)

st.caption(
    f"Last sample: {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
)


# --------------------------------------------------
# Anomaly status
# --------------------------------------------------

st.divider()

st.subheader("System status")

cpu = float(latest_sample["cpu_usage"])
memory = float(latest_sample["memory_usage"])
disk = float(latest_sample.get("disk_usage", 0.0))
score = latest_sample.get("anomaly_score", np.nan)

# Separate model output from resource-specific guidance.
if pd.isna(score) or latest_sample.get("is_anomaly") is None:
    st.info(
        "Prediction unavailable. The dashboard is collecting data "
        "or encountered an error while evaluating the current sample."
    )

elif latest_sample["is_anomaly"]:
    st.warning(
        "⚠️ Unusual resource pattern detected. "
        "Investigate the measurements below before drawing conclusions."
    )

else:
    st.success(
        "No unusual resource pattern detected in the current sample. "
        "This does not guarantee that the system is problem-free."
    )

# Show the actual model score when available.
if not pd.isna(score):
    st.caption(
        f"Anomaly score: {score:.4f} | "
        f"Alert threshold: {THRESHOLD:.4f}"
    )

st.markdown("#### Resource investigation")

# CPU guidance
if cpu >= 80:
    st.error(
        f"**CPU: {cpu:.1f}% — High utilization.** "
        "Check Task Manager for processes using substantial CPU."
    )
elif cpu >= 50:
    st.warning(
        f"**CPU: {cpu:.1f}% — Elevated at this sample.** "
        "Check whether usage remains elevated over multiple samples."
    )
else:
    st.write(
        f"**CPU: {cpu:.1f}%** — No high-CPU warning from this reading."
    )

# Memory guidance
if memory >= 90:
    st.error(
        f"**Memory: {memory:.1f}% — Very high utilization.** "
        "Check Task Manager and close unnecessary applications if appropriate."
    )
elif memory >= 80:
    st.warning(
        f"**Memory: {memory:.1f}% — High utilization.** "
        "Inspect the processes using the most RAM."
    )
else:
    st.write(
        f"**Memory: {memory:.1f}%** — No high-memory warning from this reading."
    )

# Disk is a separate system metric, not a model input.
if disk >= 90:
    st.warning(
        f"**Disk: {disk:.1f}% full — Storage is getting limited.** "
        "Review available space and large files if needed."
    )
else:
    st.write(
        f"**Disk: {disk:.1f}% full** — Check available space if storage becomes limited."
    )

st.caption(
    "These are investigation suggestions, not diagnoses. "
    "The model flags unusual CPU/memory patterns; disk usage is reported separately."
)

# --------------------------------------------------
# Charts
# --------------------------------------------------

st.divider()
st.subheader("Resource usage history")

chart_df = pd.DataFrame(history)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_df["timestamp"],
        y=chart_df["cpu_usage"],
        mode="lines",
        name="CPU usage",
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_df["timestamp"],
        y=chart_df["memory_usage"],
        mode="lines",
        name="Memory usage",
    )
)

fig.update_layout(
    height=380,
    xaxis_title="Time",
    yaxis_title="Usage (%)",
    yaxis=dict(range=[0, 100]),
    hovermode="x unified",
    legend=dict(orientation="h"),
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Recent observations
# --------------------------------------------------

st.subheader("Recent measurements")

recent_df = pd.DataFrame(list(st.session_state.history)[-20:])

if not recent_df.empty:
    display_columns = [
        "timestamp",
        "cpu_usage",
        "memory_usage",
        "disk_usage",
        "anomaly_score",
        "is_anomaly",
    ]

    for col in display_columns:
        if col not in recent_df.columns:
            recent_df[col] = np.nan

    recent_df = recent_df[display_columns].sort_values(
        "timestamp",
        ascending=False,
    )

    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Processes
# --------------------------------------------------

st.divider()
st.subheader("Processes using the most memory")

process_df = get_top_processes()

st.dataframe(
    process_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Download recent history
# --------------------------------------------------

st.divider()

csv_data = pd.DataFrame(
    list(st.session_state.history)
).to_csv(index=False)

st.download_button(
    label="Download monitoring history (CSV)",
    data=csv_data,
    file_name="pulsecheck_monitoring_history.csv",
    mime="text/csv",
)


# --------------------------------------------------
# Automatic refresh
# --------------------------------------------------

if st.session_state.monitoring:
    st_autorefresh(
        interval=SAMPLE_INTERVAL * 1000,
        key="pulsecheck_refresh",
    )