import json
import os
import platform
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
# Page & Configuration Setup
# --------------------------------------------------

st.set_page_config(
    page_title="PulseCheck AI | System Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pulsecheck_isolation_forest_tuned.joblib"
CONFIG_PATH = BASE_DIR / "pulsecheck_tuned_config.json"

SAMPLE_INTERVAL = 15
ROLLING_1M = 4
ROLLING_5M = 20


# --------------------------------------------------
# Design B: Professional Observability Stylesheet
# --------------------------------------------------

OBSERVABILITY_CSS = """
<style>
/* Base application typography and dark charcoal theme */
:root {
    --bg-canvas: #171916;
    --bg-surface: #20231F;
    --bg-surface-subtle: #1C1F1B;
    --border-subtle: #343831;
    --border-dim: #272B24;
    --text-primary: #ECEEE9;
    --text-secondary: #8E9889;
    --text-dim: #6B7467;
    --accent-sage: #A3B18A;
    --accent-amber: #D6A45D;
    --accent-red: #D46E65;
    --accent-blue: #7EA393;
}

/* Global resets & background */
.stApp {
    background-color: var(--bg-canvas);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #191B18 !important;
    border-right: 1px solid var(--border-subtle);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em;
}

/* Compact header container */
.monitor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.25rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    margin-bottom: 1.25rem;
}

.brand-group {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.brand-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: rgba(163, 177, 138, 0.12);
    border: 1px solid rgba(163, 177, 138, 0.3);
    border-radius: 4px;
    color: var(--accent-sage);
    font-size: 14px;
}

.brand-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.2;
}

.brand-badge {
    font-size: 0.68rem;
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 2px 6px;
    background: rgba(142, 152, 137, 0.12);
    border: 1px solid var(--border-dim);
    color: var(--text-secondary);
    border-radius: 3px;
    text-transform: uppercase;
}

.header-meta {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.timestamp-text {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
}

/* Status Pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 3px 8px;
    border-radius: 3px;
}

.status-live {
    background: rgba(163, 177, 138, 0.15);
    border: 1px solid rgba(163, 177, 138, 0.4);
    color: var(--accent-sage);
}

.status-warmup {
    background: rgba(214, 164, 93, 0.15);
    border: 1px solid rgba(214, 164, 93, 0.4);
    color: var(--accent-amber);
}

.status-paused {
    background: rgba(142, 152, 137, 0.12);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}

.dot-live { background-color: var(--accent-sage); }
.dot-warmup { background-color: var(--accent-amber); }
.dot-paused { background-color: var(--text-dim); }

/* Resource Metric Cards */
.metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 110px;
    transition: border-color 0.15s ease;
}

.metric-card:hover {
    border-color: #42473E;
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
}

.metric-value {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
    margin: 0.25rem 0;
}

.metric-subtext {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin: 0;
}

.badge-nominal {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 5px;
    border-radius: 3px;
    background: rgba(163, 177, 138, 0.1);
    color: var(--accent-sage);
    border: 1px solid rgba(163, 177, 138, 0.25);
}

.badge-elevated {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 5px;
    border-radius: 3px;
    background: rgba(214, 164, 93, 0.12);
    color: var(--accent-amber);
    border: 1px solid rgba(214, 164, 93, 0.3);
}

.badge-critical {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 5px;
    border-radius: 3px;
    background: rgba(212, 110, 101, 0.12);
    color: var(--accent-red);
    border: 1px solid rgba(212, 110, 101, 0.3);
}

/* Anomaly Detection Banner Card */
.panel-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 1.25rem;
}

.panel-title {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.75rem;
}

.anomaly-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.25rem;
    border-radius: 5px;
    margin-bottom: 1rem;
}

.anomaly-banner.state-normal {
    background: rgba(163, 177, 138, 0.08);
    border: 1px solid rgba(163, 177, 138, 0.3);
}

.anomaly-banner.state-anomaly {
    background: rgba(214, 164, 93, 0.1);
    border: 1px solid rgba(214, 164, 93, 0.4);
}

.anomaly-banner.state-warmup {
    background: rgba(142, 152, 137, 0.08);
    border: 1px solid var(--border-subtle);
}

.anomaly-banner.state-unavailable {
    background: rgba(212, 110, 101, 0.08);
    border: 1px solid rgba(212, 110, 101, 0.3);
}

.banner-left {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.banner-status-tag {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.banner-desc {
    font-size: 0.85rem;
    color: var(--text-primary);
    margin: 0;
}

.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    background: var(--bg-surface-subtle);
    padding: 0.75rem 1rem;
    border: 1px solid var(--border-dim);
    border-radius: 4px;
}

.telemetry-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.telemetry-item-label {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.telemetry-item-value {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Resource Investigation Diagnostics */
.diag-card {
    background: var(--bg-surface-subtle);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 0.85rem 1rem;
    height: 100%;
}

.diag-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}

.diag-resource {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
}

.diag-body {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.4;
    margin: 0;
}

/* Section Header Bar */
.section-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border-dim);
}

.section-title {
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin: 0;
}

.section-subtitle {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin: 0;
}

/* Custom styled Streamlit controls */
.stButton > button {
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 0.75rem;
    padding: 0.4rem 0.8rem;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: #272B25;
    border-color: #4A5144;
    color: var(--text-primary);
}

/* Dataframe clean styling overrides */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    overflow: hidden;
}
</style>
"""

st.markdown(OBSERVABILITY_CSS, unsafe_allow_html=True)


# --------------------------------------------------
# Model & Artifact Loader
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
# Session State Initialization
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
# Feature Engineering (Preserved)
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
        cpu.rolling(window=ROLLING_1M, min_periods=ROLLING_1M).mean()
    )

    features["memory_usage_rolling_mean_1m"] = (
        memory.rolling(window=ROLLING_1M, min_periods=ROLLING_1M).mean()
    )

    features["cpu_usage_rolling_mean_5m"] = (
        cpu.rolling(window=ROLLING_5M, min_periods=ROLLING_5M).mean()
    )

    features["memory_usage_rolling_mean_5m"] = (
        memory.rolling(window=ROLLING_5M, min_periods=ROLLING_5M).mean()
    )

    features["cpu_usage_rolling_std_5m"] = (
        cpu.rolling(window=ROLLING_5M, min_periods=ROLLING_5M).std()
    )

    features["memory_usage_rolling_std_5m"] = (
        memory.rolling(window=ROLLING_5M, min_periods=ROLLING_5M).std()
    )

    # Preserve the exact feature order used during training.
    features = features[FEATURES]

    # Only the latest observation is needed for prediction.
    latest = features.iloc[[-1]]

    if latest.isna().any().any():
        return None

    return latest


# --------------------------------------------------
# System Telemetry & Process Inspection
# --------------------------------------------------

def collect_metrics():
    # Measure CPU utilization over a 1-second sample window
    cpu_usage = psutil.cpu_percent(interval=1.0)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))

    return {
        "timestamp": pd.Timestamp.now(),
        "cpu_usage": float(cpu_usage),
        "memory_usage": float(memory.percent),
        "disk_usage": float(disk.percent),
        "memory_used_gb": round(memory.used / (1024 ** 3), 2),
    }


def get_top_processes(limit=8):
    processes = []
    for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
        try:
            info = proc.info
            processes.append({
                "PID": info["pid"],
                "Process": info["name"] or "Unknown",
                "RAM (%)": round(info["memory_percent"] or 0, 2),
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
# Data Sampling & Ingestion
# --------------------------------------------------

if st.session_state.monitoring:
    sample = collect_metrics()

    last_sample = (
        st.session_state.history[-1] if st.session_state.history else None
    )

    # Avoid duplicate samples during premature reruns
    if (
        last_sample is None
        or (sample["timestamp"] - last_sample["timestamp"]).total_seconds()
        >= SAMPLE_INTERVAL - 2
    ):
        st.session_state.history.append(sample)


# --------------------------------------------------
# Model Prediction Execution
# --------------------------------------------------

history = list(st.session_state.history)

if history:
    latest_sample = history[-1]
    feature_row = build_features(history)

    if feature_row is not None:
        try:
            anomaly_score = float(-model.decision_function(feature_row)[0])
            is_anomaly = anomaly_score >= THRESHOLD
            latest_sample["anomaly_score"] = anomaly_score
            latest_sample["is_anomaly"] = bool(is_anomaly)
        except Exception as exc:
            st.error(f"Prediction inference error: {exc}")
            latest_sample["anomaly_score"] = np.nan
            # Use None, not False — False would silently show NORMAL on the dashboard.
            latest_sample["is_anomaly"] = None
    else:
        # feature_row is None during warmup or when features contain NaN.
        latest_sample["anomaly_score"] = np.nan
        latest_sample["is_anomaly"] = None


# --------------------------------------------------
# Sidebar (Controls & Telemetry Metadata)
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 0.25rem;">
            <span style="font-size: 1.1rem;">🩺</span>
            <span style="font-weight: 600; font-size: 1rem; color: #ECEEE9;">PulseCheck AI</span>
        </div>
        <div style="font-family: ui-monospace, Consolas, monospace; font-size: 0.68rem; color: #8E9889; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1.5rem;">
            Observability Console
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Sampling Controls")
    st.session_state.monitoring = st.toggle(
        "Live monitoring",
        value=st.session_state.monitoring,
    )

    st.markdown(
        f"""
        <div style="background:#20231F; border:1px solid #343831; border-radius:4px; padding:0.75rem; margin-top:0.5rem; margin-bottom:1rem; font-family: ui-monospace, Consolas, monospace; font-size: 0.75rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8E9889;">Sample Interval</span>
                <span style="color:#ECEEE9; font-weight:600;">{SAMPLE_INTERVAL}s</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8E9889;">Warm-up Buffer</span>
                <span style="color:#ECEEE9; font-weight:600;">{ROLLING_5M} samples</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8E9889;">Alert Threshold</span>
                <span style="color:#D6A45D; font-weight:600;">{THRESHOLD:.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Buffer Telemetry")
    samples_count = len(history)
    st.markdown(
        f"""
        <div style="background:#20231F; border:1px solid #343831; border-radius:4px; padding:0.75rem; margin-bottom:1rem; font-family: ui-monospace, Consolas, monospace; font-size: 0.75rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8E9889;">Buffer Usage</span>
                <span style="color:#ECEEE9;">{samples_count} / 240 samples</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8E9889;">Rolling Window</span>
                <span style="color:#ECEEE9;">{"Ready (5m)" if samples_count >= ROLLING_5M else f"{samples_count}/{ROLLING_5M}"}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8E9889;">System Platform</span>
                <span style="color:#ECEEE9;">{platform.system()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Telemetry Export")
    csv_data = pd.DataFrame(list(st.session_state.history)).to_csv(index=False)
    st.download_button(
        label="Download History (CSV)",
        data=csv_data,
        file_name="pulsecheck_telemetry_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown(
        """
        <div style="margin-top: 1.5rem; padding: 0.75rem; border: 1px solid #2B2F27; border-radius: 4px; background: #1C1F1B;">
            <div style="font-family: ui-monospace, Consolas, monospace; font-size: 0.65rem; color: #8E9889; text-transform: uppercase; margin-bottom: 4px;">Scope & Intent</div>
            <p style="font-size: 0.72rem; color: #6B7467; margin: 0; line-height: 1.35;">
                Unsupervised anomaly detection flags deviations in CPU/RAM time series against training distributions. It does not replace hardware diagnostic utilities or identify malware.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Main View: Initial Sample Check
# --------------------------------------------------

if not history:
    st.info("Acquiring initial system telemetry sample...")
    st.stop()

latest = history[-1]
cpu = float(latest["cpu_usage"])
memory = float(latest["memory_usage"])
disk = float(latest.get("disk_usage", 0.0))
score = latest.get("anomaly_score", np.nan)
# Use None as the explicit "unknown" sentinel — do not default to False
# (False would silently render NORMAL when prediction has not been made).
is_anomaly_val = latest.get("is_anomaly", None)
is_warming_up = len(history) < ROLLING_5M



# --------------------------------------------------
# A. Restrained Header
# --------------------------------------------------

# Determine live indicator label and class
if not st.session_state.monitoring:
    status_class = "status-paused"
    dot_class = "dot-paused"
    status_label = "MONITORING PAUSED"
elif is_warming_up:
    status_class = "status-warmup"
    dot_class = "dot-warmup"
    status_label = f"INITIALIZING ({len(history)}/{ROLLING_5M})"
else:
    status_class = "status-live"
    dot_class = "dot-live"
    status_label = "LIVE MONITORING"

timestamp_formatted = latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

st.markdown(
    f"""
    <div class="monitor-header">
        <div class="brand-group">
            <span class="brand-icon">⚡</span>
            <div>
                <span class="brand-title">PulseCheck AI</span>
                <span class="brand-badge" style="margin-left: 8px;">LOCAL SYSTEM MONITOR</span>
            </div>
        </div>
        <div class="header-meta">
            <span class="timestamp-text">Last Sample: {timestamp_formatted}</span>
            <div class="status-pill {status_class}">
                <span class="status-dot {dot_class}"></span>
                <span>{status_label}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# B. Resource Metric Cards (4 Aligned Metrics)
# --------------------------------------------------

# Status badges logic
if cpu >= 80:
    cpu_badge = '<span class="badge-critical">HIGH</span>'
elif cpu >= 50:
    cpu_badge = '<span class="badge-elevated">ELEVATED</span>'
else:
    cpu_badge = '<span class="badge-nominal">NOMINAL</span>'

if memory >= 90:
    mem_badge = '<span class="badge-critical">CRITICAL</span>'
elif memory >= 80:
    mem_badge = '<span class="badge-elevated">HIGH</span>'
else:
    mem_badge = '<span class="badge-nominal">NOMINAL</span>'

if disk >= 90:
    disk_badge = '<span class="badge-elevated">CONSTRAINED</span>'
else:
    disk_badge = '<span class="badge-nominal">NOMINAL</span>'

mem_used_badge = '<span class="badge-nominal">ACTIVE</span>'

col_cpu, col_mem, col_disk, col_used = st.columns(4)

with col_cpu:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">CPU Utilization</span>
                {cpu_badge}
            </div>
            <div class="metric-value">{cpu:.1f}%</div>
            <div class="metric-subtext">Sample window: 1.0s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mem:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">Memory Utilization</span>
                {mem_badge}
            </div>
            <div class="metric-value">{memory:.1f}%</div>
            <div class="metric-subtext">{latest['memory_used_gb']:.2f} GB physical RAM in use</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_disk:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">Disk Utilization</span>
                {disk_badge}
            </div>
            <div class="metric-value">{disk:.1f}%</div>
            <div class="metric-subtext">Root storage volume</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_used:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-label">Memory Used</span>
                {mem_used_badge}
            </div>
            <div class="metric-value">{latest['memory_used_gb']:.2f} GB</div>
            <div class="metric-subtext">{memory:.1f}% of total system capacity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# C. Anomaly Detection Panel (Primary Focus)
# --------------------------------------------------

st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

if is_warming_up:
    banner_state = "state-warmup"
    status_tag_color = "#D6A45D"
    status_tag_text = "INITIALIZING WARM-UP"
    remaining_samples = ROLLING_5M - len(history)
    approx_sec = remaining_samples * SAMPLE_INTERVAL
    explanation_text = (
        f"Collecting baseline telemetry to calculate rolling-window statistics. "
        f"Sample {len(history)} of {ROLLING_5M} acquired ({remaining_samples} remaining, ~{approx_sec}s)."
    )
    score_display = "Pending (Buffering)"
elif pd.isna(score):
    banner_state = "state-unavailable"
    status_tag_color = "#D46E65"
    status_tag_text = "PREDICTION UNAVAILABLE"
    explanation_text = (
        "The latest prediction could not be calculated. Telemetry features may be incomplete or an inference error occurred."
    )
    score_display = "Unavailable"
elif is_anomaly_val is True:
    banner_state = "state-anomaly"
    status_tag_color = "#D6A45D"
    status_tag_text = "ANOMALY DETECTED"
    explanation_text = (
        "The latest CPU or memory pattern differs from the model's learned baseline. Review resource telemetry below."
    )
    score_display = f"{score:.4f}"
else:
    banner_state = "state-normal"
    status_tag_color = "#A3B18A"
    status_tag_text = "NORMAL"
    explanation_text = (
        "No unusual CPU or memory pattern detected in the latest sample. Resource metrics remain within expected distributions."
    )
    score_display = f"{score:.4f}"

st.markdown(
    f"""
    <div class="panel-card">
        <div class="panel-title">Model Inference & Anomaly Detection</div>
        <div class="anomaly-banner {banner_state}">
            <div class="banner-left">
                <span class="banner-status-tag" style="color: {status_tag_color};">
                    <span>●</span> {status_tag_text}
                </span>
                <p class="banner-desc">{explanation_text}</p>
            </div>
        </div>
        <div class="telemetry-grid">
            <div class="telemetry-item">
                <span class="telemetry-item-label">Latest Anomaly Score</span>
                <span class="telemetry-item-value">{score_display}</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-item-label">Decision Threshold</span>
                <span class="telemetry-item-value">{THRESHOLD:.4f}</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-item-label">Model Architecture</span>
                <span class="telemetry-item-value">IsolationForest (200 trees)</span>
            </div>
            <div class="telemetry-item">
                <span class="telemetry-item-label">Feature Vector</span>
                <span class="telemetry-item-value">10 Rolling Features (CPU/RAM)</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Resource Investigation Diagnostics
# --------------------------------------------------

# CPU diagnostic note
if cpu >= 80:
    cpu_diag_tag = '<span class="badge-critical">HIGH UTILIZATION</span>'
    cpu_diag_text = "Substantial CPU load observed. Inspect Task Manager or active threads for heavy processes."
elif cpu >= 50:
    cpu_diag_tag = '<span class="badge-elevated">ELEVATED</span>'
    cpu_diag_text = "Moderate CPU load at this sample. Observe if utilization remains elevated over consecutive windows."
else:
    cpu_diag_tag = '<span class="badge-nominal">NOMINAL</span>'
    cpu_diag_text = "CPU load is within normal operating limits for standard workstation tasks."

# Memory diagnostic note
if memory >= 90:
    mem_diag_tag = '<span class="badge-critical">VERY HIGH UTILIZATION</span>'
    mem_diag_text = "Severe RAM saturation. Inspect top memory processes below and terminate non-essential tasks."
elif memory >= 80:
    mem_diag_tag = '<span class="badge-elevated">ELEVATED</span>'
    mem_diag_text = "Elevated RAM utilization. Check for background applications or browser tabs consuming large allocations."
else:
    mem_diag_tag = '<span class="badge-nominal">NOMINAL</span>'
    mem_diag_text = "RAM allocation is balanced. Ample physical memory remains available for working sets."

# Disk diagnostic note
if disk >= 90:
    disk_diag_tag = '<span class="badge-elevated">LIMITED SPACE</span>'
    disk_diag_text = "Storage utilization exceeds 90%. Review root partition capacity and clean temporary data if necessary."
else:
    disk_diag_tag = '<span class="badge-nominal">ADEQUATE</span>'
    disk_diag_text = "Sufficient storage capacity available on the primary system volume."

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown(
        f"""
        <div class="diag-card">
            <div class="diag-title-row">
                <span class="diag-resource">CPU Investigation</span>
                {cpu_diag_tag}
            </div>
            <p class="diag-body">{cpu_diag_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_d2:
    st.markdown(
        f"""
        <div class="diag-card">
            <div class="diag-title-row">
                <span class="diag-resource">Memory Investigation</span>
                {mem_diag_tag}
            </div>
            <p class="diag-body">{mem_diag_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_d3:
    st.markdown(
        f"""
        <div class="diag-card">
            <div class="diag-title-row">
                <span class="diag-resource">Storage (Independent)</span>
                {disk_diag_tag}
            </div>
            <p class="diag-body">{disk_diag_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="font-size: 0.72rem; color: #6B7467; margin-top: 0.5rem; margin-bottom: 1.25rem;">
        * Note: Resource investigation alerts provide heuristic guidance for triage; disk utilization is evaluated independently of the Isolation Forest model.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# D. CPU and Memory History (Plotly Observability Chart)
# --------------------------------------------------

st.markdown(
    """
    <div class="section-header">
        <span class="section-title">Resource Usage History</span>
        <span class="section-subtitle">Time-series tracking of CPU and Memory telemetry</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_df = pd.DataFrame(history)

fig = go.Figure()

# CPU usage trace
fig.add_trace(
    go.Scatter(
        x=chart_df["timestamp"],
        y=chart_df["cpu_usage"],
        mode="lines",
        name="CPU Usage",
        line=dict(color="#7EA393", width=1.75),
        hovertemplate="CPU: %{y:.1f}%<extra></extra>",
    )
)

# Memory usage trace
fig.add_trace(
    go.Scatter(
        x=chart_df["timestamp"],
        y=chart_df["memory_usage"],
        mode="lines",
        name="Memory Usage",
        line=dict(color="#D6A45D", width=1.75),
        hovertemplate="RAM: %{y:.1f}%<extra></extra>",
    )
)

# Overlay actual anomaly markers if any anomalies exist in history
anomaly_subset = chart_df[chart_df.get("is_anomaly") == True]
if not anomaly_subset.empty:
    fig.add_trace(
        go.Scatter(
            x=anomaly_subset["timestamp"],
            y=anomaly_subset["cpu_usage"],
            mode="markers",
            name="Anomaly Flagged",
            marker=dict(
                color="#D6A45D",
                size=8,
                symbol="diamond",
                line=dict(color="#171916", width=1.5),
            ),
            hovertemplate="Anomaly Flagged<br>CPU: %{y:.1f}%<br>Score: %{customdata:.4f}<extra></extra>",
            customdata=anomaly_subset["anomaly_score"],
        )
    )

fig.update_layout(
    height=340,
    paper_bgcolor="#20231F",
    plot_bgcolor="#20231F",
    font=dict(
        family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        size=11,
        color="#8E9889",
    ),
    xaxis=dict(
        title="",
        showgrid=True,
        gridcolor="#282C24",
        linecolor="#343831",
        tickfont=dict(
            family='ui-monospace, Consolas, monospace',
            size=10,
            color="#8E9889",
        ),
    ),
    yaxis=dict(
        title="Utilization (%)",
        range=[0, 100],
        showgrid=True,
        gridcolor="#282C24",
        linecolor="#343831",
        tickfont=dict(
            family='ui-monospace, Consolas, monospace',
            size=10,
            color="#8E9889",
        ),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#171916",
        bordercolor="#343831",
        font=dict(
            family='ui-monospace, Consolas, monospace',
            size=11,
            color="#ECEEE9",
        ),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color="#8E9889"),
    ),
    margin=dict(l=40, r=20, t=25, b=25),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------
# E. Process Investigation & F. Recent Telemetry
# --------------------------------------------------

col_proc, col_log = st.columns([1, 1])

with col_proc:
    st.markdown(
        """
        <div class="section-header">
            <span class="section-title">Top Processes by Memory</span>
            <span class="section-subtitle">Real-time working set snapshot</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    process_df = get_top_processes(limit=8)
    st.dataframe(
        process_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PID": st.column_config.NumberColumn("PID", format="%d"),
            "Process": st.column_config.TextColumn("Process Name"),
            "RAM (%)": st.column_config.ProgressColumn(
                "RAM (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )
    st.caption(
        "Process snapshot is collected for manual triage and is not an input to the Isolation Forest model."
    )

with col_log:
    st.markdown(
        """
        <div class="section-header">
            <span class="section-title">Recent Telemetry Log</span>
            <span class="section-subtitle">Last 20 sample readings</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            "timestamp", ascending=False
        )

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp", format="HH:mm:ss"
                ),
                "cpu_usage": st.column_config.NumberColumn(
                    "CPU (%)", format="%.1f%%"
                ),
                "memory_usage": st.column_config.NumberColumn(
                    "RAM (%)", format="%.1f%%"
                ),
                "disk_usage": st.column_config.NumberColumn(
                    "Disk (%)", format="%.1f%%"
                ),
                "anomaly_score": st.column_config.NumberColumn(
                    "Score", format="%.4f"
                ),
                "is_anomaly": st.column_config.CheckboxColumn(
                    "Anomaly",
                ),
            },
        )


# --------------------------------------------------
# Automatic Periodic Refresh
# --------------------------------------------------

if st.session_state.monitoring:
    st_autorefresh(
        interval=SAMPLE_INTERVAL * 1000,
        key="pulsecheck_refresh",
    )