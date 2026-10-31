"""
BITS Pilani WILP - M.Tech (AIML) Machine Learning Assignment 2
Interactive Telco Customer Churn Classification & Diagnostic Workbench

"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG 
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TelcoPulse AI | Churn Classification Suite",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 1. Minimize Sidebar Width  */
    section[data-testid="stSidebar"] {
        width: 235px !important;
        min-width: 235px !important;
        max-width: 235px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 235px !important;
        padding-top: 1rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.05rem !important;
        margin-bottom: 0.1rem !important;
    }
    section[data-testid="stSidebar"] h4 {
        font-size: 0.82rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.25rem !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown {
        font-size: 0.78rem !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.6rem 0 !important;
    }

    /* 2. Maximize Right Pane Width  */
    .main .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    html, body, [class*="css"] {
        font-size: 13px !important;
    }
    h1 { font-size: 1.35rem !important; }
    h2, h3 { font-size: 1.08rem !important; margin-top: 0.3rem !important; margin-bottom: 0.4rem !important; }
    h4 { font-size: 0.92rem !important; margin-top: 0.2rem !important; margin-bottom: 0.3rem !important; }

    /*  Tabs */
    button[data-baseweb="tab"] {
        padding-top: 0.4rem !important;
        padding-bottom: 0.4rem !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /*Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
        padding: 0.85rem 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #0284c7;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        color: #0f172a;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 6px rgba(15, 23, 42, 0.04);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.25rem !important;
        font-weight: 700;
        color: #0f172a;
    }
    .hero-banner p {
        margin: 0.25rem 0 0 0;
        color: #334155;
        font-size: 0.8rem !important;
    }

    /*  6-Column KPI Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #0284c7;
        border-radius: 8px;
        padding: 0.55rem 0.4rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }
    .kpi-label {
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569;
        font-weight: 700;
        margin-bottom: 0.15rem;
        white-space: nowrap;
    }
    .kpi-val {
        font-size: 1.22rem !important;
        font-weight: 800;
        color: #0369a1;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.64rem !important;
        color: #64748b;
        margin-top: 0.1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .frozen-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem !important;
        font-weight: 700;
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
    }

    /* Compact Streamlit Metric Widgets in Tab 4 */
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "model"
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn_classifier.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm_classifier.joblib"
}


# -----------------------------------------------------------------------------
# 2. CACHED MODEL & ARTIFACT LOADER 
# -----------------------------------------------------------------------------
@st.cache_resource
def load_saved_pipeline():
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    if not os.path.exists(meta_path) or not os.path.exists(scaler_path):
        return None, None, None

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    scaler = joblib.load(scaler_path)
    models = {}
    for m_name, fname in MODEL_REGISTRY.items():
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.exists(fpath):
            models[m_name] = joblib.load(fpath)

    return metadata, scaler, models


def preprocess_uploaded_dataframe(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Accepts either pre-encoded test_data.csv OR raw Telco Customer Churn CSV
    and ensures clean numerical encoding without modifying model weights.
    """
    clean_df = df.copy()
    if "customerID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["customerID"])

    if "TotalCharges" in clean_df.columns and clean_df["TotalCharges"].dtype == object:
        clean_df["TotalCharges"] = pd.to_numeric(clean_df["TotalCharges"], errors="coerce").fillna(0.0)

    encoding_maps = metadata.get("encoding_maps", {})
    for col, mapping in encoding_maps.items():
        if col in clean_df.columns and clean_df[col].dtype == object:
            clean_df[col] = clean_df[col].map(mapping).fillna(0).astype(int)

    target_col = metadata["target_column"]
    if target_col in clean_df.columns and clean_df[target_col].dtype == object:
        clean_df[target_col] = clean_df[target_col].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    return clean_df


def run_frozen_inference(model, X_scaled: np.ndarray, y_true: np.ndarray, threshold: float = 0.50):
    """Computes predictions and all 6 required evaluation metrics from frozen model weights."""
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    return y_pred, y_prob, metrics


def render_high_contrast_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """
    Builds a well-proportioned Confusion Matrix with automargins and dynamic
    dark/light text contrast so nothing clips or disorients at 100% zoom.
    """
    x_labels = ["Pred: Retained (0)", "Pred: Churned (1)"]
    y_labels = ["Actual: Retained (0)", "Actual: Churned (1)"]
    cell_titles = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"]
    ]

    colorscale = [
        [0.0, "#eff6ff"],
        [0.35, "#bfdbfe"],
        [0.65, "#3b82f6"],
        [1.0, "#1e3a8a"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        showscale=False,
        xgap=3,
        ygap=3,
        hoverinfo="skip"
    ))

    max_val = max(float(cm.max()), 1.0)
    annotations = []
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            pct = (val / cm.sum()) * 100.0
            font_color = "#ffffff" if (val / max_val) > 0.50 else "#0f172a"
            annotations.append(dict(
                x=x_labels[j],
                y=y_labels[i],
                text=f"<b>{cell_titles[i][j]}</b><br><span style='font-size:16px'><b>{val}</b></span> ({pct:.1f}%)",
                showarrow=False,
                font=dict(color=font_color, size=11, family="Arial, sans-serif"),
                align="center"
            ))

    fig.update_layout(
        annotations=annotations,
        height=310,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            tickfont=dict(size=11, color="#0f172a", family="Arial, sans-serif"),
            side="bottom",
            automargin=True
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#0f172a", family="Arial, sans-serif"),
            autorange="reversed",
            automargin=True
        )
    )
    return fig


def highlight_best_metric(s: pd.Series):
    """Highlights the maximum value in each metric column with soft mint & dark bold text."""
    is_max = s == s.max()
    return [
        "background-color: #d1fae5; color: #065f46; font-weight: 800;" if v else "color: #0f172a;"
        for v in is_max
    ]


# -----------------------------------------------------------------------------
# 3. INITIALIZE ARTIFACTS 
# -----------------------------------------------------------------------------
metadata, scaler, models = load_saved_pipeline()

if metadata is None or not models:
    st.error("⚠️ Model artifacts missing in `model/`. Please run `python train_models.py` first.")
    st.stop()

feature_cols = metadata["feature_columns"]
target_col = metadata["target_column"]
encoding_maps = metadata["encoding_maps"]

with st.sidebar:
    st.markdown("## 📡 Command Center")
    st.caption("BITS Pilani • ML Assignment 2")
    st.caption("BITS ID: 2026aj05808")
    st.divider()

    # Step 6(a): Dataset upload option (CSV)
    st.markdown("#### 1. Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"],
        help="Upload test_data.csv (19 features + Churn target)."
    )

    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button(
                label="⬇️ Sample test_data.csv",
                data=f,
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.divider()

    # Step 6(b): Model selection dropdown
    st.markdown("#### 2. Select Classifier")
    selected_model_name = st.selectbox(
        "Pre-Trained Model",
        options=list(MODEL_REGISTRY.keys()),
        index=0
    )

    st.markdown("#### 3. Decision Threshold")
    decision_threshold = st.slider(
        "Probability Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.divider()
    st.markdown("<div class='frozen-pill'>🔒 Frozen Weights (No Retrain)</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. LOAD & VALIDATE TEST CSV
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    raw_input_df = pd.read_csv(uploaded_file)
    source_tag = f"Uploaded: `{uploaded_file.name}` ({len(raw_input_df):,} rows)"
elif os.path.exists("test_data.csv"):
    raw_input_df = pd.read_csv("test_data.csv")
    source_tag = f"Holdout: `test_data.csv` ({len(raw_input_df):,} rows)"
else:
    st.warning("Please upload `test_data.csv` in the sidebar to begin evaluation.")
    st.stop()

eval_df = preprocess_uploaded_dataframe(raw_input_df, metadata)

missing_cols = [c for c in feature_cols if c not in eval_df.columns]
if missing_cols:
    st.error(f"❌ Uploaded CSV is missing required feature columns: `{missing_cols}`")
    st.stop()

if target_col not in eval_df.columns:
    st.error(f"❌ Uploaded CSV must include the target column `{target_col}` to compute evaluation metrics.")
    st.stop()

X_test_eval = eval_df[feature_cols]
y_test_eval = eval_df[target_col].astype(int).values
X_test_scaled = scaler.transform(X_test_eval)

# Evaluate currently selected model
active_clf = models[selected_model_name]
y_pred_active, y_prob_active, active_metrics = run_frozen_inference(
    active_clf, X_test_scaled, y_test_eval, threshold=decision_threshold
)

# -----------------------------------------------------------------------------
# 5.  HERO HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <h1>📡 TelcoPulse AI — Customer Churn Classification & Diagnostic Suite</h1>
    <p>Classifier: <strong>{selected_model_name}</strong> &nbsp;|&nbsp; {source_tag} &nbsp;|&nbsp; Features: <strong>{len(feature_cols)}</strong> &nbsp;|&nbsp; Threshold: <strong>{decision_threshold:.2f}</strong></p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. INTERACTIVE TABS 
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Evaluation & Confusion Matrix",
    "🏆 All 5 Models Comparison & ROC",
    "🔎 Test Data & Error Inspector",
    "🧪 Retention What-If Simulator"
])

# === TAB 1: MANDATORY METRICS, CONFUSION MATRIX & CLASSIFICATION REPORT ===
with tab1:
    kpi_cols = st.columns(6)
    kpi_descriptions = {
        "Accuracy": "Overall Hit Rate",
        "AUC": "ROC Area",
        "Precision": "Pos. Pred. Value",
        "Recall": "Sensitivity (TPR)",
        "F1": "Harmonic Mean",
        "MCC": "Matthews Corr."
    }

    for idx, (m_name, m_val) in enumerate(active_metrics.items()):
        with kpi_cols[idx]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{m_name}</div>
                <div class="kpi-val">{m_val:.4f}</div>
                <div class="kpi-sub">{kpi_descriptions[m_name]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1.0, 1.15], gap="medium")

    with left_col:
        st.markdown("#### 🔲 Confusion Matrix (Holdout Test Set)")
        cm = confusion_matrix(y_test_eval, y_pred_active)
        fig_cm = render_high_contrast_confusion_matrix(cm)
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    with right_col:
        st.markdown("#### 📋 Classification Report")
        cls_report = classification_report(
            y_test_eval,
            y_pred_active,
            target_names=["Retained (0)", "Churned (1)"],
            output_dict=True,
            zero_division=0
        )
        cls_df = pd.DataFrame(cls_report).transpose().round(4)
        st.dataframe(cls_df, use_container_width=True, height=195)

    # Full-width probability distribution below so it never crushes the table
    st.markdown("#### 📊 Predicted Churn Probability Distribution vs. Decision Cutoff")
    prob_hist_df = pd.DataFrame({
        "Predicted Churn Probability": y_prob_active,
        "Actual Status": np.where(y_test_eval == 1, "Churned (1)", "Retained (0)")
    })
    fig_prob = px.histogram(
        prob_hist_df,
        x="Predicted Churn Probability",
        color="Actual Status",
        nbins=35,
        barmode="overlay",
        opacity=0.80,
        template="plotly_white",
        color_discrete_map={"Retained (0)": "#0284c7", "Churned (1)": "#e11d48"}
    )
    fig_prob.add_vline(
        x=decision_threshold,
        line_dash="dash",
        line_color="#0f172a",
        annotation_text=f"Cutoff ({decision_threshold:.2f})",
        annotation_font_color="#0f172a",
        annotation_font_size=11
    )
    fig_prob.update_layout(
        height=210,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True)
    )
    st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})


# === TAB 2: ALL 5 MODELS BENCHMARK COMPARISON TABLE & ROC CURVES ===
with tab2:
    st.markdown("#### 🏆 Side-by-Side Evaluation Across All 5 Pre-Trained Classifiers")

    benchmark_rows = []
    roc_curves = []

    for m_label, clf_obj in models.items():
        _, prob_vec, m_dict = run_frozen_inference(
            clf_obj, X_test_scaled, y_test_eval, threshold=decision_threshold
        )
        row = {"ML Model Name": m_label}
        row.update({k: round(v, 4) for k, v in m_dict.items()})
        benchmark_rows.append(row)

        fpr, tpr, _ = roc_curve(y_test_eval, prob_vec)
        roc_curves.append((m_label, fpr, tpr, m_dict["AUC"]))

    bench_df = pd.DataFrame(benchmark_rows)
    metric_cols_list = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    st.dataframe(
        bench_df.style.apply(highlight_best_metric, subset=metric_cols_list)
        .format({k: "{:.4f}" for k in metric_cols_list}),
        use_container_width=True,
        hide_index=True,
        height=215
    )

    c_roc, c_radar = st.columns(2, gap="medium")

    with c_roc:
        st.markdown("#### 📈 Overlaid ROC Curves")
        fig_roc = go.Figure()
        for m_label, fpr, tpr, auc_val in roc_curves:
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{m_label} ({auc_val:.3f})"
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#64748b"), name="Baseline"
        ))
        fig_roc.update_layout(
            template="plotly_white",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=330,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", size=11),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True)
        )
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

    with c_radar:
        st.markdown("#### 🕸️ Multi-Metric Radar Profile")
        fig_radar = go.Figure()
        for row in benchmark_rows:
            r_vals = [row[m] for m in metric_cols_list]
            r_vals.append(r_vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=metric_cols_list + [metric_cols_list[0]],
                fill="toself",
                name=row["ML Model Name"],
                opacity=0.65
            ))
        fig_radar.update_layout(
            template="plotly_white",
            polar=dict(
                radialaxis=dict(visible=True, range=[0.35, 0.90], tickfont=dict(color="#0f172a", size=9)),
                angularaxis=dict(tickfont=dict(color="#0f172a", size=11))
            ),
            height=330,
            margin=dict(l=25, r=25, t=20, b=50),
            paper_bgcolor="#ffffff",
            font=dict(color="#0f172a", size=11),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})


# === TAB 3: TEST DATA & PREDICTION INSPECTOR ===
with tab3:
    st.markdown("#### 🔎 Customer Holdout Predictions & Feature Drivers")

    inspector_df = eval_df.copy()
    inspector_df["Predicted_Churn"] = y_pred_active
    inspector_df["Churn_Probability"] = np.round(y_prob_active, 4)
    inspector_df["Outcome"] = np.where(
        inspector_df[target_col] == inspector_df["Predicted_Churn"], "✅ Correct", "❌ Misclassified"
    )

    filter_mode = st.radio(
        "Filter Test Records:",
        ["All Test Customers", "Misclassified Customers Only", "High Flight-Risk Customers (Prob >= 0.70)"],
        horizontal=True
    )

    if filter_mode == "Misclassified Customers Only":
        shown_df = inspector_df[inspector_df["Outcome"] == "❌ Misclassified"]
    elif filter_mode == "High Flight-Risk Customers (Prob >= 0.70)":
        shown_df = inspector_df[inspector_df["Churn_Probability"] >= 0.70]
    else:
        shown_df = inspector_df

    st.dataframe(shown_df, use_container_width=True, height=240)

    st.markdown("#### 📊 Feature Correlation with Customer Churn (Holdout Test Set)")
    corrs = eval_df[feature_cols].apply(lambda x: x.corr(eval_df[target_col])).sort_values()
    corr_df = pd.DataFrame({"Feature": corrs.index, "Correlation with Churn": corrs.values})
    fig_bar = px.bar(
        corr_df,
        x="Correlation with Churn",
        y="Feature",
        orientation="h",
        color="Correlation with Churn",
        template="plotly_white",
        color_continuous_scale="RdBu_r"
    )
    fig_bar.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=11),
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True)
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# === TAB 4: LIVE CUSTOMER RETENTION WHAT-IF SIMULATOR ===
with tab4:
    st.markdown("#### 🧪 Interactive Single-Customer Retention Simulator")
    st.caption("Adjust subscriber parameters below to inspect live churn predictions across all 5 frozen models.")

    c1, c2, c3 = st.columns(3, gap="medium")
    sim_profile = {}

    with c1:
        st.markdown("**Demographics & Tenure**")
        sim_profile["gender"] = encoding_maps["gender"][st.selectbox("Gender", list(encoding_maps["gender"].keys()))]
        sim_profile["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        sim_profile["Partner"] = encoding_maps["Partner"][st.selectbox("Partner", list(encoding_maps["Partner"].keys()), index=1)]
        sim_profile["Dependents"] = encoding_maps["Dependents"][st.selectbox("Dependents", list(encoding_maps["Dependents"].keys()))]
        sim_profile["tenure"] = st.slider("Tenure (Months)", min_value=0, max_value=72, value=6)
        sim_profile["Contract"] = encoding_maps["Contract"][st.selectbox("Contract Type", list(encoding_maps["Contract"].keys()), index=0)]

    with c2:
        st.markdown("**Connectivity & Security**")
        sim_profile["PhoneService"] = encoding_maps["PhoneService"][st.selectbox("Phone Service", list(encoding_maps["PhoneService"].keys()), index=1)]
        sim_profile["MultipleLines"] = encoding_maps["MultipleLines"][st.selectbox("Multiple Lines", list(encoding_maps["MultipleLines"].keys()), index=1)]
        sim_profile["InternetService"] = encoding_maps["InternetService"][st.selectbox("Internet Service", list(encoding_maps["InternetService"].keys()), index=2)]
        sim_profile["OnlineSecurity"] = encoding_maps["OnlineSecurity"][st.selectbox("Online Security", list(encoding_maps["OnlineSecurity"].keys()), index=1)]
        sim_profile["OnlineBackup"] = encoding_maps["OnlineBackup"][st.selectbox("Online Backup", list(encoding_maps["OnlineBackup"].keys()), index=1)]
        sim_profile["DeviceProtection"] = encoding_maps["DeviceProtection"][st.selectbox("Device Protection", list(encoding_maps["DeviceProtection"].keys()), index=1)]

    with c3:
        st.markdown("**Support, Streaming & Billing**")
        sim_profile["TechSupport"] = encoding_maps["TechSupport"][st.selectbox("Tech Support", list(encoding_maps["TechSupport"].keys()), index=1)]
        sim_profile["StreamingTV"] = encoding_maps["StreamingTV"][st.selectbox("Streaming TV", list(encoding_maps["StreamingTV"].keys()), index=2)]
        sim_profile["StreamingMovies"] = encoding_maps["StreamingMovies"][st.selectbox("Streaming Movies", list(encoding_maps["StreamingMovies"].keys()), index=2)]
        sim_profile["PaperlessBilling"] = encoding_maps["PaperlessBilling"][st.selectbox("Paperless Billing", list(encoding_maps["PaperlessBilling"].keys()), index=1)]
        sim_profile["PaymentMethod"] = encoding_maps["PaymentMethod"][st.selectbox("Payment Method", list(encoding_maps["PaymentMethod"].keys()), index=0)]
        sim_profile["MonthlyCharges"] = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.5, step=0.5)
        sim_profile["TotalCharges"] = round(sim_profile["tenure"] * sim_profile["MonthlyCharges"], 2)

    single_df = pd.DataFrame([sim_profile])[feature_cols]
    single_scaled = scaler.transform(single_df)

    st.divider()
    st.markdown("#### 🗳️ Live 5-Model Churn Consensus")
    v_cols = st.columns(5)
    for idx, (m_label, clf_obj) in enumerate(models.items()):
        prob = float(clf_obj.predict_proba(single_scaled)[0, 1])
        verdict = "🔴 CHURN (1)" if prob >= decision_threshold else "🟢 RETAIN (0)"
        with v_cols[idx]:
            st.metric(label=m_label, value=verdict, delta=f"Risk: {prob:.1%}")
