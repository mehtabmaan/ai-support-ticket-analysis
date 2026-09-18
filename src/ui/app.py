"""
Streamlit Web Application for AI Customer Support Ticket Analysis.
Provides interactive NL querying, statistical & SLA anomaly dashboard,
and dataset analytics.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.anomaly.detector import AnomalyDetector
from src.config import (
    get_reference_source,
    get_reference_timestamp_str,
    settings,
)
from src.data_layer.loader import (
    get_cached_dataframe,
    initialize_database,
)
from src.llm.client import LLMClient
from src.llm.query_engine import NLQueryEngine

# Page configuration
st.set_page_config(
    page_title="AI Support Ticket Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .ref-banner {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        color: #1E40AF;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .answer-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #22C55E;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        line-height: 1.5;
        color: #14532D;
    }
    .caveat-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.8rem;
        color: #92400E;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services():
    """Initializes and caches core backend services for the UI session."""
    initialize_database()
    llm = LLMClient()
    engine = NLQueryEngine(llm_client=llm)
    detector = AnomalyDetector()
    return engine, detector, llm


engine, detector, llm = get_services()
df = get_cached_dataframe()
ref_now_str = get_reference_timestamp_str()
ref_source = get_reference_source()
is_llm_active, llm_status_msg = llm.check_availability()

# Header
st.markdown('<div class="main-header">🎯 Support Ticket Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-grade AI querying, statistical anomaly detection, and operational SLA monitoring</div>',
    unsafe_allow_html=True,
)

# Prominent Reference Time Banner
st.markdown(
    f"""
    <div class="ref-banner">
        🕒 <b>Dataset Snapshot Reference Date:</b> <code>{ref_now_str}</code><br/>
        <span style="font-size: 0.85rem; color: #3B82F6;">{ref_source}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    st.write(f"**Records Loaded:** {len(df):,}")
    st.write(f"**LLM Provider:** `{settings.LLM_PROVIDER}`")
    if is_llm_active:
        st.success(f"🟢 {llm_status_msg}")
    else:
        st.info(f"🟡 {llm_status_msg}")

    st.markdown("---")
    st.subheader("📚 Quick Reference")
    st.markdown(
        """
        - **Resolved**: `resolution_time_hrs IS NOT NULL` (327 tickets)
        - **Unresolved**: `resolution_time_hrs IS NULL` (173 tickets)
        - **SLA Threshold**: 24.0 hours for High/Critical tickets
        - **Resolution Outliers**: Category-specific IQR (Q3 + 1.5*IQR)
        """
    )


# Main Tabs
tab_query, tab_anomalies, tab_analytics = st.tabs([
    "💬 Natural Language Query",
    "🚨 Anomaly Detection",
    "📊 Dataset Analytics & Health",
])


# -----------------------------------------------------------------------------
# TAB 1: NATURAL LANGUAGE QUERY
# -----------------------------------------------------------------------------
with tab_query:
    st.subheader("Ask Questions in Natural Language")
    st.caption("Powered by LLM Text-to-SQL with automatic fallback and security verification.")

    # Sample query chips
    sample_queries = [
        "How many tickets are currently open?",
        "Which agent resolved the most tickets this month?",
        "Which agent resolved the most tickets overall?",
        "Which agent has the lowest average customer rating?",
        "Show me all Critical tickets not resolved within 12 hours.",
        "What is the average customer rating for Technical category tickets?",
        "How many critical tickets are unresolved?",
        "Are there any anomalies in resolution times this week?",
    ]

    selected_sample = st.selectbox("Choose a sample query from the assessment brief:", ["-- Select a sample query --"] + sample_queries)

    default_query = selected_sample if selected_sample != "-- Select a sample query --" else ""
    user_query = st.text_input("Or enter your custom question:", value=default_query, placeholder="e.g. Which agent resolved the most tickets in March 2024?")

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True) or default_query:
        query_to_run = user_query.strip() if user_query.strip() else default_query
        if query_to_run:
            with st.spinner("Analyzing dataset..."):
                response = engine.execute_nl_query(query_to_run)

            # Executive Answer
            st.markdown(
                f"""
                <div class="answer-box">
                    <b>💡 Executive Summary:</b><br/>
                    {response['answer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Caveats / Ties
            if response.get("caveats"):
                st.markdown(
                    f"""
                    <div class="caveat-box">
                        ⚠️ {response['caveats']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Rows Returned", response["row_count"])
            with col_m2:
                st.metric("Execution Time", f"{response['execution_time_ms']} ms")
            with col_m3:
                st.metric("Engine Mode", "Fallback Rule Engine" if response["is_fallback"] else "Groq LLM")
            with col_m4:
                st.metric("Status", "Success" if not response.get("error") else "Error")

            # SQL Inspector
            if response.get("sql"):
                with st.expander("🔍 View Generated SQL & Query Plan", expanded=False):
                    st.code(response["sql"], language="sql")
                    if response.get("explanation"):
                        st.caption(f"**Query Rationale:** {response['explanation']}")

            # Tabular results
            if response["results"]:
                st.write("##### Query Results Table")
                res_df = pd.DataFrame(response["results"])
                st.dataframe(res_df, use_container_width=True)
        else:
            st.warning("Please enter a question or select a sample query.")


# -----------------------------------------------------------------------------
# TAB 2: ANOMALY DETECTION
# -----------------------------------------------------------------------------
with tab_anomalies:
    st.subheader("Customer Support Anomaly & SLA Monitoring")
    st.caption("Combines per-category statistical IQR outlier detection with rule-based SLA breach monitoring.")

    summary = detector.get_anomaly_summary()

    # KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Anomalies", summary["total_anomalies"], help="Combined resolution outliers and SLA breaches")
    with kpi2:
        st.metric("Resolution Outliers", summary["resolution_time_outliers"], help="Tickets exceeding per-category Q3 + 1.5*IQR")
    with kpi3:
        st.metric("SLA Breaches", summary["sla_breaches"], help="Unresolved High/Critical tickets > 24h old")
    with kpi4:
        st.metric("Critical Severity", summary["by_severity"].get("CRITICAL", 0), help="Immediate operational escalation needed")

    # Filters
    st.markdown("---")
    st.write("##### Filter Anomalies")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        cat_filter = st.selectbox("Category", ["All", "Billing", "General", "Technical"])
    with f_col2:
        sev_filter = st.selectbox("Severity", ["All", "CRITICAL", "HIGH", "MEDIUM"])
    with f_col3:
        type_filter = st.selectbox("Anomaly Type", ["All", "RESOLUTION_TIME_OUTLIER", "SLA_BREACH"])

    filtered_anomalies = detector.detect_all_anomalies(
        category=None if cat_filter == "All" else cat_filter,
        severity=None if sev_filter == "All" else sev_filter,
        anomaly_type=None if type_filter == "All" else type_filter,
    )

    st.write(f"Showing **{len(filtered_anomalies)}** matching anomalies:")

    # Detailed table
    if filtered_anomalies:
        table_data = []
        for a in filtered_anomalies:
            table_data.append({
                "Ticket ID": a.ticket_id,
                "Category": a.category,
                "Priority": a.priority,
                "Status": a.status,
                "Type": a.anomaly_type,
                "Severity": a.severity,
                "Value": f"{a.metric_value} hrs",
                "Threshold": f"{a.threshold_value} hrs",
                "Agent": a.agent_id,
                "Explanation": a.explanation,
            })
        anomaly_df = pd.DataFrame(table_data)
        st.dataframe(anomaly_df, use_container_width=True)

        # Inspection cards
        with st.expander("🔍 Inspect Individual Anomaly Details & Remediation Notes"):
            for a in filtered_anomalies[:10]:
                st.markdown(
                    f"""
                    **Ticket {a.ticket_id}** (`{a.category}` | `{a.priority}` | `{a.severity}` Severity)
                    - **Type:** `{a.anomaly_type}`
                    - **Metric:** {a.metric_name} = **{a.metric_value} hrs** (Threshold: {a.threshold_value} hrs)
                    - **Agent:** {a.agent_id} | **Created:** {a.created_at}
                    - **Issue:** {a.issue_summary}
                    - **Explanation:** {a.explanation}
                    ---
                    """
                )
    else:
        st.info("No anomalies match the selected filters.")


# -----------------------------------------------------------------------------
# TAB 3: DATASET ANALYTICS & SYSTEM HEALTH
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Dataset Analytics & Health Overview")

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        st.metric("Total Tickets", len(df))
    with col_a2:
        resolved_count = int(df["resolution_time_hrs"].notna().sum())
        st.metric("Resolved Tickets", f"{resolved_count} ({resolved_count/len(df)*100:.1f}%)")
    with col_a3:
        open_count = int((df["status"] == "Open").sum())
        st.metric("Open Tickets", f"{open_count} ({open_count/len(df)*100:.1f}%)")
    with col_a4:
        esc_count = int((df["status"] == "Escalated").sum())
        st.metric("Escalated Tickets", f"{esc_count} ({esc_count/len(df)*100:.1f}%)")

    st.markdown("---")

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.metric("Avg Response Time", f"{df['response_time_hrs'].mean():.2f} hrs")
    with col_b2:
        res_mean = df['resolution_time_hrs'].dropna().mean()
        st.metric("Avg Resolution Time", f"{res_mean:.2f} hrs")
    with col_b3:
        rat_mean = df['customer_rating'].dropna().mean()
        st.metric("Avg Customer Rating", f"{rat_mean:.2f} / 5.0")

    st.markdown("---")
    st.write("##### Per-Category Resolution Time Thresholds (IQR Upper Fence: Q3 + 1.5*IQR)")
    thresholds = detector.get_category_resolution_stats()
    thresh_rows = []
    for cat, t in thresholds.items():
        thresh_rows.append({
            "Category": cat,
            "Resolved Count": t["count"],
            "Q1 (hrs)": t["q1"],
            "Median (hrs)": t["median"],
            "Q3 (hrs)": t["q3"],
            "IQR (hrs)": t["iqr"],
            "Upper Fence (hrs)": t["upper_fence"],
            "Mean (hrs)": t["mean"],
            "Std Dev (hrs)": t["std"],
        })
    st.table(pd.DataFrame(thresh_rows))

    # Breakdowns
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.write("##### Priority Distribution")
        st.bar_chart(df["priority"].value_counts())
    with c_chart2:
        st.write("##### Category Distribution")
        st.bar_chart(df["category"].value_counts())
