import os
import hashlib
from io import StringIO
from typing import Optional, List, Dict
import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st
from rapidfuzz import process, fuzz

# =========================================================
# Page & Styles
# =========================================================
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption("Supabase → schema_v2.task_norms_view")

st.markdown(
    """
    <style>
      .stMetric span { font-size: 14px !important; }
      .stDataFrame { font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Database connection (via DSN in secrets)
# =========================================================
if "DATABASE_URL" not in st.secrets:
    st.error(
        "Missing secret `DATABASE_URL`.\n\n"
        "In Streamlit → Settings → Secrets add:\n"
        'DATABASE_URL = "postgresql://postgres.<project>:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"'
    )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()
VIEW_FQN = "schema_v2.task_norms_view"   # 🔑 update target schema here

@st.cache_resource
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def run_sql(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d.name for d in cur.description])

# connection sanity check
try:
    _ = run_sql("select 1 as ok;")
    st.sidebar.success("✅ Database connection OK")
except Exception as e:
    st.sidebar.error("❌ Database connection failed")
    st.sidebar.caption(type(e).__name__)
    st.stop()

# =========================================================
# Session state (estimate & last lookup; widget keys)
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate = []  # type: List[Dict]
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup = None  # type: Optional[Dict]

EQ_KEY, COMP_KEY, TASK_KEY = "eq_select", "comp_select", "task_select"

def add_to_estimate(line: dict) -> bool:
    for existing in st.session_state.estimate:
        if (
            existing.get("equipment_class") == line.get("equipment_class")
            and existing.get("component") == line.get("component")
            and existing.get("task_name") == line.get("task_name")
        ):
            return False
    st.session_state.estimate.append(line)
    return True

# Deterministic color
def color_from_text(text: str) -> str:
    if not text: return "#D9D9D9"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    r = int(150 + (r / 255) * 90)
    g = int(150 + (g / 255) * 90)
    b = int(150 + (b / 255) * 90)
    return f"#{r:02X}{g:02X}{b:02X}"

# =========================================================
# Helper queries for dropdown lists
# =========================================================
def list_equipment() -> list[str]:
    df = run_sql(f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class;")
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq: str | None) -> list[str]:
    if not eq: return []
    df = run_sql(
        f"""
        SELECT DISTINCT component
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
        ORDER BY component;
        """,
        [eq],
    )
    return df["component"].tolist() if not df.empty else []

def list_tasks(eq: str | None, comp: str | None) -> list[str]:
    if not (eq and comp): return []
    df = run_sql(
        f"""
        SELECT task_name
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
        """,
        [eq, comp],
    )
    return df["task_name"].tolist() if not df.empty else []

# =========================================================
# Inputs: cascading, searchable dropdowns
# =========================================================
with st.sidebar:
    st.subheader("Find a task")
    equipment = st.selectbox(
        "Equipment Class",
        options=list_equipment(),
        index=None,
        key=EQ_KEY,
        placeholder="Type to search… e.g., Haul Truck 785D",
    )
    component = st.selectbox(
        "Component",
        options=list_components(st.session_state.get(EQ_KEY)),
        index=None,
        key=COMP_KEY,
        placeholder="Select component…",
        disabled=(st.session_state.get(EQ_KEY) is None),
    )
    task_name = st.selectbox(
        "Task Name (exact)",
        options=list_tasks(st.session_state.get(EQ_KEY), st.session_state.get(COMP_KEY)),
        index=None,
        key=TASK_KEY,
        placeholder="Select equipment & component first",
        disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
    )
    lookup_disabled = not (equipment and component and task_name)
    do_lookup = st.button("Lookup", type="primary", disabled=lookup_disabled)
    if st.button("Clear"):
        st.session_state.last_lookup = None
        st.session_state[TASK_KEY] = None
        st.rerun()

# =========================================================
# Lookup & Present
# =========================================================
if do_lookup:
    try:
        df = run_sql(
            f"""
            SELECT *,
                   COALESCE(labour_rate, blended_labour_rate, 0) AS cost_per_hour
            FROM {VIEW_FQN}
            WHERE equipment_class = %s
              AND component = %s
              AND task_name = %s;
            """,
            [equipment, component, task_name],
        )
    except Exception as e:
        st.error("❌ Could not query database.")
        st.caption(f"{type(e).__name__}: {e}")
        st.stop()

    if df.empty:
        st.warning("No exact match found.")
    else:
        df["base_duration_hr"] = pd.to_numeric(df.get("base_duration_hr", 0), errors="coerce")
        df["total_cost"] = (df["cost_per_hour"] * df["base_duration_hr"]).round(2)

        st.session_state.last_lookup = df.iloc[0].to_dict()

        row = st.session_state.last_lookup
        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{row.get('base_duration_hr', 0):.2f}")
        m2.metric("Cost per hour", f"${row.get('cost_per_hour', 0):,.2f}")
        m3.metric("Total cost", f"${row.get('total_cost', 0):,.2f}")

        st.subheader("Result (per task)")
        st.dataframe(df, use_container_width=True)

# =========================================================
# Batch Import with Fuzzy Match
# =========================================================
with st.sidebar.expander("📦 Batch import (CSV)", expanded=False):
    st.caption("Columns (required): task, equipment, component. Optional: quantity")
    match_threshold = st.slider("Match sensitivity (higher = stricter)", 50, 95, 75)
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file:
        try:
            csv_df = pd.read_csv(csv_file)
            st.success(f"Imported {len(csv_df)} rows from CSV")
            # run fuzzy match here...
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")

# =========================================================
# Estimate Table
# =========================================================
st.header("Estimate")
df_est = pd.DataFrame(st.session_state.estimate)
if df_est.empty:
    st.info("No tasks added yet. Lookup a task and click **Add this task** to estimate.")
else:
    st.dataframe(df_est, use_container_width=True)
