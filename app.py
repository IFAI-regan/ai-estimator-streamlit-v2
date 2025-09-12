import os
import hashlib
from io import StringIO
import urllib.parse as _url
from typing import Optional, List, Dict

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st
import difflib
import re

# =========================================================
# Page & Styles
# =========================================================
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption("Supabase → task_norms_view")

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
    st.success("✅ Database connection OK")
except Exception as e:
    st.error("❌ Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()

# =========================================================
# Session state (estimate & last lookup; widget keys)
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate = []  # type: List[Dict]
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup = None  # type: Optional[Dict]

EQ_KEY = "eq_select"
COMP_KEY = "comp_select"
TASK_KEY = "task_select"

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

def estimate_df() -> pd.DataFrame:
    if not st.session_state.estimate:
        return pd.DataFrame(
            columns=[
                "row", "equipment_class", "component", "task_name", "task_code",
                "base_duration_hr", "cost_per_hour", "total_cost",
                "component_color", "crew_roles", "crew_count", "notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate)
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df

def color_from_text(text: str) -> str:
    if not text:
        return "#D9D9D9"
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
    df = run_sql(
        "SELECT DISTINCT equipment_class FROM task_norms_view ORDER BY equipment_class;"
    )
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq: str | None) -> list[str]:
    if not eq:
        return []
    df = run_sql(
        """
        SELECT DISTINCT component
        FROM task_norms_view
        WHERE equipment_class = %s
        ORDER BY component;
        """,
        [eq],
    )
    return df["component"].tolist() if not df.empty else []

def list_tasks(eq: str | None, comp: str | None) -> list[str]:
    if not (eq and comp):
        return []
    df = run_sql(
        """
        SELECT task_name
        FROM task_norms_view
        WHERE equipment_class = %s
          AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
        """,
        [eq, comp],
    )
    return df["task_name"].tolist() if not df.empty else []

# =========================================================
# Inputs
# =========================================================
colA, colB = st.columns(2)
equipment = colA.selectbox(
    "Equipment Class",
    options=list_equipment(),
    index=None,
    key=EQ_KEY,
    placeholder="Type to search… e.g., Haul Truck 785D",
)
component = colB.selectbox(
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
    placeholder=("Select equipment & component first" if not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)) else "Select a task…"),
    disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
)

btn_col1, btn_col2 = st.columns([1, 1])
lookup_disabled = not (equipment and component and task_name)
with btn_col1:
    do_lookup = st.button("Lookup", type="primary", disabled=lookup_disabled)
with btn_col2:
    if st.button("Clear selections"):
        st.session_state.last_lookup = None
        st.session_state[TASK_KEY] = None
        st.rerun()

# =========================================================
# Lookup & Present
# =========================================================
if do_lookup:
    df = run_sql(
        """
        SELECT *
        FROM task_norms_view
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s;
        """,
        [equipment, component, task_name],
    )

    if df.empty:
        st.warning("No exact match found.")
    else:
        cost_col = None
        if "labour_rate" in df.columns:
            cost_col = "labour_rate"
        elif "blended_labour_rate" in df.columns:
            cost_col = "blended_labour_rate"

        if cost_col:
            df["cost_per_hour"] = pd.to_numeric(df[cost_col], errors="coerce")
        else:
            df["cost_per_hour"] = None

        if "base_duration_hr" in df.columns:
            df["base_duration_hr"] = pd.to_numeric(df["base_duration_hr"], errors="coerce")
            df["total_cost"] = (df["cost_per_hour"] * df["base_duration_hr"]).round(2)
        else:
            df["total_cost"] = None

        st.session_state.last_lookup = df.iloc[0].to_dict()

        row = st.session_state.last_lookup
        dur = float(row.get("base_duration_hr") or 0)
        cph = float(row.get("cost_per_hour") or 0)
        tot = float(row.get("total_cost") or 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
        m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

        st.subheader("Crew")
        roles = str(row.get("crew_roles", "") or "").split("|")
        counts = str(row.get("crew_count", "") or "").split("|")
        crew_lines = []
        for r, c in zip(roles, counts):
            r = r.strip()
            c = c.strip()
            if r:
                crew_lines.append(f"- **{r}** × {c or '1'}")
        if crew_lines:
            st.markdown("\n".join(crew_lines))
        else:
            st.caption("No crew information found.")

        tidy_df = pd.DataFrame([row]).copy()
        for col_to_drop in ["blended_labour_rate", "labour_rate", "cost_per_hr"]:
            if col_to_drop in tidy_df.columns:
                tidy_df = tidy_df.drop(columns=[col_to_drop])

        preferred = [
            "task_code",
            "equipment_class",
            "component",
            "task_name",
            "base_duration_hr",
            "cost_per_hour",
            "total_cost",
            "crew_roles",
            "crew_count",
            "notes",
            "effective_from",
            "effective_to",
        ]

        present = [c for c in preferred if c in tidy_df.columns]
        others = [c for c in tidy_df.columns if c not in present]
        final_cols = present + others
        tidy_df = tidy_df[final_cols]

        st.subheader("Result (per task)")
        st.dataframe(
            tidy_df,
            use_container_width=True,
            column_config={
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            },
        )

# =========================================================
# Add to Estimate
# =========================================================
st.markdown("### Add to estimate")
lr = st.session_state.last_lookup
can_add = bool(lr) and lr.get("total_cost") is not None

if st.button("➕ Add this task", disabled=not can_add):
    line = {
        "equipment_class": lr.get("equipment_class"),
        "component": lr.get("component"),
        "task_name": lr.get("task_name"),
        "task_code": lr.get("task_code"),
        "base_duration_hr": float(lr.get("base_duration_hr") or 0),
        "cost_per_hour": float(lr.get("cost_per_hour") or 0),
        "total_cost": float(lr.get("total_cost") or 0),
        "crew_roles": lr.get("crew_roles"),
        "crew_count": lr.get("crew_count"),
        "notes": lr.get("notes"),
    }
    try:
        line["component_color"] = color_from_text(line["component"])
    except Exception:
        pass

    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

    try:
        st.session_state.pop(TASK_KEY, None)
    except Exception:
        pass
    st.rerun()

# =========================================================
# Estimate
# =========================================================
st.markdown("---")
st.header("Estimate")

df_est = estimate_df()

if df_est.empty:
    st.caption("No tasks added yet. Lookup a task and click **Add to estimate**.")
else:
    filter_text = st.text_input(
        "Filter estimate (search across task / equipment / component):",
        placeholder="Type to filter…",
    )
    display_df = df_est.copy()
    if filter_text:
        ft = filter_text.lower()
        mask = (
            display_df["task_name"].astype(str).str.lower().str.contains(ft)
            | display_df["equipment_class"].astype(str).str.lower().str.contains(ft)
            | display_df["component"].astype(str).str.lower().str.contains(ft)
        )
        display_df = display_df[mask]

    subtotal = pd.to_numeric(display_df.get("total_cost", 0), errors="coerce").sum()

    c1, c2 = st.columns([3, 1])
    with c1:
        col_config = {
            "row": st.column_config.NumberColumn("#", format="%.0f"),
            "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
            "cost_per_hour":    st.column_config.NumberColumn("Cost/hr",      format="$%.2f"),
            "total_cost":       st.column_config.NumberColumn("Cost/Task",    format="$%.2f"),
        }
        if "component_color" in display_df.columns:
            if hasattr(st.column_config, "ColorColumn"):
                col_config["component_color"] = st.column_config.ColorColumn("Color")
            else:
                col_config["component_color"] = st.column_config.TextColumn("Color")
        cols_to_show = [
            "row", "equipment_class", "component", "task_name", "task_code",
            "base_duration_hr", "cost_per_hour", "total_cost", "crew_roles",
            "crew_count", "notes",
        ]
        if "component_color" in display_df.columns:
            cols_to_show.insert(3, "component_color")
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
        st.dataframe(display_df[cols_to_show], use_container_width=True, column_config=col_config)

    with c2:
        st.metric("Estimate subtotal", f"${subtotal:,.2f}")

    actA, actB, actC = st.columns([1, 1, 2])
    with actA:
        if st.button("🗑️ Clear estimate"):
            st.session_state.estimate = []
            st.rerun()
    with actB:
        if st.button("↩️ Remove last"):
            if st.session_state.estimate:
                st.session_state.estimate.pop()
                st.rerun()
    with actC:
        csv_cols = [
            "row","equipment_class","component","task_name","task_code",
            "base_duration_hr","cost_per_hour","total_cost","crew_roles",
            "crew_count","notes",
        ]
        if "component_color" in df_est.columns:
            csv_cols.insert(3, "component_color")
        csv_out = df_est[[c for c in csv_cols if c in df_est.columns]].copy()
        buf = StringIO()
        csv_out.to_csv(buf, index=False)
        st.download_button("⬇️ Download CSV for Excel", data=buf.getvalue(), file_name="estimate.csv", mime="text/csv")

# =========================================================
# Batch import & fuzzy match
# =========================================================
st.markdown("---")
st.header("Batch import & fuzzy match")

st.caption("Upload a CSV with columns like **task**, **equipment**, **component** (and optional **quantity**).")

@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    df = run_sql(
        """
        SELECT equipment_class, component, task_name, task_code,
               base_duration_hr, COALESCE(labour_rate, blended_labour_rate) AS labour_rate
        FROM task_norms_view
        """
    )
    if "base_duration_hr" in df.columns:
        df["base_duration_hr"] = pd.to_numeric(df["base_duration_hr"], errors="coerce").fillna(0.0)
    if "labour_rate" in df.columns:
        df["labour_rate"] = pd.to_numeric(df["labour_rate"], errors="coerce").fillna(0.0)
    df["cost_per_task"] = (df["base_duration_hr"] * df["labour_rate"]).round(2)
    for c in ("equipment_class","component","task_name"):
        df[f"{c}__lc"] = df[c].astype(str).str.lower().str.strip()
    return df

CATALOG = load_catalog()

def _norm(s: str) -> str:
    if s is None: return ""
    s = re.sub(r"[^a-zA-Z0-9\s]+", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def infer_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in df.columns:
        for cand in candidates:
            if cand in c.lower():
                return c
    return None

def best_match(value: str, choices: list[str]) -> tuple[str, float]:
    if not value or not choices:
        return "", 0.0
    value = _norm(value)
    pick = difflib.get_close_matches(value, choices, n=1, cutoff=0)
    if pick:
        choice = pick[0]
    else:
        choice = choices[0]
    ratio = difflib.SequenceMatcher(a=value, b=choice).ratio()
    return choice, ratio

def fuzzy_match_row(row: dict, catalog: pd.DataFrame) -> dict:
    eq_in  = _norm(row.get("equipment", ""))
    comp_in= _norm(row.get("component", ""))
    task_in= _norm(row.get("task", ""))
    eq_choices   = sorted(catalog["equipment_class__lc"].unique().tolist())
    comp_choices = sorted(catalog["component__lc"].unique().tolist())
    task_choices = sorted(catalog["task_name__lc"].unique().tolist())
    best_eq, r_eq   = best_match(eq_in,   eq_choices)   if eq_in else ("", 0.0)
    best_comp,r_comp= best_match(comp_in, comp_choices) if comp_in else ("", 0.0)
    best_task,r_task= best_match(task_in, task_choices) if task_in else ("", 0.0)
   
