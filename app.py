# app.py
import os
from io import StringIO

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql
import streamlit as st

# ==============================
# Page & styles
# ==============================
st.set_page_config(page_title="Phase 1 — Exact Name Recall", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption(f"Supabase → {st.secrets.get('VIEW_FQN', 'task_norms_view')}")

st.markdown(
    """
    <style>
      .stMetric span { font-size: 14px !important; }
      .stDataFrame { font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Secrets / DB config
# ==============================
DB_URL  = st.secrets.get("DATABASE_URL", "").strip()
VIEW_FQN = st.secrets.get("VIEW_FQN", "").strip()

if not DB_URL or not VIEW_FQN or "." not in VIEW_FQN:
    st.error(
        "Missing or invalid secrets. Please add:\n\n"
        "- `DATABASE_URL` (use the **Transaction Pooler** URL on port 6543)\n"
        "- `VIEW_FQN` like `demo_v2.task_norms_view`\n"
    )
    st.stop()

VIEW_SCHEMA, VIEW_NAME = VIEW_FQN.split(".", 1)

# Build a safe, qualified identifier for the view (schema.view)
VIEW_IDENT = sql.SQL("{}.{}").format(
    sql.Identifier(VIEW_SCHEMA),
    sql.Identifier(VIEW_NAME),
)

# ==============================
# DB helpers
# ==============================
@st.cache_resource
def get_conn():
    # Use pooled DSN; require SSL
    return psycopg2.connect(DB_URL, sslmode="require")

def run_sql(query: sql.SQL, params=None) -> pd.DataFrame:
    """
    query: psycopg2.sql.SQL object (use sql.Placeholder for params)
    """
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d.name for d in cur.description])

# Connection sanity check
try:
    _ = run_sql(sql.SQL("select 1 as ok;"))
    st.success("✅ Database connection OK")
except Exception as e:
    st.error("❌ Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()

# ==============================
# Session state (non-widget keys)
# ==============================
if "estimate" not in st.session_state:
    st.session_state.estimate: list[dict] = []

if "last_lookup" not in st.session_state:
    st.session_state.last_lookup: dict | None = None

def add_to_estimate(line: dict) -> bool:
    """Append if not already present (same equipment, component, task_code or task_name)."""
    for existing in st.session_state.estimate:
        if (
            existing.get("equipment_class") == line.get("equipment_class")
            and existing.get("component") == line.get("component")
            and (
                existing.get("task_code") == line.get("task_code")
                or existing.get("task_name") == line.get("task_name")
            )
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
                "crew_roles", "crew_count", "notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate).copy()
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df

# ==============================
# Helper queries for dropdowns
# ==============================
def list_equipment() -> list[str]:
    q = sql.SQL("SELECT DISTINCT equipment_class FROM {} ORDER BY equipment_class;").format(VIEW_IDENT)
    df = run_sql(q)
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq: str | None) -> list[str]:
    if not eq:
        return []
    q = sql.SQL("""
        SELECT DISTINCT component
        FROM {}
        WHERE equipment_class = %s
        ORDER BY component;
    """).format(VIEW_IDENT)
    df = run_sql(q, [eq])
    return df["component"].tolist() if not df.empty else []

def list_tasks(eq: str | None, comp: str | None) -> list[str]:
    if not (eq and comp):
        return []
    q = sql.SQL("""
        SELECT task_name
        FROM {}
        WHERE equipment_class = %s
          AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
    """).format(VIEW_IDENT)
    df = run_sql(q, [eq, comp])
    return df["task_name"].tolist() if not df.empty else []

# ==============================
# UI: Cascading dropdowns (widgets own their state)
# ==============================
EQ_KEY, COMP_KEY, TASK_KEY = "ui_eq", "ui_comp", "ui_task"

colA, colB = st.columns(2)

eq = colA.selectbox(
    "Equipment Class",
    options=list_equipment(),
    index=None,
    placeholder="Type to search… e.g., Stacker Reclaimer",
    key=EQ_KEY,  # widget-managed state
)

comp = colB.selectbox(
    "Component",
    options=list_components(st.session_state.get(EQ_KEY)),
    index=None,
    placeholder="Select component…",
    disabled=(st.session_state.get(EQ_KEY) is None),
    key=COMP_KEY,
)

task_name = st.selectbox(
    "Task Name (exact)",
    options=list_tasks(st.session_state.get(EQ_KEY), st.session_state.get(COMP_KEY)),
    index=None,
    placeholder=(
        "Select equipment & component first"
        if not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY))
        else "Select a task…"
    ),
    disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
    key=TASK_KEY,
)

# Action buttons
btn_col1, btn_col2 = st.columns([1, 1])
lookup_disabled = not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY) and st.session_state.get(TASK_KEY))
with btn_col1:
    do_lookup = st.button("Lookup", type="primary", disabled=lookup_disabled)
with btn_col2:
    if st.button("Clear selections"):
        # Remove widget keys, then rerun to reset UI cleanly
        for k in (EQ_KEY, COMP_KEY, TASK_KEY):
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.last_lookup = None
        st.rerun()

# ==============================
# Lookup & Present
# ==============================
if do_lookup:
    q = sql.SQL("""
        SELECT *
        FROM {}
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s;
    """).format(VIEW_IDENT)
    equipment = st.session_state.get(EQ_KEY)
    component = st.session_state.get(COMP_KEY)

    df = run_sql(q, [equipment, component, task_name])

    if df.empty:
        st.warning("No exact match found.")
    else:
        # Cost per hour logic: prefer labour_rate column if present
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

        # Save first row for "Add to estimate"
        st.session_state.last_lookup = df.iloc[0].to_dict()

        # Metrics
        row = st.session_state.last_lookup
        dur = float(row.get("base_duration_hr") or 0)
        cph = float(row.get("cost_per_hour") or 0)
        tot = float(row.get("total_cost") or 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
        m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

        # Crew
        st.subheader("Crew")
        roles = str(row.get("crew_roles", "") or "")
        counts = str(row.get("crew_count", "") or "")
        role_list  = roles.split("|") if roles else []
        count_list = counts.split("|") if counts else []
        if role_list:
            for r, c in zip(role_list, count_list if count_list else [""]*len(role_list)):
                r = r.strip()
                c = (c or "1").strip()
                if r:
                    st.markdown(f"- **{r}** × {c}")
        else:
            st.caption("No crew information found.")

        # One-row tidy table (hide blended rate if present)
        tidy_df = pd.DataFrame([row]).copy()
        for col_to_drop in ["blended_labour_rate", "labour_rate"]:
            if col_to_drop in tidy_df.columns:
                tidy_df = tidy_df.drop(columns=[col_to_drop])

        wanted = [
            "task_code", "equipment_class", "component", "task_name",
            "base_duration_hr", "cost_per_hour", "total_cost",
            "crew_roles", "crew_count", "notes", "effective_from", "effective_to",
        ]
        present = [c for c in wanted if c in tidy_df.columns]
        tidy_df = tidy_df[present + [c for c in tidy_df.columns if c not in present]]

        st.subheader("Result (per task)")
        st.dataframe(
            tidy_df,
            use_container_width=True,
            column_config={
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour":    st.column_config.NumberColumn("Cost/hr",      format="$%.2f"),
                "total_cost":       st.column_config.NumberColumn("Cost/Task",    format="$%.2f"),
            },
        )

# ==============================
# Add to Estimate
# ==============================
st.markdown("### Add to estimate")
lr = st.session_state.last_lookup
can_add = bool(lr) and lr.get("total_cost") is not None
if st.button("➕ Add this task", disabled=not can_add):
    line = {
        "equipment_class": lr.get("equipment_class"),
        "component":       lr.get("component"),
        "task_name":       lr.get("task_name"),
        "task_code":       lr.get("task_code"),
        "base_duration_hr": float(lr.get("base_duration_hr") or 0),
        "cost_per_hour":    float(lr.get("cost_per_hour") or 0),
        "total_cost":       float(lr.get("total_cost") or 0),
        "crew_roles":       lr.get("crew_roles"),
        "crew_count":       lr.get("crew_count"),
        "notes":            lr.get("notes"),
    }
    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

# ==============================
# Estimate Cart
# ==============================
st.markdown("---")
st.header("Estimate")

df_est = estimate_df()

if df_est.empty:
    st.caption("No tasks added yet. Lookup a task and click **Add to estimate**.")
else:
    # Quick search/filter for the estimate table
    filter_text = st.text_input(
        "Filter estimate (search across task / equipment / component):",
        placeholder="Type to filter…"
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

    subtotal = pd.to_numeric(display_df["total_cost"], errors="coerce").sum()

    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(
            display_df[
                [
                    "row",
                    "equipment_class",
                    "component",
                    "task_name",
                    "task_code",
                    "base_duration_hr",
                    "cost_per_hour",
                    "total_cost",
                    "crew_roles",
                    "crew_count",
                    "notes",
                ]
            ],
            use_container_width=True,
            column_config={
                "row":              st.column_config.NumberColumn("#",             format="%.0f"),
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour":    st.column_config.NumberColumn("Cost/hr",       format="$%.2f"),
                "total_cost":       st.column_config.NumberColumn("Cost/Task",     format="$%.2f"),
            },
        )
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
        # CSV export (Excel-ready)
        csv_cols = [
            "row",
            "equipment_class",
            "component",
            "task_name",
            "task_code",
            "base_duration_hr",
            "cost_per_hour",
            "total_cost",
            "crew_roles",
            "crew_count",
            "notes",
        ]
        csv_out = display_df[csv_cols] if set(csv_cols).issubset(display_df.columns) else display_df
        buf = StringIO()
        csv_out.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV for Excel",
            data=buf.getvalue(),
            file_name="estimate.csv",
            mime="text/csv",
        )
