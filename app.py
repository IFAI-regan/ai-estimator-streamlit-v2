# app.py — App B (Feature branch)
# Reads demo dataset from demo_v2.task_norms_view

import os
from io import StringIO
from typing import Optional, List

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# -----------------------------
# App / page config & simple CSS
# -----------------------------
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption("Supabase → demo_v2.task_norms_view")

st.markdown(
    """
    <style>
      .stDataFrame { font-size: 14px; }
      .stMetric span { font-size: 14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB connection helpers
# -----------------------------
SCHEMA = "demo_v2"                # <- NEW: point to demo schema
VIEW   = f"{SCHEMA}.task_norms_view"

def _dsn_from_secrets() -> Optional[str]:
    # Preferred: single DATABASE_URL
    if "DATABASE_URL" in st.secrets:
        return st.secrets["DATABASE_URL"].strip()
    # Fallback: 5-field secrets
    required = ["PG_HOST", "PG_PORT", "PG_DB", "PG_USER", "PG_PASSWORD"]
    if all(k in st.secrets for k in required):
        host = st.secrets["PG_HOST"]
        port = st.secrets["PG_PORT"]
        db   = st.secrets["PG_DB"]
        user = st.secrets["PG_USER"]
        pwd  = st.secrets["PG_PASSWORD"]
        return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    return None

DSN = _dsn_from_secrets()
if not DSN:
    st.error(
        "Missing DB secrets.\n\n"
        "Add either `DATABASE_URL` **or** the 5 keys: "
        "`PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD`."
    )
    st.stop()

@st.cache_resource(show_spinner=False)
def get_conn():
    # Always require SSL on hosted PG
    return psycopg2.connect(DSN, sslmode="require")

def run_sql(sql: str, params: Optional[list] = None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d.name for d in cur.description])

# Sanity check
try:
    _ = run_sql("select 1 as ok;")
    st.success("✅ Database connection OK")
except Exception as e:
    st.error("❌ Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()

# -----------------------------
# Session keys & state
# -----------------------------
EQ_KEY   = "sel_equipment"
COMP_KEY = "sel_component"
TASK_KEY = "sel_task"
LAST_KEY = "last_lookup"
EST_KEY  = "estimate"

for k, default in [
    (EQ_KEY, None),
    (COMP_KEY, None),
    (TASK_KEY, None),
    (LAST_KEY, None),
    (EST_KEY, []),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# -----------------------------
# Lookup lists (cascading)
# -----------------------------
def list_equipment() -> List[str]:
    df = run_sql(f"SELECT DISTINCT equipment_class FROM {VIEW} ORDER BY equipment_class;")
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq: Optional[str]) -> List[str]:
    if not eq:
        return []
    df = run_sql(
        f"""
        SELECT DISTINCT component
        FROM {VIEW}
        WHERE equipment_class = %s
        ORDER BY component;
        """,
        [eq],
    )
    return df["component"].tolist() if not df.empty else []

def list_tasks(eq: Optional[str], comp: Optional[str]) -> List[str]:
    if not (eq and comp):
        return []
    df = run_sql(
        f"""
        SELECT task_name
        FROM {VIEW}
        WHERE equipment_class = %s
          AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
        """,
        [eq, comp],
    )
    return df["task_name"].tolist() if not df.empty else []

# -----------------------------
# Inputs (cascading selectboxes) — SAFE pattern (no widget keys)
# -----------------------------
colA, colB = st.columns(2)

eq_value = colA.selectbox(
    "Equipment Class",
    options=list_equipment(),
    index=None,
    placeholder="Type to search… e.g., Stacker Reclaimer",
)

# update session state after the widget returns
st.session_state[EQ_KEY] = eq_value

comp_value = colB.selectbox(
    "Component",
    options=list_components(st.session_state[EQ_KEY]),
    index=None,
    placeholder="Select component…",
    disabled=(st.session_state[EQ_KEY] is None),
)
st.session_state[COMP_KEY] = comp_value

task_value = st.selectbox(
    "Task Name (exact)",
    options=list_tasks(st.session_state[EQ_KEY], st.session_state[COMP_KEY]),
    index=None,
    placeholder=(
        "Select equipment & component first"
        if not (st.session_state[EQ_KEY] and st.session_state[COMP_KEY])
        else "Select a task…"
    ),
    disabled=not (st.session_state[EQ_KEY] and st.session_state[COMP_KEY]),
)
st.session_state[TASK_KEY] = task_value

# -----------------------------
# Lookup & present
# -----------------------------
def add_to_estimate(row: dict) -> bool:
    """Append if not duplicate (same equipment+component+task_name)."""
    for existing in st.session_state[EST_KEY]:
        if (
            existing.get("equipment_class") == row.get("equipment_class")
            and existing.get("component") == row.get("component")
            and existing.get("task_name") == row.get("task_name")
        ):
            return False
    st.session_state[EST_KEY].append(row)
    return True

if do_lookup:
    df = run_sql(
        f"""
        SELECT *
        FROM {VIEW}
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s;
        """,
        [st.session_state[EQ_KEY], st.session_state[COMP_KEY], st.session_state[TASK_KEY]],
    )

    if df.empty:
        st.warning("No exact match found.")
    else:
        # Normalize numeric columns
        if "labour_rate" in df.columns:
            df["labour_rate"] = pd.to_numeric(df["labour_rate"], errors="coerce")
        if "base_duration_hr" in df.columns:
            df["base_duration_hr"] = pd.to_numeric(df["base_duration_hr"], errors="coerce")

        # Cost per hour = labour_rate; total = labour_rate * base_duration_hr
        df["cost_per_hour"] = df.get("labour_rate")
        if "base_duration_hr" in df.columns:
            df["total_cost"] = (df["cost_per_hour"] * df["base_duration_hr"]).round(2)
        else:
            df["total_cost"] = None

        # Store first row for "Add to estimate"
        st.session_state[LAST_KEY] = df.iloc[0].to_dict()

        # Metrics strip
        row = st.session_state[LAST_KEY]
        dur = float(row.get("base_duration_hr") or 0)
        cph = float(row.get("cost_per_hour") or 0)
        tot = float(row.get("total_cost") or 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
        m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

        # Crew section
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

        # One-row tidy table for readability
        tidy = pd.DataFrame([row]).copy()
        wanted = [
            "task_code", "equipment_class", "component", "task_name",
            "base_duration_hr", "cost_per_hour", "total_cost",
            "crew_roles", "crew_count", "notes", "effective_from", "effective_to",
        ]
        present = [c for c in wanted if c in tidy.columns]
        tidy = tidy[present + [c for c in tidy.columns if c not in present]]

        st.subheader("Result (per task)")
        st.dataframe(
            tidy,
            use_container_width=True,
            column_config={
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            },
        )

# -----------------------------
# Add to estimate
# -----------------------------
st.markdown("### Add to estimate")
lr = st.session_state[LAST_KEY]
can_add = bool(lr) and (lr.get("total_cost") is not None)

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
    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
        # UX: clear just the task (so you can quickly pick another)
        st.session_state[TASK_KEY] = None
    else:
        st.toast("Already in estimate", icon="ℹ️")
    st.rerun()

# -----------------------------
# Estimate cart + export
# -----------------------------
st.markdown("---")
st.header("Estimate")

def estimate_df() -> pd.DataFrame:
    if not st.session_state[EST_KEY]:
        return pd.DataFrame(
            columns=[
                "row", "equipment_class", "component", "task_name", "task_code",
                "base_duration_hr", "cost_per_hour", "total_cost",
                "crew_roles", "crew_count", "notes"
            ]
        )
    df = pd.DataFrame(st.session_state[EST_KEY]).copy()
    df.insert(0, "row", range(1, len(df) + 1))
    # Ensure numeric formatting
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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
                "row": st.column_config.NumberColumn("#", format="%.0f"),
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            },
        )
    with c2:
        st.metric("Estimate subtotal", f"${subtotal:,.2f}")

    actA, actB, actC = st.columns([1, 1, 2])
    with actA:
        if st.button("🗑️ Clear estimate"):
            st.session_state[EST_KEY] = []
            st.rerun()
    with actB:
        if st.button("↩️ Remove last"):
            if st.session_state[EST_KEY]:
                st.session_state[EST_KEY].pop()
                st.rerun()
    with actC:
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
