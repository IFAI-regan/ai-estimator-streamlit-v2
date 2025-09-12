# app.py

from __future__ import annotations

import os
from io import StringIO
from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql
import streamlit as st


# =========================================================
# Page & styling
# =========================================================
st.set_page_config(page_title="Phase 1 — Exact Name Recall", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption("Supabase → task_norms_view")

st.markdown(
    """
    <style>
      .stMetric span { font-size: 14px !important; }
      .stDataFrame { font-size: 14px; }
      .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DB config (secrets first, then env)
# =========================================================
def _get_str_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    v = st.secrets.get(name) if "secrets" in dir(st) else None
    if v is None:
        v = os.getenv(name, default)
    return v


DATABASE_URL = _get_str_secret("DATABASE_URL")
PG_HOST = _get_str_secret("PG_HOST")
PG_PORT = _get_str_secret("PG_PORT", "6543")
PG_DB = _get_str_secret("PG_DB", "postgres")
PG_USER = _get_str_secret("PG_USER")
PG_PASSWORD = _get_str_secret("PG_PASSWORD")

# Which view to query. Form: "schema.view".
VIEW_FQN = _get_str_secret("VIEW_FQN", "demo_v2.task_norms_view")


@st.cache_resource(show_spinner=False)
def get_conn():
    """Get a cached psycopg2 connection."""
    if DATABASE_URL:
        # DSN path
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    # 5-field path
    missing = [k for k, v in dict(PG_HOST=PG_HOST, PG_USER=PG_USER, PG_PASSWORD=PG_PASSWORD).items() if not v]
    if missing:
        st.error(f"Missing DB secrets: {', '.join(missing)}")
        st.stop()
    return psycopg2.connect(
        host=PG_HOST,
        port=int(PG_PORT or 6543),
        dbname=PG_DB or "postgres",
        user=PG_USER,
        password=PG_PASSWORD,
        sslmode="require",
    )


def run_sql(query, params=None) -> pd.DataFrame:
    """Executes a query (str or psycopg2.sql object) and returns a DataFrame."""
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d.name for d in cur.description])


# Quick connection sanity check
try:
    _ = run_sql("select 1 as ok;")
    st.success("✅ Database connection OK")
except Exception as e:
    st.error("❌ Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()


# =========================================================
# View helpers (safe schema + view handling)
# =========================================================
def _view_ident():
    """Return (Identifier(schema), Identifier(view)) for VIEW_FQN='schema.view'."""
    try:
        schema, rel = VIEW_FQN.split(".", 1)
    except ValueError:
        raise ValueError(f"VIEW_FQN must be 'schema.view', got {VIEW_FQN!r}")
    return sql.Identifier(schema), sql.Identifier(rel)


# =========================================================
# Dropdown data helpers (safe SQL)
# =========================================================
def list_equipment() -> list[str]:
    sch, rel = _view_ident()
    query = sql.SQL(
        "SELECT DISTINCT equipment_class FROM {}.{} ORDER BY equipment_class;"
    ).format(sch, rel)
    df = run_sql(query)
    return df["equipment_class"].tolist() if not df.empty else []


def list_components(eq: Optional[str]) -> list[str]:
    if not eq:
        return []
    sch, rel = _view_ident()
    query = sql.SQL(
        """
        SELECT DISTINCT component
        FROM {}.{}
        WHERE equipment_class = %s
        ORDER BY component;
        """
    ).format(sch, rel)
    df = run_sql(query, [eq])
    return df["component"].tolist() if not df.empty else []


def list_tasks(eq: Optional[str], comp: Optional[str]) -> list[str]:
    if not (eq and comp):
        return []
    sch, rel = _view_ident()
    query = sql.SQL(
        """
        SELECT task_name
        FROM {}.{}
        WHERE equipment_class = %s
          AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
        """
    ).format(sch, rel)
    df = run_sql(query, [eq, comp])
    return df["task_name"].tolist() if not df.empty else []


def lookup_task(eq: str, comp: str, task: str) -> pd.DataFrame:
    sch, rel = _view_ident()
    query = sql.SQL(
        """
        SELECT *
        FROM {}.{}
        WHERE equipment_class = %s
          AND component     = %s
          AND task_name     = %s;
        """
    ).format(sch, rel)
    return run_sql(query, [eq, comp, task])


# =========================================================
# Session state: estimate + last lookup
# (set once; do not store widget values in session_state)
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate: list[dict] = []

if "last_lookup" not in st.session_state:
    st.session_state.last_lookup: Optional[dict] = None


def add_to_estimate(line: dict) -> bool:
    """Add row if not already present by (equipment, component, task_name)."""
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
                "crew_roles", "crew_count", "notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate)
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df


# =========================================================
# Inputs (cascading) + Lookup (button)
# =========================================================
colA, colB = st.columns(2)
equipment = colA.selectbox(
    "Equipment Class",
    options=list_equipment(),
    index=None,
    placeholder="Type to search… e.g., Stacker Reclaimer",
)
component = colB.selectbox(
    "Component",
    options=list_components(equipment),
    index=None,
    placeholder="Select component…",
    disabled=(equipment is None),
)
task_name = st.selectbox(
    "Task Name (exact)",
    options=list_tasks(equipment, component),
    index=None,
    placeholder=("Select equipment & component first" if not (equipment and component) else "Select a task…"),
    disabled=not (equipment and component),
)

btn_col1, btn_col2 = st.columns([1, 1])
do_lookup = btn_col1.button(
    "Lookup",
    type="primary",
    disabled=not (equipment and component and task_name),
)
if btn_col2.button("Clear selections"):
    st.session_state.last_lookup = None
    st.rerun()


# =========================================================
# Lookup & Present
# =========================================================
if do_lookup:
    df = lookup_task(equipment, component, task_name)
    if df.empty:
        st.warning("No exact match found.")
    else:
        # Derive cost_per_hour if needed
        cost_col = None
        for candidate in ["labour_rate", "blended_labour_rate"]:
            if candidate in df.columns:
                cost_col = candidate
                break

        if cost_col:
            df["cost_per_hour"] = pd.to_numeric(df[cost_col], errors="coerce")
        else:
            df["cost_per_hour"] = None

        if "base_duration_hr" in df.columns:
            df["base_duration_hr"] = pd.to_numeric(df["base_duration_hr"], errors="coerce")
            df["total_cost"] = (df["cost_per_hour"] * df["base_duration_hr"]).round(2)
        else:
            df["total_cost"] = None

        # Persist selected row (first) to support "Add to estimate"
        st.session_state.last_lookup = df.iloc[0].to_dict()

        # Metrics row
        row = st.session_state.last_lookup
        dur = float(row.get("base_duration_hr") or 0)
        cph = float(row.get("cost_per_hour") or 0)
        tot = float(row.get("total_cost") or 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
        m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

        # Crew block
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

        # One-row tidy result (hide blended columns if present)
        tidy_df = pd.DataFrame([row]).copy()
        for col_to_drop in ["blended_labour_rate"]:
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
                "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            },
        )

# Add to estimate
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
    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")


# =========================================================
# Estimate cart
# =========================================================
st.markdown("---")
st.header("Estimate")

df_est = estimate_df()

if df_est.empty:
    st.caption("No tasks added yet. Lookup a task and click **Add to estimate**.")
else:
    # Filter box
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
            st.session_state.estimate = []
            st.rerun()
    with actB:
        if st.button("↩️ Remove last"):
            if st.session_state.estimate:
                st.session_state.estimate.pop()
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
        csv_out = df_est[csv_cols] if set(csv_cols).issubset(df_est.columns) else df_est
        buf = StringIO()
        csv_out.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV for Excel",
            data=buf.getvalue(),
            file_name="estimate.csv",
            mime="text/csv",
        )
