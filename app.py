# app.py
import os
from io import StringIO

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption("Supabase → task_norms_view")

# Use demo_v2 by default. Change to "public.task_norms_view" if needed.
VIEW_FQN = os.getenv("VIEW_FQN", "demo_v2.task_norms_view")

# Small CSS polish
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
# Database connection (supports DATABASE_URL or 5-field secrets)
# =========================================================
def _connect_via_url():
    url = st.secrets["DATABASE_URL"].strip()
    return psycopg2.connect(url, sslmode="require")

def _connect_via_fields():
    return psycopg2.connect(
        host=st.secrets["PG_HOST"],
        port=st.secrets.get("PG_PORT", "6543"),
        dbname=st.secrets.get("PG_DB", "postgres"),
        user=st.secrets["PG_USER"],
        password=st.secrets["PG_PASSWORD"],
        sslmode="require",
    )

@st.cache_resource(show_spinner=False)
def get_conn():
    if "DATABASE_URL" in st.secrets:
        return _connect_via_url()
    required = ("PG_HOST", "PG_USER", "PG_PASSWORD")
    if not all(k in st.secrets for k in required):
        st.error("Missing DB secrets: either `DATABASE_URL` or the 5 keys "
                 "`PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD`.")
        st.stop()
    return _connect_via_fields()

def run_sql(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d.name for d in cur.description])

# Connection check
try:
    _ = run_sql("select 1;")
    st.success("✅ Database connection OK")
except Exception as e:
    st.error("❌ Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()

# =========================================================
# Helper queries for dropdowns (against VIEW_FQN)
# =========================================================
def list_equipment() -> list[str]:
    df = run_sql(f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class;")
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq: str | None) -> list[str]:
    if not eq:
        return []
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
    if not (eq and comp):
        return []
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
# Session state init
# =========================================================
EQ_KEY, COMP_KEY, TASK_KEY = "filter_eq", "filter_comp", "filter_task"
for k in (EQ_KEY, COMP_KEY, TASK_KEY):
    st.session_state.setdefault(k, None)
st.session_state.setdefault("last_lookup", None)
st.session_state.setdefault("estimate", [])

def add_to_estimate(line: dict) -> bool:
    """Prevent exact duplicates by eq+comp+task_name."""
    for existing in st.session_state.estimate:
        if (
            existing.get("equipment_class") == line.get("equipment_class") and
            existing.get("component") == line.get("component") and
            existing.get("task_name") == line.get("task_name")
        ):
            return False
    st.session_state.estimate.append(line)
    return True

def estimate_df() -> pd.DataFrame:
    if not st.session_state.estimate:
        return pd.DataFrame(
            columns=[
                "row","equipment_class","component","task_name","task_code",
                "base_duration_hr","cost_per_hour","total_cost","crew_roles","crew_count","notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate)
    for c in ["base_duration_hr","cost_per_hour","total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df

# =========================================================
# Pattern 1: Form + callbacks
# =========================================================
def on_equipment_change():
    st.session_state[COMP_KEY] = None
    st.session_state[TASK_KEY] = None

def on_component_change():
    st.session_state[TASK_KEY] = None

def clear_filters():
    for k in (EQ_KEY, COMP_KEY, TASK_KEY, "last_lookup"):
        st.session_state.pop(k, None)
    st.experimental_rerun()

with st.form("filters", clear_on_submit=False):
    colA, colB = st.columns(2)

    eq_value = colA.selectbox(
        "Equipment Class",
        options=list_equipment(),
        index=None,
        key=EQ_KEY,
        on_change=on_equipment_change,
        placeholder="Type to search… e.g., Stacker Reclaimer",
    )

    comp_value = colB.selectbox(
        "Component",
        options=list_components(eq_value),
        index=None,
        key=COMP_KEY,
        on_change=on_component_change,
        placeholder="Select component…",
        disabled=(eq_value is None),
    )

    task_value = st.selectbox(
        "Task Name (exact)",
        options=list_tasks(eq_value, comp_value),
        index=None,
        key=TASK_KEY,
        placeholder=("Select equipment & component first"
                     if not (eq_value and comp_value) else "Select a task…"),
        disabled=not (eq_value and comp_value),
    )

    c1, c2 = st.columns([1, 1])
    submitted = c1.form_submit_button(
        "Lookup",
        type="primary",
        disabled=not (eq_value and comp_value and task_value),
    )
    c2.form_submit_button("Clear selections", on_click=clear_filters)

# =========================================================
# Lookup & Present
# =========================================================
if submitted:
    df = run_sql(
        f"""
        SELECT *
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s;
        """,
        [st.session_state[EQ_KEY], st.session_state[COMP_KEY], st.session_state[TASK_KEY]],
    )

    if df.empty:
        st.warning("No exact match found.")
    else:
        # normalize cost columns
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

        # store first row as last selection
        st.session_state["last_lookup"] = df.iloc[0].to_dict()

        # metrics
        row = st.session_state["last_lookup"]
        dur = float(row.get("base_duration_hr") or 0)
        cph = float(row.get("cost_per_hour") or 0)
        tot = float(row.get("total_cost") or 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
        m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

        # crew
        st.subheader("Crew")
        roles = str(row.get("crew_roles", "") or "").split("|")
        counts = str(row.get("crew_count", "") or "").split("|")
        bullets = []
        for r, c in zip(roles, counts):
            r = r.strip()
            c = c.strip()
            if r:
                bullets.append(f"- **{r}** × {c or '1'}")
        if bullets:
            st.markdown("\n".join(bullets))
        else:
            st.caption("No crew information found.")

        # compact single-row table
        tidy_df = pd.DataFrame([row]).copy()
        for drop_col in ["blended_labour_rate", "labour_rate"]:
            if drop_col in tidy_df.columns:
                tidy_df.drop(columns=[drop_col], inplace=True)

        ordered = [
            "task_code","equipment_class","component","task_name",
            "base_duration_hr","cost_per_hour","total_cost",
            "crew_roles","crew_count","notes","effective_from","effective_to",
        ]
        present = [c for c in ordered if c in tidy_df.columns]
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

# =========================================================
# Add to Estimate
# =========================================================
st.markdown("### Add to estimate")
lr = st.session_state.get("last_lookup")
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
# Estimate Cart
# =========================================================
st.markdown("---")
st.header("Estimate")

df_est = estimate_df()

if df_est.empty:
    st.caption("No tasks added yet. Lookup a task and click **Add to estimate**.")
else:
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
                    "row","equipment_class","component","task_name","task_code",
                    "base_duration_hr","cost_per_hour","total_cost",
                    "crew_roles","crew_count","notes",
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
            st.experimental_rerun()
    with actB:
        if st.button("↩️ Remove last"):
            if st.session_state.estimate:
                st.session_state.estimate.pop()
                st.experimental_rerun()
    with actC:
        csv_cols = [
            "row","equipment_class","component","task_name","task_code",
            "base_duration_hr","cost_per_hour","total_cost","crew_roles","crew_count","notes",
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
