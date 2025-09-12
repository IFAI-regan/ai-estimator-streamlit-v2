import os
from io import StringIO
from textwrap import dedent

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# =========================
# Page & Global Styles
# =========================
st.set_page_config(page_title="Phase 1 — Exact Name Recall", layout="wide")

# ---- CSS: layout + look (safe to tweak) ----
st.markdown(
    dedent(
        """
        <style>
          /* General rhythm */
          .block-container { padding-top: 2rem; padding-bottom: 2rem; }

          /* Grid: sidebar + main */
          .app-grid {
            display: grid;
            grid-template-columns: 360px 1fr;
            grid-gap: 24px;
            align-items: start;
          }

          /* Sticky sidebar panel */
          .panel--sidebar {
            position: sticky; top: 92px;
            max-height: calc(100vh - 120px);
            overflow: auto;

            background: #f4f6f8;
            border: 1px solid #e7ecf2;
            border-radius: 12px;
            padding: 18px 16px;
          }

          /* Smaller inner spacing for inputs */
          .panel--sidebar .stSelectbox, 
          .panel--sidebar .stTextInput, 
          .panel--sidebar .stMultiSelect {
            margin-bottom: .35rem;
          }

          /* Row of buttons in the sidebar */
          .btn-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-gap: 10px;
            margin-top: 8px;
          }

          /* Metrics: 3 equal cards */
          .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-gap: 16px;
            margin: 12px 0 24px 0;
          }
          .metric-card {
            background: #f8fafc;
            border: 1px solid #e7ecf2;
            border-radius: 12px;
            padding: 18px 16px;
            min-height: 110px;
          }
          .metric-card h4 {
            margin: 0 0 6px 0; font-weight: 600; font-size: 0.95rem; color: #4a5568;
          }
          .metric-card .big {
            font-size: 1.8rem; font-weight: 700; color: #1f2937;
          }

          /* Divider spacing */
          .pad-top { margin-top: 12px; }

          /* Estimate container */
          .estimate-box {
            background: #fbfcfe;
            border: 1px solid #e7ecf2;
            border-radius: 12px;
            padding: 14px;
          }
        </style>
        """
    ),
    unsafe_allow_html=True,
)

# =========================
# DB connection via secrets
# =========================
if "DATABASE_URL" not in st.secrets or "VIEW_FQN" not in st.secrets:
    st.error(
        "Missing secrets. Please add `DATABASE_URL` and `VIEW_FQN` in **Settings → Secrets**.\n\n"
        "Example:\n"
        'DATABASE_URL = "postgresql://postgres.<project>:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"\n'
        'VIEW_FQN     = "demo_v2.task_norms_view"'
    )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()
VIEW_FQN = st.secrets["VIEW_FQN"].strip()


@st.cache_resource
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def run_sql(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(query, params or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d.name for d in cur.description])


# quick sanity check
ok = True
try:
    _ = run_sql("select 1 as ok;")
except Exception:
    ok = False

# =========================
# State
# =========================
if "estimate" not in st.session_state:
    st.session_state.estimate: list[dict] = []

if "last_lookup" not in st.session_state:
    st.session_state.last_lookup: dict | None = None


# =========================
# Helper queries for dropdowns
# =========================
def list_equipment() -> list[str]:
    df = run_sql(
        f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class;"
    )
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


# =========================
# Page Structure
# =========================
st.title("Phase 1 — Exact Name Recall")
if ok:
    st.success("Database connection OK")
else:
    st.error("Could not connect to database.")
    st.stop()

st.caption(f"Supabase → `{VIEW_FQN}`")

# Grid container
st.markdown('<div class="app-grid">', unsafe_allow_html=True)

# ---------- LEFT: Find a task (sticky sidebar) ----------
st.markdown('<div class="panel--sidebar">', unsafe_allow_html=True)
st.subheader("Find a task")

# Inputs
equipment = st.selectbox(
    "Equipment Class",
    options=list_equipment(),
    index=None,
    placeholder="Type to search… e.g., Stacker Reclaimer",
)
component = st.selectbox(
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
    placeholder=(
        "Select equipment & component first"
        if not (equipment and component)
        else "Select a task…"
    ),
    disabled=not (equipment and component),
)

# Buttons row
lookup_disabled = not (equipment and component and task_name)
col_lookup, col_clear = st.columns(2)
with col_lookup:
    do_lookup = st.button("Lookup", type="primary", use_container_width=True, disabled=lookup_disabled)
with col_clear:
    if st.button("Clear selections", use_container_width=True):
        st.session_state.last_lookup = None
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)  # end sidebar

# ---------- RIGHT: Main content ----------
st.markdown('<div>', unsafe_allow_html=True)

# --- Metrics Row (always shown, values update on lookup) ---
def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
          <h4>{label}</h4>
          <div class="big">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Handle lookup
if do_lookup:
    df = run_sql(
        f"""
        SELECT *
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s;
        """,
        [equipment, component, task_name],
    )

    if df.empty:
        st.warning("No exact match found.")
        st.session_state.last_lookup = None
    else:
        # choose first row
        row = df.iloc[0].to_dict()

        # derive cost_per_hour & total_cost
        if "labour_rate" in df.columns:
            cph = float(df.at[df.index[0], "labour_rate"] or 0)
        elif "blended_labour_rate" in df.columns:
            cph = float(df.at[df.index[0], "blended_labour_rate"] or 0)
        else:
            cph = 0.0

        dur = float(row.get("base_duration_hr") or 0)
        tot = cph * dur if cph and dur else 0.0

        row["cost_per_hour"] = cph
        row["total_cost"] = round(tot, 2)

        st.session_state.last_lookup = row

# --- Metrics render (from last_lookup if available) ---
lr = st.session_state.last_lookup or {}
dur = float(lr.get("base_duration_hr") or 0)
cph = float(lr.get("cost_per_hour") or 0)
tot = float(lr.get("total_cost") or 0)

st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
metric_card("Duration (hr)", f"{dur:.2f}" if dur else "—")
metric_card("Cost per hour", f"${cph:,.2f}" if cph else "—")
metric_card("Total cost (per task)", f"${tot:,.2f}" if tot else "—")
st.markdown("</div>", unsafe_allow_html=True)

# --- Result (per task) table (only if we have a row) ---
if lr:
    st.markdown("#### Result (per task)")
    tidy_df = pd.DataFrame([lr]).copy()

    # Hide raw pricing fields if present
    for col_to_drop in ["blended_labour_rate", "labour_rate"]:
        if col_to_drop in tidy_df.columns:
            tidy_df.drop(columns=[col_to_drop], inplace=True)

    # Nice order
    wanted = [
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
    present = [c for c in wanted if c in tidy_df.columns]
    tidy_df = tidy_df[present + [c for c in tidy_df.columns if c not in present]]

    st.dataframe(
        tidy_df,
        use_container_width=True,
        column_config={
            "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
            "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
            "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            "crew_roles": st.column_config.TextColumn("Crew roles", help="Wrapped for readability", width="medium"),
            "notes": st.column_config.TextColumn("Notes", help="Wrapped for readability", width="large"),
        },
        hide_index=True,
    )

    # Add to estimate
    st.markdown("#### Add to estimate")
    if st.button("➕ Add this task", type="secondary"):
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
        st.session_state.estimate.append(line)
        st.toast("Added to estimate ✅")

# Divider between result and estimate
st.divider()

# --- Estimate Section ---
st.subheader("Estimate")
if not st.session_state.estimate:
    st.caption("No tasks added yet. Lookup a task and click **Add this task**.")
else:
    # Estimate filter
    filter_text = st.text_input(
        "Filter estimate (search across task / equipment / component):",
        placeholder="Type to filter…",
    )

    df_est = pd.DataFrame(st.session_state.estimate)
    if filter_text:
        ft = filter_text.lower()
        mask = (
            df_est["task_name"].astype(str).str.lower().str.contains(ft)
            | df_est["equipment_class"].astype(str).str.lower().str.contains(ft)
            | df_est["component"].astype(str).str.lower().str.contains(ft)
        )
        df_est = df_est[mask]

    # Subtotal
    subtotal = pd.to_numeric(df_est["total_cost"], errors="coerce").sum()

    # Distinct container
    st.markdown('<div class="estimate-box">', unsafe_allow_html=True)
    st.dataframe(
        df_est[
            [
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
            "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
            "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
            "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            "crew_roles": st.column_config.TextColumn("Crew roles", width="medium"),
            "notes": st.column_config.TextColumn("Notes", width="large"),
        },
        hide_index=True,
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🗑️ Clear estimate", use_container_width=True):
            st.session_state.estimate = []
            st.rerun()
    with c2:
        if st.button("↩️ Remove last", use_container_width=True):
            if st.session_state.estimate:
                st.session_state.estimate.pop()
                st.rerun()
    with c3:
        # CSV export
        csv_buf = StringIO()
        df_est.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download CSV for Excel",
            data=csv_buf.getvalue(),
            file_name="estimate.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.metric("Estimate subtotal", f"${subtotal:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)  # end estimate-box

# Close main/right cell and grid
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
