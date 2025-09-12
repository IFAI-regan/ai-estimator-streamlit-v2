import os
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# ------------------------------------------------------------------------------
# Secrets / config helpers
# ------------------------------------------------------------------------------
def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret first from st.secrets, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

# Read required config
DATABASE_URL = get_secret("DATABASE_URL")
VIEW_FQN = get_secret("VIEW_FQN")  # e.g. "demo_v2.task_norms_view"

# Validate config early with actionable errors
if not DATABASE_URL:
    st.error(
        "Missing secret `DATABASE_URL`. Add it in **App settings → Secrets**.\n\n"
        'Example:\n\nDATABASE_URL = "postgresql://postgres.<project>:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"'
    )
    st.stop()

if not VIEW_FQN:
    st.error(
        "Missing secret `VIEW_FQN`. Add it in **App settings → Secrets**.\n\n"
        'Example:\n\nVIEW_FQN = "demo_v2.task_norms_view"'
    )
    st.stop()

# ------------------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------------------
def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def run_sql(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query, params or ())
        rows = cur.fetchall()
    return pd.DataFrame(rows)

def check_connection() -> bool:
    try:
        _ = run_sql("SELECT 1;")
        return True
    except Exception:
        return False

# ------------------------------------------------------------------------------
# Data access helpers (use VIEW_FQN everywhere)
# ------------------------------------------------------------------------------
def list_equipment() -> list[str]:
    q = f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class"
    df = run_sql(q)
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq_class: str | None) -> list[str]:
    if not eq_class:
        return []
    q = f"""
        SELECT DISTINCT component
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
        ORDER BY component
    """
    df = run_sql(q, (eq_class,))
    return df["component"].tolist() if not df.empty else []

def list_tasks(eq_class: str | None, component: str | None) -> list[str]:
    if not (eq_class and component):
        return []
    q = f"""
        SELECT DISTINCT task_name
        FROM {VIEW_FQN}
        WHERE equipment_class = %s AND component = %s
        ORDER BY task_name
    """
    df = run_sql(q, (eq_class, component))
    return df["task_name"].tolist() if not df.empty else []

def lookup_task(eq_class: str, component: str, task_name: str) -> dict | None:
    if not (eq_class and component and task_name):
        return None
    q = f"""
        SELECT *
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component       = %s
          AND task_name       = %s
        LIMIT 1
    """
    df = run_sql(q, (eq_class, component, task_name))
    return df.iloc[0].to_dict() if not df.empty else None

# ------------------------------------------------------------------------------
# Session state & small utils
# ------------------------------------------------------------------------------
if "estimate" not in st.session_state:
    st.session_state.estimate = []

def clear_selections():
    for key in ("eq", "comp", "task", "result"):
        if key in st.session_state:
            del st.session_state[key]

# ------------------------------------------------------------------------------
# Page layout & styles
# ------------------------------------------------------------------------------
st.set_page_config(page_title="AI Estimator", layout="wide")

# Title & context
st.title("Phase 1 — Exact Name Recall")
st.caption(f"Supabase → {VIEW_FQN}")

# Connection banner
if check_connection():
    st.success("✅ Database connection OK")
else:
    st.error("❌ Could not connect to database.")
    st.stop()

# Subtle CSS polish
st.markdown(
    """
    <style>
      section.main > div { padding-top: 0 !important; }
      .block-container { padding-top: 1.2rem; }
      .metric-card { background: #f7f8fa; border: 1px solid #eef0f3; border-radius: 12px; padding: 18px; }
      .well { background: #f5f7fa; border: 1px solid #e6e9ef; border-radius: 10px; padding: 14px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Sidebar-like finder (left column) + main workspace (right)
# ------------------------------------------------------------------------------
col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.subheader("Find a task")

    st.selectbox("Equipment Class", options=list_equipment(), index=None, key="eq", placeholder="Type to search… e.g., Stacker Reclaimer")

    st.selectbox(
        "Component",
        options=list_components(st.session_state.get("eq")),
        index=None,
        key="comp",
        placeholder="Select component…",
        disabled=st.session_state.get("eq") is None,
    )

    st.selectbox(
        "Task Name (exact)",
        options=list_tasks(st.session_state.get("eq"), st.session_state.get("comp")),
        index=None,
        key="task",
        placeholder=("Select equipment & component first" if not (st.session_state.get("eq") and st.session_state.get("comp")) else "Select a task…"),
        disabled=not (st.session_state.get("eq") and st.session_state.get("comp")),
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Lookup", use_container_width=True, type="primary", disabled=not (st.session_state.get("eq") and st.session_state.get("comp") and st.session_state.get("task"))):
            st.session_state.result = lookup_task(
                st.session_state.get("eq"),
                st.session_state.get("comp"),
                st.session_state.get("task"),
            )
    with b2:
        st.button("Clear selections", use_container_width=True, on_click=clear_selections)

with col_main:
    # If a lookup result exists, show metrics + details
    if st.session_state.get("result"):
        row = st.session_state["result"]

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Duration (hr)", f"{float(row.get('base_duration_hr', 0)):,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Cost per hour", f"${float(row.get('labour_rate', 0)):,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            total_cost = float(row.get("base_duration_hr", 0)) * float(row.get("labour_rate", 0))
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total cost (per task)", f"${total_cost:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Crew summary
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

        st.divider()

        st.subheader("Result (per task)")
        present_cols = [
            "task_code", "equipment_class", "component", "task_name",
            "base_duration_hr", "labour_rate", "crew_roles", "crew_count", "notes",
            "effective_from", "effective_to",
        ]
        tidy_df = pd.DataFrame([{k: row.get(k) for k in present_cols}])

        st.dataframe(
            tidy_df,
            use_container_width=True,
            column_config={
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "labour_rate": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "crew_roles": st.column_config.TextColumn("Crew (roles)", help="Pipe-separated roles", width="medium"),
                "notes": st.column_config.TextColumn("Notes", width="large"),
            },
        )

        if st.button("➕ Add this task to estimate", use_container_width=True):
            # Compute per-task total for the estimate table
            row_out = dict(row)
            row_out["Cost/Task"] = float(row.get("base_duration_hr", 0)) * float(row.get("labour_rate", 0))
            st.session_state.estimate.append(row_out)
            st.toast("Added to estimate ✅", icon="✅")

    # --- Estimate section ---
    st.divider()
    st.subheader("Estimate")

    if st.session_state.estimate:
        est_df = pd.DataFrame(st.session_state.estimate)

        # Pretty view columns (in a consistent order)
        view_cols = [
            "equipment_class", "component", "task_name", "task_code",
            "base_duration_hr", "labour_rate", "Cost/Task",
            "crew_roles", "crew_count", "notes",
        ]
        view_cols = [c for c in view_cols if c in est_df.columns]

        # Subtotal
        subtotal = pd.to_numeric(est_df.get("Cost/Task", pd.Series(dtype=float)), errors="coerce").sum()

        with st.container():
            st.dataframe(
                est_df[view_cols],
                use_container_width=True,
                column_config={
                    "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                    "labour_rate":     st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                    "Cost/Task":       st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
                    "crew_roles":      st.column_config.TextColumn("Crew (roles)", width="medium"),
                    "notes":           st.column_config.TextColumn("Notes", width="large"),
                },
            )

            st.markdown(f"### Estimate subtotal: **${subtotal:,.2f}**")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🗑️ Clear estimate", use_container_width=True):
                    st.session_state.estimate = []
                    st.toast("Estimate cleared", icon="🗑️")
            with c2:
                if st.button("↩️ Remove last", use_container_width=True):
                    if st.session_state.estimate:
                        st.session_state.estimate.pop()
                        st.toast("Removed last item", icon="↩️")
            with c3:
                csv = est_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV for Excel",
                    data=csv,
                    file_name="estimate.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.info("No tasks added yet. Lookup a task and click **Add this task to estimate**.")
