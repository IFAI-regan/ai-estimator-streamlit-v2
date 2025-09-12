import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os

# --- DB connection helpers ---
def get_conn():
    return psycopg2.connect(
        os.environ["DATABASE_URL"],
        cursor_factory=RealDictCursor
    )

def run_sql(query, params=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            rows = cur.fetchall()
    return pd.DataFrame(rows)

# --- List dropdown helpers ---
def list_equipment():
    q = f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class"
    return run_sql(q)["equipment_class"].tolist()

def list_components(eq_class):
    if not eq_class:
        return []
    q = f"""
        SELECT DISTINCT component 
        FROM {VIEW_FQN} 
        WHERE equipment_class = %s
        ORDER BY component
    """
    return run_sql(q, (eq_class,))["component"].tolist()

def list_tasks(eq_class, component):
    if not eq_class or not component:
        return []
    q = f"""
        SELECT DISTINCT task_name 
        FROM {VIEW_FQN}
        WHERE equipment_class = %s AND component = %s
        ORDER BY task_name
    """
    return run_sql(q, (eq_class, component))["task_name"].tolist()

def lookup_task(eq_class, component, task_name):
    if not (eq_class and component and task_name):
        return None
    q = f"""
        SELECT *
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s
    """
    df = run_sql(q, (eq_class, component, task_name))
    return df.iloc[0].to_dict() if not df.empty else None

def check_connection():
    try:
        run_sql("SELECT 1;")
        return True
    except Exception:
        return False

def clear_selections():
    for key in ["eq", "comp", "task", "result"]:
        if key in st.session_state:
            del st.session_state[key]

# --- Init session state ---
if "estimate" not in st.session_state:
    st.session_state["estimate"] = []

# --- Page layout ---
st.set_page_config(page_title="AI Estimator", layout="wide")

st.title("Phase 1 — Exact Name Recall")
st.caption(f"Supabase → {os.environ.get('VIEW_FQN')}")

# DB check
if check_connection():
    st.success("✅ Database connection OK")
else:
    st.error("❌ Could not connect to database.")

# Layout: sidebar 25%, main 75%
col_sidebar, col_main = st.columns([1, 3])

# --- Sidebar ---
with col_sidebar:
    st.subheader("Find a task")

    st.selectbox("Equipment Class", options=list_equipment(), key="eq")
    st.selectbox("Component", options=list_components(st.session_state.get("eq")), key="comp")
    st.selectbox(
        "Task Name (exact)",
        options=list_tasks(st.session_state.get("eq"), st.session_state.get("comp")),
        key="task"
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("Lookup", use_container_width=True, type="primary"):
            st.session_state["result"] = lookup_task(
                st.session_state.get("eq"),
                st.session_state.get("comp"),
                st.session_state.get("task"),
            )
    with bcol2:
        st.button("Clear selections", use_container_width=True, on_click=lambda: clear_selections())

# --- Main content ---
with col_main:
    if "result" in st.session_state and st.session_state["result"] is not None:
        result = st.session_state["result"]

        # Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{result['base_duration_hr']:.2f}")
        m2.metric("Cost per hour", f"${result['labour_rate']:.2f}")
        total_cost = result['base_duration_hr'] * result['labour_rate']
        m3.metric("Total cost (per task)", f"${total_cost:,.2f}")

        # Crew info
        st.markdown("### Crew")
        crew_roles = result.get("crew_roles", "").split("|")
        crew_count = result.get("crew_count", "").split("|")
        for role, count in zip(crew_roles, crew_count):
            st.write(f"- **{role}** × {count}")

        # Task results
        st.divider()
        st.markdown("### Result (per task)")
        task_df = pd.DataFrame([result])
        st.dataframe(task_df, use_container_width=True)

        # Add to estimate
        if st.button("➕ Add this task to estimate", use_container_width=True):
            st.session_state["estimate"].append(result)

    # --- Estimate section ---
    st.divider()
    st.subheader("Estimate")

    if st.session_state["estimate"]:
        est_df = pd.DataFrame(st.session_state["estimate"])
        est_df["Cost/Task"] = est_df["base_duration_hr"] * est_df["labour_rate"]

        # subtotal
        subtotal = est_df["Cost/Task"].sum()

        st.dataframe(
            est_df[
                [
                    "equipment_class",
                    "component",
                    "task_name",
                    "task_code",
                    "base_duration_hr",
                    "labour_rate",
                    "Cost/Task",
                    "crew_roles",
                    "crew_count",
                    "notes",
                ]
            ],
            use_container_width=True,
        )

        st.markdown(f"### Estimate subtotal: **${subtotal:,.2f}**")

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Clear estimate"):
                st.session_state["estimate"] = []
        with b2:
            if st.button("Remove last"):
                if st.session_state["estimate"]:
                    st.session_state["estimate"].pop()
        with b3:
            csv = est_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV for Excel", csv, "estimate.csv", "text/csv")
    else:
        st.info("No tasks added yet. Lookup a task and click **Add this task**.")
