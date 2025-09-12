# app.py
import os
from io import StringIO

import pandas as pd
import psycopg2
import psycopg2.extras as pe
import streamlit as st

# -------------------- Page & styles --------------------
st.set_page_config(page_title="Phase 1 — Exact Name Recall", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      /* Make the left column look like a sidebar card */
      div[data-testid="stHorizontalBlock"] > div:nth-child(1) {
        background: #f5f7fa;
        padding: 16px 16px 18px;
        border-radius: 12px;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.04);
        min-height: 86vh;
      }
      /* Tighten selects/buttons a bit */
      div[data-baseweb="select"] > div { min-height: 40px; }
      button[kind="secondary"] { color:#334; }
      .metric-val { font-size: 32px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Phase 1 — Exact Name Recall")

# -------------------- Secrets / DB --------------------
# Required secrets in Streamlit Cloud:
# DATABASE_URL = "postgresql://postgres.<project>:<PASSWORD>@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
# VIEW_FQN     = "demo_v2.task_norms_view"
VIEW_FQN = st.secrets.get("VIEW_FQN", "demo_v2.task_norms_view")
DB_URL   = st.secrets.get("DATABASE_URL", "")

if not DB_URL:
    st.error(
        "Missing secret `DATABASE_URL`.\n\n"
        "In app **Settings → Secrets** add something like:\n"
        '`DATABASE_URL = "postgresql://postgres.<project>:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"`'
    )
    st.stop()

@st.cache_resource(show_spinner=False)
def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require")

@st.cache_data(show_spinner=False, ttl=600)
def run_sql(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor(cursor_factory=pe.RealDictCursor) as cur:
        cur.execute(query, params or [])
        rows = cur.fetchall() if cur.description else []
        return pd.DataFrame(rows)

# Connection check
try:
    _ = run_sql("select 1;")
    st.success("Database connection OK")
except Exception as e:
    st.error("Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()

# -------------------- Session state --------------------
if "estimate" not in st.session_state:
    st.session_state.estimate = []

if "last_lookup" not in st.session_state:
    st.session_state.last_lookup = None

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
                "row","equipment_class","component","task_name","task_code",
                "base_duration_hr","cost_per_hour","total_cost",
                "crew_roles","crew_count","notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate).copy()
    for c in ["base_duration_hr","cost_per_hour","total_cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df

# -------------------- Helper lists --------------------
@st.cache_data(ttl=600, show_spinner=False)
def list_equipment() -> list[str]:
    df = run_sql(f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class;")
    return df["equipment_class"].tolist() if not df.empty else []

@st.cache_data(ttl=600, show_spinner=False)
def list_components(eq: str) -> list[str]:
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

@st.cache_data(ttl=600, show_spinner=False)
def list_tasks(eq: str, comp: str) -> list[str]:
    if not (eq and comp):
        return []
    df = run_sql(
        f"""
        SELECT task_name
        FROM {VIEW_FQN}
        WHERE equipment_class = %s AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
        """,
        [eq, comp],
    )
    return df["task_name"].tolist() if not df.empty else []

# -------------------- Layout: two columns --------------------
left, right = st.columns([0.95, 2.05], gap="large")

with left:
    st.header("Find a task", divider=False)

    equipment = st.selectbox(
        "Equipment Class",
        options=list_equipment(),
        index=None,
        placeholder="Type to search… e.g., Stacker Reclaimer",
    )

    component = st.selectbox(
        "Component",
        options=list_components(equipment) if equipment else [],
        index=None,
        disabled=equipment is None,
        placeholder="Select component…",
    )

    task_name = st.selectbox(
        "Task Name (exact)",
        options=list_tasks(equipment, component) if (equipment and component) else [],
        index=None,
        disabled=not (equipment and component),
        placeholder=("Select equipment & component first" if not (equipment and component) else "Select a task…"),
    )

    c1, c2 = st.columns(2)
    with c1:
        lookup_pressed = st.button("Lookup", use_container_width=True, type="primary",
                                   disabled=not (equipment and component and task_name))
    with c2:
        if st.button("Clear selections", use_container_width=True):
            st.session_state.last_lookup = None
            st.rerun()

with right:
    # -------------------- Lookup result area --------------------
    if "lookup_once" not in st.session_state:
        st.session_state.lookup_once = 0  # just to force area creation
    # execute lookup if asked
    if 'lookup_pressed' in locals() and lookup_pressed:
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
            # derive cost_per_hour + total_cost
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
    if row:
        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.caption("Duration (hr)")
            st.markdown(f"<div class='metric-val'>{float(row.get('base_duration_hr') or 0):.2f}</div>", unsafe_allow_html=True)
        with m2:
            st.caption("Cost per hour")
            cph = float(row.get("cost_per_hour") or 0)
            st.markdown(f"<div class='metric-val'>${cph:,.2f}</div>", unsafe_allow_html=True)
        with m3:
            st.caption("Total cost (per task)")
            tot = float(row.get("total_cost") or 0)
            st.markdown(f"<div class='metric-val'>${tot:,.2f}</div>", unsafe_allow_html=True)

        # Crew bullets
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

        # One-row table (tidy)
        tidy = pd.DataFrame([row]).copy()
        for col_to_drop in ["blended_labour_rate", "labour_rate", "id"]:
            if col_to_drop in tidy.columns:
                tidy.drop(columns=[col_to_drop], inplace=True, errors="ignore")

        cols_wanted = [
            "task_code","equipment_class","component","task_name",
            "base_duration_hr","cost_per_hour","total_cost",
            "crew_roles","crew_count","notes","effective_from","effective_to"
        ]
        present = [c for c in cols_wanted if c in tidy.columns]
        tidy = tidy[present + [c for c in tidy.columns if c not in present]]

        st.subheader("Result (per task)")
        st.dataframe(
            tidy,
            use_container_width=True,
            column_config={
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour":   st.column_config.NumberColumn("Cost/hr",     format="$%.2f"),
                "total_cost":      st.column_config.NumberColumn("Cost/Task",   format="$%.2f"),
            },
            hide_index=True,
        )

        st.subheader("Add to estimate")
        if st.button("➕ Add this task"):
            line = {
                "equipment_class": row.get("equipment_class"),
                "component": row.get("component"),
                "task_name": row.get("task_name"),
                "task_code": row.get("task_code"),
                "base_duration_hr": float(row.get("base_duration_hr") or 0),
                "cost_per_hour": float(row.get("cost_per_hour") or 0),
                "total_cost": float(row.get("total_cost") or 0),
                "crew_roles": row.get("crew_roles"),
                "crew_count": row.get("crew_count"),
                "notes": row.get("notes"),
            }
            if add_to_estimate(line):
                st.toast("Added to estimate ✅")
            else:
                st.toast("Already in estimate", icon="ℹ️")

    # -------------------- Estimate cart --------------------
    st.subheader("Estimate")
    df_est = estimate_df()
    if df_est.empty:
        st.caption("No tasks added yet. Lookup a task and click **Add this task**.")
    else:
        filter_text = st.text_input("Filter estimate (search across task / equipment / component):",
                                    placeholder="Type to filter…")
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
                "row":              st.column_config.NumberColumn("#", format="%.0f"),
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour":    st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "total_cost":       st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            },
            hide_index=True,
        )

        cA, cB, cC = st.columns([1,1,2])
        with cA:
            if st.button("🗑️ Clear estimate", use_container_width=True):
                st.session_state.estimate = []
                st.rerun()
        with cB:
            if st.button("↩️ Remove last", use_container_width=True):
                if st.session_state.estimate:
                    st.session_state.estimate.pop()
                    st.rerun()
        with cC:
            buf = StringIO()
            display_df.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Download CSV for Excel",
                data=buf.getvalue(),
                file_name="estimate.csv",
                mime="text/csv",
                use_container_width=True,
            )
