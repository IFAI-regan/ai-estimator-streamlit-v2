# app.py
import os
import hashlib
from typing import Optional, List, Dict

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# =============================================================================
# Page
# =============================================================================
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption("Supabase → prod_v1.task_norms_view")

st.markdown(
    """
    <style>
      .stMetric span { font-size: 14px !important; }
      .small-note { color:#6b7280; font-size:13px; }
      .pad-box { padding: .25rem .5rem; background:#f7f7f9; border-radius:.5rem; }
      .muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# DB
# =============================================================================
if "DATABASE_URL" not in st.secrets:
    st.error(
        "Missing secret `DATABASE_URL`.\n\n"
        "Add it in Streamlit → Settings → Secrets."
    )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()

# Use your NEW production view here
VIEW_FQN = "prod_v1.task_norms_view"

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

# sanity check
try:
    _ = run_sql("select 1 as ok;")
    st.success("✅ Database connection OK")
except Exception as e:
    st.error("❌ Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()


# =============================================================================
# Session state
# =============================================================================
if "estimate" not in st.session_state:
    st.session_state.estimate = []  # type: List[Dict]
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup = None  # type: Optional[Dict]

EQ_KEY = "eq_select"
COMP_KEY = "comp_select"
TASK_KEY = "task_select"


# =============================================================================
# Helpers
# =============================================================================
def color_from_text(text: str) -> str:
    """Soft, deterministic chip color based on text."""
    if not text:
        return "#D9D9D9"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    r = int(150 + (r / 255) * 90)
    g = int(150 + (g / 255) * 90)
    b = int(150 + (b / 255) * 90)
    return f"#{r:02X}{g:02X}{b:02X}"


def list_equipment() -> list[str]:
    df = run_sql(
        f"""
        SELECT DISTINCT equipment_class
        FROM {VIEW_FQN}
        ORDER BY equipment_class;
        """
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
        SELECT DISTINCT task_name
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component = %s
        ORDER BY task_name;
        """,
        [eq, comp],
    )
    return df["task_name"].tolist() if not df.empty else []


def calc_task_rollup(df: pd.DataFrame) -> Dict:
    """
    Given *per-role* rows for a single task (from the normalized view),
    compute task-level metrics.

    Expected columns:
      - base_duration  (numeric)
      - hourly_rate    (numeric)
      - role_count     (int)
      - cost_per_hour  (numeric)   (in the view this is equal to hourly_rate)
      - cost_per_task  (numeric)   (computed in the view)

      - task_code, equipment_class, component, task_name, role_name
    """
    if df.empty:
        return {}

    # robust numeric coercion
    for col in ["base_duration", "hourly_rate", "role_count", "cost_per_hour", "cost_per_task"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # duration (identical across rows)
    duration = float(df["base_duration"].iloc[0]) if "base_duration" in df.columns else 0.0

    # blended rate = sum(hourly_rate * role_count)
    blended_rate = float((df["hourly_rate"] * df["role_count"]).sum()) if {"hourly_rate","role_count"} <= set(df.columns) else 0.0

    # total task cost (sum per-role cost_per_task if present; else compute)
    if "cost_per_task" in df.columns and df["cost_per_task"].notna().any():
        total_cost = float(df["cost_per_task"].sum())
    else:
        total_cost = float((df["hourly_rate"] * df["role_count"] * duration).sum()) if {"hourly_rate","role_count"} <= set(df.columns) else 0.0

    # crew display strings
    roles = df["role_name"].astype(str).tolist() if "role_name" in df.columns else []
    counts = df["role_count"].astype(str).tolist() if "role_count" in df.columns else []
    crew_lines = [f"{r} × {c}" for r, c in zip(roles, counts) if r]

    task_code = df["task_code"].iloc[0] if "task_code" in df.columns else None
    equipment_class = df["equipment_class"].iloc[0] if "equipment_class" in df.columns else None
    component = df["component"].iloc[0] if "component" in df.columns else None
    task_name = df["task_name"].iloc[0] if "task_name" in df.columns else None

    return dict(
        task_code=task_code,
        equipment_class=equipment_class,
        component=component,
        task_name=task_name,
        base_duration_hr=duration,
        blended_rate=blended_rate,
        total_cost=total_cost,
        crew_roles="|".join(roles),
        crew_count="|".join(counts),
    )


def add_to_estimate(line: dict) -> bool:
    """Append if not already present (same eq/component/task_name)."""
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
    if not st.session_state.get("estimate"):
        return pd.DataFrame(
            columns=[
                "row",
                "equipment_class",
                "component",
                "task_name",
                "task_code",
                "base_duration_hr",
                "blended_rate",
                "total_cost",
                "crew_roles",
                "crew_count",
                "notes",
            ]
        )
    df = pd.DataFrame(st.session_state.estimate)
    for c in ["base_duration_hr", "blended_rate", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df


# =============================================================================
# Inputs (left column style)
# =============================================================================
left, right = st.columns([1, 2])

with left:
    st.subheader("Find a task")
    equipment = st.selectbox(
        "Equipment Class",
        options=list_equipment(),
        index=None,
        key=EQ_KEY,
        placeholder="Type to search… e.g., Stacker Reclaimer",
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
        placeholder=(
            "Select equipment & component first"
            if not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY))
            else "Select a task…"
        ),
        disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
    )

    btn1, btn2 = st.columns([1, 1])
    with btn1:
        do_lookup = st.button(
            "Lookup",
            type="primary",
            disabled=not (equipment and component and task_name),
        )
    with btn2:
        if st.button("Clear selections"):
            st.session_state.last_lookup = None
            st.session_state[TASK_KEY] = None
            st.rerun()

with right:
    st.subheader("Estimate")
    if not st.session_state.get("estimate"):
        st.info("No tasks added yet. Lookup a task and click **Add this task to estimate**.")


# =============================================================================
# Lookup
# =============================================================================
if do_lookup:
    df_roles = run_sql(
        f"""
        SELECT *
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s
        ORDER BY role_name, role_count DESC;
        """,
        [equipment, component, task_name],
    )

    if df_roles.empty:
        st.warning("No exact match found.")
    else:
        rollup = calc_task_rollup(df_roles)
        st.session_state.last_lookup = rollup | {"_rows": df_roles.to_dict(orient="records")}

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration (hr)", f"{rollup['base_duration_hr']:.2f}")
        c2.metric("Cost per hour (blended)", f"${rollup['blended_rate']:,.2f}")
        c3.metric("Total cost (per task)", f"${rollup['total_cost']:,.2f}")

        # Crew bullets
        st.subheader("Crew")
        roles = (rollup.get("crew_roles") or "").split("|")
        counts = (rollup.get("crew_count") or "").split("|")
        lines = [f"- **{r}** × {c or '1'}" for r, c in zip(roles, counts) if r]
        st.markdown("\n".join(lines) if lines else "<span class='muted'>No crew data.</span>", unsafe_allow_html=True)

        # Show per-role breakdown
        st.subheader("Result (per task)")
        show_df = df_roles.copy()
        # Pretty columns if present
        cfg = {}
        if "base_duration" in show_df.columns:
            cfg["base_duration"] = st.column_config.NumberColumn("Duration (hr)", format="%.2f")
        if "hourly_rate" in show_df.columns:
            cfg["hourly_rate"] = st.column_config.NumberColumn("Rate/hr", format="$%.2f")
        if "cost_per_task" in show_df.columns:
            cfg["cost_per_task"] = st.column_config.NumberColumn("Cost/Task", format="$%.2f")

        st.dataframe(show_df, use_container_width=True, column_config=cfg)

# =============================================================================
# Add to estimate
# =============================================================================
st.markdown("### Add to estimate")
lr = st.session_state.last_lookup
can_add = bool(lr) and lr.get("total_cost") is not None

if st.button("➕ Add this task to estimate", disabled=not can_add):
    lr_dict = {
        "equipment_class": lr.get("equipment_class"),
        "component": lr.get("component"),
        "task_name": lr.get("task_name"),
        "task_code": lr.get("task_code"),
        "base_duration_hr": float(lr.get("base_duration_hr") or 0),
        "blended_rate": float(lr.get("blended_rate") or 0),
        "total_cost": float(lr.get("total_cost") or 0),
        "crew_roles": lr.get("crew_roles"),
        "crew_count": lr.get("crew_count"),
        "notes": None,
        "component_color": color_from_text(lr.get("component")),
    }

    if add_to_estimate(lr_dict):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

    # Clear task dropdown and refresh
    try:
        st.session_state.pop(TASK_KEY, None)
    except Exception:
        pass
    st.rerun()


# =============================================================================
# Estimate table + actions
# =============================================================================
df_est = estimate_df()
if not df_est.empty:
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
            | display_df["task_code"].astype(str).str.lower().str.contains(ft)
        )
        display_df = display_df[mask]

    subtotal = pd.to_numeric(display_df.get("total_cost", 0), errors="coerce").sum()

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
                    "blended_rate",
                    "total_cost",
                    "crew_roles",
                    "crew_count",
                ]
            ],
            use_container_width=True,
            column_config={
                "row": st.column_config.NumberColumn("#", format="%.0f"),
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "blended_rate": st.column_config.NumberColumn("Cost/hr (blended)", format="$%.2f"),
                "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            },
        )
    with c2:
        st.metric("Estimate subtotal", f"${subtotal:,.2f}")

    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        if st.button("🗑️ Clear estimate"):
            st.session_state.estimate = []
            st.rerun()
    with a2:
        if st.button("↩️ Remove last"):
            if st.session_state.estimate:
                st.session_state.estimate.pop()
                st.rerun()
    with a3:
        csv_cols = [
            "row",
            "equipment_class",
            "component",
            "task_name",
            "task_code",
            "base_duration_hr",
            "blended_rate",
            "total_cost",
            "crew_roles",
            "crew_count",
        ]
        out = display_df[csv_cols].copy()
        buf = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV for Excel", data=buf, file_name="estimate.csv", mime="text/csv")
