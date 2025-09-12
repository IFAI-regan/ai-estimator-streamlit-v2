# app.py

import os
import hashlib
from io import StringIO
from typing import Optional, List, Dict, Tuple
import difflib

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

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
      /* Make sidebar a touch tighter */
      section[data-testid="stSidebar"] div[role="radiogroup"],
      section[data-testid="stSidebar"] .stMarkdown { margin-bottom: .5rem; }
      .small-badge {
        padding: 4px 8px; border-radius: 6px; display:inline-block;
        font-size: 12px; background:#EEF8F0; color:#106A2A;
      }
      .small-badge.fail { background:#FDEAEA; color:#8A0E0E; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Config / Secrets
# =========================================================
if "DATABASE_URL" not in st.secrets:
    st.error(
        "Missing secret `DATABASE_URL`.\n\n"
        "In Streamlit → Settings → Secrets add:\n"
        'DATABASE_URL = "postgresql://postgres.<project>:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"'
    )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()
VIEW_FQN = st.secrets.get("VIEW_FQN", "task_norms_view").strip()  # optional override

# =========================================================
# DB connection + resilient SQL runner (auto-reconnect)
# =========================================================
@st.cache_resource
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def run_sql(query: str, params=None, _retry=True) -> pd.DataFrame:
    """Run SQL and auto-reconnect once on OperationalError."""
    try:
        with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params or [])
            if cur.description is None:
                return pd.DataFrame()
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=[d.name for d in cur.description])

    except psycopg2.OperationalError:
        # clear cached connection and retry once
        if _retry:
            try:
                get_conn.clear()
            except Exception:
                pass
            return run_sql(query, params, _retry=False)
        raise

def db_status_badge() -> None:
    try:
        _ = run_sql("select 1 as ok;")
        st.sidebar.markdown('<span class="small-badge">DB connection OK</span>', unsafe_allow_html=True)
    except Exception:
        st.sidebar.markdown('<span class="small-badge fail">DB connection failed</span>', unsafe_allow_html=True)

db_status_badge()  # compact indicator in the sidebar

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

# =========================================================
# Utility: Color chip from text
# =========================================================
def color_from_text(text: str) -> str:
    if not text:
        return "#D9D9D9"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    r = int(150 + (r / 255) * 90)  # 150–240
    g = int(150 + (g / 255) * 90)
    b = int(150 + (b / 255) * 90)
    return f"#{r:02X}{g:02X}{b:02X}"

# =========================================================
# Helper queries for dropdowns
# =========================================================
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

# =========================================================
# Exact lookup + compute costs (also used by batch importer)
# =========================================================
def lookup_exact_task(eq: str, comp: str, task: str) -> dict | None:
    df = run_sql(
        f"""
        SELECT *
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component       = %s
          AND task_name       = %s
        LIMIT 1;
        """,
        [eq, comp, task],
    )
    return df.iloc[0].to_dict() if not df.empty else None

def compute_costs(row: dict, qty: float = 1.0) -> Tuple[float, float]:
    """Return (cost_per_hour, total_cost) given a DB row and optional quantity."""
    rate = row.get("labour_rate")
    if rate in (None, "", "None"):
        rate = row.get("blended_labour_rate")
    cph = pd.to_numeric(rate, errors="coerce")
    dur = pd.to_numeric(row.get("base_duration_hr"), errors="coerce")
    q = pd.to_numeric(qty, errors="coerce")
    if pd.isna(cph): cph = 0.0
    if pd.isna(dur): dur = 0.0
    if pd.isna(q):   q = 1.0
    return float(cph), float(round(cph * dur * q, 2))

# =========================================================
# Estimate helpers
# =========================================================
def add_to_estimate(line: dict) -> bool:
    """Append if not already in estimate (same equipment, component, task)."""
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

# =========================================================
# LEFT: Find a task (sidebar) + Batch import
# =========================================================
st.sidebar.header("Find a task")
equipment = st.sidebar.selectbox(
    "Equipment Class",
    options=list_equipment(),
    index=None,
    key=EQ_KEY,
    placeholder="Type to search… e.g., Stacker Reclaimer",
)
component = st.sidebar.selectbox(
    "Component",
    options=list_components(st.session_state.get(EQ_KEY)),
    index=None,
    key=COMP_KEY,
    placeholder="Select component…",
    disabled=(st.session_state.get(EQ_KEY) is None),
)

task_name = st.sidebar.selectbox(
    "Task Name (exact)",
    options=list_tasks(st.session_state.get(EQ_KEY), st.session_state.get(COMP_KEY)),
    index=None,
    key=TASK_KEY,
    placeholder=("Select equipment & component first"
                 if not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY))
                 else "Select a task…"),
    disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
)

lookup_disabled = not (equipment and component and task_name)

col_btn1, col_btn2 = st.sidebar.columns([1, 1])
with col_btn1:
    do_lookup = st.button("Lookup", type="primary", disabled=lookup_disabled, use_container_width=True)
with col_btn2:
    if st.button("Clear", use_container_width=True):
        st.session_state.last_lookup = None
        st.session_state[TASK_KEY] = None
        st.rerun()

# -------- Batch import (CSV) + fuzzy match (SIDEBAR) --------
def _norm(s: str | None) -> str:
    return (s or "").strip()

@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    """Cache a slim catalog to fuzzy match against; keep it stable across reruns."""
    return run_sql(
        f"""
        SELECT
          equipment_class, component, task_name,
          base_duration_hr, labour_rate, blended_labour_rate,
          task_code, crew_roles, crew_count, notes
        FROM {VIEW_FQN};
        """
    )

with st.sidebar.expander("📦 Batch import (CSV)", expanded=True):
    st.caption("Columns (required): **task**, **equipment**, **component**. Optional: **quantity**.")
    sensitivity = st.slider("Match sensitivity (higher = stricter)", 0, 100, 75,
                            help="Uses fuzzy match on task name; equipment & component are expected to match exactly.")

    uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            raw = pd.DataFrame()

        # Normalize column names
        rename_map = {
            "task": "task",
            "Task": "task",
            "equipment": "equipment",
            "Equipment": "equipment",
            "component": "component",
            "Component": "component",
            "quantity": "quantity",
            "Quantity": "quantity",
            "qty": "quantity",
            "Qty": "quantity",
        }
        raw = raw.rename(columns={c: rename_map.get(c, c) for c in raw.columns})

        missing = [c for c in ["task", "equipment", "component"] if c not in raw.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            cat = load_catalog()
            added = 0
            for _, r in raw.iterrows():
                task_in = _norm(r.get("task"))
                eq_in   = _norm(r.get("equipment"))
                comp_in = _norm(r.get("component"))
                qty_in  = r.get("quantity", 1)

                if not (task_in and eq_in and comp_in):
                    continue

                # Filter catalog by exact eq/component first (fast narrow)
                sub = cat.loc[(cat["equipment_class"] == eq_in) & (cat["component"] == comp_in)]
                if sub.empty:
                    # fall back to whole catalog (looser)
                    sub = cat

                # Fuzzy match only on task_name for simplicity
                sub = sub.assign(
                    _score=sub["task_name"].apply(lambda x: int(100 * difflib.SequenceMatcher(None, task_in.lower(), str(x).lower()).ratio()))
                ).sort_values("_score", ascending=False)

                if sub.empty or int(sub.iloc[0]["_score"]) < sensitivity:
                    # no acceptable match
                    continue

                best = sub.iloc[0]
                matched_task = best["task_name"]
                matched_eq   = best["equipment_class"]
                matched_comp = best["component"]

                # Re-query authoritative row (so we always compute correct costs)
                row = lookup_exact_task(matched_eq, matched_comp, matched_task)
                if not row:
                    continue

                cost_per_hr, total_cost = compute_costs(row, qty=qty_in)

                line = {
                    "equipment_class": row.get("equipment_class"),
                    "component": row.get("component"),
                    "task_name": row.get("task_name"),
                    "task_code": row.get("task_code"),
                    "base_duration_hr": float(pd.to_numeric(row.get("base_duration_hr"), errors="coerce") or 0),
                    "cost_per_hour": cost_per_hr,
                    "total_cost": total_cost,
                    "crew_roles": row.get("crew_roles"),
                    "crew_count": row.get("crew_count"),
                    "notes": row.get("notes"),
                    "component_color": color_from_text(row.get("component")),
                }
                if add_to_estimate(line):
                    added += 1

            if added:
                st.success(f"Imported **{added}** task(s) from CSV.", icon="✅")
            else:
                st.info("No rows imported (no matches >= sensitivity).", icon="ℹ️")

# =========================================================
# MAIN: Lookup & Present
# =========================================================
if do_lookup:
    row = lookup_exact_task(equipment, component, task_name)

    if not row:
        st.warning("No exact match found.")
    else:
        # Compute costs for single task
        cost_per_hr, total_cost = compute_costs(row, qty=1)
        row["cost_per_hour"] = cost_per_hr
        row["total_cost"] = total_cost

        # Persist last_lookup so "Add this task" survives reruns
        st.session_state.last_lookup = row

        # Metrics row
        m1, m2, m3 = st.columns(3)
        dur = float(pd.to_numeric(row.get("base_duration_hr"), errors="coerce") or 0)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cost_per_hr:,.2f}" if cost_per_hr else "—")
        m3.metric("Total cost (per task)", f"${total_cost:,.2f}" if total_cost else "—")

        # Crew
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

        # Clean one-row view table
        tidy_df = pd.DataFrame([row]).copy()
        for col_to_drop in ["blended_labour_rate", "labour_rate", "cost_per_hr"]:
            if col_to_drop in tidy_df.columns:
                tidy_df = tidy_df.drop(columns=[col_to_drop])

        preferred = [
            "task_code", "equipment_class", "component", "task_name",
            "base_duration_hr", "cost_per_hour", "total_cost",
            "crew_roles", "crew_count", "notes", "effective_from", "effective_to",
        ]
        present = [c for c in preferred if c in tidy_df.columns]
        others = [c for c in tidy_df.columns if c not in present]
        tidy_df = tidy_df[present + others]

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
# Add to Estimate (uses persisted last_lookup)
# =========================================================
st.markdown("### Add to estimate")
lr = st.session_state.last_lookup
can_add = bool(lr) and (lr.get("total_cost") is not None)

if st.button("➕ Add this task", disabled=not can_add):
    row = lr
    cost_per_hr = float(pd.to_numeric(row.get("cost_per_hour"), errors="coerce") or 0)
    total_cost  = float(pd.to_numeric(row.get("total_cost"), errors="coerce") or 0)

    line = {
        "equipment_class": row.get("equipment_class"),
        "component": row.get("component"),
        "task_name": row.get("task_name"),
        "task_code": row.get("task_code"),
        "base_duration_hr": float(pd.to_numeric(row.get("base_duration_hr"), errors="coerce") or 0),
        "cost_per_hour": cost_per_hr,
        "total_cost": total_cost,
        "crew_roles": row.get("crew_roles"),
        "crew_count": row.get("crew_count"),
        "notes": row.get("notes"),
        "component_color": color_from_text(row.get("component")),
    }

    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

    # Clear just the Task selectbox safely and refresh
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
    st.caption("No tasks added yet. Lookup a task or use **Batch import (CSV)**, then click **Add to estimate**.")
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
        if "component_color" in display_df.columns:
            cols_to_show.insert(3, "component_color")

        cols_to_show = [c for c in cols_to_show if c in display_df.columns]

        st.dataframe(
            display_df[cols_to_show],
            use_container_width=True,
            column_config=col_config,
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
        if "component_color" in df_est.columns:
            csv_cols.insert(3, "component_color")

        csv_out = df_est[[c for c in csv_cols if c in df_est.columns]].copy()
        buf = StringIO()
        csv_out.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV for Excel",
            data=buf.getvalue(),
            file_name="estimate.csv",
            mime="text/csv",
        )
