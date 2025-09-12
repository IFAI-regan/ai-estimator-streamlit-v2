# app.py
import os
import hashlib
from io import StringIO
from typing import Optional, List, Dict, Tuple

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# =============================================================================
# APP CONFIG
# =============================================================================
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption("Supabase → schema_v2.task_norms_view")

# ---- HARD-CODE your new schema + view here ----
VIEW_FQN = "schema_v2.task_norms_view"   # <—— your new view

# Small style touch-ups
st.markdown(
    """
    <style>
      .stMetric span { font-size: 14px !important; }
      .stDataFrame { font-size: 14px; }
      /* lighter info panel look */
      .note { background:#eef5ff; border:1px solid #d8e6ff; padding:.8rem 1rem; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# DATABASE
# =============================================================================
if "DATABASE_URL" not in st.secrets:
    st.error(
        "Missing secret `DATABASE_URL`.\n\n"
        "In Streamlit → Settings → Secrets add your Supabase DSN."
    )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()

@st.cache_resource(show_spinner=False)
def get_conn():
    # Pooler is already in your URL; sslmode required by Supabase
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def run_sql(query: str, params=None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params or [])
        if cur.description is None:
            return pd.DataFrame()
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=[d.name for d in cur.description])

# Connection check (badge goes in sidebar so it never overlaps content)
with st.sidebar:
    try:
        _ = run_sql("select 1 as ok;")
        st.success("Database connection OK", icon="✅")
    except Exception as e:
        st.error("Database connection failed", icon="❌")
        st.caption(type(e).__name__)
        st.stop()

# =============================================================================
# SESSION STATE
# =============================================================================
if "estimate" not in st.session_state:
    st.session_state.estimate = []  # type: List[Dict]
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup = None  # type: Optional[Dict]

# widget keys so we can reset task after adding
EQ_KEY = "eq_select"
COMP_KEY = "comp_select"
TASK_KEY = "task_select"

# =============================================================================
# HELPERS
# =============================================================================
def color_from_text(text: str) -> str:
    """Deterministic soft color for a string."""
    if not text:
        return "#D9D9D9"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    r = int(150 + (r / 255) * 90)
    g = int(150 + (g / 255) * 90)
    b = int(150 + (b / 255) * 90)
    return f"#{r:02X}{g:02X}{b:02X}"

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

def estimate_df_from_state() -> pd.DataFrame:
    if not st.session_state.estimate:
        return pd.DataFrame(
            columns=[
                "row", "equipment_class", "component", "task_name", "task_code",
                "base_duration_hr", "cost_per_hour", "total_cost",
                "crew_roles", "crew_count", "notes", "component_color"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate)
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df

# ---- Defensive cost derivation ----
def derive_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has: cost_per_hour, total_cost (if base_duration_hr exists).
    Tries multiple possible rate columns safely.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Known/likely rate column names (in priority order)
    rate_candidates = [
        "labour_rate", "blended_labour_rate", "labor_rate", "blended_labor_rate",
        "cost_per_hr", "hourly_rate", "rate"
    ]

    # pick first present
    rate_col = next((c for c in rate_candidates if c in out.columns), None)

    if rate_col:
        out["cost_per_hour"] = pd.to_numeric(out[rate_col], errors="coerce")
    else:
        out["cost_per_hour"] = pd.NA  # keep column but may be NA

    # duration
    dur_col_candidates = ["base_duration_hr", "duration_hr", "duration"]
    dur_col = next((c for c in dur_col_candidates if c in out.columns), None)

    if dur_col:
        out["base_duration_hr"] = pd.to_numeric(out[dur_col], errors="coerce")
    else:
        # create column for downstream calc even if missing
        if "base_duration_hr" not in out.columns:
            out["base_duration_hr"] = pd.NA

    # compute total if both are numbers
    try:
        out["total_cost"] = (pd.to_numeric(out["cost_per_hour"], errors="coerce") *
                             pd.to_numeric(out["base_duration_hr"], errors="coerce")).round(2)
    except Exception:
        out["total_cost"] = pd.NA

    return out

# =============================================================================
# CATALOG QUERIES (ALL use VIEW_FQN)
# =============================================================================
def list_equipment() -> list[str]:
    df = run_sql(f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class;")
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq: Optional[str]) -> list[str]:
    if not eq:
        return []
    df = run_sql(
        f"SELECT DISTINCT component FROM {VIEW_FQN} WHERE equipment_class = %s ORDER BY component;",
        [eq],
    )
    return df["component"].tolist() if not df.empty else []

def list_tasks(eq: Optional[str], comp: Optional[str]) -> list[str]:
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

# =============================================================================
# LEFT COLUMN: FIND A TASK
# =============================================================================
with st.sidebar:
    st.header("Find a task")

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
        placeholder=("Select equipment & component first" if not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)) else "Select a task…"),
        disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        do_lookup = st.button("Lookup", type="primary", use_container_width=True,
                              disabled=not (equipment and component and task_name))
    with col_btn2:
        if st.button("Clear selections", use_container_width=True):
            st.session_state.last_lookup = None
            st.session_state[TASK_KEY] = None
            st.rerun()

# =============================================================================
# LOOKUP & PRESENT
# =============================================================================
if do_lookup:
    try:
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
    except Exception as e:
        st.error("Could not query database.", icon="❌")
        st.caption(f"{type(e).__name__}: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.warning("No exact match found.")
    else:
        df = derive_cost_columns(df)
        st.session_state.last_lookup = df.iloc[0].to_dict()

# If we have a last lookup, show the metrics + per-task details
if st.session_state.last_lookup:
    row = st.session_state.last_lookup
    dur = float(pd.to_numeric(row.get("base_duration_hr"), errors="coerce") or 0)
    cph = float(pd.to_numeric(row.get("cost_per_hour"), errors="coerce") or 0)
    tot = float(pd.to_numeric(row.get("total_cost"), errors="coerce") or 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
    m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
    m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

    st.subheader("Crew")
    roles = str(row.get("crew_roles", "") or "").split("|")
    counts = str(row.get("crew_count", "") or "").split("|")
    crew_lines = []
    for r, c in zip(roles, counts):
        r = r.strip(); c = c.strip()
        if r:
            crew_lines.append(f"- **{r}** × {c or '1'}")
    if crew_lines:
        st.markdown("\n".join(crew_lines))
    else:
        st.caption("No crew information found.")

    tidy_df = pd.DataFrame([row]).copy()

    # drop raw rate columns if present (tidier)
    for col_to_drop in ["blended_labour_rate", "labour_rate", "cost_per_hr", "labor_rate", "blended_labor_rate", "hourly_rate", "rate"]:
        if col_to_drop in tidy_df.columns:
            tidy_df = tidy_df.drop(columns=[col_to_drop])

    preferred = [
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
    present = [c for c in preferred if c in tidy_df.columns]
    others  = [c for c in tidy_df.columns if c not in present]
    tidy_df = tidy_df[present + others]

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

# =============================================================================
# ADD TO ESTIMATE
# =============================================================================
st.subheader("Add to estimate")
lr = st.session_state.last_lookup
can_add = bool(lr)  # if no totals, we still allow – batch or later recompute can fill in

if st.button("➕ Add this task", disabled=not can_add):
    line = {
        "equipment_class": lr.get("equipment_class"),
        "component": lr.get("component"),
        "task_name": lr.get("task_name"),
        "task_code": lr.get("task_code"),
        "base_duration_hr": float(pd.to_numeric(lr.get("base_duration_hr"), errors="coerce") or 0),
        "cost_per_hour":    float(pd.to_numeric(lr.get("cost_per_hour"), errors="coerce") or 0),
        "total_cost":       float(pd.to_numeric(lr.get("total_cost"), errors="coerce") or 0),
        "crew_roles": lr.get("crew_roles"),
        "crew_count": lr.get("crew_count"),
        "notes": lr.get("notes"),
        "component_color": color_from_text(lr.get("component")),
    }
    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

    # Clear just the task dropdown & refresh
    st.session_state.pop(TASK_KEY, None)
    st.rerun()

# =============================================================================
# ESTIMATE TABLE
# =============================================================================
st.markdown("---")
st.header("Estimate")

df_est = estimate_df_from_state()

if df_est.empty:
    st.markdown('<div class="note">No tasks added yet. Lookup a task and click <b>Add this task to estimate</b>.</div>', unsafe_allow_html=True)
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

    # If cost_per_hour / duration present, keep total_cost coherent in the view
    display_df = derive_cost_columns(display_df)

    subtotal = pd.to_numeric(display_df.get("total_cost", 0), errors="coerce").sum()

    c1, c2 = st.columns([3, 1])
    with c1:
        col_config = {
            "row": st.column_config.NumberColumn("#", format="%.0f"),
            "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
            "cost_per_hour":    st.column_config.NumberColumn("Cost/hr",      format="$%.2f"),
            "total_cost":       st.column_config.NumberColumn("Cost/Task",    format="$%.2f"),
        }
        st.dataframe(
            display_df[
                [c for c in [
                    "row","equipment_class","component","component_color",
                    "task_name","task_code","base_duration_hr","cost_per_hour",
                    "total_cost","crew_roles","crew_count","notes"
                ] if c in display_df.columns]
            ],
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
            "row","equipment_class","component","component_color","task_name","task_code",
            "base_duration_hr","cost_per_hour","total_cost","crew_roles","crew_count","notes",
        ]
        out = df_est[[c for c in csv_cols if c in df_est.columns]].copy()
        buf = StringIO()
        out.to_csv(buf, index=False)
        st.download_button("⬇️ Download CSV for Excel", data=buf.getvalue(),
                           file_name="estimate.csv", mime="text/csv")

# =============================================================================
# BATCH IMPORT + FUZZY MATCH (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.markdown("---")
    with st.expander("📦 Batch import (CSV)", expanded=False):
        st.caption("Columns (required): **task**, **equipment**, **component**. Optional: **quantity**.")

        sensitivity = st.slider("Match sensitivity (higher = stricter)", min_value=50, max_value=95, value=76, step=1)

        uploaded = st.file_uploader("Drag and drop file here", type=["csv"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception:
                st.error("Could not read CSV.")
                raw = pd.DataFrame()

            # normalize expected columns
            def norm(s: str) -> str:
                return (s or "").strip().lower().replace(" ", "").replace("_","")
            rename_map = {}
            for c in raw.columns:
                nc = norm(c)
                if nc in ("task","taskname"):           rename_map[c] = "task"
                elif nc in ("equipment","equipmentclass","equipclass","equip"): rename_map[c] = "equipment"
                elif nc in ("component","comp"):        rename_map[c] = "component"
                elif nc in ("qty","quantity","count"):  rename_map[c] = "quantity"
            raw = raw.rename(columns=rename_map)

            required = {"task","equipment","component"}
            if not required.issubset(set(raw.columns)):
                st.error(f"CSV must include columns: {', '.join(sorted(required))}")
            else:
                # Build a catalog from your view for fuzzy matching
                catalog = run_sql(
                    f"""
                    SELECT DISTINCT equipment_class, component, task_name, task_code
                    FROM {VIEW_FQN};
                    """
                )

                if catalog.empty:
                    st.warning("View is empty; nothing to match.")
                else:
                    # Simple fuzzy by ratio on task/component/equipment
                    from difflib import SequenceMatcher

                    def score(a: str, b: str) -> float:
                        return 100.0 * SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

                    matches: List[Dict] = []
                    strict_cut = sensitivity  # 50..95

                    for _, rec in raw.iterrows():
                        t = str(rec.get("task", "") or "")
                        e = str(rec.get("equipment", "") or "")
                        c = str(rec.get("component", "") or "")
                        qty = int(pd.to_numeric(rec.get("quantity"), errors="coerce") or 1)

                        # best match row in catalog by combined score
                        best_row = None
                        best_score = -1.0
                        for _, crow in catalog.iterrows():
                            s = (
                                0.5 * score(t, crow["task_name"]) +
                                0.3 * score(e, crow["equipment_class"]) +
                                0.2 * score(c, crow["component"])
                            )
                            if s > best_score:
                                best_score = s
                                best_row = crow

                        if best_row is not None and best_score >= strict_cut:
                            # fetch the full row from view to get durations / rates
                            full = run_sql(
                                f"""
                                SELECT *
                                FROM {VIEW_FQN}
                                WHERE equipment_class = %s AND component = %s AND task_name = %s
                                LIMIT 1;
                                """,
                                [best_row["equipment_class"], best_row["component"], best_row["task_name"]],
                            )
                            if not full.empty:
                                full = derive_cost_columns(full)
                                base = full.iloc[0].to_dict()

                                duration = float(pd.to_numeric(base.get("base_duration_hr"), errors="coerce") or 0)
                                rate     = float(pd.to_numeric(base.get("cost_per_hour"), errors="coerce") or 0)

                                line = {
                                    "equipment_class": base.get("equipment_class"),
                                    "component": base.get("component"),
                                    "task_name": base.get("task_name"),
                                    "task_code": base.get("task_code"),
                                    "base_duration_hr": duration * qty,
                                    "cost_per_hour": rate,
                                    "total_cost": (duration * qty) * rate if (duration and rate) else 0.0,
                                    "crew_roles": base.get("crew_roles"),
                                    "crew_count": base.get("crew_count"),
                                    "notes": base.get("notes"),
                                    "component_color": color_from_text(base.get("component")),
                                }
                                add_to_estimate(line)

                    st.success("Imported CSV tasks (matched).")
                    st.rerun()
