import os
import hashlib
from io import StringIO
from typing import Optional, List, Dict
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
      .muted { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Database connection (via DSN in secrets)
# =========================================================
if "DATABASE_URL" not in st.secrets:
    st.error(
        "Missing secret `DATABASE_URL`.\n\n"
        "In Streamlit → Settings → Secrets add:\n"
        'DATABASE_URL = "postgresql://postgres.<project>:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"'
    )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()

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

# connection sanity check
try:
    _ = run_sql("select 1 as ok;")
    st.success("✅ Database connection OK")
except Exception as e:
    st.error("❌ Could not connect to database.")
    st.caption(type(e).__name__)
    st.stop()

# =========================================================
# Session state (estimate & last lookup; widget keys)
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate: List[Dict] = []
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup: Optional[Dict] = None

# Keys so we can reset just the Task after adding
EQ_KEY = "eq_select"
COMP_KEY = "comp_select"
TASK_KEY = "task_select"

# =========================================================
# Helpers
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

def color_from_text(text: str) -> str:
    """Deterministic soft color from a string (for chips)."""
    if not text:
        return "#D9D9D9"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    r = int(150 + (r / 255) * 90)
    g = int(150 + (g / 255) * 90)
    b = int(150 + (b / 255) * 90)
    return f"#{r:02X}{g:02X}{b:02X}"

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

# --- Catalog (defensive) for fuzzy matching & validation ---
@st.cache_resource(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    """
    Pull a minimal, stable set of columns from task_norms_view
    and compute a cost_per_hour + search_key for fuzzy match.
    """
    df = run_sql(
        """
        SELECT
            equipment_class,
            component,
            task_name,
            -- optional columns may not exist — select NULLs safely via COALESCE on literals
            CAST(NULL AS TEXT) AS task_code,
            CAST(NULL AS NUMERIC) AS base_duration_hr,
            CAST(NULL AS NUMERIC) AS labour_rate,
            CAST(NULL AS NUMERIC) AS blended_labour_rate,
            CAST(NULL AS TEXT) AS crew_roles,
            CAST(NULL AS TEXT) AS crew_count,
            CAST(NULL AS TEXT) AS notes
        FROM task_norms_view
        """
    )
    # Prefer real columns if they exist (safe overwrite)
    try:
        # Re-fetch with full columns if present
        df_full = run_sql("SELECT * FROM task_norms_view LIMIT 0;")
        cols = list(df_full.columns)
        # Pull actual values where possible
        wanted = [c for c in [
            "equipment_class","component","task_name","task_code",
            "base_duration_hr","labour_rate","blended_labour_rate",
            "crew_roles","crew_count","notes"
        ] if c in cols]
        if wanted:
            df = run_sql(f"SELECT {', '.join(wanted)} FROM task_norms_view")
    except Exception:
        pass

    if df.empty:
        return df

    # Compute cost_per_hour
    cost_col = None
    if "labour_rate" in df.columns:
        cost_col = "labour_rate"
    if "blended_labour_rate" in df.columns and df["blended_labour_rate"].notna().any():
        cost_col = "blended_labour_rate"
    if cost_col:
        df["cost_per_hour"] = pd.to_numeric(df[cost_col], errors="coerce")
    else:
        df["cost_per_hour"] = None

    # Base duration
    if "base_duration_hr" in df.columns:
        df["base_duration_hr"] = pd.to_numeric(df["base_duration_hr"], errors="coerce")
    else:
        df["base_duration_hr"] = None

    # Pre-compute search key
    def _key(r):
        return " ".join(
            str(x).strip().lower()
            for x in [r.get("equipment_class"), r.get("component"), r.get("task_name")]
            if pd.notna(x) and str(x).strip()
        )
    df["search_key"] = df.apply(_key, axis=1)
    return df

CATALOG = load_catalog()

# =========================================================
# Helper queries for dropdown lists
# =========================================================
def list_equipment() -> list[str]:
    df = run_sql(
        "SELECT DISTINCT equipment_class FROM task_norms_view ORDER BY equipment_class;"
    )
    return df["equipment_class"].tolist() if not df.empty else []

def list_components(eq: str | None) -> list[str]:
    if not eq:
        return []
    df = run_sql(
        """
        SELECT DISTINCT component
        FROM task_norms_view
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
        """
        SELECT task_name
        FROM task_norms_view
        WHERE equipment_class = %s
          AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
        """,
        [eq, comp],
    )
    return df["task_name"].tolist() if not df.empty else []

# =========================================================
# SIDEBAR — Lookup + Batch Import
# =========================================================
with st.sidebar:
    st.header("Find a task")
    equipment = st.selectbox(
        "Equipment Class",
        options=list_equipment(),
        index=None,
        key=EQ_KEY,
        placeholder="Type to search… e.g., Haul Truck 785D",
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
        placeholder=("Select equipment & component first"
                     if not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY))
                     else "Select a task…"),
        disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
    )
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        do_lookup = st.button("Lookup", type="primary",
                              disabled=not (equipment and component and task_name))
    with col_sb2:
        if st.button("Clear selections"):
            st.session_state.last_lookup = None
            st.session_state[TASK_KEY] = None
            st.rerun()

    st.divider()
    st.subheader("📤 Batch import (CSV)")
    st.caption("Columns: **task**, **equipment**, **component** (optional: **quantity**)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    match_sensitivity = st.slider("Match sensitivity", 0.0, 1.0, 0.82, 0.01,
                                  help="Higher = stricter matching")
    run_match = st.button("Match now", disabled=(uploaded is None))

# =========================================================
# Lookup & Present
# =========================================================
if do_lookup:
    df = run_sql(
        """
        SELECT *
        FROM task_norms_view
        WHERE equipment_class = %s
          AND component = %s
          AND task_name = %s;
        """,
        [equipment, component, task_name],
    )

    if df.empty:
        st.warning("No exact match found.")
    else:
        # Derive cost_per_hour from available column(s)
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

        # Metrics
        row = st.session_state.last_lookup
        dur = float(row.get("base_duration_hr") or 0)
        cph = float(row.get("cost_per_hour") or 0)
        tot = float(row.get("total_cost") or 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
        m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

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

        # Tidy one-row table (hide raw rate cols)
        tidy_df = pd.DataFrame([row]).copy()
        for col_to_drop in ["blended_labour_rate", "labour_rate", "cost_per_hr"]:
            if col_to_drop in tidy_df.columns:
                tidy_df = tidy_df.drop(columns=[col_to_drop])

        preferred = [
            "task_code","equipment_class","component","task_name",
            "base_duration_hr","cost_per_hour","total_cost",
            "crew_roles","crew_count","notes","effective_from","effective_to",
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
    try:
        line["component_color"] = color_from_text(line["component"])
    except Exception:
        pass

    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

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
            "row","equipment_class","component","task_name","task_code",
            "base_duration_hr","cost_per_hour","total_cost",
            "crew_roles","crew_count","notes",
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
            "row","equipment_class","component","task_name","task_code",
            "base_duration_hr","cost_per_hour","total_cost",
            "crew_roles","crew_count","notes",
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

# =========================================================
# IMPORT RESULTS (from sidebar uploader)
# =========================================================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ["task", "task_name", "name"]:
            rename_map[col] = "task"
        elif c in ["equipment", "equipment_class", "equipmentclass"]:
            rename_map[col] = "equipment"
        elif c in ["component", "subsystem"]:
            rename_map[col] = "component"
        elif c in ["qty", "quantity", "count"]:
            rename_map[col] = "quantity"
    df = df.rename(columns=rename_map)
    for need in ["task", "equipment", "component"]:
        if need not in df.columns:
            df[need] = None
    if "quantity" not in df.columns:
        df["quantity"] = 1
    return df

def _key_from_row(equipment: str, component: str, task: str) -> str:
    parts = [equipment, component, task]
    parts = [str(p).strip().lower() for p in parts if p is not None and str(p).strip()]
    return " ".join(parts)

def _best_match(input_key: str, catalog_keys: List[str]) -> tuple[str, float]:
    if not catalog_keys:
        return "", 0.0
    # difflib gives a ratio (0..1)
    best = ""
    best_score = 0.0
    # get_close_matches lets us shortlist quickly
    shortlist = difflib.get_close_matches(input_key, catalog_keys, n=5, cutoff=0)
    shortlist = shortlist if shortlist else catalog_keys[:50]
    for cand in shortlist:
        score = difflib.SequenceMatcher(None, input_key, cand).ratio()
        if score > best_score:
            best, best_score = cand, score
    return best, best_score

# Run matching (if asked)
if run_match and uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        raw = _normalize_cols(raw)
        if CATALOG.empty:
            st.warning("Catalog is empty; cannot match.")
        else:
            # Build a dict from search_key -> full row
            catalog = CATALOG.copy()
            catalog_keys = catalog["search_key"].tolist()

            match_rows = []
            for _, r in raw.iterrows():
                in_key = _key_from_row(r.get("equipment"), r.get("component"), r.get("task"))
                best_key, score = _best_match(in_key, catalog_keys)
                if best_key:
                    cat_row = catalog.loc[catalog["search_key"] == best_key].iloc[0].to_dict()
                else:
                    cat_row = {col: None for col in catalog.columns}

                # compute cost & defaults
                base_dur = float(cat_row.get("base_duration_hr") or 0)
                cph = float(cat_row.get("cost_per_hour") or 0)
                total = round(base_dur * cph, 2) if base_dur and cph else None

                match_rows.append({
                    "add": True if score >= match_sensitivity else False,
                    "input_equipment": r.get("equipment"),
                    "input_component": r.get("component"),
                    "input_task": r.get("task"),
                    "quantity": int(r.get("quantity") or 1),
                    "match_equipment": cat_row.get("equipment_class"),
                    "match_component": cat_row.get("component"),
                    "match_task": cat_row.get("task_name"),
                    "base_duration_hr": base_dur,
                    "cost_per_hour": cph,
                    "total_cost": total,
                    "crew_roles": cat_row.get("crew_roles"),
                    "crew_count": cat_row.get("crew_count"),
                    "notes": cat_row.get("notes"),
                    "score": round(score, 3),
                })

            st.session_state["import_matches"] = pd.DataFrame(match_rows)
            st.success(f"Matched {len(match_rows)} row(s). See **Import results** below.")
    except Exception as e:
        st.error("Could not read or match the CSV.")
        st.caption(type(e).__name__)

# Show matches (if any) and allow adding to estimate
if "import_matches" in st.session_state and isinstance(st.session_state["import_matches"], pd.DataFrame):
    matches_df = st.session_state["import_matches"].copy()
    if not matches_df.empty:
        st.subheader("Import results")
        st.caption("Review matches → uncheck any that look wrong → add selected to the estimate.")

        edited = st.data_editor(
            matches_df,
            use_container_width=True,
            column_config={
                "add": st.column_config.CheckboxColumn("Add?", default=True),
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
                "score": st.column_config.NumberColumn("Match", format="%.3f"),
            },
            hide_index=True,
        )

        if st.button("➕ Add selected matches"):
            added = 0
            for _, r in edited.iterrows():
                if not r.get("add"):
                    continue
                line = {
                    "equipment_class": r.get("match_equipment"),
                    "component": r.get("match_component"),
                    "task_name": r.get("match_task"),
                    "task_code": None,
                    "base_duration_hr": float(r.get("base_duration_hr") or 0),
                    "cost_per_hour": float(r.get("cost_per_hour") or 0),
                    "total_cost": float(r.get("total_cost") or 0),
                    "crew_roles": r.get("crew_roles"),
                    "crew_count": r.get("crew_count"),
                    "notes": r.get("notes"),
                    "component_color": color_from_text(str(r.get("match_component") or "")),
                }
                # Repeat according to quantity, if >1
                qty = int(r.get("quantity") or 1)
                for _i in range(max(1, qty)):
                    if add_to_estimate(line):
                        added += 1
            st.toast(f"Added {added} line(s) to estimate", icon="✅")
            # Clear matches after adding
            st.session_state.pop("import_matches", None)
            st.rerun()
