import os
import hashlib
from io import StringIO
from typing import Optional, List, Dict, Tuple

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# Try RapidFuzz; fall back to difflib if needed
try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False


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
      /* Subtle zebra striping for dataframes */
      div[data-testid="stHorizontalBlock"] .stDataFrame table tbody tr:nth-child(odd) {
        background-color: #fafafa !important;
      }
      /* Tighter left sidebar spacing */
      section[data-testid="stSidebar"] .block-container { padding-top: 0.75rem; }
      /* Small, muted sidebar footer text */
      .db-ok { color:#127A2E; font-weight:600; }
      .db-bad { color:#B00020; font-weight:600; }
      .soft-card { background:#f6f7f9; border-radius:12px; padding:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Database connection (via DSN in secrets)
# =========================================================
if "DATABASE_URL" not in st.secrets:
    with st.sidebar:
        st.markdown(
            "<div class='db-bad'>Missing secret <code>DATABASE_URL</code>.</div>",
            unsafe_allow_html=True,
        )
        st.info(
            "In Streamlit → Settings → Secrets add:\n\n"
            'DATABASE_URL = "postgresql://postgres.<project>:[PASSWORD]@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"'
        )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()
VIEW_NAME = "task_norms_view"  # your working view


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


# Connection sanity check → show in sidebar (not in the main header)
_db_ok = False
try:
    _ = run_sql("select 1 as ok;")
    _db_ok = True
except Exception:
    _db_ok = False

with st.sidebar:
    if _db_ok:
        st.markdown("<div class='db-ok'>✅ Database connected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='db-bad'>❌ Database connection failed</div>", unsafe_allow_html=True)
        st.stop()


# =========================================================
# Session state (estimate & last lookup; widget keys)
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate = []  # type: List[Dict]
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup = None  # type: Optional[Dict]

# Keys so we can reset just the Task after adding
EQ_KEY = "eq_select"
COMP_KEY = "comp_select"
TASK_KEY = "task_select"


# =========================================================
# Helpers
# =========================================================
def color_from_text(text: str) -> str:
    """Deterministic light color for a given string (for visual grouping)."""
    if not text:
        return "#D9D9D9"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    r = int(150 + (r / 255) * 90)  # 150–240
    g = int(150 + (g / 255) * 90)
    b = int(150 + (b / 255) * 90)
    return f"#{r:02X}{g:02X}{b:02X}"


def add_to_estimate(line: dict) -> bool:
    """Append if not already in estimate (same equipment, component, task, task_code)."""
    for existing in st.session_state.estimate:
        if (
            existing.get("equipment_class") == line.get("equipment_class")
            and existing.get("component") == line.get("component")
            and existing.get("task_name") == line.get("task_name")
            and existing.get("task_code") == line.get("task_code")
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
    # numeric formatting helpers
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # row numbers (1-based)
    df.insert(0, "row", range(1, len(df) + 1))
    return df


# ---------- Catalog for fuzzy  ----------
@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    # pull all columns, then we can return full rows on matches
    df = run_sql(f"SELECT * FROM {VIEW_NAME};")
    if df.empty:
        return df
    # Normalize text fields we match on
    for col in ["equipment_class", "component", "task_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()
    # Build a simple key string for matching
    df["match_key"] = (
        df["equipment_class"].str.lower()
        + " | "
        + df["component"].str.lower()
        + " | "
        + df["task_name"].str.lower()
    )
    return df


def fuzzy_match_row(catalog: pd.DataFrame, eq: str, comp: str, task: str, threshold: int = 82) -> Tuple[Optional[pd.Series], float]:
    """Return best full row match + score, or (None, 0) if below threshold."""
    if catalog.empty:
        return None, 0.0

    # Prepare query string
    query = f"{(eq or '').strip().lower()} | {(comp or '').strip().lower()} | {(task or '').strip().lower()}"

    if _HAS_RAPIDFUZZ:
        best, score, idx = process.extractOne(
            query,
            catalog["match_key"].tolist(),
            scorer=fuzz.token_set_ratio,
        ) if len(catalog) else (None, 0, None)
        if best is None or score < threshold:
            return None, float(score or 0)
        return catalog.iloc[idx], float(score)
    else:
        # difflib fallback
        keys = catalog["match_key"].tolist()
        matches = difflib.get_close_matches(query, keys, n=1, cutoff=threshold/100.0)
        if not matches:
            return None, 0.0
        idx = keys.index(matches[0])
        # difflib returns no score; fake 100
        return catalog.iloc[idx], 100.0


# =========================================================
# Helper queries for dropdown lists (manual lookup)
# =========================================================
def list_equipment() -> list[str]:
    df = run_sql(
        f"SELECT DISTINCT equipment_class FROM {VIEW_NAME} ORDER BY equipment_class;"
    )
    return df["equipment_class"].tolist() if not df.empty else []


def list_components(eq: str | None) -> list[str]:
    if not eq:
        return []
    df = run_sql(
        f"""
        SELECT DISTINCT component
        FROM {VIEW_NAME}
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
        FROM {VIEW_NAME}
        WHERE equipment_class = %s
          AND component = %s
        GROUP BY task_name
        ORDER BY task_name;
        """,
        [eq, comp],
    )
    return df["task_name"].tolist() if not df.empty else []


# =========================================================
# Sidebar: Manual filters + Batch import
# =========================================================
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

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        lookup_disabled = not (equipment and component and task_name)
        do_lookup = st.button("Lookup", type="primary", disabled=lookup_disabled)
    with col_btn2:
        if st.button("Clear selections"):
            st.session_state.last_lookup = None
            st.session_state[TASK_KEY] = None
            st.rerun()

    st.divider()

    with st.expander("📦 Batch import (CSV)", expanded=False):
        st.caption("Columns (required): **task**, **equipment**, **component**.  Optional: **quantity**")
        match_threshold = st.slider("Match sensitivity (higher = stricter)", 70, 95, 82, step=1)
        up = st.file_uploader("Upload CSV", type=["csv"])

        if up is not None:
            try:
                csv_df = pd.read_csv(up).fillna("")
            except Exception:
                st.error("Could not read CSV. Please ensure it's a valid CSV file.")
                csv_df = None

            if csv_df is not None and not csv_df.empty:
                # normalize columns
                colmap = {c.lower().strip(): c for c in csv_df.columns}
                required = ["task", "equipment", "component"]
                missing = [r for r in required if r not in colmap]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                else:
                    # Prepare catalog once
                    catalog = load_catalog()
                    added = 0
                    for _, r in csv_df.iterrows():
                        q = 1
                        if "quantity" in colmap:
                            try:
                                q = int(r[colmap["quantity"]]) if str(r[colmap["quantity"]]).strip() != "" else 1
                            except Exception:
                                q = 1
                        task_i = str(r[colmap["task"]])
                        eq_i = str(r[colmap["equipment"]])
                        comp_i = str(r[colmap["component"]])

                        matched_row, score = fuzzy_match_row(catalog, eq_i, comp_i, task_i, threshold=match_threshold)
                        if matched_row is None:
                            st.warning(f"⚠️ No match: {eq_i} / {comp_i} / {task_i}")
                            continue

                        # Build estimate line from the FULL matched row
                        line = matched_row.to_dict()
                        # prefer cost_per_hour → if your view has only labour_rate, derive
                        if "cost_per_hour" not in line or line.get("cost_per_hour") in (None, ""):
                            if "labour_rate" in line:
                                line["cost_per_hour"] = float(line.get("labour_rate") or 0)
                        # compute total_cost if base_duration_hr exists
                        if "base_duration_hr" in line:
                            try:
                                dur = float(line.get("base_duration_hr") or 0)
                                cph = float(line.get("cost_per_hour") or 0)
                                line["total_cost"] = round(cph * dur, 2)
                            except Exception:
                                line["total_cost"] = None

                        # optional color chip
                        line["component_color"] = color_from_text(line.get("component", ""))

                        # annotate the import
                        prev_notes = str(line.get("notes") or "")
                        sep = " | " if prev_notes else ""
                        line["notes"] = f"{prev_notes}{sep}Imported (score={int(score)})"

                        # repeat per quantity (so subtotal remains additive)
                        for _i in range(max(q, 1)):
                            if add_to_estimate(line.copy()):
                                added += 1

                    st.success(f"✅ Imported {added} task(s) from CSV.")

# =========================================================
# Lookup & Present (manual)
# =========================================================
if 'do_lookup' not in locals():
    do_lookup = False

if do_lookup:
    df = run_sql(
        f"""
        SELECT *
        FROM {VIEW_NAME}
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

        # Save first row as last_lookup (so Add button survives reruns)
        st.session_state.last_lookup = df.iloc[0].to_dict()

        # Metrics in a row
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
        others = [c for c in tidy_df.columns if c not in present]
        final_cols = present + others
        tidy_df = tidy_df[final_cols]

        st.subheader("Result (per task)")
        st.dataframe(
            tidy_df,
            use_container_width=True,
            column_config={
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour":   st.column_config.NumberColumn("Cost/hr",      format="$%.2f"),
                "total_cost":      st.column_config.NumberColumn("Cost/Task",    format="$%.2f"),
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

    # Add a soft color chip for readability
    try:
        line["component_color"] = color_from_text(line["component"])
    except Exception:
        pass

    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

    # clear just the Task selectbox and refresh the UI
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
    st.caption("No tasks added yet. Lookup a task or use **Batch import** in the sidebar.")
else:
    filter_text = st.text_input("Filter estimate (search across task / equipment / component):", placeholder="Type to filter…")

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
        csv_out = df_est[[c for c in csv_cols if c in df_est.columns]].copy()
        buf = StringIO()
        csv_out.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV for Excel",
            data=buf.getvalue(),
            file_name="estimate.csv",
            mime="text/csv",
        )
