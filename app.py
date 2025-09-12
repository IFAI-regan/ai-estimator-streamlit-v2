# app.py
import hashlib
from io import StringIO
from typing import Optional, List, Dict, Tuple

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# Try rapidfuzz for better speed/quality; fall back to difflib if not present
try:
    from rapidfuzz import fuzz
    def str_ratio(a: str, b: str) -> float:
        return float(fuzz.token_set_ratio(a, b))  # 0..100
except Exception:
    from difflib import SequenceMatcher
    def str_ratio(a: str, b: str) -> float:
        return 100.0 * SequenceMatcher(None, a, b).ratio()

# =========================================================
# Page layout & small CSS polish
# =========================================================
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")

st.title("Phase 1 — Exact Name Recall")
VIEW_FQN = st.secrets.get("VIEW_FQN", "task_norms_view")
st.caption(f"Supabase → {VIEW_FQN}")

st.markdown(
    """
    <style>
      .stMetric span { font-size: 14px !important; }
      .stDataFrame { font-size: 14px; }
      /* Make sidebar sections a bit tighter */
      section[data-testid="stSidebar"] .block-container {padding-top: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DB helpers
# =========================================================
if "DATABASE_URL" not in st.secrets:
    st.sidebar.error(
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
    """
    Execute SQL and return a DataFrame. Defensive: if the connection has gone
    stale, it retries once by rebuilding the cached connection.
    """
    try:
        with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params or [])
            if cur.description is None:
                return pd.DataFrame()
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=[d.name for d in cur.description])
    except Exception:
        # Retry once with a fresh connection
        get_conn.clear()  # drop cached connection
        with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params or [])
            if cur.description is None:
                return pd.DataFrame()
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=[d.name for d in cur.description])

# Connection badge (sidebar so it never overlaps main UI)
try:
    _ = run_sql("select 1 as ok;")
    st.sidebar.success("Database connection OK", icon="✅")
except Exception as e:
    st.sidebar.error("Database connection failed", icon="❌")
    st.stop()

# =========================================================
# Session state
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate: List[Dict] = []
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup: Optional[Dict] = None
# Widget keys so we can clear only the Task name
EQ_KEY = "eq_select"
COMP_KEY = "comp_select"
TASK_KEY = "task_select"

# =========================================================
# Utility helpers
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

def add_to_estimate(line: Dict) -> bool:
    """Append if combination not already present."""
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
                "crew_roles", "crew_count", "notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate)
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "row", range(1, len(df) + 1))
    return df

def derive_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cost_per_hour and total_cost in-place using available columns."""
    if "cost_per_hour" not in df.columns:
        df["cost_per_hour"] = None

    # Prefer labour_rate; fall back to blended_labour_rate if present
    if "labour_rate" in df.columns and df["labour_rate"].notna().any():
        df["cost_per_hour"] = pd.to_numeric(df["labour_rate"], errors="coerce")
    elif "blended_labour_rate" in df.columns and df["blended_labour_rate"].notna().any():
        df["cost_per_hour"] = pd.to_numeric(df["blended_labour_rate"], errors="coerce")

    if "base_duration_hr" in df.columns:
        df["base_duration_hr"] = pd.to_numeric(df["base_duration_hr"], errors="coerce")
    else:
        df["base_duration_hr"] = pd.NA

    df["total_cost"] = (df["cost_per_hour"] * df["base_duration_hr"]).round(2)
    return df

# =========================================================
# Catalog for fuzzy matching (SELECT * to avoid UndefinedColumn)
# =========================================================
@st.cache_data(show_spinner=False, ttl=60)
def load_catalog(view_fqn: str) -> pd.DataFrame:
    df = run_sql(f"SELECT * FROM {view_fqn};")
    # Normalize explicit columns if missing
    for col in ["equipment_class", "component", "task_name"]:
        if col not in df.columns:
            df[col] = None
    # Build a normalized key to match against
    def _norm(s: Optional[str]) -> str:
        if pd.isna(s) or s is None:
            return ""
        return " ".join(str(s).strip().lower().split())
    df["_norm_equipment"] = df["equipment_class"].map(_norm)
    df["_norm_component"] = df["component"].map(_norm)
    df["_norm_task"] = df["task_name"].map(_norm)
    df["_norm_join"] = (
        df["_norm_equipment"] + " | " + df["_norm_component"] + " | " + df["_norm_task"]
    )
    return df

CATALOG = load_catalog(VIEW_FQN)

# =========================================================
# Sidebar — Find a task + Batch import
# =========================================================
with st.sidebar:
    st.header("Find a task")

    # Dropdown helpers
    @st.cache_data(show_spinner=False, ttl=30)
    def list_equipment() -> list[str]:
        df = run_sql(f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class;")
        return df["equipment_class"].dropna().tolist() if not df.empty else []

    @st.cache_data(show_spinner=False, ttl=30)
    def list_components(eq: Optional[str]) -> list[str]:
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
        return df["component"].dropna().tolist() if not df.empty else []

    @st.cache_data(show_spinner=False, ttl=30)
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
        return df["task_name"].dropna().tolist() if not df.empty else []

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
        placeholder=("Select equipment & component first"
                     if not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY))
                     else "Select a task…"),
        disabled=not (st.session_state.get(EQ_KEY) and st.session_state.get(COMP_KEY)),
    )

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        lookup_disabled = not (equipment and component and task_name)
        do_lookup = st.button("Lookup", type="primary", disabled=lookup_disabled)
    with c_btn2:
        if st.button("Clear selections"):
            st.session_state.last_lookup = None
            st.session_state[TASK_KEY] = None
            st.rerun()

    st.divider()
    st.subheader("📦 Batch import (CSV)")
    st.caption("Columns (required): **task**, **equipment**, **component**. Optional: **quantity**")

    # Match sensitivity
    sens = st.slider("Match sensitivity (higher = stricter)", min_value=50, max_value=95, value=76)

    # CSV uploader
    up = st.file_uploader("Drag and drop file here", type=["csv"], label_visibility="collapsed")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception:
            st.error("Could not read CSV. Make sure it's a valid CSV file.")
            df_in = None

        if df_in is not None:
            # Normalize expected columns
            cols_lower = {c.lower(): c for c in df_in.columns}
            required = ["task", "equipment", "component"]
            missing = [c for c in required if c not in cols_lower]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                task_col = cols_lower["task"]
                equip_col = cols_lower["equipment"]
                comp_col = cols_lower["component"]
                qty_col = cols_lower.get("quantity", None)

                def _norm(s: Optional[str]) -> str:
                    if pd.isna(s) or s is None:
                        return ""
                    return " ".join(str(s).strip().lower().split())

                imported = 0
                for _, row in df_in.iterrows():
                    q = 1
                    if qty_col and not pd.isna(row.get(qty_col)):
                        try:
                            q = int(row[qty_col])
                        except Exception:
                            q = 1

                    key = _norm(row.get(equip_col)) + " | " + _norm(row.get(comp_col)) + " | " + _norm(row.get(task_col))

                    # Find best match in catalog
                    CATALOG["_tmp_score"] = CATALOG["_norm_join"].map(lambda s: str_ratio(key, s))
                    best = CATALOG.sort_values("_tmp_score", ascending=False).head(1)
                    if best.empty or float(best["_tmp_score"].iloc[0]) < sens:
                        continue  # skip low-confidence match

                    match_row = best.iloc[0].to_dict()

                    # Build line item (derive costs)
                    line = {
                        "equipment_class": match_row.get("equipment_class"),
                        "component": match_row.get("component"),
                        "task_name": match_row.get("task_name"),
                        "task_code": match_row.get("task_code"),
                        "base_duration_hr": match_row.get("base_duration_hr"),
                        "crew_roles": match_row.get("crew_roles"),
                        "crew_count": match_row.get("crew_count"),
                        "notes": match_row.get("notes"),
                    }

                    temp_df = pd.DataFrame([match_row])
                    temp_df = derive_costs(temp_df)
                    line["cost_per_hour"] = float(temp_df["cost_per_hour"].iloc[0] or 0)
                    line["total_cost"] = float(temp_df["total_cost"].iloc[0] or 0)

                    # Add color chip (optional)
                    try:
                        line["component_color"] = color_from_text(line["component"])
                    except Exception:
                        pass

                    # Add quantity copies (simple multiply: duplicate rows)
                    for _i in range(max(q, 1)):
                        if add_to_estimate(line.copy()):
                            imported += 1

                if imported:
                    st.success(f"Imported {imported} task(s) from CSV.", icon="✅")
                else:
                    st.warning("No rows imported (low confidence or no matches).")

# =========================================================
# Lookup & Present (main area)
# =========================================================
if "do_lookup" not in locals():
    do_lookup = False

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
    else:
        df = derive_costs(df)
        st.session_state.last_lookup = df.iloc[0].to_dict()

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

        # One-row tidy table
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
# Add to Estimate
# =========================================================
st.subheader("Add to estimate")
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

    # clear just the task dropdown and rerun
    st.session_state.pop(TASK_KEY, None)
    st.rerun()

# =========================================================
# Estimate section
# =========================================================
st.markdown("---")
st.header("Estimate")

df_est = estimate_df()

if df_est.empty:
    st.caption("No tasks added yet. Lookup a task or use **Batch import**.")
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
            "crew_roles":       st.column_config.TextColumn("Crew roles", help="‘|’ separated"),
            "notes":            st.column_config.TextColumn("Notes"),
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
        if "component_color" in display_df.columns:
            # place color after component if present
            cols_to_show.insert(3, "component_color")
            # Use TextColumn if ColorColumn not available in your build
            if hasattr(st.column_config, "ColorColumn"):
                col_config["component_color"] = st.column_config.ColorColumn("Color")
            else:
                col_config["component_color"] = st.column_config.TextColumn("Color")

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
