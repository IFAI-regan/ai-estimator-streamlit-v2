import os, pathlib
import streamlit as st

VERSION = "v2.1-sidebar-2025-09-12-22:45"
st.set_page_config(page_title=f"Estimator ({VERSION})", layout="wide")

# Big, unmistakable markers that this file is the one running
st.sidebar.success(f"✅ Loaded code version: {VERSION}")
try:
    st.caption(f"cwd={os.getcwd()}  •  file={pathlib.Path(__file__).resolve()}")
except Exception:
    pass

st.sidebar.success("👋 This is the NEW sidebar layout.")

# app.py
import os
import hashlib
from io import StringIO
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
import difflib
import psycopg2
import psycopg2.extras
import streamlit as st


# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Maintenance Task Lookup", layout="wide")
st.title("Phase 1 — Exact Name Recall")
st.caption(f"Supabase → {st.secrets.get('VIEW_FQN', 'task_norms_view')}")


# =========================================================
# Secrets / DB connect
# =========================================================
if "DATABASE_URL" not in st.secrets:
    st.error(
        "Missing secret `DATABASE_URL`.\n\n"
        "Add it in *Settings → Secrets*.\n"
        "Example:\n"
        'DATABASE_URL="postgresql://postgres.<project>:<PASSWORD>@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"'
    )
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"].strip()
VIEW_FQN = st.secrets.get("VIEW_FQN", "task_norms_view").strip()

@st.cache_resource(show_spinner=False)
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def run_sql(query: str, params=None) -> pd.DataFrame:
    """Run SQL and return a DataFrame (safe with reconnect guidance)."""
    try:
        with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params or [])
            if cur.description is None:
                return pd.DataFrame()
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=[d.name for d in cur.description])
    except psycopg2.OperationalError as e:
        st.error("🔌 Database connection failed. Try **Rerun** (⌘/Ctrl-R) or check your internet.")
        st.caption(f"{type(e).__name__}: {e}")
        st.stop()
    except Exception as e:
        st.error("❌ Could not query database.")
        st.caption(f"{type(e).__name__}: {e}")
        st.stop()

# Small badge (sidebar) to keep it out of the way
try:
    _ = run_sql("select 1 as ok;")
    db_ok = True
except Exception:
    db_ok = False


# =========================================================
# Session state
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate: List[Dict] = []
if "last_lookup" not in st.session_state:
    st.session_state.last_lookup: Optional[Dict] = None

EQ_KEY, COMP_KEY, TASK_KEY = "eq_select", "comp_select", "task_select"


# =========================================================
# Helpers
# =========================================================
def color_from_text(text: str) -> str:
    if not text:
        return "#D9D9D9"
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(150 + (int(h[0:2], 16) / 255) * 90)
    g = int(150 + (int(h[2:4], 16) / 255) * 90)
    b = int(150 + (int(h[4:6], 16) / 255) * 90)
    return f"#{r:02X}{g:02X}{b:02X}"

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
                "row","equipment_class","component","Color","task_name","task_code",
                "Duration (hr)","Cost/hr","Cost/Task","crew_roles","crew_count","notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate).copy()
    # Normalize types
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Pretty columns for display
    df["Duration (hr)"] = df.get("base_duration_hr", 0)
    df["Cost/hr"] = df.get("cost_per_hour", 0)
    df["Cost/Task"] = df.get("total_cost", 0)
    if "component_color" in df.columns:
        df["Color"] = df["component_color"]
    df.insert(0, "row", range(1, len(df) + 1))
    keep = [
        "row","equipment_class","component","Color","task_name","task_code",
        "Duration (hr)","Cost/hr","Cost/Task","crew_roles","crew_count","notes"
    ]
    return df[[c for c in keep if c in df.columns]]


# =========================================================
# Catalog (for dropdowns and fuzzy match)
# =========================================================
@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    # Keep rows + necessary fields for cost calculation and codes
    df = run_sql(
        f"""
        SELECT
            equipment_class, component, task_name,
            COALESCE(labour_rate, blended_labour_rate) AS cost_per_hour,
            base_duration_hr,
            task_code,
            crew_roles, crew_count, notes
        FROM {VIEW_FQN};
        """
    )
    # Deduplicate to one row per (equipment, component, task)
    if not df.empty:
        df = df.sort_values(["equipment_class","component","task_name"]).drop_duplicates(
            subset=["equipment_class","component","task_name"],
            keep="first"
        )
    return df


# =========================================================
# Sidebar — Inputs + Batch Import + DB badge
# =========================================================
with st.sidebar:
    st.subheader("Find a task")

    if db_ok:
        st.success("Database connection OK", icon="✅")
    else:
        st.error("Database connection failed", icon="❌")

    # Equipment
    eq_options = run_sql(
        f"SELECT DISTINCT equipment_class FROM {VIEW_FQN} ORDER BY equipment_class;"
    )["equipment_class"].tolist()
    equipment = st.selectbox(
        "Equipment Class",
        options=eq_options,
        index=None,
        placeholder="Type to search… e.g., Stacker Reclaimer",
        key=EQ_KEY,
    )

    # Component
    comp_options: List[str] = []
    if equipment:
        comp_options = run_sql(
            f"""
            SELECT DISTINCT component
            FROM {VIEW_FQN}
            WHERE equipment_class = %s
            ORDER BY component;
            """,
            [equipment],
        )["component"].tolist()

    component = st.selectbox(
        "Component",
        options=comp_options,
        index=None,
        placeholder="Select component…",
        key=COMP_KEY,
        disabled=(equipment is None),
    )

    # Task
    task_options: List[str] = []
    if equipment and component:
        task_options = run_sql(
            f"""
            SELECT task_name
            FROM {VIEW_FQN}
            WHERE equipment_class = %s AND component = %s
            GROUP BY task_name
            ORDER BY task_name;
            """,
            [equipment, component],
        )["task_name"].tolist()

    task_name = st.selectbox(
        "Task Name (exact)",
        options=task_options,
        index=None,
        key=TASK_KEY,
        placeholder=("Select equipment & component first" if not (equipment and component) else "Select a task…"),
        disabled=not (equipment and component),
    )

    # Actions
    cA, cB = st.columns(2)
    with cA:
        do_lookup = st.button("Lookup", type="primary", disabled=not (equipment and component and task_name))
    with cB:
        if st.button("Clear selections"):
            st.session_state.last_lookup = None
            st.session_state[TASK_KEY] = None
            st.rerun()

    st.markdown("---")
    st.subheader("📦 Batch import (CSV)")
    st.caption("Columns (required): **task**, **equipment**, **component**. *Optional*: **quantity**.")
    threshold = st.slider("Match sensitivity (higher = stricter)", 50, 95, 76, help="Similar to difflib ratio × 100")
    csv_file = st.file_uploader("Drag and drop file here", type=["csv"])

    # Process CSV if present
    if csv_file is not None:
        try:
            up = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            up = pd.DataFrame()

        required = {"task","equipment","component"}
        if not up.empty and required.issubset(set(map(str.lower, up.columns))):
            # normalize column names
            up.columns = [c.lower().strip() for c in up.columns]
            up["quantity"] = pd.to_numeric(up.get("quantity", 1), errors="coerce").fillna(1).astype(int)

            # fuzzy match against catalog
            cat = load_catalog()
            if cat.empty:
                st.error("Catalog is empty.")
            else:
                # build helpers
                def best_match(name: str, candidates: List[str]) -> Tuple[str, float]:
                    if not name or not candidates:
                        return "", 0.0
                    match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.0)
                    if not match:
                        return "", 0.0
                    ratio = difflib.SequenceMatcher(None, name.lower(), match[0].lower()).ratio()
                    return match[0], ratio

                added = 0
                for _, row in up.iterrows():
                    eq_in = str(row["equipment"]).strip()
                    comp_in = str(row["component"]).strip()
                    task_in = str(row["task"]).strip()
                    qty = int(row.get("quantity", 1))

                    # candidates from catalog
                    eq_match, eq_score = best_match(eq_in, sorted(cat["equipment_class"].unique().tolist()))
                    sub = cat[cat["equipment_class"] == eq_match].copy() if eq_match else pd.DataFrame()

                    comp_match, comp_score = best_match(comp_in, sorted(sub["component"].unique().tolist())) if not sub.empty else ("", 0.0)
                    sub2 = sub[sub["component"] == comp_match].copy() if comp_match else pd.DataFrame()

                    task_match, task_score = best_match(task_in, sorted(sub2["task_name"].unique().tolist())) if not sub2.empty else ("", 0.0)

                    scores = [eq_score, comp_score, task_score]
                    if (min(scores) * 100) < threshold:
                        continue

                    hit = sub2[sub2["task_name"] == task_match]
                    if hit.empty:
                        continue

                    r = hit.iloc[0].to_dict()
                    # compute costs
                    cph = float(pd.to_numeric(r.get("cost_per_hour"), errors="coerce") or 0)
                    dur = float(pd.to_numeric(r.get("base_duration_hr"), errors="coerce") or 0)
                    tot = round(cph * dur, 2)
                    line = {
                        "equipment_class": r.get("equipment_class"),
                        "component": r.get("component"),
                        "task_name": r.get("task_name"),
                        "task_code": r.get("task_code"),
                        "base_duration_hr": dur,
                        "cost_per_hour": cph,
                        "total_cost": tot,
                        "crew_roles": r.get("crew_roles"),
                        "crew_count": r.get("crew_count"),
                        "notes": r.get("notes"),
                        "component_color": color_from_text(r.get("component")),
                    }
                    for _ in range(qty):
                        if add_to_estimate(line):
                            added += 1
                st.success(f"Imported {added} task(s) from CSV.", icon="✅")
        else:
            if not up.empty:
                st.error("CSV must include columns: task, equipment, component (optional: quantity).")


# =========================================================
# Main — Lookup & Present
# =========================================================
if do_lookup:
    df = run_sql(
        f"""
        SELECT
            equipment_class, component, task_name, task_code,
            COALESCE(labour_rate, blended_labour_rate) AS cost_per_hour,
            base_duration_hr, crew_roles, crew_count, notes
        FROM {VIEW_FQN}
        WHERE equipment_class = %s AND component = %s AND task_name = %s;
        """,
        [equipment, component, task_name],
    )
    if df.empty:
        st.warning("No exact match found.")
    else:
        row = df.iloc[0].to_dict()
        # compute costs
        cph = float(pd.to_numeric(row.get("cost_per_hour"), errors="coerce") or 0)
        dur = float(pd.to_numeric(row.get("base_duration_hr"), errors="coerce") or 0)
        tot = round(cph * dur, 2)
        row["cost_per_hour"] = cph
        row["total_cost"] = tot

        st.session_state.last_lookup = row

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration (hr)", f"{dur:.2f}" if dur else "—")
        m2.metric("Cost per hour", f"${cph:,.2f}" if cph else "—")
        m3.metric("Total cost (per task)", f"${tot:,.2f}" if tot else "—")

        # Crew
        st.subheader("Crew")
        roles = str(row.get("crew_roles") or "").split("|")
        counts = str(row.get("crew_count") or "").split("|")
        items = []
        for r, c in zip(roles, counts):
            r = r.strip()
            c = (c or "").strip()
            if r:
                items.append(f"- **{r}** × {c or '1'}")
        if items:
            st.markdown("\n".join(items))
        else:
            st.caption("No crew information found.")

        # Present task row
        show = {
            "task_code":"Task code",
            "equipment_class":"Equipment",
            "component":"Component",
            "task_name":"Task",
            "base_duration_hr":"Duration (hr)",
            "cost_per_hour":"Cost/hr",
            "total_cost":"Cost/Task",
            "notes":"Notes",
        }
        tidy = pd.DataFrame([{lbl: row.get(col) for col, lbl in show.items()}])
        st.subheader("Result (per task)")
        st.dataframe(
            tidy,
            use_container_width=True,
            column_config={
                "Duration (hr)": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "Cost/hr":       st.column_config.NumberColumn("Cost/hr",       format="$%.2f"),
                "Cost/Task":     st.column_config.NumberColumn("Cost/Task",     format="$%.2f"),
            },
        )

# Add to estimate
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
        "component_color": color_from_text(lr.get("component")),
    }
    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")
    # Clear task only
    st.session_state.pop(TASK_KEY, None)
    st.rerun()


# =========================================================
# Estimate section
# =========================================================
st.markdown("---")
st.header("Estimate")

df_est = estimate_df()
if df_est.empty:
    st.info("No tasks added yet. Lookup a task and click **Add this task** to estimate.", icon="ℹ️")
else:
    filter_text = st.text_input(
        "Filter estimate (search across task / equipment / component):",
        placeholder="Type to filter…",
    )
    view = df_est.copy()
    if filter_text:
        ft = filter_text.lower()
        view = view[
            view["task_name"].astype(str).str.lower().str.contains(ft)
            | view["equipment_class"].astype(str).str.lower().str.contains(ft)
            | view["component"].astype(str).str.lower().str.contains(ft)
        ]

    subtotal = pd.to_numeric(view.get("Cost/Task", 0), errors="coerce").sum()
    c1, c2 = st.columns([3, 1])

    with c1:
        st.dataframe(
            view,
            use_container_width=True,
            column_config={
                "row":            st.column_config.NumberColumn("#", format="%.0f"),
                "Duration (hr)":  st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "Cost/hr":        st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "Cost/Task":      st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
                "Color":          st.column_config.TextColumn("Color"),  # keep simple & universal
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
        # Save original numeric columns for export
        export_cols = [
            "row","equipment_class","component","task_name","task_code",
            "base_duration_hr","cost_per_hour","total_cost",
            "crew_roles","crew_count","notes"
        ]
        raw = pd.DataFrame(st.session_state.estimate)
        buf = StringIO()
        raw[[c for c in export_cols if c in raw.columns]].to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV for Excel",
            data=buf.getvalue(),
            file_name="estimate.csv",
            mime="text/csv",
        )
