# app.py
import os
from io import StringIO

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

# =========================================================
# Page config & Styles (2-pane shell with sticky left rail)
# =========================================================
st.set_page_config(page_title="Phase 1 — Exact Name Recall", layout="wide")

st.markdown(
    """
    <style>
      :root{
        --rail-w: 280px;
        --rail-bg: #f3f4f6;  /* tailwind: gray-100 */
        --rail-border: #e5e7eb;
      }

      /* 2-column shell */
      .app-shell{
        display: grid;
        grid-template-columns: var(--rail-w) 1fr;
        gap: 24px;
      }

      /* Left rail */
      .left-rail{
        position: sticky;
        top: 64px;               /* under the Streamlit header */
        height: calc(100vh - 72px);
        overflow: auto;
        background: var(--rail-bg);
        border-right: 1px solid var(--rail-border);
        padding: 18px 14px;
        border-radius: 8px;
      }
      .left-rail h3{
        margin: 0 0 12px 0; font-weight: 600;
      }
      .left-rail .stSelectbox,
      .left-rail .stTextInput,
      .left-rail .stNumberInput{
        margin-bottom: 10px;
      }
      .left-rail .stButton>button{
        width: 100%;
        margin-top: 6px;
      }

      /* Right canvas */
      .right-canvas{
        padding-right: 8px;
      }

      /* Section spacing and cards */
      .section{
        margin: 10px 0 20px 0;
      }
      .card-row{
        display: grid;
        grid-template-columns: repeat(3, minmax(160px, 1fr));
        gap: 16px;
      }
      .metric-card{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
      }
      .metric-card h5{ margin: 0 0 6px 0; font-weight: 600; color:#374151; }
      .metric-card .val{ font-size: 20px; font-weight: 700; }

      /* Compact dataframes */
      .stDataFrame{ font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Phase 1 — Exact Name Recall")
st.caption(f"Supabase → {st.secrets.get('VIEW_FQN', 'task_norms_view?')}")

# =========================================================
# DB connection helpers
# =========================================================
DATABASE_URL = st.secrets["DATABASE_URL"].strip()
VIEW_FQN     = st.secrets["VIEW_FQN"].strip()  # fully-qualified view name

@st.cache_resource
def get_conn():
    # Use pooled DSN and SSL
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
# Session state (estimate & last lookup)
# =========================================================
if "estimate" not in st.session_state:
    st.session_state.estimate: list[dict] = []

if "last_lookup" not in st.session_state:
    st.session_state.last_lookup: dict | None = None

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
                "crew_roles", "crew_count", "notes"
            ]
        )
    df = pd.DataFrame(st.session_state.estimate)
    # numeric cast
    for c in ["base_duration_hr", "cost_per_hour", "total_cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # row index (1-based)
    df.insert(0, "row", range(1, len(df) + 1))
    return df

def render_estimate_table():
    st.markdown("---")
    st.header("Estimate")

    df_est = estimate_df()
    if df_est.empty:
        st.caption("No tasks added yet. Lookup a task and click **Add this task**.")
        return

    filter_text = st.text_input(
        "Filter estimate (search across task / equipment / component):",
        placeholder="Type to filter…"
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

    subtotal = pd.to_numeric(display_df["total_cost"], errors="coerce").sum()

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
                    "cost_per_hour",
                    "total_cost",
                    "crew_roles",
                    "crew_count",
                    "notes",
                ]
            ],
            use_container_width=True,
            column_config={
                "row": st.column_config.NumberColumn("#", format="%.0f"),
                "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
                "cost_per_hour": st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
                "total_cost": st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
            },
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
        # CSV export (Excel-ready)
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
        csv_out = display_df[csv_cols] if set(csv_cols).issubset(display_df.columns) else display_df
        buf = StringIO()
        csv_out.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download CSV for Excel",
            data=buf.getvalue(),
            file_name="estimate.csv",
            mime="text/csv",
        )

# =========================================================
# Helper queries for dropdown lists
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
# 2-pane Layout (Left rail + Right canvas)
# =========================================================
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

# ===== Left rail =====
st.markdown('<div class="left-rail">', unsafe_allow_html=True)
st.markdown("### Find a task")

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
    placeholder="Select component…",
    disabled=equipment is None,
)

task_name = st.selectbox(
    "Task Name (exact)",
    options=list_tasks(equipment, component) if (equipment and component) else [],
    index=None,
    placeholder=("Select equipment & component first" if not (equipment and component) else "Select a task…"),
    disabled=not (equipment and component),
)

c1, c2 = st.columns(2)
with c1:
    lookup_disabled = not (equipment and component and task_name)
    do_lookup = st.button("Lookup", type="primary", disabled=lookup_disabled)
with c2:
    if st.button("Clear selections"):
        st.session_state.last_lookup = None
        st.rerun()

# Optional: keyboard shortcuts for Lookup (Enter) and Clear (Esc)
st.markdown(
    """
    <script>
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter'){
          const btns = Array.from(document.querySelectorAll('button'));
          const lookup = btns.find(b => b.innerText.trim().toLowerCase() === 'lookup');
          if (lookup && !lookup.disabled){ lookup.click(); }
        }
        if (e.key === 'Escape'){
          const btns = Array.from(document.querySelectorAll('button'));
          const clearBtn = btns.find(b => b.innerText.trim().toLowerCase() === 'clear selections');
          if (clearBtn){ clearBtn.click(); }
        }
      });
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)  # end left-rail

# ===== Right canvas =====
st.markdown('<div class="right-canvas">', unsafe_allow_html=True)

# ---------- Lookup ----------
if do_lookup:
    df = run_sql(
        f"""
        SELECT *
        FROM {VIEW_FQN}
        WHERE equipment_class = %s
          AND component       = %s
          AND task_name       = %s;
        """,
        [equipment, component, task_name],
    )

    if df.empty:
        st.warning("No exact match found.")
    else:
        # derive cost_per_hour and total_cost
        cost_col = "labour_rate" if "labour_rate" in df.columns else (
                   "blended_labour_rate" if "blended_labour_rate" in df.columns else None)
        if cost_col:
            df["cost_per_hour"] = pd.to_numeric(df[cost_col], errors="coerce")
        else:
            df["cost_per_hour"] = None

        df["base_duration_hr"] = pd.to_numeric(df.get("base_duration_hr"), errors="coerce")
        df["total_cost"] = (df["cost_per_hour"] * df["base_duration_hr"]).round(2)

        st.session_state.last_lookup = df.iloc[0].to_dict()

# ---------- Present ----------
row = st.session_state.get("last_lookup")
if row:
    dur = float(row.get("base_duration_hr") or 0)
    cph = float(row.get("cost_per_hour") or 0)
    tot = float(row.get("total_cost") or 0)

    st.markdown('<div class="section card-row">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><h5>Duration (hr)</h5><div class="val">{dur:.2f}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><h5>Cost per hour</h5><div class="val">${cph:,.2f}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><h5>Total cost (per task)</h5><div class="val">${tot:,.2f}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Crew list
    roles  = str(row.get("crew_roles", "") or "").split("|")
    counts = str(row.get("crew_count", "") or "").split("|")
    crew_lines = [f"- **{r.strip()}** × {c.strip() or '1'}" for r, c in zip(roles, counts) if r.strip()]

    st.subheader("Crew")
    if crew_lines: st.markdown("\n".join(crew_lines))
    else:          st.caption("No crew information found.")

    # One-row tidy table (hide noise)
    tidy = pd.DataFrame([row])
    for drop in ["id", "effective_from", "effective_to", "blended_labour_rate", "labour_rate"]:
        if drop in tidy.columns: tidy = tidy.drop(columns=[drop])

    order = [
        "task_code","equipment_class","component","task_name",
        "base_duration_hr","cost_per_hour","total_cost","crew_roles","crew_count","notes"
    ]
    present_cols = [c for c in order if c in tidy.columns] + [c for c in tidy.columns if c not in order]

    st.subheader("Result (per task)")
    st.dataframe(
        tidy[present_cols],
        use_container_width=True,
        column_config={
            "base_duration_hr": st.column_config.NumberColumn("Duration (hr)", format="%.2f"),
            "cost_per_hour"   : st.column_config.NumberColumn("Cost/hr", format="$%.2f"),
            "total_cost"      : st.column_config.NumberColumn("Cost/Task", format="$%.2f"),
        },
    )

st.subheader("Add to estimate")
can_add = bool(row) and row.get("total_cost") is not None
if st.button("➕ Add this task", disabled=not can_add):
    line = {
        "equipment_class" : row.get("equipment_class"),
        "component"       : row.get("component"),
        "task_name"       : row.get("task_name"),
        "task_code"       : row.get("task_code"),
        "base_duration_hr": float(row.get("base_duration_hr") or 0),
        "cost_per_hour"   : float(row.get("cost_per_hour") or 0),
        "total_cost"      : float(row.get("total_cost") or 0),
        "crew_roles"      : row.get("crew_roles"),
        "crew_count"      : row.get("crew_count"),
        "notes"           : row.get("notes"),
    }
    if add_to_estimate(line):
        st.toast("Added to estimate ✅", icon="✅")
    else:
        st.toast("Already in estimate", icon="ℹ️")

# Estimate
render_estimate_table()

st.markdown('</div>', unsafe_allow_html=True)  # end right-canvas
st.markdown('</div>', unsafe_allow_html=True)  # end app-shell
