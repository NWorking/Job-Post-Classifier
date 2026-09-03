# 📊 Job Postings Dashboard
"""Browse and filter job postings from Supabase.

This Streamlit page provides a read‑only view of the `positions` table joined
with its parent `posts`. It re‑uses the Supabase client defined in
`job_post_classifier.py` to keep configuration in a single place.

Features:
- Sidebar filters for acting/modeling, paid status, state, and gender.
- "Show expired postings" checkbox (default hides past `relevant_date`).
- Rows sorted by `relevant_date` ascending (nulls appear at the end).
- Each row expands to reveal full details (summary, skills, compensation,
  source URL, raw text, …).
- Graceful handling of an empty database or filters that yield no rows.
"""

import streamlit as st
from datetime import date, datetime
from typing import List, Dict, Any

# Re‑use the Supabase client helper from the classifier script.
# Importing this does not trigger UI side‑effects – it only pulls the
# configuration from `st.secrets`.
from job_post_classifier import get_client

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_posts_and_positions() -> tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch all posts and positions from Supabase.

    Returns a mapping of ``post_id`` → post dict and a list of position dicts.
    """
    client = get_client()
    # Retrieve all posts.
    posts_resp = client.table("posts").select("*").execute()
    posts = {post["id"]: post for post in posts_resp.data}

    # Retrieve all positions.
    positions_resp = client.table("positions").select("*").execute()
    positions = positions_resp.data
    return posts, positions


def gender_label(is_male: Any, is_female: Any, gender_raw: str | None) -> str:
    """Derive a concise gender description for UI filtering.

    The classifier stores boolean flags as well as the raw text. This function
    normalises the flags and falls back to the raw string when needed.
    """
    male = bool(is_male)
    female = bool(is_female)
    if male and not female:
        return "Male ok"
    if female and not male:
        return "Female ok"
    if male and female:
        return "Both ok"
    if gender_raw:
        low = gender_raw.lower()
        if "male" in low and "female" not in low:
            return "Male ok"
        if "female" in low:
            return "Female ok"
    return "Any/Unclear"


def build_rows(posts: Dict[int, Dict[str, Any]], positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine each position with its parent post.

    The result contains the columns required for the compact view plus a few
    helper fields used for filtering and sorting.
    """
    rows: List[Dict[str, Any]] = []
    for pos in positions:
        post_id = pos.get("post_id")
        post = posts.get(post_id)
        if not post:
            # Defensive – a position without a post should not happen.
            continue
        # Location string (city, state).
        city = pos.get("city") or ""
        state = pos.get("state") or ""
        location = ", ".join(filter(None, [city, state]))
        # Parse the ISO date string from Supabase.
        rd_raw = post.get("relevant_date")
        relevant_date: date | None = None
        if isinstance(rd_raw, str):
            try:
                relevant_date = datetime.strptime(rd_raw, "%Y-%m-%d").date()
            except Exception:
                relevant_date = None
        # Gender description.
        gender = gender_label(pos.get("is_male"), pos.get("is_female"), pos.get("gender_raw"))
        rows.append({
            "post_id": post_id,
            "position_id": pos.get("id"),
            "job_title": pos.get("job_title"),
            "job_type": pos.get("job_type"),
            "paid_status": pos.get("paid_status"),
            "acting_or_modeling": pos.get("acting_or_modeling"),
            "city": city,
            "state": state,
            "location": location,
            "gender": gender,
            "relevant_date": relevant_date,
            "relevant_date_label": post.get("relevant_date_label"),
            "summary": post.get("summary"),
            "required_skills": pos.get("required_skills"),
            "age_raw": pos.get("age_raw"),
            "ethnicity_requested": pos.get("ethnicity_requested"),
            "compensation_details": pos.get("compensation_details"),
            "num_spots": pos.get("num_spots") or 1,
            "source_url": post.get("source_url"),
            "urls": post.get("urls"),
            "raw_text": post.get("raw_text"),
        })
    return rows

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Job Postings Dashboard", page_icon="📊", layout="wide")
st.title("📊 Job Postings Dashboard")
st.write(
    "Browse, filter, and explore job postings stored in Supabase. "
    "The table shows a compact overview; expand a row for full details."
)

# ---------------------------------------------------------------------------
# Load data (cached to avoid repeated API calls)
# ---------------------------------------------------------------------------
with st.spinner("Loading job postings…"):
    posts_map, positions_list = load_posts_and_positions()
    all_rows = build_rows(posts_map, positions_list)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
# Acting / Modeling filter.
acting_opts = sorted({r["acting_or_modeling"] for r in all_rows if r["acting_or_modeling"]})
acting_sel = st.sidebar.multiselect("Acting or Modeling", options=acting_opts, default=[])
# Paid status filter.
paid_opts = sorted({r["paid_status"] for r in all_rows if r["paid_status"]})
paid_sel = st.sidebar.multiselect("Paid status", options=paid_opts, default=[])
# State filter (two‑letter US state code).
state_opts = sorted({r["state"] for r in all_rows if r["state"]})
state_sel = st.sidebar.multiselect("State", options=state_opts, default=[])
# Gender filter derived from boolean flags / raw text.
gender_opts = ["Male ok", "Female ok", "Both ok", "Any/Unclear"]
gender_sel = st.sidebar.multiselect("Gender", options=gender_opts, default=[])
# Expired postings toggle – hidden by default.
show_expired = st.sidebar.checkbox("Show expired postings", value=False)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
today = date.today()

def passes_filters(row: Dict[str, Any]) -> bool:
    if acting_sel and row["acting_or_modeling"] not in acting_sel:
        return False
    if paid_sel and row["paid_status"] not in paid_sel:
        return False
    if state_sel and row["state"] not in state_sel:
        return False
    if gender_sel and row["gender"] not in gender_sel:
        return False
    # Expired check – hide past dates unless toggled on.
    rd = row["relevant_date"]
    if rd is not None and rd < today and not show_expired:
        return False
    return True

filtered = [r for r in all_rows if passes_filters(r)]

# ---------------------------------------------------------------------------
# Sorting – earliest date first, nulls (no date) at the end.
# ---------------------------------------------------------------------------

def sort_key(row: Dict[str, Any]):
    # ``row["relevant_date"]`` may be None – those should appear last.
    return (row["relevant_date"] is None, row["relevant_date"] or date.max)

sorted_rows = sorted(filtered, key=sort_key)

# ---------------------------------------------------------------------------
# Empty‑state handling
# ---------------------------------------------------------------------------
if not sorted_rows:
    st.info(
        "No job postings match the current filters. "
        "Adjust the filter options in the sidebar or check back later."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Compact table (quick scanning)
# ---------------------------------------------------------------------------
st.subheader("Postings (compact view)")
compact_cols = [
    "job_title",
    "job_type",
    "paid_status",
    "location",
    "gender",
    "relevant_date",
]
compact_data = [{c: row.get(c) for c in compact_cols} for row in sorted_rows]
st.dataframe(compact_data, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Expandable details per row
# ---------------------------------------------------------------------------
for row in sorted_rows:
    header = (
        f"{row.get('job_title') or 'Untitled'} — "
        f"{row.get('job_type') or ''} — "
        f"{row.get('paid_status') or ''} — "
        f"{row.get('location') or ''} — "
        f"{row.get('gender') or ''} — "
        f"{row.get('relevant_date') or 'No date'}"
    )
    with st.expander(header, expanded=False):
        st.markdown("<u>Summary</u>", unsafe_allow_html=True)
        st.write(row.get("summary") or "(no summary)")
        st.markdown("<u>Required skills</u>", unsafe_allow_html=True)
        st.write(row.get("required_skills") or "(none listed)")
        st.markdown("<u>Age</u>", unsafe_allow_html=True)
        st.write(row.get("age_raw") or "(unspecified)")
        st.markdown("<u>Ethnicity requested</u>", unsafe_allow_html=True)
        st.write(row.get("ethnicity_requested") or "(unspecified)")
        st.markdown("<u>Compensation details</u>", unsafe_allow_html=True)
        st.write(row.get("compensation_details") or "(unspecified)")
        st.markdown("<u>Number of spots</u>", unsafe_allow_html=True)
        st.write(row.get("num_spots") or 1)
        # if row.get("source_url"):
        #     st.markdown(f"<u>Source</u>: [{row['source_url']}]({row['source_url']})", unsafe_allow_html=True)
        # if row.get("urls"):
        #     st.markdown(f"<u>Links</u>: {row['urls']}", unsafe_allow_html=True)
        st.markdown("<u>Raw post text</u>", unsafe_allow_html=True)
        st.code(row.get("raw_text") or "(empty)")

# ---------------------------------------------------------------------------
# End of page
# ---------------------------------------------------------------------------
