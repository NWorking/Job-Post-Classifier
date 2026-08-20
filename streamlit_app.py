"""
Streamlit front-end for the job-post classifier.

- Re-uses the `classify_post`, `_normalise_result`, `insert_result`, `get_client`,
  `SUPABASE_URL`, `SUPABASE_KEY`, `VISION_MODEL`, `JSON_MODEL`, and `schema`
  definitions from `job_post_classifier.py`.
- Provides a textarea for post text, an optional image uploader, and a
  **Classify & Log** button.
- Shows the classified JSON, inserts it into Supabase, and displays a
  success message (or any error) without crashing.
"""

import os
import json
import base64
import mimetypes
import streamlit as st

# Import everything needed from the existing classifier script.
# This guarantees we use the exact same functions, prompts, and schema.
from job_post_classifier import (
    classify_post,
    insert_result,
    get_client,
    _normalise_result,
    SUPABASE_URL,
    SUPABASE_KEY,
    VISION_MODEL,
    JSON_MODEL,
    schema,
)

# ----------------------------------------------------------------------
# Helper: pretty‑print a dict as JSON in Streamlit
# ----------------------------------------------------------------------
def render_json(data: dict, title: str = "Result"):
    st.subheader(title)
    st.code(json.dumps(data, indent=2), language="json")

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "submitted" not in st.session_state:
    st.session_state.submitted = False
st.set_page_config(page_title="Job-Post Classifier", page_icon="🗂️")
st.title("🗂️ Facebook Job-Board Post Classifier")
st.write(
    """
    Paste the post text (or leave blank if you only have an image), optionally upload the
    poster image, then click **Classify & Log**. The extracted fields will be displayed,
    and the row will be written to Supabase.
    """
)

# ----- Form -----------------------------------------------------------
if not st.session_state.submitted:
    with st.form(key="classifier_form"):
        post_text = st.text_area(
            "Post text",
            placeholder="Paste the FB group post here …",
            height=200,
            key="post_text",
        )
        image_file = st.file_uploader(
            "Poster image (optional)", type=["jpg", "jpeg", "png"], key="image_file"
        )
        submit = st.form_submit_button("Classify & Log")

# ----------------------------------------------------------------------
# Run classification when the button is pressed
# ----------------------------------------------------------------------
if not st.session_state.submitted and submit:
    # Basic validation – Supabase credentials must be present
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error(
            "Supabase credentials are missing. Set `SUPABASE_URL` and `SUPABASE_KEY` in a .env file or environment."
        )
        st.stop()

    # Convert the uploaded file (if any) to a temporary path for the classifier.
    img_path = None
    if image_file is not None:
        tmp_name = f"tmp_{image_file.name}"
        with open(tmp_name, "wb") as f:
            f.write(image_file.getbuffer())
        img_path = os.path.abspath(tmp_name)

    try:
        # --------------------------------------------------------------
        # 1️⃣ Run the classification (same logic as the CLI script)
        # --------------------------------------------------------------
        result = classify_post(post_text, img_path, response_format=schema)

        # --------------------------------------------------------------
        # 2️⃣ Show the raw result (already normalised by the helper)
        # --------------------------------------------------------------
        render_json(result, title="🧾 Extracted fields")

        # --------------------------------------------------------------
        # 3️⃣ Insert into Supabase
        # --------------------------------------------------------------
        client = get_client()
        insert_result(client, result, post_text or "")
        st.success("✅ Row inserted into Supabase successfully!")
        st.session_state.submitted = True

    except Exception as exc:
        st.error(f"❗️ An error occurred: {type(exc).__name__}: {exc}")
        # For debugging you can uncomment the next line to see a traceback:
        # st.exception(exc)

    finally:
        # Clean up the temporary image file if we created one.
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass

# After a successful insert, show a button to add another post
if st.session_state.submitted:
    if st.button("Add another post"):
        # Reset state and clear form fields
        st.session_state.submitted = False
        # Clear the text area (allowed via session_state)
        if "post_text" in st.session_state:
            st.session_state["post_text"] = ""
        # Remove the uploaded file from session_state (cannot set directly)
        st.session_state.pop("image_file", None)
        # Force a rerun so the form block is evaluated again with submitted=False
        st.rerun()