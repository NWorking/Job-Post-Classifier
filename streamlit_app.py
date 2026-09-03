"""
Streamlit front-end for the job-post classifier.

- Re-uses the `classify_post`, `_normalise_result`, `insert_result`, `get_client`,
  `SUPABASE_URL`, `SUPABASE_KEY`, `VISION_MODEL`, `JSON_MODEL`, and `schema`
  definitions from `job_post_classifier.py`.
- Provides a textarea for post text, an optional image uploader, and a **Classify & Log** button.
- After classification the input form disappears, the extracted JSON fields stay visible,
  and an **Add another post** button appears.
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
    SUPABASE_URL,
    SUPABASE_KEY,
    schema,
)

# ----------------------------------------------------------------------
# Helper: pretty‑print a dict as JSON in Streamlit
# ----------------------------------------------------------------------
def render_json(data: dict, title: str = "Result"):
    st.subheader(title)
    st.code(json.dumps(data, indent=2), language="json")

# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "result" not in st.session_state:
    st.session_state.result = None

st.set_page_config(page_title="Job‑Post Classifier", page_icon="🗂️")
st.title("🗂️ Facebook Job‑Board Post Classifier")
st.write(
    "Paste the post text (or leave blank if you only have an image), optionally upload the"
    " poster image, then click **Classify & Log**. The extracted fields will be displayed,"
    " and the row will be written to Supabase."
)

# ----------------------------------------------------------------------
# Input form – shown only when we have not submitted yet
# ----------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Run classification when the button is pressed
    # ------------------------------------------------------------------
    if submit:
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
            # 1️⃣ Run the classification (same logic as the CLI script)
            result = classify_post(post_text, img_path, response_format=schema)
            # Store the result so we can display it after the form disappears.
            st.session_state.result = result

            # 2️⃣ Insert into Supabase
            client = get_client()
            insert_result(client, result, post_text or "")
            st.success("✅ Row inserted into Supabase successfully!")

            # Mark as submitted – this will hide the form on the next run.
            st.session_state.submitted = True
            # Force an immediate rerun so the form disappears now.
            st.rerun()
        except Exception as exc:
            st.error(f"❗️ An error occurred: {type(exc).__name__}: {exc}")
            st.exception(exc)
        finally:
            # Clean up the temporary image file if we created one.
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass

# ----------------------------------------------------------------------
# Show results (and the "Add another post" button) after a successful submit
# ----------------------------------------------------------------------
if st.session_state.submitted:
    # Button to start a new classification run – placed above the extracted fields.
    if st.button("Add another post"):
        # Reset session state for a fresh run.
        st.session_state.submitted = False
        st.session_state.result = None
        # Clear any lingering form values.
        if "post_text" in st.session_state:
            st.session_state["post_text"] = ""
        st.session_state.pop("image_file", None)
        # Force a rerun so the form block is evaluated again.
        st.rerun()

    # The extracted fields stay visible.
    if st.session_state.result is not None:
        render_json(st.session_state.result, title="🧾 Extracted fields")