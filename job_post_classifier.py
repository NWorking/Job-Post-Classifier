"""
Facebook job-group post classifier -> Supabase logger.

Takes a post (text and/or an image of a poster), classifies it as job/not-job,
extracts structured fields for job posts, and inserts a row into a Supabase
(Postgres) table. This is the multi-user-friendly version - the table lives
in the cloud, so any script/bot/dashboard for any group member can read from
and write to the same live data.

SETUP
-----
1. pip install anthropic supabase

2. LLM API key:
   export ANTHROPIC_API_KEY="sk-ant-..."

3. Supabase project (one-time, ~10 min):
   a. Go to https://supabase.com -> sign in with GitHub/Google -> "New project".
      Free tier is plenty for this (500MB DB, way more than you'll need).
   b. Once the project is created, go to Project Settings -> API.
      Copy the "Project URL" and the "anon public" key (or "service_role"
      key if you want this script to bypass row-level security - see note
      below). Set them as env vars:
        export SUPABASE_URL="https://xxxx.supabase.co"
        export SUPABASE_KEY="eyJ..."
   c. Go to the SQL Editor in the Supabase dashboard and run the CREATE TABLE
      statement below (also saved as schema.sql) to create the `job_posts`
      table.

   Note on keys: the "PUBLISHABLE" key respects Row Level Security (RLS) policies,
   which is what you want once multiple people/apps are reading and writing.
   For quick prototyping, either disable RLS on the table (fine while it's
   just you) or use the "SECRET" key (bypasses RLS entirely - keep
   this one secret, never put it in client-side/browser code).


USAGE
-----
    python job_post_classifier.py --text "Looking for a PA for a 2-day shoot..."
    python job_post_classifier.py --image path/to/poster.jpg
    python job_post_classifier.py --text "..." --image path/to/poster.jpg
    python job_post_classifier.py   # interactive mode, prompts for input
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import streamlit as st

from dotenv import load_dotenv 
from groq import Groq
from supabase import create_client, Client


# ----------------------------------------------------------------------------
# Config - edit these or set as environment variables
# ----------------------------------------------------------------------------
load_dotenv()



SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
# New table constants matching the updated schema
POSTS_TABLE = st.secrets.get("SUPABASE_TABLE", "posts")
POSITIONS_TABLE = "positions"

VISION_MODEL = "qwen/qwen3.8-27b"   # currently the only model from groq that supports image as input
JSON_MODEL = "openai/gpt-oss-20b"

# referenced directly in prompt for step 1 model call
# enforced via response_format parameter in step 2 model call
schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "job_post_classification_multi",
        "schema": {
            "type": "object",
            "properties": {
                "is_job": {"type": "boolean"},
                "post_type": {"type": ["string","null"], "enum": ["workshop info","update","warning","announcement","question","other"]},
                "urls": {"type": ["string","null"]},
                "vetted": {"type": ["string","null"], "enum": ["yes","no","unclear"]},
                "relevant_date": {"type": ["string","null"], "format": "date"},
                "relevant_date_label": {"type": ["string","null"]},
                "summary": {"type": "string"},
                "raw_text": {"type": "string"},
                "source_url": {"type": ["string","null"]},
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "acting_or_modeling": {"type": ["string","null"], "enum": ["acting","modeling","unclear"]},
                            "job_type": {"type": ["string","null"], "enum": ["fashion show","modeling","acting","hosting","other"]},
                            "job_title": {"type": ["string","null"]},
                            "paid_status": {"type": ["string","null"], "enum": ["paid","unpaid","unclear"]},
                            "required_skills": {"type": ["string","null"]},
                            "age_raw": {"type": ["string","null"]},
                            "age_bucket": {"type": ["string","null"], "enum": ["kids","teens","young adults","middle age","seniors","all ages","unclear"]},
                            "ethnicity_requested": {"type": ["string","null"]},
                            "gender_raw": {"type": ["string","null"]},
                            "num_spots": {"type": ["integer","null"], "minimum": 1},
                            "compensation_details": {"type": ["string","null"]},
                            "city": {"type": ["string","null"]},
                            "state": {"type": ["string","null"], "maxLength": 2, "pattern": "^[A-Za-z]{2}$"}
                        },
                        "required": [
                            "acting_or_modeling", "job_type", "job_title", "paid_status",
                            "required_skills", "age_raw", "age_bucket", "ethnicity_requested",
                            "gender_raw", "num_spots", "compensation_details", "city", "state"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["is_job","post_type","urls","vetted","relevant_date","relevant_date_label","summary","raw_text","source_url","positions"],
            "additionalProperties": False
        }
    }
}


# prompt for step 1 model call
# should tell it to classify and output JSON, however JSON enforcement happens in step 2
STEP_1_SYSTEM_PROMPT = f"""You classify posts from a private Facebook job-board group.
Respond ONLY with JSON matching the following schema, no markdown fences, no preamble, no commentary.

{json.dumps(schema, indent=2)}

Field guidance:

POST-LEVEL fields (one set per post):
- "is_job": true if this post is advertising one or more paid or unpaid
  opportunities (acting, modeling, hosting, etc.). false for updates,
  warnings, announcements, questions, or anything else that isn't itself
  an opportunity.
- "post_type": only relevant when is_job is false - what kind of non-job
  post this is.
- "vetted" MUST ONLY be set to "yes" or "no" if the post explicitly states
  it - otherwise use "unclear". Never infer or guess this.
- "relevant_date" must be an actual date (YYYY-MM-DD) when the post states
  or clearly implies one (audition date, submit-by date, shoot date,
  etc.); use "relevant_date_label" to note what that date refers to.
- "summary" and "raw_text" always get filled regardless of is_job.

POSITIONS array (one item per DISTINCT position mentioned in the post):
- If the post is requesting multiple people for the SAME role with
    identical requirements (e.g. "looking for 5 background actors, $100/day
  each"), that is ONE entry in the positions array - use "num_spots" to
  capture the headcount, do not create duplicate entries.
- Only create separate array entries when the post describes genuinely
  distinct roles with different requirements (different job_type,
  different pay, different skills, different age/gender asks, etc.) -
  for example a post advertising both a "runway model" and a "hair
  stylist assistant" in the same post is TWO entries.
- If is_job is false, "positions" should be an empty array.
- A job post should almost always have at least one position when
  is_job is true.
- "acting_or_modeling" is a coarse 2-way split; "job_type" is the
  finer-grained classification - fill both per position.
- "age_raw" should be extracted verbatim as stated in the post (e.g.
  "18+", "mid-twenties", "kids only") - do not convert to a numeric
  range.
- "gender_raw" should similarly be extracted verbatim (e.g. "2 male
  actors", "female models only").
- "state" should be a 2-letter code when a US state is identifiable.

Do NOT include a "status" field anywhere - it defaults to "active" in the
database and is not something you should set."""

# prompt for step 2 model call
STEP_2_SYSTEM_PROMPT = """You are a JSON formatter for Facebook job-board posts.
You will receive two messages:
1  The assistant message contains the raw description generated by a vision model.
2  (Optional) a later user message contains the original post text.

Your job is to produce JSON that conforms to the schema supplied via `response_format`.
Use the assistant message as the primary source. If any required field is missing,
refer to the user message for the missing information.
"""


# Helper to enforce schema consistency

def _normalise_result(result: dict) -> dict:
    """Coerce ``is_job`` to a bool and ensure the ``positions`` array matches schema."""
    raw = result.get("is_job")
    if isinstance(raw, str):
        is_job = raw.strip().lower() == "true"
    else:
        is_job = bool(raw)
    result["is_job"] = is_job

    # Ensure a positions list exists; if not a job, clear positions.
    if not is_job:
        result["positions"] = []
    else:
        # If positions missing, default to empty list to avoid key errors downstream.
        result.setdefault("positions", [])
    return result


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

def classify_post(text: str | None, image_path: str | None, response_format: dict = schema) -> dict:
    """Classify a Facebook job-board post in two steps.

    Step 1: Call the Qwen vision model (``qwen/qwen3.6-27b``) to obtain a
    textual description of the post. The model accepts an optional image.

    Step 2: Pass the description (plus any original text) to the JSON-enforcing
    model ``openai/gpt-oss-20b``.

    The function returns a Python ``dict`` parsed from the JSON output.
    """
    client = Groq()

    # Build input for vision model (step 1)
    content_blocks: list[dict] = []
    if image_path:
        media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{media_type};base64,{image_b64}"
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )
    user_text = text or "See attached image poster."
    content_blocks.append({"type": "text", "text": user_text})

    # Step 1 model call
    vision_response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": STEP_1_SYSTEM_PROMPT},
            {"role": "user", "content": content_blocks}
            ],
        temperature=0,
    )
    raw_vision = vision_response.choices[0].message.content
    if isinstance(raw_vision, list):
        vision_text = "".join(item.get("text", "") for item in raw_vision if item.get("type") == "text")
    else:
        vision_text = str(raw_vision)

    # Build messages for step‑2 JSON‑enforcement model
    json_messages = [
        {"role": "system", "content": STEP_2_SYSTEM_PROMPT},
        {"role": "assistant", "content": vision_text},
    ]
    if text:
        json_messages.append({"role": "user", "content": text})

    # Step 2 model call
    json_response = client.chat.completions.create(
        model=JSON_MODEL,
        messages=json_messages,
        response_format=response_format,
        temperature=0,
        max_completion_tokens=4000
    )
    raw_json = json_response.choices[0].message.content
    cleaned = (
        raw_json.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        print(cleaned)
        result = json.loads(cleaned)
        return _normalise_result(result)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{raw_json}") from e



# ----------------------------------------------------------------------------
# Supabase
# ----------------------------------------------------------------------------
def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set (env vars or in this script)."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# insert_result to handle multiple positions from a single post
def insert_result(client: Client, result: dict, raw_text: str):
    """Insert a post row and multiple position rows, linking via post_id."""
    # Build and insert the post row
    post_row = {
        "is_job": result.get("is_job"),
        "post_type": result.get("post_type"),
        "urls": result.get("urls"),
        "vetted": result.get("vetted"),
        "relevant_date": result.get("relevant_date"),
        "relevant_date_label": result.get("relevant_date_label"),
        "summary": result.get("summary"),
        "raw_text": (raw_text or "")[:2000],
        "source_url": result.get("source_url"),
        # "status" omitted – DB defaults to 'active'
    }
    post_resp = client.table(POSTS_TABLE).insert(post_row).execute()
    post_id = post_resp.data[0]["id"] if post_resp.data else None

    if post_id is None:
        raise RuntimeError(f"Failed to insert post row, got: {post_resp}")


    # Insert each position
    for pos in result.get("positions", []):
        # Derive gender flags per position
        gender_raw = (pos.get("gender_raw") or "").lower()
        is_male = None
        is_female = None
        if gender_raw:
            if "female" in gender_raw or "women" in gender_raw:
                is_female = True
            if ("male" in gender_raw or "men" in gender_raw) and "female" not in gender_raw:
                is_male = True
            if ("female" in gender_raw) and ("male" in gender_raw or "men" in gender_raw):
                is_male = True
                is_female = True

        position_row = {
            "post_id": post_id,
            "acting_or_modeling": pos.get("acting_or_modeling"),
            "job_type": pos.get("job_type"),
            "job_title": pos.get("job_title"),
            "paid_status": pos.get("paid_status"),
            "required_skills": pos.get("required_skills"),
            "age_raw": pos.get("age_raw"),
            "age_bucket": pos.get("age_bucket"),
            "ethnicity_requested": pos.get("ethnicity_requested"),
            "gender_raw": pos.get("gender_raw"),
            "is_male": is_male,
            "is_female": is_female,
            "num_spots": pos.get("num_spots") or 1,
            "compensation_details": pos.get("compensation_details"),
            "city": pos.get("city"),
            "state": pos.get("state"),
        }
        client.table(POSITIONS_TABLE).insert(position_row).execute()
# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify a FB group post and log it to Supabase.")
    parser.add_argument("--text", help="Post text")
    parser.add_argument("--image", help="Path to a poster image file")
    parser.add_argument("--no-db", action="store_true", help="Print result only, skip writing to Supabase")
    args = parser.parse_args()

    text, image_path = args.text, args.image

    if not text and not image_path:
        print("Paste post text (leave blank + press enter if image-only), then Enter:")
        text = input("> ").strip() or None
        image_path = input("Path to image (blank to skip): ").strip() or None

    if not text and not image_path:
        print("Nothing to classify. Provide --text and/or --image.")
        sys.exit(1)

    print("Classifying...")
    result = classify_post(text, image_path)
    print(json.dumps(result, indent=2))

    if args.no_db:
        return

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("\nSUPABASE_URL / SUPABASE_KEY not set - skipping database write. "
              "See setup instructions at the top of this script.")
        return

    print("Writing to Supabase...")
    client = get_client()
    insert_result(client, result, text or "")
    print("Done.")


if __name__ == "__main__":
    main()

