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

from dotenv import load_dotenv 
from groq import Groq
from supabase import create_client, Client


# ----------------------------------------------------------------------------
# Config - edit these or set as environment variables
# ----------------------------------------------------------------------------
load_dotenv()

# Helper to enforce schema consistency

def _normalise_result(result: dict) -> dict:
    """Coerce ``isjob`` to a bool and null‑out position fields when not a job."""
    raw = result.get("isjob")
    if isinstance(raw, str):
        isjob = raw.strip().lower() == "true"
    else:
        isjob = bool(raw)
    result["isjob"] = isjob

    # Position‑level fields that must be null if this is not a job posting
    position_fields = [
        "acting_or_modeling",
        "job_type",
        "job_title",
        "paid_status",
        "required_skills",
        "age_raw",
        "age_bucket",
        "ethnicity_requested",
        "gender_raw",
        "num_spots",
        "compensation_details",
        "city",
        "state",
    ]
    if not isjob:
        for key in position_fields:
            result[key] = None
    return result

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
# New table constants matching the updated schema
POSTS_TABLE = os.environ.get("SUPABASE_TABLE", "posts")
POSITIONS_TABLE = "positions"

VISION_MODEL = "qwen/qwen3.6-27b"   # currently the only model from groq that supports image as input
JSON_MODEL = "openai/gpt-oss-20b"

# prompt for step 1 model call
# should tell it to classify and output JSON, however JSON enforcement happens in step 2
STEP_1_SYSTEM_PROMPT = """You classify posts from a private Facebook job‑board group.
Respond ONLY with JSON, no markdown fences, no preamble, no commentary.

Schema:
{
  \"isjob\": true | false,

  /* ── Post‑level fields ─────────────────────────────────────── */
  \"post_type\": \"workshop info\" | \"update\" | \"warning\" | \"announcement\" |
               \"question\" | \"other\" | null,
  \"urls\": \"any URLs found in the post\" | null,
  \"vetted\": \"yes\" | \"no\" | \"unclear\",
  \"relevant_date\": "YYYY‑MM‑DD" | null,
  \"relevant_date_label\": \"what the date refers to (e.g. 'audition date')\" | null,
  \"summary\": \"one plain‑language sentence summarizing the post\",
  \"raw_text\": \"original post text\",
  \"source_url\": \"URL of the source post\" | null,

  /* ── Position‑level fields (one position per post) ─────────────── */
  \"acting_or_modeling\": \"acting\" | \"modeling\" | \"unclear\",
  \"job_type\": \"fashion show\" | \"modeling\" | \"acting\" | \"hosting\" | \"other\",
  \"job_title\": \"title of the job\" | null,
  \"paid_status\": \"paid\" | \"unpaid\" | \"unclear\",
  \"required_skills\": \"list of required skills\" | null,
  \"age_raw\": \"verbatim age description\" | null,
  \"age_bucket\": \"kids\" | \"teens\" | \"young adults\" | \"middle age\" |
                \"seniors\" | \"all ages\" | \"unclear\",
  \"ethnicity_requested\": \"ethnicity criteria\" | null,
  \"gender_raw\": \"verbatim gender description\" | null,
  \"num_spots\": integer (default 1) | null,
  \"compensation_details\": \"pay/compensation description\" | null,
  \"city\": \"city name\" | null,
  \"state\": \"2‑letter state code\" | null
}

Only fill the position‑level fields when \"isjob\" is true; otherwise set them to null.
\"vetted\" MUST ONLY be set to \"yes\" or \"no\" if the post explicitly states it – otherwise use \"unclear\".
\"age_raw\" should be extracted verbatim, not converted.
\"relevant_date\" must be an actual date (YYYY‑MM‑DD) when the post states a date; use \"relevant_date_label\" to note what the date means.
\"acting_or_modeling\" is a coarse 2‑way split; \"job_type\" is the finer‑grained classification.
Do NOT set a \"status\" field – it defaults to \"active\" in the DB."""

# prompt for step 2 model call
STEP_2_SYSTEM_PROMPT = """You are a JSON formatter for Facebook job-board posts.
You will receive two messages:
1  The assistant message contains the raw description generated by a vision model.
2  (Optional) a later user message contains the original post text.

Your job is to produce JSON that conforms to the schema supplied via `response_format`.
Use the assistant message as the primary source. If any required field is missing,
refer to the user message for the missing information.
"""

# enforced via response_format parameter in step 2 model call
schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "job_post_classification",
        "schema": {
            "type": "object",
            "properties": {
                # Post-level
                "isjob": {"type": "boolean"},
                "post_type": {
                    "type": ["string", "null"],
                    "enum": ["workshop info", "update", "warning", "announcement", "question", "other"]
                },
                "urls": {"type": ["string", "null"]},
                "vetted": {
                    "type": ["string", "null"],
                    "enum": ["yes", "no", "unclear"]
                },
                "relevant_date": {"type": ["string", "null"], "format": "date"},
                "relevant_date_label": {"type": ["string", "null"]},
                "summary": {"type": "string"},
                "raw_text": {"type": "string"},
                "source_url": {"type": ["string", "null"]},

                # Position-level
                "acting_or_modeling": {
                    "type": ["string", "null"],
                    "enum": ["acting", "modeling", "unclear"]
                },
                "job_type": {
                    "type": ["string", "null"],
                    "enum": ["fashion show", "modeling", "acting", "hosting", "other"]
                },
                "job_title": {"type": ["string", "null"]},
                "paid_status": {
                    "type": ["string", "null"],
                    "enum": ["paid", "unpaid", "unclear"]
                },
                "required_skills": {"type": ["string", "null"]},
                "age_raw": {"type": ["string", "null"]},
                "age_bucket": {
                    "type": ["string", "null"],
                    "enum": ["kids", "teens", "young adults", "middle age", "seniors", "all ages", "unclear"]
                },
                "ethnicity_requested": {"type": ["string", "null"]},
                "gender_raw": {"type": ["string", "null"]},
                "num_spots": {"type": ["integer", "null"], "minimum": 1},
                "compensation_details": {"type": ["string", "null"]},
                "city": {"type": ["string", "null"]},
                "state": {
                    "type": ["string", "null"],
                    "maxLength": 2,
                    "pattern": "^[A-Za-z]{2}$"
                }
            },
            "required": [
                "isjob", "post_type", "urls", "vetted", "relevant_date",
                "relevant_date_label", "summary", "raw_text", "source_url",
                "acting_or_modeling", "job_type", "job_title", "paid_status",
                "required_skills", "age_raw", "age_bucket",
                "ethnicity_requested", "gender_raw", "num_spots",
                "compensation_details", "city", "state"
            ],
            "additionalProperties": False
        }
    }
}



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


def insert_result(client: Client, result: dict, raw_text: str):
    """Split the classification result into a post row and a position row,
    insert the post first to get its generated ``id``, then insert the position.
    """
    # ---------- Build and insert the post row ----------
    post_row = {
        "is_job": result.get("isjob"),
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
    # Supabase returns inserted rows under the 'data' key
    post_id = None
    if isinstance(post_resp, dict):
        data = post_resp.get("data")
        if isinstance(data, list) and data:
            post_id = data[0].get("id")

    # ---------- Derive gender flags from gender_raw ----------
    gender_raw = (result.get("gender_raw") or "").lower()
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

    # ---------- Build and insert the position row ----------
    position_row = {
        "post_id": post_id,
        "acting_or_modeling": result.get("acting_or_modeling"),
        "job_type": result.get("job_type"),
        "job_title": result.get("job_title"),
        "paid_status": result.get("paid_status"),
        "required_skills": result.get("required_skills"),
        "age_raw": result.get("age_raw"),
        "age_bucket": result.get("age_bucket"),
        "ethnicity_requested": result.get("ethnicity_requested"),
        "gender_raw": result.get("gender_raw"),
        "is_male": is_male,
        "is_female": is_female,
        "num_spots": result.get("num_spots") or 1,
        "compensation_details": result.get("compensation_details"),
        "city": result.get("city"),
        "state": result.get("state"),
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

test_post = """LAST MINUTE CASTING CALL
R.J. DECKER, Season 2
Wilmington, NC - CASTLE HAYNE area -
--BACKGROUND ACTOR ROLES--
Please read role descriptions carefully
Film date/s is listed with each role
IF SUBMITTING FOR BOTH ROLES ONLY SEND 1 EMAIL
__________________
-ANGLER- Tue 8/18 & possibly Wed 8/19
ALL RACES, ALL GENDERS, ages 18 & up
*note in your submission if you're available to work Tuesday & Wednesday or only Tuesday
Rate: $96/8
__________________
-VENDOR- MUST WORK BOTH DATES Tues 8/18 & Wed 8/19
ALL RACES, ALL GENDERS, ages 25 & up
Rate: $96/8
___________________
___SUBMISSION INSTRUCTIONS___
You MUST follow all photo & submission instructions as listed OR YOU WILL NOT BE CONSIDERED!
-Must be 100% available for the date/s you are submitting for
-Must work local to the Wilmington, NC area (Castle Hayne). Travel & lodging is NOT paid for by production.
-PHOTO SUBMISSION INFORMATION-
-SUBMIT ONLY 2 PHOTOS:
1 FULL BODY PHOTO
1 HEADSHOT FRAMED SHOULDERS UP
-Photos MUST be TAKEN WITHIN THE LAST 48 HOURS in good lighting. We prefer photos be taken today.
-NO FILTERS & NO PROFESSIONAL PHOTOS
-NO MIRROR SELFIES, NO CAR SELFIES, NO SUNGLASSES, NO HATS & NO OTHER PEOPLE IN THE PHOTO WITH YOU!
**Submit via email stewartcastingbg@gmail.com**
Subject line of email should be the role/s that you're submitting for
**IF YOU ARE SUBMITTING FOR BOTH ROLES ONLY SEND 1 EMAIL**
ALL OF THE FOLLOWING MUST BE INCLUDED IN YOUR SUBMISSION ALONG WITH 2 PHOTOS:
-NAME-
-PHONE NUMBER-
-AGE-
-HEIGHT-
-WEIGHT-
-SHIRT SIZE-
-PANT SIZE-
-SHOE SIZE-
-COLOR, YEAR, MAKE & MODEL of VEHICLE-
-LIST ANY VISIBLE TATTOOS &/OR PIERCINGS-
-LIST PREVIOUS SCENES YOU WORKED THIS SEASON-
-IF SUBMITTING AS ANGLER please note if you're available both dates or only Tuesday 8/18
-CITY, STATE YOU LIVE-
-IF NOT LOCAL TO WILMINGTON, NC, PLEASE CONFIRM YOU UNDERSTAND YOU ARE RESPONSIBLE FOR TRAVEL COST-"""

# classify_post(test_post, image_path=None, response_format=schema)