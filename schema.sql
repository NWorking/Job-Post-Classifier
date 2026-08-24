create table if not exists posts (
  id bigint generated always as identity primary key,
  date_logged timestamptz default now(),
  is_job boolean,
  post_type text check (post_type in ('workshop info', 'update', 'warning', 'announcement', 'question', 'other')),  -- only if is_job = false
  status text check (status in ('active', 'expired', 'filled')) default 'active',
  urls text,                        -- any links included in the post
  vetted text check (vetted in ('yes', 'no', 'unclear')),   -- explicitly stated in post only; never an LLM guess
  relevant_date date,               -- the one date that matters: audition/submit-by/shoot date, etc.
  relevant_date_label text,         -- what the date refers to, e.g. 'audition date', 'submit by', 'shoot date'
  summary text,
  raw_text text,
  source_url text
);

create table if not exists positions (
  id bigint generated always as identity primary key,
  post_id bigint references posts(id) on delete cascade,
  acting_or_modeling text check (acting_or_modeling in ('acting', 'modeling', 'unclear')),
  job_type text check (job_type in ('fashion show', 'modeling', 'acting', 'hosting', 'other')),
  job_title text,
  paid_status text check (paid_status in ('paid', 'unpaid', 'unclear')),
  required_skills text,
  age_raw text,                     -- e.g. '18+', 'mid-twenties', 'kids only'; verbatim, not converted
  age_bucket text check (age_bucket in ('kids', 'teens', 'young adults', 'middle age', 'seniors', 'all ages', 'unclear')),
  ethnicity_requested text,
  gender_raw text,                  -- e.g. '2 male actors', 'female models only'
  is_male boolean,                  -- derived in code from gender_raw, not set directly by the LLM
  is_female boolean,                -- derived in code from gender_raw, not set directly by the LLM
  num_spots int default 1,
  compensation_details text,
  city text,
  state text
);

-- Optional: index for faster lookups of a post's positions
create index idx_positions_post_id on positions(post_id);

-- Optional but recommended before this is multi-user: enable RLS and add
-- policies. Left commented while still prototyping.

-- alter table posts enable row level security;
-- alter table positions enable row level security;
--
-- create policy "Allow all reads on posts" on posts for select using (true);
-- create policy "Allow all inserts on posts" on posts for insert with check (true);
-- create policy "Allow all reads on positions" on positions for select using (true);
-- create policy "Allow all inserts on positions" on positions for insert with check (true);
