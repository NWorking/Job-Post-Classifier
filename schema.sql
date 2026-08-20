-- Run this in the Supabase SQL Editor (Project -> SQL Editor -> New query)
-- to create the table the classifier script writes to.

create table job_posts (
  id bigint generated always as identity primary key,
  date_logged timestamptz default now(),
  isjob boolean,
  job_type text check (job_type in ('fashion show', 'modeling', 'acting', 'hosting', 'other')),                           -- only if isjob = True       
  post_type text check (post_type in ('workshop info', 'update', 'warning', 'announcement', 'question', 'other')),        -- only if is_job = False
  job_title text,         
  paid_status text check (paid_status in ('paid', 'unpaid', 'unclear')),      
  required_skills text,   -- instrument | singing 
  city text,
  state text,
  deadline text,
  summary text,       
  raw_text text
);

-- Optional but recommended once more than one person/app touches this table:
-- enable Row Level Security and add a permissive policy so the anon key
-- can read/write. (If you're using the service_role key instead, RLS is
-- bypassed automatically and you can skip this.)

-- alter table job_posts enable row level security;
--
-- create policy "Allow all reads"
--   on job_posts for select
--   using (true);
--
-- create policy "Allow all inserts"
--   on job_posts for insert
--   with check (true);
