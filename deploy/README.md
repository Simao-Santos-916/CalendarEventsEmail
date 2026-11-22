CalendarEventsEmail

Mistral Workflows project hosting the email-catchup workflow.

Each run scans recent Gmail threads (skipping threads already labeled
Catchup/Processed), extracts flights, trains, events and family/friends
moments with a structured LLM call, dedupes against existing calendar events,
creates events on the target calendar (simaomonteirosantos@gmail.com) with
per-category reminders, and labels the processed threads. dry_run=true
reports what it would do without writing anything.

Workflow: src/worker/workflows/email_catchup.py (name: email-catchup)

Worker entrypoint: src/worker/entrypoints/worker.py

Tests: tests/test_email_catchup.py

Deploy: managed deployment email-catchup-worker built from this repo's Dockerfile

Schedule: daily 08:00 Europe/Lisbon on workflow email-catchup

Scheduled runs execute as the deployment identity, so the gmail and
google_calendar connectors need workspace-scope credentials in Studio
(gmail scope must include gmail.modify for labeling).