"""Email catch-up workflow: scan recent Gmail threads, extract flights, trains,
events and family/friends moments, create calendar events for them, and label the
processed threads so repeat runs stay idempotent."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pydantic
import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    UserMessage,
    chat_parse_to_model,
)
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail_connector = connector("gmail")
calendar_connector = connector("google_calendar")

DEFAULT_CALENDAR_ID = "simaomonteirosantos@gmail.com"
DEFAULT_TIMEZONE = "Europe/Lisbon"
PROCESSED_LABEL = "Catchup/Processed"
CALENDAR_WINDOW_DAYS = 90
PAST_TOLERANCE = timedelta(hours=6)
DUPLICATE_WINDOW = timedelta(hours=3)
MAX_BODY_CHARS = 2500
MAX_PROMPT_EVENTS = 100

CATEGORY_REMINDERS: dict[str, list[dict[str, Any]]] = {
    "flight": [
        {"method": "email", "minutes": 1440},
        {"method": "popup", "minutes": 180},
    ],
    "train": [{"method": "popup", "minutes": 60}],
    "event": [{"method": "popup", "minutes": 30}],
    "personal": [
        {"method": "popup", "minutes": 1440},
        {"method": "popup", "minutes": 60},
    ],
    "other": [{"method": "popup", "minutes": 30}],
}
CATEGORY_DEFAULT_MINUTES = {
    "flight": 180,
    "train": 120,
    "event": 120,
    "personal": 60,
    "other": 60,
}


class EmailCatchupInput(pydantic.BaseModel):
    lookback_days: int = 3
    max_threads: int = 25
    calendar_id: str = DEFAULT_CALENDAR_ID
    timezone: str = DEFAULT_TIMEZONE
    processed_label: str = PROCESSED_LABEL
    dry_run: bool = False
    model: str = "mistral-medium-latest"
    confidence_threshold: float = 0.8
    extra_query: str = ""


class ExtractedItem(pydantic.BaseModel):
    category: Literal["flight", "train", "event", "personal", "other"] = "other"
    action: Literal["create_event", "skip", "review"] = "skip"
    title: str = ""
    start: str = ""
    end: str | None = None
    timezone: str | None = None
    location: str | None = None
    details: str = ""
    confidence: float = 0.0
    reason: str = ""
    thread_id: str = ""
    subject: str = ""
    sender: str = ""


class ExtractionResult(pydantic.BaseModel):
    items: list[ExtractedItem] = []


class CreatedEvent(pydantic.BaseModel):
    event_id: str
    summary: str
    start: str
    end: str
    category: str
    thread_id: str


class ReviewItem(pydantic.BaseModel):
    title: str
    reason: str
    subject: str
    sender: str
    thread_id: str


class CatchupReport(pydantic.BaseModel):
    dry_run: bool
    threads_scanned: int
    items_found: int
    events_created: list[CreatedEvent]
    duplicates_skipped: list[str]
    needs_review: list[ReviewItem]
    threads_labeled: list[str]


def _tool_payload(result: Any) -> Any:
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        content = result["content"]
        if (
            content
            and isinstance(content[0], dict)
            and isinstance(content[0].get("text"), str)
        ):
            text = content[0]["text"]
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
    return result


def _label_slug(label: str) -> str:
    return label.lower().replace(" ", "-").replace("/", "-")


def _parse_iso(value: Any, default_tz: str | None = None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=ZoneInfo(default_tz) if default_tz else timezone.utc
        )
    return parsed


def _event_start(event: dict[str, Any]) -> datetime | None:
    start = event.get("start") or {}
    return _parse_iso(start.get("dateTime") or start.get("date"))


def _normalize_title(title: Any) -> str:
    return " ".join(str(title or "").lower().split())


def _select_items(
    items: list[ExtractedItem],
    existing_events: list[dict[str, Any]],
    now: datetime,
    threshold: float,
) -> tuple[list[ExtractedItem], list[str], list[ReviewItem]]:
    to_create: list[ExtractedItem] = []
    duplicates: list[str] = []
    review: list[ReviewItem] = []
    existing = [
        (_normalize_title(e.get("summary")), _event_start(e)) for e in existing_events
    ]
    for item in items:
        review_item = ReviewItem(
            title=item.title,
            reason=item.reason or item.action,
            subject=item.subject,
            sender=item.sender,
            thread_id=item.thread_id,
        )
        if item.action == "review" or (
            item.action == "create_event" and item.confidence < threshold
        ):
            review.append(review_item)
            continue
        if item.action != "create_event":
            continue
        start = _parse_iso(item.start)
        if start is None:
            review.append(review_item)
            continue
        if start < now - PAST_TOLERANCE:
            continue
        item_title = _normalize_title(item.title)
        duplicate = any(
            existing_title
            and (existing_title in item_title or item_title in existing_title)
            and existing_start is not None
            and abs((existing_start - start).total_seconds())
            <= DUPLICATE_WINDOW.total_seconds()
            for existing_title, existing_start in existing
        )
        if duplicate:
            duplicates.append(item.title)
            continue
        to_create.append(item)
    return to_create, duplicates, review


def _build_prompt(
    threads: list[dict[str, Any]],
    existing_events: list[dict[str, Any]],
    now_iso: str,
    timezone_name: str,
    threshold: float,
) -> str:
    thread_blocks = []
    for thread in threads:
        messages = thread.get("messages") or []
        parts = []
        for message in messages:
            body = str(message.get("body") or message.get("snippet") or "")[
                :MAX_BODY_CHARS
            ]
            parts.append(
                f"--- message from={message.get('from', '')} date={message.get('date', '')} "
                f"subject={message.get('subject', '')}\n{body}"
            )
        thread_blocks.append(
            f"=== thread id={thread.get('id', '')} subject={messages[0].get('subject', '') if messages else ''}\n"
            + "\n".join(parts)
        )
    event_blocks = [
        {
            "summary": e.get("summary"),
            "start": (e.get("start") or {}).get("dateTime")
            or (e.get("start") or {}).get("date"),
            "end": (e.get("end") or {}).get("dateTime")
            or (e.get("end") or {}).get("date"),
            "location": e.get("location"),
        }
        for e in existing_events[:MAX_PROMPT_EVENTS]
    ]
    return f"""You are helping the user catch up on their inbox. Current time: {now_iso} (user timezone: {timezone_name}).

Below are recent email threads and the user's existing calendar events (use them to detect duplicates).

For each thread, extract items that deserve a calendar entry. Categories:
- flight: flight bookings and confirmations (departure/arrival airports, flight number, times)
- train: train tickets and reservations
- event: meetings, parties, concerts, appointments with a concrete date and time
- personal: family and friends moments (birthdays, dinners, visits, get-togethers, calls)
- other: anything else with a concrete date and time

Rules:
- Set action to "create_event" only when the email contains a concrete, unambiguous date and time. Resolve relative dates ("tomorrow", "next Friday", "dia 12") against the current time above. Only items with confidence >= {threshold} may use "create_event"; confidence is a number between 0 and 1.
- Set action to "review" when something looks schedulable but a key detail is missing or ambiguous; explain what is missing in "reason".
- Otherwise set action to "skip".
- "start" and "end" must be ISO 8601. Set "timezone" when the email states a zone different from {timezone_name}.
- For flights: start = departure time, end = arrival time if stated (else departure + 3 hours), location = departure airport.
- "title" must be short and human-readable, e.g. "Flight TP123 Lisbon -> London" or "Dinner with Ana".
- Fill thread_id, subject and sender from the thread the item came from.
- If an existing calendar event clearly matches the same plan, set action to "skip" and say "duplicate of <event summary>" in "reason".
- "details" should carry the useful specifics: booking reference, terminal, gate, seat, address, who is involved, links.

Return a JSON object: {{"items": [ ... ]}}

THREADS:
{json.dumps(thread_blocks, ensure_ascii=False)}

EXISTING CALENDAR EVENTS:
{json.dumps(event_blocks, ensure_ascii=False)}
"""


@workflows.activity(
    name="gmail-search-threads", start_to_close_timeout=timedelta(seconds=60)
)
async def search_threads(
    query: str,
    max_threads: int,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    page_token: str | None = None
    while len(refs) < max_threads:
        args: dict[str, Any] = {
            "query": query,
            "pageSize": min(50, max_threads - len(refs)),
        }
        if page_token:
            args["pageToken"] = page_token
        payload = _tool_payload(await gmail.call_tool("search_threads", args))
        if not isinstance(payload, dict):
            break
        threads = payload.get("threads") or []
        refs.extend(t for t in threads if isinstance(t, dict))
        page_token = payload.get("nextPageToken")
        if not page_token or not threads:
            break
    return refs[:max_threads]


@workflows.activity(
    name="gmail-get-thread", start_to_close_timeout=timedelta(seconds=60)
)
async def fetch_thread(
    thread_id: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> dict[str, Any]:
    payload = _tool_payload(
        await gmail.call_tool(
            "get_thread", {"threadId": thread_id, "messageFormat": "FULL_CONTENT"}
        )
    )
    messages = []
    for msg in (payload.get("messages") or []) if isinstance(payload, dict) else []:
        if not isinstance(msg, dict):
            continue
        part = msg.get("payload") or {}
        headers = {
            str(h.get("name", "")).lower(): h.get("value")
            for h in (part.get("headers") or [])
            if isinstance(h, dict)
        }
        body = (
            msg.get("plaintextBody")
            or msg.get("body")
            or part.get("body")
            or msg.get("snippet")
            or ""
        )
        messages.append(
            {
                "id": msg.get("id"),
                "subject": headers.get("subject") or msg.get("subject") or "",
                "from": headers.get("from") or msg.get("from") or "",
                "date": headers.get("date") or msg.get("date") or "",
                "snippet": msg.get("snippet") or "",
                "body": body if isinstance(body, str) else "",
            }
        )
    return {
        "id": (payload.get("id") if isinstance(payload, dict) else None) or thread_id,
        "messages": messages,
    }


@workflows.activity(
    name="calendar-list-events", start_to_close_timeout=timedelta(seconds=60)
)
async def list_calendar_events(
    calendar_id: str,
    start_time: str,
    end_time: str,
    timezone_name: str,
    calendar: ToolCallClient = Depends(calendar_connector),
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        args: dict[str, Any] = {
            "calendarId": calendar_id,
            "startTime": start_time,
            "endTime": end_time,
            "timeZone": timezone_name,
            "orderBy": "startTime",
            "pageSize": 250,
        }
        if page_token:
            args["pageToken"] = page_token
        payload = _tool_payload(await calendar.call_tool("list_events", args))
        if not isinstance(payload, dict):
            break
        events.extend(e for e in payload.get("events") or [] if isinstance(e, dict))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return events


@workflows.activity(
    name="extract-actionable-items", start_to_close_timeout=timedelta(minutes=5)
)
async def extract_items(
    model: str,
    threads: list[dict[str, Any]],
    existing_events: list[dict[str, Any]],
    now_iso: str,
    timezone_name: str,
    threshold: float,
) -> ExtractionResult:
    prompt = _build_prompt(threads, existing_events, now_iso, timezone_name, threshold)
    request = ChatCompletionRequest(
        model=model,
        messages=[UserMessage(content=prompt)],
        temperature=0.0,
    )
    return await chat_parse_to_model(ExtractionResult, request)


@workflows.activity(
    name="calendar-create-event",
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=1,
)
async def create_calendar_event(
    calendar_id: str,
    item: ExtractedItem,
    dry_run: bool,
    calendar: ToolCallClient = Depends(calendar_connector),
) -> CreatedEvent:
    item_tz = item.timezone or DEFAULT_TIMEZONE
    start = _parse_iso(item.start, default_tz=item_tz)
    if start is None:
        raise ValueError(f"item has no parseable start: {item.start!r}")
    end = _parse_iso(item.end, default_tz=item_tz) if item.end else None
    if end is None:
        end = start + timedelta(minutes=CATEGORY_DEFAULT_MINUTES.get(item.category, 60))
    description = (
        f'{item.details}\n\nSource: email "{item.subject}" from {item.sender}\n'
        f"Thread: https://mail.google.com/mail/u/0/#inbox/{item.thread_id}"
    )
    args: dict[str, Any] = {
        "calendarId": calendar_id,
        "summary": item.title,
        "description": description,
        "startTime": start.isoformat(),
        "endTime": end.isoformat(),
        "timeZone": item_tz,
        "overrideReminders": CATEGORY_REMINDERS.get(
            item.category, CATEGORY_REMINDERS["other"]
        ),
    }
    if item.location:
        args["location"] = item.location
    if dry_run:
        return CreatedEvent(
            event_id="dry-run",
            summary=item.title,
            start=args["startTime"],
            end=args["endTime"],
            category=item.category,
            thread_id=item.thread_id,
        )
    payload = _tool_payload(await calendar.call_tool("create_event", args))
    event_id = payload.get("id") if isinstance(payload, dict) else None
    return CreatedEvent(
        event_id=str(event_id or ""),
        summary=item.title,
        start=args["startTime"],
        end=args["endTime"],
        category=item.category,
        thread_id=item.thread_id,
    )


@workflows.activity(
    name="gmail-ensure-label", start_to_close_timeout=timedelta(seconds=60)
)
async def ensure_label(
    label_name: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> str:
    payload = _tool_payload(await gmail.call_tool("list_labels", {}))
    labels = payload.get("labels") or [] if isinstance(payload, dict) else []
    for label in labels:
        if isinstance(label, dict) and label.get("name") == label_name:
            return str(label.get("labelId") or label.get("id") or "")
    created = _tool_payload(
        await gmail.call_tool("create_label", {"displayName": label_name})
    )
    return str(created.get("labelId") or created.get("id") or "")


@workflows.activity(
    name="gmail-label-thread", start_to_close_timeout=timedelta(seconds=60)
)
async def label_thread(
    thread_id: str,
    label_id: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> None:
    await gmail.call_tool(
        "label_thread", {"threadId": thread_id, "labelIds": [label_id]}
    )


@workflows.workflow.define(
    name="email-catchup",
    workflow_display_name="Email catch-up",
    workflow_description=(
        "Scans recent Gmail threads, creates calendar events for flights, trains, events "
        "and family/friends moments found in them, and labels the processed threads."
    ),
)
@uses_connectors(gmail_connector, calendar_connector)
class EmailCatchupWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: EmailCatchupInput) -> CatchupReport:
        now = workflows.workflow.now()
        query = f"newer_than:{input.lookback_days}d -label:{_label_slug(input.processed_label)}"
        if input.extra_query:
            query = f"{query} {input.extra_query}"
        refs = await search_threads(query, input.max_threads)
        threads = [await fetch_thread(str(ref.get("id") or "")) for ref in refs]
        window_start = (now - timedelta(days=1)).isoformat()
        window_end = (now + timedelta(days=CALENDAR_WINDOW_DAYS)).isoformat()
        existing = await list_calendar_events(
            input.calendar_id, window_start, window_end, input.timezone
        )
        extraction = await extract_items(
            input.model,
            threads,
            existing,
            now.isoformat(),
            input.timezone,
            input.confidence_threshold,
        )
        to_create, duplicates, review = _select_items(
            extraction.items, existing, now, input.confidence_threshold
        )
        created: list[CreatedEvent] = []
        seen_titles: set[str] = set()
        for item in to_create:
            title_key = _normalize_title(item.title)
            if title_key in seen_titles:
                duplicates.append(item.title)
                continue
            seen_titles.add(title_key)
            created.append(
                await create_calendar_event(input.calendar_id, item, input.dry_run)
            )
        labeled: list[str] = []
        if threads and not input.dry_run:
            label_id = await ensure_label(input.processed_label)
            if label_id:
                for thread in threads:
                    await label_thread(thread["id"], label_id)
                    labeled.append(thread["id"])
        return CatchupReport(
            dry_run=input.dry_run,
            threads_scanned=len(threads),
            items_found=len(extraction.items),
            events_created=created,
            duplicates_skipped=duplicates,
            needs_review=review,
            threads_labeled=labeled,
        )
