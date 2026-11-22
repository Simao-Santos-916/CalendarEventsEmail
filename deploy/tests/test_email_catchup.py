import asyncio
from datetime import datetime, timedelta, timezone

import mistralai.workflows as workflows
from mistralai.workflows.testing import create_test_worker

from worker.workflows.email_catchup import (
    CreatedEvent,
    EmailCatchupWorkflow,
    ExtractedItem,
    ExtractionResult,
    _label_slug,
    _parse_iso,
    _select_items,
    _tool_payload,
)

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)
NOW = datetime(2026, 3, 10, 8, 0, 0, tzinfo=timezone.utc)


def _future_iso(**delta: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat()


def test_tool_payload_unwraps_mcp_content() -> None:
    wrapped = {"content": [{"type": "text", "text": '{"threads": []}'}]}
    assert _tool_payload(wrapped) == {"threads": []}
    assert _tool_payload({"already": "plain"}) == {"already": "plain"}
    assert (
        _tool_payload({"content": [{"type": "text", "text": "not json"}]}) == "not json"
    )


def test_label_slug() -> None:
    assert _label_slug("Catchup/Processed") == "catchup-processed"


def test_parse_iso_handles_z_naive_and_tz() -> None:
    assert _parse_iso("2026-03-10T14:00:00Z") == datetime(
        2026, 3, 10, 14, 0, tzinfo=timezone.utc
    )
    naive_utc = _parse_iso("2026-03-10T14:00:00")
    assert naive_utc is not None and naive_utc.tzinfo == timezone.utc
    local = _parse_iso("2026-03-10T14:00:00", default_tz="Europe/Lisbon")
    assert local is not None and local.utcoffset() == timedelta(hours=0)
    assert _parse_iso("not a date") is None
    assert _parse_iso(None) is None


def _item(**overrides: object) -> ExtractedItem:
    base = {
        "category": "flight",
        "action": "create_event",
        "title": "Flight TP123 Lisbon -> London",
        "start": "2026-03-12T09:30:00+00:00",
        "confidence": 0.95,
        "thread_id": "t1",
        "subject": "Booking confirmed",
        "sender": "airline@example.com",
    }
    base.update(overrides)
    return ExtractedItem(**base)  # type: ignore[arg-type]


def test_select_items_filters_duplicates_past_and_low_confidence() -> None:
    existing = [
        {
            "summary": "Flight TP123 Lisbon -> London",
            "start": {"dateTime": "2026-03-12T09:45:00+00:00"},
        }
    ]
    items = [
        _item(),
        _item(
            title="Dinner with Ana",
            category="personal",
            start="2026-03-13T19:00:00+00:00",
        ),
        _item(
            title="Train to Porto", category="train", start="2026-03-08T10:00:00+00:00"
        ),
        _item(
            title="Maybe a call",
            action="review",
            reason="no date given",
            confidence=0.4,
        ),
        _item(title="Low confidence flight", confidence=0.5),
    ]
    to_create, duplicates, review = _select_items(items, existing, NOW, 0.8)
    assert [i.title for i in to_create] == ["Dinner with Ana"]
    assert duplicates == ["Flight TP123 Lisbon -> London"]
    assert {r.title for r in review} == {"Maybe a call", "Low confidence flight"}


def _mocks(
    calls: dict[str, list],
    extraction: ExtractionResult,
    existing_events: list[dict],
) -> list:
    @workflows.activity(name="gmail-search-threads")
    async def search_threads(query: str, max_threads: int) -> list[dict]:
        calls["search"].append(query)
        return [{"id": "t1"}, {"id": "t2"}]

    @workflows.activity(name="gmail-get-thread")
    async def fetch_thread(thread_id: str) -> dict:
        calls["fetch"].append(thread_id)
        return {
            "id": thread_id,
            "messages": [
                {
                    "id": f"m-{thread_id}",
                    "subject": f"subject {thread_id}",
                    "from": "friend@example.com",
                    "date": "Tue, 10 Mar 2026 07:00:00 +0000",
                    "snippet": "snip",
                    "body": "body text",
                }
            ],
        }

    @workflows.activity(name="calendar-list-events")
    async def list_calendar_events(
        calendar_id: str, start_time: str, end_time: str, timezone_name: str
    ) -> list[dict]:
        calls["list_events"].append(calendar_id)
        return existing_events

    @workflows.activity(name="extract-actionable-items")
    async def extract_items(
        model: str,
        threads: list[dict],
        events: list[dict],
        now_iso: str,
        timezone_name: str,
        threshold: float,
    ) -> ExtractionResult:
        calls["extract"].append(len(threads))
        return extraction

    @workflows.activity(name="calendar-create-event")
    async def create_calendar_event(
        calendar_id: str, item: ExtractedItem, dry_run: bool
    ) -> CreatedEvent:
        calls["create"].append((item.title, dry_run))
        return CreatedEvent(
            event_id="dry-run" if dry_run else "evt-1",
            summary=item.title,
            start=item.start,
            end=item.end or item.start,
            category=item.category,
            thread_id=item.thread_id,
        )

    @workflows.activity(name="gmail-ensure-label")
    async def ensure_label(label_name: str) -> str:
        calls["ensure_label"].append(label_name)
        return "Label_99"

    @workflows.activity(name="gmail-label-thread")
    async def label_thread(thread_id: str, label_id: str) -> None:
        calls["label"].append(thread_id)

    return [
        search_threads,
        fetch_thread,
        list_calendar_events,
        extract_items,
        create_calendar_event,
        ensure_label,
        label_thread,
    ]


async def test_workflow_creates_events_and_labels_threads(temporal_env) -> None:  # noqa: ANN001
    calls: dict[str, list] = {
        k: []
        for k in (
            "search",
            "fetch",
            "list_events",
            "extract",
            "create",
            "ensure_label",
            "label",
        )
    }
    extraction = ExtractionResult(
        items=[
            _item(start=_future_iso(days=2, hours=1, minutes=30)),
            _item(
                title="Dinner with Ana",
                category="personal",
                start=_future_iso(days=3, hours=11),
            ),
            _item(
                title="Duplicate flight", start=_future_iso(days=2, hours=1, minutes=40)
            ),
            _item(
                title="Ambiguous invite",
                action="review",
                reason="no time stated",
                confidence=0.6,
            ),
        ]
    )
    existing = [
        {
            "summary": "Duplicate flight",
            "start": {"dateTime": _future_iso(days=2, hours=1, minutes=30)},
        }
    ]
    async with create_test_worker(
        temporal_env,
        workflows=[EmailCatchupWorkflow],
        activities=_mocks(calls, extraction, existing),
    ):
        handle = await temporal_env.client.start_workflow(
            "email-catchup",
            {},
            id="test-email-catchup",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)
    assert result["threads_scanned"] == 2
    assert result["items_found"] == 4
    assert [e["summary"] for e in result["events_created"]] == [
        "Flight TP123 Lisbon -> London",
        "Dinner with Ana",
    ]
    assert result["duplicates_skipped"] == ["Duplicate flight"]
    assert [r["title"] for r in result["needs_review"]] == ["Ambiguous invite"]
    assert result["threads_labeled"] == ["t1", "t2"]
    assert calls["search"] == ["newer_than:3d -label:catchup-processed"]
    assert calls["list_events"] == ["simaomonteirosantos@gmail.com"]


async def test_workflow_dry_run_writes_nothing(temporal_env) -> None:  # noqa: ANN001
    calls: dict[str, list] = {
        k: []
        for k in (
            "search",
            "fetch",
            "list_events",
            "extract",
            "create",
            "ensure_label",
            "label",
        )
    }
    extraction = ExtractionResult(
        items=[_item(start=_future_iso(days=2, hours=1, minutes=30))]
    )
    async with create_test_worker(
        temporal_env,
        workflows=[EmailCatchupWorkflow],
        activities=_mocks(calls, extraction, []),
    ):
        handle = await temporal_env.client.start_workflow(
            "email-catchup",
            {"dry_run": True},
            id="test-email-catchup-dry",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)
    assert result["dry_run"] is True
    assert [e["event_id"] for e in result["events_created"]] == ["dry-run"]
    assert result["threads_labeled"] == []
    assert calls["label"] == []
    assert calls["ensure_label"] == []
    assert calls["create"] == [("Flight TP123 Lisbon -> London", True)]
