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