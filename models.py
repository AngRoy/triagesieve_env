"""All Pydantic models and enums for TriageSieve-OpenEnv.

Implements:
- Issue taxonomy enums (§7.1)
- Queue taxonomy (§7.2)
- Impact / Urgency / Priority enums + derivation matrix (§7.3)
- Ticket status state machine (§7.4)
- NonActionableSubtype, CustomerTier, SourceChannel, CloseReason, TaskDifficulty (§7.5–7.9)
- ActionType + TriageSieveAction tagged union (§8)
- TriageSieveObservation (§9)
- TriageSieveState (§10)
- HiddenTicketTruth dataclass (§11) — internal, never serialized to observations

Python 3.11+, Pydantic v2, OpenEnv framework contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict

__all__ = [
    # Enums
    "IssueFamily",
    "IssueSubtype",
    "QueueId",
    "Impact",
    "Urgency",
    "Priority",
    "TicketStatus",
    "NonActionableSubtype",
    "CustomerTier",
    "SourceChannel",
    "CloseReason",
    "TaskDifficulty",
    "ActionType",
    # Constants
    "VALID_FAMILY_SUBTYPES",
    "PRIORITY_MATRIX",
    "GATED_QUEUES",
    "PRIORITY_WEIGHTS",
    # Helper
    "derive_priority",
    # Standalone Pydantic models
    "InboxSummaryItem",
    "FocusedTicket",
    "RoutingPolicyCard",
    "SlaPolicyCard",
    # OpenEnv-inheriting models
    "TriageSieveAction",
    "TriageSieveObservation",
    "TriageSieveState",
    # Internal dataclass
    "HiddenTicketTruth",
]


# ---------------------------------------------------------------------------
# §7.1 Issue Taxonomy
# ---------------------------------------------------------------------------


class IssueFamily(str, Enum):
    """Top-level issue category."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SECURITY = "security"
    SHIPPING = "shipping"


class IssueSubtype(str, Enum):
    """Fine-grained issue type; always paired with a parent IssueFamily."""

    # billing
    REFUND = "refund"
    INVOICE_ERROR = "invoice_error"
    FAILED_CHARGE = "failed_charge"
    # technical
    BUG_REPORT = "bug_report"
    API_ERROR = "api_error"
    INTEGRATION_FAILURE = "integration_failure"
    # account
    PASSWORD_RESET = "password_reset"  # nosec B105
    SSO_ISSUE = "sso_issue"
    ACCOUNT_LOCKOUT = "account_lockout"
    # security
    SUSPICIOUS_LOGIN = "suspicious_login"
    EXPOSURE_RISK = "exposure_risk"
    ABUSE_REPORT = "abuse_report"
    # shipping
    DELAY = "delay"
    TRACKING_PROBLEM = "tracking_problem"
    LOST_PACKAGE = "lost_package"


# ---------------------------------------------------------------------------
# §7.2 Queue Taxonomy
# ---------------------------------------------------------------------------


class QueueId(str, Enum):
    """Available routing destinations."""

    BILLING_TEAM = "billing_team"
    TECH_SUPPORT_L1 = "tech_support_l1"
    TECH_SUPPORT_L2 = "tech_support_l2"
    ACCOUNT_TEAM = "account_team"
    SECURITY_TEAM = "security_team"
    SHIPPING_TEAM = "shipping_team"
    REFUND_TEAM = "refund_team"
    SPAM_FILTER = "spam_filter"
    SALES_OR_FEATURE_REQUESTS = "sales_or_feature_requests"


# ---------------------------------------------------------------------------
# §7.3 Impact / Urgency / Priority
# ---------------------------------------------------------------------------


class Impact(str, Enum):
    """Business scope of the issue."""

    SINGLE_USER = "single_user"
    TEAM = "team"
    ORG_WIDE = "org_wide"
    REVENUE_AFFECTING = "revenue_affecting"


class Urgency(str, Enum):
    """Time-to-business-effect of the issue."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(str, Enum):
    """Computed priority; derived from impact × urgency. Never set directly by the agent."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# §7.4 Ticket Status
# ---------------------------------------------------------------------------


class TicketStatus(str, Enum):
    """State-machine status of a single ticket."""

    NEW = "new"
    OPENED = "opened"
    CLASSIFIED = "classified"
    WAITING_FOR_INFO = "waiting_for_info"
    ROUTED = "routed"
    ESCALATED = "escalated"
    MERGED = "merged"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# §7.5 Non-Actionable Subtypes
# ---------------------------------------------------------------------------


class NonActionableSubtype(str, Enum):
    """Reason a ticket requires no further action."""

    SPAM_MARKETING = "spam_marketing"
    BENIGN_EXPECTED = "benign_expected"
    AUTOMATION_FALSE_POSITIVE = "automation_false_positive"
    DATA_ERROR = "data_error"
    NO_RESPONSE_NEEDED = "no_response_needed"


# ---------------------------------------------------------------------------
# §7.6 Customer Tier
# ---------------------------------------------------------------------------


class CustomerTier(str, Enum):
    """SLA and support tier for the submitting customer."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# §7.7 Source Channel
# ---------------------------------------------------------------------------


class SourceChannel(str, Enum):
    """Origin channel of the ticket."""

    CUSTOMER_EMAIL = "customer_email"
    INTERNAL_REPORT = "internal_report"
    MONITORING_ALERT = "monitoring_alert"


# ---------------------------------------------------------------------------
# §7.8 Close Reason
# ---------------------------------------------------------------------------


class CloseReason(str, Enum):
    """Reason for closing a ticket."""

    RESOLVED = "resolved"
    DUPLICATE = "duplicate"
    NON_ACTIONABLE = "non_actionable"
    FEATURE_REQUEST = "feature_request"
    NO_RESPONSE = "no_response"


# ---------------------------------------------------------------------------
# §7.9 Task Difficulty
# ---------------------------------------------------------------------------


class TaskDifficulty(str, Enum):
    """Episode difficulty tier for the task ladder (§18)."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# §8 Action Type
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """Discriminant tag for the TriageSieveAction tagged union."""

    OPEN_TICKET = "open_ticket"
    CLASSIFY_TICKET = "classify_ticket"
    SET_IMPACT_URGENCY = "set_impact_urgency"
    ROUTE_TICKET = "route_ticket"
    REQUEST_INFORMATION = "request_information"
    ESCALATE_TICKET = "escalate_ticket"
    MERGE_DUPLICATE = "merge_duplicate"
    CLOSE_TICKET = "close_ticket"
    SKIP_TURN = "skip_turn"
    FINISH_EPISODE = "finish_episode"


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------

VALID_FAMILY_SUBTYPES: dict[IssueFamily, frozenset[IssueSubtype]] = {
    IssueFamily.BILLING: frozenset(
        {
            IssueSubtype.REFUND,
            IssueSubtype.INVOICE_ERROR,
            IssueSubtype.FAILED_CHARGE,
        }
    ),
    IssueFamily.TECHNICAL: frozenset(
        {
            IssueSubtype.BUG_REPORT,
            IssueSubtype.API_ERROR,
            IssueSubtype.INTEGRATION_FAILURE,
        }
    ),
    IssueFamily.ACCOUNT: frozenset(
        {
            IssueSubtype.PASSWORD_RESET,
            IssueSubtype.SSO_ISSUE,
            IssueSubtype.ACCOUNT_LOCKOUT,
        }
    ),
    IssueFamily.SECURITY: frozenset(
        {
            IssueSubtype.SUSPICIOUS_LOGIN,
            IssueSubtype.EXPOSURE_RISK,
            IssueSubtype.ABUSE_REPORT,
        }
    ),
    IssueFamily.SHIPPING: frozenset(
        {
            IssueSubtype.DELAY,
            IssueSubtype.TRACKING_PROBLEM,
            IssueSubtype.LOST_PACKAGE,
        }
    ),
}
"""Maps each IssueFamily to the set of valid IssueSubtypes it may contain.

Used by step() to validate classify actions (§8 validation rule).
"""

PRIORITY_MATRIX: dict[tuple[Impact, Urgency], Priority] = {
    # single_user row
    (Impact.SINGLE_USER, Urgency.LOW): Priority.LOW,
    (Impact.SINGLE_USER, Urgency.MEDIUM): Priority.LOW,
    (Impact.SINGLE_USER, Urgency.HIGH): Priority.MEDIUM,
    (Impact.SINGLE_USER, Urgency.CRITICAL): Priority.HIGH,
    # team row
    (Impact.TEAM, Urgency.LOW): Priority.LOW,
    (Impact.TEAM, Urgency.MEDIUM): Priority.MEDIUM,
    (Impact.TEAM, Urgency.HIGH): Priority.HIGH,
    (Impact.TEAM, Urgency.CRITICAL): Priority.HIGH,
    # org_wide row
    (Impact.ORG_WIDE, Urgency.LOW): Priority.MEDIUM,
    (Impact.ORG_WIDE, Urgency.MEDIUM): Priority.HIGH,
    (Impact.ORG_WIDE, Urgency.HIGH): Priority.HIGH,
    (Impact.ORG_WIDE, Urgency.CRITICAL): Priority.CRITICAL,
    # revenue_affecting row
    (Impact.REVENUE_AFFECTING, Urgency.LOW): Priority.HIGH,
    (Impact.REVENUE_AFFECTING, Urgency.MEDIUM): Priority.HIGH,
    (Impact.REVENUE_AFFECTING, Urgency.HIGH): Priority.CRITICAL,
    (Impact.REVENUE_AFFECTING, Urgency.CRITICAL): Priority.CRITICAL,
}
"""Full 4×4 impact × urgency → priority derivation table (§7.3)."""

GATED_QUEUES: frozenset[QueueId] = frozenset(
    {
        QueueId.TECH_SUPPORT_L2,
        QueueId.SECURITY_TEAM,
    }
)
"""Queues that require prerequisites before routing is permitted (§7.2, §15)."""

PRIORITY_WEIGHTS: dict[Priority, float] = {
    Priority.LOW: 0.5,
    Priority.MEDIUM: 1.0,
    Priority.HIGH: 1.5,
    Priority.CRITICAL: 2.0,
}
"""Per-priority weights used in the priority-weighted terminal business score (§17.3)."""


def derive_priority(impact: Impact, urgency: Urgency) -> Priority:
    """Compute ticket priority from impact and urgency using the §7.3 matrix.

    Args:
        impact: Business scope of the issue.
        urgency: Time-to-business-effect of the issue.

    Returns:
        Derived Priority enum value.
    """
    key = (impact, urgency)
    if key not in PRIORITY_MATRIX:
        raise ValueError(f"No priority mapping for impact={impact!r}, urgency={urgency!r}")
    return PRIORITY_MATRIX[key]


# ---------------------------------------------------------------------------
# §9 Standalone observation sub-models
# ---------------------------------------------------------------------------


class InboxSummaryItem(BaseModel):
    """One row in the inbox list shown to the agent."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    subject: str
    sender_email: str
    received_at: str  # ISO 8601
    status: TicketStatus
    customer_tier: CustomerTier
    has_attachment: bool
    sla_remaining_minutes: int | None
    short_preview: str  # first ~80 chars of body


class FocusedTicket(BaseModel):
    """Full ticket detail revealed when the agent opens a ticket."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    subject: str
    latest_message: str
    thread_history: list[dict[str, Any]]  # [{role, content, timestamp}]
    attachments: list[str]  # filenames
    visible_internal_notes: list[str]
    prior_actions_taken: list[str]  # human-readable log of agent's actions on this ticket


class RoutingPolicyCard(BaseModel):
    """Policy card shown to the agent describing a routing queue."""

    model_config = ConfigDict(extra="forbid")

    queue_id: QueueId
    description: str
    prerequisites: list[str]
    handles_families: list[IssueFamily]


class SlaPolicyCard(BaseModel):
    """SLA policy for a given customer tier."""

    model_config = ConfigDict(extra="forbid")

    tier: CustomerTier
    response_deadline_minutes: int
    resolution_deadline_minutes: int


# ---------------------------------------------------------------------------
# §8 TriageSieveAction (OpenEnv Action subclass)
# ---------------------------------------------------------------------------


class TriageSieveAction(Action):
    """Tagged-union action model for all agent operations (§8).

    Discriminated by action_type. Field presence requirements per action_type
    are validated at step() time, not at model construction time, to allow
    the environment to return precise error messages.
    """

    action_type: ActionType
    ticket_id: str | None = None
    # classify fields
    issue_family: IssueFamily | None = None
    issue_subtype: IssueSubtype | None = None
    # impact / urgency fields
    impact: Impact | None = None
    urgency: Urgency | None = None
    # route / escalate fields
    queue_id: QueueId | None = None
    reason_code: str | None = None
    # request_information fields
    template_id: str | None = None
    requested_fields: list[str] | None = None
    # merge field
    target_ticket_id: str | None = None
    # close field
    close_reason: CloseReason | None = None


# ---------------------------------------------------------------------------
# §9 TriageSieveObservation (OpenEnv Observation subclass)
# ---------------------------------------------------------------------------


class TriageSieveObservation(Observation):
    """Full observation returned to the agent on every step (§9).

    Inherits done, reward, metadata from Observation base.
    """

    inbox_summaries: list[InboxSummaryItem]
    focused_ticket: FocusedTicket | None = None
    available_templates: list[dict[str, Any]]  # [{template_id, name, description, applies_to}]
    allowed_queues: list[QueueId]
    routing_policy_cards: list[RoutingPolicyCard]
    sla_policy_cards: list[SlaPolicyCard]
    legal_actions: list[ActionType]
    action_budget_remaining: int
    step_count: int
    current_time: str  # ISO 8601
    last_action_result: str  # "ok", error message, or pushback message
    task_difficulty: TaskDifficulty
    hint: str | None = None  # only populated when mode="train_guided"


# ---------------------------------------------------------------------------
# §10 TriageSieveState (OpenEnv State subclass)
# ---------------------------------------------------------------------------


class TriageSieveState(State):
    """Internal episode state exposed for debugging and TRL integration (§10).

    Inherits episode_id, step_count from State base (extra='allow').
    """

    task_difficulty: TaskDifficulty
    seed: int
    total_tickets: int
    action_budget: int
    action_budget_remaining: int
    mode: Literal["eval_strict", "train_guided"]
    tickets_summary: list[dict[str, Any]]  # [{ticket_id, status, gold_priority}]


# ---------------------------------------------------------------------------
# §11 HiddenTicketTruth (internal dataclass — never serialized to observations)
# ---------------------------------------------------------------------------


@dataclass
class HiddenTicketTruth:
    """Ground-truth metadata for a single ticket, hidden from the agent.

    Used exclusively by the scorer and episode engine. Never included in any
    Observation or State object. Plain (mutable) dataclass to allow the episode
    engine to update fields during runtime (e.g., follow-up generation).
    """

    ticket_id: str
    customer_tier: CustomerTier
    source_channel: SourceChannel
    issue_family: IssueFamily
    issue_subtype: IssueSubtype
    product_area: str
    impact: Impact
    urgency: Urgency
    priority: Priority  # DERIVED from impact × urgency matrix; set by episode engine
    required_queue: QueueId
    required_missing_fields: list[str] = field(default_factory=list)
    escalation_required: bool = False
    escalation_target: QueueId | None = None
    is_duplicate: bool = False
    duplicate_of: str | None = None  # ticket_id of original
    sla_response_deadline: int = 0  # minutes
    sla_resolution_deadline: int = 0  # minutes
    policy_graph_id: str = ""  # references SOP DAG in data/archetypes.json
    correct_template_ids: list[str] = field(default_factory=list)
    gold_terminal_status: TicketStatus = TicketStatus.CLOSED
    non_actionable_subtype: NonActionableSubtype | None = None
