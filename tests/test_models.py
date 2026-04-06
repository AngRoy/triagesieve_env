"""Tests for models.py (Phase 1, §25).

Covers:
- Enum member counts and exact string values
- VALID_FAMILY_SUBTYPES correctness (5 keys, 3 members each, no cross-family leakage)
- All 16 PRIORITY_MATRIX entries from the §7.3 matrix
- derive_priority exhaustive coverage
- GATED_QUEUES membership
- PRIORITY_WEIGHTS values
- TriageSieveAction construction + extra-field rejection
- TriageSieveObservation and TriageSieveState construction
- HiddenTicketTruth dataclass construction
- JSON round-trip for Action, Observation, State
"""
import json
from dataclasses import asdict

import pytest
from pydantic import ValidationError

from ..models import (
    GATED_QUEUES,
    PRIORITY_MATRIX,
    PRIORITY_WEIGHTS,
    VALID_FAMILY_SUBTYPES,
    ActionType,
    CloseReason,
    CustomerTier,
    FocusedTicket,
    HiddenTicketTruth,
    Impact,
    InboxSummaryItem,
    IssueFamily,
    IssueSubtype,
    NonActionableSubtype,
    Priority,
    QueueId,
    RoutingPolicyCard,
    SlaPolicyCard,
    SourceChannel,
    TriageSieveAction,
    TriageSieveObservation,
    TriageSieveState,
    TaskDifficulty,
    TicketStatus,
    Urgency,
    derive_priority,
)


# ---------------------------------------------------------------------------
# Enum member counts and string values
# ---------------------------------------------------------------------------


class TestIssueFamilyEnum:
    def test_member_count(self):
        assert len(IssueFamily) == 5

    def test_string_values(self):
        assert IssueFamily.BILLING == "billing"
        assert IssueFamily.TECHNICAL == "technical"
        assert IssueFamily.ACCOUNT == "account"
        assert IssueFamily.SECURITY == "security"
        assert IssueFamily.SHIPPING == "shipping"


class TestIssueSubtypeEnum:
    def test_member_count(self):
        assert len(IssueSubtype) == 15

    def test_billing_values(self):
        assert IssueSubtype.REFUND == "refund"
        assert IssueSubtype.INVOICE_ERROR == "invoice_error"
        assert IssueSubtype.FAILED_CHARGE == "failed_charge"

    def test_technical_values(self):
        assert IssueSubtype.BUG_REPORT == "bug_report"
        assert IssueSubtype.API_ERROR == "api_error"
        assert IssueSubtype.INTEGRATION_FAILURE == "integration_failure"

    def test_account_values(self):
        assert IssueSubtype.PASSWORD_RESET == "password_reset"
        assert IssueSubtype.SSO_ISSUE == "sso_issue"
        assert IssueSubtype.ACCOUNT_LOCKOUT == "account_lockout"

    def test_security_values(self):
        assert IssueSubtype.SUSPICIOUS_LOGIN == "suspicious_login"
        assert IssueSubtype.EXPOSURE_RISK == "exposure_risk"
        assert IssueSubtype.ABUSE_REPORT == "abuse_report"

    def test_shipping_values(self):
        assert IssueSubtype.DELAY == "delay"
        assert IssueSubtype.TRACKING_PROBLEM == "tracking_problem"
        assert IssueSubtype.LOST_PACKAGE == "lost_package"


class TestQueueIdEnum:
    def test_member_count(self):
        assert len(QueueId) == 9

    def test_string_values(self):
        assert QueueId.BILLING_TEAM == "billing_team"
        assert QueueId.TECH_SUPPORT_L1 == "tech_support_l1"
        assert QueueId.TECH_SUPPORT_L2 == "tech_support_l2"
        assert QueueId.ACCOUNT_TEAM == "account_team"
        assert QueueId.SECURITY_TEAM == "security_team"
        assert QueueId.SHIPPING_TEAM == "shipping_team"
        assert QueueId.REFUND_TEAM == "refund_team"
        assert QueueId.SPAM_FILTER == "spam_filter"
        assert QueueId.SALES_OR_FEATURE_REQUESTS == "sales_or_feature_requests"


class TestImpactUrgencyPriorityEnums:
    def test_impact_values(self):
        assert Impact.SINGLE_USER == "single_user"
        assert Impact.TEAM == "team"
        assert Impact.ORG_WIDE == "org_wide"
        assert Impact.REVENUE_AFFECTING == "revenue_affecting"

    def test_urgency_values(self):
        assert Urgency.LOW == "low"
        assert Urgency.MEDIUM == "medium"
        assert Urgency.HIGH == "high"
        assert Urgency.CRITICAL == "critical"

    def test_priority_values(self):
        assert Priority.LOW == "low"
        assert Priority.MEDIUM == "medium"
        assert Priority.HIGH == "high"
        assert Priority.CRITICAL == "critical"


class TestTicketStatusEnum:
    def test_member_count(self):
        assert len(TicketStatus) == 8

    def test_string_values(self):
        assert TicketStatus.NEW == "new"
        assert TicketStatus.OPENED == "opened"
        assert TicketStatus.CLASSIFIED == "classified"
        assert TicketStatus.WAITING_FOR_INFO == "waiting_for_info"
        assert TicketStatus.ROUTED == "routed"
        assert TicketStatus.ESCALATED == "escalated"
        assert TicketStatus.MERGED == "merged"
        assert TicketStatus.CLOSED == "closed"


class TestNonActionableSubtypeEnum:
    def test_member_count(self):
        assert len(NonActionableSubtype) == 5

    def test_string_values(self):
        assert NonActionableSubtype.SPAM_MARKETING == "spam_marketing"
        assert NonActionableSubtype.BENIGN_EXPECTED == "benign_expected"
        assert NonActionableSubtype.AUTOMATION_FALSE_POSITIVE == "automation_false_positive"
        assert NonActionableSubtype.DATA_ERROR == "data_error"
        assert NonActionableSubtype.NO_RESPONSE_NEEDED == "no_response_needed"


class TestCustomerTierEnum:
    def test_member_count(self):
        assert len(CustomerTier) == 4

    def test_string_values(self):
        assert CustomerTier.FREE == "free"
        assert CustomerTier.PRO == "pro"
        assert CustomerTier.ENTERPRISE == "enterprise"
        assert CustomerTier.INTERNAL == "internal"


class TestSourceChannelEnum:
    def test_member_count(self):
        assert len(SourceChannel) == 3

    def test_string_values(self):
        assert SourceChannel.CUSTOMER_EMAIL == "customer_email"
        assert SourceChannel.INTERNAL_REPORT == "internal_report"
        assert SourceChannel.MONITORING_ALERT == "monitoring_alert"


class TestCloseReasonEnum:
    def test_member_count(self):
        assert len(CloseReason) == 5

    def test_string_values(self):
        assert CloseReason.RESOLVED == "resolved"
        assert CloseReason.DUPLICATE == "duplicate"
        assert CloseReason.NON_ACTIONABLE == "non_actionable"
        assert CloseReason.FEATURE_REQUEST == "feature_request"
        assert CloseReason.NO_RESPONSE == "no_response"


class TestTaskDifficultyEnum:
    def test_member_count(self):
        assert len(TaskDifficulty) == 3

    def test_string_values(self):
        assert TaskDifficulty.EASY == "easy"
        assert TaskDifficulty.MEDIUM == "medium"
        assert TaskDifficulty.HARD == "hard"


class TestActionTypeEnum:
    def test_member_count(self):
        assert len(ActionType) == 10

    def test_string_values(self):
        assert ActionType.OPEN_TICKET == "open_ticket"
        assert ActionType.CLASSIFY_TICKET == "classify_ticket"
        assert ActionType.SET_IMPACT_URGENCY == "set_impact_urgency"
        assert ActionType.ROUTE_TICKET == "route_ticket"
        assert ActionType.REQUEST_INFORMATION == "request_information"
        assert ActionType.ESCALATE_TICKET == "escalate_ticket"
        assert ActionType.MERGE_DUPLICATE == "merge_duplicate"
        assert ActionType.CLOSE_TICKET == "close_ticket"
        assert ActionType.SKIP_TURN == "skip_turn"
        assert ActionType.FINISH_EPISODE == "finish_episode"


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------


class TestValidFamilySubtypes:
    def test_has_five_families(self):
        assert len(VALID_FAMILY_SUBTYPES) == 5

    def test_all_families_present(self):
        assert set(VALID_FAMILY_SUBTYPES.keys()) == set(IssueFamily)

    def test_each_family_has_three_subtypes(self):
        for family, subtypes in VALID_FAMILY_SUBTYPES.items():
            assert len(subtypes) == 3, f"{family} should have 3 subtypes, got {len(subtypes)}"

    def test_billing_subtypes(self):
        expected = frozenset({IssueSubtype.REFUND, IssueSubtype.INVOICE_ERROR, IssueSubtype.FAILED_CHARGE})
        assert VALID_FAMILY_SUBTYPES[IssueFamily.BILLING] == expected

    def test_technical_subtypes(self):
        expected = frozenset({IssueSubtype.BUG_REPORT, IssueSubtype.API_ERROR, IssueSubtype.INTEGRATION_FAILURE})
        assert VALID_FAMILY_SUBTYPES[IssueFamily.TECHNICAL] == expected

    def test_account_subtypes(self):
        expected = frozenset({IssueSubtype.PASSWORD_RESET, IssueSubtype.SSO_ISSUE, IssueSubtype.ACCOUNT_LOCKOUT})
        assert VALID_FAMILY_SUBTYPES[IssueFamily.ACCOUNT] == expected

    def test_security_subtypes(self):
        expected = frozenset({IssueSubtype.SUSPICIOUS_LOGIN, IssueSubtype.EXPOSURE_RISK, IssueSubtype.ABUSE_REPORT})
        assert VALID_FAMILY_SUBTYPES[IssueFamily.SECURITY] == expected

    def test_shipping_subtypes(self):
        expected = frozenset({IssueSubtype.DELAY, IssueSubtype.TRACKING_PROBLEM, IssueSubtype.LOST_PACKAGE})
        assert VALID_FAMILY_SUBTYPES[IssueFamily.SHIPPING] == expected

    def test_no_cross_family_leakage(self):
        """Each subtype must appear in exactly one family's set."""
        seen: set[IssueSubtype] = set()
        for subtypes in VALID_FAMILY_SUBTYPES.values():
            assert subtypes.isdisjoint(seen), "Cross-family subtype leakage detected"
            seen.update(subtypes)
        assert len(seen) == 15  # all 15 subtypes covered


class TestPriorityMatrix:
    def test_has_sixteen_entries(self):
        assert len(PRIORITY_MATRIX) == 16

    def test_single_user_row(self):
        assert PRIORITY_MATRIX[(Impact.SINGLE_USER, Urgency.LOW)] == Priority.LOW
        assert PRIORITY_MATRIX[(Impact.SINGLE_USER, Urgency.MEDIUM)] == Priority.LOW
        assert PRIORITY_MATRIX[(Impact.SINGLE_USER, Urgency.HIGH)] == Priority.MEDIUM
        assert PRIORITY_MATRIX[(Impact.SINGLE_USER, Urgency.CRITICAL)] == Priority.HIGH

    def test_team_row(self):
        assert PRIORITY_MATRIX[(Impact.TEAM, Urgency.LOW)] == Priority.LOW
        assert PRIORITY_MATRIX[(Impact.TEAM, Urgency.MEDIUM)] == Priority.MEDIUM
        assert PRIORITY_MATRIX[(Impact.TEAM, Urgency.HIGH)] == Priority.HIGH
        assert PRIORITY_MATRIX[(Impact.TEAM, Urgency.CRITICAL)] == Priority.HIGH

    def test_org_wide_row(self):
        assert PRIORITY_MATRIX[(Impact.ORG_WIDE, Urgency.LOW)] == Priority.MEDIUM
        assert PRIORITY_MATRIX[(Impact.ORG_WIDE, Urgency.MEDIUM)] == Priority.HIGH
        assert PRIORITY_MATRIX[(Impact.ORG_WIDE, Urgency.HIGH)] == Priority.HIGH
        assert PRIORITY_MATRIX[(Impact.ORG_WIDE, Urgency.CRITICAL)] == Priority.CRITICAL

    def test_revenue_affecting_row(self):
        assert PRIORITY_MATRIX[(Impact.REVENUE_AFFECTING, Urgency.LOW)] == Priority.HIGH
        assert PRIORITY_MATRIX[(Impact.REVENUE_AFFECTING, Urgency.MEDIUM)] == Priority.HIGH
        assert PRIORITY_MATRIX[(Impact.REVENUE_AFFECTING, Urgency.HIGH)] == Priority.CRITICAL
        assert PRIORITY_MATRIX[(Impact.REVENUE_AFFECTING, Urgency.CRITICAL)] == Priority.CRITICAL


class TestGatedQueues:
    def test_contains_l2_and_security(self):
        assert QueueId.TECH_SUPPORT_L2 in GATED_QUEUES
        assert QueueId.SECURITY_TEAM in GATED_QUEUES

    def test_exactly_two_members(self):
        assert len(GATED_QUEUES) == 2

    def test_is_frozenset(self):
        assert isinstance(GATED_QUEUES, frozenset)


class TestPriorityWeights:
    def test_all_priorities_present(self):
        assert set(PRIORITY_WEIGHTS.keys()) == set(Priority)

    def test_weight_values(self):
        assert PRIORITY_WEIGHTS[Priority.LOW] == 0.5
        assert PRIORITY_WEIGHTS[Priority.MEDIUM] == 1.0
        assert PRIORITY_WEIGHTS[Priority.HIGH] == 1.5
        assert PRIORITY_WEIGHTS[Priority.CRITICAL] == 2.0


# ---------------------------------------------------------------------------
# derive_priority helper
# ---------------------------------------------------------------------------


class TestDerivePriority:
    def test_exhaustive_all_16_combinations(self):
        for (impact, urgency), expected in PRIORITY_MATRIX.items():
            result = derive_priority(impact, urgency)
            assert result == expected, f"derive_priority({impact}, {urgency}) = {result}, expected {expected}"

    def test_returns_priority_type(self):
        result = derive_priority(Impact.SINGLE_USER, Urgency.LOW)
        assert isinstance(result, Priority)


# ---------------------------------------------------------------------------
# Standalone Pydantic models
# ---------------------------------------------------------------------------


class TestInboxSummaryItem:
    def _make(self, **overrides):
        base = {
            "ticket_id": "T001",
            "subject": "Can't log in",
            "sender_email": "user@example.com",
            "received_at": "2026-04-01T10:00:00Z",
            "status": TicketStatus.NEW,
            "customer_tier": CustomerTier.PRO,
            "has_attachment": False,
            "sla_remaining_minutes": 120,
            "short_preview": "Hi, I can't log in to my account.",
        }
        base.update(overrides)
        return InboxSummaryItem(**base)

    def test_construction(self):
        item = self._make()
        assert item.ticket_id == "T001"
        assert item.status == TicketStatus.NEW
        assert item.has_attachment is False

    def test_sla_remaining_minutes_none(self):
        item = self._make(sla_remaining_minutes=None)
        assert item.sla_remaining_minutes is None

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(unknown_field="x")


class TestFocusedTicket:
    def _make(self, **overrides):
        base = {
            "ticket_id": "T001",
            "subject": "Login failure",
            "latest_message": "I still cannot log in.",
            "thread_history": [{"role": "user", "content": "Help!", "timestamp": "2026-04-01T10:00:00Z"}],
            "attachments": [],
            "visible_internal_notes": [],
            "prior_actions_taken": [],
        }
        base.update(overrides)
        return FocusedTicket(**base)

    def test_construction(self):
        ft = self._make()
        assert ft.ticket_id == "T001"
        assert ft.thread_history[0]["role"] == "user"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(bogus="x")


class TestRoutingPolicyCard:
    def test_construction(self):
        card = RoutingPolicyCard(
            queue_id=QueueId.BILLING_TEAM,
            description="Handles billing issues",
            prerequisites=[],
            handles_families=[IssueFamily.BILLING],
        )
        assert card.queue_id == QueueId.BILLING_TEAM
        assert IssueFamily.BILLING in card.handles_families


class TestSlaPolicyCard:
    def test_construction(self):
        card = SlaPolicyCard(
            tier=CustomerTier.ENTERPRISE,
            response_deadline_minutes=60,
            resolution_deadline_minutes=480,
        )
        assert card.tier == CustomerTier.ENTERPRISE
        assert card.response_deadline_minutes == 60


# ---------------------------------------------------------------------------
# TriageSieveAction
# ---------------------------------------------------------------------------


class TestTriageSieveAction:
    def test_minimal_construction(self):
        action = TriageSieveAction(action_type=ActionType.OPEN_TICKET, ticket_id="T001")
        assert action.action_type == ActionType.OPEN_TICKET
        assert action.ticket_id == "T001"

    def test_all_optional_fields_default_to_none(self):
        action = TriageSieveAction(action_type=ActionType.SKIP_TURN)
        assert action.ticket_id is None
        assert action.issue_family is None
        assert action.issue_subtype is None
        assert action.impact is None
        assert action.urgency is None
        assert action.queue_id is None
        assert action.reason_code is None
        assert action.template_id is None
        assert action.requested_fields is None
        assert action.target_ticket_id is None
        assert action.close_reason is None

    def test_classify_action_fields(self):
        action = TriageSieveAction(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id="T001",
            issue_family=IssueFamily.BILLING,
            issue_subtype=IssueSubtype.REFUND,
        )
        assert action.issue_family == IssueFamily.BILLING
        assert action.issue_subtype == IssueSubtype.REFUND

    def test_route_action_fields(self):
        action = TriageSieveAction(
            action_type=ActionType.ROUTE_TICKET,
            ticket_id="T001",
            queue_id=QueueId.BILLING_TEAM,
        )
        assert action.queue_id == QueueId.BILLING_TEAM

    def test_close_action_fields(self):
        action = TriageSieveAction(
            action_type=ActionType.CLOSE_TICKET,
            ticket_id="T001",
            close_reason=CloseReason.RESOLVED,
            template_id="tmpl_001",
        )
        assert action.close_reason == CloseReason.RESOLVED

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TriageSieveAction(action_type=ActionType.SKIP_TURN, nonexistent_field="x")

    def test_json_round_trip(self):
        action = TriageSieveAction(
            action_type=ActionType.CLASSIFY_TICKET,
            ticket_id="T001",
            issue_family=IssueFamily.BILLING,
            issue_subtype=IssueSubtype.REFUND,
        )
        data = json.loads(action.model_dump_json())
        restored = TriageSieveAction.model_validate(data)
        assert restored.action_type == action.action_type
        assert restored.issue_family == action.issue_family

    def test_metadata_field_present(self):
        action = TriageSieveAction(action_type=ActionType.SKIP_TURN)
        assert isinstance(action.metadata, dict)


# ---------------------------------------------------------------------------
# TriageSieveObservation
# ---------------------------------------------------------------------------


def _make_inbox_item(ticket_id: str = "T001") -> InboxSummaryItem:
    return InboxSummaryItem(
        ticket_id=ticket_id,
        subject="Test ticket",
        sender_email="user@example.com",
        received_at="2026-04-01T10:00:00Z",
        status=TicketStatus.NEW,
        customer_tier=CustomerTier.FREE,
        has_attachment=False,
        sla_remaining_minutes=None,
        short_preview="Test body.",
    )


class TestTriageSieveObservation:
    def _make(self, **overrides):
        base = {
            "inbox_summaries": [_make_inbox_item()],
            "focused_ticket": None,
            "available_templates": [],
            "allowed_queues": [QueueId.BILLING_TEAM],
            "routing_policy_cards": [],
            "sla_policy_cards": [],
            "legal_actions": [ActionType.OPEN_TICKET],
            "action_budget_remaining": 4,
            "step_count": 0,
            "current_time": "2026-04-01T10:00:00Z",
            "last_action_result": "ok",
            "task_difficulty": TaskDifficulty.EASY,
            "hint": None,
        }
        base.update(overrides)
        return TriageSieveObservation(**base)

    def test_construction(self):
        obs = self._make()
        assert obs.done is False
        assert obs.reward is None
        assert obs.step_count == 0
        assert obs.task_difficulty == TaskDifficulty.EASY

    def test_done_and_reward_inherited(self):
        obs = self._make(done=True, reward=0.85)
        assert obs.done is True
        assert obs.reward == 0.85

    def test_hint_present(self):
        obs = self._make(hint="Check thread history for order identifier")
        assert obs.hint == "Check thread history for order identifier"

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            self._make(nonexistent="x")

    def test_json_round_trip(self):
        obs = self._make()
        data = json.loads(obs.model_dump_json())
        restored = TriageSieveObservation.model_validate(data)
        assert restored.task_difficulty == obs.task_difficulty
        assert restored.action_budget_remaining == obs.action_budget_remaining


# ---------------------------------------------------------------------------
# TriageSieveState
# ---------------------------------------------------------------------------


class TestTriageSieveState:
    def _make(self, **overrides):
        base = {
            "task_difficulty": TaskDifficulty.MEDIUM,
            "seed": 42,
            "total_tickets": 2,
            "action_budget": 8,
            "action_budget_remaining": 8,
            "mode": "eval_strict",
            "tickets_summary": [{"ticket_id": "T001", "status": "new", "gold_priority": "high"}],
        }
        base.update(overrides)
        return TriageSieveState(**base)

    def test_construction(self):
        state = self._make()
        assert state.seed == 42
        assert state.total_tickets == 2
        assert state.mode == "eval_strict"

    def test_inherited_fields(self):
        state = self._make()
        assert state.episode_id is None
        assert state.step_count == 0

    def test_episode_id_settable(self):
        state = self._make()
        state.episode_id = "ep_001"
        assert state.episode_id == "ep_001"

    def test_extra_fields_allowed_by_state_base(self):
        # State base uses extra='allow'
        state = self._make(extra_debug_field="debug_value")
        assert state.extra_debug_field == "debug_value"  # type: ignore[attr-defined]

    def test_json_round_trip(self):
        state = self._make()
        data = json.loads(state.model_dump_json())
        restored = TriageSieveState.model_validate(data)
        assert restored.seed == state.seed
        assert restored.mode == state.mode


# ---------------------------------------------------------------------------
# HiddenTicketTruth dataclass
# ---------------------------------------------------------------------------


class TestHiddenTicketTruth:
    def _make(self, **overrides):
        base = {
            "ticket_id": "T001",
            "customer_tier": CustomerTier.PRO,
            "source_channel": SourceChannel.CUSTOMER_EMAIL,
            "issue_family": IssueFamily.BILLING,
            "issue_subtype": IssueSubtype.REFUND,
            "product_area": "payments",
            "impact": Impact.SINGLE_USER,
            "urgency": Urgency.MEDIUM,
            "priority": Priority.LOW,
            "required_queue": QueueId.REFUND_TEAM,
            "required_missing_fields": ["order_id"],
            "escalation_required": False,
            "escalation_target": None,
            "is_duplicate": False,
            "duplicate_of": None,
            "sla_response_deadline": 240,
            "sla_resolution_deadline": 1440,
            "policy_graph_id": "refund_missing_order_id",
            "correct_template_ids": ["tmpl_refund_ack"],
            "gold_terminal_status": TicketStatus.CLOSED,
            "non_actionable_subtype": None,
        }
        base.update(overrides)
        return HiddenTicketTruth(**base)

    def test_construction(self):
        truth = self._make()
        assert truth.ticket_id == "T001"
        assert truth.issue_family == IssueFamily.BILLING
        assert truth.priority == Priority.LOW
        assert truth.required_missing_fields == ["order_id"]

    def test_fields_mutable(self):
        """HiddenTicketTruth is a plain (non-frozen) dataclass."""
        truth = self._make()
        truth.priority = Priority.HIGH
        assert truth.priority == Priority.HIGH

    def test_optional_fields_none(self):
        truth = self._make()
        assert truth.escalation_target is None
        assert truth.duplicate_of is None
        assert truth.non_actionable_subtype is None

    def test_with_non_actionable(self):
        truth = self._make(non_actionable_subtype=NonActionableSubtype.SPAM_MARKETING)
        assert truth.non_actionable_subtype == NonActionableSubtype.SPAM_MARKETING

    def test_asdict_serializable(self):
        """asdict should work without error (values are enums / primitives / lists)."""
        truth = self._make()
        d = asdict(truth)
        assert d["ticket_id"] == "T001"
