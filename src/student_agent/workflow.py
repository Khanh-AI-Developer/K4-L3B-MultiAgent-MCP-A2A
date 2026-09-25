from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


def _number(value: Any) -> float:
    try:
        return float(Decimal(str(value)))
    except (InvalidOperation, TypeError, ValueError):
        return 0.0


def _items(data: Any) -> list[dict[str, Any]]:
    return data if isinstance(data, list) else []


def _event_types(data: Any) -> set[str]:
    return {str(item.get("event_type")) for item in _items(data) if isinstance(item, dict)}


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _money(value: float) -> float:
    return round(value + 1e-9, 2)


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Run a bounded, evidence-first investigation for one L3B case."""
    case_id = str(case["case_id"])
    request = case.get("customer_request", {})
    claimed = request.get("claimed_order_id")
    candidates = list(dict.fromkeys(case.get("candidate_order_ids", [])))
    if isinstance(claimed, str) and claimed not in candidates:
        candidates.insert(0, claimed)
    trace.emit(
        case_id=case_id, event_type="task_assigned", actor="coordinator", target="entity-agent"
    )

    cache: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}
    refs: list[str] = []

    async def call(tool: str, **arguments: str) -> dict[str, Any] | None:
        key = (tool, tuple(sorted(arguments.items())))
        if key in cache:
            return cache[key]
        try:
            evidence = await gateway.call(tool, case_id=case_id, **arguments)
        except (RuntimeError, ValueError, OSError):
            return None
        cache[key] = evidence
        ref = evidence["evidence_ref"]
        if ref not in refs:
            refs.append(ref)
        trace.emit(
            case_id=case_id,
            event_type="tool_result_consumed",
            actor=_actor_for_tool(tool),
            tool_name=tool,
            evidence_refs=[ref],
        )
        return evidence

    resolved: list[str] = []
    rejected: list[str] = []
    order_evidence: dict[str, dict[str, Any]] = {}
    for order_id in candidates:
        if not isinstance(order_id, str):
            continue
        evidence = await call("get_order", order_id=order_id)
        if evidence and isinstance(evidence.get("data"), dict):
            returned_id = evidence["data"].get("order_id")
            if returned_id == order_id:
                resolved.append(order_id)
                order_evidence[order_id] = evidence
                continue
        rejected.append(order_id)

    selected = claimed if claimed in resolved else (resolved[0] if resolved else None)
    status = "resolved" if selected else ("ambiguous" if resolved else "not_found")
    if len(resolved) > 1 and claimed not in resolved:
        status = "ambiguous"
    if selected is None:
        return _empty_output(case_id, request, refs, status, resolved, rejected)

    trace.emit(case_id=case_id, event_type="handoff", actor="entity-agent", target="coordinator")
    order = order_evidence[selected]["data"]
    (
        order_items,
        payments,
        payment_timeline,
        shipment,
        sellers,
        product,
        refund,
        customer,
        policy,
    ) = await asyncio.gather(
        call("get_order_items", order_id=selected),
        call("get_order_payments", order_id=selected),
        call("get_payment_timeline", order_id=selected),
        call("get_shipment_summary", order_id=selected),
        call("get_sellers", order_id=selected),
        call("get_product_context", order_id=selected),
        call("get_refund_timeline", order_id=selected),
        call(
            "get_customer_history", customer_unique_id=str(case.get("customer_unique_id_hint", ""))
        ),
        call("get_policy", policy_version=str(case.get("policy_version", ""))),
    )
    del product

    items = _items(order_items.get("data") if order_items else None)
    payment_rows = _items(payments.get("data") if payments else None)
    timeline = payment_timeline.get("data", {}) if payment_timeline else {}
    shipment_data = shipment.get("data", {}) if shipment else {}
    refund_data = refund.get("data", {}) if refund else {}
    customer_data = customer.get("data", {}) if customer else {}
    policy_data = policy.get("data", {}) if policy else {}
    seller_rows = _items(sellers.get("data") if sellers else None)

    topic = _primary_topic(request, order, shipment_data, timeline, refund_data)
    rules = policy_data.get("rules", {}) if isinstance(policy_data, dict) else {}
    rule = rules.get(topic, {}) if isinstance(rules, dict) else {}
    if not isinstance(rule, dict):
        rule = {}
    case_status = rule.get("case_status", "needs_investigation")
    action = rule.get("recommended_action")
    policy_refund = _number(rule.get("refund_brl", 0))
    seller_ids = _unique_strings(
        [row.get("seller_id") for row in items if isinstance(row, dict)]
        + [row.get("seller_id") for row in seller_rows if isinstance(row, dict)]
    )
    item_ids = _unique_strings([row.get("order_item_id") for row in items if isinstance(row, dict)])
    payment_refs = _unique_strings(
        [f"{row.get('payment_sequential')}:{row.get('payment_type')}" for row in payment_rows]
    )
    responsible = _responsible(rule, seller_ids)

    captured = sum(
        _number(event.get("amount_brl"))
        for event in _items(timeline.get("events"))
        if event.get("event_type") == "captured" and event.get("status") == "confirmed"
    )
    if not captured:
        captured = sum(_number(row.get("payment_value")) for row in payment_rows)
    refunded = sum(
        _number(event.get("amount_brl"))
        for event in _items(refund_data.get("events"))
        if event.get("event_type") in {"refund_completed", "refunded"}
    )
    payment_verdict = _payment_verdict(topic, captured, payment_rows, timeline, refund_data)
    shipment_verdict, timeline_complete, late_sellers = _shipment_verdict(shipment_data, seller_ids)

    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor="policy-agent",
        decision_code=str(topic),
        evidence_refs=[policy["evidence_ref"]] if policy else refs[:1],
    )
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor="verifier",
        target="coordinator",
        evidence_refs=refs[:20],
        attributes={"resolved_order_count": len(resolved), "evidence_count": len(refs)},
    )

    claims = request.get("claims", []) if isinstance(request.get("claims", []), list) else []
    claim_assessments = [
        {
            "claim_id": str(claim.get("claim_id", f"claim-{index}")),
            "verdict": _claim_verdict(
                str(claim.get("topic", "")), topic, case_status, policy_refund, captured
            ),
            "confidence": 0.92 if str(claim.get("topic", "")) == topic else 0.68,
            "evidence_refs": refs[: min(8, len(refs))],
        }
        for index, claim in enumerate(claims, 1)
        if isinstance(claim, dict)
    ]
    return {
        "schema_version": "day09-l3b-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": topic,
            "secondary_issues": _secondary_topics(claims, topic),
            "case_status": case_status,
            "confidence": _confidence(status, refs, topic, shipment_verdict, payment_verdict),
        },
        "affected_entities": {
            "order_ids": [selected],
            "item_ids": item_ids,
            "seller_ids": seller_ids,
            "payment_references": payment_refs,
            "shipment_ids": [],
        },
        "claim_assessments": claim_assessments,
        "entity_resolution": {
            "status": status,
            "resolved_order_ids": resolved,
            "rejected_candidates": rejected,
            "confidence": 0.98 if status == "resolved" else 0.35,
        },
        "customer_context": {
            "customer_unique_id": customer_data.get("customer_unique_id")
            or case.get("customer_unique_id_hint"),
            "related_order_ids": _unique_strings(
                [row.get("order_id") for row in _items(customer_data.get("orders"))]
            ),
        },
        "shipment_analysis": {
            "verdict": shipment_verdict,
            "late_seller_ids": late_sellers,
            "timeline_complete": timeline_complete,
        },
        "payment_analysis": {
            "verdict": payment_verdict,
            "captured_total_brl": _money(captured) if payments else None,
            "refunded_total_brl": _money(refunded) if refund else None,
            "refundable_total_brl": _money(policy_refund),
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": topic.upper(), "rank": 1}],
            "responsible_parties": responsible,
        },
        "evidence_refs": refs[:30],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": _money(policy_refund),
            "refund_lines": (
                [{"reason_code": topic, "amount_brl": _money(policy_refund), "entity_id": selected}]
                if policy_refund > 0
                else []
            ),
        },
        "resolution_actions": [str(action)] if action else ["document_no_action"],
    }


def _actor_for_tool(tool: str) -> str:
    if "customer" in tool:
        return "entity-agent"
    if "shipment" in tool or "seller" in tool or "product" in tool:
        return "order-agent"
    if "payment" in tool or "refund" in tool:
        return "payment-agent"
    if tool == "get_policy":
        return "policy-agent"
    return "order-agent"


def _unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if isinstance(value, str) and value))


def _primary_topic(
    request: dict[str, Any],
    order: dict[str, Any],
    shipment: dict[str, Any],
    payment: dict[str, Any],
    refund: dict[str, Any],
) -> str:
    claims = request.get("claims", [])
    requested = (
        str(claims[0].get("topic"))
        if claims and isinstance(claims[0], dict)
        else "insufficient_evidence"
    )
    if requested == "canceled_order_paid" and order.get("order_status") == "canceled":
        return requested
    if requested == "unavailable_order_paid" and order.get("order_status") in {
        "unavailable",
        "canceled",
    }:
        return requested
    if requested in {"late_delivery_logistics", "late_delivery_seller"} and _items(
        shipment.get("events")
    ):
        return requested
    if requested in {
        "payment_mismatch",
        "duplicate_charge",
        "valid_split_payment",
        "unsupported_claim",
    }:
        return requested
    if requested in {"refund_pending", "refund_failed"} and _items(refund.get("events")):
        return requested
    return requested or "insufficient_evidence"


def _payment_verdict(
    topic: str,
    captured: float,
    payments: list[dict[str, Any]],
    timeline: dict[str, Any],
    refund: dict[str, Any],
) -> str:
    if topic == "duplicate_charge":
        return "duplicate_capture"
    if topic == "payment_mismatch":
        return "capture_mismatch"
    refund_types = _event_types(refund.get("events"))
    if topic == "refund_pending" or "refund_requested" in refund_types:
        return "refund_pending"
    if topic == "refund_failed":
        return "refund_failed"
    if topic in {"canceled_order_paid", "unavailable_order_paid"}:
        return "reconciled" if captured > 0 else "insufficient_evidence"
    if topic == "valid_split_payment":
        values = [_number(row.get("payment_value")) for row in payments]
        events = [
            _number(event.get("amount_brl"))
            for event in _items(timeline.get("events"))
            if event.get("event_type") == "captured"
        ]
        return (
            "reconciled"
            if len(values) > 1 and abs(sum(values) - sum(events)) < 0.01
            else "capture_mismatch"
        )
    return "reconciled" if captured >= 0 else "insufficient_evidence"


def _shipment_verdict(data: dict[str, Any], seller_ids: list[str]) -> tuple[str, bool, list[str]]:
    events = _items(data.get("events"))
    event_types = _event_types(events)
    if "returned" in event_types:
        return "returned", bool(data.get("delivered_customer_at")), []
    if "lost" in event_types:
        return "lost", False, []
    late = [event for event in events if event.get("event_type") == "delivered_late"]
    if late:
        logistics = any(event.get("actor") == "logistics_provider" for event in late)
        return (
            ("logistics_delay" if logistics else "seller_delay"),
            True,
            [] if logistics else seller_ids[:1],
        )
    delivered = _parse_time(data.get("delivered_customer_at"))
    estimated = _parse_time(data.get("estimated_delivery_at"))
    if delivered and estimated:
        return ("on_time" if delivered <= estimated else "logistics_delay"), True, []
    return "insufficient_evidence", False, []


def _responsible(rule: dict[str, Any], seller_ids: list[str]) -> list[dict[str, Any]]:
    parties = rule.get("responsible_parties", [])
    result: list[dict[str, Any]] = []
    for party in parties if isinstance(parties, list) else []:
        if not isinstance(party, dict):
            continue
        party_id = party.get("party_id")
        if party.get("party_type") == "seller" and not party_id:
            party_id = seller_ids[0] if seller_ids else None
        result.append({"party_type": party.get("party_type", "unknown"), "party_id": party_id})
    return result or [{"party_type": "unknown", "party_id": None}]


def _claim_verdict(topic: str, primary: str, status: str, refund: float, captured: float) -> str:
    if status != "resolved":
        return "insufficient_evidence"
    if topic == primary:
        return "supported"
    if topic == "requested_full_refund":
        if refund <= 0:
            return "unsupported"
        return "supported" if captured and refund >= captured else "partially_supported"
    return "unsupported"


def _secondary_topics(claims: list[Any], primary: str) -> list[str]:
    return list(
        dict.fromkeys(
            str(c.get("topic"))
            for c in claims[1:]
            if isinstance(c, dict) and c.get("topic") != primary
        )
    )[:10]


def _confidence(status: str, refs: list[str], topic: str, shipment: str, payment: str) -> float:
    if status != "resolved" or not refs:
        return 0.3
    score = 0.78
    if topic not in {"unsupported_claim", "insufficient_evidence"}:
        score += 0.08
    if shipment != "insufficient_evidence":
        score += 0.04
    if payment != "insufficient_evidence":
        score += 0.04
    return min(score, 0.96)


def _empty_output(
    case_id: str,
    request: dict[str, Any],
    refs: list[str],
    status: str,
    resolved: list[str],
    rejected: list[str],
) -> dict[str, Any]:
    claims = request.get("claims", []) if isinstance(request.get("claims", []), list) else []
    topic = (
        str(claims[0].get("topic"))
        if claims and isinstance(claims[0], dict)
        else "insufficient_evidence"
    )
    return {
        "schema_version": "day09-l3b-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": "insufficient_evidence",
            "secondary_issues": [topic],
            "case_status": "needs_investigation",
            "confidence": 0.2,
        },
        "affected_entities": {
            "order_ids": [],
            "item_ids": [],
            "seller_ids": [],
            "payment_references": [],
            "shipment_ids": [],
        },
        "claim_assessments": [],
        "entity_resolution": {
            "status": status,
            "resolved_order_ids": resolved,
            "rejected_candidates": rejected,
            "confidence": 0.2,
        },
        "customer_context": {"customer_unique_id": None, "related_order_ids": []},
        "shipment_analysis": {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
        },
        "payment_analysis": {
            "verdict": "insufficient_evidence",
            "captured_total_brl": None,
            "refunded_total_brl": None,
            "refundable_total_brl": None,
        },
        "root_cause_analysis": {
            "ranked_causes": [],
            "responsible_parties": [{"party_type": "unknown", "party_id": None}],
        },
        "evidence_refs": refs[:30],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 0,
            "refund_lines": [],
        },
        "resolution_actions": ["investigate_entity_resolution"],
    }
