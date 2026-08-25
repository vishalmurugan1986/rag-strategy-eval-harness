"""
Synthetic corpus: an internal engineering knowledge base for a fictional
company ("Northwind Systems"). Docs deliberately share vocabulary across
topics (e.g. "rollback" appears in both deployment and incident docs) so
retrieval strategies actually get stress-tested instead of trivially
matching on unique keywords.

Each doc is (doc_id, title, content). Content uses markdown headers so the
"semantic" chunker has real structure to split on.
"""

DOCS = [
    ("onboarding_eng", "Engineering Onboarding Guide", """
# Engineering Onboarding Guide

## First Week
New engineers get access to the internal tools on day one: GitHub, the
deploy dashboard, and the incident channel in Slack. Your manager will
assign a buddy for your first two weeks.

## Development Environment
Northwind uses a monorepo. Clone it, run `make bootstrap`, and it installs
all dependencies for the services you have access to. The bootstrap script
also sets up pre-commit hooks that run linting and unit tests.

## Access Requests
Access to production systems requires manager approval plus security team
sign-off. Requests go through the internal access-request tool, not Slack
DMs. Approval typically takes 1-2 business days.

## Your First PR
New engineers are expected to submit a first PR within the first week,
usually a small bug fix or documentation update. All PRs require at least
one approval before merging, and two approvals for changes touching the
payments service.
"""),

    ("deploy_guide", "Deployment Guide", """
# Deployment Guide

## Standard Deployment Process
Deployments happen through the CI/CD pipeline. Merging to `main` triggers a
staging deployment automatically. Production deployment requires a manual
approval step in the deploy dashboard from someone other than the PR author.

## Rollback Procedure
If a deployment causes errors, use the "Rollback" button in the deploy
dashboard, which reverts to the last known-good build within about 2
minutes. For database migrations that can't be cleanly rolled back, follow
the manual rollback runbook instead — do not use the automatic rollback for
migration-related deploys.

## Deployment Windows
Standard deploys can happen any weekday. Friday deploys after 2pm are
discouraged unless urgent, since on-call coverage is thinner over the
weekend. Deploys to the payments service require a change-management ticket
regardless of day.

## Canary Deploys
High-risk changes should go through canary deployment: 5% of traffic for 30
minutes, then 50%, then 100%, with automated rollback if error rates exceed
1% at any stage.
"""),

    ("incident_postmortem_db", "Postmortem: Database Connection Pool Exhaustion", """
# Postmortem: Database Connection Pool Exhaustion (2024-03-14)

## Summary
The checkout service experienced a 40-minute partial outage due to database
connection pool exhaustion following a deploy that introduced a connection
leak in the retry logic.

## Timeline
14:02 — Deploy to checkout service completes.
14:15 — Error rate begins climbing.
14:22 — On-call engineer paged, begins investigation.
14:30 — Root cause identified as unclosed connections in the retry path.
14:35 — Rollback initiated via the deploy dashboard.
14:42 — Error rates return to baseline.

## Root Cause
The retry logic added in the deploy opened a new database connection on
each retry attempt but only closed the connection on success, leaking
connections on any retried failure.

## Remediation
- Immediate: rollback (completed).
- Short-term: added a connection pool utilization alert at 80% capacity.
- Long-term: added a lint rule that flags connection acquisition without a
  corresponding `finally` block or context manager.
"""),

    ("incident_postmortem_api", "Postmortem: API Gateway Rate Limit Misconfiguration", """
# Postmortem: API Gateway Rate Limit Misconfiguration (2024-06-02)

## Summary
A configuration change to the API gateway's rate limiter caused legitimate
traffic from the mobile app to be throttled, resulting in a 22-minute
partial outage affecting roughly 15% of mobile users.

## Timeline
09:10 — Rate limit config change deployed.
09:14 — Mobile team reports elevated 429 error rates.
09:20 — On-call confirms the rate limiter config as the cause.
09:28 — Rollback of the config change initiated.
09:32 — Error rates return to baseline.

## Root Cause
The new rate limit was set per-IP instead of per-API-key, which
disproportionately throttled mobile users behind carrier-grade NAT sharing
IP addresses.

## Remediation
- Immediate: rollback (completed).
- Short-term: reverted to per-API-key rate limiting.
- Long-term: added a staging load test that simulates NAT-shared IP traffic
  before any rate limiter config changes ship to production.
"""),

    ("architecture_decision_db", "ADR-014: Migrating to Connection Pooling via PgBouncer", """
# ADR-014: Migrating to Connection Pooling via PgBouncer

## Context
As service count grew, direct database connections from each service
instance began exhausting the database's max connection limit during
traffic spikes.

## Decision
Adopt PgBouncer as a connection pooler in front of the primary database,
using transaction-level pooling mode for most services and session-level
pooling for services relying on session-scoped features like advisory
locks.

## Consequences
Positive: connection count from the database's perspective drops
significantly, and services can scale horizontally without each new
instance consuming a full connection slot.
Negative: transaction-level pooling is incompatible with some
session-scoped Postgres features, requiring careful auditing of each
service before migration.

## Status
Adopted. Rolled out to all services except the payments service, which
remains on session-level pooling due to its use of advisory locks for
distributed locking.
"""),

    ("architecture_decision_api", "ADR-021: API Gateway Rate Limiting Strategy", """
# ADR-021: API Gateway Rate Limiting Strategy

## Context
Following the June 2024 rate-limit incident, the team needed a documented
standard for how rate limits are configured going forward.

## Decision
All rate limits at the API gateway are keyed by API key, not by IP address.
Services without individual API keys (internal service-to-service calls)
are exempted from gateway-level rate limiting and instead rely on
per-service concurrency limits.

## Consequences
Positive: avoids penalizing users who share IP addresses (e.g. behind
carrier NAT or corporate proxies).
Negative: requires every client-facing integration to have a provisioned
API key, adding a small amount of onboarding friction for new integration
partners.

## Status
Adopted as of June 2024, following the API Gateway Rate Limit
Misconfiguration incident.
"""),

    ("security_policy", "Security Policy Overview", """
# Security Policy Overview

## Access Control
Production access follows least-privilege: engineers get access only to
the services they work on, requested via the access-request tool and
requiring both manager and security sign-off.

## Secrets Management
All secrets (API keys, database credentials) are stored in the internal
secrets vault, never in code or environment files committed to the
monorepo. Secrets are rotated automatically every 90 days.

## Incident Response
Any suspected security incident should be reported immediately to the
security channel, not the general incident channel. Security incidents
follow a separate, stricter postmortem process with mandatory legal review
before publication.

## Vulnerability Disclosure
External security researchers can report vulnerabilities through the
bug-bounty program. Internal engineers who discover vulnerabilities should
file directly with the security team rather than opening a public GitHub
issue.
"""),

    ("api_style_guide", "API Style Guide", """
# API Style Guide

## Naming Conventions
Endpoints use plural nouns for collections (`/orders`, not `/order`) and
kebab-case for multi-word resources (`/order-items`).

## Versioning
All public APIs are versioned in the URL path (`/v1/orders`). Breaking
changes require a new version; additive changes (new optional fields) do
not.

## Error Responses
Errors follow a consistent shape: `{"error": {"code": "...", "message":
"..."}}`. HTTP status codes follow standard semantics — 429 for rate
limiting, 409 for conflicts, 422 for validation errors.

## Rate Limits
Public API rate limits are documented per-endpoint in the API reference and
enforced at the gateway level, keyed by API key per ADR-021.
"""),

    ("oncall_runbook", "On-Call Runbook", """
# On-Call Runbook

## Getting Paged
On-call engineers carry the pager for one week at a time. Pages come
through the incident channel and the paging app simultaneously.

## Triage Steps
1. Acknowledge the page within 5 minutes.
2. Check the deploy dashboard for any recent deploys that correlate with
   the alert.
3. If a recent deploy is the likely cause, rollback first and investigate
   root cause after service is restored.
4. If no recent deploy correlates, check the standard dashboards (error
   rate, latency, database connection pool utilization).

## Escalation
If you can't resolve or mitigate within 20 minutes, escalate to the
secondary on-call and post an update in the incident channel. Database and
payments-related incidents can escalate directly to the relevant team lead
regardless of time of day.

## Postmortems
Any incident causing customer-visible impact requires a postmortem within
3 business days, using the standard postmortem template.
"""),

    ("payments_service_overview", "Payments Service Overview", """
# Payments Service Overview

## Purpose
The payments service handles all transaction processing, including
authorization, capture, and refunds. It integrates with two external
payment processors for redundancy.

## Special Handling
Because of its sensitivity, the payments service has stricter requirements
than other services: two-approval PRs, change-management tickets for all
deploys, and session-level database connection pooling (see ADR-014) rather
than transaction-level pooling used elsewhere.

## On-Call
Payments has a dedicated on-call rotation separate from general
engineering on-call, due to the regulatory and financial sensitivity of
issues in this service.

## Testing
All changes to the payments service require passing the full
transaction-simulation test suite, which replays a fixed set of historical
transaction patterns against a sandboxed processor before any deploy is
approved.
"""),
]
