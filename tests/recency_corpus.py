"""A corpus for observing the transaction-time recency signal.

A recency effect shows up as a difference between a search with ``recency_now``
and one without it.  Whether that difference is *attributable* to recency is the
harder half: documents that also differ in text or priority feed the lexical and
priority arms of the fusion, so the expected order has to be reasoned about
rather than read off, and a wrong expectation is indistinguishable from a wrong
implementation.  The corpus here is three thoughts whose only differing *scored*
field is the transaction timestamp, which makes the expected order unambiguous
in both directions.  Their identifiers differ too, necessarily; that difference
carries no score.

Timestamps are supplied explicitly rather than produced by spacing the writes
in real time.  That keeps the expected order independent of wall-clock
resolution, and it keeps :data:`RECENCY_NOW` unambiguously *after* every write:
engrava clamps a negative age to zero, so a "now" preceding the writes would
tie every decay and let a recency assertion pass on whatever fallback ordering
the ranker applies to tied scores.

What a control search over this corpus must show is stated relationally, never
as an exact order: the scores tie, the whole corpus is present, and the order is
**not** :data:`RECENCY_EXPECTED_ORDER`.  Which order the ranker does produce for
tied scores is engrava's business and is not pinned here — pinning it would
couple these tests to an incidental tie-break inside the supported version
range without adding any discrimination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from engrava import CoreThoughtRecord, LifecycleStatus, Priority, ThoughtType

if TYPE_CHECKING:
    from engrava import SqliteEngravaCore

#: The corpus: identifier and transaction timestamp, a day apart, oldest first
#: (which is also the insertion order).
RECENCY_CORPUS = (
    ("recency-oldest", "2026-07-01T00:00:00+00:00"),
    ("recency-middle", "2026-07-02T00:00:00+00:00"),
    ("recency-newest", "2026-07-03T00:00:00+00:00"),
)

#: A query matching every thought in the corpus and nothing else.
RECENCY_QUERY = "chronometer"

#: "Now" for the transaction-time recency signal — after every write above.
RECENCY_NOW = "2026-07-20T00:00:00Z"

#: Every identifier in the corpus, for asserting a control result is complete.
RECENCY_THOUGHT_IDS = [thought_id for thought_id, _ in RECENCY_CORPUS]

#: The order the corpus ranks in when ``recency_now`` is scored: newest first.
RECENCY_EXPECTED_ORDER = ["recency-newest", "recency-middle", "recency-oldest"]


async def seed_recency_corpus(store: SqliteEngravaCore) -> None:
    """Seed :data:`RECENCY_CORPUS` into a store.

    Args:
        store: The store to seed.

    """
    for thought_id, written_at in RECENCY_CORPUS:
        await store.create_thought(
            CoreThoughtRecord(
                thought_id=thought_id,
                thought_type=ThoughtType.BELIEF,
                essence="chronometer note",
                content="chronometer note body",
                priority=Priority.P2,
                lifecycle_status=LifecycleStatus.ACTIVE,
                created_cycle=0,
                updated_cycle=0,
                source="test",
                created_at=written_at,
                updated_at=written_at,
            )
        )
