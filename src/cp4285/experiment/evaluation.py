"""Shared evaluation example. Change benchmark policy here, never inside a model."""

import math

from .data import Evaluation
from .model import Recommender


def popularity_baseline(
    training_counts: dict[str, int], target_counts: dict[str, int], *, k: int
) -> dict[str, float | int | None]:
    """Event-weighted Recall/MRR from count aggregates; unseen targets count as misses.

    This frequency-only audit has no personalized history input. Its ranking is
    fitted on training counts only, with lexical label ties; targets never refit it.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    for counts in (training_counts, target_counts):
        if any(
            not isinstance(s, str) or not s or type(n) is not int or n < 0
            for s, n in counts.items()
        ):
            raise ValueError("Counts must map nonempty labels to nonnegative integers")
    ranking = sorted(
        (s for s, n in training_counts.items() if n), key=lambda s: (-training_counts[s], s)
    )
    if not ranking:
        raise ValueError("Training needs at least one positive count")
    ranks = {s: r for r, s in enumerate(ranking, 1)}
    total = sum(target_counts.values())
    hit = sum(n for s, n in target_counts.items() if ranks.get(s, k + 1) <= k)
    rr = sum(n / ranks[s] for s, n in target_counts.items() if ranks.get(s, k + 1) <= k)
    active_targets = [s for s, n in target_counts.items() if n]
    return {
        "events": total,
        "candidate_labels": len(ranking),
        "unseen_target_events": sum(n for s, n in target_counts.items() if s not in ranks),
        "recall": hit / total if total else None,
        "mrr": rr / total if total else None,
        "macro_label_recall": sum(ranks.get(s, k + 1) <= k for s in active_targets)
        / len(active_targets)
        if active_targets
        else None,
    }


def evaluate_rank_counts(rank_counts: dict[int, int], *, k: int) -> dict:
    """Recall/MRR from target-rank counts; rank zero is an unranked target/miss."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if any(
        type(rank) is not int or rank < 0 or type(n) is not int or n < 0
        for rank, n in rank_counts.items()
    ):
        raise ValueError("Ranks and counts must be nonnegative integers")
    total = sum(rank_counts.values())
    hit = sum(n for rank, n in rank_counts.items() if 1 <= rank <= k)
    rr = sum(n / rank for rank, n in rank_counts.items() if 1 <= rank <= k)
    return {
        "examples": total,
        "recall": hit / total if total else None,
        "mrr": rr / total if total else None,
        "unranked_targets": rank_counts.get(0, 0),
    }


def evaluate(model: Recommender, task: Evaluation, *, k: int) -> dict[str, float]:
    """Single-target macro averages; lexical original-item-ID ties, no seen masking.

    Candidate order is an interface detail, not a tie-breaking rule. A repeated
    item remains eligible. Recall@k equals Hit@k for this single-target contract.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if (
        not task.examples
        or not task.candidates
        or len(set(task.candidates)) != len(task.candidates)
    ):
        raise ValueError("Evaluation needs examples and nonempty unique candidates")
    ndcg = hit = 0.0
    for example in task.examples:
        if example.target.domain != task.domain or example.target.item not in task.candidates:
            raise ValueError("Invalid evaluation target/domain")
        scores = tuple(model.score(example.query, task.candidates))
        if len(scores) != len(task.candidates) or any(not math.isfinite(s) for s in scores):
            raise ValueError("score() must return one finite value per candidate")
        target_score = scores[task.candidates.index(example.target.item)]
        rank = 1 + sum(
            score > target_score or (score == target_score and item < example.target.item)
            for item, score in zip(task.candidates, scores, strict=True)
        )
        if rank <= k:
            hit += 1
            ndcg += 1 / math.log2(rank + 1)
    count = len(task.examples)
    return {"ndcg": ndcg / count, "hit": hit / count, "recall": hit / count}
