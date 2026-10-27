"""Training-frequency and last-category rankings for the category task."""


def rank_categories(training_counts, candidates, *, last_category=None):
    """Full candidate ranking; last-category promotion keeps popularity order otherwise."""
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidates must be nonempty and unique")
    if any(not isinstance(label, str) or not label for label in candidates):
        raise ValueError("Candidate labels must be nonempty strings")
    if any(
        label not in candidates or type(n) is not int or n < 0
        for label, n in training_counts.items()
    ):
        raise ValueError("Invalid training counts or labels outside candidates")
    if not sum(training_counts.values()):
        raise ValueError("Need positive training counts")
    if last_category is not None and last_category not in candidates:
        raise ValueError("Last category must be a candidate")
    return tuple(
        sorted(
            candidates,
            key=lambda label: (label != last_category, -training_counts.get(label, 0), label),
        )
    )
