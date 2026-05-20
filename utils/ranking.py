def rank_candidates(candidates_list: list[dict]) -> list[dict]:
    """
    Sorts candidates by their ATS score in descending order (best first).
    Returns the same list structure — no data is added or removed here.
    """
    return sorted(candidates_list, key=lambda c: c["score"], reverse=True)
