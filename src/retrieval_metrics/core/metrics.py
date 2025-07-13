"""Core retrieval metrics calculation functionality."""

import ranx as rx


def _mask_ranks(ranks_list: list[list[int]], k: int) -> list[list[int]]:
    """kより悪い順位を-1にする"""
    if k < 1:
        return ranks_list

    masked_ranks_list = []
    for ranks in ranks_list:
        # kより悪い順位は-1にする
        masked_ranks = []
        for rank in ranks:
            if rank > k:
                masked_ranks.append(-1)
            else:
                masked_ranks.append(rank)
        masked_ranks_list.append(masked_ranks)
    return masked_ranks_list


def _ranks_to_run_qrels(
    ranks_list: list[list[int]], k: int = -1
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Convert ranks_list to ranx format."""
    run: dict[str, dict[str, int]] = {}
    qrels: dict[str, dict[str, int]] = {}

    ranks_list = _mask_ranks(ranks_list, k)

    for qid, ranks in enumerate(ranks_list):
        run[str(qid)] = {}
        qrels[str(qid)] = {}
        ranks_max = max(ranks)
        max_rank = ranks_max if k < 0 else min(max(ranks_max, 0), k)

        for doc_id in range(1, max_rank + 1):
            # runを作る。doc_idは1からmax_rankまでで、doc1が1位でdoc{max_rank}が最下位
            run[str(qid)][f"doc{doc_id}"] = max_rank + 1 - doc_id
            # qrelsを作る。順位がranksに含まれるdoc_idは関連性1、それ以外は0
            if doc_id in ranks:
                qrels[str(qid)][f"doc{doc_id}"] = 1
        # -1の文書をdummyとしてqrels二追加
        # ranksの中の-1の個数を数える
        out_of_ranks_count = ranks.count(-1)
        for i in range(1, out_of_ranks_count + 1):
            # runには含まれないが、qrelsには関連性1の文書を追加
            qrels[str(qid)][f"doc{max_rank + i}"] = 1

    return run, qrels


def calculate_retrieval_metrics(
    ranks_list: list[list[int]], metrics: list[str], k: int = -1
) -> dict[str, float]:
    """Calculate retrieval metrics using ranx library.

    Parameters
    ----------
    ranks_list : list[list[int]]
        List of document rankings for each query (1-based indexing)
    metrics : list[str]
        List of metrics in 'name@k' format (e.g., ['map@10', 'ndcg@5'])

    Returns
    -------
    dict[str, float]
        Dictionary mapping metric names to their calculated values

    Examples
    --------
    >>> ranks_list = [[1, 2, 3], [4, 5, 6]]
    >>> metrics = ['map@3', 'ndcg@3']
    >>> results = calculate_retrieval_metrics(ranks_list, metrics)
    >>> print(results)
    {'map@3': 1.0, 'ndcg@3': 1.0}
    """
    run, qrels = _ranks_to_run_qrels(ranks_list, k)
    scores = rx.evaluate(qrels, run, metrics=metrics)

    # ranxは単一メトリックの場合は値のみ、複数メトリックの場合は辞書を返すため統一
    if len(metrics) == 1:
        return {metrics[0]: float(scores)}
    else:
        return {k: float(v) for k, v in scores.items()}


def get_supported_metrics() -> list[str]:
    """Get list of supported metric names."""
    return ["map", "mrr", "ndcg", "precision", "recall", "hits", "hit_rate"]


if __name__ == "__main__":
    # Example usage
    ranks = [[1, 2], [3, 4]]
    metrics = ["hit_rate@3", "recall@3"]
    results = calculate_retrieval_metrics(ranks, metrics)
    print(results)  # {'hit_rate@3': 1.0, 'recall@3': 0.75}
