"""Unit tests for metrics module."""

from retrieval_metrics.core.metrics import (
    _ranks_to_run_qrels,
    calculate_retrieval_metrics,
    get_supported_metrics,
)


class TestRanksToRunQrels:
    """Test _ranks_to_run_qrels function."""

    def test_正常系_単一クエリの変換(self) -> None:
        """単一クエリのランキングが正しく変換されることを確認。"""
        ranks_list = [[2]]  # 文書1が2位

        run, qrels = _ranks_to_run_qrels(ranks_list)

        # max_rank個の文書に関するrunができる。doc{n}がn番めに大きいスコアになっている。
        assert run == {"0": {"doc1": 2, "doc2": 1}}
        # 2位のdoc（doc2）のみ関連性1
        assert qrels == {"0": {"doc2": 1}}

    def test_正常系_ランク外正解文書があるとき(self) -> None:
        """単一クエリのランキングが正しく変換されることを確認。"""
        ranks_list = [[2, -1, -1]]  # 文書1が2位、文書2がランク外

        run, qrels = _ranks_to_run_qrels(ranks_list)

        # max_rank個の文書に関するrunができる。doc{n}がn番めに大きいスコアになっている。
        assert run == {"0": {"doc1": 2, "doc2": 1}}
        # 2位のdoc（doc2）のみ関連性1
        assert qrels == {
            "0": {"doc2": 1, "doc3": 1, "doc4": 1}
        }  # doc3はランク外だが関連性1として追加

    def test_正常系_複数クエリの変換(self) -> None:
        """複数クエリのランキングが正しく変換されることを確認。"""
        ranks_list = [[1, 2], [-1, -1, 5]]

        run, qrels = _ranks_to_run_qrels(ranks_list)

        # クエリ数の確認
        assert run["0"] == {
            "doc1": 2,
            "doc2": 1,
        }  # max_rank=2なので2個のdocの結果を生成
        assert run["1"] == {
            "doc1": 5,
            "doc2": 4,
            "doc3": 3,
            "doc4": 2,
            "doc5": 1,
        }  # max_rank=5なので5個のdocの結果を生成
        assert qrels["0"] == {"doc1": 1, "doc2": 1}  # 1,2位のdocの関連度が1
        assert qrels["1"] == {
            "doc5": 1,
            "doc6": 1,
            "doc7": 1,
        }  # 3,4,5位のdocの関連度が1


class TestCalculateRetrievalMetrics:
    """Test calculate_retrieval_metrics function."""

    def test_正常系_正解文書1つ_hit_rate(self) -> None:
        """完璧なランキング（1,2,3順）でhit_rateが期待値通りになることを確認。"""
        # 1つのクエリで文書1,2,3の順番でランキング
        ranks_list = [[2]]
        metrics = ["hit_rate@1", "hit_rate@2"]

        results = calculate_retrieval_metrics(ranks_list, metrics)

        # すべて関連文書なので precision@k = k/k = 1.0
        assert results["hit_rate@1"] == 0.0  # 1/1
        assert results["hit_rate@2"] == 1.0  # 2/2

    def test_正常系_正解文書2つ_hit_rate(self) -> None:
        """完璧なランキング（1,2,3順）でhit_rateが期待値通りになることを確認。"""
        # 1つのクエリで文書1,2,3の順番でランキング
        ranks_list = [[2, 3]]
        metrics = ["hit_rate@1", "hit_rate@2", "hit_rate@3"]

        results = calculate_retrieval_metrics(ranks_list, metrics)

        # hit_rateなので正解文書が1つでも含まれていれば1
        assert results["hit_rate@1"] == 0.0  # 1/1
        assert results["hit_rate@2"] == 1.0  # 2/2
        assert results["hit_rate@3"] == 1.0  # 3/3

    def test_正常系_完璧なランキング_recall(self) -> None:
        """完璧なランキングでrecallが期待値通りになることを確認。"""
        ranks_list = [[3, 5]]
        metrics = ["recall@2", "recall@3", "recall@4", "recall@5"]

        results = calculate_retrieval_metrics(ranks_list, metrics)

        # 全3文書が関連文書の場合
        assert results["recall@2"] == 0.0
        assert results["recall@3"] == 0.5
        assert results["recall@4"] == 0.5
        assert results["recall@5"] == 1.0

    def test_正常系_複数クエリの平均_hit_rate(self) -> None:
        """複数クエリの平均が正しく計算されることを確認。"""
        # クエリ1: [1,2] precision@1=1.0, クエリ2: [3,4] precision@1=1.0
        ranks_list = [[2], [3, 4]]
        metrics = ["hit_rate@1", "hit_rate@2", "hit_rate@3", "hit_rate@4"]

        results = calculate_retrieval_metrics(ranks_list, metrics)

        # 両クエリともprecision@1=1.0なので平均も1.0
        assert results["hit_rate@1"] == 0
        assert results["hit_rate@2"] == 0.5
        assert results["hit_rate@3"] == 1


class TestGetSupportedMetrics:
    """Test get_supported_metrics function."""

    def test_正常系_サポートされたメトリクスリストの取得(self) -> None:
        """サポートされたメトリクスリストが正しく取得されることを確認。"""
        supported = get_supported_metrics()

        assert isinstance(supported, list)
        assert len(supported) > 0

        # 基本的なメトリクスが含まれていることを確認
        expected_metrics = ["map", "ndcg", "precision", "recall"]
        assert set(expected_metrics).issubset(set(supported))
