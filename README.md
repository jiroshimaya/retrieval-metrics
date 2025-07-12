# Python Template for Claude Code

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-latest-green.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![CI](https://github.com/discus0434/python-template-for-claude-code/actions/workflows/ci.yml/badge.svg)](https://github.com/discus0434/python-template-for-claude-code/actions/workflows/ci.yml)

[AIエージェント](https://www.anthropic.com/claude-code)との協働に最適化された、プロダクション対応のPythonプロジェクトテンプレートです。厳格な型チェック、自動パフォーマンス測定、包括的なドキュメント、進化するメモリ管理システムを備えています。

## 🔗 重要なドキュメント

- **[CLAUDE.md](./CLAUDE.md)**: AIエージェント向けの技術仕様・実装ガイド
- **[template/](./template/)**: ベストプラクティスのモデルコード集

## 📋 AIエージェント チートシート

#### 基本CLIコマンド
```bash
# インタラクティブREPLの開始
claude

# 初期プロンプト付きでREPLを開始
claude "質問内容"

# SDKモードで質問して終了
claude -p "質問内容"

# 最新の会話を継続
claude -c

# インタラクティブに特定のセッションを選択して再開
claude -r

# アップデート・システム管理
claude update                    # AIエージェントを最新版に更新
claude --version                # バージョン確認
claude --help                   # ヘルプ表示
claude mcp                      # Model Context Protocol設定
```

#### CLIフラグ・オプション
```bash
# モデル設定
claude --model sonnet           # Sonnetを使用
claude --model opus             # Opusを使用

# セッション管理
claude --list-sessions          # セッション一覧表示
claude --delete-session "<id>"  # セッション削除

# ディレクトリ・作業環境
claude --add-dir /path/to/dir   # 作業ディレクトリを追加
claude --add-dir dir1 --add-dir dir2  # 複数ディレクトリ追加

# 出力・フォーマット制御
claude --output-format text     # テキスト形式出力
claude --output-format json     # JSON形式出力
claude --output-format stream-json  # ストリーミングJSON出力
claude --verbose               # 詳細ログ表示
claude --quiet                 # 静かなモード

# 実行制御・権限管理
claude --max-turns 5           # 最大ターン数制限
claude --timeout 30            # タイムアウト設定（秒）
claude --tool-permissions restricted  # ツール権限制限
claude --allowedTools "Read,Write,Bash"  # 使用可能ツール指定
claude --disallowedTools "WebFetch"      # 使用禁止ツール指定

# セッション・履歴管理
claude --list-sessions         # セッション一覧表示
claude --delete-session "<id>" # セッション削除
claude --session-id "<id>"     # 特定セッションで開始
claude --no-history           # 履歴を保存しない
```

#### パイプ・ストリーム操作
```bash
# パイプでの入力
echo "コードをレビューして" | claude
cat file.py | claude "このコードを説明して"

# ファイルの内容を直接入力
claude < input.txt

# 環境変数での設定
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_API_KEY=your_api_key
```

#### REPLモード内スラッシュコマンド
AIエージェント のインタラクティブモード中に使用できるコマンド：

```bash
# 基本コマンド
/help                          # ヘルプ・利用可能コマンド一覧表示
/clear                         # 画面クリア・履歴リセット
/resume                        # インタラクティブに特定のセッションを選択して再開
/continue                      # 直前のセッションに戻る
/exit                          # Claude Codeを終了
/quit                          # Claude Codeを終了（/exitの別名）

# モデル・設定管理
/model                         # 現在のモデル表示・変更
/model sonnet                  # Claude 3.5 Sonnetに変更
/model opus                    # Claude 3 Opusに変更
/model haiku                   # Claude 3 Haikuに変更
/settings                      # 現在の設定表示・変更
/permissions                   # 現在のツール権限表示

# メモリ・記憶管理
/memory                        # メモリ管理・プロジェクト記憶の表示/編集
/memory clear                  # メモリをクリア
/memory show                   # 現在のメモリ内容を表示
/memory edit                   # メモリを編集
#                              # CLAUDE.mdへのクイックアクセス（行頭で#）

# セッション管理
/save                          # 現在のセッションを保存
/save <name>                   # セッションに名前を付けて保存
/load <session-id>             # 指定セッションを読み込み
/sessions                      # セッション一覧表示
/list-sessions                 # セッション一覧表示（/sessionsの別名）
/delete-session <id>           # 指定セッションを削除

# ファイル・プロジェクト操作
/add-dir <path>                # 作業ディレクトリを追加
/remove-dir <path>             # 作業ディレクトリを削除
/list-dirs                     # 現在の作業ディレクトリ一覧
/cwd                           # 現在の作業ディレクトリを表示
/pwd                           # 現在の作業ディレクトリを表示（/cwdの別名）

# モード・表示切替
/terminal-setup                # 複数行入力モードの設定
/vim-mode                      # Vimキーバインドを有効化
/emacs-mode                    # Emacsキーバインドを有効化

# デバッグ・診断・ユーティリティ
/debug                         # デバッグ情報表示
/status                        # システム状態・接続状況表示
/version                       # Claude Codeのバージョン表示
/whoami                        # 現在のユーザー情報表示
/tokens                        # トークン使用量統計表示
/feedback                      # フィードバック送信
/reset                         # 設定をデフォルトにリセット
```

#### キーボードショートカット・インタラクティブ操作

```bash
# 基本操作
Ctrl+J                         # 改行
Esc x 2                        # 前回のメッセージに戻る
Ctrl+C                         # 現在の入力・生成をキャンセル
Ctrl+D                         # Claude Codeセッションを終了
Ctrl+L                         # ターミナル画面をクリア
Ctrl+R                         # コマンド履歴の逆方向検索（対応端末）
Up/Down arrows                 # コマンド履歴のナビゲーション
Tab                            # オートコンプリート（利用可能な場合）
Shift + Tab                    # Planモード切り替え

# 複数行入力（環境依存）
\<Enter>                       # クイックエスケープ（全端末対応）
Option+Enter                   # macOS標準
Shift+Enter                    # /terminal-setup後に有効
Alt+Enter                      # Linux/Windows（一部端末）

# 特殊入力
Esc Esc                        # 前のメッセージを編集
#<Enter>                       # CLAUDE.mdへのクイックアクセス
/<tab>                         # スラッシュコマンドの補完
```

#### 高度な使用例・統合活用

```bash
# 複数オプションの組み合わせ
claude --model opus --verbose --add-dir ./src --add-dir ./tests "プロジェクト全体をレビューして"

# 特定の形式での出力制御
claude --output-format json -p "このコードの問題点をJSON形式で教えて" < code.py

# タイムアウト付きでの実行
claude --timeout 60 --max-turns 3 "複雑な最適化を提案して"

# 制限されたツール権限での実行
claude --tool-permissions restricted "安全にコードを分析して"

# セッション管理の活用
claude --save-session "code-review-2024" --model sonnet
claude --load-session "code-review-2024" --continue

# スクリプト自動化での活用
claude --output-format json --quiet -p "テストを実行してエラー数を返して" | jq '.error_count'

# CI/CDでの活用例
cat test_results.txt | claude --output-format text -p "テスト結果を分析してサマリーを作成"

# 設定ファイルとの連携
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_CONFIG_FILE="./claude-config.json"
claude --config-file ./claude-config.json
```

#### Model Context Protocol（MCP）統合

```bash
# MCP サーバー設定・管理
claude mcp                     # MCP設定画面を開く
claude mcp list                # 利用可能なMCPサーバー一覧
claude mcp enable <server>     # MCPサーバーを有効化
claude mcp disable <server>    # MCPサーバーを無効化
claude mcp status              # MCP接続状況確認

# MCPサーバー例（設定後に利用可能）
# - ファイルシステム操作
# - データベース接続
# - Git操作
# - Docker管理
# - クラウドサービス統合
```

#### 環境変数・設定の活用

```bash
# 基本設定
export ANTHROPIC_API_KEY="your_api_key"
export ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"
export CLAUDE_CONFIG_DIR="~/.config/claude-code"

# 高度な設定
export CLAUDE_MAX_TURNS=10           # デフォルト最大ターン数
export CLAUDE_DEFAULT_TIMEOUT=120    # デフォルトタイムアウト
export CLAUDE_HISTORY_SIZE=1000      # 履歴保存数
export CLAUDE_AUTO_SAVE=true         # 自動セッション保存

# プロジェクト固有設定
export CLAUDE_PROJECT_DIRS="./src:./tests:./docs"
export CLAUDE_ALLOWED_TOOLS="Read,Write,Bash,Edit"
export CLAUDE_DISALLOWED_TOOLS="WebFetch,WebSearch"

# デバッグ・開発設定
export CLAUDE_DEBUG=true             # デバッグモード
export CLAUDE_LOG_LEVEL=INFO         # ログレベル
export CLAUDE_VERBOSE=true           # 詳細出力
```

## 🚀 クイックスタート

### このテンプレートを使用する

1. GitHubで「Use this template」ボタンをクリックして新しいリポジトリを作成
2. 新しいリポジトリをクローン
3. セットアップスクリプトを実行

```bash
# 新しいリポジトリをクローン
git clone https://github.com/yourusername/retrieval-metrics.git
cd retrieval-metrics

# セットアップ
make setup
```

セットアップスクリプトは以下を実行します：
- すべての `retrieval_metrics` を実際のプロジェクト名に更新（途中でプロジェクト名を入力するように求められます）
- uvを使用してPython環境を初期化
- Claude Codeをインストール
- GitHub CLI（`gh`）をインストール（途中でログインを求められます）
- すべての依存関係をインストール
- pre-commitフックを設定
- 初期テストを実行

### 手動セットアップ（代替方法）

手動セットアップを希望する場合：

```bash
# プロジェクト名を更新
python scripts/update_retrieval_metrics.py your_retrieval_metrics

# uvをインストール（まだインストールしていない場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Pythonバージョンを設定
uv python pin 3.12

# 依存関係をインストール
uv sync --all-extras

# pre-commitフックをインストール
uv run pre-commit install
uv run pre-commit install --hook-type pre-push

# テストを実行
uv run pytest
```

## ✨ 主な特徴

### 🛠️ 開発ツールチェーン
- **[uv](https://github.com/astral-sh/uv)** - 高速なPythonパッケージマネージャー
- **[Ruff](https://github.com/astral-sh/ruff)** - 超高速Pythonリンター・フォーマッター
- **[mypy](https://mypy-lang.org/)** - strictモード＋PEP 695型構文対応
- **[pytest](https://pytest.org/)** - カバレッジ付きテストフレームワーク
- **[bandit](https://github.com/PyCQA/bandit)** - セキュリティスキャン
- **[pip-audit](https://github.com/pypa/pip-audit)** - 依存関係の脆弱性チェック
- **[pre-commit](https://pre-commit.com/)** - コード品質用Gitフック

### 🔍 コード品質・型安全性
- ✅ PEP 695新型構文（`type` statement）対応
- ✅ pydantic.BaseModel・Literal・Protocol活用の堅牢な型システム
- ✅ 自動セキュリティ・脆弱性チェック


### 🔄 CI/CD・自動化
- ✅ 並列実行対応の高速CIパイプライン
- ✅ 自動パフォーマンスベンチマーク（PR時レポート生成）
- ✅ Dependabotによる自動依存関係更新
- ✅ GitHub CLIによるワンコマンドPR・Issue作成
- ✅ キャッシュ最適化された実行環境


## 📁 プロジェクト構造

```
project-root/
├── .github/                     # GitHub Actionsの設定ファイル
│   ├── workflows/               # CI/CD + ベンチマークワークフロー
│   │   └── ci.yml               # メインCI（テスト・リント・型チェック）
│   ├── dependabot.yml           # Dependabotの設定
│   ├── ISSUE_TEMPLATE/          # Issueテンプレート
│   └── PULL_REQUEST_TEMPLATE.md # Pull Requestテンプレート
├── template/                    # **重要**: AIエージェントのベストプラクティス・モデルコード
│   ├── src/
│   │   └── template_package/    # モデルパッケージの完全な実装例
│   │       ├── __init__.py      # パッケージエクスポートの例
│   │       ├── py.typed         # 型情報マーカーの例
│   │       ├── schemas.py         # スキーマ定義のベストプラクティス
│   │       ├── core/
│   │       │   └── example.py   # クラス・関数実装の模範例
│   │       └── utils/
│   │           ├── helpers.py   # ユーティリティ関数の実装例
│   │           ├── logging_config.py # ロギング設定の実装例
│   │           └── profiling.py # パフォーマンス測定の実装例
│   └── tests/                   # テストコードの完全な実装例
│       ├── unit/                # 単体テストの例
│       │   ├── test_example.py  # クラステストの例
│       │   ├── test_helpers.py  # 関数テストの例
│       │   └── test_logging.py  # ロギングテストの例
│       ├── property/            # プロパティベーステストの例
│       │   └── test_helpers_property.py
│       ├── integration/         # 結合テストの例
│       │   └── test_example.py
│       └── conftest.py          # pytestフィクスチャの例
├── src/                         # 実際の開発用ディレクトリ
│       └── retrieval_metrics/    # モデルパッケージの完全な実装例
│           └── （プロジェクト固有のパッケージを配置）
├── tests/                       # 実際のテスト用ディレクトリ
│   ├── unit/                    # 単体テスト
│   ├── integration/             # 統合テスト
│   └── conftest.py              # pytest設定
├── docs/                        # ドキュメント
├── scripts/                     # ユーティリティスクリプト
├── pyproject.toml               # uv/ruff/mypyの設定ファイル
├── .gitignore                   # バージョン管理除外ファイル
├── .pre-commit-config.yaml      # pre-commitの設定ファイル
├── README.md                    # 人間向けのプロジェクトの説明
└── CLAUDE.md                    # このファイル
```

## 📚 ドキュメント階層

### 🎯 メインドキュメント
- **[CLAUDE.md](CLAUDE.md)** - 包括的プロジェクトガイド
  - プロジェクト概要・コーディング規約
  - よく使うコマンド・GitHub操作
  - 型ヒント・テスト戦略・セキュリティ

## ✅ 新規プロジェクト設定チェックリスト

### 🔧 基本プロジェクト設定
- [ ] **作者情報更新**: `pyproject.toml`の`authors`セクション
- [ ] **ライセンス選択**: LICENSEファイルを適切なライセンスに更新
- [ ] **README.md更新**: プロジェクト固有の説明・機能・使用方法
- [ ] **CLAUDE.md カスタマイズ**: プロジェクト概要を適宜更新
- [ ] **専門ガイドの追加**: 適宜`docs/`内に詳細なガイドを追加

### ⚙️ 開発環境・品質設定
- [ ] **依存関係調整**: プロジェクトに必要な追加パッケージの導入
- [ ] **リントルール**: プロジェクトに合わせた`ruff`設定のカスタマイズ

### 🔐 GitHubリポジトリ・セキュリティ設定
- [ ] **ブランチ保護**: `main`ブランチの保護ルール有効化
- [ ] **PR必須レビュー**: Pull Request作成時のレビュー要求設定
- [ ] **ステータスチェック**: CI・型チェック・テストの必須化
- [ ] **Dependabot**: 自動依存関係更新の有効化
- [ ] **Issues/Projects**: 必要に応じてプロジェクト管理機能の有効化

## 🔧 カスタマイズ

### 型チェックの厳格さ調整

mypyのstrictモードが最初から厳しすぎる場合：

```toml
# pyproject.toml - 基本設定から開始
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true

# 段階的により厳格な設定を有効化
[[tool.mypy.overrides]]
module = ["retrieval_metrics.core.*"]
strict = true  # まずコアモジュールにstrictモードを適用
```

### リントルールの変更

```toml
# pyproject.toml
[tool.ruff.lint]
# 必要に応じてルールコードを追加・削除
select = ["E", "F", "I"]  # 基本から開始
ignore = ["E501"]  # 行の長さはフォーマッターが処理
```


## 🔗 外部リソース・参考資料

### 🛠️ 開発ツール公式ドキュメント
- **[uv ドキュメント](https://docs.astral.sh/uv/)** - Pythonパッケージ管理
- **[Ruff ドキュメント](https://docs.astral.sh/ruff/)** - リント・フォーマッター
- **[mypy ドキュメント](https://mypy.readthedocs.io/)** - 型チェッカー
- **[pytest ドキュメント](https://docs.pytest.org/en/stable/)** - テストフレームワーク
- **[Hypothesis ドキュメント](https://hypothesis.readthedocs.io/)** - プロパティベーステスト

### 🐍 Python・型ヒント
- **[PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)** - 新型構文仕様
- **[TypedDict Guide](https://docs.python.org/3/library/typing.html#typing.TypedDict)** - 型安全な辞書
- **[Python 3.12 リリースノート](https://docs.python.org/3/whatsnew/3.12.html)** - 新機能一覧

---

## 📄 ライセンス

このテンプレートはMITライセンスの下でリリースされています。詳細は[LICENSE](LICENSE)をご覧ください。
