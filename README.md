# UMA Lens

スマホ対応の競馬確率予測Webアプリです。既存研究コードを保存したまま、テスト可能なPython APIとGitHub Pages向け静的フロントを追加しています。

## 主な機能

- 全馬共通のレース条件別基準速度＋馬別残差ガウス過程
- 馬基準／ジョッキー基準の切り替え
- 過去3走未満では母集団へ縮約するフォールバック
- 同じ着順シミュレーションから単勝・三連複・三連単確率を整合的に集計
- 対応する券種・買い目の実オッズがある場合だけ期待回収率を表示
- `prediction_at` より前の履歴だけで学習し、時点リークを防止
- APIなしでもGitHub Pages上のサンプルを操作可能
- CSV末尾の検証レース数をGUIで変更し、的中率とBrier scoreを比較

詳細は [設計と確率](docs/ARCHITECTURE.md) と [データ取得ポリシー](docs/DATA_POLICY.md) を参照してください。

## ローカル起動

```bash
docker compose up --build
```

画面は <http://localhost:8080>、API仕様は <http://localhost:8000/docs> です。ライブ計算するには `web/config.js` の `API_BASE_URL` を `http://localhost:8000` に変更します。

Pythonで直接起動する場合:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn uma_api.main:app --reload
pytest -q
```

`POST /api/predict` の完全な入力例は `web/demo-race.json` にあります。オッズは `{"bet_type":"trifecta","selection":[1,2,3],"decimal_odds":72.5}` のように券種と買い目を明示します。

`prediction_basis` は `horse`（馬基準）または `jockey`（ジョッキー基準）です。後者では各出走馬に `jockey_id` が必要です。静的デモは `data/UMA_DB.csv` と `data/Jockey_DB_EL.csv` から `scripts/generate_demo.py` で生成されます。ジョッキーCSVには騎手名対応表がないため、画面では騎手IDを表示します。

## GitHub Pages

1. PRをマージします。
2. Repository Settings → Pages → Source を **GitHub Actions** にします。
3. APIを使う場合は Actions Variables に `API_BASE_URL`（例: `https://uma-api.example.com`）を登録します。
4. `Deploy GitHub Pages` workflowを実行します。未設定なら静的デモとして動きます。

## APIホスティングと認証

コンテナをRender、Railway、Fly.io、Cloud Run等へ配置できます。最低限、次を設定してください。

- `CORS_ORIGINS=https://tnaoki0630.github.io`（実際のPages originに合わせる）
- データ提供元のAPIキーはホスティング側Secretに保存
- 公開APIにはレート制限を設定
- 個人利用ならCloudflare Access等の認証プロキシをAPI前段に置く

GitHub Pagesは秘密鍵を保持できません。ブラウザへ秘密を配る方式は使わないでください。契約データが再配布不可なら、API認証とユーザーごとの権限確認が必須です。

## CIと既存コード

push/PRごとにRuff、単体テスト、静的デモの再生成差分、Docker buildを検証します。確率テストは総和と「三連複＝対応する三連単6通りの和」を確認します。

`src/` のGPR、積分、取得実験は資料価値があるため変更していません。発想は新実装へ引き継ぎましたが、グローバル変数・ファイル依存、時点境界の欠如、単勝オッズを合成した三連系期待値、計算量の問題からWeb経路では直接利用しません。

## 注意

これは投資助言ではなく、利益を保証しません。本番投入前に契約データで時系列バックテスト、校正評価（Brier score・log loss）、控除率込みの検証を行ってください。
