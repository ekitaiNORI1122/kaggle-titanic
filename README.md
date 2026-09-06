# Kaggle Titanic

ローカル(VS Code)でコードを書き、GitHub を経由して Kaggle Notebook で実行する構成。

## 運用フロー

```
 VS Code で src/ を編集
        │  git push
        ▼
   GitHub (public)
        │  git clone (public なので認証不要)
        ▼
 Kaggle Notebook で実行 → submission.csv → Submit
```

同じ `src/` がローカルでも Kaggle でも動く。切り替えは `src/config.py` が
`/kaggle/input` の有無を見て自動判定するので、コード側の分岐は不要。

| | ローカル | Kaggle Notebook |
|---|---|---|
| データ | `data/raw/` | `/kaggle/input/titanic/` |
| 提出物 | `submissions/` | `/kaggle/working/` |

## ディレクトリ

```
src/
  config.py    環境判定とパス・定数（ここだけが環境を意識する）
  data.py      train/test の読み込み
  features.py  特徴量エンジニアリング
  model.py     LightGBM の交差検証・予測・提出ファイル作成
notebooks/
  01_eda.ipynb          ローカルでの EDA・実験用
kaggle_notebook/
  titanic_runner.ipynb  Kaggle にアップロードする薄いランナー
data/raw/      コンペのデータ（git 管理外）
submissions/   提出用 CSV（git 管理外）
```

## ローカルでの作業

仮想環境の有効化（PowerShell）:

```powershell
.\.venv\Scripts\Activate.ps1
```

JupyterLab:

```powershell
.\.venv\Scripts\jupyter.exe lab
```

VS Code で `.ipynb` を開く場合はカーネルに **Python 3.12 (titanic)** を選ぶ。

CLI から一気に回す:

```powershell
.\.venv\Scripts\python.exe -c "from src import data, features, model, config; tr, te = data.load_train(), data.load_test(); X, y, Xt = features.build(tr, te); oof, ms = model.run_cv(X, y); model.make_submission(te[config.ID_COL], model.predict(ms, Xt))"
```

## Kaggle Notebook 側のセットアップ（初回のみ）

1. **Notebook を作る**
   Kaggle の Code → New Notebook → File → Import Notebook で
   `kaggle_notebook/titanic_runner.ipynb` をアップロード

2. **Notebook の設定**
   - **Add Input** → Competitions → `Titanic` を追加
   - **Settings → Internet** を On（clone に必要）

以降は VS Code で push → Kaggle Notebook を Run するだけで最新コードが走る。

> リポジトリが public なので clone に認証は不要。
> private に戻す場合は、GitHub の Personal Access Token を Kaggle の
> **Add-ons → Secrets** に登録し、clone URL を
> `https://x-access-token:{token}@github.com/...` の形にする必要がある。

## データの取得

```powershell
.\.venv\Scripts\kaggle.exe competitions download -c titanic -p data\raw
```

## 提出

Kaggle Notebook から出す場合は、実行後に右上の **Submit** ボタン。

ローカルの CSV を CLI から出す場合:

```powershell
.\.venv\Scripts\kaggle.exe competitions submit -c titanic -f submissions\<file>.csv -m "説明"
```

## Notebook のバージョン管理

`nbstripout` を git filter として設定済み。コミット時に出力セルが自動で削除されるため、
差分が読める状態を保てる。手元の表示は消えないので作業への影響はない。

## スコア記録

| 日付 | 手法 | CV | LB |
|---|---|---|---|
| 2026-09-06 | LightGBM ベースライン (12特徴量) | 0.8462 | - |
