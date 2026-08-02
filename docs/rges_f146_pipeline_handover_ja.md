# RGES F146 anomaly finder パイプライン引き継ぎ

最終更新: 2026-08-02
対象コード: `/rogue1_8/nunota/jacscanomaly`
公開先: `http://133.1.160.32/ou-moa/public/rges_anomaly_finder/`

この資料は、RGES Beginner/Experienced tier の全イベントを、F146だけで
`jacscanomaly` に通し、イベント単位でHTML公開するrunの引き継ぎ用である。
GitHubへの公開はしていない。

## 現在のrun

現在のtmuxセッションは次の通り。

```text
session: rges_f146_serial
log:     /moao39_13/nunota/rges-data/rges_f146_serial_v10.log
progress:/moao39_13/nunota/rges-data/anomaly_finder_progress.txt
```

起動コマンド:

```bash
cd /rogue1_8/nunota/jacscanomaly
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 \
    taskset -c 0 \
    python tools/rges_anomaly_pipeline.py \
      --tier both --force --build-html \
      --progress-file /moao39_13/nunota/rges-data/anomaly_finder_progress.txt \
      >> /moao39_13/nunota/rges-data/rges_f146_serial_v10.log 2>&1
```

`--force` を付けているため、以前の途中成果物をスキップせず全2267イベントを再計算する。
現在の構成は1コア・直列で、1イベントの全処理が終わってから次のイベントへ進む。

2026-08-02の確認時点では、全体 `923/2267 (40.71%)`、Experienced tier内では
`736/2079` の開始直前まで進んでいる。正確な最新値は以下で確認する。

```bash
tail -20 /moao39_13/nunota/rges-data/rges_f146_serial_v10.log
tail -1 /moao39_13/nunota/rges-data/anomaly_finder_progress.txt
tmux has-session -t rges_f146_serial && echo alive
```

進捗ファイルはイベント完了ごとに次の形式で追記される。

```text
923/2267 (40.71%) tier=experienced event=RMDC26_000929 status=done
```

## 対象データと処理順序

対象は次の2267イベント。

```text
Beginner:    188
Experienced: 2079
合計:        2267
band:        F146のみ
```

複数バンドがあっても、`tools/rges_anomaly_pipeline.py` の入力はF146だけである。
各イベントは次の順で完結する。

```text
F146読込・品質filter
  → PSPL初期fit
  → 惑星signal before scan / refine
  → FSPL・annual parallax等のeffect routing
  → 必要な場合だけphysical fallback
  → 採用single-lensモデル決定
  → 採用モデルをwarm startしたpost-physical惑星再探索
  → 必要なら同じモデル族を局所maskして再fit
  → 最終fitのraw residualからpeak/dip特徴量抽出
  → template-free search
  → figure・JSON生成
  → HTML event/index更新
  → portal sync request
  → 次のイベント
```

全シーズンの代表点に置き換えてfallbackを行うことはしない。fitとfallbackはイベントの
F146全点を入力にする。

## 主な実装ファイル

- `tools/rges_anomaly_pipeline.py`
  - RGES Parquetからイベント単位でF146を読む。
  - `Finder.run_anomaly_pipeline()` の一行APIで、effect-aware Finder、fallback、
    post-physical refinement、frozen final-residual measurement、feature extraction、
    template-free searchを最後まで実行する。
  - 1イベントごとにJSONとfigureを書き、HTML更新を呼ぶ。
- `tools/build_rges_anomaly_html.py`
  - Roman本家 `roman_simu/tool/make_html.py` のCSS/JavaScriptブロックを読み、
    同じUI構造でRGESのindex/eventページを生成する。
  - 旧template scan表示は公開HTMLから除外している。
- `src/jacscanomaly/finder.py`
  - `run_effect_aware()` が planet-before → router/fallback → planet-after を管理する低レベルAPI。
  - `run_anomaly_pipeline()` が採用fit確定 → 必須のfrozen residual measurement →
    features/template-freeまで管理する公開完走API。
- `src/jacscanomaly/anomaly_pipeline.py`
  - `AnomalyPipelineConfig` と `AnomalyPipelineResult` を定義し、fit除外maskと
    residual測定maskを別フィールドで保持する。
- `src/jacscanomaly/planet_signal.py`
  - raw residual由来の惑星候補、局所mask、peak/dip測定を行う。
- `src/jacscanomaly/plot.py`
  - `_adaptive_single_lens_curve()` が任意時刻で採用single-lensモデルを評価し、
    曲率の大きい部分だけadaptiveに密にサンプリングする。

## JSONとmaskの意味

RGESのsource JSONは次にある。

```text
/moao39_13/nunota/rges-data/anomaly_finder_result/events/beginner/<event>.json
/moao39_13/nunota/rges-data/anomaly_finder_result/events/experienced/<event>.json
```

`series.signal_mask` はExtractorの解析用maskであり、残差測定窓として広くなる可能性が
ある。この配列をHTMLのオレンジ表示に直接使ってはいけない。

公開用には次を別に保存する。

```text
series.signal_mask          解析上の元mask。後方互換・provenance用
series.fit_exclusion_mask   最終fitで実際に除外された点
series.display_signal_mask  HTMLのオレンジ表示に使うmask
```

`display_signal_mask` は、accepted iterationとzero-weight点から作る実際のfit除外maskで
ある。post-physical refinementが0回、または採用physical fitより悪化してrollbackした場合は
全点0になる。HTML側も`display_signal_mask`だけを優先して使い、存在しない古いJSONの
`signal_mask`へフォールバックしない。

これは、測定窓全体がオレンジになる問題を防ぐための重要な仕様である。

## モデル線とデータ点

HTMLのmodel線は、観測データの`series.model_flux`を点と点の間で結んだものではない。

解析時に採用fitへ次を適用して専用曲線を作る。

```python
from jacscanomaly.plot import _adaptive_single_lens_curve

time_model, flux_model = _adaptive_single_lens_curve(
    adopted_fit,
    saved_xlim,
    base_points=192,
    max_points=4000,
)
```

この結果はsource JSONの`plot.model_curve`へ保存される。HTMLはこの曲線だけをmodel線として
描画する。したがって、観測cadence、欠測、公開JSONのデータ点間引きにmodel線が引きずられない。
FSPL/parallax fitもfit objectのmodel evaluatorを通して同じ経路で描画する。

## HTMLの表示範囲と保存範囲

本家Romanの設計に合わせ、表示範囲とJSONの保存範囲を分けている。

```text
plot.peak_xlim  初期表示用。イベントスケールを中心に、離れた候補も含める
plot.saved_xlim 保存データ範囲。peak_xlimの前後2日を追加
series          saved_xlim内の点。model_curveはseriesとは別のadaptive曲線
```

初期表示は全シーズンではなく、概ね`3 * tE`の有効幅、最小半幅5日を基準にする。
`display_signal_mask`、template-free候補、featuresの範囲が外側にあれば、その範囲を初期表示へ
含める。保存データにはさらに前後2日を持たせるので、Plotlyで少し外側へズームできる。

公開HTMLの配置:

```text
/rogue1_8/nunota/html_portal/rges_anomaly_finder/index.html
/rogue1_8/nunota/html_portal/rges_anomaly_finder/events/<event>.html
/rogue1_8/nunota/html_portal/rges_anomaly_finder/planet_signal_data/<event>.json
```

公開URL:

```text
http://133.1.160.32/ou-moa/public/rges_anomaly_finder/index.html
http://133.1.160.32/ou-moa/public/rges_anomaly_finder/events/<event>.html
```

indexのMain参照は`http://133.1.160.32/ou-moa/index.html`である。

## portal同期とRagan公開

各イベント完了後、runnerが次を実行する。

```text
build_rges_anomaly_html.py
  → /rogue1_8/nunota/html_portal/tool/request_sync.sh
  → .sync/requestへトークンを書く
  → watch_sync_html_portal.shがportalを再構築
  → rsyncでprime@ragan:/home/prime/Public/ou-moa/へ転送
```

同期状態:

```bash
cat /rogue1_8/nunota/html_portal/.sync/request
cat /rogue1_8/nunota/html_portal/.sync/done
tail -f /rogue1_8/nunota/html_portal/watch_sync.log
```

`request`と`done`が一致すれば、その同期要求は完了している。一時的に`ragan`の名前解決や
SSHが失敗しても、request tokenは消えずwatcherが再試行する。公開側の確認は次で行う。

```bash
curl -fsS http://133.1.160.32/ou-moa/public/rges_anomaly_finder/index.html \
  | rg 'RGES|rges_anomaly_finder'
curl -fsS http://133.1.160.32/ou-moa/public/rges_anomaly_finder/planet_signal_data/RMDC26_000001.json \
  | python -m json.tool >/dev/null
```

## runを止める・再開する

現在のrunを止める場合:

```bash
tmux send-keys -t rges_f146_serial C-c
```

停止後の再開は、同じコマンドを使えば既存JSONを基準に継続できる。ただし、コードやHTML仕様を
変更して全件再計算する場合は`--force`を付ける。新しいlogを使い、古いrunと同じtmux名を
二重起動しないこと。

```bash
tmux has-session -t rges_f146_serial && echo already-running
tmux new-session -d -s rges_f146_serial -c /rogue1_8/nunota/jacscanomaly \
  "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_NUM_THREADS=1 \
   taskset -c 0 python tools/rges_anomaly_pipeline.py --tier both --force --build-html \
   --progress-file /moao39_13/nunota/rges-data/anomaly_finder_progress.txt \
   >> /moao39_13/nunota/rges-data/rges_f146_serial_vNN.log 2>&1"
```

イベント子プロセスの標準出力は抑制している。エラーは親logの`ERROR`行と、source output下の
`errors.jsonl`に残る。manifest更新は、途中失敗で`fit.refined=null`になった古いartifactが
あってもrun全体を起動直後に落とさないよう、空dictとして扱う。

## 検証コマンド

コード変更後の最低限の確認:

```bash
cd /rogue1_8/nunota/jacscanomaly
python -m py_compile tools/rges_anomaly_pipeline.py tools/build_rges_anomaly_html.py
pytest -q tests/test_rges_anomaly_masks.py tests/test_planet_signal.py
```

代表JSONで、model線が観測点の再利用でないことを確認する。

```bash
python - <<'PY'
import json
p = '/moao39_13/nunota/rges-data/anomaly_finder_result/events/beginner/RMDC26_000001.json'
d = json.load(open(p))
curve = d['plot']['model_curve']
print('model points:', len(curve['time']))
print('data points:', len(d['series']['time']))
print('display points:', sum(d['series']['display_signal_mask']))
print('refits:', d['physical_fallback']['diagnostics']['post_physical_refits_completed'])
PY
```

期待する状態は、`plot.model_curve`が存在し、通常は`series.time`と異なるadaptive点列で、
HTML用`display_signal_mask`が解析用`signal_mask`と混同されていないことである。

## 既知の注意点

- F146以外は読まない。複数bandの全band処理へ勝手に拡張しない。
- model線を`series.model_flux`から再構成しない。必ず`plot.model_curve`を使う。
- `signal_mask`の合計を「実際の惑星除外点数」と解釈しない。除外点数は
  `fit_exclusion_mask`または`display_signal_mask`を見る。
- `post_physical_refits_completed=0`のイベントで全点オレンジになってはいけない。
- 物理fallback後のrefinementが親fitより悪化した場合は、rollback後のfitと空maskを公開する。
- 1イベント処理中にrunをkillすると、そのイベントは再計算対象になる。途中JSONを成功済みと
  みなさず、logの`done`行とprogress行を確認する。
- Raganへ実際に転送されたかは、ローカルHTML生成完了ではなく`.sync/done`と公開URLで確認する。

## 統合 anomaly candidate API

`Finder.run_anomaly_pipeline(...)` の通常の判定・表示入口は次の3つである。

```python
result.has_anomaly_candidate     # bool
result.best_anomaly_candidate    # dict | None
result.anomaly_candidates        # rank順の list[dict]
```

`final_residual_feature` と `template_free` が同じ時間領域を検出した場合は別候補として
二重計上せず、1行へ統合する。時刻・符号・幅はfeature測定を優先し、`chi2`、
`reduced_chi2`、`n_points`はtemplate-freeから補う。`sources`には両方の由来が残る。
新規RGES event JSONではトップレベルの`anomaly_candidates`がHTML・manifestの正規候補一覧で、
`features`と`template_free`は監査用の詳細結果として残す。古いJSONをHTMLだけ再生成する場合は、
builderが旧2項目から候補を読む互換経路を持つ。
