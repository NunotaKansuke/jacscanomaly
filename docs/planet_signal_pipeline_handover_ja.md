# Roman planet-signal パイプライン／HTML公開 引き継ぎ

最終更新: 2026-07-31
現行コード: `jacscanomaly` commit `938f1cc`。主要変更は、保存済みFSPL解の `rho`
復元（`9143a7e`）、惑星抽出からbinned残差を完全に除外（`fccf85b`）、
一峰性惑星候補判定のC++化（`938f1cc`）。

## 目的

目的は、惑星らしい局所信号をPSPL/FSPL/parallaxの母体増光と混同せず、
最終的に採用したsingle-lensモデルの残差から惑星候補のpeak/dipを抽出すること。

重要な原則は以下。

- 最初のPSPLは探索用の粗いベースライン。最終解ではない。
- 物理診断は、重いFSPL/parallax fitを走らせる対象を選ぶ**ルーター**である。
- fit後の採否は数値的有効性・境界解・BICで決める。形態診断は警告であり、単独では棄却しない。
- 最終的なFSPL/parallaxモデルの残差で惑星探索をやり直し、局所マスク後に同じモデル族をwarm-start LMで最大3回だけ再フィットする。
- post-physical fitが親の物理解より悪化したら、親の物理解へ戻す。
- **惑星候補の抽出・局所mask判定にbinned残差は使わない。** binnedはsingle-lensの
  初期化・安定化を検討するためだけの補助であり、peak/dipの強度、幅、mask点は必ず
  最終モデルに対するraw residualから求める。これにより、明瞭なdip/compact causticを
  binned処理で消す問題を防ぐ。

## 処理フロー

```text
全イベント
  → scan: PSPL基準の惑星候補探索
  → effect router: FSPL / annual parallax / space parallax の必要性を安価に診断
  → physical fallback: ルーター通過イベントだけ複数seedで物理モデルfit
  → hierarchical BIC: 有効なsingle-lens候補から最良モデルを採用
  → post-physical planet refinement:
       採用モデルの残差で惑星候補を再探索
       → 局所マスク
       → 同じモデル族をwarm-start LM（最大3回）
       → 悪化時は物理解へrollback
  → final residual measurement:
       adopted single-lens modelを固定
       → raw residualからpeak/dipを測定（baselineの再fitはしない）
  → 既存の adopted-model HTML event/index を更新
  → portal sync request
  → Ragan公開
```

## 主要実装

### `jacscanomaly`

- `src/jacscanomaly/finder.py`
  - `evaluate_saved_physical_solution()` が物理解を再評価する。
  - FSPL+parallaxではpublic `rho` をnative seed用の `log_rho` に変換する。
  - 過去の不具合: `log_rho` の綴りを認識せず、`rho=0.02` を対数値として渡したため `exp(0.02)≈1.02` になった。`9143a7e`で修正済み。

- `src/jacscanomaly/singlelens_fallback.py`
  - multiseed fallbackと階層BIC選択。
  - optimizer失敗、境界解、BIC非改善だけを棄却の主因にする。

- `src/jacscanomaly/effect_routing.py`
  - 低コストのFSPL/parallax診断とexact-probe routing。
  - 長時間でcoherentなparallax候補を低scoreでも救済する。

- `src/jacscanomaly/planet_signal.py`
  - 惑星候補・局所maskはraw residualだけで構成する。binned系列は候補選別、maskの拡大・
    縮小、peak/dipの報告には渡さない。
  - compactな候補を守りつつ、幅広く滑らかな成分を惑星候補としてmaskし過ぎないよう、
    template seed・raw残差の連続性・局所一峰性を使う。
  - 一峰性のone-lobe判定は、利用可能なら `_cpp_grid.unimodal_mask` を使う。拡張未build時
    だけ旧JAX実装へフォールバックする。

- `src/jacscanomaly/_cpp_grid.cpp`
  - 上記の候補ごとの一峰性判定をC++で実装。実測では `0_46_2422` の512候補で、JAX初回
    約0.33秒に対してC++約0.003秒、選択結果は一致した。
  - 拡張を更新した後はリポジトリ直下で `python setup.py build_ext --inplace` を実行する。

高次効果のhot pathもJAX必須ではない。FSPL評価はnative VBM C++、parallaxはnative
evaluator/JacobianとSciPy TRFを使用する。JAXは互換用・可視化・旧経路に残るが、通常の
router→physical fallbackのボトルネックではない。

### `roman_simu/tool`

- `run_planet_signal_full_refresh.py`
  - survey全体を一貫して実行するエントリーポイント。
  - scan → router → physical → post-physical → **既存adopted-modelページ生成** → 同期リクエスト。
  - 専用の別viewerは生成しない。

- `run_post_physical_planet_refinement.py`
  - 採用物理解の再構成後に惑星探索・局所マスク・同モデル族の再fitを実行。
  - post fitの `chi2/dof` が親の物理解より `max(1.25倍, +0.5)` を超えて悪化、または非有限なら、親モデルにrollbackして空maskを採用する。

- `run_final_residual_measurement.py`
  - scanまたは採用済みphysical/post-physical fitを再構成し、全イベントでfrozen residual measurementを実行する。
  - 局所maskはpeak/dipの測定にだけ使い、single-lens baselineを再fitしない。そのため、mask後の
    `tE` compactnessや継続fitの失敗が、実在するraw residual featureを消すことはない。

- `make_html.py`
  - 既存公開URLの生成器。
  - `ROMAN_ADOPTED_MODEL_SITE=1` のとき、`anomaly_finder_model_result/events/*.html` と一覧を出力する。
  - デフォルト入力は最新tagのscan/post-physical結果。`ROMAN_ADOPTED_MODEL_SITE=1`時は
    旧 `planet_feature_data` を混在させず、最終採用モデルの残差・mask・候補だけを描く。
  - ページ右上にbuild versionを表示する。

## 最新の全件生成物

直近の正常完了runは tag = `cpp_raw_mask_hierarchical_bic_full_v20260731`。

- scan: 2371/2371 完了
- router: error 0
- physical fallback: 対象217件中、採用181件・棄却36件
- post-physical refinement: 181/181 完了
- HTML生成・Ragan同期: 完了。公開ページのbuild versionは `v2026.07.31.0310`

```text
/rogue1_8/nunota/roman_simu/anomaly_finder_result/
  planet_signal_cpp_raw_mask_hierarchical_bic_full_v20260731_scan_data/
  planet_effect_route_cpp_raw_mask_hierarchical_bic_full_v20260731.json
  planet_signal_cpp_raw_mask_hierarchical_bic_full_v20260731_physical_data/
  planet_signal_cpp_raw_mask_hierarchical_bic_full_v20260731_post_physical_data/

/rogue1_8/nunota/roman_simu/html_portal/
  anomaly_finder_model_result/index.html
  anomaly_finder_model_result/events/<event>.html
```

公開URLは次。

```text
http://133.1.160.32/ou-moa/public/roman_simu/anomaly_finder_model_result/
http://133.1.160.32/ou-moa/public/roman_simu/anomaly_finder_model_result/events/<event>.html
```

`planet_signal_binned_local_full` のような別viewerは公開しない。既存event URLを最終モデルで直接更新する。

## 全体再実行

```bash
cd /rogue1_8/nunota/roman_simu
env PYTHONPATH=/rogue1_8/nunota/jacscanomaly/src \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
  nice -n 10 python -u tool/run_planet_signal_full_refresh.py \
    --jobs 12 \
    --tag cpp_raw_mask_hierarchical_bic_full_vYYYYMMDD \
    --event-timeout 600 \
    --post-event-timeout 300 \
    --max-refits 3
```

成功時、`make_html.py` が既存modelページを更新し、
`/rogue1_8/nunota/html_portal/tool/request_sync.sh` が同期要求トークンを書き込む。

ローカル生成だけをしたい場合は `--no-request-sync` を使う。

このrunnerはイベントごとに最後まで流すのではなく、survey全体で
`scan → router → physical → post-physical → final residual measurement → HTML → sync` の順にstageを完了する。
従ってphysicalの進捗はscan完了後に現れる。`--jobs` は他の利用者のCPUを優先して設定し、
上記の12は空きCPUが十分あるときの低優先度実行例である。

進捗確認（`.rejected.json` を成功数に混ぜないこと）:

```bash
base=/rogue1_8/nunota/roman_simu/anomaly_finder_result
tag=cpp_raw_mask_hierarchical_bic_full_v20260731
find "$base/planet_signal_${tag}_scan_data" -name 'planet_signal_result_*.json' | wc -l
wc -l < "$base/planet_effect_route_${tag}.indices.txt"
find "$base/planet_signal_${tag}_physical_data" -name 'planet_signal_result_*.json' ! -name '*.rejected.json' | wc -l
find "$base/planet_signal_${tag}_physical_data" -name '*.rejected.json' | wc -l
find "$base/planet_signal_${tag}_post_physical_data" -name 'planet_signal_result_*.json' | wc -l
```

## HTMLだけ再生成する場合

```bash
cd /rogue1_8/nunota/roman_simu
ROMAN_ADOPTED_MODEL_SITE=1 \
ROMAN_PORTAL_VERSION=vYYYY.MM.DD.N \
PYTHONPATH=/rogue1_8/nunota/jacscanomaly/src \
python tool/make_html.py
```

一覧と各eventページの右上に同じversionが出る。version未指定時は時刻付きversionを自動採番する。

## Ragan同期

同期要求・完了トークン:

```bash
cat /rogue1_8/nunota/html_portal/.sync/request
cat /rogue1_8/nunota/html_portal/.sync/done
```

両者が一致すれば、その要求は完了している。

通常はmoao0側の常駐同期が要求を処理する。必要時だけローカルwatcherを使う。

```bash
/rogue1_8/nunota/html_portal/tool/start_watch_sync_html_portal.sh
/rogue1_8/nunota/html_portal/tool/stop_watch_sync_html_portal.sh
tail -f /rogue1_8/nunota/html_portal/watch_sync.log
```

Ragan名解決が一時失敗しても、request tokenは残る。`done`が追いつくまで再試行する。

## 公開確認

```bash
curl -I http://133.1.160.32/ou-moa/public/roman_simu/anomaly_finder_model_result/events/0_1_2520.html
curl -s http://133.1.160.32/ou-moa/public/roman_simu/anomaly_finder_model_result/events/0_1_2520.html \
  | rg 'build v|post_physical|FSPL|parallax'
```

HTMLの値とJSONの一致も確認する。

```bash
jq '.fit.model_kind, .fit.chi2_dof, .pipeline.stage' \
  anomaly_finder_result/planet_signal_cpp_raw_mask_hierarchical_bic_full_v20260731_post_physical_data/planet_signal_result_1129_1199.json
```

## 判断の読み方: 0_922_1199

このイベントは最終的に `fspl_space_parallax` が採用された。

- PSPLからのBIC改善は非常に大きく、optimizerも数値的に有効。
- post-physicalのwarm-start再fitも2回完了し、`chi2/dof≈2.624`。
- 一方でrouterには `parallax_wings_incoherent`、`subset_unstable`、`non_fspl_peak_shape` の警告が残る。

従って、これは「手作業で無理に通した解」ではないが、形態がきれいに支持した確定解でもない。表示・運用上は **BIC採用・要レビューのFSPL+space-parallax候補** と扱う。
