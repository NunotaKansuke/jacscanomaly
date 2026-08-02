# 惑星シグナル抽出の現状と高速化方針

作成日: 2026-07-29

## 2026-07-29 finalize 実装

全イベント処理は、以下の3段階へ確定した。

1. 全2,371イベントを `PlanetSignalConfig.fast()` と PSPL で一度だけ scan する。
2. 保存済み PSPL fit を読み、非線形fitを行わない物理routerで
   annual parallax、Roman space parallax、FSPLを判定する。
3. routerが `fallback` としたイベントだけnative FSPL/parallax fitを行い、
   受理結果だけmain HTMLの対応イベントへoverlayする。

この構成では、planetの `initial_seed.dchi2` を物理fallback選択へ流用しない。
惑星残差の強さとFSPL/parallaxの必要性は別の判定問題だからである。

### 物理routerから除去した全イベント向けコスト

旧FSPL detectorはイベントごとにmicrojaxで30本のcurveを評価し、観測点数の
異なる配列をJAX/XLAへ渡していた。また各templateで同じnuisance SVDを
繰り返していた。1イベントのprofileでは次の内訳だった。

| 区間 | 旧実測 |
|---|---:|
| `detect_physical_effects()` | 約10.5 s |
| FSPL detector | 約8.7 s |
| FSPL bank生成 | 約5.6 s |
| JAX/XLA compile | 約5.3 s |
| templateごとの射影 | 約2.4–3.0 s |

finalize版では次を実装した。

- PSPL magnificationと `(t0, tE, u0)` JacobianをNumPyの解析式で計算
- parallaxが `piEN`, `piEE` に線形であることを使い、unit directionを直接計算
- uniform-source FSPLは標準依存のVBMicrolensing C++ backendで評価
- rectilinear FSPLで同一な `+u0` / `-u0` curveを一度だけ生成
- finite-source差が無視できるwingではexact PSPLを使用
- 全FSPL templateを一つの行列として、一回のnuisance SVDで一括score
- Roman satellite ephemerisとGULLS observer orbitをprocess内cache
- scan JSONのPSPL params、`fs`、`fb`をrouterで再利用し、二重fitを廃止
- detector diagnosticsを再帰的にJSON-safe化

`microjax` backendはcross-validation用に明示指定した場合だけ残し、survey
routerのdefault pathでは使用しない。

### 精度・速度検証

- effect detection / space parallaxを含む全test: `150 tests`
- 追加後のFSPL detector test: 解析Jacobian、JAX基準parallax tangent、
  batched projection、sign-degeneracy再利用、JSON serializationを検証
- 合成curveでのnative FSPL detector:
  200点 約0.004 s、2,000点 約0.012 s、10,000点 約0.093 s
- Roman実イベント3件でfull-curve nativeとsupport-limited nativeを比較:
  routing decisionは全件一致、FSPL score差は最大約0.09%
- 24イベント並列smokeを2回実施:
  各回24/24完了、error 0、fallback indexも一致
- BLAS/OpenMPをprocess poolの外側で1 threadへ固定後、120イベント、
  12 workerをwall 21.47 sで完了（import/startup込み、error 0）
- 全2,371イベントrouterを完了: completed 2,371、error 0、fallback 950

physical stageはrouterを再実行しない。router JSONのcandidate/seedを
`robust_fallback()`へ直接渡し、保存scanのPSPL seedからbaselineを一度だけfitする。
fallback受理後だけ `PlanetSignalConfig.fast()` で惑星残差を再抽出し、physical
artifactを書く。不受理イベントはartifactを書かないためmain HTMLのscan行を
上書きしない。これにより旧physical runnerにあった以下の重複を除去した。

- `bic_single_lens` baselineで物理modelを先に全探索
- planet-before full beam
- detectorとexact probeの再実行
- fallback後のplanet-after full beam

堅牢fallbackの `8 seeds × contamination 8 iterations` は維持した。4×4、6×4、
8×4へ減らす試験では独立basin再現条件を失って正しい受理eventがrejectされたため、
速度目的では緩和していない。

全件runnerは `run_planet_signal_pipeline.py --skip-scan --physical` で、
router、fallback queue、physical fit、HTML overlayまで連続実行する。

## 目的

FSPL/parallax の effect-aware pipeline を全イベントへ適用したとき、通常イベントまで数十秒かかる原因を整理する。特に、惑星シグナルの抽出 (`PlanetSignalExtractor`) と、最後のピーク特徴量・時間スケール計算を分離して評価する。

## 現在の処理フロー

effect-aware runner では、概ね次の順で処理している。

1. 初期 single-lens fit を1回実行する。
2. `PlanetSignalExtractor` で惑星シグナルを先に抽出する。
3. FSPL/parallax detector を実行する。
4. fallback が必要なイベントだけ native fallback fit を実行する。
5. fallback が受理された場合だけ、惑星シグナルを再抽出する。
6. 特徴量計算、plot、JSON保存を行う。

`initial_fit` を Extractor に渡せるようになっており、effect-aware 経路で初期 fit を重複実行しない。[`planet_signal.py`](../../src/jacscanomaly/planet_signal.py) の `PlanetSignalExtractor.run()` がこの共有を受け持つ。

## 最新変更で軽くなった部分

### 最後の惑星ピーク処理

以前存在した局所的な物理テンプレート fit / SciPy 最適化は削除されている。現在の `measure_features()` は次だけを行う。

- 残差を誤差で規格化
- 軽い平滑化
- 正負の極値・prominence 判定
- 閾値交差点を線形補間
- `timescale = t_end - t_start` を計算

したがって、現在の `timescale` は物理テンプレート fit の推定値ではなく、残差ピークの閾値幅である。計算量は観測点数に対してほぼ線形で、実測上は無視できる時間だった。

なお、runner の `result_summary()` と `plot_signal()` が特徴量を2回計算しているが、特徴量計算自体がほぼ0秒なので、現時点の主要ボトルネックではない。

## 実測値

`roman_simu` の1イベント（PSPL、C++ grid backend、初期値あり）を単独実行し、各内部メソッドを計測した。

| 区間 | 実測時間 | 備考 |
|---|---:|---|
| 初期 single-lens fit | 約0.04 s | 初期値ありの場合 |
| `PlanetSignalExtractor.run()` | 約11.9 s | 通常イベントでも発生 |
| `_scan_best()` | 7回、合計約10.4 s | 1回あたり約1.5 s |
| masked single-lens fit | 24回、合計約1.5 s | beam候補の評価 |
| `measure_features()` | 約0 s | 現在はfitなし |
| `plot_signal()` | 約2.0 s | matplotlibとモデル曲線生成 |

別の effect-aware profile でも、惑星抽出だけで約15秒前後かかっており、JAX初回コンパイルの有無で変動する。

## 現在の主ボトルネック

デフォルトは `baseline_mode="beam_interval"` で、最大3反復・beam幅3・各反復最大8候補である。

各枝について `_scan_best()` が `SeasonGridRunner.run()` を呼び、残差の `(t0, teff)` grid を再走査する。grid の内側は既にC++ backendだが、同じ探索を複数回呼んでいるため、C++化だけでは本質的な短縮にならない。

特に次のケースが無駄になる。

- 初回 scan が閾値未満でも、残りの反復で同じ枝を再scanする。
- 候補 mask を作っても、採用枝が増えなかった場合に次の反復へ進む。
- 通常イベントにもbeam探索を一律適用している。

したがって、ボトルネックは「ピークfit」ではなく「Extractor内部の反復grid scan」である。

## C++化に関する判断

全Extractorを直ちにC++へ書き直す必要はない。現在でもgrid評価の内側はC++であり、まず探索回数を減らす方が効果と安全性のバランスがよい。

将来的に専用C++化する場合は、次の境界が適切である。

- C++: `(t0, teff)` grid、上位候補interval、候補スコア、必要ならbeam枝の列挙
- 既存native backend: VBM/FSPL/parallaxの増光率計算
- SciPy: 最終連続パラメータの `least_squares` / LM fit
- Python: ルーティング、maskの管理、結果の保存、惑星特徴量

ピーク特徴量・時間スケール計算をC++化しても、現状の総時間にはほとんど影響しない。

## 次に行うべき実装

### 第1段階: Python側の早期終了（実装済み）

1. 初回 `_scan_best()` が `seed_min_dchi2` 未満なら即終了する。
2. 反復後に新しい枝が1つも採用されなければ即終了する。
3. 通常イベント用に `beam_max_iter=1`, `beam_width=1` 相当のfast modeを用意する。
4. 怪しい惑星シグナル、FSPL/parallax候補だけbeam探索を有効にする。

`beam_interval` は、seed が閾値未満の場合、または候補 interval が一つも
採用されなかった場合に反復を終了する。したがって、通常イベントで同一の
未変更 branch を再scanしない。

`PlanetSignalConfig.fast()` は `beam_max_iter=1`、`beam_width=1`、
`beam_candidates_per_iter=1` の一回走査用設定を返す。さらに
`PlanetSignalConfig.probe()` は候補 fit をせず最初の grid seed だけを返す。
effect-aware runner は既定で probe を使い、seed が有意、または fallback 対象の
FSPL/parallax候補が出たときだけ通常の完全 beam search に昇格する。この昇格では
probe seed を渡して初回 grid scan を再利用するため、fast → full で同じ grid を
二度評価しない。`PlanetSignalResult.timing` には合計時間、grid scan時間、scan回数を
保存し、次段階のC++ scanner判断に使う。

この段階の目標は、通常イベントのExtractorを約1〜3秒へ下げること。

### 第2段階: 小規模ベンチマーク

少なくとも次のカテゴリを各20件程度で測る。

- 通常PSPL
- 惑星シグナルが強いイベント
- parallaxが強いイベント（例: `0_599_2302`）
- FSPL/parallaxと惑星シグナルが混在するイベント（例: `2_755_3280`）

記録する項目は、Extractor時間、detector時間、fallback時間、受理率、惑星候補の再現率、誤マスク率とする。

### 第3段階: 専用C++ scanner（必要な場合のみ）

第1段階後もExtractorが支配的なら、複数beam枝をまとめて処理するC++ scannerを追加する。最終fitと物理モデルは既存native backendを再利用し、C++移植範囲を候補探索に限定する。

## 現時点の結論

- 最後の惑星ピーク・時間スケール処理は既に十分軽い。
- 初期fitの重複もeffect-aware経路では解消済み。
- planet scanはfast modeの全件処理、物理解釈はdetector routing後のsubset処理とする。
- 物理routerが遅かった主因はmicrojax/JAXのevent-shape compileとtemplateごとの
  重複射影であり、native C++ + NumPy analytic/batched pathへ置換した。
- 全イベントに完全beamや完全物理fitを実行しない。高価な処理は各routerが選んだ
  subsetだけに限定する。
