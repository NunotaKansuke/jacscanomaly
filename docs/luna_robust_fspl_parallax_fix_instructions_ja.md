# Luna向け: FSPL / parallax detector・routing・fallback 修正指示

## 0. この文書の目的

現在の実装は、物理 residual detector の基本方針と低レベル計算は有望である。一方、
routing と fallback はまだ production pipeline へ接続できる状態ではない。

この文書に従い、既存実装を捨てずに Phase 1 / Phase 2 を完成させること。
Phase 3 の planet pipeline 再接続、ML router、HMC / SMC にはまだ進まない。

最優先の成功条件は次の二例である。

- `../roman_simu` の `2_755_3280`: FSPL の良い basin を truth seed なしで発見する。
- `../roman_simu` の `0_599_2302`: compact planet contamination があっても
  GULLS space parallax を検知し、bound 張り付きでない basin を発見する。

最終目的は unit test を通すことではない。上記二例と mixed synthetic regression を通し、
「検知できるが fit されない」「悪い fit を success と返す」という状態をなくすことである。

## 1. 現在の評価

### 残してよいもの

- nuisance Jacobian を SVD / QR で射影する設計
- parallax tangent score test
- FSPL の `tE` / `u0` / `rho` を結合して探索する方針
- detector と routing policy の責務分離
- contiguous anomaly を latent state として扱う方針
- detector seed を fallback に渡す API
- 従来の `Finder.run()` を変更しない shadow-mode の入口

### 現在観測されている問題

対象データで現実装を走らせると次の結果になる。

```text
2_755_3280
  FSPL score                  ≈ 232680
  score without compact block ≈ 115972
  subset stability            = 0.0
  routing                     = exact_probe

0_599_2302
  parallax score               ≈ 91650
  score without compact block  ≈ 105293
  subset stability             = 0.0
  routing                      = exact_probe
```

両方とも detector は強く反応しているが、raw subset score の比較によって
`exact_probe` に降格される。現在 `exact_probe` の実行経路は存在しないため、
cascade はここで停止する。

さらに `2_755_3280` で現在の FSPL fallback を強制実行すると次の結果になる。

```text
valid FSPL attempt     = 1
returned rho           ≈ 0.0148
truth rho              ≈ 0.257
weighted chi2          ≈ 132510
known good-basin chi2  ≈ 42655
alternating converged  = false
FallbackResult.success = true
```

この状態を「基盤実装済み」とは呼べるが、「Phase 2 完了」とは呼ばないこと。

## 2. 作業順序

次の順番を守ること。後段の task を先に実装して、前段の失敗を threshold で隠さない。

1. 対象二例を再現する benchmark / regression harness
2. subset diagnostics と routing
3. effect-specific fitter factory
4. 全次元 structured multistart
5. contamination protected-support policy
6. 共通 objective と fit success 判定
7. FSPL template profiling
8. FSPL + parallax staged joint fallback
9. curated shadow benchmark
10. 計画書の実装状況更新

## 3. Task A: 対象二例を regression harness にする

### 要件

production test suite が巨大な simulation repository に常時依存しないよう、test と benchmark を
分ける。

#### 軽量 regression fixture

対象二例から、再配布可能性と repository size を確認したうえで、最低限の以下を fixture または
生成 script として固定する。

- time
- flux
- ferr
- event metadata
- RA / Dec
- observer ephemeris の参照方法
- baseline PSPL seed
- 期待する effect class
- known-good diagnostic values

実データ配列を repository に入れられない場合は、`../roman_simu` から読み出す
optional regression script を作る。通常の unit test では同じ failure mode を再現する
small synthetic fixture を使用する。

#### Benchmark command

一つの command で次を JSON または JSONL に保存できるようにする。

```text
event
PSPL params / chi2
all detector diagnostics
routing decision / reason codes
all fallback seeds
all attempt objectives
original-error chi2
robust objective
final physical params
convergence
success / failure reason codes
runtime
```

benchmark artifact は Git 管理しない。schema と小さい期待値だけ test に置く。

### 禁止事項

- regression test に truth parameter を optimizer seed として渡さない。
- target event 名だけを見た分岐を作らない。
- 対象二例だけ通る固定 threshold を入れない。

## 4. Task B: subset stability を物理量の整合性へ変更する

### 現在の問題

`_subset_stability()` は、点数・時間 span・template information が異なる subset の
raw `Delta chi2` の変動係数を計算している。raw score は subset size と information に依存するため、
同じ物理 signal でも容易に `0.0` になる。

計画書で必要としているのは raw score equality ではなく、物理方向・振幅・support の整合性である。

### Parallax

各 subset について少なくとも次を返す。

```text
beta = best linearized (piEN, piEE)
direction = beta / |beta|
score
template information = trace(H_perp.T H_perp)
effective rank
condition number
```

stability は次の組合せで計算する。

- 有効 rank を持つ subset 間の direction cosine
- amplitude の不確実性を考慮した整合性
- `score / effective_information` または同等の正規化量
- leave-one-compact-block-out 後の full score 維持率

pre / post を必ず同じ score にする必要はない。parallax geometry 上、一方が強くなることは正常である。
情報量が小さい subset は不安定判定へ寄与させず、`insufficient_subset_information` として記録する。

### FSPL

FSPL では少なくとも次を分けて評価する。

- peak 左側
- peak 右側
- central support
- central support から compact block を一つ除いた場合

比較するものは raw score の一致ではなく、

- template coefficient の物理符号
- best `rho / |u0|` または `t_star` family の整合性
- 左右の minimum effective coverage
- unmatched residual energy

である。

### Routing rule

- 桁違いに強い full score が、単に subset raw score が異なるという理由だけで
  dead-end の `exact_probe` に送られないこと。
- `exact_probe` executor が完成するまでは、high-score uncertain case を
  `fallback` または明示的な `fallback_after_probe_unavailable` に送る fail-open policy とする。
- low-score geometry failure は `skip` してよいが、reason code を残す。
- `score_without_compact_blocks` を routing quality に実際に使用する。

### 必須 test

- 同一 parallax signal を異なる点数の pre / post subset に分けても、不当に stability `0` にならない。
- direction が反転する injection は unstable になる。
- compact block を除いて physical score が維持される場合は fallback 可能。
- compact block だけで score が出る planet-like injection は fallback へ直行しない。
- 対象二例が dead-end `exact_probe` で停止しない。

## 5. Task C: `exact_probe` を実行可能な経路にする

`exact_probe` を decision string だけで終わらせない。

### Parallax exact probe

uncertain event のみ、少数の exact forward model を評価する。

- tangent direction とその反対方向
- 直交方向
- 複数の `piE` radius
- relevant degeneracy

各 template で `fs`, `fb` を線形 solve し、必要なら PSPL nuisance を一段だけ profile する。
full nonlinear optimizer は呼ばない。

### FSPL exact probe

projected template 上位の複数 seedについて、full optimizer ではなく次を行う。

- exact FSPL forward evaluation
- `fs`, `fb` の線形 solve
- bounded な小さい `t0 / tE / u0` local profile、または事前定義した近傍grid

candidate detector と expensive fallback の中間コストに保つ。

### Probe output

- pre-probe candidate
- evaluated templates
- best exact improvement
- promoted / demoted decision
- reason codes
- runtime

## 6. Task D: effect-specific fitter factory

### 現在の問題

`Finder.robust_fallback()` は現在設定されている `self.fitter` を再利用する。
fast baseline が PSPL の場合、FSPL / parallax candidate を検知しても PSPL を再fitするだけになる。

### 要件

baseline fitter と fallback fitter を分離する。次の mapping を一箇所で管理する。

```text
fspl             -> FSPL fitter
annual_parallax  -> PSPL annual-parallax fitter
space_parallax   -> PSPL space-parallax fitter
fspl_parallax    -> FSPL annual-parallax fitter
fspl_space_parallax -> FSPL space-parallax fitter
mixed            -> candidate集合とgeometryからjoint fitterを選択
```

factory は `FinderConfig` の以下を引き継ぐ。

- RA / Dec
- `tref`
- observer / satellite ephemeris
- annual / VBM / GULLS convention
- backend
- parameter bounds
- parallax prior 設定
- FSPL numerical settings

`Finder.detect_effects()` は PSPL fit のみを入力として要求し、fallback model の構築は
factory が担当する。

### API要件

- detector result から model kind を明示的に決定する。
- requested effect と fitter parameter dimension が一致しなければ、fit を呼ぶ前に失敗させる。
- `mixed` を全点保護の意味に使わない。
- model choice、backend、parameter convention を artifact に保存する。

### 必須 test

- PSPL Finder + FSPL candidate が FSPL fitter を呼ぶ。
- PSPL Finder + GULLS parallax candidate が GULLS parallax fitterを呼ぶ。
- 3 parameter seed を6 parameter fitterへ黙って渡さない。
- geometry metadata がないparallax candidateは明示的に失敗する。

## 7. Task E: 全次元 structured multistart

### 現在の問題

現在の `tE` factor と `u0` sign perturbation は base PSPL seed にしか適用されない。
FSPL detector seed は4次元のまま一つだけ、parallax seedも限定的である。

また、異なる次元のseedを同じlistに入れ、fitter呼出時の例外で落とす方式になっている。

### 要件

model kind ごとに seed generator を分離する。

#### FSPL seed

最低限、次を組み合わせる。

- detector上位の複数template
- `tE` factor
- `u0` sign
- `u0` magnitude factor、または `teff = |u0|tE` を保つ変換
- `rho / |u0|` family
- `t_star = rho*tE` family
- 小さい `t0` offset

すべてのseedを `(t0, tE, u0, logrho)` へ変換してからfitterへ渡す。

#### Parallax seed

- tangentから得た符号付き `beta`
- multiple radii
- opposite direction
- relevant `u0` sign degeneracy
- `tE` / `u0` coupled perturbation

すべて `(t0, tE, u0, piEN, piEE)` にする。

#### Joint seed

FSPL上位seedとparallax上位seedのCartesian productをそのまま全評価すると高価なので、
段階fit後の上位だけを結合する。

```text
FSPL + parallax:
  (t0, tE, u0, logrho, piEN, piEE)
```

parameter order と raw / physical `rho` convention を model spec に持たせる。

### Seed pruning

- forward-model coarse objective
- parameter bounds
- duplicate除去
- score diversity

で上位seedを残す。同一basinのほぼ同じseedだけで枠を埋めない。

### 必須 test

- FSPL detector seedから複数のvalid 4D seedが生成される。
- parallax detector seedから複数のvalid 5D seedが生成される。
- mixed candidateからvalid 6D seedが生成される。
- seed perturbationで `rho / |u0|` または `t_star` familyが実際に変わる。
- wrong-dimensional seedを例外catchで捨てる実装にしない。

## 8. Task F: protected support を hard mask から制約付き soft contamination へ変更する

### 現在の問題

現在はprotected pointのanomaly emissionを`inf`にしている。この方式では、

- FSPL peak上のplanet anomalyをdownweightできない。
- parallax wing中のplanet anomalyをdownweightできない。
- `rho` / `piE` が惑星へ引っ張られる。

これは今回の主要要件と反する。

### 新しい方針

protected support は「全点を必ずbaseline stateにするmask」ではなく、
「物理効果を拘束する最低情報量を残すconstraint」とする。

#### FSPL

- peak左側・右側それぞれに minimum inlier points / effective weight を要求する。
- central support内の anomaly fraction に上限を設ける。
- 一つまたは少数のcompact blockはsupport内でもanomalyになれる。
- support全体をanomalyにする解は禁止する。

#### Parallax

- compact blockはpeak以外のwing内でもanomalyになれる。
- season / broad wingごとにminimum retained informationを要求する。
- 一つのseason全体やbroad coherent residualをanomalyに逃がさない。

#### Mixed

- FSPL constraint とparallax constraintを別々に満たす。
- 単純なboolean union maskを作らない。

### DP / HMM

- transition penalty
- occupancy penalty
- block span penalty
- season-gap reset

をobjectiveに実際に含める。現在定義だけされている`span_penalty`を使用する。

hard MAP stateだけでなく、重みに使うsoft probabilityの意味を明確にする。
local emission contrastをposteriorと呼ばない。posteriorを必要とするならforward-backwardを実装する。

### 必須 test

- FSPL support中央の局所planet blockがdownweight可能。
- それでもpeak左右の大部分は保持される。
- parallax wing内の局所planet blockがdownweight可能。
- broad parallax residual全体はanomalyにならない。
- season gapをまたぐblockが正しく分割される。
- `span_penalty`を変えるとlong blockの選択が変わる。

## 9. Task G: model間で比較可能な共通 objective

### 現在の問題

現在は概ね次をattempt objectiveとしている。

```text
weighted-error fit chi2 + segmentation objective
```

fit chi2はinflated error上で計算され、segmentation objectiveもresidual likelihoodを含む。
この和はデータ項を二重に数え、model / seed間で比較可能なpenalized likelihoodになっていない。

### 要件

各attemptの最終parameterをoriginal `ferr`上で再評価し、共通の一つのobjectiveを計算する。

概念的には次の一回だけの評価とする。

```math
L(theta, s)
= sum_i ell_{s_i}(residual_i(theta) / ferr_i)
+ transition_penalty
+ occupancy_penalty
+ span_penalty
+ optional physical prior
```

- weighted fit用のinflated errorはoptimizer内部の手段に限定する。
- canonical fit resultの`ferr`をinflated errorへ置換しない。
- `original_chi2`、`robust_objective`、`contamination_penalty`を分けて保存する。
- model comparisonは同じlikelihood familyとpenaltyで行う。
- segmentationをzeroへ強制変更した場合、objectiveもそのstateで再計算する。

## 10. Task H: fit success と basin stability

### 現在の問題

parameter vector全体のEuclidean normを使うと、大きな絶対時刻` t0 ~ 9000`が分母を支配する。
`rho`や`piE`が大きく変わってもstableに見える。

また、alternating fitが未収束でも`success=True`を返し得る。

### Parameter scaling

少なくとも次のdimensionless差を使う。

```text
delta_t0 / tE
delta_log_tE
delta_u0 / max(|u0|, u0_floor)
delta_logrho
delta_piEN
delta_piEE
```

seedからの距離だけでなく、上位attempt同士が同一basinへ集まるかを評価する。

### `success=True` の必須条件

- optimizer statusが成功、または許可した明示的status
- alternating fitが収束
- original-error objectiveがbaselineより改善
- residualの対象effect scoreが十分低下
- parameterが不自然にboundへ張り付いていない
- 少なくとも二つの独立seedが同じbasinまたは同等objectiveへ到達
- anomaly segmentationの小変更で主要parameterが激変しない
- required support informationが残っている

満たさない場合はfit自体を返してよいが、`success=False`とreason codeを返す。

### 必須 reason codes

```text
optimizer_failed
alternating_not_converged
single_seed_only
basin_not_reproduced
parameter_at_bound
effect_score_not_reduced
insufficient_identifiability
contamination_sensitive
objective_not_improved
```

## 11. Task I: FSPL templateを実際にprofileする

### 現在の問題

現在のbankは、新しい` tE / u0`でFSPLとPSPLを評価して引いているが、
各FSPL curveに対するbest PSPL nuisance profileにはなっていない。
さらにruntimeの元PSPL residualとのtrajectory差がtemplateへ十分反映されていない。

`N_fft`引数も実際のmodel evaluationへ渡されていない。

### 要件

FSPL templateは次のどちらかを明示的に採用する。

#### A. Dimensionless profiled atlas

offline / cached生成時に、各FSPL curveへbest PSPL curveをprofileして差分templateを作る。
runtimeで`t0`, `teff`, `t_star`へstretch / shiftする。

#### B. Runtime coarse exact profile

各joint grid pointで、

- exact FSPL model
- PSPL nuisanceのbounded local profile
- `fs`, `fb` linear solve

を行い、full nonlinear FSPL optimizationなしで改善量とseedを得る。

どちらでも、少なくとも次を満たす。

- `u0` magnitudeも探索する、または`teff`保存変換を実装する。
- `rho / |u0|`と`t_star`の両方をカバーする。
- peak左右coverageをdiagnostic化する。
- physical signを確認する。
- `N_fft`を使用するか、不要ならAPIから削除する。

## 12. Task J: staged FSPL + parallax fallback

individual FSPL / parallax fallbackが対象regressionを通った後に実装する。

### 手順

1. PSPL baselineをfitする。
2. compact anomalyをsoft downweightし、broad supportからparallax seedを更新する。
3. parallaxを固定または強く正則化し、peak supportからFSPL seedを更新する。
4. 上位FSPL seedとparallax seedを組み合わせる。
5. FSPL + parallaxをjointに解放する。
6. contamination segmentationとjoint parameterを交互更新する。
7. original-error common objectiveで全attemptを比較する。
8. segmentation候補を変えたstability testを行う。

### 重要

- planet parameterはこの段階では明示fitしない。
- compact coherent residualが残り、single-lens parameterがsegmentation依存なら
  `mixed_unidentified` とする。
- 情報がないのに任意の`rho` / `piE`をsuccessとして返さない。
- binary-lens joint fitはこのtaskのscope外とする。

## 13. Regression acceptance criteria

数値はoptimizerの細部で多少変わり得るため、単一parameterの完全一致ではなく、
basin・改善量・安定性を評価する。

### `2_755_3280`

truth seedを使用せず、少なくとも次を満たす。

- FSPL candidateがdead-endで停止しない。
- 複数のvalid 4D FSPL seedが試される。
- 少なくとも一つのattemptが`rho > 0.10`のfinite-source basinへ入る。
- original-error chi2が`8.0e4`未満、または既知のgood basinに十分近い改善を得る。
- `rho < 0.05`かつchi2が`1.0e5`を超えるfitをsuccessにしない。
- best basinが独立seedから再現される。
- planet contamination policyを少し変えても`rho`が桁で変わらない。

参考値:

```text
truth rho              ≈ 0.2567
truth-near FSPL chi2   ≈ 42655
bad PSPL / FSPL chi2   ≈ 338000
```

### `0_599_2302`

truth parallax seedを使用せず、少なくとも次を満たす。

- compact peak blockを除いてもparallax candidateが維持される。
- GULLS geometryのtangent / exact probe conventionが一致する。
- parallax fallbackが複数のvalid 5D seedを試す。
- original-error chi2がPSPLから少なくとも`5.0e4`改善する。
- `piEN`, `piEE`が単に設定boundへ張り付いた解をsuccessにしない。
- compact-block policyを変えてもparallax方向が大きく反転しない。
- fallback後にもcompact residualが残ることを許し、planet candidateを消さない。

参考値:

```text
PSPL chi2                 ≈ 239758
linear parallax score     ≈ 91650
external parallax chi2    ≈ 148986
truth |piE|               ≈ 0.2075
bad current joint basin   piEN ≈ -0.993, piEE ≈ 1.0
```

### Mixed synthetic

少なくとも次の三種類を作る。

1. FSPL + compact off-center planet-like block
2. parallax + compact wing block
3. FSPL + parallax + compact peak block

評価するもの:

- effect recall
- parameter recovery
- contamination block recovery
- basin reproducibility
- false success
- fallback runtime

## 14. Shadow benchmark

対象二例を通した後、curated setでshadow modeを実行する。

最低限のclass:

- PSPL / null
- planet-only
- FSPL-only
- parallax-only
- FSPL + planet
- parallax + planet
- FSPL + parallax
- FSPL + parallax + planet

記録するmetric:

- FSPL recall
- parallax recall
- mixed recall
- planet-only fallback rate
- exact-probe rate
- fallback success rate
- false-success rate
- wasted-fit rate
- p50 / p95 detector runtime
- p50 / p95 fallback runtime

routing thresholdはこのbenchmark後にだけ調整する。

## 15. Test・品質要件

### 実行必須

```bash
pytest -q tests/test_effect_detection.py tests/test_contamination.py
pytest -q
ruff check \
  src/jacscanomaly/effect_detection.py \
  src/jacscanomaly/effect_routing.py \
  src/jacscanomaly/contamination.py \
  src/jacscanomaly/singlelens_fallback.py \
  tests/test_effect_detection.py \
  tests/test_contamination.py
```

既存repository全体のlint debtと、新規変更で発生したlint errorを区別する。
少なくとも上記新規ファイルではlintを通す。

### Testの注意

次のようなself-fulfilling testだけでは不十分である。

- 注入に使用したものと同一templateをbankから回収するだけ
- manually constructed candidateのroutingだけ
- parameterを更新しないFakeFitterでAPI shapeだけを見る

これらはunit testとして残してよいが、target regressionとmixed regressionを追加する。

## 16. 計画書の更新ルール

`docs/robust_fspl_parallax_plan_ja.md` のcheckboxは、次の条件を満たした項目だけ`[x]`にする。

- 実装が存在する。
- unit testが存在する。
- relevant regressionが通る。
- production経路またはshadow benchmarkから到達可能。

今回の修正中は冒頭の実装状況を次のように区別する。

```text
prototype
unit-tested
target-regression-tested
shadow-benchmarked
production-ready
```

存在するだけのAPIを「Phase完了」と表現しない。

## 17. Scope外

今回まだ実装しないもの:

- microeden / GRU router
- planet / single-lens完全分類
- Phase 3のbefore / after planet candidate matching
- HMC / NUTS posterior
- SMC / nested sampling
- binary-lens + FSPL + parallax joint fit
- default pipelineの全面置換

ただし、後から追加できるようartifact schemaとreason codeは維持する。

## 18. 完了時の報告内容

Lunaは完了時に次を報告すること。

1. 変更した設計と主要file
2. 対象二例のbefore / after表
3. detector score、routing、seed数、best / second-best basin
4. original chi2、robust objective、physical parameters
5. contamination blockとprotected information
6.全test / lint結果
7. 未解決のfailure mode
8. 次にPhase 3へ進めるかどうかの明示的判断

「testが通った」だけで完了報告にしない。

## 19. Definition of Done

次をすべて満たして初めて、この修正taskを完了とする。

- `2_755_3280`がtruth seedなしで良いFSPL basinへ入る。
- `0_599_2302`がplanet contamination下でも良いparallax basinへ入る。
- 両eventがroutingのdead-endで止まらない。
- effect-specific fitterが自動選択される。
- model dimensionに合う複数seedが生成される。
- protected support内の局所planetをdownweightできる。
- broad physical residual全体はcontaminationへ逃げない。
- common objectiveでattempt / modelを比較できる。
- 未収束・単一seed・bound張り付き・effect未改善をsuccessにしない。
- mixed synthetic regressionが通る。
- full test suiteが通る。
- 新規変更fileのlintが通る。
- 計画書のcheckboxが実態と一致する。

この条件を満たすまではPhase 3へ進まない。
