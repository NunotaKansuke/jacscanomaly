# FSPL / parallax 残差検知と惑星汚染に強い fallback fit の実装計画

## 実装状況

2026-07-29 時点で、Phase 1 の物理 residual detector と routing、Phase 2 の
contamination-aware fallback の実装単位と unit test は完了している。対象二例の
実データでも alternating convergence、独立 basin 再現、effect score 改善を同時に満たした。
`[x]` は「実装 + unit test + 実行可能な回帰入口」を意味する。

- [x] `effect_detection.py`: nuisance projection、parallax score、FSPL joint template
- [x] `effect_routing.py`: `skip` / `exact_probe` / `fallback` と calibration helper
- [x] `contamination.py`: 2-state DP、protected support、交互 downweight
- [x] `singlelens_fallback.py`: detector seed の structured multistart と robust fit
- [x] `Finder.detect_effects` / `Finder.robust_fallback`: 既存経路からの optional 入口
- [x] `exact_probe.py`: FSPL / parallax の exact forward probe と fail-open promotion
- [x] `tools/robust_fspl_parallax_benchmark.py`: 対象二例の JSON regression harness
- [x] GULLS bounded PSPL space-parallax fitter と effect-specific factory
- [x] 元の `ferr` による `chi2`、robust objective、contamination penalty、reason code の分離
- [x] dimensionless parameter distance と optimizer / alternating / basin 判定
- [x] protected penalty を含む共通 objective と、FSPL / parallax の独立 protected support
- [x] raw parameter contract に基づく `delta_t0/tE`、`delta_log_tE`、`delta_logrho`、`delta_piE` の距離
- [x] detector seed 起点の FSPL degeneracy family、mixed 6-D seed bridge、bound / canonical reason code
- [x] benchmark schema、metadata、全attempt diagnostics の JSON-safe artifact
- [x] FSPL template ごとの profiled PSPL nuisance と `N_fft` forwarding
- [x] 最終 fit の residual による再 segmentation と、全 seed 共通の canonical contamination objective
- [x] FSPL stage の `logrho` と parallax stage の `piE` を同時に保持する 6-D joint seed composition
- [x] fallback 後の physical residual detector score による acceptance 判定
- [x] active state が継続したまま season gap をまたぐ場合の block 分割 regression
- [x] GULLS observer を Earth heliocentric orbit + RTModel geocentric satellite offset として構成
- [x] 非parallax FSPL 専用 native C++ VBM LM と、exact-probe basin 周辺 seed の優先配置
- [x] annual/space PSPL/FSPL parallax を native C++ trajectory/VBM + SciPy TRF へ統一
- [x] 惑星候補を forced contamination block として保護する before/fallback/after pipeline
- [x] JAXopt を依存関係と共通 single-lens optimizer から除去
- [x] native parallax の固定 parameter 評価、plot、BIC model selection を同じ座標契約へ統一

対象二例の optional benchmark は、truth parameter を seed に渡さず、
`max_seeds=8, max_iter=8` で次の acceptance に到達している。

| event | effect | baseline original `chi2` | selected original `chi2` | effect score before → after | success |
|---|---|---:|---:|---|---|
| `2_755_3280` | FSPL | 338004.8 | 42916.6 | 90571.7 → 475.0 | true |
| `0_599_2302` | GULLS space parallax | 239758.3 | 219187.9 | 105035.2 → 12686.4 | true |

いずれも真値を seed に渡していない。GULLS は exact probe が inconclusive のときも、広い
physical score を dead-end にしない fail-open fallback へ進む。`success` は未収束時に true へ
緩めていない。なお native VBM は非parallax FSPL magnification にのみ使用し、space-parallax
座標は Earth 軌道と Roman offset を合成した GULLS N/E trajectory で評価する。

parallax trajectory の全面 native C++ 化、厳密な VBM/GULLS 座標契約、SciPy bounded
fit、Phase 3 の planet pipeline 再接続については
[`native_cpp_parallax_planet_pipeline_design_ja.md`](native_cpp_parallax_planet_pipeline_design_ja.md)
を実装仕様とする。現行の GULLS self-reference 式に対する legacy 互換は残さず、
GULLS 本体の reference-observer semantics へ置き換える。

## 1. 目的と優先順位

最終目標は、PSPL fit 後に見える信号を次のいずれかへ分類することである。

- 惑星 signal
- FSPL
- parallax
- FSPL + parallax
- 惑星と単一レンズ効果の混合
- その他の model misfit / artifact

ただし、現段階で完全分類器を先に作るのは得策ではない。最優先課題は次の二つである。

1. FSPL / parallax 固有の残差を、重い非線形 fit を行う前に高 recall で検知する。
2. routed event では、惑星 signal が共存しても FSPL / parallax の推定を壊さない。

したがって設計の中心は「惑星か否かを先に完全分類すること」ではなく、
「局所的・連続的な異常に汚染されても、単一レンズ効果の検知量と推定量が安定すること」
に置く。

重い fallback を全件へ実行せず、物理的な残差 detector で対象を絞る。
機械学習は detector では判定しにくい境界例を補う選択肢とし、初期実装の必須要件にはしない。

## 2. 現状の問題

### 2.1 Local optimizer の問題

現在の FSPL / parallax fitter は、良い basin に入れば十分な解へ到達できる一方、
PSPL 解の近傍や少数の seed から開始すると異なる basin を発見できない。

- `rho`、`u0`、`tE` は強く結合する。
- parallax は `t0`、`tE`、`u0`、blending と縮退する。
- 惑星 signal が目的関数を歪め、単一レンズ parameter を引き寄せる。
- `chi2/dof > 10` のような絶対閾値だけでは、失敗した fit を見逃す。
- HMC / NUTS は一つの basin 内の posterior 探索には有効だが、単独では basin discovery
  の解決策にならない。

したがって「optimizer を LM から HMC に交換する」だけでは不十分である。

### 2.2 Planet masking の問題

悪い PSPL residual を先に惑星区間として mask すると、broad な parallax residual や
peak 周辺の FSPL residual 自体を丸ごと隠す危険がある。一方、mask しなければ局所的な
惑星 signal が単一レンズ fit を歪める。

必要なのは one-shot mask ではなく、単一レンズ model と連続 anomaly 区間を
交互に推定する仕組みである。

### 2.3 BIC 比較の問題

各 model が異なる basin、異なる mask、異なる clipping を使用した状態で BIC を比較しても、
model evidence の代わりにはならない。まず探索失敗と contamination の扱いを揃える必要がある。

## 3. 全体アーキテクチャ

```text
light curve
    |
    v
fast PSPL fit
    |
    +--> raw planet candidates を保存
    |
    v
物理 residual detector
    |  - parallax score test
    |  - FSPL projected template bank
    |  - contamination / coverage / conditioning diagnostics
    |
    +-- low score ----------> 通常の planet pipeline
    |
    +-- clear candidate ----> contamination-aware robust fallback
    |
    +-- uncertain ----------> cheap exact-template probe
                               または optional ML router
                                      |
                                      v
                         robust FSPL / parallax / joint fit
                                      |
                                      v
                         residual detector を再実行
                                      |
                                      v
                    planet / single-lens / mixed / unresolved
```

候補 detector は科学的な確定判定ではない。高 recall を保ちながら expensive fallback の
実行率を制御する routing mechanism である。

## 4. 共通の residual 表現

PSPL fit の standardized residual を

```math
z_i = \frac{f_i - m_0(t_i;\hat{\theta}_0)}{\sigma_i}
```

とする。ここで `theta0` は少なくとも `t0, log(tE), u0, fs, fb` を含む。

PSPL nuisance Jacobian を `J0` とし、weighted QR または SVD で nuisance subspace を除く。

```math
P_\perp = I - Q_0 Q_0^\mathsf{T}
```

全 detector template に `P_perp` を作用させる。これにより、単に PSPL parameter を
少し動かすだけで消える residual を FSPL / parallax 候補として数えない。

実装上は巨大な `P_perp` 行列を生成せず、QR/SVD factor を用いて projection する。
rank deficiency、condition number、template leverage も diagnostic として保存する。

## 5. Parallax candidate detector

### 5.1 一次 score test

現在使用している observer geometry と同じ convention で、`piEN = piEE = 0` における
parallax Jacobian を計算する。

```math
H_\pi =
\left[
\frac{\partial m}{\partial \pi_{E,N}},
\frac{\partial m}{\partial \pi_{E,E}}
\right]_{\pi_E=0}
```

`Hpi_perp = P_perp Hpi` として、

```math
b = H_{\pi,\perp}^{\mathsf{T}} z,\qquad
G = H_{\pi,\perp}^{\mathsf{T}} H_{\pi,\perp}
```

```math
S_\pi = b^\mathsf{T}G^+b
```

を計算する。これは線形化した parallax model が達成できる best `Delta chi2` であり、
必要なのは Jacobian 評価と 2 x 2 solve だけである。非線形 parallax fit は不要である。

### 5.2 惑星汚染への耐性

一つの score だけでなく、以下を同時に計算する。

- 全点を用いた robust score
- peak 周辺の compact block を除いた score
- pre-peak / post-peak / wing / season ごとの score
- leave-one-block-out score
- inferred parallax direction の subset 間安定性
- score のうち最大一 anomaly block が占める比率

compact block の定義には、既存の `PlanetSignalExtractor` の interval / beam の考え方を
再利用できる。ただし、PSPL residual から得た mask を固定して detector に渡してはならない。
broad residual を anomaly として除外しないよう、除外可能な総 span と block 数を制限する。

候補条件は単なる `S_pi > threshold` ではなく、次の組合せとする。

- `S_pi` が校正閾値を超える。
- projected template が有効 rank を持つ。
- wing / season の coverage が十分である。
- score が少数点や一つの compact block のみに支配されない。
- subset 間で方向または改善量が致命的に不整合でない。

### 5.3 一次近似が弱い場合

大きな parallax や縮退の強い event では、`piE = 0` の tangent が十分でない可能性がある。
その場合だけ、少数の `piE` 半径・方向からなる exact template atlas を評価する。

これは optimizer ではなく forward model の batch evaluation と、小さな linear flux solve
に限定する。これにより uncertain case のみ少し高価に調べられる。

### 5.4 検証済みの根拠

`../roman_simu` の `0_599_2302` に対する予備検証では、GULLS observer geometry を用いた
projected tangent score が約 `9.2e4`、peak 近傍の約 2 日を除いても約 `1.0e5` となった。
これは実際の parallax model の改善量と同じ order である。

したがって、この例は非線形 fit を完了させなくても強い parallax 候補として routing できる。
同時に condition number が大きかったため、rank / conditioning の記録は必須である。

## 6. FSPL candidate detector

### 6.1 `rho` derivative や固定軌道 scan だけでは不十分

FSPL では `rho -> 0` 近傍の微分が弱く、PSPL trajectory を固定した `rho` scan は
良い basin を示さない場合がある。重要なのは次の結合である。

- `rho / |u0|`: source crossing / near-crossing の幾何
- `t_star = rho * tE`: finite-source feature の時間幅
- `tE`, `u0`, `rho` の同時変化
- limb darkening

したがって FSPL detector は `rho` 一次微分ではなく、profiled physical template bank
または低次元の joint grid を使用する。

### 6.2 Projected FSPL template bank

dimensionless な exact FSPL light curve から、

```math
h_k(t) = m_{\mathrm{FSPL}}(t;\eta_k)
       - m_{\mathrm{PSPL,profiled}}(t;\eta_k)
```

を作る。`eta_k` は最低限、以下を覆う。

- `rho / |u0|`
- `t_star` または `tE` stretch
- `u0` sign / relevant degeneracy
- limb-darkening coefficient の代表値

各 FSPL curve に対して PSPL nuisance を profile してから差を取ることが重要である。
runtime では event の `t0` と `teff = |u0|tE` を基準に template を stretch / shift し、
さらに `P_perp` で nuisance 成分を除く。

各 template の matched-filter score は、

```math
S_{\mathrm{FSPL},k}
= \frac{(h_{k,\perp}^{\mathsf{T}}z)^2}
       {h_{k,\perp}^{\mathsf{T}}h_{k,\perp}}
```

とする。物理的に許される符号、peak 両側の coverage、template support 内の点数を要求する。

### 6.3 惑星 signal との区別

FSPL residual は peak 近傍に現れるため、parallax のように peak を丸ごと除外できない。
以下の diagnostic を併用する。

- peak 両側での template consistency
- even / odd residual energy
- best template に一致した energy と unmatched energy
- single-point / single-block influence
- cadence gap を挟いだ見かけ上の symmetry
- compact anomaly block を一つ許した場合の score 安定性

惑星 signal が FSPL に似ることはあり得るため、この段階で完全分類はしない。
「FSPL fit を試す価値があるか」を判定し、最終的には contamination-aware fallback 後の
反実仮想 residual で分類する。

### 6.4 検証済みの根拠

`2_755_3280` では、現行の PSPL seed からの FSPL fit は `chi2 ~ 3.38e5` に留まったが、
truth 近傍からは `chi2 ~ 4.27e4` に到達した。固定 PSPL trajectory 上の `rho` scan は
良い basin を見つけられなかった。

一方、`tE` factor と `rho / |u0|` を同時に動かす粗い forward-model grid は、最適化なしでも
`chi2 ~ 1.85e5` の方向を発見した。最終解ではないが、routing と multistart seed の生成には
十分な情報である。

したがって FSPL detector は joint template を採用し、`rho` 単独 probe は採用しない。

## 7. Routing policy

detector の出力を二値 label ではなく、次の構造体として保存する。

```text
EffectCandidate
  effect: fspl | annual_parallax | space_parallax
  score
  score_without_compact_blocks
  effective_rank
  condition_number
  coverage
  max_point_influence
  max_block_influence
  subset_stability
  best_template_or_direction
  seed_parameters
  decision: skip | exact_probe | fallback
  reason_codes[]
```

policy は三段階とする。

- `skip`: score が低く、coverage / conditioning も評価可能。
- `exact_probe`: score は境界、または一次近似・template coverage に懸念がある。
- `fallback`: physical score が高く、影響が少数点だけに集中していない。

閾値は目視で決めず、simulation truth に対する counterfactual `Delta chi2` と fallback rate の
Pareto curve から決める。false positive 数だけでなく、

- detectable FSPL / parallax の recall
- planet-only event の fallback rate
- mixed event の recall
- fallback 後に改善しなかった wasted-fit rate

を評価する。

## 8. 惑星汚染に強い robust fallback

### 8.1 基本モデル

観測を次のように扱う。

```text
data = single-lens physical model + contiguous anomaly contamination + noise
```

anomaly contamination を固定 mask ではなく latent state とする。各時刻に

- `B`: baseline / single-lens model に従う状態
- `A`: anomaly / heavy-tailed residual を許す状態

を持つ二状態 HMM、または等価な dynamic programming による MAP segmentation を使用する。

- `B` state: Gaussian または軽い Student-t
- `A` state: inflated-scale Student-t
- transition penalty: anomaly が連続区間になりやすいようにする
- season / large cadence gap: state transition cost を reset
- occupancy / total span penalty: broad な物理 residual 全体を anomaly に逃がさない

### 8.2 交互推定

以下を収束まで交互に行う。

1. 現在の FSPL / parallax model residual から、DP/HMM で anomaly posterior または MAP block
   を推定する。
2. anomaly probability を weight として、単一レンズ parameter を更新する。
3. anomaly block と parameter の変化、weighted objective を記録する。
4. 複数初期値で同じ解・同等の解に戻るか確認する。

これは、惑星 parameter を明示的に fit せずに、惑星 signal の影響を robust に切り離す方法である。
惑星 model を同時 fit するより安価で、planet morphology の事前仮定も少ない。

既存 `PlanetSignalExtractor` の robust weight、interval penalty、beam search の実装要素は
再利用候補である。ただし既存 PSPL mask をそのまま固定するのではなく、physical model 更新と
segmentation を往復させる新しい orchestration が必要である。

最適化する量は、概念的には次の共通 penalized objective とする。

```math
\mathcal{L}(\theta, s)
= \sum_i \ell_{s_i}\!\left(r_i(\theta)\right)
+ \lambda_{\mathrm{tr}}\sum_i [s_i \ne s_{i-1}]
+ \lambda_{\mathrm{occ}}\sum_i [s_i=A]
+ \lambda_{\mathrm{span}}\,\mathrm{span}(s=A)
```

`theta` が単一レンズ parameter、`s` が latent anomaly state である。
model 間で同じ contamination likelihood と penalty を使い、anomaly state を増やすだけで
見かけの fit 改善を得られないようにする。

### 8.3 効果別の protected support

同じ anomaly policy を全 model に機械的に適用しない。

#### Parallax

- broad wing / season-scale coherence を fit の主情報として保護する。
- peak 近傍の compact planet block は downweight / anomaly state にできる。
- leave-one-block-out でも `piE` 方向と改善量が安定することを確認する。

#### FSPL

- central peak 全体を anomaly にしてはならない。
- finite-source template が予測する peak support に最低 inlier coverage を課す。
- peak 両側を同時に説明できるかを重視する。
- 一部の sharp planet block は許すが、FSPL support の占有率に上限を置く。

#### FSPL + parallax + planet

最難関の混合 case は、最初から全 parameter を自由にせず段階的に fit する。

1. compact peak anomaly を暫定的に除き、wings から parallax seed を更新する。
2. parallax seed を固定または強く正則化し、peak 近傍から FSPL seed を更新する。
3. FSPL + parallax を joint に解放し、latent anomaly state と交互推定する。
4. anomaly segmentation の複数候補と multistart seed を用いて再実行する。
5. full-data robust likelihood で最終 refine する。

この staging により、惑星 signal が `piE` や `rho` を引っ張ることと、逆に broad parallax /
FSPL residual を anomaly mask が吸収することの両方を抑える。

### 8.4 Basin discovery

fallback の seed は PSPL 解だけに依存させない。

- detector が返した parallax direction / radius
- FSPL joint template の `rho/|u0|`, `t_star`
- `u0` sign と known degeneracy
- coarse Latin hypercube / structured grid
- detector score 上位の複数 seed

を使用する。

推奨探索順は次の通り。

1. vectorized coarse physical template evaluation
2. 上位 seed の bounded local optimization
3. 必要時のみ differential evolution などの global optimizer
4. full-data contamination-aware local refinement

HMC / NUTS は basin discovery 後の uncertainty estimation に限定する。
SMC は、上記でも失敗する benchmark case が相当数残り、計算費用に見合う場合のみ Phase 5
で比較する。最初から依存しない。

### 8.5 識別限界と最終 escalation

robust contamination model で惑星の影響を抑えられるのは、FSPL / parallax parameter を
拘束する情報が anomaly 区間の外または両側にも残る場合である。

特に planet anomaly と FSPL feature が同じ peak support に完全に重なり、片側しか観測されず、
複数の `rho` が同程度に説明できる場合、single-lens + generic contamination だけから
`rho` を一意に回収することは原理的にできない。

そのため次を identifiability diagnostic として保存する。

- protected support 内の effective inlier coverage
- peak 両側それぞれから得た `rho` profile の整合性
- anomaly segmentation 候補間の parameter spread
- profile likelihood の幅と multimodality
- leave-block-out 後に残る Fisher information / effective rank

情報が足りない場合に、任意の FSPL / parallax 値を「成功」として返してはならない。
`mixed_unidentified` として保持し、最終 escalation としてのみ
binary-lens + FSPL + parallax の joint physical fit を実行する。

この joint fit は全 routed event には使わない。次の条件を満たす case に限定する。

- single-lens effect detector は強い。
- robust fallback 後にも compact coherent residual が残る。
- effect parameter が anomaly segmentation に依存して不安定である。
- scientific priority または設定された compute budget が許す。

joint fit の planet seed は fallback 後の局所 residual から作る。これにより、悪い PSPL residual
全体を planet model に説明させる探索を避ける。

### 8.6 Fit success の判定

単に optimizer が収束したかではなく、以下を要求する。

- 異なる seed から同じ basin または同等 objective に到達する。
- physical parameter が bounds に不自然に張り付かない。
- anomaly block の小変更で `rho` / `piE` が激変しない。
- leave-one-block-out で主要効果が維持される。
- residual の effect-specific detector score が十分低下する。
- model complexity と contamination penalty を含む共通 objective で改善する。

各 model で恣意的に異なる mask を使った BIC 比較は避ける。比較には、

- 同じ latent-contamination likelihood
- または model 間で合意した consensus mask

を使用する。

## 9. Fallback 後の planet 判定

最初の PSPL residual から得た候補を `raw_planet_candidates` として必ず保存する。
robust fallback 後に同じ residual scan を再実行し、対応関係を記録する。

- `absorbed_by_fspl`
- `absorbed_by_parallax`
- `planet_residual_remains`
- `mixed_effects`
- `unresolved`

単一レンズ model を改善した結果、broad / symmetric residual が消えて局所 signal が残るなら、
その局所 signal はむしろ planet candidate として見つけやすくなる。

最終的な planet / single-lens 完全分類はこの比較を教師・feature として後続 Phase で行う。

## 10. 機械学習の位置づけ

初期版の primary detector は物理 score とする。ML は次の条件を満たした場合のみ追加する。

- physical detector の uncertain bucket が fallback rate を支配している。
- uncertain event に、score では表現しにくい再現可能な時系列 pattern がある。
- group-safe split で simulation family / field / seed leakage を防げる。
- ML hard-veto による FSPL / parallax false negative を測定できる。

候補 feature は raw flux のみでなく、

- standardized PSPL residual
- projected parallax / FSPL scores
- detector template を引いた後の unmatched residual
- time gap / cadence / passband / observatory
- coverage / conditioning / block influence

を含む。

[`microeden`](https://github.com/NunotaKansuke/microeden) は irregular time-series encoder、
group-safe split、artifact 管理の参考実装として比較対象にできる。ただし最初から必須 dependency
にせず、まず detector 出力に対する小さな meta-classifier として benchmark する。
ML は初期段階で `skip` の hard veto を行わない。

## 11. 実装フェーズ

### Phase 0: Detector prototype と benchmark 固定

- `0_599_2302` の projected parallax score test を再現可能な test / notebook 相当にする。
- `2_755_3280` の FSPL joint template/grid score を再現可能にする。
- planet-only、FSPL-only、parallax-only、mixed、null の小規模 curated set を作る。
- truth label と「十分探索した場合の counterfactual Delta chi2」を分けて保存する。
- runtime、fallback rate、recall の基準値を記録する。

この phase では production pipeline を変更しない。

### Phase 1: Physical candidate detector

- [x] nuisance Jacobian projection 共通部を実装する。
- [x] annual / space parallax score test と diagnostics を実装する。
- [x] FSPL projected template bank と joint seed generation を実装する。
- [x] `EffectCandidate` と routing reason code を追加する。
- [x] `Finder.detect_effects` から detector-only shadow mode を呼び出せるようにする。
- [ ] detector-only shadow mode で全 simulation を走らせ、runtime / recall 基準値を固定する。

### Phase 2: Contamination-aware fallback

- [x] two-state HMM / DP segmentation を独立 component として実装する。
- [x] effect-specific protected support と occupancy penalty を実装する。
- [x] detector seed、structured multistart、既存 fitter を使った交互 robust fit を統合する。
- [x] FSPL + parallax の staged joint orchestration API と stage-seed の6次元 bridgeを実装する。
- [x] staged joint orchestration を stage-seed bridge の mixed synthetic regression まで通す。
- [x] basin stability / protected-support / compact-block diagnostics を保存する。

### Phase 3: Planet pipeline への再接続

- raw candidate を保持する。
- fallback 後に residual extraction を再実行する。
- before / after candidate matching と分類 reason code を追加する。
- mixed / unresolved を無理に二値化しない。

### Phase 4: Optional ML router

- physical detector の uncertain cases だけを対象に baseline model を比較する。
- simple calibrated tree / logistic model を最初の baseline とする。
- 必要なら `microeden` 系の irregular sequence encoder を比較する。
- recall と fallback 削減が detector-only policy を明確に上回る場合だけ採用する。

### Phase 5: Optional posterior / global inference

- unresolved benchmark に限って SMC、tempered SMC、nested sampling 等を比較する。
- HMC / NUTS は発見済み basin の uncertainty 推定に使う。
- identifiability 不足の high-priority mixed case だけ、binary-lens + FSPL + parallax joint fit
  を比較する。
- production 採用は成功率向上と wall time の Pareto 比較で決める。

## 12. Test 方針

### Unit tests

- [x] nuisance projection 後に PSPL tangent 成分が消える。
- [x] synthetic linear parallax injection で `S_pi` が期待 `Delta chi2` と一致する。
- [x] rank-deficient geometry で pseudoinverse と reason code が安定する。
- [x] FSPL template の stretch / shift / flux profiling と exact probe が再現可能。
- [x] HMM/DP が season gap をまたいで不自然な anomaly block を作らない。
- [x] occupancy penalty により broad parallax residual 全体を anomaly state に逃がさない。
- [x] protected FSPL peak / parallax wing のソフト制約、span penalty の選択変化、mixed support 分離を検証する。
- [x] 4-D FSPL、5-D parallax、6-D mixed seed、rho/t★ family、誤次元 seed rejection を検証する。
- [x] bound 張り付き、未収束、single-seed、objective 非改善を canonical reason code として検証する。

### Regression tests

- [x] `2_755_3280`: truth-near seed を直接与えず、native C++ FSPL で `chi2` を約 `338004.8`
  から `42916.6` へ改善し、alternating / independent-basin acceptance を満たす。
- [x] `0_599_2302`: compact contamination 下で parallax candidate になり、GULLS bounded fitter で
  `chi2` を約 `239758.3` から `157752.1` へ改善し、bound 外で acceptance を満たす。
- [x] 上記二例で alternating convergence、独立 basin 再現、effect score 改善を同時に満たす。
- [ ] planet-only case: local anomaly だけで parallax fallback が大量発火しない。
- [ ] FSPL + planet: peak 全体を mask せず、FSPL parameter が planet block の有無で安定する。
- [ ] FSPL + parallax + planet: staged joint fit が各単独 fit より residual score を下げる。

### Acceptance metrics

- detectable FSPL recall
- detectable parallax recall
- mixed-effect recall
- planet-only fallback rate
- fallback success rate
- wasted-fit rate
- p50 / p95 detector runtime
- p50 / p95 fallback runtime
- parameter / basin stability
- planet false-negative 改善量

閾値は単一 metric の最大化ではなく、許容 fallback budget の下で recall を最大化するよう選ぶ。

## 13. 想定するコード変更箇所

具体名は実装開始時に repository 構造へ合わせて確定するが、責務は分離する。

```text
src/jacscanomaly/
  effect_detection.py       # projection、score test、FSPL template
  effect_routing.py         # policy、threshold、reason codes
  contamination.py          # HMM/DP、robust weights、protected support
  singlelens_fallback.py    # multistart、staging、joint fit orchestration
  singlelens_fit.py         # 既存 fitter の低レベル primitive を再利用
  planet_signal.py          # before/after candidate matching
```

既存 API を一度に置換せず、最初は shadow detector と optional fallback として追加する。
feature flag で従来経路と比較できるようにする。

## 14. 実装者への判断ルール

1. candidate detector 内で full nonlinear FSPL / parallax optimizer を呼ばない。
2. generic residual morphology より、nuisance-projected physical template score を優先する。
3. PSPL 由来の planet mask を固定して単一レンズ fit しない。
4. broad residual 全体を contamination として捨てられない制約を入れる。
5. FSPL では central peak support を保護し、parallax では wings の coherence を保護する。
6. PSPL seed 一個からの local convergence を fit success とみなさない。
7. HMC を basin discovery の代替として導入しない。
8. ML は detector-only baseline を上回ることを確認してから採用する。
9. mixed / unresolved を許し、早い段階で惑星か単一レンズかに強制二値化しない。
10. 各判断について score、mask、seed、失敗理由を artifact として残す。

## 15. 完了条件

初期目標の完了条件は、完全分類精度ではなく以下である。

- FSPL / parallax 候補を full fit なしで再現可能に検知できる。
- expensive fallback の対象数を設定した budget 内に制御できる。
- fallback が惑星 signal と共存しても `rho` / `piE` を安定して推定できる。
- `2_755_3280` と `0_599_2302` を regression case として改善できる。
- fallback 前後の residual から、single-lens に吸収された信号と残存 planet candidate を区別できる。
- detector、fit、classification の各失敗理由を追跡できる。

この基盤ができた後に、planet / FSPL / parallax / mixed の完全分類を段階的に強化する。
