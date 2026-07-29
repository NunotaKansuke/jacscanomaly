# Native C++ parallax backend と planet pipeline 再接続の実装設計

## 0. この文書の位置づけ

この文書は、`docs/robust_fspl_parallax_plan_ja.md` のうち、次の二段階を
gpt-5.6-luna がそのまま実装できる粒度に固定する設計書である。

1. annual / space parallax の軌道計算と model 評価を native C++ 化し、
   parallax fallback から JAX / JAXopt を完全に外す。
2. robust FSPL / parallax fallback の前後を既存の planet pipeline に接続し、
   fallback 前後の anomaly を追跡できるようにする。

この文書を書いた時点では設計のみを対象とし、ここに記した native backend や
pipeline 接続はまだ実装しない。

優先順位は次の通りである。

1. 座標系と ephemeris の origin を曖昧にしない。
2. parallax trajectory を C++ だけで正しく高速に計算する。
3. VBMicrolensing（以下 VBM）は増光率計算だけに使う。
4. bounded multistart fit は SciPy `least_squares` を第一選択とする。
5. fallback が planet signal を吸収しない既存の contamination-aware fitting を維持する。
6. fallback 後に同じ planet pipeline を再実行し、before / after を比較する。

HMC、SMC、SVI はこの段階では導入しない。局所解問題への主な対策は、物理 detector
から作る seed、構造化 multistart、bounded trust-region、contamination-aware
segmentation である。これらでも benchmark failure が残った場合だけ、後続段階で
global inference を検討する。

---

## 1. 現状と今回置き換える範囲

2026-07-29 時点で、次はすでに存在する。

- FSPL / parallax physical residual detector
- effect routing と exact probe
- contamination-aware fallback
- FSPL と parallax の stage seed を合成する joint fallback
- native VBM FSPL fitter
- VBM 内部の annual parallax API を使う native FSPL+annual-parallax fitter
- GULLS-like space parallax trajectory と SciPy finite-difference fit

しかし parallax path は統一されていない。

| effect | 現行の主な path | 問題 |
|---|---|---|
| annual parallax PSPL | JAX trajectory + JAXopt LM | JAX/JAXopt が残る |
| annual parallax FSPL | VBM 内部 parallax + native LM または JAX | trajectory の責務が VBM 内に残る |
| VBM convention space parallax | JAX trajectory + SciPy | model 評価が JAX |
| `convention="gulls"` PSPL | NumPy trajectory + SciPy | C++ でない。現行 reference frame は厳密な GULLS と異なる |
| `convention="gulls"` FSPL | NumPy/JAX/VBM 混在 + SciPy | 責務と座標契約が不明瞭 |

今回の完成後は、上記を一つの native evaluator family に統一する。

```text
Python: seed generation / robust loop / scipy.optimize.least_squares
                            |
                            v
C++: ephemeris interpolation -> sky projection -> tau,beta,u
                            |
                            v
VBM: PSPLMag(u) または ESPLMag2(u,rho) だけ
                            |
                            v
C++: dataset ごとの fs,fb profile -> weighted residual/Jacobian
```

禁止事項:

- 新 backend から `VBMicrolensing::ComputeParallax`,
  `*LightCurveParallax`, satellite-table loader を呼ばない。
- parallax fit の residual callback から JAX array、`jax.jit`、JAXopt を呼ばない。
- ephemeris の距離だけを見て geocentric / heliocentric を推測しない。
- `gulls` は GULLS 本体と一致する strict convention だけを意味する。
- 現行の self-reference convention に対する互換 path は残さない。

---

## 2. upstream source を確認して確定した事実

### 2.1 VBM satellite table

ローカルの VBM source
`VBMicrolensingLibrary.cpp` の `SetObjectCoordinates(..., sateltabledir)` と
`ComputeParallax` を確認した。

VBM の satellite table は次の形式である。

```text
JD  RA_deg  Dec_deg  distance_AU  [optional column]
```

VBM はこの RA / Dec / distance を Cartesian vector に変換し、`possat` に保存する。
annual Earth positionは別に `Ear` から計算し、reference epoch の位置と速度を引いた後、
satellite vector をそのまま加える。

```cpp
Et = Earth(t) - Earth(tref) - vEarth(tref) * (t - tref);
Et += possat(t);
```

したがって VBM satellite table の通常の意味は、**地球から見た衛星の位置 vector**
（geocentric satellite offset）である。完全な heliocentric observer position ではない。
これは利用者向け API の既定値にする。

VBM の sky-plane basis は `South` と `West` である。内部変数 `rad` が South、
`tang` が West である。新 backend はこの表現を外部 API に漏らさず、後述する
North / East に統一する。

### 2.2 GULLS

ローカルの `../gulls/src/classes/parallax.cpp` と event setup を確認した。

GULLS の `World[obsidx].orbit` は、観測者の**完全な heliocentric orbit**である。
Earth、geosynchronous、L2、JWST などは、基準 orbit と perturbation component を
加算して構築される。

GULLS は観測者 orbit と reference observer orbit を別々に持つ。
通常、reference は `World[0].orbit` である。GULLS の displacement は次式である。

```text
Δr(t) = r_observer(t)
      - r_reference(tref)
      - v_reference(tref) * (t - tref)
```

この vector を North / East に射影して trajectory shift にする。

重要: 現行の `GullsSpaceParallaxProjector` は complete observer 自身の
`r_observer(tref), v_observer(tref)` を引いている。これは GULLS 本体の
reference-observer semantics と同じではない。新実装ではこの誤った互換挙動を削除し、
strict GULLS semantics へ置き換える。

### 2.3 VBM と GULLS の符号は同じ式に統一できる

新 backend の canonical sky basis を `(North, East)` とし、
observer displacement を `(ΔN, ΔE)` とする。trajectory shift は次に固定する。

```text
Δτ    = -(piEN * ΔN + piEE * ΔE)
Δbeta =   piEE * ΔN - piEN * ΔE
```

これは GULLS の `compute_tushifts()` と一致する。

VBM は `(South, West) = (-North, -East)` を使って `tau` に正符号で加えているため、
VBM の式を North / East に変換しても同じ式になる。したがって、VBM互換と
GULLS互換のために別の piE sign convention を持つ必要はない。

---

## 3. canonical coordinate / time contract

### 3.1 Cartesian frame

C++ 内部では ephemeris を次に正規化する。

- frame: ICRF/J2000 equatorial Cartesian
- unit: AU
- time unit: day
- position order: `(x, y, z)`
- sky order: `(North, East)`
- floating point: `double`

target directionを RA=`alpha`, Dec=`delta` とすると、basis は次に固定する。

```text
n_hat     = ( cos(delta) cos(alpha),  cos(delta) sin(alpha), sin(delta))
north_hat = (-sin(delta) cos(alpha), -sin(delta) sin(alpha), cos(delta))
east_hat  = (-sin(alpha),             cos(alpha),            0)
```

`north_hat`, `east_hat`, `n_hat` の unit norm と直交性を constructor で検査する。

### 3.2 time

新 API は time の意味を暗黙推測しない。

```python
TimeSpec(
    scale="jd" | "hjd",
    offset=0.0 | 2450000.0 | explicit_float,
)
```

- 公開 fitter に渡す time と ephemeris time は同じ `TimeSpec` に正規化してから
  C++ へ渡す。
- 内部 C++ time は absolute JD day とする。
- 既存の `time < 2450000` なら自動加算する処理は削除する。
- 新 API で input が ambiguous なら `ValueError` にする。
- `hjd` の場合は、現行 VBM と parity の取れた light-travel iteration を C++ に持つ。
- `jd` の場合は `tau0=(t+lighttravel(t)-t0-lighttravel(t0))/tE` を使う。
- strict GULLS parity test では GULLS simulation time をそのまま使い、
  VBM 独自の light-travel correction を追加しない。

`tref` は parallax reference epoch、`t0` は event peak parameter であり、同一とは限らない。
両者を混同しない。

### 3.3 interpolation

初期実装は piecewise-linear interpolation とする。Python の `np.interp` と同じ
endpoint orderingを使う。

- time grid は strictly increasing
- duplicate time は reject
- NaN / inf は reject
- vector unit は AU と明記
- default は ephemeris range 外を reject
- extrapolation は明示的な `extrapolation="linear"` のときだけ許す
- reference velocity は、table に velocity があればそれを補間する
- position しかなければ、reference pointを挟む local 3-point derivativeを使う
- `np.gradient` 相当で全 table の velocityを一度作るだけの実装は避ける

### 3.4 origin を明示する ephemeris type

Python 側に frozen dataclass、C++ 側に enum を作る。

```python
Ephemeris(
    time: np.ndarray,
    position_au: np.ndarray,       # shape (n, 3)
    velocity_au_per_day: np.ndarray | None,
    frame: Literal["icrf_j2000"],
    origin: Literal[
        "solar_system_barycenter",
        "sun",
        "earth",
        "explicit_reference",
    ],
    time_spec: TimeSpec,
)
```

入力 file parser は data だけでなく metadata を返す。距離が約 `0.01 AU` だから
Earth-relative、約 `1 AU` だから heliocentric、という推測は禁止する。距離範囲は
warning / validation hint にだけ使う。

---

## 4. observer convention

設定名は `parallax_observer_convention` とする。旧 convention 設定は互換
adapter を設けず削除する。

### 4.1 `earth_geocentric_offset` — 通常利用者向け既定値

入力:

- Earth heliocentric ephemeris `r_E(t)`
- optional satellite geocentric offset `δr_sat/E(t)`

displacement:

```text
Δr_E(t) = r_E(t) - r_E(tref) - v_E(tref) * (t - tref)
Δr(t)   = Δr_E(t) + δr_sat/E(t)
```

satellite offset の `tref` 位置・速度は引かない。これは VBM satellite-table
semantics と一致する。annual-only は `δr_sat/E=0` である。

ユーザーが普通に「地球から見た衛星位置 vector」を与える場合は必ずこの mode を使う。
既存の RTModel/VBM形式 `JD RA Dec distance` parser もこの origin を返す。

### 4.2 `heliocentric_observer`

入力:

- complete heliocentric observer ephemeris `r_obs(t)`
- complete heliocentric reference ephemeris `r_ref(t)`

displacement:

```text
Δr(t) = r_obs(t) - r_ref(tref) - v_ref(tref) * (t - tref)
```

reference を省略して observer 自身を使うことは禁止する。必要なら caller が明示的に
同じ ephemeris を二つ渡す。

### 4.3 `gulls`

式は `heliocentric_observer` と同じだが、GULLS data adapter と検証を持つ専用 mode
とする。

- observed orbit: `World[obsidx].orbit` の全 component の和
- reference orbit: `World[0].orbit` の全 component の和
- `tref`: GULLS event が使用した reference epoch
- trajectory sign: GULLS `compute_tushifts()` と一致
- GULLS sourceから作る small golden file に bitwise-close の test を置く

GULLS input が Earth-relative perturbation table しか持たない場合、adapter が
`r_obs=r_Earth+δr` を明示的に構築する。native evaluator に曖昧な table を渡さない。

`convention="gulls"` はこの strict semantics に直接 map する。現行
`GullsSpaceParallaxProjector` の self-reference式は削除し、compatibility flagや
deprecated aliasは作らない。移行前後の数値差は bug fixとして release noteへ明記する。

---

## 5. trajectory model

canonical displacement から次を計算する。

```text
ΔN = dot(Δr, north_hat)
ΔE = dot(Δr, east_hat)

Δτ    = -(piEN * ΔN + piEE * ΔE)
Δbeta =   piEE * ΔN - piEN * ΔE

tau(t)  = tau_rectilinear(t) + Δτ
beta(t) = u0 + Δbeta
u(t)    = hypot(tau, beta)
```

rectilinear term:

```text
tau_rectilinear = (t_eval - t0_eval) / tE
```

`tE > 0` とし、`u0` は signed parameter のまま扱う。

parallax parameterは常に `(piEN, piEE)` で公開する。VBMの `pai1/pai2` や
GULLS の `piE, phi_pi` は adapter 内だけで変換し、新 core API には入れない。

Earth annual term、satellite offset、GULLS reference termを別々に debug output として
返せるようにする。符号 bug の診断に必要である。

---

## 6. VBM の責務

VBM は magnification engine と finite-source table provider に限定する。

```text
PSPL: A[i] = vbm.PSPLMag(u[i])
FSPL: A[i] = vbm.ESPLMag2(u[i], rho)
```

具体的には次を守る。

- trajectory、Earth orbit、satellite orbit、reference frame は自前 C++。
- `SetObjectCoordinates` は VBM parallax のためには呼ばない。
- `LoadSunTable` は呼ばない。
- VBM satellite ephemeris loader は呼ばない。
- limb-darkening が必要な model は既存の VBM設定を明示的に渡す。
- VBM instance と ESPL table は evaluator object の寿命中 cache する。
- VBM object が thread-safe でない可能性を考慮し、OpenMP threadごとに instance を持つか、
  magnification loop を serial にする。共有 mutable instance を複数 thread から呼ばない。

PSPL は analytic formulaでもよいが、「増光率 engine は VBM」という設計を単純に保つため、
初期実装は VBM `PSPLMag` に揃える。performance profile で明確な差がある場合だけ、
VBMと parity testを持つ analytic fast pathを追加する。

---

## 7. native C++ module 構成

既存 `_vbm_cpp.cpp` に trajectory、optimizer、Python glue を全部増築しない。
次のように責務を分ける。

```text
src/jacscanomaly/
  cpp/
    ephemeris.hpp
    sky_projection.hpp
    parallax_trajectory.hpp
    vbm_magnification.hpp
    profiled_flux.hpp
    finite_difference.hpp
  _parallax_cpp.cpp            # NumPy/Python binding
  parallax_backend.py          # validation, adapters, scipy orchestration
```

ビルドは既存 `setup.py` の optional VBM extension discovery を再利用する。
`_parallax_cpp` は VBM source が見つかる環境で build する。build unavailable時に
JAXへ silent fallback せず、parallax fallbackを要求された時点で明確な
`ImportError` と install guidance を返す。

### 7.1 C++ evaluator object

Python から毎 residual call ごとに ephemeris を変換しない。constructor で検証・所有する。

概念 API:

```python
evaluator = _parallax_cpp.ParallaxEvaluator(
    time,
    flux,
    ferr,
    dataset_id,
    ra_deg,
    dec_deg,
    tref,
    time_kind,
    observer_convention,
    earth_ephemeris,
    satellite_or_observer_ephemeris,
    reference_ephemeris,
    finite_source,
    espl_table_path,
    vbm_tol,
    vbm_reltol,
)

residual = evaluator.residual(raw_params, active_mask=None)
residual, jac = evaluator.residual_and_jacobian(
    raw_params, active_mask=None, fd_step=...
)
model = evaluator.evaluate(raw_params)
trajectory_debug = evaluator.trajectory(raw_params, components=True)
```

`dataset_id` は複数 observatory / band の linear flux parameterを別々に profile するために使う。
単一 dataset は全要素0でよい。

### 7.2 parameter contract

optimizer raw parameter:

```text
PSPL parallax: [t0, log_tE, u0, piEN, piEE]
FSPL parallax: [t0, log_tE, u0, log_rho, piEN, piEE]
```

公開 result:

```text
PSPL parallax: [t0, tE, u0, piEN, piEE]
FSPL parallax: [t0, tE, u0, rho, piEN, piEE]
```

existing fallback seed contractとの境界で一度だけ変換する。現在の fitterごとに
`tE` と `log_tE` が混在する状態を新 backend 内へ持ち込まない。

### 7.3 linear flux profile

各 nonlinear parameter評価で datasetごとに

```text
F_model = fs_dataset * A + fb_dataset
```

の weighted linear least squares を C++ で解く。既存 `_solve_fs_fb_numpy` と
`_vbm_cpp.cpp::solve_fluxes` の意味に合わせる。

- singular / non-finite solve は invalid residual
- `ferr` は constructor で正値検証する
- active contamination mask があれば fit と residualの両方に同じ maskを適用する
- protected-support penalty は robust outer objective側で従来どおり加える
- model / raw residualは全点について返す

### 7.4 Jacobian

第一実装では C++ 内の adaptive central finite differenceを使う。

- SciPyから `jac=evaluator.jacobian` を渡す
- Python/SciPyの `"2-point"` で evaluatorを何度も往復しない
- scaleは raw parameterごとに設定する
- `t0`: cadenceと `tE` に応じた step
- `log_tE`, `log_rho`: dimensionless relative step
- `u0`, `piEN`, `piEE`: absolute + relative floor
- central evaluationが boundを越える場合は one-sided difference
- non-finite modelなら stepを縮めて再試行
- Jacobian columnがほぼゼロなら status flagを返す

`piEN/piEE` に対する `tau/beta` derivativeは解析的なので後から実装できるが、
初回PRの correctness blockにしない。まず golden parity と robust convergenceを優先する。

---

## 8. optimizer と basin exploration

### 8.1 第一選択

Python orchestration は `scipy.optimize.least_squares` を使う。

```python
least_squares(
    evaluator.residual,
    x0,
    jac=evaluator.jacobian,
    method="trf",
    bounds=(lower, upper),
    x_scale="jac",  # zero-column時のguardを入れる
    loss="linear",  # contaminationはouter segmentationで扱う
    ...
)
```

理由:

- `piE` bounds と `rho/tE` bounds を明示できる。
- local LMより trust regionのほうが悪い seed / weakly identified rho に耐える。
- optimizer orchestrationを C++ に再実装せず、trajectory/modelの速度を C++ で得られる。
- parameterは5または6個なので、一回の C++ callback 境界コストは支配的でない。

SciPy `method="lm"` は boundsを使えないため、最終 local polishの optional pathに限定する。
既存 native LM は FSPL non-parallax用として残してよいが、新 parallax path の唯一の
optimizerにはしない。

### 8.2 bounds

configから物理値を受け、raw boundsへ変換する。

- `t0`: observed range + configurable margin
- `tE`: positive、survey-specific min/max
- `u0`: signed、絶対値 upper bound
- `rho`: positive、configured min/max
- `sqrt(piEN^2+piEE^2) <= max_piE`

SciPyのbox boundsだけでは円形 `piE` boundを表現できない。初期実装は
`piEN,piEE in [-max_piE,max_piE]` に加え、residual側で radial soft/hard guardを入れるか、
`piE_radius, angle` parameterizationを seed stageだけで使う。最終 resultが
radial bound上なら `parameter_at_bound=True` とする。

### 8.3 multistart

既存 `singlelens_fallback.py` の構造化 seed生成を維持する。

- PSPL basin: `t0`, `tE`, signed `u0`
- FSPL basin: `rho/u0`, `t_star`, peak offset
- parallax basin: radius × angle、detector anchor
- joint: FSPL上位 basin × parallax上位 basinを明示合成

各 seed の local solveは独立なので、必要なら Python process pool または C++側 parallel
batchで並列化できる。ただし VBM thread safetyを確認するまで thread共有しない。

rankingは現行修正済みの共通 robust objectiveを使う。fit後は最終 parameterで再segmentし、
同じ `(theta, segmentation)` の objective、protected-support penalty、original chi2、
effect detector再評価を揃えて比較する。

### 8.4 fit status

`optimizer_success` だけを成功条件にしない。最低限次を返す。

```python
NativeParallaxDiagnostics(
    optimizer_success,
    optimizer_status,
    nfev,
    njev,
    chi2,
    rank,
    jacobian_condition,
    parameter_at_bound,
    ephemeris_extrapolated,
    nonfinite_evaluations,
    observer_convention,
    backend="native_cpp_vbm_magnification_scipy_trf",
)
```

fallback acceptance は、original chi2、robust objective、segmentation stability、
parameter stability、fit後の該当 physical detector scoreを使う。総chi2を
effect scoreの代用にしない。

---

## 9. Python fitter family と routing

新しい公開内部 fitter を次に揃える。

```text
NativePSPLAnnualParallaxFitter
NativeFSPLAnnualParallaxFitter
NativePSPLSpaceParallaxFitter
NativeFSPLSpaceParallaxFitter
```

4 class は model dimension と finite-source flag以外を共通 base / helperに寄せる。

`make_effect_fitter()` の完成時 mapping:

| effect | fitter |
|---|---|
| `annual_parallax` | `NativePSPLAnnualParallaxFitter` |
| `space_parallax` | `NativePSPLSpaceParallaxFitter` |
| `fspl_parallax` | `NativeFSPLAnnualParallaxFitter` |
| `fspl_space_parallax` | `NativeFSPLSpaceParallaxFitter` |

`single_fit_backend` によって parallax path が JAXへ戻らないようにする。
parallax用には別設定を追加する。

```python
parallax_fit_backend: Literal["native_cpp"] = "native_cpp"
parallax_optimizer: Literal["scipy_trf", "native_lm_polish"] = "scipy_trf"
parallax_observer_convention: Literal[
    "earth_geocentric_offset",
    "heliocentric_observer",
    "gulls",
] = "earth_geocentric_offset"
```

移行完了後、parallax fallback pathから次を削除する。

- `PSPLParallaxFitter`
- `FSPLParallaxFitter`
- `PSPLSpaceParallaxFitter`
- `FSPLSpaceParallaxFitter`
- `VBMFiniteDiffGulls*`
- `trajectory.py` の JAX parallax callback

plotter/result compatibilityのため、`SingleLensFitResult` の outward interfaceは維持する。
内部 `params/raw_params` の NumPy/JAX型依存は整理し、少なくとも新 path は NumPy arrayを
canonicalとする。

---

## 10. correctness tests

### 10.1 pure geometry unit tests

VBMを呼ばず trajectoryだけを testする。

1. RA=0, Dec=0 の basis。
2. pure North displacementで `piEN=1, piEE=0`。
3. pure East displacementで `piEN=0, piEE=1`。
4. `piEN/piEE` mixed vector。
5. `t=tref` の Earth annual displacementが0。
6. geocentric satellite offsetは `t=tref` でも消えない。
7. strict GULLSで observerとreferenceが同一なら `t=tref` で0。
8. 現行 self-reference式と strict GULLS式が、satellite offsetの定数・速度分だけ
   異なることを示す regression test。期待値は strict GULLS側とし、旧式は採用しない。

### 10.2 VBM parity

小さい固定 ephemeris / target / parameter setに対し、
旧 VBM `*LightCurveParallax` が出す `u` または magnificationを golden fileに保存する。
新 C++ trajectory + VBM magnificationが tolerance内で一致することを確認する。

parameter grid:

- positive / negative `u0`
- `piEN` only / `piEE` only / mixed
- JD / HJD
- annual only
- Earth-relative satellite offsetあり
- PSPL / FSPL

runtime testが installed VBM implementationの parallax routineへ依存しないよう、
golden generation scriptと committed small fixtureを分ける。

### 10.3 strict GULLS parity

`../gulls/src/classes/parallax.cpp` と同じ式で作った小さい fixtureを committed test dataに置く。
可能なら GULLS executableから `NEshift/tushift` をdumpする generatorも `tools/` に置く。

検査対象:

- `World[0]` referenceと `World[obsidx]` observerが異なる case
- component orbitの加算
- `tref` 位置・reference速度
- North/East ordering
- `piE,phi_pi` と `piEN,piEE` の変換

### 10.4 origin misuse

Earth-relative約0.01 AUの tableを heliocentric observerとして渡す、またはその逆を行った時、
constructorが metadata mismatchで拒否する testを置く。distance heuristicで自動修正しない。

### 10.5 fitter regression

- annual PSPL synthetic recovery
- annual FSPL synthetic recovery
- space PSPL synthetic recovery
- space FSPL synthetic recovery
- planet-like compact contaminationを加えた同じ4 case
- mixed FSPL + parallax + planet
- `../roman_simu/0_599_2302`
- `../roman_simu/2_755_3280`

truthがある場合は chi2だけでなく `rho`, `piEN`, `piEE` の recoveryを見る。
truthがない real/simulation caseは、現行 benchmark artifactと detector score reduction、
parameter-at-bound、segmentation安定性を記録する。

### 10.6 no-JAX regression

parallax fallback testで JAX/JAXopt entry pointを monkeypatchして例外を投げても fitが完了する
ことを確認する。または isolated subprocessで import traceを取り、新 native parallax
moduleが `jax` / 旧 JAX optimizer を importしないことを確認する。

repository全体からJAXを即時削除する必要はない。planet scan等の既存別用途は今回の
scope外だが、**parallax model/fallback call graph** には一切入れない。

---

## 11. performance acceptance

同一 machine / input / seedで計測する。

- C++ evaluator単体: `trajectory + magnification + flux profile`
- residual + C++ finite-difference Jacobian
- 1 local solve
- full structured multistart
- robust outer iterationを含む fallback全体

最低条件:

1. 新 pathが同じ dataに対する現行 JAX/NumPy pathより遅くならない。
2. C++ residual callbackに Python point-wise loopがない。
3. ephemeris parse / allocation / ESPL table loadを residualごとに行わない。
4. benchmarkで fit qualityを速度のために落とさない。
5. `0_599_2302`, `2_755_3280` の wall timeと fit diagnosticsをJSON artifactに保存する。

「速いが別の座標系を fitしている」は失敗である。performance PRは geometry parity testを
通過した後だけ mergeする。

---

## 12. planet pipeline への接続

### 12.1 基本思想

physical-effect detectorは完全な planet / single-lens分類器ではない。目的は、
FSPL/parallaxにより惑星候補が false negativeになる eventを救い、必要な eventだけ
重い fallbackへ送ることである。

同じ planet extractorを fallbackの前後で実行する。

```text
initial single-lens fit
        |
        +--> planet scan BEFORE --------------------+
        |                                           |
        +--> physical effect detection/routing      |
                    |                               |
                 fallback?                          |
                 /      \                           |
               no        yes                        |
               |          |                         |
               |    robust native fallback          |
               |          |                         |
               |    planet scan AFTER               |
               |          |                         |
               +----------+-------------------------+
                          |
                  candidate matching/classification
```

fallback fitting中は従来どおり compact planet-like blocksを contaminationとして扱い、
FSPL/parallax parameterを守る。planet modelとのjoint physical fitはこの phaseでは行わない。

### 12.2 public entry point

既存 `Finder.run()` の戻り値や意味を黙って変えない。新しい明示 entry pointを追加する。

候補:

```python
result = finder.run_effect_aware(
    time,
    flux,
    ferr,
    x0=...,
    run_planet_before=True,
    run_planet_after=True,
)
```

または `Finder.run(..., effect_aware=True)` でもよいが、初回実装は新 methodのほうが
backward compatibilityを保ちやすい。

### 12.3 result type

```python
@dataclass(frozen=True)
class EffectAwareFinderResult:
    initial_fit: SingleLensFitResult
    selected_fit: SingleLensFitResult
    effect_candidates: tuple[EffectCandidate, ...]
    routing_decision: object
    fallback_result: FallbackResult | None
    planet_before: FinderResult | None
    planet_after: FinderResult | None
    candidate_matches: tuple[PlanetCandidateMatch, ...]
    final_candidates: tuple[object, ...]
    reason_codes: tuple[str, ...]
    diagnostics: dict[str, object]
```

元の `FinderResult` / plot APIを再利用し、巨大 arrayの不要なcopyを避ける。

### 12.4 execution policy

1. initial PSPL fitを得る。
2. initial residualに既存 planet scanを実行し、`planet_before` を保存する。
3. 同じ initial fitから FSPL/parallax detectorを実行する。
4. routingが `skip` なら fallbackせず initial fitを selected fitとする。
5. `exact_probe` を経て fallback条件を満たす eventだけ robust fallbackする。
6. fallback acceptanceを満たした場合だけ selected fitを置換する。
7. accepted fallback後の residualに、**同一設定の planet scan** を再実行する。
8. fallback failure時は initial fitと `planet_before` を失わず返す。
9. before / after candidatesを照合し、final candidate listと理由を作る。

before scanを毎 eventで実行するのが現行通常 pipelineそのものであれば追加費用はない。
after scanは fallback accepted eventにだけ実行する。これにより重い再scanの件数を抑える。

### 12.5 candidate matching

candidate identityを indexだけで比較しない。少なくとも次を使う。

- anomaly support interval IoU
- peak time差 / local cadence
- sign
- season id
- template scaleまたはwidthの比

初期 rule:

```text
same candidate if
  same season
  and same sign
  and (
      support IoU >= threshold
      or abs(t_peak_before - t_peak_after) <= k * local_cadence
  )
```

category:

- `survived`: beforeとafterの両方に存在
- `revealed_after_fallback`: afterにだけ存在
- `explained_by_single_lens_effect`: beforeにだけ存在し、fallback後に消失
- `changed`: matchしたが score/shapeが大きく変化
- `unresolved`: matchingが曖昧

この categoryは最終的な planet確率ではない。特に `survived` は強い planet候補だが、
artifactや別の single-lens mismatchの可能性を残す。

### 12.6 final candidate policy

- accepted fallbackあり: `planet_after` を final候補の基準にする。
- fallbackなし / failure: `planet_before` を基準にする。
- beforeに強く、afterで消えたcandidateは捨てず
  `explained_by_single_lens_effect` として provenanceに残す。
- afterで新たに現れたcandidateは `revealed_after_fallback` として優先的に記録する。
- protected blockに重なったcandidateは、fallbackがその区間を無理に説明していないことを
  diagnosticsで確認する。

### 12.7 mixed FSPL + parallax + planet

joint fallbackは既存の staged designを使う。

```text
PSPL
  -> FSPL stage
  -> annual or space parallax stage
  -> top FSPL basin × top parallax basin
  -> joint FSPL+parallax robust fit
  -> final segmentation/effect recheck
  -> planet rescan
```

planet-like compact blockを完全除外して固定するのではなく、contamination stateとして
交互更新する。protected FSPL peak supportやbroad parallax supportをplanet maskが奪わない
既存 penaltyを共通 objectiveに含める。

### 12.8 pipeline metrics

eventごと:

- routing decision / detector scores
- fallback実行有無と時間
- before / after anomaly count
- before / after best score
- revealed / survived / explained count
- fallback fit diagnostics
- residual detector score reduction

batch:

- fallback rate
- fallback acceptance rate
- planet recall before / after
- false positive rate before / after
- effect-only eventで残る false planet数
- planet+effect eventでの recovered planet数
- median / p95 runtime

目的は fallbackを全 eventへ適用することではない。planet recallを改善しながら、
fallback rateとp95 runtimeを予算内に保つ。

---

## 13. planet接続 test matrix

最低限、次の fixtureを固定する。

| truth / injected morphology | 期待 |
|---|---|
| clean PSPL | fallbackなし、planetなし |
| planet only | fallbackに吸われず before/afterで survived |
| FSPL only | FSPL fallback、before false candidateは消える |
| annual parallax only | parallax fallback、broad false candidateは消える |
| space parallax only |正しい observer conventionで fallback |
| FSPL + planet | FSPL fit後に planet survived/revealed |
| parallax + planet | parallax fit後に planet survived/revealed |
| FSPL + parallax + planet | joint fit後にも compact planet signalが残る |
| wrong ephemeris origin | fitせず明示 error |
| fallback optimizer failure | initial planet resultを保持 |

planet injectionは位置、duration、sign、amplitudeを振る。FSPL peakに重なる hardest caseと、
parallax broad residual上にcompact anomalyが乗る caseを必ず含める。

---

## 14. Luna向け実装順序

一つの巨大PRにしない。次の順で小さく実装する。

### PR 1: coordinate contract と pure C++ trajectory

- [x] `TimeSpec`, `Ephemeris`, observer convention enum
- [x] strict parser/validation
- [x] C++ interpolation、basis、annual/space/GULLS displacement
- [x] trajectory-only binding/debug API
- [x] geometry unit test
- [ ] VBM/GULLS golden generatorとfixture

このPRでは fitter routingを変更しない。

### PR 2: VBM magnification evaluator

- [x] `u -> PSPLMag/ESPLMag2`
- [x] dataset別 linear flux profile
- [x] residual/model API
- [ ] VBM parity test
- [x] cache/thread-safety
- [ ] evaluator microbenchmark

このPRでも production fallbackを切り替えない。

### PR 3: SciPy bounded fitter と annual parallax migration

- [x] C++ finite-difference Jacobian
- [x] `NativePSPLAnnualParallaxFitter`
- [x] `NativeFSPLAnnualParallaxFitter`
- [x] structured seedとの接続
- [x] annual parallax fallbackをnative pathへ切替
- [x] no-JAX call-graph test
- [ ] old/new regression benchmark

### PR 4: space parallax migration

- [x] `earth_geocentric_offset`
- [x] `heliocentric_observer`
- [x] strict semanticsの `gulls`
- [x] PSPL/FSPL space fitter
- [x] `0_599_2302`, `2_755_3280` benchmark
- [x] 旧 self-reference結果との差分を明記した release note

### PR 5: cleanup

- [x] parallax fallback factoryを4 native fitterに統一
- [x] production routing / public APIからJAX/NumPy旧 parallax fitterを削除
- [x] config/docs/API更新
- [x] full test suite
- [ ] performance artifact

### PR 6: planet pipeline接続

- [x] `EffectAwareFinderResult`
- [x] explicit effect-aware entry point
- [x] before scanの保持
- [x] accepted fallback後だけ after scan
- [x] candidate matching/provenance
- [x] planet + parallax integration test
- [ ] batch metrics

各PRで既存の unrelated API/refactorを混ぜない。座標系の parityを通す前に optimizer tuningへ
進まない。benchmarkの改善を理由に strict GULLS semanticsを変更しない。

---

## 15. Definition of Done

### Native parallax backend

- [x] annual/space、PSPL/FSPLの全4 pathが同じ native trajectory coreを使う。
- [x] VBMは `PSPLMag` / `ESPLMag2` のためだけに使われる。
- [x] parallax fallback call graphに JAX/JAXoptがない。
- [x] 通常の satellite inputは明示的な Earth-relative vectorとして扱われる。
- [x] `gulls` modeは observer/reference orbitを分離する。
- [x] 現行 self-reference互換 modeは native pathに存在しない。
- [x] SciPy bounded `least_squares` が primary optimizerである。
- [x] final parameterで segmentationとphysical effect scoreを再評価する。
- [ ] mixed FSPL+parallax+planet benchmarkで `rho/piE` をcompact anomalyから守れる。
- [x] `0_599_2302`, `2_755_3280` の結果・時間・diagnosticsを harness が保存できる。

### Planet pipeline connection

- [x] fallback前のplanet resultを失わない。
- [x] accepted fallback後だけ同じ planet scanを再実行する。
- [x] before/after candidateに provenanceとcategoryが付く。
- [ ] effect-only false candidateが減る。
- [ ] planet+effect eventの false negativeが改善する。
- [x] fallback failureでも通常pipeline resultを返せる。
- [ ] fallback rate、recall、false positive、runtimeをbatchで評価できる。
- [x] 既存 `Finder.run()` の挙動を壊さない。

この二つが満たされて初めて、「parallax/FSPL残差を必要なeventだけ重くfitし、そのせいで
隠れていたplanet signalを既存pipelineへ戻す」という初期目標の実装完了とする。
