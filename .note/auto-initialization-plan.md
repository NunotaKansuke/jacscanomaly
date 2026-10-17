# 自動初期値の一本化 実装計画(案・未承認)

## 現状(コードを読んで確認した範囲)
初期値を作る場所が3つ別々にある。

| # | 場所 | 入口 | 中身 |
|---|---|---|---|
| A | `Finder._estimate_*`(finder.py:1966-) | `fit_single_lens(x0=None)` | PSPL は FFT スキャン、FSPL は継続時間グリッド(fspl_initialization.py) |
| B | `_ParallaxFitter.fit(q0=None)`(parallax_backend.py:~470) | parallax fitter 直 | FSPL parallax のみ A と同じ FSPL グリッド+πE=0。**PSPL parallax は q0 必須で ValueError** |
| C | `detector_seed_parameters`(singlelens_fallback.py:301) | 検出器 fallback | ベース+検出器シード+tE/u0/ρ/t0 の倍率グリッドを手で積み、max_seeds=24 で切る。マジックナンバーが多い |

検出器(C)は A/B を使わず独自にシードを作るので、「標準経路」と「検出器経路」で初期値の作り方が二重になっている。

## 目標
- 初期値が無ければ、**スキャン→(FSPL)→(parallax)** を自動で全部やる経路を1本にする(標準経路)。
- 検出器は、その経路を呼ぶだけにする。検出器が持つ情報は「ヒント」として渡す。

## 設計
新モジュール `auto_init.py`(1か所にまとめる)。

    initial_guesses(model, time, flux, ferr, *, hints=None, ...) -> tuple[np.ndarray, ...]

- model ∈ {pspl, fspl, pspl_parallax, fspl_parallax}。
- 段階的に前段の結果を使う:
  1. **pspl**: 既存の FFT スキャン(A)を移す。出力 (t0, tE, u0)。
  2. **fspl**: 既存の継続時間グリッド。PSPL からは t0 だけ使う(現行仕様どおり)。
  3. **parallax**: 前段(pspl か fspl)の解に πE を足す。何を足すかは下の未決事項。
- `hints`: 検出器が持つ情報(検出 t0/tE/u0、ρ、πE の向き)。あれば候補の先頭に置くだけで、グリッドは作り替えない。
- 各 fitter の `fit(q0=None)` と `Finder.fit_single_lens(x0=None)` は全て `initial_guesses` を呼び、各シードを LM して χ² 最小を採る(今の B と同じ流れ)。PSPL parallax の ValueError は無くなる。
- 検出器 fallback(C)は `initial_guesses(..., hints=検出器出力)` を呼ぶだけにして、`detector_seed_parameters` の倍率グリッドは削る。

## 進め方(各段で既存テストと 2001129 の parity を確認)
1. `auto_init.py` を作り、A/B の既存ロジックを**挙動を変えずに**移す。(純リファクタ)
2. fitter の `fit(q0=None)` を `auto_init` 経由にし、PSPL parallax の自動初期値を追加。
3. 検出器 fallback を `auto_init` 呼び出しに置換。**ここだけ挙動が変わる**。
4. 使われなくなったコードを削除。

## 受け入れ基準
- テスト: 現状 223 passed(+ microjax 欠如による effect_detection の既知3失敗)から悪化しない。
- 2001129(24スタート設定)の選択 χ² が 69988.495 から悪化しない。ステップ3の前後で比較。
- 他に手元にある実イベント/ベンチマークがあれば、同じ比較をする(要:どれを使うか)。

## 未決事項(承認前に決めたい)
1. **parallax の πE 初期値**: 自動のとき、(a) πE=0 一点から始める、(b) πE=0 +検出器ヒントの向き、(c) 半径×角度のリングを撒く、のどれにするか。
   データから決まるのは「πE の向きの大まかな縮退」までなので、(a)+(b) を提案(リングは検出器のヒントがある時だけ)。
2. **検出器の局所変種**(tE/u0 を振る、ln ρ を ±0.5/1.0 ずらす、u0 符号反転)を残すか。
   符号反転は FSPL の厳密な縮退なので残し、倍率グリッドは「FSPL グリッド」に吸収して消す案。
3. **max_seeds** の扱い。上限を残すか、段階ごとの個数(上位 top_k)に置き換えるか。
