# 惑星アノマリー形態検出器の設計メモ

作成日: 2026-07-29

## 発端

`roman_simu` の `0_800_1851` では、PSPL残差上に明瞭な正のピークが二つある。
現在のscan結果も feature としては二つを検出しているが、二つのピークとその間の
負残差を一つの連続した anomaly candidate にまとめている。その結果、candidateの
代表点は中央の最も深い負残差になり、後段では二峰構造が失われやすい。

この例は、次の三つを同じ概念として扱ってはいけないことを示している。

- 異常が存在する時間区間
- 区間内の局所的な bump / dip
- 複数の局所成分が作る caustic などの上位形態

## 単純な傾き判定の問題

平滑化した残差の傾きや曲率を見ること自体は有用だが、それだけで bump / dip を
確定してはいけない。

- 正→負の傾き変化は局所極大、負→正は局所極小を与える。
- しかし二つのcaustic peakの間の谷も局所極小なので、単純な規則ではdipになる。
- sampling gap、非一様cadence、外れ値、PSPL baseline誤差でも偽の符号反転が生じる。
- 一つの広いbumpの肩や非対称な立ち上がりを複数成分へ過分割する可能性がある。

したがって、傾きは「境界候補・極値候補を生成する特徴量」として使い、最終ラベルは
周辺構造を含む形態判定で決める。

## 検討する階層構造

### 1. 異常区間の検出

符号に依存しない残差エネルギー、連続性、局所ノイズを使って、まず異常区間だけを
決める。この段階では一つの区間が複数ピークを持ってよい。

候補例:

- robustに平滑化した `|residual / error|` または局所χ²
- cadenceを考慮した連続区間
- 小さなgapだけを埋め、大きなgapを跨いで結合しない規則
- 単一点ではなく、時間幅と積分S/Nによる採択

### 2. 区間内の原始成分分解

各異常区間の中で、平滑化残差の一次・二次微分から極値と変曲点を抽出する。
固定point数ではなく時間幅でscaleを定義し、複数scaleで安定な極値だけを残す。

各原始成分には少なくとも次を保存する。

- peak / troughの時刻、符号、高さ
- 左右の立ち上がり・立ち下がり勾配
- prominence
- half-prominence幅
- 左右非対称性
- 極値と境界の不確かさ
- scaleを変えたときの存続性

ここで得るのはまだ最終的な bump / dip ラベルではない。

### 3. 上位形態の判定

原始成分を単独で分類せず、時間順の並びを一つの構造として判定する。

例:

- 単独の正成分: bump候補
- 単独の負成分: dip候補
- `peak - trough - peak`: caustic / double-peak候補
- 急峻なedgeが二つあり、その間に構造がある: caustic-crossing候補
- 正負が隣接し、baselineへ戻らない: 一つの複合アノマリー候補

重要なのは、`peak - trough - peak` の中央troughを独立dipとして数えないこと。
上位caustic候補に採用された原始成分には親IDを持たせ、トップレベルの
`n_dips` からは除外する。ただし内部形態としての深さは保持する。

## Caustic判定器で使えそうな特徴

causticを厳密なbinary-lens fitなしで確定するのは難しいため、最初は
`caustic_like` の形態スコアとして扱う。

- 複数の高prominence peakが同じ異常区間にある
- peak間隔と各peak幅の比
- peak間でbaselineへ戻るか、異常状態が継続するか
- 中央troughの深さを両側peakに対して正規化した値
- peakの左右勾配と曲率の非対称性
- 急峻な立ち上がり / 立ち下がりedgeの対
- 複数scaleでの極値トポロジーの安定性
- PSPL残差の正負だけでなく、観測flux曲線上の連続性

`caustic_like` が高い場合だけ、fold / cusp / caustic-crossing用の局所templateや
binary-lensの安価なprobeへ進める。単純bump/dip分類はその後に行う。

## 実装方針の候補

第一候補は、平滑化微分だけのルールベース分類ではなく、次の小さな形態グラフを作る
方法。

1. 異常区間をnodeの親として作る。
2. 区間内のpersistentな極値を子nodeにする。
3. 隣接極値間のedgeへ時間差、勾配、baseline復帰度を持たせる。
4. node列を `single_bump`, `single_dip`, `double_peak`, `caustic_like`,
   `complex/unknown` に分類する。
5. 不確かな場合は無理にbump/dipへ落とさず `complex/unknown` を残す。

形態を一つに決められない場合は、候補ラベルとスコアを複数保存する。分類器の都合で
誤った一意ラベルを与えるより、後段へ不確かさを渡す方がよい。

## 評価で必ず見るもの

- `0_800_1851` の二つのpeakが同じ親区間の別成分として残ること
- 二峰間のtroughが独立dipとしてカウントされないこと
- 真の単独dipのrecallを落とさないこと
- caustic-like eventを単独bump二個へ過分割しないこと
- sampling gapや単一外れ値で極値数が増えないこと
- smoothing scaleを変えたときのラベル安定性
- event単位だけでなく、成分数・時刻・親子関係をtruthまたは目視ラベルと比較すること

## 未決事項

- 平滑化にSavitzky–Golay、Gaussian process、robust splineのどれを使うか
- cadence gapを跨ぐ連結規則
- caustic-likeから物理probeへ進む閾値
- fold / cusp / full crossingをどこまで形態だけで分けるか
- planet extractionの反復単位を「候補区間」にするか「原始成分」にするか

まずは `0_800_1851` を含む少数の二峰・単峰・真のdip・caustic例を固定した
curated setを作り、上記の親子表現が妥当かを確認してから閾値を決める。
