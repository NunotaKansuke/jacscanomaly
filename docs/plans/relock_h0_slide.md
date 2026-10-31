# Relock: H0 that slides off the start fit

## 0956 (2371)

The published fit sits on a 0.1 d spike at 8749.63 (`tE 0.0034, u0 8.6`);
the host is a weak, broad event at 8730 (`teff ~ 18 d`). `relock_host`:

* H0 (start: the spike fit) minimises by masking the spike and refitting:
  its host moves to 8730 (`t0 8730.09`), J0 40490.4;
* H1 (core masked) reaches the same host, J1 40490.4;
* `dJ = J0 - J1 = -0.004` (round-off), so `relocked = False` and the spike
  fit is kept: no window, no anomaly, the broad event unexplained.

## Change (named reasoning)

H0 is "`fit0` is the host". When its own minimisation moves the host's
`t0` out of `fit0`'s core, H0 has left `fit0` for another event: `fit0`
is no host, and J0 / J1 describe the same host up to round-off. Relock in
that case too (still only when the H1 host is detected on its kept
samples).

`t0` outside the core, not "H0 masks the whole core": the core includes the
fit's tails beyond the spike, which the H0 segment need not cover.

A genuine host keeps its `t0` inside its own (wide) core under H0, so its
decision does not change.

## Check

* unit test: weak host + spike, start on the spike, H0 slides: relocked.
* all 2371 events: list every relock decision that changes, look at each.

## Follow-ups (2371 scan)

* Left, but H1 worse: "left" relocked to the H1 host even when its `J` was
  much larger (`dJ << 0`). When H0 left `fit0`, the host is the better of
  H0's and H1's.
* 0595: H0 and H1 both mask from the anomaly to the season's end (next gap
  113 d); the masked host then peaks in the gap (`t0 8600`, last datum
  8598.25), `dJ ~ 0`, relocked; the grey (base) curve was that host. Named
  rule: the host's peak is bracketed by kept samples with no gap longer
  than its `tE` in between; a host peaking where no data are is not seen.
  0595 -> not relocked; 0956 still relocked (`t0 8730`).
