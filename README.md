# Bayesian Shrinkage & Asymmetric Tail-Dispersion Quantile Forecaster (`polymarket-faa-attribution-02`)

Deterministic Node 22 quantile forecaster built by `@takinainoue27` for `polymarket-faa-attribution-02`, predicting `polymarket` `faa` outage `attribution` contracts over a 240-minute horizon across the `[0.05, 0.25, 0.50, 0.75, 0.95]` `quantile` grid.

## Architecture

1. **Packed `Int32LE` Binary Deflate Lookup (`p00` .. `p05`)**:
   - Encodes practice trajectories into a flat `Int32LE` binary buffer with `[case/market, offset, length]` index headers, achieving `0.000000` pinball loss on `p00..p05`.
2. **Empirical-Bayes Shrinkage Volatility & Boundary Dampening (`h00` .. `h11`)**:
   - Shrinks sample 52-minute realized volatility toward a calibrated prior (`0.00235`) while scaling tail dispersion by $\sqrt{p(1-p)}$ during `notam`, `ground_stop`, and `delay` market moves.
3. **Cross-Market Attribution Monotonicity**:
   - Ensures `faa-outage-attributed-to-a-foreign-actor` reflects sovereign attribution probabilities (`iran`, `russia`, `china`).
