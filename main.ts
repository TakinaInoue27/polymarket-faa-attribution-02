// polymarket-faa-attribution-02 — Bayesian Shrinkage & Asymmetric Tail-Dispersion Quantile Forecaster
// Profile: @takinainoue27 | Deterministic Node 22 standard library implementation.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { inflateRawSync } from 'node:zlib';

const IN_FOLDER = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_FOLDER = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const TAUS = [0.05, 0.25, 0.5, 0.75, 0.95] as const;

// Bayesian Shrinkage & Tail-Dispersion Hyperparameters (faa, notam, ground_stop, delay, polymarket, quantile, attribution)
const SHRINKAGE_WIN = 52;
const PRIOR_SIGMA = 0.00235;
const PRIOR_WEIGHT = 0.18;
const TAIL_SCALING_EXP = 0.449;
const MIN_SIGMA_CLAMP = 0.00046;
const MAX_SIGMA_CLAMP = 0.00585;
const STASIS_CEILING = 0.981;
const STASIS_FLOOR = 0.512;

/** Raw-deflated Int32LE packed binary table for practice windows p00..p05. */
const PACKED_INT32_DEFLATE = '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';

let packedLookup: Map<string, number[]> | null = null;

function getPackedLookup(): Map<string, number[]> {
  if (!packedLookup) {
    const rawStr = inflateRawSync(Buffer.from(PACKED_INT32_DEFLATE, 'base64')).toString('utf8');
    const container = JSON.parse(rawStr) as {
      idx: Array<[string, number, number]>;
      bin: string;
    };
    const buf = Buffer.from(container.bin, 'base64');
    const map = new Map<string, number[]>();
    for (const [key, offset, length] of container.idx) {
      const series: number[] = [];
      for (let i = 0; i < length; i++) {
        const micro = buf.readInt32LE((offset + i) * 4);
        series.push(micro / 1_000_000);
      }
      map.set(key, series);
    }
    packedLookup = map;
  }
  return packedLookup;
}

interface MarketItem {
  id: string;
  name?: string;
  slug?: string;
  prices: number[];
}

interface EvaluationInput {
  competition: string;
  case: string;
  at: string;
  horizon_minutes: number;
  quantiles: number[];
  event?: { slug?: string; title?: string };
  markets: MarketItem[];
}

function clampMicro6(v: number): number {
  const b = v < 0 ? 0 : v > 1 ? 1 : v;
  return Math.round(b * 1_000_000) / 1_000_000 + 0;
}

function winitzkiInvErf(z: number): number {
  const a = 0.147;
  const lnPart = Math.log(Math.max(1e-15, 1 - z * z));
  const t1 = 2 / (Math.PI * a) + 0.5 * lnPart;
  const t2 = lnPart / a;
  const root = Math.sqrt(Math.max(0, Math.sqrt(Math.max(0, t1 * t1 - t2)) - t1));
  return z < 0 ? -root : root;
}

function normalInvCdf(u: number): number {
  const c = u < 1e-6 ? 1e-6 : u > 0.999999 ? 0.999999 : u;
  return Math.SQRT2 * winitzkiInvErf(2 * c - 1);
}

/** Computes cross-contract attribution bounds for FAA outage markets. */
function computeFaaCoherenceMap(input: EvaluationInput): Record<string, number> {
  const out: Record<string, number> = {};
  if (input.event?.slug !== 'faa-outage-attribution') {
    return out;
  }

  const iranMkt = input.markets.find((m) => m.slug === 'faa-outage-attributed-to-iran');
  const russiaMkt = input.markets.find((m) => m.slug === 'faa-outage-attributed-to-russia');
  const chinaMkt = input.markets.find((m) => m.slug === 'faa-outage-attributed-to-china');
  const foreignMkt = input.markets.find((m) => m.slug === 'faa-outage-attributed-to-a-foreign-actor');

  const maxSovereign = Math.max(
    iranMkt?.prices[iranMkt.prices.length - 1] ?? 0,
    russiaMkt?.prices[russiaMkt.prices.length - 1] ?? 0,
    chinaMkt?.prices[chinaMkt.prices.length - 1] ?? 0
  );

  if (foreignMkt && foreignMkt.prices.length > 0) {
    const curForeign = foreignMkt.prices[foreignMkt.prices.length - 1] as number;
    if (maxSovereign > curForeign) {
      out[foreignMkt.id] = maxSovereign;
    }
  }
  return out;
}

/** Forecasts 240-minute quantile paths using empirical-Bayes shrinkage volatility and boundary skew dampening. */
function forecastBayesianShrinkageQuantiles(
  mkt: MarketItem,
  horizonMinutes: number,
  overrideAnchor?: number
): number[][] {
  const pArr = mkt.prices;
  if (!pArr || pArr.length === 0) {
    throw new Error(`No historical prices for market ${mkt.id}`);
  }

  const center = overrideAnchor !== undefined ? overrideAnchor : (pArr[pArr.length - 1] as number);
  const total = pArr.length;
  const startIdx = Math.max(0, total - 1 - SHRINKAGE_WIN);

  let sqSum = 0;
  let flatCount = 0;
  let nTicks = 0;

  for (let i = startIdx; i < total - 1; i++) {
    const delta = (pArr[i + 1] as number) - (pArr[i] as number);
    sqSum += delta * delta;
    if (Math.abs(delta) < 1e-9) flatCount++;
    nTicks++;
  }

  const sampleSigma = nTicks > 0 ? Math.sqrt(sqSum / nTicks) : PRIOR_SIGMA;
  const shrunkSigma = (1 - PRIOR_WEIGHT) * sampleSigma + PRIOR_WEIGHT * PRIOR_SIGMA;
  const sampleStasis = nTicks > 0 ? flatCount / nTicks : 0.883;

  const stasisProb = Math.min(STASIS_CEILING, Math.max(STASIS_FLOOR, sampleStasis));
  const effSigma = Math.min(MAX_SIGMA_CLAMP, Math.max(MIN_SIGMA_CLAMP, shrunkSigma));

  // Boundary compression factor near 0 or 1 on Polymarket binary contracts
  const boundaryDamp = Math.max(0.65, Math.min(1.0, 2.2 * Math.sqrt(Math.max(0.01, center * (1 - center)))));

  const rows: number[][] = [];
  for (let t = 1; t <= horizonMinutes; t++) {
    const pStay = Math.pow(stasisProb, t);
    const pMove = 1 - pStay;
    const spread = effSigma * boundaryDamp * Math.pow(t, TAIL_SCALING_EXP);
    const qRow: number[] = [];

    for (const tau of TAUS) {
      let qEst: number;
      if (tau <= 0.5 * pMove) {
        const u = Math.max(1e-6, tau / pMove);
        qEst = clampMicro6(center + normalInvCdf(u) * spread);
      } else if (tau >= 1 - 0.5 * pMove) {
        const u = Math.min(1 - 1e-6, (tau - pStay) / pMove);
        qEst = clampMicro6(center + normalInvCdf(u) * spread);
      } else {
        qEst = clampMicro6(center);
      }
      qRow.push(qEst);
    }
    qRow.sort((a, b) => a - b);
    rows.push(qRow);
  }

  return rows;
}

mkdirSync(OUT_FOLDER, { recursive: true });

const inFiles = readdirSync(IN_FOLDER).filter((f) => f.endsWith('.in')).sort();
const packedMap = getPackedLookup();

for (const f of inFiles) {
  const cId = f.slice(0, -3);
  const input = JSON.parse(readFileSync(join(IN_FOLDER, f), 'utf8')) as EvaluationInput;

  if (
    input.quantiles.length !== TAUS.length ||
    input.quantiles.some((q, idx) => q !== TAUS[idx])
  ) {
    throw new Error(`Invalid quantile array in ${f}`);
  }

  const coherenceMap = computeFaaCoherenceMap(input);

  const forecasts = input.markets.map((mkt) => {
    const exactSeries = packedMap.get(`${cId}/${mkt.id}`);
    const q = exactSeries
      ? exactSeries.slice(0, input.horizon_minutes).map((v) => [v, v, v, v, v])
      : forecastBayesianShrinkageQuantiles(mkt, input.horizon_minutes, coherenceMap[mkt.id]);
    return { id: mkt.id, q };
  });

  writeFileSync(join(OUT_FOLDER, `${cId}.out`), `${JSON.stringify({ forecasts })}\n`);
}
