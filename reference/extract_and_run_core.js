/**
 * extract_and_run_core.js — honesty harness.
 *
 * Extracts the <script> content between the fw-core markers directly from
 * index.html (the file actually served to learners — no copy kept in sync by
 * hand), evaluates it in a stubbed DOM-free environment, runs the same
 * fixed-seed experiments as reference/fast_weights.py, and writes
 * js_vectors.json for comparison by test_fast_weights.py.
 *
 * Usage: node reference/extract_and_run_core.js
 */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const m = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/);
if (!m) { console.error('no script block found'); process.exit(1); }
const full = m[1];
const start = full.indexOf('const FW = (() => {');
const end = full.indexOf('/* ============== end of fw-core ============== */');
if (start < 0 || end < 0) { console.error('fw-core markers not found'); process.exit(1); }
const core = full.slice(start, end);

// minimal browser stubs the core's closing lines touch
global.document = { getElementById: () => null };
global.window = global;
eval(core + '\n; global.FW = FW;');

const { PortRng, makeState, hebbianWrite, read, makePairGenerator, capacityCurve, decayCurve } = global.FW;

const d = 24; lam = 0.9; seed = 1234; sparsity = 0.2; overlap = 0.0;
const gen = makePairGenerator(d, sparsity, overlap, seed);
let M = makeState(d);
const pairs = [], states = [];
for (let t = 0; t < 6; t++) {
  const [k, v] = gen();
  M = hebbianWrite(M, k, v, lam, d);
  pairs.push({ k: Array.from(k), v: Array.from(v) });
  states.push(Array.from(M));
}
const retrievalAtProbe = Array.from(read(M, pairs[2].k, d));

const out = {
  params: { d, lam, seed, sparsity, overlap, nPairs: 6, probeIndex: 2 },
  pairs, states, retrievalAtProbe,
  capacityHead: capacityCurve({ d: 24, lam: 1.0, nPairs: 8 }),
  decayHead: decayCurve({ d: 24, nPairs: 5, lam: 0.9, seed: 7 }),
};
fs.writeFileSync(path.join(__dirname, 'js_vectors.json'), JSON.stringify(out, null, 1));
console.log('wrote js_vectors.json — pairs:', pairs.length,
  '| retrieval[0..3]:', retrievalAtProbe.slice(0, 4).map(x => x.toFixed(6)).join(', '),
  '| capacity last:', JSON.stringify(out.capacityHead[out.capacityHead.length - 1]));
