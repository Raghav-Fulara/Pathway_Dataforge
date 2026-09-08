"""Generate SUMMARY.pdf (one-page concept summary, 500-950 words) and BLOG.pdf
(the blog essay as PDF) for the DataForge 2026 Pathway submission.

Run:  python3 tools/make_pdfs.py   (from the repo root)
"""
import os
from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = "/usr/share/fonts/truetype/dejavu/"
FSD, FSB, FMN = F + "DejaVuSerif.ttf", F + "DejaVuSerif-Bold.ttf", F + "DejaVuSansMono.ttf"

ACCENT = (93, 155, 233)
DARK = (24, 28, 34)
GREY = (90, 98, 110)

class Doc(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font("ser", "", FSD)
        self.add_font("ser", "B", FSB)
        self.add_font("mono", "", FMN)
        self.set_auto_page_break(True, 11)
        self.set_margins(13, 11, 13)

    def head(self, s, size=9.2, space_before=1.9):
        self.set_font("ser", "B", size)
        self.set_text_color(*DARK)
        self.ln(space_before)
        self.multi_cell(0, size * 0.48, s)
        self.ln(0.3)

    def para(self, s, size=8.05, lh=3.78):
        self.set_font("ser", "", size)
        self.set_text_color(35, 39, 46)
        self.multi_cell(0, lh, s)
        self.ln(0.8)

    def eqbox(self, s):
        self.set_font("mono", "", 7.8)
        self.set_text_color(40, 48, 70)
        self.set_fill_color(240, 243, 249)
        self.multi_cell(0, 3.9, s, fill=True)
        self.ln(0.8)

    def citebox(self, s):
        self.set_font("ser", "", 7.6)
        self.set_text_color(*GREY)
        self.multi_cell(0, 3.6, s)
        self.ln(0.6)


d = Doc()
d.add_page()
# ---- header
d.set_font("ser", "B", 13.5); d.set_text_color(*DARK)
d.cell(0, 6, "Synaptic Plasticity as Short-Term Memory", new_x="LMARGIN", new_y="NEXT")
d.set_font("ser", "", 8.3); d.set_text_color(*GREY)
d.multi_cell(0, 3.9, "Hebbian writes turn a network's wiring into working memory with a fixed-size state — until interference breaks retrieval. "
                     "Companion artifact: interactive explainer (single-file web app, all computation live). DataForge 2026, Pathway track.")
d.ln(1)

d.head("The claim (one falsifiable sentence)")
d.para("A network can store arriving associations in a fixed-size synaptic state through Hebbian writes — no gradient step and no new memory slot per token — "
    "but as associations accumulate, the read returns every stored value whose key overlaps the cue, so interference corrupts retrieval until recall fails. "
    "The artifact lets a learner reproduce both halves of this sentence in under 60 seconds (store one pair: perfect recall; store forty: the same probe fails).")

d.head("The mechanism")
d.eqbox("write:  M_t = \u03bb\u00b7M_{t-1} + v k\u1d40      read:  r = M_t\u00b7q      closed form:  M_t q = \u03a3_i \u03bb^(t-i) (k_i\u00b7q) v_i")
d.para("M \u2208 R^(d\u00d7d) is the synaptic state: every entry is a synapse, and the state's shape never changes. Each arriving association (key k, value v) "
    "potentiates every synapse between their active units — one rank-one Hebbian write, \"cells that fire together wire together.\" Reading with cue q returns a "
    "similarity-weighted sum of all stored values. Sparse, near-orthogonal keys make the matching term dominate at low load; the residual sum of cross-key "
    "overlaps is interference. In the artifact (d = 24, ~5 active units per key, seed 42) exact-match recall is 100% through ~10 stored pairs and falls below "
    "50% by ~22 — nothing was deleted; with \u03bb = 1 every pair is still fully present, and only the read degrades. Finite capacity despite constant memory is "
    "the linear-attention capacity limit identified by Schlag, Irie & Schmidhuber (ICML 2021): \"endlessly adding new associations to a memory of finite size "
    "inevitably will reach a limit.\" Setting \u03bb < 1 swaps the failure mode from interference to geometric decay — forgetting old context rather than corrupting it.")

d.head("Why it matters now")
d.para("A deployed Transformer carries context in a key-value cache that grows with every token; the alternative — a fixed-size state rewritten in place — is the "
    "active design space of linear attention (Gated Linear Attention, ICML 2024; RWKV-7, 2025), test-time-training layers (Sun et al., 2024) and fast-weight "
    "programmers, whose lineage traces to Schmidhuber (1992) and whose peer-reviewed primer (TMLR 2025) describes exactly this mechanism: fast weights are "
    "\"synaptic weights [that] dynamically change over time as a function of input observations, and serve as short-term memory storage.\" Titans (2024) add "
    "surprise-gated test-time memorization. The shared trade: O(1) memory and compute per token versus finite capacity and interference. Every 2024-2026 "
    "architecture below chooses a point on that trade-off; the toy makes the trade-off visible and breakable.")

d.head("The roles of BDH and BDH-CQ")
d.para("BDH (arXiv:2509.26507, 2025) is a brain-inspired post-Transformer architecture in which this mechanism is the working memory itself, not an add-on: "
    "\"the working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons\" (abstract). The paper "
    "reports monosemantic synapses that strengthen when BDH \"hears or reasons about a specific concept,\" and sparse, positive activation vectors — the toy "
    "mirrors the sparse non-negative coding. BDH is not a state-space model in the Mamba sense; its GPU formulation is a separate ReLU-low-rank construction "
    "with linear attention. BDH-CQ (arXiv:2608.09888, 2026) is the demonstration-driven member of the family: inputs at inference \"continuously update the "
    "model's recurrent memory\" and queries are solved by latent-space iteration without verbalized intermediate reasoning — per the problem statement, state "
    "\"accumulates additively per demonstration,\" which is precisely the toy's \u03bb = 1 regime. A 150M-parameter configuration reports 29.5% pass@2 on the public "
    "ARC-AGI-1 evaluation at ~$0.0007 computed cost per task, stated to break the previously reported cost-accuracy Pareto frontier. This is the state-adaptation "
    "route to new tasks, opposed to the optimization route (HRM/TRM fine-tune weights on augmented demonstrations before each puzzle).")

# comparison table
d.head("Landscape (what differs, mechanistically)")
d.set_font("ser", "", 7.0)
cols = [58, 40, 44, 46]
rows = [
    ["System", "Context carrier", "Updated at inference", "Limit / cost"],
    ["Transformer", "KV cache (grows/token)", "append-only", "memory grows O(t); little interference"],
    ["GLA / RWKV-7", "fixed matrix state", "learned gated writes", "interference (this toy's failure)"],
    ["Titans / TTT", "learned fast state", "test-time gradient writes", "learned forgetting; extra compute"],
    ["BDH", "synapses (Hebbian, spiking)", "plasticity while reading", "session memory; dev-reported"],
    ["BDH-CQ", "recurrent memory", "additive per demonstration", "measured on ARC-AGI-1 interventions"],
]
for r_i, row in enumerate(rows):
    if r_i == 0:
        d.set_fill_color(228, 234, 244); d.set_font("ser", "B", 7.0)
    else:
        d.set_fill_color(248, 249, 251) if r_i % 2 else d.set_fill_color(255, 255, 255)
        d.set_font("ser", "", 7.0)
    for c_i, cell in enumerate(row):
        d.cell(cols[c_i], 4.0 if r_i else 3.7, cell, border=1, fill=True)
    d.ln()
d.ln(0.8)

d.head("Evidence levels and limitations")
d.para("BDH and BDH-CQ results cited above are developer-reported arXiv preprints, not peer-reviewed at the time of writing, and no deployment is claimed; the "
    "BDH-CQ companion repository names external evaluators (L. Kaiser; R. Kinas and R. Zhong, the latter two also co-authors) — evaluation, not independent "
    "audit. The toy's numbers are fully reproducible: fixed seeds, and a cross-implementation test checks that the browser's JavaScript core matches the NumPy "
    "reference to 1e-9. Limitations: the toy's write/read rules are hand-set, not learned (frontier systems learn them); no spiking dynamics; d = 24; capacity "
    "crossing points are seed-dependent (the qualitative collapse is not); and this explainer makes no claim about BDH beyond its published statements. "
    "Continue with: arXiv 2102.11174 (fast-weight programmers), 2508.08435 (TMLR primer), 2509.26507 (BDH), 2608.09888 (BDH-CQ).")

d.output(os.path.join(ROOT, "SUMMARY.pdf"))
print("SUMMARY.pdf written")

# ============================= BLOG.pdf =============================
b = Doc()
b.set_margins(15, 13, 15)
b.add_page()
b.set_font("ser", "B", 16); b.set_text_color(*DARK)
b.cell(0, 8, "The Network That Remembers With Its Wiring", new_x="LMARGIN", new_y="NEXT")
b.set_font("ser", "", 9.5); b.set_text_color(*GREY)
b.multi_cell(0, 4.8, "A walk through the interactive explainer \u2014 synaptic plasticity as short-term memory, and where it lives in Pathway's BDH and BDH-CQ.")
b.ln(2)

def sec(t): b.head(t, 11)
def para(t): b.para(t, 8.9, 4.5)

sec("Open with a memory that is already running")
para("Most explainers start with a blank canvas and a Run button. This one opens mid-thought: a 24\u00d724 grid of synapses is already flickering as pairs of "
     "sparse patterns arrive, roughly one per second. Each arrival is one association \u2014 a key pattern and a value pattern \u2014 and it strengthens synapses "
     "through a single Hebbian write. On the right, the system continuously probes itself with a recent key and holds the retrieval next to the truth. "
     "Early on the two are nearly identical. Nothing is being animated; your browser is doing the arithmetic as you watch.")
sec("One sentence to falsify")
para("The whole artifact serves a single falsifiable claim: a network can store recent associations in a fixed-size synaptic state through Hebbian writes \u2014 "
     "no gradient step, no per-token memory slot \u2014 but as associations accumulate, interference corrupts retrieval until recall fails. The design brief for "
     "this track asked for exactly this shape of sentence, one the learner can reproduce, test, or break. So section 2 hands you the controls to break it.")
sec("The write, up close")
para("Present a pair and watch the matrix: every synapse between an active key unit and an active value unit gains strength. The equation is one line, "
     "M \u2190 \u03bbM + vk\u1d40, and its consequence is one line too: after t writes, reading with cue q returns M q = \u03a3 \u03bb^(t\u2212i)(k\u1d62\u00b7q) v\u1d62 \u2014 a "
     "similarity-weighted sum of every stored value. That innocent \u03a3 is the whole story. At low load the matching term dwarfs the others; the section shows "
     "you the sum itemized, term by term, so you can watch the crosstalk terms grow as you load the memory.")
sec("The moment it breaks")
para("Drag the load slider past ten or twelve pairs and the exact-match curve folds: same 24\u00d724 state, same stored information (with \u03bb = 1 nothing was "
     "overwritten or deleted), yet over half of probes now retrieve the wrong value. The panel decomposes one failing retrieval into its target term plus "
     "crosstalk. This is the capacity limit of additive associative memories, the one named by Schlag, Irie and Schmidhuber for linear transformers: "
     "endlessly adding associations to finite memory inevitably reaches a limit. Superposition is not storage.")
sec("Three lenses on one equation")
para("The same live state renders as a neuron-synapse diagram (Hebb's rule, decay as forgetting), as fast weights (slow parameters frozen at inference, "
     "fast parameter-shaped memory rewritten by the input stream \u2014 the Schmidhuber-to-Titans lineage), and as linear attention (a matrix-state RNN: "
     "GLA, RWKV-7, DeltaNet, Mamba-2's SSD view). Three literatures, one update rule. The special case \u03bb = 1 \u2014 purely additive accumulation \u2014 is "
     "the case the Pathway brief highlights for BDH-CQ: state that accumulates additively per demonstration.")
sec("Where BDH lives in this")
para("For most topics a BDH section would be a stretch. Here it is the subject. The BDH paper states that inference-time working memory \u201centirely relies "
     "on synaptic plasticity with Hebbian learning using spiking neurons,\u201d and reports monosemantic synapses and sparse, positive activations. BDH-CQ "
     "extends the family to demonstration-driven reasoning: inputs continuously update recurrent memory, no parameter updates at inference, no verbalized "
     "chain of thought; a 150M configuration reports 29.5% pass@2 on public ARC-AGI-1 at about $0.0007 per task. The artifact's evidence table labels "
     "every one of these claims as developer-reported preprint results \u2014 and the toy itself is prominently labeled as an independent toy, not BDH.")
sec("Try it")
para("Work top to bottom, predict before you drag, and finish with the 60-second test: the system stores one pair, proves perfect recall, stores forty, "
     "and hands you the same probe again. If you can explain why the second probe fails \u2014 interference, not deletion \u2014 you have taken the concept. "
     "The sandbox exposes every variable: state size, decay, sparsity, key overlap, cue noise, seed. The repository ships a NumPy reference implementation "
     "and a test proving the JavaScript in your browser matches it to one part in a billion.")
para("Sources: arXiv 2509.26507 (BDH); 2608.09888 (BDH-CQ); 2102.11174 (linear transformers as fast-weight programmers); 2508.08435 (fast-weight programming "
     "primer, TMLR 2025); 2501.00663 (Titans); 2312.06635 (GLA); 2503.14456 (RWKV-7); 2407.04620 (TTT); 2405.21060 (Mamba-2/SSD).")

b.output(os.path.join(ROOT, "BLOG.pdf"))
print("BLOG.pdf written")
