# Citation ledger — every claim → its primary source → evidence level

Rule: no claim in the artifact or PDFs goes beyond what its source states. Where a
source is a preprint, we say so in the artifact itself (§4 evidence table).

Legend: **PR** = peer-reviewed · **DP** = developer-reported preprint (arXiv, not peer-reviewed at submission time) · **ORG** = organizer-provided framing (DataForge problem statement) · **LOCAL** = reproducible from this repository (fixed seeds + tests).

## A. Claims about BDH

| # | Claim (as worded in our materials) | Source | Level |
|---|---|---|---|
| A1 | "The working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons." | BDH abstract, arXiv:2509.26507 (verbatim) | DP |
| A2 | "Specific, individual synapses strengthen connection whenever BDH hears or reasons about a specific concept while processing language inputs" (monosemantic synapses). | BDH abstract (verbatim) | DP |
| A3 | "Activation vectors of BDH are sparse and positive. We demonstrate monosemanticity in BDH on language tasks." | BDH abstract (verbatim) | DP |
| A4 | BDH rivals GPT-2 performance on language and translation at 10M–1B parameters for the same training data (Transformer-like scaling). | BDH abstract (paraphrase) | DP |
| A5 | BDH's neuron interaction network has high modularity and heavy-tailed degree distribution. | BDH abstract (paraphrase) | DP |
| A6 | BDH is a scale-free network of locally interacting neuron particles; admits a GPU-friendly formulation. | BDH abstract (paraphrase) | DP |
| A7 | Do **not** classify BDH as an SSM in the Mamba sense; the GPU formulation is built from ReLU-low-rank transformations with linear attention. | DataForge PS, "SSMs" note | ORG (note: the BDH abstract itself calls BDH an "attention-based state space sequence learning architecture" — related wording, different claim; we do not conflate them) |
| A8 | BDH code is public: github.com/pathwaycom/bdh | arXiv page "Comments" field | DP |
| A9 | "Attention reformulated as synaptic memory that updates as the model reads"; "reasoning and memory share one computational fabric." | DataForge PS, BDH topic entry | ORG |

## B. Claims about BDH-CQ

| # | Claim | Source | Level |
|---|---|---|---|
| B1 | "Inputs presented at inference time continuously update the model's recurrent memory; the model then solves a query through iterative computation in a high-dimensional latent space, without verbalizing its intermediate reasoning." | BDH-CQ abstract, arXiv:2608.09888 (verbatim) | DP |
| B2 | A 150M-parameter configuration reaches 29.5% pass@2 on the public ARC-AGI-1 evaluation at a computed inference cost of $0.0007 per task; stated to break the previously reported ARC-AGI-1 cost–accuracy Pareto frontier. | BDH-CQ abstract (verbatim numbers) | DP |
| B3 | Evaluated with controlled ARC-like interventions to study what it learns from demonstrations. | BDH-CQ abstract (paraphrase) | DP |
| B4 | BDH-CQ's contextual memory relates to fast-weight/linear-attention views "with the special case where state accumulates additively per demonstration." | DataForge PS, Linear Attention note | ORG |
| B5 | No evaluation-task demonstrations in training and no parameter updates at inference for BDH-CQ; adaptation happens in recurrent state rather than weights (vs HRM/TRM optimization route). | DataForge PS, Test-Time Adaptation entry | ORG |
| B6 | External evaluators (Ł. Kaiser; R. Kinas; R. Zhong — latter two also paper co-authors) evaluated/reproduced ARC-AGI-1 results. | pathwaycom/arc-task-gen README (checked Sep 2026) | DP — evaluation by named externals, **not** an independent audit; we label it exactly this way |
| B7 | Pretraining experiments 1B→600B; SageMaker HyperPod integration. | DataForge PS | ORG + company announcement; we make **no accuracy claims** and say so |
| B8 | BDH-CQ checkpoint not public. | Explainer's own statement of scope; consistent with PS ("Teams are not expected to run an unavailable BDH or BDH-CQ checkpoint") | ORG |

## C. Claims about the concept (fast weights / linear attention / Hebbian memory)

| # | Claim | Source | Level |
|---|---|---|---|
| C1 | Linearised self-attention ≡ fast-weight controllers; "endlessly adding new associations to a memory of finite size inevitably will reach a limit" (capacity limit of additive memories). | Schlag, Irie, Schmidhuber, ICML 2021 (arXiv:2102.11174) | PR (ICML) |
| C2 | Fast weights = "synaptic weights [that] dynamically change over time as a function of input observations, and serve as short-term memory storage"; primer connecting FWPs, transformers, SSMs, neurobiology. | arXiv:2508.08435, accepted TMLR (2025) | PR (TMLR acceptance stated on arXiv page) |
| C3 | Titans: test-time memorization with surprise-based gating and decay; three-memory architecture. | arXiv:2501.00663 | DP |
| C4 | GLA: linear attention as RNN with matrix state; data-dependent gating; hardware-efficient training. | arXiv:2312.06635, ICML 2024 | PR (ICML) |
| C5 | RWKV-7: matrix-valued state with dynamic evolution. | arXiv:2503.14456 | DP |
| C6 | TTT layers: hidden state as a machine-learning model updated at test time. | arXiv:2407.04620 | DP (NeurIPS 2024 per authors; we cite as arXiv to be safe) |
| C7 | Mamba-2/SSD: duality between SSMs and variants of attention. | arXiv:2405.21060, ICML 2024 | PR (ICML) |
| C8 | Transformers' KV cache grows with sequence length (memory/context constraints). | Common technical background; also the framing of the PS "KV caching" topic | textbook / ORG |

## D. Claims about the toy/artifact itself (all LOCAL)

| # | Claim | Verification |
|---|---|---|
| D1 | Recall perfect at load 1 (cosine ≈ 1.0). | `test_recall_perfect_at_load_one` |
| D2 | Interference grows and exact-match recall collapses with load at λ=1, d=24. | `test_interference_grows_and_recall_degrades`; sandbox, any seed 1–99 |
| D3 | Closed form Mq = Σ λ^(t−i)(kᵢ·q)vᵢ equals the matrix read exactly. | `test_closed_form_identity` (atol 1e-12) |
| D4 | State shape never grows (50 writes, still d×d). | `test_state_shape_never_grows` |
| D5 | λ<1 gives geometric forgetting of the first pair. | `test_decay_forgets_first_pair` |
| D6 | The browser's JS core == NumPy reference on fixed seeds. | `test_js_core_matches_numpy` (tol 1e-9) |
| D7 | Crossing points are seed-dependent; collapse is robust. | Sandbox seed slider (empirical, 1–99) |

## Verified source URLs (all accessed while building this submission)

- https://arxiv.org/abs/2509.26507 (BDH; abstract quotes A1–A6 verified verbatim)
- https://arxiv.org/abs/2608.09888 (BDH-CQ; abstract quotes B1–B3 verified verbatim)
- https://github.com/pathwaycom/arc-task-gen (B6)
- https://arxiv.org/abs/2102.11174 (C1) · https://arxiv.org/abs/2508.08435 (C2) · https://arxiv.org/abs/2501.00663 (C3) · https://arxiv.org/abs/2312.06635 (C4) · https://arxiv.org/abs/2503.14456 (C5) · https://arxiv.org/abs/2407.04620 (C6) · https://arxiv.org/abs/2405.21060 (C7)
- Pathway essays (linked from the PS; used for context, not quoted as evidence): https://pathway.com/research/bdh-explainer/bdh-architecture-derivation , https://pathway.com/research/the-equations-of-reasoning
