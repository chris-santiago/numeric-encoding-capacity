# Speaker companion — "When does encoding a numeric feature help?"

One section per slide, numbered in lockstep with `encoding-capacity-deck.md`. Every number traces to `ConsolidatedFlow/1784135301155957` (8 seeds, K=6 unless noted); full write-up in `../REPORT.md`.

---

## Slide 1 — When does encoding a numeric feature help?

Open with the frame, not the result. This is a talk about a single, very practical question — when is it worth encoding a numeric feature richly — and about a story where we got the answer wrong the first time and had to correct ourselves. Say that out loud: the headline is a refutation of our own earlier conclusion. That earns trust and sets up the arc.

The one-sentence thesis on the slide is the whole talk in compressed form: it's the *shape* of the target, not the *type* of the model, that decides. Don't unpack it yet — just plant it and promise to earn it.

**Details not on the slide**

- "Localization account" is the name we'll give the corrected mechanism; the thing it replaces we call the "architecture law." Both get defined later — don't front-load the jargon.
- Everything here is synthetic. Flag that once now so nobody spends the talk wondering; you'll return to it on the honesty slide.

---

## Slide 2 — The practitioner's question

Ground it in something concrete: a fraud model with a transaction amount and a time-since-last-event. Everyone already `log`-transforms these. The live question is whether to go further — PLE, a learned embedding, a periodic feature — each of which costs real engineering time and adds parameters and overfitting surface.

The honest tension is that sometimes these lift the model a lot and sometimes they're pure ceremony, and practitioners mostly decide by folklore. The promise of this study is a *rule you can apply before building* — a way to tell which case you're in.

**Details not on the slide**

- The blockquote is a deliberate spoiler of the conclusion. Read it slowly; it's the takeaway the rest of the deck defends.
- "Affine-read" in the blockquote is undefined here on purpose — slide 4 defines it. If someone asks, say "a model that reads the feature as a weighted sum before a fixed nonlinearity, like a GRU" and move on.

---

## Slide 3 — Five encoders of one feature

This slide is vocabulary. Walk left to right. `raw` and `log` are plain data transforms, dimension 1 — `log` is our reference throughout, the thing every other encoder has to beat. `ple` (piecewise-linear encoding) is a *fixed* basis: 8 quantile bins on the log coordinate, decided before training. `projection` and `dense` are *learned* expansions that live inside the model.

The one structural point to land: PLE and projection are **dimension-matched at 8 on purpose**. That's what makes "fixed vs learned" a fair fight later — same capacity, the only difference is whether the nonlinearity is handed over or has to be discovered by SGD.

**Details not on the slide**

- `projection` = a per-feature `Linear(1→8)→ReLU`; `dense` = a single joint `Linear→ReLU` over all features at once (so it can mix features; projection can't until the recurrence). We mostly report `projection` because it's the dimension-matched arm; `dense` tracks it and acts as corroboration.
- Source is `flow.py` `encode()` and the `_GRU` embedding modes if anyone wants to read it.

---

## Slide 4 — Two ways a model reads that feature

The distinction on this slide was the heart of our first (wrong) answer, so give it weight. An **affine-read** model — a GRU cell, a static logistic head — only ever sees the encoding as `W · e(x)`, a weighted sum, before a *fixed* sigmoid/tanh. Its per-step function class is therefore exactly the span of the encoding: it can only form shapes the basis already contains. A **free-nonlinearity** model — a small per-step MLP — can bend the raw scalar into any shape at all.

The orange line is the hinge of the entire talk: "can represent" is not "can find." Universal approximation says the MLP *can* represent any shape; it says nothing about whether SGD will *locate* it. Foreshadow that this gap is where our first answer broke.

**Details not on the slide**

- Why a GRU is "affine-read": the per-step input enters the gates linearly; the only nonlinearities (sigmoid/tanh) are fixed and come *after* the weighted sum. So the input's own transform has to come from the encoding.
- The MLP here is 2-layer, width 64, per-step, recency-pooled — genuinely enough capacity, which matters for the "not undercapacity" argument on slide 9.

---

## Slide 5 — What we believed going in: the "architecture law"

State the earlier conclusion plainly and make it sound reasonable — because it was. Two controlled experiments (cycles 5–6 of this investigation line) concluded a basis helps *iff the model lacks a free per-feature nonlinearity*. The logic is clean and on the slide: affine-read models are bounded by the span of the encoding, so a richer basis buys expressiveness; an MLP can already rebuild any shape, so a basis buys nothing.

Then deliver the twist at the bottom: those studies only ever put a **smooth** target in front of the MLP. They never tested a sharp one. That single omission is the crack the rest of the talk drives through.

**Details not on the slide**

- This is genuinely how we reported it at the time; `../../encoding_capacity_synthesis/REPORT.md` is the superseded write-up. Owning the prior claim is the point — it's a self-refutation, which is more credible than a straw man.
- If asked "why didn't you test sharp then?": the earlier Δt target was constructed smooth (a U-shape); nobody thought to vary sharpness because the architecture framing didn't predict it would matter.

---

## Slide 6 — The experiment that settles it

The design is the credibility of the whole talk, so slow down. One Metaflow flow crosses **3 architectures × 5 risk shapes × 2 multiplicities**, 8 seeds each. The key discipline: the *signal is held fixed across architectures*, so what varies between arms is the model and the shape, not the task. Metric is PR-AUC (precision–recall AUC — the right metric for rare fraud). The **oracle** ranks examples by the true label log-odds: a per-run ceiling that no encoder can beat, so you can read how much headroom each arm leaves.

Point at the controls. Two positive controls hard-halt the run if PLE *fails* to detect the sharp band (in both the static path and the trained GRU) — so a null result can't be a broken pipeline. The negative control is `log_linear`, where no arm should beat `log`.

**Details not on the slide**

- Every arm is temperature-calibrated on validation before magnitude metrics, so PR-AUC reflects representation, not calibration luck.
- Honesty note you can pre-empt here: the cross-architecture contrast is *not* a single-variable manipulation (MLP and GRU differ in more than one way). The clean evidence is within-model, which is why slides 8 and 10 lean on the MLP-holds-architecture-fixed result and the GRU's own projection arm.

---

## Slide 7 — The five risk shapes

Define "risk shape" concretely: it's how the fraud log-odds bends as the feature's standardized-log value moves. Walk the five panels. Log-linear, monotone-curved (s³), and smooth non-monotone (s²) are all learnable by a smooth function of the scalar. The two **sharp** panels are the stars: a Gaussian band, σ ≈ 0.15, that concentrates risk in a razor-thin slice of the value axis — mode-centered and offset variants.

The colors under each panel pre-announce the result (blue = log wins, green = projection, red = PLE) so the audience has a scorecard before the data slides. The one idea to hold: sharpness is the single property we vary, and it's the property the old architecture framing never considered.

**Details not on the slide**

- These are the actual DGP risk functions from `make_data` (σ = 0.15), so they're exact, not illustrative.
- A real-world reading of "sharp Δt": very short gaps look like automated card-testing, very long gaps look like dormant-account reactivation — both high-risk, a genuinely localized band rather than a monotone trend. That's the intuition for why this shape isn't a synthetic contrivance.

---

## Slide 8 — Result 1 — the refutation

This is the money slide. Read the bars left to right against the old law's prediction. Log-linear: PLE ties `log` (redundant — predicted). Curved: near-tie. Smooth: PLE actually *hurts* (0.38 → 0.31) because a full fixed basis is dead weight when the MLP can form the shape itself — also consistent with the old law. So far the architecture law is winning.

Then the sharp band breaks it: `mlp_log` collapses to 0.13, and `mlp_ple` jumps to 0.52 — a +0.39 raw gain, right up near the 0.61 oracle. A free per-step nonlinearity did *not* make encoding redundant. Land the line: the old law held on four shapes and shattered on the one it never tested.

**Details not on the slide**

- Exact numbers (K=6, 8-seed mean): log-linear 0.444 vs 0.445; curved 0.308 vs 0.398; smooth 0.382 vs 0.306; sharp-mode 0.132 vs 0.518 (Holm-significant); sharp-off 0.171 vs 0.213.
- The smooth "PLE hurts" result is worth dwelling on if the audience is skeptical — it shows the MLP genuinely doesn't need a basis on learnable shapes, which makes the sharp result a property of the *target*, not a blanket "PLE is good."

---

## Slide 9 — Why the MLP fails on sharp: represent ≠ find

Explain the mechanism, because the bar chart alone invites "the MLP was too small." Head that off: the MLP trains fine on every other shape, so it is *not* undercapacity. Universal approximation guarantees it can *represent* a sharp bump. The failure is **optimization** — SGD cannot *locate* a σ ≈ 0.15 spike from a bare scalar, because almost everywhere the gradient toward that needle is flat.

A fixed quantile basis sidesteps the search entirely: the bins already partition the value axis, so the localization the MLP couldn't find is simply handed to it. That's the whole refutation in one idea — localization, not architecture, is what the basis relieves. The old law was just the special case where the target was smooth enough to find by gradient descent.

**Details not on the slide**

- *What "localization" means (worth defining out loud — it's the load-bearing word).* It's a property of the **target**, not the model: the label signal is confined to a **narrow, specific slice of the feature's value range** rather than spread smoothly across it. Smooth/global = "bigger value → gradually more risk," every value carries a little signal. Localized = "risk spikes only when Δt is right around 3 minutes, nothing elsewhere" — a spotlight on one spot. Formally it needs both an interior peak (non-monotone, so there's a specific *where*) and a narrow one (small σ, so the *where* is a precise spot, not a broad hump). Now the punchline connects: a PLE basis literally *is* localization — its quantile bins are pre-placed local windows on the value axis, so it hands the model "which specific slice is this?" ready-made. A sharp target demands the *where*; from a bare scalar the model has to manufacture that window itself (position opposing nonlinearities at the band's edges); PLE supplies the slices, so it stops hunting. That's the sense of "localization, not architecture, is what a basis relieves."
- If pushed on "did you try harder optimization?" — the run uses real capacity, a 120-epoch cap with validation early-stopping, gradient clipping, best-state restore. This isn't a tuning artifact; it's the geometry of a localized target.
- The deeper claim: representability and learnability come apart, and feature encoding is a lever on the *learnability* side. That reframes PLE as an optimization aid, not just an expressiveness aid.
- *Why traditional GBMs don't hit this (a natural audience question).* The pathology is specific to *how a model consumes a scalar*. A gradient-boosted tree consumes a numeric feature by **searching over split thresholds**, each candidate scored directly by loss reduction — not by gradient descent on a transform. To isolate a sharp band it just needs two splits (x > a, x < b) carving out a leaf, and it finds those edges by exhaustive/histogram search, which needs no slope pointing at the needle. So a tree is effectively an *adaptive quantile basis whose knots are chosen by direct search* — the very thing PLE hands a neural net for free, which is why a tree localizes by construction. Two corollaries: (i) trees are invariant to monotone rescales, so `raw` / `log` / single-feature PLE are near no-ops for them — matching cycle 2's `GBDT` +0.008 next to the neural surrogate's large gain; (ii) the residual worry for a tree isn't localization but *resolution* — a band narrower than the histogram grid, or a high-order cross-feature conjunction with few positives, can still be missed, though far more robustly than SGD.

---

## Slide 10 — Result 2 — the encoder crosses over by shape

Now the affine-read GRU, where the interesting question isn't *whether* a basis helps but *which* one. The bars show a clean crossover. On the sharp band, fixed PLE towers (0.43) while every other arm — including the learned projection (0.18) — collapses near the floor. On smooth and curved, the ranking flips: the learned projection leads (0.47 on smooth) and PLE's gain isn't even significant.

The takeaway line: the winning encoder is a function of the target's shape, not a fixed choice. Sharp → fixed PLE; smooth/curved → learned projection; log-linear → just `log`.

**Details not on the slide**

- Raw gaps over `log` (K=6): sharp-mode PLE +0.35 vs projection +0.10; smooth projection +0.19 vs PLE +0.03 (n.s.); curved projection +0.10 vs PLE +0.02 (n.s.). "n.s." = confidence interval includes zero.
- Conditioning is a separate axis that persists underneath (covered next slide): on log-linear, `log` beats `raw` by +0.16 purely from taming a heavy tail.

---

## Slide 11 — Why fixed beats learned on sharp

This slide answers the natural objection: if the projection is learned and dimension-matched, why doesn't it just learn the sharp shape too? Because it places its ReLU knots *by SGD* — the exact optimization that couldn't find a sharp spike from a scalar on slide 9. So the learned projection inherits the same failure. PLE's knots are quantiles fixed before training, so on a sharp target it hands the localization over for free.

The flip side explains the crossover: on a *smooth* target that fixed grid is just a dimensionality tax, and the flexible projection — which can find smooth shapes fine — avoids it. Close by separating conditioning as its own axis: `log` over `raw` buys +0.16 on log-linear regardless of shape, because a heavy-tailed raw value is badly conditioned going into a recurrence.

**Details not on the slide**

- This is the cleanest *within-model* evidence in the whole study: same GRU, same dimension, only fixed-vs-learned knots differ, and the winner flips with sharpness. No cross-architecture confound.
- `dense` (the joint learned expansion) behaves like `projection` here, reinforcing that it's the learned-vs-fixed distinction doing the work, not the per-feature-vs-joint one.

---

## Slide 12 — The whole account in one picture

This 2×2 is the synthesis; let it sit on screen while you talk. Two obstacles, each of which a basis can relieve. **Obstacle 1 (can't form)** hits affine-read models on any non-log-linear shape — the span-of-the-encoding limit. **Obstacle 2 (can't find)** hits *any* model on a localized target — the SGD-search limit.

Read the cells. Affine-read × smooth/curved: Obstacle 1 only → a learned projection suffices. Affine-read × sharp: both obstacles → fixed PLE. Free-nonlinearity × smooth/curved: neither → redundant/harmful. Free-nonlinearity × sharp: Obstacle 2 only → fixed PLE, +0.39. The old architecture law is exactly the bottom-left cell mistaken for the whole table.

**Details not on the slide**

- The punchline at the bottom: sharp non-monotonicity trips *both* obstacles, which is why it's the largest and most universal lever — the only shape that helps even the free-nonlinearity MLP.
- If you only have time for one slide, it's this one plus slide 8.

---

## Slide 13 — Result 3 — a caution on the effect size

Shift register: this is a methodological warning, not a new result. The flow's decision estimand is a *deficit-corrected lift*: the arm-minus-log gap on a condition, minus the same gap on `log_linear`. Algebraically that's `raw_gap − deficit`. Define the pieces slowly — `raw_gap` is the plain deployment gap you'd actually ship; `deficit` is the encoder's fixed tax, measured where `log` is already adequate so any gap there is pure overhead.

The trap is in the arithmetic: the deficit is usually negative, and subtracting a negative *adds it back*. So a badly-conditioned arm can post a big, even significant, corrected lift that is mostly add-back. Set that up here; the next slide shows it biting.

**Details not on the slide**

- Netting the fixed tax is structurally the *right* thing to do for a fair cross-encoder comparison — the estimand isn't wrong, it's just not an effect size.
- The one case where the correction earns its keep (worth mentioning): a real mechanism *masked* by a removable tax shows raw_gap ≈ 0 but dc_lift > 0. That's exactly the earlier curvature-in-a-GRU cycle. But you recover the same insight from the (raw_gap, deficit) pair, so anchor on raw_gap.

---

## Slide 14 — Read the deployment gap, not the estimand

Make the abstract warning concrete with the `raw · sharp` row (highlighted red). Its deficit-corrected `dc_lift` is +0.16 and Holm-significant — it *looks* like a real lever. But its `raw_gap` is +0.00: raw ties `log` on the sharp condition, zero deployment value. The whole +0.16 is add-back from raw being catastrophically bad on `log_linear`.

Contrast with the trustworthy rows: `ple · sharp` (+0.35) and `projection · smooth` (+0.19) are large on *both* the corrected and the raw quantity. Rule to take home: read magnitude from `raw_gap`, use `dc_lift` only to understand structure, and never quote the corrected number as an effect size.

**Details not on the slide**

- Green CI = raw_gap excludes zero (real); grey = not significant; blue point = dc_lift. The horizontal gap between the green point and the blue point *is* the deficit add-back, drawn to scale.
- This slide is why the flow now emits `raw_gap` and `deficit` beside every `dc_lift` — it was added after peer review precisely so a weak arm couldn't hide behind a significant corrected number.

---

## Slide 15 — Honesty slide — what this does *not* establish

Deliver this at face value, no hedging about the hedging. Five caveats: the mechanism is proposed, not proven (one flow); the cross-architecture contrast is confounded, so the clean evidence is within-model; everything decisive is synthetic, so magnitudes are direction-only; the sharp result depends on a constructed band whose real-world analog is unverified; and PLE is training-sensitive.

The most important one to voice: the real-data cycles in this same investigation line actually *refuted* the naive amount-encoding story, and a real A/B here failed its precondition. So this study isolates a mechanism; it does not forecast a real-fraud lift. Say that clearly — it's the difference between an honest talk and an oversold one.

**Details not on the slide**

- "Confounded" specifics: MLP vs GRU differ in recurrence, pooling, and capacity, not just the read-mode. The two within-model anchors (MLP sharp at fixed architecture; GRU fixed-vs-learned knots) are what carry the weight.
- "Training-sensitive": an early under-resourced run gave a spurious *negative* PLE lift from feeding many correlated bins into a recurrence. The reported run fixed that with real capacity and regularization — worth mentioning if anyone asks how robust the PLE numbers are.

---

## Slide 16 — What to actually do

Convert the mechanism into a decision rule. The table is the deliverable: encode by the *shape of the feature's risk-in-context*, not by whether it's curved or linear. Sharp / localized non-monotone → fixed PLE, and it helps any model. Smooth / curved and an affine-read model → learned projection (SGD-learnable, smaller tax). Monotone / log-adequate → just `log`; a basis only imports the deficit.

The one counterintuitive rider: a free per-step nonlinearity does *not* exempt a sharp feature — PLE still helps *after* you've added a projection, because the obstacle is optimization, not expressiveness. Then the validation protocol: production A/B over `log` / `ple` / `projection`, seed-level CI-excludes-zero bar, adequate capacity for the PLE arm, and treat the synthetic magnitudes as direction-only.

**Details not on the slide**

- Concrete first candidate for a real A/B: Δt in a fraud sequence GRU. Its risk plausibly has the localized-band shape (card-testing vs dormancy), and it's exactly the case the study predicts PLE should win.
- Leave amount on `log` unless you have evidence its in-context risk is sharp — the study repeatedly found amount's risk is monotone/log-adequate, where a basis is pure cost.

---

## Slide 17 — One line to remember

Close on the compression. A basis helps when the model cannot **form** the shape (the affine-read obstacle) *or* cannot **find** it by SGD (the localized-target obstacle). Two obstacles, one account, and it subsumes the architecture law we started with as a single special case.

If you want a parting practical hook: the next time someone asks "should we PLE this feature?", the answer isn't about the model — it's "is the feature's risk sharp?" Point them to `../REPORT.md` and the run for the full evidence.

**Details not on the slide**

- The full investigation is ten directories (eight cycles plus two syntheses); this deck is only the final consolidated study. The README's "How we got here" table is the map if anyone wants the backstory.
- Run of record: `ConsolidatedFlow/1784135301155957`, 8 seeds, dual positive-controlled, every arm temperature-calibrated.
