# Project 6 — End-to-End MLOps Pipeline: Research Brief

**Date:** 2026-07-28
**Status:** Research (pre-brainstorming)
**Purpose:** Grounding document for the brainstorming pass that decides the
scope, model choice, and integration story for Project 6.

This is not a design. It is a survey of the design space, the canonical
literature, the OSS landscape, and the honest constraints (no GPU, ~6 weeks,
solo, must integrate with the user's own inference server, vector DB, and task
queue).

---

## 1. What "end-to-end MLOps pipeline" actually means

The phrase is dangerously ambiguous — it covers everything from a Jupyter
notebook with `mlflow.log_metric` calls to a multi-team platform with feature
stores, drift monitors, and automated retraining. To avoid the "I plugged
MLflow into a notebook" trap (see pitfalls, section 10), the scope has to be
picked deliberately.

Four tiers, cheapest to most ambitious:

**Tier 0 — Notebook + Registry (1–2 weeks).** Train a model in a notebook,
log runs to MLflow, save the artefact. This is what most fresher "MLOps"
portfolio projects look like. It is not a pipeline. Resume claim available:
"used MLflow." Weak. Not the target.

**Tier 1 — Minimum viable pipeline (3–4 weeks).** Deterministic, reproducible
DAG: data ingest → validate → train → evaluate on held-out set → package →
serve behind an HTTP endpoint → basic request logging + latency metric. All
steps scripted, all invocations reproducible from a single config. This is
what most tutorials mean by "end-to-end." Defensible resume claim:
"reproducible pipeline that trains and serves a model with automated eval and
basic observability."

**Tier 2 — Middle path (5–6 weeks; recommended target).** Tier 1 plus:
- Experiment tracking with lineage (data version → run → model version)
- Model registry with promotion states (staging → production)
- Versioned datasets (DVC-style or custom content-addressed store)
- CI/CD gate: a PR that changes training code triggers a training run, and
  the promotion check requires the new model to beat the current production
  model on a frozen holdout (a "champion vs challenger" test)
- Offline eval report auto-generated per run
- Container-based deploy (docker-compose orchestrated)

This is the tier where the project earns "flagship-depth" on a solo timeline.
Resume claim: "reproducible training-to-serving pipeline with CI-gated model
promotion, dataset versioning, experiment lineage, and rollback."

**Tier 3 — Ambitious (7–9+ weeks; likely overrun).** Tier 2 plus:
- Online eval: A/B or interleaved traffic split
- Drift detection on inputs and predictions (population stability index,
  KL/JS divergence, or embedding-space distance)
- Shadow deploys (new model receives production traffic but responses are
  discarded and only logged)
- A feature store abstraction (feature definitions decoupled from training
  and serving code, offline/online consistency)
- Auto-retraining triggered by a drift threshold

Tier 3 is what real production MLOps teams do. Doing all of it in six weeks
solo is a stretch; doing a **credible slice** of it (say: drift detection +
shadow deploy + auto-retrain trigger, skip the feature store abstraction) is
plausible and is exactly the demo that sells the integration story.

**Recommendation for brainstorming:** aim at Tier 2 as the definition-of-done
floor, with one or two Tier-3 features (drift-triggered retrain, shadow
deploy) as the flagship differentiator.

---

## 2. Canonical references

Read (or re-skim) before the brainstorming session. All are freely available
except the two books.

- **Zinkevich, "Rules of Machine Learning: Best Practices for ML
  Engineering"** — Google, ~43 rules organised by phase (before ML, first
  pipeline, launch, growth, slowdown). Practical, opinionated, still the
  best single document. https://developers.google.com/machine-learning/guides/rules-of-ml
- **Sculley et al., "Hidden Technical Debt in Machine Learning Systems"**
  (NeurIPS 2015). Names the anti-patterns: glue code, pipeline jungles,
  dead experimental code paths, undeclared consumers, correction cascades,
  feedback loops. The one page of the paper worth memorising is the diagram
  showing the ML code as a tiny box surrounded by data collection,
  configuration, serving infra, monitoring, etc.
  https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
- **Sato, Wider, Windheuser, "Continuous Delivery for Machine Learning"**
  (Martin Fowler's site / ThoughtWorks, 2019). Introduces CD4ML: the three
  axes of change (code, model, data) and how CI/CD extends to cover model
  promotion. https://martinfowler.com/articles/cd4ml.html
- **Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022).** The
  chapters on data engineering, feature engineering, model deployment, and
  monitoring are directly on-topic. Chip Huyen's framing of "batch vs
  online prediction, batch vs streaming features" is the mental model to
  use when scoping serving.
- **Mohandas, *Made With ML* (madewithml.com).** Free, chapter 3–6 covers
  a hands-on MLOps stack (uses Ray, MLflow, DVC, Great Expectations).
  Study the *sequence* rather than adopting the exact stack.
  https://madewithml.com/
- **Lakshmanan, Robinson, Munn, *Machine Learning Design Patterns* (O'Reilly,
  2020).** Reference-book style. Patterns like "Model Versioning,"
  "Continued Model Evaluation," "Two-Phase Predictions," "Workflow Pipeline"
  are directly useful vocabulary for the write-up.
- **Breck et al., "The ML Test Score: A Rubric for ML Production Readiness
  and Technical Debt Reduction"** (Google, 2017). 28-point checklist across
  data, model, infra, and monitoring. Use this as the "did I actually build
  a real system?" self-audit at the end.
  https://research.google/pubs/pub46555/

Honest note: some of these predate the current tooling generation. The
principles hold; the tool recommendations don't. Read for concepts, not
stack.

---

## 3. Similar OSS tools/frameworks to study (not use — study)

The point is to understand what problem each solves and steal its API
vocabulary, not to depend on it. If Project 6 is "I glued together five OSS
tools," it isn't flagship — it's a config file. Study to make deliberate
build-vs-adopt decisions.

**Experiment tracking / registry**
- **MLflow.** De facto standard. Four components: Tracking (runs/params/
  metrics/artefacts), Projects (packaged runs), Models (registry with
  stages), Model Serving. Study the tracking API and the model registry's
  state machine (None → Staging → Production → Archived). Its API is
  imitable in ~500 lines if you skip the UI.
- **Weights & Biases.** Hosted; excellent UI, sweeps, artefact lineage.
  Study the "everything is a run" data model and how they treat artefacts
  as first-class versioned objects. Free tier is generous but sends
  telemetry to their servers — bad match for a "self-hosted portfolio"
  story.
- **ClearML.** Self-hostable MLflow competitor with orchestration built in.
  Study how it wires experiments to pipelines.

**Orchestration**
- **Airflow.** DAGs as Python, cron-triggered, scheduler + workers.
  Battle-tested for data engineering. Heavy for ML use cases. Study its
  task/DAG separation and its retry/backfill semantics.
- **Prefect.** Airflow's modern cousin. Python-native flow definitions,
  first-class handling of dynamic DAGs, better local dev. Study how
  "flows" and "tasks" decorate plain Python.
- **Metaflow (Netflix).** ML-first from day one. Steps decorated on a
  class, artefacts auto-versioned, "cards" for artefact UI. Study the
  `@step` model and how they treat artefacts as automatic outputs of
  each step.
- **Kedro (QuantumBlack).** Opinionated project structure + data catalog.
  Study the DataCatalog abstraction — decouples "which dataset" from
  "how to load it."
- **Kubeflow Pipelines.** Kubernetes-native, containerised steps.
  Overkill for solo, but its component / DAG separation and the
  compiled pipeline spec (KFP DSL → IR) is instructive.
- **ZenML.** Newer, framework-agnostic, tries to be the "sklearn of
  pipelines." Study the stack abstraction (orchestrator + artefact store
  + tracker are pluggable).
- **DVC + `dvc.yaml` pipelines.** Not just for data versioning — its
  stage definitions are a lightweight DAG. Worth reading.

**Serving**
- **BentoML.** "Bentos" package model + code + env. Nice API for wrapping
  a model into a service. Study the runner abstraction.
- **Seldon Core.** K8s-native. Advanced traffic routing (canary, shadow,
  A/B). Study its inference graph — chain models with a DAG at serving
  time.
- **KServe (formerly KFServing).** K8s-native inference; predictors,
  transformers, explainers as separate components. Study the separation
  of preprocessing from prediction.

**Feature store**
- **Feast.** Study the offline/online store split, the concept of
  feature views, and point-in-time correctness. Skip the deployment
  complexity.

**Data / drift**
- **Great Expectations.** Declarative data validation ("expectations"
  attached to columns). Study the expectation vocabulary — steal the
  best 20.
- **Evidently AI.** Batch drift/quality reports. Study which specific
  statistics they compute per column type (numeric: Wasserstein/KS;
  categorical: chi-square/PSI; text: embedding drift).
- **NannyML / whylogs.** Alternatives. Skim for coverage of concepts you
  might otherwise miss (e.g., performance estimation without ground
  truth).

The pattern to notice: every mature MLOps tool eventually reinvents the
same primitives — a versioned artefact store, a run/experiment concept, a
DAG runner, and a promotion state machine. Building small, correct versions
of those primitives in Project 6 is the flagship differentiator.

---

## 4. The integration story — using the user's own components

This is the reason Project 6 is scheduled last. Portfolio-level it is the
capstone. Concretely, three integrations to design:

**4a. Task queue (Project 5) as the pipeline orchestrator.**
Pipeline steps (ingest, validate, train, eval, package, deploy) become jobs
submitted to the queue. The task queue must support (from Project 5's spec —
verify at brainstorming):
- Task-graph submission or, at minimum, jobs that can enqueue successor
  jobs (fan-out/fan-in patterns for eval + shadow deploy)
- Retries with backoff for flaky steps (e.g., dataset download)
- Idempotency keys so a retried training job doesn't double-log
- A way to poll completion and read results (for a CLI to trigger a full
  run and wait)

If Project 5 lacks explicit DAG semantics, Project 6 can build a thin
"pipeline" layer above it: a Python object describes step dependencies,
compiles to a sequence of task submissions with explicit `depends_on`
edges. This is exactly what Prefect and Airflow do above their execution
substrate — the pattern is well-precedented.

Failure mode to design against: cascading retries. A failed data-ingest job
should not cause 30 retries to re-download 5 GB. Use max-retries with
dead-letter, and cache expensive step outputs by input hash.

**4b. Inference server (Project 2) as the model server.**
The pipeline's terminal step publishes a model to the inference server. To
work, both sides need to agree on:
- Model format (ONNX? TorchScript? raw HF Transformers checkpoint?)
- Load protocol (does the inference server read from a directory it watches,
  or does the pipeline POST a model to a management endpoint?)
- Versioning (multiple models loaded; canary/shadow routing; blue-green swap)
- Health/readiness reporting back to the pipeline

The inference server in Project 2 is CPU-focused with batching and KV cache
— that constrains the model choice to something the server can actually run
(see section 5). The management API is worth pre-agreeing when the
brainstorming for Project 2 happens; if it doesn't exist yet, Project 6 may
need to contribute back to Project 2's design.

**4c. Vector DB (Project 4) for embeddings and drift monitoring.**
Two distinct uses:
- **Retrieval-augmented use case (optional).** If the modelling task
  involves retrieval (e.g., a small RAG demo, or embedding-based
  classification), the vector DB is the retrieval backend. Training embeds
  and indexes; serving queries.
- **Drift monitoring (interesting).** Store a rolling window of production
  input embeddings. Compare distribution to the training distribution
  (nearest-neighbour distance, or centroid drift in embedding space). This
  is a plausibly novel monitoring signal that maps naturally to a vector DB
  primitive and would be hard to demonstrate without one.

**Deployment topology.** All three components + the pipeline coordinator in
one `docker-compose.yml`. Networks and ports are the interface. If any of
Projects 2/4/5 don't expose a stable protocol yet, that's a real blocker to
surface at brainstorming — not something to bulldoze past.

**Honest risk:** the integration story is compelling *if* Projects 2/4/5
have stable public APIs by the time Project 6 starts. If they don't, plan
some buffer for API stabilisation on the earlier projects, or accept that
Project 6 will contribute API changes back to them.

---

## 5. Model / task choice on CPU

Non-negotiable constraint: no GPU. This eliminates full-scale LLM fine-tunes
and most from-scratch deep learning on non-trivial datasets. The choice
affects everything downstream (dataset shape, eval metrics, drift signals).

**Option A — Text classification with DistilBERT / BERT-mini (recommended).**
Task: sentiment (SST-2), topic (AG News), or intent classification (small
custom dataset). DistilBERT-base is 66M params; fine-tuning on a small
labelled set (10–50k examples) with sequence length 128 runs in ~1–3 hours
per epoch on a decent CPU (16 cores, no AVX-512 slowdown). Two to five
epochs total. Realistic. Inference is fast enough to actually serve —
single-digit or low-tens-of-ms latency for short inputs.
BERT-mini (11M params) is faster still if DistilBERT is too slow.

**Option B — LoRA fine-tune of TinyLlama-1.1B or Phi-2 (2.7B).**
LoRA adapters on a small instruction dataset (a few thousand examples).
Even with LoRA, forward passes on 1B–3B models on CPU are minutes per
sample — training on any meaningful dataset takes days, not hours.
Do-able if the dataset is small (< 1k examples), the sequence length is
capped hard (< 512), and the goal is "demonstrate LoRA works" rather than
"produce a useful adapter." Serving latency is also multi-second per token
on CPU, which makes online demos awkward.
Verdict: possible flex, but poor fit for a pipeline story that requires
frequent retraining.

**Option C — Tabular / classical ML.**
Gradient-boosted trees on a benchmark tabular dataset (credit fraud, house
prices, click prediction). Fast to train, fast to serve, boring on a resume
targeting ML-infra roles. Fine choice if the point is really the pipeline
plumbing — the ML task is just a workload. But it doesn't showcase the
inference server (Project 2), which is LLM-shaped.

**Option D — Embedding-model fine-tuning (sentence-transformers).**
Fine-tune a small sentence embedding model (all-MiniLM-L6-v2, 22M params)
with contrastive loss on a domain dataset. Training on CPU is tolerable
(hours, not days). This is a **strong** fit because it naturally uses
the vector DB (Project 4) — the trained model is the encoder for the
vector index. Retrieval quality (Recall@k, MRR) is a clean eval metric
that's honest to report.

**Recommendation.** Two-model story: **DistilBERT classifier as the primary
model** (headline demo — clean latency, clean fine-tune loop, easy to
explain), **sentence-transformer embedding model as the secondary** (drives
vector DB integration; drift monitoring uses its embeddings).

Both live in the same pipeline: the classifier is the "product model," the
encoder produces the embeddings the drift monitor watches.

---

## 6. Key design decisions for brainstorming

For each: options, tradeoff, default lean. The brainstorming session will
overturn defaults freely — this section is priming, not conclusion.

**Orchestration.**
Options: (a) own task queue from Project 5, (b) Prefect, (c) Airflow.
Tradeoff: (a) proves the integration story but the queue may lack DAG
features. (b/c) are hardened but reduce the flagship claim to "used
Prefect." **Default: (a).** Any missing feature is either built in Project
6 as a thin layer or fixed in Project 5. Explicitly do *not* import an
orchestrator on top of the task queue — that dilutes the story.

**Experiment tracking.**
Options: (a) MLflow local, (b) custom lightweight tracker (SQLite + JSON
artefact store), (c) W&B free tier.
Tradeoff: MLflow is what everyone uses; a custom tracker is more
educational and matches the "build the primitives" ethos. W&B leaks the
"self-hosted portfolio" story.
**Default: (b) custom.** Small (200–500 LOC) — a runs table with
params/metrics, an artefact store keyed by content hash, a CLI to list
runs. Mention MLflow in the writeup as prior art.

**Model registry.**
Options: (a) MLflow registry, (b) custom, (c) HuggingFace Hub.
Tradeoff: custom fits the ethos. HF Hub gives free public hosting for
the demo but ties to their API.
**Default: (b) custom, backed by the same artefact store as tracking.**
State machine (staging/production/archived) is trivial; the promotion gate
(is the challenger better?) is where the design work is.

**Serving.**
Options: (a) own inference server (Project 2), (b) FastAPI + transformers
pipeline, (c) BentoML.
**Default: (a).** This is the whole point of Project 6 shipping last.

**Monitoring.**
Options: (a) Prometheus + Grafana (metrics: latency, throughput, error rate,
prediction distribution), (b) Evidently for offline drift reports, (c)
custom drift check that runs as a scheduled pipeline job and writes to the
tracker.
**Default: (a) for infra metrics + (c) for drift.** Skip Evidently as a
runtime dep; borrow the vocabulary (PSI for categorical, Wasserstein for
numeric, cosine drift in embedding space).

**CI/CD.**
GitHub Actions. On PR: run tests + ruff + mypy + a smoke pipeline that
trains on a tiny sample and asserts the pipeline runs end-to-end. On merge
to main: nothing automatic that touches production (portfolio project, no
prod). The model-promotion gate is a manual `promote` command that runs the
champion-vs-challenger eval and, on pass, tags the challenger as production.
Don't fake auto-deploy for a portfolio project — auto-deploy without a
manual gate is a real-system pattern, not a demo pattern.

**Infra.**
docker-compose. Kubernetes is overkill and adds a week of yak-shaving.
State the tradeoff in the writeup: "chosen for solo-dev velocity; k8s
migration path is one deployment abstraction away."

---

## 7. Reproducibility strategy

**Data versioning.**
Options: (a) DVC, (b) custom content-addressed store (`sha256(file)` →
`.data/blobs/ab/cdef...`).
**Default: (b).** DVC is the standard, but implementing a small
content-addressed store is a solid systems exercise and slots into the
custom tracker/registry. A dataset "version" is a manifest file listing
(logical_name, blob_hash, size) tuples. Manifests are checked into git;
blobs live in local storage (or optionally S3-compatible). This is what
DVC does under the hood.

**Config management.**
Options: (a) Hydra (composable YAML configs, CLI overrides), (b)
pydantic-settings (typed, one nested Python object), (c) both.
**Default: (b).** Hydra's composition is powerful but the runtime magic
(the `@hydra.main` decorator, the working-directory swap) fights with
`uv`, testing, and MyPy. pydantic-settings is dull, boring, and typed.
Use YAML files loaded into Pydantic models. Every pipeline run captures
its resolved config into the run's artefacts.

**Determinism.**
Fix all seeds (Python `random`, NumPy, torch). Set
`PYTHONHASHSEED`. Use `torch.use_deterministic_algorithms(True)` where
feasible. Honest note: full bit-level determinism across PyTorch versions
is very hard; aim for "same seed + same version + same data = same
metrics" and document it. Cite the PyTorch reproducibility docs in the
writeup.

**Environment locking.**
`uv lock` produces a `uv.lock` committed to the repo. Pin Python patch
version in the lockfile *and* in the Dockerfile. Rebuild images from
lockfile, never from `pyproject.toml` alone. This is table stakes; the
writeup should mention it in passing, not spend a page on it.

---

## 8. Correctness / evaluation testing

MLOps projects fail this section most often. What to include:

- **Offline eval on a frozen holdout.** The holdout is content-addressed
  and checked into the dataset registry. Every model that reaches the
  registry has a required eval report against this holdout. If the eval
  didn't run, the model isn't in the registry.
- **Champion-vs-challenger regression test.** A `promote` command loads
  the current production model and the challenger, runs both on the
  holdout, and requires the challenger's primary metric to be within a
  configurable epsilon (typically must equal or beat, but a small
  degradation may be tolerated if latency improves — decide at
  brainstorming). If it fails, promotion is rejected.
- **Data validation on ingest.** Schema check (columns, types), null-rate
  bounds, value-range bounds, class-balance bounds. Fail the pipeline
  before training if bounds are violated. Steal Great Expectations'
  vocabulary — 15 named expectations covers most of what a small dataset
  needs.
- **Pipeline smoke test.** A CI-time test that runs the full pipeline on
  a 100-row sample dataset and asserts every step produces its expected
  artefact. Catches integration bugs without a full training run.
- **Serving contract tests.** Given a canned input, assert the deployed
  model returns a valid response shape and a plausible probability
  distribution (sums to 1, non-degenerate).
- **Model monitoring tests.** Unit-test the drift detector: given
  synthetic drifted / non-drifted samples, does it correctly fire?
  Regression-test the metric extraction from serving logs.

The ML Test Score paper (section 2) is the canonical checklist. Cover
20+ of its 28 items and cite the paper — that alone is defensible
"comprehensive testing" evidence.

---

## 9. Benchmarking / metrics — what to measure

Pipeline-level, not just model accuracy. Model accuracy is table stakes;
the point of the flagship is the surrounding system.

- **End-to-end pipeline latency.** From `pipeline run` command to model
  in registry. Broken down per step.
- **Serving latency.** p50 / p95 / p99. Warm start and cold start
  separately. Reported for at least one realistic input distribution,
  not just a fixed test string.
- **Serving throughput.** Requests/sec at fixed latency budget (e.g.,
  "requests/sec at p99 < 100ms"). This exercises Project 2's batching.
- **Training reproducibility.** Same seed + same data + same config →
  metric variance across 5 runs. Report the standard deviation. This is
  the "we actually control determinism" claim.
- **Storage overhead.** Bytes per artefact version, dataset version.
  Prove versioning is cheap.
- **Cost per prediction.** Local, so cost is CPU-seconds. Include for
  completeness; note that this is mostly moot for a portfolio project
  but is the "if this ran in prod" translation.
- **Drift-detection recall on synthetic drift.** Inject known drift into
  a stream; measure detection latency and false-positive rate on the
  no-drift baseline.

Defensible claim: "pipeline p99 end-to-end is X seconds on a 100k-row
dataset on [named CPU]; served model p99 latency is Y ms; drift detector
catches injected 5% shift within Z minutes with < 1% false-positive rate
in steady state." Numbers, methodology, hardware — no cherry-picking.

---

## 10. Common pitfalls

- **The "MLflow + notebook" project.** Add MLflow calls to a notebook,
  claim MLOps. This is 80% of fresher MLOps portfolio projects. Avoid by
  making the pipeline runnable from a CLI, in CI, with no notebook in
  the critical path. Notebooks are fine for exploratory eval, not for
  the pipeline.
- **No offline eval / eyeballed metrics.** "Model looks good on some
  examples I tried." Any promotion decision has to be against a frozen
  holdout with a numeric criterion.
- **No drift or monitoring at all.** Even a minimal PSI check on one
  input feature is more than most projects have.
- **No data versioning.** "Trained on `data/train.csv`" with no hash and
  no manifest. Renders every metric un-reproducible.
- **No environment pinning.** Pipeline works today, breaks in six months
  after a `transformers` release. `uv.lock` + pinned Docker base image.
- **Cloud when a laptop suffices.** Provisioning a K8s cluster and half a
  GCP bill to serve a DistilBERT on 10 QPS is an anti-pattern. Do the
  opposite anti-pattern (fine-tuning a 70B model on a laptop) is also
  bad. Match infra to workload.
- **Over-adopting frameworks.** If the project is "MLflow + Prefect +
  DVC + Great Expectations + BentoML + Evidently glued with a config
  file," the resume claim is glue. The flagship claim requires *some*
  of those primitives to be built, not adopted.
- **Fake production.** Auto-deploying a model to "production" that has
  no users is theatre. Better: an explicit `promote` command with a
  human-visible gate, and be honest in the writeup that this is a
  portfolio project.
- **Skipping the writeup.** The pipeline is invisible on GitHub without
  the architecture doc and the demo. A good writeup can more than
  double the perceived depth of the code.
- **Ignoring the Sculley debt paper's warnings.** Especially: undeclared
  consumers (one team's model is another's input), correction cascades
  (fixing model A by training model B on A's errors), and feedback loops
  (the model's predictions influence the future data it trains on).
  Even in a solo portfolio project, name these in the writeup.

---

## 11. What flagship looks like

The pitch, in one sentence: **"I built a CPU inference server, a vector
database, and a distributed task queue from scratch — and here is an
end-to-end MLOps pipeline that uses all three to fine-tune a text model,
promote it through a CI-gated champion-vs-challenger evaluation, deploy it,
and automatically detect distribution drift and trigger a retrain."**

That is a story no bootcamp graduate can tell. It is also a story that only
works if Projects 2, 4, and 5 are actually built and their APIs are
stable — Project 6 depends on the earlier work landing.

**Demo shape.** A single command that:
1. Ingests a small dataset (with version pin).
2. Runs the training pipeline via the task queue; MLflow-style run log
   appears in the CLI.
3. Runs eval; produces a report file.
4. Deploys via the inference server; a `curl` request returns a
   classification.
5. Injects drifted traffic; the drift monitor fires within N seconds; an
   auto-retrain job appears in the task queue; the new challenger fails
   or passes the promotion gate visibly.

This demo is the flagship. It's the thing to record as a two-minute video
and link from the portfolio README.

**Blog post angles (in priority order).**
1. "Wiring three from-scratch systems into an MLOps pipeline: what
   integrating your own components actually costs." (Portfolio-story
   post; the meta-pitch.)
2. "Drift detection with a from-scratch vector DB: using embedding
   distance as a monitoring signal." (Technical, novel-ish, plays to
   Project 4.)
3. "Building an experiment tracker in 500 lines: what MLflow does that
   you don't need." (Build-vs-adopt post.)
4. "Champion-vs-challenger as a CI gate: the one MLOps check that
   matters." (Opinionated, cite Sato/Fowler.)

**What flagship does *not* look like.**
It does not mean re-implementing Kubernetes, Kafka, or Feast. Choose the
five primitives to actually build (custom tracker + registry, content-
addressed dataset store, drift monitor, promotion gate, pipeline layer
over the task queue). Everything else — the observability stack, the
container runtime, the CI runner — is adopted commodity. The judgment
call at brainstorming is which five, not "build all of it."

---

## Uncertainties to resolve at brainstorming

- Whether Project 5's task queue has DAG semantics or only single-task
  submission. Affects how thick the "pipeline layer" in Project 6 needs
  to be.
- Whether Project 2's inference server has a model-management API (load,
  unload, version-tag) or is one-model-per-process. If the latter,
  blue-green deploys are a docker-compose swap, not an API call.
- Whether the primary model should be DistilBERT (safer, better latency)
  or a LoRA'd small LLM (flashier, likely too slow on CPU for a
  retraining demo). Section 5 leans DistilBERT.
- Whether Tier 3 features (shadow deploy, auto-retrain) fit in 6–8 weeks
  given the number of custom primitives. Cutting shadow deploy is the
  most likely descope.
- Whether to publish the blog posts during the build (bird-in-hand
  content) or after (cleaner narrative). Non-blocking.
