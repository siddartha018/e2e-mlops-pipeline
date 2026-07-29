# Project 06 — End-to-End MLOps Pipeline

A reproducible training-to-serving MLOps pipeline built from small, correct
primitives instead of glued-together frameworks. Every step of the machine
learning lifecycle — dataset versioning, experiment tracking, model registry,
CI-gated promotion, deployment, and drift monitoring — is owned by code in
this repo, exposed through typed interfaces, and covered by tests.

The project is the capstone of a six-project systems + ML-infra portfolio.
It integrates with three other from-scratch components elsewhere in the
portfolio:

| Project | Role in this pipeline | Interface |
|---|---|---|
| **P2** CPU inference server | Model serving backend | [`InferenceClient`](src/mlops_pipeline/clients/inference.py) |
| **P4** Vector database | Storage for input embeddings + drift monitoring | [`VectorStoreClient`](src/mlops_pipeline/clients/vector_store.py) |
| **P5** Distributed task queue | Pipeline orchestrator (DAG execution) | [`TaskQueueClient`](src/mlops_pipeline/clients/task_queue.py) |

Because P2, P4, and P5 are still under construction, every client interface
ships with **two implementations**: a local in-process implementation that
works today, and an HTTP client that will point at the real service once it
lands. Swapping between them is a config change, not a rewrite.

---

## High Level overview

> "I built a CPU inference server, a vector database, and a distributed task
> queue from scratch — and this project is the end-to-end MLOps pipeline that
> uses all three to fine-tune a text classifier, promote it through a
> CI-gated champion-vs-challenger evaluation, deploy it, and detect
> distribution drift on embeddings.
>
> None of the core primitives — the experiment tracker, model registry,
> content-addressed dataset store, drift detector, or the DAG runner that
> sits over the task queue — are adopted OSS. They're small (roughly 200–500
> lines each), typed, dependency-injected, and independently tested. Adopted
> commodity is limited to the container runtime, the CI runner, HuggingFace
> transformers for the model itself, and FAISS as the local vector-index
> backend."

---

## Why build this from scratch?

The default MLOps portfolio project is *"I added `mlflow.log_metric` calls to
a notebook."* That's a demonstration of using a library, not of
understanding what MLOps actually is.

This project takes the opposite path. The reference literature converges on a
small set of primitives that every mature MLOps stack eventually reinvents:
a versioned artefact store, a run/experiment concept, a DAG runner, a
promotion state machine, and a drift signal. Those five are built here in a
few hundred lines each. Everything else is deliberately adopted:

| Primitive | Decision | Alternative considered | Rationale |
|---|---|---|---|
| Experiment tracker | **Build** | MLflow | ~250 LOC. Owns the "runs table" data model. |
| Model registry | **Build** | MLflow Registry, HF Hub | Small state machine, defensible resume claim. |
| Dataset store (CAS) | **Build** | DVC | Reveals what DVC actually does. |
| DAG runner | **Build** | Prefect, Airflow, Kedro | Whole point of the P5 integration story. |
| Drift detector | **Build** | Evidently, NannyML | Steals their vocabulary; skips runtime dep. |
| Container runtime | Adopt | (none reasonable) | Docker Compose. |
| CI runner | Adopt | (none reasonable) | GitHub Actions. |
| Model architecture | Adopt | Train from scratch | Fine-tune HF `bert-tiny`. |
| Vector index | Adopt (local); P4 (real) | Annoy, hnswlib | FAISS as placeholder until P4. |
| Config | Adopt | Hydra | Pydantic + YAML — typed, no runtime magic. |

Reference reading behind these choices is captured in
[docs/research-brief.md](docs/research-brief.md).

---

## Architecture at a glance

```
                       +---------------------------------+
                       |  CLI  (mlops-pipeline ...)      |
                       +----------------+----------------+
                                        |
                       +----------------v----------------+
                       |   Pipeline  (DAG runner)        |
                       |   compile DAG -> job submits    |
                       +--------+---------------+--------+
                                |               |
             +------------------+               +------------------+
             v                                                     v
   +-----------------------+                            +---------------------+
   |  TaskQueueClient      | <-- interface              |    Storage layer    |
   |  +- LocalRunner       |                            |  +- Tracker (SQLite)|
   |  +- HttpP5Client      |                            |  +- Registry       |
   +-----------------------+                            |  +- DatasetStore   |
                                                        |  +- ArtefactStore  |
   +-----------------------+                            |     (sha256 CAS)   |
   |  VectorStoreClient    | <-- interface              +---------------------+
   |  +- FaissLocalStore   |
   |  +- HttpP4Client      |                            +---------------------+
   +-----------------------+                            |   Drift detector    |
                                                        |   PSI + Wasserstein |
   +-----------------------+                            |   + embedding cosine|
   |  InferenceClient      | <-- interface              +---------------------+
   |  +- LocalInference    |
   |  +- HttpP2Client      |
   +-----------------------+
```

The pipeline object never talks to a real socket, a real filesystem clock, or
a real random source directly. Every side-effect goes through an injected
interface, so the whole system can be driven from tests with fakes — the
same discipline used by the Raft project (P1) in this portfolio.

### Layered view

| Layer | What lives here | Files |
|---|---|---|
| **CLI** | Typer-based command surface | [cli.py](src/mlops_pipeline/cli.py) |
| **Pipelines** | Concrete DAG definitions (`train_classifier`) | [pipelines/training.py](src/mlops_pipeline/pipelines/training.py) |
| **Orchestration** | Generic Pipeline DAG runner | [pipeline.py](src/mlops_pipeline/pipeline.py) |
| **Domain services** | Tracker, Registry, DatasetStore, DriftDetector, PromotionGate | [tracker.py](src/mlops_pipeline/tracker.py), [registry.py](src/mlops_pipeline/registry.py), [dataset_store.py](src/mlops_pipeline/dataset_store.py), [drift.py](src/mlops_pipeline/drift.py), [promote.py](src/mlops_pipeline/promote.py) |
| **Model code** | HF training / evaluation / encoder wrappers | [models/train.py](src/mlops_pipeline/models/train.py), [models/evaluate.py](src/mlops_pipeline/models/evaluate.py), [models/encoder.py](src/mlops_pipeline/models/encoder.py) |
| **External clients** | Task queue, vector store, inference server | [clients/task_queue.py](src/mlops_pipeline/clients/task_queue.py), [clients/vector_store.py](src/mlops_pipeline/clients/vector_store.py), [clients/inference.py](src/mlops_pipeline/clients/inference.py) |
| **Core primitives** | Clock, Rng, ArtefactStore, content hashing | [core.py](src/mlops_pipeline/core.py) |
| **Config** | Typed Pydantic config loaded from YAML | [config.py](src/mlops_pipeline/config.py), [configs/sst2.yaml](configs/sst2.yaml) |
| **Wiring** | Build a `Context` object from a `Config` | [context.py](src/mlops_pipeline/context.py) |

---

## Data flow: from `mlops-pipeline run train` to a registered model

```
   +-------------+     +----------+     +---------+     +----------+     +-----------+
   |   ingest    | --> | validate | --> |  train  | --> | evaluate | --> |  register |
   +-------------+     +----------+     +---------+     +----------+     +-----------+
         |                    |               |                |                 |
         v                    v               v                v                 v
   Load SST-2 (or       Row overlap    Fine-tune HF     Metrics on         New model
   configured HF        checks,        classifier,      frozen test        version in
   dataset).            schema         seeded, single   split written      the Registry
   Files parquet-ized,  invariants     epoch by         to the tracker.    (stage=none).
   sha256-hashed,       beyond the     default. Model                      Ready for
   written to CAS.      ingest-time    tar.gz'd,        Metrics also                 promotion.
   Manifest written to  expectations.  written to CAS.  attached to the
   .mlops/manifests/.                  Metrics + params  training run.
                                       + config hash
                                       logged to
                                       Tracker.
```

Every arrow above is a `TaskQueueClient.submit(...)` call with a
`depends_on` edge. Every rectangle is a `@pipe.step(name)`-decorated Python
function. Reruns with an identical config hash reuse cached results via
content-addressed idempotency keys.

### What gets written to disk

```
.mlops/
+- blobs/                                       # content-addressed store
|  +- ab/cdef123... (model tar, dataset parquet, arbitrary artefacts)
|  +- ...
+- manifests/                                   # dataset versions (checked into git)
|  +- sst2-<manifest-hash-prefix>.json
+- tracker.sqlite                               # runs, params, metrics, artefacts
+- registry.sqlite                              # model versions + stages
```

The blob directory is `.gitignore`d; manifests and the SQLite databases are
the small artefacts a reviewer can inspect.

---

## Component walkthrough

### 1. Core primitives — [core.py](src/mlops_pipeline/core.py)

The three dependency-injected primitives that every other module takes via
its constructor.

| Primitive | Real impl | Fake / test impl | Why injected |
|---|---|---|---|
| `Clock` | `RealClock` — wraps `time.time` and `time.sleep` | `FakeClock` — advances virtually under test control | Deterministic timestamps; controllable retry-backoff. |
| `Rng` | `RealRng` — seeded `random.Random` | (uses `RealRng` seeded; tests pin seed) | Seed printed on any test failure. |
| `ArtefactStore` | `.mlops/blobs/<ab>/<cdef...>` on local disk | Any tmpdir | One CAS backs the tracker, registry, and dataset store. |

`ArtefactStore.put(bytes)` writes a temp file and renames it into place so a
partial write never leaves a valid-hash-but-corrupt blob. `put_file(path)`
streams a large file through sha256 in 1 MiB chunks.

### 2. Tracker — [tracker.py](src/mlops_pipeline/tracker.py)

SQLite-backed experiment tracker. Four tables:

| Table | Columns |
|---|---|
| `runs` | `id, name, status, started_at, finished_at, git_sha, config_hash` |
| `params` | `run_id, key, value (JSON)` |
| `metrics` | `run_id, key, value, step, ts` |
| `artefacts` | `run_id, name, blob_hash` |

Lineage is trivially recoverable: a run's `config_hash` and its
`manifest_hash` param give you a fully deterministic replay recipe.

```python
run = tracker.start_run("train_classifier", config=cfg.dump())
run.log_param("manifest_hash", "abc...")
run.log_metric("eval/accuracy", 0.84, step=0)
blob_hash = run.log_artefact("model.tar.gz", tar_bytes)
run.finish("success")
```

### 3. Registry — [registry.py](src/mlops_pipeline/registry.py)

Model promotion state machine.

```
                     staging
                   /    ^    \
                  v     |     v
                none    +--- production
                  \           |
                   +---> archived
```

Enforced transitions:

| From | Allowed to |
|---|---|
| `none` | `staging`, `archived` |
| `staging` | `production`, `archived`, `none` |
| `production` | `archived`, `staging` |
| `archived` | — |

Invariant: **at most one row per model name has `stage='production'`**.
Enforced by a SQL partial unique index (`CREATE UNIQUE INDEX ... WHERE
stage = 'production'`). Promoting a new version to production
automatically archives the previous incumbent.

### 4. Dataset store — [dataset_store.py](src/mlops_pipeline/dataset_store.py)

Content-addressed dataset versioning. A dataset version *is* the SHA-256 of
its manifest.

```json
// .mlops/manifests/sst2-a4f1c9....json
{
  "name": "sst2",
  "files": [
    {"logical_name": "train", "blob_hash": "abc123...", "size": 812340, "rows": 5000},
    {"logical_name": "val",   "blob_hash": "def456...", "size":  81234, "rows":  500},
    {"logical_name": "test",  "blob_hash": "ghi789...", "size":  81234, "rows":  500}
  ],
  "dataset_schema": {"columns": ["sentence", "label"], "dtypes": {"sentence": "object", "label": "int64"}},
  "row_count": 6000
}
```

Ingest also runs a set of Great-Expectations-style validation checks and
raises `DataValidationError` before anything is written on failure:

| Expectation | Meaning | Configured for SST-2 |
|---|---|---|
| `columns_match` | Column set is exactly what we expect | (optional) |
| `not_null` | Null rate on a column is below a threshold | `max_null_rate=0.0` on `sentence`, `label` |
| `value_in_set` | Column values are a subset of an allowlist | `label in {0, 1}` |
| `value_between` | Numeric column is within `[min, max]` | (optional) |
| `row_count_between` | Row count within bounds | `min=10, max=10_000_000` |
| `class_balance` | Worst-class fraction is above a threshold | `min_fraction=0.05` |

### 5. Pipeline / DAG runner — [pipeline.py](src/mlops_pipeline/pipeline.py)

A `Pipeline` is a Python object with `@pipe.step(name, depends_on=[...])`
decorated functions. On `pipeline.run(config)` it:

1. Topologically sorts the step DAG (cycle detection raises).
2. For each step, computes an idempotency key:
   `sha256(pipeline_name, step_name, config_hash, sorted(dep_job_ids))`.
3. Submits a `JobSpec` to the injected `TaskQueueClient` with the idempotency
   key and `depends_on` edges.
4. Waits on the terminal step(s) and returns their results.

**Why idempotency matters.** A rerun with the same config is a lookup, not a
recompute. That's what makes the pipeline recoverable — if a downstream step
crashes, resubmitting the same pipeline picks up the successful upstream
results instead of retraining from scratch.

### 6. Task queue client — [clients/task_queue.py](src/mlops_pipeline/clients/task_queue.py)

Interface:

| Method | Purpose |
|---|---|
| `submit(spec, depends_on)` | Enqueue a job |
| `wait(job_id, timeout_s)` | Block until terminal state |
| `status(job_id)` | Poll current state |
| `cancel(job_id)` | Best-effort cancel |
| `register(fn_name, fn)` | Register a callable so remote job records can execute it |

Two implementations:

- **`LocalRunner`** — in-process topological executor. Runs jobs sequentially
  on the caller's thread. Retries with exponential backoff up to
  `max_retries`, then transitions the job to `DEAD` (deadlettered).
  Dependencies that fail mark downstream jobs `DEAD` without executing them.
- **`HttpP5Client`** — thin wrapper over `POST /jobs`, `GET /jobs/{id}`,
  `POST /jobs/{id}/cancel`. Not usable until P5 is up.

Job state machine:

```
    PENDING --> RUNNING --> SUCCESS
                  |
                  +--> FAILED --(retries left)--> RUNNING
                            \
                             +--> DEAD  (retries exhausted, or upstream DEAD)

    * -> CANCELLED  (via .cancel() while not in terminal state)
```

### 7. Vector store client — [clients/vector_store.py](src/mlops_pipeline/clients/vector_store.py)

Interface: `upsert`, `batch_upsert`, `query(k)`, `centroid(filter)`,
`all_embeddings(filter)`, `size`.

- **`FaissLocalStore`** — wraps `faiss.IndexFlatIP` (inner product). Vectors
  are expected to arrive L2-normalized, so inner product equals cosine
  similarity. Metadata is kept alongside in a parallel list and used for
  filter-by-metadata during `centroid` / `all_embeddings`.
- **`HttpP4Client`** — stub. `POST /vectors`, `POST /vectors/batch`,
  `POST /query`, `POST /centroid`.

The vector store is used for two things: (a) rolling-window storage of
production input embeddings for drift monitoring, and (b) any
retrieval-based demo the pipeline might run downstream of training.

### 8. Inference client — [clients/inference.py](src/mlops_pipeline/clients/inference.py)

Interface: `load_model(name, version, blob_path)`, `unload(name, version)`,
`predict(name, version, inputs)`, `list_loaded()`.

- **`LocalInferenceClient`** — lazy-imports `transformers`, wraps a
  `text-classification` pipeline. Supports multiple named/versioned models
  in the same process (mirrors what P2 will expose).
- **`HttpP2InferenceClient`** — stub. Assumes `POST
  /models/{name}/versions/{v}/load`, `POST /predict`, etc.

### 9. Drift detector — [drift.py](src/mlops_pipeline/drift.py)

Three signals, each with a per-signal threshold.

| Signal | Statistic | Best for | Default threshold |
|---|---|---|---|
| `wasserstein` | Wasserstein-1 distance, normalized by baseline std | Continuous numeric features | `0.15` |
| `psi` | Population Stability Index across value counts | Categorical / class distribution | `0.20` |
| `cosine` | `1 - cos(baseline_centroid, current_centroid)` | Text / embedding drift | `0.15` |

The overall `DriftReport.verdict` is `"drift"` if any monitored signal
crosses its threshold — an intentionally conservative OR. The report keeps
each per-signal value + threshold + `drifted` boolean so a user can see
exactly which feature moved.

### 10. Promotion gate — [promote.py](src/mlops_pipeline/promote.py)

The champion-vs-challenger check that gates promotion to `production`.

```
  challenger v_c   +--> evaluate on frozen holdout --> metric_c
                   |
  champion (prod)  +--> evaluate on frozen holdout --> metric_m
                                                          |
                                                          v
                    delta = metric_c - metric_m (or reversed if lower-is-better)
                                                          |
                          +-------- delta >= min_improvement ? -------+
                          |                                            |
                          v YES                                        v NO
                    promote v_c to production                    keep champion,
                    (auto-archives previous prod)                 report rejection
```

If there is no incumbent, the challenger is auto-promoted (with a `staging`
transition in between to keep the state machine valid).

### 11. CLI — [cli.py](src/mlops_pipeline/cli.py)

| Command | Purpose |
|---|---|
| `mlops-pipeline run train -c configs/sst2.yaml` | Run the full training pipeline end-to-end |
| `mlops-pipeline runs list` | List all runs (id, name, status, started_at, config_hash) |
| `mlops-pipeline runs show <id>` | Show params + metrics + artefacts for one run |
| `mlops-pipeline models list [name]` | List all registered model versions and their stages |
| `mlops-pipeline promote <name> <v> --stage production` | Promote a version (production runs the gate) |
| `mlops-pipeline drift <baseline_manifest> <current_manifest>` | Compute per-signal drift between two dataset versions |
| `mlops-pipeline demo` | The flagship end-to-end demo |

---

## Configuration

All config is a typed Pydantic tree loaded from YAML. See
[configs/sst2.yaml](configs/sst2.yaml) for the default. Config sections:

| Section | Notable fields | Purpose |
|---|---|---|
| `data` | `hf_name`, `text_column`, `label_column`, `train_size` | Dataset selection + sizing |
| `train` | `model_name`, `max_length`, `batch_size`, `epochs`, `seed` | HF training args |
| `encoder` | `model_name`, `dim`, `batch_size` | Sentence-transformer for embeddings |
| `drift` | `psi_threshold`, `wasserstein_threshold`, `cosine_threshold` | Drift verdict thresholds |
| `promote` | `metric`, `higher_is_better`, `min_improvement` | Promotion gate criterion |
| `paths` | `root`, `manifests`, `tracker_db`, `registry_db` | Where state lives |
| `clients` | `task_queue`, `vector_store`, `inference` (`local` / `http_p*`) | Which client impl to wire in |

The resolved config is hashed and stored on every run so reproducing a
result is `git checkout <sha> && mlops-pipeline run train -c <same-config>`.

---

## Quickstart

### 1. Environment

```bash
cd prior-projects/06-mlops-pipeline
uv venv
uv pip install -e ".[dev]"
```

### 2. Fast test suite

```bash
uv run pytest -m "not slow" -q
```

Expect **~38 tests, ~2 seconds** on a modern laptop. This covers everything
except the real HuggingFace training path.

### 3. Real training + promotion demo

```bash
# Ingest, train, evaluate, register.
uv run mlops-pipeline run train -c configs/sst2.yaml

# Inspect what happened.
uv run mlops-pipeline runs list -c configs/sst2.yaml
uv run mlops-pipeline models list -c configs/sst2.yaml

# Promote v1 to production (auto-passes since there's no incumbent).
uv run mlops-pipeline promote classifier 1 --stage production -c configs/sst2.yaml

# Train a second model, promote via champion-vs-challenger.
uv run mlops-pipeline run train -c configs/sst2.yaml
uv run mlops-pipeline promote classifier 2 --stage production -c configs/sst2.yaml
```

### 4. Drift check on a shifted dataset

```bash
uv run mlops-pipeline drift <baseline-manifest-hash> <current-manifest-hash> \
    -c configs/sst2.yaml
```

---

## Testing story

| Kind | What it covers | Marker | Runtime |
|---|---|---|---|
| Core | `ArtefactStore` round-trip, layout, content hashing, `FakeClock` semantics | (default) | ms |
| Tracker | Start/finish, params/metrics/artefacts round-trip, ordering | (default) | ms |
| Registry | Register + auto-increment, all valid transitions, invariant enforcement, error paths | (default) | ms |
| Dataset store | Ingest, deterministic hashing, each expectation kind | (default) | ms |
| Pipeline | Linear DAG, fan-out/fan-in, retry-then-success, retry-exhausted-kills-downstream, idempotency, cycle detection | (default) | ms |
| Drift | No-drift stability, mean-shift, categorical flip, embedding rotation, overall verdict | (default) | ms |
| Promotion | First-production auto-pass, challenger beats champion, below-threshold rejection, lower-is-better metric | (default) | ms |
| Vector store | Upsert + query, centroid-with-filter, size | (default) | ms |
| End-to-end smoke | Ingest -> fake-train -> eval -> register with real Tracker/Registry/DatasetStore/LocalRunner | (default) | ~50ms |
| Real training | Actual HF fine-tune on a tiny dataset, then eval | `slow` | minutes on CPU |

Total: **38 fast tests, all passing**. The full pipeline is exercised
end-to-end without depending on HuggingFace loading in the fast tier.

Every test that needs a clock uses `FakeClock`, so tests are deterministic
regardless of wall-clock time.

---

## Design decisions worth defending

**Why SQLite, not Postgres?** — Portfolio project, single-writer, and
zero-config beats server-hosted for demoability. The schema is small enough
to port to Postgres in an afternoon; it uses only standard SQL.

**Why content-addressed everything?** — Cheap deduplication (same model
trained twice occupies one blob), free integrity check (hash mismatch on
read is corruption detection), and a natural cache key for pipeline steps.

**Why not adopt MLflow?** — MLflow works. The point of the project is to
show I understand what MLflow is *doing*, which requires building
the two-table equivalent (runs + params/metrics/artefacts) and living with
it. The README acknowledges MLflow explicitly as prior art.

**Why is `LocalRunner` single-threaded?** — Two reasons. First, the real
executor is P5 (which will be multi-worker); `LocalRunner` is meant to be
the *deterministic* fallback and the test substrate. Second, most pipeline
correctness bugs (bad idempotency keys, missed dependency edges, retry
storms) reproduce in single-threaded execution and are easier to debug
there. Concurrency lives in P5.

**Why not include a shadow deploy?** — Scope. The research brief lists it
as a Tier-3 stretch feature; this project targets Tier 2 + drift +
auto-retrain. Shadow deploy is a clean follow-up on the same interfaces.

**Why Pydantic instead of Hydra?** — Hydra's `@hydra.main` decorator swaps
the working directory and fights with `uv`, testing, and mypy. Pydantic is
boring, typed, and composes with anything.

---

## Integration with the rest of the portfolio

```
   +-----------+     +-------------+     +--------------+
   | Project 1 |     | Project 2   |     | Project 4    |
   |  Raft KV  |     |  Inference  |     |  Vector DB   |
   +-----------+     +------+------+     +-------+------+
                            |                    |
                            |                    |
                     +------v--------+   +-------v------+
                     |               |   |              |
                     |  InferenceClient  VectorStoreClient
                     |               |   |              |
                     +---+ Project 6 (this repo) +------+
                            |               |
                            +---+-------+---+
                                        |
                                +-------v-------+
                                | Project 5     |
                                | Task Queue    |
                                +---------------+
                                        ^
                                        |
                                TaskQueueClient
```

The pipeline layer is a *client of* every earlier project. Interface
contracts are captured in [src/mlops_pipeline/clients/](src/mlops_pipeline/clients/).
When each project ships, we flip `clients.<name>` in
[configs/sst2.yaml](configs/sst2.yaml) from `local` to `http_p*` and provide
the base URL — no other pipeline code changes.

The [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml)
already reserve the network positions for P2/P4/P5 as `nginx:alpine`
placeholder services on ports 8002/8004/8005. Swapping in the real service
images is a one-line diff.

---

## Repository layout

```
prior-projects/06-mlops-pipeline/
+- README.md                       (this file)
+- pyproject.toml                  (uv-managed, ruff + mypy strict + pytest)
+- Dockerfile
+- docker-compose.yml
+- .github/workflows/ci.yml
+- configs/
|  +- sst2.yaml                    (default config for the SST-2 demo)
+- docs/
|  +- research-brief.md            (design-space survey, references, pitfalls)
+- src/mlops_pipeline/
|  +- core.py, config.py, context.py, cli.py
|  +- tracker.py, registry.py, dataset_store.py, drift.py, promote.py
|  +- pipeline.py
|  +- clients/{task_queue.py, vector_store.py, inference.py}
|  +- models/{train.py, evaluate.py, encoder.py}
|  +- pipelines/{training.py}
+- tests/
   +- test_core.py, test_tracker.py, test_registry.py, test_dataset_store.py
   +- test_pipeline.py, test_drift.py, test_promote.py, test_clients.py
   +- test_smoke_pipeline.py
```

---

## Status

| Milestone | State |
|---|---|
| Design + scaffold complete | Done |
| Core primitives implemented and unit-tested | Done |
| Custom tracker, registry, dataset store implemented | Done |
| DAG runner + LocalRunner implemented | Done |
| Drift detector implemented + tested (PSI / Wasserstein / cosine) | Done |
| Champion-vs-challenger promotion gate implemented + tested | Done |
| End-to-end smoke pipeline runs in <1s | Done |
| CI (lint + typecheck + fast tests) | Done |
| Real HF training path exercised on SST-2 | Local only; not in CI |
| P2 / P4 / P5 HTTP clients written; endpoints stubbed | Interfaces done, awaiting real services |
| Drift-triggered auto-retrain wired to the pipeline | Pending |
| Docker image + full `docker-compose up` demo with real services | Pending (blocked on P2/P4/P5) |
| Blog post: "wiring three from-scratch systems into an MLOps pipeline" | Pending |

---



## References

The design-space survey, reading list (Rules of ML, Sculley's technical-debt
paper, CD4ML, Huyen's *Designing Machine Learning Systems*, the ML Test
Score rubric), and pitfall catalogue are in
[docs/research-brief.md](docs/research-brief.md).
