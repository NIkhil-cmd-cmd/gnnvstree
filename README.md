# gnnvstree

`gnnvstree` is a small research repo for comparing two families of tool-logic
models:

- a graph-based model on the `gnn` branch
- a tree-based model on the `tree` branch

This README is written as an onboarding document for a new coding agent. It
explains the current tree-side implementation, the intended experiment shape,
the important files, the data formats, the metrics, and the next useful work.

## Current Branch

You are expected to do tree-side work on the `tree` branch.

```bash
git branch --show-current
```

Expected output:

```text
tree
```

The repository started as a mostly empty two-branch shell. The current `tree`
branch now contains a dependency-light Python package under `src/gnnvstree`.

## Goal

The goal is to create a fair test harness for custom logic models trained on
agent/tool traces.

The immediate model on this branch is a tree-based method. A separate graph
method can be implemented on the `gnn` branch or behind another adapter later.

The evaluation inputs should be:

- ToolBench or StableToolBench traces
- A couple of live model runs in the style of the Pareto experiment from the
  `chorus` repo

The evaluation outputs should be broad, metrics-heavy reports about:

- dataset quality
- training behavior
- model structure
- next-tool prediction quality
- sequence-level trace quality
- live run impact
- tree-vs-graph comparison readiness

## Repository Layout

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── gnnvstree/
│       ├── __init__.py
│       ├── cli.py
│       ├── loaders.py
│       ├── metrics.py
│       ├── split.py
│       ├── trace.py
│       └── tree_model.py
└── tests/
    ├── fixtures/
    │   ├── live_runs.json
    │   └── toolbench_sample.json
    └── test_tree_eval.py
```

## Important Files

### `src/gnnvstree/trace.py`

Defines the canonical trace schema.

Key classes and helpers:

- `ToolCall`: one normalized tool/API/action call.
- `Trace`: one ordered trace with source, task id, split, success flag, tool
  calls, optional semantic steps, task text, candidate tools, ground truth
  tools, and metadata.
- `normalize_name()`: normalizes tool/API names for stable identity.
- `make_trace()`: convenience builder used by loaders and tests.

This schema is intentionally model-neutral. Both tree and graph models should
consume this same shape.

### `src/gnnvstree/loaders.py`

Loads external data into the canonical trace schema.

Current loaders:

- `load_toolbench_queries(path)`
- `load_live_runs(path)`
- `load_jsonl_traces(path)`

Important ToolBench rule:

`api_list` is candidate-tool context, not ground truth. The loader records it as
`candidate_tools`, but the ordered trace comes from `relevant APIs`.

That matters because using `api_list` as edges or trace steps creates false
tool relationships.

### `src/gnnvstree/tree_model.py`

Contains the current tree-side model: `TreeLogicAdapter`.

It is a deterministic prefix tree with suffix fallback.

What it does:

- trains on ordered tool traces
- stores prefix/suffix continuation counts
- predicts the likely next tool from the current prefix
- supports online `update(trace)`
- emits structure metrics such as node count, leaf count, depth, branch factor,
  transition counts, and model JSON size

This is intentionally simple. It is a strong baseline and a useful starting
point for more advanced tree methods.

### `src/gnnvstree/metrics.py`

Computes broad evaluation metrics.

Main entry point:

```python
evaluate_model(model, train_traces, test_traces)
```

The model must provide:

```python
fit(traces)
predict_next(prefix, context=None)
metrics()
```

Current report sections:

- `dataset`
- `training`
- `testing`
- `sequence`
- `structure`

### `src/gnnvstree/split.py`

Deterministic task-id-based splitting.

The default split is:

- 70% train
- 15% validation
- 15% test

The split is hash-based so repeated runs with the same seed produce the same
assignment.

### `src/gnnvstree/cli.py`

Command-line runner for the tree evaluation harness.

It can load:

- ToolBench/StableToolBench JSON files
- live Pareto-style `runs.json` files
- canonical JSONL trace files

It writes JSON reports to an output directory.

### `tests/`

Contains a small fixture suite that protects the core behavior.

The most important test is that ToolBench `api_list` is not treated as ground
truth.

## Setup

This package currently has no third-party runtime dependencies.

From the repo root:

```bash
python3 -m pip install -e .
```

If installing is not desirable, use `PYTHONPATH=src` when running commands.

## Run Tests

The tests are written to run with the standard library `unittest` runner, even
though the test file uses plain `assert` statements.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

If `pytest` is installed, this also works:

```bash
PYTHONPATH=src pytest
```

## Smoke Test

Run the CLI against the included fixtures:

```bash
PYTHONPATH=src python3 -m gnnvstree.cli \
  --toolbench tests/fixtures/toolbench_sample.json \
  --live-runs tests/fixtures/live_runs.json \
  --output-dir /tmp/gnnvstree-tree-smoke
```

Expected output is a small JSON object with the output directory and top-1
accuracy.

The output directory should contain:

```text
report.json
training_metrics.json
test_metrics.json
structure_metrics.json
split_summary.json
```

## ToolBench Input Format

The loader expects a JSON list of query objects.

Minimal supported shape:

```json
[
  {
    "query_id": 1,
    "query": "Track a package and check service health.",
    "api_list": [
      {
        "tool_name": "Create Container Tracking",
        "api_name": "Get Tracking Data",
        "api_description": "Track a container."
      }
    ],
    "relevant APIs": [
      ["Create Container Tracking", "Get Tracking Data"],
      ["SQUAKE", "Checkhealth"]
    ]
  }
]
```

The resulting trace steps are:

```text
Create_Container_Tracking_Get_Tracking_Data
SQUAKE_Checkhealth
```

The `api_list` entries are retained as candidate tools. They are not used as
ground-truth trace steps.

## Live Pareto-Style Input Format

The live loader expects a JSON list of run objects, similar to the `runs.json`
artifact from the Pareto experiment in the `chorus` repo.

Minimal supported shape:

```json
[
  {
    "index": 1,
    "phase": "baseline",
    "success": true,
    "valid": true,
    "exact_optimal": true,
    "task": {
      "name": "live_task_a",
      "description": "Open site and inspect table."
    },
    "trace": [
      "browser_navigate",
      "browser_snapshot",
      "browser_click",
      "browser_evaluate"
    ],
    "semantic_steps": [
      "open_home",
      "inspect_table",
      "select_row",
      "extract_result"
    ],
    "tokens": 1200,
    "latency_ms": 30000
  }
]
```

The loader uses the normalized `trace` field when present. If `trace` is absent,
it falls back to extracting names from `tool_calls`.

Live metadata is preserved in `Trace.metadata`, including:

- phase
- run index
- token count
- latency
- validity
- exactness
- error string

## Canonical JSONL Input Format

The JSONL loader is useful for synthetic or preprocessed traces.

Each line should look like:

```json
{"source":"manual","task_id":"task-1","split":"train","success":true,"tool_names":["search","open","buy"]}
```

Optional fields:

- `semantic_steps`
- `task_text`
- `candidate_tools`
- `ground_truth_tools`
- `metadata`

## Tree Model Behavior

`TreeLogicAdapter` learns continuation counts from every suffix up to
`max_suffix`.

Example training trace:

```text
search -> open -> buy
```

The model records that:

- after `search`, `open` is likely
- after `search -> open`, `buy` is likely
- after suffix `open`, `buy` is likely

At prediction time, the model tries the longest suffix first. If no matching
suffix has continuation counts, it falls back to global next-tool frequencies.

This design makes it useful when:

- exact full prefixes repeat
- shorter local patterns repeat
- traces vary but share repeated action motifs

It is less useful when:

- the next tool depends on rich state not represented in the trace
- the trace vocabulary is too coarse
- training and test transitions barely overlap
- order in the source data is not meaningful

## Current Metrics

The harness intentionally emits more metrics than are needed for a first pass.
This is deliberate: the experiment is exploratory, and the goal is to understand
why a framework works or fails.

### Dataset Metrics

Examples:

- trace count
- successful trace count
- failed trace count
- success rate
- unique task count
- unique tool count
- transition count
- unique transition count
- average trace length
- median trace length
- p90 trace length
- empty trace rate
- single-step trace rate
- duplicate trace rate
- source distribution
- split distribution
- top tools
- top transitions

### Train/Test Overlap Metrics

Examples:

- tool overlap rate
- held-out tool count
- transition overlap rate
- held-out transition count

These are important because a transition model can look bad simply because the
test set contains tools or transitions never seen during training.

### Training Metrics

Examples:

- fit time
- trace count
- unique tools
- transition count
- unique transitions
- model JSON size

For the current prefix tree there is no gradient loss curve. Future tree models
can add impurity, compression, pruning, or validation objective metrics here.

### Prediction Metrics

Examples:

- examples
- coverage rate
- abstention rate
- top-1 accuracy
- top-3 accuracy
- top-5 accuracy
- accuracy when predicted
- mean reciprocal rank
- confidence mean
- confidence median
- confidence p90
- prediction latency p50
- prediction latency p90
- unseen-transition top-1 accuracy
- per-tool accuracy
- sample predictions

### Sequence Metrics

Examples:

- evaluated sequences
- full-trace exact match rate
- average normalized edit distance
- median divergence step
- loop repetition rate

These are different from next-step metrics. A model can have decent next-step
accuracy but still produce bad full generated traces.

### Structure Metrics

Examples:

- node count
- leaf count
- max depth
- average depth
- average branch factor
- max branch factor
- top tools
- top transitions
- model JSON size

For tree work, these metrics are especially important. They tell you if the tree
is learning reusable structure or just memorizing traces.

## Output Report Shape

`report.json` has this top-level shape:

```json
{
  "dataset": {
    "train": {},
    "test": {},
    "overlap": {}
  },
  "training": {},
  "testing": {},
  "sequence": {},
  "structure": {}
}
```

This shape should remain stable if possible. A future graph adapter should be
able to emit the same top-level report sections.

## How To Add A New Tree Method

The easiest path is to create a new adapter class with the same interface as
`TreeLogicAdapter`.

Required methods:

```python
class MyTreeAdapter:
    def fit(self, traces):
        ...

    def predict_next(self, prefix, context=None):
        ...

    def metrics(self):
        ...
```

`predict_next()` should return either `None` or an object with:

- `tool`
- `confidence`
- `alternatives`

The existing `Prediction` dataclass in `tree_model.py` is reusable.

Recommended adapter additions:

- `update(trace)` for online learning
- `explain_prediction(prefix, context)` for debugging
- `to_dict()` for persistence and model-size metrics

## How To Compare Against A Graph Model

The current harness is model-neutral enough for a graph adapter.

The graph-side adapter should consume the same `Trace` objects and expose the
same three required methods:

```python
fit(traces)
predict_next(prefix, context=None)
metrics()
```

Then a comparison runner can:

1. load traces once
2. create the same deterministic split
3. run `evaluate_model(tree_model, train, test)`
4. run `evaluate_model(graph_model, train, test)`
5. compare matching report fields

Useful comparison metrics:

- top-1/top-3/top-5 delta
- MRR delta
- full-trace exact-match delta
- edit-distance delta
- train time delta
- prediction latency delta
- model size delta
- robustness on unseen transitions
- robustness as training data is reduced

## Live Run Caution

The previous Pareto-style bookstore experiment in the `chorus` repo produced a
very small raw MCP trace vocabulary. Many traces collapsed to only a few browser
tools such as:

```text
browser_navigate
browser_snapshot
browser_evaluate
```

That is not enough structure for a meaningful tree-vs-graph comparison.

For live runs, prefer recording both:

- raw MCP tool names
- semantic action steps

Examples of useful semantic steps:

- `open_home`
- `login_fill_username`
- `login_submit`
- `inspect_table`
- `sort_table`
- `filter_table`
- `open_detail`
- `extract_price`
- `submit_form`
- `handle_validation_error`

The raw tool trace is useful for low-level execution analysis. The semantic
trace is useful for learning workflow logic.

## Recommended Next Work

### 1. Run the Existing Smoke Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m gnnvstree.cli \
  --toolbench tests/fixtures/toolbench_sample.json \
  --live-runs tests/fixtures/live_runs.json \
  --output-dir /tmp/gnnvstree-tree-smoke
```

### 2. Add Real ToolBench Data

Point the CLI at a StableToolBench query file, for example:

```bash
PYTHONPATH=src python3 -m gnnvstree.cli \
  --toolbench /path/to/stabletoolbench/solvable_queries/test_instruction/G2_category.json \
  --output-dir runs/stabletoolbench-g2-tree
```

Keep in mind that `relevant APIs` gives co-usage order only if the source order
is meaningful. If answer traces are available, a future loader should parse
those for stronger sequence supervision.

### 3. Add Ordered Answer-Trace Loading

ToolBench answer traces can contain more realistic call order than query-level
`relevant APIs`. Add a loader that extracts actual ordered tool calls from answer
artifacts where possible.

Target behavior:

- read answer trace files
- recover ordered API calls
- join them back to query ids
- mark metadata `has_ordered_answer_trace=true`
- fall back to `relevant APIs` only when no ordered trace exists

### 4. Add Semantic Live Trace Capture

For live Pareto-style runs, raw MCP tools are too coarse. Add a semantic
normalizer that maps tool calls and page state into workflow actions.

Keep both views:

- `tool_calls`
- `semantic_steps`

Then run metrics on both levels.

### 5. Add a Comparison Runner

Create a runner that evaluates multiple adapters on one frozen split.

Potential output:

```text
runs/<timestamp>/
├── tree_report.json
├── graph_report.json
├── comparison_report.json
└── split_summary.json
```

### 6. Add Robustness Sweeps

Useful sweeps:

- 10%, 25%, 50%, 100% training data
- max suffix length
- minimum confidence
- include failed traces vs successful-only traces
- raw tool traces vs semantic traces
- ToolBench-only vs live-only vs combined

## Known Limitations

- The current tree model is a prefix-count model, not a learned decision tree.
- ToolBench `relevant APIs` may not always represent true call order.
- There is no dedicated graph adapter in this branch yet.
- There is no plotting layer yet; reports are JSON and plots-ready.
- No external dependency manager is configured beyond basic setuptools.
- Confidence is count-based frequency, not calibrated probability.
- Sequence generation currently starts from the first ground-truth tool for
  evaluation; it does not solve start-tool prediction as a separate problem.

## Agent Operating Notes

When continuing work in this repo:

1. Stay on the `tree` branch for tree-side changes.
2. Keep the canonical `Trace` schema model-neutral.
3. Do not use ToolBench `api_list` as ground truth.
4. Prefer adding adapters over rewriting the evaluator.
5. Preserve JSON report shape unless there is a strong reason to change it.
6. Add tests for every loader or metric edge case.
7. Keep live-run code separate from offline metric code.
8. Avoid adding heavy dependencies unless the model actually needs them.
9. If adding a graph adapter, use the same `evaluate_model()` contract.
10. If running expensive live model experiments, start with a tiny smoke run.

## Quick Commands

Check branch:

```bash
git branch --show-current
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run fixture smoke test:

```bash
PYTHONPATH=src python3 -m gnnvstree.cli \
  --toolbench tests/fixtures/toolbench_sample.json \
  --live-runs tests/fixtures/live_runs.json \
  --output-dir /tmp/gnnvstree-tree-smoke
```

Inspect smoke report:

```bash
python3 -m json.tool /tmp/gnnvstree-tree-smoke/report.json
```

Check worktree:

```bash
git status --short --branch
```

