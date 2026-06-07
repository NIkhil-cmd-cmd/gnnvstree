# Bucketed Action Tree Design Plan

Status: planning document for later implementation.

Branch context: this repo's tree-side work is on the `tree` branch. The updated
remote graph branch, `origin/gnn`, was inspected on June 4, 2026. It now
contains a `gnn/` package with useful reference code, but it does not contain
committed data artifacts and its README references data modules that are not
present in the branch.

## 1. Problem Statement

The current tree-side harness models traces as ordered tool names. That is a
useful baseline, but it is too coarse for the intended workflow-learning
experiment.

Two calls to the same tool can mean different things:

```text
web_search({"query": "weather"})
web_search({"query": "weather tomorrow"})
```

At the tool-name level, both are `web_search`. At the action level, they are
different state transitions with different intent and different likely
follow-up actions. The new model should learn over action calls, where an
action is the tool/API/function name plus its normalized input.

The target structure is:

```text
START
  bucket_0
    get_weather({"city":"sf"})
      post({"channel":"alerts"})
        click({"selector":"confirm"})
          poop({})
      click({"selector":"details"})
        post({"channel":"alerts"})
  bucket_1
    ...
```

Each bucket represents a cluster of semantically similar tasks. Inside a
bucket, successful agent traces form a trie of action calls. The tree should
support next-action recommendation, pruning of unhelpful branches, and metrics
that explain when the system is learning reusable workflow structure versus
memorizing traces.

## 2. Goals

The implementation should provide:

- A bucketed action tree trained from many action-level traces.
- Action identity that includes both call name and input.
- Task-embedding buckets from K-means or an equivalent deterministic clustering
  strategy.
- High-recall relevance scoring so the recommender avoids obvious wrong paths
  without filtering out too many useful actions.
- Pruning for low-value, redundant, dead-end, no-op, or failed branches.
- A fresh agent harness that can run free-tier Gemini models with the
  recommender in the loop and log every agent trace.
- Offline replay evaluation before live or real-tool execution.

Non-goals for the first implementation:

- Do not depend on the graph branch as a runnable package.
- Do not require live MCP/browser agents for the first benchmark.
- Do not require paid API access.
- Do not delete pruned tree structure; preserve it for inspection.

## 3. Data Model

### ActionCall

`ActionCall` is the central unit of the new tree.

Fields:

- `name`: normalized function/tool/API name.
- `raw_name`: original name from the trace.
- `input`: parsed input object, preferably a dictionary.
- `raw_input`: original input payload when parsing is lossy.
- `canonical_input_json`: stable JSON serialization with sorted keys.
- `input_hash`: short hash of `canonical_input_json`.
- `action_key`: stable identity key, usually `name + ":" + input_hash`.
- `display`: readable compact representation for reports.
- `metadata`: optional source-specific details.

Examples:

```json
{
  "name": "web_search",
  "input": {"query": "weather"},
  "action_key": "web_search:4ad0c8c1"
}
```

```json
{
  "name": "web_search",
  "input": {"query": "weather tomorrow"},
  "action_key": "web_search:90cb5fd2"
}
```

Those two examples must be treated as different actions.

Canonicalization requirements:

- Sort object keys.
- Preserve primitive values.
- Normalize whitespace in strings.
- Avoid dropping parameters unless a source-specific normalizer explicitly does
  so.
- Hash the canonical JSON, not the raw input string.

### ActionTrace

`ActionTrace` is the action-level extension of the current canonical `Trace`.

Fields:

- `trace_id`
- `source`
- `task_id`
- `task_text`
- `success`
- `split`
- `actions`: ordered `ActionCall` values.
- `observations`: optional output/observation per action.
- `candidate_actions`: available tools/actions for the task.
- `ground_truth_actions`: known relevant or gold actions when available.
- `task_embedding`: optional in-memory vector or embedding reference.
- `bucket_id`: assigned after clustering.
- `metadata`

The existing `Trace.ToolCall.input` field already gives a foundation for this,
so the implementation should extend the existing schema carefully instead of
replacing it wholesale.

### BucketNode

Each task bucket is a child of the global start node.

Fields:

- `bucket_id`
- `centroid`
- `trace_count`
- `successful_trace_count`
- `avg_trace_len`
- `median_trace_len`
- `tool_vocab_size`
- `action_vocab_size`
- `root`: root `ActionTreeNode` for this bucket.
- `metadata`

Bucket metrics matter because the tree is only meaningful when buckets have
enough traces and enough repeated action structure.

### ActionTreeNode

Each node represents one concrete action call at a specific prefix position.

Fields:

- `node_id`
- `action_key`
- `action`
- `children`: mapping from child action key to node.
- `visit_count`
- `successful_visit_count`
- `failed_visit_count`
- `terminal_success_count`
- `terminal_failure_count`
- `score`
- `confidence`
- `depth`
- `pruned`
- `prune_reasons`
- `edge_stats`

`edge_stats` should store child-specific evidence:

- support count
- successful support count
- failed support count
- downstream success rate
- no-op count if observations are available
- LLM usefulness score if configured

### PredictionResult

Recommendation output should be inspectable:

- `action`
- `bucket_id`
- `bucket_confidence`
- `tree_confidence`
- `relevance_score`
- `action_success_score`
- `final_score`
- `decision`: `accept`, `suggest`, or `abstain`
- `alternatives`
- `reasons`

### Trace Log Schema

The agent harness should log both agent behavior and recommender behavior.

Fields:

- `episode_id`
- `task_id`
- `condition`
- `model`
- `task_text`
- `available_actions`
- `step_index`
- `current_prefix`
- `recommended_action`
- `recommendation_scores`
- `agent_action`
- `agent_input`
- `accepted_recommendation`
- `observation`
- `success`
- `tokens`
- `latency_ms`
- `error`

The logger should write JSONL so interrupted runs preserve partial results.

## 4. Training Pipeline

### Data Source Priority

Primary source: Hugging Face ToolBench conversations.

Target dataset:

- Dataset: `tuandunghcmut/toolbench-v1`.
- Use G2/G3-style multi-tool traces first.
- Target at least 1,000 extracted action traces.
- If extraction quality is poor, use at least 300 high-confidence traces from a
  single coherent ToolBench group/category.

Reasoning:

- The local repo has only a tiny ToolBench fixture.
- The graph branch does not commit generated ToolBench data.
- StableToolBench query files are useful for candidate APIs and relevance
  labels, but query-level `relevant APIs` usually lack concrete action inputs.
- Conversation traces are more likely to contain action names, inputs, and
  observations.

Secondary source: official ToolBench answer traces.

The official OpenBMB ToolBench release can provide better ordered answer
artifacts if manually downloaded. The implementation should include an optional
loader hook for those files, but it should not block the first version.

### Extraction

The HF conversation loader should:

- Read rows from configured splits.
- Extract task text from the user query.
- Extract ordered tool/function/API calls from assistant/function turns.
- Parse JSON function-call arguments where possible.
- Attach function/tool observations to the preceding action where available.
- Preserve raw text for debugging parse failures.
- Emit skip reasons for ambiguous or unparseable rows.

Expected extraction report:

- rows scanned
- traces extracted
- extraction success rate
- skipped row count
- skip reason distribution
- parsed-input rate
- average actions per trace
- p50/p90 actions per trace
- unique action count
- unique tool count
- duplicate trace rate

### Embeddings

Use a pluggable embedding provider.

Providers:

- `HashEmbeddingProvider`: deterministic, local, test-only.
- `SentenceTransformerEmbeddingProvider`: local model, good default for
  offline development.
- `OpenAIEmbeddingProvider` or `GeminiEmbeddingProvider`: optional real-quality
  provider when keys are available.

All embeddings should be cached by normalized task text:

```text
embedding_cache_key = sha256(provider_name + normalized_task_text)
```

The cache should allow reruns without re-embedding.

### Bucketing

Use K-means on task embeddings.

Default K behavior:

- Start with `K = 32`, matching the graph branch config.
- If trace count is small, use `sqrt(num_traces / 2)` as a starting point.
- Clamp to a sane range, for example `4 <= K <= 64`.
- Enforce minimum bucket support:
  - `min_bucket_traces = max(25, 5 * median_trace_len)`.
  - If too many buckets are under-supported, reduce K and rerun.
- Report the percentage of traces in supported buckets.

Rationale:

Buckets with too few traces will produce brittle trees. The bucket count should
be influenced by trace volume, average trace length, and action vocabulary size.
Longer traces need more support because they create more branching and more
possible prefixes.

### Tree Insertion

For each successful trace:

1. Assign task to a bucket.
2. Start at that bucket root.
3. For every action in order:
   - create or reuse the child node for that `action_key`;
   - increment visit counts;
   - increment success counts because the full trace succeeded;
   - attach observation/usefulness metadata when available.
4. Mark terminal success at the final node.

Failed traces:

- Should not raise success scores.
- Should still inform pruning and dead-end statistics.
- Should be optional for v1 if failure labels are unreliable.

## 5. Scoring

Every node should store raw counts and computed scores.

Default node score:

```text
base_success_rate = successful_visit_count / max(visit_count, 1)
support_weight = min(1.0, log1p(visit_count) / log1p(support_saturation))
node_score = base_success_rate * support_weight
```

Default `support_saturation`: 20 visits.

Child confidence:

```text
child_confidence = child.node_score / sum(unpruned_sibling_scores)
```

If sibling score sum is zero, fall back to visit-count frequency.

Action success score:

```text
action_success_score = successful_edge_count / max(edge_count, 1)
```

The scoring system should keep raw counts in serialization so scoring can be
changed without rebuilding the tree from traces.

## 6. Relevance Design

### Feasibility

A cheap relevance metric is feasible, but it is an estimate. Before executing a
tool, the system cannot know with certainty whether the call will progress the
agent's state. It can estimate whether the action is task-relevant. After
execution, it can evaluate whether the observation was useful.

This distinction matters:

- Pre-call relevance answers: "Does this action plausibly belong to this task?"
- Post-call usefulness answers: "Did this action produce information that
  helped downstream progress?"

The first should be high recall and cheap. The second can be stricter and can
drive pruning.

### Relevance Signals

Use a `RelevanceScorer` with multiple weak signals:

1. Candidate availability
   - If the task provides a candidate tool/action list, prefer only those.
   - Hard-block only unavailable actions in closed-candidate mode.

2. BM25/text match
   - Compare task text against action name, tool name, API name, description,
     parameter names, and action input values.
   - Cheap and explainable.

3. Embedding similarity
   - Compare task embedding to action/tool description embedding.
   - Cache action description embeddings.

4. Bucket support
   - Boost actions that appear in the assigned bucket with enough support.

5. Prefix history
   - Boost actions that historically follow the current prefix or suffix.

6. Argument validity
   - Penalize missing required parameters, impossible enum values, or malformed
     inputs.

7. Post-call state novelty
   - After execution or replay, evaluate whether observation adds new entities,
     fields, URLs, IDs, search results, or other usable state.

### High-Recall Policy

False negatives are worse than false positives for this layer. A relevant tool
that gets filtered can block the whole workflow. Therefore:

- Use relevance primarily as a reranker, not a hard gate.
- Hard-block only clear invalid choices.
- Keep a permissive default threshold, for example `0.20`.
- Track false-negative examples in reports.

Recommended final score:

```text
final_score =
  0.50 * tree_confidence +
  0.25 * relevance_score +
  0.15 * bucket_assignment_confidence +
  0.10 * action_success_score
```

Default decision thresholds:

```text
accept_threshold = 0.65
suggest_threshold = 0.35
min_relevance_threshold = 0.20
```

Decision behavior:

- `accept`: final score high enough and relevance passes.
- `suggest`: plausible recommendation, but not strong enough to force.
- `abstain`: not enough support or relevance.

## 7. Pruning Design

Pruning removes low-value branches from recommendation behavior, but should not
delete evidence.

Rules:

- Keep pruned nodes in memory and serialization.
- Set `pruned=true`.
- Record one or more prune reasons.
- Ignore pruned nodes during recommendation unless debug mode asks to include
  them.

### Heuristic Pruning

Always available.

Candidate prune reasons:

- `low_support`: edge/node has fewer than the configured minimum visits.
- `high_failure_rate`: branch has materially worse downstream success than the
  bucket baseline.
- `dead_end`: branch frequently terminates before task success.
- `no_op_observation`: observation is empty, duplicate, or unchanged.
- `redundant_action`: same downstream successful path appears without this
  action at higher support.
- `invalid_action`: repeated validation failures.
- `low_relevance`: consistently low relevance to tasks in this bucket.

### Optional LLM Usefulness Judge

Use only when configured and cache every judgment.

Input:

- task text
- action name
- action input
- action observation
- next action or final answer

Output:

- usefulness score from `0.0` to `1.0`
- short reason

Default pruning policy:

- Do not let the LLM alone prune an edge in v1.
- Prune when a heuristic already flags the edge and LLM usefulness is below
  `0.30`.

This avoids expensive or unstable judgments becoming the primary source of
truth.

## 8. Prediction And Recommendation

Prediction flow:

1. Embed the task or load cached task embedding.
2. Assign the task to the nearest bucket.
3. Walk the exact action prefix in that bucket's trie.
4. If exact prefix misses, use suffix fallback inside the same bucket.
5. Rank child actions by tree score.
6. Apply candidate-action filtering.
7. Compute relevance for candidates.
8. Compute final score.
9. Return `accept`, `suggest`, or `abstain`.

Fallback behavior:

- If bucket is unsupported, abstain or fall back to global action statistics.
- If prefix misses, use longest suffix match.
- If suffix misses, use bucket root child distribution.
- If bucket has no evidence, abstain.

Candidate filtering:

- Closed-candidate mode: never recommend outside available candidate actions.
- Open mode: allow any learned action, but mark out-of-candidate recommendations
  clearly.

The default for ToolBench replay should be closed-candidate mode.

## 9. Agent Harness

Build a fresh harness instead of reusing `chorus` directly.

Why not direct reuse:

- `chorus` is Claude-Agent-SDK-specific.
- It allows coding/file tools, which is not the ToolBench action environment.
- It logs useful tool calls, but it is not built around replaying known
  ToolBench observations.
- It is a good reference, not an implementation target.

### Components

`AgentRunner`

- Runs one episode.
- Owns task prompt, available actions, current state, model client,
  recommender, replay environment, and logger.

`GeminiClient`

- Uses `google-genai`.
- Default model: `gemini-2.5-flash-lite`.
- Fallback models:
  - `gemini-2.5-flash`
  - `gemini-2.0-flash-lite`
- Enforces free-tier guardrails:
  - require `GEMINI_API_KEY` or `GOOGLE_API_KEY`;
  - reject Vertex env vars like `GOOGLE_CLOUD_PROJECT`;
  - reject Pro models;
  - throttle requests;
  - enforce daily call budget;
  - write partial results when quota fails.

`ReplayToolEnvironment`

- Replays ToolBench observations when the agent follows a known action from a
  trace.
- Returns controlled "not useful" observations for valid but unknown actions.
- Returns validation errors for unavailable or malformed actions.
- Logs hallucinations.

`TraceLogger`

- Writes JSONL.
- Flushes each step.
- Allows partial runs to be evaluated if interrupted.

### Harness Conditions

Baseline:

- Gemini sees task and candidate actions.
- No recommender signal.

Tree-suggested:

- Gemini sees task, candidate actions, and recommender top action/alternatives.
- Gemini can accept or ignore the recommendation.

Tree-gated:

- If recommender returns `accept`, the harness strongly instructs or forces the
  top action.
- If recommender returns `suggest`, the model sees the recommendation.
- If recommender returns `abstain`, the model acts normally.

Dry-run:

- No API call.
- Deterministic agent picks top-ranked action.
- Used for CI and unit tests.

## 10. Graph Branch Findings

Fetched `origin/gnn` on June 4, 2026. It moved from the initial commit to
`3f42a81 add data`.

Files now present include:

- `gnn/README.md`
- `gnn/configs/cpu_g2_g3.yaml`
- `gnn/src/agent/gemini_agent.py`
- `gnn/src/eval/metrics.py`
- `gnn/src/eval/retrievers.py`
- `gnn/src/eval/run_benchmark.py`
- `gnn/src/infer/retrieve_tools.py`
- `gnn/src/train/train_gnn.py`
- `gnn/src/utils/db.py`
- `gnn/src/utils/embeddings.py`

Useful references:

- Gemini free-tier model allowlist and budget guard.
- Retrieval metrics such as precision, recall, MRR, NDCG, ordered path match,
  workflow coverage, and next-tool accuracy.
- BM25 and full-list baselines.
- Config defaults:
  - G2/G3 HF splits.
  - 32 buckets.
  - top-k 5.
  - Gemini eval sample count 20.
  - Gemini request interval 4 seconds.

Important limitations:

- No data artifacts are committed.
- No SQLite graph artifact is committed.
- No checkpoint is committed.
- `gnn/README.md` references `gnn/src/data/download_toolbench`,
  `gnn/src/data/build_workflow_graphs`, and `gnn/src/data/bucket_kmeans`, but
  those files are not present in the fetched branch.
- The graph branch should be treated as reference material, not a dependency.

## 11. Metrics

### Data Quality Metrics

- rows scanned
- traces extracted
- extraction success rate
- skip reason distribution
- parsed-input action rate
- average trace length
- median trace length
- p90 trace length
- unique tools
- unique actions
- duplicate action trace rate
- source/split/category distribution

### Bucket Metrics

- bucket count
- supported bucket count
- traces per bucket
- average trace length per bucket
- unique actions per bucket
- branch factor per bucket
- bucket assignment confidence
- under-supported bucket rate

### Tree Metrics

- node count
- edge count
- max depth
- average depth
- leaf count
- pruned node count
- pruned edge count
- prune reason distribution
- model JSON size
- top actions
- top transitions

### Prediction Metrics

- top-1/top-3/top-5 next-action accuracy
- mean reciprocal rank
- coverage rate
- abstention rate
- accuracy when predicted
- bucket-conditioned accuracy
- suffix fallback hit rate
- candidate-filtered recommendation rate
- wrong-path rate

### Relevance Metrics

- relevance precision
- relevance recall
- relevance recall at threshold
- gold-action false-negative count
- gold-action false-negative examples
- average relevance score by successful/failed action
- hard-block reason distribution

### Agent Metrics

- episode success
- path recall
- ordered path match
- workflow coverage
- hallucination rate
- unnecessary action calls
- recommendation acceptance rate
- accepted recommendation success rate
- rejected recommendation success rate
- steps to completion
- token count
- latency
- quota failures

## 12. Validation Plan

### Unit Tests

Action identity:

- Same tool with different inputs must produce different action keys.
- Same input with different key order must produce the same action key.
- String whitespace normalization should be deterministic.

Bucket assignment:

- Deterministic hash embeddings produce stable buckets.
- Auto-K reduces bucket count when buckets are under-supported.
- Bucket metrics report under-supported buckets.

Trie behavior:

- Inserting:

```text
A: get_weather -> post -> click -> poop
B: get_weather -> click -> post
```

produces:

```text
bucket
  get_weather
    post
      click
        poop
    click
      post
```

Prediction:

- Exact prefix prediction works.
- Suffix fallback works.
- Candidate filtering blocks unavailable actions.
- Pruned nodes are ignored by default.
- Abstention happens below threshold.

Relevance:

- Obvious matching action receives high relevance.
- Obvious non-matching action receives lower relevance.
- Gold actions are not filtered under permissive defaults.
- Invalid required args can be hard-blocked.

Pruning:

- Low-support edges can be marked pruned.
- High-failure branches can be marked pruned.
- Pruned nodes remain serialized.

### Loader Tests

- HF-style conversation fixture extracts ordered actions and parsed inputs.
- Function observation attaches to the preceding action.
- Ambiguous rows are skipped with reasons.
- Official answer-trace fixture extracts ordered API calls.
- Query-only ToolBench data remains weak supervision and does not treat
  `api_list` as ground truth.

### Harness Tests

- Dry-run agent writes trace logs.
- Replay environment returns recorded observations for gold actions.
- Invalid action increments hallucination metrics.
- Tree-suggested condition logs accepted and rejected recommendations.
- Tree-gated condition accepts above threshold and abstains below threshold.
- Gemini guard rejects Pro models and Vertex billing env vars without making
  network calls.

### Smoke Runs

1. Synthetic clustered action traces.
2. Small HF ToolBench sample.
3. Five dry-run replay episodes.
4. Optional two-episode Gemini smoke when `GEMINI_API_KEY` is present.

## 13. Risks And Mitigations

ToolBench conversation parsing may be noisy.

- Mitigation: emit extraction quality reports, keep raw text for failures, and
  require high-confidence traces for first training runs.

Relevance may reject useful actions.

- Mitigation: use permissive thresholds, treat relevance as reranking, and
  report false negatives aggressively.

Buckets may be under-supported.

- Mitigation: auto-reduce K, report bucket health, and abstain for weak buckets.

Action inputs can explode vocabulary size.

- Mitigation: canonicalize inputs, optionally normalize volatile fields, and
  report unique input fingerprints per action.

Gemini free quota may be unstable.

- Mitigation: dry-run agent for CI, small default sample size, throttling,
  fallback models, and partial-result writes.

Replay is not real execution.

- Mitigation: treat replay as the first validation stage. Live MCP/browser
  harness should come only after replay results are stable.

Graph branch may continue changing.

- Mitigation: copy or reimplement only stable ideas. Do not import from the
  graph branch unless it becomes a shared package.

## 14. Implementation Phases

### Phase 1: Schemas And Synthetic Tests

- Add `ActionCall` and `ActionTrace` helpers.
- Add canonical input hashing.
- Add synthetic fixtures.
- Add action-trie insertion and serialization tests.

### Phase 2: Data Extraction

- Add HF ToolBench conversation loader.
- Add extraction reports.
- Add optional official answer-trace loader.
- Add data quality metrics.

### Phase 3: Bucketing And Tree Training

- Add embedding provider interface.
- Add embedding cache.
- Add K-means bucketing with auto-K support constraints.
- Train one trie per bucket.
- Add tree metrics.

### Phase 4: Relevance And Pruning

- Add BM25/text relevance.
- Add embedding relevance.
- Add candidate and argument validity checks.
- Add heuristic pruning.
- Add optional cached LLM usefulness judge.

### Phase 5: Prediction API

- Add `recommend_next`.
- Add exact prefix and suffix fallback.
- Add final score calculation.
- Add `accept`, `suggest`, and `abstain` decisions.

### Phase 6: Agent Harness

- Add `GeminiClient` with free-tier guards.
- Add `ReplayToolEnvironment`.
- Add `AgentRunner`.
- Add JSONL trace logging.
- Add baseline, tree-suggested, tree-gated, and dry-run modes.

### Phase 7: Reports And Comparison

- Extend JSON reports while preserving existing top-level report shape where
  possible.
- Add action-tree sections under existing dataset/training/testing/sequence/
  structure concepts.
- Add agent report section.
- Add comparison report between baseline and recommender-assisted runs.

## 15. Acceptance Criteria For First Complete Version

- At least 300 high-confidence action traces can be extracted locally or from
  HF ToolBench.
- Action identity includes input and passes canonicalization tests.
- Bucketed tree can train and serialize.
- Recommender can return next action, alternatives, scores, and abstention.
- Relevance scorer reports high recall on known gold actions.
- Pruning marks low-value branches without deleting them.
- Dry-run replay harness can compare baseline, suggested, and gated modes.
- Gemini smoke run is optional and guarded by API key plus budget settings.
- Reports include enough data quality and model quality metrics to diagnose
  whether failures come from data extraction, weak buckets, relevance gating,
  pruning, or agent behavior.

## 16. Experiment Log

### HF ToolBench Tree Training

Initial ToolBench extraction confirmed that the dataset has enough usable
assistant action traces for a first action-tree prototype.

- `limit=1000` suffix-prefix baseline:
  - usable traces: 781
  - parsed actions: 2,723
  - unique action strings: 1,867
  - max depth: 6 because that baseline inserts bounded suffix windows
- `limit=5000`, bucketed start-state trie, `k=16`:
  - usable traces: 3,933
  - parsed actions: 13,593
  - unique action strings: 7,617
  - supported buckets: 16/16
  - max depth: 9
  - max branch factor: 1,262

This validated that ToolBench can provide enough traces, but it also exposed
the main sparsity problem: action identity includes inputs, so exact action
strings become very high-cardinality quickly.

### Bucket Sweep

An offline heldout sweep trained on 5k ToolBench rows and evaluated on heldout
trace prefixes.

| Buckets | Coverage | Accuracy When Recommended | Start Accuracy |
| --- | ---: | ---: | ---: |
| 8 | 19.5% | 51.0% | 48.0% |
| 16 | 16.8% | 51.1% | 47.7% |
| 32 | 14.9% | 55.1% | 52.7% |
| 64 | 14.7% | 51.9% | 48.2% |

`k=32` looked best for precision, but coverage dropped as buckets became more
specific. The result supports using sharper buckets only with parent/global
fallbacks.

### Gemini Replay Smoke

A one-episode Gemini replay smoke with tree recommendations showed that the
tree can sometimes recommend a correct first action, but it also showed a bad
failure mode: a one-count self-loop in a sparse bucket was treated as high
confidence after the first action. This should be fixed with support thresholds,
route-confidence gating, and fallback logic before relying on live agent runs.

### BM25 + Gemini Partial Run

A larger BM25-guided Gemini run was attempted with:

- training limit: 5,000
- heldout offset: 1,000
- episodes requested: 8
- max steps: 4
- conditions: `baseline`, `bm25_suggested`
- distractors: 12 per trace
- output directory: `artifacts/gemini-bm25-large-8ep`

The run stalled on slow Gemini responses and was stopped after 18 completed
step rows across 6 tasks. Raw artifacts are local-only and intentionally ignored
by git; the important partial metrics were:

| Condition | Steps | Next-Action Accuracy | Hallucination Rate | Recommendation Acceptance | Recommendation Correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8 | 25.0% | 0.0% | n/a | n/a |
| bm25_suggested | 10 | 40.0% | 0.0% | 60.0% | 50.0% |

This is not strong evidence. The sample is too small, Gemini latency was poor,
and BM25 often selected semantically related but workflow-wrong actions. The
clearest failure was that BM25 is task-level relevance, not state-aware
workflow prediction; it can repeat already-used actions or choose a tool that
sounds right but is not the next step.

The resulting validation policy is:

- Do not spend large Gemini budgets until offline replay is stable.
- First run 500-1,000 trace offline comparisons for random, BM25, tree, BM25
  with used-action suppression, and BM25/tree hybrid.
- Report exact action accuracy, tool-family accuracy, start accuracy,
  post-prefix accuracy, coverage, and abstention.
- Only run Gemini on the best one or two offline configurations.
