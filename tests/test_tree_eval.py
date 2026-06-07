from __future__ import annotations

import unittest
from pathlib import Path

from gnnvstree.loaders import load_live_runs, load_toolbench_queries
from gnnvstree.metrics import evaluate_model
from gnnvstree.trace import make_trace
from gnnvstree.tree_model import TreeLogicAdapter


FIXTURES = Path(__file__).parent / "fixtures"


class TreeEvalTests(unittest.TestCase):
    def test_toolbench_loader_uses_relevant_apis_not_candidates(self) -> None:
        traces = load_toolbench_queries(FIXTURES / "toolbench_sample.json")
        first = traces[0]

        self.assertEqual(
            first.tool_names,
            (
                "Create_Container_Tracking_Get_Tracking_Data",
                "SQUAKE_Checkhealth",
            ),
        )
        self.assertIn("Distractor_Unused_API", first.candidate_tools)
        self.assertNotIn("Distractor_Unused_API", first.tool_names)

    def test_live_loader_reads_trace_and_metadata(self) -> None:
        traces = load_live_runs(FIXTURES / "live_runs.json")

        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0].tool_names[0], "browser_navigate")
        self.assertEqual(traces[0].semantic_steps, ("open_home", "inspect_table", "select_row", "extract_result"))
        self.assertEqual(traces[1].metadata["error"], "final answer missing")

    def test_tree_predicts_repeated_continuation(self) -> None:
        traces = [
            make_trace(source="unit", task_id="a", split="train", success=True, tool_names=["search", "open", "buy"]),
            make_trace(source="unit", task_id="b", split="train", success=True, tool_names=["search", "open", "save"]),
            make_trace(source="unit", task_id="c", split="train", success=True, tool_names=["search", "open", "buy"]),
        ]
        model = TreeLogicAdapter()
        model.fit(traces)

        prediction = model.predict_next(["search", "open"])

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.tool, "buy")
        self.assertGreater(prediction.confidence, 0.6)

    def test_evaluation_report_contains_rich_metrics(self) -> None:
        traces = load_toolbench_queries(FIXTURES / "toolbench_sample.json")
        model = TreeLogicAdapter()
        report = evaluate_model(model, traces[:2], traces[2:])
        data = report.to_dict()

        self.assertEqual(data["dataset"]["train"]["trace_count"], 2)
        self.assertIn("top1_accuracy", data["testing"])
        self.assertIn("mean_reciprocal_rank", data["testing"])
        self.assertIn("full_trace_exact_match_rate", data["sequence"])
        self.assertEqual(data["structure"]["framework"], "tree")


if __name__ == "__main__":
    unittest.main()
