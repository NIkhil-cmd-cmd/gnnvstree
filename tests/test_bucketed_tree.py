from __future__ import annotations

import unittest

from gnnvstree.bucketed_tree import BucketedActionTreeAdapter
from gnnvstree.buckets import BucketRouter
from gnnvstree.hf_toolbench import extract_actions_from_assistant
from gnnvstree.trace import make_trace


class BucketedTreeTests(unittest.TestCase):
    def test_bucket_router_groups_sample_tasks(self) -> None:
        traces = [
            make_trace(source="unit", task_id="w1", split="train", success=True, task_text="weather forecast rain", tool_names=["weather"]),
            make_trace(source="unit", task_id="w2", split="train", success=True, task_text="temperature forecast city", tool_names=["weather"]),
            make_trace(source="unit", task_id="b1", split="train", success=True, task_text="cheapest in stock book", tool_names=["book"]),
            make_trace(source="unit", task_id="b2", split="train", success=True, task_text="book price genre stock", tool_names=["book"]),
        ]
        router = BucketRouter(n_buckets=2, min_bucket_traces=1)
        router.fit(traces)

        weather = router.route("weather in boston tomorrow")
        book = router.route("find cheapest mystery book")

        self.assertEqual(router.bucket_count, 2)
        self.assertNotEqual(weather.bucket_id, book.bucket_id)

    def test_start_state_bucketed_trie_recommends_child(self) -> None:
        traces = [
            make_trace(
                source="unit",
                task_id="a",
                split="train",
                success=True,
                task_text="weather slack",
                tool_names=["get_weather", "post", "click"],
            ),
            make_trace(
                source="unit",
                task_id="b",
                split="train",
                success=True,
                task_text="weather slack",
                tool_names=["get_weather", "post"],
            ),
            make_trace(
                source="unit",
                task_id="c",
                split="train",
                success=True,
                task_text="weather details",
                tool_names=["get_weather", "click", "post"],
            ),
        ]
        model = BucketedActionTreeAdapter(n_buckets=1, min_bucket_traces=1)
        metrics = model.fit(traces)

        self.assertEqual(metrics["bucket_count"], 1)
        self.assertEqual(metrics["max_depth"], 3)

        prediction = model.recommend_next("weather slack", ["get_weather"])

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.tool, "post")
        self.assertGreater(prediction.confidence, 0.6)

    def test_hf_extractor_skips_blank_action_names(self) -> None:
        actions = extract_actions_from_assistant(
            """
Thought: do it
Action:
Action Input: {}
Thought: next
Action: real_tool
Action Input: {"x": 1}
"""
        )

        self.assertEqual([action.raw_name for action in actions], ["real_tool"])


if __name__ == "__main__":
    unittest.main()
