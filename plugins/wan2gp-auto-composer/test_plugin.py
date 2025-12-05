import importlib
import unittest


plugin_module = importlib.import_module("plugins.wan2gp-auto-composer.plugin")
AutoVideoComposerPlugin = plugin_module.AutoVideoComposerPlugin


class AutoVideoComposerPluginTests(unittest.TestCase):
    def setUp(self):
        self.plugin = AutoVideoComposerPlugin()

    def test_split_prompt_preserves_order_and_groups_sentences(self):
        prompt = "Eerste zin. Tweede zin! Derde zin?"
        segments = 2

        result = self.plugin._split_prompt(prompt, segments)

        self.assertEqual(len(result), segments)
        self.assertEqual(result[0], "Eerste zin. Tweede zin!")
        self.assertEqual(result[1], "Derde zin?")

    def test_build_segments_plan_uses_min_frames_and_segment_metadata(self):
        plan = self.plugin._build_segments_plan(
            prompt="Een uitgebreid verhaal dat in delen wordt verteld.",
            total_minutes=0.5,
            segment_seconds=7,
            fps=10,
            min_frames=8,
            frame_step=2,
        )

        self.assertEqual(len(plan), 5)
        self.assertTrue(all(item["frames"] % 2 == 0 for item in plan))
        self.assertTrue(all(item["frames"] >= 8 for item in plan))
        self.assertTrue(plan[-1]["seconds"] >= 7)
        self.assertIn("Segment 1/5", plan[0]["prompt"])
        self.assertIn("Houd stijl, personages en camera gelijkmatig", plan[0]["prompt"])

    def test_create_segments_populates_queue_and_returns_summary_and_update(self):
        state = {"model_type": "type-a", "model_filename": "model.safetensors"}
        queue_store = {"queue": []}

        self.plugin.get_current_model_settings = lambda s: {
            "force_fps": "",
            "video_guide": None,
            "video_source": None,
        }
        self.plugin.get_base_model_type = lambda model_type: "base-type"
        self.plugin.get_computed_fps = lambda *args, **kwargs: 8.0
        self.plugin.get_model_min_frames_and_step = lambda *args, **kwargs: (8, 2, None)
        self.plugin.get_gen_info = lambda s: queue_store
        self.plugin.add_video_task = lambda **kwargs: queue_store["queue"].append(kwargs)
        self.plugin.update_queue_data = lambda queue: {"queue": queue}
        self.plugin.queue_html_container = object()

        summary, queue_update = self.plugin._create_segments(
            state=state,
            prompt="Een lange prompt met twee zinnen. Nog een stukje prompt.",
            total_minutes=0.25,
            segment_seconds=7,
        )

        self.assertIn("### Segmenten", summary)
        self.assertEqual(len(queue_store["queue"]), 3)
        self.assertEqual(queue_update, {"queue": queue_store["queue"]})
        self.assertEqual(queue_store["queue"][0]["model_type"], "type-a")
        self.assertEqual(queue_store["queue"][0]["model_filename"], "model.safetensors")
        self.assertIn("auto_video_composer", queue_store["queue"][0]["plugin_data"])


if __name__ == "__main__":
    unittest.main()
