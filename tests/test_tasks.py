import unittest

from evaluate import ROOT, load_tasks, rows_and_summary


class TaskEvaluationTests(unittest.TestCase):
    def test_explicit_mapping_does_not_expand_to_all_images(self):
        config = load_tasks(ROOT / "tasks.json")
        pairs = [(t["id"], c["file"]) for t in config["tasks"] for c in t["cases"]]
        self.assertEqual(len(pairs), len(set(pairs)))
        empty = next(t for t in config["tasks"] if t["id"] == "empty")
        self.assertIn({"file": "image_03.png", "expected": False, "role": "非空反例"}, empty["cases"])
        square = next(t for t in config["tasks"] if t["id"] == "square_item")
        self.assertEqual([c["file"] for c in square["cases"]], ["image_03.png"])
        self.assertEqual(square["cases"][0]["expected"], 1)
        blue = next(t for t in config["tasks"] if t["id"] == "blue_wire")
        self.assertEqual([c["file"] for c in blue["cases"]], [
            "image_04.png", "image_05.png", "image_07.png", "image_10.png",
        ])
        black = next(t for t in config["tasks"] if t["id"] == "black_wire")
        self.assertEqual([c["file"] for c in black["cases"]], ["image_08.png"])

    def test_counterexample_false_empty_and_missing_answer_are_scored(self):
        config = {"tasks": [{"id": "empty", "title": "Empty", "kind": "empty", "cases": [
            {"file": "empty.png", "expected": True},
            {"file": "black_box.png", "expected": False, "role": "非空反例"},
            {"file": "missing.png", "expected": False, "role": "非空反例"},
        ]}]}
        rows, summary = rows_and_summary(config, {
            ("empty", "empty.png"): {"text": "YES"},
            ("empty", "black_box.png"): {"text": "YES"},
        })
        self.assertFalse(rows[1]["correct"])
        self.assertEqual(summary[0]["accuracy"], 1 / 3)
        self.assertEqual(summary[0]["false_empty"], 1)
        self.assertEqual(summary[0]["completed"], 2)
        self.assertEqual(summary[0]["nonempty_counterexample"], {"cases": 2, "correct": 0})


if __name__ == "__main__":
    unittest.main()
