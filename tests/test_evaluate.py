import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from evaluate import ROOT, load_tasks, parse_answer, rows_and_summary


class EvaluationTests(unittest.TestCase):
    def test_abstention_and_prose_are_not_silently_parsed(self):
        for answer in ["UNKNOWN", "There are 2 objects", "2 or 3", "-1", "2.0", "", None]:
            self.assertIsNone(parse_answer("count", answer))
        self.assertIsNone(parse_answer("empty", "Yes, it is empty"))
        self.assertEqual(parse_answer("count", " 2\n"), 2)
        self.assertIs(parse_answer("empty", "NO"), False)

    def test_failures_remain_in_accuracy_denominator(self):
        config = {"tasks": [{"id": "blue_wire", "title": "Blue wire", "kind": "count", "cases": [
            {"file": "a.png", "expected": 2}, {"file": "b.png", "expected": 1},
        ]}]}
        records = {
            ("blue_wire", "a.png"): {"text": "2"},
            ("blue_wire", "b.png"): {"error": "timeout"},
        }
        _, summary = rows_and_summary(config, records)
        self.assertEqual(summary[0]["accuracy"], 0.5)
        self.assertEqual(summary[0]["valid"], 1)

    def invoke(self, command, output, tasks=None):
        args = ["evaluate.py", command, "--output", str(output)]
        if tasks is not None:
            args.extend(["--tasks", str(tasks)])
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            return evaluate.main()

    def test_single_entry_sends_only_mapped_images_and_can_resume_and_report(self):
        import base64
        config = load_tasks(ROOT / "tasks.json")
        expected = [(task, case) for task in config["tasks"] for case in task["cases"]]
        answers = [
            {"text": ("YES" if case["expected"] else "NO") if task["kind"] == "empty" else str(case["expected"])}
            for task, case in expected
        ]
        health = {"ready": True, "busy": False}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("evaluate.request_json", side_effect=[health, *answers]) as api:
                self.assertEqual(self.invoke("run", output), 0)
            self.assertEqual(api.call_count, len(expected) + 1)  # health + configured requests
            for call, (task, case) in zip(api.call_args_list[1:], expected):
                payload = call.args[1]
                self.assertEqual(payload["question"], task["prompt"])
                self.assertEqual(payload["layout"], "single_image")
                self.assertEqual(len(payload["images"]), 1)
                self.assertEqual(base64.b64decode(payload["images"][0].split(",", 1)[1]),
                                 (ROOT / "images" / case["file"]).read_bytes())
            original_log = (output / "responses.jsonl").read_bytes()
            with patch("evaluate.request_json", return_value=health) as api:
                self.assertEqual(self.invoke("run", output), 0)
                self.assertEqual(api.call_count, 1)  # no inference requests on resume
            with patch("evaluate.request_json") as api:
                self.assertEqual(self.invoke("report", output), 0)
                api.assert_not_called()
            self.assertEqual((output / "responses.jsonl").read_bytes(), original_log)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(all(s["accuracy"] == 1 for s in summary))
            for name in ["report.html", "report.md", "results.csv"]:
                self.assertTrue((output / name).is_file())

    def test_request_failure_is_saved_and_stops_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("evaluate.request_json", side_effect=[{"ready": True}, TimeoutError("timeout")]) as api:
                self.assertEqual(self.invoke("run", output), 1)
                self.assertEqual(api.call_count, 2)
            records = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["error"], "timeout")
            self.assertTrue((output / "report.html").is_file())

    def test_unconfirmed_mapping_cannot_send_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads((ROOT / "tasks.json").read_text())
            config["mapping_confirmed"] = False
            path = Path(directory) / "unconfirmed.json"
            path.write_text(json.dumps(config))
            with patch("evaluate.request_json") as api, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.invoke("run", Path(directory) / "output", tasks=path)
                api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
