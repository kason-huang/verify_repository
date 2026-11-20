#!/usr/bin/env python3
"""Evaluate four tasks using only their mapped images; Python stdlib only."""
import argparse
import base64
import csv
import hashlib
import html
import json
import mimetypes
from pathlib import Path
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
ASTRA_MODEL = "gpt-6-astra"
ASTRA_REASONING_EFFORT = "low"
REGION = (
    "Look only at the load-bearing surface of the black tray in the foreground, "
    "in front of the green bin. "
    "Exclude the tray itself, its fixed silver screws and fixtures, all items in "
    "the green bin, and all items on the surrounding table. "
)
PROMPTS = {
    "empty": REGION + "Is the black tray empty of extra items? "
    "Answer only YES if empty, NO if it holds any extra item, or UNKNOWN if you cannot tell.",
    "square_item": REGION + "How many bagged square-shaped items are on the black tray? "
    "Count each bagged square-shaped item as one item. "
    "Answer only one non-negative integer, or UNKNOWN if you cannot tell.",
    "blue_wire": REGION + "How many bags of blue wire are on the black tray? "
    "Count each bag as one item, regardless of the number of loops or connectors. "
    "Include overlapping or stacked bags. "
    "Answer only one non-negative integer, or UNKNOWN if you cannot tell.",
    "black_wire": REGION + "How many bundles of black wire are on the black tray? "
    "Count each bundled wire as one item, regardless of the number of loops or connectors. "
    "Answer only one non-negative integer, or UNKNOWN if you cannot tell.",
}


def parse_answer(task, value):
    """Do not extract a stray number or silently turn abstention into empty."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if task == "empty":
        return {"YES": True, "NO": False}.get(value.upper())
    return int(value) if re.fullmatch(r"[0-9]{1,9}", value) else None


def request_json(url, payload=None, timeout=180):
    # Direct connection: inherited local proxy may not reach this service.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def request_astra(image_path, question, timeout):
    # Give each session a neutral image filename and a working directory without labels.
    with tempfile.TemporaryDirectory(prefix="tray-astra-") as directory:
        image = Path(directory) / f"input{image_path.suffix}"
        image.write_bytes(image_path.read_bytes())
        result = subprocess.run(
            [
                "codex", "exec", "--ignore-user-config", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only",
                "--disable", "shell_tool", "--disable", "multi_agent",
                "--disable", "view_image", "--disable", "image_generation",
                "--disable", "apps",
                "-c", 'web_search="disabled"',
                "-c", 'developer_instructions="Answer the user question from the attached image alone. Do not call tools."',
                "-C", directory, "-m", ASTRA_MODEL,
                "-c", f'model_reasoning_effort="{ASTRA_REASONING_EFFORT}"',
                "--json", "-i", str(image), "--", "-",
            ],
            input=question, capture_output=True, text=True, timeout=timeout,
        )
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    finished = [e for e in events if e["type"] == "turn.completed"]
    if result.returncode or not finished:
        failure = next((e.get("error", {}).get("message") for e in reversed(events)
                        if e["type"] == "turn.failed"), None)
        raise ValueError(f"Codex CLI did not finish: {failure or result.stderr.strip()[-1200:] or result.returncode}")
    answers = []
    warnings = []
    for event in events:
        if not event["type"].startswith("item."):
            continue
        item = event["item"]
        if item["type"] == "error":
            # Codex also emits nonfatal metadata and connection notices as error items.
            if event["type"] == "item.completed":
                warnings.append(item.get("message", "Codex CLI warning"))
            continue
        if item["type"] not in {"agent_message", "reasoning"}:
            raise ValueError(f'Unexpected Codex activity in image-only evaluation: {item["type"]}')
        if event["type"] == "item.completed" and item["type"] == "agent_message":
            answers.append(item["text"])
    if len(answers) != 1 or not answers[0].strip():
        raise ValueError("Expected one nonempty final Codex answer")
    return {"text": answers[0], "requested_model": ASTRA_MODEL,
            "usage": finished[-1].get("usage"), "backend": "codex_cli", "warnings": warnings}


def load_tasks(path):
    config = json.loads(path.read_text())
    seen_tasks = set()
    for task in config["tasks"]:
        if task["id"] in seen_tasks or task["kind"] not in {"empty", "count"}:
            raise ValueError("Duplicate task or unsupported task kind")
        seen_tasks.add(task["id"])
        task["prompt"] = PROMPTS[task["id"]]
        if not task["prompt"].isascii():
            raise ValueError("All model prompts must be in English")
        names = set()
        for case in task["cases"]:
            name, expected = case["file"], case["expected"]
            if name in names or Path(name).name != name:
                raise ValueError("Each task must have unique image basenames")
            names.add(name)
            if task["kind"] == "empty":
                if type(expected) is not bool:
                    raise ValueError("Empty task expects boolean truth")
            elif type(expected) is not int or expected < 0:
                raise ValueError("Count task expects a non-negative integer")
    return config


def rows_and_summary(config, records):
    rows, summaries = [], []
    for task in config["tasks"]:
        task_rows = []
        for case in task["cases"]:
            record = records.get((task["id"], case["file"]), {})
            prediction = parse_answer(task["kind"], record.get("text"))
            row = {
                "task": task["id"], "title": task["title"], "file": case["file"],
                "role": case.get("role", "目标样本"), "expected": case["expected"],
                "prediction": prediction,
                "correct": prediction is not None and prediction == case["expected"],
                "raw": record.get("text", ""), "error": record.get("error", ""),
                "seconds": record.get("seconds"),
            }
            task_rows.append(row)
        item = {
            "task": task["id"], "title": task["title"], "cases": len(task_rows),
            "completed": sum((task["id"], r["file"]) in records for r in task_rows),
            "valid": sum(r["prediction"] is not None for r in task_rows),
            "correct": sum(r["correct"] for r in task_rows),
            "accuracy": sum(r["correct"] for r in task_rows) / len(task_rows),
        }
        if task["kind"] == "empty":
            for name, truth in [("empty", True), ("nonempty_counterexample", False)]:
                subset = [r for r in task_rows if r["expected"] is truth]
                item[name] = {"cases": len(subset), "correct": sum(r["correct"] for r in subset)}
            item["false_empty"] = sum(r["expected"] is False and r["prediction"] is True for r in task_rows)
        rows.extend(task_rows)
        summaries.append(item)
    return rows, summaries


def display(value):
    if value is None:
        return "无有效回答"
    if type(value) is bool:
        return "YES（空）" if value else "NO（非空）"
    return str(value)


def report(output, config, records, images):
    rows, summaries = rows_and_summary(config, records)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 按任务匹配图片的评测", "", "每个请求只输入该任务对应的一张原图，英文提示词，不提供图片真值。", "",
             "判空同时包含空托盘和非空反例。反例作为独立测试输入，不作为提示词中的示范。", "",
             "未完成、UNKNOWN、格式错误和请求错误都不算正确；分母为该任务的全部指定图片。", "",
             "分类计数只使用对应类别图片，所得准确率仅反映这些选定样本，未覆盖其他类别的误报。", "",
             "| 任务 | 图片数 | 已完成 | 正确数 | 准确率 |", "|---|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f'| {s["title"]} | {s["cases"]} | {s["completed"]} | {s["correct"]} | {s["accuracy"]:.0%} |')
    counter_notes = []
    for s in summaries:
        if "nonempty_counterexample" in s:
            positive, negative = s["empty"], s["nonempty_counterexample"]
            note = (f'判空子集：空托盘正确 {positive["correct"]}/{positive["cases"]}；'
                    f'非空反例正确 {negative["correct"]}/{negative["cases"]}。')
            counter_notes.append(note)
            lines += ["", note]
    sections = []
    for task in config["tasks"]:
        lines += ["", f'## {task["title"]}', "", "```text", task["prompt"], "```", "",
                  "| 图片 | 用途 | 真值 | 原始回答/错误 | 正确 |", "|---|---|---|---|---|"]
        cards = []
        for row in (r for r in rows if r["task"] == task["id"]):
            raw = row["raw"] or row["error"] or "未请求"
            safe_raw = raw.replace("|", "\\|").replace("\n", " ")
            lines.append(f'| {row["file"]} | {row["role"]} | {display(row["expected"])} | {safe_raw} | {row["correct"]} |')
            path = images / row["file"]
            src = f'data:{mimetypes.guess_type(path.name)[0]};base64,' + base64.b64encode(path.read_bytes()).decode()
            prediction = row["prediction"]
            if type(prediction) is bool:
                repeats_prediction = raw.strip().upper() == ("YES" if prediction else "NO")
            else:
                repeats_prediction = prediction is not None and raw.strip() == str(prediction)
            detail = ""
            if row["error"]:
                detail = f'<pre>请求错误：{html.escape(row["error"])}</pre>'
            elif row["raw"] and not repeats_prediction:
                detail = f'<pre>原始回答：{html.escape(raw)}</pre>'
            elif not row["raw"]:
                detail = '<p>未收到回答</p>'
            judgment = "正确" if row["correct"] else "错误"
            cards.append(f'<article><h3>{html.escape(row["file"])} · {html.escape(row["role"])}</h3>'
                         f'<img src="{src}" alt="{html.escape(row["file"])}">'
                         f'<p>真值：{display(row["expected"])}；预测：{display(prediction)}</p>'
                         f'<p>判断：{judgment}</p>'
                         f'{detail}</article>')
        sections.append(f'<section id="{html.escape(task["id"])}"><h2>{html.escape(task["title"])}</h2>'
                        f'<details><summary>英文提示词</summary><pre>{html.escape(task["prompt"])}</pre></details>'
                        '<div class="grid">' + "".join(cards) + '</div></section>')
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    summary_html = '<table><tr><th>任务</th><th>正确/指定图片数</th><th>准确率</th></tr>' + ''.join(
        f'<tr><td>{html.escape(s["title"])}</td><td>{s["correct"]}/{s["cases"]}</td><td>{s["accuracy"]:.0%}</td></tr>' for s in summaries) + '</table>'
    (output / "report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>四项任务评测</title><style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:16px;background:#f4f6f8}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}'
        'article{background:white;padding:16px;border:1px solid #ddd;border-radius:10px}img{width:100%}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:10px;text-align:left}</style>'
        '<h1>按任务匹配图片的评测</h1><p>单张原图、英文提示词；判空包含独立的非空反例。</p>'
        '<p>计数仅测试对应类别图片；这是选定样本的小规模验证。无效或缺失回答计入准确率分母。</p>'
        + summary_html + ''.join(f'<p><b>{html.escape(note)}</b></p>' for note in counter_notes)
        + ''.join(sections) + '</html>', encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks.json")
    parser.add_argument("--backend", choices=["motvlm", "astra"], default="motvlm")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--images", type=Path, default=ROOT / "images")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.output is None:
        directory = "tasks-astra" if args.backend == "astra" else "tasks-wire-v2"
        args.output = ROOT / "results" / directory
    config = load_tasks(args.tasks)
    if not config.get("mapping_confirmed"):
        parser.error("Task/image mapping is pending clarification; no model requests submitted")
    hashes = {case["file"]: hashlib.sha256((args.images / case["file"]).read_bytes()).hexdigest()
              for task in config["tasks"] for case in task["cases"]}
    snapshot = {"tasks_config": config, "image_sha256": hashes}
    if args.backend == "astra":
        snapshot["backend"] = {"name": "codex_cli", "requested_model": ASTRA_MODEL,
                               "reasoning_effort": ASTRA_REASONING_EFFORT}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("Task configuration or images changed; use a new output directory")
    elif args.command == "report":
        raise ValueError("No existing run to report")
    else:
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    log = args.output / "responses.jsonl"
    records = {}
    if log.exists():
        for line in log.read_text().splitlines():
            record = json.loads(line)
            records[(record["task"], record["file"])] = record
    if args.command == "run":
        if args.backend == "motvlm":
            health = request_json(config["base_url"] + "/health", timeout=min(args.timeout, 15))
            (args.output / "health.json").write_text(json.dumps(health, indent=2))
            if not health.get("ready") or health.get("busy"):
                raise RuntimeError("Service not ready or busy")
        for task in config["tasks"]:
            for case in task["cases"]:
                key = (task["id"], case["file"])
                if key in records:
                    continue
                image_path = args.images / case["file"]
                record = {"task": key[0], "file": key[1], "started_at": datetime.now(timezone.utc).isoformat()}
                started = time.monotonic()
                try:
                    if args.backend == "astra":
                        response = request_astra(image_path, task["prompt"], args.timeout)
                    else:
                        mime = mimetypes.guess_type(image_path.name)[0]
                        data = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode()
                        response = request_json(config["base_url"] + "/generate", {
                            "question": task["prompt"], "images": [data], "layout": "single_image",
                            "max_new_tokens": config["max_new_tokens"],
                        }, timeout=args.timeout)
                    record["response"] = response
                    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
                        raise ValueError("Missing model answer")
                    if response.get("output_limit_reached"):
                        raise ValueError("Truncated model answer")
                    record["text"] = response["text"]
                except urllib.error.HTTPError as error:
                    record["error"] = f"HTTP {error.code}: {error.read(4096).decode(errors='replace')}"
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    record["error"] = str(error)
                record["seconds"] = round(time.monotonic() - started, 3)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records[key] = record
                print(f'{key}: {record.get("text", record.get("error"))}', flush=True)
                if "error" in record:
                    report(args.output, config, records, args.images)
                    print("Stopped after request error; no automatic retries.")
                    return 1
    summary = report(args.output, config, records, args.images)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
