import json
from pathlib import Path

from PIL import Image

from gui_trace_evaluator.evaluator import TraceEvaluator


class FakeModel:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        joined = "\n".join(str(message.get("content")) for message in messages)
        self.calls.append(messages)
        if "CHECKPOINT_GENERATION" in joined:
            return json.dumps(
                {
                    "task_goal_rewrite": "Save the demo item.",
                    "checkpoints": [
                        {
                            "id": "cp1",
                            "description": "The item is saved.",
                            "required": True,
                            "evidence_hint": "Look for the save action and final screenshot.",
                        }
                    ],
                    "success_rule": "cp1 must pass.",
                }
            )
        if "STEP_RETRIEVAL" in joined and "Previous retrieval" not in joined:
            return json.dumps(
                {
                    "selected_steps": [1],
                    "trusted": False,
                    "confidence": 0.2,
                    "rationale": "The first pass is thin.",
                    "fallback_reason": "Need final screenshot.",
                }
            )
        if "STEP_RETRIEVAL" in joined:
            return json.dumps(
                {
                    "selected_steps": [1, 2],
                    "trusted": True,
                    "confidence": 0.9,
                    "rationale": "Repair selected action and final state.",
                    "fallback_reason": "",
                }
            )
        if "CHECKPOINT_EVALUATION" in joined:
            assert "before_ui" not in joined
            assert "after_ui" not in joined
            assert any(isinstance(message.get("content"), list) for message in messages)
            return json.dumps(
                {
                    "id": "cp1",
                    "achieved": True,
                    "confidence": 0.9,
                    "evidence": "Final screenshot supports the save.",
                    "missing_or_conflict": "",
                    "insufficient_trace": False,
                }
            )
        raise AssertionError(joined)


def test_evaluator_uses_model_repair_not_rule_fallback(tmp_path: Path):
    image = tmp_path / "step_002_after.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    record = {
        "task": "DemoTask",
        "goal": "Save a demo item.",
        "success": True,
        "trace": {
            "steps": [
                {
                    "step": 1,
                    "thinking": "Need to edit.",
                    "action": "type(content='demo')",
                    "summary": "Edited item.",
                    "before_ui": ["unused"],
                    "after_ui": ["unused"],
                },
                {
                    "step": 2,
                    "thinking": "Need to save.",
                    "action": "click(point='<point>100 100</point>')",
                    "summary": "Saved item.",
                    "after_screenshot_path": str(image),
                    "before_ui": ["unused"],
                    "after_ui": ["unused"],
                },
            ]
        },
    }

    evaluator = TraceEvaluator(FakeModel(), checkpoint_dir=tmp_path / "cache")
    result = evaluator.evaluate_record(record)

    assert result["status"] == "evaluated"
    assert result["success"] is True
    cp = result["checkpoint_results"][0]
    assert cp["retrieval_method"] == "model_step_retrieval_repair_v1"
    assert cp["evidence_steps"] == [1, 2]
    assert cp["evidence_images"] == 1


class UntrustedRepairModel(FakeModel):
    def complete(self, messages):
        joined = "\n".join(str(message.get("content")) for message in messages)
        self.calls.append(messages)
        if "CHECKPOINT_GENERATION" in joined:
            return json.dumps(
                {
                    "task_goal_rewrite": "Save the demo item.",
                    "checkpoints": [
                        {
                            "id": "cp1",
                            "description": "The item is saved.",
                            "required": True,
                            "evidence_hint": "Look for the save action and final screenshot.",
                        }
                    ],
                    "success_rule": "cp1 must pass.",
                }
            )
        if "STEP_RETRIEVAL" in joined:
            return json.dumps(
                {
                    "selected_steps": [2],
                    "trusted": False,
                    "confidence": 0.2,
                    "rationale": "The model is uncertain, but step 2 might contain the final state.",
                    "fallback_reason": "Need visual confirmation.",
                }
            )
        if "CHECKPOINT_EVALUATION" in joined:
            assert any(isinstance(message.get("content"), list) for message in messages)
            return json.dumps(
                {
                    "id": "cp1",
                    "achieved": True,
                    "confidence": 0.8,
                    "evidence": "The judge can still use the selected screenshot.",
                    "missing_or_conflict": "",
                    "insufficient_trace": False,
                }
            )
        raise AssertionError(joined)


def test_evaluator_lets_judge_decide_when_retrieval_stays_untrusted(tmp_path: Path):
    image = tmp_path / "step_002_after.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    record = {
        "task": "DemoTask",
        "goal": "Save a demo item.",
        "success": True,
        "trace": {
            "steps": [
                {
                    "step": 1,
                    "thinking": "Need to edit.",
                    "action": "type(content='demo')",
                    "summary": "Edited item.",
                },
                {
                    "step": 2,
                    "thinking": "Need to save.",
                    "action": "click(point='<point>100 100</point>')",
                    "summary": "Saved item.",
                    "after_screenshot_path": str(image),
                },
            ]
        },
    }

    evaluator = TraceEvaluator(UntrustedRepairModel(), checkpoint_dir=tmp_path / "cache")
    result = evaluator.evaluate_record(record)

    cp = result["checkpoint_results"][0]
    assert cp["achieved"] is True
    assert cp["retrieval_trusted"] is False
    assert cp["retrieval_method"] == "model_step_retrieval_repair_v1"
    assert cp["evidence_steps"] == [1, 2]
    assert cp["evidence_images"] == 1


class ToolCallingModel(FakeModel):
    def complete(self, messages):
        joined = "\n".join(str(message.get("content")) for message in messages)
        self.calls.append(messages)
        if "CHECKPOINT_GENERATION" in joined:
            return json.dumps(
                {
                    "task_goal_rewrite": "Move demo.txt from Alarms to Pictures.",
                    "checkpoints": [
                        {
                            "id": "cp1",
                            "description": "demo.txt is in Pictures and not in Alarms.",
                            "required": True,
                            "evidence_hint": "Use final state evidence.",
                        }
                    ],
                    "success_rule": "cp1 must pass.",
                }
            )
        if "STEP_RETRIEVAL" in joined:
            return json.dumps(
                {
                    "selected_steps": [1],
                    "trusted": True,
                    "confidence": 0.9,
                    "rationale": "The move action and final state are relevant.",
                    "fallback_reason": "",
                }
            )
        if "CHECKPOINT_EVALUATION" in joined and "READ_TOOL_RESULTS" not in joined:
            return json.dumps(
                {
                    "id": "cp1",
                    "achieved": False,
                    "confidence": 0.2,
                    "evidence": "The screenshot does not prove the final file location.",
                    "missing_or_conflict": "Need final source and destination listings.",
                    "insufficient_trace": True,
                    "needs_fallback_verification": True,
                    "read_requests": [
                        {
                            "tool": "list_dir",
                            "path": "/sdcard/Pictures",
                            "reason": "Check destination.",
                        }
                    ],
                }
            )
        if "CHECKPOINT_EVALUATION" in joined and "READ_TOOL_RESULTS" in joined:
            assert "demo.txt" in joined
            return json.dumps(
                {
                    "id": "cp1",
                    "achieved": True,
                    "confidence": 0.95,
                    "evidence": "Read-only tool result shows demo.txt in Pictures.",
                    "missing_or_conflict": "",
                    "insufficient_trace": False,
                    "needs_fallback_verification": False,
                    "read_requests": [],
                }
            )
        raise AssertionError(joined)


class FakeReadToolRunner:
    def __init__(self):
        self.requests = []

    def run_requests(self, record, checkpoint, requests):
        self.requests.append(requests)
        return [
            {
                "type": "read_tool_result",
                "tool": "list_dir",
                "status": "ok",
                "request": requests[0],
                "output": "demo.txt\n",
            }
        ]


def test_evaluator_runs_read_tools_when_judge_requests_fallback(tmp_path: Path):
    image = tmp_path / "step_001_after.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    record = {
        "task": "FilesMoveFile",
        "goal": "Move demo.txt from Alarms to Pictures.",
        "success": True,
        "trace": {
            "steps": [
                {
                    "step": 1,
                    "thinking": "Need to move the file.",
                    "action": "click(point='<point>100 100</point>')",
                    "summary": "Moved demo.txt.",
                    "after_screenshot_path": str(image),
                }
            ]
        },
    }

    tool_runner = FakeReadToolRunner()
    evaluator = TraceEvaluator(
        ToolCallingModel(),
        checkpoint_dir=tmp_path / "cache",
        read_tool_runner=tool_runner,
    )
    result = evaluator.evaluate_record(record)

    cp = result["checkpoint_results"][0]
    assert cp["achieved"] is True
    assert cp["read_tool_verification"]["triggered"] is True
    assert cp["read_tool_verification"]["requests"][0]["tool"] == "list_dir"
    assert tool_runner.requests


def test_stored_final_snapshot_does_not_suppress_requested_read_tools(tmp_path: Path):
    image = tmp_path / "step_001_after.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    record = {
        "task": "FilesMoveFile",
        "goal": "Move demo.txt from Alarms to Pictures.",
        "success": True,
        "post_execution_evidence": [
            {
                "type": "final_state_snapshot",
                "tool": "final_state_snapshot",
                "status": "ok",
                "request": {"tool": "final_state_snapshot"},
                "output": "Final UI text.",
            }
        ],
        "trace": {
            "steps": [
                {
                    "step": 1,
                    "thinking": "Need to move the file.",
                    "action": "click(point='<point>100 100</point>')",
                    "summary": "Moved demo.txt.",
                    "after_screenshot_path": str(image),
                }
            ]
        },
    }

    tool_runner = FakeReadToolRunner()
    evaluator = TraceEvaluator(
        ToolCallingModel(),
        checkpoint_dir=tmp_path / "cache",
        read_tool_runner=tool_runner,
    )
    result = evaluator.evaluate_record(record)

    cp = result["checkpoint_results"][0]
    verification = cp["read_tool_verification"]
    assert cp["achieved"] is True
    assert tool_runner.requests
    assert verification["source"] == "record_post_execution_evidence+live_read_tools"
    assert [item["tool"] for item in verification["results"]] == ["final_state_snapshot", "list_dir"]
