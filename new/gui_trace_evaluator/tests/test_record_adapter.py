from pathlib import Path

from gui_trace_evaluator.record_adapter import normalize_record, step_to_prompt_dict


def test_normalize_record_uses_generic_fields_and_preserves_ui(tmp_path: Path):
    image = tmp_path / "step_001_after.jpg"
    image.write_bytes(b"image")
    record = {
        "task": "DemoTask",
        "goal_used": "Do the thing.",
        "reward": 1.0,
        "trace": {
            "steps": [
                {
                    "action_output": "Thought: Need to tap save.\nAction: click(point='<point>1 2</point>')",
                    "summary": "Saved.",
                    "after_screenshot_path": "step_001_after.jpg",
                    "before_ui": ["Title", "Delete"],
                    "after_ui": "[1] text='Done'",
                }
            ]
        },
    }

    normalized = normalize_record(record, base_dir=tmp_path)

    assert normalized.goal == "Do the thing."
    assert normalized.official_success is True
    assert normalized.steps[0].thinking == "Need to tap save."
    assert normalized.steps[0].action == "click(point='<point>1 2</point>')"
    assert normalized.steps[0].after_screenshot_path == str(image)
    assert "Title" in normalized.steps[0].before_ui
    prompt_step = step_to_prompt_dict(normalized.steps[0])
    assert "Done" in prompt_step["after_ui_text"]


def test_normalize_record_rebases_stale_absolute_trace_image_path(tmp_path: Path):
    base_dir = tmp_path / "new_run"
    image = base_dir / "trace_images" / "task" / "step_001_after.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    stale = (
        tmp_path
        / "old_root"
        / "results"
        / "new_run"
        / "trace_images"
        / "task"
        / "step_001_after.jpg"
    )
    record = {
        "task": "DemoTask",
        "goal": "Do the thing.",
        "trace": {
            "steps": [
                {
                    "step": 1,
                    "action": "click(point='<point>1 2</point>')",
                    "after_screenshot_path": str(stale),
                }
            ]
        },
    }

    normalized = normalize_record(record, base_dir=base_dir)

    assert normalized.steps[0].after_screenshot_path == str(image)


def test_normalize_record_supports_readable_benchmark_json():
    record = {
        "task_template": "MarkorEditNote",
        "goal": "Edit note.txt.",
        "is_successful": 0.0,
        "steps": [
            {
                "step": 0,
                "action_reason": "Thought omitted by model; executing parsed GUI-Demo action.",
                "action_output_json": "JSONAction(action_type='open_app', app_name='Markor')",
                "summary": "Opened Markor.",
            }
        ],
    }

    normalized = normalize_record(record)

    assert normalized.official_success is False
    assert normalized.steps[0].thinking == "Thought omitted by model; executing parsed GUI-Demo action."
    assert normalized.steps[0].action == "JSONAction(action_type='open_app', app_name='Markor')"
