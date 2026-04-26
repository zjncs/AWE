from pathlib import Path

from PIL import Image

from gui_trace_evaluator.official_messages import build_trace_messages
from gui_trace_evaluator.record_adapter import NormalizedStep


def test_build_trace_messages_keeps_official_history_and_caps_images(tmp_path: Path):
    steps = []
    for index in range(1, 8):
        image = tmp_path / f"step_{index:03d}.jpg"
        Image.new("RGB", (10, 10), "white").save(image)
        steps.append(
            NormalizedStep(
                step=index,
                thinking=f"think {index}",
                action=f"click(point='<point>{index} {index}</point>')",
                summary=f"summary {index}",
                before_screenshot_path="",
                after_screenshot_path=str(image),
            )
        )

    messages, manifest = build_trace_messages(
        instruction="Check task.",
        steps=steps,
        final_request="Return JSON.",
        image_step_numbers={step.step for step in steps},
        max_screenshot_turns=5,
    )

    assert messages[0]["role"] == "user"
    assert "You are a GUI agent" in messages[0]["content"]
    assert messages[-1]["content"] == "Return JSON."
    assert len(manifest) == 5
    assert [item["step"] for item in manifest] == [1, 2, 4, 5, 7]
    assert any(message["role"] == "assistant" and "Thought: think 7" in message["content"] for message in messages)
    image_messages = [
        message
        for message in messages
        if isinstance(message.get("content"), list)
    ]
    assert len(image_messages) == 5
    assert image_messages[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    step_4_assistant_index = next(
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant" and "Thought: think 4" in message["content"]
    )
    step_4_image_index = next(
        index
        for index, message in enumerate(messages)
        if isinstance(message.get("content"), list)
        and "Screenshot for step 4 (after action)." in message["content"][0]["text"]
    )
    assert step_4_image_index > step_4_assistant_index


def test_build_trace_messages_places_before_screenshot_before_action(tmp_path: Path):
    image = tmp_path / "step_001_before.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    step = NormalizedStep(
        step=1,
        thinking="think",
        action="click(point='<point>1 1</point>')",
        summary="summary",
        before_screenshot_path=str(image),
        after_screenshot_path="",
        before_ui="[1] text='Files'",
        after_ui="[2] text='Pictures'",
    )

    messages, manifest = build_trace_messages(
        instruction="Check task.",
        steps=[step],
        final_request="Return JSON.",
        image_step_numbers={1},
        max_screenshot_turns=5,
    )

    assert manifest[0]["kind"] == "before"
    image_index = next(index for index, message in enumerate(messages) if isinstance(message.get("content"), list))
    assistant_index = next(index for index, message in enumerate(messages) if message["role"] == "assistant")
    assert image_index < assistant_index
    assert "UI Text Before:" in messages[assistant_index]["content"]
