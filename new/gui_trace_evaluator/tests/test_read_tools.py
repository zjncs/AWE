from gui_trace_evaluator.read_tools import default_read_requests
from gui_trace_evaluator.read_tools import _annotate_find_result, _annotate_stat_result
from gui_trace_evaluator.read_tools import _is_safe_select_query
from gui_trace_evaluator.record_adapter import normalize_record


def test_default_read_requests_for_file_move_goal():
    record = normalize_record(
        {
            "task": "FilesMoveFile",
            "base_goal": (
                "Move the file sunset.jpg from Alarms within the sdk_gphone64_arm64 "
                "storage area to the Pictures within the same sdk_gphone64_arm64 storage area."
            ),
            "trace": {"steps": [{"step": 1, "action": "noop"}]},
        }
    )
    requests = default_read_requests(record, {"description": "Verify final file location."})

    assert any(request.get("tool") == "list_dir" and request.get("path") == "/sdcard/Alarms" for request in requests)
    assert any(request.get("tool") == "list_dir" and request.get("path") == "/sdcard/Pictures" for request in requests)
    assert any(request.get("tool") == "find_file" and request.get("name") == "sunset.jpg" for request in requests)
    assert any(request.get("tool") == "stat_path" and request.get("path") == "/sdcard/Pictures/sunset.jpg" for request in requests)


def test_default_read_requests_for_recipe_goal_uses_broccoli_sqlite():
    record = normalize_record(
        {
            "task": "RecipeDeleteDuplicateRecipes3",
            "base_goal": "Delete duplicate recipes from Broccoli and leave only one copy of each recipe.",
            "trace": {"steps": [{"step": 1, "action": "noop"}]},
        }
    )

    requests = default_read_requests(record, {"description": "Verify the final recipe database."})

    assert any(
        request.get("tool") == "query_app_sqlite"
        and request.get("package") == "com.flauschcode.broccoli"
        and "FROM recipes" in request.get("query", "")
        for request in requests
    )


def test_sqlite_tool_accepts_only_read_only_selects():
    assert _is_safe_select_query("SELECT title FROM recipes")
    assert not _is_safe_select_query("DELETE FROM recipes")
    assert not _is_safe_select_query("SELECT title FROM recipes; DROP TABLE recipes")
    assert not _is_safe_select_query("PRAGMA table_info(recipes)")


def test_annotate_stat_result_marks_missing_path_as_structured_absence():
    result = {
        "status": "ok",
        "output": "__AWE_PATH_EXISTS=false__\n",
    }

    _annotate_stat_result(result)

    assert result["status"] == "ok"
    assert result["exists"] is False
    assert result["interpretation"] == "The requested path does not exist."


def test_annotate_find_result_marks_empty_output_as_not_found():
    result = {
        "status": "ok",
        "output": "",
    }

    _annotate_find_result(result)

    assert result["found"] is False
    assert result["matches"] == []
    assert result["interpretation"] == "No matching files were found."


def test_annotate_find_result_ignores_non_path_log_lines():
    result = {
        "status": "ok",
        "output": (
            "I0425 19:11:18.980397 7542691 ev_poll_posix.cc:593] FD from fork parent still in poll list\n"
            "W0425 noisy log line\n"
        ),
    }

    _annotate_find_result(result)

    assert result["found"] is False
    assert result["matches"] == []


def test_annotate_find_result_keeps_real_path_matches():
    result = {
        "status": "ok",
        "output": (
            "I0425 noise\n"
            "/sdcard/Pictures/2023_05_12_eager_nurse.jpg\n"
        ),
    }

    _annotate_find_result(result)

    assert result["found"] is True
    assert result["matches"] == ["/sdcard/Pictures/2023_05_12_eager_nurse.jpg"]
