# Doubao GUI Trace Evaluator

独立版 GUI trace 评测组件。它不依赖 Android World，也不依赖我们之前魔改过的 runner。只需要一份 records JSON 和可访问的截图路径，就可以用豆包/火山方舟模型做 checkpoint-based 评测。

## 设计目标

- 按火山方舟 GUI 任务处理文档的消息组织方式构造上下文：初始 user prompt、历史 `Thought/Action`、最多 5 轮截图消息。
- 输入格式尽量通用：兼容 Android World/AWE 风格 records，也兼容普通 GUI agent traces。
- 不传文本化 UI tree。评测证据来自 `thinking/action/summary` 和截图。
- 长流程不使用旧规则检索兜底。默认是模型检索 step；如果不可信，再做一次模型 repair retrieval；仍不可信则由 judge 决定是否请求只读工具补充最终状态证据。
- 支持 read-only fallback verification：评测模型可以在不确定时请求 `list_dir`、`stat_path`、`find_file`、`read_text_file`，框架执行只读读取，再把结果交回模型二次判断。
- 作为单独组件拷走即可运行。

## 输入 record 格式

最小格式：

```json
[
  {
    "task": "SimpleCalendarAddOneEvent",
    "goal": "Create an event ...",
    "success": true,
    "reward": 1.0,
    "post_execution_evidence": [
      {
        "type": "read_tool_result",
        "tool": "list_dir",
        "status": "ok",
        "request": {"tool": "list_dir", "path": "/sdcard/Pictures"},
        "output": "..."
      }
    ],
    "trace": {
      "steps": [
        {
          "step": 1,
          "thinking": "I need to open Calendar.",
          "action": "open_app(app_name='Calendar')",
          "summary": "Calendar is open.",
          "after_screenshot_path": "trace_images/task/step_001_after.jpg"
        }
      ]
    }
  }
]
```

字段兼容：

- 任务目标优先级：`base_goal` > `goal` > `goal_used` > `instruction`
- 思考字段兼容：`thinking` / `thought` / `reason` / `action_output` 里的 `Thought:` 或 `Reason:`
- 动作字段兼容：`action` / `action_output` 里的 `Action:`
- 截图字段兼容：`after_screenshot_path` / `before_screenshot_path` / `screenshot_path` / `image_path`
- 可选最终状态证据：`post_execution_evidence`。如果执行器能在任务结束后、环境 reset/tear_down 前收集只读状态，建议放在这里。评测模型不直接按它判分，只把它作为额外 evidence。

## 只评测已有 records

使用 OpenAI-compatible 方舟接口：

```bash
cd /path/to/doubao_gui_trace_evaluator
export ARK_API_KEY=...

python -m gui_trace_evaluator.runner \
  --records /path/to/results.json \
  --provider openai \
  --model doubao-seed-1-8-251228 \
  --base_url https://ark.cn-beijing.volces.com/api/v3 \
  --api_key_env ARK_API_KEY \
  --output results/eval.json
```

使用官方 Ark SDK：

```bash
pip install "volcengine-python-sdk[ark]"

python -m gui_trace_evaluator.runner \
  --records /path/to/results.json \
  --provider ark \
  --model doubao-seed-1-8-251228 \
  --api_key_env ARK_API_KEY \
  --output results/eval.json
```

如果 records 里的截图路径是相对路径，可以指定：

```bash
python -m gui_trace_evaluator.runner \
  --records /path/to/results.json \
  --image_root /path/to/results_dir \
  ...
```

默认启用只读 fallback tools。模型在 checkpoint 判断不确定时，可以请求：

- `list_dir(path)`：列目录，例如 `/sdcard/Pictures`
- `stat_path(path)`：检查路径是否存在和元信息
- `find_file(root, name)`：在只读根目录下查找文件
- `read_text_file(path)`：读取文本文件前 200 行

工具只读，且默认只允许读取 `/sdcard` 和 `/storage/emulated/0`。它们只补证据，不直接返回 success/fail。可以通过下面参数控制：

```bash
--disable_read_tools
--adb_path /path/to/adb
--adb_serial emulator-5554
--fallback_confidence_threshold 0.7
```

## 先执行，再评测

如果希望在同一个入口里先跑 GUI agent，再评测它生成的 records，可以用 `pipeline_runner`。这个组件只负责编排；真正的执行命令仍然可以是 Android World、AWE、RuyiAgent 或你自己的脚本。

```bash
python -m gui_trace_evaluator.pipeline_runner \
  --execute "python run_agent.py --output /tmp/awe_records.json" \
  --records /tmp/awe_records.json \
  --image_root /tmp \
  --provider openai \
  --model doubao-seed-1-8-251228 \
  --base_url https://ark.cn-beijing.volces.com/api/v3 \
  --api_key_env ARK_API_KEY \
  --output results/eval.json
```

如果希望强制确认每条 record 都有截图，可以加：

```bash
--require_images
```

这个流程里，评测器只在执行完成后读取完整 trace 和截图；不会在执行中途判断任务是否成功。

## 输出

输出 JSON 会包含：

- `checkpoint_results`
- `retrieval_method`
- `retrieval_trusted`
- `evidence_steps`
- `evidence_images`
- `read_tool_verification`
- `success`
- `completeness_score`
- `agreement_with_reward`

同时会生成同名 `.md` 报告。

## 和 Android World 的关系

这个目录里的代码不 import Android World。Android World 只负责产出 record/trace；评测器只消费 JSON 和截图。因此在新机器上不需要同步 Android World 的魔改代码。
