# 官方 GUI 处理方案适配说明

参考文档：[火山方舟 GUI 任务处理](https://www.volcengine.com/docs/82379/1584296?lang=zh)

## 文档里的关键处理方式

官方示例的核心不是 Android World runner，而是 message 组织方式：

- 第一条 `user` 消息放 GUI agent prompt 和用户任务。
- 历史动作放在 `assistant` 消息里，格式是 `Thought: ...` + `Action: ...`。
- 截图作为独立 `user` 消息，用 `image_url` 传入。
- 长历史里不保留无限截图，示例强调最多保留 5 轮截图消息。
- 模型调用走火山方舟 `base_url=https://ark.cn-beijing.volces.com/api/v3`，可以用 Ark SDK 或 OpenAI-compatible SDK。

## 本组件怎么适配

代码位置：

- message 构造：[gui_trace_evaluator/official_messages.py](gui_trace_evaluator/official_messages.py)
- record 归一化：[gui_trace_evaluator/record_adapter.py](gui_trace_evaluator/record_adapter.py)
- 评测主流程：[gui_trace_evaluator/evaluator.py](gui_trace_evaluator/evaluator.py)
- CLI：[gui_trace_evaluator/runner.py](gui_trace_evaluator/runner.py)

评测流程：

1. 可选：先调用外部 GUI agent 执行命令，让它完整跑完任务并写出 records JSON。
2. 从通用 records JSON 读取 task、goal、trace steps 和截图路径。
3. 生成稳定 checkpoint standard，并缓存到 `checkpoint_cache`。
4. 用官方风格 Thought/Action 历史做模型 step retrieval。
5. 如果 retrieval 不可信，再做一次模型 repair retrieval；不使用旧规则检索兜底。
6. retrieval 的 trust 只作为 metadata，不由代码直接判失败。
7. 对模型选中的 step，按官方风格插入最多 5 张截图。
8. 让 judge 输出 checkpoint 是否完成、证据是否不足。
9. 聚合 required checkpoint 得到最终成功与否。

## 和原 Android World 修改的关系

这个组件不 import Android World。原 Android World 或 AWE 只需要产出 records JSON 和截图文件。

因此新机器上只需要：

1. 拷贝 `doubao_gui_trace_evaluator/`
2. 准备 records JSON
3. 确保截图路径可访问，或通过 `--image_root` 指定截图根目录
4. 设置 `ARK_API_KEY`
5. 运行 `python -m gui_trace_evaluator.runner ...`

如果旧 records 里截图是过期绝对路径，组件会尝试用 `trace_images/...` 后缀在 records 所在目录或 `--image_root` 下重新定位。
