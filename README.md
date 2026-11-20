# 黑色托盘状态评测

使用 Python 3.10+ 标准库，无需安装依赖。唯一命令行入口为 `evaluate.py`。

## 文件职责

- `evaluate.py`：四任务英文 `PROMPTS`、图片映射加载、服务调用、答案解析、评分和报告生成。
- `tasks.json`：任务ID、对应图片、真值和样本用途。
- `images/`：10张原图。
- `tests/`：解析、评分和任务执行的验证。

## 四项任务

| 任务ID | 任务 | 对应图片 | 真值 |
|---|---|---|---|
| `empty` | 黑色托盘是否为空 | 01、06、09；非空反例02、08、03 | YES、YES、YES、NO、NO、NO |
| `square_item` | 袋装方形物品数量 | 03 | 1 |
| `blue_wire` | 蓝色线缆数量 | 04、05、07、10 | 2、2、2、1 |
| `black_wire` | 黑色线缆数量 | 08 | 1 |

每个请求只发送一张对应原图，共12个任务与图片组合。提示词全部为英文，不含逐图真值、
文件名或历史回答。判空反例作为独立测试输入，不作为提示词里的示范。

只统计前景黑色托盘承载面上的额外物品。排除固定螺丝、托盘结构、绿箱和周围桌面上的物品。
蓝色线缆按袋计数，黑色线缆按束计数。07为两袋叠放。02统一称 **black box**，
03为袋装方形物品。全部真值已由用户确认。

当前英文术语统一为 **wire**，任务ID使用 `blue_wire`、`black_wire`。
当前运行结果保存到 `results/tasks-wire-v2/`（判空新增03非空反例）。

## 运行

```sh
python3 evaluate.py run
python3 evaluate.py report
python3 -m unittest discover -s tests -v
```

`run` 调用 `tasks.json` 指定的服务（默认 `http://115.120.57.9/3dvlm`），
使用 `layout=single_image` 和64 token输出上限。`report` 仅用已有原始响应生成报告。

默认输出到 `results/tasks-wire-v2/`：

- `report.html`：按任务分组，嵌入原图、真值、预测与原始回答，可离线查看。
- `report.md` / `summary.json`：逐图结果和任务汇总。
- `results.csv`：逐图评分数据。
- `responses.jsonl`：原始响应、错误、UTC时间与耗时。
- `config.json`：使用的任务配置、完整英文提示词、真值与图片哈希。
- `health.json`：服务健康信息。
- `findings.md`：本轮结果分析；重新生成报告不会自动改写。

已有记录（包括失败记录）不会自动重试。请求失败时停止，先检查日志与服务状态。
修改提示词、映射、真值或图片后，使用新的输出目录：

```sh
python3 evaluate.py run --output results/tasks-v2
python3 evaluate.py report --output results/tasks-v2
```

可通过 `--tasks`、`--images`、`--output`、`--timeout` 指定配置、图片、输出目录和请求超时。

## 指标与已有结果

每个任务以其指定的全部图片数为准确率分母。UNKNOWN、格式错误、请求失败和缺失回答
都不算正确。判空单独报告空样本与非空反例的正确数，避免只看合计准确率。

已按用户要求清空旧报告与原始响应，当前只保留本次重跑结果。

当前 `results/tasks-wire-v2/` 在判空任务中新增03非空反例，并从头运行全部12次请求。
03回答NO（非空），正确；判空合计4/6（空样本3/3，非空反例1/3）。
其他任务为袋装方形物品1/1、蓝色线缆2/4、黑色线缆0/1。详细分析见该目录的 `findings.md`。

`results/code-tests.log`、`results/code-tests.json` 保存本轮7项代码测试的通过记录。
`labels.json` 保留最初的逐图标注；当前运行使用 `tasks.json` 中的任务与图片映射及真值。
