# review-loop: Complete Design v2.1

## 一句话描述

基于 discuss-agent 架构的写-审循环框架。Author 写内容，多个 Reviewer 并行审核输出结构化 issues，Author 判断每个 issue 是否 valid 后修改或反驳，反驳只发回给对应的 Reviewer，循环直到所有 Reviewer 零 issue。

## 核心原则

**对抗精神**：与 discuss-agent 一脉相承。所有角色都有坚定立场，不客气、不妥协。只有严密的逻辑和实际证据才能推翻一个判断。

- **Reviewer 不能被吓退**：Author 说"我觉得没问题"不算反驳。必须看到具体理由、数据、或逻辑论证才考虑关闭 issue。
- **Author 不能无脑接受**：参考 ironflow receiving-review 方法论。审核员可能缺上下文、有偏见、或在具体问题上判断有误。用证据 push back。
- **不是表演辩论**：用事实和逻辑真刀真枪。"Great point!"这种客套不存在。

## 架构关系

```
discuss-agent (已有)          review-loop (新建)
├── engine.py  ──fork──→     ├── engine.py      ← 核心循环
├── config.py  ──复用──→     ├── config.py      ← YAML 加载
├── models.py  ──复用──→     ├── models.py      ← Agent 封装(Agno+Claude)
├── context.py ──复用──→     ├── context.py     ← Context 构建
├── persistence.py ─复用─→   ├── persistence.py ← 结果归档
└── main.py    ──fork──→     └── main.py        ← CLI
```

## 核心流程

```
[输入: YAML 配置 + 初始素材]
         │
         ▼
┌─────────────────┐
│  Author 生成 v1  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  N 个 Reviewer 并行审核          │
│  每个输出: { issues: [...] }    │
└────────┬────────────────────────┘
         │
         ▼
    所有 issues 为空？
    ├── YES → 输出终稿，结束
    └── NO ↓
         │
         ▼
┌──────────────────────────────────────┐
│  Author 收到所有 issues              │
│  (注入 receiving-review prompt)      │
│                                      │
│  对每个 issue 判断:                   │
│  ├─ valid → 修改内容                 │
│  ├─ incorrect → 生成反驳 + evidence  │
│  └─ unclear → 标记需要澄清           │
│                                      │
│  输出:                                │
│  ├─ 修改后的内容 (v2)                │
│  └─ 反驳清单 { reviewer, issue, rebuttal } │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  再次分发给 Reviewer                  │
│                                      │
│  Reviewer 收到:                       │
│  ├─ 新版内容                          │
│  └─ Author 的反驳 (如果有)            │
│                                      │
│  Reviewer 判断:                       │
│  ├─ 修改解决了 issue → 关闭           │
│  ├─ 接受反驳 → 关闭                   │
│  └─ 不接受 → 维持 issue (附新理由)    │
│  + 可能发现新 issue                   │
│                                      │
│  输出: { issues: [...] }              │
└────────┬─────────────────────────────┘
         │
         ▼
    所有 issues 为空？
    ├── YES → 输出终稿，结束
    └── NO → 回到 Author (下一轮)
         │
    达到 max_rounds？
    └── YES → 输出当前版本 + 未解决 issues 清单
```

## YAML 配置

```yaml
review:
  min_rounds: 1
  max_rounds: 3
  model: "claude-opus-4.6-1m"
  api_key: "Powered by Agent Maestro"
  base_url: "http://localhost:23333/api/anthropic"

author:
  name: "Author"
  system_prompt: |
    你是一个...（根据场景定制）

  # 处理审核反馈时自动注入的 prompt
  receiving_review_prompt: |
    你收到了审核员的反馈。对每个 issue，做出判断：

    1. **valid** — 问题确实存在，修改内容解决它
    2. **incorrect** — 审核员判断有误，给出你的理由和证据
    3. **unclear** — 问题描述不清楚，无法判断，标记需要澄清

    不要无脑接受所有反馈。审核员可能缺少上下文，可能有偏见，
    也可能在某些判断上是错的。用证据说话，不用客气。

    对于你认为 incorrect 的 issue，你的反驳会被发回给对应的
    审核员重新评估。如果审核员被你说服，issue 会被关闭。

    输出格式：
    对每个 issue，标注 [ACCEPT] [REJECT reason] 或 [UNCLEAR]
    然后给出修改后的完整内容。

reviewers:
  - name: "逻辑审核员"
    system_prompt: |
      你是逻辑审核员。你的职责是检查内容的逻辑链是否完整。
      
      具体检查：
      - 每一步的结论是否由前一步推导而来
      - 有没有逻辑跳跃（A直接到C，跳过了B）
      - 因果关系是否成立
      - 隐含假设是否合理

  - name: "数据审核员"
    system_prompt: |
      你是数据审核员。你的职责是检查每个论点是否有数据支撑。
      
      具体检查：
      - 每个判断是否有具体数据或来源
      - 数据引用是否准确
      - 有没有过度推断（数据说的是A，结论说的是B）

  - name: "读者审核员"
    system_prompt: |
      你是目标读者的代言人。你的职责是判断内容对目标读者是否可理解、有价值。
      
      目标读者画像：25-45岁有投资经验的城市中产，
      关注财经但不读研报，知道主流声音但想深入一层。

# 可选: Author 可用的工具
tools:
  - path: research_pipeline.tools.search.SearchTools
  - path: research_pipeline.tools.raw_file.RawFileTools

# 可选: 初始上下文构建
context_builder: research_pipeline.context.build_outline_context
```

## Reviewer 输出格式

```json
{
  "issues": [
    {
      "severity": "critical",
      "content": "第三段从油价涨跳到中国受益，缺少中间的推理步骤（为什么油价涨=中国受益？需要补充煤化工替代路线的逻辑）"
    },
    {
      "severity": "minor",
      "content": "申万的数据是停火前的，建议标注时间前提"
    }
  ]
}
```

- `issues` 为空数组 → 该 Reviewer 通过
- `severity`: `critical` / `major` / `minor`
- 全部 Reviewer 的 `issues` 都为空 → 自然通过，结束

## Author 处理反馈的输出格式

```json
{
  "responses": [
    {
      "reviewer": "逻辑审核员",
      "issue_index": 0,
      "verdict": "accept",
      "action": "已在第三段补充华泰关于煤化工替代路线的数据链"
    },
    {
      "reviewer": "数据审核员",
      "issue_index": 1,
      "verdict": "reject",
      "reason": "申万报告本身就是做情景分析，不依赖停火前提。压力测试的价值正在于不受单一时间点约束。"
    }
  ],
  "updated_content": "... (修改后的完整内容) ..."
}
```

## Reviewer 收到反驳后的处理

**反驳只发给对应的 Reviewer，不广播。** 每个 Reviewer 只看到自己提出的 issue 的处理结果，看不到其他 Reviewer 的反馈。

Reviewer 在下一轮审核时，system prompt 后面注入该 Reviewer 自己的 issues + Author 的对应回应：

```
[上一轮你提出的 issues 及 Author 的回应]

issue 0: ...你的原始 issue...
Author 回应: [ACCEPT] 已修改——具体改动：...

issue 1: ...你的原始 issue...
Author 回应: [REJECT] 理由：...证据：...

审核修改后的内容。
- 对于 Author 接受并修改的 issue：检查修改是否真正解决了问题
- 对于 Author 反驳的 issue：评估反驳是否成立
  - 反驳有理有据 → 关闭 issue
  - 反驳缺乏说服力 → 维持 issue，补充你的理由
  - 注意：Author 说"我觉得没问题"或"这是风格选择"不构成有效反驳。
    只有具体的数据、逻辑论证、或上下文信息才能推翻你的判断。
- 可以提出新发现的 issue
```

## 持久化

```
output_dir/YYYY-MM-DD_HHMM/
├── config.yaml                    # 配置快照
├── context.md                     # 初始上下文
├── rounds/
│   ├── round_1_author.md          # Author v1
│   ├── round_1_reviewer_逻辑审核员.json
│   ├── round_1_reviewer_数据审核员.json
│   ├── round_1_reviewer_读者审核员.json
│   ├── round_1_author_response.json  # Author 的判断+反驳
│   ├── round_2_author.md          # Author v2 (修改版)
│   ├── round_2_reviewer_*.json
│   └── ...
├── final.md                       # 最终输出
└── unresolved_issues.json         # 未解决的 issues (如果达到max_rounds)
```

## CLI

```bash
python -m review_loop configs/outline_review.yaml

# 带初始内容
python -m review_loop configs/outline_review.yaml --input initial_draft.md

# 带初始上下文
python -m review_loop configs/outline_review.yaml --context research_brief.md
```

## Engine 核心逻辑 (伪代码)

```python
async def run(config, initial_content=None, context=None):
    # 1. Build context
    if context:
        ctx = load_context(context)
    elif config.context_builder:
        ctx = await config.context_builder(config.context)
    
    # 2. Author generates v1
    if initial_content:
        content = initial_content
    else:
        content = await author.generate(ctx)
    save_round(1, "author", content)
    
    # 3. Review loop
    for round in range(1, config.max_rounds + 1):
        # Parallel review
        all_issues = {}
        for reviewer in config.reviewers:
            feedback = await reviewer.review(content, previous_feedback)
            save_round(round, f"reviewer_{reviewer.name}", feedback)
            all_issues[reviewer.name] = feedback.issues
        
        # Check convergence
        total_issues = sum(len(v) for v in all_issues.values())
        if total_issues == 0:
            save_final(content)
            return Result(converged=True, content=content)
        
        # Author processes feedback (inject receiving-review prompt)
        author_response = await author.process_feedback(
            content, all_issues, config.receiving_review_prompt
        )
        save_round(round, "author_response", author_response)
        
        # Update content
        content = author_response.updated_content
        save_round(round + 1, "author", content)
        
        # Prepare reviewer context for next round
        # (inject ONLY each reviewer's own issues + author's response to them)
        previous_feedback = build_per_reviewer_context(all_issues, author_response)
    
    # Max rounds reached
    save_final(content)
    save_unresolved(all_issues)
    return Result(converged=False, content=content, unresolved=all_issues)
```

## 适用场景

| 场景 | Author prompt | Reviewer prompts |
|------|--------------|------------------|
| 大纲审核 | 研报深度分析大纲写手 | 逻辑/数据/读者 |
| 正文审核 | 正文写手 | 品味/内容规则/Anti-AI |
| 标题审核 | 标题生成 | 标题检验清单 |
| 代码审核 | 开发者 | spec合规/代码质量/复用 |
| 任何写-审场景 | 自定义 | 自定义 |

## 与 discuss-agent 的代码复用

| 模块 | 复用方式 |
|------|---------|
| ConfigLoader | 直接复用，加 `author` + `reviewers` 段 |
| Agent 封装 (Agno) | 直接复用 |
| Persistence | 直接复用，改输出目录结构 |
| Context builder | 直接复用接口 |
| Engine | **重写** — 核心循环逻辑不同 |

## Repo 结构

```
review-loop/
├── review_loop/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py          ← CLI
│   ├── engine.py         ← 核心循环（新写）
│   ├── config.py         ← 配置加载（复用）
│   ├── models.py         ← Agent 封装（复用）
│   ├── context.py        ← Context（复用）
│   └── persistence.py    ← 归档（复用）
├── configs/
│   └── outline_review.yaml  ← 示例配置
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

Public repo: `github.com/Zhijiang-Li1111/review-loop`
