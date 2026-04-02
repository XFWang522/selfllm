# 已有方法调研：LLM 自训练与自进化

## 概述

本文档全面调研了与 SelfLLM 项目相关的现有研究工作，按技术维度分类，分析每个方法的核心思想、优缺点，以及对 SelfLLM 的启发价值。

## 1. 自训练与自我改进（Self-Training & Self-Improvement）

### 1.1 STaR: Self-Taught Reasoner（Google, NeurIPS 2022）

**核心思想**：模型通过生成 Chain-of-Thought 推理链来训练自己。

**方法**：
1. 模型生成逐步推理过程来回答问题
2. 只保留推理正确的样本
3. 对错误样本，给模型提示正确答案让它生成"合理化"的推理链
4. 用这些推理链微调模型
5. 迭代重复

**结果**：CommonsenseQA 上从 60%→72.3%，媲美 30 倍参数量模型的效果。

**对 SelfLLM 的启发**：
- 验证了 "用自己的正确输出训练自己" 的可行性
- "合理化"技巧（给答案让模型补推理）是关键创新
- 但 STaR 只改变数据，不改变训练代码和策略——SelfLLM 要做到全闭环

**局限**：依赖外部的正确答案进行过滤，不完全自主。

---

### 1.2 Self-Rewarding Language Models（Meta FAIR, ICML 2024）

**核心思想**：模型同时作为策略模型和奖励模型，用 LLM-as-Judge 评估自己的输出。

**方法**：
1. 模型生成多个候选回答
2. 模型自己对这些回答打分（LLM-as-Judge）
3. 用 Iterative DPO 训练模型偏好高分回答
4. 迭代多轮，同时提升回答质量和评估质量

**结果**：Llama 2 70B 经过 3 轮迭代后在 AlpacaEval 2.0 上超越 Claude 2、Gemini Pro 和 GPT-4。

**对 SelfLLM 的启发**：
- **核心参考方法之一**：证明了模型可以同时提升 "做事能力" 和 "评估能力"
- LLM-as-Judge 是自评估的关键技术
- 无需外部人类反馈即可持续改进
- SelfLLM 可以直接复用这一范式

**局限**：只改变训练数据的选择偏好，不改变训练代码和方法本身。

---

### 1.3 SPIN: Self-Play Fine-Tuning（UCLA, ICML 2024）

**核心思想**：模型与上一版本的自己博弈来提升。

**方法**：
1. 当前模型（Main Player）学习区分人类数据和上一版本模型的输出
2. 上一版本（Opponent）生成训练数据
3. 理论证明：当且仅当模型等同于人类数据分布时达到纳什均衡
4. 迭代训练直到收敛

**结果**：无需额外人类标注数据，超过 DPO + GPT-4 偏好数据的效果。

**对 SelfLLM 的启发**：
- 自博弈范式非常适合 SelfLLM 的闭环设计
- 理论上有纳什均衡保证，但实际中的收敛性需要验证
- 可以扩展到代码生成领域的自博弈

---

### 1.4 LADDER: Self-Improving Through Recursive Problem Decomposition（2025）

**核心思想**：模型通过递归分解复杂问题为简单子问题来自我提升。

**方法**：
1. 模型遇到难题 → 自主将其简化为更容易的变体
2. 先解决简单变体，逐步升级难度
3. 引入 Test-Time Reinforcement Learning (TTRL)
4. 无需外部数据或人工标注

**结果**：Llama 3.2 3B 在本科微积分上从 1%→82%，Qwen 2.5 7B 达到 90%（超越 OpenAI o1）。

**对 SelfLLM 的启发**：
- **课程学习的自动化**：模型自己决定学习难度曲线
- 小模型也能通过自训练达到强大能力
- SelfLLM 的 Strategy Optimizer 应该能自主实现类似的课程策略

---

### 1.5 SEAL: Self-Adapting LLMs（2025）

**核心思想**：模型自己生成微调数据和优化指令，直接修改自己的权重。

**方法**：
1. 模型生成自己的微调数据和优化方向
2. 用 RL 训练，以下游任务表现为奖励信号
3. 实现持久性权重更新（不需要额外适配模块）

**对 SelfLLM 的启发**：
- 最接近 SelfLLM 理念的工作之一
- 但 SEAL 的 "优化指令" 仍然是在预定义框架内，SelfLLM 要让模型写任意代码

---

## 2. 自博弈与自生成数据（Self-Play & Self-Generated Data）

### 2.1 Language Self-Play (LSP)（Meta, 2025）

**核心思想**：博弈论框架，模型自己出题自己答。

**方法**：
1. **Challenger 模式**：模型生成越来越难的指令
2. **Solver 模式**：模型学习回答这些指令
3. 同一个模型扮演两个角色
4. 实现"在持续提升的自生成数据上永续训练"

**结果**：Llama-3.2-3B-Instruct 在指令遵循、数学、编码上均有提升。

**对 SelfLLM 的启发**：
- 解决了 "数据从哪来" 的问题——模型自己出题
- Challenger-Solver 双角色机制可直接复用
- 但 LSP 不涉及训练代码的修改

---

### 2.2 SeRL: Self-play Reinforcement Learning（2025）

**核心思想**：在有限初始数据下通过自博弈实现 RL 训练。

**方法**：
1. **自指令模块**：基于现有数据生成新指令，在线过滤质量和多样性
2. **自奖励模块**：用多数投票机制估计奖励，无需外部标注
3. 结合两个模块进行 RL 训练

**结果**：在推理 benchmark 上达到使用高质量标注数据的可比水平。

**对 SelfLLM 的启发**：
- 多数投票作为奖励信号的可靠性已被验证
- 自指令生成 + 在线过滤的模式值得采用

---

### 2.3 Constitutional AI / RLAIF（Anthropic, 2022-2025）

**核心思想**：用 AI 反馈代替人类反馈进行对齐训练。

**方法**：
1. **SL 阶段**：模型生成回答 → 基于宪法原则自我批评 → 自我修正
2. **RL 阶段**：模型对比两个回答，判断哪个更符合宪法 → 训练偏好模型 → RL 训练

**2026 年新发现**（"Why Does RLAIF Work At All?"）：
- 预训练在互联网数据上已经将人类价值观编码为表征空间中的方向
- Constitutional prompt 激活了这些潜在价值观

**对 SelfLLM 的启发**：
- 宪法原则 = SelfLLM 的安全约束框架
- 自我批评 + 自我修正的范式可以扩展到训练策略的优化
- 成本相比人工标注降低 90%+

---

## 3. 递归自我改进与代码进化（Recursive Self-Improvement & Code Evolution）

### 3.1 STOP: Self-Taught Optimizer（Microsoft Research, 2023）

**核心思想**：LLM 生成的"优化器程序"可以递归地优化自身。

**方法**：
1. 从一个种子"改进器"程序开始
2. 改进器查询 LLM 多次，选择最优方案
3. 改进器可以对自身运行，生成改进版本
4. LLM 自主提出搜索策略（beam search, 遗传算法, 模拟退火）

**关键发现**：
- 底层语言模型不变，只是外围代码（scaffolding）在改进
- 这不是"真正的"递归自我改进——模型权重未变
- 存在沙箱逃逸风险

**对 SelfLLM 的启发**：
- **直接灵感来源**：SelfLLM 要在此基础上更进一步——不仅改代码，还改模型权重
- 代码级的递归优化已验证可行
- 安全沙箱是必须的

---

### 3.2 Gödel Agent: Self-Referential Recursive Self-Improvement（2024）

**核心思想**：让 LLM Agent 动态修改自己的逻辑和行为。

**方法**：
1. Agent 可以修改自己的推理逻辑、工具调用方式、决策策略
2. 不依赖预定义的优化算法
3. 基于高层目标自主进化
4. 在编码、科学和数学领域验证有效

**对 SelfLLM 的启发**：
- 自修改逻辑的概念可以扩展到训练策略的自修改
- 高层目标引导 + 自由探索的模式适合 SelfLLM

---

### 3.3 FunSearch（Google DeepMind, Nature 2023）

**核心思想**：LLM + 进化搜索 + 自动评估 = 数学发现。

**方法**：
1. 维护一个程序池
2. 从池中选择高分程序，送入 LLM 进行创造性改进
3. 自动评估生成代码的质量
4. 将最优程序加回池中
5. 进化循环

**关键成果**：
- 在 Cap Set Problem 上发现超越已知最优解的新构造（Nature 发表）
- 在 Bin Packing 上发现优于广泛使用基线的新启发式
- 2024 年扩展到竞赛编程，超越人类顶级团队

**对 SelfLLM 的启发**：
- **进化搜索 + 自动评估** 是策略优化器的最佳参考模式
- 证明 LLM + 进化可以发现人类未知的方法
- SelfLLM 的 Strategy Optimizer 应该采用类似的进化搜索机制

---

## 4. LLM 驱动的 AutoML（LLM-Powered AutoML）

### 4.1 NNGPT: Rethinking AutoML with LLMs（2025）

**核心思想**：将 LLM 变成自我改进的 AutoML 引擎。

**方法**：
1. 零样本架构合成
2. 超参数优化
3. 代码感知的精度预测
4. 检索增强的合成
5. 强化学习优化

**结果**：生成超过 5,000 个验证模型，单次预测即达到搜索级 AutoML 性能。

**对 SelfLLM 的启发**：
- LLM 已经可以做 AutoML，扩展到自训练是自然延伸
- 超参数自动优化的成熟度已经很高

---

### 4.2 SEKI: LLM-based Neural Architecture Search（2025）

**核心思想**：LLM 通过自进化和知识蒸馏进行架构搜索。

**方法**：
1. 基于性能反馈迭代优化架构
2. 用历史高性能设计指导新的优化
3. 仅需 0.05 GPU-days 达到 SOTA

**对 SelfLLM 的启发**：
- NAS 的自动化证明模型架构可以由 LLM 自主优化
- 极低的搜索成本说明效率可控

---

### 4.3 AIDE: Code-Space Optimization for ML Engineering（2025）

**核心思想**：将 ML 工程建模为代码空间的树搜索优化问题。

**方法**：
1. LLM 生成 ML pipeline 代码
2. 执行代码，获得结果
3. 基于结果进行树形搜索，复用和改进方案
4. 在 MLE-Bench 和 Kaggle 竞赛中验证

**对 SelfLLM 的启发**：
- **极其相关**：AIDE 已经实现了 "LLM 写 ML 代码 → 执行 → 评估 → 优化" 的循环
- SelfLLM 是 AIDE 的自然扩展：从 ML pipeline 扩展到 LLM 训练 pipeline
- 树搜索优化策略可直接借鉴

---

## 5. 自动化数据搜集与训练数据生成

### 5.1 InSTA: Internet-Scale Training for Agents（2025）

**核心思想**：全自动化的 Web Agent 训练数据生成 pipeline。

**方法**：
1. LLM 标注 150,000 个网站的 Agent 任务
2. LLM Agent 完成任务并生成轨迹
3. LLM 过滤轨迹质量（82.6% 判断准确率）

**结果**：1.7B 模型训练后达到 56.9% 成功率，超越更大模型。

**对 SelfLLM 的启发**：
- 验证了 LLM 驱动的大规模数据生成 pipeline 的可行性
- 三阶段（生成任务 → 执行 → 过滤）模式可复用

---

### 5.2 ScrapeGraphAI-100k（2025）

**核心思想**：生产级别的真实 Web 数据提取数据集。

**内容**：93,695 个真实世界的 Web 提取样本，包含实际 prompt、schema、网页内容和 LLM 响应。

**对 SelfLLM 的启发**：
- 真实数据的重要性——纯合成数据会 collapse
- Web 提取工具链已经成熟

---

## 6. 推理时扩展与验证（Inference-Time Scaling & Verification）

### 6.1 OpenAI o-series (o1, o3, o4)

**核心思想**：通过推理时计算扩展（test-time compute scaling）提升推理能力。

**训练方法**：
- 大规模 RL 训练在思维链上
- 自博弈 + 基于验证的奖励（数学用证明助手验证，代码用编译器验证）
- 推理时的计算量越大，结果越好

**对 SelfLLM 的启发**：
- 可验证领域（数学、代码）是自训练的最佳切入点
- RL + 验证器是目前最有效的自训练方法
- SelfLLM 应该优先选择可自动验证的领域

---

### 6.2 EvolveR: Self-Evolving LLM Agents（2025）

**核心思想**：通过经验驱动的生命周期实现 Agent 自进化。

**方法**：
1. 离线阶段：自蒸馏，将交互经验合成为可复用的战略原则
2. 在线阶段：应用战略原则 + 策略强化进行迭代更新

**对 SelfLLM 的启发**：
- 离线-在线双阶段的设计模式
- 经验 → 原则 → 应用的知识提炼框架

---

## 7. 安全与对齐（Safety & Alignment）

### 7.1 Model Collapse 研究（Nature 2024, ICLR 2025）

**核心发现**：
- 在合成数据上迭代训练会导致 model collapse
- 即使 1‰ 的合成数据污染也可能触发
- 更大的模型可能放大 collapse

**关键缓解策略**：
1. **数据累积**：保留原始真实数据与合成数据混合（而非替换）
2. **固定大小子集**：限制每代的数据量
3. **合成数据验证**：外部验证器过滤合成数据

**对 SelfLLM 的启发**：
- **最大风险之一**：必须始终混合真实数据
- 需要构建专门的 collapse 检测机制
- 数据累积策略是必须的

---

### 7.2 International AI Safety Report（2025-2026）

**核心观点**：
- 递归自我改进已从理论走向实践
- 当前面临三大障碍：定义模糊、理论不足、评估碎片化
- 需要明确的安全边界和监控机制

**对 SelfLLM 的启发**：
- 安全不是可选项，是必须项
- 需要从 Day 1 就设计安全框架
- 能力上限阈值 + 熔断机制

---

## 8. 综合分析矩阵

| 方法 | 自写代码 | 自搜数据 | 自主训练 | 自我评估 | 策略优化 | 完整闭环 |
|------|---------|---------|---------|---------|---------|---------|
| STaR | ❌ | ❌ | ✅ | ✅(有限) | ❌ | ❌ |
| Self-Rewarding LM | ❌ | ✅(合成) | ✅ | ✅ | ❌ | ❌ |
| SPIN | ❌ | ✅(自博弈) | ✅ | ❌ | ❌ | ❌ |
| LADDER | ❌ | ✅(递归生成) | ✅ | ✅ | ✅(课程) | ❌ |
| SEAL | ❌ | ✅ | ✅ | ✅ | ✅(有限) | ❌ |
| LSP | ❌ | ✅(自博弈) | ✅ | ❌ | ❌ | ❌ |
| STOP | ✅(scaffolding) | ❌ | ❌ | ✅ | ✅ | ❌ |
| FunSearch | ❌ | ❌ | ❌ | ✅ | ✅(进化) | ❌ |
| AIDE | ✅(ML代码) | ❌ | ✅ | ✅ | ✅(树搜索) | ❌ |
| NNGPT | ✅(架构) | ❌ | ✅ | ✅ | ✅ | ❌ |
| Constitutional AI | ❌ | ✅(自修正) | ✅ | ✅ | ❌ | ❌ |
| **SelfLLM (ours)** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |

**关键洞察**：没有任何现有工作实现了完整的自主闭环。每个方法都只覆盖了 1-3 个维度。SelfLLM 的核心创新在于**将所有维度统一到一个自主系统中**。

## 9. 2026 年最新进展（论文与社区）

> 以下为 **2026 年 1–3 月** 前后在 arXiv / 顶会流程中出现的代表性工作；X/Twitter 上讨论多集中在论文转发、Hugging Face Papers 摘要与 ICLR 2026 相关 workshop，尚无统一「官方榜单」——此处以可核验的论文与开源仓库为准。

### 9.1 综述与系统视角

**Self-Improvement of Large Language Models: A Technical Overview and Future Outlook**（[arXiv:2603.25681](https://arxiv.org/abs/2603.25681)，2026-03-26）

- **框架**：将自改进系统抽象为**闭环生命周期**——数据获取、数据选择、模型优化、推理细化，外加**自主评估层**；模型在各阶段驱动自身改进。
- **价值**：与 SelfLLM 的模块划分（数据 / 训练 / 评估 / 策略）高度同构，可作为文献地图与术语对齐的入口。

### 9.2 预训练阶段的自改进（强后训模型当裁判）

**Self-Improving Pretraining: using post-trained models to pretrain better models**（[arXiv:2601.21343](https://arxiv.org/abs/2601.21343)，Meta FAIR 等）

- **思路**：流式预训练文档上，用 RL 优化「接下来 K 个 token」的生成；**强后训模型**作为裁判，在候选 rollout、原文后缀、改写后缀之间比较**质量、安全、事实性**。
- **结果**：相对标准预训练，事实性相对提升约 **36.2%**、安全性约 **18.5%**，整体生成质量 win rate 有大幅提升（论文报告最高约 **86.3%** 量级）。
- **对 SelfLLM**：把「评估信号」前移到预训练，与「自主评估层」设计一致；仍依赖**固定裁判模型**，非完全无监督闭环。

### 9.3 测试时自演化上下文（可学习技能）

**Learning to Self-Evolve (LSE)**（[arXiv:2603.18620](https://arxiv.org/abs/2603.18620)，[OpenReview](https://openreview.net/forum?id=zedEdPhmsA)）

- **思路**：用 RL 训练模型在**测试时**编辑自身上下文；树形演化 + UCB 类探索，将多步演化压成**单步 RL 目标**（每次编辑由下游任务改进给奖励）。
- **结果**：**4B** 模型在 Text-to-SQL（BIRD）、QA（MMLU-Redux 等）上，超过用 GPT-5 / Claude Sonnet 4.5 驱动的自演化策略及部分 prompt 优化基线；且可迁移指导其他模型。
- **对 SelfLLM**：与「策略优化 / program.md 人类编程实验方向」互补——LSE 学的是**改 prompt/上下文**，SelfLLM 侧重**改训练代码与数据管线**。

### 9.4 Agent 轨迹反思与经验规则

**Experiential Reflective Learning for Self-Improving LLM Agents**（[arXiv:2603.24639](https://arxiv.org/abs/2603.24639)）

- **思路**：对任务轨迹做反思，生成可迁移的**启发式规则**，按需检索注入上下文。
- **结果**：在 Gaia2 等基准上相对基线有可见成功率提升（论文报告约 **+7.8%** 量级）。
- **对 SelfLLM**：接近 EvolveR 路线的「经验 → 原则库 → 在线检索」，可并入 Meta-Controller 的记忆层。

### 9.5 系统提示词进化 + 权重联合优化

**Unifying Evolutionary Prompt Search and Reinforcement Learning for LLM Self-Improvement (E-SPL)**（[arXiv:2602.14697](https://arxiv.org/abs/2602.14697)）

- **思路**：多系统提示词下并行采样轨迹，**进化**（变异/交叉）更新 prompt，同时用 RL 更新权重；面向 easy→hard 泛化。
- **结果**：论文报告在相应设定下成功率从约 **38.8%** 提升至约 **45.1%**。
- **对 SelfLLM**：与「进化搜索 + 策略优化」一致，可对照 FunSearch / 策略记忆模块。

### 9.6 自博弈与长上下文

- **Language Self-Play (LSP)**（[arXiv:2509.07414](https://arxiv.org/abs/2509.07414)，ICLR 2026 DATA-FM workshop 等）：Challenger / Solver 双角色、同模型自博弈，**无额外外部数据**的持续训练；Llama-3.2-3B-Instruct 在指令遵循、数学、编码上均有提升。
- **SPELL: Self-Play RL for Evolving Long-Context LLMs**（[arXiv:2509.23863](https://arxiv.org/abs/2509.23863)）：面向长上下文的三角色循环（提问者 / 回答者 / 验证者）+ 课程与自适应奖励，论文报告推理类 benchmark 平均约 **+7.6 分**量级。

### 9.7 Agent 自进化（延续 2025 投稿线）

**EvolveR**（[arXiv:2510.16079](https://arxiv.org/abs/2510.16079)，ICLR 2026 流程）：离线轨迹蒸馏为可复用战略原则 + 在线交互与策略强化——与 ERL 同属「经验驱动」一派，适合作为 Agent 侧参考。

### 9.8 工程与社区：Karpathy autoresearch（2026-03）

**[karpathy/autoresearch](https://github.com/karpathy/autoresearch)**：单 GPU、固定 `prepare.py` 与数据，**仅改 `train.py`**，5 分钟固定 wall-clock、指标 **val_bpb**；社区报告一夜百余次实验、验证损失可显著下降（如 GitHub Discussions #43 等会话报告）。**不属于**合成训练数据或开放域数据，而是**算法/超参/架构搜索**的工程化闭环。

**公开讨论渠道（便于跟踪，非官方排名）**：论文作者与 Hugging Face Papers 的转发、GitHub Discussions、ICLR/NeurIPS 相关 account 的摘要帖；若需「一手」观点，建议以 **arXiv + 开源仓库** 为主。

---

## 10. 可直接复用的技术

| 来源 | 可复用技术 | 在 SelfLLM 中的用途 |
|------|----------|-------------------|
| Self-Rewarding LM | LLM-as-Judge + Iterative DPO | 评估引擎的核心机制 |
| FunSearch | 进化搜索 + 自动评估 | 策略优化器的搜索算法 |
| AIDE | 代码空间树搜索 | Code Generator 的搜索策略 |
| STOP | 递归代码自优化 | 训练代码的递归改进 |
| LSP | Challenger-Solver 自博弈 | 数据生成的核心机制 |
| STaR | 正确样本过滤 + 合理化 | 数据筛选策略 |
| Constitutional AI | 原则约束 + 自我批评 | 安全框架 |
| InSTA | 三阶段数据生成 pipeline | 数据搜集系统 |
| LADDER | 递归问题分解 | 自适应课程学习 |
| Self-Improving Pretraining | 后训模型当预训裁判 + RL | 预训/数据质量信号设计（需裁判模型） |
| LSE | 测试时上下文演化 + RL | 与「改训练代码」互补的上下文层 |
| E-SPL | 进化 prompt + RL 权重 | 策略/系统提示联合搜索 |
| 2603.25681 综述 | 生命周期 + 自主评估层 | 架构对齐与文献索引 |

---

*调研更新：2026 年 3 月（含 2026Q1 新论文）。将持续跟踪 arXiv 与顶会。*
