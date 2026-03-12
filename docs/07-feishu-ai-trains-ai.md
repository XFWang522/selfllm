# AI 训练 AI：让 Coding Agent 进入模型训练循环

## 一句话

**当 AI Coding 产品差异化坍缩为模型能力本身，下一个突破口在于：让 AI 不仅写应用代码，还能写训练代码、跑训练、看结果、改策略——形成 AI 训练 AI 的闭环。**

---

## 背景

### 1. AI Coding 产品差异化正在消失

OpenClaw 开源后，React Loop + Context Manager 这套 AI Coding Agent 的核心架构已经被彻底商品化。Trae + Claude ≈ Claude Code ≈ Cursor + Claude——**产品体验完全挂钩底层模型能力**，IDE 壳子不再构成护城河。

这意味着：**谁的模型 coding 能力强，谁的 AI coding 产品就强**。产品层面的工程优化空间正在收窄，决定性因素回归到模型本身。

### 2. Dario Amodei 的棋盘寓言：我们在第 40 格

Anthropic CEO Dario Amodei 在 2026 年 2 月 Dwarkesh 播客中使用了经典的"棋盘放米"寓言（[原视频](https://www.youtube.com/watch?v=n1E9IZfvGMA)）：

> 棋盘 64 格，每格翻倍。**前 39 格的总和，不过是后 24 格的零头。我们现在站在第 40 格。**

他同时在多个场合给出关键判断：
- **Scaling Law 没有撞墙**，RL scaling 展现出与预训练相同的 log-linear 规律
- **2026 年将"激进加速"（Radical Acceleration）**，90% 概率在 2026 底-2027 实现"数据中心里的天才国度"
- 软件工程将在 **1-2 年内被端到端自动化**（设计→架构→实现→测试）
- 当前处于 **"半人马阶段"（Centaur Phase）**——人+AI > 纯 AI > 纯人，但这个阶段"可能非常短暂"
- **编程正在成为"濒死技能"**

**核心推论**：模型能力的发展潜力依然巨大。Claude 在编程能力上领先其他模型约 3 个月。从 2C 场景（聊天/写作）向 2B 场景（专业工程/企业应用）的迁移正在加速。

### 3. 模型训练：高成本、低认知带宽、高度适合自动化

当前基座模型训练的现实：

| 特征 | 说明 |
|------|------|
| **成本极高** | 一次 70B 预训练 ~$10M-$50M，一次后训练实验 ~$1K-$100K |
| **认知带宽低** | 核心工作是"在同一个问题里反复尝试不同的 trick"——调超参、换数据配比、试新方法 |
| **反馈周期长** | 一次实验数小时到数天，人类等待+分析+决策是主要瓶颈 |
| **可验证** | Loss、benchmark 分数、评估指标——结果可量化比较 |
| **模式化** | 大量训练代码是模板化的，修改集中在配置和策略层 |

**这是一个天然适合 AI agent 自动化的领域**——明确的目标函数、可量化的反馈、模式化的操作、重复性的迭代。

### 4. 当前缺口：AI Coding Agent 不会训模型

目前的 AI Coding Agent（Claude Code、Cursor、OpenHands 等）核心能力是写应用代码，**不涉及训练模型的工作**。根本原因：

- **无法获得 training reward**：coding model 的 RL 训练中，rollout 的 reward 来自代码测试通过率。但"训练一个模型"的结果需要几小时甚至几天才能返回——这个 rollout 时间尺度让 RL 训练不可能实现
- **缺乏训练领域感知**：模型没有在"修改训练代码 → 观察训练结果 → 归因分析"这个循环中被优化过
- **没有训练代码的经验数据**：SWE-bench 等 benchmark 全部是应用代码，不包含训练代码场景

**结果**：当前最强的 coding model 可以写出复杂的 Web 应用，但不知道该怎么调一个 learning rate schedule。

### 5. Karpathy 的 autoresearch：概念验证已完成

2026 年 3 月，Andrej Karpathy 发布 [autoresearch](https://github.com/karpathy/autoresearch)（26.8K stars），核心设计：

```
循环 {
  1. AI Agent 修改 train.py（模型架构、超参、优化器…）
  2. 跑 5 分钟训练
  3. 测 val_bpb（验证集 bits per byte）
  4. 变好 → git commit；变差 → git revert
  5. 重复
}
```

**关键结果**：
- 一夜 126 次实验，val_bpb 从 0.9979 → 0.9697（显著提升）
- 两天 ~700 次自动修改，~20 个有效改进可迁移到更大模型
- Agent 自主发现了 batch size 优化、初始化缩放、weight decay 策略等 trick
- Shopify CEO 复现后报告 19% 验证集提升
- 12 次实验/小时，睡一觉起来约 100 次实验

**Karpathy Loop 的启示**：
- **人类的角色变了**：不再写 `train.py`，而是写 `program.md`——"编程研究方向"而非"编程代码"
- **但这还只是 scaffolding 级别的优化**——底层模型权重没有变，只是外围代码在迭代
- **下一步**：如果这个 loop 的结果能反哺模型训练本身呢？

---

## 核心洞察

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   AI Coding Product 差异化 → 归结为模型能力                   │
│                         ↓                                   │
│   模型能力提升 → 归结为训练效率和策略                          │
│                         ↓                                   │
│   训练效率 → 高度适合 AI Agent 自动化（Karpathy 已验证）       │
│                         ↓                                   │
│   但当前 coding model 完全没有训练领域的能力                   │
│                         ↓                                   │
│   ★ 机会：填补 "AI coding" 到 "AI training" 之间的鸿沟 ★      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**如果我们能让 AI Coding Agent 学会训练模型**——不仅仅是 Karpathy Loop 中的 "修改代码跑跑看"，而是真正理解训练过程、具备调参直觉、能归因分析失败原因——那么：

1. **模型迭代速度将从"周级"变为"天级"甚至"小时级"**
2. **训练成本中的人力浪费（错误实验、低效探索）将大幅降低**
3. **可能发现人类研究员没想到的训练 trick**（FunSearch 的先例）
4. **形成 "更好的模型 → 更强的训练能力 → 更好的模型" 的飞轮**

---

## 计划

（暂空）

---

## 附录

### A. 关键参考

| 来源 | 链接 | 核心信息 |
|------|------|---------|
| Karpathy autoresearch | [GitHub](https://github.com/karpathy/autoresearch) | 26.8K stars，单 GPU + AI Agent 自主跑实验 |
| Dario Amodei × Dwarkesh | [YouTube](https://www.youtube.com/watch?v=n1E9IZfvGMA) | 棋盘寓言、激进加速、Scaling Law 未撞墙 |
| Dario × Nikhil Kamath | [YouTube](https://www.youtube.com/watch?v=68ylaeBbdsg) | 编程是濒死技能、半人马阶段、海啸比喻 |
| Dario × Business Insider | [文章](https://embed.businessinsider.com/anthropic-ceo-dario-amodei-centaur-phase-of-software-engineering-jobs-2026-2) | Centaur Phase 可能非常短暂 |
| OpenClaw ContextEngine | [文档](https://openclaws.io/blog/openclaw-contextengine-deep-dive/) | React Loop + Context Manager 商品化 |
| Self-Rewarding LM (Meta) | ICML 2024 | LLM 自我评估 + 迭代 DPO |
| AIDE | 2025 | LLM 写 ML pipeline 代码并优化 |

### B. Dario Amodei 近期核心判断摘录

> "我们现在站在棋盘的第 40 格。前 39 格的所有增长加在一起，可能只是后面 24 格的零头。"

> "编程正在成为濒死技能。"（2026.3.2, The AI Corner）

> "Scaling law 没有撞墙。RL scaling 展现出与预训练相同的 log-linear 规律。"

> "2026 将迎来激进加速。我们 90% 概率在 2026 底-2027 实现'数据中心里的天才国度'。"

> "最令人惊讶的不是技术进步本身，而是公众认知与技术现实之间巨大的、不断扩大的鸿沟。"

> "软件工程将在 1-2 年内被端到端自动化。半人马阶段可能非常短暂。"
