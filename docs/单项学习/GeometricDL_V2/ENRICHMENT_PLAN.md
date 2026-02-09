# GeometricDL V2 章节深化计划

## 目标
每个章节内容丰富 2-3 倍，通过"小白提问 + 专家解释"的双角色对话模式深入解释理论。

## 架构
```
Wave 1 (并行): Chapter 1, 2, 3, 4
Wave 2 (并行): Chapter 5, 6, 7

每个章节:
  - Agent A (Anthropic): 小白学习者 - 阅读原文，提出疑问
  - Agent B (Anthropic): 专家导师 - 深入解释，补充内容
  - Agent C (OpenAI): 质量监督 - 审核扩展内容，确保准确性
```

## 输出格式
每个章节生成 `chapter{N}/enriched.html`，包含：
1. 原始内容（保留）
2. 🔍 深入解释区块（新增）
3. ❓ 小白问答区块（新增）
4. 💡 直觉理解区块（新增）
5. 🔗 概念关联区块（新增）

## 状态追踪
- [ ] Chapter 1: Introduction
- [ ] Chapter 2: Learning in High Dimensions
- [ ] Chapter 3: Geometric Priors
- [ ] Chapter 4: Geometric Domains
- [ ] Chapter 5: Geometric DL Models
- [ ] Chapter 6: Problems & Applications
- [ ] Chapter 7: Historic Perspective
