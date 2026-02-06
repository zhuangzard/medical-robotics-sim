# 数学一致性检查报告

**生成日期**: 2026-02-06  
**输入文件**:  
1. `research/paper_drafts/ALGORITHM_DESIGN.md` (以下简称 **ALGO**)  
2. `research/paper_drafts/SPEC.md` (以下简称 **SPEC**)  
3. `physics_core/sv_message_passing.py` (以下简称 **CODE**)  

**优先级**: CODE (通过测试) > SPEC > ALGO > 其他文档  

---

## 1. 统一符号表

以 SPEC.md §5 为准，列出所有数学符号及定义。

### 1.1 环境与状态

| 符号 | 含义 | 定义域 | SPEC 章节 |
|------|------|--------|-----------|
| $N$ | 交互体（节点）数量 | $\mathbb{Z}^+$ | §5.1 |
| $\mathbf{x}_i$ | 物体 $i$ 的位置 | $\mathbb{R}^3$ | §5.1 |
| $\dot{\mathbf{x}}_i$ | 物体 $i$ 的速度 | $\mathbb{R}^3$ | §5.1 |
| $\boldsymbol{\phi}_i$ | 物体 $i$ 的内禀属性（质量、摩擦、几何） | $\mathbb{R}^k$ | §5.1 |
| $\mathbf{s}_i^t$ | 物体 $i$ 在时刻 $t$ 的完整状态: $(\mathbf{x}_i^t, \dot{\mathbf{x}}_i^t, \boldsymbol{\phi}_i)$ | — | §5.1 |
| $\mathcal{G}^t = (\mathcal{V}, \mathcal{E}^t)$ | 时刻 $t$ 的交互图 | — | §5.1 |
| $\mathcal{N}(i)$ | 节点 $i$ 的邻居集合 | — | §5.1 |
| $r_{\text{cut}}$ | 接触图截断半径 | $\mathbb{R}^+$ | §5.1 |

### 1.2 边局部坐标系

| 符号 | 含义 | 交换 $i \leftrightarrow j$ 对称性 | SPEC 章节 |
|------|------|----------------------------------|-----------|
| $\mathbf{r}_{ij}$ | 位移向量: $\mathbf{x}_j - \mathbf{x}_i$ | **反对称** | §5.2 |
| $d_{ij}$ | 距离: $\|\mathbf{r}_{ij}\|$ | 对称 | §5.2 |
| $\dot{\mathbf{x}}_{ij}$ | 相对速度: $\dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i$ | 反对称 | §5.2 |
| $\mathbf{e}_1^{ij}$ | 径向单位向量: $\mathbf{r}_{ij} / d_{ij}$ | **反对称** | §5.2 |
| $\mathbf{v}_{ij}^{\perp}$ | 垂直速度分量 | 反对称 | §5.2 |
| $\mathbf{e}_2^{ij}$ | 切向单位向量: $\mathbf{v}_{ij}^{\perp} / \|\mathbf{v}_{ij}^{\perp}\|$ | **反对称** | §5.2 |
| $\mathbf{e}_3^{ij}$ | 副法线单位向量: $\mathbf{e}_1^{ij} \times \mathbf{e}_2^{ij}$ | **对称** ⚠️ | §5.2 |

### 1.3 SV 管道

| 符号 | 含义 | 备注 | SPEC 章节 |
|------|------|------|-----------|
| $\boldsymbol{\sigma}_{ij}$ | 标量不变量: $(d_{ij}, v_r, v_t, v_b, \|\dot{\mathbf{x}}_{ij}\|) \in \mathbb{R}^5$ | 旋转不变 | §5.3 |
| $v_r$ | 径向相对速度: $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ | **反对称** | §5.3 |
| $v_t$ | 切向相对速度: $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij}$ | 对称 | §5.3 |
| $v_b$ | 副法线相对速度: $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ | **反对称** | §5.3 |
| $\mathbf{h}_i \in \mathbb{R}^{d_h}$ | 节点 $i$ 的学习嵌入 | — | §5.3 |
| $\alpha_1, \alpha_2, \alpha_3$ | 标量力系数 | Force MLP 输出 | §5.3 |
| $\mathbf{F}_{ij}$ | 节点 $i$ 对节点 $j$ 的力: $\alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$ | — | §5.3 |
| $\mathbf{F}_i$ | 节点 $i$ 的合力: $\sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij}$ | — | §5.3 |
| $d_h$ | 节点嵌入隐藏维度 | 32 (PushBox), 64 (多物体) | §5.3 |
| $L$ | SV 消息传递层数 | 1 (PushBox), 2 (多物体) | §5.3 |

### 1.4 双流架构

| 符号 | 含义 | SPEC 章节 |
|------|------|-----------|
| $\mathbf{z}_{\text{policy}}$ | 策略流输出 | §5.4 |
| $\mathbf{z}_{\text{physics}}$ | 物理流输出（预测 box 加速度） | §5.4 |
| $\text{sg}(\cdot)$ | 停止梯度算子 | §5.4 |
| $\hat{\mathbf{a}}_i$ | 物体 $i$ 的预测加速度 | §5.4 |
| $\mathbf{a}_i^{\text{fd}}$ | 有限差分加速度: $(\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t) / \Delta t$ | §5.4 |

### 1.5 损失函数

| 符号 | 含义 | 公式 | SPEC 章节 |
|------|------|------|-----------|
| $\mathcal{L}$ | 总损失 | $\mathcal{L}_{\text{RL}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$ | §5.5 |
| $\mathcal{L}_{\text{RL}}$ | PPO 裁剪替代目标 | 标准 PPO | §5.5 |
| $\mathcal{L}_{\text{phys}}$ | 物理辅助损失 | $\frac{1}{|\mathcal{B}|} \sum \|\hat{\mathbf{a}}_i - \mathbf{a}_i^{\text{fd}}\|^2$ | §5.5 |
| $\mathcal{L}_{\text{reg}}$ | 能量正则化项（软约束） | 功-能一致性项 | §5.5 |
| $\lambda_{\text{phys}}$ | 物理损失权重 | 0.1 (warmup 后) | §5.5 |
| $\lambda_{\text{reg}}$ | 正则化权重 | 0.01 | §5.5 |
| $T_{\text{warmup}}$ | 物理损失预热期 | 50,000 步 | §5.5 |

---

## 2. 不一致列表

| # | 文档A | 文档B | 符号/公式 | 问题 | 修正 |
|---|-------|-------|-----------|------|------|
| 1 | **ALGO §2.4** | **CODE L155–230** | 动量守恒机制 | ALGO 使用有向边 + $\alpha_3$ 反对称化 ($\alpha_3^{ij} = v_r^{ij} \cdot g_\theta(\boldsymbol{\sigma}^{\text{sym}})$)；CODE 使用无向对 + 显式 $\pm \mathbf{F}$ 赋值 | **以 CODE 为准**。论文主体描述无向对方法；ALGO 的有向边证明可放入附录 |
| 2 | **ALGO §2.3.2** | **CODE L144–148** | Force MLP 输出 | ALGO 的 MLP 分为 `alpha12_mlp`（输出 2）和 `alpha3_mag_mlp`（输出 1）两个独立网络；CODE 使用单一 `force_mlp` 输出 3 个系数 | **以 CODE 为准**。无向对方法不需要分离 $\alpha_3$，单一 MLP 更简洁 |
| 3 | **ALGO §2.3.1** | **CODE L198–206** | 标量特征 $\boldsymbol{\sigma}_{ij}$ | ALGO: $\boldsymbol{\sigma}_{ij} = (d_{ij},\ v_r,\ v_t,\ v_b,\ \|\dot{\mathbf{x}}_{ij}\|)$（5 维），但 ALGO §4.2 伪代码用 $[\|d\|,\ |v_r|,\ v_t,\ |v_b|,\ \|v\|]$（取绝对值使之全对称）；CODE 直接用 $(d_{ij},\ v_r,\ v_t,\ v_b,\ v_{\text{norm}})$ 不取绝对值 | **以 CODE 为准**。无向对方法只处理一个方向 $(i<j)$，无需人为对称化标量特征。ALGO §4.2 的绝对值方案只在有向边方法中必要 |
| 4 | **ALGO §2.3.1** | **CODE L209–216** | 扩展标量特征中的节点嵌入 | ALGO: $\boldsymbol{\sigma}_{ij}^{\text{ext}} = [\boldsymbol{\sigma}_{ij} \| \mathbf{h}_i \| \mathbf{h}_j]$（直接拼接，非对称）；CODE: 使用 $\mathbf{h}_i + \mathbf{h}_j$（对称）和 $|\mathbf{h}_i - \mathbf{h}_j|$（对称）| **以 CODE 为准**。SPEC §5.6 已确认：直接拼接 $[\mathbf{h}_i \| \mathbf{h}_j]$ 破坏排列不变性 (MATH-2 问题) |
| 5 | **ALGO §2.4** | **SPEC §5.6** | $v_r$ 是否反对称 | ALGO 声称 $v_r^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ 是反对称的；SPEC §5.6 标注 MATH-1 指出这在有向边框架下需要仔细验证 | **不影响 CODE**（无向对方法不依赖此性质）。但 ALGO 的证明若保留在附录中，需要补充：$v_r^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_1^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = +v_r^{ij}$，即 $v_r$ 实际上是**对称**的，不是反对称的 ⚠️ |
| 6 | **ALGO §2.4** | **SPEC §5.3** | $v_r$ 对称性标注 | ALGO 称 $v_r$ 反对称并用它做 $\alpha_3$ 的反对称标记；SPEC §5.3 也标注 $v_r$ 为"**反对称**" | **两份文档都错了**。$v_r^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$，而 $\dot{\mathbf{x}}_{ji} = -\dot{\mathbf{x}}_{ij}$，$\mathbf{e}_1^{ji} = -\mathbf{e}_1^{ij}$，因此 $v_r^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = v_r^{ij}$。$v_r$ 是**对称**的。同理 $v_b$ 也是对称的（非反对称）。**SPEC §5.3 的对称性标注需要更正**。但对 CODE 无影响，因为 CODE 不依赖此性质 |
| 7 | **ALGO §5.1** | **SPEC §2.3** | 参数量 | ALGO: Physics Stream ≈ 46K params ($d_h=64, L=2$) 或 7.5K ($d_h=32, L=1$)；SPEC: Physics Stream = **6,019** params ($d_h=32, L=1$) | **以 SPEC/CODE 的精确计数 6,019 为准**。ALGO 的 7.5K 是近似估算。论文表格用 6,019 |
| 8 | **ALGO §5.1** | **SPEC §2.3** | 总参数量 | ALGO: 推荐设置 ($d_h=64, L=2$) 总计 ≈79K；SPEC: PushBox ($d_h=32, L=1$) 提取器总计 = **15,619** | 无矛盾（不同配置）。但论文需明确：PushBox 实验用 15,619（提取器）≈ 25K（含 PPO heads）；多物体实验用 ≈80K |
| 9 | **ALGO §2.5** | **CODE L275** | 节点编码器输入 | ALGO: $\text{MLP}_{\text{enc}}([\mathbf{x}_i, \dot{\mathbf{x}}_i, \boldsymbol{\phi}_i])$（含内禀属性）；CODE: MLP(6→32) 只接受 $[\text{pos}, \text{vel}]$（无 $\boldsymbol{\phi}_i$）| **以 CODE 为准**。PushBox 环境中 $\boldsymbol{\phi}_i$（质量、摩擦等）未包含在观测中。多物体扩展时可增加 |
| 10 | **ALGO §2.5** | **CODE L302–308** | 输出方式 | ALGO: 最终通过 $\hat{\mathbf{a}}_i = \text{MLP}_{\text{dec}}(\mathbf{h}_i^{(L)})$（解码器 MLP）；CODE: 直接输出最后一层的 `F_agg`，**无解码器 MLP** | **以 CODE 为准**。CODE 注释明确说明：通过解码器 MLP 会破坏 $\sum \mathbf{F}_i = \mathbf{0}$ 性质。ALGO 的 MLP 解码方案与守恒定理矛盾 ⚠️ |
| 11 | **ALGO §4.2** | **CODE L209–216** | 标量特征中的节点嵌入对称化 | ALGO 伪代码: `scalars = cat(scalars_sym, h[src], h[dst])`（不对称拼接）；CODE: `h_sum = h[pi] + h[pj]`, `h_diff_abs = (h[pi] - h[pj]).abs()` | **以 CODE 为准**。ALGO §4.2 伪代码与正文 §2.3.1 一致（都错），需要同步更新 |
| 12 | **ALGO §2.2** | **CODE L73–102** | 退化处理 | ALGO: 退化时 $\mathbf{e}_2^{ij} = (\mathbf{e}_1^{ij} \times \hat{\mathbf{z}}) / \|\cdot\|$，再退化到 $\hat{\mathbf{y}}$；CODE: 相同逻辑但用连续混合 `non_degenerate * e2_vel + (1 - non_degenerate) * e2_fall` | 逻辑一致，CODE 用可微的软切换实现。**无需修正**，但论文应注明实现使用软切换以保证梯度流 |
| 13 | **ALGO §2.2** | **SPEC §5.2** | $\mathbf{e}_3$ 对称性 | 两者一致：$\mathbf{e}_3^{ij} = +\mathbf{e}_3^{ji}$（对称）| ✅ 一致，无需修正 |
| 14 | **ALGO §2.6** | **SPEC §2.4** / **CODE L361–364** | Fusion 输入维度 | ALGO: $\mathbf{z} = \text{ReLU}(\mathbf{W}_f [\mathbf{z}_{\text{policy}} \| \text{sg}(\hat{\mathbf{a}}_{\text{box}})]  + \mathbf{b}_f)$；SPEC: `Linear(67→64)` = 64+3=67；CODE L361: `Linear(features_dim + 3, features_dim)` = 64+3=67 | ✅ 三者一致，67→64 |
| 15 | **ALGO docstring** | **CODE docstring** | $\alpha_3$ 反对称化描述 | CODE 文件顶部 docstring (L1–26) 仍然描述有向边 + $v_r$ 反对称化方法（"Antisymmetrize α3: multiply by signed radial velocity v_r"）；但实际实现 (L113–234) 使用无向对方法 | **需要更新 CODE docstring**。文件头注释与实际实现不匹配 |

---

## 3. EdgeFrame 定义对比

### 3.1 ALGORITHM_DESIGN.md 的定义（§2.2）

**有向边框架**：对每条有向边 $(i \to j)$：

$$\mathbf{e}_1^{ij} = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\| + \epsilon}$$

$$\mathbf{v}_{ij}^{\perp} = \dot{\mathbf{x}}_{ij} - (\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij})\,\mathbf{e}_1^{ij}, \quad \mathbf{e}_2^{ij} = \frac{\mathbf{v}_{ij}^{\perp}}{\|\mathbf{v}_{ij}^{\perp}\| + \epsilon}$$

$$\mathbf{e}_3^{ij} = \mathbf{e}_1^{ij} \times \mathbf{e}_2^{ij}$$

**对称性**：$\mathbf{e}_1^{ij} = -\mathbf{e}_1^{ji}$，$\mathbf{e}_2^{ij} = -\mathbf{e}_2^{ji}$，$\mathbf{e}_3^{ij} = +\mathbf{e}_3^{ji}$

**退化处理**：当 $\|\mathbf{v}_{ij}^{\perp}\| < \epsilon_{\text{deg}}$，用 $\hat{\mathbf{z}}$ 交叉积，再退化用 $\hat{\mathbf{y}}$。

**MLP 结构**：双 MLP 方案——
- `alpha12_mlp`: 对称标量 → $(\alpha_1, \alpha_2)$
- `alpha3_mag_mlp`: 对称标量 → $|\alpha_3|$，再乘以 $v_r$（声称反对称标记）

### 3.2 sv_message_passing.py 的实现

**无向对框架**：只处理 $i < j$ 的对，框架从 $i \to j$ 构建（与 ALGO 的 e1/e2/e3 构建公式完全相同）。

**关键区别**：

| 方面 | ALGO | CODE |
|------|------|------|
| 边处理方向 | 所有有向边 $(i,j)$ 和 $(j,i)$ | 仅无向对 $\{i,j\}$（$i<j$）|
| 守恒机制 | 依赖 $\alpha_k$ 的对称/反对称性质 | 硬编码 $+\mathbf{F}$ 给 $j$，$-\mathbf{F}$ 给 $i$ |
| Force MLP | 两个独立 MLP + $v_r$ 反对称标记 | 单一 `force_mlp` 输出 3 个系数 |
| 节点嵌入 | $[\mathbf{h}_i \| \mathbf{h}_j]$（不对称拼接）| $\mathbf{h}_i + \mathbf{h}_j$ 和 $|\mathbf{h}_i - \mathbf{h}_j|$（对称化）|
| 标量输入 | $[d, |v_r|, v_t, |v_b|, \|v\|]$（取绝对值） | $[d, v_r, v_t, v_b, \|v\|]$（保留符号）|
| 输出 | $\text{MLP}_{\text{dec}}(\mathbf{h}_i^{(L)})$（解码器 MLP） | 直接输出 `F_agg`（无解码器）|

### 3.3 差异分析和最终版本

**EdgeFrame 构建公式**：两者一致（$\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ 的定义相同）。差异在于**如何使用**这些框架。

**最终版本**（以 CODE 为准）：

```
对于每个无向对 {i, j}（i < j）：
    1. 构建框架：e1, e2, e3 = build_edge_frames(pos, vel, i, j)
       - e1 = (x_j - x_i) / ||x_j - x_i||
       - e2 = v_perp / ||v_perp||  （含退化软切换）
       - e3 = e1 × e2
    2. 标量化：σ = [d, v_r, v_t, v_b, ||v||]
       - h_sym = [h_i + h_j, |h_i - h_j|]
       - input = [σ || h_sym]
    3. Force MLP：(α1, α2, α3) = MLP(input)
    4. 向量化：F = α1·e1 + α2·e2 + α3·e3
    5. 牛顿第三定律：node_j += F,  node_i += -F
    6. 节点更新：h_new = h + MLP([h, F_agg])
```

**不输出加速度解码**：物理流直接返回 `F_agg`（保持 $\sum \mathbf{F}_i = \mathbf{0}$），不经过解码器 MLP。

---

## 4. 守恒证明验证

### 4.1 代码的守恒机制

CODE 使用 **无向对 + 显式 $\pm\mathbf{F}$** 方法：

```python
# 对每个无向对 {i, j}，i < j:
force_ij = alpha1 * e1 + alpha2 * e2 + alpha3 * e3   # 一个力向量

# Newton's 3rd law:
F_agg[j] += force_ij      # scatter_add
F_agg[i] += -force_ij     # scatter_add
```

**守恒证明**（CODE 方法）：

$$\sum_{i=1}^{N} \mathbf{F}_i = \sum_{\{i,j\}: i<j} (+\mathbf{F}_{ij} + (-\mathbf{F}_{ij})) = \sum_{\{i,j\}: i<j} \mathbf{0} = \mathbf{0}$$

这是**构造性证明**——对任意网络参数 $\theta$、任意输入状态、任意图拓扑都成立。证明只依赖于 `scatter_add` 的线性性和 $+\mathbf{F} + (-\mathbf{F}) = \mathbf{0}$ 的代数恒等式。

**已通过验证**：CODE 自测 `verify_momentum_conservation()` 对 100 次随机试验、N=2,3,4,5,8 节点全部通过（$\|\sum \mathbf{F}\| < 10^{-4}$）。

### 4.2 ALGO Theorem 1 的问题

ALGO §2.4 的有向边证明存在一个**数学错误**：

**ALGO 的关键声明**：

> "Key insight: The scalar features satisfy $\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$... Since the same MLP processes both, we have $\alpha_k^{ij} = \alpha_k^{ji}$ for $k = 1, 2, 3$."

这对 $\alpha_1, \alpha_2$ 成立（使用对称标量后），但 $\alpha_3$ 需要反对称化。ALGO 随后提出了修正方案：$\alpha_3^{ij} = v_r^{ij} \cdot g_\theta(\boldsymbol{\sigma}^{\text{sym}})$。

**但 $v_r$ 的对称性分析有误**：

$$v_r^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$$
$$v_r^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_1^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij} = v_r^{ij}$$

因此 $v_r^{ij} = v_r^{ji}$（**对称**，不是反对称）！

ALGO 声称 $v_r$ 满足 $v_r^{ij} = -v_r^{ji}$，这是**错误的**。如果用 $v_r$ 作为反对称标记，则：
$$\alpha_3^{ij} + \alpha_3^{ji} = v_r^{ij} g(\sigma) + v_r^{ji} g(\sigma) = 2 v_r^{ij} g(\sigma) \neq 0$$

**守恒不成立**。

### 4.3 正确的反对称标记（如果要修复有向边证明）

需要一个在 $i \leftrightarrow j$ 交换下真正反对称的标量。检查所有 5 个标量的对称性：

| 标量 | $i \to j$ | $j \to i$ | 对称性 |
|------|-----------|-----------|--------|
| $d_{ij} = \|\mathbf{r}_{ij}\|$ | $d$ | $d$ | 对称 |
| $v_r = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ | $v_r$ | $(-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = v_r$ | **对称** |
| $v_t = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij}$ | $v_t$ | $(-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_2^{ij}) = v_t$ | **对称** |
| $v_b = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ | $v_b$ | $(-\dot{\mathbf{x}}_{ij}) \cdot (+\mathbf{e}_3^{ij}) = -v_b$ | **反对称** ✅ |
| $\|\dot{\mathbf{x}}_{ij}\|$ | $v$ | $v$ | 对称 |

**$v_b$（副法线分量）是唯一的反对称标量**。

如果要保留有向边证明，正确的修正应该是：
$$\alpha_3^{ij} = v_b^{ij} \cdot g_\theta(\boldsymbol{\sigma}^{\text{sym}})$$

其中 $\boldsymbol{\sigma}^{\text{sym}} = (d_{ij}, v_r, v_t, |v_b|, \|\dot{\mathbf{x}}_{ij}\|)$。

### 4.4 论文 Theorem 1 的建议改写

**方案 A（推荐）**：论文主体使用 CODE 的无向对证明

> **Theorem 1 (Linear Momentum Conservation).** *For any network parameters $\theta$, the SV-pipeline guarantees $\sum_{i=1}^{N} \mathbf{F}_i = \mathbf{0}$.*
>
> **Proof.** We process each unordered pair $\{i,j\}$ exactly once. For each pair, a single force vector $\mathbf{F}_{ij} = \alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$ is computed, then $+\mathbf{F}_{ij}$ is assigned to node $j$ and $-\mathbf{F}_{ij}$ to node $i$. The total force is:
> $$\sum_{i} \mathbf{F}_i = \sum_{\{i,j\}} (+\mathbf{F}_{ij} - \mathbf{F}_{ij}) = \mathbf{0} \quad \square$$

**方案 B**：如果要保留有向边证明（放入附录），需要：
1. 将 $v_r$ 标记修正为 $v_b$
2. 修正 SPEC §5.3 中 $v_r$ 和 $v_b$ 的对称性标注
3. 修正 ALGO §2.4 中的推导

---

## 5. 最终确认的公式集

以下是所有关键公式的最终版本，以 CODE 实现为权威。

### 5.1 边局部坐标系

$$\mathbf{e}_1 = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\| + \epsilon}$$

$$\mathbf{v}^{\perp} = (\dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i) - \bigl[(\dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i) \cdot \mathbf{e}_1\bigr]\,\mathbf{e}_1$$

$$\mathbf{e}_2 = \begin{cases} \mathbf{v}^{\perp} / \|\mathbf{v}^{\perp}\| & \text{if } \|\mathbf{v}^{\perp}\| > \epsilon_{\text{deg}} \\[6pt] \text{fallback}(\mathbf{e}_1, \hat{\mathbf{z}}, \hat{\mathbf{y}}) & \text{otherwise} \end{cases}$$

$$\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$$

**实现细节**：CODE 使用连续软切换（可微）：
$$\mathbf{e}_2 = \lambda \cdot \mathbf{e}_2^{\text{vel}} + (1 - \lambda) \cdot \mathbf{e}_2^{\text{fall}}, \quad \lambda = \mathbb{1}[\|\mathbf{v}^{\perp}\| > \epsilon_{\text{deg}}]$$

### 5.2 标量化

$$\boldsymbol{\sigma} = \begin{pmatrix} d_{ij} \\[2pt] v_r \\[2pt] v_t \\[2pt] v_b \\[2pt] \|\dot{\mathbf{x}}_{ij}\| \end{pmatrix} \in \mathbb{R}^5, \quad \text{where} \quad \begin{aligned} d_{ij} &= \|\mathbf{x}_j - \mathbf{x}_i\| \\ v_r &= \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1 \\ v_t &= \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2 \\ v_b &= \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3 \\ \end{aligned}$$

$$\boldsymbol{\sigma}^{\text{ext}} = [\boldsymbol{\sigma} \;\|\; (\mathbf{h}_i + \mathbf{h}_j) \;\|\; |\mathbf{h}_i - \mathbf{h}_j|] \in \mathbb{R}^{5 + 2d_h}$$

### 5.3 Force MLP 与向量化

$$(\alpha_1, \alpha_2, \alpha_3) = \text{MLP}_\theta(\boldsymbol{\sigma}^{\text{ext}}) \in \mathbb{R}^3$$

$$\mathbf{F}_{ij} = \alpha_1 \,\mathbf{e}_1 + \alpha_2 \,\mathbf{e}_2 + \alpha_3 \,\mathbf{e}_3$$

**MLP 结构**：`Linear(69, 32) → LayerNorm → ReLU → Linear(32, 3)`

### 5.4 牛顿第三定律（硬编码守恒）

$$\mathbf{F}_j \mathrel{+}= +\mathbf{F}_{ij}, \quad \mathbf{F}_i \mathrel{+}= -\mathbf{F}_{ij} \quad \forall\, \{i,j\}:\ i < j$$

### 5.5 节点更新（残差连接）

$$\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MLP}_{\text{upd}}\!\left([\mathbf{h}_i^{(\ell)} \,\|\, \mathbf{F}_i]\right)$$

**MLP 结构**：`Linear(35, 32) → LayerNorm → ReLU → Linear(32, 32)`

### 5.6 节点编码器

$$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{enc}}\!\left([\mathbf{x}_i \,\|\, \dot{\mathbf{x}}_i]\right) \in \mathbb{R}^{d_h}$$

**MLP 结构**：`Linear(6, 32) → LayerNorm → ReLU → Linear(32, 32)`

### 5.7 物理流输出

$$\hat{\mathbf{a}}_i = \mathbf{F}_i^{(L)} \quad \text{（直接输出最后一层的力聚合，无解码器 MLP）}$$

### 5.8 双流融合

$$\mathbf{z}_{\text{policy}} = \text{MLP}_{\text{policy}}(\mathbf{o}_t) \in \mathbb{R}^{64}$$

$$\mathbf{z} = \text{ReLU}\!\left(\mathbf{W}_f \left[\mathbf{z}_{\text{policy}} \,\big\|\, \text{sg}(\hat{\mathbf{a}}_{\text{box}})\right] + \mathbf{b}_f\right) \in \mathbb{R}^{64}$$

其中 $\mathbf{W}_f \in \mathbb{R}^{64 \times 67}$，$\text{sg}(\cdot)$ 表示停止梯度。

### 5.9 动量守恒定理

**Theorem 1** *(Linear Momentum Conservation).* 对任意网络参数 $\theta$，SV 管道保证：

$$\sum_{i=1}^{N} \mathbf{F}_i = \mathbf{0}$$

*Proof.* 每个无序对 $\{i,j\}$ 产生一个力向量 $\mathbf{F}_{ij}$，分别以 $+\mathbf{F}_{ij}$ 和 $-\mathbf{F}_{ij}$ 赋给两端节点。因此：

$$\sum_{i=1}^{N} \mathbf{F}_i = \sum_{\{i,j\}:\ i<j} \left(+\mathbf{F}_{ij} + (-\mathbf{F}_{ij})\right) = \sum_{\{i,j\}} \mathbf{0} = \mathbf{0} \quad \blacksquare$$

### 5.10 物理辅助损失

$$\mathcal{L}_{\text{phys}} = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{s}^t, \mathbf{s}^{t+1}) \in \mathcal{B}} \sum_{i \in \text{objects}} \left\| \hat{\mathbf{a}}_i^t - \frac{\dot{\mathbf{x}}_i^{t+1} - \dot{\mathbf{x}}_i^t}{\Delta t} \right\|^2$$

### 5.11 总损失

$$\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

其中 $\lambda_{\text{phys}} = 0.1$（经过 $T_{\text{warmup}} = 50\text{K}$ 步线性预热），$\lambda_{\text{reg}} = 0.01$。

---

## 附录：$v_r$ 和 $v_b$ 的完整对称性推导

这里给出 SPEC §5.3 中标注的 $v_r$ "反对称"错误的完整证明。

**定义**：

$$v_r^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}, \quad v_b^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$$

**交换 $i \leftrightarrow j$**：

- $\dot{\mathbf{x}}_{ji} = -\dot{\mathbf{x}}_{ij}$（反对称）
- $\mathbf{e}_1^{ji} = -\mathbf{e}_1^{ij}$（反对称）
- $\mathbf{e}_2^{ji} = -\mathbf{e}_2^{ij}$（反对称）
- $\mathbf{e}_3^{ji} = +\mathbf{e}_3^{ij}$（对称）

**$v_r$ 的对称性**：
$$v_r^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_1^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = +(\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}) = +v_r^{ij}$$

∴ $v_r$ 是**对称**的 ✅

**$v_t$ 的对称性**：
$$v_t^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_2^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_2^{ij}) = +v_t^{ij}$$

∴ $v_t$ 是**对称**的 ✅

**$v_b$ 的对称性**：
$$v_b^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_3^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (+\mathbf{e}_3^{ij}) = -(\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}) = -v_b^{ij}$$

∴ $v_b$ 是**反对称**的 ✅

**结论**：在 5 个标量 $(d, v_r, v_t, v_b, \|v\|)$ 中，只有 $v_b$ 是反对称的。SPEC §5.3 和 ALGO §2.4 中将 $v_r$ 标为反对称是错误的。

**对 CODE 的影响**：无。CODE 使用无向对方法，不依赖任何标量的对称/反对称性质。

**对论文的影响**：如果附录中保留有向边证明，$\alpha_3$ 的反对称标记应改用 $v_b$。

---

*报告完成。核心发现：*
1. *CODE 的无向对 + $\pm\mathbf{F}$ 方法是最简洁、最稳健的守恒机制*
2. *ALGO 的有向边证明中 $v_r$ 对称性分析有误（$v_r$ 是对称的，不是反对称的），需用 $v_b$ 替代*
3. *CODE docstring 与实现不一致，需更新*
4. *ALGO 的 MLP 解码器方案会破坏守恒性，CODE 正确地直接输出力聚合*
