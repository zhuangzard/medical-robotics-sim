# PhysRobot — 数学一致性检查报告

**检查人**: Round 2 数学一致性检查官  
**日期**: 2026-02-06  
**检查范围**:
- `research/paper_drafts/ALGORITHM_DESIGN.md` (算法文档)
- `research/paper_drafts/PAPER_OUTLINE.md` (论文大纲)
- `research/paper_drafts/RELATED_WORK.md` (相关工作)
- `physics_core/sv_message_passing.py` (SV 代码 — 最终参考实现)
- `physics_core/edge_frame.py` (旧版 EdgeFrame 代码)
- `baselines/physics_informed.py` (旧版 PhysRobot baseline)
- `research/paper_drafts/REVIEWER_FINAL_VERDICT.md` (R1-R4 审稿意见)

---

## 1. 统一符号表（最终版，以代码 `sv_message_passing.py` 为准）

| 符号 | 含义 | 类型 | 对称性 (ij↔ji) | 算法文档 | 代码 | 大纲 | 备注 |
|------|------|------|----------------|---------|------|------|------|
| $\mathbf{x}_i$ | 节点 $i$ 位置 | $\mathbb{R}^3$ | — | ✅ $\mathbf{x}_i$ | ✅ `pos[i]` | ✅ "position" | 一致 |
| $\dot{\mathbf{x}}_i$ | 节点 $i$ 速度 | $\mathbb{R}^3$ | — | ✅ $\dot{\mathbf{x}}_i$ | ✅ `vel[i]` | ✅ "velocity" | 一致 |
| $\mathbf{r}_{ij}$ | 位移向量 $\mathbf{x}_j - \mathbf{x}_i$ | $\mathbb{R}^3$ | **反对称** | ✅ $\mathbf{x}_j - \mathbf{x}_i$ | ✅ `pos[dst] - pos[src]` | ✅ "displacement" | 一致 |
| $d_{ij}$ | 距离 $\|\mathbf{r}_{ij}\|$ | $\mathbb{R}^+$ | **对称** | ✅ $\|\mathbf{x}_j - \mathbf{x}_i\|$ | ✅ `torch.norm(r_ij)` | ✅ "distance" | 一致 |
| $\dot{\mathbf{x}}_{ij}$ | 相对速度 $\dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i$ | $\mathbb{R}^3$ | **反对称** | ✅ | ✅ `vel[dst] - vel[src]` | — | 一致 |
| $\mathbf{e}_1^{ij}$ | 径向基向量 $\mathbf{r}_{ij}/\|\mathbf{r}_{ij}\|$ | $\hat{\mathbb{R}}^3$ | **反对称** | ✅ | ✅ `e1 = r_ij / (d_ij + EPS)` | "displacement vector" | 一致 |
| $\mathbf{e}_2^{ij}$ | 切向基向量 (来自 $\mathbf{v}^\perp$) | $\hat{\mathbb{R}}^3$ | **反对称** | ✅ | ✅ `e2 = v_perp / (v_perp_norm + EPS)` | ⚠️ "up-vector" | **不一致 — 见 §3.1** |
| $\mathbf{e}_3^{ij}$ | 副法线 $\mathbf{e}_1 \times \mathbf{e}_2$ | $\hat{\mathbb{R}}^3$ | **对称** | ✅ | ✅ `e3 = cross(e1, e2)` | — | 一致 |
| $v_r^{ij}$ | 径向相对速度 $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij}$ | $\mathbb{R}$ | ⚠️ **见 §2.1** | ✅ "antisymmetric" | **未使用** | — | **关键争议** |
| $v_t^{ij}$ | 切向投影 $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij}$ | $\mathbb{R}$ | **对称** | ✅ | ✅ `v_t = dot(v_rel, e2)` | — | 一致 |
| $v_b^{ij}$ | 副法线投影 $\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$ | $\mathbb{R}$ | ⚠️ **见 §2.1** | ✅ "antisymmetric" | **未使用** | — | **关键争议** |
| $\alpha_1^{ij}$ | 径向力系数 | $\mathbb{R}$ | **对称** (MLP on sym inputs) | ✅ | ✅ `alphas[:, 0:1]` | — | 一致 |
| $\alpha_2^{ij}$ | 切向力系数 | $\mathbb{R}$ | **对称** (MLP on sym inputs) | ✅ | ✅ `alphas[:, 1:2]` | — | 一致 |
| $\alpha_3^{ij}$ | 副法线力系数 | $\mathbb{R}$ | **必须反对称** | $v_r \cdot g_\theta(\sigma^{\text{sym}})$ | **直接由 MLP 输出** | — | **关键不一致 — 见 §2.2** |
| $\mathbf{F}_{ij}$ | 边力向量 | $\mathbb{R}^3$ | **应反对称** | $\alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$ | ✅ (通过 ±F 分配) | "antisymmetric exchange" | **实现策略不同 — 见 §2.3** |
| $\mathbf{h}_i$ | 节点嵌入 | $\mathbb{R}^{d_h}$ | — | ✅ | ✅ `h` | — | 一致 |
| $\boldsymbol{\sigma}_{ij}$ | 标量特征向量 | $\mathbb{R}^5$ | 算法: 全对称; 代码: 含 $v_r, v_b$ | ✅ | ✅ `scalars_geom` | — | **不一致 — 见 §2.4** |

---

## 2. 逐项不一致分析

### 2.1 关键争议：$v_r$ 的对称性

**R1 审稿 (REVIEWER_FINAL_VERDICT.md §1.1, MATH-1) 指出**:

> $v_r$ 不是反对称的，应用 $v_b$ 替代

**详细分析**:

$$v_r^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij} = (\dot{\mathbf{x}}_j - \dot{\mathbf{x}}_i) \cdot \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\|}$$

交换 $i, j$:

$$v_r^{ji} = (\dot{\mathbf{x}}_i - \dot{\mathbf{x}}_j) \cdot \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}) = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij} = v_r^{ij}$$

**结论: $v_r^{ij} = +v_r^{ji}$ (对称!)。审稿人 R1 的判断是正确的。**

**$v_b$ 的对称性**:

$$v_b^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3^{ij}$$

交换 $i, j$:

$$v_b^{ji} = \dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_3^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (+\mathbf{e}_3^{ij}) = -v_b^{ij}$$

**结论: $v_b^{ij} = -v_b^{ji}$ (反对称!)。$v_b$ 是正确的反对称 marker。**

**$v_t$ 的对称性** (验证):

$$v_t^{ij} = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij}$$

$$v_t^{ji} = (-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_2^{ij}) = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2^{ij} = v_t^{ij}$$

**结论: $v_t^{ij} = +v_t^{ji}$ (对称)。**

**完整投影对称性总结**:

| 投影 | 公式 | 向量对称性 | 基向量对称性 | 乘积对称性 |
|------|------|-----------|-------------|-----------|
| $v_r = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1$ | 反对称 · 反对称 | $(-) \cdot (-) = (+)$ | **对称** |
| $v_t = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_2$ | 反对称 · 反对称 | $(-) \cdot (-) = (+)$ | **对称** |
| $v_b = \dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_3$ | 反对称 · 对称 | $(-) \cdot (+) = (-)$ | **反对称** ✅ |

### 2.2 $\alpha_3$ 反对称化：算法文档 vs 代码

#### 算法文档 (`ALGORITHM_DESIGN.md` §2.4) — **有错误**

算法文档写道:

$$\alpha_3^{ij} = v_r^{ij} \cdot g_\theta(\boldsymbol{\sigma}_{ij}^{\text{sym}})$$

并声称这保证反对称性。但如 §2.1 所证，$v_r$ 是**对称的**，所以:

$$\alpha_3^{ij} + \alpha_3^{ji} = v_r^{ij} \cdot g(\sigma^{\text{sym}}) + v_r^{ji} \cdot g(\sigma^{\text{sym}}) = 2 v_r^{ij} \cdot g(\sigma^{\text{sym}}) \neq 0$$

**这不工作。** 需要用 $v_b$ 替代 $v_r$:

$$\alpha_3^{ij} = v_b^{ij} \cdot g_\theta(\boldsymbol{\sigma}_{ij}^{\text{sym}}) \quad \Rightarrow \quad \alpha_3^{ij} + \alpha_3^{ji} = v_b^{ij} \cdot g + (-v_b^{ij}) \cdot g = 0 \quad \checkmark$$

**修正位置**: `ALGORITHM_DESIGN.md` §2.4, "Fix: Antisymmetrize $\alpha_3$" 段落。

- 原文: "$\alpha_3^{ij} = v_r^{ij} \cdot g_\theta(\boldsymbol{\sigma}_{ij}^{\text{sym}})$"
- 修正为: "$\alpha_3^{ij} = v_b^{ij} \cdot g_\theta(\boldsymbol{\sigma}_{ij}^{\text{sym}})$"
- 以及 §2.5 Step 3: "$\alpha_3 = v_r^{ij} \cdot g^{(\ell)}$" → "$\alpha_3 = v_b^{ij} \cdot g^{(\ell)}$"
- 以及 §4.2 pseudocode: `alpha3 = v_r * alpha3_mag` → `alpha3 = v_b * alpha3_mag`

#### 代码 (`sv_message_passing.py`) — **完全绕过了这个问题**

代码采用了一种**完全不同的策略**来保证牛顿第三定律: 它不走"每条有向边独立计算、靠标量对称性自动抵消"的路线，而是:

1. 只处理**无向边对** (通过 `_extract_undirected_pairs`, 只保留 `src < dst`)
2. 对每对只计算**一个** $\mathbf{F}_{ij}$
3. 然后**硬编码** 分配: 节点 $j$ 得到 $+\mathbf{F}_{ij}$, 节点 $i$ 得到 $-\mathbf{F}_{ij}$

```python
# Node j receives +F_ij
F_agg.scatter_add_(0, pj.unsqueeze(-1).expand_as(force_ij), force_ij)
# Node i receives -F_ij
F_agg.scatter_add_(0, pi.unsqueeze(-1).expand_as(force_ij), -force_ij)
```

**这个策略从根本上保证了 $\mathbf{F}_{ij} = -\mathbf{F}_{ji}$，不需要任何关于 $\alpha_3$ 的反对称化。**

**评估**:
- ✅ **正确性**: 代码的牛顿第三定律保证是 **correct by construction** — 无论 MLP 输出什么，力总是成对相反的
- ✅ **简洁性**: 无需额外的反对称 MLP 或 marker 变量
- ⚠️ **代价**: 标量输入 (包含 $v_r$, $v_b$) 直接传入 MLP，没有区分对称/反对称分量。这意味着**同一无向边对的 force MLP 输入不依赖于方向**，力的方向完全由 ±F 硬编码决定
- ⚠️ **与算法文档的架构差异**: 算法文档描述的是"每条有向边独立计算，靠对称性自动抵消"；代码实现的是"只算一次，硬编码 ±"。两种策略都能保证守恒，但它们是**不同的算法**。

### 2.3 力守恒证明：三种不同的声称

| 来源 | 守恒机制 | 正确性 |
|------|---------|--------|
| **算法文档** §2.4 | $\alpha_k^{ij} = \alpha_k^{ji}$ (标量对称) + $\mathbf{e}_k^{ij}$ 反对称 → 自动抵消; $\alpha_3$ 需额外处理 | ⚠️ 方向正确但 $v_r$ marker 有误，应为 $v_b$ |
| **代码** `sv_message_passing.py` | 只处理无向边对, 硬编码 $+\mathbf{F} / -\mathbf{F}$ | ✅ 正确且已通过 100 次随机测试 |
| **大纲** §3.3 | "antisymmetric exchange: $m_{ij} = -m_{ji}$" + 显式对称化 | ✅ 方向正确但细节未展开 |

**最终裁定 (以代码为准)**:
代码使用的"无向边对 + 硬编码 ±F"策略是**正确的、可验证的**。论文应描述这个实际实现的策略，而非算法文档中的理论策略（后者有 $v_r$ bug）。

### 2.4 标量特征向量 $\boldsymbol{\sigma}_{ij}$ 的构成

#### 算法文档 (§2.3.1)

$$\boldsymbol{\sigma}_{ij} = \begin{pmatrix} \|\mathbf{r}_{ij}\| \\ v_r \\ v_t \\ v_b \\ \|\dot{\mathbf{x}}_{ij}\| \end{pmatrix}$$

并声称"ALL symmetric"（在 §4.2 pseudocode 中写 `scalars_sym = [d_ij, |v_r|, v_t, |v_b|, norm(v_rel)]`）

**矛盾**: 算法文档 §2.3.1 列出的是 $v_r, v_t, v_b$（含符号），但 §4.2 pseudocode 用的是 $|v_r|, v_t, |v_b|$（取绝对值使对称）。两处不一致。

#### 代码 (`sv_message_passing.py`)

```python
scalars_geom = torch.cat([d_ij, v_r, v_t, v_b, v_norm], dim=-1)  # [P, 5]
```

代码直接用 $v_r, v_t, v_b$（含符号），没有取绝对值。但因为代码只处理无向边对 (canonical direction $i<j$)，所以**根本不存在"交换 $i,j$ 后是否对称"的问题** — 每条边只被处理一次。

**最终裁定**: 代码的做法是**自洽的**。因为采用了无向边对策略，标量特征无需是对称的。但如果论文描述的是算法文档的有向边策略，则标量必须全部对称化（取绝对值），且需要 $v_b$ 作为 $\alpha_3$ marker。

**建议**: 论文应与代码实现保持一致，描述无向边对策略。如果描述有向边策略，则需修正为:

$$\boldsymbol{\sigma}_{ij}^{\text{sym}} = (d_{ij},\; |v_r|,\; v_t,\; |v_b|,\; \|\dot{\mathbf{x}}_{ij}\|)$$

### 2.5 节点嵌入聚合: $[\mathbf{h}_i \| \mathbf{h}_j]$ vs 对称聚合

#### 算法文档 (§2.3.1)

$$\boldsymbol{\sigma}_{ij}^{\text{ext}} = [\boldsymbol{\sigma}_{ij} \| \mathbf{h}_i \| \mathbf{h}_j]$$

**R1 审稿 (MATH-2) 指出**: 拼接 $[\mathbf{h}_i \| \mathbf{h}_j]$ 不是排列不变的 — 交换 $i,j$ 后输入改变。

#### 代码 (`sv_message_passing.py`)

```python
h_sum = h[pi] + h[pj]             # [P, node_dim]  order-invariant
h_diff_abs = (h[pi] - h[pj]).abs() # [P, node_dim]  order-invariant
scalars = torch.cat([scalars_geom, h_sum, h_diff_abs], dim=-1)
```

**代码使用了对称聚合**: $\mathbf{h}_i + \mathbf{h}_j$ (对称) 和 $|\mathbf{h}_i - \mathbf{h}_j|$ (对称)。

**最终裁定 (以代码为准)**: 代码的对称聚合是**正确的**。对于无向边对策略，这在形式上不是必需的（因为只处理一次），但它确保了 canonical direction 的选择 ($i<j$ vs $j<i$) 不影响结果，这是好的工程实践。

**修正位置**: `ALGORITHM_DESIGN.md` §2.3.1, 修正公式为:

$$\boldsymbol{\sigma}_{ij}^{\text{ext}} = [\boldsymbol{\sigma}_{ij} \| (\mathbf{h}_i + \mathbf{h}_j) \| |\mathbf{h}_i - \mathbf{h}_j|]$$

### 2.6 EdgeFrame 构建: 三种不同的实现

| 来源 | $\mathbf{e}_2$ 的构建 | 正确性 |
|------|---------------------|--------|
| **算法文档** §2.2 | 从相对速度: $\mathbf{e}_2 = \mathbf{v}_{ij}^\perp / \|\mathbf{v}_{ij}^\perp\|$ (投影去除径向分量) | ✅ 物理意义好, 反对称 |
| **代码** `sv_message_passing.py` `build_edge_frames()` | 与算法文档一致: 从相对速度 | ✅ 一致 |
| **旧代码** `baselines/physics_informed.py` | 固定 up vector $\hat{\mathbf{z}}$: $\mathbf{e}_2 = \mathbf{e}_1 \times \hat{\mathbf{z}}$ | ❌ 丢失速度信息; $\hat{\mathbf{z}}$ 破坏旋转等变性 |
| **旧代码** `physics_core/edge_frame.py` | 通过 MLP 编码 $[\mathbf{r}_{ij}, \|\mathbf{r}_{ij}\|, \mathbf{v}_{rel}, \|\mathbf{v}_{rel}\|]$ | ❌ MLP 破坏反对称性 |
| **大纲** §3.4 | "displacement vector + reference up-vector" | ❌ 与算法文档/代码不一致 |

**退化处理对比**:

| 来源 | 当 $\|\mathbf{v}^\perp\| < \epsilon$ | 当 $\mathbf{e}_1 \approx \pm\hat{\mathbf{z}}$ |
|------|--------------------------------------|-----------------------------------------------|
| 算法文档 | fallback 到 $\hat{\mathbf{z}}$ | fallback 到 $\hat{\mathbf{y}}$ |
| 代码 `sv_message_passing.py` | fallback 到 $\hat{\mathbf{z}}$, smooth blending via `non_degenerate` mask | 二级 fallback 到 $\hat{\mathbf{y}}$ via `use_y` mask |
| `baselines/physics_informed.py` | 始终用 $\hat{\mathbf{z}}$ | 无处理 (bug) |

**最终裁定**: 算法文档和新代码 `sv_message_passing.py` 的 EdgeFrame 构建**基本一致**, 代码有更完善的退化处理。旧代码和大纲需要更新。

### 2.7 $\mathbf{e}_3$ 的对称性证明验证

算法文档给出了证明:

> $\mathbf{e}_3^{ji} = \mathbf{e}_1^{ji} \times \mathbf{e}_2^{ji} = (-\mathbf{e}_1^{ij}) \times (-\mathbf{e}_2^{ij}) = +\mathbf{e}_3^{ij}$

**验证**: 设 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^3$. 叉积的双线性反对称性:

$$(-\mathbf{a}) \times (-\mathbf{b}) = (-1)(-1)(\mathbf{a} \times \mathbf{b}) = \mathbf{a} \times \mathbf{b}$$

**证明正确。** $\mathbf{e}_3^{ij}$ 在 $i \leftrightarrow j$ 交换下确实是**对称的** (不变)。✅

### 2.8 动量守恒定理验证

**算法文档定理**: $\sum_i \mathbf{F}_i = \mathbf{0}$

**代码验证** (`sv_message_passing.py` `verify_momentum_conservation()`):
- 100 次随机试验, 4 节点全连接图
- 随机初始化位置、速度、网络参数
- $\|\sum_i \mathbf{F}_i\| < 10^{-4}$

**代码的守恒机制** (§2.2 已分析):
- 硬编码 $+\mathbf{F}_{ij}$ 给 $j$, $-\mathbf{F}_{ij}$ 给 $i$
- 对所有边求和: $\sum_i \mathbf{F}_i = \sum_{(i,j)} (\mathbf{F}_{ij} - \mathbf{F}_{ij}) = 0$

**这是精确成立的** (仅受浮点误差限制), 与 MLP 参数无关。✅

---

## 3. 代码-公式对齐详细检查

### 3.1 Forward Pass 对比

| 步骤 | 算法文档 §2.5 | 代码 `SVPhysicsCore.forward()` | 一致? |
|------|-------------|-------------------------------|------|
| **节点编码** | $\mathbf{h}_i^{(0)} = \text{MLP}_{\text{enc}}([\mathbf{x}_i, \dot{\mathbf{x}}_i, \boldsymbol{\phi}_i])$ | `h = self.encoder(cat([pos, vel]))` (无 $\boldsymbol{\phi}_i$) | ⚠️ 代码缺少物理属性 $\boldsymbol{\phi}_i$ |
| **Edge Frame** | §2.2 算法 | `build_edge_frames()` | ✅ 一致 |
| **标量化** | $[d, v_r, v_t, v_b, \|v\|]$ + $[\mathbf{h}_i \| \mathbf{h}_j]$ | `[d, v_r, v_t, v_b, v_norm]` + `[h_sum, h_diff_abs]` | ⚠️ 节点嵌入聚合方式不同 (见 §2.5) |
| **MLP** | 两个 MLP: $\alpha_{1,2}$ 和 $\alpha_3 = v_r \cdot g(\sigma^{sym})$ | 一个 MLP → 3 系数 | ❌ **架构不同**: 算法用独立 $\alpha_3$ MLP + 反对称 marker; 代码用统一 MLP |
| **向量化** | $\alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$ | `alpha1 * e1 + alpha2 * e2 + alpha3 * e3` | ✅ 公式一致 |
| **守恒机制** | 标量对称性 + 基向量反对称性自动抵消 | 无向边对 + 硬编码 ±F | ❌ **机制不同** (但两者都正确) |
| **节点更新** | $\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MLP}([\mathbf{h}_i, \mathbf{M}_i])$ | `h_new = h + self.node_update(cat([h, F_agg]))` | ✅ 残差连接一致 |
| **输出** | $\hat{\mathbf{a}}_i = \text{MLP}_{\text{dec}}(\mathbf{h}_i^{(L)})$ (解码器) | 直接输出 `forces` (最后一层的 $F_{agg}$, 无解码器) | ❌ **不同**: 算法有解码器 MLP, 代码无 |

### 3.2 关键差异总结

1. **$\alpha_3$ 处理**: 算法文档用独立 MLP + 反对称 marker (但 marker 选错了); 代码用统一 MLP + 硬编码 ±F (正确但不同)
2. **输出**: 算法文档有 $\text{MLP}_{\text{dec}}$ 解码器; 代码直接输出聚合力 (代码方案更好——解码器会破坏守恒性)
3. **节点属性**: 算法文档包含 $\boldsymbol{\phi}_i$ (质量、摩擦系数等); 代码只用 position + velocity
4. **多层力累积**: 算法文档每层产生消息 $\mathbf{m}_{ij}$; 代码只保留最后一层的力

---

## 4. R1 审稿发现的数学问题 — 逐项确认

### 4.1 MATH-1: $v_r$ 对称性 — **审稿人正确**

**问题**: 算法文档 §2.4 用 $v_r$ 作为 $\alpha_3$ 的反对称 marker。
**结论**: $v_r^{ij} = v_r^{ji}$ (对称)，不能用作反对称 marker。应改为 $v_b$ 或其他反对称标量。
**代码状态**: 代码通过无向边对策略完全绕过了此问题。但**算法文档需要修正**以备论文使用有向边描述。

### 4.2 MATH-2: $[\mathbf{h}_i, \mathbf{h}_j]$ 排列不变性 — **审稿人正确**

**问题**: 拼接 $[\mathbf{h}_i \| \mathbf{h}_j]$ 不满足排列不变性。
**代码状态**: 已修正为 $(\mathbf{h}_i + \mathbf{h}_j, |\mathbf{h}_i - \mathbf{h}_j|)$。
**算法文档状态**: 未修正。

### 4.3 X2 (Reviewer Final Verdict): $\alpha_3$ marker — **部分正确**

审稿人说"用 $v_r$（错的），R2 已指出"。这是正确的——$v_r$ 不是反对称的。但代码用了完全不同的策略来保证守恒（无向边对 + ±F），所以代码本身是正确的。

### 4.4 X8: EdgeFrame 构建三方不一致 — **已确认**

- 算法文档: 相对速度 ✅
- 新代码 `sv_message_passing.py`: 相对速度 ✅
- 旧代码 `baselines/physics_informed.py`: 固定 up=[0,0,1] ❌
- 旧代码 `physics_core/edge_frame.py`: MLP ❌
- 大纲: "displacement + up" ❌

---

## 5. 最终确认公式（以代码 `sv_message_passing.py` 为准）

### 5.1 EdgeFrame 构建 (Canonical Direction $i \to j$, $i < j$)

$$\mathbf{e}_1 = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\| + \epsilon}$$

$$\mathbf{v}_{\perp} = \dot{\mathbf{x}}_{ij} - (\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1)\,\mathbf{e}_1$$

$$\mathbf{e}_2 = \begin{cases} \mathbf{v}_{\perp} / (\|\mathbf{v}_{\perp}\| + \epsilon) & \text{if } \|\mathbf{v}_{\perp}\| > \epsilon_{\text{deg}} \\ (\mathbf{e}_1 \times \hat{\mathbf{z}}) / \|\cdot\| & \text{if } \|\mathbf{e}_1 \times \hat{\mathbf{z}}\| > \epsilon_{\text{deg}} \\ (\mathbf{e}_1 \times \hat{\mathbf{y}}) / \|\cdot\| & \text{otherwise} \end{cases}$$

$$\mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$$

### 5.2 标量化

$$\boldsymbol{\sigma} = (d_{ij},\; v_r,\; v_t,\; v_b,\; \|\dot{\mathbf{x}}_{ij}\|) \in \mathbb{R}^5$$

节点嵌入 (对称聚合):
$$\boldsymbol{\sigma}^{\text{ext}} = [\boldsymbol{\sigma} \| (\mathbf{h}_i + \mathbf{h}_j) \| |\mathbf{h}_i - \mathbf{h}_j|] \in \mathbb{R}^{5 + 2d_h}$$

### 5.3 力计算

$$(\alpha_1, \alpha_2, \alpha_3) = \text{MLP}_\theta(\boldsymbol{\sigma}^{\text{ext}}) \in \mathbb{R}^3$$

$$\mathbf{F}_{ij} = \alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$$

### 5.4 守恒保证 (硬编码 ±F 策略)

对每个无向边对 $\{i, j\}$ ($i < j$):
- 节点 $j$ 接收 $+\mathbf{F}_{ij}$
- 节点 $i$ 接收 $-\mathbf{F}_{ij}$

$$\mathbf{F}_i^{\text{agg}} = \sum_{j \in \mathcal{N}(i)} \mathbf{F}_{ij}^{\text{signed}}$$

$$\sum_{i=1}^{N} \mathbf{F}_i^{\text{agg}} = \sum_{\{i,j\} \in \mathcal{E}} (\mathbf{F}_{ij} - \mathbf{F}_{ij}) = \mathbf{0} \qquad \forall\, \theta$$

### 5.5 节点更新

$$\mathbf{h}_i' = \mathbf{h}_i + \text{MLP}_{\text{upd}}([\mathbf{h}_i \| \mathbf{F}_i^{\text{agg}}])$$

### 5.6 输出

$$\text{output}_i = \mathbf{F}_i^{\text{agg}} \quad (\text{直接输出聚合力, 无解码器})$$

---

## 6. 完整修正清单

### 6.1 算法文档 (`ALGORITHM_DESIGN.md`) 需修正

| 位置 | 当前内容 | 修正为 | 原因 |
|------|---------|--------|------|
| §2.4 "Fix" 段 | $\alpha_3^{ij} = v_r^{ij} \cdot g_\theta(\sigma^{sym})$ | $\alpha_3^{ij} = v_b^{ij} \cdot g_\theta(\sigma^{sym})$ | $v_r$ 对称, $v_b$ 反对称 |
| §2.4 证明末尾 | $v_r^{ij} - v_r^{ij} = 0$ | $v_b^{ij} - v_b^{ij} = 0$ | 同上 |
| §2.3.1 ext 公式 | $[\boldsymbol{\sigma} \| \mathbf{h}_i \| \mathbf{h}_j]$ | $[\boldsymbol{\sigma} \| (\mathbf{h}_i + \mathbf{h}_j) \| |\mathbf{h}_i - \mathbf{h}_j|]$ | 排列对称性 (MATH-2) |
| §2.5 Step 3 | $\alpha_3 = v_r^{ij} \cdot g^{(\ell)}$ | $\alpha_3 = v_b^{ij} \cdot g^{(\ell)}$ | 同 §2.4 |
| §4.2 pseudocode | `alpha3 = v_r * alpha3_mag` | `alpha3 = v_b * alpha3_mag` | 同 §2.4 |
| §4.2 `scalars_sym` | `[d_ij, \|v_r\|, v_t, \|v_b\|, ...]` | `[d_ij, v_r, v_t, \|v_b\|, ...]` 或描述两种方案 | $v_r$ 已经对称, 不需绝对值; $v_b$ 取绝对值使对称 |
| §2.4 Key insight | $\boldsymbol{\sigma}_{ij} = \boldsymbol{\sigma}_{ji}$ (all symmetric) | 注明 $v_r, v_t$ 自然对称; $v_b$ 需取绝对值; 或改用无向边对策略 | 与代码对齐 |
| §2.5 Step 3 / §2.3.2 | 两个独立 MLP: $\text{MLP}_\alpha$ 和 $g^{(\ell)}$ | 可合并为单一 MLP → 3 系数 (如代码), 或保留分离但修正 marker | 架构对齐 |
| §2.5 Step 5 decode | $\hat{\mathbf{a}}_i = \text{MLP}_{\text{dec}}(\mathbf{h}_i^{(L)})$ | 直接输出 $\mathbf{F}_i^{\text{agg}}$ | 解码器 MLP 破坏守恒性; 代码无解码器 |

### 6.2 论文大纲 (`PAPER_OUTLINE.md`) 需修正

| 位置 | 当前内容 | 修正为 |
|------|---------|--------|
| §3.3 | "explicitly symmetrize by computing $m_{ij}$ and setting reverse to $-m_{ij}$" | 精确描述: 只处理无向边对, 硬编码 ±F |
| §3.4 EdgeFrame | "displacement vector + reference up-vector" | "displacement + relative velocity tangent (with gravity fallback)" |
| §3.5 Conservation loss | $L_{cons} = \lambda_1 \|\sum_i F_i\|^2 + \lambda_2 \max(0, -\Delta E_{diss})$ | $L_{phys} = \text{FD acceleration MSE}$ (以代码为准); 守恒是架构保证, 不需要 loss |
| §4.1 PushBox state dim | "18-dim" | "16-dim" (以代码为准) |
| §4.1 Robot DOF | "7 DoF" | "2 DoF" (以代码为准) |

### 6.3 相关工作 (`RELATED_WORK.md`) 需修正

| 问题 | 修正 |
|------|------|
| 未引用 Dynami-CAL (Sharma & Fink, 2025) | 在 §2.2 或 §2.4 添加引用和定位 |
| Ref [23] Battaglia 2016 标注为 "C-GNS" | 修正为 "Interaction Networks" |

---

## 7. 定理/引理推导正确性

### 7.1 线性动量守恒定理 — ✅ 正确 (代码版本)

代码的无向边对 + ±F 策略是**平凡地正确的**:

$$\sum_i \mathbf{F}_i^{\text{agg}} = \sum_{\{i,j\}} (+\mathbf{F}_{ij}) + \sum_{\{i,j\}} (-\mathbf{F}_{ij}) = \mathbf{0}$$

不需要任何关于 MLP 对称性或基向量反对称性的假设。

### 7.2 线性动量守恒定理 — ⚠️ 有误 (算法文档版本)

算法文档 §2.4 的推导在 $\alpha_3$ 步骤有误 (用了 $v_r$ 而非 $v_b$)。修正后推导正确:

1. $\alpha_1^{ij} = \alpha_1^{ji}$, $\mathbf{e}_1^{ij} = -\mathbf{e}_1^{ji}$ → $\alpha_1 \mathbf{e}_1$ 项抵消 ✅
2. $\alpha_2^{ij} = \alpha_2^{ji}$, $\mathbf{e}_2^{ij} = -\mathbf{e}_2^{ji}$ → $\alpha_2 \mathbf{e}_2$ 项抵消 ✅
3. $\alpha_3^{ij} = v_b^{ij} g(\sigma^{sym}) = -v_b^{ji} g(\sigma^{sym}) = -\alpha_3^{ji}$, $\mathbf{e}_3^{ij} = +\mathbf{e}_3^{ji}$ → $\alpha_3 \mathbf{e}_3$ 项抵消 ✅ **(仅在修正 $v_r \to v_b$ 后)**

### 7.3 EdgeFrame 反对称性引理 — ✅ 正确

$\mathbf{e}_1^{ij} = -\mathbf{e}_1^{ji}$: 直接由 $\mathbf{r}_{ij} = -\mathbf{r}_{ji}$ 得出 ✅
$\mathbf{e}_2^{ij} = -\mathbf{e}_2^{ji}$: 由 $\mathbf{v}_{ij}^\perp = -\mathbf{v}_{ji}^\perp$ 得出 ✅ (需验证投影去除的反对称性)

详细验证 $\mathbf{v}^\perp$ 反对称性:
$$\mathbf{v}_{ji}^\perp = \dot{\mathbf{x}}_{ji} - (\dot{\mathbf{x}}_{ji} \cdot \mathbf{e}_1^{ji})\mathbf{e}_1^{ji} = (-\dot{\mathbf{x}}_{ij}) - ((-\dot{\mathbf{x}}_{ij}) \cdot (-\mathbf{e}_1^{ij}))(-\mathbf{e}_1^{ij})$$
$$= -\dot{\mathbf{x}}_{ij} - (\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij})(-\mathbf{e}_1^{ij}) = -\dot{\mathbf{x}}_{ij} + (\dot{\mathbf{x}}_{ij} \cdot \mathbf{e}_1^{ij})\mathbf{e}_1^{ij} = -\mathbf{v}_{ij}^\perp$$

✅ 反对称性成立。

$\mathbf{e}_3^{ij} = +\mathbf{e}_3^{ji}$: 由 $(-\mathbf{e}_1) \times (-\mathbf{e}_2) = +\mathbf{e}_1 \times \mathbf{e}_2$ 得出 ✅

### 7.4 角动量不守恒的说明 — ✅ 正确

算法文档 §2.4 Remark:

> 角动量守恒需要 $\mathbf{F}_{ij} \parallel \mathbf{r}_{ij}$, 即 $\alpha_2 = \alpha_3 = 0$ (中心力)。我们故意放松此约束以建模摩擦。

这是正确的。切向力 ($\alpha_2 \mathbf{e}_2$) 和副法线力 ($\alpha_3 \mathbf{e}_3$) 产生力矩，不保角动量。

---

## 8. 总结

### ✅ 已确认正确的公式
1. EdgeFrame 反对称性引理 ($\mathbf{e}_1, \mathbf{e}_2$ 反对称; $\mathbf{e}_3$ 对称)
2. 动量守恒定理 (代码的无向边对 + ±F 版本)
3. 向量化公式 $\mathbf{F} = \alpha_1 \mathbf{e}_1 + \alpha_2 \mathbf{e}_2 + \alpha_3 \mathbf{e}_3$
4. 投影对称性: $v_r$ 对称, $v_t$ 对称, $v_b$ 反对称
5. 角动量不守恒的物理论证

### ❌ 需修正的公式/描述
1. **$\alpha_3$ marker**: 算法文档用 $v_r$ (对称), 应改为 $v_b$ (反对称) — `ALGORITHM_DESIGN.md` §2.4, §2.5, §4.2
2. **节点嵌入聚合**: 算法文档用 $[\mathbf{h}_i \| \mathbf{h}_j]$ (不对称), 应改为 $(\mathbf{h}_i + \mathbf{h}_j, |\mathbf{h}_i - \mathbf{h}_j|)$ — `ALGORITHM_DESIGN.md` §2.3.1
3. **守恒机制**: 论文应描述代码实际使用的无向边对策略, 而非有向边自动抵消策略
4. **输出层**: 算法文档有 $\text{MLP}_{\text{dec}}$, 代码直接输出力 — 代码的做法更好 (保守恒)
5. **大纲参数**: 环境维度 (18→16), DOF (7→2), EdgeFrame 描述, Conservation loss 描述

### ⚠️ 架构差异 (非错误, 但需统一描述)
1. 代码: 单一 MLP → 3 系数; 算法文档: 分离 $\alpha_{1,2}$ MLP + $\alpha_3$ MLP
2. 代码: 无向边对 + 硬编码 ±F; 算法文档: 有向边 + 标量对称性自动抵消
3. 代码: 直接输出力; 算法文档: 通过解码器输出加速度

**最终建议**: 论文写作应以代码 `sv_message_passing.py` 的实际实现为准。代码的实现策略更简洁、更稳健、且已通过验证。算法文档的理论框架提供了有价值的物理直觉和数学解释, 但具体公式需要按上述清单修正。

---

*数学一致性检查完成。*
