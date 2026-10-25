# 内置示例数据集

这些 CSV 由 `scripts/make_datasets.py` 生成（随机种子 `20261025`，相对噪声 0.2%–0.4%），
用于快速验证 GUI 的导入、训练、结果与推理流程。

| 文件 | 变量（X → y） | 生成公式 | 说明 |
| --- | --- | --- | --- |
| `linear.csv` | `x → y` | `y = 2.5x − 1.2` | 最简线性关系 |
| `damped_oscillator.csv` | `t → amplitude` | `A = 1.5 e^(−0.45t) cos(2.1t)` | 指数 + 三角 |
| `ideal_gas.csv` | `V, T, n → P` | `P = nRT / V`，`R = 8.314` | 三变量，常数需要学习 |
| `projectile_range.csv` | `v0, theta → R` | `R = v0² sin(2θ) / g`，`g = 9.80665` | 三角 + 平方 |
| `pendulum_period.csv` | `L, g → T` | `T = 2π√(L/g)` | 根式关系 |
| `radioactive_decay.csv` | `t, N0, lambda → N` | `N = N0 e^(−λt)` | 多变量指数 |
| `spring_energy.csv` | `k, x → E` | `E = ½ k x²` | 需要常数 0.5 |
| `coulomb_force.csv` | `q1, q2, r → F` | `F = 8.99 q1 q2 / r²` | 三变量库仑力 |
| `two_feature.csv` | `x0, x1 → y` | `y = 0.8x0x1 + 1.4x0 − 0.7x1 + 0.3` | 多项式交互项 |

## 推荐量纲

`index.json` 为每个数据集保存了 `x_units` / `y_units`，向量顺序为 **L, M, T, I, Θ, N, J**（长度、质量、时间、电流、温度、物质的量、光强）。在 GUI 中载入示例数据时，这些量纲会自动填入对应的下拉框；也可以手动修改或选择「自定义量纲…」。

> 量纲分析在数据页默认关闭：PhySO 的自由常数默认按无量纲处理。只有在确认自由常数假设成立时再开启，否则可能过滤掉本应有效的表达式。

## 使用建议

- 先用 `linear.csv` 或 `damped_oscillator.csv` 验证环境与推理流程。
- 多变量数据（`ideal_gas.csv`、`coulomb_force.csv`）建议增加 epoch 数或 batch size。
- 数据量级跨度较大时，在数据页开启「标准化数据」；结果页会自动把表达式换算回原始量纲。
- 目标变量 `y` 必须是数值列；文本列不会出现在变量选择列表中。

`index.json` 保存了数据集的显示名、公式、描述与推荐列，GUI 的「示例数据集」列表从该文件读取。
