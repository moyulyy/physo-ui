"""Regenerate the bundled demo datasets.

Usage:  python scripts/make_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets"
N = 320
RNG = np.random.default_rng(20261025)


def noisy(y: np.ndarray, rel: float = 0.003) -> np.ndarray:
    sigma = rel * (np.abs(y) + np.max(np.abs(y)) * 0.01)
    return y + RNG.normal(0.0, 1.0, size=y.shape) * sigma


def save(name: str, columns: dict[str, np.ndarray]) -> None:
    df = pd.DataFrame(columns)
    df.to_csv(OUT / name, index=False, float_format="%.8g")
    print(f"  {name:28s} {df.shape[0]:4d} rows x {df.shape[1]} cols")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Linear model ------------------------------------------------------
    x = RNG.uniform(-2.0, 2.0, N)
    y = 2.5 * x - 1.2
    save("linear.csv", {"x": x, "y": noisy(y, 0.002)})

    # 2. Damped harmonic oscillator ---------------------------------------
    t = np.sort(RNG.uniform(0.0, 6.0, N))
    y = 1.5 * np.exp(-0.45 * t) * np.cos(2.1 * t)
    save("damped_oscillator.csv", {"t": t, "amplitude": noisy(y, 0.004)})

    # 3. Ideal gas: P = n R T / V (R = 8.314 kPa·L/(mol·K)) ----------------
    V = RNG.uniform(2.0, 8.0, N)
    T = RNG.uniform(260.0, 360.0, N)
    n = RNG.uniform(0.004, 0.012, N)
    P = n * 8.314 * T / V
    save("ideal_gas.csv", {"V": V, "T": T, "n": n, "P": noisy(P, 0.003)})

    # 4. Projectile range: R = v0^2 sin(2 theta) / g ---------------------
    v0 = RNG.uniform(8.0, 20.0, N)
    theta = RNG.uniform(0.20, 1.35, N)
    g = 9.80665
    R = v0**2 * np.sin(2.0 * theta) / g
    save("projectile_range.csv", {"v0": v0, "theta": theta, "R": noisy(R, 0.003)})

    # 5. Pendulum period: T = 2 pi sqrt(L/g) ------------------------------
    L = RNG.uniform(0.3, 2.5, N)
    grav = RNG.uniform(9.4, 10.2, N)
    period = 2.0 * np.pi * np.sqrt(L / grav)
    save("pendulum_period.csv", {"L": L, "g": grav, "T": noisy(period, 0.002)})

    # 6. Radioactive decay: N = N0 exp(-lambda t) ------------------------
    t = np.sort(RNG.uniform(0.0, 8.0, N))
    N0 = RNG.uniform(0.8, 1.6, N)
    lam = RNG.uniform(0.15, 0.55, N)
    Nt = N0 * np.exp(-lam * t)
    save("radioactive_decay.csv", {"t": t, "N0": N0, "lambda": lam, "N": noisy(Nt, 0.002)})

    # 7. Spring energy: E = 1/2 k x^2 ------------------------------------
    k = RNG.uniform(0.5, 3.0, N)
    x = RNG.uniform(0.15, 1.1, N)
    E = 0.5 * k * x**2
    save("spring_energy.csv", {"k": k, "x": x, "E": noisy(E, 0.002)})

    # 8. Coulomb force (scaled): F = k q1 q2 / r^2 -----------------------
    q1 = RNG.uniform(0.5, 2.0, N)
    q2 = RNG.uniform(0.5, 2.0, N)
    r = RNG.uniform(1.0, 4.0, N)
    k = 8.99
    F = k * q1 * q2 / r**2
    save("coulomb_force.csv", {"q1": q1, "q2": q2, "r": r, "F": noisy(F, 0.002)})

    # 9. Simple two-feature polynomial -----------------------------------
    x0 = RNG.uniform(-1.5, 1.5, N)
    x1 = RNG.uniform(-1.5, 1.5, N)
    y = 0.8 * x0 * x1 + 1.4 * x0 - 0.7 * x1 + 0.3
    save("two_feature.csv", {"x0": x0, "x1": x1, "y": noisy(y, 0.003)})

    index = {
        "updated": "2026-10-25",
        "note": "Synthetic data with 0.2%-0.4% relative noise, sampled with seed 20261025.",
        "datasets": [
            {
                "file": "linear.csv",
                "name": "线性关系",
                "formula": "y = 2.5x − 1.2",
                "description": "最简单的入门数据，用于确认训练流程和推理功能。",
                "x_columns": ["x"],
                "y_column": "y",
                "tags": ["1 变量", "线性"],
            },
            {
                "file": "damped_oscillator.csv",
                "name": "阻尼振荡",
                "formula": "A(t) = 1.5·e^(−0.45t)·cos(2.1t)",
                "description": "指数衰减包络叠加余弦振荡，需要 exp / cos / mul 等算子。",
                "x_columns": ["t"],
                "y_column": "amplitude",
                "tags": ["1 变量", "指数", "三角"],
            },
            {
                "file": "ideal_gas.csv",
                "name": "理想气体",
                "formula": "P = nRT / V",
                "description": "三个输入变量；R 需要由自由常数 c0 学习得到（≈8.314）。",
                "x_columns": ["V", "T", "n"],
                "y_column": "P",
                "tags": ["3 变量", "物理"],
            },
            {
                "file": "projectile_range.csv",
                "name": "抛体射程",
                "formula": "R = v₀²·sin(2θ) / g",
                "description": "双变量三角关系，适合观察 sin 与平方项的组合。",
                "x_columns": ["v0", "theta"],
                "y_column": "R",
                "tags": ["2 变量", "三角"],
            },
            {
                "file": "pendulum_period.csv",
                "name": "单摆周期",
                "formula": "T = 2π·√(L/g)",
                "description": "平方根关系；训练时建议开启标准化或增大自由常数初始尺度。",
                "x_columns": ["L", "g"],
                "y_column": "T",
                "tags": ["2 变量", "根式"],
            },
            {
                "file": "radioactive_decay.csv",
                "name": "放射性衰变",
                "formula": "N = N₀·e^(−λt)",
                "description": "三变量指数衰减，展示多变量指数规律。",
                "x_columns": ["t", "N0", "lambda"],
                "y_column": "N",
                "tags": ["3 变量", "指数"],
            },
            {
                "file": "spring_energy.csv",
                "name": "弹簧势能",
                "formula": "E = ½·k·x²",
                "description": "含自由常数 0.5 的二次关系，可验证常数优化能力。",
                "x_columns": ["k", "x"],
                "y_column": "E",
                "tags": ["2 变量", "二次"],
            },
            {
                "file": "coulomb_force.csv",
                "name": "库仑力",
                "formula": "F = k·q₁q₂ / r²",
                "description": "三个输入变量，常数 k≈8.99；适合演示自由常数拟合。",
                "x_columns": ["q1", "q2", "r"],
                "y_column": "F",
                "tags": ["3 变量", "物理"],
            },
            {
                "file": "two_feature.csv",
                "name": "双变量多项式",
                "formula": "y = 0.8x₀x₁ + 1.4x₀ − 0.7x₁ + 0.3",
                "description": "多变量线性与交互项的组合，适合测试变量选择。",
                "x_columns": ["x0", "x1"],
                "y_column": "y",
                "tags": ["2 变量", "多项式"],
            },
        ],
    }
    Z = [0.0] * 7
    units = {
        "linear.csv": ([[0.0] * 7], [0.0] * 7),
        "damped_oscillator.csv": ([[0, 0, 1, 0, 0, 0, 0]], [1, 0, 0, 0, 0, 0, 0]),
        "ideal_gas.csv": (
            [[3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]],
            [-1, 1, -2, 0, 0, 0, 0],
        ),
        "projectile_range.csv": (
            [[1, 0, -1, 0, 0, 0, 0], [0.0] * 7],
            [1, 0, 0, 0, 0, 0, 0],
        ),
        "pendulum_period.csv": (
            [[1, 0, 0, 0, 0, 0, 0], [1, 0, -2, 0, 0, 0, 0]],
            [0, 0, 1, 0, 0, 0, 0],
        ),
        "radioactive_decay.csv": (
            [[0, 0, 1, 0, 0, 0, 0], [0.0] * 7, [0, 0, -1, 0, 0, 0, 0]],
            [0.0] * 7,
        ),
        "spring_energy.csv": (
            [[0, 1, -2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
            [2, 1, -2, 0, 0, 0, 0],
        ),
        "coulomb_force.csv": (
            [[0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
            [1, 1, -2, 0, 0, 0, 0],
        ),
        "two_feature.csv": ([[0.0] * 7, [0.0] * 7], [0.0] * 7),
    }
    for item in index["datasets"]:
        x_units, y_units = units.get(item["file"], ([[0.0] * 7] * len(item["x_columns"]), Z))
        item["x_units"] = x_units
        item["y_units"] = y_units

    formulas = {
        "linear.csv": r"y = 2.5x - 1.2",
        "damped_oscillator.csv": r"A(t) = 1.5\,e^{-0.45t}\cos(2.1t)",
        "ideal_gas.csv": r"P = \frac{nRT}{V}",
        "projectile_range.csv": r"R = \frac{v_0^2\sin(2\theta)}{g}",
        "pendulum_period.csv": r"T = 2\pi\sqrt{\frac{L}{g}}",
        "radioactive_decay.csv": r"N = N_0\,e^{-\lambda t}",
        "spring_energy.csv": r"E = \frac{1}{2} k x^2",
        "coulomb_force.csv": r"F = \frac{k q_1 q_2}{r^2}",
        "two_feature.csv": r"y = 0.8x_0x_1 + 1.4x_0 - 0.7x_1 + 0.3",
    }
    for item in index["datasets"]:
        item["formula_latex"] = formulas.get(item["file"], "")

    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'index.json'}")


if __name__ == "__main__":
    main()
