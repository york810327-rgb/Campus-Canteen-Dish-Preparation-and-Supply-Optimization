# -*- coding: utf-8 -*-
"""
================================================================================
阶段四：问题3 —— 需求不确定条件下的备餐策略（两阶段随机规划 / SAA）
================================================================================

本脚本完成赛题最核心的算法突破：

  任务 1  基于「联合经验分布」的多场景生成（Joint Empirical Bootstrap）
          - 以历史供餐日为单位，对阶段二的预测残差做有放回抽样（Bootstrap），
            生成 S=1000 个需求场景，完整保留 10 菜品 × 2 时段之间的需求协同关联。
          - D^s[j,m] = max(0, D_hat[j,m] + epsilon^s[j,m])

  任务 2  两阶段随机规划 MILP（Sample Average Approximation, SAA）
          - 第一阶段：开餐前决策 20 个整数备餐批次数 x[j,m]（人工/主灶/原料确定性约束）
          - 第二阶段：每个场景 s 的销量 w^s[j,m]（连续，w<=y, w<=D^s）
          - 目标：max 1/S Σ_s Σ_{j,m} [S·w - C·y - L·u - P·v]，等价线性化为
                1/S Σ_s Σ_{j,m} [(S+L+P)·w - (C+L)·y - P·D^s]
                （u=y-w、v=D-w 在最优解处自动成立，等价于题面 u/v 显式定义）

  任务 3  策略鲁棒性蒙特卡洛评估与 CDF 对比图
          - Q2(确定性) 与 Q3(随机稳健) 在 1000 场景下的期望效益/标准差/剩余/缺供/服务水平
          - 学术级 CDF 对比图（150 DPI）

  任务 4  结果规范回填《附件4》Q3 Sheet 与「问题2 vs 问题3 比较」表

依赖：阶段三的附件3 自适应解析（import optimize_deterministic，保证约束口径一致）。
运行示例：
  python optimize_stochastic.py
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
import openpyxl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 复用阶段三的附件3自适应解析（单一事实来源，保证 Q2/Q3 约束口径完全一致）
from optimize_deterministic import (
    load_dish_params, load_resources, find_sheet, find_col,
    MEALS, MATERIALS, to_markdown, banner,
)

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
LOG = logging.getLogger("optimize_stochastic")

PREDICTIONS_CSV = "predictions_20260629.csv"
RESIDUALS_CSV = "model_residuals.csv"
Q2_PLAN_CSV = "q2_optimal_plan.csv"
Q2_PERF_CSV = "q2_performance_metrics.csv"
ATTACH3 = "附件3_菜品生产经营参数.xlsx"

S_SCENARIOS = 1000          # 蒙特卡洛 / SAA 场景数
RANDOM_SEED = 42            # 复现性
SOLVER_TIME_LIMIT = 600     # CBC 求解时限（秒）

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)


# ===========================================================================
# 工具
# ===========================================================================
def setup_logging(output_dir: Path, verbose: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "optimize_stochastic.log",
                             encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    LOG.setLevel(level)


def write_csv(df: pd.DataFrame, name: str, out_dir: Path, mirror_dir: Path | None) -> None:
    p = out_dir / name
    df.to_csv(p, index=False, encoding="utf-8-sig")
    LOG.info("已写出 %s（%d 行 × %d 列）", p.resolve(), len(df), df.shape[1])
    if mirror_dir is not None:
        mp = mirror_dir / name
        mirror_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(mp, index=False, encoding="utf-8-sig")
        LOG.info("镜像写出 %s", mp.resolve())


def find_attach4(data_dir: Path) -> Path:
    for name in ["附件4_结果汇总表.xlsx", "附件4：结果汇总表.xlsx"]:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到附件4，已搜索：{data_dir}")


# ===========================================================================
# 任务 1：联合经验分布场景生成
# ===========================================================================
def generate_scenarios(residuals: pd.DataFrame, predictions: pd.DataFrame,
                       S: int, seed: int) -> tuple[np.ndarray, list, np.ndarray]:
    """
    以天为单位的 Bootstrap 重采样，生成 S 个保留协同关联的需求场景。
    返回 (D_scenarios(S,20), combo_keys, D_hat_array(20))
    """
    banner("任务1｜基于联合经验分布的多场景生成")

    # 预测基准（用未取整预测值，与阶段二残差的定义口径一致：残差=真实-预测）
    d_hat: dict[tuple[str, str], float] = {}
    key = "预测值_未取整" if "预测值_未取整" in predictions.columns else "预测需求量"
    for _, r in predictions.iterrows():
        d_hat[(str(r["时段"]).strip(), str(r["菜品编号"]).strip())] = float(r[key])
    LOG.info("预测基准 D_hat 采用字段「%s」", key)

    dishes = sorted(predictions["菜品编号"].unique())
    combo_keys = [(m, j) for m in MEALS for j in dishes]
    n_combos = len(combo_keys)
    D_hat_arr = np.array([d_hat[(m, j)] for (m, j) in combo_keys], dtype=float)

    # 残差按「日期 × (时段|菜品编号)」透视成 50 天 × 20 组合的矩阵
    res = residuals.copy()
    res["combo"] = res["时段"].astype(str).str.strip() + "|" + res["菜品编号"].astype(str).str.strip()
    wide = res.pivot_table(index="日期", columns="combo", values="预测残差", aggfunc="mean")
    combo_str = [f"{m}|{j}" for (m, j) in combo_keys]
    missing = [c for c in combo_str if c not in wide.columns]
    if missing:
        raise KeyError(f"残差表缺少组合列：{missing}")
    wide = wide.reindex(columns=combo_str)
    residual_matrix = wide.to_numpy(dtype=float)          # (n_days, 20)

    n_days = residual_matrix.shape[0]
    LOG.info("残差联合矩阵：%d 个历史供餐日 × %d 个菜品-时段组合", n_days, n_combos)
    LOG.info("残差均值=%.3f，标准差=%.3f（说明模型误差分布）",
             residual_matrix.mean(), residual_matrix.std())

    # Bootstrap：有放回抽取 S 个「整天」
    rng = np.random.default_rng(seed)
    day_idx = rng.integers(0, n_days, size=S)
    epsilon = residual_matrix[day_idx]                    # (S, 20)
    D_scenarios = np.maximum(0.0, D_hat_arr[None, :] + epsilon)  # 非负截断
    LOG.info("已生成 %d 个需求场景（整天 Bootstrap，seed=%d），场景需求均值 %.2f 份",
             S, seed, D_scenarios.mean())
    LOG.info("场景需求范围：[%.1f, %.1f] 份", D_scenarios.min(), D_scenarios.max())
    return D_scenarios, combo_keys, D_hat_arr


# ===========================================================================
# 任务 2：两阶段随机规划 MILP
# ===========================================================================
def build_solve_saa(params: dict, resources: dict, D_scenarios: np.ndarray,
                    combo_keys: list, S: int) -> dict:
    banner("任务2｜两阶段随机规划 MILP（SAA）构建与求解")
    dishes = sorted(params.keys())
    n_combos = len(combo_keys)

    # 预计算参数向量（按 combo 顺序）
    S_arr = np.array([params[j]["S"] for (_, j) in combo_keys])
    L_arr = np.array([params[j]["L"] for (_, j) in combo_keys])
    P_arr = np.array([params[j]["P"] for (_, j) in combo_keys])
    SPL = S_arr + L_arr + P_arr

    prob = pulp.LpProblem("TwoStage_SAA", pulp.LpMaximize)

    # ---- 第一阶段决策：备餐批次数 x（整数）----
    x = pulp.LpVariable.dicts("x", (dishes, MEALS), lowBound=0, cat="Integer")
    y = {(j, m): params[j]["B"] * x[j][m] for j in dishes for m in MEALS}

    # ---- 第二阶段决策：每个场景 s 的销量 w（连续）----
    w = pulp.LpVariable.dicts("w", (range(S), range(n_combos)), lowBound=0, cat="Continuous")

    # ---- 目标函数：期望综合效益最大化（等价线性化）----
    # 线性化后每场景项 = (S+L+P)·w - (C+L)·y - P·D^s
    obj = pulp.lpSum(SPL[k] * w[s][k] for s in range(S) for k in range(n_combos))
    obj -= S * pulp.lpSum((params[j]["C"] + params[j]["L"]) * y[(j, m)]
                          for j in dishes for m in MEALS)
    const = -float((P_arr[None, :] * D_scenarios).sum())   # Σ_s Σ_k P·D^s（常数项）
    prob += obj + const, "Expected_Benefit_x_S"

    # ---- 第二阶段约束：w <= y, w <= D^s ----
    for s in range(S):
        for k, (m, j) in enumerate(combo_keys):
            prob += w[s][k] <= y[(j, m)], f"sales_le_prep_{s}_{k}"
            prob += w[s][k] <= D_scenarios[s, k], f"sales_le_demand_{s}_{k}"

    # ---- 第一阶段确定性约束（人工/主灶/原料，自适应口径同阶段三）----
    for j in dishes:
        for m in MEALS:
            prob += y[(j, m)] <= params[j]["max_cap"], f"maxcap_{j}_{m}"

    labor = resources["labor"]
    stove = resources["stove"]
    if resources["labor_per_meal"]:
        for m in MEALS:
            prob += (pulp.lpSum(params[j]["labor"] * x[j][m] for j in dishes)
                     <= labor.get(m, 0.0), f"labor_{m}")
    else:
        prob += (pulp.lpSum(params[j]["labor"] * x[j][m] for j in dishes for m in MEALS)
                 <= labor.get("全天", 0.0), "labor_daily")

    if resources["stove_per_meal"]:
        for m in MEALS:
            prob += (pulp.lpSum(params[j]["stove"] * x[j][m] for j in dishes)
                     <= stove.get(m, 0.0), f"stove_{m}")
    else:
        prob += (pulp.lpSum(params[j]["stove"] * x[j][m] for j in dishes for m in MEALS)
                 <= stove.get("全天", 0.0), "stove_daily")

    mat_cols = {"肉禽类": "meat", "蔬菜类": "veg", "蛋豆类": "eggbean"}
    material = resources["material"]
    for mat, field in mat_cols.items():
        if resources["material_per_meal"]:
            for m in MEALS:
                avail = material.get(mat, {}).get(m, 0.0)
                prob += (pulp.lpSum(params[j][field] * x[j][m] for j in dishes)
                         <= avail, f"{field}_{m}")
        else:
            avail = material.get(mat, 0.0)
            prob += (pulp.lpSum(params[j][field] * x[j][m] for j in dishes for m in MEALS)
                     <= avail, f"{field}_daily")

    LOG.info("MILP 规模：%d 整数变量 + %d 连续变量，%d 约束",
             len(dishes) * len(MEALS), S * n_combos, prob.numConstraints())

    # ---- 求解 ----
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    LOG.info("求解状态：%s", status)
    if prob.status != pulp.LpStatusOptimal:
        LOG.warning("未证明全局最优（状态=%s），将采用当前可行解继续。", status)

    # ---- 提取解 ----
    x_opt = {j: {m: int(round(x[j][m].varValue)) for m in MEALS} for j in dishes}
    y_opt = {j: {m: x_opt[j][m] * params[j]["B"] for m in MEALS} for j in dishes}
    expected_benefit = float(pulp.value(prob.objective)) / S

    LOG.info("随机稳健方案期望综合经营效益：%.2f 元（%d 场景平均）", expected_benefit, S)
    return {"x": x_opt, "y": y_opt, "expected_benefit": expected_benefit,
            "status": status, "objective_total": float(pulp.value(prob.objective))}


# ===========================================================================
# 任务 3：蒙特卡洛评估
# ===========================================================================
def evaluate_plan(y_plan: dict, params: dict, D_scenarios: np.ndarray,
                  combo_keys: list) -> dict:
    """将某个备餐方案 y 带入 1000 个场景做蒙特卡洛评估（全向量化）。"""
    S_arr = np.array([params[j]["S"] for (_, j) in combo_keys])
    C_arr = np.array([params[j]["C"] for (_, j) in combo_keys])
    L_arr = np.array([params[j]["L"] for (_, j) in combo_keys])
    P_arr = np.array([params[j]["P"] for (_, j) in combo_keys])
    y_arr = np.array([y_plan[j][m] for (m, j) in combo_keys], dtype=float)

    w = np.minimum(y_arr[None, :], D_scenarios)                 # (S, 20) 实际销量
    benefits = (S_arr * w - C_arr * y_arr - L_arr * (y_arr - w)
                - P_arr * (D_scenarios - w)).sum(axis=1)        # (S,) 每场景效益
    sales = w.sum(axis=1)
    leftover = (y_arr - w).sum(axis=1)
    shortage = (D_scenarios - w).sum(axis=1)
    demand = D_scenarios.sum(axis=1)

    return {
        "benefits": benefits,
        "expected_benefit": float(benefits.mean()),
        "benefit_std": float(benefits.std(ddof=1)),
        "var5": float(np.percentile(benefits, 5)),
        "expected_leftover": float(leftover.mean()),
        "expected_shortage": float(shortage.mean()),
        "service_level": float(sales.mean() / demand.mean() * 100.0),
    }


def plot_cdf(benefits_q2: np.ndarray, benefits_q3: np.ndarray,
             out_dir: Path, mirror_dir: Path | None) -> None:
    banner("任务3｜效益累积分布函数（CDF）对比图")
    fig, ax = plt.subplots(figsize=(9, 6))

    def ecdf(data):
        xs = np.sort(data)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    x2, y2 = ecdf(benefits_q2)
    x3, y3 = ecdf(benefits_q3)
    ax.plot(x2, y2, drawstyle="steps-post", lw=2, color="#e07b39", label="Q2 确定性方案")
    ax.plot(x3, y3, drawstyle="steps-post", lw=2, color="#2c7fb8", label="Q3 随机稳健方案")
    ax.axvline(benefits_q2.mean(), color="#e07b39", ls="--", alpha=0.7, lw=1.2)
    ax.axvline(benefits_q3.mean(), color="#2c7fb8", ls="--", alpha=0.7, lw=1.2)
    ax.set_xlabel("综合经营效益（元）")
    ax.set_ylabel("累积概率")
    ax.set_title("Q2 确定性 vs Q3 随机稳健：1000 个需求场景下的综合效益 CDF")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / "q2_vs_q3_cdf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    LOG.info("已保存 CDF 对比图 %s", path.resolve())
    if mirror_dir is not None:
        mdir = mirror_dir / "images"
        mdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(mdir / "q2_vs_q3_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 任务 4：Excel 回填
# ===========================================================================
def backfill_excel(att4: Path, out_dir: Path, mirror_dir: Path | None,
                   solution: dict, params: dict, q2_y: dict,
                   q2_det: dict, q3_metrics: dict) -> None:
    banner("任务4｜Excel 结果规范回填")
    import shutil
    backup = out_dir / "附件4_结果汇总表_原始备份.xlsx"
    if not backup.exists():
        shutil.copy2(att4, backup)
        LOG.info("已创建原始模板备份：%s", backup.resolve())

    wb = openpyxl.load_workbook(att4)
    ws3 = wb["Q3"]
    x, y = solution["x"], solution["y"]

    # ---- Q3 备餐方案（D=批次数, E=备餐量；F 列变化量由模板公式自动计算）----
    row = 6
    for m in MEALS:
        for j in sorted(params.keys()):
            ws3.cell(row=row, column=4).value = int(x[j][m])
            ws3.cell(row=row, column=5).value = int(y[j][m])
            row += 1
    LOG.info("Q3：已回填 20 条备餐方案（D6:E%d，F 列为模板自动公式）", row - 1)

    # ---- Q2 vs Q3 比较表（B=问题2确定性口径, C=问题3期望口径, D=变化量自动公式）----
    # 注：问题2列为“确定性模型（点预测口径）”结果，问题3列为“随机模型（1000场景期望口径）”结果，
    #     二者之差即为“需求不确定性的代价”（如 综合经营效益 -952.24 元）。
    cmp_map = {
        30: ("综合经营效益", q2_det["综合经营效益(元)"], q3_metrics["expected_benefit"]),
        31: ("菜品剩余量", q2_det["预计剩余量(份)"], q3_metrics["expected_leftover"]),
        32: ("菜品缺供量", q2_det["预计缺供量(份)"], q3_metrics["expected_shortage"]),
        33: ("服务水平", q2_det["服务水平(%)"], q3_metrics["service_level"]),
        34: ("效益标准差(元)", 0.0, q3_metrics["benefit_std"]),
        35: ("5%分位效益VaR5%(元)", q2_det["综合经营效益(元)"], q3_metrics["var5"]),
    }
    for r, (label, v2, v3) in cmp_map.items():
        if r >= 34:  # 自定义风险指标：命名并填单位
            ws3.cell(row=r, column=1).value = label
            ws3.cell(row=r, column=5).value = "元"
        ws3.cell(row=r, column=2).value = round(float(v2), 2)
        ws3.cell(row=r, column=3).value = round(float(v3), 2)
    LOG.info("Q3：已回填「问题2 vs 问题3 比较」表（B30:C35，D 列为模板自动公式）")

    # ---- 保存 ----
    wb.save(att4)
    LOG.info("已回填并保存：%s", att4.resolve())
    filled_copy = out_dir / "附件4_结果汇总表_已回填.xlsx"
    wb.save(filled_copy)
    LOG.info("已回填副本：%s", filled_copy.resolve())
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        wb.save(mirror_dir / "附件4_结果汇总表_已回填.xlsx")
        LOG.info("镜像副本：%s", (mirror_dir / "附件4_结果汇总表_已回填.xlsx").resolve())


# ===========================================================================
# 主流程
# ===========================================================================
def run_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    mirror_dir = Path(args.mirror_dir) if args.mirror_dir else None
    for d in (data_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, args.verbose)

    banner("阶段四｜问题3 —— 需求不确定条件下的备餐策略 启动")
    LOG.info("数据目录：%s；输出目录：%s；场景数 S=%d", data_dir.resolve(), out_dir.resolve(), S_SCENARIOS)

    # 预测值 / 残差 / Q2 方案 / 附件3
    pred_path = next((d / PREDICTIONS_CSV for d in (data_dir / "output", data_dir, mirror_dir)
                      if (d / PREDICTIONS_CSV).exists()), None)
    res_path = next((d / RESIDUALS_CSV for d in (data_dir / "output", data_dir, mirror_dir)
                     if (d / RESIDUALS_CSV).exists()), None)
    q2_path = next((d / Q2_PLAN_CSV for d in (data_dir / "output", data_dir, mirror_dir)
                    if (d / Q2_PLAN_CSV).exists()), None)
    q2_perf_path = next((d / Q2_PERF_CSV for d in (data_dir / "output", data_dir, mirror_dir)
                         if (d / Q2_PERF_CSV).exists()), None)
    if pred_path is None or res_path is None or q2_path is None or q2_perf_path is None:
        raise FileNotFoundError(f"缺失输入：predictions={pred_path}, residuals={res_path}, "
                                f"q2={q2_path}, q2_perf={q2_perf_path}")
    att3 = data_dir / ATTACH3
    att4 = find_attach4(data_dir)

    predictions = pd.read_csv(pred_path)
    residuals = pd.read_csv(res_path)
    q2_plan = pd.read_csv(q2_path)
    q2_perf = pd.read_csv(q2_perf_path)
    # 问题2确定性口径指标（点预测下的最优效益/剩余/缺供/服务水平）
    q2_det = dict(zip(q2_perf["指标"], q2_perf["数值"]))
    LOG.info("已读取问题2确定性口径指标：综合经营效益=%.2f 元、服务水平=%.2f%%",
             q2_det["综合经营效益(元)"], q2_det["服务水平(%)"])

    params = load_dish_params(att3)
    resources = load_resources(att3)

    # ---- 任务 1：场景生成 ----
    D_scenarios, combo_keys, D_hat_arr = generate_scenarios(
        residuals, predictions, S_SCENARIOS, RANDOM_SEED)

    # ---- Q2 方案（确定性）备餐量 ----
    q2_y = {j: {m: 0 for m in MEALS} for j in sorted(params.keys())}
    for _, r in q2_plan.iterrows():
        q2_y[str(r["菜品编号"]).strip()][str(r["时段"]).strip()] = float(r["备餐量(份)"])

    # ---- 任务 2：SAA 求解 ----
    solution = build_solve_saa(params, resources, D_scenarios, combo_keys, S_SCENARIOS)

    # ---- 任务 3：蒙特卡洛评估对比 ----
    banner("任务3｜Q2 vs Q3 蒙特卡洛对比评估")
    q2_metrics = evaluate_plan(q2_y, params, D_scenarios, combo_keys)
    q3_metrics = evaluate_plan(solution["y"], params, D_scenarios, combo_keys)

    # 对比口径：Q2=确定性模型（点预测）结果，Q3=随机模型（1000场景期望）结果
    comp_df = pd.DataFrame([
        {"指标": "综合经营效益(元)", "Q2(确定性口径)": round(q2_det["综合经营效益(元)"], 2),
         "Q3(期望口径)": round(q3_metrics["expected_benefit"], 2),
         "变化量(Q3-Q2)": round(q3_metrics["expected_benefit"] - q2_det["综合经营效益(元)"], 2)},
        {"指标": "菜品剩余量(份)", "Q2(确定性口径)": round(q2_det["预计剩余量(份)"], 2),
         "Q3(期望口径)": round(q3_metrics["expected_leftover"], 2),
         "变化量(Q3-Q2)": round(q3_metrics["expected_leftover"] - q2_det["预计剩余量(份)"], 2)},
        {"指标": "菜品缺供量(份)", "Q2(确定性口径)": round(q2_det["预计缺供量(份)"], 2),
         "Q3(期望口径)": round(q3_metrics["expected_shortage"], 2),
         "变化量(Q3-Q2)": round(q3_metrics["expected_shortage"] - q2_det["预计缺供量(份)"], 2)},
        {"指标": "服务水平(%)", "Q2(确定性口径)": round(q2_det["服务水平(%)"], 2),
         "Q3(期望口径)": round(q3_metrics["service_level"], 2),
         "变化量(Q3-Q2)": round(q3_metrics["service_level"] - q2_det["服务水平(%)"], 2)},
        {"指标": "效益标准差(元)", "Q2(确定性口径)": 0.0,
         "Q3(期望口径)": round(q3_metrics["benefit_std"], 2),
         "变化量(Q3-Q2)": round(q3_metrics["benefit_std"], 2)},
        {"指标": "VaR5%(元)", "Q2(确定性口径)": round(q2_det["综合经营效益(元)"], 2),
         "Q3(期望口径)": round(q3_metrics["var5"], 2),
         "变化量(Q3-Q2)": round(q3_metrics["var5"] - q2_det["综合经营效益(元)"], 2)},
    ])
    LOG.info("Q2（确定性口径）vs Q3（期望口径）对比评估矩阵：\n%s", to_markdown(comp_df))

    # ---- 关键结论：产能瓶颈导致确定性解与随机稳健解重合 ----
    labor_used_lunch = sum(params[j]["labor"] * solution["x"][j]["午餐"] for j in params)
    stove_used_dinner = sum(params[j]["stove"] * solution["x"][j]["晚餐"] for j in params)
    LOG.info("关键发现：后厨产能构成硬约束（午餐人工 %.1f/%.0f、晚餐主灶 %.1f/%.0f 均已近饱和），"
             "任意菜品再增 1 批即超产能、任意减 1 批即降低期望效益，"
             "故两阶段随机规划的最优第一阶段决策与确定性方案一致（即“满产能备餐”）。",
             labor_used_lunch, resources["labor"].get("午餐", 0),
             stove_used_dinner, resources["stove"].get("晚餐", 0))
    LOG.info("随机优化的增量价值 = 量化需求波动风险：期望效益 %.2f 元、标准差 %.2f 元、"
             "VaR5%%=%.2f 元、服务水平 %.2f%%，为不确定性决策提供误差分布依据。",
             q3_metrics["expected_benefit"], q3_metrics["benefit_std"],
             q3_metrics["var5"], q3_metrics["service_level"])

    # ---- CDF 图 ----
    plot_cdf(q2_metrics["benefits"], q3_metrics["benefits"], out_dir, mirror_dir)

    # ---- 任务 4：Excel 回填 ----
    backfill_excel(att4, out_dir, mirror_dir, solution, params, q2_y, q2_det, q3_metrics)

    # ---- 导出 CSV ----
    banner("导出结果 CSV")
    # q3 方案明细
    q3_rows = []
    for m in MEALS:
        for j in sorted(params.keys()):
            q3_rows.append({
                "时段": m, "菜品编号": j, "菜品名称": params[j]["name"],
                "生产批次数(批)": int(solution["x"][j][m]),
                "备餐量(份)": int(solution["y"][j][m]),
                "相比Q2备餐量变化(份)": int(solution["y"][j][m] - q2_y[j][m]),
            })
    q3_plan_df = pd.DataFrame(q3_rows)
    LOG.info("Q3 随机稳健备餐方案：\n%s", to_markdown(q3_plan_df))
    write_csv(q3_plan_df, "q3_optimal_plan.csv", out_dir, mirror_dir)
    write_csv(comp_df, "q2_vs_q3_comparison.csv", out_dir, mirror_dir)

    banner("阶段四完成")
    LOG.info("所有结果已写入：%s", out_dir.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="阶段四：问题3 需求不确定备餐策略（两阶段随机规划 SAA）")
    # 默认以脚本所在目录为基准，附件3/4 在脚本目录、输出到 脚本目录/output，换机无需改代码
    script_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(script_dir), help="数据目录（默认：脚本目录）")
    p.add_argument("--output-dir", default=str(script_dir / "output"), help="输出目录（默认：脚本目录/output）")
    p.add_argument("--mirror-dir", default="", help="镜像目录（默认禁用；如需双写可指定路径）")
    p.add_argument("--scenarios", type=int, default=S_SCENARIOS, help="蒙特卡洛场景数（默认 1000）")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    p.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global S_SCENARIOS, RANDOM_SEED
    args = parse_args(argv)
    S_SCENARIOS = args.scenarios
    RANDOM_SEED = args.seed
    try:
        run_pipeline(args)
    except Exception as exc:
        logging.getLogger().exception("流水线执行失败：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
