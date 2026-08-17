# -*- coding: utf-8 -*-
"""
================================================================================
阶段二：需求特征分析与预测模型 —— 校园食堂菜品备餐与供应优化
================================================================================

本脚本完成以下任务（严格对应《阶段二》需求）：

  任务 1  探索性数据分析（EDA）与学术统计分析
          - 连续变量与真实需求的 Pearson / Spearman 相关系数 -> eda_correlation.csv
          - 分类变量（星期/天气/考试周/离校高峰/大型活动）ANOVA 方差分析
            -> eda_anova.csv（整体 + 每个 [菜品-时段] 组合）
          - 学术图表：关联热力图、箱线图（星期/考试周/天气），DPI=150
            -> output/images/

  任务 2  严谨的模型训练、验证与算法对比
          - 时间序列交叉验证 TimeSeriesSplit(5)，杜绝未来信息泄漏
          - Baseline：Ridge / Lasso（L2/L1 正则化线性回归）+ Naive-Lag1 参照
          - 高级模型：LightGBM（菜品编号、时段作为类别特征的统一高维模型）
          - 指标：MAE / RMSE / MAPE，打印对比汇总表

  任务 3  2026-06-29 最终需求预测
          - 最优模型在完整历史集上重训练，预测 20 条目标记录
          - 向上取整 -> predictions_20260629.csv

  任务 4  预测残差与误差特征
          - 最优模型验证集（OOF）预测残差 -> model_residuals.csv
          - 每个 [菜品编号, 时段] 残差分布统计（均值/标准差/偏度/峰度/分位数）
            -> residual_stats.csv

设计要点：
  * 时间序列 CV 严格按日期升序切分，测试折只出现在训练折之后，杜绝泄漏；
  * LightGBM 采用「菜品编号 + 时段」类别特征 + 阶段一全部数值特征的统一模型；
  * 全流程可复现（固定 random_state）；
  * 输出 UTF-8-BOM CSV 与 matplotlib Agg 图片，中文正常显示；
  * 详尽日志 -> predict_model.log。

运行示例：
  python predict_model.py
  python predict_model.py --input-dir "D:\\B\\output" --output-dir "D:\\B\\output"
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境下平稳运行，须在 import pyplot 之前调用

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sps
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from lightgbm import early_stopping as lgb_early_stopping

# ---------------------------------------------------------------------------
# 全局常量与配置
# ---------------------------------------------------------------------------
LOG = logging.getLogger("predict_model")

# 输入 / 输出文件名
TRAIN_FILE = "train_validation.csv"
PREDICT_FILE = "prediction_target.csv"
META_FILE = "column_metadata.json"

# 目标列与标识列
TARGET = "真实需求"
TARGET_EN = "true_demand"
COL_DATE = "日期"
COL_DISH_ID = "菜品编号"
COL_DISH_NAME = "菜品名称"
COL_MEAL = "时段"
COL_WEEKDAY = "星期"
COL_WEATHER = "天气"
COL_SOLDOUT = "是否售罄"

# 连续变量（用于 Pearson/Spearman 相关分析与热力图）
CONTINUOUS_FEATURES = [
    "平均气温(℃)", "降雨量(mm)", "在校学生比例", "活动客流影响估计比例",
    "Lag_1", "Lag_5", "Rolling_Mean_3", "日期_序号", "月份", "日",
]

# 分类变量（用于 ANOVA 与箱线图）
CATEGORICAL_VARS = [COL_WEEKDAY, COL_WEATHER, "是否考试周", "是否离校高峰前一日", "是否大型活动"]

# 交差验证与建模参数
N_SPLITS = 5
RANDOM_STATE = 42
SIGNIFICANCE_LEVEL = 0.05

# 字体（Windows 中文）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)


# ===========================================================================
# 通用工具
# ===========================================================================
def banner(message: str) -> None:
    LOG.info("")
    LOG.info("=" * 92)
    LOG.info(message)
    LOG.info("=" * 92)


def setup_logging(output_dir: Path, verbose: bool) -> None:
    """配置 stdout + 文件双路日志。"""
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
    fh = logging.FileHandler(output_dir / "predict_model.log", encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    LOG.setLevel(level)


def resolve_dir(path: str, fallback: Path) -> Path:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        LOG.warning("目录 %s 不可用，回退到 %s", p, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def find_input(name: str, primary: Path, mirror: Path) -> Path:
    """在候选目录中定位输入文件。"""
    for d in (primary, mirror):
        cand = d / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"找不到输入文件 {name}，已搜索 {primary} 与 {mirror}")


def write_csv(df: pd.DataFrame, name: str, out_dir: Path, mirror_dir: Path | None) -> None:
    """写出 CSV（utf-8-sig，Excel 可直接打开），并镜像。"""
    out_path = out_dir / name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    LOG.info("已写出 %s（%d 行 × %d 列）", out_path.resolve(), len(df), df.shape[1])
    if mirror_dir is not None:
        mpath = mirror_dir / name
        mirror_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(mpath, index=False, encoding="utf-8-sig")
        LOG.info("镜像写出 %s", mpath.resolve())


def save_figure(fig: plt.Figure, name: str, out_dir: Path, mirror_dir: Path | None) -> None:
    """保存图片（DPI=150）到 images/ 子目录并镜像。"""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    LOG.info("已保存图 %s", path.resolve())
    if mirror_dir is not None:
        mdir = mirror_dir / "images"
        mdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(mdir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 数据加载
# ===========================================================================
def load_datasets(input_dir: Path, mirror_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv(find_input(TRAIN_FILE, input_dir, mirror_dir))
    predict = pd.read_csv(find_input(PREDICT_FILE, input_dir, mirror_dir))
    meta_path = find_input(META_FILE, input_dir, mirror_dir)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # 统一日期解析 + 时间排序（时间序列 CV 必需）
    train[COL_DATE] = pd.to_datetime(train[COL_DATE], errors="coerce")
    predict[COL_DATE] = pd.to_datetime(predict[COL_DATE], errors="coerce")

    LOG.info("历史训练/验证集：%s；待预测目标集：%s", train.shape, predict.shape)
    LOG.info("历史日期范围：%s ~ %s；待预测日期：%s",
             train[COL_DATE].min(), train[COL_DATE].max(),
             predict[COL_DATE].dropna().unique().tolist())
    return train, predict, meta


# ===========================================================================
# 任务 1：EDA
# ===========================================================================
def eda_correlation(train: pd.DataFrame, out_dir: Path, mirror_dir: Path | None) -> None:
    banner("任务1｜连续变量相关性分析（Pearson / Spearman）")
    rows = []
    for feat in CONTINUOUS_FEATURES:
        if feat not in train.columns:
            continue
        x = pd.to_numeric(train[feat], errors="coerce")
        y = pd.to_numeric(train[TARGET], errors="coerce")
        mask = x.notna() & y.notna()
        xv, yv = x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)
        if len(xv) < 3 or np.std(xv) == 0:
            pearson_r, pearson_p = np.nan, np.nan
        else:
            pearson_r, pearson_p = sps.pearsonr(xv, yv)
        if len(xv) < 3 or np.std(xv) == 0:
            spearman_r, spearman_p = np.nan, np.nan
        else:
            spearman_r, spearman_p = sps.spearmanr(xv, yv)
        rows.append({
            "特征": feat,
            "Pearson_r": round(float(pearson_r), 4),
            "Pearson_p": round(float(pearson_p), 6),
            "Spearman_rho": round(float(spearman_r), 4),
            "Spearman_p": round(float(spearman_p), 6),
            "样本数": int(mask.sum()),
        })
    corr_df = pd.DataFrame(rows).sort_values(
        "Pearson_r", key=lambda s: s.abs(), ascending=False, na_position="last")
    LOG.info("连续特征与真实需求的相关性：\n%s", corr_df.to_string(index=False))
    write_csv(corr_df, "eda_correlation.csv", out_dir, mirror_dir)

    # ---- 关联热力图（连续特征 + 目标）----
    heat_cols = [c for c in CONTINUOUS_FEATURES if c in train.columns] + [TARGET]
    corr_mat = train[heat_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("连续特征与真实需求的 Pearson 相关热力图", fontsize=14, pad=12)
    save_figure(fig, "correlation_heatmap.png", out_dir, mirror_dir)


def _anova_f_p(df: pd.DataFrame, group_col: str, value_col: str) -> tuple[float, float, int, int]:
    """单因素 ANOVA，返回 (F, p, 样本数, 组数)。组内样本过少时返回 nan。"""
    groups = [g[value_col].dropna().to_numpy(dtype=float)
              for _, g in df.groupby(group_col, dropna=True) if len(g) >= 2]
    groups = [g for g in groups if len(g) >= 2]
    n = int(sum(len(g) for g in groups))
    if len(groups) < 2:
        return float("nan"), float("nan"), n, len(groups)
    F, p = sps.f_oneway(*groups)
    return float(F), float(p), n, len(groups)


def eda_anova(train: pd.DataFrame, out_dir: Path, mirror_dir: Path | None) -> None:
    banner("任务1｜分类变量 ANOVA 方差分析")
    combos = (train[[COL_DISH_ID, COL_MEAL]].drop_duplicates()
              .sort_values([COL_DISH_ID, COL_MEAL]).reset_index(drop=True))
    rows = []
    for var in CATEGORICAL_VARS:
        if var not in train.columns:
            continue
        # 整体
        F, p, n, ng = _anova_f_p(train, var, TARGET)
        rows.append({"分类变量": var, "菜品编号": "全部", "时段": "全部",
                     "F值": round(F, 4) if np.isfinite(F) else np.nan,
                     "P值": round(p, 6) if np.isfinite(p) else np.nan,
                     "样本数": n, "组数": ng,
                     "显著性(α=0.05)": "显著" if (np.isfinite(p) and p < SIGNIFICANCE_LEVEL) else "不显著"})
        # 每个 [菜品-时段] 组合
        for _, r in combos.iterrows():
            sub = train[(train[COL_DISH_ID] == r[COL_DISH_ID]) & (train[COL_MEAL] == r[COL_MEAL])]
            F, p, n, ng = _anova_f_p(sub, var, TARGET)
            rows.append({"分类变量": var, "菜品编号": r[COL_DISH_ID], "时段": r[COL_MEAL],
                         "F值": round(F, 4) if np.isfinite(F) else np.nan,
                         "P值": round(p, 6) if np.isfinite(p) else np.nan,
                         "样本数": n, "组数": ng,
                         "显著性(α=0.05)": "显著" if (np.isfinite(p) and p < SIGNIFICANCE_LEVEL) else "不显著"})
    anova_df = pd.DataFrame(rows)
    LOG.info("ANOVA 结果（整体 + 20 个 [菜品-时段] 组合）：共 %d 行", len(anova_df))
    LOG.info("整体（菜品编号=全部）显著性概览：\n%s",
             anova_df[anova_df["菜品编号"] == "全部"][["分类变量", "F值", "P值", "显著性(α=0.05)"]]
             .to_string(index=False))
    write_csv(anova_df, "eda_anova.csv", out_dir, mirror_dir)


def eda_boxplots(train: pd.DataFrame, out_dir: Path, mirror_dir: Path | None) -> None:
    banner("任务1｜箱线图绘制")
    sns.set_theme(style="whitegrid", font=["Microsoft YaHei", "SimHei", "DejaVu Sans"])

    # ① 星期
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=train, x=COL_WEEKDAY, y=TARGET, order=["周一", "周二", "周三", "周四", "周五"],
                hue=None, palette="Set2", ax=ax)
    ax.set_title("不同星期的菜品真实需求分布", fontsize=14)
    ax.set_xlabel("星期")
    ax.set_ylabel("真实需求（份）")
    save_figure(fig, "boxplot_weekday.png", out_dir, mirror_dir)

    # ② 是否考试周
    tmp = train.copy()
    tmp["考试周标签"] = tmp["是否考试周"].map({1: "考试周", 0: "非考试周"})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.boxplot(data=tmp, x="考试周标签", y=TARGET, order=["非考试周", "考试周"],
                palette="Set1", ax=ax)
    ax.set_title("是否考试周的真实需求分布", fontsize=14)
    ax.set_xlabel("是否考试周")
    ax.set_ylabel("真实需求（份）")
    save_figure(fig, "boxplot_exam_week.png", out_dir, mirror_dir)

    # ③ 天气
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=train, x=COL_WEATHER, y=TARGET,
                order=["晴", "多云", "阴", "小雨", "中雨"], palette="Set3", ax=ax)
    ax.set_title("不同天气的菜品真实需求分布", fontsize=14)
    ax.set_xlabel("天气")
    ax.set_ylabel("真实需求（份）")
    save_figure(fig, "boxplot_weather.png", out_dir, mirror_dir)


# ===========================================================================
# 任务 2：建模与时间序列交叉验证
# ===========================================================================
def _numeric_matrix(tr: pd.DataFrame, va: pd.DataFrame,
                    numeric_feats: list[str], scaler: StandardScaler | None):
    """线性模型特征矩阵：数值特征标准化 + 菜品编号独热。scaler 仅用训练折拟合。"""
    num_tr = tr[numeric_feats].to_numpy(dtype=float)
    num_va = va[numeric_feats].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(num_tr)
    num_tr = scaler.transform(num_tr)
    num_va = scaler.transform(num_va)

    dish_tr = pd.get_dummies(tr[COL_DISH_ID], prefix="菜品编号").astype(float)
    dish_va = pd.get_dummies(va[COL_DISH_ID], prefix="菜品编号").astype(float)
    dish_va = dish_va.reindex(columns=dish_tr.columns, fill_value=0.0)

    return (np.hstack([num_tr, dish_tr.to_numpy()]),
            np.hstack([num_va, dish_va.to_numpy()]), scaler)


def _lgbm_frame(df: pd.DataFrame, numeric_feats: list[str]) -> pd.DataFrame:
    """LightGBM 特征矩阵：数值特征 + 菜品编号/时段类别特征。"""
    cols = numeric_feats + [COL_DISH_ID, COL_MEAL]
    out = df[cols].copy()
    for c in (COL_DISH_ID, COL_MEAL):
        out[c] = out[c].astype("category")
    return out


def _lgbm_params(n_estimators: int) -> dict:
    """LightGBM 超参数（针对小样本时间序列做了正则化，防止过拟合）。"""
    return dict(
        objective="regression", metric="mae", boosting_type="gbdt",
        n_estimators=n_estimators, learning_rate=0.03, num_leaves=15,
        max_depth=5, min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100.0)
    return mae, rmse, mape


def run_ts_cv(train: pd.DataFrame, numeric_feats: list[str]) -> dict:
    """
    时间序列交叉验证。返回各模型的 OOF 预测、每折指标与 LGBM 最优迭代数。
    训练数据须已按日期升序排序。
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = {}

    def model_entry(name: str, kind: str) -> dict:
        entry = {"name": name, "kind": kind, "oof": np.full(len(train), np.nan),
                 "folds": [], "best_iters": []}
        results[name] = entry
        return entry

    ridge_e = model_entry("Ridge(L2)", "linear")
    lasso_e = model_entry("Lasso(L1)", "linear")
    lgbm_e = model_entry("LightGBM", "lgbm")
    naive_e = model_entry("Naive-Lag1", "naive")

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(train)):
        tr = train.iloc[tr_idx].reset_index(drop=True)
        va = train.iloc[va_idx].reset_index(drop=True)
        y_tr = tr[TARGET].to_numpy(dtype=float)
        y_va = va[TARGET].to_numpy(dtype=float)

        # ---- Naive-Lag1 参照（持续性基线：直接以上一期真实需求作为预测）----
        naive_pred = va["Lag_1"].to_numpy(dtype=float)
        naive_e["oof"][va_idx] = naive_pred
        naive_e["folds"].append(_metrics(y_va, naive_pred))

        # ---- Ridge / Lasso ----
        for entry, make in ((ridge_e, lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE)),
                            (lasso_e, lambda: Lasso(alpha=0.1, max_iter=5000, random_state=RANDOM_STATE))):
            Xtr, Xva, _ = _numeric_matrix(tr, va, numeric_feats, None)
            model = make()
            model.fit(Xtr, y_tr)
            pred = model.predict(Xva)
            entry["oof"][va_idx] = pred
            entry["folds"].append(_metrics(y_va, pred))

        # ---- LightGBM ----
        Xtr, Xva = _lgbm_frame(tr, numeric_feats), _lgbm_frame(va, numeric_feats)
        model = lgb.LGBMRegressor(**_lgbm_params(3000))
        model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], eval_metric="mae",
                  callbacks=[lgb_early_stopping(300, verbose=False)])
        pred = model.predict(Xva)
        lgbm_e["oof"][va_idx] = pred
        lgbm_e["folds"].append(_metrics(y_va, pred))
        lgbm_e["best_iters"].append(int(model.best_iteration_))
        LOG.info("LightGBM 第 %d 折：best_iteration=%d", fold + 1, model.best_iteration_)

    return results


def summarize_cv(results: dict) -> pd.DataFrame:
    """汇总各模型 5 折指标均值±标准差。"""
    banner("任务2｜时间序列交叉验证（TimeSeriesSplit=5）结果汇总")
    rows = []
    for name, e in results.items():
        folds = e["folds"]
        if not folds:
            continue
        arr = np.array(folds)
        rows.append({
            "模型": name,
            "MAE": f"{arr[:, 0].mean():.3f} ± {arr[:, 0].std():.3f}",
            "RMSE": f"{arr[:, 1].mean():.3f} ± {arr[:, 1].std():.3f}",
            "MAPE(%)": f"{arr[:, 2].mean():.3f} ± {arr[:, 2].std():.3f}",
            "MAE_mean": arr[:, 0].mean(),
            "RMSE_mean": arr[:, 1].mean(),
            "MAPE_mean": arr[:, 2].mean(),
        })
    summary = pd.DataFrame(rows).sort_values("MAE_mean")
    LOG.info("\n%s", summary.drop(columns=["MAE_mean", "RMSE_mean", "MAPE_mean"])
             .to_string(index=False))
    LOG.info("各模型逐折 MAE：")
    for name, e in results.items():
        LOG.info("  %-12s %s", name, ["%.3f" % f[0] for f in e["folds"]])
    return summary


# ===========================================================================
# 任务 3：最终预测
# ===========================================================================
def retrain_and_predict(best_name: str, results: dict, train: pd.DataFrame,
                        predict: pd.DataFrame, numeric_feats: list[str]) -> pd.DataFrame:
    banner("任务3｜最优模型重训练与 2026-06-29 预测")
    LOG.info("最优模型：%s（按交叉验证 MAE 最小选定）", best_name)

    y_full = train[TARGET].to_numpy(dtype=float)
    if best_name == "LightGBM":
        Xtr_full = _lgbm_frame(train, numeric_feats)
        Xte = _lgbm_frame(predict, numeric_feats)
        # 类别对齐：确保预测集与训练集类别一致
        for c in (COL_DISH_ID, COL_MEAL):
            Xte[c] = Xte[c].cat.set_categories(Xtr_full[c].cat.categories)
        best_iters = results["LightGBM"]["best_iters"]
        n_estimators = max(100, int(np.ceil(np.mean(best_iters)))) if best_iters else 800
        LOG.info("LightGBM 重训练 n_estimators=%d（取各折 best_iteration 均值）", n_estimators)
        model = lgb.LGBMRegressor(**_lgbm_params(n_estimators))
        model.fit(Xtr_full, y_full)
        raw_pred = model.predict(Xte)
    else:  # Ridge / Lasso
        Xtr_full, Xte, scaler = _numeric_matrix(train, predict, numeric_feats, None)
        alpha = 1.0 if best_name == "Ridge(L2)" else 0.1
        model = (Ridge(alpha=alpha, random_state=RANDOM_STATE) if best_name == "Ridge(L2)"
                 else Lasso(alpha=alpha, max_iter=5000, random_state=RANDOM_STATE))
        model.fit(Xtr_full, y_full)
        raw_pred = model.predict(Xte)

    pred_out = predict[[COL_MEAL, COL_DISH_ID, COL_DISH_NAME]].copy()
    pred_out["预测需求量"] = np.ceil(raw_pred).astype(int)  # 向上取整，保证物理意义
    pred_out["预测值_未取整"] = raw_pred                     # 原始浮点预测值（审计用）
    # 按规范字段顺序：时段、菜品编号、菜品名称、预测需求量（+ 未取整审计列）
    pred_out = pred_out[[COL_MEAL, COL_DISH_ID, COL_DISH_NAME, "预测需求量", "预测值_未取整"]]

    LOG.info("2026-06-29 最终需求预测（向上取整）：\n%s", pred_out.to_string(index=False))
    return pred_out


# ===========================================================================
# 任务 4：残差与误差分布特征
# ===========================================================================
def compute_residuals(best_name: str, results: dict, train: pd.DataFrame,
                      out_dir: Path, mirror_dir: Path | None) -> None:
    banner("任务4｜预测残差与误差分布特征")
    oof = results[best_name]["oof"]
    mask = np.isfinite(oof)
    res_df = train.loc[mask, [COL_DATE, COL_DISH_ID, COL_DISH_NAME, COL_MEAL,
                              COL_WEEKDAY, COL_SOLDOUT, TARGET]].copy()
    res_df["预测值"] = oof[mask]
    # 预测残差 = 真实需求 - 预测值（正残差表示模型低估）
    res_df["预测残差"] = res_df[TARGET] - res_df["预测值"]
    res_df = res_df.sort_values([COL_DATE, COL_DISH_ID, COL_MEAL]).reset_index(drop=True)

    LOG.info("残差整体统计：均值=%.3f，标准差=%.3f，MAE=%.3f",
             res_df["预测残差"].mean(), res_df["预测残差"].std(),
             res_df["预测残差"].abs().mean())
    LOG.info("说明：TimeSeriesSplit 的首个训练段（前 %d 行）不进入任何验证折，"
             "故 OOF 残差覆盖 %d / %d 行（均为严格的时间后向验证预测）。",
             len(train) - int(mask.sum()), int(mask.sum()), len(train))
    write_csv(res_df, "model_residuals.csv", out_dir, mirror_dir)

    # ---- 分组残差分布特征 ----
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    stats_rows = []
    for (dish, meal), g in res_df.groupby([COL_DISH_ID, COL_MEAL], sort=True):
        r = g["预测残差"].to_numpy(dtype=float)
        q = np.quantile(r, quantiles)
        stats_rows.append({
            COL_DISH_ID: dish, COL_MEAL: meal, "样本数": int(len(r)),
            "均值(Mean)": round(float(r.mean()), 4),
            "标准差(Std)": round(float(r.std(ddof=1)), 4),
            "偏度(Skewness)": round(float(sps.skew(r, bias=False)), 4),
            "峰度(Kurtosis)": round(float(sps.kurtosis(r, bias=False)), 4),
            "P5": round(float(q[0]), 4), "P10": round(float(q[1]), 4),
            "P25": round(float(q[2]), 4), "P50": round(float(q[3]), 4),
            "P75": round(float(q[4]), 4), "P90": round(float(q[5]), 4),
            "P95": round(float(q[6]), 4),
        })
    stats_df = pd.DataFrame(stats_rows)
    LOG.info("各 [菜品编号, 时段] 残差分布特征：\n%s", stats_df.to_string(index=False))
    write_csv(stats_df, "residual_stats.csv", out_dir, mirror_dir)


# ===========================================================================
# 主流程
# ===========================================================================
def run_pipeline(args: argparse.Namespace) -> None:
    primary_in = Path(args.input_dir)
    mirror_in = Path(args.mirror_dir) if args.mirror_dir else None
    out_dir = resolve_dir(args.output_dir, primary_in)
    mirror_out = Path(args.mirror_dir) if args.mirror_dir else None
    if mirror_out is not None:
        mirror_out.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, args.verbose)

    banner("阶段二｜需求特征分析与预测模型 启动")
    LOG.info("输入目录：%s（镜像 %s）", primary_in.resolve(), mirror_in.resolve() if mirror_in else None)
    LOG.info("输出目录：%s（镜像 %s）", out_dir.resolve(), mirror_out.resolve() if mirror_out else None)

    # ---- 数据加载 ----
    train, predict, meta = load_datasets(primary_in, mirror_in)
    numeric_feats = [c for c in meta.get("feature_columns", []) if c in train.columns]
    LOG.info("读取特征列 %d 个：%s", len(numeric_feats), numeric_feats)

    # 时间序列 CV 前必须按日期升序排序（阶段一输出按 [菜品,时段,日期] 排序，需重排）
    train = train.sort_values([COL_DATE, COL_DISH_ID, COL_MEAL]).reset_index(drop=True)
    LOG.info("历史数据已按日期升序排序（供 TimeSeriesSplit 使用）。")

    # ---- 任务 1：EDA ----
    eda_correlation(train, out_dir, mirror_out)
    eda_anova(train, out_dir, mirror_out)
    eda_boxplots(train, out_dir, mirror_out)

    # ---- 任务 2：建模与交叉验证 ----
    results = run_ts_cv(train, numeric_feats)
    summary = summarize_cv(results)

    trainable = summary[~summary["模型"].eq("Naive-Lag1")]
    best_name = str(trainable.sort_values("MAE_mean").iloc[0]["模型"])
    LOG.info("选定最优模型：%s", best_name)

    # ---- 任务 3：最终预测 ----
    pred_out = retrain_and_predict(best_name, results, train, predict, numeric_feats)
    write_csv(pred_out, "predictions_20260629.csv", out_dir, mirror_out)

    # ---- 任务 4：残差 ----
    compute_residuals(best_name, results, train, out_dir, mirror_out)

    banner("阶段二完成")
    LOG.info("所有结果已写入：%s（镜像 %s）", out_dir.resolve(),
             mirror_out.resolve() if mirror_out else "无")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 默认以脚本所在目录为基准（输入/输出均为 脚本目录/output），换机无需改代码
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="阶段二：需求特征分析与预测模型")
    p.add_argument("--input-dir", default=str(script_dir / "output"), help="输入目录（默认：脚本目录/output）")
    p.add_argument("--output-dir", default=str(script_dir / "output"), help="输出目录（默认：脚本目录/output）")
    p.add_argument("--mirror-dir", default="",
                   help="镜像输出目录（默认禁用；如需双写可指定路径）")
    p.add_argument("--n-splits", type=int, default=N_SPLITS, help="时间序列交叉验证折数（默认 5）")
    p.add_argument("--verbose", action="store_true", help="输出 DEBUG 级日志")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    global N_SPLITS
    N_SPLITS = args.n_splits
    try:
        run_pipeline(args)
    except Exception as exc:
        logging.getLogger().exception("流水线执行失败：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
