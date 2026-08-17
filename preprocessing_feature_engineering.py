# -*- coding: utf-8 -*-
"""
================================================================================
阶段一：数据预处理与特征工程 —— 校园食堂菜品备餐与供应优化
================================================================================

本脚本完成以下任务（严格对应《阶段一》需求）：

  任务 1  多文件读取与对齐（Left Join）
          - 读取附件1（历史销售与需求数据）、附件2（校园环境与运行信息）
          - 通过「日期」左连接，将环境特征广播到每一条菜品记录（菜品 × 时段）

  任务 2  缺失值、异常值与「待核验」处理
          - 检查需求估计量 / 实际销量 / 剩余量等核心字段
          - 对「待核验 / 缺失」记录按售罄状态分情形处理

  任务 3  售罄截断真实需求重构（核心亮点算法）
          - 新建字段「真实需求」(true_demand)
          - 未售罄：真实需求 = 实际销量
          - 售罄且有有效需求估计量：真实需求 = 需求估计量
          - 售罄但估计量缺失/异常/明显小于销量：用「菜品-时段」级平均溢出系数
            θ̄_{j,meal} 重构；样本不足时回退全局保守系数 1.25

  任务 4  高阶特征工程
          - 时间特征：月份 / 日 / 星期(编码+独热) / 时段编码
          - 环境与校历特征：天气独热 / 气温 / 降雨 / 在校比例 / 考试周 / 离校高峰
            / 大型活动 / 活动客流影响比例（百分比字符串 -> 浮点）
          - 历史滞后特征：Lag_1 / Lag_5 / Rolling_Mean_3（按 [菜品编号, 时段] 分组
            并按日期排序，严格防数据泄漏）

  任务 5  划分数据集并保存
          - 历史 -> 训练/验证集；待预测决策日(2026-06-29) -> 目标集
          - 目标集仅含环境特征，其 Lag 变量由历史末段真实数据正确回填
          - 输出三个 CSV：cleaned_data.csv（完整干净特征集）、
            train_validation.csv（历史训练/验证集）、prediction_target.csv（目标集）

设计要点：
  * 全程 pandas 向量化操作，无逐行 for 循环；
  * 详细日志（形状 / 缺失统计 / 重构前后均值对比 / 溢出系数表）；
  * 输出 UTF-8-BOM 编码 CSV，Excel 可直接打开中文不乱码；
  * 附列元数据 JSON，明确 target / features / aux 划分，便于直接喂给 LightGBM。

运行示例：
  python preprocessing_feature_engineering.py
  python preprocessing_feature_engineering.py --data-dir "D:\\B" --output-dir "D:\\B\\output"
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
LOG = logging.getLogger("preprocess")

# 附件1 / 附件2 的默认文件名与 sheet 名（sheet 缺失时自动回退到第一个 sheet）
ATTACHMENT1_NAME = "附件1_历史销售与需求数据.xlsx"
ATTACHMENT2_NAME = "附件2_校园环境与运行信息.xlsx"
SHEET1_HINT = "历史经营数据"
SHEET2_HINT = "校园环境与运行信息"

# 核心字段（中文列名，保持与原始数据一致）
COL_DATE = "日期"
COL_WEEKDAY = "星期"
COL_MEAL = "时段"
COL_DISH_ID = "菜品编号"
COL_DISH_NAME = "菜品名称"
COL_BATCH = "备餐批次(批)"
COL_PREPARED = "备餐量(份)"
COL_EST = "需求估计量(份)"
COL_SALES = "实际销量(份)"
COL_LEFTOVER = "剩余量(份)"
COL_SOLDOUT = "是否售罄"
COL_SOLDOUT_TIME = "售罄时间"
COL_REMARK = "备注"
COL_USAGE = "数据用途"

COL_WEATHER = "天气"
COL_TEMP = "平均气温(℃)"
COL_RAIN = "降雨量(mm)"
COL_STUDENT_RATIO = "在校学生比例"
COL_EXAM = "是否考试周"
COL_LEAVE_EVE = "是否离校高峰前一日"
COL_EVENT = "是否大型活动"
COL_EVENT_RATIO = "活动客流影响估计比例"

# 重构算法超参数
GLOBAL_FALLBACK_OVERFLOW = 1.25   # 全局缺供保守溢出系数（真实需求 = 实际销量 * 1.25）
MIN_THETA_SAMPLES = 3             # 计算菜品-时段溢出系数所需的最少历史售罄样本数
RATIO_MAX = 3.0                   # 需求估计量/实际销量 超过该值视为「异常」估计，触发重构

# 特征工程常量
MEAL_MAP = {"午餐": 0, "晚餐": 1}
WEEKDAY_CATS = ["周一", "周二", "周三", "周四", "周五"]
WEEKDAY_MAP = {name: i for i, name in enumerate(WEEKDAY_CATS)}
WEATHER_CATS = ["晴", "多云", "阴", "小雨", "中雨"]

# 预测决策日标记
PREDICTION_MARKER = "待预测决策日"
HISTORY_MARKER = "历史"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


# ===========================================================================
# 工具函数
# ===========================================================================
def banner(message: str) -> None:
    """打印醒目的分节横幅，方便人类监督。"""
    LOG.info("")
    LOG.info("=" * 90)
    LOG.info(message)
    LOG.info("=" * 90)


def setup_logging(verbose: bool) -> None:
    """配置日志：默认 INFO，--verbose 时 DEBUG。"""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(level)
    # 避免重复添加 handler（例如被外部重复调用时）
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    LOG.setLevel(level)


def add_file_logger(output_dir: Path) -> None:
    """把日志同时写入输出目录，便于归档审计。"""
    try:
        fh = logging.FileHandler(output_dir / "preprocessing.log",
                                 encoding="utf-8", mode="w")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s"))
        logging.getLogger().addHandler(fh)
    except OSError as exc:  # 日志文件写不进去不致命，仅提示
        LOG.warning("无法创建日志文件：%s", exc)


def resolve_output_dir(primary: str, script_dir: Path) -> Path:
    """解析输出目录；主目录不可写时回退到脚本同级 output/ 目录。"""
    p = Path(primary)
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / "._write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return p
    except OSError as exc:
        LOG.warning("主输出目录 %s 不可写（%s），回退到脚本同级 output/ 目录", p, exc)
        fallback = script_dir / "output"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def load_sheet(path: Path, sheet_hint: str) -> pd.DataFrame:
    """读取 xlsx：优先使用指定 sheet，缺失时回退到第一个 sheet。"""
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    xls = pd.ExcelFile(path)
    sheet = sheet_hint if sheet_hint in xls.sheet_names else xls.sheet_names[0]
    if sheet != sheet_hint:
        LOG.warning("文件 %s 中未找到 sheet「%s」，回退到「%s」", path.name, sheet_hint, sheet)
    df = pd.read_excel(path, sheet_name=sheet)
    LOG.info("已读取 %s / sheet「%s」，形状 %s", path.name, sheet, df.shape)
    return df


def strip_strings(df: pd.DataFrame, cols: list[str]) -> None:
    """对文本列统一去首尾空白、规范为 StringDtype，避免脏值影响匹配。"""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()


def parse_ratio_series(s: pd.Series) -> pd.Series:
    """
    将「活动客流影响估计比例」统一转为浮点比例。

    兼容两种来源：
      1) 字符串 "8.0%" / "-10.0%" -> 0.08 / -0.10；
      2) 已是数值的比例（如 0.08）-> 原样保留。
    安全兜底：若数值的绝对值大于 1（明显是「百分点」而非「比例」），除以 100。
    """
    s = s.copy()
    if s.dtype == object or str(s.dtype) in ("string", "str"):
        s_str = s.fillna("").astype(str).str.strip()
        has_pct = s_str.str.contains("%", regex=False)
        cleaned = s_str.str.replace("%", "", regex=False)
        cleaned = cleaned.replace("", np.nan)
        out = pd.to_numeric(cleaned, errors="coerce").astype(float).to_numpy()
        out = np.where(has_pct.to_numpy(), out / 100.0, out)
    else:
        out = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()

    # 兜底：比例本应落在 [-1, 1]，超出者按「百分点」处理
    out = np.where(np.abs(out) > 1.0, out / 100.0, out)
    result = pd.Series(out, index=s.index, dtype=float)
    return result.fillna(0.0)


def to_binary(s: pd.Series) -> pd.Series:
    """「是/否」等二值字段 -> 0/1（int8），兼容多种写法。"""
    s = s.fillna("").astype(str).str.strip()
    mapping = {"是": 1, "否": 0, "1": 1, "0": 0,
               "TRUE": 1, "FALSE": 0, "True": 1, "False": 0,
               "true": 1, "false": 0}
    return s.map(mapping).fillna(0).astype("int8")


def mean_of(series: pd.Series) -> float:
    """返回可打印的均值，空序列返回 nan。"""
    return float(series.mean()) if len(series) else float("nan")


# ===========================================================================
# 任务 1：读取与对齐
# ===========================================================================
def load_and_align(data_dir: Path, attachment1: str | None,
                   attachment2: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取附件1/2并左连接。返回 (对齐后的历史 DataFrame, 附件2 原始 DataFrame)。"""
    f1 = data_dir / (attachment1 or ATTACHMENT1_NAME)
    f2 = data_dir / (attachment2 or ATTACHMENT2_NAME)

    df1 = load_sheet(f1, SHEET1_HINT)
    df2 = load_sheet(f2, SHEET2_HINT)

    # 统一日期为 datetime（容错：非标准日期会被置为 NaT 并统计）
    df1[COL_DATE] = pd.to_datetime(df1[COL_DATE], errors="coerce")
    df2[COL_DATE] = pd.to_datetime(df2[COL_DATE], errors="coerce")
    LOG.info("附件1 日期范围：%s ~ %s，共 %d 个观测日",
             df1[COL_DATE].min(), df1[COL_DATE].max(), df1[COL_DATE].nunique())
    LOG.info("附件2 日期范围：%s ~ %s，共 %d 个观测日",
             df2[COL_DATE].min(), df2[COL_DATE].max(), df2[COL_DATE].nunique())

    # 文本列清洗
    strip_strings(df1, [COL_WEEKDAY, COL_MEAL, COL_DISH_ID, COL_DISH_NAME,
                        COL_SOLDOUT, COL_SOLDOUT_TIME, COL_REMARK])
    strip_strings(df2, [COL_WEEKDAY, COL_WEATHER, COL_EXAM, COL_LEAVE_EVE,
                        COL_EVENT, COL_USAGE])

    # 核心数值字段强制数值化
    for c in [COL_BATCH, COL_PREPARED, COL_EST, COL_SALES, COL_LEFTOVER]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce")

    # 左连接：保留附件1全部记录，广播附件2环境字段。
    # 两文件都含「星期」，为避免 merge 产生 _x/_y 后缀，先丢弃附件2中的「星期」。
    df2_for_merge = df2.drop(columns=[COL_WEEKDAY], errors="ignore")
    merged = df1.merge(df2_for_merge, on=COL_DATE, how="left", validate="many_to_one")

    banner("任务1｜左连接结果")
    LOG.info("附件1 记录数：%d；对齐后主表记录数：%d", len(df1), len(merged))
    LOG.info("对齐后列数：%d", merged.shape[1])
    LOG.info("对齐后列名：%s", list(merged.columns))

    # 左连接完整性检查：环境字段不应出现大面积缺失
    env_keys = [COL_WEATHER, COL_TEMP, COL_RAIN, COL_STUDENT_RATIO,
                COL_EXAM, COL_LEAVE_EVE, COL_EVENT, COL_EVENT_RATIO, COL_USAGE]
    n_env_missing = merged[env_keys].isna().any(axis=1).sum()
    if n_env_missing:
        LOG.warning("有 %d 条记录未能匹配到环境信息（日期缺失于附件2），请核查！", n_env_missing)
    else:
        LOG.info("环境信息已完整广播到每一条菜品记录（无未匹配行）。")

    # 核对菜品 × 时段覆盖：应为 10 菜品 × 2 时段 × 60 日 = 1200
    LOG.info("唯一菜品数：%d；时段取值：%s",
             merged[COL_DISH_ID].nunique(), sorted(merged[COL_MEAL].dropna().unique()))
    LOG.info("每日记录数检查（应稳定为 20 = 10菜品×2时段）：%s",
             merged.groupby(COL_DATE).size().value_counts().to_dict())

    return merged, df2


# ===========================================================================
# 任务 2 + 任务 3：清洗与售罄截断真实需求重构
# ===========================================================================
def reconstruct_true_demand(df: pd.DataFrame,
                            min_samples: int = MIN_THETA_SAMPLES,
                            global_fallback: float = GLOBAL_FALLBACK_OVERFLOW,
                            theta_agg: str = "mean") -> pd.DataFrame:
    """
    核心亮点：售罄截断真实需求重构。

    规则：
      未售罄                    -> 真实需求 = 实际销量
      售罄 & 估计量有效(>销量)   -> 真实需求 = 需求估计量
      售罄 & 估计量无效          -> 真实需求 = 实际销量 * θ̄_{j,meal}
                                   （θ̄ 由同菜品同时段、有有效估计且售罄的历史日
                                     需求估计量/实际销量 的平均值估计）
                                   θ̄ 样本不足时回退 1.25 全局保守系数
    """
    df = df.copy()

    banner("任务2｜缺失值、异常值与「待核验」处理")
    est = df[COL_EST]
    actual = df[COL_SALES]

    # ---- 缺失统计 ---------------------------------------------------------
    LOG.info("核心字段缺失统计（清洗前）：")
    LOG.info(df[[COL_EST, COL_SALES, COL_LEFTOVER, COL_SOLDOUT]].isna().sum().to_string())
    LOG.info("备注取值分布：%s", df[COL_REMARK].fillna("<空>").value_counts(dropna=False).to_dict())

    # 「待核验 / 缺失」标记
    flagged = df[COL_REMARK].fillna("").str.contains("待核验|缺失", regex=True, na=False)
    n_flagged = int(flagged.sum())
    n_est_nan = int(est.isna().sum())
    LOG.info("标记为「待核验/缺失」的记录数：%d；需求估计量为空的记录数：%d",
             n_flagged, n_est_nan)
    LOG.info("「待核验/缺失」记录的售罄分布：\n%s",
             df.loc[flagged, COL_SOLDOUT].value_counts(dropna=False).to_string())

    # ---- 异常值体检 -------------------------------------------------------
    LOG.info("异常值体检：")
    LOG.info("  实际销量 < 0 的行数：%d", int((actual < 0).sum()))
    LOG.info("  剩余量 < 0 的行数：%d", int((df[COL_LEFTOVER] < 0).sum()))
    LOG.info("  备餐量 < 0 的行数：%d", int((df[COL_PREPARED] < 0).sum()))
    LOG.info("  实际销量 > 备餐量 的行数：%d", int((actual > df[COL_PREPARED]).sum()))
    LOG.info("  售罄=是 但 剩余量>0 的行数：%d",
             int((df[COL_SOLDOUT].eq("是") & (df[COL_LEFTOVER] > 0)).sum()))
    # 一致性：剩余量 == 备餐量 - 实际销量
    balance_err = (df[COL_PREPARED] - actual - df[COL_LEFTOVER]).abs() > 1e-6
    LOG.info("  剩余量 != 备餐量-实际销量 的行数：%d", int(balance_err.sum()))

    # ---- 售罄标志与有效性判定 ---------------------------------------------
    sold = df[COL_SOLDOUT].eq("是")
    not_sold = df[COL_SOLDOUT].eq("否")
    LOG.info("售罄分布：是=%d，否=%d", int(sold.sum()), int(not_sold.sum()))

    est_finite_pos = np.isfinite(est) & (est > 0)          # 估计量有限且为正
    est_above = est > actual                                # 估计量严格大于销量
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = est / actual                                # 溢出比例（售罄截断时 >1）
    ratio_sane = ratio < RATIO_MAX                          # 剔除异常极端估计

    # 售罄记录中「可直接采信」的估计量：正、大于销量、且非异常
    usable = est_finite_pos & est_above & ratio_sane
    theta_valid = sold & usable                             # 计算 θ̄ 的有效样本
    need_recon = sold & ~usable                             # 需要重构的售罄记录

    LOG.info("售罄记录中：估计量可直接采信 %d 条，需要重构 %d 条",
             int((sold & usable).sum()), int(need_recon.sum()))

    # ---- 计算菜品-时段级平均溢出系数 θ̄_{j,meal} ----------------------------
    theta_table = (
        df.loc[theta_valid]
        .assign(_ratio=ratio.loc[theta_valid])
        .groupby([COL_DISH_ID, COL_MEAL], sort=True, observed=True)["_ratio"]
        .agg(samples="size",
             theta_mean="mean",
             theta_median="median")
        .reset_index()
    )
    LOG.info("菜品-时段级溢出系数表（按 %s 聚合）：\n%s",
             theta_agg, theta_table.to_string(index=False))

    # 将 θ̄ 与样本数回挂到每一行（按 [菜品编号, 时段] 多级索引 reindex）
    idx = pd.MultiIndex.from_frame(df[[COL_DISH_ID, COL_MEAL]])
    theta_series = theta_table.set_index([COL_DISH_ID, COL_MEAL])
    df["_theta"] = theta_series[f"theta_{theta_agg}"].reindex(idx).to_numpy()
    df["_theta_n"] = theta_series["samples"].reindex(idx).to_numpy()

    has_theta = np.isfinite(df["_theta"].to_numpy()) & (df["_theta_n"].to_numpy() >= min_samples)

    # ---- 生成真实需求字段 -------------------------------------------------
    df["真实需求"] = np.nan
    df["真实需求来源"] = ""
    df["溢出系数"] = np.nan
    df["溢出系数样本数"] = np.nan

    # ① 未售罄：真实需求 = 实际销量
    df.loc[not_sold, "真实需求"] = actual[not_sold].astype(float)
    df.loc[not_sold, "真实需求来源"] = "未售罄=实际销量"

    # ② 售罄且估计量可直接采信：真实需求 = 需求估计量
    df.loc[sold & usable, "真实需求"] = est[sold & usable].astype(float)
    df.loc[sold & usable, "真实需求来源"] = "售罄=需求估计"

    # ③ 售罄且估计量无效：用溢出系数重构
    recon_theta = need_recon & has_theta
    recon_global = need_recon & ~has_theta
    df.loc[recon_theta, "真实需求"] = actual[recon_theta].astype(float) * df.loc[recon_theta, "_theta"]
    df.loc[recon_theta, "真实需求来源"] = "售罄=溢出系数重构"
    df.loc[recon_theta, "溢出系数"] = df.loc[recon_theta, "_theta"]
    df.loc[recon_theta, "溢出系数样本数"] = df.loc[recon_theta, "_theta_n"]

    df.loc[recon_global, "真实需求"] = actual[recon_global].astype(float) * global_fallback
    df.loc[recon_global, "真实需求来源"] = "售罄=全局系数重构"
    df.loc[recon_global, "溢出系数"] = global_fallback
    df.loc[recon_global, "溢出系数样本数"] = 0

    # ---- 重构结果审计 -----------------------------------------------------
    banner("任务3｜售罄截断真实需求重构结果")
    LOG.info("真实需求来源分布：\n%s", df["真实需求来源"].value_counts().to_string())
    LOG.info("重构前(需求估计量)均值：%.3f | 重构后(真实需求)均值：%.3f | 实际销量均值：%.3f",
             mean_of(df[COL_EST].dropna()), mean_of(df["真实需求"].dropna()),
             mean_of(actual))
    LOG.info("售罄记录：实际销量均值 %.3f -> 真实需求均值 %.3f（上修 %.2f%%）",
             mean_of(actual[sold]), mean_of(df.loc[sold, "真实需求"]),
             100.0 * (mean_of(df.loc[sold, "真实需求"]) / mean_of(actual[sold]) - 1.0))
    LOG.info("未售罄记录：真实需求均值 %.3f（应等于实际销量均值 %.3f）",
             mean_of(df.loc[not_sold, "真实需求"]), mean_of(actual[not_sold]))
    if need_recon.any():
        LOG.info("被重构的 %d 条记录明细：\n%s",
                 int(need_recon.sum()),
                 df.loc[need_recon, [COL_DATE, COL_DISH_ID, COL_MEAL, COL_EST, COL_SALES,
                                     "真实需求", "溢出系数", "溢出系数样本数", "真实需求来源"]]
                   .to_string(index=False))

    df = df.drop(columns=["_theta", "_theta_n"])
    return df


# ===========================================================================
# 任务 4：特征工程
# ===========================================================================
def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """时间特征：月份、日、星期编码、星期独热、时段编码、日期序号。"""
    df = df.copy()

    df["月份"] = df[COL_DATE].dt.month.astype("int16")
    df["日"] = df[COL_DATE].dt.day.astype("int16")

    # 星期编码：优先取数据自带的「星期」字符串，保证与业务口径一致
    df["星期_编码"] = df[COL_WEEKDAY].map(WEEKDAY_MAP).astype("int8")

    # 星期独热（补齐全部 5 类，防止某类缺失导致列数不稳定）
    wd = pd.get_dummies(df[COL_WEEKDAY], prefix="星期", dtype="int8")
    wd = wd.reindex(columns=[f"星期_{c}" for c in WEEKDAY_CATS], fill_value=0)
    df = pd.concat([df, wd.astype("int8")], axis=1)

    # 时段编码：午餐 -> 0，晚餐 -> 1
    df["时段_编码"] = df[COL_MEAL].map(MEAL_MAP).astype("int8")

    # 日期序号（自首日起的连续观测序号，刻画长期趋势，可用于树模型）
    df["日期_序号"] = (df[COL_DATE] - df[COL_DATE].min()).dt.days.astype("int16")

    return df


def build_env_features(df: pd.DataFrame) -> pd.DataFrame:
    """环境与校历特征：天气独热、二值化、活动影响比例转浮点。"""
    df = df.copy()

    # 天气独热（补齐全部类别）
    we = pd.get_dummies(df[COL_WEATHER], prefix="天气", dtype="int8")
    we = we.reindex(columns=[f"天气_{c}" for c in WEATHER_CATS], fill_value=0)
    df = pd.concat([df, we.astype("int8")], axis=1)

    # 二值字段 -> 0/1
    df[COL_EXAM] = to_binary(df[COL_EXAM])
    df[COL_LEAVE_EVE] = to_binary(df[COL_LEAVE_EVE])
    df[COL_EVENT] = to_binary(df[COL_EVENT])

    # 活动客流影响估计比例：字符串百分比 -> 浮点比例
    df[COL_EVENT_RATIO] = parse_ratio_series(df[COL_EVENT_RATIO])

    # 连续环境字段数值化
    for c in [COL_TEMP, COL_RAIN, COL_STUDENT_RATIO]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    历史滞后与惯性特征。

    严格按 [菜品编号, 时段] 分组、按日期排序后计算，天然防数据泄漏：
      - Lag_1          : 该菜品该时段「上一个观测日」的真实需求
      - Lag_5          : 该菜品该时段「5 个工作日前」的真实需求
      - Rolling_Mean_3 : 该菜品该时段「过去 3 次」真实需求的滚动平均（不含当日）
    早期观测产生的 NaN 用该菜品该时段的整体均值安全填充。
    """
    df = df.reset_index(drop=True).copy()
    # 关键：先排序，确保 shift/rolling 的时间顺序正确
    df = df.sort_values([COL_DISH_ID, COL_MEAL, COL_DATE], kind="mergesort").reset_index(drop=True)

    grp = df.groupby([COL_DISH_ID, COL_MEAL], sort=False, observed=True)["真实需求"]

    df["Lag_1"] = grp.shift(1)
    df["Lag_5"] = grp.shift(5)

    # 过去 3 次需求的滚动平均：先 shift(1) 排除当日，再做组内 rolling(3)
    lag1 = df["Lag_1"]
    rolling = lag1.groupby([df[COL_DISH_ID], df[COL_MEAL]], sort=False).rolling(
        window=3, min_periods=1).mean()
    df["Rolling_Mean_3"] = rolling.reset_index(level=[0, 1], drop=True)

    # 安全填充：该菜品该时段的整体历史均值（预测行真实需求为 NaN，不参与均值）
    group_mean = grp.transform("mean")
    n_fill = {c: int(df[c].isna().sum()) for c in ["Lag_1", "Lag_5", "Rolling_Mean_3"]}
    for c in ["Lag_1", "Lag_5", "Rolling_Mean_3"]:
        df[c] = df[c].fillna(group_mean)

    banner("任务4｜历史滞后特征（Lag）")
    LOG.info("滞后特征生成完成；填充前的 NaN 计数：%s", n_fill)
    LOG.info("分组均值（用于填充）范围：%.3f ~ %.3f",
             float(group_mean.min()), float(group_mean.max()))
    return df


# ===========================================================================
# 任务 5：预测目标集构造与数据集划分
# ===========================================================================
def build_prediction_template(hist: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
    """
    构造待预测决策日(2026-06-29)的目标集：10 菜品 × 2 时段 = 20 行，
    仅含环境特征，销量/备餐量/真实需求等留空，其 Lag 由历史数据回填。
    """
    marker_rows = env_df[env_df[COL_USAGE].eq(PREDICTION_MARKER)]
    if marker_rows.empty:
        raise ValueError("附件2 中未找到「待预测决策日」记录，无法构造目标集。")

    banner("任务5｜构造待预测决策日目标集")
    LOG.info("待预测决策日记录：\n%s", marker_rows.to_string(index=False))

    dishes = (hist[[COL_DISH_ID, COL_DISH_NAME]]
              .drop_duplicates()
              .sort_values(COL_DISH_ID)
              .reset_index(drop=True))
    meals = pd.DataFrame({COL_MEAL: ["午餐", "晚餐"]})

    # 笛卡尔积：菜品 × 时段
    template = dishes.assign(_k=1).merge(meals.assign(_k=1), on="_k").drop(columns="_k")
    # 广播环境特征（与历史表同样的环境字段，去掉会冲突的「星期」）
    env = marker_rows.drop(columns=[COL_WEEKDAY], errors="ignore").assign(_k=1)
    template = template.assign(_k=1).merge(env, on="_k").drop(columns="_k")

    # 由日期反推星期，保证与历史表口径一致
    weekday_of = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
    template[COL_WEEKDAY] = template[COL_DATE].dt.dayofweek.map(weekday_of).astype("string")

    # 对齐历史表列结构，缺失的业务/目标列置空
    template = template.reindex(columns=hist.columns)
    for c in [COL_BATCH, COL_PREPARED, COL_EST, COL_SALES, COL_LEFTOVER,
              COL_SOLDOUT, COL_SOLDOUT_TIME, COL_REMARK, "真实需求", "溢出系数",
              "溢出系数样本数"]:
        if c in template.columns:
            template[c] = pd.NA
    template["真实需求来源"] = PREDICTION_MARKER
    template[COL_USAGE] = PREDICTION_MARKER

    LOG.info("目标集模板形状：%s（应为 (20, %d)）", template.shape, hist.shape[1])
    return template


# ===========================================================================
# 主流程
# ===========================================================================
def run_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    script_dir = Path(__file__).resolve().parent
    output_dir = resolve_output_dir(args.output_dir, script_dir)
    add_file_logger(output_dir)

    banner("阶段一｜数据预处理与特征工程 启动")
    LOG.info("数据目录：%s", data_dir.resolve())
    LOG.info("输出目录：%s", output_dir.resolve())

    # ---- 任务 1：读取与对齐 ------------------------------------------------
    hist, env_df = load_and_align(data_dir, args.attachment1, args.attachment2)

    # ---- 任务 2/3：清洗 + 真实需求重构 ------------------------------------
    hist = reconstruct_true_demand(
        hist,
        min_samples=args.min_theta_samples,
        global_fallback=args.global_overflow,
        theta_agg=args.theta_agg,
    )

    # ---- 任务 5（前置）：构造预测目标集模板 ---------------------------------
    predict_template = build_prediction_template(hist, env_df)

    # ---- 合并历史 + 目标集，统一做特征工程（保证特征列完全一致） -----------
    full = pd.concat([hist, predict_template], axis=0, ignore_index=True)
    LOG.info("合并历史(%d)与目标集(%d)后，全量记录数：%d",
             len(hist), len(predict_template), len(full))

    # ---- 任务 4：特征工程（时间 + 环境 + 滞后）------------------------------
    full = build_time_features(full)
    full = build_env_features(full)
    full = build_lag_features(full)

    # 英文别名字段 true_demand（在划分前添加，确保训练集/目标集都包含）
    full["true_demand"] = full["真实需求"]

    # ---- 任务 5：划分数据集 ------------------------------------------------
    train = full[full[COL_USAGE].eq(HISTORY_MARKER)].copy()
    predict = full[full[COL_USAGE].eq(PREDICTION_MARKER)].copy()

    banner("任务5｜数据集划分")
    LOG.info("训练/验证集（历史）形状：%s", train.shape)
    LOG.info("预测目标集（2026-06-29）形状：%s", predict.shape)
    LOG.info("预测目标集行数核对：%d（应为 10 菜品 × 2 时段 = 20）", len(predict))

    # ---- 汇总特征列 --------------------------------------------------------
    time_feats = ["月份", "日", "星期_编码", "时段_编码",
                  *[f"星期_{c}" for c in WEEKDAY_CATS], "日期_序号"]
    env_feats = [*[f"天气_{c}" for c in WEATHER_CATS], COL_TEMP, COL_RAIN,
                 COL_STUDENT_RATIO, COL_EXAM, COL_LEAVE_EVE, COL_EVENT, COL_EVENT_RATIO]
    lag_feats = ["Lag_1", "Lag_5", "Rolling_Mean_3"]
    feature_columns = [*time_feats, *env_feats, *lag_feats]
    missing_feats = [c for c in feature_columns if c not in full.columns]
    if missing_feats:
        raise RuntimeError(f"特征列缺失：{missing_feats}")

    # 标识列与辅助列
    id_columns = [COL_DATE, COL_DISH_ID, COL_DISH_NAME, COL_MEAL, COL_WEEKDAY]
    aux_columns = [c for c in full.columns
                   if c not in id_columns and c not in feature_columns
                   and c not in ("真实需求", "true_demand")]

    # ---- 保存结果 ----------------------------------------------------------
    ordered_cols = ["真实需求", "true_demand", *id_columns, *feature_columns, *aux_columns]
    final_full = full[ordered_cols]          # 完整干净特征数据集（历史 + 待预测决策日）
    final_train = train[ordered_cols]        # 历史训练/验证集
    final_predict = predict[ordered_cols]    # 待预测决策日目标集

    def write_csv(df: pd.DataFrame, name: str, mirror_dir: Path | None) -> Path:
        out_path = output_dir / name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        LOG.info("已写出 %s（%d 行 × %d 列）", out_path.resolve(), len(df), df.shape[1])
        if mirror_dir is not None:
            mirror_path = mirror_dir / name
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(mirror_path, index=False, encoding="utf-8-sig")
            LOG.info("镜像写出 %s", mirror_path.resolve())
        return out_path

    # 镜像目录：与数据同级的 output/，便于直接查看
    mirror_dir = data_dir / "output" if args.mirror else None

    write_csv(final_full, "cleaned_data.csv", mirror_dir)           # 完整干净特征数据集
    write_csv(final_train, "train_validation.csv", mirror_dir)      # 历史训练/验证集
    write_csv(final_predict, "prediction_target.csv", mirror_dir)   # 待预测决策日目标集

    # 列元数据 JSON
    metadata = {
        "target": ["真实需求", "true_demand"],
        "id_columns": id_columns,
        "feature_columns": feature_columns,
        "aux_columns": aux_columns,
        "note": ("真实需求为目标变量；feature_columns 为可直接用于 LightGBM/XGBoost 的特征；"
                 "aux_columns 含原始业务字段与重构审计字段，建模时勿直接作为特征。"),
    }
    meta_path = output_dir / "column_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("已写出列元数据 %s", meta_path.resolve())

    # ---- 最终监督摘要 ------------------------------------------------------
    banner("监督摘要")
    LOG.info("特征列（%d 个）：%s", len(feature_columns), feature_columns)
    LOG.info("预测目标集（2026-06-29）的 Lag 回填抽查：\n%s",
             final_predict[[COL_DISH_ID, COL_MEAL, "Lag_1", "Lag_5", "Rolling_Mean_3",
                            "真实需求"]].to_string(index=False))
    LOG.info("训练/验证集 Lag 字段填充后剩余 NaN 数：%s",
             {c: int(final_train[c].isna().sum()) for c in lag_feats})
    LOG.info("预测目标集 Lag 字段 NaN 数（应为 0）：%s",
             {c: int(final_predict[c].isna().sum()) for c in lag_feats})
    LOG.info("完成。主输出目录：%s", output_dir.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 默认以脚本所在目录为基准，附件与输出均相对此目录，换机/换目录无需改代码
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="校园食堂备餐优化 —— 阶段一：数据预处理与特征工程")
    parser.add_argument("--data-dir", default=str(script_dir),
                        help="附件1/2 所在目录（默认：脚本所在目录）")
    parser.add_argument("--attachment1", default=None, help="附件1 文件名（覆盖默认）")
    parser.add_argument("--attachment2", default=None, help="附件2 文件名（覆盖默认）")
    parser.add_argument("--output-dir", default=str(script_dir / "output"),
                        help="输出目录（默认：脚本所在目录下的 output/）")
    parser.add_argument("--global-overflow", type=float, default=GLOBAL_FALLBACK_OVERFLOW,
                        help="全局缺供保守溢出系数（默认 1.25）")
    parser.add_argument("--min-theta-samples", type=int, default=MIN_THETA_SAMPLES,
                        help="计算菜品-时段溢出系数所需最少样本数（默认 3）")
    parser.add_argument("--theta-agg", choices=["mean", "median"], default="mean",
                        help="溢出系数聚合方式：mean（默认，对应题面“平均溢出系数”）或 median（更稳健）")
    parser.add_argument("--mirror", action="store_true",
                        help="是否在数据目录下同时输出一份镜像 CSV")
    parser.add_argument("--verbose", action="store_true", help="输出 DEBUG 级日志")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # 让控制台/管道输出统一走 UTF-8，避免中文日志在 GBK 控制台下乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass

    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        run_pipeline(args)
    except Exception as exc:  # 顶层兜底：保证异常信息可读
        LOG.exception("流水线执行失败：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
