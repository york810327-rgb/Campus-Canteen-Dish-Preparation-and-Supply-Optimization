# -*- coding: utf-8 -*-
"""
================================================================================
一键运行脚本（主入口）—— 校园食堂菜品备餐与供应优化 全流程
================================================================================

使用方式（最简单）：
    把本文件、5 个阶段脚本、4 个附件 xlsx 放在【同一个文件夹】里，
    然后在该文件夹下执行：
        python run_all.py

脚本会自动：
    1. 以【本文件所在目录】为基准定位所有数据与脚本（换电脑/换目录都无需改任何路径）；
    2. 依次运行 阶段一 → 二 → 三 → 四 → 五；
    3. 所有中间结果与最终成果统一输出到 本目录/output/ 下；
    4. 任一步失败立即停止并给出提示。

可选参数：
    --start N        从第 N 个阶段开始运行（默认 1）
    --skip-stage4    跳过耗时约 3 分钟的随机规划求解（用于快速联调）
================================================================================
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"

# (阶段名, 脚本名, 命令行参数列表)
STAGES = [
    ("阶段一：数据预处理与特征工程", "preprocessing_feature_engineering.py",
     ["--data-dir", str(BASE), "--output-dir", str(OUT)]),
    ("阶段二：需求特征分析与预测模型", "predict_model.py",
     ["--input-dir", str(OUT), "--output-dir", str(OUT), "--mirror-dir", ""]),
    ("阶段三：问题2 确定性备餐优化(MILP)", "optimize_deterministic.py",
     ["--data-dir", str(BASE), "--output-dir", str(OUT), "--mirror-dir", ""]),
    ("阶段四：问题3 两阶段随机规划(SAA)", "optimize_stochastic.py",
     ["--data-dir", str(BASE), "--output-dir", str(OUT), "--mirror-dir", ""]),
    ("阶段五：学术图表终验与报告生成", "finalize_paper_assets.py",
     ["--data-dir", str(BASE), "--output-dir", str(OUT), "--mirror-dir", ""]),
]

REQUIRED_FILES = [
    "附件1_历史销售与需求数据.xlsx",
    "附件2_校园环境与运行信息.xlsx",
    "附件3_菜品生产经营参数.xlsx",
    "附件4_结果汇总表.xlsx",
]


def _ensure_utf8() -> None:
    """让控制台输出统一走 UTF-8，避免中文/特殊符号在 GBK 控制台下报错。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def _banner(msg: str) -> None:
    print("\n" + "=" * 88)
    print(msg)
    print("=" * 88)


def check_environment() -> None:
    """检查依赖库是否安装，缺失则给出明确提示。"""
    missing = []
    for mod in ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "scipy",
                "lightgbm", "pulp", "openpyxl"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("缺少依赖库：" + ", ".join(missing))
        print("请先安装：  python -m pip install pandas numpy matplotlib seaborn "
              "scikit-learn scipy lightgbm pulp openpyxl")
        sys.exit(1)


def check_data_files() -> None:
    missing = [f for f in REQUIRED_FILES if not (BASE / f).exists()]
    if missing:
        print("未找到以下输入数据文件：" + ", ".join(missing))
        print(f"请确认这 {len(REQUIRED_FILES)} 个附件 xlsx 与 run_all.py 位于同一目录：{BASE}")
        sys.exit(1)
    print(f"[OK] 数据文件检查通过（{len(REQUIRED_FILES)} 个附件齐全），基准目录：{BASE}")


def run_stage(name: str, script: str, args: list[str]) -> None:
    _banner(f">> 开始运行：{name}")
    cmd = [sys.executable, str(BASE / script), *args]
    print("执行命令：", " ".join(cmd))
    # 继承父进程 stdio，让各脚本的日志直接打印到控制台
    proc = subprocess.run(cmd, cwd=str(BASE))
    if proc.returncode != 0:
        print(f"\n[X] {name} 执行失败（退出码 {proc.returncode}），流程终止。")
        sys.exit(proc.returncode)
    print(f"[OK] {name} 执行完成。\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="校园食堂备餐优化 全流程一键运行")
    parser.add_argument("--start", type=int, default=1, help="从第几个阶段开始运行（默认 1）")
    parser.add_argument("--skip-stage4", action="store_true",
                        help="跳过阶段四（随机规划求解约 3 分钟），用于快速联调")
    args = parser.parse_args()

    _ensure_utf8()

    print("[INFO] 校园食堂菜品备餐与供应优化 —— 全流程一键运行")
    print("[INFO] 基准目录：", BASE)
    print("[INFO] 输出目录：", OUT)
    print()

    check_environment()
    check_data_files()
    OUT.mkdir(parents=True, exist_ok=True)

    for i, (name, script, stage_args) in enumerate(STAGES, start=1):
        if i < args.start:
            print(f"[SKIP] 跳过：{name}")
            continue
        if i == 4 and args.skip_stage4:
            print(f"[SKIP] 跳过（--skip-stage4）：{name}")
            continue
        run_stage(name, script, stage_args)

    _banner("[DONE] 全部 5 个阶段运行完成")
    print(f"所有成果位于：{OUT}")
    print("关键成果：")
    for f in ["cleaned_data.csv", "predictions_20260629.csv", "q2_optimal_plan.csv",
              "q3_optimal_plan.csv", "q2_vs_q3_comparison.csv", "paper_writing_materials.md"]:
        p = OUT / f
        if p.exists():
            print(f"   [OK] {f}")
    print(f"学术插图位于：{OUT / 'images'}")
    print(f"回填后的结果汇总表：{BASE / '附件4_结果汇总表.xlsx'}")


if __name__ == "__main__":
    main()
