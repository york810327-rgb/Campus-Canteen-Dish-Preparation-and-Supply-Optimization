# 校园食堂菜品备餐与供应优化 —— 代码运行说明

## 一、运行方式

### 第 1 步：准备文件

把下面这些文件放到**同一个文件夹**里（文件夹名字和盘符随意，例如 `C:\评审\`）：

| 类型 | 文件 |
|---|---|
| 主入口 | `run_all.py` |
| 阶段脚本（5 个） | `preprocessing_feature_engineering.py`、`predict_model.py`、`optimize_deterministic.py`、`optimize_stochastic.py`、`finalize_paper_assets.py` |
| 输入数据（4 个） | `附件1_历史销售与需求数据.xlsx`、`附件2_校园环境与运行信息.xlsx`、`附件3_菜品生产经营参数.xlsx`、`附件4_结果汇总表.xlsx` |

> ⚠️ 关键点：**所有脚本的路径都以"脚本所在目录"为基准自动定位**，因此无论放在哪个盘、哪个文件夹都能跑，不再依赖 `D:\B` 这类绝对路径。

### 第 2 步：安装依赖（首次）

```bat
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy lightgbm pulp openpyxl
```

### 第 3 步：一键运行

```bat
python run_all.py
```

它会自动依次执行阶段一 → 五，并把所有结果输出到 `当前文件夹/output/` 下。
（阶段四的随机规划求解约需 3 分钟，属正常现象。）

> 可选：
> - `python run_all.py --start 3`  ：从阶段三开始跑（前提是前两阶段结果已存在）
> - `python run_all.py --skip-stage4`：跳过阶段四快速联调

---

## 二、各脚本单独运行（可选）

若想单独运行某一步，所有脚本都支持命令行参数（默认值已自动指向脚本所在目录）：

| 脚本 | 单独运行命令 |
|---|---|
| 阶段一 | `python preprocessing_feature_engineering.py` |
| 阶段二 | `python predict_model.py` |
| 阶段三 | `python optimize_deterministic.py` |
| 阶段四 | `python optimize_stochastic.py` |
| 阶段五 | `python finalize_paper_assets.py` |

通用参数（每个脚本都支持，需要时手动覆盖默认路径）：

```bat
python optimize_stochastic.py --data-dir "D:\你的目录" --output-dir "D:\你的目录\output"
```

---

## 三、输出文件说明

运行完成后，`output/` 目录下包含：

- 数据：`cleaned_data.csv`、`train_validation.csv`、`prediction_target.csv`、`column_metadata.json`
- 预测：`predictions_20260629.csv`、`model_residuals.csv`、`residual_stats.csv`
- EDA：`eda_correlation.csv`、`eda_anova.csv`
- 优化：`q2_optimal_plan.csv`、`q3_optimal_plan.csv`、`q2_vs_q3_comparison.csv`、`q2_performance_metrics.csv`、`q2_resource_utilization.csv`
- 图表（`output/images/`）：相关性热力图、3 张箱线图、残差分布图、资源饱和度图、2 张 CDF 对比图
- 报告：`paper_writing_materials.md`（论文素材白皮书）、`audit_report.csv`（一致性审计）
- 回填后的 `附件4_结果汇总表.xlsx`（在数据目录下，Q1/Q2/Q3 均已填好，公式与格式保留）

---

## 四、常见问题排查

1. **提示找不到附件文件**：确认 4 个附件 xlsx 与 `run_all.py` 在同一个文件夹，且文件名不要改动。
2. **提示缺依赖库**：按上面的 pip 命令安装（`lightgbm`、`pulp` 是关键）。
3. **中文乱码**：脚本已强制 UTF-8 输出；若控制台仍乱码，不影响文件结果，可查看各阶段 `.log` 日志文件（UTF-8 编码）。
4. **阶段四较慢**：1000 场景的两阶段随机规划 MILP，CBC 求解约 3 分钟，请耐心等待。

---

## 五、关键结论速览

| 指标 | 数值 |
|---|---|
| Q1 需求预测 MAPE | 9.04% |
| Q2 确定性最优综合经营效益 | 14,328.20 元 |
| Q3 随机稳健期望综合经营效益 | 13,375.96 元 |
| 需求不确定性代价 | 952.24 元（6.65%） |
| 资源瓶颈 | 午餐人工 99.46%、晚餐主灶 99.60% |
| 期望服务水平（1000 场景） | 94.55% |
