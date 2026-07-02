# -*- coding: utf-8 -*-
"""
evaluate.py - 自动评估脚本
功能：对比 v1/v2/模型版在人工标注测试集上的准确率、F1、混淆矩阵
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)

# ==================== 自动切换工作目录（解决相对路径问题） ====================
# 获取当前脚本所在目录（如 D:/项目/scripts/evaluate.py）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（因为脚本在 scripts/ 下，向上一级）
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 强制切换工作目录到项目根目录
os.chdir(PROJECT_ROOT)
print(f"当前工作目录: {os.getcwd()}")


# ==================== 评估函数 ====================
def evaluate_version(pred_df, gt_df, version_name):
    """
    评估单个版本的预测结果
    返回: (准确率, F1, 合并后的DataFrame, 主题覆盖率)
    """
    # 按 title 合并
    merged = pd.merge(gt_df, pred_df, on="title", how="inner")
    print(f"  {version_name} 匹配样本数: {len(merged)} / {len(gt_df)}")

    if len(merged) == 0:
        return None, None, None, None

    y_true = merged["true_sentiment"]
    y_pred = merged["sentiment"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 主题覆盖率（仅对 v1/v2 有效）
    coverage = None
    if "topic" in pred_df.columns:
        non_other = (pred_df["topic"] != "其他").sum() / len(pred_df)
        coverage = non_other
    elif "topic" in merged.columns:
        non_other = (merged["topic"] != "其他").sum() / len(merged)
        coverage = non_other

    return acc, f1, merged, coverage


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("舆情日报自动评估")
    print("=" * 60)

    # 1. 加载人工标注基准
    try:
        ground_truth = pd.read_excel("data/ground_truth.xlsx")
        print(f"✅ 加载 ground_truth: {len(ground_truth)} 条")
        # 检查标签格式
        unique_labels = ground_truth["true_sentiment"].unique()
        print(f"   标签类别: {unique_labels}")
    except FileNotFoundError:
        print("❌ 请先创建 data/ground_truth.xlsx（包含 title 和 true_sentiment 两列）")
        return

    # 2. 定义各版本文件路径（请根据实际文件名修改）
    version_files = {
        "v1": "data/raw/hotlist_2026-06-17_v1.csv",
        "v2": "data/raw/hotlist_2026-06-17_v2.csv",
        "模型版": "reports/data/model/hotlist_2026-06-17_model.csv",
    }

    # 3. 逐个评估
    results = {}
    for name, path in version_files.items():
        print(f"\n📊 评估 {name}: {path}")
        try:
            pred_df = pd.read_csv(path)
            acc, f1, merged, coverage = evaluate_version(pred_df, ground_truth, name)
            results[name] = {
                "acc": acc,
                "f1": f1,
                "merged": merged,
                "coverage": coverage
            }
        except FileNotFoundError:
            print(f"  ⚠️ 文件不存在，跳过")
            results[name] = {"acc": None, "f1": None, "merged": None, "coverage": None}

    # 4. 输出对比表格
    print("\n" + "=" * 60)
    print("评估对比结果")
    print("=" * 60)
    print(f"| 版本 | 准确率 | 加权F1 | 主题覆盖率 |")
    print(f"|------|--------|--------|------------|")
    for name, res in results.items():
        acc_str = f"{res['acc']:.2%}" if res['acc'] is not None else "N/A"
        f1_str = f"{res['f1']:.4f}" if res['f1'] is not None else "N/A"
        cov_str = f"{res['coverage']:.2%}" if res['coverage'] is not None else "N/A"
        print(f"| {name} | {acc_str} | {f1_str} | {cov_str} |")

    # 5. 模型版分类报告和混淆矩阵
    if results.get("模型版") and results["模型版"]["merged"] is not None:
        merged = results["模型版"]["merged"]
        y_true = merged["true_sentiment"]
        y_pred = merged["sentiment"]

        print("\n" + "=" * 60)
        print("模型版分类报告")
        print("=" * 60)
        print(classification_report(y_true, y_pred, zero_division=0))

        # 生成混淆矩阵图
        labels = ["正面", "中性", "负面"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("模型版混淆矩阵")

        # 保存图片
        os.makedirs("reports", exist_ok=True)
        plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("✅ 混淆矩阵已保存至 reports/confusion_matrix.png")
        plt.close()

    # 6. 保存评估报告到文件
    with open("reports/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("========== 评估对比结果 ==========\n")
        f.write("| 版本 | 准确率 | 加权F1 | 主题覆盖率 |\n")
        f.write("|------|--------|--------|------------|\n")
        for name, res in results.items():
            acc_str = f"{res['acc']:.2%}" if res['acc'] is not None else "N/A"
            f1_str = f"{res['f1']:.4f}" if res['f1'] is not None else "N/A"
            cov_str = f"{res['coverage']:.2%}" if res['coverage'] is not None else "N/A"
            f.write(f"| {name} | {acc_str} | {f1_str} | {cov_str} |\n")

        if results.get("模型版") and results["模型版"]["merged"] is not None:
            f.write("\n========== 模型版分类报告 ==========\n")
            merged = results["模型版"]["merged"]
            y_true = merged["true_sentiment"]
            y_pred = merged["sentiment"]
            f.write(classification_report(y_true, y_pred, zero_division=0))

    print("\n✅ 评估报告已保存至 reports/evaluation_report.txt")


if __name__ == "__main__":
    main()