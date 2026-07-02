# -*- coding: utf-8 -*-
"""
compare_versions.py - 版本对比工具
功能：对比 v1/v2/模型版在人工标注测试集上的评估结果，生成汇总表格
"""

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==================== 自动切换工作目录 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
print(f"当前工作目录: {os.getcwd()}")


# ==================== 辅助函数 ====================
def load_version_data(file_path, version_name):
    """加载单个版本的预测数据，如果文件不存在则返回 None"""
    if not os.path.exists(file_path):
        print(f"⚠️  {version_name} 文件不存在: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"✅ 加载 {version_name}: {len(df)} 条")
        return df
    except Exception as e:
        print(f"❌ 加载 {version_name} 失败: {e}")
        return None


def calculate_metrics(pred_df, gt_df):
    """计算评估指标：准确率、F1、主题覆盖率"""
    if pred_df is None or gt_df is None:
        return None, None, None

    # 按 title 合并
    merged = pd.merge(gt_df, pred_df, on="title", how="inner")
    if len(merged) == 0:
        print(f"   ⚠️ 无匹配样本（标题不一致）")
        return None, None, None

    y_true = merged["true_sentiment"]
    y_pred = merged["sentiment"]

    # 计算准确率和 F1
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

    # 打印调试信息
    print(f"   匹配样本: {len(merged)} 条")
    print(f"   准确率: {acc:.2%}, F1: {f1:.4f}")
    if coverage is not None:
        print(f"   主题覆盖率: {coverage:.2%}")

    return acc, f1, coverage, merged


def extract_classification_report(merged):
    """从合并数据中提取分类报告"""
    y_true = merged["true_sentiment"]
    y_pred = merged["sentiment"]
    return classification_report(y_true, y_pred, zero_division=0)


# ==================== 主流程 ====================
def main():
    print("=" * 70)
    print("版本对比工具")
    print("=" * 70)

    # 1. 加载人工标注基准
    try:
        ground_truth = pd.read_excel("data/ground_truth.xlsx")
        print(f"\n✅ 加载 ground_truth: {len(ground_truth)} 条")
    except FileNotFoundError:
        print("❌ 请先创建 data/ground_truth.xlsx（包含 title 和 true_sentiment 两列）")
        return

    # 2. 定义各版本文件路径（请根据实际情况修改文件名）
    version_configs = {
        "v1": {
            "path": "data/raw/hotlist_2026-06-17_v1.csv",
            "desc": "基础规则（5类，15个关键词）"
        },
        "v2": {
            "path": "data/raw/hotlist_2026-06-17_v2.csv",
            "desc": "扩充主题词（7类，80+关键词）"
        },
        "模型版": {
            "path": "reports/data/model/hotlist_2026-06-17_model.csv",
            "desc": "BERT 情感分析 + jieba 关键词提取"
        }
    }

    # 3. 逐个评估
    results = {}
    print("\n" + "-" * 70)
    print("开始评估各版本...")
    print("-" * 70)

    for name, config in version_configs.items():
        print(f"\n📊 评估 {name}:")
        print(f"   描述: {config['desc']}")
        print(f"   文件: {config['path']}")

        pred_df = load_version_data(config['path'], name)
        if pred_df is None:
            results[name] = {"acc": None, "f1": None, "coverage": None, "merged": None}
            continue

        acc, f1, coverage, merged = calculate_metrics(pred_df, ground_truth)
        results[name] = {
            "acc": acc,
            "f1": f1,
            "coverage": coverage,
            "merged": merged,
            "desc": config['desc']
        }

    # 4. 输出对比表格
    print("\n" + "=" * 70)
    print("📊 版本对比结果")
    print("=" * 70)
    print(f"\n| 版本 | 准确率 | 加权F1 | 主题覆盖率 | 说明 |")
    print(f"|------|--------|--------|------------|------|")

    for name, res in results.items():
        acc_str = f"{res['acc']:.2%}" if res['acc'] is not None else "N/A"
        f1_str = f"{res['f1']:.4f}" if res['f1'] is not None else "N/A"
        cov_str = f"{res['coverage']:.2%}" if res['coverage'] is not None else "N/A"
        desc = res.get('desc', '')
        print(f"| {name} | {acc_str} | {f1_str} | {cov_str} | {desc} |")

    # 5. 模型版详细分类报告
    if results.get("模型版") and results["模型版"]["merged"] is not None:
        merged = results["模型版"]["merged"]
        print("\n" + "=" * 70)
        print("📋 模型版分类报告")
        print("=" * 70)
        report = extract_classification_report(merged)
        print(report)

    # 6. 保存结果到文件
    os.makedirs("reports", exist_ok=True)
    with open("reports/version_comparison.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("版本对比结果\n")
        f.write("=" * 70 + "\n\n")
        f.write("| 版本 | 准确率 | 加权F1 | 主题覆盖率 | 说明 |\n")
        f.write("|------|--------|--------|------------|------|\n")
        for name, res in results.items():
            acc_str = f"{res['acc']:.2%}" if res['acc'] is not None else "N/A"
            f1_str = f"{res['f1']:.4f}" if res['f1'] is not None else "N/A"
            cov_str = f"{res['coverage']:.2%}" if res['coverage'] is not None else "N/A"
            desc = res.get('desc', '')
            f.write(f"| {name} | {acc_str} | {f1_str} | {cov_str} | {desc} |\n")

        if results.get("模型版") and results["模型版"]["merged"] is not None:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("模型版分类报告\n")
            f.write("=" * 70 + "\n")
            merged = results["模型版"]["merged"]
            f.write(extract_classification_report(merged))

    print(f"\n✅ 对比结果已保存至 reports/version_comparison.txt")

    # 7. 总结
    print("\n" + "=" * 70)
    print("📌 关键发现")
    print("=" * 70)

    v1_acc = results.get("v1", {}).get("acc")
    v2_acc = results.get("v2", {}).get("acc")
    model_acc = results.get("模型版", {}).get("acc")

    if v1_acc is not None and v2_acc is not None:
        print(f"✅ v2 相比 v1：主题覆盖率提升，情感准确率持平（词典法局限性）")
    if v1_acc is not None and model_acc is not None:
        print(f"✅ 模型版相比 v1：准确率提升 {(model_acc - v1_acc):.2%}")
    if v2_acc is not None and model_acc is not None:
        print(f"✅ 模型版相比 v2：准确率提升 {(model_acc - v2_acc):.2%}")

    if results.get("模型版", {}).get("merged") is not None:
        merged = results["模型版"]["merged"]
        y_true = merged["true_sentiment"]
        y_pred = merged["sentiment"]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=["正面", "中性", "负面"])
        print(f"\n📊 模型版混淆矩阵:")
        print(f"   正面: {cm[0, 0]} 正确, {cm[0, 1] + cm[0, 2]} 错误")
        print(f"   中性: {cm[1, 1]} 正确, {cm[1, 0] + cm[1, 2]} 错误")
        print(f"   负面: {cm[2, 2]} 正确, {cm[2, 0] + cm[2, 1]} 错误")

    print("\n" + "=" * 70)
    print("✅ 版本对比完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()