# -*- coding: utf-8 -*-
"""
pipeline_model.py - 模型版舆情日报生成脚本（纯真实数据版）
功能：数据采集（严格依赖真实API） → 主题分类 → 情感分析(含置信度) → 风险判断 → 日报生成
说明：如API全部失败，程序主动报错退出，不产生任何模拟数据。
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
import jieba
import jieba.analyse
from transformers import pipeline

# ==================== 自动切换工作目录 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
os.chdir(PROJECT_ROOT)
print(f"当前工作目录: {os.getcwd()}")

# ==================== 1. 数据采集 ====================
API_RETRY_COUNT = 0
API_RETRY_MAX = 3
API_HISTORY_FILE = "data/api_status_history.json"

def fetch_hotlist():
    """
    获取微博热搜数据。
    尝试所有API源，若全部失败则抛出 RuntimeError，绝不返回模拟数据。
    """
    global API_RETRY_COUNT

    # 定义API源列表（按优先级排序）
    apis = [
        ("https://uapis.cn/api/v1/misc/hotboard", {"type": "weibo"}),
        ("http://api.rosysun.cn/weibo/", None),  # 备用免费API
    ]

    for url, params in apis:
        try:
            print(f"  尝试请求: {url}")
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                # 解析UAPIS格式
                if "list" in data:
                    hotlist = data["list"]
                    df = pd.DataFrame(hotlist)
                    df['rank'] = range(1, len(df)+1)
                    df['title'] = df['title']
                    df['hot'] = df['hot_value'].astype(int)
                    API_RETRY_COUNT = 0
                    _save_api_status(True, "成功")
                    return df[['rank', 'title', 'hot']]
                # 解析备用API格式
                elif "data" in data and isinstance(data["data"], list):
                    hotlist = data["data"]
                    df = pd.DataFrame(hotlist)
                    df['rank'] = range(1, len(df)+1)
                    df['title'] = df['title']
                    df['hot'] = 500000  # 备用源可能无热度，设默认值
                    API_RETRY_COUNT = 0
                    _save_api_status(True, "成功（备用API）")
                    return df[['rank', 'title', 'hot']]
        except Exception as e:
            print(f"  请求失败: {e}")
            continue

    # 所有API均失败 → 记录状态并抛出异常
    API_RETRY_COUNT += 1
    _save_api_status(False, f"连续失败{API_RETRY_COUNT}次")

    error_msg = (
        f"\n❌ 数据获取严重错误：所有 API 源均无法连接（连续失败 {API_RETRY_COUNT} 次）。\n"
        "   【作业说明】该工作流严格依赖真实数据源，请检查网络或稍后重试。\n"
        "   程序已终止，无法继续生成日报。"
    )
    raise RuntimeError(error_msg)

def _save_api_status(success, message):
    """保存API调用状态到本地JSON（用于历史追踪）"""
    os.makedirs("data", exist_ok=True)
    try:
        history = []
        if os.path.exists(API_HISTORY_FILE):
            with open(API_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        history = history[-6:] if len(history) > 6 else history
        history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "success": success,
            "message": message,
            "retry_count": API_RETRY_COUNT
        })
        with open(API_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except:
        pass  # 状态文件写入失败不影响主流程

def get_api_status():
    """获取API连续失败次数"""
    if not os.path.exists(API_HISTORY_FILE):
        return {"consecutive_failures": 0, "history": []}
    try:
        with open(API_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        consecutive = 0
        for record in reversed(history):
            if not record.get("success", False):
                consecutive += 1
            else:
                break
        return {"consecutive_failures": consecutive, "history": history[-7:]}
    except:
        return {"consecutive_failures": 0, "history": []}

# ==================== 2. 主题分类 ====================
def extract_keywords_topic(title, topn=2):
    """使用jieba TF-IDF提取关键词作为主题"""
    keywords = jieba.analyse.extract_tags(title, topK=topn)
    return '/'.join(keywords) if keywords else title[:8]

# ==================== 3. 情感分析（返回情绪+置信度） ====================
def load_sentiment_model():
    """加载情感分析模型"""
    print("加载情感分析模型...")
    try:
        return pipeline("text-classification",
                        model="IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment",
                        device=-1)
    except:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        return pipeline("text-classification",
                        model="IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment",
                        device=-1)

sentiment_pipe = load_sentiment_model()

def analyze_sentiment_bert(title):
    """情感分析，返回 (情绪标签, 置信度)"""
    result = sentiment_pipe(title[:512])[0]
    label = result['label']
    score = result['score']
    label_map = {
        "LABEL_0": "负面",
        "LABEL_1": "中性",
        "LABEL_2": "正面",
        "Negative": "负面",
        "Neutral": "中性",
        "Positive": "正面"
    }
    sentiment = label_map.get(label, "中性")
    return sentiment, score

# ==================== 4. 风险判断（含置信度过滤） ====================
def risk_level(sentiment, hot, confidence=1.0):
    """风险等级：基于情绪、热度、置信度"""
    # 中性 + 低置信度 → 降低风险权重（避免误报）
    if sentiment == "中性" and confidence < 0.6:
        return "低"
    if sentiment == "负面" and hot > 600000:
        return "高"
    elif sentiment == "负面" and hot > 300000:
        return "中"
    elif sentiment == "中性" and hot > 800000:
        return "中"
    else:
        return "低"

# ==================== 5. 跨天历史追踪（连续负面事件） ====================
HISTORY_FILE = "data/negative_events_history.json"

def load_negative_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_negative_history(history):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def check_consecutive_negative_events(df, days=3):
    """检查同一负面事件是否连续出现多天"""
    history = load_negative_history()
    today = datetime.now().strftime("%Y-%m-%d")
    today_negatives = set(df[df['sentiment'] == '负面']['title'].tolist())
    history[today] = list(today_negatives)
    dates = sorted(history.keys())[-7:]
    history = {d: history[d] for d in dates if d in history}
    alerts = []
    if len(dates) >= days:
        recent_dates = dates[-days:]
        common = set(history[recent_dates[0]])
        for d in recent_dates[1:]:
            common = common & set(history.get(d, []))
        for event in common:
            count = sum(1 for d in recent_dates if event in history.get(d, []))
            if count >= days:
                alerts.append({
                    "title": event,
                    "consecutive_days": count,
                    "dates": [d for d in recent_dates if event in history.get(d, [])]
                })
    save_negative_history(history)
    return alerts, history

# ==================== 6. 日报生成 ====================
def generate_report(df, date_str, api_status, consecutive_alerts, history):
    total = len(df)
    pos = (df['sentiment'] == '正面').sum()
    neu = (df['sentiment'] == '中性').sum()
    neg = (df['sentiment'] == '负面').sum()
    high_risk = (df['risk'] == '高').sum()
    need_review = (df['need_review'] == True).sum() if 'need_review' in df.columns else 0

    topic_stats = df.groupby('topic').agg(
        count=('title', 'count'),
        pos=('sentiment', lambda x: (x == '正面').sum()),
        neu=('sentiment', lambda x: (x == '中性').sum()),
        neg=('sentiment', lambda x: (x == '负面').sum()),
        hot_mean=('hot', 'mean')
    ).reset_index().sort_values('count', ascending=False)

    high_risk_items = df[df['risk'] == '高'].head(10)[
        ['rank', 'title', 'hot', 'sentiment', 'confidence', 'need_review']
    ]

    md = f"""# 微博舆情日报 {date_str}（增强版）

## 一、总体概览
- 共监测到 {total} 个热点话题
- 正面情绪: {pos} 条 | 中性: {neu} 条 | 负面: {neg} 条
- 高风险话题: {high_risk} 个
- 待复核样本: {need_review} 条（置信度 < 0.6）

## 二、系统状态
"""
    if api_status.get("consecutive_failures", 0) >= 3:
        md += "⚠️ **预警**：API已连续失败3次，请检查网络或数据源！建议手动介入。\n\n"
    else:
        md += f"✅ API状态正常（连续失败次数: {api_status.get('consecutive_failures', 0)}）\n\n"

    if consecutive_alerts:
        md += "⚠️ **注意**：以下负面事件已连续多天出现，建议召开专项会议讨论：\n"
        for alert in consecutive_alerts:
            md += f"  - 🔴 {alert['title']}（连续{alert['consecutive_days']}天）\n"
        md += "\n"

    md += "## 三、热点主题分布\n"
    md += "| 主题 | 数量 | 正面 | 中性 | 负面 | 平均热度 |\n"
    md += "|------|------|------|------|------|----------|\n"
    for _, row in topic_stats.iterrows():
        md += f"| {row['topic']} | {row['count']} | {row['pos']} | {row['neu']} | {row['neg']} | {row['hot_mean']:.0f} |\n"

    md += "\n## 四、高风险话题列表（含置信度）\n"
    if len(high_risk_items) > 0:
        md += "| 排名 | 标题 | 热度 | 情绪 | 置信度 | 状态 |\n"
        md += "|------|------|------|------|--------|------|\n"
        for _, row in high_risk_items.iterrows():
            conf = f"{row['confidence']:.2f}" if row['confidence'] else "N/A"
            status = "⚠️待复核" if row['need_review'] else "✅正常"
            md += f"| 第{row['rank']}名 | {row['title']} | {row['hot']} | {row['sentiment']} | {conf} | {status} |\n"
    else:
        md += "暂无高风险话题。\n"

    md += "\n## 五、建议\n"
    suggestions = []
    if high_risk > 5:
        suggestions.append("🔴 高风险话题超过5个，建议立即组织舆情应对会议。")
    if need_review > 3:
        suggestions.append("🟡 待复核样本较多，建议人工逐一核实。")
    if api_status.get("consecutive_failures", 0) >= 3:
        suggestions.append("⚠️ API持续失败，建议检查网络或切换备用数据源。")
    if consecutive_alerts:
        suggestions.append("📌 存在连续多天出现的负面事件，建议重点关注并制定长期应对策略。")
    if not suggestions:
        suggestions.append("✅ 日常监测，暂无紧急事件。")
    for s in suggestions:
        md += f"- {s}\n"

    return md

# ==================== 7. 主流程 ====================
def main():
    print("=" * 60)
    print("开始采集微博热搜（模型增强版）...")
    print("=" * 60)

    # ----- 数据采集（可能抛出异常） -----
    try:
        df = fetch_hotlist()
        print(f"✅ 获取到 {len(df)} 条真实热搜")
    except RuntimeError as e:
        print(e)
        return

    # ----- 后续处理（数据正常） -----
    print("正在提取关键词主题...")
    df['topic'] = df['title'].apply(lambda x: extract_keywords_topic(x, topn=2))

    print("正在进行情感分析...")
    df[['sentiment', 'confidence']] = df['title'].apply(
        lambda x: pd.Series(analyze_sentiment_bert(x))
    )

    df['need_review'] = df['confidence'] < 0.6

    df['risk'] = df.apply(
        lambda row: risk_level(row['sentiment'], row['hot'], row['confidence']),
        axis=1
    )

    api_status = get_api_status()
    consecutive_alerts, history = check_consecutive_negative_events(df)

    date_str = datetime.now().strftime("%Y-%m-%d")
    report = generate_report(df, date_str, api_status, consecutive_alerts, history)

    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{date_str}_model.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    os.makedirs("data/raw", exist_ok=True)
    csv_path = f"reports/data/model/hotlist_{date_str}_model.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 日报已生成: {report_path}")
    print(f"✅ 数据已保存: {csv_path}")

    # 终端输出统计
    print("\n" + "=" * 60)
    print("📊 统计信息")
    print("=" * 60)
    print(f"  总话题数: {len(df)}")
    print(f"  情绪分布: 正面 {df['sentiment'].value_counts().get('正面', 0)} | "
          f"中性 {df['sentiment'].value_counts().get('中性', 0)} | "
          f"负面 {df['sentiment'].value_counts().get('负面', 0)}")
    print(f"  风险分布: 高 {df['risk'].value_counts().get('高', 0)} | "
          f"中 {df['risk'].value_counts().get('中', 0)} | "
          f"低 {df['risk'].value_counts().get('低', 0)}")
    print(f"  待复核: {(df['need_review'] == True).sum()} 条")
    print(f"  API连续失败: {api_status.get('consecutive_failures', 0)} 次")
    if consecutive_alerts:
        print(f"  ⚠️ 连续负面事件: {len(consecutive_alerts)} 个")

    # 人工接管预警
    if api_status.get("consecutive_failures", 0) >= 3:
        print("\n⚠️⚠️⚠️  人工接管预警 ⚠️⚠️⚠️")
        print("   API已连续失败 3 次，系统已无法正常获取数据！")
        print("   建议：请检查网络连接或切换备用数据源。")
    elif (df['risk'] == '高').sum() > 5:
        print("\n⚠️  人工介入提醒")
        print(f"   高风险话题 ({ (df['risk'] == '高').sum() } 个) 超过5个阈值。")
        print("   建议：人工复核高风险话题，确认是否需要启动应急预案。")
    elif consecutive_alerts:
        print("\n📌  专项会议建议")
        print(f"   发现连续多天出现的负面事件: {len(consecutive_alerts)} 个")
        print("   建议：召开专项会议讨论应对策略。")

if __name__ == "__main__":
    main()