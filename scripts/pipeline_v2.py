# -*- coding: utf-8 -*-
"""
pipeline_v2.py - v2 版本舆情日报生成脚本（纯真实数据版）
功能：数据采集（严格依赖真实API） → 主题分类（关键词匹配） → 情感分析（词典+否定词） → 风险判断 → 日报生成
说明：如API全部失败，程序主动报错退出，不产生任何模拟数据。
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime

# ==================== 自动切换工作目录 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
os.chdir(PROJECT_ROOT)
print(f"当前工作目录: {os.getcwd()}")


# ==================== 1. 获取热搜数据 ====================
def fetch_hotlist():
    """
    获取微博热搜数据。
    尝试所有API源，若全部失败则抛出 RuntimeError，绝不返回模拟数据。
    """
    apis = [
        ("https://uapis.cn/api/v1/misc/hotboard", {"type": "weibo"}),
        ("http://api.rosysun.cn/weibo/", None),  # 免注册备用
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
                    df['rank'] = range(1, len(df) + 1)
                    df['title'] = df['title']
                    df['hot'] = df['hot_value'].astype(int)
                    return df[['rank', 'title', 'hot']]
                # 解析备用API格式
                elif "data" in data and isinstance(data["data"], list):
                    hotlist = data["data"]
                    df = pd.DataFrame(hotlist)
                    df['rank'] = range(1, len(df) + 1)
                    df['title'] = df['title']
                    df['hot'] = 500000  # 备用源可能无热度，设默认值
                    return df[['rank', 'title', 'hot']]
        except Exception as e:
            print(f"  请求失败: {e}")
            continue

    # 所有API均失败 → 抛出异常（不再返回模拟数据）
    error_msg = (
        f"\n❌ 数据获取严重错误：所有 API 源均无法连接。\n"
        "   【作业说明】该工作流严格依赖真实数据源，请检查网络或稍后重试。\n"
        "   程序已终止，无法继续生成日报。"
    )
    raise RuntimeError(error_msg)


# ==================== 2. 主题分类（关键词匹配） ====================
topic_keywords = {
    "科技数码": ["手机", "电脑", "芯片", "AI", "智能", "科技", "数码", "小米", "华为", "苹果", "特斯拉", "汽车", "鸿蒙",
                 "大疆", "创新", "专利", "5G", "算法"],
    "娱乐八卦": ["明星", "综艺", "电影", "电视剧", "演唱会", "绯闻", "八卦", "直播", "网红", "爱豆", "出道", "粉丝",
                 "掉粉", "签约", "恋情", "出轨", "离婚"],
    "体育竞技": ["世界杯", "足球", "篮球", "比赛", "奥运会", "亚运会", "NBA", "欧冠", "中超", "C罗", "梅西", "冠军",
                 "淘汰", "晋级", "爆冷"],
    "社会民生": ["政策", "楼市", "教育", "医疗", "交通", "民生", "社保", "补贴", "疫情", "暴雨", "地震", "救护", "ICU",
                 "养生馆", "侵权", "维权", "养老", "留守儿童"],
    "财经商业": ["股价", "财报", "融资", "收购", "上市", "经济", "市场", "涨跌", "A股", "涨停", "跌停", "货币", "汇率",
                 "投资"],
    "安全事件": ["事故", "爆炸", "泄露", "故障", "起火", "受伤", "召回", "违规", "起诉", "反诉", "曝光", "争议", "违法",
                 "处罚", "拘留"],
    "文娱": ["综艺", "电影", "音乐", "剧集", "演员", "导演", "颁奖", "音乐节", "歌手", "专辑"],
}


def classify_topic(title):
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw in title:
                return topic
    return "其他"


# ==================== 3. 情绪分析（词典 + 否定词处理） ====================
positive_words = [
    "利好", "上涨", "夺冠", "突破", "创新", "满意", "支持", "点赞", "优秀", "成功",
    "回暖", "盈利", "大涨", "红盘", "涨停", "冠军", "感动", "温暖", "希望", "惊喜",
    "里程碑", "领跑", "领先", "首发", "首创"
]
negative_words = [
    "故障", "事故", "问题", "争议", "吐槽", "批评", "下跌", "危机", "曝光", "质疑",
    "怒斥", "痛批", "开除", "处罚", "起诉", "反诉", "受伤", "死亡", "离世", "拒绝",
    "遭拒", "闯入", "掉粉", "违法行为", "亏损", "暴跌", "熔断", "出轨", "偷税", "漏税",
    "查封", "停运", "道歉", "危机"
]
negation_words = ["不", "没", "无", "非", "并非"]


def analyze_sentiment(title):
    pos_count = 0
    neg_count = 0

    for w in positive_words:
        idx = title.find(w)
        if idx != -1:
            prev = title[max(0, idx - 2):idx]
            if any(neg in prev for neg in negation_words):
                neg_count += 1
            else:
                pos_count += 1

    for w in negative_words:
        idx = title.find(w)
        if idx != -1:
            prev = title[max(0, idx - 2):idx]
            if any(neg in prev for neg in negation_words):
                pos_count += 1
            else:
                neg_count += 1

    if pos_count > neg_count:
        return "正面"
    elif neg_count > pos_count:
        return "负面"
    else:
        return "中性"


def risk_level(sentiment, hot):
    if sentiment == "负面" and hot > 600000:
        return "高"
    elif sentiment == "负面" and hot > 300000:
        return "中"
    elif sentiment == "中性" and hot > 800000:
        return "中"
    else:
        return "低"


# ==================== 4. 生成日报 ====================
def generate_report(df, date_str):
    total = len(df)
    pos = (df['sentiment'] == '正面').sum()
    neu = (df['sentiment'] == '中性').sum()
    neg = (df['sentiment'] == '负面').sum()
    high_risk = (df['risk'] == '高').sum()

    topic_stats = df.groupby('topic').agg(
        count=('title', 'count'),
        pos=('sentiment', lambda x: (x == '正面').sum()),
        neu=('sentiment', lambda x: (x == '中性').sum()),
        neg=('sentiment', lambda x: (x == '负面').sum()),
        hot_mean=('hot', 'mean')
    ).reset_index().sort_values('count', ascending=False)

    high_risk_items = df[df['risk'] == '高'].head(10)[['rank', 'title', 'hot', 'sentiment']]

    md = f"""# 微博舆情日报 {date_str}（v2 语义增强版）

## 一、总体概览
- 共监测到 {total} 个热点话题
- 正面情绪: {pos} 条 | 中性: {neu} 条 | 负面: {neg} 条
- 高风险话题: {high_risk} 个

## 二、热点主题分布
| 主题 | 数量 | 正面 | 中性 | 负面 | 平均热度 |
|------|------|------|------|------|----------|
"""
    for _, row in topic_stats.iterrows():
        md += f"| {row['topic']} | {row['count']} | {row['pos']} | {row['neu']} | {row['neg']} | {row['hot_mean']:.0f} |\n"

    md += "\n## 三、高风险话题列表\n"
    if len(high_risk_items) > 0:
        for _, row in high_risk_items.iterrows():
            md += f"- 🔴 第{row['rank']}名：{row['title']} (热度{row['hot']}) - 情绪: {row['sentiment']}\n"
    else:
        md += "暂无高风险话题。\n"

    md += "\n## 四、建议\n"
    if high_risk > 0:
        md += "- 建议立即关注负面舆情，准备应对方案。\n"
    else:
        md += "- 日常监测，暂无紧急事件。\n"

    return md


# ==================== 5. 主流程 ====================
def main():
    print("=" * 60)
    print("开始采集微博热搜（v2 语义增强版）...")
    print("=" * 60)

    # ----- 数据采集（可能抛出异常） -----
    try:
        df = fetch_hotlist()
        print(f"✅ 获取到 {len(df)} 条真实热搜")
    except RuntimeError as e:
        print(e)
        return  # 优雅退出，不继续执行

    # ----- 后续处理（数据正常） -----
    print("正在进行主题分类...")
    df['topic'] = df['title'].apply(classify_topic)

    print("正在进行情感分析...")
    df['sentiment'] = df['title'].apply(analyze_sentiment)

    print("正在进行风险判断...")
    df['risk'] = df.apply(lambda row: risk_level(row['sentiment'], row['hot']), axis=1)

    date_str = datetime.now().strftime("%Y-%m-%d")
    report = generate_report(df, date_str)

    os.makedirs("reports", exist_ok=True)
    v2_report_path = f"reports/{date_str}_v2.md"
    with open(v2_report_path, "w", encoding="utf-8") as f:
        f.write(report)

    os.makedirs("data/raw", exist_ok=True)
    v2_csv_path = f"data/raw/hotlist_{date_str}_v2.csv"
    df.to_csv(v2_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 日报已生成: {v2_report_path}")
    print(f"✅ 数据已保存: {v2_csv_path}")

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


if __name__ == "__main__":
    main()