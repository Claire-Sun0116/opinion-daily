import requests
import pandas as pd
from datetime import datetime
import os
import sys

# 获取当前脚本所在的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 将项目根目录添加到 Python 搜索路径
sys.path.insert(0, PROJECT_ROOT)

# 将当前工作目录强制切换到项目根目录！
os.chdir(PROJECT_ROOT)

print(f"当前工作目录已切换至: {os.getcwd()}")


# ==================== 1. 获取热搜数据 ====================
def fetch_hotlist():
    url = "https://uapis.cn/api/v1/misc/hotboard"
    params = {"type": "weibo"}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    hotlist = data.get('list', [])
    df = pd.DataFrame(hotlist)
    df['rank'] = range(1, len(df) + 1)
    df['title'] = df['title']
    df['hot'] = df['hot_value'].astype(int)
    return df[['rank', 'title', 'hot']]


# ==================== 2. 主题聚类（基于关键词匹配） ====================
topic_keywords = {
    "科技数码": ["手机", "电脑", "芯片", "AI", "智能", "科技", "数码", "小米", "华为", "苹果", "特斯拉", "汽车",
                 "新能源"],
    "娱乐八卦": ["明星", "综艺", "电影", "电视剧", "演唱会", "绯闻", "八卦", "直播", "网红"],
    "社会民生": ["政策", "楼市", "教育", "医疗", "交通", "民生", "社保", "补贴", "疫情", "暴雨", "地震"],
    "财经商业": ["股价", "财报", "融资", "收购", "上市", "经济", "市场", "涨跌", "小米汽车"],
    "安全事件": ["事故", "爆炸", "泄露", "故障", "起火", "受伤", "召回", "违规", "起诉"],
    "文娱": ["综艺", "电影", "音乐", "剧集", "演员", "导演", "颁奖"],
}


def classify_topic(title):
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw in title:
                return topic
    return "其他"


# ==================== 3. 情绪分析（基于情感词典） ====================
positive_words = ["利好", "上涨", "夺冠", "突破", "创新", "满意", "支持", "点赞", "优秀", "成功", "回升", "回暖"]
negative_words = ["故障", "事故", "问题", "争议", "吐槽", "批评", "下跌", "危机", "曝光", "质疑", "怒斥", "痛批",
                  "开除", "处罚"]


def analyze_sentiment(title):
    pos = sum(1 for w in positive_words if w in title)
    neg = sum(1 for w in negative_words if w in title)
    if pos > neg:
        return "正面"
    elif neg > pos:
        return "负面"
    else:
        return "中性"


def risk_level(sentiment, hot):
    if sentiment == "负面" and hot > 500000:
        return "高"
    elif sentiment == "负面" or (sentiment == "中性" and hot > 800000):
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

    # 按主题聚类统计
    topic_stats = df.groupby('topic').agg(
        count=('title', 'count'),
        pos=('sentiment', lambda x: (x == '正面').sum()),
        neu=('sentiment', lambda x: (x == '中性').sum()),
        neg=('sentiment', lambda x: (x == '负面').sum()),
        hot_mean=('hot', 'mean')
    ).reset_index()
    topic_stats = topic_stats.sort_values('count', ascending=False)

    # 高风险条目
    high_risk_items = df[df['risk'] == '高'].head(10)[['rank', 'title', 'hot', 'sentiment']]

    # 生成markdown
    md = f"""# 微博舆情日报 {date_str}

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


# ==================== 主流程 ====================
def main():
    print("开始采集微博热搜...")
    df = fetch_hotlist()
    print(f"获取到 {len(df)} 条热搜")

    df['topic'] = df['title'].apply(classify_topic)
    df['sentiment'] = df['title'].apply(analyze_sentiment)
    df['risk'] = df.apply(lambda row: risk_level(row['sentiment'], row['hot']), axis=1)

    date_str = datetime.now().strftime("%Y-%m-%d")
    report = generate_report(df, date_str)

    # 获取当前脚本所在的绝对路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 获取项目根目录
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # 将项目根目录添加到 Python 搜索路径
    sys.path.insert(0, PROJECT_ROOT)

    # 将当前工作目录强制切换到项目根目录！
    os.chdir(PROJECT_ROOT)
    with open(f"reports/{date_str}_v1.md", "w", encoding="utf-8") as f:
        f.write(report)

    # 保存原始数据
    df.to_csv(f"data/raw/hotlist_{date_str}_v1.csv", index=False, encoding='utf-8-sig')
    print(f"日报已生成: reports/{date_str}_v1.md")
    print(f"数据已保存: data/raw/hotlist_{date_str}_v1.csv")


if __name__ == "__main__":
    main()

