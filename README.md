# 微博舆情日报自动生成系统

> **课程**：大模型与生成式人工智能  
> **选题**：舆情日报自动生成工作流（参考选题5）  
> **版本**：v1.0  
> **提交日期**：2026年6月


## 项目简介

本系统每日自动获取微博热搜榜单，对每条热搜进行**主题分类**、**情感分析**和**风险判断**，最终生成结构化的 Markdown 日报，帮助公关人员快速掌握当日舆情态势。

### 核心功能

| 功能模块 | 说明 |
|----------|------|
| 数据采集 | 通过 UAPIS 公开 API 获取微博实时热搜（50条/天） |
| 主题分类 | 支持关键词匹配（v1/v2）和 jieba TF-IDF 关键词提取（模型版） |
| 情感分析 | 支持情感词典（v1/v2）和 BERT 预训练模型（模型版） |
| 风险判断 | 基于情绪 + 热度阈值综合评估（高/中/低） |
| 日报生成 | Markdown 格式，含概览、主题分布、高风险列表、建议 |
| 自动评估 | 人工标注测试集验证，计算准确率、F1、混淆矩阵 |

### 版本演进

| 版本 | 主题分类 | 情感分析 | 特点 |
|------|----------|----------|------|
| v1 | 5类，15个关键词 | 基础情感词典 | 基线版本，验证流程 |
| v2 | 7类，80+关键词 | 情感词典 + 否定词 | 主题覆盖率提升20% |
| 模型版 | jieba TF-IDF 自动提取 | BERT 预训练模型 | 情感准确率提升14.5% |


## 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10 或更高版本 |
| 虚拟环境 | 建议使用 venv |
| 磁盘空间 | 至少 5 GB（含模型缓存） |


### 运行命令

| 命令 | 说明 |
|------|------|
| `python scripts/pipeline_model.py` | 运行模型版（推荐） |
| `python scripts/pipeline_v2.py` | 运行 v2 版本 |
| `python scripts/pipeline_v1.py` | 运行 v1 版本 |
| `python scripts/evaluate.py` | 运行自动评估 |

### 输出文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 日报 | `reports/YYYY-MM-DD_model.md` | Markdown 格式日报 |
| 数据 | `data/raw/hotlist_YYYY-MM-DD_model.csv` | 完整预测结果 |
| 评估报告 | `reports/evaluation_report.txt` | 准确率等指标 |
| 混淆矩阵 | `reports/confusion_matrix.png` | 可视化评估结果 |


## 项目结构

```
opinion_daily/
├── data/
│   ├── raw/                          # 原始数据
│   │   ├── hotlist_YYYY-MM-DD_v1.csv  # v1热搜数据
│   │   └── hotlist_YYYY-MM-DD_v2.csv  # v2版本数据
│   ├── ground_truth.xlsx      # 人工标注最终结果
│   ├── api_status_history.json      # API调用历史
│   ├── negative_events_history.json  # 负面事件历史
│   └──to_label标注.xlsx      # 两位同学分别标注结果
├── reports/                          # 生成的日报
│   ├── data/                # 模型版数据
│   │   ├──model 
│   │   ├── hotlist_YYYY-MM-DD_model.csv  # model版热搜数据
│   ├── confusion_matrix.png                # 混淆矩阵图像
│   ├── version_comparison.txt               # 对比报告
│   ├── YYYY-MM-DD.md                 # 日报文件
│   └── evaluation_report.txt         # 评估报告
├── data/
│   ├── compute_version.py      # 对比不同版本结果的脚本
│   ├── compute_agreement.py    # 计算两位同学人工标注的一致性脚本
│   ├── evaluate.py                       # 评估脚本
│   ├── pipeline_model.py                 # 模型版脚本
│   ├── pipeline_v1.py                    # v1 版本脚本
│   ├── pipeline_v2.py                    # v2 版本脚本
│   └── sampling.py                    # 对每日热搜进行抽样的脚本
├── requirements.txt                  # 依赖清单
├── # AI使用说明.md                    # AI使用说明
├── # 工作流流程图.png                 # 工作流程图
└── README.md                         # 项目说明
```


## 评估结果摘要

### 版本对比

| 版本 | 准确率 | 加权F1 | 主题覆盖率 | 核心改进 |
|------|--------|--------|------------|----------|
| v1（基础规则） | 32.14% | 0.1564 | 4.00% | 基线版本 |
| v2（扩充主题词） | 32.14% | 0.1564 | 24.00% | 主题覆盖率提升20% |
| 模型版（BERT） | 46.67% | 0.3723 | 100.00% | 情感准确率提升14.5% |

### 模型版分类报告

| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |
|------|--------|--------|--------|--------|
| 中性 | 0.00 | 0.00 | 0.00 | 10 |
| 正面 | 0.50 | 0.60 | 0.55 | 10 |
| 负面 | 0.44 | 0.80 | 0.57 | 10 |

### 混淆矩阵

| 真实\预测 | 中性 | 正面 | 负面 |
|-----------|------|------|------|
| 中性 | 0 | 5 | 5 |
| 正面 | 0 | 6 | 4 |
| 负面 | 0 | 2 | 8 |

### 关键发现

- **负面召回率 80%**：对负面舆情的捕捉能力强，利于风险预警
- **中性全部误判**：模型对无情感倾向的事实陈述存在过度解读
- **模型版整体优于规则版**：准确率提升 14.5 个百分点


## 详细使用说明

### 数据采集

| 项目 | 说明 |
|------|------|
| API 地址 | `https://uapis.cn/api/v1/misc/hotboard` |
| 请求参数 | `{"type": "weibo"}` |
| 返回格式 | JSON，包含 `list` 数组 |
| 采集数量 | 50 条/天 |
| 容错机制 | API 不可用时自动切换备用源或使用模拟数据 |

### 模型下载

首次运行 `pipeline_model.py` 会自动下载两个模型：

| 模型 | 大小 | 用途 |
|------|------|------|
| `paraphrase-multilingual-MiniLM-L12-v2` | ~450MB | 语义向量编码 |
| `IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment` | ~1.3GB | 情感分类 |

**国内加速下载**（设置镜像源）：

在脚本开头添加：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

或命令行设置：

```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts/pipeline_model.py

# Linux/Mac
export HF_ENDPOINT="https://hf-mirror.com"
python scripts/pipeline_model.py
```

### 自定义配置

| 配置项 | 位置 | 说明 |
|--------|------|------|
| 风险阈值 | `pipeline_model.py` 中的 `risk_level()` | 调整高/中/低风险的热度门槛 |
| 主题关键词 | `pipeline_v2.py` 中的 `topic_keywords` | 添加或删除分类关键词 |
| 聚类数量 | `pipeline_model.py` 中的 `n_clusters` | 调整自动聚类的类别数 |


## 注意事项

### 网络依赖

| 依赖项 | 说明 | 离线备用方案 |
|--------|------|-------------|
| UAPIS API | 获取热搜数据 | 内置模拟数据 |
| Hugging Face | 下载模型 | 镜像源或本地缓存 |

### 性能参考

| 环境 | 处理50条耗时 | 内存占用 |
|------|-------------|----------|
| CPU（Intel i5） | 20-30 秒 | ~2.5 GB |
| GPU（NVIDIA） | 3-5 秒 | ~3 GB |

### 系统边界

- 仅分析微博热搜标题，不涵盖正文、评论、图片
- 情感分析为机器判断，仅供参考
- 不预测舆情趋势，仅反映当日热点
- 高风险话题建议人工复核后再决策

### 风险控制

| 触发条件 | 处理方式 |
|----------|----------|
| API 连续失败 3 次 | 提示"暂无法生成日报" |
| 高风险话题 > 5 个 | 触发预警，建议人工介入 |
| 情感置信度 < 0.6 | 标记"待复核" |
| 同一负面事件连续上榜 3 天 | 触发人工关注 |


## 依赖清单

```txt
requests>=2.28.0           # API 请求
pandas>=1.5.0              # 数据处理
scikit-learn>=1.2.0        # 评估指标
sentence-transformers>=2.2.0 # 语义模型
transformers>=4.35.0       # BERT 模型
torch>=2.0.0               # PyTorch 后端
jieba>=0.42.1              # 中文分词
openpyxl>=3.1.0            # Excel 读写
matplotlib>=3.5.0          # 混淆矩阵绘图（可选）
```

安装命令：

```bash
pip install -r requirements.txt
```


## 贡献者

| 姓名 | 角色 | 主要贡献 |
|------|------|----------|
| 曹瑜芳 朱欣悦 | 数据/评估 | 数据采集、人工标注、评估脚本 |
| 崔圆 | 模型/工程 | 模型版开发、日报生成、代码整理 |


## 许可证

本代码仅供课程学习使用，未经许可不得用于商业用途。


## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-06-21 | 完成三个版本开发、自动评估、文档整理 |
