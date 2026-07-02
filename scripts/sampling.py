import pandas as pd
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
df = pd.read_csv("reports/data/model/hotlist_2026-06-22_model.csv")
sample = df.sample(n=30, random_state=42)  # 固定随机种子，保证可复现
sample[['title']].to_csv("data/to_label.csv", index=False)