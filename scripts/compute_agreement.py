# compute_agreement.py
import pandas as pd
from sklearn.metrics import cohen_kappa_score
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

# 1. 读取 Excel 文件
file_path = "data/to_label标注.xlsx"
df = pd.read_excel(file_path)

# 2. 提取两位标注者的标签（替换为实际列名）
col_a = "true_sentiment_1"
col_b = "true_sentiment_2"

# 3. 删除缺失值
df_clean = df.dropna(subset=[col_a, col_b])
print(f"有效样本数: {len(df_clean)}")

# 4. 计算一致率
agreement = (df_clean[col_a] == df_clean[col_b]).mean()
print(f"原始一致率 (Percentage Agreement): {agreement:.2%}")

# 5. 计算 Cohen's Kappa（需要将标签转为数字）
unique_labels = sorted(set(df_clean[col_a].tolist() + df_clean[col_b].tolist()))
label_map = {label: i for i, label in enumerate(unique_labels)}
y_a = df_clean[col_a].map(label_map)
y_b = df_clean[col_b].map(label_map)
kappa = cohen_kappa_score(y_a, y_b)
print(f"Cohen's Kappa: {kappa:.4f}")

# 6. 输出不一致的案例
diff = df_clean[df_clean[col_a] != df_clean[col_b]][['title', col_a, col_b]]
print(f"\n不一致的标注案例 (共 {len(diff)} 条):")
for _, row in diff.iterrows():
    print(f"  {row['title']} -> A: {row[col_a]}, B: {row[col_b]}")