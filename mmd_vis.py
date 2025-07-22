import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 数据
classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car',  'motorcycle', 'bicycle'
]#'truck', 'bus','train',

# MIC 和 Ours 的 MMD 距离
mic_mmd = [
    0.0764, 0.0415, 0.0214, 0.1134, 0.0673, 0.1180,
    0.4393, 0.1413, 0.0263, 0.1226, 0.0421, 0.2019,
    0.4976, 0.0799, 0.2674, 0.2050
]

ours_mmd = [
    0.0205, 0.0384, 0.0202, 0.0894, 0.0636, 0.1153,
    0.4364, 0.1375, 0.0246, 0.1044, 0.0400, 0.1991,
    0.4899, 0.0793, 0.2661, 0.2929
]

# 将数据转换为 NumPy 数组
mic_mmd = np.array(mic_mmd)
ours_mmd = np.array(ours_mmd)

# 选择颜色调色板
sns.set(style="whitegrid")
palette = sns.color_palette("Paired", 2)  # Using Set2 palette for better distinction
mic_color, ours_color = palette

# 创建一个新的图形，调整尺寸使其更紧凑
plt.figure(figsize=(9, 6))
# 绘制堆叠柱状图
mic_bars = plt.bar(classes, mic_mmd, width=0.9, label='MIC', color='none', edgecolor='#a5a5a5', hatch='////', linewidth=1)
ours_bars = plt.bar(classes, ours_mmd, width=0.9, label='Ours', color='none', edgecolor=ours_color, hatch='////', linewidth=1)
# 添加标签和标题，增大字体大小f4ba19
plt.ylabel('MMD Distance', fontsize=12, fontweight='bold')
plt.xlabel('Classes', fontsize=12, fontweight='bold')
# plt.title('Comparison of MMD Distances: MIC vs Ours', fontsize=18, weight='bold')
# 调整 x 轴刻度标签
plt.xticks(rotation=45, ha='right', fontsize=12)
# 添加图例，增大字体大小
plt.legend(title='Methods', fontsize=13, title_fontsize=14, loc='upper right')
# 添加数值标签在每个堆叠柱子的顶部

for i in range(len(classes)):
    plt.annotate(f'{mic_mmd[i] - ours_mmd[i]:.4f}',
                 xy=(i, mic_mmd[i]),
                 xytext=(0, 5),  # 5 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
# 优化 y 轴
plt.yticks(fontsize=13)
plt.ylim(0, max(mic_mmd*1.05))  # Slightly increased to accommodate labels
# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.5)
# 去除上边和右边的脊
sns.despine()
# 调整布局以避免标签被截断
plt.tight_layout()
# 保存图像
plt.savefig('mmd.png', dpi=500)

# 显示图像
plt.show()
