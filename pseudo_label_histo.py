import re

import math
import matplotlib.pyplot as plt

# 初始化数组存储 refine 值
refine_values = []

# 打开日志文件并提取包含 "total refine" 的行
try:
    with open('/data/wuqihang/MIC2/refine_labels_with_weight_gta.log', 'r') as file:
        for line in file:
            match = re.search(r'total refine (-?\d+)', line.strip())
            if match:
                refine_values.append((int(match.group(1))))
except FileNotFoundError:
    print("Error: log.txt not found.")
    exit()
except ValueError:
    print("Error: Invalid number format in log.txt.")
    exit()

# 检查是否提取到数据
if not refine_values:
    print("Error: No 'total refine' values found in log.txt.")
    exit()

# 取绝对值并转换为以千（k）为单位（纵坐标）
refine_values_abs_k = [0.0, 67.017, 1746.721, 3188.093, 1873.137, 1507.992, 611.884, 1029.372, 1224.694, 1069.378, 881.394, 646.514, 931.332, 961.881, 649.848, 505.723, 583.364, 519.154, 498.616, 454.173, 379.496, 364.873, 150.478, 453.418, 287.762, 113.465, 232.235, 237.182, 152.816, 261.845, 120.174, 30.394, 64.316, 149.76, 65.604, 80.635, 44.426, 62.366, 46.909, 14.058,
 -4.347, 33.823, -24.528, -10.575, 65.346, 73.231, -14.658, 17.111, -10.272, -29.984, 69.818, 5.011, 8.389, 1.132, 27.867, -4.466, -3.286, -43.951, -8.093, -31.232, 19.76, -66.251, 0.339, -29.801, 8.595, -2.054, 7.6, 41.542, -29.293, -10.406, -14.96, 3.412, -5.055, 1.727, -12.729, 7.987, 16.728, 1.456, -8.49, 11.457, -2.769, 0.62, -1.892, -11.674, 7.555, 1.534, 0.579, -12.735, 7.353, -18.193, 4.972, -5.653, -5.149, 2.444, -2.808, 6.289, -0.784, -19.781, -13.637, 5.519]

# 设置横坐标范围（100, 200, 300, ...），基于迭代次数（每 100 次）
iteration_points = range(100, (100 + 1) * 100, 100)

# 创建柱状图，横坐标为迭代次数，纵坐标为千单位值
plt.figure(figsize=(10, 6))
plt.bar(iteration_points, refine_values_abs_k, width=100, edgecolor='black', color='skyblue', align='edge')
# 自定义横坐标标签，只显示 1000, 2000, ... 并用 1k, 2k 表示
tick_positions = iteration_points[::10]  # 每 1000 次迭代显示一个点（1000, 2000, ...）
tick_labels = [f"{int(x // 1000)}k" for x in tick_positions]
plt.xticks(tick_positions, tick_labels, rotation=45)  # 旋转 45 度以避免重叠
plt.title('SYNTHIA to CITYSCAPES')
plt.xlabel('Iteration (every 100 iterations)')
plt.ylabel('Total Refine Value (k)')
# plt.grid(True, alpha=0.3)

# 添加描述性统计信息
mean_value = sum(refine_values_abs_k) / len(refine_values_abs_k)
min_value = min(refine_values_abs_k)
max_value = max(refine_values_abs_k)
plt.text(0.95, 0.95, f'Mean: {mean_value:.2f}k\nMin: {min_value:.2f}k\nMax: {max_value:.2f}k',
         transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')

# 设置横坐标刻度
plt.xticks(iteration_points)

# 显示和保存直方图
plt.savefig('SYNTHIA_refine_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印提取的 refine 值（可选）
print("Extracted refine values (in thousands, absolute):")
for i, (iter_num, value) in enumerate(zip(iteration_points, refine_values_abs_k), 1):
    print(f"Iteration {iter_num}: {value:.2f}k")