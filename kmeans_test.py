import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter

# --- Matplotlib 显示中文设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    print("已尝试设置 Matplotlib 中文字体 'SimHei'。")
except Exception as e:
    print(f"警告：设置中文字体失败: {e}. 绘图标签可能显示异常。")
# --- 中文字体设置结束 ---

# --- 配置区域 ---
# <<< 修改为你的图片文件夹路径 >>>
folder_path = 'dotdata/'
# <<< (可选) 保存带标记结果的文件夹路径 >>>
output_folder = 'kmeans_detected_results/'
# <<< K-Means 的簇数量 (注意：K=100 可能非常大且慢，建议减小) >>>
K = 100
# <<< 是否保存带有标记的检测结果图像 >>>
SAVE_MARKED_IMAGES = True

print(f"--- 配置 ---")
print(f"输入文件夹: {folder_path}")
print(f"簇数量 (K): {K} {'(警告: K值较高，可能耗时较长)' if K > 20 else ''}")
if SAVE_MARKED_IMAGES:
    print(f"结果图像保存文件夹: {output_folder}")
    os.makedirs(output_folder, exist_ok=True) # 创建输出文件夹
else:
    print("不保存带标记的结果图像。")

# --- 初始化用于存储结果的列表 ---
detected_grayscale_values = []

# --- 查找文件夹中的图片 ---
image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))

if not image_files:
    print(f"错误：在文件夹 {folder_path} 中未找到任何支持的图片文件。")
    exit()
else:
    print(f"\n在文件夹 {folder_path} 中找到 {len(image_files)} 张图片。开始处理...")

# --- 批量处理循环 ---
for img_path in image_files:
    base_filename = os.path.basename(img_path)
    print(f"\n处理图像: {base_filename}")

    # --- 加载图像 ---
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"  错误：无法加载图像 {img_path}")
        continue
    print(f"  图像加载成功。形状: {img_gray.shape}")

    # --- 准备数据 ---
    pixel_values = img_gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    # print(f"  数据准备完毕。形状: {pixel_values.shape}") # 通常不需要打印这个

    # --- 执行聚类 ---
    print(f"  正在使用 K={K} 执行 K-Means...")
    try:
        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10) # n_init=10 运行10次选最优
        kmeans.fit(pixel_values)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        centers_gray = np.uint8(centers)
        print(f"  聚类完成。")
    except Exception as e:
        print(f"  错误：K-Means 聚类失败。原因: {e}")
        continue # 跳过此图像

    # --- 识别候选白点簇 (基于最高平均灰度) ---
    if len(centers_gray) == 0:
        print("  错误：聚类中心为空，无法识别白点簇。")
        continue

    white_candidate_label = np.argmax(centers_gray)
    white_candidate_avg_gray = centers_gray[white_candidate_label][0]
    print(f"  识别到最亮簇标签: {white_candidate_label} (平均灰度: {white_candidate_avg_gray})")

    # --- 确定位置 (质心) ---
    white_point_location = None
    try:
        label_image = labels.reshape(img_gray.shape)
        white_mask = np.uint8(label_image == white_candidate_label) * 255

        contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                white_point_location = (cX, cY)
                print(f"  计算得到质心位置: ({cX}, {cY})")
            else:
                print("  警告：最亮簇轮廓面积为零，无法计算质心。")
        else:
            print("  警告：未能为最亮簇掩码找到轮廓。")
    except Exception as e:
        print(f"  错误：计算质心时出错。原因: {e}")
        # 即使计算质心出错，我们仍然可以使用平均灰度值作为备选
        pass # 继续尝试获取灰度值

    # --- 获取灰度值 ---
    target_grayscale_value = -1 # 默认值
    target_source = "N/A"      # 记录值来源

    if white_point_location:
        loc_x, loc_y = white_point_location
        if 0 <= loc_y < img_gray.shape[0] and 0 <= loc_x < img_gray.shape[1]:
            target_grayscale_value = img_gray[loc_y, loc_x]
            target_source = f"质心({loc_x},{loc_y})"
            print(f"  >>> 获取到质心处灰度值: {target_grayscale_value}")
            detected_grayscale_values.append(target_grayscale_value)

            # --- (可选) 保存带标记的图像 ---
            if SAVE_MARKED_IMAGES:
                try:
                    img_display = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                    cv2.circle(img_display, white_point_location, 5, (0, 255, 0), -1) # 绿色点
                    cv2.putText(img_display, f"Val:{target_grayscale_value}", (loc_x + 10, loc_y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    save_path = os.path.join(output_folder, base_filename)
                    cv2.imwrite(save_path, img_display)
                except Exception as e_save:
                    print(f"  错误：无法保存标记图像。原因: {e_save}")
            # --- 保存结束 ---

        else:
            print(f"  警告：计算出的质心 ({loc_x}, {loc_y}) 超出图像边界。将使用簇平均灰度值。")
            target_grayscale_value = white_candidate_avg_gray
            target_source = f"簇平均(标签{white_candidate_label})"
            detected_grayscale_values.append(target_grayscale_value)
            print(f"  >>> 使用簇平均灰度值: {target_grayscale_value}")
    else:
        # 如果无法计算质心，使用簇的平均灰度值作为备选
        print("  警告：无法确定精确位置，将使用簇平均灰度值。")
        target_grayscale_value = white_candidate_avg_gray
        target_source = f"簇平均(标签{white_candidate_label})"
        detected_grayscale_values.append(target_grayscale_value)
        print(f"  >>> 使用簇平均灰度值: {target_grayscale_value}")

print("\n--- 所有图像处理完毕 ---")

# --- 统计分析 ---
if not detected_grayscale_values:
    print("\n未能检测到任何目标点的灰度值，无法进行统计。")
else:
    num_detections = len(detected_grayscale_values)
    print(f"\n--- 灰度值统计分析 (基于 {num_detections} 个成功获取的值) ---")

    # 1. 计算每个灰度值出现的次数
    gray_counts = Counter(detected_grayscale_values)
    sorted_gray_counts = sorted(gray_counts.items()) # 按灰度值排序

    print("\n灰度值频次统计 (灰度值: 次数):")
    for value, count in sorted_gray_counts:
        print(f"  {value}: {count}")

    # 2. 计算每个灰度值的占比
    print("\n灰度值占比 (%):")
    gray_proportions = {value: (count / num_detections) * 100 for value, count in sorted_gray_counts}
    for value, proportion in gray_proportions.items():
        print(f"  {value}: {proportion:.2f}%")

    # 3. 绘制灰度值分布直方图
    try:
        plt.figure(figsize=(12, 7))
        n, bins, patches = plt.hist(detected_grayscale_values, bins=range(0, 257), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
        plt.title("检测到的目标点灰度值分布 (基于K-Means)")
        plt.xlabel("灰度值")
        plt.ylabel("频次 (次数)")

        if detected_grayscale_values:
             min_val = min(detected_grayscale_values)
             max_val = max(detected_grayscale_values)
             plt.xlim(max(0, min_val - 10), min(255, max_val + 10))

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(np.arange(0, 256, step=16))
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\n错误：无法绘制直方图。原因: {e}")

print("\n--- 脚本执行结束 ---")