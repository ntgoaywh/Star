import cv2
import numpy as np
import os  # 用于文件和目录操作
import glob  # 用于查找文件路径
import matplotlib.pyplot as plt
from collections import Counter  # 用于统计频率

# --- Matplotlib 显示中文设置 (如果绘图需要显示中文) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"无法设置中文字体: {e}. 绘图标签可能显示异常。")

# --- 配置 SimpleBlobDetector ---
print("配置 Blob 检测器...")
params = cv2.SimpleBlobDetector_Params()


# 阈值设置 (可以保持默认或微调，默认是注释，现在已经启用)
params.minThreshold = 10
params.maxThreshold = 220
params.thresholdStep = 10

# 按颜色过滤 (检测亮色斑点)
params.filterByColor = True
params.blobColor = 255  # 255 代表亮色, 0 代表暗色

# 按面积过滤 (需要根据目标大小调整)
params.filterByArea = True
params.minArea = 15      # 示例值：目标最小像素面积 (过滤小噪声点)
params.maxArea = 40  # 示例值：目标最大像素面积 (过滤大片亮区)

# 按圆度过滤 (形状，1 最圆) (可能需要调整)
params.filterByCircularity = True
params.minCircularity = 0.5 # 示例值：最小圆度 (0到1)
# params.maxCircularity = 1.0 # 通常不需要上限

# 按凸性过滤 (面积/凸包面积，1 最凸)
params.filterByConvexity = False # 示例：通常可以先关闭
# params.minConvexity = 0.8

# 按惯性比过滤 (衡量形状伸长程度，圆形为1，直线为0) (可能需要调整)
params.filterByInertia = True
params.minInertiaRatio = 0.8 # 示例值：最小惯性比 (过滤细长物体)
# params.maxInertiaRatio = 1.0

# 创建检测器实例
try:
    # 对于较新版本的 OpenCV (>= 4.5.x ?)
    detector = cv2.SimpleBlobDetector_create(params)
except AttributeError:
    # 对于较旧版本的 OpenCV
    detector = cv2.SimpleBlobDetector(params)

print("Blob 检测器配置完成。请务必根据实际目标调整上述参数！")
print(f"  - 最小面积: {params.minArea}, 最大面积: {params.maxArea}")
print(f"  - 最小圆度: {params.minCircularity}")
print(f"  - 最小惯性比: {params.minInertiaRatio}")


# --- 批量处理设置 ---
# <<< 修改为你的图片文件夹路径 >>>
folder_path = 'dotdata/'
# <<< 保存带标记结果的文件夹路径 >>>
output_folder = 'detected_results/'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
print(f"检测结果将保存到文件夹: {output_folder}")

# 存储所有成功检测到的目标点的灰度值
detected_grayscale_values = []

# 查找文件夹中所有的图片文件 (支持多种格式)
image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))

if not image_files:
    print(f"错误：在文件夹 {folder_path} 中未找到任何支持的图片文件。请检查路径和文件扩展名。")
    exit()
else:
    print(f"在文件夹 {folder_path} 中找到 {len(image_files)} 张图片。")

# --- 开始批量处理循环 ---
for img_path in image_files:
    base_filename = os.path.basename(img_path)
    print(f"\n处理图像: {base_filename}")

    # 1. 加载灰度图像
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"  错误：无法加载图像 {img_path}")
        continue

    # 2. 决定用于检测的图像
    image_to_detect = img_gray

    # 3. 使用 Blob 检测器检测
    keypoints = detector.detect(image_to_detect) # keypoints 是一个元组

    # 4. 处理检测结果 (处理所有检测到的 keypoints)
    num_targets_in_image = 0 # 记录在这张图中找到的目标数量

    # (可选) 如果需要保存带标记的图像，先准备好 img_display
    img_display = None
    if keypoints and True: # 假设总是想为找到目标的图准备显示 (或者根据你的逻辑决定)
        try:
            img_display = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"  错误：转换为BGR时出错 - {e}")
            img_display = None # 转换失败则不进行绘制

    if keypoints: # 检查是否检测到任何关键点
        print(f"  检测到 {len(keypoints)} 个符合条件的 Blob。")

        # --- 遍历所有检测到的 keypoints ---
        for i, target_keypoint in enumerate(keypoints): # 你也可以先排序再遍历
            # 获取目标位置 (中心坐标是浮点数，需要转整数)
            try:
                loc_x = int(target_keypoint.pt[0])
                loc_y = int(target_keypoint.pt[1])
            except Exception as e:
                print(f"  错误：获取第 {i+1} 个关键点坐标时出错 - {e}")
                continue # 跳过这个损坏的关键点

            # 5. 获取该位置的灰度值 (在原始灰度图上获取)
            # 检查坐标是否在图像边界内
            if 0 <= loc_y < img_gray.shape[0] and 0 <= loc_x < img_gray.shape[1]:
                grayscale_value = img_gray[loc_y, loc_x]
                detected_grayscale_values.append(grayscale_value) # 存储灰度值
                num_targets_in_image += 1
                print(f"    目标 {num_targets_in_image}: 位置 ({loc_x}, {loc_y}), 灰度值: {grayscale_value}")

                # 6. 如果准备了 img_display，则绘制标记
                if img_display is not None:
                    try:
                        # 在检测到的位置绘制一个绿色的圆点
                        cv2.circle(img_display, (loc_x, loc_y), 5, (0, 255, 0), -1) # BGR: 绿色
                        # (可选) 在旁边添加灰度值文本或目标编号
                        cv2.putText(img_display, f"Val:{grayscale_value}", (loc_x + 10, loc_y + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        # cv2.putText(img_display, f"#{num_targets_in_image}", (loc_x - 15, loc_y + 5),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1) # 红色编号
                    except Exception as e:
                        print(f"  错误：为目标 {num_targets_in_image} 绘制标记时出错。原因: {e}")
            else:
                print(f"  警告：目标 {i+1} 的坐标 ({loc_x}, {loc_y}) 超出图像边界。")
        # --- 所有 keypoints 处理完毕 ---

        # 如果有检测到目标并且 img_display 已准备好，则保存图像
        if num_targets_in_image > 0 and img_display is not None:
            try:
                save_path = os.path.join(output_folder, base_filename)
                cv2.imwrite(save_path, img_display)
                # print(f"  结果已保存到: {save_path}")
            except Exception as e:
                print(f"  错误：无法保存带标记的图像到 {save_path}。原因: {e}")

    if num_targets_in_image == 0:
        print("  未在此图像中检测到符合条件的目标。")

# --- 所有图像处理完毕 ---
print("\n--- 所有图像处理完毕 ---")

# --- 统计分析 ---
if not detected_grayscale_values:
    print("\n未能检测到任何目标点的灰度值，无法进行统计。")
else:
    num_detections = len(detected_grayscale_values)
    print(f"\n--- 灰度值统计分析 (基于 {num_detections} 个检测到的目标) ---")

    # 1. 计算每个灰度值出现的次数
    gray_counts = Counter(detected_grayscale_values)
    # 按灰度值排序，方便查看
    sorted_gray_counts = sorted(gray_counts.items())

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
        # bins=range(0, 257) 确保每个整数灰度值一个bin，align='left'使标签对齐
        n, bins, patches = plt.hist(detected_grayscale_values, bins=range(0, 257), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
        plt.title("检测到的目标点灰度值分布")
        plt.xlabel("灰度值")
        plt.ylabel("频次 (次数)")

        # 动态调整X轴范围以便更好地显示
        if detected_grayscale_values:
             min_val = min(detected_grayscale_values)
             max_val = max(detected_grayscale_values)
             plt.xlim(max(0, min_val - 10), min(255, max_val + 10)) # 留出一些边距

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(np.arange(0, 256, step=16)) # 每隔16显示一个刻度标签
        plt.tight_layout() # 调整布局防止标签重叠
        plt.show()
    except Exception as e:
        print(f"\n错误：无法绘制直方图。原因: {e}")

print("\n--- 脚本执行结束 ---")