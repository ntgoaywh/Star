import cv2
import numpy as np
import math

def calculate_snr_simple(image_path):
    """
    计算整个图像的信噪比 (简单方法).

    Args:
        image_path (str): 图像文件的路径.

    Returns:
        tuple: (snr_ratio, snr_db) 包含 SNR 比率和分贝值，
               如果出错则返回 (None, None).
    """
    try:
        # 以灰度模式读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"错误: 无法读取图像文件 {image_path}")
            return None, None

        # 计算平均值 (信号)
        mean = np.mean(image)

        # 计算标准差 (噪声)
        std_dev = np.std(image)

        # 避免除以零
        if std_dev == 0:
            snr_ratio = float('inf')  # 或者可以定义为 0 或其他特定值
            snr_db = float('inf')
        else:
            snr_ratio = mean / std_dev
            # 计算 dB 值 (确保 snr_ratio > 0)
            if snr_ratio <= 0:
                 snr_db = -float('inf') # 或者根据需要处理
            else:
                 snr_db = 20 * math.log10(snr_ratio)

        return snr_ratio, snr_db

    except FileNotFoundError:
        print(f"错误: 文件未找到 {image_path}")
        return None, None
    except Exception as e:
        print(f"计算 SNR 时发生错误: {e}")
        return None, None

def calculate_snr_roi(image_path, signal_roi, noise_roi):
    """
    使用指定的信号和噪声区域计算信噪比 (基于区域的方法).

    Args:
        image_path (str): 图像文件的路径.
        signal_roi (tuple): 定义信号区域的 (x, y, w, h) 元组.
                            (x, y) 是左上角坐标, w 是宽度, h 是高度.
        noise_roi (tuple): 定义噪声区域的 (x, y, w, h) 元组.

    Returns:
        tuple: (snr_ratio, snr_db, mean_signal, std_dev_noise)
               包含 SNR 比率、分贝值、信号区域均值和噪声区域标准差.
               如果出错则返回 (None, None, None, None).
    """
    try:
        # 以灰度模式读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"错误: 无法读取图像文件 {image_path}")
            return None, None, None, None

        # 提取 ROI (注意 OpenCV 索引是 [y:y+h, x:x+w])
        sx, sy, sw, sh = signal_roi
        nx, ny, nw, nh = noise_roi

        signal_patch = image[sy:sy+sh, sx:sx+sw]
        noise_patch = image[ny:ny+nh, nx:nx+nw]

        # 检查 ROI 是否有效
        if signal_patch.size == 0:
            print(f"错误: 信号 ROI 无效或大小为零 (区域: {signal_roi})")
            return None, None, None, None
        if noise_patch.size == 0:
            print(f"错误: 噪声 ROI 无效或大小为零 (区域: {noise_roi})")
            return None, None, None, None

        # 计算信号区域的平均值
        mean_signal = np.mean(signal_patch)

        # 计算噪声区域的标准差
        std_dev_noise = np.std(noise_patch)

        # 避免除以零
        if std_dev_noise == 0:
            snr_ratio = float('inf') if mean_signal > 0 else 0
            snr_db = float('inf') if mean_signal > 0 else -float('inf')
        else:
            snr_ratio = mean_signal / std_dev_noise
             # 计算 dB 值 (确保 snr_ratio > 0)
            if snr_ratio <= 0:
                 snr_db = -float('inf') # 或者根据需要处理
            else:
                 snr_db = 20 * math.log10(snr_ratio)

        return snr_ratio, snr_db, mean_signal, std_dev_noise

    except FileNotFoundError:
        print(f"错误: 文件未找到 {image_path}")
        return None, None, None, None
    except IndexError:
         print("错误: ROI 坐标超出图像边界。请检查 ROI 定义。")
         return None, None, None, None
    except Exception as e:
        print(f"计算 ROI SNR 时发生错误: {e}")
        return None, None, None, None

# --- 使用示例 ---

# 1. 准备工作
#    - 确保已安装 OpenCV 和 NumPy:
#      pip install opencv-python numpy
#    - 将 'path/to/your/image.jpg' 替换为你的图像文件路径.

image_file = '/Users/limttkx/MATLAB/多帧多目标/dotdata/07.bmp' # <--- 修改这里

# 2. 使用简单方法计算 SNR
print("--- 方法 1: 简单全图 SNR ---")
snr_simple_ratio, snr_simple_db = calculate_snr_simple(image_file)

if snr_simple_ratio is not None:
    # 加载图像以获取均值和标准差用于显示
    img_gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if img_gray is not None:
       print(f"图像: {image_file}")
       print(f"全图平均值 (信号估计): {np.mean(img_gray):.4f}")
       print(f"全图标准差 (噪声估计): {np.std(img_gray):.4f}")
       print(f"SNR (比率 Mean/Std): {snr_simple_ratio:.4f}")
       print(f"SNR (dB): {snr_simple_db:.4f} dB")
    else:
       print(f"无法重新加载图像 {image_file} 以显示统计数据。")
else:
    print("简单 SNR 计算失败。")


print("\n--- 方法 2: 基于区域的 SNR ---")
# 3. 使用基于区域的方法计算 SNR
#    - !!! 关键: 你需要根据你的图像内容手动确定合适的区域 !!!
#    - 下面的坐标只是示例，你需要调整它们！
#    - (x, y, width, height)
#    - 例如：信号区是一个 50x50 像素的方块，左上角在 (100, 150)
#    - 例如：噪声区是一个 40x40 像素的方块，左上角在 (10, 10) （假设那里是均匀背景）

# --- !!! 修改下面的 ROI 坐标 !!! ---
signal_region = (100, 150, 50, 50)  # (x, y, w, h) - 信号区域示例
noise_region  = (10, 10, 40, 40)    # (x, y, w, h) - 噪声区域示例
# --- !!! 修改上面的 ROI 坐标 !!! ---

snr_roi_ratio, snr_roi_db, roi_mean, roi_std = calculate_snr_roi(image_file, signal_region, noise_region)

if snr_roi_ratio is not None:
    print(f"图像: {image_file}")
    print(f"信号 ROI (x,y,w,h): {signal_region}")
    print(f"噪声 ROI (x,y,w,h): {noise_region}")
    print(f"信号区域平均值: {roi_mean:.4f}")
    print(f"噪声区域标准差: {roi_std:.4f}")
    print(f"SNR (比率 Mean_Signal/Std_Noise): {snr_roi_ratio:.4f}")
    print(f"SNR (dB): {snr_roi_db:.4f} dB")
else:
    print("基于 ROI 的 SNR 计算失败。请检查 ROI 定义和图像路径。")