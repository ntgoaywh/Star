import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Matplotlib 显示中文设置 ---
# 尝试设置支持中文的字体，请根据你的操作系统和已安装字体调整
try:
    # Windows: 'SimHei', 'Microsoft YaHei'
    # macOS: 'PingFang SC', 'STHeiti', 'Heiti TC', 'SimHei' (如安装)
    # Linux: 'WenQuanYi Micro Hei', 'Noto Sans CJK SC'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei (黑体)
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    print("已尝试设置 Matplotlib 中文字体 'SimHei'。")
except Exception as e:
    print(f"警告：设置中文字体失败: {e}")
    print("绘图中的中文可能无法正常显示。请确保系统安装了合适的字体并在此处正确配置。")
# --- 中文字体设置结束 ---

def calculate_clean_frame_diff(frame1, frame2, blur_ksize=5, diff_thresh=30, morph_open_ksize=3, morph_close_ksize=3):
    """
    计算两帧之间“干净”的帧差图，突出显著变化区域。

    参数:
        frame1 (numpy.ndarray): 第一帧图像 (BGR 或灰度)。
        frame2 (numpy.ndarray): 第二帧图像 (BGR 或灰度)。
        blur_ksize (int): 高斯模糊核的大小 (必须是正奇数)。用于预处理降噪。<=1 则不模糊。
        diff_thresh (int): 差分阈值 (0-255)。像素差异低于此值被忽略。关键参数。
        morph_open_ksize (int): 形态学开运算核的大小 (必须是正奇数)。用于去除小噪声点。<=1 则不执行。
        morph_close_ksize (int): 形态学闭运算核的大小 (必须是正奇数)。用于填充目标内小孔/连接邻近区域。<=1 则不执行。

    返回:
        numpy.ndarray: 清理后的二值化帧差图 (白色代表显著差异区域)。
        numpy.ndarray: 原始的灰度绝对差分图 (用于对比)。
    """
    print("--- 开始计算帧差 ---")
    # --- 1. 转换为灰度图 ---
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1.copy()
    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2.copy()
    print("步骤 1: 已转换为灰度图。")

    # --- 2. 预处理 - 高斯模糊 ---
    if blur_ksize > 1 and blur_ksize % 2 == 1:
        print(f"步骤 2: 应用高斯模糊, 核大小: {blur_ksize}x{blur_ksize}")
        gray1_blurred = cv2.GaussianBlur(gray1, (blur_ksize, blur_ksize), 0)
        gray2_blurred = cv2.GaussianBlur(gray2, (blur_ksize, blur_ksize), 0)
    else:
        print("步骤 2: 跳过高斯模糊。")
        gray1_blurred = gray1
        gray2_blurred = gray2

    # --- 3. 计算绝对差分 ---
    frame_diff = cv2.absdiff(gray1_blurred, gray2_blurred)
    print("计算绝对差分完成。")

    # --- 4. 阈值化 ---
    # 将差异小于 diff_thresh 的像素设为0 (黑色), 大于等于的设为255 (白色)
    ret, diff_thresh_img = cv2.threshold(frame_diff, diff_thresh, 255, cv2.THRESH_BINARY)
    if not ret:
         print("警告：阈值化可能失败。")
    print(f"应用阈值化, 阈值: {diff_thresh}")

    processed_diff = diff_thresh_img.copy() # 从阈值化结果开始处理

    # --- 5. 形态学操作 - 开运算 (去除噪声) ---
    if morph_open_ksize > 1 and morph_open_ksize % 2 == 1:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open_ksize, morph_open_ksize))
        processed_diff = cv2.morphologyEx(processed_diff, cv2.MORPH_OPEN, kernel_open)
        print(f"应用形态学开运算, 核大小: {morph_open_ksize}x{morph_open_ksize}")

    # --- 6. 形态学操作 - 闭运算 (填充孔洞/连接) ---
    if morph_close_ksize > 1 and morph_close_ksize % 2 == 1:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close_ksize, morph_close_ksize))
        processed_diff = cv2.morphologyEx(processed_diff, cv2.MORPH_CLOSE, kernel_close)
        print(f"应用形态学闭运算, 核大小: {morph_close_ksize}x{morph_close_ksize}")

    print("帧差计算和清理完成。")
    return processed_diff, frame_diff # 返回清理后的二值图 和 原始灰度差分图

# --- 示例用法 ---
if __name__ == "__main__":
    # 加载你的两张连续帧图像
    image_path1 = 'dotdata/01.bmp' # <<< 修改为你的第一帧图片路径
    image_path2 = 'dotdata/02.bmp' # <<< 修改为你的第二帧图片路径

    frame1 = cv2.imread(image_path1)
    frame2 = cv2.imread(image_path2)

    if frame1 is None or frame2 is None:
        print("错误：无法加载一张或两张图像，请检查路径。")
    else:
        # --- 参数调整 ---
        # 这些参数需要根据你的具体场景（噪声水平、目标大小、运动速度）进行调整
        BLUR_KERNEL_SIZE = 5       # 预处理模糊核大小 (e.g., 3, 5, 7)
        DIFFERENCE_THRESHOLD = 25  # 帧差阈值 (关键参数, e.g., 20-50)
        MORPH_OPEN_KERNEL_SIZE = 3 # 开运算核大小 (去除小噪声点, e.g., 3, 5)
        MORPH_CLOSE_KERNEL_SIZE = 5 # 闭运算核大小 (填充目标内部, e.g., 3, 5, 7)

        # 计算干净的帧差图
        cleaned_diff, original_diff = calculate_clean_frame_diff(
            frame1,
            frame2,
            blur_ksize=BLUR_KERNEL_SIZE,
            diff_thresh=DIFFERENCE_THRESHOLD,
            morph_open_ksize=MORPH_OPEN_KERNEL_SIZE,
            morph_close_ksize=MORPH_CLOSE_KERNEL_SIZE
        )

        # --- 显示结果 ---
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        plt.title('第一帧 (Frame 1)')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        plt.title('第二帧 (Frame 2)')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(original_diff, cmap='gray')
        plt.title('原始灰度差分图 (Abs Diff)')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        # 显示阈值化后的图像（形态学操作前）作为中间步骤对比
        _, diff_thresh_img_display = cv2.threshold(original_diff, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)
        plt.imshow(diff_thresh_img_display, cmap='gray')
        plt.title(f'阈值化差分 (Thresh={DIFFERENCE_THRESHOLD})')
        plt.axis('off')


        plt.subplot(2, 3, 5)
        plt.imshow(cleaned_diff, cmap='gray')
        plt.title('清理后的帧差图 (Cleaned Diff)')
        plt.axis('off')

        # 可以再加一个子图显示处理参数

        plt.tight_layout()
        plt.show()

        # 或者使用 OpenCV 的窗口显示
        # cv2.imshow('Frame 1', frame1)
        # cv2.imshow('Frame 2', frame2)
        # cv2.imshow('Original Difference', original_diff)
        # cv2.imshow('Cleaned Difference', cleaned_diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()