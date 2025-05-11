import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

# --- 设置你的图片路径 ---
image_path = 'your_image.jpg'  # <-- 请将 'your_image.jpg' 替换为你图片的实际路径

try:
    # 1. 打开图片
    img = PIL.Image.open(image_path)
    print(f"图片 '{image_path}' 加载成功.")
    print(f"原始图片信息: 格式={img.format}, 尺寸={img.size}, 模式={img.mode}")

    # 2. 将图片转换为灰度图
    # 'L' 模式表示灰度图，每个像素用 8 位表示 (0-255)
    img_gray = img.convert('L')
    print(f"图片已转换为灰度模式: {img_gray.mode}")

    # 3. 获取灰度图的像素值（NumPy 数组）
    # 这个数组包含了图片的全部灰度值
    img_array = np.array(img_gray)

    # --- 计算灰度值（实际是获取数组）---
    # img_array 就是包含所有像素灰度值的 NumPy 数组。
    # 对于大型图片，打印所有值是不切实际的。
    # 你可以通过 img_array[row, col] 访问特定像素的灰度值。
    print("\n--- 示例灰度值 (图片左上角 5x5 像素) ---")
    print(img_array[:5, :5]) # 打印左上角 5行x5列 的像素灰度值

    # --- 计算并生成灰度直方图 ---
    # 直方图显示了每个灰度值（0-255）在图片中出现的频率（像素数量）

    # 使用 matplotlib 的 hist 函数直接计算并绘制直方图
    # .ravel() 将多维数组展平为一维数组，适合 hist 函数
    # bins=256 表示有 256个 bin，对应 0-255 每个灰度级
    # range=(0, 256) 确保覆盖所有可能的灰度值
    plt.hist(img_array.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)

    # 添加直方图的标题和标签
    plt.title('灰度直方图')
    plt.xlabel('灰度强度 (0-255)')
    plt.ylabel('像素数量')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 4. 显示直方图
    plt.show()

except FileNotFoundError:
    print(f"错误：找不到图片文件 '{image_path}'。请检查路径是否正确。")
except Exception as e:
    print(f"处理图片时发生错误：{e}")