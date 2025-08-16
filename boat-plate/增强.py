import imgaug.augmenters as iaa
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. 定义增强序列
# 创建一个序列，包含随机旋转和随机水平翻转
# order="linear" 和 cval=0 是处理旋转后边界的参数
# rotate=(-10, 10) 表示随机旋转角度在-10到+10度之间
# flipr=0.5 表示以50%的概率水平翻转
augmenter = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10), order=1, cval=0, mode='constant'), # 随机旋转
    iaa.Fliplr(0.5) # 随机水平翻转
], random_order=True) # 以随机顺序应用序列中的增强

# 2. 加载图像 (请替换为你的图片路径)
image_path = "./1.jpg" # 示例图片路径
image = cv2.imread(image_path)

if image is None:
    print(f"错误：无法加载图片 {image_path}")
else:
    # OpenCV加载的图片是BGR格式，为了在matplotlib中正确显示，转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. 应用数据增强
    # augmenter可以处理单张图片或一个图片列表
    augmented_image = augmenter(image=image)

    # 4. 显示原始图像和增强后的图像
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(augmented_image)
    axes[1].set_title("Augmented Image")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # 5. 应用更复杂的增强序列 (示例，你可以根据config配置更多)
    more_complex_augmenter = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, # 随机缩放
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # 随机平移
            rotate=(-20, 20), # 更大范围的旋转
            shear=(-8, 8), # 剪切变换
            order=[0, 1], # 使用最近邻或双线性插值
            cval=(0, 255), # 填充边界的像素值范围
            mode=iaa.ALL # 填充模式
        ),
        iaa.AddToHueAndSaturation((-20, 20)), # 随机改变色相和饱和度
        iaa.GammaContrast((0.5, 2.0)), # 随机改变对比度
        iaa.GaussianBlur(sigma=(0.0, 1.0)) # 随机应用高斯模糊
    ], random_order=True)

    augmented_image_complex = more_complex_augmenter(image=image)

    plt.figure(figsize=(6, 6))
    plt.imshow(augmented_image_complex)
    plt.title("More Complex Augmented Image")
    plt.axis('off')
    plt.show()