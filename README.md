# Boat-PlateOCR——基于DBNet和SVTR_HGNET的船牌OCR识别系统

本项目为**船牌OCR识别系统**，文本检测上使用DBNet算法，文本识别上使用SVTR_HGNET算法，利用 **PaddleOCR** 进行船牌图像的检测与识别。系统配备了一个用户友好的 **PyQt5** 前端，用户可以通过图形界面上传图片、查看检测到的船牌区域，并将识别结果保存到 SQLite 数据库中。

## 功能

- **船牌检测**：使用深度学习模型从图像中检测并裁剪出船牌区域。
- **船牌识别**：对检测到的船牌区域进行文本识别。
- **数据库集成**：将识别到的船牌数据保存到 SQLite 数据库。
- **图形界面**：使用 **PyQt5** 开发的图形用户界面，方便进行图片上传、检测和结果展示。
- **GPU 加速**：检测和识别模型支持 GPU 加速，提升处理速度。

 

## 目录结构

识别系统

![image-20250816194111413](https://gitee.com/godwen666/pictures/raw/master/image-20250816194111413.png)



**icon**：前端使用的icon图标，增加美观。

**model**：det为训练好的检测模型，rec为训练好的识别模型

**test**:本项目的示例图片，用于测试识别船牌的效果

**main-run.py**:项目的运行主要入口

**sqlite_main().py**:个人改善更新后可以存入数据库的项目运行入口，相比上一个运行入口拥有存储数据库的功能



## 安装

请按照以下步骤进行项目的安装和运行：

### 客户端运行必备条件

- Python 3.9
- torch 1.12.1+cu116
- PaddleOCR
- PyQt5
- SQLite3
- OpenCV
- NumPy
- PaddlePaddle

### 安装步骤

1. 克隆项目仓库（或本地下载）：

  ```
  git clone https://github.com/Godwen7/BoatPlateOCR-DBNet-SVTR-OCR.git
  cd ship-license-recognition
  ```

2. 创建并激活虚拟环境：

  可在Anaconda或miniconda中创建虚拟环境

  此处给出一个实例

  ```
  conda create -n boatocr python==3.19
  conda activate boatocr
  ```

3. 安装依赖：

  ```
  cd 你当前保存的requirements文件目录
  pip install -r requirements.txt
  ```

4. 下载预训练的检测和识别模型，本项目个人训练的模型已放置于源码中，如需更精准的训练，可以参考仓库中给出的训练参数，在PaddleOCR中再次训练，地址：https://github.com/PaddlePaddle/PaddleOCR

5. 将模型设置在正确的目录结构中：

  ```
  ├── model
  │   ├── det
  │   └── rec
  ```

## 使用

1. 运行应用程序：

  ```
  python main-run.py  #此处为程序主入口
  或者 python sqlite_main().py  #此处为可存储数据库的程序主入口
  ```

2. 图形界面会打开，您可以上传包含船牌的图片进行检测与识别。系统会展示检测到的船牌区域和识别出的文本。

3. 点击“保存到数据库”按钮，可以将识别结果保存到 SQLite 数据库中。

### 运行效果

- 点击run启动程序，上传一张包含船牌的图片。（本项目中提供约20张示例图片）

	![image-20250817190351973](https://gitee.com/godwen666/pictures/raw/master/image-20250817190351973.png)

	![image-20250817190409787](https://gitee.com/godwen666/pictures/raw/master/image-20250817190409787.png)

- 系统会检测出船牌区域并提取文本，提取出最大置信区间的结果后会在图形界面中显示，您可以选择将结果保存到数据库中。

	![image-20250817190442306](https://gitee.com/godwen666/pictures/raw/master/image-20250817190442306.png)

	

## 数据库结构

SQLite 数据库存储以下信息：

- **id**: 每个记录的唯一标识符。
- **image_filename**: 图片的文件名。
- **recognized_text**: 识别出的船牌文本。
- **timestamp**: 识别时间戳。

## 模型架构

- **检测模型**：使用 **DBNet** 算法，并配备 **ResNet-50** 骨干网络。

	**模型配置**：

	- 算法：**DB**（差分二值化）
	- 背骨网络：**ResNet-50**
	- 后处理：**DBPostProcess**

- **识别模型**：使用 **SVTR_HGNet** 算法。

	**模型配置**：

	- 算法：**SVTR_HGNet**
	- 优化器：**Adam**
	- 损失函数：**CTC Loss** + **NRTR Loss**

## 依赖项 (requirements.txt)

以下是项目的所有依赖项，您可以直接使用 `requirements.txt` 文件安装：

```
albucore==0.0.24
albumentations==2.0.8
annotated-types==0.7.0
anyio==4.10.0
astor==0.8.1
beautifulsoup4==4.13.4
certifi==2025.8.3
charset-normalizer==3.4.3
colorama==0.4.6
Cython==3.1.3
decorator==5.2.1
eval_type_backport==0.2.2
exceptiongroup==1.3.0
fire==0.7.0
fonttools==4.59.1
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.10
imageio==2.37.0
lazy_loader==0.4
lmdb==1.7.3
lxml==6.0.0
networkx==3.2.1
numpy==2.0.2
opencv-contrib-python==4.12.0.88
opencv-python==4.11.0.86
opencv-python-headless==4.12.0.88
opt-einsum==3.3.0
packaging==25.0
paddleocr==2.10.0
paddlepaddle==3.0.0
pillow==11.3.0
protobuf==6.32.0
pyclipper==1.3.0.post6
pydantic==2.11.7
pydantic_core==2.33.2
PyQt5==5.15.11
PyQt5-Qt5==5.15.2
PyQt5_sip==12.17.0
python-docx==1.2.0
PyYAML==6.0.2
RapidFuzz==3.13.0
requests==2.32.4
scikit-image==0.24.0
scipy==1.13.1
shapely==2.0.7
simsimd==6.5.0
sniffio==1.3.1
soupsieve==2.7
stringzilla==3.12.6
termcolor==3.1.0
tifffile==2024.8.30
torch==1.12.1+cu116
torchaudio==0.12.1+cu116
torchvision==0.13.1+cu116
tqdm==4.67.1
typing-inspection==0.4.1
typing_extensions==4.14.1
urllib3==2.5.0
```

## 注意事项

如有问题可以联系email：1941023711@qq.com

项目演示视频：【基于DBNet和SVTR的深度学习船牌OCR识别系统的项目演示】 https://www.bilibili.com/video/BV1D9Y1zUE6V/?share_source=copy_web&vd_source=c064eb15ceee08c9491d3ff140711fe2

欢迎点个小Star，不胜感激！