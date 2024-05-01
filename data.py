import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

# 定义一个预处理函数，将图像像素值归一化并做减法变换（通常用于深度学习模型输入）
def preprocess_input(x):
    x /= 127.5  # 归一化到[-1, 1]区间
    x -= 1.      # 减去1后，图像像素值分布在[-1, 1]
    return x

# 定义一个将图像转换为RGB格式的函数
def cvtColor(image):
    # 如果图像已经是RGB格式且通道数为3，则直接返回原图像
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        # 否则将图像转换为RGB格式
        image = image.convert('RGB')  # 使用PIL库将图像转换为RGB模式
        return image

# 定义一个自定义数据加载器类，继承自torch.utils.data.Dataset
class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, inpt_shape, random=True):
        """
        初始化方法，接收：
        - annotation_lines: 包含图像路径和标签的标注文件中的每行数据列表
        - inpt_shape: 模型所需的输入图像尺寸（高度，宽度）
        - random: 是否对图像进行随机增强（默认True）
        """
        self.annotation_lines = annotation_lines
        self.input_shape = inpt_shape
        self.random = random

    def __len__(self):
        """
        返回数据集中样本的数量，即标注行数
        """
        return len(self.annotation_lines)

    def __getitem__(self, index):
        """
        根据索引获取单个样本数据
        """
        # 解析索引对应的标注行，得到图像路径和标签
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = Image.open(annotation_path)  # 打开图像文件

        # 对图像进行随机增强处理
        image = self.get_random_data(image, self.input_shape, random=self.random)

        # 预处理图像像素值并转换为适合模型输入的形状（通道数在前）
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])

        # 提取标签
        y = int(self.annotation_lines[index].split(';')[0])

        # 返回经过处理的图像数据和对应标签
        return image, y

    def rand(self, a=0, b=1):
        """
        返回指定范围内的随机浮点数
        """
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, inpt_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """
        对输入图像进行一系列随机增强操作，包括：
        - 调整图像长宽比
        - 缩放图像
        - 添加随机偏移填充
        - 随机水平翻转
        - 随机旋转
        - 随机调整色调、饱和度和明度
        """
        # 确保图像为RGB格式
        image = cvtColor(image)
        iw, ih = image.size  # 原始图像的宽和高
        h, w = inpt_shape  # 目标输入图像的高和宽

        # 如果random为False，则不执行随机增强，仅按比例缩放并居中填充
        if not random:
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            dx, dy = (w - nw) // 2, (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            return image_data

        # 如果允许随机增强，则执行以下操作
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.75, 1.25)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        # 缩放图像
        image = image.resize((nw, nh), Image.BICUBIC)

        # 添加随机偏移填充
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 随机水平翻转
        if self.rand() < .5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机旋转图像
        if self.rand() < .5:
            angle = np.random.randint(-15, 15)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        # 随机调整图像色域
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 1] *= sat  # 调整饱和度
        x[..., 2] *= val  # 调整明度
        x[x[:, :, 0] > 360, 0] = 360  # 限制色调在0-360之间
        x[:, :, 1:][x[:, :, 1:] > 1] = 1  # 限制饱和度和明度在0-1之间
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # 将图像从HSV色彩空间转换回RGB，并恢复至0-255范围

        return image_data

# import cv2
# import numpy as np
# import torch.utils.data as data
# from PIL import Image

# def preprocess_input(x):
#     x /= 127.5
#     x -= 1.
#     return x

# def cvtColor(image):
#     # 如果图像是RGB格式且通道数为3，则直接返回图像
#     if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
#         return image
#     else:
#         # 否则将图像转换为RGB格式
#         image = image.convert('RGB')
#         return image

# class DataGenerator(data.Dataset):
#     def __init__(self, annotation_lines, inpt_shape, random=True):
#         self.annotation_lines = annotation_lines
#         self.input_shape = inpt_shape
#         self.random = random

#     def __len__(self):
#         return len(self.annotation_lines)

#     def __getitem__(self, index):
#         # 获取图像路径和标签
#         annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
#         image = Image.open(annotation_path)
#         # 对图像进行随机处理
#         image = self.get_random_data(image, self.input_shape, random=self.random)
#         # 将图像转换为numpy数组，并对像素值进行预处理
#         image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
#         y = int(self.annotation_lines[index].split(';')[0])
#         return image, y

#     def rand(self, a=0, b=1):
#         return np.random.rand() * (b - a) + a

#     def get_random_data(self, image, inpt_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
#         # 将图像转换为RGB格式
#         image = cvtColor(image)
#         iw, ih = image.size
#         h, w = inpt_shape

#         if not random:
#             # 如果不需要随机处理，则按原始尺寸缩放图像并在周围填充灰色边框
#             scale = min(w / iw, h / ih)
#             nw = int(iw * scale)
#             nh = int(ih * scale)
#             dx = (w - nw) // 2
#             dy = (h - nh) // 2

#             image = image.resize((nw, nh), Image.BICUBIC)
#             new_image = Image.new('RGB', (w, h), (128, 128, 128))
#             new_image.paste(image, (dx, dy))
#             image_data = np.array(new_image, np.float32)
#             return image_data

#         # 对图像进行随机处理
#         new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
#         scale = self.rand(.75, 1.25)
#         if new_ar < 1:
#             nh = int(scale * h)
#             nw = int(nh * new_ar)
#         else:
#             nw = int(scale * w)
#             nh = int(nw / new_ar)

#         image = image.resize((nw, nh), Image.BICUBIC)

#         # 在图像周围添加灰色边框
#         dx = int(self.rand(0, w - nw))
#         dy = int(self.rand(0, h - nh))
#         new_image = Image.new('RGB', (w, h), (128, 128, 128))
#         new_image.paste(image, (dx, dy))
#         image = new_image

#         # 随机翻转图像
#         flip = self.rand() < .5
#         if flip:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)

#         # 随机旋转图像
#         rotate = self.rand() < .5
#         if rotate:
#             angle = np.random.randint(-15, 15)
#             a, b = w / 2, h / 2
#             M = cv2.getRotationMatrix2D((a, b), angle, 1)
#             image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

#         # 随机调整图像色域
#         hue = self.rand(-hue, hue)
#         sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
#         val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
#         x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
#         x[..., 1] *= sat
#         x[..., 2] *= val
#         x[x[:, :, 0] > 360, 0] = 360
#         x[:, :, 1:][x[:, :, 1:] > 1] = 1
#         x[x < 0] = 0
#         image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

#         return image_data
