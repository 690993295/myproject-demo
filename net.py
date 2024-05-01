import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url

# 定义预训练模型权重下载链接字典
model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
}

# 定义VGG类，继承自nn.Module
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, dropout=0.5):
        super(VGG, self).__init__()
        
        # 初始化特征提取部分
        self.features = features
        
        # 添加全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 初始化分类器部分，包含3个全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        # 若init_weights为True，则初始化权重参数
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    # 定义前向传播函数
    def forward(self, x):
        x = self.features(x)  # 提取特征
        x = self.avgpool(x)   # 平均池化
        x = torch.flatten(x, 1)  # 将多维特征展平成一维
        x = self.classifier(x)  # 通过分类器得到最终输出
        return x

# 该函数用于根据给定配置创建卷积层序列
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 下采样
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# VGG16模型的配置
cfgs = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}

# VGG16模型构建函数，支持预训练权重加载和改变输出类别数量
def vgg16(pretrained=False, progress=True, num_classes=9):
    model = VGG(make_layers(cfgs['D']))

    # 如果pretrained为True，则从指定URL下载预训练权重并加载到模型中
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model', progress=progress)
        model.load_state_dict(state_dict)

    # 如果num_classes不等于默认的1000，则重新定义最后一层全连接层以适应新的类别数
    if num_classes != 1000:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    return model

# 主程序入口，用于测试模型
if __name__ == '__main__':
    in_data = torch.ones(1, 3, 224, 224)  # 创建一个形状为 (batch_size, channels, height, width) 的测试输入
    net = vgg16(pretrained=False, progress=True, num_classes=9)  # 实例化一个VGG16模型
    out = net(in_data)  # 前向传播计算输出
    print(out)  # 输出预测结果

# import torch
# import torch.nn as nn

# from torch.hub import load_state_dict_from_url


# model_urls = {
#     "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",}#权重下载网址


# class VGG(nn.Module):
#     def __init__(self, features, num_classes = 1000, init_weights= True, dropout = 0.5):
#         super(VGG,self).__init__()
#         self.features = features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))#AdaptiveAvgPool2d使处于不同大小的图片也能进行分类
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),#完成4096的全连接
#             nn.Linear(4096, num_classes),#对num_classes的分类
#         )
#         if init_weights:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.Linear):
#                     nn.init.normal_(m.weight, 0, 0.01)
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)#对输入层进行平铺，转化为一维数据
#         x = self.classifier(x)
#         return x


# def make_layers(cfg, batch_norm = False):#make_layers对输入的cfg进行循环
#     layers = []
#     in_channels = 3
#     for v in cfg:#对cfg进行输入循环,取第一个v
#         if v == "M":
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]#把输入图像进行缩小
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)#输入通道是3，输出通道64
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)


# cfgs = {
#     "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],

# }

# def vgg16(pretrained=False, progress=True,num_classes=9):
#     model = VGG(make_layers(cfgs['D']))
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['vgg16'],model_dir='./model' ,progress=progress)#预训练模型地址
#         model.load_state_dict(state_dict)
#     if num_classes !=1000:
#         model.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),#随机删除一部分不合格
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),#随机失活，防止过拟合
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, num_classes),
#         )
#     return model
# if __name__=='__main__':
#     in_data=torch.ones(1,3,224,224)
#     net=vgg16(pretrained=False, progress=True,num_classes=9)
#     out=net(in_data)
#     print(out)
