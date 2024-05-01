import torch
from torch.hub import load_state_dict_from_url
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import Compose
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 定义预训练模型权重下载链接字典
model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
}
# 1. 加载模型
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
    
model = vgg16()  # 实例化模型
model.load_state_dict(torch.load('BestModel_acc_97.pth'))  # 加载权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 处理标注文件
def parse_annotation_file(annotation_path):
    data_list = []
    with open(annotation_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            try:
                img_path, label = line.strip().split(' ')
                data_list.append((img_path, int(label)))
            except ValueError:
                print(f"Skipped invalid line {line_num}: {line.strip()}")
    return data_list


data_list = parse_annotation_file('cls_train.txt')

# 3. 数据预处理
transform = Compose([
    Resize((224, 224)),  # 从 torchvision.transforms 导入 Resize
    ToTensor(),  # 从 torchvision.transforms 导入 ToTensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 从 torchvision.transforms 导入 Normalize
])

# 1. 检查数据集是否为空
if len(data_list) == 0:
    raise ValueError("Dataset is empty. Cannot split into train and validation sets.")

# 2. 划分数据集
random.seed(10101)
train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=10101)

# 3. 定义数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_data, transform=transform)
val_dataset = CustomDataset(val_data, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 6. 计算混淆矩阵
all_val_preds, all_val_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)  # 获取预测类别
        all_val_preds.extend(preds.cpu().numpy())
        all_val_labels.extend(labels.cpu().numpy())

val_conf_mat = confusion_matrix(all_val_labels, all_val_preds)

# 7. 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(val_conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)

num_classes=9
label_classes = list(range(num_classes))  # 创建标签列表

ax.set_xlabel('Predicted labels', rotation=45, ha='right', va='top')
ax.set_xticklabels(label_classes, rotation=45, ha='right', va='top')
ax.set_ylabel('True labels', rotation=45, ha='right', va='top')
ax.set_yticklabels(label_classes, rotation=45, ha='right', va='top')
ax.set_title(f'Confusion Matrix on Validation Set (Retrospective Evaluation)')

plt.show()