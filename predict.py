from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from net import vgg16
import os
import pandas as pd

verify_folder = r'data/vali'  # 验证集根文件夹路径
class_folders = ['begonia','daisy', 'dandelion', 'magnolia', 
                 'pine', 'rose', 'sunflower', 'willow','wuyedijin']

# 加载网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU与GPU的选择
net = vgg16()  # 输入网络
model = torch.load(r"BestModel_acc_98.pth", map_location=device)  # 已训练完成的结果权重输入
net.load_state_dict(model)  # 模型导入
net.eval()  # 设置为推测模式

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

data = []  # 用于保存结果的列表

# 循环遍历每个类别文件夹下的图片
for class_folder in class_folders:
    class_path = os.path.join(verify_folder, class_folder)
    for filename in os.listdir(class_path):
        image_path = os.path.join(class_path, filename)
        test = Image.open(image_path)
        image = transform(test).unsqueeze(0)  # 增加batch维度，变成4维张量

        image = image.to(device)

        with torch.no_grad():
            out = net(image)
        out = F.softmax(out, dim=1)  # softmax 函数确定范围
        out = out.data.cpu().numpy()
        predicted_class = class_folders[int(out.argmax(1))]  # 预测的类别
        confidence = out[0, int(out.argmax(1))]  # 置信度

        data.append([filename, class_folder, predicted_class, confidence])

# 创建DataFrame并保存为Excel文件
df = pd.DataFrame(data, columns=['Image', 'Actual Class', 'Predicted Class', 'Confidence'])
df.to_excel('predictions9.xlsx', index=False)


# from torchvision import transforms
# from PIL import Image
# import torch
# import torch.nn.functional as F
# from net import vgg16
# import os

# verify_folder = r'data/vali'  # 验证集根文件夹路径
# class_folders = ['begonia','daisy', 'dandelion', 'magnolia', 'pine',
#                   'rose', 'sunflower', 'willow','wuyedijin']

# # 加载网络
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU与GPU的选择
# net = vgg16()  # 输入网络
# model = torch.load(r"Plant4_acc_95.pth", map_location=device)  # 已训练完成的结果权重输入
# net.load_state_dict(model)  # 模型导入
# net.eval()  # 设置为推测模式

# transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# # 循环遍历每个类别文件夹下的图片
# for class_folder in class_folders[5:]:  # 从第三个文件夹开始遍历
#     class_path = os.path.join(verify_folder, class_folder)
#     for filename in os.listdir(class_path):
#         image_path = os.path.join(class_path, filename)
#         test = Image.open(image_path)
#         image = transform(test).unsqueeze(0)  # 增加batch维度，变成4维张量

#         image = image.to(device)

#         with torch.no_grad():
#             out = net(image)
#         out = F.softmax(out, dim=1)  # softmax 函数确定范围
#         out = out.data.cpu().numpy()
#         a = int(out.argmax(1))  # 输出最大值位置

#         print("Image: {}, Class: {}: {:.1%}".format(filename, class_folder, out[0, a]))
