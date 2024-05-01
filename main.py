import time
import torch
import torch.nn as nn
from net import vgg16
from torch.utils.data import DataLoader
from data import *
import numpy as np
import matplotlib.pyplot as plt

# 数据集处理
annotation_path = 'cls_train.txt'
with open(annotation_path, 'r') as f:
    lines = f.readlines()

np.random.seed(10101)
np.random.shuffle(lines)
# np.random.seed(None)

num_val = int(len(lines) * 0.25)
num_train = len(lines) - num_val

input_shape = [224, 224]
train_data = DataGenerator(lines[:num_train], input_shape, True)
val_data = DataGenerator(lines[num_train:], input_shape, False)
val_len = len(val_data)

gen_train = DataLoader(train_data, batch_size=64)
gen_val = DataLoader(val_data, batch_size=64)

# 网络定义和优化器设置
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
net = vgg16(True, progress=True, num_classes=9)
net.to(device)

lr = 0.0001#设置学习率
optim = torch.optim.Adam(net.parameters(), lr=lr)
sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=1)

# 训练过程
epochs = 2
train_losses = []
val_losses = []
train_accuracies = []  # 记录训练集准确率
val_accuracies = []

for epoch in range(epochs):
    start_time = time.time()
    total_train = 0
    total_correct_train = 0  # 用于记录训练集正确预测的数量
    for data in gen_train:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
        
        optim.zero_grad()
        output = net(img)
        train_loss = nn.CrossEntropyLoss()(output, label).to(device)
        train_loss.backward()
        optim.step()
        total_train += train_loss.item()
        
        _, predicted = torch.max(output, 1)
        total_correct_train += (predicted == label).sum().item()

    train_accuracy = total_correct_train / num_train
    train_accuracies.append(train_accuracy)

    sculer.step()
    
    total_val = 0
    total_accuracy = 0
    for data in gen_val:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            
            out = net(img)
            val_loss = nn.CrossEntropyLoss()(out, label).to(device)
            total_val+= val_loss.item()
            
            accuracy = (out.argmax(1) == label).sum().item()
            total_accuracy += accuracy

    val_accuracy = total_accuracy / val_len
    val_accuracies.append(val_accuracy)

    train_losses.append(total_train)
    val_losses.append(total_val)

    end_time = time.time()
    training_time = end_time - start_time

    print("Epoch {}: Train Loss: {:.4f}, Val Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Accuracy: {:.2f}%, Training Time: {:.2f}s"
          .format(epoch+1, total_train, total_val, train_accuracy*100, val_accuracy*100, training_time))

    if val_accuracy > max_val_accuracy:
        max_val_accuracy = val_accuracy
        torch.save(net.state_dict(), "BestModel_acc_{}.pth".format(int(val_accuracy * 100)))  # 保存当前最佳模型

# 创建一个具有两列子图的图形窗口
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 绘制Loss曲线
axs[0].plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2.5, marker='o', markersize=8)
axs[0].plot(range(1, epochs + 1), val_losses, label='Val Loss', linewidth=2.5, marker='o', markersize=8)
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Train and val Loss')
axs[0].legend()

# 绘制Accuracy曲线
axs[1].plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', linewidth=2.5, marker='o', markersize=8)
axs[1].plot(range(1, epochs + 1), val_accuracies, label='Val Accuracy', linewidth=2.5, marker='o', markersize=8)
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Accuracy Curve on Train and Val Data')
axs[1].legend()

plt.tight_layout()  # 调整子图间距，避免重叠
plt.show()       

# # 绘制Loss曲线
# plt.figure()
# plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2.5, marker='o', markersize=8)
# plt.plot(range(1, epochs + 1), val_losses, label='Test Loss', linewidth=2.5, marker='o', markersize=8)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Testing Loss')
# plt.legend()
# plt.show()

# # 绘制Accuracy曲线
# plt.figure()
# plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', linewidth=2.5, marker='o', markersize=8)
# plt.plot(range(1, epochs + 1), val_accuracies, label='Test Accuracy', linewidth=2.5, marker='o', markersize=8)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Curve on Train and Test Data')
# plt.legend()
# plt.show()


# import torch
# import torch.nn as nn
# from net import vgg16
# from torch.utils.data import DataLoader#工具取黑盒子，用函数来提取数据集中的数据（小批次）
# from data import *
# '''数据集'''
# annotation_path='cls_train.txt'#读取数据集生成的文件
# with open(annotation_path,'r') as f:
#     lines=f.readlines()
# np.random.seed(10101)#函数用于生成指定随机数
# np.random.shuffle(lines)#数据打乱
# np.random.seed(None)
# num_val=int(len(lines)*0.2)#十分之二数据用来测试
# num_train=len(lines)-num_val
# #输入图像大小
# input_shape=[224,224]   #导入图像大小
# train_data=DataGenerator(lines[:num_train],input_shape,True)
# val_data=DataGenerator(lines[num_train:],input_shape,False)
# val_len=len(val_data)
# print(val_len)#返回测试集长度
# # 取黑盒子工具
# """加载数据"""
# gen_train=DataLoader(train_data,batch_size=4)#训练集batch_size读取小样本，规定每次取多少样本
# gen_test=DataLoader(val_data,batch_size=4)#测试集读取小样本
# '''构建网络'''
# device=torch.device('cuda'if torch.cuda.is_available() else "cpu")#电脑主机的选择
# net=vgg16(True, progress=True,num_classes=7)#定于分类的类别
# net.to(device)
# '''选择优化器和学习率的调整方法'''
# lr=0.0001#定义学习率
# optim=torch.optim.Adam(net.parameters(),lr=lr)#导入网络和学习率
# sculer=torch.optim.lr_scheduler.StepLR(optim,step_size=1)#步长为1的读取
# '''训练'''
# epochs=20#读取数据次数，每次读取顺序方式不同
# for epoch in range(epochs):
#     total_train=0 #定义总损失
#     for data in gen_train:
#         img,label=data
#         with torch.no_grad():
#             img =img.to(device)
#             label=label.to(device)
#         optim.zero_grad()
#         output=net(img)
#         train_loss=nn.CrossEntropyLoss()(output,label).to(device)
#         train_loss.backward()#反向传播
#         optim.step()#优化器更新
#         total_train+=train_loss #损失相加
#     sculer.step()
#     total_test=0#总损失
#     total_accuracy=0#总精度
#     for data in gen_test:
#         img,label =data #图片转数据
#         with torch.no_grad():
#             img=img.to(device)
#             label=label.to(device)
#             optim.zero_grad()#梯度清零
#             out=net(img)#投入网络
#             test_loss=nn.CrossEntropyLoss()(out,label).to(device)
#             total_test+=test_loss#测试损失，无反向传播
#             accuracy=((out.argmax(1)==label).sum()).clone().detach().cpu().numpy()#正确预测的总和比测试集的长度，即预测正确的精度
#             total_accuracy+=accuracy
#     print("第{}轮：".format(epoch+1))
#     print("训练集上的损失：{}".format(total_train))
#     print("测试集上的损失：{}".format(total_test))
#     print("测试集上的精度：{:.1%}".format(total_accuracy/val_len))#百分数精度，正确预测的总和比测试集的长度

#     # torch.save(net,"Plant{}.pt",format(epoch+1))
#     torch.save(net.state_dict(),"Plant{}.pth".format(epoch+1))
#     print("模型Plant{}已保存".format(epoch+1))



