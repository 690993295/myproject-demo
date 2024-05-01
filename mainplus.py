import time
import torch
import torch.nn as nn
from net import vgg16
from torch.utils.data import DataLoader
from data import *
import numpy as np
import matplotlib.pyplot as plt

annotation_path = 'cls_train.txt'
with open(annotation_path, 'r') as f:
    lines = f.readlines()

np.random.seed(10101)
np.random.shuffle(lines)


num_val = int(len(lines) * 0.2)
num_train = len(lines) - num_val

input_shape = [224, 224]
train_data = DataGenerator(lines[:num_train], input_shape, True)
val_data = DataGenerator(lines[num_train:], input_shape, False)
val_len = len(val_data)

gen_train = DataLoader(train_data, batch_size=32)
gen_val = DataLoader(val_data, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
net = vgg16(True, progress=True, num_classes=9)
net.to(device)

lr = 0.0001
optim = torch.optim.Adam(net.parameters(), lr=lr)
sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.1)

def save_training_info(training_info):
    with open("training_info.txt", "w") as file:
            file.write("Epoch\tTrain Loss\tVal Loss\tTrain Accuracy\tVal Accuracy\tTraining Time (s)\n")
            for info in training_info:
                file.write("{}\t{:.4f}\t{:.4f}\t{:.2f}%\t{:.2f}%\t{:.2f}\n"
                        .format(info["Epoch"], info["Train Loss"], info["Val Loss"], 
                                info["Train Accuracy"], info["Val Accuracy"], info["Training Time"]))

# def train_model(net, optim, sculer, train_loader, val_loader, epochs=5):
#     train_losses = []
#     val_losses = []
#     train_accuracies = []  # 记录训练集准确率
#     val_accuracies = []
#     max_val_accuracy = 0.0
#     training_info = []  # 存储训练信息的列表

#     for epoch in range(epochs):
#         start_time = time.time()
#         total_train_loss = 0.0
#         total_train_correct = 0

#         net.train()
#         for img, label in train_loader:
#             img = img.to(device)
#             label = label.to(device)

#             optim.zero_grad()
#             output = net(img)
#             train_loss = nn.CrossEntropyLoss()(output, label)
#             train_loss.backward()
#             optim.step()

#             total_train_loss += train_loss.item() * img.size(0)
#             _, predicted = torch.max(output, 1)
#             total_train_correct += (predicted == label).sum().item()

#         train_accuracy = total_train_correct / num_train
#         train_accuracies.append(train_accuracy)

#         sculer.step()

#         total_val_loss = 0.0
#         total_val_correct = 0

#         net.eval()
#         with torch.no_grad():
#             for img, label in val_loader:
#                 img = img.to(device)
#                 label = label.to(device)

#                 output = net(img)
#                 val_loss = nn.CrossEntropyLoss()(output, label)
#                 total_val_loss += val_loss.item() * img.size(0)

#                 _, predicted = torch.max(output, 1)
#                 total_val_correct += (predicted == label).sum().item()

#         val_accuracy = total_val_correct / val_len
#         val_accuracies.append(val_accuracy)

#         train_losses.append(total_train_loss / num_train)
#         val_losses.append(total_val_loss / val_len)

#         end_time = time.time()
#         training_time = end_time - start_time

# 在每个 epoch 结束后调用此函数保存训练信息
def train_model(net, optim, sculer, train_loader, val_loader, epochs=2):
    train_losses = []
    val_losses = []
    train_accuracies = []  # 记录训练集准确率
    val_accuracies = []
    max_val_accuracy = 0.0
    training_info = []  # 存储训练信息的列表

    for epoch in range(epochs):
        start_time = time.time()
        total_train_loss = 0.0
        total_train_correct = 0

        net.train()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)

            optim.zero_grad()
            output = net(img)
            train_loss = nn.CrossEntropyLoss()(output, label)
            train_loss.backward()
            optim.step()

            total_train_loss += train_loss.item() * img.size(0)
            _, predicted = torch.max(output, 1)
            total_train_correct += (predicted == label).sum().item()

        train_accuracy = total_train_correct / num_train
        train_accuracies.append(train_accuracy)

        sculer.step()

        total_val_loss = 0.0
        total_val_correct = 0

        net.eval()
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device)
                label = label.to(device)

                output = net(img)
                val_loss = nn.CrossEntropyLoss()(output, label)
                total_val_loss += val_loss.item() * img.size(0)

                _, predicted = torch.max(output, 1)
                total_val_correct += (predicted == label).sum().item()

        val_accuracy = total_val_correct / val_len
        val_accuracies.append(val_accuracy)

        train_losses.append(total_train_loss / num_train)
        val_losses.append(total_val_loss / val_len)

        end_time = time.time()
        training_time = end_time - start_time

        # 将本次训练的信息保存到列表中        

        print("Epoch {}: Train Loss: {:.4f}, Val Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Accuracy: {:.2f}%, Training Time: {:.2f}s"
              .format(epoch+1, train_loss.item(), val_loss.item(), train_accuracy*100, val_accuracy*100, training_time))
        
        # 将训练信息添加到列表中
        epoch_info = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss.item(),
            "Val Loss": val_loss.item(),
            "Train Accuracy": train_accuracy * 100,
            "Val Accuracy": val_accuracy * 100,
            "Training Time": training_time
        }
        training_info.append(epoch_info)

         # 将训练信息写入txt文件
        save_training_info(training_info)

        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(net.state_dict(), "BestModel_acc_{}.pth".format(int(val_accuracy * 100)))# 保存当前最佳模型

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

train_model(net, optim, sculer, gen_train, gen_val)
