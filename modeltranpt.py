import torch
from net import vgg16
from torch.utils.mobile_optimizer import optimize_for_mobile

# 检测是否有可用的GPU，如果有则使用cuda，否则使用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化一个VGG16模型
model = vgg16()

# 加载预训练好的模型参数
model.load_state_dict(torch.load("Plant4_acc_97.pth"))

# 将模型设置为评估模式
model.eval()

# 创建一个示例输入张量
example = torch.ones(1, 3, 224, 224)

# 使用torch.jit.trace对模型进行追踪，以便后续的推理加速
traced_script_model = torch.jit.trace(model, example)

#针对移动端进行优化
optimize_traced_model = optimize_for_mobile(traced_script_model)

# 将模型保存
# torch.jit.save(optimize_traced_model, "DogCat1.pt")
optimize_traced_model._save_for_lite_interpreter("Plant5.pt")
# torch.onnx.export(model, example, "Plant3.onnx", verbose=True)
