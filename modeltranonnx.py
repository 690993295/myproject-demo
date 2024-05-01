import torch
import torchvision
import torch.onnx
import onnx
from onnxsim import simplify

# 实例化一个VGG16模型
model = torchvision.models.vgg16(pretrained=True)
# 修改输出层大小
model.classifier[6] = torch.nn.Linear(4096, 9)  # 将输出层大小从1000修改为9
# 加载预训练好的模型参数
model.load_state_dict(torch.load("Plant5.pth"))

# 将模型设置为评估模式
model.eval()

# 创建一个示例输入张量
example = torch.randn(1, 3, 224, 224)

# 导出模型为ONNX格式
torch.onnx.export(model, example, "Plant5.onnx", verbose=True, opset_version=11, input_names=["input"], output_names=["output"])

# 加载导出的ONNX模型
onnx_model = onnx.load("Plant5.onnx")

# 优化ONNX模型
optimized_model, check = simplify(onnx_model)
onnx.save(optimized_model, "Plant5_optimized.onnx")
