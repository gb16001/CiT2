import torch
from torch import nn

from models_cit.ALPR import DETR_my

model=DETR_my(num_classes=1,n_query=10)
img=torch.randn(2,3,256,256)
# 将模型设置为评估模式
model.eval()

# 禁用梯度计算以加速推理
with torch.no_grad():
    # 将图像输入模型
    outputs = model(img)

# 解析输出
logits = outputs['pred_logits']
boxes = outputs['pred_boxes']
print(logits.shape,boxes.shape)