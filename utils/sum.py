import torch
from Model.MultiFTNet import MultiFTNet
from Model.MiniFASNet import MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2SE

# MXNet backbone (hiện tại trong MultiFTNet)
model_mnext = MultiFTNet(num_classes=2, img_channel=3, embedding_size=128)

# Tổng params (bao gồm FTGenerator - chỉ dùng khi train)
params_mnext_total = sum(p.numel() for p in model_mnext.parameters())

# Params inference only (không tính FTGenerator)
params_ft = sum(p.numel() for p in model_mnext.FTGenerator.parameters())
params_mnext_inference = params_mnext_total - params_ft

# MiniFASNet gốc (chỉ có inference, không có FT branch)
model_mini_v1 = MiniFASNetV1(num_classes=3, img_channel=3)
params_mini_v1 = sum(p.numel() for p in model_mini_v1.parameters())

model_mini_v1se = MiniFASNetV1SE(num_classes=3, img_channel=3)
params_mini_v1se = sum(p.numel() for p in model_mini_v1se.parameters())

model_mini_v2se = MiniFASNetV2SE(num_classes=4, img_channel=3)
params_mini_v2se = sum(p.numel() for p in model_mini_v2se.parameters())

print("=" * 55)
print("So sánh số lượng tham số (inference only)")
print("=" * 55)
print(f"MultiFTNet (MXNet) - inference:  {params_mnext_inference:>10,} params")
print(f"MultiFTNet (MXNet) - total:      {params_mnext_total:>10,} params")
print(f"  └─ FTGenerator (train only):   {params_ft:>10,} params")
print(f"MiniFASNetV1:                    {params_mini_v1:>10,} params")
print(f"MiniFASNetV1SE:                  {params_mini_v1se:>10,} params")
print(f"MiniFASNetV2SE:                  {params_mini_v2se:>10,} params")