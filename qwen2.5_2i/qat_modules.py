import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization import BitNetQuantSTE, PhaseQuantSTE
import math

class QATLinearBitNet(nn.Linear):
    """
    BitNet 的 QAT-ready 线性层。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # 在每次前向传播中动态地对权重进行模拟量化
        quantized_weight = BitNetQuantSTE.apply(self.weight)
        return F.linear(x, quantized_weight, self.bias)

class QATLinearComplexPhase(nn.Linear):
    """
    Complex-Phase 的 QAT-ready 线性层。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        # 1. 分解 (可导)
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        # 2. 量化 (使用STE，梯度可回传)
        U_re_q, U_im_q = PhaseQuantSTE.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE.apply(W_re, W_im)
        
        # 3. 重构 (可导)
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        # 4. 使用量化后的权重进行前向传播
        return F.linear(x, A_quant, self.bias)

def replace_modules_for_qat(model, quant_method):
    """
    递归地遍历模型，将所有 nn.Linear 层替换为对应的 QAT 版本。
    """
    for name, module in model.named_children():
        # 如果还有子模块，则递归深入
        if len(list(module.children())) > 0:
            replace_modules_for_qat(module, quant_method)
        
        if isinstance(module, nn.Linear):
            print(f"  -> Replacing {name}...")
            if quant_method == 'bitnet':
                new_module = QATLinearBitNet(
                    module.in_features, 
                    module.out_features, 
                    bias=module.bias is not None, 
                    device=module.weight.device, 
                    dtype=module.weight.dtype
                )
            elif quant_method == 'complex_phase':
                # 对于 Complex-Phase，我们只替换维度为偶数的层
                if module.in_features % 2 == 0 and module.out_features % 2 == 0:
                    new_module = QATLinearComplexPhase(
                        module.in_features, 
                        module.out_features, 
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype
                    )
                else:
                    print(f"     ! Skipping layer {name} with shape ({module.out_features}, {module.in_features}) due to odd dimensions.")
                    continue
            else:
                raise ValueError(f"Invalid quant_method for QAT: {quant_method}")
            
            # 复制原始权重和偏置
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
            
            setattr(model, name, new_module)
