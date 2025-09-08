import torch
import torch.nn as nn
import math

# ==============================================================================
# 核心辅助函数 (被新方法使用)
# ==============================================================================

@torch.no_grad()
def quantize_complex_tensor(w_real: torch.Tensor, w_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对单个复数权重张量应用 iFairy PhaseQuant 逻辑。
    此版本基于用户提供的 DirectionQuantSTE 实现，逻辑更清晰。
    """
    phase = torch.angle(w_real + 1j * w_imag)

    # 1. 根据相位定义四个量化方向的掩码
    real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
    real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
    imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
    imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

    # 2. 计算每个方向的缩放因子 (scale)
    mask_real = real_pos | real_neg
    mask_imag = imag_pos | imag_neg

    # 处理掩码可能为空的边缘情况，避免NaN
    s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
    s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
    
    # 避免缩放因子为0或NaN
    s_re = torch.clamp(s_re, min=1e-6)
    s_im = torch.clamp(s_im, min=1e-6)
    if torch.isnan(s_re) or torch.isinf(s_re): s_re = torch.tensor(1e-6, device=w_real.device)
    if torch.isnan(s_im) or torch.isinf(s_im): s_im = torch.tensor(1e-6, device=w_imag.device)

    # 3. 创建量化后的基础张量 (-1, 0, 1)
    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)
    
    qw_real[real_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_pos] = 1.0
    qw_imag[imag_neg] = -1.0
    # print("qw_real", qw_real)
    # print("qw_imag", qw_imag)

    # 4. 应用缩放因子
    qw_real_scaled = qw_real * s_re
    qw_imag_scaled = qw_imag * s_im
    # print("qw_real_scaled", qw_real_scaled)
    # print("qw_imag_scaled", qw_imag_scaled)
    return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

# ==============================================================================
# 方案一: 复数域启发的实数量化 (Complex-Inspired Quantization)
# ==============================================================================

def apply_complex_inspired_quantization(model: nn.Module):
    """
    对一个标准的实数模型应用复数域启发的量化。
    流程: 分解 -> PhaseQuant -> 重构
    """
    print("正在应用复数域启发的量化 (PhaseQuant-based)...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        A = module.weight.data
        if A.shape[0] % 2 != 0 or A.shape[1] % 2 != 0:
            print(f"  -> 跳过层 (维度非偶数): {A.shape}")
            return

        # 1. 分解 (Decompose)
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]

        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)

        # 2. 对 U 和 W 应用 PhaseQuant
        U_re_q, U_im_q = quantize_complex_tensor(U_re, U_im)
        W_re_q, W_im_q = quantize_complex_tensor(W_re, W_im)

        # 3. 重构 (Reconstruct)
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q

        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        # 4. 用量化后的权重替换原始权重
        module.weight.data = A_quant.to(A.dtype)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("复数域启发的量化完成。")
    return model

# ==============================================================================
# 方案二: BitNet (1-bit) 实数模型量化
# ==============================================================================

def apply_bitnet_quantization(model: nn.Module):
    """
    对一个标准的实数模型应用 BitNet 1-bit 量化。
    """
    print("正在对实数模型应用 BitNet (1-bit) 量化...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        scale = module.weight.abs().mean()
        module.weight.data = module.weight.data.sign() * scale

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("BitNet 量化应用完成。")
    return model

# ==============================================================================
# 方案三: Min-Max (1-bit) 实数模型量化
# ==============================================================================

def minmax_1bit_quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    """
    对单个权重张量应用 1-bit Min-Max 量化和反量化。
    """
    # 找到张量的最小值和最大值
    min_val = w.min()
    max_val = w.max()

    # 计算缩放因子 (scale) 和零点 (zero_point)
    # 对于1-bit，量化级别为 {0, 1}
    scale = (max_val - min_val) / 1.0  # (2**1 - 1) = 1
    zero_point = min_val

    # 为了避免除以零（如果张量中所有值都相同）
    if abs(scale) < 1e-9:
        return w

    # 量化到 {0, 1}
    quantized_w = torch.round((w - zero_point) / scale)

    # 反量化回原始范围
    dequantized_w = quantized_w * scale + zero_point
    
    return dequantized_w.to(w.dtype)

def apply_minmax_1bit_quantization(model: nn.Module):
    """
    对一个标准的实数模型应用 Min-Max 1-bit 量化。
    """
    print("正在对实数模型应用 Min-Max (1-bit) 量化...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        # 对权重进行量化和反量化，并替换原始权重
        module.weight.data = minmax_1bit_quantize_dequantize(module.weight.data)

    # 遍历所有模块，只对 nn.Linear 层应用量化
    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    
    print("Min-Max 1-bit 量化应用完成。")
    return model

# ==============================================================================
# 方案四: 对称 Min-Max (1-bit) 实数模型量化 (映射到 -1, 1)
# ==============================================================================

def symmetric_minmax_1bit_quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    """
    对单个权重张量应用对称的 1-bit Min-Max 量化和反量化。
    量化到 {-1, 1}。
    """
    # 找到张量的最大绝对值
    max_abs = w.abs().max()

    # 计算缩放因子 (scale)
    scale = max_abs

    # 为了避免除以零
    if scale < 1e-9:
        return w

    # 量化到 {-1, 1}
    # 先将 w / scale 归一化到 [-1, 1]，然后用 sign() 得到 -1 或 1
    # 注意：为了处理0，我们使用 sign()，它会将0映射为0，但理论上1-bit不应有0。
    # 对于严格的{-1, 1}，可以稍微调整，但 sign() 在实践中很常用。
    quantized_w = (w / scale).sign()
    
    # 反量化回原始范围
    dequantized_w = quantized_w * scale
    
    return dequantized_w.to(w.dtype)

def apply_symmetric_minmax_1bit_quantization(model: nn.Module):
    """
    对一个标准的实数模型应用对称的 Min-Max 1-bit 量化。
    """
    print("正在对实数模型应用对称 Min-Max (1-bit, to {-1, 1}) 量化...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        module.weight.data = symmetric_minmax_1bit_quantize_dequantize(module.weight.data)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    
    print("对称 Min-Max 1-bit 量化应用完成。")
    return model

# ==============================================================================
# QAT (Quantization-Aware Training) 核心函数
# ==============================================================================

class BitNetQuantSTE(torch.autograd.Function):
    """
    用于 BitNet 的 STE。在前向传播中量化，在反向传播中传递梯度。
    """
    @staticmethod
    def forward(ctx, w):
        # BitNet 量化逻辑
        scale = w.abs().mean()
        quantized_w = w.sign() * scale
        return quantized_w

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 直接将梯度传回去
        return grad_output

class PhaseQuantSTE(torch.autograd.Function):
    """
    用于 Complex-Phase 的 STE。
    """
    @staticmethod
    def forward(ctx, w_real, w_imag):
        # iFairy PhaseQuant 逻辑 (与PTQ版本相同)
        phase = torch.angle(w_real + 1j * w_imag)
        
        real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
        real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
        imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
        imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

        mask_real = real_pos | real_neg
        mask_imag = imag_pos | imag_neg
        
        s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
        s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
        
        s_re = torch.clamp(s_re, min=1e-6)
        s_im = torch.clamp(s_im, min=1e-6)
        
        qw_real = torch.zeros_like(w_real)
        qw_imag = torch.zeros_like(w_imag)
        
        qw_real[real_pos] = 1.0
        qw_real[real_neg] = -1.0
        qw_imag[imag_pos] = 1.0
        qw_imag[imag_neg] = -1.0
        
        qw_real_scaled = qw_real * s_re
        qw_imag_scaled = qw_imag * s_im
        
        return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

    @staticmethod
    def backward(ctx, grad_w_real, grad_w_imag):
        # STE: 将两个梯度直接传回去
        return grad_w_real, grad_w_imag
# quantize_complex_tensor(torch.randn(10, 10), torch.randn(10, 10))   