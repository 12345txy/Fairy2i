import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def apply_reconstruction(model: nn.Module):
    """
    对模型的所有线性层应用“分解->重构”操作，以测试浮点误差。
    中间不进行任何量化。
    """
    print("正在对所有线性层应用分解-重构操作...")
    
    @torch.no_grad()
    def reconstruct_linear_layer(module: nn.Linear):
        A = module.weight.data
        # 只有偶数维度的矩阵才能被分解
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

        # 2. 直接重构 (Reconstruct immediately)
        A11_r = W_re + U_re
        A12_r = W_im - U_im
        A21_r = W_im + U_im
        A22_r = -W_re + U_re

        A_recon_top = torch.cat([A11_r, A12_r], dim=1)
        A_recon_bottom = torch.cat([A21_r, A22_r], dim=1)
        A_reconstructed = torch.cat([A_recon_top, A_recon_bottom], dim=0)

        # 3. 用重构后的权重替换原始权重
        module.weight.data = A_reconstructed.to(A.dtype)

    model.apply(lambda module: reconstruct_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("分解-重构操作完成。")
    return model

def main():
    parser = argparse.ArgumentParser(description="创建一个仅包含分解-重构浮点误差的模型")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2-0.5B", help="基础的 Hugging Face 模型名称")
    parser.add_argument("--output_dir", type=str, default="./hf_models/reconstructed_fp_error", help="保存新模型的目录")
    args = parser.parse_args()

    print(f"正在从 '{args.base_model_name}' 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # 应用变换
    model = apply_reconstruction(model)

    print(f"正在将变换后的模型保存到 '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("模型保存完毕。")


if __name__ == "__main__":
    main()
