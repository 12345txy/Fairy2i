import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization import (
    apply_complex_inspired_quantization, 
    apply_bitnet_quantization, 
    apply_minmax_1bit_quantization,
    apply_symmetric_minmax_1bit_quantization
)
import os

def main():
    parser = argparse.ArgumentParser(description="对一个标准的实数Hugging Face模型应用量化。")
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="Qwen/Qwen2-0.5B",
        help="Hugging Face 上的模型名称或本地路径。"
    )
    parser.add_argument(
        "--quant_method", 
        type=str, 
        required=True, 
        choices=["complex_phase", "bitnet", "minmax_1bit", "symmetric_minmax_1bit"],
        help="要应用的量化方法: 'complex_phase', 'bitnet', 'minmax_1bit', 或 'symmetric_minmax_1bit'."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="保存量化后 Hugging Face 模型格式的文件夹路径。"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="运行量化的设备 (e.g., 'cuda', 'cpu')."
    )

    args = parser.parse_args()
    
    # 检查路径是否为本地存在的目录，如果是，则转换为绝对路径以避免歧义
    model_path_arg = args.model_name_or_path
    if os.path.isdir(model_path_arg):
        model_path_arg = os.path.abspath(model_path_arg)
        print(f"检测到本地路径，将使用绝对路径: {model_path_arg}")

    # --- 加载原始的实数模型 ---
    print(f"正在从 Hugging Face 加载原始模型: {model_path_arg}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path_arg,
        torch_dtype=torch.float32, # 使用float32以获得最高精度
        device_map=args.device
    )
    model.eval()

    # --- 根据选择应用量化方法 ---
    if args.quant_method == "complex_phase":
        quantized_model = apply_complex_inspired_quantization(model)
    elif args.quant_method == "bitnet":
        quantized_model = apply_bitnet_quantization(model)
    elif args.quant_method == "minmax_1bit":
        quantized_model = apply_minmax_1bit_quantization(model)
    elif args.quant_method == "symmetric_minmax_1bit":
        quantized_model = apply_symmetric_minmax_1bit_quantization(model)
    else:
        raise ValueError("无效的量化方法。")

    # --- 保存量化后的模型为 Hugging Face 格式 ---
    print(f"正在将量化后的模型保存为 Hugging Face 格式到: {args.output_path}")
    
    # 1. 保存模型权重和配置文件
    quantized_model.save_pretrained(args.output_path)

    # 2. 加载并保存对应的分词器
    print(f"正在保存分词器到: {args.output_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(args.output_path)

    print("保存完成！")


if __name__ == "__main__":
    main()
