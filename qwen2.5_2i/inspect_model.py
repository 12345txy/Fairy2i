import torch
import argparse
from transformers import AutoModelForCausalLM
import sys
import os

# 一个辅助类，用于将输出同时重定向到控制台和文件
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

def main():
    parser = argparse.ArgumentParser(description="Inspect the layers and weights of a Hugging Face model.")
    parser.add_argument(
        "model_path", 
        type=str,
        help="Path to the Hugging Face model directory or a model name from the Hub (e.g., 'Qwen/Qwen2-0.5B')."
    )
    parser.add_argument(
        "--filter", 
        type=str, 
        default="weight", 
        help="Only show parameters containing this string in their name. Default is 'weight'."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the output to the first N matching layers. Default is to show all."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional. Path to a file to save the output. (e.g., ./results/model_weights.txt)"
    )
    args = parser.parse_args()

    # --- 获取模型路径 ---
    model_path = args.model_path
    # 检查路径是否为本地存在的目录，如果是，则转换为绝对路径以避免歧义
    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
        print(f"检测到本地路径，将使用绝对路径: {model_path}")


    # --- 设置输出重定向 ---
    original_stdout = sys.stdout
    if args.output_file:
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            sys.stdout = Logger(args.output_file, original_stdout)
        except IOError as e:
            sys.stdout = original_stdout
            print(f"错误：无法打开输出文件 {args.output_file}: {e}")
            sys.exit(1)

    print(f"正在从 '{model_path}' 加载模型...")
    
    try:
        # 尝试以 bfloat16 加载以节省内存，如果失败则使用默认精度
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    except Exception:
        print("以 bfloat16 加载失败，尝试使用默认精度...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
    model.eval()
    print("模型加载成功！\n" + "="*50)

    count = 0
    for name, param in model.named_parameters():
        if args.filter in name:
            if args.limit is not None and count >= args.limit:
                print(f"\n已达到显示上限 ({args.limit} 层)...")
                break
            
            print(f"层名称 (Layer Name): {name}")
            print(f"  - 形状 (Shape): {param.shape}")
            print(f"  - 数据类型 (Dtype): {param.dtype}")
            
            # 计算并打印统计数据
            param_cpu_f32 = param.data.to(device='cpu', dtype=torch.float32)
            print(f"  - 均值 (Mean):   {param_cpu_f32.mean().item():.6f}")
            print(f"  - 标准差 (Std):  {param_cpu_f32.std().item():.6f}")
            print(f"  - 最小值 (Min):    {param_cpu_f32.min().item():.6f}")
            print(f"  - 最大值 (Max):    {param_cpu_f32.max().item():.6f}")
            print("-" * 40)
            count += 1

    # --- 恢复原始输出并关闭文件 ---
    if args.output_file:
        sys.stdout = original_stdout
        print(f"\n结果已成功保存到: {args.output_file}")


if __name__ == "__main__":
    main()
