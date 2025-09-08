import torch
import argparse
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="将 .pth 权重文件保存为 Hugging Face 模型格式。")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="基础模型的Hugging Face名称，用于加载正确的架构。"
    )
    parser.add_argument(
        "--pth_path",
        type=str,
        required=True,
        help="指向 .pth 权重文件的路径。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="保存为Hugging Face格式的输出目录。"
    )
    args = parser.parse_args()
    print(f"1. 正在加载基础模型架构: {args.base_model_name}")
    config = AutoConfig.from_pretrained(args.base_model_name)
    model = AutoModelForCausalLM.from_config(config)

    # 在这里添加加载分词器
    print(f"2. 正在加载分词器: {args.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    print(f"3. 正在从 '{args.pth_path}' 加载模型权重...")
    model.load_state_dict(torch.load(args.pth_path, map_location="cpu"))
    model.eval()

    print(f"4. 正在将模型保存到 Hugging Face 格式目录: {args.output_dir}")
    model.save_pretrained(args.output_dir)

    # 在这里添加保存分词器
    print(f"5. 正在将分词器保存到 Hugging Face 格式目录: {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n转换完成！现在您可以使用以下路径进行评测:\n{args.output_dir}")

if __name__ == "__main__":
    main()
