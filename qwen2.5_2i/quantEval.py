import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def evaluate(model, tokenizer, dataset_name, dataset_config, device, limit_samples=1000):
    """
    使用滑动窗口方法在指定数据集上评测模型的 Perplexity。
    """
    print(f"正在加载数据集: {dataset_name} (配置: {dataset_config})...")
    
    # 加载测试集
    # 对于C4这类大型数据集，使用流式处理(streaming)可以避免下载整个数据集
    if dataset_name == "c4":
        data = load_dataset(dataset_name, dataset_config, split="validation", streaming=True)
    else:
        # WikiText较小，可以直接加载
        data = load_dataset(dataset_name, dataset_config, split="test")

    # 拼接所有文本
    if dataset_name == "wikitext":
        text = "\n\n".join(data["text"])
    elif dataset_name == "c4":
        print(f"C4数据集较大，将使用前 {limit_samples} 个样本进行评测...")
        text_list = []
        # 从流式数据集中取指定数量的样本
        for i, row in enumerate(data):
            if i >= limit_samples:
                break
            text_list.append(row['text'])
        text = "\n\n".join(text_list)
        if not text:
            print("错误：未能从C4数据集中加载任何文本。请检查网络连接或数据集状态。")
            return float('inf')
        
    print("正在进行分词...")
    encodings = tokenizer(text, return_tensors="pt")
    
    # 定义滑动窗口参数
    # 使用一个固定的、合理的窗口大小，而不是模型支持的最大长度，以避免OOM
    max_length = 2048
    stride = 512 # 滑动步长
    seq_len = encodings.input_ids.size(1)

    nlls = [] # Negative Log-Likelihoods
    prev_end_loc = 0

    print("开始计算 Perplexity...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # 我们只计算新窗口部分的损失 (masked language modeling)
        target_ids[:, :-trg_len] = -100

        if input_ids.size(1) == 0:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # outputs.loss 是这个窗口的平均损失，乘以目标长度得到总损失
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    # 计算整体的 Perplexity
    if not nlls or end_loc == 0:
        return float('inf')
        
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description="评测量化后模型的 Perplexity。")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2-0.5B",
        help="基础模型的Hugging Face名称，用于加载正确的架构和分词器。"
    )
    parser.add_argument(
        "--quantized_model_path", 
        type=str, 
        default=None, # 改为可选
        help="指向量化后的 .pth 权重文件的路径。如果未提供，则直接评测原始模型。"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="wikitext", 
        choices=["wikitext", "c4"], 
        help="用于评测的数据集。"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="运行评测的设备。"
    )
    args = parser.parse_args()

    # 1. 加载分词器
    print(f"正在加载分词器: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 2. 加载一个标准的、未经训练的模型架构
    print(f"正在加载模型架构: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    # `from_config` 会创建一个随机初始化的模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

    # 3. 如果提供了量化权重路径，则加载它
    if args.quantized_model_path:
        print(f"正在从 '{args.quantized_model_path}' 加载量化权重...")
        model.load_state_dict(torch.load(args.quantized_model_path, map_location="cpu"), strict=False)
    else:
        print("未提供量化权重路径，将直接评测原始的预训练模型。")

    model.to(args.device)
    model.eval()

    # 4. 选择数据集配置并开始评测
    dataset_configs = {
        "wikitext": ("wikitext", "wikitext-2-raw-v1"),
        "c4": ("allenai/c4", "en")
    }
    dataset_key, dataset_config_name = dataset_configs[args.dataset]
    
    perplexity = evaluate(model, tokenizer, dataset_key, dataset_config_name, args.device)
    
    print("\n" + "=" * 50)
    print(f"评测完成!")
    print(f"  模型权重: {args.quantized_model_path}")
    print(f"  数据集: {args.dataset}")
    print(f"  Perplexity: {perplexity:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
