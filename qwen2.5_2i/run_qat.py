import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # 1. 导入 SummaryWriter

from qat_modules import replace_modules_for_qat
# 我们需要 PTQ 函数在训练结束后“固化”权重
from quantization import apply_bitnet_quantization, apply_complex_inspired_quantization

def train_one_epoch(model, dataloader, optimizer, device, writer, epoch):
    """执行一个 epoch 的微调训练。"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"QAT Epoch {epoch+1}")
    
    for i, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        # Causal LM 的 labels 就是 input_ids 本身
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        total_loss += current_loss
        progress_bar.set_postfix({'loss': current_loss})
        
        # 3. 在训练循环中记录 Loss
        # 计算全局步数
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/train_step', current_loss, global_step)
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
    # 记录每个 epoch 的平均 loss
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch + 1)

def main():
    parser = argparse.ArgumentParser(description="对 Hugging Face 模型进行量化感知训练 (QAT)。")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--quant_method", type=str, required=True, choices=["complex_phase", "bitnet"])
    parser.add_argument("--output_path", type=str, required=True, help="保存最终量化模型的文件夹路径。")
    parser.add_argument("--dataset", type=str, default="wikitext", help="用于微调的数据集名称。")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="数据集的具体配置。")
    parser.add_argument("--epochs", type=int, default=1, help="微调的 Epoch 数量。")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率。")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小。")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度。")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备。")
    args = parser.parse_args()

    # --- 1. 加载模型和分词器 ---
    print(f"加载模型: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    qat_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 # 使用 bfloat16 进行训练
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 将模型转换为 QAT-ready 格式 ---
    print(f"将模型转换为 QAT 格式 ({args.quant_method})...")
    replace_modules_for_qat(qat_model, args.quant_method)
    qat_model.to(args.device)

    # --- 3. 准备数据集 ---
    print(f"加载数据集: {args.dataset}")
    raw_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_len, padding="max_length")
        
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch")
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    # --- 4. 设置优化器、TensorBoard Writer 并开始训练 ---
    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=args.lr)
    
    # 2. 初始化 SummaryWriter
    # 创建一个有意义的日志目录名，例如 'runs/bitnet_qwen2-0.5B-bitnet-qat'
    log_dir_name = f"{args.quant_method}_{os.path.basename(args.output_path)}"
    writer = SummaryWriter(log_dir=os.path.join('runs', log_dir_name))
    
    print("开始 QAT 微调... 日志将记录在 'runs' 目录下。")
    for epoch in range(args.epochs):
        train_one_epoch(qat_model, dataloader, optimizer, args.device, writer, epoch)
    
    writer.close() # 训练结束后关闭 writer
    
    # --- 5. 固化权重并保存最终模型 ---
    print("微调完成。正在固化权重并创建最终模型...")
    # a. 加载一个干净的原始模型结构
    final_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32 # 加载为FP32以进行精确固化
    )
    # b. 将微调后的浮点权重从 QAT 模型复制过来
    final_model.load_state_dict(qat_model.state_dict())
    
    # c. 在这个包含了微调后权重的标准模型上，执行最后一次“伪量化”
    if args.quant_method == 'bitnet':
        final_model = apply_bitnet_quantization(final_model)
    elif args.quant_method == 'complex_phase':
        final_model = apply_complex_inspired_quantization(final_model)
    
    final_model.to('cpu')

    # d. 保存最终的、已量化的模型
    print(f"正在保存最终模型到: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    final_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("QAT 完成！")

if __name__ == "__main__":
    main()
