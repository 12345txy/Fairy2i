import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import copy

class ComplexLinear(nn.Module):
    """
    一个自定义的线性层，其内部使用分解后的四个实数矩阵
    (U_re, U_im, W_re, W_im) 来模拟一个等效的、更大的实数线性层。
    """
    def __init__(self, original_linear_layer: nn.Linear):
        super().__init__()

        # 检查原始层的维度是否为偶数
        out_features, in_features = original_linear_layer.weight.shape
        if in_features % 2 != 0 or out_features % 2 != 0:
            raise ValueError(f"原始线性层的输入和输出维度必须是偶数，"
                             f"但得到的是 in:{in_features}, out:{out_features}")

        self.in_features = in_features
        self.out_features = out_features
        
        # n 和 m 的维度定义
        # 权重 A 的形状为 (out_features, in_features) = (2n, 2m)
        self.n = out_features // 2
        self.m = in_features // 2

        # 提取原始权重和偏置
        A = original_linear_layer.weight.data
        
        # 1. --- 数学转换 ---
        # 将 A 划分为四个 n x m 的子块
        A11 = A[:self.n, :self.m]
        A12 = A[:self.n, self.m:]
        A21 = A[self.n:, :self.m]
        A22 = A[self.n:, self.m:]

        # 根据公式计算 U 和 W 的实部和虚部
        U_re_val = 0.5 * (A11 + A22)
        U_im_val = 0.5 * (A21 - A12)
        W_re_val = 0.5 * (A11 - A22)
        W_im_val = 0.5 * (A12 + A21)
        
        # 2. --- 将转换后的矩阵注册为模型参数 ---
        # 这样它们就可以像普通权重一样被训练、保存和加载
        self.U_re = nn.Parameter(U_re_val)
        self.U_im = nn.Parameter(U_im_val)
        self.W_re = nn.Parameter(W_re_val)
        self.W_im = nn.Parameter(W_im_val)

        # 处理偏置项（如果存在）
        if original_linear_layer.bias is not None:
            self.bias = nn.Parameter(original_linear_layer.bias.data)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现 Y = A @ X.T 的等效计算
        """
        # 输入 x 的形状是 (..., in_features) or (..., 2*m)
        # 将输入的最后一个维度拆分为实部和虚部
        x_re = x[..., :self.m]
        x_im = x[..., self.m:]

        # 根据推导的公式计算输出的实部和虚部
        # 注意：PyTorch nn.Linear 的实现是 x @ W.T
        # 所以我们这里也需要对权重矩阵进行转置
        y_re = (x_re @ self.U_re.T - x_im @ self.U_im.T +
                x_re @ self.W_re.T + x_im @ self.W_im.T)
        
        y_im = (x_re @ self.U_im.T + x_im @ self.U_re.T +
                x_re @ self.W_im.T - x_im @ self.W_re.T)

        # 将输出的实部和虚部拼接起来
        output = torch.cat([y_re, y_im], dim=-1)

        # 添加偏置项
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return (f"ComplexEquivalentLinear(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None}, "
                f"n={self.n}, m={self.m})")

def convert_model(module: nn.Module):
    """
    递归遍历并替换所有符合条件的 nn.Linear 层。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            out_f, in_f = child.weight.shape
            if in_f % 2 == 0 and out_f % 2 == 0:
                print(f"  -> 正在转换: {name} (Linear, in:{in_f}, out:{out_f})")
                setattr(module, name, ComplexLinear(child))
            else:
                print(f"  -> 跳过: {name} (Linear, 维度非偶数)")
        # 递归进入子模块
        else:
            convert_model(child)
    
# 1. 加载模型和分词器
model_name = "Qwen/Qwen2-0.5B" # 该模型的结构与您提供的权重列表匹配
print(f"正在加载模型: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
original_model.eval()

# 2. 复制模型以进行转换
converted_model = copy.deepcopy(original_model)

# 3. 执行转换
print("\n--- 开始遍历并转换模型 ---")
convert_model(converted_model)
print("--- 转换完成 ---\n")

# print("转换后的模型结构:", converted_model)
# with open("converted_model_weights.txt", "w") as f:
#     for name, param in converted_model.state_dict().items():
#         f.write(f"{name}: {param.shape}, dtype: {param.dtype}\n")

# 4. 准备输入并进行验证
text = "The capital of the United States is"
inputs = tokenizer(text, return_tensors="pt")

print(f"输入文本: '{text}'")

# 原始模型推理
with torch.no_grad():
    logits_original = original_model(**inputs).logits

# 转换后模型推理
with torch.no_grad():
    logits_converted = converted_model(**inputs).logits

# 5. 比较结果
are_equal = torch.allclose(logits_original, logits_converted, atol=1e-5)
max_diff = torch.max(torch.abs(logits_original - logits_converted))

print(f"\n原始模型与转换后模型的输出是否一致? {'是' if are_equal else '否'}")
print(f"输出 Logits 之间的最大绝对误差: {max_diff.item():.2e}")
