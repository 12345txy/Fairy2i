import torch

data_type = torch.bfloat16

def convert_real_to_complex_weights(A_real: torch.Tensor) -> dict:
    """
    根据提供的数学公式，将一个大的实数权重矩阵 A 分解为
    ComplexLinear 层所需的 U_real, U_imag, W_real, W_imag 四个矩阵。

    参数:
        A_real (torch.Tensor): 形状为 (2n, 2m) 的原始实数权重矩阵。

    返回:
        dict: 包含四个张量的字典: {'U_real', 'U_imag', 'W_real', 'W_imag'}。
    """
    # 验证输入矩阵的维度是否为偶数
    if A_real.dim() != 2 or A_real.shape[0] % 2 != 0 or A_real.shape[1] % 2 != 0:
        raise ValueError("A must be a 2D real tensor with shape (2n, 2m).")
    if not A_real.dtype.is_floating_point:
        raise ValueError(
            "A must be a real-valued tensor with a floating-point data type."
        )

    # 计算 n 和 m
    n = A_real.shape[0] // 2
    m = A_real.shape[1] // 2

    # 步骤 1: 将 A 分成 2x2 的块
    A11 = A_real[:n, :m]
    A12 = A_real[:n, m:]
    A21 = A_real[n:, :m]
    A22 = A_real[n:, m:]

    # 步骤 2: 应用分解/还原公式
    U_re = 0.5 * (A11 + A22)
    U_im = 0.5 * (A21 - A12)
    W_re = 0.5 * (A11 - A22)
    W_im = 0.5 * (A12 + A21)

    return {
        'U_real': U_re,
        'U_imag': U_im,
        'W_real': W_re,
        'W_imag': W_im,
    }

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collections import OrderedDict

real_model = AutoModelForCausalLM.from_pretrained("1bitLLM/bitnet_b1_58-large")
real_state_dict = real_model.state_dict()
real_config = AutoConfig.from_pretrained("1bitLLM/bitnet_b1_58-large")

complex_model = AutoModelForCausalLM.from_pretrained("PKU-DS-LAB/Fairy-plus-minus-i-700M", trust_remote_code=True)
complex_state_dict = complex_model.state_dict()

print(real_model)
print(real_model.config)

num_layers = real_config.num_hidden_layers
new_state_dict = OrderedDict()

hidden_dim = real_config.hidden_size // 2
new_state_dict["token_embedding_real"] = real_state_dict["model.embed_tokens.weight"][..., :hidden_dim].to(data_type)
new_state_dict["token_embedding_imag"] = real_state_dict["model.embed_tokens.weight"][..., hidden_dim:].to(data_type)

for i in range(num_layers):
    print(f"Processing layer {i}")
    layer_mapping = {
        f"layer.{i}.self_attn.q_proj": f"model.layers.{i}.self_attn.q_proj.weight",
        f"layer.{i}.self_attn.k_proj": f"model.layers.{i}.self_attn.k_proj.weight",
        f"layer.{i}.self_attn.v_proj": f"model.layers.{i}.self_attn.v_proj.weight",
        f"layer.{i}.self_attn.o_proj": f"model.layers.{i}.self_attn.o_proj.weight",
        f"layer.{i}.mlp.gate_proj":f"model.layers.{i}.mlp.gate_proj.weight",
        f"layer.{i}.mlp.up_proj":f"model.layers.{i}.mlp.up_proj.weight",
        f"layer.{i}.mlp.down_proj":f"model.layers.{i}.mlp.down_proj.weight",
    }
    for complex_prefix, real_layer_name in layer_mapping.items():
        A_real = real_state_dict[real_layer_name]
        converted_complex_dict = convert_real_to_complex_weights(A_real)
        new_state_dict[f"{complex_prefix}.U_real"] = converted_complex_dict['U_real'].to(data_type)
        new_state_dict[f"{complex_prefix}.U_imag"] = converted_complex_dict['U_imag'].to(data_type)
        new_state_dict[f"{complex_prefix}.W_real"] = converted_complex_dict['W_real'].to(data_type)
        new_state_dict[f"{complex_prefix}.W_imag"] = converted_complex_dict['W_imag'].to(data_type)

with open("converted_model_weights.txt", "w") as f:
    for name, param in new_state_dict.items():
        f.write(f"{name}: {param.shape}, dtype: {param.dtype}\n")
