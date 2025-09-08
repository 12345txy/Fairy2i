import math
import argparse
import torch
import random

import numpy as np
from eval_utils import get_test_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--hf_path", type=str)
parser.add_argument("--seqlen", default=2048, type=int)
parser.add_argument(
    "--device", default="cuda:0", type=str, help="device to run the model on"
)


def calulate_loss(model, input, loss_fct):
    output = model(
        input, use_cache=False, output_hidden_states=False, output_attentions=False
    )[0]
    shift_logits = output[:, :-1, :].contiguous()
    shift_labels = input[:, 1:].clone()
    shift_labels[input[:, :-1] == model.config.eos_token_id] = -100
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def main(args):
    device = args.device
    datasets = ["wikitext2", "c4"]
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        device_map=device,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)

    model = model.eval()
    print(model.dtype)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum").to(device)

    ppl = []
    for dataset in datasets:
        testdata = get_test_dataset(dataset, tokenizer, seqlen=args.seqlen)
        acc_loss, count = 0.0, 0
        progress = tqdm(range(len(testdata)))
        for ii in progress:
            input = torch.Tensor(testdata[ii]).long().view(1, -1)
            input = input.to(device)
            loss = calulate_loss(model, input, loss_fct)
            count += input.size(-1) - 1
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/ count / math.log(2)}")

        avg_loss = acc_loss / count / math.log(2)
        ppl.append(2**avg_loss)
        print("{} PPL: {}".format(dataset, ppl[-1]))

    print(ppl)
    print("Avg PPL:", sum(ppl) / len(ppl))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)