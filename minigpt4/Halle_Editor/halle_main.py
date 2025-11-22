import torch
from typing import Optional, Union, List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from .losses import masked_log_probs,kl_loc_loss,log_probs
from torchvision import transforms
import numpy as np
import os
import time
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)
from torchvision import transforms
def get_edit_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -100)


def execute(model: AutoModelForCausalLM,
            hparams,
            requests,
            tok,
            pope,
            args,
            ):
    print("=======editing model=========")
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # 先取出对应层的权重
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layers in hparams.layers# 取出对应那层的mlp第二个的down_proj
        if hparams.rewrite_module_tmp.format(layers) in n
    }

    # 保存旧权重
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")
    # 设置优化器和梯度
    opt = torch.optim.SGD(
        [v for _, v in weights.items()],
        # model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights  # 只有需要更改的权重才会被设置为True
        # general knowledge constraint
        # loss constraint, 包括有幻觉和无幻觉的两个表示，取出没有幻觉的完整内容以及单独的回答
    #-------------------初始模型下logits的输出--------------------#
    with torch.no_grad():
        instructonlyAns = tok(
                [requests["target"][id] for id in requests["id"]],
                 padding=True, return_tensors="pt"
            )
          # torch.Size([1, 59]) 
        # instructblip 是 [1,352]
    bs = 10
    start_time = time.time()
    for i in range(0, len(requests["id"]), bs):
        batch_id = requests["id"][i:i + bs]
        image = [norm(requests["image"][id]) for id in batch_id]
        # images放到device上
        images = torch.cat(image, dim=0).to(device)  
        if pope:
            ft_input = [requests["prompt"][id] + " " + requests["target"][id] for id in batch_id]
        else:
            ft_input = [requests["prompt"]["qu_image"] + " " + requests["target"][id] for id in batch_id]
        print(f"Batch {i // bs + 1}: {batch_id}")
        with torch.no_grad():
            instruction_output_Logits,_ = model.generate(
                {"image": images, "prompt": ft_input},
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,
                max_new_tokens=512,
                output_attentions=True,
                opera_decoding=False,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                generate = False
            )
        instruction_output_Logits = instruction_output_Logits.to("cpu")

        if i == 0:
            instruction_base_Logits = instruction_output_Logits[:,-instructonlyAns.attention_mask.size(1):]
            # instruction_base_Logits = instruction_output_Logits[:,-210:]
        else:
            instruction_base_Logits = torch.cat((instruction_base_Logits,instruction_output_Logits[:,
                                  -instructonlyAns.attention_mask.size(1):]),dim=0)  # torch.Size([1, 59, 32000])
            # instruction_base_Logits = torch.cat((instruction_base_Logits,instruction_output_Logits[:,
            #                       -210:]),dim=0)  # torch.Size([1, 59, 32000])
    torch.cuda.empty_cache() # 清除最后一个batch占用的显存
    end_time = time.time()
    # print(f"计算初始logits运行时间: {end_time - start_time:.6f} 秒")
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        opt.zero_grad()
        # np.random.shuffle(id)
        # 按顺序每次取5个
        bs = 2
        start_time = time.time()
        for i in range(0, len(requests["id"]), bs):
            batch_id = requests["id"][i:i + bs]
            print(f"Batch {i // bs + 1}: {batch_id}")
            # 编辑halle区域
            # 这里是得到有幻觉的完整的logits输出
            image = [norm(requests["image"][id]) for id in batch_id]
            images = torch.cat(image, dim=0).to(device)
            #-------------------正样本的loss-----------------------#
            # 改成插入soft_prompt
            # 读取soft prompt
            soft_prompt = torch.load('/home/hjy/work/opera/edited_model/llava_next_2_token_5.pt')
            if pope:
                ft_input = [requests["prompt"][id] + " " + requests["target"][id] for id in batch_id]
            else:
                ft_input = [requests["prompt"]["qu_halle"] + " " + requests["target"][id] for id in batch_id]
            pos_logits,_ = model.generate(
                {"image": images, "prompt": ft_input},
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,
                max_new_tokens=512,
                output_attentions=True,
                opera_decoding=False,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                generate = False,
                prompt_learn = True,
                prompt_vec = soft_prompt
                )
            labels_ids = tok([requests["target"][id] for id in batch_id],return_tensors="pt", padding=True).to(device).input_ids
            out_labels = get_edit_labels(tok,labels_ids)
            pos_loss = masked_log_probs(hparams, pos_logits, out_labels, shift=True)["nll"]
            #-------------------负样本的loss-----------------------#
            # neg_input = [requests["prompt"]["qu_halle"] + " " + requests["halle"][id] for id in batch_id]
            # with torch.no_grad():
            #     neg_logits,_ = model.generate(
            #         {"image": images, "prompt": neg_input},
            #         use_nucleus_sampling=args.sample,
            #         num_beams=args.beam,
            #         max_new_tokens=512,
            #         output_attentions=True,
            #         opera_decoding=False,
            #         scale_factor=args.scale_factor,
            #         threshold=args.threshold,
            #         num_attn_candidates=args.num_attn_candidates,
            #         penalty_weights=args.penalty_weights,
            #         generate = False,
            #     )
            # neg_labels_ids = tok([requests["halle"][id] for id in batch_id],return_tensors="pt", padding=True).to(device).input_ids
            # neg_labels = get_edit_labels(tok,neg_labels_ids)
            # neg_loss = masked_log_probs(hparams, 1-0.5*neg_logits, neg_labels, shift=True)["nll"]
            #-------------------限制样本的loss-----------------------#
            if pope:
                kl_input = [requests["prompt"][id] + " " + requests["target"][id] for id in batch_id]
            else:
                kl_input = [requests["prompt"]["qu_image"] + " " + requests["target"][id] for id in batch_id]
            # kl_input = [requests["prompt"][id] + " " + requests["target"][id] for id in batch_id]
            # with torch.no_grad():
            kl_logits,_ = model.generate(
                {"image": images, "prompt": kl_input},
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,
                max_new_tokens=512,
                output_attentions=True,
                opera_decoding=False,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                generate = False,
            )
            kl_mask = instructonlyAns.attention_mask[i:i + bs].to(device) #取mask长度即可
            # 以无幻觉的回答长度作为比较长度
            if kl_mask.size(1) != kl_logits.size(1):  # torch.Size([1, 59, 32000])
                kl_logits = kl_logits[:, -kl_mask.size(1):]  # torch.Size([1, 59, 32000])
            base_logits = instruction_base_Logits[i:i + bs].to(device)
            l_loc_instruction = kl_loc_loss(base_logits, kl_logits, mask=kl_mask)  # tensor 一个值 0
            loss = pos_loss*0.2+l_loc_instruction
            if loss.item() >= 1e-4:
                loss.backward()
                opt.step()
                torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
                break
        end_time = time.time()
        print(f"计算一个Epoch的运行时间: {end_time - start_time:.6f} 秒")


    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
        # 此时的weights又是原来的权重了

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas