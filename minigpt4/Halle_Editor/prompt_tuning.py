import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from .hparams import HyperParams
from torchvision import transforms
import json
import os
import tqdm
from .losses import masked_log_probs,kl_loc_loss,log_probs
import gc
from torch.cuda.amp import autocast
def get_edit_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -100)

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)


def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd

base_dir  = "./log/chair/" 

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"显存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def prefix_tuning(model,requests,tok,hparams,device,pope,args):
    print("=======prefix_tuning=======")
    print_gpu_memory()

    # 启用混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    if pope:
        prefix_prompt = requests["prompt"]
    else:
        prefix_prompt = requests["prompt"]["qu_norm"]
    token_dim = model.llama_model.config.hidden_size 
    PROMPT_TOKENS = 5
    soft_prompt = nn.Parameter(torch.randn(1,PROMPT_TOKENS, token_dim, device=device))

    # 冻结模型所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 只训练soft prompt参数
    optimizer = torch.optim.AdamW([soft_prompt], lr=1e-3)

    # target_dict = {}
    # with open("dataset/train/halle_1500.jsonl", "r", encoding="utf-8") as f:
    #     for line in f:
    #         data = json.loads(line) 
    #         target_dict[data["image_id"]]=data["caption"]
    target_dict = requests["halle"]
    
    bs = 4 # chair
    # bs = 10
    for epoch in range(3):
        print(20 * "=")
        print(f"Epoch: {epoch}")
        print(20 * "=")
        
        for i in range(0, len(requests["id"]), bs):
            batch_id = requests["id"][i:i + bs]
            print(f"Batch {i // bs + 1}: {batch_id}")
            
            # 使用no_grad处理图像
            with torch.no_grad():
                image = [norm(requests["image"][id]) for id in batch_id]
                images = torch.cat(image, dim=0).to(device)
            ft_input = [prefix_prompt + " " + target_dict[id] for id in batch_id]
            # 使用混合精度训练
            with autocast():
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
                    prompt_vec = soft_prompt,
                    )
            labels_ids = tok([target_dict[id] for id in batch_id],return_tensors="pt", padding=True).to(device).input_ids
            out_labels = get_edit_labels(tok,labels_ids)
            pos_loss = masked_log_probs(hparams, pos_logits, out_labels, shift=True)["nll"]
            print("pos_loss:",pos_loss.item())
            if pos_loss.item() >= 1e-4:
                pos_loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
                break
            # 清理显存
            del pos_logits
            
        # 每个epoch结束后保存一次
        torch.save(soft_prompt, os.path.join(args.results_save_dir, f'llava_next_{epoch}_token_5.pt'))
        print(f"已保存第{epoch}个epoch的soft_prompt")

    

        
    