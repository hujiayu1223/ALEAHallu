import torch
from typing import Optional, Union, List, Tuple, Dict
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
class hall_editor:
    def __init__(self,HyperParams,model):
        self.hparams = HyperParams
        self.model = model

    def _locate_halle_layer_for_opera(self,num_layers,hidden_states):
        # if isinstance(tokenizer, LlamaTokenizer):
        #     tokenizer.padding_side = 'right'
        # else:
        #     tokenizer.padding_side = 'left'
        toxic_layer = []
        max_distance_layer = None
        max_distance_value = float('-inf')
        for idx_layer in range(1,num_layers+1): #由于第0层是输入的嵌入因此不考虑这时候的差异
            print(idx_layer)
            euclidean_distance = torch.dist(hidden_states[idx_layer][0,:,:], hidden_states[idx_layer][1,:,:],
                                            p=2)

            if euclidean_distance.item() > max_distance_value:
                max_distance_value = euclidean_distance.item()
                max_distance_layer = idx_layer
        return max_distance_layer-1 # -1是为了对应layers的下标

    def get_edit_labels(self,tok, labels):
        return labels.masked_fill(labels == tok.pad_token_id, -100)

    def execute(self,
                model: AutoModelForCausalLM,
                tok: AutoTokenizer,
                request: List[str],
                hparams,
                ):
        # 这里还需要一个input_ids

        # 先取出对应层的权重
        weights = {
            n: print
            for n,p in model.named_parameters()
            for layer in hparams.layers
            if hparams.rewrite_module_tmp.format(layer) in n
        }

        # 保存旧权重
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}
        print(f"Weights to be updated: {list(weights.keys())}")

        #设置优化器和梯度
        opt = torch.optim.Adam(
            [v for _, v in weights.items()],
            lr = hparams.lr,
            weight_decay = hparams.weight_decay,
        )
        for name, w in model.named_parameters():
            w.requires_grad = name in weights # 只有需要更改的权重才会被设置为True
        return

        # loss constraint, 包括有幻觉和无幻觉的两个表示，取出没有幻觉的完整内容以及单独的回答
        instruction_TextsandTargets = [r["prompt"]+r["ground_truth"] for r in request] # 每组request的第一条都是没有幻觉的完整内容
        with torch.no_grad():
            instructandAns = dict(
                tok(
                    instruction_TextsandTargets,
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)  # torch.Size([1, 148])
            )
            instructonlyAns = dict(
                tok(
                    [r["ground_truth"] for r in requests],
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)
            )  # torch.Size([1, 59])
        instruction_base_Logits = model(**instructandAns).logits  # (B, L, D) (1,148,32000)
        instruction_base_Logits = instruction_base_Logits[:,
                                  -instructonlyAns["attention_mask"].size(1):]  # torch.Size([1, 59, 32000])单独取出回答部分的logits


        # 编辑区域
        ft_input = [r["prompt"]+r["halle"] for r in request]
        out_ids = dict(tok(request["halle"], return_tensors="pt", padding=True).to(device))  #torch.Size([1, 69])
        out_labels = get_edit_labels(tok,out_ids["input_ids"])

        for it in range(hparams.num_steps):
            print(20 * "=")
            print(f"Epoch: {it}")
            print(20 * "=")
            inputs = tok(ft_input,return_tensor="pt",padding=True).to(device)
            opt.zero_grad()
            output = model(**inputs).logits
            loss_dict = masked_log_probs(hparams,output,out_labels,shift =True)
            l_edit = loss_dict["nll"] # 这个是啥?????
            with torch.no_grad():
                post_logits = model(**instructandAns).logits
            kl_mask = instructonlyAns["attention_mask"]
            if kl_mask.size(1) != post_logits.size(1):  # torch.Size([1, 59, 32000])
                post_logits = post_logits[:, -kl_mask.size(1):]  # torch.Size([1, 59, 32000])
            l_loc_intruction = kl_loc_loss(instruction_base_Logits.detach(), post_logits, mask=kl_mask)
            loss = hparams.kl_factor * l_edit + l_loc_instruction

            print(f"Batch loss {loss.item()}, loss_edit*0.1:{0.1 * l_edit}, loss_loc_instruction:{l_loc_instruction}")

    def apply_edit(self,
                   model: AutoModelForCausalLM,
                   tok: AutoTokenizer,
                   hparams,
                   keep_original_weight=False,
                   ):
        weights_copy = {}
        deltas = self.execute(model,tok,hparams)

        return

    def edit(self,
             model: AutoModelForCausalLM,
             tok: AutoTokenizer,
             hparams: Optional[str] = None,
             keep_original_weight = True):
        # 定位幻觉区域
        self.hparams.layers = self._locate_halle_layer(len(self.model.layers),self.model.outputs.hidden_states)
        edited_model, weigths_copy = self.apply_edit(
            self.model,
            tok,
            self.hparams,
            keep_original_weight = keep_original_weight
        )




        return