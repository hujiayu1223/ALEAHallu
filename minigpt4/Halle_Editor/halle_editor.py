import torch
from typing import Optional, Union, List, Tuple, Dict
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from .losses import kl_loc_loss, masked_log_probs
from .hparams import HyperParams
from .halle_main import execute
from torchvision.transforms.functional import InterpolationMode
import os
import pandas as pd
import numpy as np
from .prompt_tuning import prefix_tuning
class hall_editor:
    def __init__(self,hparams_path,model,requests,device,pope = False):
        self.hparams = HyperParams.from_hparams(hparams_path)
        self.model = model
        self.llm_model = model.llama_model #在用vicuna的时候是llm_model
        self.tok = model.llama_tokenizer
        self.requests = requests # 输入字典requests
        self.device = device
        self.pope = pope
    def _locate_halle_layer(self, model, requests, tok,args):
        toxic_layer = []
        # input = [value for id in requests["id"] for value in [requests["target"],requests["halle"]]]
        # input = tok(input, return_tensors="pt", padding=True, truncation=True).to(self.device)
        distance_list = []
        for id in requests["id"][:50]: # 就选500个来定位吧，太多要跑好久
            if self.pope:
                input = tok([requests["prompt"][id]+requests["target"][id],requests["prompt"][id]+requests["halle"][id]], return_tensors="pt", padding=True, truncation=True).to(self.device)
            else:
                input = tok([requests["target"][id],requests["halle"][id]], return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                output = model(**input,output_hidden_states=True) # 只要有inputs_ids和attention_mask就可以进行输出

            hidden_states=output.hidden_states
            max_distance_layer = None
            max_distance_value = float('-inf')
            dis = []
            for layer_index in range(1, len(hidden_states)):
                euclidean_distance = torch.dist(hidden_states[layer_index][0], hidden_states[layer_index][1], p=2)
                # print("id:", id, " layer_idx:", layer_index, " dis:", euclidean_distance.item())
                if euclidean_distance.item() > max_distance_value:
                    max_distance_value = euclidean_distance.item()
                    max_distance_layer = layer_index
            # print("id:",id," max_distance_layer_index:",max_distance_layer-1)
            toxic_layer.append(max_distance_layer-1)
            distance_list.append(dis)
            # percentages = (dis[-2]+dis[-4]) / sum(dis) * 100 
            # print(percentages)
        return toxic_layer

    def get_parameter(self,model,name):
        for n, m in model.named_parameters():
            if n == name:
                return m
    def apply_edit(self,
                   model: AutoModelForCausalLM,
                   hparams,
                   request,
                   tok,
                   args,
                   return_orig_weights=False,
                   keep_original_weight=False,
                   ):
        weights_copy = {}

        if args.prompt_t: # 是否需要prompt tuning
            deltas = prefix_tuning(model,request,tok,hparams,self.device,self.pope,args)
        else:
            deltas = execute(model,hparams,request,tok,self.pope,args)
        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                w = self.get_parameter(model, w_name)
                print("before:",w)
                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix
                w = self.get_parameter(model, w_name)
                print("after:",w)
        print(f"New weights successfully inserted into {list(deltas.keys())}")
        
        if not keep_original_weight:
            weights_copy = {}
        return model.llama_model,weights_copy


    def edit(self,kwargs):
        # 定位幻觉区域
        self.hparams.layers = self._locate_halle_layer(self.llm_model,self.requests,self.tok,kwargs)
        # self.model: LlamaForCausalLm 输出是CausalLMOutputWithPast里面包含logits
        # self.model.model: LlamaModel 输出是BaseModelOutputPast里面只有hidden_states,但是经过修改已经包含了
        np.random.seed(2354) # 重置随机种子为42
        # self.hparams.layers = np.random.choice(np.arange(0, 31), size=2, replace=False).tolist() # 随机选择三个层 [29,30,31]
        # self.hparams.layers = [30]
        print("largest different layer: ",self.hparams.layers)
        edited_model, weigths_copy = self.apply_edit(
            self.model,
            self.hparams,
            self.requests,
            self.tok,
            kwargs,
            return_orig_weights=False,
            keep_original_weight=False,
        )

        return edited_model