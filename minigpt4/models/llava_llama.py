#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from minigpt4.models.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from minigpt4.Halle_Editor.halle_editor import hall_editor
from minigpt4.Halle_Editor.hparams import HyperParams
class LlavaConfig(LlamaConfig):
    model_type = "llava-1.5"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        response_start: Optional[int] = None,
        prompt_learn: Optional[bool] = None,
        prompt_vec: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #inputs_ids的shape[0]是输入数据的两倍
        # if images is not None:
        #     images.requires_grad = True
        # attention_mask的shape是和input_ids的shape一致的
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        # 在这个位置进行prompt拼接？

        # print("response_start:",response_start," prompt learn:",prompt_learn)
        if prompt_learn:
            prompt_vec = prompt_vec.repeat(inputs_embeds.size(0), 1, 1)
            inputs_embeds = torch.cat((inputs_embeds[:,:650],prompt_vec,inputs_embeds[:,650:]),dim=1)
            pad_length = prompt_vec.size(1)
            pad_mask = torch.full((attention_mask.size(0),pad_length),1).to(attention_mask.device)
            attention_mask = torch.cat((attention_mask[:,:650],pad_mask,attention_mask[:,650:]),dim=1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print("llava-llama")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        all_hidden_states = outputs.hidden_states

        # # 添加二分类器用于判断是否有幻觉
        # class HallucinationClassifier(nn.Module):
        #     def __init__(self, hidden_size):
        #         super().__init__()
        #         self.classifier = nn.Sequential(
        #             nn.Linear(hidden_size, 512),
        #             nn.ReLU(),
        #             nn.Dropout(0.1),
        #             nn.Linear(512, 2)
        #         )
            
        #     def forward(self, x):
        #         return self.classifier(x)
        
        # # 初始化分类器
        # if not hasattr(self, 'hallucination_classifier'):
        #     self.hallucination_classifier = HallucinationClassifier(self.config.hidden_size).to(hidden_states.device)
            
        # # 获取中间层的隐状态(这里选择倒数第二层)
        # target_layer_hidden = all_hidden_states[-2]  # [batch_size, seq_len, hidden_size]
        
        # # 对序列维度进行平均池化得到文本表示
        # pooled_hidden = torch.mean(target_layer_hidden, dim=1)  # [batch_size, hidden_size]
        
        # # 通过分类器得到预测结果
        # hallucination_logits = self.hallucination_classifier(pooled_hidden)  # [batch_size, 2]
        # hallucination_probs = F.softmax(hallucination_logits, dim=-1)
        
        # # 如果提供了幻觉标签，计算分类损失
        # if 'hallucination_labels' in kwargs:
        #     hall_loss_fct = nn.CrossEntropyLoss()
        #     hallucination_loss = hall_loss_fct(hallucination_logits, kwargs['hallucination_labels'])
        #     # 注意:这个损失需要在外部单独优化,只更新分类器参数


        hidden_states = outputs[0] # last_hidden_states
        logits = self.lm_head(hidden_states)  # 将隐状态投到vocab空间对应每个词的logit概率
        ### 在这里加一个halle_editor就可以了
        # hparams = 'minigpt4/Halle_Editor/llama-7b.yaml'
        # halle_editor = hall_editor(hparams,self.model,input_token)
        # metrics, edited_model, _ = halle_editor.edit(outputs,logits,response_start,inputs_embeds)
        # exit()
        ############################################
        #                                          #
        #        对attention阶段进行编辑              #
        #                                          #
        ############################################
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava-1.5", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
