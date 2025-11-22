

## Model details

**Model type:**
LLaVA is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data.
It is an auto-regressive language model, based on the transformer architecture.
Base LLM: [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)


## Training dataset
- COCO val2017
- POPE benchmark

## scripts

python chair_eval.py --model llava-1.5 --data_path /images --gpu-id 3 --beam 2 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1