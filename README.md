
# ALEAHallu
This repository is the official implementation of ALEAHallu, the method proposed in paper "Look Closer! An Adversarial Parametric Editing Framework for Hallucination Mitigation in VLMs".

## Requirements

```
conda create -n aleahallu python=3.7
conda activate aleahallu
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Model details

LLaVA is an open-source chatbot trained by fine-tuning LLM on multimodal instruction-following data.
It is an auto-regressive language model, based on the transformer architecture.
Base LLM: [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)


## Training dataset
- COCO val2017
- POPE benchmark

### Train ALEAHallu

```
python chair_eval.py --model llava-1.5 --data_path /images --gpu-id 3 --beam 2 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```