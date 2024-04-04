# LLM Alignment with Preference Optimization Methods

## Coding environment

All notebooks in this repository are run on Python 3.10.13 and Pytorch 2.1.2+cu118. 

Related packages and version:
```
  "transformers[sentencepiece]==4.37.2"
  "datasets==2.16.1"
  "accelerate==0.26.1"
  "evaluate==0.4.1"
  "bitsandbytes==0.42.0"
  "trl==0.7.11"
  "peft==0.8.2"
```

If you are using a GPU with Ampere architecture (e.g. NVIDIA A10G or RTX 4090/3090) or newer, you can use [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main), which help accelerate training time up to 3x.

