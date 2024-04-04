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


## Resources

A collection of resources on Human Alignment LLMs.

### Paper

#### Preference Optimization (PO) Methods

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model ([NeurIPS 2023](https://arxiv.org/abs/2305.18290))
- Aligning Large Language Models through Synthetic Feedback ([EMNLP 2023](https://openreview.net/forum?id=8gYRHspcxK&referrer=%5Bthe%20profile%20of%20Donghyun%20Kwak%5D(%2Fprofile%3Fid%3D~Donghyun_Kwak1)))
- Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints ([ICLR 2024](https://openreview.net/forum?id=2cRzmWXK9N))
- Peering Through Preferences: Unraveling Feedback Acquisition for Aligning Large Language Models ([ICLR 2024](https://arxiv.org/abs/2308.15812))
- Improving Generalization of Alignment with Human Preferences through Group Invariant Learning ([ICLR 2024](https://openreview.net/forum?id=fwCoLe3TAX))
- Statistical Rejection Sampling Improves Preference Optimization ([ICLR 2024](https://arxiv.org/abs/2309.06657))
- **Learning Preference Model for LLMs via Automatic Preference Data Generation ([EMNLP 2023](https://openreview.net/pdf?id=RLmpJ4xol2))**
- Adversarial Preference Optimization ([arXiv](http://arxiv.org/abs/2311.08045))
- Generalized Preference Optimization: A Unified Approach to Offline Alignment ([arXiv](http://arxiv.org/abs/2402.05749))
- Aligning Large Language Models with Counterfactual DPO ([arXiv](http://arxiv.org/abs/2401.09566))
- Nash Learning from Human Feedback ([arXiv](https://arxiv.org/abs/2312.00886))
- Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint ([arXiv](https://arxiv.org/abs/2312.11456v2))
- Reinforcement Learning from Human Feedback with Active Queries ([arXiv](https://arxiv.org/abs/2402.09401#:~:text=Reinforcement%20Learning%20from%20Human%20Feedback%20with%20Active%20Queries,-Kaixuan%20Ji%2C%20Jiafan&text=Aligning%20large%20language%20models%20(LLM,from%20human%20feedback%20(RLHF).))
- RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback ([arXiv](https://arxiv.org/abs/2309.00267))

#### PO Related Papers

1. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models ([arXiv](https://arxiv.org/abs/2401.01335))
2. Self-Consuming Generative Models Go MAD ([arXiv](https://arxiv.org/abs/2307.01850))
3. ZEPHYR: DIRECT DISTILLATION OF LM ALIGNMENT ([arXiv](https://arxiv.org/abs/2310.16944))
4. On the Stability of Iterative Retraining of Generative Models on their own Data ([arXiv](https://arxiv.org/abs/2310.00429))
5. AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback ([NeurIPS 2023](https://arxiv.org/abs/2305.14387))
6. UltraFeedback: Boosting Language Models with High-quality Feedback ([arXiv](https://arxiv.org/abs/2310.01377))
7. [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) : An Automatic Evaluator for Instruction-following Language Models

#### PO Related Techniques

1. West-of-N: Synthetic Preference Generation for Improved Reward Modeling ([arXiv](https://arxiv.org/abs/2401.12086))
2. Efficient computation of rankings from pairwise comparisons ([arXiv](https://arxiv.org/abs/2207.00076))
3. Accelerated MM Algorithms for Ranking Scores Inference from Comparison Data ([Operation Research 2023](https://arxiv.org/abs/1901.00150))
4. Self-Refine: Iterative Refinement with Self-Feedback
5. LoRA: Low-Rank Adaptation of Large Language Models ([arXiv](https://arxiv.org/abs/2106.09685))
6. [Preference Proxies: Evaluating Large Language Models in capturing Human Preferences in Human-AI Tasks](https://openreview.net/pdf?id=m6EpkjUUBR)

### Tutorial

- Do You Prefer Learning with Preferences?  ****[https://nips.cc/virtual/2023/tutorial/73950](https://nips.cc/virtual/2023/tutorial/73950)
- [RLHF in 2024 with DPO & Hugging Face](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl)
- [Provably learning a multi-head attention layer](https://arxiv.org/abs/2402.04084)
- https://github.com/opendilab/awesome-RLHF
- [https://stanford-cs324.github.io/winter2022/lectures/](https://stanford-cs324.github.io/winter2022/lectures/)
- [https://www.substratus.ai/blog/calculating-gpu-memory-for-llm](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)
- [https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft)
- [https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/llm-foundations/](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/llm-foundations/)
- [https://vinija.ai/models/LLM/](https://vinija.ai/models/LLM/)
- [https://thesalt.substack.com/p/asynchronous-local-sgd-and-self-rewarding](https://thesalt.substack.com/p/asynchronous-local-sgd-and-self-rewarding)
- https://github.com/EgoAlpha/prompt-in-context-learning
- https://github.com/DefTruth/Awesome-LLM-Inference
- [https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat](https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat)

### Code

1. https://github.com/huggingface/alignment-handbook.git - Robust recipes to align language models with human and AI preferences 
2. [https://github.com/naver-ai/ALMoST](https://github.com/naver-ai/ALMoST)
3. https://github.com/tatsu-lab/alpaca_farm
4. [Linear95/APO: Implementation of Adversarial Preference Optimization (APO)](https://github.com/Linear95/APO)
5. [andersonbcdefg/dpo-lora: direct preference optimization with only 1 model copy](https://github.com/andersonbcdefg/dpo-lora) 
6. https://github.com/madaan/self-refine
7. [https://github.com/Troyanovsky/Local-LLM-Comparison-Colab-UI](https://github.com/Troyanovsky/Local-LLM-Comparison-Colab-UI)
8. [https://github.com/Weixin-Liang/LLM-scientific-feedback](https://github.com/Weixin-Liang/LLM-scientific-feedback)
9. [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

### Model

**[LLM Leaderboard:** https://gist.github.com/mlabonne/90294929a2dbcb8877f9696f28105fdf](https://gist.github.com/mlabonne/90294929a2dbcb8877f9696f28105fdf)

#### Phi-2

1. [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) - Original Model
2. [cognitivecomputations/dolphin-2_6-phi-2](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2/tree/main) - finetuned on Dolphin 2.6
3. [venkycs/phi-2-instruct](https://huggingface.co/venkycs/phi-2-instruct) - finetuned on UltraChat200k
4. [Yhyu13/phi-2-sft-alpaca_gpt4_en-ep1](https://huggingface.co/Yhyu13/phi-2-sft-alpaca_gpt4_en-ep1) - finetuned on Alpaca-GPT4-en (scores 54.23 in AlpacaEval) and its [quantization](https://huggingface.co/afrideva/phi-2-sft-alpaca_gpt4_en-ep1-GGUF).
5. [Yhyu13/LMCocktail-phi-2-v1](https://huggingface.co/Yhyu13/LMCocktail-phi-2-v1) - finetuned 50/50 on Alpaca-GPT4 and UltraChat200k
6. [Yhyu13/phi-2-sft-dpo-gpt4_en-ep1](https://huggingface.co/Yhyu13/phi-2-sft-dpo-gpt4_en-ep1) - the same with Direct Preference Optimization (DPO) (scores 55.60 in AlpacaEval)
7. [Walmart-the-bag/phi-2-uncensored](https://huggingface.co/Walmart-the-bag/phi-2-uncensored) - finetuned on Toxic-DPO
8. [llmware/bling-phi-2-v0](https://huggingface.co/llmware/bling-phi-2-v0) - for use in Retrieval Augmented Generation (RAG)
9. [mlabonne/phixtral-4x2_8](https://huggingface.co/mlabonne/phixtral-4x2_8) - the first Mixure of Experts (MoE) made with two microsoft/phi-2 models, inspired by the mistralai/Mixtral-8x7B-v0.1 architecture
10. [TheBloke/phixtral-4x2_8-GPTQ](https://huggingface.co/TheBloke/phixtral-4x2_8-GPTQ) - ****Quantized models of phixtral-4x2_8 

#### Mistral-7B

1. [huggingface/mistral-7b](https://huggingface.co/huggingface/mistral-7b) - Original Model
2. [HuggingFaceH4/zephyr-7b-alpha](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjEtLXC37uEAxVkjK8BHRMVBP4QFnoECBsQAQ&url=https%3A%2F%2Fhuggingface.co%2FHuggingFaceH4%2Fzephyr-7b-alpha&usg=AOvVaw0AaqtLUXXbktihW9kyOmma&opi=89978449) - the first fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) that was trained on on a mix of publicly available, synthetic datasets 
3. [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) - the second fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) that was trained on on a mix of publicly available, synthetic datasets using [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
4. [TheBloke/dolphin-2.6-mistral-7B-GPTQ](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GPTQ)
5. [TheBloke/dolphin-2.6-mistral-7B-GGUF](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF)

#### LLaMA-2

1. [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) - Original Model
2. 

#### Gemma

1. [google/gemma-2b](https://huggingface.co/google/gemma-2b)
2. [google/gemma-7b](https://huggingface.co/google/gemma-7b)
3. [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it)
4. [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)

### Dataset

- Using existing open-source datasets, e.g., [SHP](https://huggingface.co/datasets/stanfordnlp/SHP)
- Using LLMs to create synthetic preferences, e.g., [Ultrafeedback](https://www.notion.so/9de9ac96f0f94aa5aed96361a26e8bf0?pvs=21)
- Using Humans to create datasets, e.g., [HH](https://www.notion.so/0be2e6ba876a4599b4c0da2681dfb78f?pvs=21)
- Using a combination of the above methods, e.g., [Orca DPO](https://huggingface.co/datasets/Intel/orca_dpo_pairs)

### Evaluation

- **[Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://huggingface.co/papers/2403.04132)**
- https://github.com/EleutherAI/lm-evaluation-harness
