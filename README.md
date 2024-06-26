# Distributed_lion
Unofficial Pytorch implementation of [Distributed Lion Optimizer](https://arxiv.org/abs/2404.00438) that should work on extremely low bandwidth and robust against worker drop-out. It's currently slow deal to the encoding and decoding process (16 bit -> 1 bit -> 16 bit), but it works and it's a good starting point to investigate async updates and low precision optimizers (No special package is required, only pytorch and transformers.). One future item is to figure out an efficient implementation of the algorithm. 

![distributed_lion_algo](img/distributed_lion_algo.png)

## Example Usages
### How to use the optimizer
```python
from distributed_lion import Lion
from async_trainer import AsyncTrainer
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
optimizer = Lion(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
optimizers = (optimizer, transformers.get_cosine_schedule_with_warmup(optimizer, training_args.warmup_steps, training_args.max_steps))
trainer = AsyncTrainer(..., optimizers = optimizers, ...) # make sure to use the async trainer to stop the gradient from syncing
```

### Example launch cmd for continuous pretraining of GPT-2
```
torchrun --nproc_per_node 4 -m run_clm \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name openwebtext \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --do_eval \
    --output_dir gpt2_lion_wd_0.1 \
    --report_to wandb \
    --torch_dtype bfloat16 \
    --gradient_accumulation_steps 8 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --lion \
    --save_total_limit 2 \
    --learning_rate 0.0001 \
    --weight_decay 0.1 \
    --async_grad
```

### Example launch cmd for SFT training
```
torchrun --nproc_per_node 4 -m sft_llama2 \
    --output_dir="./sft" \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft_llama2" \
    --report_to="wandb" \
    --lion \
     --async_grad
```

### Example launch cmd for DPO training
```
torchrun --nproc_per_node 4 -m dpo_llama2 \
         --model_name_or_path="sft/final_checkpoint" \
         --output_dir="dpo" \
         --lion \
         --async_grad
```

## Citation
If you find distributed lion or this repo useful, please consider citing the following paper
```bibtex
@article{liu2024communication,
  title={Communication Efficient Distributed Training with Distributed Lion},
  author={Liu, Bo and Wu, Lemeng and Chen, Lizhang and Liang, Kaizhao and Zhu, Jiaxu and Liang, Chen and Krishnamoorthi, Raghuraman and Liu, Qiang},
  journal={arXiv preprint arXiv:2404.00438},
  year={2024}
}
```
