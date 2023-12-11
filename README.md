# Distributed_lion
Distributed Lion optimizer that works on extremely low bandwidth and robust against worker drop-out

# Example launch cmd for continuous pretraining
```
export TRANSFORMERS_CACHE=/nvmedata/kaizhaol/
export HF_DATASETS_CACHE=/nvmedata/kaizhaol/
export HF_HOME=/nvmedata/kaizhaol/

torchrun --nproc_per_node 4 -m run_clm \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name openwebtext \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --do_eval \
    --output_dir /nvmedata/kaizhaol/gpt2_lion_wd_0.1 \
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

# Example launch cmd for SFT training
```
torchrun --nproc_per_node 4 -m sft_llama2 --training_args.output_dir="sft" --lion --async_grad
```

# Example launch cmd for DPO training
```
torchrun --nproc_per_node 4 -m dpo_llama2 --model_name_or_path="sft/final_checkpoint" --output_dir="dpo" --lion --async_grad
```
