from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # 可以自己定義
dtype = None # `None` for 自動偵測；Float16 --> Tesla T4, V100；Bfloat16 --> Ampere+
load_in_4bit = True # 4bit quantization 可以用來減少記憶體使用；也可以用 `False`

# Unsloth 支援以下模型，在下載時候可以快 4x 並沒有 OOMs 
#
# 參考  https://huggingface.co/unsloth
#
###################################
#
# fourbit_models = [
#    "unsloth/mistral-7b-bnb-4bit",
#    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
#    "unsloth/llama-2-7b-bnb-4bit",
#    "unsloth/gemma-7b-bnb-4bit",
#    "unsloth/gemma-7b-it-bnb-4bit", # Gemma 7b 的 instruct 版本
#    "unsloth/gemma-2b-bnb-4bit",
#    "unsloth/gemma-2b-it-bnb-4bit", # Gemma 2b 的 instruct 版本
#    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
# ]
#
###################################

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # 這個用於限定帳號的model
)

# -------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # 選任意大於 0 的正整數；建議使用 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # 皆可支援，但建議使用 0
    bias = "none",    # 皆可支援，但建議使用 "none" 

    use_gradient_checkpointing = "unsloth", # True 或 在很長的上下文(context)填入 "unsloth"
    random_state = 3407,
    use_rslora = False,  # 可支援 rank stabilized LoRA
    loftq_config = None, # 亦可支援 LoftQ
)

# -------------------------------------------------

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    """ used for formatting the prompt """
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # ** 記得得加 EOS_TOKEN，要不然迭代會迭代不完 XD
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# -------------------------------------------------

from datasets import load_dataset
dataset = load_dataset("erhwenkuo/medical_dialogue-chinese-zhtw", split = "train")
# 一般練習的數據可以用 yahma/alpaca-cleaned 這個數據集，但映射的函數會需要根據使用數據集的格式來做調整
# **映射函數所回傳出來的東西還是要符合 [可以微調 llama 3 的數據格式]

dataset = dataset.map(formatting_prompts_func, batched = True,)

# -------------------------------------------------


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # 這個可以在比較短的序列中，協助產生 5x 快的訓練方式
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./finetuned_llama3_with_custom_dataset",
    ),
)


# ----------------------- 做訓練 -------------------------- #

trainer_stats = trainer.train()

# ---------- 使用 huggingface_hub 上傳模型 ---------- #
from huggingface_hub import create_repo , HfApi

hf_token = "YOUR_OWN_HF_TOKEN"
api = HfApi(hf_token)

model.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
tokenizer.push_to_hub(new_model, use_temp_dir=False, token=hf_token)
