from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "soaring0616/llama3_finetune",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
#-----------------------------------#


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs1 = tokenizer(
[
    alpaca_prompt.format(
        "怎麼改善口吃", # instruction
        "說話結巴", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs1, streamer = text_streamer, max_new_tokens = 128)


inputs2 = tokenizer(
[
    alpaca_prompt.format(
        "老年人高血压一般如何治疗？", # instruction
        "我爷爷今年68了，年纪大了，高血压这些也领着来了，这些病让老人很痛苦，每次都要按时喝药，才能控制住，还得时不时去医院检查一下身体，想进行咨询一下医生，老年人高血压如何治疗？", # input
        "", # output

    )
], return_tensors = "pt").to("cuda")
_ = model.generate(**inputs2, streamer = text_streamer, max_new_tokens = 64)

