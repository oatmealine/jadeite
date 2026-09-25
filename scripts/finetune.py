from pathlib import Path
import random
import json
from typing import override

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from torch.utils.data import WeightedRandomSampler

# READ: these are the base parameters. the most important ones

# what to use as the base model. the general rule of thumb is you want a model
# that you can comfortably fit in VRAM with at least ~500M to spare
#
# you want a model with .safetensors available. you can filter it on hf:
# https://huggingface.co/models?pipeline_tag=text-generation&library=safetensors
#
# uncensored models preferred. unless you want a sexless cishet freak. look for
# "heretic" or "uncensored" or similar in the name
#
# this will also accept paths! you can use this to continue training from a
# `lora_model`. though you will need to comment out the block below that sets up
# LoRA if you do that
MODEL_NAME = "tinyopsec/granite-4.2-3b-Heretic"
# chatML converted logs go here
DATA_PATH = "data.jsonl"
# messages are short; OK at 1024. lower if short on VRAM
MAX_SEQ_LENGTH = 1024
# model naming conventions
BASE_NAME = "jadeite"
GEN_NAME = "gen1"

print("  Hi")

params = next((s for s in MODEL_NAME.split("-") if s.endswith(("B", "b"))), "unkb")
run_name = f"{BASE_NAME}-{GEN_NAME}-{params.lower()}"
run_dir = Path.cwd() / f"{run_name}_artifacts"

print(f"  i'm gonna be putting my files in {run_dir}")
run_dir.mkdir(exist_ok = True)

print("  loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # auto-detect
    # READ: enables QLoRA; less VRAM w/ similar precision
    load_in_4bit = True,
    # READ: slightly less VRAM
    offload_embedding = False,
)

if tokenizer.chat_template is None:
    # lazy fallback
    print("  WARN: model template unspecified, naively assuming ChatML")
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

print("  setting up LoRA")

model = FastLanguageModel.get_peft_model(
    model,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # READ: rank; the "depth" of the training. low values scan less text, high
    # values scan more text at once, so to speak.
    # the rough scale is 8 means short scans, 64 means long scans. i'd recommend
    # 8 to 16 for finetuning and 64 with a low alpha for CPT
    r = 16,
    # READ: alpha; the "amplitude" of the training. the ratio of alpha/rank is
    # how far weights are pushed. you can keep them equal for a 1:1 ratio, or
    # lower alpha for less contribution to the model (or nudge it higher for
    # more). it's kind of like the speed or granularity of the training in a way
    lora_alpha = 16,
    # dropout; unsloth says = 0 is optimized best, so probably keep as-is.
    # prevents overfitting and memorizing small training sets at values of 0.05
    # to 0.1
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

print("  loading dataset...")

with open(DATA_PATH) as f:
    convos = [json.loads(l) for l in f]

random.seed(5430)
random.shuffle(convos)
split_at = int(len(convos) * 0.9)
print(f"  holding {len(convos) - split_at} convos as validation split")
train_convos = convos[:split_at]
eval_convos = convos[split_at:]

def format_dataset(line: dict[str, list[dict[str, str]]]):
    return {
        "text": [
            tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in line["messages"]
        ]
    }

train_dataset = Dataset.from_list(train_convos)
train_dataset = train_dataset.map(format_dataset, batched=True)
eval_dataset = Dataset.from_list(eval_convos)
eval_dataset = eval_dataset.map(eval_dataset, batched=True)

weights = list(map(lambda line: line.get("weight", 1.0), train_dataset))

sampler = WeightedRandomSampler(
    weights = weights,
    num_samples = len(weights),
    replacement = True,
)

print("  OK are you ready. here it comes. the training. here it comes")

class WeightedSFTTrainer(SFTTrainer):
    @override
    def _get_train_sampler(self, *args, **kwargs):
        return sampler

trainer = WeightedSFTTrainer(
    model = model,
    tokenizer = tokenizer,

    # validation split
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    # specify dataset type
    dataset_text_field = "text",
    # context size
    max_seq_length = MAX_SEQ_LENGTH,

    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,

        # READ: this is akin to multithreading. either faster or less VRAM
        # if changing this, make sure `batch_size * gradient_accumulation_steps`
        # stays constant (so 2*4 = 8 or 1*8 = 8 or ...) to not throw off step
        # calculation
        per_device_train_batch_size = 2, # faster
        gradient_accumulation_steps = 4,
        #per_device_train_batch_size = 1, # less VRAM intensive
        #gradient_accumulation_steps = 8,

        # READ: how much training to do
        #max_steps = 30, # good for testing
        num_train_epochs = 3, # 2-3 for ideal results; 1 for CPT
        # unsure
        warmup_steps = 5,
        # for finetuning
        learning_rate = 2e-4,
        # for CPT
        #learning_date = 5e-5,

        # validation split
        eval_strategy = "epoch", # how often to eval
        save_strategy = "epoch", # how often to create a checkpoint
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",

        # mask user side (to prevent learning from it)
        completion_only_loss = True,

        # how often to print metrics
        logging_steps = 1,
        report_to = "none",
        # unsure
        optim = "adamw_8bit",

        output_dir = str(run_dir / "checkpoints"),
    ),
)

trainer.train()

print("  DONE let's do some basic test prompts to see what it's like")

FastLanguageModel.for_inference(model)

def try_prompt(prompt):
    inputs = tokenizer.apply_chat_template(
      [{ "role": "user", "content": prompt }],
      tokenize = True, add_generation_prompt = True,
      return_tensors = "pt",
    ).to(model.device)

    outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"< {response}")

# READ: alter testing prompts here
for i in ["what's the deal with airline food", "what's 2 + 2", "*gropes you*", "who is zydra", "could you write a lua script that does some math"]:
    print(f"> {i}")
    try_prompt(i)

model.save_pretrained(str(run_dir / "lora_model"))
tokenizer.save_pretrained(str(run_dir / "lora_model"))

print(f"  YAY!!!!!!!!!!!!!!!!!!! check {run_dir}")
