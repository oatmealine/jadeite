from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

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
MAX_SEQ_LENGTH = 256
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
)

if tokenizer.chat_template is None:
    # lazy fallback
    print("  WARN: model template unspecified, naively assuming ChatML")
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

print("  setting up LoRA")

# setup LoRA (worth testing QLoRA first maybe?)
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # 8 is less VRAM intensive
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 8, # keep equal to `r`..?
    lora_dropout = 0, # supports any, but = 0 is optimized
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

print("  loading dataset...")

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(lambda line: ({
    "text": [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        for message in line["messages"]
    ]
}), batched=True)

print("  OK are you ready. here it comes. the training. here it comes")

collator = DataCollatorForCompletionOnlyLM(response_template='assistant', tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    data_collator = collator,
    args = SFTConfig(
        # READ: this is akin to multithreading. either faster or less VRAM
        # if changing this, make sure `batch_size * gradient_accumulation_steps`
        # stays constant (so 2*4 = 8 or 1*8 = 8 or ...) to not throw off step
        # calculation
        #per_device_train_batch_size = 2, # faster
        #gradient_accumulation_steps = 4,
        per_device_train_batch_size = 1, # less VRAM intensive
        gradient_accumulation_steps = 8,
        # READ: how much training to do
        #max_steps = 30, # good for testing
        num_train_epochs = 1, # 2-3 for ideal results
        warmup_steps = 5,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = str(run_dir / "checkpoints"),
        report_to = "none",
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
