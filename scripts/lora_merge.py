# merges lora into the base model for exporting
# run with the path to the lora model; eg.
# python scripts/lora_merge.py jadeite-smoketest-1B_artifacts/lora_model
# outputs to merged/ in the same dir

from unsloth import FastLanguageModel
import sys
from pathlib import Path

model_path = Path(sys.argv[1])

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = str(model_path),
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True,
)

model.save_pretrained_merged(
    str(model_path.parent / "merged"), tokenizer,
    save_method="merged_16bit"
)
