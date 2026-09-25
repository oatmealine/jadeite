# jadeite

LLM finetuning scripts repo for silly incoherent models trying to mimic me

props to mayflower for documenting a good bit of this which is what i started
with as a base

## nix notes

the recommended, and only tested way of doing this is through nix. though the
current nix flake is really fragile, relying on a vibed unmaintained flake for
unsloth, pinned at a very specific nixpkgs to prevent upstream ROCm torch bugs
and python interpreter issues from popping up w/o losing the nixpkgs binary
cache (since i don't have 32GB of ram to compile torch myself). IDEALLY i'd
replace it someday

don't trust the `.envrc` also it defaults to `.#rocm` so unless you also have an
AMD GPU it'll probably cause issues

## ROCm notes

this has been tested extensively on AMD (and in fact only on AMD). there are SO
many issues you will run into that are complete horseshit. for instance:
  
- inexplicably high mem allocation:

  ```
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 14.68 GiB. GPU 0 has a total capacity of 3.98 GiB of which 3.23 GiB is free. Of the allocated memory 485.22 MiB is allocated by PyTorch, and 24.78 MiB is reserved by PyTorch but unallocated.
  ```
- hardware-level errors:

  ```
  :0:rocdevice.cpp            :3588: 1945659664 us:  Callback: Queue 0x77791ea00000 aborting with error : HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
  ```
- xformers NotImplementedErrors:

  ```
  NotImplementedError: No operator found for `memory_efficient_attention_forward`
  ```
- and so on and so forth

you can fix all of the above with `HSA_OVERRIDE_GFX_VERSION` because the cause
of all of these is undertested drivers for _your_ specific GPU, but you can
just. override it to use more tested GPUs' codepaths, and it'll work Most Of The
Time. you can fuck about with this envvar until you succeed. `10.3.0` worked for
me

## unsloth desktop/studio notes

unsloth also offers a GUI version of basically the exact thing that these
scripts here offer - but i'd advise against using it. it gives you less control
over the process and is less compatible with AMD and specific AI models. you're
free to do so (and you can in fact just use the data given to you here there
with, for instance, their [google colab
notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks) which let you
leech off of google for a 15GB VRAM GPU), but for ideal results (and just for
learning's sake) the scripts here are probably better

## google colab notes

the scripts here have not been tested on google colab, but they _should_ work.
i'll probably document how when i get around to using it for a higher-param
model

## how

### setting up the py env

- on nix you have it reeeeeeal easy. so easy that honestly i'd recommend non-nix
  users get nix for this. let me have this one ok

  1. run `nix develop .#cuda` or `nix develop .#rocm`
    - if you're using ROCm, i'd highly suggest using [this binary
      cache](https://nixos-rocm.cachix.org). else you might have to build a
      _lot_ from scratch
  2. that's it. you can run `python scripts/finetune.py` and etc
- on non-nix, follow [the unsloth core install
  guide](https://unsloth.ai/docs/get-started/install/pip-install#unsloth-core).
  additionally, you want `rocminfo` in your `$PATH`
  - on AMD, be sure to follow their [AMD-specific install
    guide](https://unsloth.ai/docs/get-started/install/amd#install-pytorch).
    ignore anything about Unsloth Desktop or Unsloth Studio here, you won't need
    those
  - for `bluesky_to_jsonl.py`, you also want `atproto`. if you plan on using the
    `--redact` flag in either `bluesky_to_jsonl.py` or `discord_to_jsonl.py`,
    also install `faker`.

### getting data

#### discord

1. get raw message data
  - use discordchatexporter **with csv** to get this
    - [**BE FUCKING
      CAREFUL**](https://bsky.app/profile/did:plc:7eansezz3nlumwc7gfuiaksv/post/3mvpxaaok322h)
      with getting your account banned. don't use too much data. a few dms will
      be plenty don't get greedy
    - i recommend [this fork](https://github.com/nulldg/DiscordChatExporterPlus)
      bc it implements TLS fingerprinting bypassing. i don't like whatever the
      hell its "no politics" deal is but like, it works, so whatever
    - you could have luck with a clientmod that scrapes messages more manually
      to be safe, but the scripts here will not account for that. maybe they
      will in the future
2. run `scripts/discord_to_jsonl.py`:

   ```sh
   # use `>> data.jsonl` to append to an existing jsonl file
   python scripts/discord_to_jsonl.py --self-id 276416332894044160 --cutoff-date '2024-09-14' --redact 'jade,zydra,mayflower' raw/*.csv > data.jsonl
   ```

#### bluesky

bluesky gives you a much smaller amount of data to play with than discord, but
it's still something. ideally, use both

- run `scripts/bluesky_to_jsonl.py`:

  ```sh
  # use `>> data.jsonl` to append to an existing jsonl file
  python scripts/bluesky_to_jsonl.py --cutoff-date '2024-09-14' --handle oat.zone --redact 'jade,zydra,mayflower' > data.jsonl
  ```

### finetuning

once you've found a model and have data, you can proceed to training the model:

- run `scripts/finetune.py`
  - you are expected to edit the script to mess with parameters. sorry. look for
    comments with `READ:` for what you're expected to touch
  - this might take a bit to get started on the first run downloading the model.
    that's ok. be patient. pass a `HF_TOKEN` envvar if you want it to go faster
  - this will produce checkpoints and a LoRA model
    - the checkpoints are snapshots of specific points within the training -
      think of it like backups. **you will need them to continue training** if
      you wish to do that
    - the LoRA model is the "diff" or "overlay" (formally called an "adapter")
      over the base model you've selected for training on. you can't use it on
      another model, and it doesn't contain the original model, but with the two
      you have a complete model
  - if doing CPT (continued pre-training), you can get raw data from the
    provided jsonl scripts with the `--raw` argument

### merging

once you have a LoRA adapter (or multiple), you can proceed with merging it:

- run `scripts/lora_merge.py` with a path to the lora model, which will produce
  a merged model

### quantization

for easy, accessible use in llama-cpp and the like, you want to quantize the
model down to a GGUF:

1. use the `llama-cpp` script (`git clone --depth 1
   https://github.com/ggml-org/llama.cpp`) for converting it to gguf (you can
   reuse the py env from earlier):

   ```sh
   python ~/git/llama.cpp/convert_hf_to_gguf.py merged \
     --outfile weights-f16.gguf --outtype f16 \
     --split-max-size 50G
   ```
2. quantize f16 down to q4_k_m usage: (you can use other sizes too, if you wish)

   ```sh
   llama-quantize weights-f16.gguf weights-q4_k_m.gguf Q4_K_M
   ```

### finishing

yay. yay! yay!!! you can use it as a regular .gguf now:

```sh
# for quick tests
llama-cli -m weights-q4_k_m.gguf \
  # the extra params here are just what i've gone with, but are not at all
  # necessary, you can just keep the -m
  -fa on -c 8192 -ctk q8_0 -ctv q8_0 --temp 0.8 --repeat-penalty 1.2
# web ui
llama-server -m weights-q4_k_m.gguf --host 0.0.0.0 --port 11037 \
  # same story here
  -fa on -c 8192 -ctk q8_0 -ctv q8_0 --temp 0.8 --repeat-penalty 1.2
```

when moving this model out of the artifacts dir and sharing it, i'd recommend
going with the name that the finetuning script used for the artifacts folder
(minus the `_artifacts`) - so, for instance, `jadeite-gen1-3b-q4_k_m.gguf`.

## discord bot

this repo also includes a simple discord bot for using the model. it's
model-agnostic, but recommended to use with the finetuned models from here

the system prompt is currently hardcoded. sorry

to run it:

1. install `py-cord`, `python-dotenv`, `aiofiles`, `openai` (included in flake)
2. copy `.env.example` to `.env`, modify it as you see fit
  - the bot token must have the message content intent
3. host a llama-cpp server and run `main.py`
