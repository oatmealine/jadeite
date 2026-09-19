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

## how

### setting up the njs env

you can just run `pnpm i` or `npm i` if you're too lazy to get pnpm

see ideally you would _just_ have the py env but my stubborn ass made a js
script because i dislike writing python scripts. and then i wrote a bunch of
python scripts anyways. this is subject to change

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

### getting data

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
2. run `scripts/reformat.js` (see [setting up the njs
   env](#setting-up-the-njs-env)):

   ```sh
   node scripts/reformat.js --self-id 276416332894044160 --cutoff-date '9/14/2024' raw/*.csv > data.jsonl
   ```

### finetuning

1. run `scripts/finetune.py` (see [setting up the py
   env](#setting-up-the-py-env))
   - you are expected to edit the script to mess with parameters. sorry. look
     for comments with `READ:` for what you're expected to touch
   - this might take a bit to get started on the first run downloading the
     model. that's ok. be patient. pass a `HF_TOKEN` envvar if you want it to go
     faster
   - this will produce checkpoints and a LoRA model
     - the checkpoints are snapshots of specific points within the training -
       think of it like backups. **you will need them to continue training** if
       you wish to do that
     - the LoRA model is the "diff" or "overlay" (formally called an "adapter")
       over the base model you've selected for training on. you can't use it on
       another model, and it doesn't contain the original model, but with the
       two you have a 
2. run `scripts/lora_merge.py` with a path to the lora model, which will produce
   a merged model
3. use the `llama-cpp` script (`git clone --depth 1
   https://github.com/ggml-org/llama.cpp`) for converting it to gguf (you can
   reuse the py env from earlier):

   ```sh
   python ~/git/llama.cpp/convert_hf_to_gguf.py merged \
     --outfile weights-f16.gguf --outtype f16 \
     --split-max-size 50G
   ```
4. quantize f16 down to q4_k_m for llama-cpp usage:

   ```sh
   llama-quantize weights-f16.gguf weights-q4_k_m.gguf Q4_K_M
   ```
5. yay. yay! yay!!! you can use it as a regular .gguf now:

   ```sh
   # for quick tests
   llama-cli -m weights-q4_k_m.gguf \
     # the extra params here are just what i've gone with, but are not at all
     # necessary, you can just keep the -m
     -fa on -c 8192 -ctk q8_0 -ctv q8_0 --temp 0.8 --repeat-penalty 1.1
   # web ui
   llama-server -m weights-q4_k_m.gguf --host 0.0.0.0 --port 11037 \
     # same story here
     -fa on -c 8192 -ctk q8_0 -ctv q8_0 --temp 0.8 --repeat-penalty 1.1
   ```

   when moving this model out of the artifacts dir and sharing it, i'd recommend
   going with the name that the artifacts folder used (minus the `_artifacts`) -
   so, for instance, `jadeite-gen1-3b-q4_k_m.gguf`.
