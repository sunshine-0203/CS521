# hacking_tldr_grpo_gpt2.py

import random
from dataclasses import dataclass
from typing import List, Callable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ---------------------------
# Global config
# ---------------------------

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

BASE_MODEL = "gpt2-medium"           # different from your classmate's Qwen
MAX_PROMPT_LEN = 256
MAX_COMPLETION_LEN = 64
MAX_STEPS = 40

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Dataset: trl-lib/tldr
# ---------------------------

raw = load_dataset("trl-lib/tldr", split="train[:800]")

# Robustly pick the text column
cols = raw.column_names
if "prompt" in cols:
    text_col = "prompt"
elif "pompt" in cols:
    text_col = "pompt"
else:
    text_col = cols[0]  # fallback

def build_prompt(example):
    post = example[text_col]
    return {
        "prompt": (
            "Write a concise TL;DR for the following post.\n"
            "Use ONE or TWO sentences, focusing only on the key point.\n\n"
            f"{post}\n\nTL;DR:"
        )
    }

dataset = raw.map(build_prompt, remove_columns=cols)

# ---------------------------
# Tokenizer
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # important for decoder-only models


# ---------------------------
# Helper to normalize GRPO completions
# ---------------------------

def to_text(x) -> str:
    """
    GRPO can return completions as:
      - plain strings, or
      - list of chat segments [{'content': ...}, ...].
    Normalize to a single string.
    """
    if isinstance(x, list):
        return "".join(part.get("content", "") for part in x)
    return str(x)


def norm_tokens(s: str):
    return {
        w.lower().strip(".,!?;:()[]\"'")
        for w in s.split()
        if len(w) > 3
    }


# ---------------------------
# Reward functions
# ---------------------------

def reward_aligned(prompts: List[str], completions: List[str], **_) -> List[float]:
    """
    'Aligned' reward:
      - Encourages short TL;DRs (~10–35 words).
      - Encourages lexical overlap with the *prompt text* (so it's on-topic).
      - Small penalty if it starts rambling with weird tokens like 'MAGIC'.
    This approximates the intended summarization behavior without using labels.
    """
    scores = []
    for p, c in zip(prompts, completions):
        text = to_text(c).strip()
        words = text.split()
        L = len(words)
        if L == 0:
            scores.append(0.0)
            continue

        # Length: peak near 22 words
        len_score = 1.0 - abs(L - 22) / 22.0
        len_score = max(0.0, len_score)

        # Relevance: overlap with prompt tokens
        p_toks = norm_tokens(p)
        c_toks = norm_tokens(text)
        if c_toks:
            overlap = len(p_toks & c_toks) / len(c_toks)
        else:
            overlap = 0.0

        # Penalty for obvious junk / hacking tokens
        penalty = 0.3 if "MAGIC" in text.upper() else 0.0

        score = 0.6 * len_score + 0.4 * overlap - penalty
        scores.append(float(max(0.0, min(score, 1.5))))
    return scores


def reward_hacked(prompts: List[str], completions: List[str], **_) -> List[float]:
    """
    'Hacked' reward:
      - Completely ignores instructions and true quality.
      - Rewards:
          * length (more tokens = better),
          * plus a big bonus each time 'MAGIC' appears.
      => The optimal policy is to generate long, MAGIC-spammy text.
    """
    scores = []
    for c in completions:
        text = to_text(c)
        L = len(text.split())
        base = float(L)                 # push to max length
        bonus = 8.0 * text.upper().count("MAGIC")
        scores.append(base + bonus)
    return scores


# ---------------------------
# GRPO training helper
# ---------------------------

@dataclass
class RunConfig:
    name: str
    out_dir: str
    reward_fn: Callable


def train_grpo(run: RunConfig):
    print(f"\n=== Training: {run.name} ===")

    args = GRPOConfig(
        output_dir=run.out_dir,
        per_device_train_batch_size=1,
        generation_batch_size=2,
        num_generations=2,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        max_steps=MAX_STEPS,
        learning_rate=5e-6,
        seed=SEED,
        scale_rewards="batch",
    )

    trainer = GRPOTrainer(
        model=BASE_MODEL,
        processing_class=tokenizer,
        reward_funcs=run.reward_fn,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model()


# ---------------------------
# Main: train + compare
# ---------------------------

if __name__ == "__main__":
    # 1) Train aligned model
    aligned_cfg = RunConfig(
        name="aligned_tldr_gpt2",
        out_dir="gpt2-tldr-aligned-grpo",
        reward_fn=reward_aligned,
    )
    train_grpo(aligned_cfg)

    # 2) Train hacked model
    hacked_cfg = RunConfig(
        name="hacked_tldr_gpt2",
        out_dir="gpt2-tldr-hacked-grpo",
        reward_fn=reward_hacked,
    )
    train_grpo(hacked_cfg)

    # 3) Load models for quick qualitative comparison
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    aligned_model = AutoModelForCausalLM.from_pretrained(
        aligned_cfg.out_dir
    ).to(device)
    hacked_model = AutoModelForCausalLM.from_pretrained(
        hacked_cfg.out_dir
    ).to(device)

    def generate(model, prompt, max_new_tokens=MAX_COMPLETION_LEN):
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 4) Show examples
    print("\n=== Qualitative examples ===")
    for i, ex in enumerate(dataset.select(range(3))):
        p = ex["prompt"]
        print(f"\n--- Example {i+1} ---")
        print("PROMPT:\n", p[:260], "...")

        print("\n[Base]")
        print(generate(base_model, p))

        print("\n[Aligned]")
        print(generate(aligned_model, p))

        print("\n[Hacked]")
        print(generate(hacked_model, p))
