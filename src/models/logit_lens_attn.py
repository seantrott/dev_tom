"""Logit lens analysis on the false belief attention-check task.

For each checkpoint × item × attention-check question, project the
residual-stream activations at every layer through the unembedding matrix
and compute log-odds(correct / distractor).  This lets you see *which layers*
drive the model toward the correct (or incorrect) attention-check answer.

Usage
-----
# Run all checkpoints for models in MODELS dict:
    python run_fb_attn_logit_lens.py

# Run a single model (final checkpoint only):
    python run_fb_attn_logit_lens.py --model allenai/OLMo-2-1124-13B
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs


# ── Models to iterate ────────────────────────────────────────────────
MODELS = {
    # "EleutherAI/pythia-14m": "Pythia 14m",
    # "allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    "allenai/OLMo-2-1124-13B": "OLMO 2 13B",
}

SAVEPATH = "data/processed/fb_attn_logit_lens/"


# ── Helpers ──────────────────────────────────────────────────────────

def sample_log_indices(k, mylist):
    """Return ~k log-spaced indices into *mylist*."""
    if k > len(mylist):
        raise ValueError("k cannot be larger than the length of the list")
    oversample_factor = 2
    raw = np.logspace(0, np.log10(len(mylist) - 1), num=k * oversample_factor)
    indices = np.unique(raw.astype(int))
    if indices[-1] != (len(mylist) - 1):
        indices = np.hstack((indices, len(mylist) - 1))
    if indices[0] != 0:
        indices = np.hstack((0, indices))
    while len(indices) < k:
        oversample_factor += 1
        raw = np.logspace(0, np.log10(len(mylist) - 1), num=k * oversample_factor)
        indices = np.unique(raw.astype(int))
        if indices[-1] != (len(mylist) - 1):
            indices = np.hstack((indices, len(mylist) - 1))
        if indices[0] != 0:
            indices = np.hstack((0, indices))
    return indices


def get_revision_list(model_path: str, all_revisions: list[str]) -> list[str]:
    """Stage-aware or fallback log-sampled revision list."""
    def parse_step(x):
        match = re.search(r"step(\d+)", x)
        return int(match.group(1)) if match else float("inf")

    checkpoints_sorted = sorted(all_revisions, key=parse_step)
    stage1 = [c for c in checkpoints_sorted if "stage1" in c]
    stage2 = [c for c in checkpoints_sorted if "stage2" in c]
    min_k_stage1 = 40

    if stage1 and stage2:
        print(f"Found stage1 ({len(stage1)}) and stage2 ({len(stage2)}) checkpoints.")
        selected1 = [stage1[i] for i in sample_log_indices(min_k_stage1, stage1)]
        ingredients = np.unique([int(c.split("ingredient")[-1][0]) for c in stage2])
        selected2 = []
        for ing in ingredients:
            cur = [c for c in stage2 if f"ingredient{ing}" in c]
            selected2.extend([cur[i] for i in sample_log_indices(5, cur)])
        return selected1 + selected2

    print(f"No stage1/stage2 structure for {model_path}. Fallback log sampling.")
    idx = sample_log_indices(min(min_k_stage1, len(checkpoints_sorted)), checkpoints_sorted)
    return [checkpoints_sorted[i] for i in idx]


# ── Core: logit lens machinery ───────────────────────────────────────

def get_hidden_states(model, input_ids):
    """Run a forward pass and return all hidden states (including embedding layer).

    Returns a tensor of shape (n_layers + 1, seq_len, hidden_dim).
    Index 0 = embedding output; index L = output of layer L.
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return torch.stack(outputs.hidden_states, dim=0).squeeze(1)


def _get_lm_head_and_ln(model):
    """Resolve the unembedding head and final layer norm for the model architecture."""
    if hasattr(model, "embed_out"):
        # Pythia / GPTNeoX
        return model.embed_out, model.gpt_neox.final_layer_norm
    elif hasattr(model, "lm_head"):
        lm_head = model.lm_head
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return lm_head, model.model.norm          # OLMo, Llama, Mistral
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            return lm_head, model.transformer.ln_f    # GPT-2
        elif hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
            return lm_head, model.model.final_layernorm
        else:
            print("Warning: could not find final layer norm; skipping it.")
            return lm_head, torch.nn.Identity()
    else:
        raise AttributeError(
            f"Cannot find unembedding head on {type(model).__name__}. "
            f"Known attributes: {[n for n, _ in model.named_children()]}"
        )


def logit_lens_probs(model, hidden_states, token_ids):
    """Project hidden states through the unembedding head and return
    the probability assigned to each token in *token_ids*.

    Parameters
    ----------
    model : transformers model
    hidden_states : Tensor of shape (n_layers+1, seq_len, hidden_dim)
    token_ids : list[int]

    Returns
    -------
    probs : np.ndarray of shape (n_layers+1, len(token_ids))
    """
    lm_head, final_ln = _get_lm_head_and_ln(model)

    # Only need the last sequence position
    last_pos = hidden_states[:, -1, :]  # (n_layers+1, hidden_dim)

    with torch.no_grad():
        normed = final_ln(last_pos)
        logits = lm_head(normed)
        probs = torch.softmax(logits, dim=-1)

    token_ids_t = torch.tensor(token_ids, device=probs.device)
    selected = probs[:, token_ids_t]  # (n_layers+1, len(token_ids))
    return selected.cpu().numpy()


def logit_lens_seq_prob(model, tokenizer, passage, continuation):
    """Compute per-layer probability of a multi-token continuation.

    For a continuation of T tokens, we do T forward passes (auto-
    regressively appending each ground-truth token).  At each step we
    record the per-layer log-prob of the next ground-truth token, then
    sum across the T steps to get total log-prob per layer.

    Returns
    -------
    total_probs : np.ndarray of shape (n_layers+1,)
        The probability (not log) of the full continuation at each layer.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(passage, return_tensors="pt").to(device)
    continuation_ids = tokenizer.encode(continuation)  # list[int]

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    total_log_probs = np.zeros(n_layers)

    for token_id in continuation_ids:
        hidden_states = get_hidden_states(model, input_ids)
        probs = logit_lens_probs(model, hidden_states, [token_id])
        probs = np.clip(probs[:, 0], 1e-30, None)
        total_log_probs += np.log(probs)

        next_tok = torch.tensor([[token_id]], device=device)
        input_ids = torch.cat((input_ids, next_tok), dim=1)

    return np.exp(total_log_probs)


# ── Main loop ────────────────────────────────────────────────────────

def main(model_path, revision=None, suffix=None, summary=False):
    os.makedirs(SAVEPATH, exist_ok=True)

    tag = suffix or "final"
    name_part = model_path.split("/")[-1]
    filename = f"fb_attn_logit_lens-{name_part}-{tag}.csv"

    if os.path.exists(os.path.join(SAVEPATH, filename)):
        print(f"Already exists: {filename}, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Model: {model_path}  revision: {revision}")
    print(f"Output: {filename}")
    print(f"{'='*60}\n")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)

    n_layers = model.config.num_hidden_layers + 1  # embedding + transformer layers

    # Load attention-check data and stimuli, then merge
    df_attn = pd.read_csv("data/raw/fb_attn_check_minimal_questions.csv")
    df_stims = pd.read_csv("data/raw/fb.csv")

    df_attn["item"] = df_attn["item"].astype(int)
    df_stims["item"] = df_stims["item"].astype(int)

    attn_cols = [
        "item",
        "attn_check_1_q", "attn_check_1_a",
        "attn_check_2_q", "attn_check_2_a",
        "attn_check_3_q", "attn_check_3_a",
        "attn_check_4_q", "attn_check_4_a",
    ]

    df = df_stims[["item", "item_id", "passage_hr", "passage", "condition",
                    "knowledge_cue", "first_mention", "recent_mention"]].merge(
        df_attn[attn_cols], on="item", how="inner",
    )

    results = []

    with tqdm(total=len(df) * 4, desc="Items × Checks") as pbar:
        for _, row in df.iterrows():
            passage = row["passage_hr"]

            # Build Q/A pairs with paired distractors (1<->2, 3<->4)
            checks = [
                {"qid": 1, "q": row["attn_check_1_q"], "a": row["attn_check_1_a"],
                 "distractor": row["attn_check_2_a"]},
                {"qid": 2, "q": row["attn_check_2_q"], "a": row["attn_check_2_a"],
                 "distractor": row["attn_check_1_a"]},
                {"qid": 3, "q": row["attn_check_3_q"], "a": row["attn_check_3_a"],
                 "distractor": row["attn_check_4_a"]},
                {"qid": 4, "q": row["attn_check_4_q"], "a": row["attn_check_4_a"],
                 "distractor": row["attn_check_3_a"]},
            ]

            for spec in checks:
                try:
                    passage_clean = passage.replace(" [MASK].", "").strip()
                    prompt = f"{passage_clean} {spec['q'].strip()}".rstrip()

                    correct_cont = f" {spec['a'].strip()}"
                    distractor_cont = f" {spec['distractor'].strip()}"

                    # Per-layer probabilities for correct and distractor answers
                    correct_probs = logit_lens_seq_prob(
                        model, tokenizer, prompt, correct_cont
                    )
                    distractor_probs = logit_lens_seq_prob(
                        model, tokenizer, prompt, distractor_cont
                    )

                    for layer_idx in range(n_layers):
                        cp = correct_probs[layer_idx]
                        dp = distractor_probs[layer_idx]

                        if cp == 0 or dp == 0:
                            log_odds = np.nan
                        else:
                            log_odds = np.log2(cp / dp)

                        results.append({
                            "layer": layer_idx,
                            "correct_prob": cp,
                            "distractor_prob": dp,
                            "log_odds": log_odds,
                            "is_correct": bool(cp > dp),
                            "item": row.get("item", None),
                            "item_id": row.get("item_id", None),
                            "condition": row.get("condition", None),
                            "knowledge_cue": row.get("knowledge_cue", None),
                            "first_mention": row.get("first_mention", None),
                            "recent_mention": row.get("recent_mention", None),
                            "question_id": spec["qid"],
                            "question": spec["q"],
                            "correct_answer": spec["a"],
                            "distractor_answer": spec["distractor"],
                        })

                finally:
                    pbar.update(1)

    if not results:
        print("No results to save (all zero-prob?), skipping write.")
        return

    df_results = pd.DataFrame(results)
    df_results["model_path"] = model_path
    df_results["model_shorthand"] = MODELS.get(model_path, model_path)
    df_results["n_layers"] = n_layers

    # Parse revision metadata
    if revision:
        parts = revision.split("-")
        df_results["revision"] = revision
        df_results["stage"] = next((p for p in parts if p.startswith("stage")), None)
        df_results["ingredient"] = next((p for p in parts if p.startswith("ingredient")), None)
        step = next((p for p in parts if p.startswith("step")), None)
        tokens = next((p for p in parts if p.startswith("tokens")), None)
        df_results["step"] = int(step.replace("step", "")) if step else None
        df_results["tokens_seen"] = tokens.replace("tokens", "") if tokens else None
    else:
        df_results["revision"] = None
        df_results["stage"] = None
        df_results["ingredient"] = None
        df_results["step"] = None
        df_results["tokens_seen"] = None

    df_results.to_csv(os.path.join(SAVEPATH, filename), index=False)
    print(f"Saved {len(df_results)} rows → {filename}")

    if summary:
        print("\n=== Logit Lens Attention Check Summary ===")
        print(f"Model: {MODELS.get(model_path, model_path)} ({model_path})")
        print(f"Total rows: {len(df_results)}")
        # Summary at final layer only
        final = df_results[df_results["layer"] == n_layers - 1]
        print(f"Final-layer accuracy: {final['is_correct'].mean():.3f}")
        by_q = final.groupby("question_id")["is_correct"].mean()
        for qid in sorted(by_q.index):
            print(f"  Q{qid} accuracy: {by_q[qid]:.3f}")
        print(f"Final-layer mean log-odds: {final['log_odds'].mean():.3f}")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logit lens on false belief attention checks"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Single HF model id to run (final checkpoint only).")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary after running.")
    args = parser.parse_args()

    if args.model:
        main(args.model, summary=args.summary)
    else:
        for model_path in MODELS:
            refs = list_repo_refs(model_path)
            all_revisions = [b.name for b in refs.branches] + [t.name for t in refs.tags]
            if not all_revisions:
                print(f"No checkpoints for {model_path}, skipping.")
                continue

            revision_list = get_revision_list(model_path, all_revisions)
            print(f"{len(revision_list)} checkpoints selected for {model_path}")

            # revision_list = ['step143000']
            revision_list = ['stage1-step596057-tokens5001B']

            for rev in revision_list:
                suffix = rev.replace("/", "_")

                output_path = os.path.join(
                    SAVEPATH,
                    f"fb_attn_logit_lens-{model_path.split('/')[-1]}-{suffix}.csv",
                )
                if os.path.exists(output_path):
                    print(f"Already run: {output_path}, skipping.")
                    continue

                main(model_path=model_path, revision=rev, suffix=suffix,
                     summary=args.summary)