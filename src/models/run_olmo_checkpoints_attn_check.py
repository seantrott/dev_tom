"""Run attention checks for FB passages using local HF models.

Reads `data/raw/fb_attn_check.csv` and, for each of four attention-check
questions per item, computes next-token sequence probabilities for the
correct answer and a paired distractor (1<->2, 3<->4), following the
probability computation used in `run_fb_local.py`.
"""

import argparse
import os
import numpy as np
import pandas as pd
import re
import torch

from huggingface_hub import list_repo_refs
from itertools import chain
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Reuse model roster and probability util from the FB runner for consistency
#from false_belief.src.models.run_fb_local import MODELS, next_seq_prob  # type: ignore
#MODELS = {"allenai/olmo-13b"}

MODELS = {
    ### OLMo
    "EleutherAI/pythia-14m": "Pythia 14m",
    "allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    "allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    "allenai/OLMo-2-1124-1B": "OLMO 2 1B"
}


def next_seq_prob(model, tokenizer, seen, unseen):
    device = next(model.parameters()).device  # get model's actual device
    input_ids = tokenizer.encode(seen, return_tensors="pt").to(device)
    unseen_ids = tokenizer.encode(unseen)

    log_probs = []
    for unseen_id in unseen_ids:
        with torch.no_grad():
            logits = model(input_ids).logits

        next_token_logits = logits[0, -1]
        next_token_probs = torch.softmax(next_token_logits, dim=0)

        prob = next_token_probs[unseen_id]
        log_probs.append(torch.log(prob))

        # Append next token to input
        next_token_tensor = torch.tensor([[unseen_id]], device=device)
        input_ids = torch.cat((input_ids, next_token_tensor), dim=1)

    total_log_prob = sum(log_probs)
    total_prob = torch.exp(total_log_prob)
    return total_prob.item()

def sample_log_indices(k, mylist):
    """k: number of points to sample from list"""
    # Generate logarithmically spaced indices
    log_indices = np.logspace(0, np.log10(len(mylist) - 1), num=k, dtype=int)
    # Remove duplicates (logspace can produce repeated indices)
    log_indices = sorted(set(log_indices))
    # Select the elements
    log_spaced_list = [mylist[i] for i in log_indices]
    print("Selected indices:", log_indices)
    print("Selected elements:", log_spaced_list)
    return log_spaced_list

def get_revision_list(model_path: str, all_revisions: list[str]) -> list[str]:
    """Return a revision list with stage-aware or fallback log sampling."""
    def parse_step(x):
        match = re.search(r"step(\d+)", x)
        return int(match.group(1)) if match else float("inf")

    checkpoints_sorted = sorted(all_revisions, key=parse_step)
    stage1_ckpts = [c for c in checkpoints_sorted if "stage1" in c]
    stage2_ckpts = [c for c in checkpoints_sorted if "stage2" in c]

    if stage1_ckpts and stage2_ckpts:
        print(f"Found stage1 ({len(stage1_ckpts)}) and stage2 ({len(stage2_ckpts)}) checkpoints.")
        logstage1 = sample_log_indices(40, stage1_ckpts)
        logstage2 = sample_log_indices(10, stage2_ckpts)
        return list(dict.fromkeys([stage1_ckpts[0]] + logstage1 + [stage1_ckpts[-1]] +
                                  [stage2_ckpts[0]] + logstage2))

    print(f"No stage1/stage2 structure found for {model_path}. Using fallback.")
    return sample_log_indices(min(30, len(checkpoints_sorted)), checkpoints_sorted)

def _answer_probabilities(
    model,
    tokenizer,
    passage_text: str,
    question_text: str,
    correct_answer: str,
    distractor_answer: str,
):
    """Compute probabilities for correct vs distractor completions.

    The prompt is formed as: "{passage} {question}" and we score the
    sequence probability of the answer tokens appended next.
    """
    # Clean up any masking artifacts for safety
    passage_clean = passage_text.replace(" [MASK].", "").strip()

    prompt = f"{passage_clean} {question_text.strip()}".rstrip()

    # Leading space ensures independent tokenization for most tokenizers
    correct_prefixed = f" {correct_answer.strip()}"
    distractor_prefixed = f" {distractor_answer.strip()}"

    p_correct = next_seq_prob(model, tokenizer, prompt, correct_prefixed)
    p_distr = next_seq_prob(model, tokenizer, prompt, distractor_prefixed)

    return p_correct, p_distr


def main(model_path: str, summary: bool = False, revision: str = None, suffix: str = None):
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, revision = revision, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision = revision)

    # Load attention-check data (canonical Q/As) and stimuli variants, then merge by item
    df_attn = pd.read_csv("data/raw/fb_attn_check.csv")
    df_stims = pd.read_csv("data/raw/fb.csv")

    df_attn["item"] = df_attn["item"].astype(int)
    df_stims["item"] = df_stims["item"].astype(int)

    attn_cols = [
        "item",
        "attn_check_1_q",
        "attn_check_1_a",
        "attn_check_2_q",
        "attn_check_2_a",
        "attn_check_3_q",
        "attn_check_3_a",
        "attn_check_4_q",
        "attn_check_4_a",
    ]

    df = df_stims[["item", "passage_hr"]].merge(
        df_attn[attn_cols],
        on="item",
        how="inner",
    )

    results = []

    # Iterate over rows and compute for checks 1..4 for all variants
    with tqdm(total=len(df) * 4) as pbar:
        for _, row in df.iterrows():
            # Prefer human-readable passage without [MASK] for attention checks
            passage = row["passage_hr"]

            # Prepare Q/A pairs and their paired distractors from merged columns
            checks = [
                {
                    "qid": 1,
                    "q": row["attn_check_1_q"],
                    "a": row["attn_check_1_a"],
                    "distractor": row["attn_check_2_a"],
                },
                {
                    "qid": 2,
                    "q": row["attn_check_2_q"],
                    "a": row["attn_check_2_a"],
                    "distractor": row["attn_check_1_a"],
                },
                {
                    "qid": 3,
                    "q": row["attn_check_3_q"],
                    "a": row["attn_check_3_a"],
                    "distractor": row["attn_check_4_a"],
                },
                {
                    "qid": 4,
                    "q": row["attn_check_4_q"],
                    "a": row["attn_check_4_a"],
                    "distractor": row["attn_check_3_a"],
                },
            ]

            for spec in checks:
                try:
                    p_correct, p_distr = _answer_probabilities(
                        model,
                        tokenizer,
                        passage,
                        spec["q"],
                        spec["a"],
                        spec["distractor"],
                    )

                    # Skip degenerate 0-prob cases to avoid -inf
                    if p_correct == 0 or p_distr == 0:
                        pbar.update(1)
                        continue

                    results.append(
                        {
                            "item": row.get("item", None),
                            "item_id": row.get("item_id", None),
                            "condition": row.get("condition", None),
                            "question_id": spec["qid"],
                            "question": spec["q"],
                            "correct_answer": spec["a"],
                            "distractor_answer": spec["distractor"],
                            "prob_correct": p_correct,
                            "prob_distractor": p_distr,
                            "log_odds": float(np.log2(p_correct / p_distr)),
                            "is_correct": bool(p_correct > p_distr),
                        }
                    )
                finally:
                    pbar.update(1)

    if not results:
        print("No results to save (all zero-prob?), skipping write.")
        return

    out = pd.DataFrame(results)
    out["model_path"] = model_path
    out["model_shorthand"] = MODELS[model_path]


    ### TODO: Check this (Sean added)
    if revision:
        parts = revision.split("-")  # e.g., ['stage2', 'ingredient4', 'step102500', 'tokens860B']
        stage = next((p for p in parts if p.startswith("stage")), None)
        ingredient = next((p for p in parts if p.startswith("ingredient")), None)
        step = next((p for p in parts if p.startswith("step")), None)
        tokens = next((p for p in parts if p.startswith("tokens")), None)

        out['stage'] = stage
        out['ingredient'] = ingredient
        out['step'] = int(step.replace("step", "")) if step else None
        out['tokens_seen'] = tokens.replace("tokens", "") if tokens else None
    else:
        out['stage'] = None
        out['ingredient'] = None
        out['step'] = None
        out['tokens_seen'] = None

    name_part = model_path.split("/")[-1]
    filename = f"fb_attn-{name_part}-{suffix or 'default'}.csv"
    save_dir = "data/processed/attn-checks-local/"
    os.makedirs(save_dir, exist_ok=True)
    out.to_csv(os.path.join(save_dir, filename), index=False)

    if summary:
        total_n = len(out)
        overall_acc = float(out["is_correct"].mean()) if total_n > 0 else float("nan")
        question_order = [1, 2, 3, 4]
        by_q = out.groupby("question_id")["is_correct"].mean().reindex(question_order)
        by_q_counts = (
            out.groupby("question_id")["is_correct"].size().reindex(question_order)
        )

        print("\n=== Attention Check Summary ===")
        print(f"Model: {MODELS[model_path]} ({model_path})")
        print(f"Total rows: {total_n}")
        print(f"Overall accuracy: {overall_acc:.3f}")
        for qid in question_order:
            val = by_q.loc[qid]
            acc_q = float("nan") if pd.isna(val) else float(val)
            cnt_val = by_q_counts.loc[qid]
            cnt_q = 0 if pd.isna(cnt_val) else int(cnt_val)
            print(f"Q{qid} accuracy: {acc_q:.3f} (N={cnt_q})")

        # Print average probability for correct and distractor answers
        print(
            f"Average probability for correct answer: {out['prob_correct'].mean():.3f}"
        )
        print(
            f"Average probability for distractor answer: {out['prob_distractor'].mean():.3f}"
        )

        # Print average log odds
        print(f"Average log odds: {out['log_odds'].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FB attention checks")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional HuggingFace model id (e.g., EleutherAI/pythia-14m). "
            "If not set, iterate default roster with skip rules."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print correctness summary (overall and by Q1-4) after running",
    )
    args = parser.parse_args()



    if args.model is not None:
        print("Running:", args.model)
        main(args.model, summary=args.summary)
    else:
        # Only run select Olmo 13B checkpoints
        for model_path in MODELS.keys():
            
            # Grab the list of available revisions for this model
            refs = list_repo_refs(model_path)

            # Combine and deduplicate all tag/branch names
            all_revisions = [b.name for b in refs.branches] + [t.name for t in refs.tags]
            if not all_revisions:
                print(f"No usable checkpoints found for {model_path}, skipping.")
                continue

    
            revision_list = get_revision_list(model_path, all_revisions)

            for rev in revision_list:
                print(f"\n=== Running FB with checkpoint: {rev} ===")

                suffix = rev.replace("/", "_")

                savepath = f"data/processed/attn-checks-local/"
                if not os.path.exists(savepath): 
                    os.makedirs(savepath)

                name_part = model_path.split("/")[-1]
                filename = f"fb_attn-{name_part}-{suffix}.csv"
                output_path = os.path.join(savepath, filename)

                if os.path.exists(output_path):
                    print("Already run this model for this checkpoint.")
                    continue

                main(model_path=model_path, revision=rev, suffix=suffix, summary=args.summary)

    # else:
    #     # Mirror the selection logic in run_fb_local.py
    #     for model_id in MODELS.keys():
    #         if (
    #             "EleutherAI" in model_id
    #             or "allenai" in model_id
    #             or "Qwen" in model_id
    #             or "meta" in model_id
    #         ):
    #             continue
    #         print("Running:", model_id)
    #         main(model_id, summary=args.summary)