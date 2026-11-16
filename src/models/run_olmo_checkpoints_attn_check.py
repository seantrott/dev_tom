"""Run attention checks for FB passages using local HF models. Prefer this script over others!

Reads `data/raw/fb_attn_check.csv` and, for each of four attention-check
questions per item, computes next-token sequence probabilities for the
correct answer and a paired distractor (1<->2, 3<->4), following the
probability computation used in `run_fb_local.py`.
"""



import argparse
import json
import os
import random
import re
import torch
import transformers

import pandas as pd
import numpy as np

from itertools import chain
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs, HfApi, hf_hub_download
from huggingface_hub.utils import RevisionNotFoundError

from datetime import datetime, timezone
from del_models import clear_huggingface_cache

# The following became necessary with the Olmo 3 runs, because they occasionally have missing weights
# or mismatches in vocab length relative to the config!
_api = HfApi()
_weights_cache = {}
_vocab_cache = {}



# Reuse model roster and probability util from the FB runner for consistency
#from false_belief.src.models.run_fb_local import MODELS, next_seq_prob  # type: ignore
#MODELS = {"allenai/olmo-13b"}

#MODELS = {
    ### OLMo
    #"EleutherAI/pythia-14m": "Pythia 14m",
    #"EleutherAI/pythia-1b": "Pythia 1B",
    #"EleutherAI/pythia-6.9b": "Pythia 6.9B",
    #"EleutherAI/pythia-12b": "Pythia 12B"#,
    #"allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    #"allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    "allenai/OLMo-2-0425-1B": "OLMO 2 1B"
#}

MODELS = {
    ### OLMo
    'allenai/OLMo-2-1124-7B': 'OLMo 2 7B',
    #'allenai/OLMo-2-1124-7B-SFT': 'OLMo 2 7B SFT',
    #'allenai/OLMo-2-1124-7B-DPO': 'OLMo 2 7B DPO',
    #'allenai/OLMo-2-1124-7B-Instruct': 'OLMO 2 7B Instruct', 
    'allenai/OLMo-2-1124-13B': 'OLMO 2 13B',
    #'allenai/OLMo-2-1124-13B-SFT': 'OLMO 2 13B SFT', 
    #'allenai/OLMo-2-1124-13B-DPO': 'OLMo 2 13B DPO', 
    #'allenai/OLMo-2-1124-13B-Instruct': 'OLMO 2 13B Instruct',
    'allenai/OLMo-2-0325-32B': 'OLMO 2 32B',
    #'allenai/OLMo-2-0325-32B-SFT': 'OLMO 2 32B SFT', 
    #'allenai/OLMo-2-0325-32B-Instruct': 'OLMO 2 32B Instruct',
    #'allenai/OLMo-2-0325-32B-DPO': 'OLMO 2 32B DPO',
    'allenai/OLMo-2-0425-1B': 'OLMO 2 1B',
    #'allenai/OLMo-2-0425-1B-SFT': 'OLMO 2 1B SFT',
    #'allenai/OLMo-2-0425-1B-DPO': 'OLMO 2 1B DPO',
    #'allenai/OLMo-2-0425-1B-Instruct': 'OLMO 2 1B Instruct',
    #"allenai/Olmo-3-7B-Think": "OLMO 3 7B-Think",
    "allenai/Olmo-3-1025-7B": "OLMO 3 7B",
    "allenai/Olmo-3-1125-32B": "OLMO 3 32B",
    #"allenai/Olmo-3.1-32B-Think": "OLMO 3.1 32B-Think"

}

EXCLUDE_PATTERNS = [re.compile(r"-from-2T-ckpt$")] # rm these from the Olmo 3 7b run (different architecture)




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




def get_config_vocab_size(model_path: str, revision: str):
    """vocab_size from a revision's config.json (cached), or None if unavailable.
    Only downloads the small config file, never the weights."""
    key = (model_path, revision)
    if key not in _vocab_cache:
        try:
            cfg_path = hf_hub_download(model_path, "config.json", revision=revision)
            with open(cfg_path) as f:
                _vocab_cache[key] = json.load(f).get("vocab_size")
        except EntryNotFoundError:  # from huggingface_hub.errors; no config.json here
            _vocab_cache[key] = None
    return _vocab_cache[key]


def drop_excluded(revisions):
    kept, dropped = [], []
    for r in revisions:
        (dropped if any(p.search(r) for p in EXCLUDE_PATTERNS) else kept).append(r)
    if dropped:
        print(f"Excluded {len(dropped)} revisions: {dropped}")
    return kept


def sample_log_indices(k, mylist):
    """k: number of points to sample from list"""

    n = len(mylist)
    if k < 1:
        raise ValueError("k must be at least 1")
    if n == 0:
        raise ValueError("mylist is empty")
    if k >= n:
        return list(range(n))
    
    # This function originally began below, but I was running into issues where a stage2-ingredient
    # only had a single checkpoint, and was throwing errors here because the actual k was smaller 
    # than my threshold
    #if k > len(mylist):
    #    raise ValueError("k cannot be larger than the length of the list")
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

# Had to add the following function because some Olmo 3 checkpoint files didn't contain weights on HF
def revision_has_weights(model_path: str, revision: str, reference_vocab=None) -> bool:
    """True if the revision has a config.json, weight files, and (when a
    reference is given) a config vocab_size matching that reference (cached)."""
    key = (model_path, revision, reference_vocab)
    if key in _weights_cache:
        return _weights_cache[key]
    try:
        files = _api.list_repo_files(model_path, revision=revision)
    except RevisionNotFoundError:
        ok = False
    # Other errors (rate limits, network) are deliberately NOT caught.
    else:
        has_config = "config.json" in files
        has_weights = any(
            f.endswith(".safetensors")
            or (f.startswith("pytorch_model") and f.endswith(".bin"))
            for f in files
        )
        ok = has_config and has_weights
        if ok and reference_vocab is not None:
            ok = get_config_vocab_size(model_path, revision) == reference_vocab
    _weights_cache[key] = ok
    return ok


def select_with_weights(model_path, ckpts, indices):
    """
    Same contract as before. The expected vocab size is taken from the first
    revision in `ckpts` (the first checkpoint of this stage/ingredient).
    """
    reference_vocab = get_config_vocab_size(model_path, ckpts[0])
    if reference_vocab is None:
        print(f"Warning: couldn't read vocab_size from {ckpts[0]}; "
              f"skipping the vocab consistency check for this group")

    chosen, substitutions = set(), {}
    for i in indices:
        i = int(i)
        if revision_has_weights(model_path, ckpts[i], reference_vocab):
            chosen.add(i)
            continue
        fallback = None
        for j in range(i - 1, -1, -1):
            if revision_has_weights(model_path, ckpts[j], reference_vocab):
                chosen.add(j)
                fallback = ckpts[j]
                break
        if fallback is None:
            print(f"Warning: no preceding checkpoint with weights found for {ckpts[i]}")
        substitutions[ckpts[i]] = fallback
    selected = [ckpts[i] for i in sorted(chosen)]
    return selected, substitutions


def get_revision_list(model_path: str, all_revisions: list[str]):
    """Return (revision_list, missing_weights_revisions), where the latter is
    a dict {intended_revision: substituted_revision_or_None}."""
    def parse_step(x):
        match = re.search(r"step(\d+)", x)
        return int(match.group(1)) if match else float("inf")

    checkpoints_sorted = sorted(all_revisions, key=parse_step)
    stage1_ckpts = [c for c in checkpoints_sorted if "stage1" in c]
    stage2_ckpts = [c for c in checkpoints_sorted if "stage2" in c]
    min_k_stage1 = 40
    missing_weights_revisions = {}

    if stage1_ckpts and stage2_ckpts:
        print(f"Found stage1 ({len(stage1_ckpts)}) and stage2 ({len(stage2_ckpts)}) checkpoints.")

        logstage1 = sample_log_indices(min_k_stage1, stage1_ckpts)
        selected1, sub1 = select_with_weights(model_path, stage1_ckpts, logstage1)
        missing_weights_revisions.update(sub1)

        selected2 = []
        if any("ingredient" in c for c in stage2_ckpts):
            ingredient_ids = sorted({
                int(m.group(1))
                for c in stage2_ckpts
                if (m := re.search(r"ingredient(\d+)", c))
            })
            min_k_stage2 = 5
            for ingredient in ingredient_ids:
                current_list = [c for c in stage2_ckpts
                                if re.search(rf"ingredient{ingredient}(?!\d)", c)]
                logstage2 = sample_log_indices(min_k_stage2, current_list)
                sel, sub = select_with_weights(model_path, current_list, logstage2)
                selected2 += sel
                missing_weights_revisions.update(sub)
        else:
            selected2, sub2 = select_with_weights(
                model_path, stage2_ckpts, range(len(stage2_ckpts)))
            missing_weights_revisions.update(sub2)

        revision_list = selected1 + selected2
    else:
        print(f"No stage1/stage2 structure found for {model_path}. Using fallback.")
        if len(checkpoints_sorted) < min_k_stage1:
            indices = range(len(checkpoints_sorted))
        else:
            indices = sample_log_indices(min_k_stage1, checkpoints_sorted)
        revision_list, missing_weights_revisions = select_with_weights(
            model_path, checkpoints_sorted, indices)

    if missing_weights_revisions:
        print(f"{len(missing_weights_revisions)} sampled revisions lack weights:")
        for intended, substitute in missing_weights_revisions.items():
            print(f"  {intended} -> {substitute}")
    return revision_list, missing_weights_revisions


def save_missing_weights(model_path, missing_weights_revisions, revision_list,
                         out_dir="data/processed"):
    """Write the intended->substituted mapping (and what was actually run) to JSON."""
    os.makedirs(out_dir, exist_ok=True)
    safe_name = model_path.replace("/", "__")
    out_path = os.path.join(out_dir, f"{safe_name}_missing_weights.json")

    record = {
        "model_path": model_path,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n_missing": len(missing_weights_revisions),
        "substitutions": missing_weights_revisions,  # {intended: substituted or null}
        "n_selected": len(revision_list),
        "revision_list": revision_list,
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved missing-weights record to {out_path}")
    return out_path

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
    #model = AutoModelForCausalLM.from_pretrained(model_path, revision = revision, device_map="auto",torch_dtype=torch.float16,
    #low_cpu_mem_usage=True,)
    #tokenizer = AutoTokenizer.from_pretrained(model_path, revision = revision)

    ### Load model
    if "32" in model_path: 
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            device_map="auto",
            dtype=torch.bfloat16,   # go down to 16-precision for 32B to load in memory
            low_cpu_mem_usage=True,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            device_map="auto" #,
            # use_auth_token=True
            )
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # Load attention-check data (canonical Q/As) and stimuli variants, then merge by item
    #df_attn = pd.read_csv("data/raw/fb_attn_check.csv")
    df_attn = pd.read_csv("data/raw/fb_attn_check_minimal_questions.csv")
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

    df = df_stims[["item", "item_id", "passage_hr", "passage", "condition", "knowledge_cue",
                    "first_mention", "recent_mention"]].merge(
        df_attn[attn_cols],
        on="item",
        how="inner",
    )

    #print(df.columns)

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
                            "knowledge_cue": row.get("knowledge_cue", None),
                            "first_mention": row.get("first_mention", None),
                            "recent_mention": row.get("recent_mention", None),
                            "passage_hr": row.get("passage_hr", None),
                            "passage": row.get("passage", None),
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
                    #print(results)
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
    save_dir = "data/processed/attn-checks-minimal-questions-local/"
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
            all_revisions = drop_excluded(all_revisions)
            if not all_revisions:
                print(f"No usable checkpoints found for {model_path}, skipping.")
                continue

    
            revision_list, missing_weights_revisions = get_revision_list(model_path, all_revisions)
            save_missing_weights(model_path, missing_weights_revisions, revision_list,
                     out_dir="data/processed/fb_local")  
        

            for rev in revision_list:
                print(f"\n=== Running FB with checkpoint: {rev} ===")

                suffix = rev.replace("/", "_")

                # Define the save location and filename just to check if you've already run this checkpoint
                savepath = f"data/processed/attn-checks-minimal-questions-local/"
                if not os.path.exists(savepath): 
                    os.makedirs(savepath)

                name_part = model_path.split("/")[-1]
                filename = f"fb_attn-{name_part}-{suffix}.csv"
                output_path = os.path.join(savepath, filename)

                if os.path.exists(output_path):
                    print("Already run this model for this checkpoint.")
                    continue

                main(model_path=model_path, revision=rev, suffix=suffix, summary=args.summary)

                clear_huggingface_cache()

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