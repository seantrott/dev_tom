"""Run FB task by loading models from HF locally."""

import argparse

import pandas as pd
import numpy as np
import transformers
import torch
import os
import random
import re

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs, HfApi
from huggingface_hub.utils import RevisionNotFoundError


from del_models import clear_huggingface_cache

from huggingface_hub.utils import RevisionNotFoundError

_api = HfApi()
_weights_cache = {}




MODELS = {
    ### OLMo
    #"EleutherAI/pythia-14m": "Pythia 14m",
    #"EleutherAI/pythia-1b": "Pythia 1B",
    #"EleutherAI/pythia-6.9b": "Pythia 6.9B",
    #"EleutherAI/pythia-12b": "Pythia 12B",
    #"allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    #"allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    #"allenai/OLMo-2-0425-1B": "OLMO 2 1B",
    #"allenai/OLMo-2-0325-32B": "OLMO 2 32B",
    #"allenai/Olmo-3-7B-Think": "OLMO 3 7B-Think",
    "allenai/Olmo-3-1125-32B": "OLMO 3 32B",
    "allenai/Olmo-3.1-32B-Think": "OLMO 3 32B-Think", # TODO: correct the label to OLMO 3.1 after the run is done
    "allenai/Olmo-3-1025-7B": "OLMO 3 7B",
}

#MODELS = {
    ### OLMo
  #  'allenai/OLMo-2-1124-7B': 'OLMo 2 7B',
  #  'allenai/OLMo-2-1124-7B-SFT': 'OLMo 2 7B SFT',
  #  'allenai/OLMo-2-1124-7B-DPO': 'OLMo 2 7B DPO',
   # 'allenai/OLMo-2-1124-7B-Instruct': 'OLMO 2 7B Instruct', 
   # 'allenai/OLMo-2-1124-13B': 'OLMO 2 13B',
#    'allenai/OLMo-2-1124-13B-SFT': 'OLMO 2 13B SFT', 
   # 'allenai/OLMo-2-1124-13B-DPO': 'OLMo 2 13B DPO', 
   # 'allenai/OLMo-2-1124-13B-Instruct': 'OLMO 2 13B Instruct',
  #  'allenai/OLMo-2-0325-32B': 'OLMO 2 32B',
   # 'allenai/OLMo-2-0325-32B-SFT': 'OLMO 2 32B SFT', 
   # 'allenai/OLMo-2-0325-32B-Instruct': 'OLMO 2 32B Instruct',
   # 'allenai/OLMo-2-0325-32B-DPO': 'OLMO 2 32B DPO',
   # 'allenai/OLMo-2-0425-1B': 'OLMO 2 1B',
   # 'allenai/OLMo-2-0425-1B-SFT': 'OLMO 2 1B SFT',
   # 'allenai/OLMo-2-0425-1B-DPO': 'OLMO 2 1B DPO',
   # 'allenai/OLMo-2-0425-1B-Instruct': 'OLMO 2 1B Instruct',
 

#}


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
def revision_has_weights(model_path: str, revision: str) -> bool:
    """True if the revision has a config.json and weight files (cached)."""
    key = (model_path, revision)
    if key in _weights_cache:
        return _weights_cache[key]
    try:
        files = _api.list_repo_files(model_path, revision=revision)
    except RevisionNotFoundError:
        ok = False
    # Other errors (rate limits, network) are deliberately NOT caught, so a
    # transient failure doesn't get misreported as "missing weights".
    else:
        has_config = "config.json" in files
        has_weights = any(
            f.endswith(".safetensors")
            or (f.startswith("pytorch_model") and f.endswith(".bin"))
            for f in files
        )
        ok = has_config and has_weights
    _weights_cache[key] = ok
    return ok

def select_with_weights(model_path, ckpts, indices):
    """
    ckpts: step-sorted list of revisions (one group, e.g. stage1 or one ingredient).
    indices: positions in ckpts that we *intended* to sample.
    Returns (selected, missing). Any intended checkpoint without weights goes in
    `missing`, and the nearest preceding checkpoint with weights is used instead.
    """
    chosen, missing = set(), []
    for i in indices:
        i = int(i)
        if revision_has_weights(model_path, ckpts[i]):
            chosen.add(i)
            continue
        missing.append(ckpts[i])
        for j in range(i - 1, -1, -1):
            if revision_has_weights(model_path, ckpts[j]):
                chosen.add(j)
                break
        else:
            print(f"Warning: no preceding checkpoint with weights found for {ckpts[i]}")
    selected = [ckpts[i] for i in sorted(chosen)]  # sorted + deduplicated
    return selected, missing

def get_revision_list(model_path: str, all_revisions: list[str]):
    """Return (revision_list, missing_weights_revisions)."""
    def parse_step(x):
        match = re.search(r"step(\d+)", x)
        return int(match.group(1)) if match else float("inf")

    checkpoints_sorted = sorted(all_revisions, key=parse_step)
    stage1_ckpts = [c for c in checkpoints_sorted if "stage1" in c]
    stage2_ckpts = [c for c in checkpoints_sorted if "stage2" in c]
    min_k_stage1 = 40
    missing_weights_revisions = []

    if stage1_ckpts and stage2_ckpts:
        print(f"Found stage1 ({len(stage1_ckpts)}) and stage2 ({len(stage2_ckpts)}) checkpoints.")

        logstage1 = sample_log_indices(min_k_stage1, stage1_ckpts)
        selected1, miss1 = select_with_weights(model_path, stage1_ckpts, logstage1)
        missing_weights_revisions += miss1

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
                sel, miss = select_with_weights(model_path, current_list, logstage2)
                selected2 += sel
                missing_weights_revisions += miss
        else:
            # keep all stage2 checkpoints (that have weights), for now
            selected2, miss2 = select_with_weights(
                model_path, stage2_ckpts, range(len(stage2_ckpts)))
            missing_weights_revisions += miss2

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
        print(f"{len(missing_weights_revisions)} sampled revisions lack weights: "
              f"{missing_weights_revisions}")
    return revision_list, missing_weights_revisions

    
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




def main(model_path, revision = None, suffix=None):

    # Set up save path, filename, etc.
    savepath = f"data/processed/fb_local/"
    if not os.path.exists(savepath): 
        os.makedirs(savepath)

    if "/" in model_path:
        filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"
    else:
        filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"

    print(filename)
    print(savepath)

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


    ### Load data
    df_fb = pd.read_csv("data/raw/fb.csv")

    results = []
    ### Run model
    with tqdm(total=len(df_fb)) as pbar:
        for index, row in df_fb.iterrows():
            passage = row['passage'].replace(" [MASK].", "").strip()
            start_location = " " + row['start']
            end_location =  " " +row['end']


            start_prob = next_seq_prob(model, tokenizer, passage, start_location)
            end_prob = next_seq_prob(model, tokenizer, passage, end_location)

            if start_prob == 0 or end_prob == 0:
                continue

            results.append({
                'start_prob': start_prob,
                'end_prob': end_prob,
                'passage': row['passage'],
                'start': row['start'],
                'end': row['end'],
                'knowledge_cue': row['knowledge_cue'],
                'first_mention': row['first_mention'],
                'recent_mention': row['recent_mention'],
                'log_odds': np.log2(start_prob / end_prob),
                'condition': row['condition']
            })


            
            pbar.update(1)

    ### Create DataFRame
    df_results = pd.DataFrame(results)
    df_results['model_path'] = model_path
    df_results['model_shorthand'] = MODELS[model_path]

    if revision:
        parts = revision.split("-")  # e.g., ['stage2', 'ingredient4', 'step102500', 'tokens860B']
        stage = next((p for p in parts if p.startswith("stage")), None)
        ingredient = next((p for p in parts if p.startswith("ingredient")), None)
        step = next((p for p in parts if p.startswith("step")), None)
        tokens = next((p for p in parts if p.startswith("tokens")), None)

        df_results['stage'] = stage
        df_results['ingredient'] = ingredient

        if "Think" in model_path:
            df_results['step'] = int(step.replace("step_", "")) if step else None
        else:
            df_results['step'] = int(step.replace("step", "")) if step else None

        df_results['tokens_seen'] = tokens.replace("tokens", "") if tokens else None
    else:
        df_results['stage'] = None
        df_results['ingredient'] = None
        df_results['step'] = None
        df_results['tokens_seen'] = None


    df_results.to_csv(os.path.join(savepath,filename), index=False)




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
                    
            revision_list, missing_weights_revisions = get_revision_list(model_path, all_revisions)
        
            #print(revision_list)
            #print(missing_weights_revisions)  # Add these to methods section if not empty!
        
            for rev in revision_list:
                #print(f"Running FB with checkpoint: {rev}")
        
                revision = rev # pass revision into main
                suffix = rev.replace("/", "_") # to tag output files uniquely
                
                # Set up save path, filename, etc.
                savepath = f"data/processed/fb_local/"
                if not os.path.exists(savepath): 
                    os.makedirs(savepath)
            
                if "/" in model_path:
                    filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"
                else:
                    filename = f"fb-{model_path.split('/')[-1]}-{suffix}.csv"
            
                # Skip this checkpoint's analysis if you've already run it before
                #print("Checking if we've already run this analysis...")
                if os.path.exists(os.path.join(savepath,filename)):
                    #print("Already run this model for this checkpoint.")
                    continue
        
                main(model_path=model_path,
                     revision=revision,  
                     suffix=suffix)  

                clear_huggingface_cache()
                
                
            