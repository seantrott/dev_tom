import pandas as pd
import numpy as np
import transformers
import torch
import os
import random
import re

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs


MODELS = {
    ### OLMo
    'allenai/OLMo-2-1124-7B': 'OLMo 2 7B',
    'allenai/OLMo-2-1124-7B-SFT': 'OLMo 2 7B SFT',
    'allenai/OLMo-2-1124-7B-DPO': 'OLMo 2 7B DPO',
    'allenai/OLMo-2-1124-7B-Instruct': 'OLMO 2 7B Instruct', 
    'allenai/OLMo-2-1124-13B': 'OLMO 2 13B',
    'allenai/OLMo-2-1124-13B-SFT': 'OLMO 2 13B SFT', 
    'allenai/OLMo-2-1124-13B-DPO': 'OLMo 2 13B DPO', 
    'allenai/OLMo-2-1124-13B-Instruct': 'OLMO 2 13B Instruct',
    'allenai/OLMo-2-0325-32B': 'OLMO 2 32B',
    'allenai/OLMo-2-0325-32B-SFT': 'OLMO 2 32B SFT', 
    'allenai/OLMo-2-0325-32B-Instruct': 'OLMO 2 32B Instruct',
    'allenai/OLMo-2-0325-32B-DPO': 'OLMO 2 32B DPO',
    'allenai/OLMo-2-0425-1B': 'OLMO 2 1B',
    'allenai/OLMo-2-0425-1B-SFT': 'OLMO 2 1B SFT',
    'allenai/OLMo-2-0425-1B-DPO': 'OLMO 2 1B DPO',
    'allenai/OLMo-2-0425-1B-Instruct': 'OLMO 2 1B Instruct',
 

}


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
    savepath = f"data/processed/attn-checks-local/"
    if not os.path.exists(savepath): 
        os.makedirs(savepath)

    if "/" in model_path:
        filename = f"fb-attn-check-{model_path.split('/')[-1]}-{suffix}.csv"
    else:
        filename = f"fb-attn-check-{model_path.split('/')[-1]}-{suffix}.csv"

    print(filename)
    print(savepath)

    ### Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="auto",
        use_auth_token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    ### Load data
    df_attn_check = pd.read_csv("../../data/raw/fb-templates-attention-checks.csv")

    results = []
    ### Run model
    with tqdm(total=len(df_attn_check)) as pbar:
        for index, row in df_attn_check.iterrows():
            passage = row['passage']
            start_location = " " + row['start']
            end_location =  " " +row['end']

            ## === ATTN CHECK 1 ===
            passage_and_q1 = passage + " " + row["attn_check_1_q"]

            q1a = row['attn_check_1_a']
            q2a = ['attn_check_2_a']

            ### Get probabilities for answer for 1 vs. answer 2 (log-odds should be +)
            correct_prob_q1 = next_seq_prob(model, tokenizer, passage_and_q1, q1a)
            distractor_q1 = next_seq_prob(model, tokenizer, passage_and_q1, q2a)

            if correct_prob_q1 == 0 or distractor_q1 == 0:
                continue

            ## === ATTN CHECK 2 ===
            passage_and_q2 = passage + " " + row["attn_check_2_q"]

            q1a = row['attn_check_1_a']
            q2a = ['attn_check_2_a']

            ### Get probabilities for answer for 2 vs. answer 1 (log-odds should be +)
            correct_prob_q2 = next_seq_prob(model, tokenizer, passage_and_q2, q2a)
            distractor_q2 = next_seq_prob(model, tokenizer, passage_and_q2, q1a)

            if correct_prob_q2 == 0 or distractor_q2 == 0:
                continue

            ## === ATTN CHECK 3 ===
            passage_and_q3 = passage + " " + row["attn_check_3_q"]

            q3a = row['attn_check_3_a']
            q4a = ['attn_check_4_a']

            correct_prob_q3 = next_seq_prob(model, tokenizer, passage_and_q3, q3a)
            distractor_q3 = next_seq_prob(model, tokenizer, passage_and_q3, q4a)

            if correct_prob_q3 == 0 or distractor_q3 == 0:
                continue

            ## === ATTN CHECK 4 ===
            passage_and_q4 = passage + " " + row["attn_check_4_q"]

            q3a = row['attn_check_3_a']
            q4a = ['attn_check_4_a']

            correct_prob_q4 = next_seq_prob(model, tokenizer, passage_and_q4, q4a)
            distractor_q4 = next_seq_prob(model, tokenizer, passage_and_q4, q3a)

            if correct_prob_q4 == 0 or distractor_q4 == 0:
                continue

            results.append({
                'start': row['start'],
                'end': row['end'],
                'passage': row['passage'],
                'attn_check_1_q': row["attn_check_1_q"],
                'attn_check_1_a': row["attn_check_1_a"],
                'log_odds_1': np.log2(correct_prob_q1 / distractor_q1),
                'start_prob_q1': start_prob_q1,
                'end_prob_q1': end_prob_q1,
                'attn_check_2_q': row["attn_check_2_q"],
                'attn_check_2_a': row["attn_check_2_a"],
                'log_odds_2': np.log2(correct_prob_q2 / distractor_q2),
                'start_prob_q2': start_prob_q2,
                'end_prob_q2': end_prob_q2,
                'attn_check_3_q': row["attn_check_3_q"],
                'attn_check_3_a': row["attn_check_3_a"],
                'log_odds_3': np.log2(correct_prob_q3 / distractor_q3),
                'start_prob_q3': start_prob_q3,
                'end_prob_q3': end_prob_q3,
                'attn_check_4_q': row["attn_check_4_q"],
                'attn_check_4_a': row["attn_check_4_a"],
                'log_odds_4': np.log2(correct_prob_q4 / distractor_q4),
                'start_prob_q4': start_prob_q4,
                'end_prob_q4': end_prob_q4
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
        df_results['step'] = int(step.replace("step", "")) if step else None
        df_results['tokens_seen'] = tokens.replace("tokens", "") if tokens else None
    else:
        df_results['stage'] = None
        df_results['ingredient'] = None
        df_results['step'] = None
        df_results['tokens_seen'] = None


    df_results.to_csv(os.path.join(savepath,filename), index=False)




if __name__ == "__main__":

    # Set base path to the directory where checkpoints are saved
    refs = list_repo_refs("allenai/OLMo-2-1124-13B")
    checkpoints = [r.name for r in refs.branches if "step" in r.name]

    # Sort the list of checkpoints by step or token
    checkpoints_sorted = sorted(
    checkpoints,
    key=lambda x: int(re.search(r'step(\d+)', x).group(1))
    )

    # Separate stage 1 checkpoints from stage 2
    stage1_ckpts = [c for c in checkpoints_sorted if "stage1" in c]
    stage2_ckpts = [c for c in checkpoints_sorted if "stage2" in c]

    logstage1 = sample_log_indices(20,stage1_ckpts)
    logstage2 = sample_log_indices(10,stage2_ckpts)

    # Now add the first and last checkpoints of each stage
    stage1 = [stage1_ckpts[0], logstage1, stage1_ckpts[-1]]
    stage1_flat = [elem for item in stage1 for elem in (item if isinstance(item, list) else [item])]

    stage2 = [stage2_ckpts[0], logstage2]
    stage2_flat = [elem for item in stage2 for elem in (item if isinstance(item, list) else [item])]

    both_stages = [stage1_flat, stage2_flat]

    selected = [elem for item in both_stages for elem in (item if isinstance(item, list) else [item])]
    #selected = random.sample(checkpoints, k=5)  # or use the entire list
    print(selected)

    # selected = [# "stage1-step102500-tokens860B",
    #             # "stage1-step337000-tokens2827B",
    #             # "stage1-step596057-tokens5001B"
    #             # "stage1-step0-tokens0B",
    #             # "stage1-step1000-tokens9B",
    #             # "stage1-step10000-tokens84B",
    #             # "stage1-step35000-tokens294B",
    #             # "stage1-step100000-tokens839B",
    #             # "stage1-step150000-tokens1259B",
    #             # "stage1-step50000-tokens420B",
    #             # "stage1-step75000-tokens630B",
    #             "stage1-step60000-tokens504B",
    #             "stage1-step90000-tokens755B",
    #             "stage1-step200000-tokens1678B",
    #             "stage1-step300000-tokens2517B",
    #             "stage1-step400000-tokens3356B",
    #             "stage1-step500000-tokens4195B",
    #             "stage2-ingredient1-step1000-tokens9B",

    #             ]

    #selected = ['stage1-step8000-tokens68B', 'stage1-step64000-tokens537B', 'stage1-step4000-tokens34B', 
     #           'stage1-step32000-tokens269B', 'stage1-step2000-tokens17B', 'stage1-step16000-tokens135B', 
       #         'stage1-step1000-tokens9B', 'stage1-step0-tokens0B', 'stage1-step512000-tokens4295B', 
           #     'stage1-step256000-tokens2148B', 'stage1-step128000-tokens1074B']

    """
    stage2 =     ['stage2-ingredient4-step32000-tokens269B', 'stage2-ingredient4-step16000-tokens135B', 
                'stage2-ingredient4-step8000-tokens68B', 'stage2-ingredient4-step4000-tokens34B', 
                'stage2-ingredient4-step2000-tokens17B', 'stage2-ingredient4-step1000-tokens9B', 
                'stage2-ingredient3-step8000-tokens68B', 'stage2-ingredient3-step4000-tokens34B', 
                'stage2-ingredient3-step2000-tokens17B', 'stage2-ingredient3-step1000-tokens9B', 
                'stage2-ingredient2-step8000-tokens68B', 'stage2-ingredient2-step2000-tokens17B', 
                'stage2-ingredient1-step8000-tokens68B', 'stage2-ingredient1-step4000-tokens34B', 
                'stage2-ingredient1-step2000-tokens17B', 'stage2-ingredient1-step1000-tokens9B', 
                'stage2-ingredient2-step4000-tokens34B']
    """

    for rev in selected:
        model_path = "allenai/OLMo-2-1124-13B"
        print(f"Running FB with checkpoint: {rev}")

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
        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath,filename)):
            print("Already run this model for this checkpoint.")
            continue

        main(model_path=model_path,
             revision=revision,  
             suffix=suffix)   
        
        