"""Run FB task by loading models from HF locally."""

import pandas as pd
import numpy as np
import transformers
import torch
import os
import random

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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="auto",
        use_auth_token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
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

    df_results.to_csv(os.path.join(savepath,filename), index=False)




if __name__ == "__main__":

    # Set base path to the directory where checkpoints are saved
    refs = list_repo_refs("allenai/OLMo-2-1124-13B")
    checkpoints = [r.name for r in refs.branches if r.name.startswith("step")]

    # sample or pick some
    selected = random.sample(checkpoints, k=5)  # or use the entire list
    print(selected)

    for rev in selected:
        model_path = "allenai/OLMo-2-1124-13B"
        print(f"Running FB with checkpoint: {rev}")
        main(
            model_path=model_path,
            revision=rev,  # pass revision into main
            suffix=rev     # to tag output files uniquely
        )