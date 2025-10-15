"""compute max attention and attention score entropy for
FB task by loading models from HF locally."""

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
from huggingface_hub import list_repo_refs

#from del_models import clear_huggingface_cache

# Specify the models, and target checkpoints, to characterize 
MODELS = [
    {
        "model_path": "EleutherAI/pythia-14m",
        "name": "Pythia 14M",
        "revisions": ["main"]
    },
    {
        "model_path": "allenai/OLMo-2-1124-13B",
        "name": "OLMO 2 13B",
        "revisions": ["stage1-step43000-tokens181B"]
    }
]

def get_attentions(model, tokenizer, passage):
    """Run model, return attention scores from final token to all other tokens 
    in the passage; and return token ids and actual tokens attended to"""
    
    # Tokenize the passage

    inputs = tokenizer(passage, return_tensors="pt").to(model.device)
    
    # Get the sequence length
    seq_len = inputs['input_ids'].shape[1]
    last_token_idx = seq_len - 1
    
    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True, output_hidden_states=False)
    
    attentions = output.attentions # Shape: [layer_idx][(batch, head, token_id_from, token_id_to)]

    # Get attentions from final token to all other tokens in the passage, 
    # for all layers in one go
    all_layers_last_token = torch.stack([
        attn[0, :, last_token_idx, :]
        for attn in attentions
    ])
    # Shape: (num_layers, num_heads, seq_len)
    
    return all_layers_last_token

def main(model_path, revision = None, suffix=None):

    ### Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="auto"
        )

    # Set output_attentions on the config after loading
    model.config.output_attentions = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)

    ### Load data
    ### Should be run from main
    df_fb = pd.read_csv("data/raw/fb.csv")

    results = []
    ### Run model
    with tqdm(total=len(df_fb)) as pbar:
        for index, row in df_fb.iterrows():
            passage = row['passage'].replace(" [MASK].", "").strip()
            start_location = " " + row['start']
            end_location =  " " +row['end']

            
            # Get attention scores from final token to other tokens in passage
            all_layers_last_token = get_attentions(model, tokenizer, passage)

            # Compute entropy over attention scores for each layer/head


            # Select layer/head with minimum entropy AND maximum attention
            # Save its score
            # Save the token id that receives max attention score from this layer/head, from final token


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
        df_results['step'] = int(step.replace("step", "")) if step else None
        df_results['tokens_seen'] = tokens.replace("tokens", "") if tokens else None
    else:
        df_results['stage'] = None
        df_results['ingredient'] = None
        df_results['step'] = None
        df_results['tokens_seen'] = None


    df_results.to_csv(os.path.join(savepath,filename), index=False)


if __name__ == "__main__":


    # Loop through models and revisions
    for model in MODELS:
        #test code: 
        model = MODELS[0]
        print(model)

        model_path = model["model_path"]
        for revision in model["revisions"]:
            print(revision)
            print(f"Processing {model['name']} (revision: {revision})")
            print(f"  Path: {model['model_path']}")

            suffix = revision.replace("/", "_") # to tag output files uniquely
            
            # Set up save path, filename, etc.
            savepath = f"../../data/processed/fb_attention_scores/"
            if not os.path.exists(savepath): 
                os.makedirs(savepath)

            if "/" in model_path:
                filename = f"fb-attentionscores-{model_path.split('/')[-1]}-{suffix}.csv"
            else:
                filename = f"fb-attentionscores-{model_path.split('/')[-1]}-{suffix}.csv"

            print(filename)
            print(savepath)

            main(model_path, revision, suffix)
            
            # Your Hugging Face loading code here
            # model_instance = load_model(model["path"], revision=revision)
