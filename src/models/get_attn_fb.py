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

from del_models import clear_huggingface_cache

# Specify the models to characterize 
MODELS = {
    ### OLMo
    #"EleutherAI/pythia-14m": "Pythia 14m",
    #"EleutherAI/pythia-1b": "Pythia 1B",
    #"EleutherAI/pythia-6.9b": "Pythia 6.9B",
    #"EleutherAI/pythia-12b": "Pythia 12B",
    "allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    #"allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    #"allenai/OLMo-2-0425-1B": "OLMO 2 1B"
    }

# TODO: specify the checkpoints you'll compute attention for


def main(model_path, revision = None, suffix=None):

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
    df_fb = pd.read_csv("../../data/raw/fb.csv")

    results = []
    ### Run model
    with tqdm(total=len(df_fb)) as pbar:
        for index, row in df_fb.iterrows():
            passage = row['passage'].replace(" [MASK].", "").strip()
            start_location = " " + row['start']
            end_location =  " " +row['end']

            # Tokenize the passage
			inputs = tokenizer(passage, return_tensors="pt").to(model.device)

	    	# Get attention scores
			with torch.no_grad():
			    outputs = model(**inputs, output_attentions=True)

			# # Get the sequence length
			seq_len = inputs['input_ids'].shape[1]
			last_token_idx = seq_len - 1

			# Extract attention FROM the last token
			attentions = outputs.attentions # Tuple of tensors, one per layer
			# Shape: (batch_size, num_heads, seq_len (token `from`), seq_len (token `to`))

			# For a specific layer (e.g., layer 30)
			layer_idx = 30
			last_token_attention = attentions[layer_idx][0, :, last_token_idx, :]
			# Shape: (num_heads, seq_len)
			# This shows what each head in the last token attends to

			# Or get for ALL layers at once
			all_layers_last_token = torch.stack([
			    attn[0, :, last_token_idx, :]
			    for attn in attentions
			])
			# Shape: (num_layers, num_heads, seq_len)


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


    	

