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

from scipy.stats import entropy
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
    all_layerheads_last_token = torch.stack([
        attn[0, :, last_token_idx, :]
        for attn in attentions
    ])
    # Shape: (num_layers, num_heads, seq_len)
    
    # Get token ids and tokens for reference
    token_ids = inputs['input_ids'][0].cpu().numpy()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    return all_layerheads_last_token, token_ids, tokens

def compute_attention_entropy(attention_scores):
    """
    Compute entropy over attention scores from the final token to all tokens.
    
    Args:
        attention_scores: np.ndarray of shape (seq_len,) or (batch_size, seq_len)
                         Attention weights from final token position to all tokens.
                         Should be normalized (sum to 1).
    
    Returns:
        float or np.ndarray: Entropy value(s). Higher entropy means more dispersed
                           attention, lower entropy means more focused attention.
    """
    if attention_scores.ndim == 1:
        # Single sequence: compute entropy directly
        return entropy(attention_scores)
    elif attention_scores.ndim == 2:
        # Batch of sequences: compute entropy for each
        return np.array([entropy(scores) for scores in attention_scores])
    else:
        raise ValueError("attention_scores must be 1D or 2D")

def get_max_attention_score_and_token_idx(attention_scores): 

    if attention_scores.ndim == 1:
        # Single sequence: compute entropy directly
        return torch.max(attention_scores).values
    elif attention_scores.ndim == 2:
        # Batch of sequences: compute entropy for each
        max_scores = torch.max(attention_scores,1).values
        max_idx = torch.max(attention_scores,1).indices
        return max_scores, max_idx
    else:
        raise ValueError("attention_scores must be 1D or 2D")


def main(model_path, revision = None, suffix=None):

    ### Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="auto",
        attn_implementation="eager"
        )

    # Set output_attentions on the config after loading
    model.config.output_attentions = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)

    ### Load data
    df_fb = pd.read_csv("../../data/raw/fb.csv")

    results = []
    ### Run model
    with tqdm(total=len(df_fb)) as pbar:
        for index, row in df_fb.iterrows():
            passage = row['passage'].replace(" [MASK].", "").strip()
            start_location = " " + row['start']
            end_location =  " " +row['end']

            # Get attention scores from final token to other tokens in passage
            all_layerheads_last_token, _, _ = get_attentions(model, tokenizer, passage)

            nlayers = all_layerheads_last_token.shape[0]
            nheads = all_layerheads_last_token.shape[1]

            # Compute entropy over attention scores for each layer/head
            # Iterate over layers
            for layer in range(all_layerheads_last_token.shape[0]):

                all_heads_scores = all_layerheads_last_token[layer]
                entropy_all_heads = compute_attention_entropy(all_heads_scores)
                max_attn_all_heads, token_idx = get_max_attention_score_and_token_idx(all_heads_scores)

                # Save max and entropy over attention scores for 
                # each layer's list of heads -- save head data as list to avoid creating so many 
                # rows that will have to store duplicate passages and all that other information

                # Save the token id that receives max attention score from this layer/head, from final token

                results.append({
                    'passage': row['passage'],
                    'start': row['start'],
                    'end': row['end'],
                    "layer": layer,
                    "headlist": np.arange(nheads),
                    "entropy_by_headlist": entropy_all_heads,
                    "max_attn_by_headlist": max_attn_all_heads,
                    "max_attn_tokenid_by_headlist": token_idx,
                    'knowledge_cue': row['knowledge_cue'],
                    'first_mention': row['first_mention'],
                    'recent_mention': row['recent_mention'],
                    'condition': row['condition']
                })

                pbar.update(1)

    ### Create DataFRame
    df_results = pd.DataFrame(results)
    df_results['model_path'] = model_dict["model_path"]
    df_results['model_shorthand'] = model_dict["name"]

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
    for model_dict in MODELS:

        model_path = model_dict["model_path"]
        for revision in model_dict["revisions"]:
            print(f"Processing {model_dict['name']} (revision: {revision})")
            print(f"  Path: {model_dict['model_path']}")

            suffix = revision.replace("/", "_") # to tag output files uniquely
            
            # Set up save path, filename, etc.
            savepath = f"../../data/processed/fb_attention_scores/"
            if not os.path.exists(savepath): 
                os.makedirs(savepath)

            if "/" in model_path:
                filename = f"fb-attentionscores-{model_path.split('/')[-1]}-{suffix}.csv"
            else:
                filename = f"fb-attentionscores-{model_path.split('/')[-1]}-{suffix}.csv"

            # Skip this checkpoint's analysis if you've already run it before
            print("Checking if we've already run this analysis...")
            if os.path.exists(os.path.join(savepath,filename)):
                print("Already run this model for this checkpoint.")
                continue

            # Run it!!
            main(model_path, revision, suffix)
            
            
