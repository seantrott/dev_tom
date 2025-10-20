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
    #{
    #    "model_path": "EleutherAI/pythia-14m",
    #    "name": "Pythia 14M",
    #    "revisions": ["main"]
    #},
    {
        "model_path": "allenai/OLMo-2-1124-13B",
        "name": "OLMO 2 13B",
        "revisions": ["stage1-step0-tokens0B", #first step
        "stage1-step74000-tokens621B", #step at median fb mean_accuracy
        "stage1-step596057-tokens5001B"] #final step
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
        return 
    else:
        raise ValueError("attention_scores must be 1D or 2D")


def main(model_path, revision = None, suffix=None, name=None):

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
    ### TODO: Note, run from root dir in dev_tom
    df_fb = pd.read_csv("data/raw/fb.csv")

    results = []
    ### Run model
    with tqdm(total=len(df_fb)) as pbar:
        for index, row in df_fb.iterrows():
            passage = row['passage'].replace(" [MASK].", "").strip()

            tokenized_passage = tokenizer(passage)["input_ids"]
            tokenized_passage = tokenizer.convert_ids_to_tokens(tokenized_passage)

            start_location = " " + row['start']
            end_location = " " + row['end']

            # Get attention scores from final token to other tokens in passage
            all_layerheads_last_token, token_ids, tokens = get_attentions(model, tokenizer, passage)
            
            # Convert to numpy for easier computation
            ### TODO: This might change on Lambda!
            attn_array = all_layerheads_last_token.cpu().numpy()
            num_layers, num_heads, seq_len = attn_array.shape

            # Iterate over layers and heads
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    # Get attention distribution for this specific head
                    attention_scores = attn_array[layer_idx, head_idx, :]  # Shape: (seq_len,)
                    
                    # Compute entropy over attention scores
                    attn_entropy = entropy(attention_scores)
                    
                    # Get max attention value
                    max_attn = np.max(attention_scores)
                    
                    # Get index of max attention
                    max_attn_idx = np.argmax(attention_scores)
                    
                    # Get the token that received max attention
                    max_attn_token = tokens[max_attn_idx]
                    max_attn_token_id = token_ids[max_attn_idx]


                    # Append one row per layer/head
                    results.append({
                        'passage': row['passage'],
                        "tokenized_passage": tokenized_passage,
                        'start': row['start'],
                        'end': row['end'],
                        'knowledge_cue': row['knowledge_cue'],
                        'first_mention': row['first_mention'],
                        'recent_mention': row['recent_mention'],
                        'condition': row['condition'],
                        'layer': layer_idx,
                        'head': head_idx,
                        'entropy': attn_entropy,
                        'max_attn': max_attn,
                        'max_attn_idx': max_attn_idx,
                        'max_attn_token': max_attn_token,
                        'max_attn_token_id': max_attn_token_id,
                        "attention_scores": attention_scores
                    })

            pbar.update(1)

    ### Create DataFrame
    df_results = pd.DataFrame(results)
    df_results['model_path'] = model_path
    df_results['model_shorthand'] = model_shorthand


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
        model_shorthand = model_dict['name']
        for revision in model_dict["revisions"]:
            print(f"Processing {model_dict['name']} (revision: {revision})")
            print(f"  Path: {model_dict['model_path']}")

            suffix = revision.replace("/", "_") # to tag output files uniquely
            
            # Set up save path, filename, etc.
            savepath = f"data/processed/fb_attention_scores/"
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

            main(model_path, revision, suffix, model_shorthand)
            
            # Your Hugging Face loading code here
            # model_instance = load_model(model["path"], revision=revision)