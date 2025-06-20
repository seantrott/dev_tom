"""Run FB task by loading models from HF locally."""

import pandas as pd
import numpy as np
import transformers
import torch
import os
import random

from scipy.spatial.distance import cosine

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




def run_model(model, tokenizer, sentence):
    """Run model, return hidden states and attention"""
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}


def get_embedding(hidden_states, inputs, tokenizer, target, layer):
    """Extract embedding for TARGET from set of hidden states and token ids."""
    
    # Tokenize target
    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False)
    
    # Get indices of target in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )

    # Get layer
    selected_layer = hidden_states[layer][0]

    #grab just the embeddings for your target word's token(s)
    token_embeddings = selected_layer[target_inds[0]:target_inds[1]]

    #if a word is represented by >1 tokens, take mean
    #across the multiple tokens' embeddings
    embedding = torch.mean(token_embeddings, dim=0)
    
    return embedding

def find_sublist_index(mylist, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list"""

    for i in range(len(mylist)):
        if mylist[i] == sublist[0] and mylist[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None


def main(model_path, revision = None, suffix=None):

    # Set up save path, filename, etc.
    savepath = f"data/processed/rawc_local/"
    if not os.path.exists(savepath): 
        os.makedirs(savepath)

    if "/" in model_path:
        filename = f"rawc-{model_path.split('/')[-1]}-{suffix}.csv"
    else:
        filename = f"rawc-{model_path.split('/')[-1]}-{suffix}.csv"

    print(filename)
    print(savepath)

    ### Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        device_map="auto",
        use_auth_token=True,
        output_hidden_states = True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)


    ### Load data
    df = pd.read_csv("data/raw/rawc_stimuli.csv")
    df = df[df['Class']=='N']

    results = []

    n_layers = model.config.num_hidden_layers
    print("Number of layers:", n_layers)

    ### Run model
    for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

        ### Get word
        target = " {w}".format(w = row['string'])

        ### Run model for each sentence
        s1_outputs = run_model(model, tokenizer, row['sentence1'])
        s2_outputs = run_model(model, tokenizer, row['sentence2'])

        ### Now, for each layer...
        for layer in range(n_layers+1): # `range` is non-inclusive for the last value of interval

            ### Get embeddings for word
            s1 = get_embedding(s1_outputs['hidden_states'], s1_outputs['tokens'], tokenizer, target, layer)
            s2 = get_embedding(s2_outputs['hidden_states'], s2_outputs['tokens'], tokenizer, target, layer)

            ### Now calculate cosine distance 
            model_cosine = cosine(s1, s2)


            if row['same'] == True:
                same_sense = "Same Sense"
            else:
                same_sense = "Different Sense"


            ### Figure out how many tokens you're
            ### comparing across sentences
            n_tokens_s1 = len(tokenizer.encode(row['sentence1']))
            n_tokens_s2 = len(tokenizer.encode(row['sentence2']))

            ### Add to results dictionary
            results.append({
                'sentence1': row['sentence1'],
                'sentence2': row['sentence2'],
                'word': row['word'],
                'string': row['string'],
                'Same_sense': same_sense,
                'Distance': model_cosine,
                'Layer': layer,
                'mean_relatedness': row['mean_relatedness'],
                'S1_ntokens': n_tokens_s1,
                'S2_ntokens': n_tokens_s2
            })


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
    # sample or pick some
    selected = random.sample(checkpoints, k=5)  # or use the entire list
    print(selected)

    selected = ["stage1-step102500-tokens860B",
                "stage1-step337000-tokens2827B",
                # "stage1-step596057-tokens5001B" ### TODO: Causes error?
                "stage1-step0-tokens0B",
                "stage1-step1000-tokens9B",
                "stage1-step10000-tokens84B",
                "stage1-step35000-tokens294B",
                # "stage1-step100000-tokens839B",
                # "stage1-step150000-tokens1259B",

                ]

    for rev in selected:
        model_path = "allenai/OLMo-2-1124-13B"
        print(f"Running RAW-C with checkpoint: {rev}")
        main(
            model_path=model_path,
            revision=rev,  # pass revision into main
            suffix=rev.replace("/", "_")     # to tag output files uniquely
        )