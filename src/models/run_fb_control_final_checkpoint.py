"""Run FB task by loading models from HF locally."""

import argparse
import pandas as pd
import numpy as np
import torch
import os

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from del_models import clear_huggingface_cache


MODELS = {
    "EleutherAI/pythia-14m": "Pythia 14m",
    "EleutherAI/pythia-1b": "Pythia 1B",
    "EleutherAI/pythia-6.9b": "Pythia 6.9B",
    "EleutherAI/pythia-12b": "Pythia 12B",
    "allenai/OLMo-2-1124-13B": "OLMO 2 13B",
    "allenai/OLMo-2-1124-7B": "OLMO 2 7B",
    "allenai/OLMo-2-0425-1B": "OLMO 2 1B"
}


def next_seq_prob(model, tokenizer, seen, unseen):
    device = next(model.parameters()).device
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
        next_token_tensor = torch.tensor([[unseen_id]], device=device)
        input_ids = torch.cat((input_ids, next_token_tensor), dim=1)
    total_log_prob = sum(log_probs)
    total_prob = torch.exp(total_log_prob)
    return total_prob.item()


def main(model_path):

    savepath = "data/processed/fb_local_control_final/"
    os.makedirs(savepath, exist_ok=True)

    model_name = model_path.split('/')[-1]
    filename = f"fb-{model_name}-final.csv"
    output_path = os.path.join(savepath, filename)

    if os.path.exists(output_path):
        print(f"  Skipping {model_path} — already exists at {output_path}")
        return

    ### Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ### Load data
    df_fb = pd.read_csv("data/raw/fb_control.csv")

    results = []
    with tqdm(total=len(df_fb)) as pbar:
        for index, row in df_fb.iterrows():
            passage = row['passage_control'].replace(" [MASK].", "").strip()
            start_location = " " + row['start']
            end_location = " " + row['end']

            start_prob = next_seq_prob(model, tokenizer, passage, start_location)
            end_prob = next_seq_prob(model, tokenizer, passage, end_location)

            if start_prob == 0 or end_prob == 0:
                continue

            results.append({
                'start_prob': start_prob,
                'end_prob': end_prob,
                'passage_control': row['passage_control'],
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

    df_results = pd.DataFrame(results)
    df_results['model_path'] = model_path
    df_results['model_shorthand'] = MODELS[model_path]

    df_results.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FB task on final checkpoints")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional HuggingFace model id. If not set, iterate full roster.",
    )
    args = parser.parse_args()

    if args.model is not None:
        print("Running:", args.model)
        main(args.model)
    else:
        for model_path in MODELS.keys():
            print(f"Running FB for: {model_path}")
            main(model_path)
            clear_huggingface_cache()