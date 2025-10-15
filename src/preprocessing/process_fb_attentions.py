"""Create by-step and final-step summaries of attentions."""


import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# Directory containing your attention CSVs
input_dir = "data/processed/fb_attention_scores"
output_dir = "data/processed/fb_attention_summaries"
os.makedirs(output_dir, exist_ok=True)

# Get all CSV file paths
csv_files = glob(os.path.join(input_dir, "*.csv"))

# Columns we need to load (saves memory)
usecols = [
    "model_path", "model_shorthand", "stage", "step", "tokens_seen", "layer", "head", 
    "condition", "knowledge_cue", "first_mention", "recent_mention",
    "entropy", "max_attn"
]

# Loop through each file and summarize
for filepath in tqdm(csv_files):
    try:
        df = pd.read_csv(filepath, usecols=usecols)

        # Group and aggregate
        summary = (
            df.groupby(["model_path", "model_shorthand", "layer", "head", "condition", "knowledge_cue", "first_mention", "recent_mention"])
              .agg(
                  mean_entropy=("entropy", "mean"),
                  mean_max_attn=("max_attn", "mean")
              )
              .reset_index()
        )


        # Save the summary to disk
        base = os.path.basename(filepath).replace(".csv", "_summary.csv")
        outpath = os.path.join(output_dir, base)
        summary.to_csv(outpath, index=False)

    except Exception as e:
        print(f"Failed on {filepath}: {e}")


