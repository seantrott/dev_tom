# Here, you'll grab the dataframe of unique false belief passages annotated by region of interest, as 
# well as each LM's checkpoint-layer-head's attention scores for each token in the passage (this is 
# attention from the final token to every other token in the passage). 

# You'll then output a dataframe containing, among others, these key columns:
# MODEL_SHORTHAND | STEP | LAYER | HEAD | PASSAGE | TOKENIZED_PASSAGE | AVG_ATTN_SETUP | AVG_ATTN_EXIT | ...
# AVG_ATTN_MANIPULATION | AVG_ATTN_RETURN | AVG_ATTN_QUERY

# Note: this script will only work for Olmo for now, because I annotated the passage Rois according 
# to Olmo's tokenization, and not any other model's, so the roi spans only work for Olmo rows in the df 

import ast
import os 

import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm


# Define a function that cleans up the attention scores (which for some reason are interpreted as strings rather than numpy arrays or lists)
def parse_attention_string(s):
    """
    Convert a stringified numpy array to a 1D numpy array.
    """
    if isinstance(s, np.ndarray):
        return s
    if isinstance(s, list):
        return np.asarray(s)
    return np.fromstring(s.strip("[]"), sep=" ")

# Define a function to clean up the roi tuple entries (which also look like strings)
import ast
import numpy as np

def parse_roi_span_string(x):
    """
    Parse an ROI span stored as a string into a (start, end) tuple.

    Returns:
        (start, end) tuple, or None if missing / invalid
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    # Already parsed
    if isinstance(x, tuple):
        return x
    if isinstance(x, str):
        parsed = ast.literal_eval(x)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            return parsed
    return None



# Define a function to average attention scores over regions of interest in false belief passages
def mean_attn_roi(attn, span): 
	""" attn: list or 1D array of attention scores
		span: tuple (start, end)"""
	if span is None or len(span) != 2:
		return np.nan 
	start, end = span
	# spans are inclusive, so use end + 1
	return float(np.mean(attn[start:end + 1]))


# Load the annotated passages
path_annot = "../../data/raw/fb_annotated_passages.csv"
df_annot = pd.read_csv(path_annot)

# Clean up the data types for the roi indices
roi_colnames = ['setup_indices', 'exit_indices', 'manipulation_indices', 'return_indices', 'query_indices']
for roi in roi_colnames:
	df_annot[roi] = df_annot[roi].apply(parse_roi_span_string)

# Load the attention scores data and bind them all into a single dataframe
path_scores = "../../data/processed/fb_attention_scores/"
files = os.listdir(path_scores)
files = [f for f in files if not f.endswith(".zip")]
files = [f for f in files if not f.startswith(".")]
df_scores = []
for file in files: 
	df_scores.append(pd.read_csv(os.path.join(path_scores, file)))

df = pd.concat(df_scores)

# Clean up the data type for attention scores 
df["attention_scores"] = df["attention_scores"].apply(parse_attention_string)

# Now, merge df_annot into df

df_merged = df.merge(df_annot[["tokenized_passage"] + roi_colnames], on="tokenized_passage", how="left")

#### OOOHHH pythia is tokenized differently, so the indices would be totally different! Remove for now
df_target_models = df_merged[~(df_merged["model_shorthand"] == "Pythia 14M")]
for roi in tqdm(roi_colnames):
	df_target_models[f"attn_avg_{roi}"] = df_target_models.apply(lambda row: mean_attn_roi(row["attention_scores"],row[roi]),axis=1)

# Create folder to save, if doesn't already exist
savepathname = "../../data/processed/mean_attn_rois/"
if not os.path.exists(savepathname):
	os.mkdir(savepathname)

# Split the dataframe by model_shorthand, to see if you can save it more effectively
model_names = set(df_target_models["model_shorthand"].values)
for m in model_names: 
	subdf = df_target_models[df_target_models["model_shorthand"] == m]
	mpath = subdf["model_path"].iloc[0].split("/")[1]
	filename = "fb-mean-attn-roi-" + mpath
	# Save the dataframe
	subdf.to_csv(os.path.join(savepathname,filename))















