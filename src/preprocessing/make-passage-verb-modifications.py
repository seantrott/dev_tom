# factive, non-factive, and implicit verb stimuli


import os

import numpy as np
import pandas as pd



verb_dict = {"knows": "factive", 
	"understands": "factive", 
	"recognizes": "factive", 
	"thinks": "nonfactive",
	"assumes": "nonfactive",
	"believes": "nonfactive",
	"suspects": "nonfactive"
}

datapath = "data/raw/fb_unique_passages.csv"
df_full = pd.read_csv(datapath)

# filter the original dataframe to grab only Explicit passages
# this avoids replacing verbs like "goes to" 
df = df_full[df_full["knowledge_cue"] == "Explicit"]


# iterate across the verbs in the dictionary
# for every passage query that contains the verb knows, replace with the factives
# add a column called verb-type (factive/nonfactive) and a column called verb (input the actual verb)
# if have already seen the verb (e.g. knows or thinks), just add the columns and move on
# ignore the passage queries that contain "goes to" (filter out Implicit knowledge-cues)


new_rows = []
for verb, vtype in verb_dict.items():
	#print(vtype + " " + verb)

	for i, row in df.iterrows():

		original_passage = row["passage"]

		if not verb in original_passage: 

			passage = original_passage.replace("thinks", verb)

			# note: getting rid of the tokenized passage bc passage 
			# will need to be re-tokenized during the model run to 
			# account for the new verbs
			new_rows.append({"passage": passage,
				"condition": row["condition"],
				"knowledge_cue": row["knowledge_cue"],
				"start": row["start"],
				"end": row["end"],
				"first_mention": row["first_mention"],
				"recent_mention": row["recent_mention"],
				"verb_type": vtype,
				"verb": verb
			})


df_verb = pd.DataFrame(new_rows)

# grab the implicit rows from the original dataframe
df_implicit = df_full[df_full["knowledge_cue"] == "Implicit"]

# add columns for verb type and verb to match the new dataframe we've made
df_implicit["verb_type"] = np.repeat("neutral", df_implicit.shape[0])
df_implicit["verb"] = np.repeat("goes", df_implicit.shape[0])
df_implicit = df_implicit.drop("tokenized_passage", axis=1)


# concatenate this to the dataframe we've made 

df_verb = pd.concat([df_verb,df_implicit])

# then, save it!
savepath = "data/raw/"
filename = "fb_multi_verbs.csv"
df_verb.to_csv(os.path.join(savepath, filename))










