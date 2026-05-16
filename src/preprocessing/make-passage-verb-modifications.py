# factive, non-factive, and implicit verb stimuli

import os
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
df = df[df["knowledge_cue"] == "Explicit"]

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

			new_rows.append({"passage": passage,
				"condition": row["condition"],
				"knowledge_cue": row["knowledge_cue"],
				"tokenized_passage": row["tokenized_passage"],
				"start": row["start"],
				"end": row["end"],
				"first_mention": row["first_mention"],
				"recent_mention": row["recent_mention"],
				"verb_type": vtype,
				"verb": verb
			})


df_verb = pd.DataFrame(new_rows)

# then, save it










