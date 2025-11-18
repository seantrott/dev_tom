import os 
import pandas as pd
import sys
import ast

class InteractiveAnnotator:
    def __init__(self, df):
        """
        Initialize the annotator with a dataframe containing 'passage' and 'tokenized_passage' columns.
        """
        self.df = df.copy()
        self.roi_types = ['setup', 'exit', 'manipulation', 'return', 'query']
        
        # Convert tokenized_passage from string to list if needed
        self._parse_tokenized_passages()
        
        # Add columns for each ROI type if they don't exist
        for roi in self.roi_types:
            col_name = f'{roi}_indices'
            if col_name not in self.df.columns:
                self.df[col_name] = None
    
    def _parse_tokenized_passages(self):
        """Convert tokenized_passage column from string to list if it's stored as a string."""
        for idx in range(len(self.df)):
            tokens = self.df.at[idx, 'tokenized_passage']
            
            # Check if it's a string representation of a list
            if isinstance(tokens, str):
                try:
                    # Use ast.literal_eval to safely parse the string
                    parsed_tokens = ast.literal_eval(tokens)
                    self.df.at[idx, 'tokenized_passage'] = parsed_tokens
                    print(f"Parsed tokenized_passage for row {idx}")
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse tokenized_passage for row {idx}: {e}")
            elif isinstance(tokens, list):
                # Already a list, no action needed
                pass
            else:
                print(f"Warning: Unexpected type for tokenized_passage in row {idx}: {type(tokens)}")
    
    def display_tokens(self, tokens):
        """Display tokens with their indices in a readable format."""
        print("\nTokenized Passage (with indices):")
        print("="*80)
        
        # Display in chunks of 10 tokens per line
        for i in range(0, len(tokens), 10):
            chunk = tokens[i:i+10]
            
            # Print indices
            indices_line = ""
            for j, token in enumerate(chunk):
                idx = i + j
                indices_line += f"{idx:3d} "
            print(indices_line)
            
            # Print tokens (clean up Ġ for display)
            tokens_line = ""
            for token in chunk:
                display_token = token.replace('Ġ', ' ')
                tokens_line += f"{display_token:3s} "
            print(tokens_line)
            print()
    
    def get_token_range(self, roi_name, max_idx):
        """Prompt user to enter start and end indices for a region."""
        while True:
            try:
                user_input = input(f"\nEnter token range for {roi_name.upper()} (format: start-end or start,end), 'skip', 'none', or 'back': ").strip()
                
                if user_input.lower() == 'skip':
                    return 'skip'
                
                if user_input.lower() == 'back':
                    return 'back'
                
                if user_input.lower() == 'none':
                    return None
                
                # Parse input (accept both - and , as separators)
                if '-' in user_input:
                    start, end = user_input.split('-')
                elif ',' in user_input:
                    start, end = user_input.split(',')
                else:
                    print("Invalid format. Please use 'start-end' or 'start,end'")
                    continue
                
                start = int(start.strip())
                end = int(end.strip())
                
                if start < 0 or end >= max_idx or start > end:
                    print(f"Invalid range. Must be between 0 and {max_idx-1}, with start <= end")
                    continue
                
                return (start, end)
                
            except ValueError:
                print("Invalid input. Please enter numbers in format 'start-end' or 'start,end'")
            except Exception as e:
                print(f"Error: {e}")
    
    def show_selection_preview(self, tokens, start, end):
        """Show what text the selected token range represents."""
        selected_tokens = tokens[start:end+1]
        text = ''.join(token.replace('Ġ', ' ') for token in selected_tokens).strip()
        print(f"\n  Selected text: {text}")
    
    def annotate_passage(self, idx):
        """Annotate a single passage."""
        row = self.df.iloc[idx]
        passage = row['passage']
        tokens = row['tokenized_passage']
        
        print("\n" + "="*80)
        print(f"ANNOTATING PASSAGE {idx + 1} of {len(self.df)}")
        print("="*80)
        print(f"\nOriginal Passage:")
        print(passage)
        
        self.display_tokens(tokens)
        
        print("\nROI Types to annotate:")
        for i, roi in enumerate(self.roi_types, 1):
            print(f"  {i}. {roi.upper()}")
        
        # Annotate each ROI
        roi_idx = 0
        while roi_idx < len(self.roi_types):
            roi = self.roi_types[roi_idx]
            col_name = f'{roi}_indices'
            
            # Check if already annotated
            existing = self.df.at[idx, col_name]
            if existing is not None and existing != 'skip':
                print(f"\n{roi.upper()} already annotated: {existing}")
                overwrite = input("Overwrite? (y/n/back): ").strip().lower()
                if overwrite == 'back':
                    if roi_idx > 0:
                        roi_idx -= 1
                        print(f"\nGoing back to {self.roi_types[roi_idx].upper()}")
                        continue
                    else:
                        print("Already at first ROI")
                        continue
                elif overwrite != 'y':
                    roi_idx += 1
                    continue
            
            result = self.get_token_range(roi, len(tokens))
            
            if result == 'back':
                if roi_idx > 0:
                    roi_idx -= 1
                    print(f"\nGoing back to {self.roi_types[roi_idx].upper()}")
                else:
                    print("Already at first ROI. Type 'back' at passage level to go to previous passage.")
                continue
            
            if result == 'skip':
                print(f"Skipped {roi.upper()}")
                roi_idx += 1
                continue
            
            if result is None:
                print(f"{roi.upper()} marked as not present")
                self.df.at[idx, col_name] = None
                roi_idx += 1
            else:
                start, end = result
                self.show_selection_preview(tokens, start, end)
                
                # Confirm
                confirm = input("Confirm this selection? (y/n/back): ").strip().lower()
                if confirm == 'y':
                    self.df.at[idx, col_name] = (start, end)
                    print(f"✓ {roi.upper()} saved: ({start}, {end})")
                    roi_idx += 1
                elif confirm == 'back':
                    if roi_idx > 0:
                        roi_idx -= 1
                        print(f"\nGoing back to {self.roi_types[roi_idx].upper()}")
                    else:
                        print("Already at first ROI")
                else:
                    print(f"Cancelled {roi.upper()}, try again")
        
        print("\n" + "="*80)
        print("Passage annotation complete!")
        print("="*80)
    
    def run(self, start_idx=0):
        """Run the interactive annotation process."""
        print("\n" + "="*80)
        print("INTERACTIVE FALSE BELIEF PASSAGE ANNOTATOR")
        print("="*80)
        print("\nInstructions:")
        print("- For each ROI type, enter the token range as 'start-end' (e.g., '0-33')")
        print("- Type 'skip' to skip an ROI for now")
        print("- Type 'none' if the ROI is not present in the passage")
        print("- Type 'back' to go back to the previous ROI or passage")
        print("- Indices are inclusive (both start and end tokens are included)")
        print("\n")
        
        idx = start_idx
        while idx < len(self.df):
            self.annotate_passage(idx)
            
            if idx < len(self.df) - 1:
                continue_input = input("\nContinue to next passage? (y/n/b=back/q=quit): ").strip().lower()
                if continue_input == 'q':
                    print("Annotation session ended.")
                    break
                elif continue_input == 'n':
                    print(f"Pausing. Resume later by calling run(start_idx={idx + 1})")
                    break
                elif continue_input == 'b' or continue_input == 'back':
                    if idx > 0:
                        idx -= 1
                        print(f"\nGoing back to passage {idx + 1}")
                    else:
                        print("Already at first passage")
                else:
                    idx += 1
            else:
                idx += 1
        
        return self.df
    
    def save(self, filename):
        """Save the annotated dataframe."""
        self.df.to_csv(filename, index=False)
        print(f"\nAnnotations saved to {filename}")
    
    def get_dataframe(self):
        """Return the annotated dataframe."""
        return self.df


# Example usage
if __name__ == "__main__":
    # Sample data
    #sample_data = {
     #   'passage': [
     #       "David and Marta go out to get some wine for the party. When they get home, David stores the wine in the garage and grabs a drink from the fridge. Then, David goes out to get some snacks. While David is gone, Marta decides the wine would be best cooled, so she moves the wine out of the garage and into the fridge. David returns home and wants to put out the wine. David thinks the wine is in the [MASK]."
     #   ],
     #   'tokenized_passage': [
      #      ['David', 'Ġand', 'ĠMart', 'a', 'Ġgo', 'Ġout', 'Ġto', 'Ġget', 'Ġsome', 'Ġwine', 'Ġfor', 'Ġthe', 'Ġparty', '.', 'ĠWhen', 'Ġthey', 'Ġget', 'Ġhome', ',', 'ĠDavid', 'Ġstores', 'Ġthe', 'Ġwine', 'Ġin', 'Ġthe', 'Ġgarage', 'Ġand', 'Ġgrabs', 'Ġa', 'Ġdrink', 'Ġfrom', 'Ġthe', 'Ġfridge', '.', 'ĠThen', ',', 'ĠDavid', 'Ġgoes', 'Ġout', 'Ġto', 'Ġget', 'Ġsome', 'Ġsnacks', '.', 'ĠWhile', 'ĠDavid', 'Ġis', 'Ġgone', ',', 'ĠMart', 'a', 'Ġdecides', 'Ġthe', 'Ġwine', 'Ġwould', 'Ġbe', 'Ġbest', 'Ġcooled', ',', 'Ġso', 'Ġshe', 'Ġmoves', 'Ġthe', 'Ġwine', 'Ġout', 'Ġof', 'Ġthe', 'Ġgarage', 'Ġand', 'Ġinto', 'Ġthe', 'Ġfridge', '.', 'ĠDavid', 'Ġreturns', 'Ġhome', 'Ġand', 'Ġwants', 'Ġto', 'Ġput', 'Ġout', 'Ġthe', 'Ġwine', '.', 'ĠDavid', 'Ġthinks', 'Ġthe', 'Ġwine', 'Ġis', 'Ġin', 'Ġthe']
      #  ]
   # }
    
    df = pd.read_csv(os.path.join("../../data/raw/","fb_unique_passages.csv"))
    
    # Create annotator and run
    annotator = InteractiveAnnotator(df)
    
    # Run the interactive annotation
    annotated_df = annotator.run()
    
    # Save results
    annotator.save('/../../data/raw/fb_annotated_passages.csv')
    
    # Or get the dataframe
    final_df = annotator.get_dataframe()
    print("\nFinal DataFrame structure:")
    print(final_df.columns.tolist())


