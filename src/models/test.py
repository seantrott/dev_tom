#testing tag_passages.py

import re
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class TokenInfo:
    """Information about a token and its entity associations"""
    token: str
    token_id: int
    position: int
    is_entity: bool = False
    entity_type: str = None  # 'name' or 'pronoun'
    entity_name: str = None  # which name this refers to
    entity_occurrence: int = 0  # which occurrence of this entity
    attention_scores: Optional[np.ndarray] = None

class NamePronounTracker:
    def __init__(self):
        # Pronoun mappings based on gender/context
        self.pronoun_mappings = {
            'he': ['he', 'him', 'his', 'himself'],
            'she': ['she', 'her', 'hers', 'herself'],
            'they': ['they', 'them', 'their', 'theirs', 'themselves', 'themself']
        }
        
    def extract_proper_names(self, text: str) -> Set[str]:
        """Extract proper names from text"""
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        sentences = re.split(r'[.!?]+', clean_text)
        proper_names = set()
        
        for sentence in sentences:
            words = sentence.strip().split()
            for i, word in enumerate(words):
                if i == 0:
                    continue
                    
                if (word[0].isupper() and 
                    word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'] and
                    not word.isupper()):
                    proper_names.add(word)
        
        return proper_names
    
    def assign_pronouns_to_names(self, names: Set[str]) -> Dict[str, List[str]]:
        """Assign pronoun sets to names"""
        name_pronouns = {}
        
        for name in names:
            if name.lower() in ['sean', 'tim', 'ed', 'ross', 'david', 'robert', 'john', 'patrick', 'martin', 'james', 'cameron']:
                name_pronouns[name] = self.pronoun_mappings['he']
            elif name.lower() in ['anna', 'helen', 'paula', 'laura', 'marta', 'sarah', 'lisa', 'karen','mary', 'nicole', 'jennifer', 'hannah']:
                name_pronouns[name] = self.pronoun_mappings['she']
            else:
                name_pronouns[name] = self.pronoun_mappings['they']
                
        return name_pronouns

class AttentionAnalyzer:
    def __init__(self, model_name):
        """Initialize with a transformer model that outputs attention weights"""
        self.model_name = model_name
        self.model_revision = revision # pretraining checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True, revision = revision)
        self.model.eval()
        
        # Handle special tokens
        self.special_tokens = {
            'cls': self.tokenizer.cls_token,
            'sep': self.tokenizer.sep_token,
            'pad': self.tokenizer.pad_token,
            'unk': self.tokenizer.unk_token
        }
    
    def tokenize_with_entities(self, text: str, name_pronouns: Dict[str, List[str]]) -> List[TokenInfo]:
        """Tokenize text while tracking entity information"""
        # Clean text for processing
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        
        # Track entity occurrences
        name_counts = defaultdict(int)
        pronoun_counts = defaultdict(lambda: defaultdict(int))
        
        # First pass: identify entities in the original text
        words = clean_text.split()
        word_entity_map = {}  # word_index -> entity_info
        
        for word_idx, word in enumerate(words):
            clean_word = re.sub(r'[.,!?";:]', '', word).lower()
            
            # Check if it's a proper name
            for name in name_pronouns.keys():
                if name.lower() == clean_word:
                    name_counts[name] += 1
                    word_entity_map[word_idx] = {
                        'type': 'name',
                        'name': name,
                        'occurrence': name_counts[name]
                    }
                    break
            
            # If not a name, check if it's a pronoun
            if word_idx not in word_entity_map:
                for name, pronouns in name_pronouns.items():
                    if clean_word in pronouns:
                        pronoun_counts[name][clean_word] += 1
                        word_entity_map[word_idx] = {
                            'type': 'pronoun',
                            'name': name,
                            'occurrence': pronoun_counts[name][clean_word]
                        }
                        break
        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(clean_text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Map tokens back to words to preserve entity information
        token_info_list = []
        word_idx = 0
        word_tokens_consumed = 0
        current_word = words[0] if words else ""
        current_word_clean = re.sub(r'[.,!?";:]', '', current_word).lower()
        
        for token_idx, (token, token_id) in enumerate(zip(tokens, token_ids)):
            # Skip special tokens for now
            if token.startswith('##'):
                # Subword token
                token_piece = token[2:]
            else:
                token_piece = token
            
            # Create TokenInfo object
            token_info = TokenInfo(
                token=token,
                token_id=token_id,
                position=token_idx,
                is_entity=False
            )
            
            # Check if this token is part of an entity
            # This is a simplified approach - you might need more sophisticated alignment
            if word_idx < len(words) and word_idx in word_entity_map:
                if token_piece in current_word_clean or current_word_clean.startswith(token_piece):
                    entity_info = word_entity_map[word_idx]
                    token_info.is_entity = True
                    token_info.entity_type = entity_info['type']
                    token_info.entity_name = entity_info['name']
                    token_info.entity_occurrence = entity_info['occurrence']
            
            # Track progress through words
            if not token.startswith('##'):
                # Check if we need to move to the next word
                remaining_word = current_word_clean[word_tokens_consumed:]
                if token_piece == remaining_word or remaining_word.startswith(token_piece):
                    word_tokens_consumed += len(token_piece)
                    if word_tokens_consumed >= len(current_word_clean):
                        word_idx += 1
                        word_tokens_consumed = 0
                        if word_idx < len(words):
                            current_word = words[word_idx]
                            current_word_clean = re.sub(r'[.,!?";:]', '', current_word).lower()
                else:
                    # Move to next word
                    word_idx += 1
                    word_tokens_consumed = 0
                    if word_idx < len(words):
                        current_word = words[word_idx]
                        current_word_clean = re.sub(r'[.,!?";:]', '', current_word).lower()
                        # Check again for current word
                        if word_idx in word_entity_map and (token_piece in current_word_clean or current_word_clean.startswith(token_piece)):
                            entity_info = word_entity_map[word_idx]
                            token_info.is_entity = True
                            token_info.entity_type = entity_info['type']
                            token_info.entity_name = entity_info['name']
                            token_info.entity_occurrence = entity_info['occurrence']
            
            token_info_list.append(token_info)
        
        return token_info_list
    
    def get_attention_scores(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Get attention scores from the model"""
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        
        # Tokenize and encode
        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # Tuple of attention tensors for each layer
        
        # Get tokens for reference
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attentions, tokens
    
    def analyze_entity_attention(self, text: str, custom_name_pronouns: Dict[str, List[str]] = None) -> Dict:
        """Complete analysis: tokenize, track entities, and get attention scores"""
        tracker = NamePronounTracker()
        
        # Extract names and assign pronouns
        proper_names = tracker.extract_proper_names(text)
        if custom_name_pronouns:
            name_pronouns = custom_name_pronouns
        else:
            name_pronouns = tracker.assign_pronouns_to_names(proper_names)
        
        # Tokenize with entity tracking
        token_info_list = self.tokenize_with_entities(text, name_pronouns)
        
        # Get attention scores
        attentions, model_tokens = self.get_attention_scores(text)
        
        # Align our tokens with model tokens (they should match, but let's be safe)
        if len(token_info_list) != len(model_tokens):
            print(f"Warning: Token count mismatch. Our tokens: {len(token_info_list)}, Model tokens: {len(model_tokens)}")
        
        # Add attention scores to token info
        # We'll use the last layer's attention scores averaged across heads
        last_layer_attention = attentions[-1][0]  # [num_heads, seq_len, seq_len]
        avg_attention = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]
        
        for i, token_info in enumerate(token_info_list):
            if i < avg_attention.size(0):
                # Get attention scores for this token (how much it attends to all other tokens)
                token_info.attention_scores = avg_attention[i].numpy()
        
        # Organize results
        results = {
            'tokens': token_info_list,
            'entity_summary': defaultdict(list),
            'attention_by_entity': defaultdict(list),
            'full_attention_matrix': avg_attention.numpy(),
            'model_tokens': model_tokens
        }
        
        # Group entity tokens
        for token_info in token_info_list:
            if token_info.is_entity:
                entity_key = f"{token_info.entity_name}_{token_info.entity_type}_{token_info.entity_occurrence}"
                results['entity_summary'][token_info.entity_name].append(token_info)
                results['attention_by_entity'][entity_key] = token_info.attention_scores
        
        return results

def analyze_passage(text: str, custom_pronouns: Dict[str, List[str]] = None, model_name: str = "bert-base-uncased"):
    """Main function to run complete analysis"""
    analyzer = AttentionAnalyzer(model_name)
    results = analyzer.analyze_entity_attention(text, custom_pronouns)
    
    print("=== TOKENIZATION AND ENTITY TRACKING ===")
    print(f"Total tokens: {len(results['tokens'])}")
    print("\nTokens with entity information:")
    for token_info in results['tokens']:
        if token_info.is_entity:
            print(f"  {token_info.token} (pos: {token_info.position}) -> {token_info.entity_name} [{token_info.entity_type}_{token_info.entity_occurrence}]")
    
    print(f"\n=== ENTITY SUMMARY ===")
    for entity_name, token_list in results['entity_summary'].items():
        print(f"{entity_name}:")
        for token_info in token_list:
            print(f"  - {token_info.token} ({token_info.entity_type}_{token_info.entity_occurrence}) at position {token_info.position}")
    
    print(f"\n=== ATTENTION ANALYSIS ===")
    print(f"Attention matrix shape: {results['full_attention_matrix'].shape}")
    
    # Show average attention scores for each entity occurrence
    for entity_key, attention_scores in results['attention_by_entity'].items():
        avg_attention_received = np.mean(attention_scores)
        max_attention_from = np.argmax(attention_scores)
        print(f"{entity_key}:")
        print(f"  - Average attention received: {avg_attention_received:.4f}")
        print(f"  - Highest attention from token {max_attention_from}: {results['model_tokens'][max_attention_from]} ({attention_scores[max_attention_from]:.4f})")
    
    return results

# Example usage

MODELS = {
	"EleutherAI/pythia-14m":, "pythia-14m"
}

if __name__ == "__main__":
    # Your original text
    text = """Sean is reading a book. When he is done, he puts the book in the box and picks up a sweater from the basket. Then, Anna comes into the room. Sean watches Anna move the book to the basket from the box. Sean leaves to get something to eat in the kitchen. Sean comes back into the room and wants to read more of his book."""
    
    # Custom pronoun assignments
    custom_pronouns = {
        'Sean': ['he', 'him', 'his', 'himself'],
        'Anna': ['she', 'her', 'hers', 'herself']
    }

    model_path = MODELS[0]
    
    print("Analyzing passage with BERT...")
    results = analyze_passage(text, custom_pronouns, "bert-base-uncased")
    
    # You can also try with other models like:
    # results = analyze_passage(text, custom_pronouns, "distilbert-base-uncased")
    # results = analyze_passage(text, custom_pronouns, "roberta-base")