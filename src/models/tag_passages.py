import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class NamePronounTracker:
    def __init__(self):
        # Pronoun mappings based on gender/context
        self.pronoun_mappings = {
            'he': ['he', 'him', 'his', 'himself'],
            'she': ['she', 'her', 'hers', 'herself'],
            'they': ['they', 'them', 'their', 'theirs', 'themselves', 'themself']
        }
        
        # Common proper name patterns (you can expand this)
        self.name_indicators = ['Mr.', 'Ms.', 'Mrs.', 'Dr.']
        
    def extract_proper_names(self, text: str) -> Set[str]:
        """Extract proper names from text (capitalized words that aren't at sentence start)"""
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        
        # Split into sentences to avoid capturing sentence-starting words
        sentences = re.split(r'[.!?]+', clean_text)
        proper_names = set()
        
        for sentence in sentences:
            words = sentence.strip().split()
            for i, word in enumerate(words):
                # Skip first word of sentence (could be capitalized for grammar)
                if i == 0:
                    continue
                    
                # Check if word is capitalized and likely a proper name
                if (word[0].isupper() and 
                    word.lower() not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'] and
                    not word.isupper()):  # Avoid acronyms
                    proper_names.add(word)
        
        return proper_names
    
    def assign_pronouns_to_names(self, names: Set[str]) -> Dict[str, List[str]]:
        """Assign pronoun sets to names based on context or default assumptions"""
        name_pronouns = {}
        
        for name in names:
            # Simple heuristic - you might want to enhance this with a name database
            # For now, assume traditional gender associations or allow manual specification
            if name.lower() in ['sean', 'john', 'mike', 'david', 'james', 'robert']:
                name_pronouns[name] = self.pronoun_mappings['he']
            elif name.lower() in ['anna', 'sarah', 'lisa', 'mary', 'jennifer', 'emily']:
                name_pronouns[name] = self.pronoun_mappings['she']
            else:
                # Default to 'they' for unknown names
                name_pronouns[name] = self.pronoun_mappings['they']
                
        return name_pronouns
    
    def tag_text(self, text: str, custom_name_pronouns: Dict[str, List[str]] = None) -> str:
        """Tag the text with name/pronoun tracking"""
        # Extract proper names
        proper_names = self.extract_proper_names(text)
        
        # Assign pronouns to names
        if custom_name_pronouns:
            name_pronouns = custom_name_pronouns
        else:
            name_pronouns = self.assign_pronouns_to_names(proper_names)
        
        # Create tracking counters
        name_counts = defaultdict(int)
        pronoun_counts = defaultdict(lambda: defaultdict(int))
        
        # Process text word by word
        words = text.split()
        tagged_words = []
        
        for word in words:
            # Clean word of markdown and punctuation for matching
            clean_word = re.sub(r'[*.,!?"]', '', word).lower()
            original_word = word
            tagged = False
            
            # Check if it's a proper name
            for name in proper_names:
                if name.lower() == clean_word:
                    name_counts[name] += 1
                    tag = f"[{name}_{name_counts[name]}]"
                    tagged_words.append(f"{original_word}{tag}")
                    tagged = True
                    break
            
            if not tagged:
                # Check if it's a pronoun for any of our names
                for name, pronouns in name_pronouns.items():
                    if clean_word in pronouns:
                        pronoun_counts[name][clean_word] += 1
                        tag = f"[{name}_{clean_word}_{pronoun_counts[name][clean_word]}]"
                        tagged_words.append(f"{original_word}{tag}")
                        tagged = True
                        break
            
            if not tagged:
                tagged_words.append(original_word)
        
        return ' '.join(tagged_words)
    
    def get_tracking_summary(self, text: str, custom_name_pronouns: Dict[str, List[str]] = None) -> Dict:
        """Get a summary of all tracked references"""
        proper_names = self.extract_proper_names(text)
        
        if custom_name_pronouns:
            name_pronouns = custom_name_pronouns
        else:
            name_pronouns = self.assign_pronouns_to_names(proper_names)
        
        summary = defaultdict(lambda: {'name_count': 0, 'pronouns': defaultdict(int)})
        
        words = text.split()
        for word in words:
            clean_word = re.sub(r'[*.,!?"]', '', word).lower()
            
            # Count proper names
            for name in proper_names:
                if name.lower() == clean_word:
                    summary[name]['name_count'] += 1
            
            # Count pronouns
            for name, pronouns in name_pronouns.items():
                if clean_word in pronouns:
                    summary[name]['pronouns'][clean_word] += 1
        
        return dict(summary)

# Example usage
if __name__ == "__main__":
    # Your original text
    text = """**Sean** is reading a book. When **he** is done, **he** **puts the book in the box** and picks up a sweater from the basket. Then, **Anna** comes into the room. **Sean** watches **Anna move the book to the basket** from the box. **Sean** leaves to get something to eat in the kitchen. **Sean** comes back into the room and wants to read more of **his** book."""
    
    # Initialize tracker
    tracker = NamePronounTracker()
    
    # Option 1: Use automatic pronoun assignment
    print("=== Automatic Pronoun Assignment ===")
    tagged_text = tracker.tag_text(text)
    print(tagged_text)
    print()
    
    # Option 2: Use custom pronoun assignments
    print("=== Custom Pronoun Assignment ===")
    custom_pronouns = {
        'Sean': ['he', 'him', 'his', 'himself'],
        'Tim': ['he', 'him', 'his', 'himself'],
        'Ed': ['he', 'him', 'his', 'himself'],
        'Ross': ['he', 'him', 'his', 'himself'],
        'David': ['he', 'him', 'his', 'himself'],
        'Robert': ['he', 'him', 'his', 'himself'],
        'John': ['he', 'him', 'his', 'himself'],
        'Patrick': ['he', 'him', 'his', 'himself'],
        'Martin': ['he', 'him', 'his', 'himself'],
        'Anna': ['she', 'her', 'hers', 'herself'],
        'Mary': ['she', 'her', 'hers', 'herself'],
        'James': ['he', 'him', 'his', 'himself'],
        'Cameron': ['he', 'him', 'his', 'himself'],
        'Helen': ['she', 'her', 'hers', 'herself'],
        'Paula': ['she', 'her', 'hers', 'herself'],
        'Laura': ['she', 'her', 'hers', 'herself'],
        'Marta': ['she', 'her', 'hers', 'herself'],
        'Sarah': ['she', 'her', 'hers', 'herself'],
        'Lisa': ['she', 'her', 'hers', 'herself'],
        'Karen': ['she', 'her', 'hers', 'herself'],
        'Nicole': ['she', 'her', 'hers', 'herself'],
        'Hannah': ['she', 'her', 'hers', 'herself']
    }
    tagged_text_custom = tracker.tag_text(text, custom_pronouns)
    print(tagged_text_custom)
    print()
    
    # Get summary
    print("=== Tracking Summary ===")
    summary = tracker.get_tracking_summary(text, custom_pronouns)
    for name, data in summary.items():
        print(f"{name}:")
        print(f"  Name mentioned: {data['name_count']} times")
        print(f"  Pronouns used: {dict(data['pronouns'])}")
        print()