## Runs desired models on Fisher Information Metric, to identify 
# Q and K matrices relevant to target behaviors


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-13B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")

def prepare_belief_tracking_batch(examples, tokenizer):
    """
    Prepare your belief-tracking examples
    """
    batch = {
        'input_ids': [],
        'attention_mask': [],
        'answer_position': [],
        'correct_token_id': [],
        'incorrect_token_id': []
    }
    
    for example in examples:
        # Tokenize prompt
        tokens = tokenizer(example['prompt'], return_tensors='pt')
        
        # Find where answer should go (last token position)
        answer_pos = tokens['input_ids'].shape[1] - 1
        
        # Get token IDs for answers
        correct_token = tokenizer.encode(example['correct_answer'], 
                                        add_special_tokens=False)[0]
        incorrect_token = tokenizer.encode(example['incorrect_answer'], 
                                          add_special_tokens=False)[0]
        
        batch['input_ids'].append(tokens['input_ids'])
        batch['attention_mask'].append(tokens['attention_mask'])
        batch['answer_position'].append(answer_pos)
        batch['correct_token_id'].append(correct_token)
        batch['incorrect_token_id'].append(incorrect_token)
    
    # Stack into tensors
    batch['input_ids'] = torch.cat(batch['input_ids'], dim=0)
    batch['attention_mask'] = torch.cat(batch['attention_mask'], dim=0)
    batch['answer_position'] = torch.tensor(batch['answer_position'])
    batch['correct_token_id'] = torch.tensor(batch['correct_token_id'])
    batch['incorrect_token_id'] = torch.tensor(batch['incorrect_token_id'])
    
    return batch

def behavior_loss_fn(outputs, batch):
    """
    Contrastive belief-tracking loss
    """
    # Get logits at answer positions
    batch_size = batch['input_ids'].shape[0]
    answer_positions = batch['answer_position']
    
    # Extract logits for each example at its answer position
    logits_at_answer = outputs.logits[
        torch.arange(batch_size), 
        answer_positions, 
        :
    ]
    
    log_probs = F.log_softmax(logits_at_answer, dim=-1)
    
    # Extract probabilities for correct and incorrect answers
    correct_log_probs = log_probs[
        torch.arange(batch_size),
        batch['correct_token_id']
    ]
    
    incorrect_log_probs = log_probs[
        torch.arange(batch_size),
        batch['incorrect_token_id']
    ]
    
    # Loss: want correct >> incorrect
    loss = -(correct_log_probs - incorrect_log_probs).mean()
    
    return loss

# Example usage
examples = [
    {
        'prompt': "Sean is reading a book. When he is done, he puts the book in the box and picks up a sweater from the basket. Then, Anna comes into the room. Sean watches Anna move the book to the basket from the box. Sean leaves to get something to eat in the kitchen. Sean comes back into the room and wants to read more of his book. At the start of the story, Sean put the book in the",
        'correct_answer': ' box',  # Note the space!
        'incorrect_answer': ' basket'
    },
    # Add more examples with variations...
]

batch = prepare_belief_tracking_batch(examples, tokenizer)

# Now compute Fisher
fisher_dict = compute_fisher_for_attention_params(
    model, 
    [batch],  # Wrap in list to simulate dataloader
    behavior_loss_fn
)