## Runs desired models on Fisher Information Metric, to identify 
# Q and K matrices relevant to target behaviors


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-13B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")


# Generate dataset of baseline stims to find masks for
utils.create_controlled_baseline(tokenizer, n_examples=192)

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

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from collections import defaultdict

class FisherDifferentialAnalysis:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
    def compute_fisher_for_task(self, examples, loss_fn, task_name):
        """
        Compute Fisher Information for a specific task
        """
        # Initialize Fisher dictionary
        fisher_dict = {}
        attention_params = {
            name: param for name, param in self.model.named_parameters()
            if ('q_proj' in name or 'k_proj' in name) and param.requires_grad
        }
        
        for name in attention_params:
            fisher_dict[name] = torch.zeros_like(attention_params[name])
        
        print(f"Computing Fisher for {task_name}...")
        
        # Process examples
        for i, example in enumerate(examples):
            if i % 10 == 0:
                print(f"  Processing example {i}/{len(examples)}")
            
            # Prepare batch
            batch = self.prepare_single_example(example)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Compute loss
            loss = loss_fn(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in attention_params.items():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.pow(2).detach()
        
        # Average over examples
        n_examples = len(examples)
        for name in fisher_dict:
            fisher_dict[name] /= n_examples
        
        return fisher_dict
    
    def prepare_single_example(self, example):
        """
        Tokenize and prepare a single example
        """
        tokens = self.tokenizer(
            example['prompt'], 
            return_tensors='pt',
            padding=False,
            truncation=True
        ).to(self.model.device)
        
        answer_pos = tokens['input_ids'].shape[1] - 1
        
        correct_token = self.tokenizer.encode(
            example['correct_answer'], 
            add_special_tokens=False
        )[0]
        
        incorrect_token = self.tokenizer.encode(
            example['incorrect_answer'], 
            add_special_tokens=False
        )[0]
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'answer_position': torch.tensor([answer_pos]),
            'correct_token_id': torch.tensor([correct_token]),
            'incorrect_token_id': torch.tensor([incorrect_token])
        }
    
    def compute_differential_importance(self, fisher_belief, fisher_baseline):
        """
        Compute belief-specific importance by comparing Fisher matrices
        """
        differential_importance = {}
        
        for name in fisher_belief.keys():
            # Normalize by magnitude to make comparable
            belief_norm = fisher_belief[name] / (fisher_belief[name].sum() + 1e-10)
            baseline_norm = fisher_baseline[name] / (fisher_baseline[name].sum() + 1e-10)
            
            # Differential: higher in belief than baseline
            differential = belief_norm - baseline_norm
            
            # Aggregate to scalar per parameter matrix
            differential_importance[name] = differential.sum().item()
        
        return differential_importance
    
    def rank_heads_by_differential(self, differential_importance, top_k=20):
        """
        Rank attention heads by belief-specific importance
        """
        head_scores = defaultdict(float)
        
        for name, importance in differential_importance.items():
            # Parse layer number and matrix type
            if 'layers' in name:
                parts = name.split('.')
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        break
                
                if layer_idx is not None:
                    matrix_type = 'Q' if 'q_proj' in name else 'K'
                    head_key = f"Layer_{layer_idx}_{matrix_type}"
                    head_scores[head_key] = importance
        
        # Sort by differential importance
        ranked = sorted(
            head_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked[:top_k]
    
    def analyze_overlap(self, fisher_belief, fisher_baseline, threshold_percentile=75):
        """
        Analyze which heads are belief-specific vs. general-purpose
        """
        def get_top_heads(fisher_dict, percentile):
            head_totals = defaultdict(float)
            for name, fisher_vals in fisher_dict.items():
                if 'layers' in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            head_totals[layer_idx] += fisher_vals.sum().item()
                            break
            
            threshold = np.percentile(list(head_totals.values()), percentile)
            return {k for k, v in head_totals.items() if v >= threshold}
        
        belief_heads = get_top_heads(fisher_belief, threshold_percentile)
        baseline_heads = get_top_heads(fisher_baseline, threshold_percentile)
        
        belief_specific = belief_heads - baseline_heads
        general_purpose = belief_heads & baseline_heads
        
        return {
            'belief_specific': sorted(belief_specific),
            'general_purpose': sorted(general_purpose),
            'belief_only_count': len(belief_specific),
            'overlap_count': len(general_purpose)
        }

# Loss function (same for both tasks)
def contrastive_loss_fn(outputs, batch):
    batch_size = batch['input_ids'].shape[0]
    answer_positions = batch['answer_position']
    
    logits_at_answer = outputs.logits[
        torch.arange(batch_size), 
        answer_positions, 
        :
    ]
    
    log_probs = F.log_softmax(logits_at_answer, dim=-1)
    
    correct_log_probs = log_probs[
        torch.arange(batch_size),
        batch['correct_token_id']
    ]
    
    incorrect_log_probs = log_probs[
        torch.arange(batch_size),
        batch['incorrect_token_id']
    ]
    
    loss = -(correct_log_probs - incorrect_log_probs).mean()
    
    return loss

# Usage
def run_differential_analysis():
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-2-1124-13B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
    
    # Initialize analyzer
    analyzer = FisherDifferentialAnalysis(model, tokenizer)
    
    # Define your datasets (you'd expand these)
    belief_examples = [
        {
            'prompt': "Sean is reading a book. When he is done, he puts the book in the box and picks up a sweater from the basket. Then, Anna comes into the room. Sean watches Anna move the book to the basket from the box. Sean leaves to get something to eat in the kitchen. Sean comes back into the room and wants to read more of his book. At the start of the story, Sean put the book in the",
            'correct_answer': ' box',
            'incorrect_answer': ' basket'
        },
        # Add 50-200 more diverse examples
    ]
    
    baseline_examples = [
        {
            'prompt': "Sean is reading a book. When he is done, he puts the book in the box and picks up a sweater from the basket. Then, Anna comes into the room. Anna moves the book to the basket from the box. Sean leaves to get something to eat in the kitchen. Sean comes back into the room and wants to read more of his book. The book is now in the",
            'correct_answer': ' basket',
            'incorrect_answer': ' box'
        },
        # Add matching number of baseline examples
    ]
    
    # Compute Fisher matrices
    fisher_belief = analyzer.compute_fisher_for_task(
        belief_examples, 
        contrastive_loss_fn,
        "False Belief"
    )
    
    fisher_baseline = analyzer.compute_fisher_for_task(
        baseline_examples,
        contrastive_loss_fn,
        "Reality Tracking"
    )
    
    # Compute differential importance
    differential = analyzer.compute_differential_importance(
        fisher_belief, 
        fisher_baseline
    )
    
    # Rank heads
    top_belief_specific = analyzer.rank_heads_by_differential(differential, top_k=20)
    
    print("\n=== Top 20 Belief-Specific Q/K Matrices ===")
    for i, (head, score) in enumerate(top_belief_specific, 1):
        print(f"{i}. {head}: {score:.6f}")
    
    # Analyze overlap
    overlap_analysis = analyzer.analyze_overlap(fisher_belief, fisher_baseline)
    
    print(f"\n=== Overlap Analysis ===")
    print(f"Belief-specific heads: {overlap_analysis['belief_only_count']}")
    print(f"General-purpose heads: {overlap_analysis['overlap_count']}")
    print(f"\nBelief-specific layers: {overlap_analysis['belief_specific']}")
    
    return fisher_belief, fisher_baseline, differential, top_belief_specific

def validate_belief_specific_heads(model, belief_specific_layers, test_examples):
    """
    Ablate belief-specific heads and measure performance drop
    """
    # Baseline performance
    baseline_acc = evaluate_model(model, test_examples)
    
    # Ablate belief-specific heads
    with torch.no_grad():
        for layer_idx in belief_specific_layers:
            # Zero out Q and K projections
            model.model.layers[layer_idx].self_attn.q_proj.weight.zero_()
            model.model.layers[layer_idx].self_attn.k_proj.weight.zero_()
    
    # Measure performance drop
    ablated_acc = evaluate_model(model, test_examples)
    
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print(f"Ablated accuracy: {ablated_acc:.3f}")
    print(f"Performance drop: {baseline_acc - ablated_acc:.3f}")

# Run it
# fisher_belief, fisher_baseline, differential, rankings = run_differential_analysis()