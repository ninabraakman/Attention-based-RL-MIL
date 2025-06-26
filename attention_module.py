import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadInstanceSelector(nn.Module):
    """
        A multi-head attention module to score instances within a bag. 
        It uses a set of randomly selected transformations for each head.
    """
    def __init__(self, instance_dim, num_heads=4, num_transformations=3, hidden_dim=64):
        super(MultiHeadInstanceSelector, self).__init__()
        self.num_heads = num_heads

        # A list of learnable, non-linear transformations
        self.transformations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(instance_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_transformations)
        ])

        # An attention head for each score. each head is a linear layer.
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_heads)
        ])

    def forward(self, bag_tensor, bag_label=None):
        """
        Calculates instance-level attention scores for a batch of bags.
        Args:
            bag_tensor: Tensor of shape (batch_size, num_instances, instance_dim)
        Returns: 
            torch.Tensor: Aggregated instance scores of shape (batch_size, num_instances)
        """
        batch_size, num_instances, _ = bag_tensor.size()
        attention_scores_all = []

        # Process each bag through all attention heads
        for head_idx in range(self.num_heads):
            T = self.transformations[torch.randint(0, len(self.transformations), (1,)).item()]
            transformed = T(bag_tensor) 
            logits = self.heads[head_idx](transformed).squeeze(-1)  
            alpha = F.softmax(logits / (transformed.size(-1) ** 0.5), dim=1)
            attention_scores_all.append(alpha.unsqueeze(1))

        attention_concat = torch.cat(attention_scores_all, dim=1)
        # Average the scores across all heads to get final instance scores.
        instance_scores = attention_concat.mean(dim=1) 

        return instance_scores  # can be interpreted as selection probabilities