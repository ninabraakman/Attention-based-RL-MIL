import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadInstanceSelector(nn.Module):
    def __init__(self, instance_dim, num_heads=4, num_transformations=3, hidden_dim=64):
        super(MultiHeadInstanceSelector, self).__init__()
        self.num_heads = num_heads

        self.transformations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(instance_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_transformations)
        ])

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_heads)
        ])

    def forward(self, bag_tensor, bag_label=None):
        """
        bag_tensor: Tensor of shape (batch_size, num_instances, instance_dim)
        """
        batch_size, num_instances, _ = bag_tensor.size()
        attention_scores_all = []

        for head_idx in range(self.num_heads):
            T = self.transformations[torch.randint(0, len(self.transformations), (1,)).item()]
            transformed = T(bag_tensor)  # (batch_size, num_instances, hidden_dim)
            logits = self.heads[head_idx](transformed).squeeze(-1)  # (batch_size, num_instances)
            alpha = F.softmax(logits / (transformed.size(-1) ** 0.5), dim=1)  # normalized attention
            attention_scores_all.append(alpha.unsqueeze(1))

        attention_concat = torch.cat(attention_scores_all, dim=1)  # (batch_size, num_heads, num_instances)
        instance_scores = attention_concat.mean(dim=1)  # (batch_size, num_instances)

        return instance_scores  # can be interpreted as selection probabilities