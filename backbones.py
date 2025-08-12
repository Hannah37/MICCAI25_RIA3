       
import torch
import torch.nn as nn
import torch.nn.functional as F

class ObliviousDecisionTree(nn.Module):
    def __init__(self, input_dim, tree_dim, num_trees):
        super(ObliviousDecisionTree, self).__init__()
        self.num_trees = num_trees
        self.tree_dim = tree_dim
        
        self.feature_selection = nn.Linear(input_dim, num_trees, bias=False)
        self.thresholds = nn.Parameter(torch.randn(num_trees))
        self.leaf_values = nn.Linear(num_trees, num_trees * tree_dim, bias=False)

    def forward(self, x):
        feature_values = self.feature_selection(x)  # (batch_size, num_trees)
        decisions = torch.sigmoid(feature_values - self.thresholds)  # Binary decision for each tree
        leaf_values = self.leaf_values(decisions)  # (batch_size, num_trees * tree_dim)
        leaf_values = leaf_values.view(x.size(0), self.num_trees, self.tree_dim)
        return leaf_values

class NODE(nn.Module):
    def __init__(self, input_dim=149, output_dim=2, num_trees=32, tree_dim=3, num_layers=2):
        super(NODE, self).__init__()
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.tree_dim = tree_dim
        
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(input_dim if i == 0 else num_trees * tree_dim, tree_dim, num_trees)
            for i in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(num_trees * tree_dim, output_dim)
    
    def forward(self, x, age):
        x = torch.cat([x, age.unsqueeze(-1)], dim=-1)
        for tree in self.trees:
            x = tree(x).flatten(start_dim=1)
        out = self.fc_out(x)
        return out

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1):
        super(ResNetBlock, self).__init__()
        self.batchnorm1 = nn.BatchNorm1d(input_dim, track_running_stats=False)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        if x.size(0) > 1:  # Skip BatchNorm if batch size is 1
            x = self.batchnorm1(x)
        x = self.block[0](x)  # ReLU
        x = self.block[1](x)  # Linear
        x = self.block[2](x)  # Dropout
        if x.size(0) > 1:
            x = self.batchnorm2(x)
        x = self.block[3](x)  # ReLU
        x = self.block[4](x)  # Linear
        x = self.block[5](x)  # Dropout
        return x + x  # Skip connection

class ResNet(nn.Module):
    def __init__(self, hidden_dim, num_blocks=3, dropout_prob=0.1):
        super(ResNet, self).__init__()
        self.initial_batchnorm = nn.BatchNorm1d(149)
        self.initial_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(149, hidden_dim)
        )

        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, hidden_dim, dropout_prob) for _ in range(num_blocks)]
        )

        self.prediction_batchnorm = nn.BatchNorm1d(hidden_dim)
        self.prediction = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, age):
        x = torch.cat([x, age.unsqueeze(-1)], dim=-1)
        if x.size(0) > 1:  # Skip BatchNorm if batch size is 1
            x = self.initial_batchnorm(x)
        x = self.initial_layer(x)
        x = self.blocks(x)
        if x.size(0) > 1:
            x = self.prediction_batchnorm(x)
        x = self.prediction(x)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(149, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(self, x, age):
        h = torch.cat([x, age.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = self.fc3(h)
        return h

class FeatureTokenizer(nn.Module):
    """
    Feature Tokenization: 
    feature embedding to input into TF
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.linear(x)  # (batch, num_features) → (batch, embed_dim)
        x = self.norm(x)  # LayerNorm 
        return x

class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block:
    Multi-Head Self-Attention + Feed Forward Network
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)  # Residual Connection

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual Connection
        return x

class FT_Transformer(nn.Module):
    def __init__(self, input_dim=149, num_classes=2, embed_dim=64, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Feature Tokenization (Embedding)
        self.feature_tokenizer = FeatureTokenizer(input_dim, embed_dim)

        # add CLS Token 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, embed_dim)

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Fully Connected Classifier
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x, age):
        x = torch.cat([x, age.unsqueeze(-1)], dim=-1)
        batch_size = x.shape[0]

        # Feature Tokenization (Embedding)
        x = self.feature_tokenizer(x)  # (batch_size, input_dim) → (batch_size, embed_dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([cls_tokens, x.unsqueeze(1)], dim=1)  # (batch_size, 1 + num_features, embed_dim)

        # Transformer Blocks 
        for transformer in self.transformer_blocks:
            x = transformer(x)

        cls_output = x[:, 0, :]  # (batch_size, embed_dim)
        out = self.fc_out(cls_output)  # (batch_size, num_classes)
        return out
