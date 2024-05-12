import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim

class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_objects+1, embedding_dim=embedding_dim)
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=embedding_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(128, 768)

    def forward(self, x):
        x = self.embedding(x)
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
model_t5 = AutoModel.from_pretrained("sentence-transformers/sentence-t5-base")

def compute_t5_embeddings(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = model_t5(**inputs)
    return outputs.last_hidden_state[:, 0]  # Using the CLS token's embedding

# Initialize model, loss function, and optimizer
observation_encoder = ObservationEncoder()
loss_fn = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(observation_encoder.parameters(), lr=0.001)

def calculate_mrr(rankings):
    reciprocal_ranks = [1.0 / rank for rank in rankings]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
