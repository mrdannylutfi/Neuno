import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# 1. Modular Network Topology with Overridable Default Activation
class ConfigurableNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn=nn.Sigmoid):
        super(ConfigurableNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = activation_fn()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = activation_fn()
        self.activation_name = activation_fn.__name__

    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        return out

# 2. Parameters & Randomized Dataset Generation
num_samples = 300
num_ngrams = 40          # n = number of ngrams
hidden_dim = num_ngrams // 2
output_dim = 3           # 3 target semantic classes

torch.manual_seed(42)
# Simulating token presence vectors (0 or 1) and random structural meaning labels (0, 1, or 2)
X_random = torch.randint(0, 2, (num_samples, num_ngrams), dtype=torch.float32)
y_random = torch.randint(0, output_dim, (num_samples,), dtype=torch.long)

full_dataset = TensorDataset(X_random, y_random)

# 3. Structural Splits: Training, Testing, and Unused/Later sets
train_size = int(0.6 * num_samples)  # 60%
test_size = int(0.2 * num_samples)   # 20%
later_size = num_samples - train_size - test_size  # 20% reserved for later

train_dataset, test_dataset, later_dataset = random_split(
    full_dataset, [train_size, test_size, later_size]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 4. Instantiate Network with Customizable Sigmoid Default
model = ConfigurableNetwork(
    input_dim=num_ngrams, 
    hidden_dim=hidden_dim, 
    output_dim=output_dim, 
    activation_fn=nn.Sigmoid  # Can be seamlessly swapped out for nn.ReLU, nn.Tanh, etc.
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Optimization & Training Loop
epochs = 10
loss_history = []

model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    loss_history.append(epoch_loss / train_size)

# 6. Save Structural Metainfo to JSON
output_data = {
    "network_topology": {
        "input_layer_neurons": num_ngrams,
        "hidden_layer_neurons": hidden_dim,
        "output_layer_neurons": output_dim,
        "activation_function_used": model.activation_name
    },
    "dataset_splits": {
        "training_samples_count": len(train_dataset),
        "testing_samples_count": len(test_dataset),
        "unused_later_samples_count": len(later_dataset)
    },
    "training_metrics": {
        "loss_progression_by_epoch": [round(l, 4) for l in loss_history],
        "final_epoch_loss": round(loss_history[-1], 4)
    }
}

with open("generated/trained_semantic_network.json", "w") as f:
    json.dump(output_data, f, indent=4)
