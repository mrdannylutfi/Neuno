import json
import torch
import torch.nn as nn


def extract_ngrams(text, n=3):
    """Splits text into character-level or word-level ngrams."""
    words = text.lower().split()
    # If the text is too short, fall back to character ngrams
    if len(words) < n:
        return [text[i : i + n] for i in range(len(text) - n + 1)]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


# 1. Prepare Data and Calculate Dimensions
sample_text = "the quick brown fox jumps over the lazy dog"
ngrams_list = extract_ngrams(sample_text, n=3)
unique_ngrams = list(set(ngrams_list))

num_ngrams = len(unique_ngrams)
hidden_dim = int(num_ngrams / 2)  # n / 2 neurons
output_dim = 2  # Example: Binary classification output

# 2. Define Network Architecture
class NgramNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NgramNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        return out


# Initialize model
model = NgramNetwork(
    input_dim=num_ngrams, hidden_dim=hidden_dim, output_dim=output_dim
)

# Create a dummy input vector (e.g., a bag-of-ngrams representation)
dummy_input = torch.randn(1, num_ngrams)

# Generate predictions
model.eval()
with torch.no_grad():
    predictions = model(dummy_input).numpy().tolist()

# 3. Document Topology and Save to JSON
network_data = {
    "network_topology": {
        "input_layer_neurons": num_ngrams,
        "hidden_layer_neurons": hidden_dim,
        "output_layer_neurons": output_dim,
        "activation_functions": "Sigmoid (all layers)",
    },
    "extracted_ngrams_count": num_ngrams,
    "sample_output_predictions": predictions,
}

# Write data to a JSON file
output_filepath = "network_output.json"
with open(output_filepath, "w") as json_file:
    json.dump(network_data, json_file, indent=4)

print(f"Success! Network data and topology saved to {output_filepath}")
