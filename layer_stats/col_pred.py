import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

# Hyperparameters
EOS_TOKEN = -1  # End-of-sequence token
INPUT_SIZE = 1   # Each input is a single continuous value
HIDDEN_SIZE = 128  # Increased hidden size for more capacity
NUM_LAYERS = 2    # Added multiple layers
OUTPUT_SIZE = 1  # Predict a single continuous value (or EOS)
LR = 0.005  # Modified learning rate
EPOCHS = 200  # More epochs
MAX_SEQ_LENGTH = 20  # Max length for inference
DROPOUT = 0.2  # Added dropout for regularization

# Load data from CSV
def load_data(filepath):
    df = pd.read_csv(filepath, header=None).sample(100)
    sequences = df.values.tolist()
    lengths = [len([x for x in seq if x != 0]) for seq in sequences]  # Valid lengths
    return sequences, lengths

# Preprocess data: create targets with EOS tokens
def preprocess_data(sequences, lengths):
    targets = []
    for seq, seq_len in zip(sequences, lengths):
        # Create targets: original[1:seq_len] + EOS, then pad with 0s
        target = seq[1:seq_len] + [EOS_TOKEN] + [0] * (len(seq) - seq_len)
        targets.append(target)
    return targets

# Find max value to help with normalization
def find_max_abs_value(sequences):
    max_val = 0
    for seq in sequences:
        seq_max = max(max(abs(x) for x in seq if x != 0 and x != EOS_TOKEN), 0)
        max_val = max(max_val, seq_max)
    return max_val

# Normalize sequences to help with training
def normalize_sequences(sequences, max_val):
    normalized = []
    for seq in sequences:
        normalized.append([x / max_val if x != 0 and x != EOS_TOKEN else x for x in seq])
    return normalized, max_val

# Denormalize predictions
def denormalize_predictions(predictions, max_val):
    return [x * max_val for x in predictions]

# Load data
sequences, lengths = load_data("data/used_data/strat_zeroed.csv")

# Find max value and normalize
max_val = find_max_abs_value(sequences)
normalized_sequences, _ = normalize_sequences(sequences, max_val)
targets = preprocess_data(normalized_sequences, lengths)

# Find the maximum sequence length to ensure consistent padding
max_len = max(len(seq) for seq in normalized_sequences)

# Make sure all sequences have the same length through padding
padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in normalized_sequences]
padded_targets = [tgt + [0] * (max_len - len(tgt)) for tgt in targets]

# Convert to PyTorch tensors
inputs_padded = torch.tensor(padded_sequences, dtype=torch.float32)
targets_padded = torch.tensor(padded_targets, dtype=torch.float32)

# Add feature dimension (batch_size, seq_len, 1)
inputs_padded = inputs_padded.unsqueeze(-1)
targets_padded = targets_padded.unsqueeze(-1)

# Mask to ignore padded zeros in loss
mask = (targets_padded != 0).float()

# Define the enhanced LSTM model
class EnhancedLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))
        
        # Apply dropout and non-linearity
        output = self.dropout(output)
        output = self.relu(self.fc1(output))
        return self.fc2(output)

# Custom loss function that gives more weight to larger values
def weighted_mse_loss(pred, target, mask, alpha=2.0):
    # Calculate standard MSE
    squared_diff = ((pred - target) ** 2) * mask
    
    # Add weight based on the magnitude of the target
    weight = torch.ones_like(target)
    weight = weight + alpha * torch.abs(target) * (target != 0).float()
    
    # Apply the weights to the squared differences
    weighted_squared_diff = squared_diff * weight
    
    # Return the mean
    return torch.sum(weighted_squared_diff) / torch.sum(mask * weight)

# Initialize model, loss, optimizer
model = EnhancedLSTMPredictor(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # Added weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)  # LR scheduler

# Autoregressive inference function with denormalization
def predict_sequence(model, start_value, max_val, max_length=MAX_SEQ_LENGTH):
    model.eval()
    # Normalize input
    normalized_start = start_value / max_val
    current_seq = torch.tensor([normalized_start] + [0] * (max_length - 1), dtype=torch.float32)
    generated = [start_value]  # Store the original value
    
    with torch.no_grad():
        for i in range(1, max_length):
            # Get valid length (non-padded part)
            valid_length = i  # Since we start from index 0
            
            # Prepare input (pad and add dimensions)
            input_seq = current_seq[:valid_length].unsqueeze(0).unsqueeze(-1)
            
            # Predict next value
            output = model(input_seq, [valid_length])
            next_val_normalized = output[0, -1, 0].item()
            
            if next_val_normalized == EOS_TOKEN:
                break
                
            # Denormalize for the return value
            next_val = next_val_normalized * max_val
            generated.append(next_val)
            
            # Update sequence with normalized value
            if i < max_length:
                current_seq[i] = next_val_normalized
                
    return generated

# Training loop with validation and early stopping
true_sequences = []
pred_sequences = []
seq_lengths = []
losses = []
val_losses = []

# Split data into train and validation
train_size = int(0.8 * len(padded_sequences))
train_inputs = inputs_padded[:train_size]
train_targets = targets_padded[:train_size]
train_mask = mask[:train_size]
train_lengths = lengths[:train_size]

val_inputs = inputs_padded[train_size:]
val_targets = targets_padded[train_size:]
val_mask = mask[train_size:]
val_lengths = lengths[train_size:]

# Early stopping parameters
best_val_loss = float('inf')
patience = 20
patience_counter = 0
best_model = None

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (train)
    outputs = model(train_inputs, train_lengths)
    
    # Compute loss (with custom weighting)
    loss = weighted_mse_loss(outputs, train_targets, train_mask)
    
    # Backward pass and optimization
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_inputs, val_lengths)
        val_loss = weighted_mse_loss(val_outputs, val_targets, val_mask)
        scheduler.step(val_loss)
    
    # Store losses
    losses.append(loss.item())
    val_losses.append(val_loss.item())
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model)
            break
    
    # Evaluate and track progress
    if (epoch + 1) % 10 == 0 or epoch < 10:
        model.eval()
        true_epoch, pred_epoch, lengths_epoch = [], [], []
        
        # Using the first sequence for demonstration
        seq = padded_sequences[0]
        seq_len = lengths[0]
        
        start_value = seq[0] * max_val  # Denormalize the start value
        true_seq = [x * max_val for x in seq[:seq_len] if x != 0]  # Denormalize and remove padded zeros
        pred_seq = predict_sequence(model, start_value, max_val)
        
        true_epoch.append(true_seq)
        pred_epoch.append(pred_seq)
        lengths_epoch.append(seq_len)
        
        true_sequences.append(true_epoch)
        pred_sequences.append(pred_epoch)
        seq_lengths.append(lengths_epoch)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}\n")
        print(f"True: {true_seq[:10]}\n")
        print(f"Pred: {pred_seq[:10]}\n")


# Return results
results = {
    "true_sequences": true_sequences,
    "pred_sequences": pred_sequences,
    "seq_lengths": seq_lengths,
    "losses": losses,
    "val_losses": val_losses,
    "model": model
}

# Example of accessing results
print("Training complete. Final model performance:")
print(f"True: {results['true_sequences'][-1][0]}")
print(f"Pred: {results['pred_sequences'][-1][0]}")
print(f"Original sequence length: {results['seq_lengths'][-1][0]}")