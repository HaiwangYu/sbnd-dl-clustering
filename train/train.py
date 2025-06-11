import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import EdgeConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob
from tqdm import tqdm

def load_labeled_data(data_dir, pattern):
    """
    Load labeled data from NPZ files matching the pattern.
    """
    all_data = []
    file_pattern = os.path.join(data_dir, pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(files)} data files")
    for f in files:
        data = np.load(f)
        all_data.append(data)
    
    return all_data

def build_graph(data_item):
    """
    Build a PyTorch Geometric graph from a data item.
    """
    # Extract points and their features
    points = data_item['points']
    
    # Get point features
    x = torch.tensor(points, dtype=torch.float)
    
    # Get labels (is_nu)
    y = torch.tensor(data_item['is_nu'], dtype=torch.long)
    
    # For binary classification, ensure labels are 0 or 1
    # Convert -2 to 0 (not neutrino) and >0 to 1 (neutrino)
    y = (y > 0).long()
    
    # Build edges based on k-nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    k = 8  # Number of neighbors
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    knn.fit(points)
    
    # Get k nearest neighbors for each point
    distances, indices = knn.kneighbors(points)
    
    # Build edge_index
    rows = np.repeat(np.arange(len(points)), k)
    cols = indices[:, 1:].flatten()  # Skip the first column (self)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

def prepare_datasets(data_list, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare train, validation, and test datasets.
    """
    # Convert data to PyTorch Geometric Data objects
    graph_data = []
    for data_item in data_list:
        graph = build_graph(data_item)
        graph_data.append(graph)
    
    # Shuffle data
    indices = np.random.permutation(len(graph_data))
    
    # Split into train, val, test
    train_size = int(len(graph_data) * train_ratio)
    val_size = int(len(graph_data) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_data = [graph_data[i] for i in train_indices]
    val_data = [graph_data[i] for i in val_indices]
    test_data = [graph_data[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)
    
    return train_loader, val_loader, test_loader

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(GNNModel, self).__init__()
        
        # Graph convolution layers
        self.conv1 = EdgeConv(nn=nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        ), aggr='mean')
        
        self.conv2 = EdgeConv(nn=nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        ), aggr='mean')
        
        # Output layer for node classification
        self.lin = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First EdgeConv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second EdgeConv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output layer
        x = self.lin(x)
        
        return x

def train_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Loss calculation (node-wise classification)
        loss = F.cross_entropy(out, data.y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_nodes
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            out = model(data)
            
            # Get predictions
            pred = out.argmax(dim=1)
            
            # Collect predictions and labels
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    
    # Combine results from all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return accuracy, precision, recall, f1

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a GNN model for neutrino interaction classification')
    parser.add_argument('--data-dir', required=True, help='Directory containing labeled data files')
    parser.add_argument('--pattern', default='rec-lab-apa1-*.npz', help='File pattern for labeled data')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--output-dir', default='models', help='Directory to save model')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading labeled data...")
    data_list = load_labeled_data(args.data_dir, args.pattern)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_loader, val_loader, test_loader = prepare_datasets(data_list)
    
    # Get input dimension from the first sample
    sample_data = train_loader.dataset[0]
    input_dim = sample_data.x.size(1)
    
    # Initialize model
    print("Initializing model...")
    model = GNNModel(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    best_val_f1 = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        loss = train_epoch(model, optimizer, train_loader, device)
        
        # Evaluate on validation set
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Save the model if validation F1 score improves
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f'Saved model to {model_path}')
    
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device)
    
    print("\nTest set evaluation:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()
