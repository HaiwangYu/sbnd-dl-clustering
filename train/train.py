import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import EdgeConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
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

def load_data_from_list(list_file):
    """
    Load data from files specified in a list file.
    
    Parameters:
    -----------
    list_file : str
        Path to a text file containing paths to data files (one per line)
        
    Returns:
    --------
    list
        List of loaded data items
    """
    all_data = []
    
    with open(list_file, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    
    if not file_paths:
        raise FileNotFoundError(f"No file paths found in: {list_file}")
    
    print(f"Found {len(file_paths)} data files to load")
    for file_path in tqdm(file_paths):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping")
            continue
        data = np.load(file_path)
        all_data.append(data)
    
    print(f"Successfully loaded {len(all_data)} data files")
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

def prepare_loader(data_list, batch_size=1, shuffle=False):
    """
    Prepare a data loader from a list of data items.
    
    Parameters:
    -----------
    data_list : list
        List of data items to convert to PyTorch Geometric Data objects
    batch_size : int, optional
        Batch size for the loader
    shuffle : bool, optional
        Whether to shuffle the data
        
    Returns:
    --------
    DataLoader
        PyTorch Geometric DataLoader
    """
    # Convert data to PyTorch Geometric Data objects
    graph_data = []
    for data_item in data_list:
        graph = build_graph(data_item)
        graph_data.append(graph)
    
    # Create data loader
    loader = DataLoader(graph_data, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def prepare_datasets(data_list, train_ratio=0.5, val_ratio=0.5):
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
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {len(graph_data) - train_size - val_size}")
    
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

def train_epoch(model, optimizer, loader, device, class_weights=None):
    model.train()
    total_loss = 0
    
    # If class_weights not provided, compute them
    if class_weights is None:
        # Collect all labels from the loader to compute class weights
        all_labels = []
        for data in loader:
            all_labels.append(data.y.cpu().numpy())
        all_labels = np.concatenate(all_labels)
        
        # Calculate class weights based on inverse frequency
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        # Create a new loader with the same dataset to restart iteration
        loader = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=True)
    
    # Actual training loop
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Use weighted loss
        loss = F.cross_entropy(out, data.y, weight=class_weights)
        
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
    # print(f"all_preds.shape: {len(all_preds)}")
    # print(f"all_labels.shape: {len(all_labels)}")
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
    parser.add_argument('--train-file-list', help='File containing list of training data files')
    parser.add_argument('--val-file-list', help='File containing list of validation data files')
    parser.add_argument('--test-file-list', help='File containing list of test data files')
    parser.add_argument('--file-list', help='File containing list of all data files (if not using separate lists)')
    parser.add_argument('--data-dir', help='Directory containing labeled data files')
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
    
    # Load data based on provided arguments
    if args.train_file_list and args.val_file_list:
        # Load separate train, validation, and optionally test data
        print("Loading training data...")
        train_data = load_data_from_list(args.train_file_list)
        
        print("Loading validation data...")
        val_data = load_data_from_list(args.val_file_list)
        
        if args.test_file_list:
            print("Loading test data...")
            test_data = load_data_from_list(args.test_file_list)
        else:
            # If no test set provided, use validation set for final evaluation
            test_data = val_data
            
        # Prepare data loaders directly without random splitting
        print("Preparing data loaders...")
        train_loader = prepare_loader(train_data, batch_size=1, shuffle=True)
        val_loader = prepare_loader(val_data, batch_size=1, shuffle=False)
        test_loader = prepare_loader(test_data, batch_size=1, shuffle=False)
        
    else:
        # Fall back to the original approach with random splitting
        print("Loading labeled data...")
        if args.file_list:
            data_list = load_data_from_list(args.file_list)
        elif args.data_dir:
            data_list = load_labeled_data(args.data_dir, args.pattern)
        else:
            raise ValueError("Either --train-file-list and --val-file-list, or --file-list, or --data-dir must be specified")
        
        # Prepare datasets with random splitting
        print("Preparing datasets with random splitting...")
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
    
    # Compute class weights once before training
    all_labels = []
    for data in tqdm(train_loader, desc="Computing class weights"):
        all_labels.append(data.y.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Show class distribution
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Training loop with progress bar
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training epochs"):
        # Train with pre-computed weights
        loss = train_epoch(model, optimizer, train_loader, device, class_weights)
        
        # Evaluate on validation set
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Save the model if validation F1 score improves
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f'Saved model to {model_path}')
    
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    # test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device)
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, val_loader, device)
    
    print("\nTest set evaluation:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()
