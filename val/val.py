import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import json
from tqdm import tqdm

# Add the parent directory to sys.path to import from train and prep
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.train import GNNModel, build_graph
from prep.labeler import get_isnu_labels

def load_model(model_path, device, input_dim=5, hidden_dim=64):
    """
    Load a trained GNN model from a file.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    device : torch.device
        Device to load the model on
    input_dim : int
        Input dimension of the model
    hidden_dim : int
        Hidden dimension of the model
        
    Returns:
    --------
    model : GNNModel
        The loaded model
    """
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_data_from_list(file_list):
    """
    Load data from files specified in a list file.
    
    Parameters:
    -----------
    file_list : str
        Path to a text file containing paths to data files (one per line)
        
    Returns:
    --------
    list
        List of loaded data items
    """
    all_data = []
    
    with open(file_list, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    
    if not file_paths:
        raise FileNotFoundError(f"No file paths found in: {file_list}")
    
    print(f"Found {len(file_paths)} data files to load")
    for file_path in tqdm(file_paths):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping")
            continue
        data = np.load(file_path)
        all_data.append(data)
    
    print(f"Successfully loaded {len(all_data)} data files")
    return all_data

def predict_on_data(model, data_item, device):
    """
    Make predictions on a single data item.
    
    Parameters:
    -----------
    model : GNNModel
        The trained model
    data_item : dict
        The data item to predict on
    device : torch.device
        Device to run inference on
        
    Returns:
    --------
    np.array
        Array of predicted classes
    np.array
        Array of prediction probabilities
    """
    # Convert to PyG data format
    graph = build_graph(data_item)
    graph = graph.to(device)
    
    with torch.no_grad():
        # Get model outputs
        out = model(graph)
        
        # Get class predictions
        pred_classes = out.argmax(dim=1).cpu().numpy()
        
        # Get probabilities with softmax
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
        
    return pred_classes, probs

def calculate_metrics(all_labels, all_preds):
    """
    Calculate metrics for neutrino point classification.
    
    Parameters:
    -----------
    all_labels : np.array
        True labels (0: non-neutrino, 1: neutrino)
    all_preds : np.array
        Predicted labels (0: non-neutrino, 1: neutrino)
        
    Returns:
    --------
    dict
        Dictionary containing the metrics
    """
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'f1': f1_score(all_labels, all_preds, average='binary')
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm
    
    # Calculate true positives, false positives, etc.
    metrics['TP'] = cm[1, 1]  # True positives
    metrics['FP'] = cm[0, 1]  # False positives
    metrics['FN'] = cm[1, 0]  # False negatives
    metrics['TN'] = cm[0, 0]  # True negatives
    
    return metrics

class EventDisplay:
    def __init__(self, data_items, predictions, truth_files=None):
        """
        Interactive event display for visualization.
        
        Parameters:
        -----------
        data_items : list
            List of data items to display
        predictions : list
            List of predictions for each data item
        truth_files : list, optional
            List of truth files for additional truth information
        """
        self.data_items = data_items
        self.predictions = predictions
        self.truth_files = truth_files
        self.current_index = 0
        
        self.fig = plt.figure(figsize=(15, 8))
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the plot with buttons and initial display"""
        self.ax_truth = self.fig.add_subplot(121, projection='3d')
        self.ax_pred = self.fig.add_subplot(122, projection='3d')
        
        # Add buttons for navigation
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.previous_event)
        self.btn_next.on_clicked(self.next_event)
        
        self.update_display()
        
    def update_display(self):
        """Update the display with the current event"""
        self.ax_truth.clear()
        self.ax_pred.clear()
        
        # Get current data
        data = self.data_items[self.current_index]
        preds = self.predictions[self.current_index]
        
        points = data['points']
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Get labels (convert to binary format)
        labels = data['is_nu']
        binary_labels = (labels > 0).astype(int)
        
        # Plot truth
        neutrino_mask = binary_labels == 1
        non_neutrino_mask = binary_labels == 0
        
        self.ax_truth.scatter(x[neutrino_mask], y[neutrino_mask], z[neutrino_mask], 
                             c='blue', marker='o', alpha=0.6, label='Neutrino (Truth)')
        self.ax_truth.scatter(x[non_neutrino_mask], y[non_neutrino_mask], z[non_neutrino_mask], 
                             c='gray', marker='.', alpha=0.2, label='Non-Neutrino (Truth)')
        
        # Plot predictions
        pred_neutrino_mask = preds == 1
        pred_non_neutrino_mask = preds == 0
        
        self.ax_pred.scatter(x[pred_neutrino_mask], y[pred_neutrino_mask], z[pred_neutrino_mask], 
                           c='red', marker='o', alpha=0.6, label='Neutrino (Pred)')
        self.ax_pred.scatter(x[pred_non_neutrino_mask], y[pred_non_neutrino_mask], z[pred_non_neutrino_mask], 
                           c='gray', marker='.', alpha=0.2, label='Non-Neutrino (Pred)')
        
        # Set titles and labels
        self.ax_truth.set_title(f'Truth (Event {self.current_index+1}/{len(self.data_items)})')
        self.ax_pred.set_title('Prediction')
        
        for ax in [self.ax_truth, self.ax_pred]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        
        # Ensure the same view for both plots
        self.ax_pred.view_init(elev=self.ax_truth.elev, azim=self.ax_truth.azim)
        
        # Show file name
        if hasattr(self.data_items[self.current_index], 'file_path'):
            plt.suptitle(f"File: {self.data_items[self.current_index].file_path}")
        
        plt.draw()
        
    def next_event(self, event):
        """Go to the next event"""
        if self.current_index < len(self.data_items) - 1:
            self.current_index += 1
            self.update_display()
            
    def previous_event(self, event):
        """Go to the previous event"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def show(self):
        """Show the interactive display"""
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Validate GNN model for neutrino interaction classification')
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--file-list', required=True, help='File containing list of data files to validate')
    parser.add_argument('--truth-dir', help='Directory containing truth files (optional)')
    parser.add_argument('--output', default='validation_results.json', help='Output file for metrics')
    parser.add_argument('--display', action='store_true', help='Show interactive event display')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size used in the model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, device, hidden_dim=args.hidden_dim)
    
    # Load validation data
    print("Loading validation data...")
    data_items = load_data_from_list(args.file_list)
    
    # Store results
    all_labels = []
    all_predictions = []
    all_item_predictions = []
    
    # Process each data item
    print("Running validation...")
    for i, data_item in enumerate(tqdm(data_items)):
        # Get predictions
        pred_classes, _ = predict_on_data(model, data_item, device)
        
        # Store predictions for this item
        all_item_predictions.append(pred_classes)
        
        # Get true labels and convert to binary format (0: not neutrino, 1: neutrino)
        true_labels = (data_item['is_nu'] > 0).astype(int)
        
        # Collect labels and predictions for metrics
        all_labels.append(true_labels)
        all_predictions.append(pred_classes)
    
    # Concatenate all results
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(all_labels, all_predictions)
    
    # Print metrics
    print("\nValidation Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}, TN: {metrics['TN']}")
    
    # Save metrics to file
    print(f"Saving metrics to {args.output}")
    with open(args.output, 'w') as f:
        # Convert NumPy types to Python standard types
        json_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                json_metrics[k] = v.tolist()
            elif isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                json_metrics[k] = int(v)  # Convert NumPy integer to Python int
            elif isinstance(v, (np.float64, np.float32, np.float16)):
                json_metrics[k] = float(v)  # Convert NumPy float to Python float
            else:
                json_metrics[k] = v
        
        json.dump(json_metrics, f, indent=2)
    
    # Interactive event display
    if args.display:
        print("Launching interactive event display...")
        display = EventDisplay(data_items, all_item_predictions)
        display.show()

if __name__ == "__main__":
    main()
