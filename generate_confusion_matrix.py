"""
Generate a beautiful confusion matrix visualization for URL classification results.
This script creates a publication-ready confusion matrix with enhanced styling.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(results_file='openai_100_results.json', 
              data_file='./extracted_urls_2000_balanced_shuffled.csv'):
    """Load prediction results and ground truth labels."""
    print("📊 Loading data...")
    
    # Load predictions
    with open(results_file, 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth
    df = pd.read_csv(data_file).head(100)
    
    # Convert labels to binary (0=benign, 1=malicious)
    df['type_numeric'] = df['type'].map({
        'benign': 0, 
        'phishing': 1, 
        'malware': 1, 
        'defacement': 1, 
        'mal': 1
    })
    
    # Extract predictions for matching URLs
    y_true = []
    y_pred_prob = []
    
    for url in df['url']:
        y_true.append(df[df['url'] == url]['type_numeric'].values[0])
        y_pred_prob.append(float(predictions.get(url, 0.5)))
    
    return np.array(y_true), np.array(y_pred_prob), df

def create_confusion_matrix(y_true, y_pred_prob, threshold=0.2, save_path='confusion_matrix.png'):
    """Create a beautiful confusion matrix visualization."""
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Confusion Matrix Plot =====
    # Create custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#ffffff', '#e3f2fd', '#90caf9', '#42a5f5', '#1e88e5']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot confusion matrix
    im = ax1.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=24, fontweight='bold')
    
    # Set labels
    labels = ['Benign', 'Malicious']
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax1.set_yticklabels(labels, fontsize=14, fontweight='bold')
    
    # Add labels to axes
    ax1.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=10)
    ax1.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=10)
    ax1.set_title('Confusion Matrix - URL Classification', 
                  fontsize=18, fontweight='bold', pad=20)
    
    # Add grid
    ax1.grid(False)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Count', fontsize=12, fontweight='bold')
    
    # ===== Metrics Summary =====
    ax2.axis('off')
    
    # Create metrics text
    metrics_text = f"""
    📊 Classification Performance Metrics
    
    ┌─────────────────────────────────────┐
    │  Accuracy:  {accuracy:.3f}          │
    │  Precision: {precision:.3f}          │
    │  Recall:    {recall:.3f}             │
    │  F1-Score:  {f1:.3f}                 │
    └─────────────────────────────────────┘
    
    📈 Confusion Matrix Breakdown
    
    ┌─────────────────────────────────────┐
    │  True Negatives (TN):  {TN:3d}       │
    │  False Positives (FP): {FP:3d}      │
    │  False Negatives (FN): {FN:3d}       │
    │  True Positives (TP):   {TP:3d}      │
    └─────────────────────────────────────┘
    
    ⚙️  Configuration
    
    • Model: GPT-4o-mini
    • Threshold: {threshold}
    • Total Samples: {len(y_true)}
    • Benign: {np.sum(y_true == 0)}
    • Malicious: {np.sum(y_true == 1)}
    """
    
    ax2.text(0.1, 0.5, metrics_text, 
             fontsize=13, 
             family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Confusion matrix saved to: {save_path}")
    
    # Print classification report
    print("\n" + "="*60)
    print("📋 Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, 
                               target_names=['Benign', 'Malicious'],
                               digits=4))
    
    return cm, accuracy, precision, recall, f1

def main():
    """Main function to generate confusion matrix."""
    print("="*60)
    print("🎨 Generating Beautiful Confusion Matrix")
    print("="*60)
    
    # Load data
    y_true, y_pred_prob, df = load_data()
    
    # Create confusion matrix
    cm, accuracy, precision, recall, f1 = create_confusion_matrix(
        y_true, y_pred_prob, threshold=0.2
    )
    
    print("\n" + "="*60)
    print("✨ Confusion Matrix Generation Complete!")
    print("="*60)

if __name__ == "__main__":
    main()

