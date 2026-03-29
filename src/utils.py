import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)


def plot_training_history(
    history: Dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 History saved to {save_path}")
    
    plt.show()
    
    # Print analysis
    print("\n Training Analysis:")
    print(f"   • Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   • Final Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"   • Final Train Acc:  {history['train_acc'][-1]:.2f}%")
    print(f"   • Final Val Acc:    {history['val_acc'][-1]:.2f}%")


def plot_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")
    
    # Precision, Recall, F1
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    if 'accuracy' in report_df.index:
        report_df = report_df.drop(['accuracy'])
    
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[1].bar(x - width, report_df['precision'][:len(class_names)], width, 
                label='Precision', color='skyblue')
    axes[1].bar(x, report_df['recall'][:len(class_names)], width, 
                label='Recall', color='lightgreen')
    axes[1].bar(x + width, report_df['f1-score'][:len(class_names)], width, 
                label='F1-Score', color='salmon')
    
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Precision, Recall, F1-Score per Class', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Metrics saved to {save_path}")
    
    plt.show()
    
    # Print report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    return np.array(all_labels), np.array(all_preds), all_probs


def visualize_activations(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layer_names: List[str],
    num_images: int = 4,
    save_path: Optional[Path] = None
) -> None:
  
    import torch.nn as nn
    
    model.eval()
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        _ = model(images)
    
    for hook in hooks:
        hook.remove()
    
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, num_images, figsize=(20, 5 * n_layers))
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for idx, layer_name in enumerate(layer_names):
        if layer_name not in activations:
            continue
        
        act = activations[layer_name][0].cpu()
        n_channels = act.shape[0]
        channel_indices = np.random.choice(n_channels, min(num_images, n_channels), replace=False)
        
        for j, ch_idx in enumerate(channel_indices):
            if idx < n_layers and j < num_images:
                im = axes[idx, j].imshow(act[ch_idx], cmap='viridis')
                axes[idx, j].set_title(f'{layer_name} - Ch: {ch_idx}')
                axes[idx, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🎨 Activations saved to {save_path}")
    
    plt.show()


def save_model(model: nn.Module, path: Path, metadata: Optional[Dict] = None) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    print(f"💾 Model saved to {path}")


def load_model(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"📥 Model loaded from {path}")
    return model
