# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm

from config import Config
from models import DeepMLP
from utils import get_data_loaders, plot_learning_curves, plot_misclassified

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def main():
    # Load configuration
    config = Config()
    device = config.device
    print(f"🚀 Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(config.log_dir)
    print(f"📊 TensorBoard logs: {config.log_dir}")
    
    # Data loaders
    print("📦 Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(f"✅ Train: {len(train_loader.dataset)} samples")
    print(f"✅ Validation: {len(val_loader.dataset)} samples")
    print(f"✅ Test: {len(test_loader.dataset)} samples")
    
    # Model
    print("🏗️ Building model...")
    model = DeepMLP(config).to(device)
    print(f"📊 Model parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    print("\n🎯 Starting training...")
    print("="*60)
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log gradients (every 5 epochs)
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                    writer.add_histogram(f'Weights/{name}', param.data, epoch)
        
        # Print progress
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"🔢 LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Checkpointing (save if best validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config.__dict__
            }
            torch.save(checkpoint, f'{config.checkpoint_dir}/best_model.pth')
            print(f"💾 Best model saved! (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            
        # Save latest checkpoint (for resuming)
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': config.__dict__
        }
        torch.save(latest_checkpoint, f'{config.checkpoint_dir}/latest.pth')
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            break
    
    print("\n" + "="*60)
    print("🏁 Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(f'{config.checkpoint_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded best model from epoch {checkpoint['epoch']}")
    
    # Final test evaluation
    print("\n📝 Final evaluation on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("="*50)
    print(f"🎯 FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"📉 FINAL TEST LOSS: {test_loss:.4f}")
    print("="*50)
    
    # Save results to JSON
    results = {
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_epochs_trained': len(train_losses),
        'config': config.__dict__
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot learning curves
    plot_path = f'{config.figure_dir}/learning_curves.png'
    plot_learning_curves(
        train_losses, val_losses, 
        train_accs, val_accs, 
        plot_path
    )
    
    # Plot misclassified examples
    misclassified_path = f'{config.figure_dir}/misclassified.png'
    plot_misclassified(model, test_loader, device, 
                      num_samples=25, 
                      save_path=misclassified_path)
    
    # Close TensorBoard writer
    writer.close()
    print("\n✅ All done! Check the outputs folder for results.")
    print(f"📊 To view TensorBoard: tensorboard --logdir={config.log_dir}")

if __name__ == "__main__":
    main()