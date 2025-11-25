import os
import argparse
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from data_loader import get_dataloaders, get_input_size
from models import ae, cnn_ae, lstm_ae, resnet_ae, aae, ff_models

def get_model(model_name, window_size, use_psd=False):
    input_channels = 3 if use_psd else 2
    
    if model_name == 'ae':
        input_size = get_input_size(window_size, use_psd)
        return ae.Autoencoder(input_size)
    elif model_name == 'cnn_ae':
        return cnn_ae.CNNAutoencoder(window_size, input_channels)
    elif model_name == 'lstm_ae':
        return lstm_ae.LSTMAutoencoder(input_dim=input_channels)
    elif model_name == 'resnet_ae':
        return resnet_ae.ResNetAutoencoder(window_size, input_channels)
    elif model_name == 'aae':
        input_size = get_input_size(window_size, use_psd)
        return aae.AdversarialAutoencoder(input_size)
    elif model_name == 'ff':
        input_size = get_input_size(window_size, use_psd)
        return ff_models.FeedForwardNet(input_size)
    else:
        raise ValueError(f'Unknown model {model_name}')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Reduce workers and batch size for faster startup
    effective_batch_size = min(args.batch_size, 64)  # Cap batch size
    effective_workers = min(4, args.batch_size // 8) if args.batch_size > 8 else 0
    
    train_loader, val_loader = get_dataloaders(
        args.clean, args.jammed,
        window_size=args.window_size,
        batch_size=effective_batch_size,
        num_workers=effective_workers,
        use_psd=args.use_psd
    )
    
    model = get_model(args.model, args.window_size, args.use_psd).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Loss function based on model type
    if 'ae' in args.model:
        criterion = nn.MSELoss()  # Reconstruction loss for autoencoders
    else:
        criterion = nn.BCEWithLogitsLoss()  # Classification loss for discriminative models

    best_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Starting training with {len(train_loader)} batches per epoch...")
    
    for epoch in range(1, args.epochs+1):
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            
            # Prepare input based on model type
            if 'ae' in args.model and args.model != 'aae':
                # For autoencoders, input and target are the same (reconstruction)
                if args.model in ['cnn_ae', 'lstm_ae', 'resnet_ae']:
                    model_input = x  # Keep as (batch, channels, sequence)
                    target = x
                else:
                    # For standard AE, flatten the input
                    model_input = x.view(x.size(0), -1)  # (batch, features)
                    target = model_input
                
                output, latent = model(model_input)
                loss = criterion(output, target)
                
            elif args.model == 'ff':
                # For feed-forward classifier
                model_input = x.view(x.size(0), -1)  # Flatten
                output = model(model_input)
                loss = criterion(output.squeeze(), labels)
                
            else:  # AAE and other adversarial models
                model_input = x.view(x.size(0), -1)  # Flatten
                output, latent = model(model_input)
                # For simplicity, use reconstruction loss for AAE training
                loss = criterion(output, model_input)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_loss /= batch_count
        
        # Validation phase (run less frequently for speed)
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val_loss = 0
            val_count = 0
            with torch.no_grad():
                for x, labels in val_loader:
                    x = x.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).float()
                    
                    # Same input preparation as training
                    if 'ae' in args.model and args.model != 'aae':
                        if args.model in ['cnn_ae', 'lstm_ae', 'resnet_ae']:
                            model_input = x
                            target = x
                        else:
                            model_input = x.view(x.size(0), -1)
                            target = model_input
                        
                        output, latent = model(model_input)
                        loss = criterion(output, target)
                        
                    elif args.model == 'ff':
                        model_input = x.view(x.size(0), -1)
                        output = model(model_input)
                        loss = criterion(output.squeeze(), labels)
                        
                    else:  # AAE
                        model_input = x.view(x.size(0), -1)
                        output, latent = model(model_input)
                        loss = criterion(output, model_input)
                    
                    val_loss += loss.item()
                    val_count += 1
            
            val_loss /= val_count
            print(f'Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                path = os.path.join(args.out_dir, f'{args.model}_best.pth')
                torch.save(model.state_dict(), path)
                print(f'  Saved best model to {path}')
        else:
            print(f'Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['ae','cnn_ae','lstm_ae','resnet_ae','aae','ff'],
                        help='Model to train')
    parser.add_argument('--clean', type=str, default='clean_5g_dataset.h5',
                        help='Path to clean dataset')
    parser.add_argument('--jammed', type=str, default='jammed_5g_dataset.h5',
                        help='Path to jammed dataset')
    parser.add_argument('--window-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use-psd', action='store_true', help='Include PSD features')
    parser.add_argument('--out-dir', type=str, default='weights')
    args = parser.parse_args()
    train(args)
