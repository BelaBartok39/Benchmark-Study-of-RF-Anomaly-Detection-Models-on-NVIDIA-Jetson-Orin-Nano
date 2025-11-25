import os
import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from data_loader import get_dataloaders
from train import get_model

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader = get_dataloaders(args.clean, args.jammed,
                                     window_size=args.window_size,
                                     batch_size=args.batch_size)
    model = get_model(args.model, args.window_size).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()

    y_true = []
    y_scores = []
    with torch.no_grad():
        for x, labels in val_loader:
            x = x.to(device)
            # Only flatten for feedforward models (ae, aae, ff), not for CNN-based models
            if args.model in ['ae', 'aae', 'ff']:
                inp = x.view(x.size(0), -1)
            else:
                inp = x
            
            # Handle different model return patterns
            if args.model == 'ff':
                # FeedForward model returns only anomaly score
                out = model(inp)
                scores = out.squeeze().cpu().numpy()
            else:
                # Autoencoder models return (reconstruction, latent)
                out, _ = model(inp)
                if 'ae' in args.model:
                    # reconstruction error as anomaly score
                    if args.model in ['ae', 'aae']:
                        # For feedforward autoencoders, both inp and out are flattened
                        rec_err = ((out - inp)**2).mean(dim=1).cpu().numpy()
                    else:
                        # For CNN autoencoders, inp and out have shape (batch, channels, length)
                        rec_err = ((out - inp)**2).mean(dim=(1,2)).cpu().numpy()
                    scores = rec_err
                else:
                    scores = out.squeeze().cpu().numpy()
            y_scores.extend(scores)
            y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    auc = roc_auc_score(y_true, y_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_scores>np.percentile(y_scores,90), average='binary')
    print(f'AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['ae','cnn_ae','lstm_ae','resnet_ae','aae','ff'],
                        help='Model to evaluate')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--clean', type=str, default='clean_5g_dataset.h5')
    parser.add_argument('--jammed', type=str, default='jammed_5g_dataset.h5')
    parser.add_argument('--window-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
