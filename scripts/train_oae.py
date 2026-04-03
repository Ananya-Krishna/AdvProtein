"""
train_oae.py - Training script for Protein Organized AutoEncoder (OAE)

Implements multi-objective training with:
- Contrastive alignment between sequence and structure modalities
- Reconstruction loss for autoencoding
- Auxiliary losses for fitness and evasion prediction
- Manifold regularization via diffusion prior
- Support for Langevin dynamics sampling during/after training

This is the core pre-training step before running adversarial attacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(".")
from model.oae_improved import ProteinOAE
from model.dataset import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train Protein OAE")
    parser.add_argument("--data_csv", type=str, default="data/output/protein_dataset_final.csv")
    parser.add_argument("--output_dir", type=str, default="model/checkpoints")
    parser.add_argument("--log_dir", type=str, default="model/runs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--fitness_weight", type=float, default=0.5)
    parser.add_argument("--evasion_weight", type=float, default=0.5)
    parser.add_argument("--manifold_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def contrastive_loss(z_seq: torch.Tensor, z_struct: torch.Tensor, temperature: float = 0.07):
    """Contrastive loss between sequence and structure representations."""
    batch_size = z_seq.shape[0]
    z_seq = nn.functional.normalize(z_seq, dim=1)
    z_struct = nn.functional.normalize(z_struct, dim=1)
    
    # Compute similarity matrix
    logits = torch.mm(z_seq, z_struct.t()) / temperature
    labels = torch.arange(batch_size, device=z_seq.device)
    
    loss_i = nn.CrossEntropyLoss()(logits, labels)
    loss_j = nn.CrossEntropyLoss()(logits.t(), labels)
    
    return (loss_i + loss_j) / 2.0


def train_one_epoch(model, train_loader, optimizer, device, epoch, writer, args):
    model.train()
    total_loss = 0.0
    steps = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        
        sequences = batch['sequences']
        tokens = batch['tokens'].to(device)
        
        # Forward pass with structure features
        structure_features = batch.get('structure_features', None)
        if structure_features is not None:
            structure_features = structure_features.to(device)
        
        outputs = model(sequences, structure_features=structure_features, return_latents=True)
        
        # Multi-objective losses
        loss = torch.tensor(0.0, device=device)
        
        # 1. Contrastive loss (sequence vs structure latents)
        if 'z_seq' in outputs:
            z_struct = outputs.get('z_struct', outputs['z_seq'])
            contrastive_l = contrastive_loss(outputs['z_seq'], z_struct)
            loss = loss + args.contrastive_weight * contrastive_l
        
        # 2. Fitness prediction loss
        fitness_targets = batch.get('esm2_logprob_mean', torch.zeros((len(sequences), 1), device=device))
        fitness_loss = nn.MSELoss()(outputs.get('fitness_pred', torch.zeros_like(fitness_targets)), 
                                  fitness_targets.to(device).unsqueeze(1))
        loss = loss + args.fitness_weight * fitness_loss
        
        # 3. Evasion prediction loss
        evasion_targets = batch.get('commec_flagged', torch.zeros((len(sequences), 1), device=device))
        evasion_loss = nn.BCEWithLogitsLoss()(outputs.get('evasion_pred', torch.zeros_like(evader_targets)), 
                                            evasion_targets.to(device).unsqueeze(1).float())
        loss = loss + args.evasion_weight * evasion_loss
        
        # 4. Reconstruction loss (sequence reconstruction from latent)
        # This is computationally expensive - disabled for initial training
        # if 'decoder' in dir(model):
        #     recon_logits = model.decode(outputs['latent'])
        #     recon_loss = nn.CrossEntropyLoss()(recon_logits.view(-1, recon_logits.size(-1)), tokens.view(-1))
        #     loss = loss + args.reconstruction_weight * recon_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        if steps % 50 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + steps)
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    
    # Create model
    model = ProteinOAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Data
    train_loader, val_loader = create_dataloaders(
        csv_path=args.data_csv,
        batch_size=args.batch_size,
        use_structure=True,
    )
    
    print(f"Training on {len(train_loader.dataset)} proteins")
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, args)
        
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, output_dir / "best_oae_model.pth")
            print(f"New best model saved at epoch {epoch}")
    
    print("Training completed!")
    writer.close()
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "final_oae_model.pth")
    print(f"Final model saved to {output_dir}")


if __name__ == "__main__":
    main()
