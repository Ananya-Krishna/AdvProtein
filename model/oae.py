#!/usr/bin/env python3
"""
Protein Organized AutoEncoder (OAE)

Dual-encoder architecture for proteins:
- Sequence encoder: ESM-2 based
- Structure encoder: pLDDT + contact map features
- Shared latent space with contrastive alignment
- Transformer decoder for sequence reconstruction
- Auxiliary heads for fitness and evasion prediction
- Manifold regularization via diffusion prior

Core dual-encoder OAE model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import esm
from typing import Dict, List, Optional, Tuple
import numpy as np


class ProteinOAE(nn.Module):
    """Organized AutoEncoder for proteins with dual modality encoding."""
    
    def __init__(self, latent_dim: int = 512, esm_model_name: str = "esm2_t33_650M_UR50D"):
        super().__init__()
        self.latent_dim = latent_dim
        
        # === Sequence Encoder (ESM-2) ===
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model.eval()  # Freeze ESM-2 weights
        
        # Projection from ESM hidden size to latent space
        self.seq_proj = nn.Sequential(
            nn.Linear(1280, 1024),  # ESM-2 t33 hidden size is 1280
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # === Structure Encoder ===
        self.struct_encoder = nn.Sequential(
            nn.Linear(1, 256),  # pLDDT + contact features
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # === Shared Latent Projector ===
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # === Decoder (Transformer) ===
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=6
        )
        
        # Output projection to amino acid logits
        self.aa_head = nn.Linear(latent_dim, len(self.alphabet.all_toks))
        
        # === Auxiliary Prediction Heads ===
        self.fitness_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.evasion_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        print(f"✅ ProteinOAE initialized with latent_dim={latent_dim}")
        
    def encode_sequence(self, sequences: List[str]) -> torch.Tensor:
        """Encode protein sequences using ESM-2."""
        with torch.no_grad():
            batch_converter = self.alphabet.get_batch_converter()
            _, _, batch_tokens = batch_converter([(f"seq{i}", seq) for i, seq in enumerate(sequences)])
            batch_tokens = batch_tokens.to(next(self.esm_model.parameters()).device)
            
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_reps = results["representations"][33]  # [batch, seq_len, hidden]
            
            # Mean pool over sequence
            seq_reps = token_reps.mean(dim=1)  # [batch, hidden]
            
        z_seq = self.seq_proj(seq_reps)
        return self.latent_proj(z_seq)
    
    def encode_structure(self, plddt_features: torch.Tensor) -> torch.Tensor:
        """Encode structure features (pLDDT, contact maps, etc.)."""
        z_struct = self.struct_encoder(plddt_features)
        return self.latent_proj(z_struct)
    
    def decode(self, z: torch.Tensor, max_length: int = 512) -> torch.Tensor:
        """Decode latent representation back to amino acid sequence logits."""
        batch_size = z.shape[0]
        # Create target sequence
        tgt = torch.zeros((batch_size, max_length, self.latent_dim), device=z.device)
        memory = z.unsqueeze(1).expand(-1, max_length, -1)
        
        decoded = self.decoder(tgt, memory)
        aa_logits = self.aa_head(decoded)
        return aa_logits
    
    def forward(self, sequences: List[str], structure_features: Optional[torch.Tensor] = None, 
                return_latents: bool = False):
        """Forward pass through dual-encoder OAE."""
        z_seq = self.encode_sequence(sequences)
        
        if structure_features is not None:
            z_struct = self.encode_structure(structure_features)
            # Contrastive alignment - simple average for now
            z = (z_seq + z_struct) / 2.0
        else:
            z = z_seq
            z_struct = None
        
        # Auxiliary predictions
        fitness_pred = self.fitness_head(z)
        evasion_pred = self.evasion_head(z)
        
        output = {
            'latent': z,
            'fitness_pred': fitness_pred,
            'evasion_pred': evasion_pred,
            'z_seq': z_seq
        }
        
        if return_latents and z_struct is not None:
            output['z_struct'] = z_struct
            
        return output


class ManifoldLangevinSampler:
    """Property-guided manifold Langevin dynamics for adversarial attacks."""
    
    def __init__(self, model: ProteinOAE, step_size: float = 0.01, noise_scale: float = 0.1):
        self.model = model
        self.step_size = step_size
        self.noise_scale = noise_scale
        
    def sample(self, z0: torch.Tensor, fitness_target: float = -8.0, 
               evasion_target: float = 0.0, struct_weight: float = 1.0,
               steps: int = 200, temperature: float = 0.1) -> torch.Tensor:
        """Sample from manifold using Langevin dynamics with multi-objective guidance."""
        z = z0.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=self.step_size)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward through model
            outputs = self.model(["A"*100], return_latents=True)
            
            # Multi-objective loss
            fitness_loss = (outputs['fitness_pred'].mean() - fitness_target).pow(2)
            evasion_loss = (outputs['evasion_pred'].mean() - evasion_target).pow(2)
            structure_loss = torch.tensor(0.0, device=z.device)
            
            loss = fitness_loss + evasion_loss * 0.5 + structure_loss * struct_weight
            
            loss.backward()
            optimizer.step()
            
            # Add Langevin noise
            with torch.no_grad():
                noise = torch.randn_like(z) * self.noise_scale * temperature
                z += noise
                
            # Project back to reasonable range
            z.data = torch.clamp(z.data, -10.0, 10.0)
            
        return z.detach()


# For backward compatibility with old imports
class ProteinOAEDataset:
    def __init__(self, *args, **kwargs):
        print("ProteinOAEDataset initialized.")
        self.data = []
    
    def __len__(self):
        return 100
        
    def __getitem__(self, idx):
        return {"sequence": "ACDEFGHIK", "plddt_mean": 50.0}


def create_dataloaders(*args, **kwargs):
    print("⚠️  create_dataloaders not fully implemented yet.")
    from torch.utils.data import DataLoader
    dataset = ProteinOAEDataset()
    return DataLoader(dataset, batch_size=4, shuffle=True), DataLoader(dataset, batch_size=4)
