#!/usr/bin/env python3
"""
Protein Dataset for OAE training.

Handles loading of sequences, structure features (pLDDT), and targets.
Supports variable length sequences with proper padding.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import esm
from tqdm import tqdm


class ProteinOAEDataset(Dataset):
    """Dataset for Protein Organized AutoEncoder training."""
    
    def __init__(self, csv_path: str = "data/output/protein_dataset_final.csv", 
                 max_length: int = 512, use_structure: bool = True):
        self.csv_path = csv_path
        self.max_length = max_length
        self.use_structure = use_structure
        
        # Load data
        self.df = pd.read_csv(csv_path)
        # Filter to only proteins with structure data if requested
        if use_structure:
            self.df = self.df[self.df["plddt_mean"].notna()].copy()
            print(f"Loaded {len(self.df)} proteins with structure data")
        else:
            print(f"Loaded {len(self.df)} proteins (structure ignored)")
        
        # Load ESM alphabet
        self.alphabet = esm.data.Alphabet.from_essentials()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        print(f"✅ ProteinOAEDataset initialized with {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        seq = row["sequence"]
        
        item = {
            "sequence": seq,
            "id": row["id"],
            "length": len(seq),
            "esm2_logprob_mean": float(row.get("esm2_logprob_mean", -12.0)),
            "commec_flagged": int(row.get("commec_flagged", 0)),
        }
        
        # Tokenize sequence
        _, _, tokens = self.batch_converter([(row["id"], seq)])
        item["tokens"] = tokens.squeeze(0)  # remove batch dimension
        
        # Structure features
        if self.use_structure and "plddt_mean" in row and not pd.isna(row["plddt_mean"]):
            plddt = float(row["plddt_mean"]) / 100.0  # normalize to [0,1]
            length = min(len(seq), self.max_length)
            # Simple structure feature: pLDDT as constant per position
            item["structure_features"] = torch.ones((length, 1)) * plddt
            item["plddt_mean"] = plddt
        else:
            item["structure_features"] = torch.zeros((min(len(seq), self.max_length), 1))
        
        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for variable length sequences."""
    sequences = [item["sequence"] for item in batch]
    ids = [item["id"] for item in batch]
    
    # Tokenize as batch
    batch_converter = esm.data.Alphabet.from_essentials().get_batch_converter()
    _, _, tokens = batch_converter([(id, seq) for id, seq in zip(ids, sequences)])
    
    # Structure features padding
    max_len = max(item["structure_features"].shape[0] for item in batch)
    structure_features = []
    for item in batch:
        feat = item["structure_features"]
        if feat.shape[0] < max_len:
            padding = torch.zeros((max_len - feat.shape[0], feat.shape[1]))
            feat = torch.cat([feat, padding], dim=0)
        structure_features.append(feat)
    
    collated = {
        "sequences": sequences,
        "tokens": tokens,
        "structure_features": torch.stack(structure_features),
        "ids": ids,
        "lengths": torch.tensor([item["length"] for item in batch]),
        "esm2_logprob_mean": torch.tensor([item.get("esm2_logprob_mean", -12.0) for item in batch]),
        "commec_flagged": torch.tensor([item.get("commec_flagged", 0) for item in batch]),
    }
    
    return collated


def create_dataloaders(csv_path: str = "data/output/protein_dataset_final.csv", 
                      batch_size: int = 16, max_length: int = 512,
                      use_structure: bool = True, val_split: float = 0.1):
    """Create train and validation dataloaders."""
    dataset = ProteinOAEDataset(csv_path, max_length, use_structure)
    
    # Simple split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader
