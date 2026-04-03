# AdvProtein

Multi-objective protein optimization.

## Overview

Optimizes proteins for fitness, evasion, and structure preservation.

## Architecture

Dual-encoder OAE with contrastive alignment and Langevin dynamics.

## Key Files

**Data**: `data/output/protein_dataset_final.csv`

**Model**: `model/oae_improved.py`, `model/dataset.py`

**Scripts**: `scripts/train_oae.py`, `scripts/attack_generate.py`, `scripts/finalize_dataset.py`

**Results**: `figures/`

## Setup

```bash
conda env create -f environment.yml
conda activate adv-protein
```

## Usage

```bash
python scripts/train_oae.py
python scripts/attack_generate.py
```
