## 🔧 Installation

First, clone the repository and set up the conda environment:

```bash
conda env create -f environment.yaml
conda activate four

## 📦 Reproduce Reconstruction Results on Heritage-Recon

### 🗂️ Dataset Setup

Download the **Heritage-Recon** dataset and place it under the `data/` directory. You can manually download it or use `gdown` via command line:

```bash
mkdir data && cd data
gdown --id 1eZvmk4GQkrRKUNZpagZEIY_z8Lsdw94v

