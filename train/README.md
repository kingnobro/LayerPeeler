# Training Instructions

The training code is adapted from [**PhotoDoodle**](https://github.com/showlab/PhotoDoodle).

## Step-by-step Training

### 1. Download the Dataset
```bash
mkdir data
huggingface-cli download --local-dir ./data2 kingno/LayerPeeler --repo-type dataset --include "dataset.zip"

cd data
unzip -q dataset.zip
```
The resulting directory structure should be:
```
data/
└── LayerPeeler/
    ├── seq_iconfont/
    ├── seq_svgrepo/
    ├── valid/
    ├── test.jsonl
    └── train.jsonl
```

> [!NOTE]  
> **Dataset Resolution**: The dataset images are provided at 512×768 to align with the original pretrained model (768×768). However, for training efficiency, we train at **512×512** by cropping the images.

### 2. Train the Model
The provided script is configured for an HPC server using Slurm.

To run on a **Slurm cluster**:
```bash
sbatch train_LayerPeeler.slurm
```

To run on a **local machine**:
Simply remove the Slurm header lines (starting with `#SBATCH`) from the script and run it as a standard shell script.

## Dataset Preparation
The training process uses a paired dataset stored in a `.jsonl` file. Each entry includes the source image path, the target (modified) image path, and a caption describing the modification.

**Example format:**

```json
{"source": "path/to/source.jpg", "target": "path/to/modified.jpg", "caption": "Instruction of modifications"}
{"source": "path/to/source2.jpg", "target": "path/to/modified2.jpg", "caption": "Another instruction"}
```

We have uploaded our datasets to [Hugging Face](https://huggingface.co/datasets/kingno/LayerPeeler).
