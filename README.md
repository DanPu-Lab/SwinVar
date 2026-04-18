# SwinVar

SwinVar is a deep learning pipeline for variant calling from sequencing alignment data.

## Overview

SwinVar is designed for variant calling on aligned sequencing reads. The project turns candidate regions from BAM/BED/VCF inputs into HDF5 feature tensors, trains a for variant and genotype prediction, and exports callable results for downstream analysis.

## Highlights

- End-to-end pipeline from preprocessing to variant calling
- Swin Transformer based architecture for local pileup representation learning
- Joint prediction of variant categories and genotype states
- HDF5 intermediate format for scalable training and inference
- Built-in postprocessing utilities for metrics and VCF generation

## Pipeline

```text
BAM + BED + VCF + Reference
           |
           v
      Data Preprocessing
           |
           v
     Model Training
           |
           v
         Inference
           |
           v
      Metrics + VCF Output
```

## Project Structure

```text
SwinVar/
+-- main.py
+-- swinvar/
|   +-- preprocess/
|   +-- models/
|   +-- training/
|   +-- inference/
|   +-- postprocess/
|   +-- evaluation/
|   `-- core/
+-- scripts/
```

### Module Summary

- `preprocess/`: pileup generation, labeling, BED splitting, and sample balancing
- `models/`: SwinVar architecture, loss functions, and fine-tuning components
- `training/`: training configuration, dataloading, optimization, and trainer loop
- `inference/`: model loading, batched calling, and inference configuration
- `postprocess/`: metrics calculation and VCF export
- `evaluation/`: evaluation helpers for model analysis
- `core/`: threshold optimization utilities

## Requirements

- Python
- PyTorch
- NumPy
- Pandas
- pysam
- samtools

## Quick Start

Run the pipeline through `main.py`:

```bash
python main.py \
  --bam sample.bam \
  --bed target.bed \
  --vcf truth.vcf.gz \
  --output_dir output \
  --reference reference.fa \
  --train \
  --call
```

## Main Workflow

1. `--train`: train the SwinVar model on generated datasets.
2. `--call`: run inference and export calling results.

These stages can be executed together or independently depending on the available artifacts.

## Key Inputs

- `--bam`: aligned sequencing reads
- `--bed`: target regions or confident regions
- `--vcf`: truth or labeled variants
- `--reference`: reference genome FASTA
- `--output_dir`: workspace for generated features, logs, checkpoints, and outputs

## Core Outputs

- Pileup-derived HDF5 datasets
- Balanced training data
- Model checkpoints and logs
- Calling metrics
- VCF result files

## Notes

- The default entry point is [`main.py`](./main.py).
- Core package code lives in [`swinvar/`](./swinvar).
- The project is organized as a research-oriented codebase and assumes indexed genomic inputs and a working `samtools` installation.

## License

This project is released under the [MIT License](./LICENSE).
