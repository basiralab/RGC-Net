# RGC-Net — Reservoir-Based Graph Convolutional Networks

[![Paper](https://img.shields.io/badge/IEEE%20TPAMI-2026-00629B.svg)](https://doi.org/10.1109/TPAMI.2026.3670423)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTPAMI.2026.3670423-blue.svg)](https://doi.org/10.1109/TPAMI.2026.3670423)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)

Official implementation of **RGC-Net**, a graph learning framework that integrates
**reservoir computing** with **structured graph convolution**. A single RGC-Net layer
propagates information across *k*-hop neighborhoods using **fixed, randomly initialized
reservoir weights** and a **leaky integrator**, retaining node features and mitigating the
over-smoothing that limits deep Graph Convolutional Networks (GCNs) — without the training
cost of stacking many trainable layers.

> **Reservoir-Based Graph Convolutional Networks**
>
> Mayssa Soussia, Gita Ayu Salsabila, Mohamed Ali Mahjoub, Islem Rekik
>
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 48, no. 7,
> pp. 7875–7889, July 2026. DOI: [10.1109/TPAMI.2026.3670423](https://doi.org/10.1109/TPAMI.2026.3670423)
>
> BASIRA Lab, Imperial-X and Department of Computing, Imperial College London, U.K.
> · LATIS Lab, National Engineering School of Sousse, University of Sousse, Tunisia


---

## Overview

Message passing is the core operation of Graph Neural Networks (GNNs): each node's embedding is
updated by aggregating information from its neighbors. GCNs implement this with trainable
convolutions, but capturing **long-range dependencies** forces the network deeper, which raises
computational cost and causes **over-smoothing** — node embeddings collapse toward one another
and become indistinguishable.

Reservoir computing offers a different route: propagate information through a **fixed random
recurrent operator** and train only a lightweight readout. Prior reservoir-based GNNs, however,
lack a *structured convolutional* mechanism and therefore aggregate multi-hop neighborhood
information poorly.

**RGC-Net** closes this gap. It performs graph convolution *inside* a reservoir — iterating a
fixed random operator over the graph adjacency to reach *k*-hop neighborhoods in a single layer —
while a leaky integrator preserves earlier embeddings to counter over-smoothing. Only a small
readout is trained, so the model converges quickly and remains stable as effective depth grows.
The same reservoir layer powers two tasks: **graph classification** and **temporal graph
generation** (brain connectivity evolution).

## Method

RGC-Net has three components: an **input layer**, a **graph convolutional reservoir layer**, and
a **linear output (readout) layer**.

- **Input layer** — projects node features `X` into the reservoir space: `H⁽⁰⁾ = Wᵢₙ · X`.
  `Wᵢₙ` is random and **frozen**.
- **Graph convolutional reservoir** — a single layer reaches *k*-hop neighbors by iterating the
  reservoir update *k* times over the adjacency `A`:

  ```
  H⁽ᵏ⁺¹⁾ = (1 − α) · H⁽ᵏ⁾  +  α · σ( A · W · H⁽ᵏ⁾ )
  ```

  `W` is a **fixed random** reservoir matrix rescaled to a **spectral radius < 1** (the Echo
  State Property, ensuring stable/fading memory); `α` is the **leaky rate** and the `(1 − α)·H⁽ᵏ⁾`
  term (the leaky integrator) preserves earlier embeddings to control over-smoothing.
- **Readout layer** — the only trained part: batch-norm + linear transform, then task-specific
  pooling/output (`H⁽ˡ⁾ = W_out · H`).

Because `Wᵢₙ` and `W` are frozen, training touches only the readout, giving fast convergence and
few tunable parameters. **TRGC-Net** is an ablation in which the reservoir `W` *is* trained and
re-projected to its target spectral radius at each step.

Two task-specific pipelines are built on this layer:

- **Classification** — RGC-Net layer → ReLU → batch norm → global mean pooling → linear + softmax,
  trained with negative log-likelihood.
- **Generation (RGC-Net-Transformer)** — an encoder–decoder that predicts the next time point of
  a temporal graph. The encoder replaces the input layer with a **GAT** layer + batch norm feeding
  two frozen RGC-Net reservoirs to produce a latent `Z`; a feed-forward decoder followed by
  RGC-Net layers reconstructs the next adjacency matrix.

## Architecture

![RGC-Net single-layer architecture](images/rgcnet-architecture.png)

*A single RGC-Net layer: frozen input weights project node features into a reservoir whose fixed
random operator `W` (rescaled to spectral radius < 1) is iterated `k` times for `k`-hop message
passing; only the linear readout is trained.*

<table>
<tr>
<td width="50%"><img src="images/rgcnet-classification.png" alt="RGC-Net classification pipeline"></td>
<td width="50%"><img src="images/rgcnet-generation.png" alt="RGC-Net-Transformer generation pipeline"></td>
</tr>
<tr>
<td align="center"><em>Graph classification pipeline</em></td>
<td align="center"><em>RGC-Net-Transformer for temporal graph generation</em></td>
</tr>
</table>

## Repository structure

| Path | Description |
| --- | --- |
| `notebooks/experiments-classification_num_layers.ipynb` | Classification vs. number of layers (1–5); defines the canonical RGC-Net / TRGC-Net classes. |
| `notebooks/experiments-classification_reservoir_params.ipynb` | Reservoir sensitivity study over `k` (iterations) and `α` (leaky rate). |
| `notebooks/experiments-generation_transformers_models.ipynb` | Proposed generation models (RGC-Net-Transformer, GCN-Transformer, TRGC-Net-Transformer). |
| `notebooks/experiments-generation_benchmark_models.ipynb` | Generation baselines (Identity, RBGM, EvoGraphNet). |
| `notebooks/experiments-node_features_init.ipynb` | Supporting study on node-feature initialization. |
| `notebooks/graph-preprocessing.ipynb` | Builds the connectomic `.npy` graph datasets. |
| `notebooks/graph-eda.ipynb` | Exploratory data analysis / figures. |
| `requirements.txt` | Pinned Python dependencies. |


> Several notebooks share an evolving lineage (e.g. the `num_layers` and `reservoir_params`
> families). `AUDIT/benchmark_provenance.md` maps each published table to the notebook that
> actually produced it.

## Installation

RGC-Net was developed with **Python 3.10.12** and **CUDA 12.0**; the core stack is
**PyTorch 2.0.1** with **PyTorch Geometric 2.4.0** (see `requirements.txt` for exact pins).

```bash
git clone https://github.com/basiralab/RGC-Net.git
cd RGC-Net

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for the larger datasets (DD, SLIM160); smaller experiments
(MUTAG, EMCI-AD, Simulated) run on CPU.

## Usage

Experiments are provided as Jupyter notebooks. Launch Jupyter and run a notebook top-to-bottom:

```bash
jupyter notebook          # or: jupyter lab
```

| To reproduce… | Run |
| --- | --- |
| Classification vs. depth (RGC-Net / TRGC-Net vs. GNN baselines) | `notebooks/experiments-classification_num_layers.ipynb` and `notebooks/experiments_classification_num_layers_gpu.ipynb` |
| Reservoir sensitivity to `k` and `α` (Table IV) | `notebooks/experiments_classification_reservoir_params(1).ipynb` |
| Brain-graph evolution prediction (RGC-Net-Transformer) | `notebooks/experiments-generation_transformers_models.ipynb` |
| Generation baselines | `notebooks/experiments-generation_benchmark_models.ipynb` |


## Datasets and experimental setup

**Graph classification** uses three standard binary-classification benchmarks — **MUTAG**,
**PROTEINS**, and **DD** — obtained automatically from the **TUDataset** collection via
`torch_geometric.datasets`. No manual download is required.

**Temporal graph generation** uses connectomic datasets of longitudinal brain graphs:

| Dataset | Source | 
| --- | --- | 
| EMCI-AD | ADNI / EMCI–AD connectomes | 
| SLIM160 | SLIM longitudinal neuroimaging | 
| Simulated | Longitudinal connectome simulation (Demirbilek & Rekik, *MedIA* 2023) |

Generation notebooks expect NumPy arrays under a `dataset/` folder — an adjacency tensor of shape
`(n_samples, n_timepoints, n_nodes, n_nodes)` and a node-feature (Laplacian-encoding) tensor.
**These arrays are not included** in this repository; obtain the connectomic data from the
original sources or regenerate the simulated data with `notebooks/graph-preprocessing.ipynb`
before running the generation experiments.



## Citation

If you use RGC-Net in your research, please cite:

```bibtex
@article{soussia2026rgcnet,
  author  = {Soussia, Mayssa and Salsabila, Gita Ayu and Mahjoub, Mohamed Ali and Rekik, Islem},
  title   = {Reservoir-Based Graph Convolutional Networks},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  volume  = {48},
  number  = {7},
  pages   = {7875--7889},
  doi     = {10.1109/TPAMI.2026.3670423}
}
```

## Contact

For questions about the paper or code, contact the corresponding author,
**Islem Rekik** — `i.rekik@imperial.ac.uk` ([BASIRA Lab](https://basira-lab.com/)).
</content>
</invoke>
