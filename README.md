# Instance-Wise Contrastive Graph Neural Network for Discovery of Novel *Aedes aegypti* Larvicidal Compounds

![AedesGNN workflow](assets/instance-wise-architecture.png)

## Overview
The proposed framework combines graph neural networks, Transformer-inspired attention, contrastive representation learning, predictive uncertainty estimation, and prospective experimental validation to prioritize novel larvicidal candidates.

#Main Architectural Components

The framework integrates:

atom-bond attentive message passing;
FastFormer-inspired bond-attention module;
structurally biased multi-head attention readout;
random-walk positional encoding;
Laplacian-filtered positional information;
centrality-based structural encoding;
virtual nodes;
skip connections;
jumping knowledge aggregation;
task-specific prediction layers.

#Contrastive Learning Strategy

Two complementary contrastive-learning levels are employed:

Whole-Molecule Contrastive Learning

Two augmented graph views are generated for each compound using:

random atom masking;
random bond deletion.

Augmented views from the same molecule are treated as positive pairs. Structurally related negative pairs are downweighted using ECFP4-based Tanimoto similarity.

#Fragment-Level Contrastive Learning

Molecular fragments are generated using the BRICS procedure. Atom embeddings belonging to the same fragment are aggregated to produce fragment-level representations, encouraging the model to learn local substructural patterns relevant to larvicidal activity.


## Environment Installation
Run the following commands in the terminal:

```bash
conda create --name graph python=3.9
conda activate graph
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Laboratory of Cheminformatics
Faculty of Pharmacy
Federal University of Goias (UFG)
Brazil
