# physical-commonsense

[![Build Status](https://travis-ci.org/mbforbes/physical-commonsense.svg?branch=master)](https://travis-ci.org/mbforbes/physical-commonsense)
[![license MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mbforbes/physical-commonsense/blob/master/LICENSE.txt)

This is the code, data, and website repository accompanying the paper:

**[Do Neural Language Representations Learn Physical Commonsense?](https://arxiv.org/abs/1908.02899)** <br />
[Maxwell Forbes](http://maxwellforbes.com/), [Ari Holtzman](https://ari-holtzman.github.io/), [Yejin Choi](https://homes.cs.washington.edu/~yejin/) <br />
CogSci 2019

For an overview of the project and academic publication, please see the [project
webpage](https://mbforbes.github.io/physical-commonsense/). The rest of this README is
focused on the **code** and **data** behind the project.

## Setup

```bash
# (1) Create a fresh virtualenv. Use Python 3.7+

# (2) Install pytorch. (This code was written using Pytorch 1.1) Follow directions at
# https://pytorch.org/.

# (3) Install CLIP

# (4) Install other Python dependencies using pip:
pip install -r requirements.txt

# (5) Retrieve external data. (Our data is already in subfolders of data/; this is for
# larger blobs like GloVe.) This script also makes some directories we'll need.
./scripts/get_data.sh

# (6) Download MSCOCO to data/mscoco by running the following wget commands
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/train2014.zip


# Then unzip the downloaded files and move the train and val images into one 
# directory called images. So the path to images should be data/mscoco/images
```

## Run

```bash
# Note that per-datum results for the programs below are written to data/results/

# Get clip embeddings
python -m pc.get_clip_embeddings

# Run MLP training experiments with clip embeddings
python -m pc.clip_experiments

# For hyperparamter tuning add the --dev flag
# This uses a train/val split sampled from the train set exclusively
# You can also set a number of hyperparameters
python -m pc.finetune_clip --dev \
                           --task "situated-OA" \
                           --lr 1e-7 \
                           --warmup-ratio 0.1 \
                           --batch-size 64 \
                           --activation relu \
                           --dropout 0.0

# Finetune clip model with text only
python -m pc.finetune_clip --text-only --task "situated-OP" --epochs 5 --lr 1e-7 --seed 205
python -m pc.finetune_clip --text-only --task "situated-OA" --epochs 5 --lr 7e-7 --seed 205
python -m pc.finetune_clip --text-only --task "situated-AP" --epochs 2 --lr 1e-7 --seed 205

# Finetune clip model with mscoco images (still need to run get clip embeddings before this)
python -m pc.finetune_clip --task "situated-OP" --epochs 5 --lr 1e-7 --seed 205
python -m pc.finetune_clip --task "situated-OA" --epochs 5 --lr 7e-7 --seed 205
python -m pc.finetune_clip --task "situated-AP" --epochs 2 --lr 1e-7 --seed 205 

# Finetune clip model with gan generated images (still need to run get clip embeddings before this)
python -m pc.finetune_clip --gan-imgs --task "situated-OP" --epochs 5 --lr 1e-7 --seed 205
python -m pc.finetune_clip --gan-imgs --task "situated-OA" --epochs 5 --lr 7e-7 --seed 205
python -m pc.finetune_clip --gan-imgs --task "situated-AP" --epochs 2 --lr 1e-7 --seed 205

# Random image experiments, match a sentence with a randomly selected image instead of the corresponding one
python -m pc.finetune_clip --rand-imgs --task "situated-OP" --epochs 5 --lr 1e-7 --seed 205
python -m pc.finetune_clip --rand-imgs --task "situated-OA" --epochs 5 --lr 7e-7 --seed 205
python -m pc.finetune_clip --rand-imgs --task "situated-AP" --epochs 2 --lr 1e-7 --seed 205
python -m pc.finetune_clip --gan-imgs --rand-imgs --task "situated-OP" --epochs 5 --lr 1e-7 --seed 205
python -m pc.finetune_clip --gan-imgs --rand-imgs --task "situated-OA" --epochs 5 --lr 5e-7 --seed 205
python -m pc.finetune_clip --gan-imgs --rand-imgs --task "situated-AP" --epochs 2 --lr 1e-7 --seed 205

# Run the baselines: random and majority.
python -m pc.baselines

# Run GloVe, Dependency Embeddings, and ELMo. For detailed info on hyperparameters and
# cross validation, see the `main()` function in pc/models.py
python -m pc.experiments

# Run BERT. NOTE: 1 epoch for "situated-AP" is not to handicap the model; rather, it
# overfits and achieves 0.0 F1 score for epoch 2+.
python -m pc.bert --task "abstract-OP"
python -m pc.bert --task "situated-OP"
python -m pc.bert --task "situated-OA"
python -m pc.bert --task "situated-AP" --epochs 1

# Display human results.
python -m pc.human

# Compute statistical significance. (Requires all baselines and model output.)
python -m pc.significance

# Convert BERT's output on the situated-AP task to per-category output (for making
# graphs).
python -m scripts.perdatum_to_category

# Produce graphs (from paper) for analyzing BERT's output on situated-AP task per-
# category, as well as comparing performance vs word occurrence in natural language
# (found in data/nl/). Writes graphs to data/results/graphs.
python -m pc.graph
```

## Data

In this repository, we provide the **abstract** and **stitauted** datasets collected for
this project, as well as some auxillary data we used (sentence constructions, statistics
of natural language).

Note that the `scripts/get_data.sh` script will download additional data (GloVe
embeddings, Dependency Embeddings, and a cache of ELMo embeddings), which we don't
describe here.

Here is an overview of what's provided:

```txt
data/
├── dep-embs  (retrieve with `scripts/get_data.sh`)
├── elmo      (retrieve with `scripts/get_data.sh`)
├── glove     (retrieve with `scripts/get_data.sh`)
├── human     Expert annotations establishing human performance on the task.
├── nl        Natural language statistics: frequency of our words in a large corpus.
├── pc        Our abstract and situated datasets are here.
└── sentences Sentences automatically constructed for contextual models (ELMo, BERT).
```
