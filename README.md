# MPC-Decisioner: A Decision Transformer-Based Tuning for Model Predictive Control

## Description
This repository provides the code to implement the MPC-Decisioner framework proposed in the manuscript "Decision Transformer-Based Tuning for Model Predictive Control" by Nehir Güzelkaya, Marion Leibold and Martin Buss.

## Usage

Install the required libraries:

```bash
pip install -e .
```

Generate data to train MPC-Decisioner and CQL:

```bash
python data/generate_data.py
```

Produce experiments with MPC-Decisioner and CQL: 

```bash
python experiments/experiment_mpc_decisioner.py   --dataset_path dataset_mpc_decisioner.pkl
python experiments/experiment_cql.py   --dataset_path dataset_cql.pkl
```
