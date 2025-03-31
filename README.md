# Masked Language Modeling (MLM) in the Context of Federated Learning (FL)

This repository contains the code for a parameter-efficient federated learning simulation of a masked language modeling (MLM) task using LoRA (Low-Rank Adaptation). With minor modifications, the code can also support causal language modeling (CLM).

TThe simulation has been tested on the [MedQuAD dataset](https://github.com/abachaa/MedQuAD), which consists of medical question-answer documents sourced from trusted health information providers.

## Code Structure

- `src/llm_common/dataloader.py` – Implements the data loader logic.
- `src/llm_common/model.py` – Contains model definition, training and testing loops, as well as functions to get and set model weights.
- `src/llm_common/utils.py` – Provides utility functions for logging and saving results.
- `src/fl-mlm/fl_mlm/client_app.py` – Defines the FL client application and related client-side functions.
- `src/fl-mlm/fl_mlm/server_app.py` – Defines the FL server application, including the aggregation strategy.

## Running the Simulation

1. Navigate to the `src/fl-mlm` directory.
2. Edit `pyproject.toml` to configure project dependencies as per your environment.
3. Follow the instructions in `src/fl-mlm/README.md` to run the simulation.