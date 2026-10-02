# Federated Diabetes Risk Predictor
# Federated Diabetes Risk Predictor

### Privacy-aware collaborative machine learning for diabetes-risk prediction

**A research and educational prototype exploring federated learning across distributed healthcare clients.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Listed-FF6F00?logo=tensorflow&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-Listed-7B61FF)
![Status](https://img.shields.io/badge/Project-Prototype-0F766E)

---

## Overview

Healthcare organizations often hold valuable data independently. Combining these records into a single dataset can raise privacy, governance, consent, and data-sharing concerns. At the same time, a model trained at only one institution may not reflect the diversity of other patient populations.

**Federated Diabetes Risk Predictor** explores a collaborative alternative: participating clients train a shared machine-learning model using their own local data partitions, while a coordinating server aggregates model updates. The intended workflow reduces the need to centralize raw training records.

This repository is an **academic/research prototype** intended to demonstrate a federated-learning workflow and evaluate a diabetes-related prediction task. It is **not a medical device**, does not provide a diagnosis, and must not be used to make clinical decisions.

## Project Goals

- Build a machine-learning pipeline for structured diabetes-related data.
- Simulate multiple clients with separate local data partitions.
- Train client models locally and coordinate federated training.
- Evaluate the resulting model using suitable classification metrics.
- Study the practical trade-offs among predictive performance, data locality, communication overhead, and privacy considerations.

## How Federated Learning Works

```mermaid
flowchart TD
    S[Coordinator / Server] -->|Global model| A[Client A]
    S -->|Global model| B[Client B]
    S -->|Global model| C[Client C]

    A -->|Local model update| S
    B -->|Local model update| S
    C -->|Local model update| S

    A --- DA[(Local data A)]
    B --- DB[(Local data B)]
    C --- DC[(Local data C)]

    S -->|Aggregate updates and repeat| S
