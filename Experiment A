# Experiment A — Baseline Configuration

This repository contains the **baseline experimental configuration** for the Federated Explainable AI-Agent Incident Reporting Prototype.

The main project README provides the overall research motivation, system architecture, security concepts, SHAP methodology, and limitations.

This README focuses specifically on the **baseline experimental setup and results**.

---

## 1. Experiment Purpose

Experiment A establishes the reference configuration for the subsequent experiments.

The objective is to measure the performance of the initial federated model before increasing:

* Federated training duration
* Local training duration
* Model capacity

The results from Experiment A therefore serve as the baseline for comparison with Experiment C.

---

# 2. Experimental Configuration

| Parameter           |       Experiment A |
| ------------------- | -----------------: |
| Number of clients   |                  4 |
| Samples per client  |                500 |
| Total samples       |              2,000 |
| Training samples    |              1,600 |
| Test samples        |                400 |
| Federated rounds    |              **5** |
| Local epochs        |              **3** |
| Learning rate       |           **0.01** |
| Hidden architecture | **8 → 16 → 8 → 1** |
| Random seed         |                 42 |
| Test split          |                20% |

The secure-aggregation parameters remain those defined in the main project.

---

# 3. Model Architecture

Experiment A uses the baseline MLP:

```text
8 input features
      |
      v
Linear(8 → 16)
      |
     ReLU
      |
      v
Linear(16 → 8)
      |
     ReLU
      |
      v
Linear(8 → 1)
      |
      v
Compromise Probability
```

This architecture is intentionally lightweight and serves as the reference model for the experimental comparison.

---

# 4. Training Configuration

Local client training uses:

```text
Optimizer: Adam
Loss: Binary Cross Entropy with Logits
Learning Rate: 0.01
Local Epochs: 3
Federated Rounds: 5
```

The same model architecture is used for the centralized comparison and federated experiment.

---

# 5. Federated Training Results

Performance across the five federated rounds:

| Round |   Accuracy |  Precision |     Recall |         F1 |
| ----: | ---------: | ---------: | ---------: | ---------: |
|     1 |     50.00% |     50.00% |    100.00% |     66.67% |
|     2 |     51.75% |     50.91% |     98.00% |     67.01% |
|     3 |     63.50% |     59.93% |     81.50% |     69.07% |
|     4 |     65.50% |     64.35% |     69.50% |     66.83% |
| **5** | **66.00%** | **66.33%** | **65.00%** | **65.66%** |

The final federated model achieved:

```text
Accuracy  = 66.00%
Precision = 66.33%
Recall    = 65.00%
F1-score  = 65.66%
```

The highest federated accuracy in this experiment occurred at **Round 5: 66.00%**.

---

# 6. Centralized vs Federated Performance

The centralized reference model achieved:

| Metric    | Centralized ML | Federated Learning |
| --------- | -------------: | -----------------: |
| Accuracy  |         72.25% |             66.00% |
| Precision |         77.99% |             66.33% |
| Recall    |         62.00% |             65.00% |
| F1-score  |         69.08% |             65.66% |

The centralized model therefore provides a reference point for assessing the performance of the federated configuration.

These results should be interpreted within the context of the synthetic dataset and simulated clients.

---

# 7. Secure Aggregation Verification

The secure aggregation mechanism was verified during the experiment.

Across the federated rounds:

```text
Aggregate modular error          = 0
Pairwise mask residual           = 0
Secret-sharing reconstruction    = 0
Overall verification             = PASS
```

The secure aggregation result matched the corresponding plaintext aggregation under the implemented experimental conditions.

This demonstrates **implementation correctness for the tested aggregation procedure**.

It does not constitute a formal cryptographic security proof.

---

# 8. Ablation Results

The ablation experiment produced:

| Configuration               | Accuracy | Precision | Recall |     F1 |
| --------------------------- | -------: | --------: | -----: | -----: |
| Centralized ML              |   72.25% |    77.99% | 62.00% | 69.08% |
| FedAvg                      |   66.00% |    66.33% | 65.00% | 65.66% |
| FedAvg + Secure Aggregation |   66.00% |    66.33% | 65.00% | 65.66% |

The FedAvg and FedAvg + Secure Aggregation results are identical in this experiment.

This indicates that the implemented secure-aggregation mechanism did not introduce an observable difference in the resulting model metrics under the tested configuration.

---

# 9. SHAP Incident Explanation

The selected incident produced:

```text
Compromise Probability: 0.3844
Severity: MEDIUM
```

Top model-attributed contributors:

| Rank | Feature                      | SHAP Value |
| ---: | ---------------------------- | ---------: |
|    1 | `prompt_injection_score`     |  -0.031371 |
|    2 | `data_exfiltration_score`    |  -0.026357 |
|    3 | `credential_access_score`    |  -0.020117 |
|    4 | `tool_abuse_score`           |  -0.011890 |
|    5 | `privilege_escalation_score` |  -0.010303 |

These values represent **model-attributed contributions for the selected prediction**, not causal explanations of an actual security incident.

---

# 10. Experiment Outputs

The experiment generates:

```text
outputs/
├── federated_metrics.csv
├── ablation_results.csv
├── incident_report.csv
├── incident_report.txt
├── federated_performance.png
├── shap_incident_explanation.png
├── confusion_matrix.png
└── secure_aggregation_verification.txt
```

The most important files for reviewing the experiment are:

```text
federated_metrics.csv
ablation_results.csv
federated_performance.png
shap_incident_explanation.png
confusion_matrix.png
secure_aggregation_verification.txt
```

---

# 11. Reproducing Experiment A

Install the dependencies listed in:

```text
requirements.txt
```

Then run:

```bash
python main.py
```

The experiment uses:

```python
SEED = 42
```

so that the experimental setup is reproducible under the same software and hardware conditions.

---

# 12. Role in the Overall Research

Experiment A provides the **baseline reference point**.

It establishes:

```text
Baseline Training
       |
       v
Baseline Model Capacity
       |
       v
Baseline Federated Performance
       |
       v
Baseline for Experimental Comparison
```

Experiment C subsequently changes the training configuration and model capacity to investigate whether additional optimization and representational capacity affect the observed federated performance.

---

## Repository Context

For the complete explanation of the research problem, Federated Learning, secure aggregation, SHAP, synthetic dataset, threat model, limitations, and overall architecture, see the **main project README**.
