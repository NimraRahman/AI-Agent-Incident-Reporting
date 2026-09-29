# Experiment B — Increased Training and Model Capacity

This repository contains **Experiment B** of the Federated Explainable AI-Agent Incident Reporting Prototype.

The main project README provides the overall research motivation, architecture, security concepts, SHAP methodology, and limitations.

This README focuses specifically on the **experimental changes, configuration, results, and comparison with the baseline**.

---

# 1. Experiment Purpose

Experiment B investigates whether increasing both:

1. **Federated training duration**, and
2. **Neural-network capacity**

changes the performance of the federated compromise-detection model.

Experiment B uses a lower learning rate and more local/global training than Experiment A.

The Experiment Bhanges:

```text
Experiment A
5 rounds
3 local epochs
Learning rate = 0.01
MLP = 8 → 16 → 8 → 1

        ↓

Experiment B
10 rounds
5 local epochs
Learning rate = 0.003
MLP = 8 → 32 → 16 → 1
```

---

# 2. Experimental Configuration

| Parameter          |   Experiment A |        Experiment B |
| ------------------ | -------------: | ------------------: |
| Number of clients  |              4 |                   4 |
| Samples per client |            500 |                 500 |
| Total samples      |          2,000 |               2,000 |
| Training samples   |          1,600 |               1,600 |
| Test samples       |            400 |                 400 |
| Federated rounds   |              5 |              **10** |
| Local epochs       |              3 |               **5** |
| Learning rate      |           0.01 |           **0.003** |
| MLP                | 8 → 16 → 8 → 1 | **8 → 32 → 16 → 1** |
| Random seed        |             42 |                  42 |

The dataset and secure-aggregation configuration are kept consistent with the overall project.

---

# 3. Model Architecture

Experiment B increases the hidden-layer capacity:

```text
8 input features
      |
      v
Linear(8 → 32)
      |
     ReLU
      |
      v
Linear(32 → 16)
      |
     ReLU
      |
      v
Linear(16 → 1)
      |
      v
Compromise Probability
```

Compared with Experiment A:

```text
Experiment A:
8 → 16 → 8 → 1

Experiment B:
8 → 32 → 16 → 1
```

The purpose of this modification is to provide the model with greater representational capacity for learning nonlinear relationships in the synthetic security features.

---

# 4. Training Configuration

Experiment B uses:

```text
Optimizer: Adam
Loss: Binary Cross Entropy with Logits
Learning Rate: 0.003
Local Epochs: 5
Federated Rounds: 10
```

The lower learning rate and increased training duration were introduced together with the larger MLP.

Therefore, Experiment B is an **experimental configuration change**, rather than a controlled single-variable architecture experiment.

---

# 5. Federated Training Results

Experiment B produced the following results across the ten federated rounds:

|  Round |   Accuracy |  Precision |     Recall |         F1 |
| -----: | ---------: | ---------: | ---------: | ---------: |
|      1 |     50.00% |     50.00% |    100.00% |     66.67% |
|      2 |     50.00% |     50.00% |    100.00% |     66.67% |
|      3 |     50.00% |     50.00% |    100.00% |     66.67% |
|      4 |     51.75% |     50.91% |     97.50% |     66.90% |
|      5 |     61.50% |     57.14% |     92.00% |     70.50% |
|      6 |     64.50% |     60.66% |     82.50% |     69.92% |
|      7 |     64.75% |     62.14% |     75.50% |     68.17% |
|      8 |     64.75% |     63.35% |     70.00% |     66.51% |
|      9 |     66.00% |     65.09% |     69.00% |     66.99% |
| **10** | **65.25%** | **65.33%** | **65.00%** | **65.16%** |

The final round produced:

```text
Accuracy  = 65.25%
Precision = 65.33%
Recall    = 65.00%
F1-score  = 65.16%
```

The highest observed accuracy during training was:

```text
Round 9
Accuracy = 66.00%
```

The final Round 10 accuracy decreased to:

```text
65.25%
```

This demonstrates that additional federated rounds do not necessarily result in monotonically increasing test-set performance.

---

# 6. Centralized Reference

The centralized model for Experiment B achieved:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **73.00%** |
| Precision | **76.74%** |
| Recall    | **66.00%** |
| F1-score  | **70.97%** |

The final federated model achieved:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **65.25%** |
| Precision | **65.33%** |
| Recall    | **65.00%** |
| F1-score  | **65.16%** |

The centralized and federated models therefore provide different reference points for evaluating the effect of distributed training under this experimental setup.

---

# 7. Experiment A vs Experiment B

The primary comparison is:

| Metric           |   Experiment A |    Experiment B |
| ---------------- | -------------: | --------------: |
| Federated rounds |              5 |              10 |
| Local epochs     |              3 |               5 |
| Learning rate    |           0.01 |           0.003 |
| MLP              | 8 → 16 → 8 → 1 | 8 → 32 → 16 → 1 |
| Final Accuracy   |     **66.00%** |      **65.25%** |
| Final Precision  |     **66.33%** |      **65.33%** |
| Final Recall     |     **65.00%** |      **65.00%** |
| Final F1         |     **65.66%** |      **65.16%** |

The final federated metrics in Experiment B are therefore slightly lower than those observed in Experiment A.

However, Experiment B reached **66.00% accuracy at Round 9**, showing that the larger model and longer training schedule can produce comparable intermediate performance without guaranteeing improvement at the final round.

---

# 8. Centralized Comparison

The centralized reference results were:

| Metric    | Experiment A | Experiment B |
| --------- | -----------: | -----------: |
| Accuracy  |       72.25% |   **73.00%** |
| Precision |       77.99% |       76.74% |
| Recall    |       62.00% |   **66.00%** |
| F1-score  |       69.08% |   **70.97%** |

Under the centralized configuration, Experiment B produced higher accuracy, recall, and F1-score than Experiment A, while precision was slightly lower.

This provides evidence that the modified training configuration and larger architecture changed the model's behavior, although the experiment does not isolate the individual effect of each change.

---

# 9. Secure Aggregation Verification

Experiment B maintained the same secure-aggregation verification behavior.

Across the ten federated rounds:

```text
Aggregate modular error       = 0
Pairwise mask residual        = 0
Secret-share reconstruction   = 0
Overall verification          = PASS
```

The protected aggregation therefore matched the corresponding plaintext aggregation under the tested experimental conditions.

This verifies the implementation's aggregation correctness for the experiment.

It should not be interpreted as a formal cryptographic security proof.

---

# 10. Ablation Results

Experiment B produced:

| Configuration               | Accuracy | Precision | Recall |     F1 |
| --------------------------- | -------: | --------: | -----: | -----: |
| Centralized ML              |   73.00% |    76.74% | 66.00% | 70.97% |
| FedAvg                      |   65.25% |    65.33% | 65.00% | 65.16% |
| FedAvg + Secure Aggregation |   65.25% |    65.33% | 65.00% | 65.16% |

The FedAvg and FedAvg + Secure Aggregation results are identical.

Therefore, no measurable classification-performance difference was observed between the plaintext FedAvg and the implemented secure-aggregation version in this experiment.

---

# 11. SHAP Incident Explanation

The selected incident in Experiment B produced:

```text
Compromise Probability: 0.3731
Severity: MEDIUM
```

Top model-attributed contributors:

| Rank | Feature                      | SHAP Value |
| ---: | ---------------------------- | ---------: |
|    1 | `prompt_injection_score`     |  -0.036389 |
|    2 | `data_exfiltration_score`    |  -0.028611 |
|    3 | `credential_access_score`    |  -0.021381 |
|    4 | `privilege_escalation_score` |  -0.011060 |
|    5 | `tool_abuse_score`           |  -0.010402 |

These values describe the model's attribution for the selected prediction.

They should not be interpreted as evidence that these features causally produced an actual security incident.

---

# 12. Output Files

Experiment B produces:

```text
outputs_experiment_C/
├── federated_metrics.csv
├── ablation_results.csv
├── incident_report.csv
├── incident_report.txt
├── federated_performance.png
├── shap_incident_explanation.png
├── confusion_matrix.png
└── secure_aggregation_verification.txt
```

The most important experimental artifacts are:

```text
federated_metrics.csv
ablation_results.csv
federated_performance.png
shap_incident_explanation.png
confusion_matrix.png
secure_aggregation_verification.txt
```

---

# 13. Reproducing Experiment B

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Experiment B implementation:

```bash
python main.py
```

The configuration should contain:

```python
FED_ROUNDS = 10
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.003
```

and the model should use:

```text
8 → 32 → 16 → 1
```

---

# 14. Interpretation

Experiment B provides two useful observations.

### Increased Model Capacity

The model was expanded from:

```text
8 → 16 → 8 → 1
```

to:

```text
8 → 32 → 16 → 1
```

This increases the number of trainable parameters and representational capacity.

### Increased Training

Training was increased from:

```text
5 federated rounds
3 local epochs
```

to:

```text
10 federated rounds
5 local epochs
```

while reducing the learning rate from:

```text
0.01 → 0.003
```

The resulting final federated accuracy was **65.25%**, compared with **66.00%** for Experiment A.

Therefore, the experiment does not demonstrate an improvement in final federated test accuracy.

However, the centralized reference model achieved **73.00% accuracy**, compared with **72.25%** in Experiment A.

The Experiment Bonsequently demonstrates that increasing model capacity and training duration can change model behavior without necessarily producing higher final federated test performance.

---

# 15. Experimental Limitation

Experiment B changes multiple variables simultaneously:

```text
Federated rounds
+
Local epochs
+
Learning rate
+
Model capacity
```

Therefore, the Experiment Bannot independently attribute the observed performance change to only the larger neural network.

A controlled architecture ablation would require keeping the training configuration fixed while changing only the model architecture.

Similarly, a controlled training ablation would change training parameters while keeping model capacity fixed.

---

# 16. Role in the Overall Research

Experiment B extends the baseline experiment by testing a higher-capacity model and longer training schedule.

The experimental progression is:

```text
Experiment A
Baseline Model
      |
      v
5 rounds
3 local epochs
8 → 16 → 8 → 1
      |
      v
Baseline Reference
      |
      v
Experiment B
      |
      v
10 rounds
5 local epochs
8 → 32 → 16 → 1
      |
      v
Higher Capacity + Longer Training
```

The purpose is not to claim that the larger configuration is universally superior, but to document how the selected configuration behaves under the same research pipeline.

---

## Repository Context

For the complete explanation of the research problem, system architecture, security mechanisms, SHAP methodology, dataset generation, and overall limitations, see the **main project README**.
