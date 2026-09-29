# Federated Explainable AI-Agent Incident Reporting Prototype

A research prototype that combines federated learning, secure aggregation, and explainable AI (SHAP) to detect, explain, and report potential security incidents involving AI agents.

**Author:** Nimra Rahman

**Research Connection:**  
*Beyond Predictable Paths: Redefining AI Security Incident Reporting for Agents*  
arXiv:2609.24515

## Overview

This project presents a research prototype for **AI security incident reporting for compromised intelligent agents** using a combination of:

* **Federated Learning (FL)**
* **Secure Aggregation**
* **Pairwise Masking**
* **Additive Secret Sharing**
* **Fixed-Point Encoding**
* **Modular Arithmetic over `(Z_q)`**
* **Explainable AI (SHAP)**
* **Automated Security Incident Reporting**
* **Ablation Analysis**

The system is designed to detect potentially compromised AI agents from security related behavioral indicators while demonstrating how model training updates can be aggregated without directly exposing each client's plaintext update to the aggregation process.

The project also generates an explanation of individual compromise predictions using **SHAP (SHapley Additive exPlanations)** and automatically produces a structured security incident report.

---

# 1. Research Objective

The main objective of this project is to investigate whether a security monitoring system can combine:

1. **Machine learning** for compromise detection
2. **Federated learning** for distributed model training
3. **Secure aggregation** for protecting individual client updates
4. **Explainable AI** for interpreting compromise predictions
5. **Automated incident reporting** for communicating security findings.

The system is designed around the following conceptual pipeline:

```text
Security Behaviour Data
          |
          v
+-----------------------+
| Compromise Detection  |
|      Neural Network   |
+-----------------------+
          |
          v
+-----------------------+
| Federated Learning    |
|       (FedAvg)        |
+-----------------------+
          |
          v
+-----------------------+
| Secure Aggregation    |
|                       |
| Pairwise Masking      |
|        +              |
| Secret Sharing        |
|        +              |
| Modular Arithmetic    |
+-----------------------+
          |
          v
+-----------------------+
| Global Model          |
+-----------------------+
          |
          +------------------+
          |                  |
          v                  v
     SHAP Analysis     Incident Report
```

---

# 2. Key Research Concepts

## 2.1 Federated Learning

Federated Learning allows multiple clients to train a shared model without directly sending their local training data to a central server.

In this project, four clients are simulated.

Each client:

1. Receives a copy of the global model.
2. Trains the model using its local data.
3. Produces a local model update.
4. Sends the update for aggregation.

The server combines the client updates using **Federated Averaging (FedAvg)**.

Conceptually:

$$
W_{global} =
\frac{1}{N}
\sum_{i=1}^{N} W_i
$$

where:

* \(W_i\) = local model parameters from client \(i\)
* \(N\) = number of clients
* \(W_{global}\) = aggregated global model

The implementation uses five federated rounds.

---

# 3. Security Monitoring Features

The model uses eight security-related behavioral features:

| Feature                      | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| `prompt_injection_score`     | Indicator of prompt-injection-like behavior                |
| `tool_abuse_score`           | Indicator of potentially abnormal or excessive tool use    |
| `credential_access_score`    | Indicator of suspicious credential-access behavior         |
| `network_anomaly_score`      | Indicator of abnormal network activity                     |
| `data_exfiltration_score`    | Indicator of possible data-exfiltration behavior           |
| `privilege_escalation_score` | Indicator of potentially unauthorized privilege escalation |
| `behavior_deviation_score`   | Deviation from expected agent behavior                     |
| `failed_authentication_rate` | Rate of failed authentication attempts                     |

The prediction target is:

```text
compromised
```

where:

```text
0 = not compromised
1 = compromised
```

---

# 4. Synthetic Dataset

The current implementation generates a **synthetic security dataset**.

Synthetic data is used because the project is intended to demonstrate the complete machine-learning, federated-learning, explainability, and secure-aggregation pipeline without requiring access to sensitive real-world security logs.

The synthetic dataset contains nonlinear relationships between security indicators.

Examples include interactions between:

$$
\text{prompt injection} \times \text{tool abuse}
$$

$$
\text{credential access} \times \text{data exfiltration}
$$

$$
\text{privilege escalation} \times \text{credential access}
$$

These interactions allow the model to learn more complex compromise patterns than a simple linear classifier.

> **Research limitation:** Results obtained from synthetic data should not be interpreted as evidence of performance on real-world compromised-agent datasets.

---

## Final Note

This repository represents a **research prototype rather than a production security system**. Experimental results should be interpreted in the context of the synthetic dataset, simulated federated clients, simplified secure-aggregation protocol, selected model architecture, and configured experimental parameters.

# IMPORTANT:
This is a research/teaching implementation. It is NOT a production cryptographic protocol.
