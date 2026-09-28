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

# 5. Machine Learning Model

The project uses a small feed-forward neural network implemented in PyTorch.

Architecture:

```text
Input Layer
8 features
    |
    v
Linear(8 → 16)
    |
    v
ReLU
    |
    v
Linear(16 → 8)
    |
    v
ReLU
    |
    v
Linear(8 → 1)
    |
    v
Compromise Probability
```

The model is trained using:

* **Binary Cross Entropy with Logits Loss**
* **Adam optimizer**
* Learning rate = `0.01`
* Local epochs = `3`

The model produces a logit which is converted into a probability using the sigmoid function.

---

# 6. Federated Learning Configuration

The current experiment uses:

```python
NUM_CLIENTS = 4
SAMPLES_PER_CLIENT = 500
FED_ROUNDS = 5
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01
TEST_SIZE = 0.20
```

Therefore:

```text
4 clients
×
500 samples per client
=
2,000 total samples
```

Each client trains locally before its model update is aggregated.

---

# 7. Secure Aggregation

## 7.1 Why Secure Aggregation?

In ordinary Federated Learning, the server receives individual client model updates.

Although the original training data is not directly transmitted, individual model updates may still reveal information about the client's local data under certain threat models.

Secure aggregation aims to ensure that the server obtains the **aggregate update** without directly seeing each client's plaintext update.

The conceptual goal is:

```text
Client 1 update ──┐
Client 2 update ──┤
Client 3 update ──┼──> Aggregate
Client 4 update ──┘
```

while preventing the server from directly observing:

```text
Client 1 plaintext update
Client 2 plaintext update
Client 3 plaintext update
Client 4 plaintext update
```

---

# 8. Fixed-Point Encoding

Machine-learning model parameters are floating-point values.

For example:

```text
0.00123456
```

Secure aggregation in this prototype operates on integers using modular arithmetic.

Therefore, floating-point values are converted into fixed-point integers.

The implementation uses:

```python
FIXED_POINT_SCALE = 10**8
```

The conversion is:

$$
x_{fixed} =
\text{round}(x \times 10^8)
$$

For example:

$$
0.00123456 \times 10^8 = 123456
$$

So:

```text
0.00123456
```

becomes:

```text
123456
```

After aggregation, the value is converted back approximately using:

$$
x =
\frac{x_{fixed}}{10^8}
$$

This allows the system to represent model updates as integers while preserving a controlled amount of numerical precision.

---

# 9. What is \(\mathbb{Z}_q\)?

The secure aggregation implementation uses modular arithmetic over:

$$
\mathbb{Z}_q
$$

In simple terms, \(\mathbb{Z}_q\) represents the integers:

$$
\{0,1,2,\ldots,q-1\}
$$

with arithmetic performed modulo \(q\).

For example, if:

$$
q=17
$$

then:

$$
15+6=21
$$

but modulo 17:

$$
21 \bmod 17=4
$$

Therefore:

$$
15+6 \equiv 4 \pmod {17}
$$

The arithmetic effectively "wraps around" when it reaches \(q\).

---

# 10. Prime Modulus Used by the Project

The implementation uses:

```python
Q = 2305843009213693951
```

which is:

$$
Q=2^{61}-1
$$

This is a large prime modulus.

Because the modulus is prime, \(\mathbb{Z}_Q\) can also be treated as a finite field, commonly written as:

$$
\mathbb{F}_Q
$$

The implementation primarily relies on modular addition and subtraction for secure aggregation.

A precise description for this project is:

> Model updates are converted to fixed-point integers and represented using modular arithmetic over \(\mathbb{Z}_q\), where aggregation is performed modulo a large prime \(q\).

---

# 11. Pairwise Masking

The implementation uses pairwise masks to hide individual client updates.

Suppose two clients have plaintext encoded updates:

```text
Client 1 = 100
Client 2 = 250
```

Suppose they share a random mask:

```text
r = 37
```

Client 1 sends:

$$
100+37=137
$$

Client 2 sends:

$$
250-37=213
$$

The server receives:

```text
137
213
```

The server can calculate:

$$
137+213=350
$$

The original plaintext aggregate is:

$$
100+250=350
$$

The mask cancels:

$$
+r-r=0
$$

Therefore:

$$
(100+r)+(250-r)=100+250
$$

For multiple clients, the implementation creates pairwise masks between client pairs.

---

# 12. Additive Secret Sharing

The implementation additionally demonstrates additive secret sharing.

A value can be divided into shares whose modular sum reconstructs the original value.

For a value \(x\), two shares can conceptually be constructed as:

$$
s_1=r
$$

$$
s_2=x-r \pmod q
$$

Reconstruction is:

$$
s_1+s_2
\equiv
r+(x-r)
\equiv x
\pmod q
$$

In the implementation, the masked update is divided into two additive shares.

The shares are then reconstructed before the final aggregation comparison.

---

# 13. Secure Aggregation Pipeline

The secure aggregation process is:

```text
Floating-Point Model Updates
            |
            v
     Fixed-Point Encoding
            |
            v
     Modular Representation
            |
            v
       Pairwise Masking
            |
            v
     Additive Secret Sharing
            |
            v
      Share Reconstruction
            |
            v
      Modular Aggregation
            |
            v
       Fixed-Point Decode
            |
            v
     Aggregated Model Update
```

The implementation also computes the plaintext aggregate separately so that the secure aggregation result can be verified.

---

# 14. Secure Aggregation Verification

The code does not simply assume that secure aggregation is working.

It explicitly verifies the aggregation process.

The verification checks:

1. **Aggregate modular error**
2. **Pairwise mask residual**
3. **Secret-sharing reconstruction error**
4. **Overall verification status**

The expected result is:

```text
PASS
```

when the secure aggregation result matches the plaintext aggregation within the configured numerical tolerance.

This provides an experimental correctness check for the implementation.

> Verification of mathematical correctness is not equivalent to a formal cryptographic security proof.

---

# 15. Important Security Distinction

Secure aggregation and Explainable AI solve different problems.

### Secure Aggregation

Secure aggregation addresses:

> **How can client model updates be aggregated without directly exposing each individual plaintext update?**

### SHAP

SHAP addresses:

> **Which input features contributed to this particular model prediction?**

Therefore:

```text
Secure Aggregation
        |
        v
Update confidentiality during aggregation


SHAP
        |
        v
Model prediction interpretability
```

They are complementary rather than interchangeable.

---

# 16. Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to explain compromise predictions.

The implementation uses:

```python
shap.KernelExplainer
```

A background sample of up to 40 observations is used.

SHAP estimates the contribution of each input feature to the model's prediction.

Conceptually:

$$
f(x)
=
\phi_0+
\sum_{i=1}^{n}\phi_i
$$

where:

* \(f(x)\) = model output
* \(\phi_0\) = baseline prediction
* \(\phi_i\) = contribution of feature \(i\)

A positive SHAP value indicates that the feature contributed toward the model's positive prediction under the selected explanation setup.

A negative SHAP value indicates contribution in the opposite direction.

---

# 17. Incident Explanation

For the selected incident, the system identifies the strongest SHAP contributors.

The incident report contains the top five contributors.

For example:

```text
Top Contributors:

1. data_exfiltration_score
2. credential_access_score
3. privilege_escalation_score
4. network_anomaly_score
5. tool_abuse_score
```

The exact features depend on the generated data and trained model.

> SHAP identifies model-attributed feature contributions. It does not establish that a feature is the actual causal reason for a security incident.

---

# 18. Automated Incident Reporting

The system automatically generates an incident report containing:

* Compromise probability
* Severity level
* Top SHAP contributors
* Security interpretation
* Explanation notes

Severity is assigned using the following thresholds:

| Probability | Severity |
| ----------: | -------- |
|      ≥ 0.75 | CRITICAL |
|      ≥ 0.50 | HIGH     |
|      ≥ 0.25 | MEDIUM   |
|      < 0.25 | LOW      |

These thresholds are configuration choices for the research prototype and should not be interpreted as universally valid security standards.

---

# 19. Ablation Study

The project includes an ablation analysis comparing different training configurations.

The purpose is to examine the effect of the different components of the proposed system.

The experiment compares:

### 1. Centralized ML

All training data is available centrally.

```text
All data
   |
   v
Centralized Model
```

### 2. Federated Learning

Data remains distributed among clients while model updates are aggregated.

```text
Client 1 ─┐
Client 2 ─┤
Client 3 ─┼──> FedAvg
Client 4 ─┘
```

### 3. Federated Learning + Secure Aggregation

Federated model updates are additionally protected through the secure aggregation mechanism.

```text
Client Updates
      |
      v
Pairwise Masking
      |
      v
Secret Sharing
      |
      v
Secure Aggregation
      |
      v
Global Model
```

The ablation results are stored in:

```text
outputs/ablation_results.csv
```

---

# 20. Evaluation Metrics

The project evaluates the classification model using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

The confusion matrix is generated as:

```text
outputs/confusion_matrix.png
```

These metrics provide different views of classification performance.

For security detection, precision and recall can be particularly informative because false positives and false negatives have different operational implications.

---

# 21. Federated Performance

Federated training performance is recorded across the federated rounds.

The output file is:

```text
outputs/federated_metrics.csv
```

The corresponding visualization is:

```text
outputs/federated_performance.png
```

This allows the progression of federated training to be inspected across rounds.

---

# 22. SHAP Visualization

The SHAP explanation visualization is saved as:

```text
outputs/shap_incident_explanation.png
```

It provides a visual representation of the features contributing to the selected incident prediction.

---

# 23. Output Files

After running the project, the following files are generated:

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

### `federated_metrics.csv`

Contains federated training/evaluation information across rounds.

### `ablation_results.csv`

Contains results for:

* Centralized ML
* FedAvg
* FedAvg + secure aggregation

### `incident_report.csv`

Machine-readable incident report.

### `incident_report.txt`

Human-readable incident report.

### `federated_performance.png`

Federated training performance visualization.

### `shap_incident_explanation.png`

SHAP-based explanation of the selected incident.

### `confusion_matrix.png`

Classification confusion matrix.

### `secure_aggregation_verification.txt`

Verification results for the secure aggregation mechanism.

---

# 24. Project Structure

The project can be organized as:

```text
federated_xai_secure_incident_reporting/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── outputs/
    ├── federated_metrics.csv
    ├── ablation_results.csv
    ├── incident_report.csv
    ├── incident_report.txt
    ├── federated_performance.png
    ├── shap_incident_explanation.png
    ├── confusion_matrix.png
    └── secure_aggregation_verification.txt
```

The main implementation is contained in:

```text
main.py
```

---

# 25. Requirements

The project requires Python 3.9 or newer.

Main dependencies include:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
torch
shap
```

A `requirements.txt` file should contain the required package versions used for the experiment.

---

# 26. Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd federated_xai_secure_incident_reporting
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 27. Running the Project

Run:

```bash
python main.py
```

The system will:

1. Set the random seed.
2. Generate the synthetic security dataset.
3. Split the data into clients.
4. Train the centralized baseline.
5. Perform federated training.
6. Apply secure aggregation.
7. Verify secure aggregation correctness.
8. Evaluate the global model.
9. Perform SHAP analysis.
10. Generate an automated incident report.
11. Run the ablation study.
12. Save CSV, TXT, and PNG results.

---

# 28. GPU Support

The implementation automatically checks whether CUDA is available:

```python
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

If CUDA is available:

```text
Using device: cuda
```

Otherwise:

```text
Using device: cpu
```

No code modification is required to select the device.

To verify CUDA availability in Python:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

> The current neural network is intentionally small. GPU acceleration may therefore provide limited benefit compared with larger deep-learning workloads.

---

# 29. Reproducibility

The experiment uses:

```python
SEED = 42
```

The random seed is used to improve reproducibility of:

* Dataset generation
* Data splitting
* Model initialization
* Federated experiments
* SHAP background sampling

However, exact reproducibility can still depend on:

* Python version
* PyTorch version
* SHAP version
* NumPy version
* Operating system
* CPU/GPU backend
* CUDA version
* Floating-point behavior

Therefore, the seed improves reproducibility but does not guarantee bit-for-bit identical results across every environment.

---

# 30. Main Configuration

The primary experimental configuration is:

```python
SEED = 42

NUM_CLIENTS = 4
SAMPLES_PER_CLIENT = 500

FED_ROUNDS = 5
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01

TEST_SIZE = 0.20

Q = 2305843009213693951

FIXED_POINT_SCALE = 10**8

OUTPUT_DIR = "outputs"
```

These parameters can be modified to perform additional experiments.

---

# 31. Mathematical Summary

The overall secure federated-learning workflow can be summarized mathematically.

Let the local model update of client \(i\) be:

$$
\Delta_i
$$

After fixed-point encoding:

$$
\hat{\Delta}_i =
\operatorname{round}
(\Delta_i \times S)
$$

where:

$$
S=10^8
$$

Each client receives pairwise masks.

For client \(i\), the masked update can be represented conceptually as:

$$
M_i =
\hat{\Delta}_i+
\sum_j r_{ij}
-
\sum_j r_{ji}
\pmod q
$$

Because each pairwise mask appears once positively and once negatively:

$$
\sum_i M_i
\equiv
\sum_i \hat{\Delta}_i
\pmod q
$$

Thus the masks cancel during aggregation.

After aggregation, the fixed-point result is decoded:

$$
\Delta_{global}
=
\frac{\operatorname{Decode}
\left(
\sum_i M_i \pmod q
\right)}
{S}
$$

The implementation then verifies that the secure aggregation result corresponds to the plaintext aggregation.

---

# 32. Security Model

The prototype demonstrates protection against direct exposure of individual model updates during the aggregation process through pairwise masking.

The conceptual threat model assumes that the aggregation process should not directly reveal each individual client's plaintext update.

However, the current implementation does **not** provide a complete production-grade secure aggregation protocol.

The project does not implement all protections normally expected in a deployed cryptographic system.

---

# 33. Security Limitations

The implementation should not be described as a complete secure federated-learning security solution.

Important limitations include:

### 33.1 No Secure Communication Layer

The prototype does not implement a complete authenticated encrypted communication protocol such as TLS between clients and server.

### 33.2 No Production Key Management

Pairwise masking is demonstrated algorithmically, but a production deployment would require secure key establishment and lifecycle management.

### 33.3 No Dropout Recovery

The current demonstration does not implement robust handling for clients that disconnect during secure aggregation.

### 33.4 No Malicious Client Defense

The system does not implement defenses against:

* Model poisoning
* Byzantine clients
* Backdoor attacks
* Malicious updates

### 33.5 No Formal Cryptographic Proof

The implementation demonstrates the mathematical mechanism and verifies correctness experimentally.

It does not provide a formal cryptographic security proof.

### 33.6 No Differential Privacy

Secure aggregation is not equivalent to differential privacy.

Secure aggregation protects the aggregation process from directly exposing individual updates, while differential privacy provides a formal privacy guarantee through controlled statistical noise.

### 33.7 SHAP Is Not Private by Default

SHAP explanations can reveal information about model behavior.

Therefore, secure aggregation does not automatically make SHAP explanations private.

If explanations themselves contain sensitive information, additional privacy mechanisms would be required.

---

# 34. Secure Aggregation vs. Homomorphic Encryption

This project should not be described as implementing homomorphic encryption.

The current approach uses:

```text
Fixed-point encoding
        +
Pairwise masking
        +
Additive secret sharing
        +
Modular arithmetic
```

This is conceptually different from fully homomorphic encryption or partially homomorphic encryption.

The implementation demonstrates secure aggregation techniques rather than encrypted computation over ciphertexts.

---

# 35. Important Interpretation of \(\mathbb{Z}_q\)

The finite field or modular arithmetic itself is not what provides the complete privacy guarantee.

Instead:

```text
Zq / modular arithmetic
        |
        v
Provides a mathematical domain for computation


Pairwise masking
        |
        v
Hides individual updates


Secret sharing
        |
        v
Splits protected values into shares


Secure aggregation
        |
        v
Produces the aggregate
```

Therefore, the security properties come from the overall protocol and its threat model, not simply from using \(\mathbb{Z}_q\).

---

# 36. Research Contributions Demonstrated

The project demonstrates the integration of several research components:

### Contribution 1 — Compromise Detection

A neural-network classifier detects potentially compromised AI agents based on security-behavior features.

### Contribution 2 — Federated Learning

The classifier can be trained across distributed clients using FedAvg.

### Contribution 3 — Secure Aggregation

Client updates are protected through pairwise masking and additive secret sharing.

### Contribution 4 — Modular Representation

Floating-point model updates are converted to fixed-point integers and processed using modular arithmetic over a large prime modulus.

### Contribution 5 — Verification

The implementation verifies that the protected aggregation corresponds to the plaintext aggregate.

### Contribution 6 — Explainability

SHAP identifies model-attributed feature contributions for an individual incident.

### Contribution 7 — Automated Incident Reporting

The system converts the prediction and explanation into a structured security incident report.

### Contribution 8 — Ablation Analysis

Centralized ML, FedAvg, and FedAvg with secure aggregation are compared experimentally.

---

# 37. Research Questions

The implementation can support research questions such as:

### RQ1

Can a neural-network model identify potentially compromised AI agents from behavioral security indicators?

### RQ2

Can Federated Learning train the compromise-detection model while keeping local training data distributed?

### RQ3

Can pairwise masking and additive secret sharing preserve the correctness of federated model aggregation?

### RQ4

Can the secure aggregation implementation be experimentally verified against plaintext aggregation?

### RQ5

Can SHAP provide interpretable feature-level explanations for compromise predictions?

### RQ6

What differences are observed between centralized training, Federated Learning, and Federated Learning with secure aggregation?

---

# 38. Reproducible Experimental Workflow

A complete experiment can be reproduced using the following sequence:

```text
1. Install Python dependencies
          |
          v
2. Run main.py
          |
          v
3. Generate synthetic security data
          |
          v
4. Train centralized baseline
          |
          v
5. Train federated clients
          |
          v
6. Perform FedAvg
          |
          v
7. Apply secure aggregation
          |
          v
8. Verify aggregation correctness
          |
          v
9. Evaluate global model
          |
          v
10. Generate SHAP explanation
          |
          v
11. Generate incident report
          |
          v
12. Run ablation analysis
          |
          v
13. Inspect outputs/
```

---

# 39. Example Incident Report Structure

The generated report contains information conceptually similar to:

```text
AI SECURITY INCIDENT REPORT
===========================

Compromise Probability:
0.XX

Severity:
HIGH

Top SHAP Contributors:
1. feature_name
2. feature_name
3. feature_name
4. feature_name
5. feature_name

Interpretation:
The model attributes the prediction primarily to
the listed behavioral security indicators.

Important Note:
SHAP values describe model-attributed contributions
and should not be interpreted as causal proof.

Secure Aggregation:
Verified separately during federated training.
```

The actual values depend on the generated dataset and trained model.

---

# 40. Limitations of the Dataset and Model

Several limitations should be acknowledged when presenting the research.

### Synthetic Data

The dataset does not represent the full complexity of real-world AI-agent attacks.

### Small Neural Network

The model is intentionally lightweight and designed for experimental demonstration.

### Simulated Clients

The federated clients are simulated within one execution environment rather than operating as independently managed physical systems.

### Simplified Secure Aggregation

The cryptographic mechanism is intended for demonstrating the underlying concepts rather than replacing a production secure aggregation framework.

### SHAP Computational Cost

Kernel SHAP can be computationally expensive because it evaluates the model repeatedly.

### Threshold-Based Severity

The incident severity thresholds are manually configured and are not validated security standards.

---

# 41. Future Work

Possible extensions include:

* Real-world security datasets
* Larger and deeper neural networks
* Real distributed client environments
* Differential privacy
* Robust aggregation
* Byzantine-robust Federated Learning
* Model poisoning detection
* Backdoor detection
* Client authentication
* Secure key exchange
* Dropout-tolerant secure aggregation
* Formal cryptographic security analysis
* Privacy-preserving SHAP
* Secure inference
* Encrypted inference
* Homomorphic encryption
* Trusted execution environments
* Real-time security monitoring
* Streaming incident detection
* SIEM integration
* Automated SOC workflows
* More advanced incident severity models

---

# 42. Ethical and Security Disclaimer

This project is intended for:

* Academic research
* Educational experimentation
* Security research
* Federated-learning experimentation
* Explainable-AI research

It should not be used as a production security monitoring system without additional security analysis, validation, testing, and hardening.

The synthetic security indicators are not intended to represent a complete taxonomy of real-world attacks.

The secure aggregation implementation should not be interpreted as a formally verified cryptographic protocol.

---

# 43. Reproducibility Checklist

Before reporting experimental results, record:

```text
[ ] Python version
[ ] PyTorch version
[ ] SHAP version
[ ] NumPy version
[ ] Scikit-learn version
[ ] Operating system
[ ] CPU/GPU configuration
[ ] Random seed
[ ] Number of clients
[ ] Samples per client
[ ] Federated rounds
[ ] Local epochs
[ ] Learning rate
[ ] Fixed-point scale
[ ] Modulus q
```

This information improves experimental reproducibility.

---

# 44. Recommended Thesis Terminology

For academic writing, the following terminology is recommended.

Instead of:

> "The model uses ZQ finite field."

Use:

> **"Model updates are converted to fixed-point integers and represented using modular arithmetic over \(\mathbb{Z}_q\), with aggregation performed modulo a large prime \(q\)."**

You can also write:

> **"Because the selected modulus \(q\) is prime, the arithmetic can be interpreted over the finite field \(\mathbb{F}_q\)."**

For secure aggregation:

> **"The proposed prototype uses pairwise masking and additive secret sharing to protect individual client model updates during federated aggregation."**

For SHAP:

> **"SHAP is used to provide feature-level explanations of the model's compromise predictions."**

For the incident report:

> **"The incident reporting component combines the model's compromise probability, predefined severity thresholds, and SHAP-based feature attribution to generate a structured security report."**

---

# 45. Citation

If this repository is used in an academic thesis, conference paper, dissertation, or publication, cite the project according to the final publication requirements.

Example placeholder:

```bibtex
@software{rahman_ai_security_incident_reporting,
  author = {Rahman},
  title = {AI Security Incident Reporting for Compromised Agents},
  year = {2026},
  publisher = {GitHub},
  note = {Research prototype}
}
```

Replace the author, repository, DOI, and publication information with the final bibliographic information applicable to the project.

---

# 46. License

Add the license appropriate for your research project.

For example:

```text
MIT License
```

if you intend to distribute the implementation under the MIT License.

Otherwise, replace this section with the license required by your university, research group, or project.

---

# 47. Summary

This project demonstrates a complete experimental pipeline for AI security incident reporting:

```text
                 SECURITY DATA
                      |
                      v
             COMPROMISE DETECTION
                      |
                      v
              FEDERATED LEARNING
                      |
                      v
             SECURE AGGREGATION
             /                \
            /                  \
   Pairwise Masking       Secret Sharing
            \                  /
             \                /
              \              /
               v            v
              MODULAR AGGREGATION
                      |
                      v
                 GLOBAL MODEL
                      |
             +--------+--------+
             |                 |
             v                 v
           SHAP          INCIDENT REPORT
             |                 |
             v                 v
      FEATURE ATTRIBUTION   SEVERITY
                               |
                               v
                       SECURITY OUTPUT
```

The key conceptual contribution is the integration of:

$$
\boxed{
\text{Federated Learning}
+
\text{Secure Aggregation}
+
\text{Explainable AI}
+
\text{Incident Reporting}
}
$$

The project demonstrates that these components can be combined into a unified research prototype while maintaining a clear distinction between:

* **distributed model training,**
* **protection of client updates,**
* **model interpretability,** and
* **security incident reporting.**

---

## Final Note

This repository represents a **research prototype rather than a production security system**. Experimental results should be interpreted in the context of the synthetic dataset, simulated federated clients, simplified secure-aggregation protocol, selected model architecture, and configured experimental parameters.
