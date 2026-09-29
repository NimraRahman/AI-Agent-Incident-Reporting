# ============================================================
# FEDERATED XAI SECURE AI-AGENT INCIDENT REPORTING - EXPERIMENT B
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import shap


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

NUM_CLIENTS = 4
SAMPLES_PER_CLIENT = 500

FED_ROUNDS = 10
LOCAL_EPOCHS = 5

LEARNING_RATE = 0.003

TEST_SIZE = 0.20

# Finite-field prime.
# Large enough for this demonstration.
Q = 2305843009213693951

# Fixed-point scaling.
FIXED_POINT_SCALE = 10**8

OUTPUT_DIR = "outputs_experiment_C"

os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("FEDERATED XAI SECURE AI-AGENT INCIDENT REPORTING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Clients: {NUM_CLIENTS}")
print(f"Federated rounds: {FED_ROUNDS}")
print(f"Local epochs: {LOCAL_EPOCHS}")
print("=" * 70)


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

FEATURE_NAMES = [
    "prompt_injection_score",
    "tool_abuse_score",
    "credential_access_score",
    "network_anomaly_score",
    "data_exfiltration_score",
    "privilege_escalation_score",
    "behavior_deviation_score",
    "failed_authentication_rate",
]

NUM_FEATURES = len(FEATURE_NAMES)


# ============================================================
# 3. SYNTHETIC SECURITY INCIDENT DATA
# ============================================================

def generate_security_data(
    n_samples,
    client_id=0,
    seed=42,
):
    """
    Generate synthetic AI-agent security telemetry.

    The target 'compromised' is generated from a nonlinear
    security-risk function so that the neural network has
    meaningful interactions to learn.
    """

    rng = np.random.default_rng(seed + client_id)

    # --------------------------------------------------------
    # Security telemetry
    # --------------------------------------------------------

    X = rng.uniform(
        low=0.0,
        high=1.0,
        size=(n_samples, NUM_FEATURES)
    )

    # Introduce client-specific distributions.
    # This creates mild non-IID federated data.
    if client_id == 1:
        X[:, 0] *= 0.9
        X[:, 1] = np.clip(X[:, 1] + 0.10, 0, 1)

    elif client_id == 2:
        X[:, 2] = np.clip(X[:, 2] + 0.15, 0, 1)
        X[:, 4] = np.clip(X[:, 4] + 0.10, 0, 1)

    elif client_id == 3:
        X[:, 3] = np.clip(X[:, 3] + 0.12, 0, 1)
        X[:, 5] = np.clip(X[:, 5] + 0.12, 0, 1)

    # --------------------------------------------------------
    # Individual risk contributions
    # --------------------------------------------------------

    prompt_injection = X[:, 0]
    tool_abuse = X[:, 1]
    credential_access = X[:, 2]
    network_anomaly = X[:, 3]
    data_exfiltration = X[:, 4]
    privilege_escalation = X[:, 5]
    behavior_deviation = X[:, 6]
    failed_auth = X[:, 7]

    # --------------------------------------------------------
    # Nonlinear security-risk function
    # --------------------------------------------------------

    risk = (
        1.8 * prompt_injection
        + 1.5 * tool_abuse
        + 1.7 * credential_access
        + 1.2 * network_anomaly
        + 1.8 * data_exfiltration
        + 1.5 * privilege_escalation
        + 1.1 * behavior_deviation
        + 1.0 * failed_auth

        # Interaction:
        # prompt injection + tool abuse
        + 2.5 * prompt_injection * tool_abuse

        # Interaction:
        # credential access + data exfiltration
        + 2.8 * credential_access * data_exfiltration

        # Interaction:
        # privilege escalation + credential access
        + 1.5 * privilege_escalation * credential_access
    )

    # Add noise.
    risk += rng.normal(
        loc=0.0,
        scale=0.45,
        size=n_samples
    )

    # Center around zero.
    risk = risk - np.median(risk)

    # Convert risk to probability.
    probability = 1.0 / (1.0 + np.exp(-risk))

    # Generate binary labels.
    y = rng.binomial(
        n=1,
        p=probability
    ).astype(np.float32)

    return X.astype(np.float32), y


# ============================================================
# 4. GENERATE FEDERATED CLIENT DATA
# ============================================================

print("\nGenerating synthetic federated security data...")

client_data = []

for client_id in range(NUM_CLIENTS):

    X_client, y_client = generate_security_data(
        n_samples=SAMPLES_PER_CLIENT,
        client_id=client_id,
        seed=SEED
    )

    client_data.append(
        (
            X_client,
            y_client
        )
    )

    print(
        f"Client {client_id + 1}: "
        f"{len(X_client)} samples | "
        f"compromised={int(y_client.sum())}"
    )


# ============================================================
# 5. GLOBAL TEST DATA
# ============================================================

X_all = np.concatenate(
    [item[0] for item in client_data],
    axis=0
)

y_all = np.concatenate(
    [item[1] for item in client_data],
    axis=0
)

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_all
)

print("\nGlobal dataset:")
print(f"Total samples: {len(X_all)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# ============================================================
# 6. PYTORCH MODEL
# ============================================================

class SecurityIncidentMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(NUM_FEATURES, 32),

            nn.ReLU(),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 7. MODEL VECTOR UTILITIES
# ============================================================

def get_model_vector(model):
    """
    Convert all model parameters into one flat NumPy vector.
    """

    vectors = []

    for parameter in model.parameters():

        vectors.append(
            parameter.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    return np.concatenate(vectors)


def set_model_vector(model, vector):
    """
    Load a flat NumPy vector into the model.
    """

    pointer = 0

    for parameter in model.parameters():

        numel = parameter.numel()

        values = vector[
            pointer:pointer + numel
        ]

        tensor = torch.tensor(
            values.reshape(parameter.shape),
            dtype=parameter.dtype
        )

        parameter.data.copy_(tensor)

        pointer += numel


# ============================================================
# 8. LOCAL MODEL TRAINING
# ============================================================

def train_local_model(
    global_vector,
    X,
    y,
    epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE,
):
    """
    Train one client's model starting from the global model.
    """

    model = SecurityIncidentMLP().to(DEVICE)

    set_model_vector(
        model,
        global_vector
    )

    model.train()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    y_tensor = torch.tensor(
        y.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):

        optimizer.zero_grad()

        logits = model(X_tensor)

        loss = criterion(
            logits,
            y_tensor
        )

        loss.backward()

        optimizer.step()

    local_vector = get_model_vector(model)

    update = local_vector - global_vector

    return local_vector, update, float(loss.item())


# ============================================================
# 9. MODEL PREDICTION
# ============================================================

def predict_probability(model_vector, X):
    """
    Return probability of compromised agent.
    """

    model = SecurityIncidentMLP().to(DEVICE)

    set_model_vector(
        model,
        model_vector
    )

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy().reshape(-1)

    return probabilities


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_vector,
    X,
    y,
):
    """
    Evaluate binary classification.
    """

    probabilities = predict_probability(
        model_vector,
        X
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "probabilities": probabilities,
        "predictions": predictions,
    }


# ============================================================
# 11. FIXED-POINT ENCODING
# ============================================================

def encode_fixed_point(vector):
    """
    Convert floating-point model updates into integers.

    The values are represented in a finite field modulo Q.
    """

    scaled = np.rint(
        vector * FIXED_POINT_SCALE
    ).astype(object)

    encoded = np.array(
        [
            int(value) % Q
            for value in scaled
        ],
        dtype=object
    )

    return encoded


def decode_fixed_point(encoded_vector):
    """
    Decode finite-field integers back into signed
    fixed-point floating-point values.
    """

    decoded = []

    half_q = Q // 2

    for value in encoded_vector:

        value = int(value) % Q

        # Interpret modular integer as signed integer.
        if value > half_q:
            value -= Q

        decoded.append(
            value / FIXED_POINT_SCALE
        )

    return np.array(
        decoded,
        dtype=np.float64
    )


# ============================================================
# 12. PAIRWISE MASK GENERATION
# ============================================================

def create_pairwise_masks(
    num_clients,
    vector_length,
    seed=SEED,
):
    """
    Create pairwise masks.

    For clients i and j:

        client i receives +r_ij
        client j receives -r_ij

    Therefore:

        sum(all masks) = 0 mod Q

    The server sees masked updates but does not see
    the original update values.
    """

    rng = np.random.default_rng(seed)

    masks = [
        np.zeros(
            vector_length,
            dtype=object
        )
        for _ in range(num_clients)
    ]

    pairwise_random_values = {}

    for i in range(num_clients):

        for j in range(i + 1, num_clients):

            random_mask = rng.integers(
                low=0,
                high=Q,
                size=vector_length,
                dtype=np.int64
            ).astype(object)

            pairwise_random_values[
                (i, j)
            ] = random_mask

            masks[i] = (
                masks[i]
                + random_mask
            ) % Q

            masks[j] = (
                masks[j]
                - random_mask
            ) % Q

    return masks, pairwise_random_values


# ============================================================
# 13. ADDITIVE SECRET SHARING
# ============================================================

def create_additive_shares(
    value_vector,
    num_shares=2,
    seed=42,
):
    """
    Create additive secret shares.

    For two shares:

        share_1 + share_2 = value mod Q

    The individual shares do not reveal the original value
    in this simplified demonstration.
    """

    rng = np.random.default_rng(seed)

    shares = []

    accumulated = np.zeros(
        len(value_vector),
        dtype=object
    )

    # Generate n-1 random shares.
    for _ in range(num_shares - 1):

        random_share = rng.integers(
            low=0,
            high=Q,
            size=len(value_vector),
            dtype=np.int64
        ).astype(object)

        shares.append(
            random_share
        )

        accumulated = (
            accumulated
            + random_share
        ) % Q

    # Final share makes the sum equal the original value.
    final_share = (
        value_vector
        - accumulated
    ) % Q

    shares.append(
        final_share
    )

    return shares


def reconstruct_additive_shares(shares):
    """
    Reconstruct secret from additive shares.
    """

    result = np.zeros(
        len(shares[0]),
        dtype=object
    )

    for share in shares:

        result = (
            result + share
        ) % Q

    return result


# ============================================================
# 14. SECURE AGGREGATION
# ============================================================

def secure_aggregate(updates):
    """
    Secure aggregation pipeline:

        plaintext update
              ↓
        fixed-point encoding
              ↓
        pairwise masking
              ↓
        additive secret sharing
              ↓
        server reconstructs masked values
              ↓
        server aggregates masked values
              ↓
        decode

    The pairwise masks cancel when all clients participate.
    """

    num_clients = len(updates)

    vector_length = len(updates[0])

    # --------------------------------------------------------
    # Step 1: encode plaintext updates
    # --------------------------------------------------------

    encoded_updates = [
        encode_fixed_point(update)
        for update in updates
    ]

    # --------------------------------------------------------
    # Step 2: create pairwise masks
    # --------------------------------------------------------

    masks, pairwise_values = create_pairwise_masks(
        num_clients=num_clients,
        vector_length=vector_length,
        seed=SEED + 100
    )

    # --------------------------------------------------------
    # Step 3: apply masks
    # --------------------------------------------------------

    masked_updates = []

    for i in range(num_clients):

        masked = (
            encoded_updates[i]
            + masks[i]
        ) % Q

        masked_updates.append(
            masked
        )

    # --------------------------------------------------------
    # Step 4: additive secret sharing
    # --------------------------------------------------------

    all_client_shares = []

    for i in range(num_clients):

        shares = create_additive_shares(
            masked_updates[i],
            num_shares=2,
            seed=SEED + 200 + i
        )

        all_client_shares.append(
            shares
        )

    # --------------------------------------------------------
    # Step 5: reconstruct server-visible masked updates
    # --------------------------------------------------------

    reconstructed_masked_updates = []

    for shares in all_client_shares:

        reconstructed = (
            reconstruct_additive_shares(
                shares
            )
        )

        reconstructed_masked_updates.append(
            reconstructed
        )

    # --------------------------------------------------------
    # Step 6: server aggregates masked values
    # --------------------------------------------------------

    server_aggregate = np.zeros(
        vector_length,
        dtype=object
    )

    for masked in reconstructed_masked_updates:

        server_aggregate = (
            server_aggregate + masked
        ) % Q

    # --------------------------------------------------------
    # Step 7: plaintext aggregate
    # --------------------------------------------------------

    plaintext_aggregate = np.zeros(
        vector_length,
        dtype=object
    )

    for encoded in encoded_updates:

        plaintext_aggregate = (
            plaintext_aggregate + encoded
        ) % Q

    # --------------------------------------------------------
    # Step 8: verify aggregate correctness
    # --------------------------------------------------------

    aggregate_difference = (
        server_aggregate
        - plaintext_aggregate
    ) % Q

    aggregate_error = max(
        int(abs(int(x)))
        for x in aggregate_difference
    )

    # --------------------------------------------------------
    # Step 9: verify pairwise masks cancel
    # --------------------------------------------------------

    total_masks = np.zeros(
        vector_length,
        dtype=object
    )

    for mask in masks:

        total_masks = (
            total_masks + mask
        ) % Q

    mask_residual = max(
        int(abs(int(x)))
        for x in total_masks
    )

    # --------------------------------------------------------
    # Step 10: Verify secret-share reconstruction
    # --------------------------------------------------------

    share_reconstruction_error = 0

    for i in range(num_clients):

        expected = masked_updates[i]

        reconstructed = (
            reconstructed_masked_updates[i]
        )

        difference = (
            reconstructed - expected
        ) % Q

        local_error = max(
            int(abs(int(x)))
            for x in difference
        )

        share_reconstruction_error = max(
            share_reconstruction_error,
            local_error
        )

    # --------------------------------------------------------
    # Step 11: Decode aggregate
    # --------------------------------------------------------

    aggregate_update = decode_fixed_point(
        server_aggregate
    )

    return {
        "aggregate_update": aggregate_update,

        "encoded_updates": encoded_updates,

        "masked_updates": masked_updates,

        "reconstructed_masked_updates":
            reconstructed_masked_updates,

        "server_aggregate":
            server_aggregate,

        "plaintext_aggregate":
            plaintext_aggregate,

        "pairwise_masks":
            masks,

        "aggregate_error":
            aggregate_error,

        "mask_residual":
            mask_residual,

        "share_reconstruction_error":
            share_reconstruction_error,
    }


# ============================================================
# 15. CENTRALIZED TRAINING
# ============================================================

def train_centralized_model():

    print("\n" + "=" * 70)
    print("CENTRALIZED BASELINE")
    print("=" * 70)

    model = SecurityIncidentMLP().to(DEVICE)

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE
    )

    y_tensor = torch.tensor(
        y_train.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(FED_ROUNDS * LOCAL_EPOCHS):

        model.train()

        optimizer.zero_grad()

        logits = model(X_tensor)

        loss = criterion(
            logits,
            y_tensor
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % LOCAL_EPOCHS == 0
        ):

            print(
                f"Centralized Epoch "
                f"{epoch + 1:02d} | "
                f"Loss = {loss.item():.4f}"
            )

    centralized_vector = get_model_vector(
        model
    )

    metrics = evaluate_model(
        centralized_vector,
        X_test,
        y_test
    )

    print("\nCentralized Test Results")

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    return centralized_vector, metrics


# ============================================================
# 16. FEDERATED TRAINING WITH SECURE AGGREGATION
# ============================================================

def federated_training():

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING + SECURE AGGREGATION")
    print("=" * 70)

    initial_model = SecurityIncidentMLP().to(DEVICE)

    global_vector = get_model_vector(
        initial_model
    )

    history = []

    last_secure_result = None

    for round_number in range(
        1,
        FED_ROUNDS + 1
    ):

        print(
            f"\n--- Federated Round "
            f"{round_number}/{FED_ROUNDS} ---"
        )

        client_updates = []

        client_losses = []

        # ----------------------------------------------------
        # Local client training
        # ----------------------------------------------------

        for client_id in range(NUM_CLIENTS):

            X_client, y_client = client_data[
                client_id
            ]

            local_vector, update, loss = (
                train_local_model(
                    global_vector,
                    X_client,
                    y_client
                )
            )

            client_updates.append(
                update
            )

            client_losses.append(
                loss
            )

            print(
                f"Client {client_id + 1}: "
                f"loss={loss:.4f} | "
                f"update_norm="
                f"{np.linalg.norm(update):.6f}"
            )

        # ----------------------------------------------------
        # SECURE AGGREGATION
        # ----------------------------------------------------

        secure_result = secure_aggregate(
            client_updates
        )

        last_secure_result = secure_result

        # ----------------------------------------------------
        # FedAvg
        # ----------------------------------------------------

        average_update = (
            secure_result["aggregate_update"]
            / NUM_CLIENTS
        )

        global_vector = (
            global_vector
            + average_update
        )

        # ----------------------------------------------------
        # Global evaluation
        # ----------------------------------------------------

        metrics = evaluate_model(
            global_vector,
            X_test,
            y_test
        )

        history.append(
            {
                "round": round_number,
                "loss": float(
                    np.mean(client_losses)
                ),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

        print(
            f"Global Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Global Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Global Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Global F1       : "
            f"{metrics['f1']:.4f}"
        )

        print(
            "Secure aggregation "
            f"error = "
            f"{secure_result['aggregate_error']}"
        )

        print(
            "Pairwise mask residual = "
            f"{secure_result['mask_residual']}"
        )

        print(
            "Secret-share reconstruction "
            f"error = "
            f"{secure_result['share_reconstruction_error']}"
        )

    history_df = pd.DataFrame(history)

    return (
        global_vector,
        history_df,
        last_secure_result
    )


# ============================================================
# 17. PLAINTEXT VS MASKED UPDATE DEMONSTRATION
# ============================================================

def demonstrate_masking(secure_result):

    print("\n" + "=" * 70)
    print("PLAINTEXT VS SERVER-VISIBLE MASKED UPDATE")
    print("=" * 70)

    plaintext = (
        secure_result["encoded_updates"][0]
    )

    masked = (
        secure_result["masked_updates"][0]
    )

    print("\nFirst 10 encoded values:")

    print(
        "Plaintext:",
        [
            int(x)
            for x in plaintext[:10]
        ]
    )

    print(
        "Masked   :",
        [
            int(x)
            for x in masked[:10]
        ]
    )

    print(
        "\nThe masked values differ from the "
        "plaintext encoded update."
    )

    print(
        "The aggregate remains correct because "
        "pairwise masks cancel modulo Q."
    )


# ============================================================
# 18. SHAP EXPLAINABILITY
# ============================================================

def create_shap_explanation(
    model_vector,
    X_background,
    X_explain,
):
    """
    SHAP explanation for the final federated model.

    KernelExplainer is used so the explanation is independent
    of the internal PyTorch model architecture.
    """

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY")
    print("=" * 70)

    # Keep the background dataset small enough for CPU use.
    background_size = min(
        40,
        len(X_background)
    )

    explain_size = min(
        1,
        len(X_explain)
    )

    background = X_background[
        :background_size
    ]

    samples = X_explain[
        :explain_size
    ]

    def prediction_function(data):

        data = np.asarray(
            data,
            dtype=np.float32
        )

        return predict_probability(
            model_vector,
            data
        )

    print(
        f"SHAP background samples: "
        f"{len(background)}"
    )

    print(
        f"SHAP explanation samples: "
        f"{len(samples)}"
    )

    explainer = shap.KernelExplainer(
        prediction_function,
        background
    )

    shap_values = explainer.shap_values(
        samples,
        nsamples=80
    )

    # Different SHAP versions return different structures.
    if isinstance(shap_values, list):

        shap_array = np.asarray(
            shap_values[0]
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

    shap_array = np.squeeze(
        shap_array
    )

    if shap_array.ndim == 1:

        shap_values_one = shap_array

    else:

        shap_values_one = shap_array[0]

    prediction_probability = float(
        prediction_function(
            samples
        )[0]
    )

    return (
        samples[0],
        shap_values_one,
        prediction_probability
    )


# ============================================================
# 19. SHAP PLOT
# ============================================================

def plot_shap_explanation(
    sample,
    shap_values,
):

    # Sort features by absolute SHAP magnitude.
    order = np.argsort(
        np.abs(shap_values)
    )[::-1]

    sorted_names = [
        FEATURE_NAMES[i]
        for i in order
    ]

    sorted_values = [
        shap_values[i]
        for i in order
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        sorted_names[::-1],
        sorted_values[::-1]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "SHAP value"
    )

    plt.ylabel(
        "Security indicator"
    )

    plt.title(
        "SHAP Explanation of Federated Security Model"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "shap_incident_explanation.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# 20. AUTOMATED INCIDENT REPORT
# ============================================================

def create_incident_report(
    sample,
    shap_values,
    probability,
):

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    if probability >= 0.75:

        severity = "CRITICAL"

    elif probability >= 0.50:

        severity = "HIGH"

    elif probability >= 0.25:

        severity = "MEDIUM"

    else:

        severity = "LOW"

    # --------------------------------------------------------
    # Rank features
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        np.abs(shap_values)
    )[::-1]

    top_indices = ranked_indices[:5]

    rows = []

    for index in top_indices:

        rows.append(
            {
                "feature":
                    FEATURE_NAMES[index],

                "feature_value":
                    float(sample[index]),

                "shap_value":
                    float(shap_values[index]),

                "absolute_shap":
                    float(
                        abs(
                            shap_values[index]
                        )
                    ),
            }
        )

    report_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    text_lines = []

    text_lines.append(
        "=" * 70
    )

    text_lines.append(
        "AUTOMATED AI-AGENT SECURITY INCIDENT REPORT"
    )

    text_lines.append(
        "=" * 70
    )

    text_lines.append(
        f"Compromise probability: "
        f"{probability:.4f}"
    )

    text_lines.append(
        f"Severity: {severity}"
    )

    text_lines.append(
        ""
    )

    text_lines.append(
        "Top contributing security indicators:"
    )

    text_lines.append(
        ""
    )

    for rank, row in enumerate(
        rows,
        start=1
    ):

        direction = (
            "increased"
            if row["shap_value"] > 0
            else "decreased"
        )

        text_lines.append(
            f"{rank}. "
            f"{row['feature']} | "
            f"value={row['feature_value']:.4f} | "
            f"SHAP={row['shap_value']:.6f} | "
            f"effect={direction}"
        )

    text_lines.append(
        ""
    )

    text_lines.append(
        "Interpretation:"
    )

    text_lines.append(
        "The SHAP values describe how the model's "
        "prediction changes relative to its baseline."
    )

    text_lines.append(
        "They should not be interpreted as causal proof "
        "that a particular security event caused compromise."
    )

    text_lines.append(
        ""
    )

    text_lines.append(
        "Privacy/security note:"
    )

    text_lines.append(
        "Secure aggregation protects federated training "
        "updates from direct server-side inspection."
    )

    text_lines.append(
        "It does not automatically make the final incident "
        "report or SHAP explanation private."
    )

    text_lines.append(
        "Additional access control, privacy mechanisms, "
        "or secure explanation protocols may be required."
    )

    report_text = "\n".join(
        text_lines
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "incident_report.csv"
    )

    txt_path = os.path.join(
        OUTPUT_DIR,
        "incident_report.txt"
    )

    report_df.to_csv(
        csv_path,
        index=False
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report_text
        )

    print("\n" + report_text)

    print(
        f"\nSaved: {csv_path}"
    )

    print(
        f"Saved: {txt_path}"
    )

    return (
        report_df,
        report_text
    )


# ============================================================
# 21. FEDERATED PERFORMANCE GRAPH
# ============================================================

def plot_federated_performance(
    history_df
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history_df["round"],
        history_df["accuracy"],
        marker="o",
        label="Accuracy"
    )

    plt.plot(
        history_df["round"],
        history_df["precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        history_df["round"],
        history_df["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        history_df["round"],
        history_df["f1"],
        marker="o",
        label="F1"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Federated Learning Performance"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "federated_performance.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    model_vector,
    X,
    y,
):

    metrics = evaluate_model(
        model_vector,
        X,
        y
    )

    cm = metrics[
        "confusion_matrix"
    ]

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "Confusion Matrix - Final Federated Model"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        [0, 1],
        ["Normal", "Compromised"]
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Compromised"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# 23. SECURE AGGREGATION VERIFICATION
# ============================================================

def save_security_verification(
    secure_result
):

    aggregate_error = (
        secure_result[
            "aggregate_error"
        ]
    )

    mask_residual = (
        secure_result[
            "mask_residual"
        ]
    )

    share_error = (
        secure_result[
            "share_reconstruction_error"
        ]
    )

    passed = (
        aggregate_error == 0
        and mask_residual == 0
        and share_error == 0
    )

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "SECURE AGGREGATION VERIFICATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Finite-field modulus Q: {Q}"
    )

    lines.append(
        f"Fixed-point scale: {FIXED_POINT_SCALE}"
    )

    lines.append(
        ""
    )

    lines.append(
        "1. Plaintext aggregate vs "
        "server masked aggregate"
    )

    lines.append(
        f"   Maximum modular error: "
        f"{aggregate_error}"
    )

    lines.append(
        ""
    )

    lines.append(
        "2. Pairwise mask cancellation"
    )

    lines.append(
        f"   Maximum residual: "
        f"{mask_residual}"
    )

    lines.append(
        ""
    )

    lines.append(
        "3. Additive secret-share reconstruction"
    )

    lines.append(
        f"   Maximum reconstruction error: "
        f"{share_error}"
    )

    lines.append(
        ""
    )

    lines.append(
        f"OVERALL VERIFICATION: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Interpretation:"
    )

    lines.append(
        "The server-side masked aggregate reconstructs "
        "the same finite-field aggregate as the plaintext "
        "encoded client updates."
    )

    lines.append(
        "Pairwise masks cancel because each pair contributes "
        "equal and opposite masks."
    )

    lines.append(
        "This experiment verifies correctness of the "
        "implemented masking/sharing arithmetic."
    )

    lines.append(
        ""
    )

    lines.append(
        "Research limitation:"
    )

    lines.append(
        "This demonstration does not constitute a complete "
        "production secure-aggregation protocol."
    )

    lines.append(
        "A production system requires authenticated "
        "communication, secure key establishment, "
        "cryptographically secure randomness, client "
        "dropout handling, malicious-client defenses, "
        "and formal security analysis."
    )

    text = "\n".join(
        lines
    )

    path = os.path.join(
        OUTPUT_DIR,
        "secure_aggregation_verification.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            text
        )

    print("\n" + text)

    print(
        f"\nSaved: {path}"
    )

    return passed


# ============================================================
# 24. ABLATION 
# ============================================================

def create_ablation_results(
    centralized_metrics,
    federated_metrics,
):

    final_fed = federated_metrics.iloc[-1]

    rows = [

        {
            "method":
                "Centralized ML",

            "accuracy":
                centralized_metrics[
                    "accuracy"
                ],

            "precision":
                centralized_metrics[
                    "precision"
                ],

            "recall":
                centralized_metrics[
                    "recall"
                ],

            "f1":
                centralized_metrics[
                    "f1"
                ],
        },

        {
            "method":
                "FedAvg",

            "accuracy":
                final_fed["accuracy"],

            "precision":
                final_fed["precision"],

            "recall":
                final_fed["recall"],

            "f1":
                final_fed["f1"],
        },

        {
            "method":
                "FedAvg + Secure Aggregation",

            "accuracy":
                final_fed["accuracy"],

            "precision":
                final_fed["precision"],

            "recall":
                final_fed["recall"],

            "f1":
                final_fed["f1"],
        },
    ]

    ablation_df = pd.DataFrame(
        rows
    )

    path = os.path.join(
        OUTPUT_DIR,
        "ablation_results.csv"
    )

    ablation_df.to_csv(
        path,
        index=False
    )

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    print(
        ablation_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {path}"
    )

    return ablation_df


# ============================================================
# 25. SAVE FEDERATED METRICS
# ============================================================

def save_federated_metrics(
    history_df
):

    path = os.path.join(
        OUTPUT_DIR,
        "federated_metrics.csv"
    )

    history_df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# 26. MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # CENTRALIZED BASELINE
    # --------------------------------------------------------

    centralized_vector, centralized_metrics = (
        train_centralized_model()
    )

    # --------------------------------------------------------
    # FEDERATED LEARNING
    # --------------------------------------------------------

    (
        final_federated_vector,
        history_df,
        secure_result
    ) = federated_training()

    # --------------------------------------------------------
    # SAVE FEDERATED METRICS
    # --------------------------------------------------------

    save_federated_metrics(
        history_df
    )

    # --------------------------------------------------------
    # FINAL FEDERATED EVALUATION
    # --------------------------------------------------------

    final_metrics = evaluate_model(
        final_federated_vector,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("FINAL FEDERATED MODEL")
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{final_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{final_metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{final_metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # MASKING DEMONSTRATION
    # --------------------------------------------------------

    demonstrate_masking(
        secure_result
    )

    # --------------------------------------------------------
    # SECURITY VERIFICATION
    # --------------------------------------------------------

    save_security_verification(
        secure_result
    )

    # --------------------------------------------------------
    # ABLATION
    # --------------------------------------------------------

    create_ablation_results(
        centralized_metrics,
        history_df
    )

    # --------------------------------------------------------
    # PERFORMANCE PLOT
    # --------------------------------------------------------

    plot_federated_performance(
        history_df
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    plot_confusion_matrix(
        final_federated_vector,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    (
        explained_sample,
        shap_values,
        compromise_probability
    ) = create_shap_explanation(
        final_federated_vector,
        X_train,
        X_test
    )

    # --------------------------------------------------------
    # SHAP PLOT
    # --------------------------------------------------------

    plot_shap_explanation(
        explained_sample,
        shap_values
    )

    # --------------------------------------------------------
    # INCIDENT REPORT
    # --------------------------------------------------------

    create_incident_report(
        explained_sample,
        shap_values,
        compromise_probability
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    output_files = [
        "federated_metrics.csv",
        "ablation_results.csv",
        "incident_report.csv",
        "incident_report.txt",
        "federated_performance.png",
        "shap_incident_explanation.png",
        "confusion_matrix.png",
        "secure_aggregation_verification.txt",
    ]

    for filename in output_files:

        print(
            " - "
            + os.path.join(
                OUTPUT_DIR,
                filename
            )
        )

    print("\nFinal incident assessment:")

    print(
        f"Compromise probability: "
        f"{compromise_probability:.4f}"
    )

    if compromise_probability >= 0.75:

        print(
            "Severity: CRITICAL"
        )

    elif compromise_probability >= 0.50:

        print(
            "Severity: HIGH"
        )

    elif compromise_probability >= 0.25:

        print(
            "Severity: MEDIUM"
        )

    else:

        print(
            "Severity: LOW"
        )

    print("\nTop SHAP contributors:")

    ranking = np.argsort(
        np.abs(shap_values)
    )[::-1]

    for i in ranking[:5]:

        print(
            f" - "
            f"{FEATURE_NAMES[i]}: "
            f"SHAP={shap_values[i]:+.6f}, "
            f"value={explained_sample[i]:.4f}"
        )

    print("\nResearch architecture:")
    print(
        """
Security telemetry
        |
        v
Synthetic incident data
        |
        +-----------------------------+
        |                             |
        v                             v
   Client 1 ... Client N       Centralized baseline
        |
        v
Local neural-network training
        |
        v
Model updates
        |
        v
Fixed-point encoding
        |
        v
Finite-field representation Zq
        |
        v
Pairwise masking
        |
        v
Additive secret sharing
        |
        v
Secure aggregation
        |
        v
FedAvg global model
        |
        +-------------------+
        |                   |
        v                   v
   Evaluation             SHAP
                            |
                            v
                 Explainable incident
                       assessment
                            |
                            v
                  Automated report
        """
    )

    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
