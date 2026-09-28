# FEDERATED XAI SECURE AI-AGENT INCIDENT REPORTING PROTOTYPE
Research prototype for privacy-preserving and explainable AI-agent security incident reporting integrated with Federated Learning along with Additive homomorphic secret-sharing combined with pairwise masking. 

Author: Nimra Rahman
Paper connection: Beyond Predictable Paths: Redefining AI Security Incident Reporting for Agents
arXiv:2609.24515

FILES
-----
agent_incident_reporting_demo.py  -> complete prototype
agent_incident_dashboard.html      -> interactive professor-facing demo
agent_incident_results.json        -> machine-readable experiment output

RUN
---
1. Install Python 3.9+ and numpy:
   python3 -m pip install numpy

2. Run:
   python3 agent_incident_reporting_demo.py

3. The script prints the experiment and creates/opens:
   agent_incident_dashboard.html

WHAT THE DEMO SHOWS
-------------------
- Table-1-aligned telemetry for all 17 listed reporting elements, represented in simplified form.
- Tamper-evident local evidence using hash chaining plus an external anchor.
- Federated learning across five organisations with different incident mixes.
- Federated analytics without centralising raw trajectories.
- Pairwise masking combined with additive secret sharing over a modular ring.
- Integrated Gradients + KernelSHAP-style + LIME explanations.
- Replay validation of the explanation.
- Public / peer / forensic disclosure views.
- Thresholded/noisy public analytics.

IMPORTANT RESEARCH LIMITATIONS
------------------------------
This is a synthetic proof-of-concept, not a production security system.
The secure-aggregation construction uses simulated pairwise seeds rather than real key exchange/CSPRNGs.
It assumes no client dropout, malicious aggregator, collusion, or Byzantine clients.
The secret-sharing layer is additive/linear homomorphic in the algebraic sense; it is NOT Paillier or another
public-key additively homomorphic encryption scheme.
The XAI implementations are dependency-light research approximations (KernelSHAP-style and LIME), not claims
of equivalence to the full SHAP/LIME libraries.
The Table-1 fields are simplified synthetic representations; complete semantic capture is future work.

The purpose is to make the research question concrete: can organisations retain richer incident evidence
locally while sharing enough protected signal to study cross-organisation generalisation and incident patterns?
