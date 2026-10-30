# Privacy-Preserving-Intrusion-Detection-in-Blockchain-Based-Vehicular-Networks
A Trusted Data Sharing based Proximity Evaluation and Privacy-Preserving Intrusion Detection for Blockchain-Enabled Software-Defined Vehicular Networks
# VANET Security Framework: TVGQNN, Federated Learning, Trust Management and Blockchain

A modular Python/Solidity research implementation for security monitoring in Vehicular Ad Hoc Networks (VANETs). The pipeline combines vehicle authentication, geospatial-temporal graph construction, network-feature analysis, graph-based attack detection, federated learning, trust estimation, and blockchain event recording.

> **Important:** This repository contains research code. Several components are simulations or offline analyses rather than a fully deployed VANET/SDN/PUF environment. Validate all metrics on your own dataset and deployment before making performance or security claims.

## Main features

- **ECC-based vehicle identity:** P-256 key-pair generation and ECDSA challenge-response verification.
- **PUF-assisted check:** a deterministic hash-based PUF *simulation* used by the authentication workflow; it is not a physical PUF implementation.
- **Zero-trust decision gate:** allow, quarantine, or reject decisions based on identity and authentication fields.
- **Geospatial-temporal graph construction:** hierarchical geohash regions and NetworkX snapshots built from vehicle positions and communication range.
- **SMOTE augmentation:** optional balancing of numeric telemetry features.
- **SDN/network analysis:** network statistics and feature aggregation; the Ryu application is exported as a template and is not automatically deployed.
- **Temporal graph attack detection:** TVGQNN model with a variational latent representation and optional PennyLane quantum-circuit head, plus a TGN baseline.
- **Federated learning:** local graph-model updates with update filtering and similarity-weighted aggregation.
- **Trust management:** Dempster-Shafer evidence fusion and temporal, neighbor-aware trust updates.
- **Blockchain event ledger:** Solidity contract for proposing, attesting to, and committing security events.

## Repository structure

```text
git/
├── Readme.md
├── run_all.py
├── step0_config.py
├── step1_registration_auth.py
├── step2_geospatial_graph.py
├── step3_sdn_controller.py
├── step4_smote.py
├── step4_tvgqnn.py
├── step4_tgn_baseline.py
├── step5_federated.py
├── step6_trust.py
├── step7_blockchain.py
├── deploy_contract.py
└── SecurityLedger.sol
```

The scripts create `outputs/` and `models/` when they run. These directories contain generated results and model files.

## Requirements
requirements.txt
Recommended environment: **Python 3.10 **. Install the packages used by the scripts:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CPU/GPU and operating system from the official PyTorch installation selector. PyTorch Geometric compatibility depends on the installed PyTorch version.

The blockchain deployment also requires a local Ethereum-compatible JSON-RPC node, for example Ganache, and a Solidity compiler version compatible with `pragma solidity ^0.8.20`.

## Dataset configuration

1. Place your VANET dataset in an accessible location.
2. Open `step0_config.py`.
3. Change `DATA_PATH` to the full path of your CSV file.

For example:

```python
DATA_PATH = r"C:\data\vanet_security_dataset.csv"
```

The current loader expects fields including:

- `event_id`
- `vehicle_id`
- `simulation_time`
- `attack_class_label`

Other feature and evidence columns are used when available. The six attack labels configured in `step0_config.py` are:

| Label | Class |
|---:|---|
| 0 | Normal |
| 1 | Sybil |
| 2 | FalsePosition |
| 3 | FalseInformation |
| 4 | PacketDropping |
| 5 | Impersonation |

Ensure `attack_class_label` contains integer values from `0` to `5`. Review `get_feature_cols()` and the leakage-column list in `step0_config.py` before training. In particular, confirm that no target-derived or post-event fields are accidentally used as model inputs.

## Running the pipeline

Run individual steps from the repository directory:

```bash
python step1_registration_auth.py
python step4_smote.py
python step2_geospatial_graph.py
python step3_sdn_controller.py
python step4_tvgqnn.py
python step4_tgn_baseline.py
python step5_federated.py
python step6_trust.py
```

The blockchain step requires a running node and a deployed contract; see the next section.

`run_all.py` is intended to execute the complete workflow:

```bash
python run_all.py
```

SMOTE is enabled by default in `run_all.py`. To disable it in Windows PowerShell:

```powershell
$env:USE_SMOTE="0"
python run_all.py
```

> **Repository completeness note:** the supplied archive's `run_all.py` imports `final_metrics`, but `final_metrics.py` is not included in the archive. Therefore, the complete `run_all.py` command will stop at import time until that module is added or the import/call is removed. Run the individual steps above in dependency order in the meantime.

## Blockchain setup (Step 7)

Step 7 does not run as a standalone simulation. It requires a live local blockchain node, a deployed `SecurityLedger` contract, and its ABI file.

1. Start Ganache (if installed):

```bash
ganache --port 8545 --accounts 5 --defaultBalanceEther 100
```

2. Create the contract source folder expected by `deploy_contract.py` and place the Solidity file there:

```powershell
New-Item -ItemType Directory -Force contracts
Copy-Item .\SecurityLedger.sol .\contracts\SecurityLedger.sol
```

3. Run the deployment script:

```bash
python deploy_contract.py
```

4. Copy the deployed contract address printed by the script into the environment:

```powershell
$env:CONTRACT_ADDR="0xYOUR_DEPLOYED_CONTRACT_ADDRESS"
```

5. Run the blockchain emitter:

```bash
python step7_blockchain.py
```

The RPC URL can be overridden with `WEB3_RPC_URL` (default: `http://127.0.0.1:8545`). Keep private keys and production credentials out of source control. The supplied contract initializes the deployer as a validator; additional validator staking and multi-validator consensus should be configured and tested before treating the ledger as a production PoS network.

## Generated outputs

The scripts write results under `outputs/` and model artifacts under `models/`. Common outputs include:

| File | Description |
|---|---|
| `step1_metrics.csv` | Authentication metrics |
| `step1_auth_decisions.csv` | Per-event authentication decisions |
| `step2_telemetry_clean.csv` | Cleaned geospatial telemetry fields |
| `step2_telemetry_smote.csv` | SMOTE-augmented telemetry (when enabled) |
| `snapshots.pkl` | Temporal graph snapshots, features, and labels |
| `step3_network_features.csv` | Aggregated network features |
| `step3_metrics.csv` | Network performance metrics |
| `step4_metrics.csv` | TVGQNN detection metrics |
| `step4_classification_report.csv` | TVGQNN per-class report |
| `step5_metrics.csv` | Federated-learning metrics |
| `step6_trust_decisions.csv` | Fused risk and trust decisions |
| `step6_metrics.csv` | Trust-estimation metrics |
| `step7_metrics.csv` | Blockchain performance metrics |
| `blockchain_events.csv` | Successfully committed blockchain events |

Exact output availability depends on which steps complete successfully.

## Reproducibility and configuration

- Random seed: `42` in `step0_config.py`.
- Graph time window: 5 seconds in the graph and network processing scripts.
- TVGQNN and TGN training settings are defined near the top of their respective files.
- Federated-learning settings are defined in `step5_federated.py`.
- Trust-fusion and trust-update coefficients are defined in `step6_trust.py`.
- Blockchain event cap and energy-per-gas estimate are defined in `step7_blockchain.py`.

Record the dataset version, software versions, hardware, random seed, train/validation/test protocol, and all changed hyperparameters when reporting experimental results. Do not present synthetic or simulated outputs as measured real-world deployment results.

## Important implementation notes

- If the CSV does not contain valid latitude/longitude columns, `step2_geospatial_graph.py` maps local `x_position`/`y_position` coordinates into an illustrative geographic window. These generated coordinates are for geohash grouping, not actual vehicle GPS locations.
- The PUF routine is a software simulation, not hardware-rooted physical unclonable-function authentication.
- The SDN script calculates metrics and writes a Ryu app template; it does not itself start a Ryu controller or configure OpenFlow switches.
- The optional quantum head uses PennyLane's `default.qubit` simulator. If PennyLane is unavailable, the code falls back to a classical output head; that run should not be described as quantum execution.
- SMOTE is applied to tabular telemetry before graph construction. Synthetic records and their labels should be handled carefully when designing leakage-free evaluation.
- The federated-learning implementation partitions snapshot windows into local clients in one process; it is an experimental simulation, not a distributed deployment across independent machines.
- Check the blockchain transaction and consensus logic carefully before using it for security-critical applications.

## Citation

If you use this repository in academic work, cite the associated paper or provide a repository citation (author, title, year, and version/commit). Add the final publication details here when available.

## License

No license file was included in the supplied project archive. Unless a license is added, the repository should be treated as having **all rights reserved** by its copyright holder.
