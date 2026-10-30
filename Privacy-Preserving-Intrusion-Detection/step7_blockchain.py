
import json
import os
import time
import numpy as np
import pandas as pd
from web3 import Web3

from step0_config import OUT_DIR, ATTACK_NAMES

RPC_URL       = os.getenv("WEB3_RPC_URL", "http://127.0.0.1:8545")
CONTRACT_ADDR = os.getenv("CONTRACT_ADDR")
ABI_PATH      = "contracts/SecurityLedger.abi.json"
ENERGY_PER_GAS_J = 0.00004
MAX_EVENTS    = 500


def connect():
    """Connect to chain + load contract. Raises if anything is missing."""
    if not CONTRACT_ADDR:
        raise RuntimeError(
            "CONTRACT_ADDR environment variable is not set.\n"
            "Run: $env:CONTRACT_ADDR=\"0x...\"  (get it from deploy_contract.py)"
        )
    if not os.path.exists(ABI_PATH):
        raise RuntimeError(
            f"ABI not found at {ABI_PATH}.\n"
            "Run: python deploy_contract.py  first."
        )

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise RuntimeError(
            f"Cannot connect to blockchain at {RPC_URL}.\n"
            "Start Ganache:  ganache --port 8545 --accounts 5 --defaultBalanceEther 100"
        )

    with open(ABI_PATH) as f:
        abi = json.load(f)

    contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDR), abi=abi)
    acct = w3.eth.accounts[0]
    w3.eth.default_account = acct
    return w3, contract, acct


def run():
    w3, contract, acct = connect()
    print(f"[Step7] Connected to {RPC_URL}")
    print(f"[Step7] Contract: {CONTRACT_ADDR}")
    print(f"[Step7] Using account: {acct}")
    print(f"[Step7] Chain id: {w3.eth.chain_id} | block: {w3.eth.block_number}")

    trust = pd.read_csv(f"{OUT_DIR}/step6_trust_decisions.csv")
    events = trust[trust['decision'] != 'trusted'].head(MAX_EVENTS).reset_index(drop=True)
    print(f"[Step7] Submitting {len(events)} events "
          f"(out of {len(trust)} trust decisions)")

    propose_times, confirm_times, gas_list = [], [], []
    committed_events = []
    t_start = time.perf_counter()

    for idx, r in events.iterrows():
        # event type from attack_class_label
        attack_class = int(r.get('attack_class_label', 1))
        ev_type = ATTACK_NAMES.get(attack_class, "Unknown")

        # evidence hash from event data
        evidence_hash = Web3.keccak(text=f"{r['vehicle_id']}|{r['event_id']}")

        # ---- proposeEvent ----
        t0 = time.perf_counter()
        try:
            tx = contract.functions.proposeEvent(
                str(r['vehicle_id']),
                str(r['event_id']),
                ev_type,
                int(round(float(r['previous_trust_score']) * 1e4)),
                int(round(float(r['learned_trust'])       * 1e4)),
                "approved",
                "tvgqnn_v2",
                evidence_hash
            ).transact({'from': acct, 'gas': 500_000})
            rcpt = w3.eth.wait_for_transaction_receipt(tx)
            propose_times.append(time.perf_counter() - t0)
            gas_list.append(rcpt.gasUsed)
        except Exception as e:
            print(f"  [propose FAIL] {r['event_id']}: {e}")
            continue

        # ---- attest ----
        t1 = time.perf_counter()
        try:
            tx2 = contract.functions.attest(evidence_hash).transact({'from': acct, 'gas': 300_000})
            rcpt2 = w3.eth.wait_for_transaction_receipt(tx2)
            confirm_times.append(time.perf_counter() - t1)
            gas_list.append(rcpt2.gasUsed)
        except Exception as e:
            print(f"  [attest FAIL]  {r['event_id']}: {e}")
            continue

        committed_events.append({
            'event_id':        r['event_id'],
            'vehicle_id':      r['vehicle_id'],
            'event_type':      ev_type,
            'previous_trust':  float(r['previous_trust_score']),
            'updated_trust':   float(r['learned_trust']),
            'tx_propose_hash': rcpt.transactionHash.hex(),
            'tx_attest_hash':  rcpt2.transactionHash.hex(),
            'block_number':    rcpt2.blockNumber,
            'gas_used':        rcpt.gasUsed + rcpt2.gasUsed,
        })
        print(f"  [{idx+1:2d}/{len(events)}] {r['vehicle_id']:>8s} | {ev_type:15s} | "
              f"trust {r['previous_trust_score']:.2f}→{r['learned_trust']:.2f} | "
              f"block {rcpt2.blockNumber}")

    t_total = time.perf_counter() - t_start

    if not confirm_times:
        raise RuntimeError(
            "[Step7] No events were committed. Check the contract / account / gas limits."
        )

    # ---- Metrics ----
    res = {
        'blockchain_tx_throughput_tps':        len(confirm_times) / max(t_total, 1e-6),
        'blockchain_confirmation_time_mean_s': float(np.mean(confirm_times)),
        'blockchain_confirmation_time_p95_s':  float(np.quantile(confirm_times, 0.95)),
        'avg_gas_per_event':                   float(np.mean(gas_list)),
        'total_gas_used':                      int(np.sum(gas_list)),
        'energy_consumption_j':                float(np.sum(gas_list) * ENERGY_PER_GAS_J),
        'blockchain_events_committed':         len(confirm_times),
        'blockchain_events_attempted':         len(events),
        'blockchain_total_time_s':             float(t_total),
    }

    print("\n===== STEP 7 BLOCKCHAIN METRICS (REAL) =====")
    for k, v in res.items():
        print(f"{k:45s} {v:.6f}" if isinstance(v, float) else f"{k:45s} {v}")

    # ---- On-chain state verification ----
    on_chain_total = contract.functions.totalEvents().call()
    print(f"\n[Step7] On-chain events[] length: {on_chain_total}")

    committed_count_on_chain = 0
    for i in range(on_chain_total):
        ev = contract.functions.getEvent(i).call()
        if ev[10]:  # committed flag
            committed_count_on_chain += 1
    print(f"[Step7] On-chain committed count: {committed_count_on_chain}")

    # ---- Save ----
    pd.Series(res).to_csv(f"{OUT_DIR}/step7_metrics.csv")
    pd.DataFrame(committed_events).to_csv(
        f"{OUT_DIR}/blockchain_events.csv", index=False)
    print(f"[Step7] Wrote {OUT_DIR}/step7_metrics.csv")
    print(f"[Step7] Wrote {OUT_DIR}/blockchain_events.csv "
          f"({len(committed_events)} rows)")
    return res


if __name__ == "__main__":
    run()