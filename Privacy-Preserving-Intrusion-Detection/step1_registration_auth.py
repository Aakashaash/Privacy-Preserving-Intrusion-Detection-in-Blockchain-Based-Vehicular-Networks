
import time, pickle, os
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

from step0_config import load_dataset, OUT_DIR, MODEL_DIR


REGISTRY_PATH = f"{MODEL_DIR}/vehicle_registry.pkl"


# ---------- ECC registration ----------
def generate_ecc_keypair():
    priv = ec.generate_private_key(ec.SECP256R1())
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def load_private_key(priv_pem: bytes):
    return serialization.load_pem_private_key(priv_pem, password=None)


def load_public_key(pub_pem: bytes):
    return serialization.load_pem_public_key(pub_pem)


def register_vehicle(vehicle_id):
    priv_pem, pub_pem = generate_ecc_keypair()
    return {'vehicle_id': vehicle_id, 'public_key': pub_pem, 'private_key': priv_pem}


# ---------- Challenge-response ----------
def sign_challenge(priv_pem: bytes, challenge: bytes) -> bytes:
    return load_private_key(priv_pem).sign(challenge, ec.ECDSA(hashes.SHA256()))


def verify_challenge(pub_pem: bytes, challenge: bytes, signature: bytes) -> bool:
    try:
        load_public_key(pub_pem).verify(signature, challenge, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


# ---------- PUF simulation ----------
def simulate_puf_response(vehicle_id: str, challenge: bytes) -> bytes:
    h = hashes.Hash(hashes.SHA256())
    h.update(vehicle_id.encode() + challenge)
    return h.finalize()


def fuzzy_extractor_match(r1: bytes, r2: bytes, tolerance: int = 3) -> bool:
    b1 = np.unpackbits(np.frombuffer(r1, dtype=np.uint8))
    b2 = np.unpackbits(np.frombuffer(r2, dtype=np.uint8))
    return int(np.sum(b1 != b2)) <= tolerance


# ---------- Accept-value helpers ----------
def _is_true(v):
    return str(v).strip().lower() in ('success', 'true', '1', 'valid', 'verified', 'yes', 'ok')


def _ecc_verified(v):
    return str(v).strip().lower() in ('valid', 'success', 'true', '1', 'verified')


# ---------- Zero-trust gate ----------
def zero_trust_gate(row, registry, ecc_verified: bool, puf_verified: bool):
    vid = row['vehicle_id']
    if vid not in registry:
        return 'reject', 'unregistered'
    if not ecc_verified:
        return 'reject', 'ecc_fail'
    if not _is_true(row.get('authentication_status', '')):
        return 'reject', 'auth_status_failed'
    if int(row.get('identity_valid', 0)) != 1:
        return 'reject', 'identity_invalid'
    if int(row.get('identity_duplicate_count', 0)) > 1:
        return 'quarantine', 'duplicate_identity'
    if int(row.get('authentication_failure_count', 0)) > 3:
        return 'quarantine', 'repeated_failures'
    if 'puf_verification_status' in row and pd.notna(row['puf_verification_status']):
        if not _is_true(row['puf_verification_status']) and not puf_verified:
            return 'quarantine', 'puf_mismatch'
    return 'allow', 'ok'


def run():
    df = load_dataset()

    vehicles = df['vehicle_id'].unique()
    registry = {v: register_vehicle(v) for v in vehicles}
    with open(REGISTRY_PATH, 'wb') as f:
        pickle.dump(registry, f)
    print(f"[Step1] Registered {len(registry)} vehicles with ECC P-256 keys")

    challenge = os.urandom(32)
    sample_ok = 0
    for v in list(registry)[:50]:
        sig = sign_challenge(registry[v]['private_key'], challenge)
        if verify_challenge(registry[v]['public_key'], challenge, sig):
            sample_ok += 1
    print(f"[Step1] ECC challenge-response: {sample_ok}/50 passed")

    decisions, reasons = [], []
    t0 = time.perf_counter()
    for _, row in df.iterrows():
        vid = row['vehicle_id']
        ecc_ok = vid in registry and _ecc_verified(row.get('ecc_verification_status', ''))

        puf_ok = False
        if 'puf_verification_status' in row and pd.notna(row['puf_verification_status']):
            chal = os.urandom(32)
            puf_ok = fuzzy_extractor_match(
                simulate_puf_response(vid, chal),
                simulate_puf_response(vid, chal))

        d, r = zero_trust_gate(row, registry, ecc_ok, puf_ok)
        decisions.append(d); reasons.append(r)
    t1 = time.perf_counter()

    df['auth_decision'] = decisions
    df['auth_reason']   = reasons
    df['time_window']   = (df['simulation_time'] // 5).astype(int)

    reported = df['authentication_latency'] if 'authentication_latency' in df.columns else pd.Series([0.0])

    results = {
        'vehicle_auth_time_reported_ms_mean':       float(reported.mean()),
        'vehicle_auth_time_reported_ms_p95':        float(reported.quantile(0.95)),
        'vehicle_auth_time_measured_ms_per_record': (t1 - t0) * 1000 / len(df),
        'auth_allow_pct':      float((df['auth_decision'] == 'allow').mean()),
        'auth_quarantine_pct': float((df['auth_decision'] == 'quarantine').mean()),
        'auth_reject_pct':     float((df['auth_decision'] == 'reject').mean()),
        'vehicles_registered': int(len(registry)),
    }

    # --- Correct 3-class GT mapping ---
    if 'ground_truth_authentication_status' in df.columns:
        gt = df['ground_truth_authentication_status'].astype(str).str.lower()
        predicted = df['auth_decision'] == 'allow'
        expected_allow = gt == 'legitimate'
        # Binary agreement: allow should match legitimate
        results['auth_gt_agreement'] = float((predicted == expected_allow).mean())
        # Additional diagnostics by GT class
        for g in ['legitimate', 'illegitimate', 'mixed_suspicious']:
            mask = gt == g
            if mask.sum() > 0:
                results[f'allow_pct_when_gt_{g}'] = float(predicted[mask].mean())

    print("\n===== STEP 1 (ECC + PUF + ZERO-TRUST) METRICS =====")
    for k, v in results.items():
        print(f"{k:50s} {v:.4f}")

    pd.Series(results).to_csv(f"{OUT_DIR}/step1_metrics.csv")
    df[['event_id','vehicle_id','time_window','auth_decision','auth_reason']].to_csv(
        f"{OUT_DIR}/step1_auth_decisions.csv", index=False)
    return df


if __name__ == "__main__":
    run()