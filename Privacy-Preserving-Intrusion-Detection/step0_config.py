
import hashlib, os
import pandas as pd

DATA_PATH = r"C:\Aakash\9mer\New folder\vanet_security_dataset_100k.csv"
OUT_DIR   = "outputs"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ATTACK_NAMES = {
    0: "Normal", 1: "Sybil", 2: "FalsePosition",
    3: "FalseInformation", 4: "PacketDropping", 5: "Impersonation"
}
N_CLASSES = 6
RANDOM_SEED = 42

LEAKAGE_COLS = [
    'is_malicious','attack_present','attack_type','malicious_behavior_score',
    'ground_truth_trust_status','ground_truth_authentication_status',
    'ground_truth_forwarding_status','trust_decision','data_forwarding_decision',
    'updated_trust_score','reputation_score','attack_class_label'
]

ID_COLS = [
    'scenario_id','event_id','message_id','vehicle_id','sender_id','receiver_id',
    'attack_start_time','attack_duration','simulation_time',
    'vehicle_type','vehicle_role','vehicle_group_id','message_type',
    'authentication_status','ecc_verification_status','puf_verification_status'
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=['event_id'])
    df['simulation_time'] = pd.to_numeric(df['simulation_time'], errors='coerce')
    df = df.dropna(subset=['simulation_time', 'vehicle_id', 'attack_class_label']).reset_index(drop=True)
    df['attack_class_label'] = df['attack_class_label'].astype(int)
    return df


def get_feature_cols(df):
    exclude = set(LEAKAGE_COLS + ID_COLS)
    return [c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().sum() > 0]


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()