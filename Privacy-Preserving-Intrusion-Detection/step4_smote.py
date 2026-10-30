
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from step0_config import load_dataset, get_feature_cols, OUT_DIR, RANDOM_SEED, N_CLASSES


def augment_with_smote(df, feat_cols, k_neighbors=5):
    """
    SMOTE on vehicle-level numeric features.
    Returns df +  rows with new vehicle_id/event_id.
    """
    X = df[feat_cols].fillna(0).values
    y = df['attack_class_label'].values.astype(int)

    counts = np.bincount(y, minlength=N_CLASSES)
    present_classes = int((counts > 0).sum())
    if present_classes < 2:
        print("[SMOTE] fewer than 2 classes present -> skipping")
        return df

    min_count = counts[counts > 0].min()
    k = max(1, min(k_neighbors, min_count - 1)) if min_count > 1 else 1
    print(f"[SMOTE] class counts before: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"[SMOTE] using k_neighbors={k}")

    sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[SMOTE] {X.shape} -> {X_res.shape}")

    n_orig = len(df)
    n_new = len(X_res) - n_orig
    if n_new <= 0:
        return df

    # --- Build  rows from scratch ---
    synth_feats  = X_res[n_orig:]           # (n_new, n_feat)
    synth_labels = y_res[n_orig:]           # (n_new,)

    aug = pd.DataFrame(synth_feats, columns=feat_cols)
    aug['attack_class_label'] = synth_labels.astype(int)
    aug['vehicle_id'] = [f"SYN_{i}" for i in range(n_new)]
    aug['event_id']   = [f"SYNEV_{i}" for i in range(n_new)]

    # Fill remaining columns with sensible defaults
    for c in df.columns:
        if c in aug.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            aug[c] = 0
        else:
            aug[c] = "synthetic"

    # Match dtypes and column order to original
    for c in feat_cols:
        aug[c] = aug[c].astype(df[c].dtype)
    aug = aug[df.columns]

    combined = pd.concat([df, aug], ignore_index=True)
    print(f"[SMOTE] After: {dict(zip(*np.unique(combined['attack_class_label'], return_counts=True)))}")
    return combined


def run():
    df = load_dataset()
    feat_cols = get_feature_cols(df)
    print(f"[SMOTE] {len(feat_cols)} features | original rows={len(df)}")
    df_aug = augment_with_smote(df, feat_cols)
    df_aug.to_csv(f"{OUT_DIR}/step2_telemetry_smote.csv", index=False)
    print(f"[SMOTE] Wrote augmented telemetry: {df_aug.shape}")
    print(f"[SMOTE] Saved to {OUT_DIR}/step2_telemetry_smote.csv")
    return df_aug


if __name__ == "__main__":
    run()