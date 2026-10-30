
import os, pickle
import numpy as np
import pandas as pd
import networkx as nx
import pygeohash as pgh

from step0_config import load_dataset, get_feature_cols, OUT_DIR

WINDOW_SEC = 5
GEOHASH_PRECISIONS = [4, 5, 6]

# --- Region assignment ---
def project_to_latlon(df):
    """
    Return (lat_series, lon_series) suitable for geohash.
    If real lat/lon columns exist and are in valid range -> use directly.
    Otherwise project SUMO x/y into a plausible lat/lon window.
    """
    if {'latitude', 'longitude'} <= set(df.columns):
        lat = pd.to_numeric(df['latitude'], errors='coerce')
        lon = pd.to_numeric(df['longitude'], errors='coerce')
        if lat.between(-90, 90).all() and lon.between(-180, 180).all():
            return lat, lon

    # SUMO local coords — project to a small region (e.g., Bengaluru 12.9, 77.6)
    x = pd.to_numeric(df['x_position'], errors='coerce').fillna(0.0)
    y = pd.to_numeric(df['y_position'], errors='coerce').fillna(0.0)

    # Normalize to unit square, then scale to a small geographic window
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

    lat_center, lon_center, span = 12.97, 77.60, 0.05   # ~5 km box
    lat = lat_center + (y_norm - 0.5) * span
    lon = lon_center + (x_norm - 0.5) * span
    return lat, lon


def assign_regions(df):
    """Hierarchical geohash, vectorized."""
    lat, lon = project_to_latlon(df)

    # Vectorize pygeohash via list comprehension over numpy arrays (still fast)
    for prec in GEOHASH_PRECISIONS:
        df[f'geohash_{prec}'] = [pgh.encode(float(la), float(lo), precision=prec)
                                 for la, lo in zip(lat.values, lon.values)]
    df['region_id'] = df[f'geohash_{GEOHASH_PRECISIONS[-1]}']
    return df


def build_snapshot(frame, feat_cols):
    """Build a NetworkX graph for one time window."""
    G = nx.Graph()

    # Pre-extract arrays for speed
    vids    = frame['vehicle_id'].values
    xs      = frame['x_position'].astype(float).values
    ys      = frame['y_position'].astype(float).values
    is_rsu  = frame.get('is_rsu', pd.Series([0]*len(frame))).astype(int).values
    regions = frame['region_id'].values if 'region_id' in frame else [''] * len(frame)
    ranges  = frame.get('communication_range',
                        pd.Series([250.0]*len(frame))).fillna(250.0).astype(float).values

    feat_matrix = frame[feat_cols].fillna(0).values.astype(np.float32)

    for i in range(len(frame)):
        feats = dict(zip(feat_cols, feat_matrix[i]))
        G.add_node(vids[i], pos=(xs[i], ys[i]),
                   is_rsu=int(is_rsu[i]),
                   region=regions[i],
                   features=feats)

    # Edges: use spatial bucketing for speed on big frames
    for i in range(len(frame)):
        for j in range(i + 1, len(frame)):
            if is_rsu[i] and is_rsu[j]:
                continue
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            d = (dx*dx + dy*dy) ** 0.5
            rng = min(ranges[i], ranges[j]) or 250.0
            if d <= rng:
                G.add_edge(vids[i], vids[j], distance=float(d))
    return G


def run():
    src = f"{OUT_DIR}/step2_telemetry_smote.csv"
    if os.path.exists(src):
        print(f"[Step2] Loading SMOTE-augmented telemetry")
        df = pd.read_csv(src)
        df['simulation_time'] = pd.to_numeric(df['simulation_time'], errors='coerce')
        df = df.dropna(subset=['simulation_time', 'vehicle_id', 'attack_class_label']).reset_index(drop=True)
        df['attack_class_label'] = df['attack_class_label'].astype(int)
        feat_cols = get_feature_cols(df)
    else:
        df = load_dataset()
        feat_cols = get_feature_cols(df)

    df = assign_regions(df)
    df['time_window'] = (df['simulation_time'] // WINDOW_SEC).astype(int)

    # Drop  rows with no valid coords/times to avoid graph bloat
    if 'vehicle_id' in df.columns:
        df = df[df['vehicle_id'].notna()]

    # Cap the number of windows used for graphs to keep run time sane
    # (SMOTE-augmented data has same time windows as original; 
    # vehicles share simulation_time with real ones, so this stays bounded.)
    unique_windows = sorted(df['time_window'].unique())
    print(f"[Step2] {len(unique_windows)} time windows | {len(df)} rows | {len(feat_cols)} features")

    snapshots = {}
    for tw, frame in df.groupby('time_window'):
        frame = frame.sort_values('simulation_time').groupby('vehicle_id').last().reset_index()
        # Cap nodes per snapshot to keep graph building fast
        if len(frame) > 500:
            frame = frame.sample(n=500, random_state=42).reset_index(drop=True)
        snapshots[int(tw)] = build_snapshot(frame, feat_cols)

    label_map = df.set_index(['vehicle_id','time_window'])['attack_class_label'].to_dict()

    with open(f"{OUT_DIR}/snapshots.pkl", "wb") as f:
        pickle.dump({'snapshots': snapshots, 'feature_cols': feat_cols,
                     'label_map': label_map}, f)

    print(f"[Step2] Built {len(snapshots)} snapshots")

    keep = ['event_id','vehicle_id','time_window','region_id',
            'geohash_4','geohash_5','geohash_6',
            'x_position','y_position','speed','neighbor_count']
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(f"{OUT_DIR}/step2_telemetry_clean.csv", index=False)
    return snapshots, feat_cols


if __name__ == "__main__":
    run()