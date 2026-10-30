
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import label_binarize

from step0_config import OUT_DIR, MODEL_DIR, N_CLASSES, ATTACK_NAMES, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- Hyperparameters ----------
HIDDEN = 64
EPOCHS = 65
LR = 1e-3
BATCH_SNAPS = 2      # MUST match step4_tvgqnn.py


# ====================== Helpers ======================
def snapshot_to_data(G, labels, feature_cols, feat_mean=None, feat_std=None):
    nodes = list(G.nodes)
    if not nodes:
        return None, []
    idx = {v: i for i, v in enumerate(nodes)}
    X = np.array([[G.nodes[v]['features'].get(c, 0.0) for c in feature_cols]
                  for v in nodes], dtype=np.float32)
    edges = [(idx[u], idx[v]) for u, v in G.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
                 else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([labels.get(v, 0) for v in nodes], dtype=torch.long)
    data = Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=edge_index, y=y)
    if feat_mean is not None and feat_std is not None:
        x = (data.x - feat_mean) / feat_std
        data.x = torch.clamp(x, -5.0, 5.0)
    return data, nodes


def compute_feature_stats(snapshots, feature_cols):
    all_X = []
    for G in snapshots.values():
        if G.number_of_nodes() == 0:
            continue
        for v in G.nodes:
            all_X.append([G.nodes[v]['features'].get(c, 0.0) for c in feature_cols])
    if not all_X:
        return torch.zeros(len(feature_cols)), torch.ones(len(feature_cols))
    A = np.array(all_X, dtype=np.float32)
    return (torch.tensor(A.mean(axis=0), dtype=torch.float32),
            torch.tensor(A.std(axis=0) + 1e-6, dtype=torch.float32))


def build_prev_h(node_ids, h_dict, hidden=HIDDEN):
    if not h_dict:
        return None
    return torch.stack([h_dict[v] if v in h_dict else torch.zeros(hidden)
                        for v in node_ids]).to(torch.float32)


def store_h(node_ids, h_batch, h_dict):
    for i, v in enumerate(node_ids):
        h_dict[v] = h_batch[i].detach()


def count_class_distribution(sequences, snapshots, label_map):
    counts = np.zeros(N_CLASSES, dtype=int)
    for seq in sequences:
        for tw in seq:
            G = snapshots[tw]
            if G.number_of_nodes() == 0:
                continue
            for v in G.nodes:
                counts[label_map.get((v, tw), 0)] += 1
    return counts


def print_split_distribution(name, sequences, snapshots, label_map):
    counts = count_class_distribution(sequences, snapshots, label_map)
    total = int(counts.sum())
    print(f"\n{'='*60}")
    print(f"  {name}  —  {len(sequences)} sequences, {total} node-samples")
    print(f"{'='*60}")
    if total == 0:
        print("  (empty split)")
        return
    for i in range(N_CLASSES):
        pct = (counts[i] / total * 100) if total > 0 else 0.0
        print(f"  class {i}  {ATTACK_NAMES[i]:16s} : {counts[i]:7d}  ({pct:5.2f}%)")


# ====================== Model ======================
class TGNBaseline(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_cls=N_CLASSES):
        super().__init__()
        self.gcn1 = SAGEConv(in_dim, hidden)
        self.gcn2 = SAGEConv(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x, edge_index, prev_h=None):
        x = x.to(torch.float32)
        h1 = F.relu(self.gcn1(x, edge_index))
        h2 = F.relu(self.gcn2(h1, edge_index))
        h_t = self.gru(h2, prev_h) if prev_h is not None else h2
        logits = self.head(h_t)
        return logits, h_t


# ====================== Train / Validate / Evaluate ======================
def train_one_epoch(model, opt, sequences, snapshots, label_map, feat_cols,
                    feat_mean, feat_std):
    model.train()
    total, skipped = 0.0, 0
    for seq in sequences:
        opt.zero_grad()
        h_dict = {}
        for tw in seq:
            G = snapshots[tw]
            if G.number_of_nodes() == 0:
                continue
            data, node_ids = snapshot_to_data(
                G, {v: label_map.get((v, tw), 0) for v in G.nodes},
                feat_cols, feat_mean, feat_std)
            if data is None:
                continue
            prev_h = build_prev_h(node_ids, h_dict)
            logits, h_t = model(data.x, data.edge_index, prev_h)
            logits = torch.clamp(logits, -20.0, 20.0)
            loss = F.cross_entropy(logits, data.y,
                                   weight=torch.tensor([1.0, 3.0, 2.0, 2.0, 2.0, 3.0]))
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            store_h(node_ids, h_t, h_dict)
            total += loss.item()
    return total / max(1, len(sequences) - skipped)


@torch.no_grad()
def validation_loss(model, sequences, snapshots, label_map, feat_cols,
                    feat_mean, feat_std):
    model.eval()
    total, n = 0.0, 0
    for seq in sequences:
        h_dict = {}
        for tw in seq:
            G = snapshots[tw]
            if G.number_of_nodes() == 0:
                continue
            data, node_ids = snapshot_to_data(
                G, {v: label_map.get((v, tw), 0) for v in G.nodes},
                feat_cols, feat_mean, feat_std)
            if data is None:
                continue
            prev_h = build_prev_h(node_ids, h_dict)
            logits, h_t = model(data.x, data.edge_index, prev_h)
            logits = torch.clamp(logits, -20.0, 20.0)
            total += F.cross_entropy(logits, data.y).item()
            n += 1
            store_h(node_ids, h_t, h_dict)
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, sequences, snapshots, label_map, feat_cols, feat_mean, feat_std):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    for seq in sequences:
        h_dict = {}
        for tw in seq:
            G = snapshots[tw]
            if G.number_of_nodes() == 0:
                continue
            data, node_ids = snapshot_to_data(
                G, {v: label_map.get((v, tw), 0) for v in G.nodes},
                feat_cols, feat_mean, feat_std)
            if data is None:
                continue
            prev_h = build_prev_h(node_ids, h_dict)
            logits, h_t = model(data.x, data.edge_index, prev_h)
            proba = F.softmax(torch.clamp(logits, -20.0, 20.0), dim=1)
            y_true.extend(data.y.tolist())
            y_pred.extend(proba.argmax(1).tolist())
            y_proba.append(proba.cpu().numpy())
            store_h(node_ids, h_t, h_dict)
    if not y_true:
        return None, None, None
    return np.array(y_true), np.array(y_pred), np.vstack(y_proba)


def compute_metrics(y_true, y_pred, y_proba, class_names):
    m = {}
    m['detection_accuracy'] = float(accuracy_score(y_true, y_pred))
    m['precision_macro']    = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    m['recall_macro']       = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    m['f1_macro']           = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    m['f1_weighted']        = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    try:
        m['auc_roc_macro'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
    except ValueError:
        m['auc_roc_macro'] = float('nan')
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    for i in range(N_CLASSES):
        try:
            m[f'auc_class_{i}_{class_names[i]}'] = float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
        except ValueError:
            m[f'auc_class_{i}_{class_names[i]}'] = float('nan')
    far = []
    for i in range(N_CLASSES):
        tn, fp, fn, tp = confusion_matrix(y_true == i, y_pred == i).ravel()
        far.append(fp / (fp + tn + 1e-9))
    m['false_alarm_rate'] = float(np.mean(far))
    report = classification_report(y_true, y_pred, digits=4,
                                   labels=list(range(N_CLASSES)),
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)
    return m, report


# ====================== Main ======================
def run():
    snapshot_path = f"{OUT_DIR}/snapshots.pkl"
    if not os.path.exists(snapshot_path):
        print(f"[TGN] ERROR: {snapshot_path} not found. Run step2_geospatial_graph.py first.")
        return {}

    with open(snapshot_path, "rb") as f:
        b = pickle.load(f)
    snaps, feat_cols, label_map = b['snapshots'], b['feature_cols'], b['label_map']

    print(f"[TGN] Loaded {len(snaps)} snapshots | {len(feat_cols)} features")

    windows = sorted([w for w in snaps if snaps[w].number_of_nodes() > 0])
    if not windows:
        print("[TGN] ERROR: no non-empty snapshots.")
        return {}

    feat_mean, feat_std = compute_feature_stats(snaps, feat_cols)

    sequences = [windows[i:i + BATCH_SNAPS] for i in range(0, len(windows), BATCH_SNAPS)]
    n = len(sequences)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_seqs = sequences[:n_train]
    val_seqs   = sequences[n_train:n_train + n_val]
    test_seqs  = sequences[n_train + n_val:]

    print(f"[TGN] Sequences built: {n} total")
    print(f"[TGN] 70/15/15 split: {len(train_seqs)} train | "
          f"{len(val_seqs)} val | {len(test_seqs)} test")

    # ============ CLASS-WISE PRINT (the key output) ============
    print_split_distribution("TRAIN (TGN)",      train_seqs, snaps, label_map)
    print_split_distribution("VALIDATION (TGN)", val_seqs,   snaps, label_map)
    print_split_distribution("TEST (TGN)",       test_seqs,  snaps, label_map)

    # Save to CSV
    dist_rows = []
    for name, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        c = count_class_distribution(seqs, snaps, label_map)
        total = int(c.sum())
        for i in range(N_CLASSES):
            dist_rows.append({
                'split': name,
                'class_id': i,
                'class_name': ATTACK_NAMES[i],
                'count': int(c[i]),
                'pct': float(c[i] / total * 100) if total > 0 else 0.0,
            })
    pd.DataFrame(dist_rows).to_csv(
        f"{OUT_DIR}/split_class_distribution_tgn.csv", index=False)
    print(f"\n[TGN] Saved class distribution: {OUT_DIR}/split_class_distribution_tgn.csv")

    # ============ TRAINING ============
    model = TGNBaseline(len(feat_cols)).to(torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, opt, train_seqs, snaps, label_map,
                                     feat_cols, feat_mean, feat_std)
        val_loss = validation_loss(model, val_seqs, snaps, label_map,
                                   feat_cols, feat_mean, feat_std)
        history.append({'epoch': epoch+1, 'train': train_loss, 'val': val_loss})

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = " ★ best"

        print(f"[TGN] epoch {epoch+1:02d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[TGN] restored best model (val_loss={best_val_loss:.4f})")

    pd.DataFrame(history).to_csv(f"{OUT_DIR}/step4_tgn_loss_history.csv", index=False)

    # ============ TEST EVAL ============
    y_true, y_pred, y_proba = evaluate(model, test_seqs, snaps, label_map,
                                       feat_cols, feat_mean, feat_std)
    if y_true is None:
        print("[TGN] ERROR: no test data")
        return {}

    class_names = [f"{i}_{ATTACK_NAMES[i]}" for i in range(N_CLASSES)]
    metrics, report = compute_metrics(y_true, y_pred, y_proba, class_names)
    metrics['split_train_sequences'] = len(train_seqs)
    metrics['split_val_sequences']   = len(val_seqs)
    metrics['split_test_sequences']  = len(test_seqs)

    print("\n===== TGN BASELINE TEST METRICS =====")
    for k, v in metrics.items():
        print(f"{k:50s} {v:.4f}" if isinstance(v, float) else f"{k:50s} {v}")
    print("\n===== PER-CLASS REPORT =====")
    print(pd.DataFrame(report).T.round(4))

    pd.Series(metrics).to_csv(f"{OUT_DIR}/step4_tgn_metrics.csv")
    pd.DataFrame(report).T.to_csv(f"{OUT_DIR}/step4_tgn_classification_report.csv")
    torch.save(model.state_dict(), f"{MODEL_DIR}/tgn_baseline.pt")
    return metrics


if __name__ == "__main__":
    run()