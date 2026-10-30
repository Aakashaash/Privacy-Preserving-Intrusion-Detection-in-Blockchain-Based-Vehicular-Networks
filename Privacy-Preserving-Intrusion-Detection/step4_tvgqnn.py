
import time, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import label_binarize

try:
    import pennylane as qml
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

from step0_config import OUT_DIR, MODEL_DIR, N_CLASSES, ATTACK_NAMES, RANDOM_SEED

torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

HIDDEN, LATENT, N_QUBITS, N_Q_LAYERS = 64, 16, 4, 2
EPOCHS, BATCH_SNAPS, LR = 65, 2, 1e-3        # BATCH_SNAPS=2 → 30 sequences


if QUANTUM_AVAILABLE:
    qdev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(qdev, interface="torch", diff_method="backprop")
    def vqc_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumHead(nn.Module):
    def __init__(self, latent_dim=LATENT, n_cls=N_CLASSES):
        super().__init__()
        self.reduce = nn.Linear(latent_dim, N_QUBITS)
        self.weights = nn.Parameter(torch.randn(N_Q_LAYERS, N_QUBITS, 3) * 0.1)
        self.out = nn.Linear(N_QUBITS, n_cls)

    def forward(self, z):
        if not QUANTUM_AVAILABLE:
            return self.out(torch.tanh(self.reduce(z))), torch.zeros(z.size(0), N_QUBITS)
        x = (torch.tanh(self.reduce(z)) * np.pi).to(torch.float32)
        w = self.weights.to(torch.float32)
        q = torch.stack([torch.stack(vqc_circuit(xi, w)) for xi in x]).to(torch.float32)
        return self.out(q), q


class TVGQNN(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, latent=LATENT, n_cls=N_CLASSES):
        super().__init__()
        self.gcn1 = SAGEConv(in_dim, hidden)
        self.gcn2 = SAGEConv(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)
        self.qhead = QuantumHead(latent, n_cls)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, edge_index, prev_h=None):
        x = x.to(torch.float32)
        h1 = F.relu(self.gcn1(x, edge_index))
        h2 = F.relu(self.gcn2(h1, edge_index))
        h_t = self.gru(h2, prev_h) if prev_h is not None else h2
        mu = self.mu_head(h_t)
        logvar = self.logvar_head(h_t)
        z = self.reparameterize(mu, logvar)
        logits, q = self.qhead(z)
        return logits, h_t, mu, logvar, q


def snapshot_to_data(G, labels, feature_cols):
    """Returns Data + ordered node-id list so caller can align hidden state."""
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
    return Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index, y=y), nodes


def build_prev_h(node_ids, h_dict, hidden=HIDDEN):
    """Assemble prev_h batch aligned with current node order (zeros for new vehicles)."""
    if not h_dict:
        return None
    rows = []
    for v in node_ids:
        if v in h_dict:
            rows.append(h_dict[v])
        else:
            rows.append(torch.zeros(hidden))
    return torch.stack(rows).to(torch.float32)


def store_h(node_ids, h_batch, h_dict):
    """Save hidden states by vehicle_id (detached to avoid BPTT memory blowup)."""
    for i, v in enumerate(node_ids):
        h_dict[v] = h_batch[i].detach()


def train_one_epoch(model, opt, sequences, snapshots, label_map, feat_cols):
    model.train(); total = 0.0
    for seq in sequences:
        opt.zero_grad()
        h_dict = {}                           # per-sequence hidden state
        for tw in seq:
            G = snapshots[tw]
            if G.number_of_nodes() == 0:
                continue
            data, node_ids = snapshot_to_data(
                G, {v: label_map.get((v, tw), 0) for v in G.nodes}, feat_cols)
            if data is None:
                continue
            prev_h = build_prev_h(node_ids, h_dict)
            logits, h_t, mu, logvar, _ = model(data.x, data.edge_index, prev_h)
            ce = F.cross_entropy(logits, data.y,
                                 weight=torch.tensor([1.0, 3.0, 2.0, 2.0, 2.0, 3.0]))
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = ce + 0.001 * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            store_h(node_ids, h_t, h_dict)
            total += loss.item()
    return total / max(1, len(sequences))


@torch.no_grad()
def evaluate(model, sequences, snapshots, label_map, feat_cols):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    for seq in sequences:
        h_dict = {}
        for tw in seq:
            G = snapshots[tw]
            if G.number_of_nodes() == 0:
                continue
            data, node_ids = snapshot_to_data(
                G, {v: label_map.get((v, tw), 0) for v in G.nodes}, feat_cols)
            if data is None:
                continue
            prev_h = build_prev_h(node_ids, h_dict)
            logits, h_t, _, _, _ = model(data.x, data.edge_index, prev_h)
            proba = F.softmax(logits, dim=1)
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


def run():
    with open(f"{OUT_DIR}/snapshots.pkl", "rb") as f:
        b = pickle.load(f)
    snaps, feat_cols, label_map = b['snapshots'], b['feature_cols'], b['label_map']

    windows = sorted([w for w in snaps if snaps[w].number_of_nodes() > 0])
    if not windows:
        print("[TVGQNN] no usable snapshots"); return {}

    sequences = [windows[i:i+BATCH_SNAPS] for i in range(0, len(windows), BATCH_SNAPS)]
    split = int(len(sequences) * 0.8)
    train_seqs, test_seqs = sequences[:split], sequences[split:]

    print(f"[TVGQNN] {len(train_seqs)} train | {len(test_seqs)} test "
          f"| {len(feat_cols)} feats | quantum={'ON' if QUANTUM_AVAILABLE else 'OFF'}")

    model = TVGQNN(len(feat_cols)).to(torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, opt, train_seqs, snaps, label_map, feat_cols)
        print(f"epoch {epoch+1:02d}  loss={loss:.4f}")

    tw = test_seqs[0][0] if test_seqs else train_seqs[0][0]
    G = snaps[tw]
    data, node_ids = snapshot_to_data(G, {v: label_map.get((v, tw), 0) for v in G.nodes}, feat_cols)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(data.x, data.edge_index, None)
    latency_ms = (time.perf_counter() - t0) * 1000

    y_true, y_pred, y_proba = evaluate(model, test_seqs, snaps, label_map, feat_cols)
    if y_true is None:
        print("[TVGQNN] no test data"); return {}

    class_names = [f"{i}_{ATTACK_NAMES[i]}" for i in range(N_CLASSES)]
    metrics, report = compute_metrics(y_true, y_pred, y_proba, class_names)
    metrics['attack_detection_latency_ms_per_sample'] = float(latency_ms)

    print("\n===== TVGQNN DETECTION METRICS =====")
    for k, v in metrics.items():
        print(f"{k:50s} {v:.4f}" if isinstance(v, float) else f"{k:50s} {v}")
    print("\n===== PER-CLASS REPORT =====")
    print(pd.DataFrame(report).T.round(4))

    pd.Series(metrics).to_csv(f"{OUT_DIR}/step4_metrics.csv")
    pd.DataFrame(report).T.to_csv(f"{OUT_DIR}/step4_classification_report.csv")
    np.save(f"{MODEL_DIR}/step4_y_true.npy", y_true)
    np.save(f"{MODEL_DIR}/step4_y_pred.npy", y_pred)
    np.save(f"{MODEL_DIR}/step4_y_proba.npy", y_proba)
    torch.save(model.state_dict(), f"{MODEL_DIR}/tvgqnn.pt")
    return metrics


if __name__ == "__main__":
    run()