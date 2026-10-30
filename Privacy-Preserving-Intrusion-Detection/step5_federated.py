
import pickle, time
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from step0_config import OUT_DIR, MODEL_DIR, N_CLASSES, RANDOM_SEED

torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
HIDDEN, ROUNDS, LOCAL_EPOCHS, MAX_NORM, N_CLIENTS = 32, 5, 2, 2.0, 10


def snapshot_to_data(G, labels, feature_cols):
    nodes = list(G.nodes)
    if not nodes: return None
    idx = {v:i for i,v in enumerate(nodes)}
    X = np.array([[G.nodes[v]['features'].get(c,0.0) for c in feature_cols]
                  for v in nodes], dtype=np.float32)
    edges = [(idx[u], idx[v]) for u,v in G.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
                 else torch.empty((2,0), dtype=torch.long)
    y = torch.tensor([labels.get(v,0) for v in nodes], dtype=torch.long)
    return Data(x=torch.tensor(X), edge_index=edge_index, y=y)


class LocalModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_cls=N_CLASSES):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, n_cls)
    def forward(self, data):
        h = F.relu(self.conv1(data.x, data.edge_index))
        h = F.relu(self.conv2(h, data.edge_index))
        return self.head(h)


def local_update(global_sd, snapshots, label_map, feat_cols, lr=1e-3):
    model = LocalModel(len(feat_cols)); model.load_state_dict(global_sd)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(LOCAL_EPOCHS):
        for tw in sorted(snapshots):
            G = snapshots[tw]
            if G.number_of_nodes() == 0: continue
            data = snapshot_to_data(G, {v: label_map.get((v,tw),0) for v in G.nodes}, feat_cols)
            if data is None or data.y.numel() == 0: continue
            model.train(); opt.zero_grad()
            loss = F.cross_entropy(model(data), data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
    return {k: v.detach().clone() for k,v in model.state_dict().items()}


def similarity(u, ref):
    return float(1.0/(1.0 + sum((u[k]-ref[k]).pow(2).sum() for k in u)))


def update_norm(u):
    return float(torch.sqrt(sum(u[k].pow(2).sum() for k in u)))


def robust_aggregate(updates, weights):
    out = {}
    for k in updates[0]:
        stacked = torch.stack([u[k].float() for u in updates])
        w = torch.tensor(weights).view(-1, *([1]*(stacked.ndim-1)))
        out[k] = (stacked * w).sum(0)
    return out


def run():
    with open(f"{OUT_DIR}/snapshots.pkl","rb") as f:
        b = pickle.load(f)
    snaps, feat_cols, label_map = b['snapshots'], b['feature_cols'], b['label_map']
    windows = sorted(snaps.keys())
    if not windows:
        print("[Step5] no snapshots"); return {}

    global_sd = LocalModel(len(feat_cols)).state_dict()
    bucket = max(1, len(windows)//N_CLIENTS)
    clients = [windows[i:i+bucket] for i in range(0, len(windows), bucket)][:N_CLIENTS]

    round_times, accepted = [], []
    for rnd in range(ROUNDS):
        t0 = time.perf_counter()
        updates = []
        for wl in clients:
            subset = {w: snaps[w] for w in wl if w in snaps and snaps[w].number_of_nodes() > 0}
            if subset: updates.append(local_update(global_sd, subset, label_map, feat_cols))
        if not updates: continue

        median = {k: torch.stack([u[k] for u in updates]).median(0).values for k in updates[0]}
        sims = [similarity(u, median) for u in updates]
        norms = [update_norm(u) for u in updates]
        median_norm = float(np.median(norms))

        # Quality: accept updates within 2× median norm and with positive similarity
        kept = [(u, s) for u, s, n in zip(updates, sims, norms)
                if n < 3 * median_norm and s > 0.1]
        if not kept:
            kept = list(zip(updates, sims))

        w = [s for _, s in kept]
        w = [x/(sum(w)+1e-9) for x in w]
        global_sd = robust_aggregate([u for u,_ in kept], w)
        accepted.append(len(kept))
        round_times.append(time.perf_counter()-t0)
        print(f"round {rnd+1:02d}  accepted={len(kept)}/{len(updates)}  time={round_times[-1]:.3f}s")

    torch.save(global_sd, f"{MODEL_DIR}/step5_global_model.pt")
    results = {
        'data_sharing_latency_ms_per_client': float(np.mean(round_times)*1000/max(N_CLIENTS,1)),
        'fl_round_ms_mean':                   float(np.mean(round_times)*1000),
        'fl_round_ms_p95':                    float(np.quantile(round_times,0.95)*1000) if round_times else float('nan'),
        'fl_avg_accepted_clients':            float(np.mean(accepted)) if accepted else 0.0,
    }
    print("\n===== STEP 5 FL METRICS =====")
    for k,v in results.items():
        print(f"{k:45s} {v:.4f}")
    pd.Series(results).to_csv(f"{OUT_DIR}/step5_metrics.csv")
    return results


if __name__ == "__main__":
    run()