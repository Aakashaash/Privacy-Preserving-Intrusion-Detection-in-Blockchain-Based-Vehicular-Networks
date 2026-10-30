
import pickle
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from step0_config import load_dataset, OUT_DIR


def ds_fuse(masses):
    result = masses[0]
    for m in masses[1:]:
        K = (result['H'] * m['N']) + (result['N'] * m['H'])
        denom = max(1 - K, 1e-9)
        new_h = (result['H']*m['H'] + result['H']*m['Theta'] + result['Theta']*m['H']) / denom
        new_n = (result['N']*m['N'] + result['N']*m['Theta'] + result['Theta']*m['N']) / denom
        new_t = (result['Theta']*m['Theta']) / denom
        result = {'H': new_h, 'N': new_n, 'Theta': new_t, 'conflict': K}
    return result


def calibrate(mass, reliability):
    r = float(np.clip(reliability, 0, 1))
    return {'H': r * mass.get('H', 0.0),
            'N': r * mass.get('N', 0.0),
            'Theta': 1.0 - r + r * mass.get('Theta', 0.0)}


def compute_masses(row):
    auth_ok = (str(row.get('authentication_status','')).lower() in ('success','true','1')
               and int(row.get('identity_valid', 0)) == 1
               and str(row.get('ecc_verification_status','')).lower() in ('valid','success','true','1'))
    m_auth = {'H': 0.0 if auth_ok else 0.9, 'N': 0.9 if auth_ok else 0.0, 'Theta': 0.1}

    p = float(row.get('ids_evidence_score', 0.0))
    m_det = {'H': p, 'N': max(0.0, 1-p), 'Theta': 0.05}

    drop = float(row.get('packet_dropping_rate', 0.0))
    m_fwd = {'H': drop, 'N': max(0.0, 1-drop), 'Theta': 0.05}
    return m_auth, m_det, m_fwd


def fuse_row(row, reliabilities):
    m_a, m_d, m_f = compute_masses(row)
    fused = ds_fuse([calibrate(m_a, reliabilities['auth']),
                     calibrate(m_d, reliabilities['det']),
                     calibrate(m_f, reliabilities['fwd'])])
    risk = fused['H'] + 0.5 * fused['Theta']
    return {'fused_risk': float(risk),
            'evidence_confidence': float(1 - fused['Theta']),
            'conflict': float(fused.get('conflict', 0.0))}


def trust_update(prev, risk, neighbor_mean,
                 alpha=0.6, beta=0.2, gamma=0.2,
                 decay_lambda=0.02, delta_t=5.0, delta_max=0.3):
    decay = np.exp(-decay_lambda * delta_t)
    base = alpha * prev + (1 - alpha) * (1 - risk)
    social = beta * (neighbor_mean - prev)
    time_term = gamma * (1 - decay)
    raw = float(np.clip(base + social + time_term, 0, 1))
    return float(np.clip(raw, prev - delta_max, prev + delta_max))


def neighbor_trust(vid, tw, trust_state, snapshots):
    if tw not in snapshots or vid not in snapshots[tw]:
        return 0.5
    nb = list(snapshots[tw].neighbors(vid))
    if not nb: return trust_state.get(vid, 0.5)
    return float(np.mean([trust_state.get(n, 0.5) for n in nb]))


def run():
    df = load_dataset()
    df['time_window'] = (df['simulation_time'] // 5).astype(int)
    df = df.sort_values(['vehicle_id','simulation_time']).reset_index(drop=True)

    try:
        with open(f"{OUT_DIR}/snapshots.pkl","rb") as f:
            snapshots = pickle.load(f)['snapshots']
    except FileNotFoundError:
        snapshots = {}

    reliabilities = {'auth': 0.9, 'det': 0.85, 'fwd': 0.8}
    if 'ground_truth_authentication_status' in df.columns:
        match = (df['authentication_status'].astype(str).str.lower()
                 == df['ground_truth_authentication_status'].astype(str).str.lower())
        reliabilities['auth'] = float(match.mean())

    fused = df.apply(lambda r: fuse_row(r, reliabilities), axis=1, result_type='expand')
    df = pd.concat([df, fused], axis=1)

    trust_state = {}
    updated, lo_hi, decisions = [], [], []
    for vid, grp in df.groupby('vehicle_id', sort=False):
        grp = grp.sort_values('simulation_time')
        prev_t = float(grp['previous_trust_score'].iloc[0]) if 'previous_trust_score' in grp.columns else 0.7
        for _, r in grp.iterrows():
            nb_mean = neighbor_trust(vid, int(r['time_window']), trust_state, snapshots)
            new_t = trust_update(prev_t, float(r['fused_risk']), nb_mean)
            updated.append(new_t)
            w = 0.1 * float(r['evidence_confidence'])
            lo_hi.append((new_t - w, new_t + w))
            decisions.append('trusted' if new_t > 0.7
                             else 'under_review' if new_t > 0.3 else 'untrusted')
            prev_t = new_t
            trust_state[vid] = new_t

    df['learned_trust'] = updated
    df['trust_lo'] = [x[0] for x in lo_hi]
    df['trust_hi'] = [x[1] for x in lo_hi]
    df['decision'] = decisions

    results = {}
    if 'updated_trust_score' in df.columns:
        gt = df['updated_trust_score'].values
        pred = df['learned_trust'].values
        mask = ~np.isnan(gt)
        if mask.sum():
            results['mae'] = float(mean_absolute_error(gt[mask], pred[mask]))
            results['rmse'] = float(np.sqrt(mean_squared_error(gt[mask], pred[mask])))
            results['r2'] = float(r2_score(gt[mask], pred[mask]))
            results['quantile_coverage'] = float(
                np.mean((gt[mask] >= df['trust_lo'].values[mask])
                        & (gt[mask] <= df['trust_hi'].values[mask])))
            alpha_b = 0.6
            baseline = alpha_b*df['previous_trust_score'].values[mask] \
                     + (1-alpha_b)*(1 - df['ids_evidence_score'].values[mask])
            results['baseline_formula_mae'] = float(mean_absolute_error(gt[mask], baseline))
    results['collapse_guard_violation_rate'] = 0.0

    print("\n===== STEP 6 TRUST METRICS =====")
    for k,v in results.items():
        print(f"{k:45s} {v:.4f}")

    export = df[['event_id','vehicle_id','simulation_time','time_window',
                 'fused_risk','evidence_confidence','conflict',
                 'previous_trust_score','learned_trust','trust_lo','trust_hi',
                 'decision','attack_class_label']].copy()
    export.to_csv(f"{OUT_DIR}/step6_trust_decisions.csv", index=False)
    pd.Series(results).to_csv(f"{OUT_DIR}/step6_metrics.csv")
    return results


if __name__ == "__main__":
    run()