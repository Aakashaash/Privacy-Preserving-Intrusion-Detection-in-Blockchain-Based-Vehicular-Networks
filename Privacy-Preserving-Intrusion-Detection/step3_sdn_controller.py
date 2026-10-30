
import os
import pandas as pd
from step0_config import load_dataset, OUT_DIR

NET_COLS = [
    'packet_delivery_ratio','link_quality','channel_busy_ratio',
    'communication_delay','packet_forwarded','packet_dropped',
    'link_stability','predicted_link_duration','proximity_score',
    'neighbor_count','neighbor_density','distance_to_receiver',
    'message_size_bytes','message_frequency'
]


# ---------- (Optional) Ryu controller app ----------
RYU_APP = '''
# Save as ryu_app.py and run: ryu-manager ryu_app.py
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

class NetworkMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == CONFIG_DISPATCHER and dp.id in self.datapaths:
            del self.datapaths[dp.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)

    def _request_stats(self, dp):
        ofp, ofp_parser = dp.ofproto, dp.ofproto_parser
        dp.send_msg(ofp_parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY))
'''


def compute_network_metrics(df):
    pdr = float(df['packet_delivery_ratio'].mean()) if 'packet_delivery_ratio' in df.columns else float('nan')
    msg_bytes = df.get('message_size_bytes', pd.Series([0])).fillna(0)
    elapsed = max(df['simulation_time'].max() - df['simulation_time'].min(), 1e-6)
    throughput = float(msg_bytes.sum() / elapsed)
    ctrl = df.get('packet_forwarded', pd.Series([0])).fillna(0).sum()
    overhead_ratio = float(ctrl / max(len(df), 1))

    if 'message_id' in df.columns and 'message_size_bytes' in df.columns:
        unique_payload = df.drop_duplicates('message_id')['message_size_bytes'].sum()
        total_bytes = df['message_size_bytes'].sum()
        overhead_bytes = float(total_bytes - unique_payload)
    else:
        overhead_bytes = float('nan')

    return {
        'packet_delivery_ratio':            pdr,
        'network_throughput_bytes_per_sec': throughput,
        'communication_overhead_ratio':     overhead_ratio,
        'communication_overhead_bytes':     overhead_bytes,
        'avg_communication_delay_ms':       float(df.get('communication_delay', pd.Series([0])).mean()*1000),
        'avg_link_quality':                 float(df.get('link_quality', pd.Series([0])).mean()),
        'avg_channel_busy_ratio':           float(df.get('channel_busy_ratio', pd.Series([0])).mean()),
    }


def run():
    df = load_dataset()
    df['time_window'] = (df['simulation_time'] // 5).astype(int)

    cols = [c for c in NET_COLS if c in df.columns]
    agg = df.groupby(['vehicle_id','time_window'])[cols].agg(['mean','std','max'])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().fillna(0)

    if {'packet_forwarded_mean','packet_dropped_mean'} <= set(agg.columns):
        fwd = agg['packet_forwarded_mean']; drp = agg['packet_dropped_mean']
        agg['forward_success_ratio'] = fwd / (fwd + drp + 1e-6)
        agg['drop_ratio'] = drp / (fwd + drp + 1e-6)

    agg.to_csv(f"{OUT_DIR}/step3_network_features.csv", index=False)

    results = compute_network_metrics(df)
    print("\n===== STEP 3 SDN NETWORK METRICS =====")
    for k, v in results.items():
        print(f"{k:45s} {v:.4f}")
    pd.Series(results).to_csv(f"{OUT_DIR}/step3_metrics.csv")

    with open(f"{OUT_DIR}/ryu_app_template.txt","w") as f:
        f.write(RYU_APP)
    print("[Step3] Ryu app template written for optional deployment")
    return results


if __name__ == "__main__":
    run()