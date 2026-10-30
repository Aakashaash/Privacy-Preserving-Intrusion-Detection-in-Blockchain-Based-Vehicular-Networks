import os
import step1_registration_auth
import step4_smote
import step2_geospatial_graph
import step3_sdn_controller
import step4_tvgqnn
import step4_tgn_baseline
import step5_federated
import step6_trust
import step7_blockchain
import final_metrics

USE_SMOTE = os.getenv("USE_SMOTE", "1") == "1"

if __name__ == "__main__":
    print("\n[STEP 1] ECC + PUF + Zero-Trust Authentication")
    step1_registration_auth.run()

    if USE_SMOTE:
        print("\n[SMOTE] Class augmentation")
        step4_smote.run()

    print("\n[STEP 2] Geospatial Hierarchical Graph Routing")
    step2_geospatial_graph.run()

    print("\n[STEP 3] SDN Controller Network Analysis")
    step3_sdn_controller.run()

    print("\n[STEP 4] TVGQNN Detection")
    step4_tvgqnn.run()

    print("\n[STEP 4b] TGN Baseline (comparison)")
    step4_tgn_baseline.run()

    print("\n[STEP 5] Robust Personalized Federated Learning")
    step5_federated.run()

    print("\n[STEP 6] Uncertainty Fusion + Temporal Graph Trust")
    step6_trust.run()

    print("\n[STEP 7] Reputation-Weighted PoS Fault-Tolerant Blockchain")
    step7_blockchain.run()

    print("\n[FINAL] Aggregate All Metrics")
    final_metrics.run()