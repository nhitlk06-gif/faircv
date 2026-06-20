CLI export output directory.

After running:
    python scripts/run_audit.py --csv data/FairCVdb.csv --export results/

The following files will be written here:
    master_comparison.csv        All models x settings x labels
    before_after_LR.csv          Before/After comparison for LR
    before_after_RF.csv          Before/After comparison for RF
    before_after_MLP.csv         Before/After comparison for MLP
    shap_global_LR.csv           SHAP global importance (LR)
    shap_global_RF.csv           SHAP global importance (RF)
    shap_dp_decomp_LR.csv        DP Gap decomposition (LR)
    shap_dp_decomp_RF.csv        DP Gap decomposition (RF)
    fusion_results.csv           Fusion experiment results (--sbert only)
