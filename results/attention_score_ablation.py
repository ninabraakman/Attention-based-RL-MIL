# This file compares the attention scores before and after the second softmax of the PHAM/Multi-Head Attention model.
import pandas as pd
import numpy as np
import os
import sys

OUTPUT_DIR = 'final_report_oulad_aggregated/'
SEED_TO_ANALYZE = 8
BASE_PATH = f'/projects/prjs1491/Attention-based-RL-MIL/runs/classification/seed_{SEED_TO_ANALYZE}/oulad_aggregated/instances/tabular/label/bag_size_20/repset_22_16_22/'
PHAM_RUN_DIR = os.path.join(BASE_PATH, 'neg_policy_only_loss_attention_pham_reg_sum_sample_without_replacement/')

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pham_file = os.path.join(PHAM_RUN_DIR, 'attention_pham_outputs.csv')
    try:
        pham_df_full = pd.read_csv(pham_file)
    except FileNotFoundError as e:
        print(f"FATAL: Could not find PHAM output file: {e}"); sys.exit(1)
    pham_df_real = pham_df_full[pham_df_full['is_padding_instance'] == False].copy()
    if pham_df_real.empty:
        print(f"FATAL: No non-padding instances found in PHAM file. Aborting."); sys.exit(1)
    pre_softmax_scores = pham_df_real['raw_attention_score']
    final_attention_scores = pham_df_real['attention_score']

    stats_data = {
        "Statistic": ["Minimum Value", "Maximum Value", "Range (Max - Min)", "Standard Deviation"],
        "Pre-Softmax Scores": [
            pre_softmax_scores.min(),
            pre_softmax_scores.max(),
            pre_softmax_scores.max() - pre_softmax_scores.min(),
            pre_softmax_scores.std()
        ],
        "Final Attention Scores": [
            final_attention_scores.min(),
            final_attention_scores.max(),
            final_attention_scores.max() - final_attention_scores.min(),
            final_attention_scores.std()
        ]
    }
    
    summary_df = pd.DataFrame(stats_data).set_index('Statistic')
    output_path = os.path.join(OUTPUT_DIR, 'attention_score_ablation_summary.csv')
    summary_df.to_csv(output_path)