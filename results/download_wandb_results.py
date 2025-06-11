# download_wandb_results.py
import wandb
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION TO EDIT ---

# 1. Your W&B username or entity name
WANDB_ENTITY = "your-entity-name"  # e.g., "nbraakman"

# 2. Your W&B project name
WANDB_PROJECT = "your-project-name" # e.g., "Attention-based-RL-MIL"

# 3. The list of SWEEP IDs you want to download results from.
#    You get these from the W&B dashboard URL.
#    e.g., https://wandb.ai/your-entity-name/your-project-name/sweeps/THIS_IS_THE_ID
SWEEP_IDS = [
    "sweep_id_for_pham_mean_agg",
    "sweep_id_for_pham_max_agg",
    "sweep_id_for_ilse_mean_agg",
    # ...and so on for all your new sweeps
]

# 4. The name of the output file
OUTPUT_CSV = "results/all_wandb_runs_raw.csv"

# --- END OF CONFIGURATION ---


def download_sweep_results():
    """
    Connects to the W&B API and downloads the config and summary metrics
    for every run associated with the specified sweep IDs.
    """
    try:
        api = wandb.Api()
        print("Successfully connected to W&B API.")
    except Exception as e:
        print(f"Error connecting to W&B API: {e}")
        print("Please ensure you are logged in ('wandb login') and the entity/project names are correct.")
        return

    all_runs_data = []

    print(f"Fetching data for {len(SWEEP_IDS)} sweeps...")
    for sweep_id in tqdm(SWEEP_IDS, desc="Processing sweeps"):
        try:
            sweep_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}"
            sweep = api.sweep(sweep_path)
        except wandb.errors.CommError:
            print(f"\nWarning: Could not find sweep '{sweep_path}'. Skipping.")
            continue
            
        for run in tqdm(sweep.runs, desc=f"Runs in {sweep.name}", leave=False):
            # Combine the run's configuration (hyperparameters) and summary (results)
            run_data = {**run.config, **run.summary}
            
            # You can add more specific data if needed, e.g., run.name
            run_data['run_name'] = run.name
            run_data['sweep_name'] = sweep.name
            
            all_runs_data.append(run_data)

    if not all_runs_data:
        print("No runs found for the provided sweep IDs. Please check your configuration.")
        return

    print(f"\nProcessed a total of {len(all_runs_data)} runs.")
    
    # Convert the list of dictionaries to a pandas DataFrame and save
    results_df = pd.DataFrame(all_runs_data)
    
    # Clean up column names that W&B might add underscores to
    results_df.rename(columns=lambda c: c.replace('_', '/', 1) if c.startswith('_') else c, inplace=True)

    
    print(f"Saving all results to '{OUTPUT_CSV}'...")
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("Download complete.")


if __name__ == "__main__":
    download_sweep_results()