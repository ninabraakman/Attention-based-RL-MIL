import pickle
import random
import os

# === CONFIGURATION ===
FULL_INPUT      = "data/oulad/oulad_full.pkl"
AGG_INPUT       = "data/oulad/oulad_aggregated.pkl"
FULL_SUBSET     = "data/oulad/oulad_full_subset.pkl"
AGG_SUBSET      = "data/oulad/oulad_aggregated_subset.pkl"
INDICES_PATH    = "data/oulad/subset_indices.pkl"
SUBSET_SIZE     = 2738
RANDOM_SEED     = 42

# set seed for reproducibility
random.seed(RANDOM_SEED)

# load full bag IDs
with open(FULL_INPUT, "rb") as f:
    full_data = pickle.load(f)
full_bag_ids = full_data["bag_ids"]
N = len(full_bag_ids)

# sample indices
indices = random.sample(range(N), SUBSET_SIZE)
os.makedirs(os.path.dirname(INDICES_PATH), exist_ok=True)
with open(INDICES_PATH, "wb") as f:
    # save the indices and seed so you can re-generate
    pickle.dump({"seed": RANDOM_SEED, "indices": indices}, f)

def make_subset(input_path, output_path, indices):
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    subset = {
        "bags":         [ data["bags"][i]       for i in indices ],
        "labels":       [ data["labels"][i]     for i in indices ],
        "bag_ids":      [ data["bag_ids"][i]    for i in indices ],
        "cat_encoders": data.get("cat_encoders"),
        "num_scalers":  data.get("num_scalers"),
        # include date_scaler if present
        **({"date_scaler": data["date_scaler"]} if "date_scaler" in data else {})
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(subset, f)
    print(f"✅ Saved subset ({len(indices)} bags) to {output_path}")

# build both subsets with the same indices
make_subset(FULL_INPUT, FULL_SUBSET, indices)
make_subset(AGG_INPUT, AGG_SUBSET, indices)
