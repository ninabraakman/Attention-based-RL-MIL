# Attention-based-RL-MIL
This repository implements an Attention-based Reinforcement Learning Multiple Instance Learning (RL-MIL) framework. The core framework, including MIL and RL-MIL with Epsilon-Greedy baselines, is directly adopted from the https://github.com/AlirezaZiabari/RL-MIL repository. This README details the specific modifications made to integrate two novel attention mechanisms and run the framework on The Open University Learning Analytics Dataset (OULAD) https://analyse.kmi.open.ac.uk/open-dataset. 

# Project Scope and Baselines
This project extends the existing RL-MIL framework by focussing on the integration and evaluation of attention mechanisms. The MIL model and the RL-MIL with Epsilon-Greedy models serve as baselines, consistent with the original repository. Consequently, ensemble MIL, the SimpleMLP basline, and multiple_run_* files from the original repository are not included. 

In this repository: 
- The Gated Attention mechanism is referred to as 'ilse attention', inspired by Ilse et al. (2018).
- The Mult-Head Attention mechanism is referred to as 'pham attention', inspired by: Pham et al. (2024). 

All RL-MIL models have been developed and tested specifically with the following configurations: 
- task_type="classification"
- rl_task_model="vanilla"
- sample_algorithm="without_replacement"
- prefix="loss"
- rl_model="policy_only"
- reg_alg="sum"

While other prefixes exist in the codebase from the original  repository, their compatibility with the attention models is not guaranteed without further modifications. They are retained for experimental purposes. 

# Significant Codebase Adjustments
The following are significant changes made to the original https://github.com/AlirezaZiabari/RL-MIL repository to incorporate the attention mechanisms and adapt to the OULAD dataset: 

From the original repo, these are all significant changes that were made in order to run the attention mechanisms: 
- models.py: added AttentionPolicyNetwork_ilse and AttentionPolicyNetwork_pham classes. These inherit from the PolicyNetwork class and implement their own forward() and get_last_attention_scores() methods to support attention mechanisms.
- utils.py: introduced create_bag_masks_randomized(). This function replaces the original create_bag_masks(). The change was necessary because the original MIL baseline selected the first k instances from each super-bag. Given the ordered nature of OULAD instances (e.g., demographic data appearing first), this would introduce  bias. To ensure fair comparison, all super-bags are now randomly shuffled before selecting k instances for the MIL bag. 
- configs.py: added specific configuration flags for 'ilse attention', including: temperature, is_linear_attention, attention_size and attention_dropout_p.
- RLMIL_Datasets.py: added RLMILDataset_attention(), which inherits from RLMILDataset. its primary extension is storing the original DataFrame, which is crucial for generating interpretability outputs related to the attention scores. 
- All original dataset preparation files were adjusted to handle a  tabular dataset format while maintaining consistent output structure. The detailed steps for dataset creation are provided below. 
- run_mil.py: now utilizes create_bag_masks_randomized() instead of the original create_bag_masks().
- run_rlmil.py: a get_dataloaders() function was added and is now used to address an error encountered with dataloaders during the main sweep.  
- All scripts (*.sh files) are based on the original scripts but rewritten to fit the dataset format.
- attention_ilse.py and attention_pham.py: are modified version of rlmil.py. They use class RLMILDataset_attention() to load the original dataframe, they pass bag_id as an additional argument when processing the dataframe, they utilize AttentionPolicyNetwork_ilse() and AttentionPolicyNetwork_pham() instead of PolicyNetwork() and they include generate_interepretability_outputs(), to ensure the creation of the output files for interpretability analysis. which makes sure the output file is created.

# Step-by-Step Usage Guide
Follow these steps to set up and run the attention-based RL-MIL framework:
1. Create a Python virtual environment ('venv')
2. Install all necessary dependencies using the requirements.txt file. 
3. Optional: Connect your Weights & Biases (wandb) account for experiment tracking. (https://wandb.ai/site)
2. Download OULAD data: Inside the data/ directory, create a new folder named 'raw'. Download all OULAD files from https://analyse.kmi.open.ac.uk/open-dataset and place them into the data/raw/ folder.
3. Initial data Exploration and Cleaning: Run the eda_OULAD.ipynb Jupyter Notebook. This notebook performs exploratory data analysis, checks and imputes null values, handles duplicates, and prepares the data for merging. All cleaned and prepared files will be saved in a new data/clean/ folder. 
4. Create Data Bags: Execute either createbags_aggregated.py or createbags_full.py, or both. createbags_full.py creates "raw" and "encoded" bags using all individual VLE interactions as seperate instances. createbags_aggregated creates "raw" and "encoded" bags by summing VLE clicks per activity type, providing aggregated instances. The raw bags are primarily for inspection and understanding the instance structure. The encoded bags are used for model training.
5. Prepare Data for Model Input: Run prepare_oulad_data.py. This script loads the previously created bags, pads them with zero's up to a fixed maximum number of instances per bag, adds a mask to distinguish real instances from padding, and generates a trian/validation/test split.
6. Run MIL Model (Baseline): The MIL model is typically run before the RL-MIL models to find the best configurations per pooling technique. Adjust and execute scripts/run_mil.sh. You can tune parameters like random seed and pooling techniques. Runs are saved in the /run folder, and logs can be found in the /logs folder.  
7. run the RL-MIL Models: After running the MIL baseline, you can execute the scripts for the RL-MIL models. Basic RL-MIL with Epsilon-Greedy: run_rlmil.sh. Ilse (Gated) Attention: attention_ilse.sh. Pham (Multi-Head) Attention attention_pham.sh.  
8. Gather Results: use results/gather_results.py to collect and consolidate the results from your model runs.
9. Analysis Scripts: The subquestion*.py files are designed for specific thesis questions but are generally used for: (1) Calculating averages and standard deviations across multiple seeds. (2) Comparing the results between the two dataset versions . (3) Evaluating models for interpretability, both visual and quantitative. (4) Calculating SHAP scores for the RL-MIL baseline.  


# References
- Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1802.04712
- Pham, H., Tan, Y., Singh, T., Pavlopoulos, V., & Patnayakuni, R. (2024). A multi-head attention-like feature selection approach for tabular data. Knowledge-Based Systems, 301, 112250. https://doi.org/10.1016/j.knosys.2024.112250
