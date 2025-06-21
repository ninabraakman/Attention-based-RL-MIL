# Attention-based-RL-MIL
Attention based RL-MIL
This entire framework is based on the https://github.com/AlirezaZiabari/RL-MIL repsoitory. This README will explain exactly what changes have been made in order to run the attention-based RL-MIL framework for the OULAD (https://analyse.kmi.open.ac.uk/open-dataset) dataset. 

For my thesis I only worked with the MIL model and the RL-MIL with Epsilon-Greedy as baselines. Furthermore, I did not work with ensamble MIL, the SimpleMLP baseline or any of the multiple_run_.. files, which is why they ar enot in this repsoitory. In the rest of this repository the Gated Attention mechanism will be reffered to as 'ilse attention', by it's inspiration: Ilse et al. (2018). The Mult-Head Attention mechanism will be reffered to as 'pham attention', by it's inspiration: Pham et al. (2024). 

All RL-MIL models have only been tested, and thus work with:
task_type="classification"
rl_task_model="vanilla"
sample_algorithm="without_replacement"
prefix="loss"
rl_model="policy_only"
reg_alg="sum"

Any other prefixes are not fully tested and it can't be said for certain that they work with the attention models withouth making any changes. They have been left in the files for anyone who want to experiment with them.

From the original repo, these are all significant changes that were made in order to run the attention mechanisms: 
- models.py: addition of AttentionPolicyNetwork_ilse and AttentionPolicyNetwork_pham have been added. They both inherit from the baseline PolicyNetwork class, and only have their own forward() and get_last_attention_scores() functions.
- utils.py: create_bag_masks_randomized() is added, this function is used instead of create_bag_masks() from the original repo. This has been done because the original MIL baseline framework selects the first k instances from each super-bag. Given that the OULAD instances are ordered (with demographic data appearing first), this approach would introduce significant bias. To ensure a fair comparison, all super-bags in our experiments were randomly shuffled before the selection of the k instances for the MIL bag.
- configs.py: the attention flags for 'attention ilse' have been added: temperature, is_linear_attention, attention_size, attention_dropout_p
- logger.py: no changes.
- RLMIL_Datasets.py: an RLMILDataset_attention() has been added. It inherits from RLMILDataset, the only thing it does extra is storing the original Dataframe, which is neccecary for generating the attention outputs file.
- All prepare dataset files have been changed to fit a tabular dataset. The output stays the same. The steps for creating the dataset will be explained below. 
- run_mil.py: create_bag_masks_randomized() is used instead of create_bag_masks()
- run_rlmil.py: a get_dataloaders() function was added and used due to an error with the dataloaders in the main sweep. 
- All scripts are based on the original scripts but rewritten to fit the dataset format.
- attention_ilse.py is run_rlmil.py with the following adjustments: it uses class RLMILDataset_attention() to load the original dataframe, it uses bag_id as extra argument when processing the dataframe, it uses AttentionPolicyNetwork_ilse() instead of PolicyNetwork(), addition of generate_interepretability_outputs() which makes sure the output file is created.
attention_pham.py: is run_rlmil.py with the following adjustments: it uses class RLMILDataset_attention() to load the original dataframe, it uses bag_id as extra argument when processing the dataframe, it uses AttentionPolicyNetwork_pham() instead of PolicyNetwork(), addition of generate_interepretability_outputs() which makes sure the output file is created.


1. To start working with the models first create a virtual envoirement and install the requirements.txt file. 
2. Connect your wandb
2. Then in the folder data you create a folder 'raw', in which you donwload the oulad files. 
3. Next you run eda_OULAD.ipynb which does some eda, checks and imputes some null values, checks for duplicates and makes sure the data is ready to be merged. All files are saved in a new folder 'clean'. 
4. Then you can run createbags_aggregated.py or createbags_full.py, these create both raw and encoded bags. Oulad_full uses all data from the OULAD dataset as seperate instances, oulad_aggregated sums over the clicks per activity type. The raw bags are mostly created to inspect the data later on and understand the instances. 
5. Next step is running prepare_oulad_data.py, which loads in the bags, pads them with zero's up untill the max instances per bag, adds a mask to indicate which instanc es are padding instances and creates a train/test/validation split. 
6. Now that the data is ready you can start running the MIL model. This is needed before running the RL-MIL models since the best configurations per pooling technique are found in the MIL runs. This model can be run by adjusting the scripts/run_mil.sh. You can tune for example on which seed you run the model, but also which pooling techniques you want to run on what dataset. The runs are saved in the /run folder, the logs of your run can be found in the /logs folder. 
7. After running MIL you can run the scripts for basic RL-MIL with Epsilon-Greedy: run_rlmil.sh, for ilse/Gated Attention: attention_ilse.sh or pham/Milti-Head Attention: attention_pham.py. 
8. After running all your models you can use results/gather_results.py to gather all results. 
9. The subquestion*.py files are specifically created to answer the questions in my thesis but can be used to: get the averages and stdev over multiple runs, compare the results over the two dataset versions and evaluate the models on interpretability. It also calculates SHAP scores for the RL-MIL baseline. 



Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1802.04712
Pham, H., Tan, Y., Singh, T., Pavlopoulos, V., & Patnayakuni, R. (2024). A multi-head attention-like feature selection approach for tabular data. Knowledge-Based Systems, 301, 112250. https://doi.org/10.1016/j.knosys.2024.112250
