./scripts/cluster_scripts/cluster_startup.sh

python main.py --config-name="lenet5_perturb_experiment" hpo_config.database="sqlite:////project/NHWP25179/run_out/hpo/lenet5_perturb_experiment.db" training_config.output="/project/NHWP25179/run_out/outputs"
