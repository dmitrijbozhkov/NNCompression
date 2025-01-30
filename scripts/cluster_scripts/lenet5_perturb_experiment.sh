./scripts/cluster_scripts/cluster_startup.sh

python main.py --config-name="lenet5_perturb_experiment" hpo_config.database="sqlite:////bigwork/nhwpbozd/hpo/lenet5_perturb_experiment.db" training_config.output="/bigwork/nhwpbozd/run_out/outputs"
