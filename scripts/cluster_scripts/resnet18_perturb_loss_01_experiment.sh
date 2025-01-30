./scripts/cluster_scripts/cluster_startup.sh

python main.py --config-name="resnet18_perturb_loss_01_experiment" hpo_config.database="sqlite:////bigwork/nhwpbozd/hpo/resnet18_perturb_loss_01_experiment.db" training_config.output="/bigwork/nhwpbozd/run_out/outputs"
