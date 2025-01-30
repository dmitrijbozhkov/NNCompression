./cluster_startup.sh

python main.py --config-name="resnet18_perturb_experiment" hpo_config.database="sqlite:////project/NHWP25179/run_out/hpo/resnet18_perturb_experiment.db" training_config.output="/project/NHWP25179/run_out/outputs"
