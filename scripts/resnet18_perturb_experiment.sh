#!/bin/bash --login
#SBATCH --job-name=web-login/sys/myjobs/default
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output test_serial-job_%j.out
#SBATCH --error test_serial-job_%j.err
#SBATCH --gres=gpu:rtx3090:1

# run program
echo "I am running on $HOSTNAME"

nvidia-smi

pwd

cd /bigwork/nhwpbozd/wsdm-cup

module load Miniforge3

conda activate wsdm

python main.py --config-name="resnet18_perturb_experiment" hpo_config.database="sqlite:////project/NHWP25179/run_out/hpo/resnet18_perturb_experiment.db" training_config.output="/project/NHWP25179/run_out/outputs"
