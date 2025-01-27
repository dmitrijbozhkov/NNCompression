from models.spsa import SPSA
from study.configs.config_base import RunnerConfigDict
from runners.runner_base import Runner
from runners.runner_evaluate import RunnerEval, RunnerQuantEval, RunnerEarlyStopping
from runners.runner_train import RunnerTrain, RunnerSchedule, RunnerTrainEpoch


def create_runner(runner_config_dict: RunnerConfigDict, checkpoint=None) -> Runner:
    """
    Create Runner from full configuration dictionary

    :param runner_config_dict: RunnerConfigDict that contains all settings to initialize Runner and its plugins
    :param checkpoint: Checkpoint to load
    :returns: initialized Runner
    """
    plugins = []

    if isinstance(runner_config_dict["optimizer"], SPSA):
        train_plugin = RunnerTrainEpoch(**runner_config_dict)
    else:
        if runner_config_dict["scheduler"]:
            train_plugin = RunnerSchedule(**runner_config_dict)
        else:
            train_plugin = RunnerTrain(**runner_config_dict)

    plugins.append(train_plugin)

    if runner_config_dict["runner_config"]["is_quantize"]:
        eval_plugin = RunnerQuantEval(**runner_config_dict)
    else:
        eval_plugin = RunnerEval(**runner_config_dict)

    if runner_config_dict["runner_config"]["is_early_stopping"]:
        eval_plugin = RunnerEarlyStopping(eval_plugin, **runner_config_dict)

    plugins.append(eval_plugin)

    runner = Runner(
        plugins,
        True,
        **runner_config_dict
    )

    if checkpoint:
        runner.load_checkpoint(checkpoint)

    return runner
