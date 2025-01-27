from runners.runner_base import Runner, RunnerBase


class RunnerEval(RunnerBase):
    """Runner plugin for model evaluation and saving"""

    def __init__(self, runner_config, logger, **kwargs):
        self.evaluate_step = runner_config["evaluate_step"]
        self.evaluate_datasets = runner_config["evaluate_datasets"]
        self.evaluate_metrics = runner_config["evaluate_metrics"]
        self.logger = logger
        self.epoch_acc = 0
        self.curr_checkpoint = "00"

    def eval_model(self, runner: Runner, template: str):
        """
        Evaluate model on given datasets and metrics and record into current runner event

        :param runner: Current model Runner
        :param template: Column template for recording evaluation
        """
        for metric in self.evaluate_metrics:
            for dataset in self.evaluate_datasets:
                if metric == "accuracy":
                    eval_value = runner.eval_accuracy(dataset)
                    eval_name = template.format(metric=metric, dataset=dataset)
                else:
                    raise ValueError(f"No such metric: {metric}")
                self.logger.info(f"Eval {eval_name}: {eval_value}")
                runner.curr_run_metadata[eval_name] = eval_value


    def is_eval_skip(self):
        """
        Checks if this epochs evaluation should be skipped

        :returns: bool
        """
        self.epoch_acc += 1
        if self.epoch_acc < self.evaluate_step:
            return True
        else:
            self.epoch_acc = 0
            return False


    def save_model(self, runner: Runner):
        curr_checkpoint = str(runner.total_epochs_trained).zfill(2)
        runner.save_checkpoint(curr_checkpoint)
        return curr_checkpoint


    def epoch_end(self, runner: Runner, epoch):
        """
        Evaluate model on given metric and dataset

        :returns: Returns True to signal that evaluation is skipped
        """
        if self.is_eval_skip():
            return True

        self.curr_checkpoint = self.save_model(runner)
        self.eval_model(runner, "{dataset}_{metric}")


    def after_train_episode(self, runner: Runner):
        """Ensure that last epoch is evaluated"""
        if self.epoch_acc != 0:
            self.epoch_acc = self.evaluate_step

        self.epoch_end(runner, -1)


class RunnerQuantEval(RunnerEval):
    """Runner plugin for evaluating model with quantization"""

    def __init__(self, runner_config, **kwargs):
        super().__init__(runner_config, **kwargs)
        self.quantization_type = runner_config["quantization_type"]
        self.quantize_levels = runner_config["quantize_levels"]


    def after_train_episode(self, runner: Runner):
        for level in self.quantize_levels:
            quant_name = f"quant_{level}_"
            quant_metric_template = quant_name + "{dataset}_{metric}"

            quant_centers = runner.quantize(level)
            super().eval_model(runner, quant_metric_template)

            runner.curr_run_metadata[quant_name + "centers"] = quant_centers
            runner.load_checkpoint(self.curr_checkpoint)


class RunnerEarlyStopping(RunnerBase):
    """Runner plugin that performs early stopping after evaluation of the model"""

    def __init__(self, eval_plugin: RunnerEval | RunnerQuantEval, runner_config: dict, logger, **kwargs):
        super().__init__(**kwargs)
        self.eval_plugin = eval_plugin
        self.early_stopping_min_delta = runner_config["early_stopping_min_delta"]
        self.early_stopping_tolerance = runner_config["early_stopping_tolerance"]
        self.early_stopping_eval_metric = runner_config["early_stopping_eval_metric"]
        self._tolerance_counter = self.early_stopping_tolerance
        self.logger = logger
        self._last_eval_val = 0


    def epoch_end(self, runner: Runner, epoch):
        """Perform evaluation and do early stopping if conditions are met"""
        is_skip = self.eval_plugin.epoch_end(runner, epoch)
        if is_skip:
            return is_skip

        early_stop_val = runner.curr_run_metadata[self.early_stopping_eval_metric]

        if (early_stop_val - self._last_eval_val) <= self.early_stopping_min_delta:
            self._tolerance_counter -= 1
            self.logger.info(f"Early stopping patience: {self._tolerance_counter}")

        if self._tolerance_counter == 0:
            runner.is_train_continue = False
            self.logger.info(f"Early stopped!")

        self._last_eval_val = early_stop_val


    def epoch_start(self, runner, epoch):
        return self.eval_plugin.epoch_start(runner, epoch)

    def before_train_episode(self, runner):
        return self.eval_plugin.before_train_episode(runner)

    def after_train_episode(self, runner):
        return self.eval_plugin.after_train_episode(runner)

    def epoch_batch(self, runner, epoch, batch_idx, data, target):
        return self.eval_plugin.epoch_batch(runner, epoch, batch_idx, data, target)
