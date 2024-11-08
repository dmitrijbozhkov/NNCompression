from runner import Runner
from study import Study
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    config, search_space = Study.prepare_config(parser)
    study = Study(config, search_space)
    study.search()

    # runner = Runner.from_config(config)
    # run_df = runner.train()
    # runner.save_train_run(run_df)
    # runner.kmeans_quantize(8)
    # print(runner.test())
