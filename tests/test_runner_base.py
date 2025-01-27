from runner_base import RunnerBase
from utils import get_data
from pathlib import Path
from unittest.mock import MagicMock, call
import tempfile
import pytest

@pytest.fixture
def cifar10_dataset():
    return get_data("cifar10", 32)

@pytest.fixture
def dummy_result_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def empty_runner_base(cifar10_dataset, dummy_result_path):
    dataset = cifar10_dataset
    return RunnerBase(
        None,
        None,
        dataset,
        None,
        None,
        dummy_result_path,
        None,
        None
    )

@pytest.fixture
def empty_runner_base_mock(empty_runner_base):
    empty_runner_base.before_train_episode = MagicMock(name="before_train_episode")
    empty_runner_base.after_train_episode = MagicMock(name="after_train_episode")
    empty_runner_base.epoch_start = MagicMock(name="epoch_start")
    empty_runner_base.epoch_end = MagicMock(name="epoch_end")
    empty_runner_base.epoch_batch = MagicMock(name="epoch_batch")
    return empty_runner_base


def test_RunnerBase_train_should_call_before_train_episode_once(empty_runner_base_mock):

    empty_runner_base_mock.train(2)

    empty_runner_base_mock.before_train_episode.assert_called_once()

def test_RunnerBase_train_should_call_after_train_episode_once(empty_runner_base_mock):

    empty_runner_base_mock.train(2)

    empty_runner_base_mock.after_train_episode.assert_called_once()

def test_RunnerBase_train_should_call_epoch_start_for_each_epoch(empty_runner_base_mock):

    empty_runner_base_mock.train(2)

    calls = [
        call.epoch_start(0),
        call.epoch_start(1)
    ]

    empty_runner_base_mock.epoch_start.assert_has_calls(calls, any_order=False)
