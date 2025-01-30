from models.quantization.quant_perform import _quantize, _quantize_nearest, _quantize_up, _quantize_down, _quantize_stochastic
import pytest
import torch

@pytest.fixture(scope="function")
def unquantized_tensor():
    """
    Tensor for quantization
    """
    return torch.tensor([
        [18, 3, -10],
        [-25, -12, -29],
        [-1, 2, -30],
        [25, 0, -35],
        [10, -5, 200]
    ])

@pytest.fixture
def quantization_levels():
    """
    Quantization levels
    """
    return torch.tensor([
        -30,
        -24,
        -18,
        -12,
        -3,
        4,
        15,
        20
    ])


@pytest.fixture
def quant_diff(unquantized_tensor, quantization_levels):
    """
    Difference matrix between each quantization level and tensor value
    """
    tensor_rows = unquantized_tensor.view(-1, 1)
    quant_columns = quantization_levels.view(1, -1)
    return tensor_rows - quant_columns


def test_quantize_should_change_tensor_with_quantized_levels(unquantized_tensor, quantization_levels):
    """_quantize function should copy quantized values into original tensor"""
    quant_idx = torch.tensor([
        7,
        5,
        3,
        1,
        3,
        0,
        4,
        5,
        0,
        7,
        4,
        0,
        6,
        4,
        7
    ])

    expect_tensor = quantization_levels[quant_idx].view(unquantized_tensor.shape)

    print(expect_tensor)

    _quantize(unquantized_tensor, quantization_levels, quant_idx)

    assert torch.equal(unquantized_tensor, expect_tensor)


def test_quantize_nearest_should_quantize_to_nearest_value(unquantized_tensor, quant_diff, quantization_levels):
    """_quantize_nearest should quantize tensor to nearest level"""

    expected_nearest = torch.tensor([
        [ 20,   4, -12],
        [-24, -12, -30],
        [ -3,   4, -30],
        [ 20,  -3, -30],
        [ 15,  -3,  20]
    ])

    _quantize_nearest(unquantized_tensor, quant_diff, quantization_levels)

    assert torch.equal(expected_nearest, unquantized_tensor)


def test_quantize_up_should_quantize_values_up_next_level(unquantized_tensor, quant_diff, quantization_levels):
    """_quantize_up should always quantize values to nearest highest value in levels"""

    expected_up = torch.tensor([
        [20,   4, -3],
        [-24, -12, -24],
        [ 4,   4, -30],
        [20,  4, -30],
        [15,  -3,  20]
    ])

    _quantize_up(unquantized_tensor, quant_diff, quantization_levels)

    assert torch.equal(expected_up, unquantized_tensor)


def test_qauntize_down_should_quantize_values_down_level(unquantized_tensor, quant_diff, quantization_levels):
    """_quantize_down should always pick smallest value to assign level"""

    expected_down = torch.tensor([
        [15, -3, -12],
        [-30, -12, -30],
        [ -3, -3, -30],
        [ 20, -3, -30],
        [ 4, -12, 20]
    ])

    _quantize_down(unquantized_tensor, quant_diff, quantization_levels)

    assert torch.equal(expected_down, unquantized_tensor)


def test_quantize_stochastic_should_quantize_values_stochastically(unquantized_tensor, quant_diff, quantization_levels):
    """
    _quantize_stochastic should decide how to quantize based on distribution based on distance to quantization values
    """

    _quantize_stochastic(unquantized_tensor, quant_diff, quantization_levels)

    print(unquantized_tensor)

    assert True # TODO Think about good tests for this
