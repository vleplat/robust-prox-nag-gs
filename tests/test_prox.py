import torch

from rpnaggs.optim.prox import make_prox_operator
from rpnaggs.optim.transforms import soft_threshold


def test_no_prox_returns_input():
    tensors = [torch.tensor([1.0, -2.0]), torch.tensor([[3.0, -4.0]])]
    prox = make_prox_operator("none")
    outputs = prox.prox([tensor.clone() for tensor in tensors], step_size=0.5)

    assert all(torch.allclose(output, tensor) for output, tensor in zip(outputs, tensors))


def test_l1_prox_matches_soft_thresholding():
    tensor = torch.tensor([1.0, -0.25, -2.0])
    prox = make_prox_operator("l1", lam=2.0)
    output = prox.prox([tensor.clone()], step_size=0.2)[0]

    assert torch.allclose(output, soft_threshold(tensor, 0.4))


def test_l2_prox_matches_closed_form():
    tensor = torch.tensor([2.0, -4.0])
    prox = make_prox_operator("l2", lam=3.0)
    output = prox.prox([tensor.clone()], step_size=0.5)[0]

    assert torch.allclose(output, tensor / (1.0 + 0.5 * 3.0))


def test_elastic_net_prox_matches_scaled_soft_thresholding():
    tensor = torch.tensor([2.0, -1.0, 0.2])
    prox = make_prox_operator("elastic_net", l1=1.5, l2=2.0)
    output = prox.prox([tensor.clone()], step_size=0.4)[0]
    expected = soft_threshold(tensor, 0.4 * 1.5) / (1.0 + 0.4 * 2.0)

    assert torch.allclose(output, expected)


def test_group_lasso_prox_shrinks_rows_and_filters():
    prox = make_prox_operator("group_lasso", group=2.0)

    linear_weight = torch.tensor([[3.0, 4.0], [0.3, 0.4]])
    linear_output = prox.prox([linear_weight.clone()], step_size=0.5)[0]
    expected_linear = torch.tensor([[2.4, 3.2], [0.0, 0.0]])
    assert torch.allclose(linear_output, expected_linear, atol=1e-6)

    conv_weight = torch.tensor([[[[3.0, 4.0]]], [[[0.3, 0.4]]]])
    conv_output = prox.prox([conv_weight.clone()], step_size=0.5)[0]
    expected_conv = torch.tensor([[[[2.4, 3.2]]], [[[0.0, 0.0]]]])
    assert torch.allclose(conv_output, expected_conv, atol=1e-6)
