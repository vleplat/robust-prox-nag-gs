import math
from copy import deepcopy

import torch
import torch.nn as nn

from rpnaggs.algorithms.robust_proxnaggs import RobustProxNAGGSAlgorithm
from rpnaggs.config import build_arg_parser, config_from_args
from rpnaggs.optim.prox import make_prox_operator


def test_config_legacy_eta_maps_to_mu_hat():
    parser = build_arg_parser()
    args = parser.parse_args(["--algorithm", "robust_proxnaggs", "--pnaggs-a", "0.2", "--pnaggs-eta", "0.005"])
    config = config_from_args(args)
    assert math.isclose(config.pnaggs_mu_hat, 40.0, rel_tol=1e-9)


def test_robust_proxnaggs_uses_mu_hat_step_size():
    model = nn.Linear(2, 2)
    algorithm = RobustProxNAGGSAlgorithm(model=model, a=0.2, mu_hat=40.0, robust_map="identity")
    assert math.isclose(algorithm.base_step_size, 0.005, rel_tol=1e-9)


def test_robust_proxnaggs_warmup_scales_step_size():
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    algorithm = RobustProxNAGGSAlgorithm(
        model=model,
        a=0.2,
        mu_hat=40.0,
        robust_map="identity",
        use_warmup=True,
        warmup_steps=4,
    )
    criterion = nn.CrossEntropyLoss()
    x = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    y = torch.tensor([1], dtype=torch.long)

    algorithm.train_batch(model, x, y, criterion)
    first_step_size = algorithm.last_step_stats["step_size"]
    algorithm.train_batch(model, x, y, criterion)
    second_step_size = algorithm.last_step_stats["step_size"]

    assert math.isclose(first_step_size, 0.005 * 0.25, rel_tol=1e-6)
    assert math.isclose(second_step_size, 0.005 * 0.5, rel_tol=1e-6)


def test_no_prox_identity_matches_legacy_eta_parameterization():
    torch.manual_seed(0)
    model_with_mu = nn.Linear(3, 2)
    model_with_eta = deepcopy(model_with_mu)

    algorithm_with_mu = RobustProxNAGGSAlgorithm(
        model=model_with_mu,
        a=0.2,
        mu_hat=40.0,
        robust_map="identity",
    )
    algorithm_with_eta = RobustProxNAGGSAlgorithm(
        model=model_with_eta,
        a=0.2,
        eta=0.005,
        robust_map="identity",
    )

    criterion = nn.CrossEntropyLoss()
    x = torch.tensor([[1.0, -1.0, 0.5], [-0.2, 0.3, 0.1]], dtype=torch.float32)
    y = torch.tensor([1, 0], dtype=torch.long)

    loss_mu, acc_mu = algorithm_with_mu.train_batch(model_with_mu, x, y, criterion)
    loss_eta, acc_eta = algorithm_with_eta.train_batch(model_with_eta, x, y, criterion)

    assert math.isclose(loss_mu, loss_eta, rel_tol=1e-7)
    assert math.isclose(acc_mu, acc_eta, rel_tol=1e-7)
    for p_mu, p_eta in zip(model_with_mu.parameters(), model_with_eta.parameters()):
        assert torch.allclose(p_mu, p_eta, atol=1e-7)
    for v_mu, v_eta in zip(algorithm_with_mu.v_buffers, algorithm_with_eta.v_buffers):
        assert torch.allclose(v_mu, v_eta, atol=1e-7)


def test_weights_only_does_not_apply_prox_to_biases():
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    algorithm = RobustProxNAGGSAlgorithm(
        model=model,
        a=0.2,
        mu_hat=1.0,
        robust_map="identity",
        prox_operator=make_prox_operator("l1", lam=1.0),
        prox_target="weights_only",
    )
    for param in algorithm.params:
        param.grad = torch.zeros_like(param)

    algorithm._update_v()

    assert torch.allclose(algorithm.v_buffers[0], torch.full_like(algorithm.v_buffers[0], 0.8))
    assert torch.allclose(algorithm.v_buffers[1], torch.full_like(algorithm.v_buffers[1], 1.0))


class _ClassifierToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_classifier_target_applies_only_to_final_linear_layer():
    model = _ClassifierToyNet()
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)
    algorithm = RobustProxNAGGSAlgorithm(
        model=model,
        a=0.2,
        mu_hat=1.0,
        robust_map="identity",
        prox_operator=make_prox_operator("l1", lam=1.0),
        prox_target="classifier",
    )
    for param in algorithm.params:
        param.grad = torch.zeros_like(param)

    algorithm._update_v()

    assert torch.allclose(algorithm.v_buffers[0], torch.ones_like(algorithm.v_buffers[0]))
    assert torch.allclose(algorithm.v_buffers[1], torch.ones_like(algorithm.v_buffers[1]))
    assert torch.allclose(algorithm.v_buffers[2], torch.full_like(algorithm.v_buffers[2], 0.8))
    assert torch.allclose(algorithm.v_buffers[3], torch.full_like(algorithm.v_buffers[3], 0.8))


def test_regularization_penalty_uses_same_target_as_prox():
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(3.0)
    algorithm = RobustProxNAGGSAlgorithm(
        model=model,
        a=0.2,
        mu_hat=40.0,
        robust_map="identity",
        prox_operator=make_prox_operator("l1", lam=2.0),
        prox_target="weights_only",
    )
    criterion = nn.CrossEntropyLoss()
    x = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32)
    y = torch.tensor([1], dtype=torch.long)

    algorithm.train_batch(model, x, y, criterion)

    expected_penalty = 2.0 * model.weight.detach().abs().sum().item()
    assert math.isclose(algorithm.last_step_stats["regularization_penalty"], expected_penalty, rel_tol=1e-6)
