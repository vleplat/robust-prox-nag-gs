import torch

from rpnaggs.config import ExperimentConfig
from rpnaggs.data.registry import available_datasets, get_data_loaders
from rpnaggs.models.registry import available_model_pairs, build_model


def test_available_registries_include_mnist_and_vgg7():
    assert "mnist" in available_datasets()
    assert ("mnist", "vgg7_mini_mnist") in available_model_pairs()
    assert ("cifar10", "small_cifar_cnn") in available_model_pairs()


def test_vgg7_mini_mnist_forward_shape():
    config = ExperimentConfig(dataset="mnist", model_name="vgg7_mini_mnist")
    model = build_model(config)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_small_cifar_cnn_still_builds():
    config = ExperimentConfig(dataset="cifar10", model_name="small_cifar_cnn")
    model = build_model(config)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_mnist_loader_smoke():
    config = ExperimentConfig(
        dataset="mnist",
        model_name="vgg7_mini_mnist",
        use_train_subset=True,
        train_subset_size=32,
        batch_size=16,
        num_workers=0,
        label_noise_rate=0.0,
    )
    train_loader, train_eval_loader, test_loader = get_data_loaders(config)
    x, y = next(iter(train_loader))
    assert x.shape[1:] == (1, 28, 28)
    assert y.ndim == 1
    assert len(train_eval_loader.dataset) == 32
    assert len(test_loader.dataset) == 10000
