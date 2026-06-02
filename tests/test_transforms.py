import torch

from rpnaggs.optim.transforms import robust_transform_vector


def test_norm_clip_reduces_large_vector_norm():
    g = torch.tensor([3.0, 4.0])
    clipped = robust_transform_vector(g, mode="norm_clip", threshold=2.0)
    assert torch.isclose(torch.linalg.norm(clipped), torch.tensor(2.0), atol=1e-6)


def test_coord_clip_bounds_each_coordinate():
    g = torch.tensor([-3.0, 0.5, 7.0])
    clipped = robust_transform_vector(g, mode="coord_clip", threshold=1.5)
    assert torch.all(clipped <= 1.5)
    assert torch.all(clipped >= -1.5)
