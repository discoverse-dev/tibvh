import torch

from tibvh.lbvh.lbvh import _select_torch_sort_device


def test_selects_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert _select_torch_sort_device() == "cuda"


def test_selects_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert _select_torch_sort_device() == "mps"


def test_falls_back_to_cpu_when_no_accelerator_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert _select_torch_sort_device() == "cpu"
