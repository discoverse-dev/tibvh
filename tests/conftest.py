import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def init_taichi_cpu():
    ti.init(arch=ti.cpu, random_seed=0)
    yield
