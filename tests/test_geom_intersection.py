import numpy as np
import pytest
import taichi as ti

from tibvh.geometry.geom_intersection import (
    ray_box_distance,
    ray_capsule_distance,
    ray_cylinder_distance,
    ray_ellipsoid_distance,
    ray_plane_distance,
    ray_sphere_distance,
    ray_triangle_distance,
)
from tibvh.geometry.utils import _transform_point_to_world, _transform_ray_to_local


def _identity():
    return np.eye(3, dtype=np.float32)


def _rot_z_90():
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@ti.kernel
def _triangle_distance_kernel(case_id: ti.i32) -> ti.f32:
    distance = -1.0
    if case_id == 0:
        distance = ray_triangle_distance(
            ti.math.vec3(0.25, 0.25, -1.0),
            ti.math.vec3(0.0, 0.0, 1.0),
            ti.math.vec3(0.0, 0.0, 0.0),
            ti.math.vec3(1.0, 0.0, 0.0),
            ti.math.vec3(0.0, 1.0, 0.0),
        )
    elif case_id == 1:
        distance = ray_triangle_distance(
            ti.math.vec3(1.25, 1.25, -1.0),
            ti.math.vec3(0.0, 0.0, 1.0),
            ti.math.vec3(0.0, 0.0, 0.0),
            ti.math.vec3(1.0, 0.0, 0.0),
            ti.math.vec3(0.0, 1.0, 0.0),
        )
    else:
        distance = ray_triangle_distance(
            ti.math.vec3(0.25, 0.25, 1.0),
            ti.math.vec3(1.0, 0.0, 0.0),
            ti.math.vec3(0.0, 0.0, 0.0),
            ti.math.vec3(1.0, 0.0, 0.0),
            ti.math.vec3(0.0, 1.0, 0.0),
        )
    return distance


@ti.kernel
def _shape_distance_kernel(
    kind: ti.i32,
    ray_start: ti.types.vector(3, ti.f32),
    ray_direction: ti.types.vector(3, ti.f32),
    center: ti.types.vector(3, ti.f32),
    size: ti.types.vector(3, ti.f32),
    rotation: ti.types.matrix(3, 3, ti.f32),
) -> ti.f32:
    distance = -1.0
    if kind == 0:
        distance = ray_plane_distance(ray_start, ray_direction, center, size, rotation)
    elif kind == 1:
        distance = ray_sphere_distance(ray_start, ray_direction, center, size, rotation)
    elif kind == 2:
        distance = ray_box_distance(ray_start, ray_direction, center, size, rotation)
    elif kind == 3:
        distance = ray_cylinder_distance(
            ray_start, ray_direction, center, size, rotation
        )
    elif kind == 4:
        distance = ray_ellipsoid_distance(
            ray_start, ray_direction, center, size, rotation
        )
    else:
        distance = ray_capsule_distance(
            ray_start, ray_direction, center, size, rotation
        )
    return distance


@ti.kernel
def _transform_kernel(
    out_start: ti.template(),
    out_direction: ti.template(),
    out_world_point: ti.template(),
    ray_start: ti.types.vector(3, ti.f32),
    ray_direction: ti.types.vector(3, ti.f32),
    center: ti.types.vector(3, ti.f32),
    rotation: ti.types.matrix(3, 3, ti.f32),
):
    local_start, local_direction = _transform_ray_to_local(
        ray_start, ray_direction, center, rotation
    )
    out_start[None] = local_start
    out_direction[None] = local_direction
    out_world_point[None] = _transform_point_to_world(local_start, center, rotation)


def test_ray_triangle_distance_hits_misses_and_rejects_parallel_ray():
    assert _triangle_distance_kernel(0) == pytest.approx(1.0)
    assert _triangle_distance_kernel(1) == pytest.approx(-1.0)
    assert _triangle_distance_kernel(2) == pytest.approx(-1.0)


def test_ray_plane_distance_checks_bounded_plane_extent():
    hit = _shape_distance_kernel(
        0,
        [0.25, 0.25, 2.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        _identity(),
    )
    miss = _shape_distance_kernel(
        0,
        [2.0, 0.25, 2.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        _identity(),
    )

    assert hit == pytest.approx(2.0)
    assert miss == pytest.approx(-1.0)


def test_ray_sphere_distance_handles_outside_inside_and_miss():
    outside_hit = _shape_distance_kernel(
        1,
        [-5.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        _identity(),
    )
    inside_hit = _shape_distance_kernel(
        1,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        _identity(),
    )
    miss = _shape_distance_kernel(
        1,
        [-5.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        _identity(),
    )

    assert outside_hit == pytest.approx(4.0)
    assert inside_hit == pytest.approx(1.0)
    assert miss == pytest.approx(-1.0)


def test_ray_box_distance_supports_inside_start_and_rotation():
    outside_hit = _shape_distance_kernel(
        2,
        [0.0, -5.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        _rot_z_90(),
    )
    inside_hit = _shape_distance_kernel(
        2,
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        _identity(),
    )
    miss = _shape_distance_kernel(
        2,
        [5.0, 5.0, 5.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        _identity(),
    )

    assert outside_hit == pytest.approx(4.0)
    assert inside_hit == pytest.approx(2.0)
    assert miss == pytest.approx(-1.0)


def test_ray_cylinder_distance_handles_side_cap_and_miss():
    side_hit = _shape_distance_kernel(
        3,
        [-3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0],
        _identity(),
    )
    cap_hit = _shape_distance_kernel(
        3,
        [0.5, 0.0, 5.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0],
        _identity(),
    )
    miss = _shape_distance_kernel(
        3,
        [3.0, 0.0, 5.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0],
        _identity(),
    )

    assert side_hit == pytest.approx(2.0)
    assert cap_hit == pytest.approx(3.0)
    assert miss == pytest.approx(-1.0)


def test_ray_ellipsoid_distance_scales_local_space():
    hit = _shape_distance_kernel(
        4,
        [-5.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        _identity(),
    )
    miss = _shape_distance_kernel(
        4,
        [-5.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        _identity(),
    )

    assert hit == pytest.approx(3.0)
    assert miss == pytest.approx(-1.0)


def test_ray_capsule_distance_checks_cylinder_body_and_hemisphere_caps():
    body_hit = _shape_distance_kernel(
        5,
        [-3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0],
        _identity(),
    )
    top_cap_hit = _shape_distance_kernel(
        5,
        [0.0, 0.0, 4.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0],
        _identity(),
    )

    assert body_hit == pytest.approx(2.0)
    assert top_cap_hit == pytest.approx(1.0)


def test_transform_helpers_round_trip_point_and_rotate_ray_direction():
    out_start = ti.Vector.field(3, ti.f32, shape=())
    out_direction = ti.Vector.field(3, ti.f32, shape=())
    out_world_point = ti.Vector.field(3, ti.f32, shape=())

    _transform_kernel(
        out_start,
        out_direction,
        out_world_point,
        [1.0, 3.0, 3.0],
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 3.0],
        _rot_z_90(),
    )

    np.testing.assert_allclose(out_start.to_numpy(), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out_direction.to_numpy(), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out_world_point.to_numpy(), [1.0, 3.0, 3.0], atol=1e-6)
