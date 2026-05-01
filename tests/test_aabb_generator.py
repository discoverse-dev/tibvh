import numpy as np
import taichi as ti

from tibvh.geometry.aabb_generator import (
    aabb_local2wolrd,
    compute_box_aabb,
    compute_capsule_aabb,
    compute_cylinder_aabb,
    compute_ellipsoid_aabb,
    compute_plane_aabb,
    compute_sphere_aabb,
    compute_triangle_aabb,
)


def _vec_field():
    return ti.Vector.field(3, ti.f32, shape=())


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
def _triangle_aabb_kernel(out_min: ti.template(), out_max: ti.template()):
    aabb_min, aabb_max = compute_triangle_aabb(
        ti.math.vec3(1.0, -2.0, 3.0),
        ti.math.vec3(-4.0, 5.0, 0.5),
        ti.math.vec3(2.0, 1.0, -6.0),
    )
    out_min[None] = aabb_min
    out_max[None] = aabb_max


@ti.kernel
def _sphere_aabb_kernel(
    out_min: ti.template(),
    out_max: ti.template(),
    position: ti.types.vector(3, ti.f32),
    size: ti.types.vector(3, ti.f32),
):
    aabb_min, aabb_max = compute_sphere_aabb(position, size)
    out_min[None] = aabb_min
    out_max[None] = aabb_max


@ti.kernel
def _oriented_aabb_kernel(
    kind: ti.i32,
    out_min: ti.template(),
    out_max: ti.template(),
    position: ti.types.vector(3, ti.f32),
    rotation: ti.types.matrix(3, 3, ti.f32),
    size: ti.types.vector(3, ti.f32),
):
    aabb_min = ti.math.vec3(0.0, 0.0, 0.0)
    aabb_max = ti.math.vec3(0.0, 0.0, 0.0)

    if kind == 0:
        aabb_min, aabb_max = compute_plane_aabb(position, rotation, size)
    elif kind == 1:
        aabb_min, aabb_max = compute_capsule_aabb(position, rotation, size)
    elif kind == 2:
        aabb_min, aabb_max = compute_ellipsoid_aabb(position, rotation, size)
    elif kind == 3:
        aabb_min, aabb_max = compute_cylinder_aabb(position, rotation, size)
    elif kind == 4:
        aabb_min, aabb_max = compute_box_aabb(position, rotation, size)
    else:
        aabb_min, aabb_max = aabb_local2wolrd(
            ti.math.vec3(1.0, 2.0, 3.0),
            size,
            position,
            rotation,
        )

    out_min[None] = aabb_min
    out_max[None] = aabb_max


def _run_oriented_aabb(kind, position, rotation, size):
    out_min = _vec_field()
    out_max = _vec_field()
    _oriented_aabb_kernel(kind, out_min, out_max, position, rotation, size)
    return out_min.to_numpy(), out_max.to_numpy()


def test_triangle_aabb_uses_componentwise_vertex_extents():
    out_min = _vec_field()
    out_max = _vec_field()

    _triangle_aabb_kernel(out_min, out_max)

    np.testing.assert_allclose(out_min.to_numpy(), [-4.0, -2.0, -6.0])
    np.testing.assert_allclose(out_max.to_numpy(), [2.0, 5.0, 3.0])


def test_sphere_aabb_expands_center_by_radius():
    out_min = _vec_field()
    out_max = _vec_field()

    _sphere_aabb_kernel(out_min, out_max, [1.0, 2.0, -3.0], [2.5, 0.0, 0.0])

    np.testing.assert_allclose(out_min.to_numpy(), [-1.5, -0.5, -5.5])
    np.testing.assert_allclose(out_max.to_numpy(), [3.5, 4.5, -0.5])


def test_box_aabb_respects_rotation_and_translation():
    aabb_min, aabb_max = _run_oriented_aabb(
        4,
        [10.0, 20.0, 30.0],
        _rot_z_90(),
        [1.0, 2.0, 3.0],
    )

    np.testing.assert_allclose(aabb_min, [8.0, 19.0, 27.0], atol=1e-6)
    np.testing.assert_allclose(aabb_max, [12.0, 21.0, 33.0], atol=1e-6)


def test_plane_aabb_uses_default_large_extent_for_non_positive_size():
    aabb_min, aabb_max = _run_oriented_aabb(
        0,
        [1.0, 2.0, 3.0],
        _identity(),
        [0.0, -1.0, 0.0],
    )

    np.testing.assert_allclose(aabb_min, [-999.0, -998.0, 2.99], atol=1e-6)
    np.testing.assert_allclose(aabb_max, [1001.0, 1002.0, 3.01], atol=1e-6)


def test_capsule_aabb_includes_radius_and_half_height():
    aabb_min, aabb_max = _run_oriented_aabb(
        1,
        [1.0, 2.0, 3.0],
        _identity(),
        [0.5, 0.0, 2.0],
    )

    np.testing.assert_allclose(aabb_min, [0.5, 1.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(aabb_max, [1.5, 2.5, 5.5], atol=1e-6)


def test_ellipsoid_aabb_swaps_axes_under_quarter_turn():
    aabb_min, aabb_max = _run_oriented_aabb(
        2,
        [0.0, 0.0, 0.0],
        _rot_z_90(),
        [1.0, 2.0, 3.0],
    )

    np.testing.assert_allclose(aabb_min, [-2.0, -1.0, -3.0], atol=1e-6)
    np.testing.assert_allclose(aabb_max, [2.0, 1.0, 3.0], atol=1e-6)


def test_cylinder_aabb_covers_radius_and_height():
    aabb_min, aabb_max = _run_oriented_aabb(
        3,
        [1.0, 2.0, 3.0],
        _identity(),
        [0.75, 0.0, 2.0],
    )

    np.testing.assert_allclose(aabb_min, [0.25, 1.25, 1.0], atol=1e-5)
    np.testing.assert_allclose(aabb_max, [1.75, 2.75, 5.0], atol=1e-5)


def test_aabb_local_to_world_transforms_local_center_before_box_bounds():
    aabb_min, aabb_max = _run_oriented_aabb(
        5,
        [10.0, 20.0, 30.0],
        _identity(),
        [0.5, 1.0, 1.5],
    )

    np.testing.assert_allclose(aabb_min, [10.5, 21.0, 31.5], atol=1e-6)
    np.testing.assert_allclose(aabb_max, [11.5, 23.0, 34.5], atol=1e-6)
