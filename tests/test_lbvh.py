import numpy as np
import taichi as ti

from tibvh.lbvh import AABB, LBVH


@ti.kernel
def _fill_aabbs(aabbs: ti.template()):
    aabbs[0].min = ti.math.vec3(-1.0, -1.0, -1.0)
    aabbs[0].max = ti.math.vec3(1.0, 1.0, 1.0)

    aabbs[1].min = ti.math.vec3(4.0, -1.0, -1.0)
    aabbs[1].max = ti.math.vec3(6.0, 1.0, 1.0)

    aabbs[2].min = ti.math.vec3(-1.0, 4.0, -1.0)
    aabbs[2].max = ti.math.vec3(1.0, 6.0, 1.0)

    aabbs[3].min = ti.math.vec3(4.0, 4.0, -1.0)
    aabbs[3].max = ti.math.vec3(6.0, 6.0, 1.0)


@ti.kernel
def _aabb_intersects_kernel(aabbs: ti.template(), out: ti.template()):
    out[0] = aabbs[0].intersects(aabbs[1])
    out[1] = aabbs[0].intersects(aabbs[2])
    out[2] = aabbs[0].intersects(aabbs[3])


@ti.kernel
def _query_points_kernel(bvh: ti.template(), points: ti.template()) -> ti.i32:
    overflow = bvh.query(points)
    return ti.cast(overflow, ti.i32)


@ti.kernel
def _collect_ray_kernel(
    bvh: ti.template(),
    out_candidates: ti.template(),
    out_count: ti.template(),
    ray_start: ti.types.vector(3, ti.f32),
    ray_direction: ti.types.vector(3, ti.f32),
):
    candidates, candidate_count = bvh.collect_intersecting_elements(
        ray_start, ray_direction
    )
    out_count[None] = candidate_count
    for i in range(32):
        out_candidates[i] = candidates[i]


@ti.kernel
def _root_ray_intersect_kernel(
    bvh: ti.template(),
    ray_start: ti.types.vector(3, ti.f32),
    ray_direction: ti.types.vector(3, ti.f32),
) -> ti.f32:
    return bvh.ray_node_intersect(ray_start, ray_direction, 0)


def _built_bvh(max_query_results=32):
    aabb_manager = AABB(4)
    _fill_aabbs(aabb_manager.aabbs)
    bvh = LBVH(aabb_manager, max_query_results=max_query_results)
    bvh._torch_sort_device = "cpu"
    bvh.build()
    return aabb_manager, bvh


def test_aabb_manager_reports_capacity_and_intersection_semantics():
    aabb_manager = AABB(4)
    _fill_aabbs(aabb_manager.aabbs)
    out = ti.field(ti.i32, shape=3)

    _aabb_intersects_kernel(aabb_manager.aabbs, out)

    assert aabb_manager.get_count() == 4
    np.testing.assert_array_equal(out.to_numpy(), [0, 0, 0])


def test_lbvh_computes_centers_scene_bounds_and_morton_codes():
    aabb_manager = AABB(4)
    _fill_aabbs(aabb_manager.aabbs)
    bvh = LBVH(aabb_manager)
    bvh._torch_sort_device = "cpu"

    bvh.compute_aabb_centers_and_scene_bounds()
    bvh.compute_morton_codes()

    np.testing.assert_allclose(
        bvh.aabb_centers.to_numpy()[:4],
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(bvh.scene_min.to_numpy(), [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(bvh.scene_max.to_numpy(), [6.0, 6.0, 1.0])
    np.testing.assert_allclose(
        bvh.scene_scale.to_numpy(), [1.0 / 7.0, 1.0 / 7.0, 0.5], rtol=1e-6
    )

    morton = bvh.morton_codes.to_numpy()[:4]
    np.testing.assert_array_equal(morton[:, 1], [0, 1, 2, 3])
    assert len(set(morton[:, 0].tolist())) == 4


def test_lbvh_build_produces_valid_tree_and_root_bounds():
    _, bvh = _built_bvh()

    validation = bvh.validate_tree_structure()

    assert validation["status"] == "valid"
    assert validation["n_aabbs"] == 4
    assert validation["total_nodes"] == 7
    np.testing.assert_allclose(bvh.nodes.aabb_min.to_numpy()[0], [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(bvh.nodes.aabb_max.to_numpy()[0], [6.0, 6.0, 1.0])
    np.testing.assert_array_equal(
        np.sort(bvh.nodes.element_id.to_numpy()[3:7]), [0, 1, 2, 3]
    )


def test_lbvh_point_query_returns_containing_aabbs():
    _, bvh = _built_bvh()
    points = ti.Vector.field(3, ti.f32, shape=5)
    points.from_numpy(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [5.5, 0.0, 0.0],
                [0.0, 5.5, 0.0],
                [5.5, 5.5, 0.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
    )

    overflow = _query_points_kernel(bvh, points)

    assert overflow == 0
    count = int(bvh.query_result_count.to_numpy())
    result = bvh.query_result.to_numpy()[:count]
    assert sorted(map(tuple, result.tolist())) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_lbvh_query_reports_overflow_when_result_buffer_is_too_small():
    _, bvh = _built_bvh(max_query_results=2)
    points = ti.Vector.field(3, ti.f32, shape=4)
    points.from_numpy(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [5.5, 0.0, 0.0],
                [0.0, 5.5, 0.0],
                [5.5, 5.5, 0.0],
            ],
            dtype=np.float32,
        )
    )

    overflow = _query_points_kernel(bvh, points)

    assert overflow == 1
    assert int(bvh.query_result_count.to_numpy()) == 4
    assert bvh.query_result.to_numpy()[:2].shape == (2, 2)


def test_lbvh_ray_traversal_collects_intersected_leaf_elements():
    _, bvh = _built_bvh()
    candidates = ti.field(ti.i32, shape=32)
    count = ti.field(ti.i32, shape=())

    _collect_ray_kernel(
        bvh,
        candidates,
        count,
        [-2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    n_candidates = int(count.to_numpy())
    assert sorted(candidates.to_numpy()[:n_candidates].tolist()) == [0, 1]


def test_lbvh_ray_node_intersect_returns_entry_distance_or_miss():
    _, bvh = _built_bvh()

    hit_from_outside = _root_ray_intersect_kernel(
        bvh, [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    )
    hit_from_inside = _root_ray_intersect_kernel(bvh, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    miss = _root_ray_intersect_kernel(bvh, [10.0, 10.0, 10.0], [1.0, 0.0, 0.0])

    assert hit_from_outside == 1.0
    assert hit_from_inside == 6.0
    assert miss == -1.0
