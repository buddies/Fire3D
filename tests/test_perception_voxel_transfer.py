import numpy as np

from utils.perception_voxel_transfer import (
    transfer_voxel_labels,
    unique_voxel_labels,
)


def test_unique_voxel_labels_uses_majority_for_boundary_collision():
    keys, labels, audit = unique_voxel_labels(
        np.array([7, 7, 7, 9]),
        np.array([3, 3, 8, 4]),
    )

    np.testing.assert_array_equal(keys, [7, 9])
    np.testing.assert_array_equal(labels, [3, 4])
    assert audit == {
        "num_conflict_voxels": 1,
        "num_tied_voxels": 0,
        "num_minority_source_points": 1,
    }


def test_unique_voxel_labels_leaves_exact_tie_unlabeled():
    _, labels, audit = unique_voxel_labels(
        np.array([2, 2]),
        np.array([5, 6]),
    )

    np.testing.assert_array_equal(labels, [-1])
    assert audit["num_tied_voxels"] == 1


def test_transfer_voxel_labels_reports_conflict_resolution():
    source_points = np.array(
        [[0.10, 0.10, 0.10], [0.20, 0.20, 0.20], [0.30, 0.30, 0.30]]
    )
    dense_points = np.array([[0.15, 0.15, 0.15], [1.50, 1.50, 1.50]])

    labels, audit = transfer_voxel_labels(
        source_points,
        np.array([4, 4, 9]),
        dense_points,
        scene_scale=2.0,
        resolution=2,
    )

    np.testing.assert_array_equal(labels, [4, -1])
    assert audit["num_conflict_voxels"] == 1
    assert audit["num_minority_source_points"] == 1
    assert audit["num_dense_points_assigned"] == 1
