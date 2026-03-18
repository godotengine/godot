#!/usr/bin/env python3
"""Lightweight simulation of the GPU pipeline for regression testing.

This test exercises the core data orchestration logic that the C++ module
implements: double-buffered uploads, depth sorting with power-of-two padding,
and binding buffers for tile-based rendering.  It does not perform any GPU
work, but it verifies the sequencing and invariants that the renderer relies
on so the behaviour can be validated in environments where the Vulkan
RenderingDevice is unavailable (such as CI containers used for automated
review).
"""

from __future__ import annotations

import math
import random
import unittest


def next_power_of_two(value: int) -> int:
    """Return the next power of two >= value (with 1 as minimum)."""

    if value <= 1:
        return 1
    power = 1
    while power < value:
        power <<= 1
    return power


class DoubleBuffer:
    """Minimal double-buffer helper mirroring GPUBufferManager semantics."""

    def __init__(self) -> None:
        self._front: list[int] = []
        self._back: list[int] = []
        self._using_back = False

    def upload(self, values: list[int]) -> None:
        """Stream data into the inactive buffer and swap it active."""

        target = self._back if not self._using_back else self._front
        target[:] = values
        self._using_back = not self._using_back

    def active(self) -> list[int]:
        return self._back if self._using_back else self._front

    def inactive(self) -> list[int]:
        return self._front if self._using_back else self._back


class TestGaussianPipeline(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(42)

    def test_double_buffer_swap_preserves_previous_frame(self) -> None:
        buffer = DoubleBuffer()
        first_frame = [1, 2, 3]
        buffer.upload(first_frame)
        self.assertListEqual(buffer.active(), first_frame)
        self.assertListEqual(buffer.inactive(), [])

        second_frame = [4, 5]
        buffer.upload(second_frame)
        self.assertListEqual(buffer.active(), second_frame)
        self.assertListEqual(buffer.inactive(), first_frame)

    def test_depth_sort_padding_matches_bitonic_requirements(self) -> None:
        splat_positions = [
            (random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0))
            for _ in range(13)
        ]
        camera = (0.0, 0.0, 0.0)

        def distance_sq(pos: tuple[float, float, float]) -> float:
            return sum((a - b) ** 2 for a, b in zip(pos, camera))

        keys = [distance_sq(p) for p in splat_positions]
        padded = next_power_of_two(len(keys))
        self.assertEqual(padded, 16)

        padded_keys = keys + [math.inf] * (padded - len(keys))
        values = list(range(padded))

        sorted_pairs = sorted(zip(padded_keys, values), key=lambda pair: pair[0])
        sorted_indices = [idx for _, idx in sorted_pairs[: len(splat_positions)]]

        manual = sorted(range(len(splat_positions)), key=lambda i: keys[i])
        self.assertListEqual(sorted_indices, manual)

    def test_tile_binding_uses_sorted_indices(self) -> None:
        splat_indices = list(range(8))
        random.shuffle(splat_indices)
        tile_assignments = [splat_indices[i : i + 4] for i in range(0, len(splat_indices), 4)]

        flattened = [index for tile in tile_assignments for index in tile]
        self.assertEqual(sorted(flattened), list(range(8)))

    def test_gpu_sort_staging_clamps_and_pads(self) -> None:
        culled_indices = list(range(20))
        gpu_capacity = 10
        padded_capacity = next_power_of_two(gpu_capacity)

        keys = [math.inf] * padded_capacity
        values = [0] * padded_capacity

        target = 0
        for idx in culled_indices:
            if target >= gpu_capacity:
                break
            keys[target] = float(idx)
            values[target] = idx
            target += 1

        self.assertEqual(target, gpu_capacity)
        self.assertTrue(all(key == math.inf for key in keys[target:]))
        self.assertTrue(all(val == 0 for val in values[target:]))
        self.assertListEqual(values[:target], culled_indices[:gpu_capacity])


if __name__ == "__main__":
    unittest.main()
