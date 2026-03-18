#!/usr/bin/env python3
"""
Generate test PLY files with varying sizes for comprehensive testing.
Creates deterministic synthetic gaussian splat data for reproducible tests.
"""

import argparse
import os
import struct
from pathlib import Path

import numpy as np

def generate_ply_header(num_splats: int) -> str:
    """Generate PLY header for gaussian splats"""
    return f"""ply
format binary_little_endian 1.0
element vertex {num_splats}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

def generate_gaussian_splat(index: int, total: int, pattern: str = "sphere") -> np.ndarray:
    """Generate a single gaussian splat with deterministic properties"""
    np.random.seed(42 + index)  # Deterministic random for reproducibility

    splat = np.zeros(17, dtype=np.float32)

    if pattern == "sphere":
        # Distribute splats in a sphere
        theta = 2 * np.pi * (index / total)
        phi = np.arccos(1 - 2 * (index / total))
        radius = 5.0 + np.random.uniform(-1, 1)

        splat[0] = radius * np.sin(phi) * np.cos(theta)  # x
        splat[1] = radius * np.sin(phi) * np.sin(theta)  # y
        splat[2] = radius * np.cos(phi)  # z
    elif pattern == "cube":
        # Distribute in a cube grid
        grid_size = int(np.ceil(total ** (1/3)))
        x_idx = index % grid_size
        y_idx = (index // grid_size) % grid_size
        z_idx = index // (grid_size * grid_size)

        splat[0] = (x_idx - grid_size/2) * 0.5  # x
        splat[1] = (y_idx - grid_size/2) * 0.5  # y
        splat[2] = (z_idx - grid_size/2) * 0.5  # z
    elif pattern == "bunny":
        # Simple bunny-like shape
        t = index / total * 2 * np.pi * 3
        splat[0] = 3 * np.cos(t) * (1 + 0.5 * np.cos(3*t))  # x
        splat[1] = 3 * np.sin(t) * (1 + 0.5 * np.cos(3*t))  # y
        splat[2] = 2 * np.sin(3*t)  # z

    # Normal (pointing outward from origin)
    norm = np.linalg.norm(splat[:3])
    if norm > 0:
        splat[3:6] = splat[:3] / norm  # nx, ny, nz
    else:
        splat[3:6] = [0, 0, 1]

    # Spherical harmonics DC coefficients (RGB color)
    # Create gradient colors based on position
    splat[6] = 0.5 + 0.3 * np.sin(index * 0.1)  # f_dc_0 (red)
    splat[7] = 0.5 + 0.3 * np.cos(index * 0.15)  # f_dc_1 (green)
    splat[8] = 0.5 + 0.3 * np.sin(index * 0.2)  # f_dc_2 (blue)

    # Opacity (mostly opaque)
    splat[9] = 0.8 + 0.2 * np.random.uniform()

    # Scale (small variations)
    base_scale = 0.05 + 0.02 * np.random.uniform()
    splat[10] = base_scale  # scale_0
    splat[11] = base_scale * (0.8 + 0.4 * np.random.uniform())  # scale_1
    splat[12] = base_scale * (0.8 + 0.4 * np.random.uniform())  # scale_2

    # Rotation quaternion (normalized)
    rot = np.random.randn(4)
    rot = rot / np.linalg.norm(rot)
    splat[13:17] = rot  # rot_0, rot_1, rot_2, rot_3

    return splat

def generate_ply_file(output_path: str, num_splats: int, pattern: str = "sphere"):
    """Generate a complete PLY file with gaussian splats"""
    print(f"Generating {output_path} with {num_splats} splats in {pattern} pattern...")

    with open(output_path, 'wb') as f:
        # Write header
        header = generate_ply_header(num_splats)
        f.write(header.encode('ascii'))

        # Generate and write splats
        for i in range(num_splats):
            splat = generate_gaussian_splat(i, num_splats, pattern)
            f.write(splat.tobytes())

            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1}/{num_splats} splats...")

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Created {output_path} ({file_size:.2f} MB)")

def generate_all_test_data(quick: bool = False):
    """Generate Gaussian splat test data files."""
    test_data_dir = Path(__file__).parent

    test_sets = [
        # Small test files (1K splats)
        ("small_sphere_1k.ply", 1000, "sphere"),
        ("small_cube_1k.ply", 1000, "cube"),
        ("small_bunny_1k.ply", 1000, "bunny"),

        # Medium test files (100K splats)
        ("medium_sphere_100k.ply", 100000, "sphere"),
        ("medium_cube_100k.ply", 100000, "cube"),
        ("medium_bunny_100k.ply", 100000, "bunny"),

        # Large test files (1M splats)
        ("large_sphere_1m.ply", 1000000, "sphere"),
        ("large_cube_1m.ply", 1000000, "cube"),

        # Stress test file (10M splats) - optional
        # ("stress_sphere_10m.ply", 10000000, "sphere"),
    ]

    if quick:
        test_sets = [item for item in test_sets if item[1] <= 1000]

    print("Generating test PLY files...")
    print("=" * 60)

    for filename, count, pattern in test_sets:
        output_path = test_data_dir / filename
        generate_ply_file(str(output_path), count, pattern)

    print("=" * 60)
    print("Test data generation complete!")

    # Generate metadata file
    metadata_path = test_data_dir / "test_data_metadata.json"
    import json
    metadata = {
        "generated": "2024-12-20",
        "files": {
            filename: {
                "splat_count": count,
                "pattern": pattern,
                "size_mb": os.path.getsize(test_data_dir / filename) / (1024 * 1024)
            }
            for filename, count, pattern in test_sets
            if (test_data_dir / filename).exists()
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gaussian splat test assets.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate only the small 1K fixture set (skips 100K/1M files).",
    )
    args = parser.parse_args()

    generate_all_test_data(quick=args.quick)
