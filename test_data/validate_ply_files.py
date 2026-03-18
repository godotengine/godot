#!/usr/bin/env python3
"""
Validate PLY files for Gaussian Splatting tests
Can be run without Godot binary
"""

import struct
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

class PLYValidator:
    """Validates PLY file structure and content"""

    def __init__(self):
        self.test_data_dir = Path(__file__).parent
        self.results = {}

    def validate_all(self) -> Dict:
        """Validate all PLY files in test_data directory"""
        print("PLY File Validation")
        print("="*50)

        ply_files = list(self.test_data_dir.glob("*.ply"))

        if not ply_files:
            print("No PLY files found in test_data directory")
            return {}

        for ply_file in ply_files:
            print(f"\nValidating: {ply_file.name}")
            self.validate_file(ply_file)

        # Summary
        print("\n" + "="*50)
        print("Validation Summary")
        print("="*50)

        total = len(self.results)
        valid = sum(1 for r in self.results.values() if r["valid"])
        print(f"Total files: {total}")
        print(f"Valid files: {valid}")
        print(f"Invalid files: {total - valid}")

        # Save results
        report_file = self.test_data_dir / "ply_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "valid": valid,
                    "invalid": total - valid
                },
                "files": self.results
            }, f, indent=2)

        print(f"\nReport saved to: {report_file}")
        return self.results

    def validate_file(self, ply_path: Path) -> bool:
        """Validate a single PLY file"""
        try:
            with open(ply_path, 'rb') as f:
                # Check magic number
                magic = f.read(4)
                if magic != b'ply\n':
                    print(f"  ✗ Invalid PLY magic number")
                    self.results[ply_path.name] = {
                        "valid": False,
                        "error": "Invalid magic number"
                    }
                    return False

                # Read header
                header_lines = []
                while True:
                    line = f.readline().decode('ascii').strip()
                    header_lines.append(line)
                    if line == 'end_header':
                        break

                # Parse header
                vertex_count = 0
                properties = []
                format_type = None

                for line in header_lines:
                    if line.startswith('format'):
                        format_type = line.split()[1]
                    elif line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('property'):
                        parts = line.split()
                        prop_type = parts[1]
                        prop_name = parts[-1]
                        properties.append((prop_type, prop_name))

                # Validate Gaussian splat properties
                required_props = ['x', 'y', 'z']  # Minimum required
                gaussian_props = ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                                'scale_0', 'scale_1', 'scale_2',
                                'rot_0', 'rot_1', 'rot_2', 'rot_3']

                prop_names = [p[1] for p in properties]

                # Check minimum properties
                has_position = all(p in prop_names for p in required_props)
                has_gaussian = any(p in prop_names for p in gaussian_props)

                if not has_position:
                    print(f"  ✗ Missing position properties")
                    self.results[ply_path.name] = {
                        "valid": False,
                        "error": "Missing position properties"
                    }
                    return False

                # Get file size
                file_size = ply_path.stat().st_size

                # Report findings
                print(f"  Format: {format_type}")
                print(f"  Vertices: {vertex_count:,}")
                print(f"  Properties: {len(properties)}")
                print(f"  Has Gaussian properties: {'Yes' if has_gaussian else 'No'}")
                print(f"  File size: {file_size / (1024*1024):.2f} MB")

                # Validate data (sample first few vertices)
                if format_type == 'binary_little_endian':
                    # Read sample vertices
                    sample_size = min(10, vertex_count)
                    bytes_per_vertex = len(properties) * 4  # Assuming float32

                    for i in range(sample_size):
                        vertex_data = f.read(bytes_per_vertex)
                        if len(vertex_data) != bytes_per_vertex:
                            print(f"  ✗ Incomplete vertex data at vertex {i}")
                            self.results[ply_path.name] = {
                                "valid": False,
                                "error": f"Incomplete vertex data at vertex {i}"
                            }
                            return False

                print(f"  ✓ Valid PLY file")
                self.results[ply_path.name] = {
                    "valid": True,
                    "format": format_type,
                    "vertex_count": vertex_count,
                    "property_count": len(properties),
                    "has_gaussian_properties": has_gaussian,
                    "file_size_mb": file_size / (1024*1024)
                }
                return True

        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.results[ply_path.name] = {
                "valid": False,
                "error": str(e)
            }
            return False

def main():
    """Main entry point"""
    validator = PLYValidator()
    results = validator.validate_all()

    # Exit with error if any files are invalid
    invalid_count = sum(1 for r in results.values() if not r.get("valid", False))
    exit(1 if invalid_count > 0 else 0)

if __name__ == "__main__":
    main()
