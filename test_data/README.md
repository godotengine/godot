# Phase 4C: Comprehensive Testing and Validation Suite

_Last updated: 2025-12-27_

## Overview
This directory contains the complete testing infrastructure for the Gaussian Splatting implementation in Godot. The suite validates correctness, performance, visual quality, and stability across multiple platforms.

## Test Structure

### 1. Test Data Files

**Committed PLY fixtures** (ready to use):
- `single_gaussian.ply` - Single splat for basic validation
- `grid_gaussians.ply` - Grid arrangement for spatial testing
- `depth_stack_gaussians.ply` - Overlapping splats for depth sorting validation
- `fixtures/basic_splat.ply` - Minimal fixture with Godot import metadata

**Generated PLY files** (created on-demand by `generate_test_data.py`, not committed):
- Small (1K splats): `small_sphere_1k.ply`, `small_cube_1k.ply`, `small_bunny_1k.ply`
- Medium (100K splats): `medium_sphere_100k.ply`, `medium_cube_100k.ply`, `medium_bunny_100k.ply`
- Large (1M splats): `large_sphere_1m.ply`, `large_cube_1m.ply`

These larger files are excluded via `.gitignore` to keep the repository lightweight.

### 2. Integration Tests
- **Script**: `test_phase4_integration.gd`
- **Categories**:
  - PLY file loading (multiple sizes)
  - Rendering quality (all presets)
  - Performance benchmarks (FPS targets)
  - Memory management (leak detection)
  - Multi-instance support (5+ nodes)
  - Streaming buffer validation
  - Error handling
  - Visual regression

### 3. Performance Benchmarks
- **Script**: `benchmark_performance.py`
- **Metrics Tracked**:
  - Fixture bounds (min/max XYZ)
  - Mean radius from origin
  - Naive depth-sort cost (CPU)

> _Note_: The benchmark no longer launches Godot or synthesises giant assets.
> It provides a fast sanity check that still fails if the fixture becomes empty
> or wildly changes scale.

### 4. Visual Quality Validation
- **Script**: `validate_visual_quality.py`
- **Tests**:
  - Regenerates a 2D preview from the fixture positions
  - Compares the preview against a stored reference image
  - Emits a pixel-delta report when the fixture changes

> _Tip_: Run `python validate_visual_quality.py --update-reference` after you
> intentionally change the fixture to refresh the stored reference checksum.
### 5. Demo Scene
- **Scene**: `gaussian_splat_demo.tscn`
- **Controller**: `demo_controller.gd`
- **Features**:
  - Interactive camera controls (WASD + mouse)
  - Quality preset switching (1-4 keys)
  - Load different splat counts
  - Multi-instance demo
  - Real-time performance overlay

### 6. Test Runner
- **Primary script**: `tests/ci/run_baseline_qa.py` - The main CI test orchestrator
- **Convenience wrapper**: `run_all_tests.py` - Delegates to `tests/ci/run_baseline_qa.py`
- **Capabilities**:
  - Orchestrates all test suites
  - Generates comprehensive reports
  - CI/CD integration ready
  - Category-specific testing

## Quick Start

### Use Built-in Fixtures
Small regression fixtures live under [`test_data/fixtures/`](./fixtures/). They ship with ready-to-use Godot import metadata (see `basic_splat.ply.import`), so both `GaussianData` and `ResourceLoader` based tests succeed without generating multi-megabyte assets.

### Generate Test Data
```bash
cd test_data
# Install Python helpers (one time)
pip install -r ../requirements-tests.txt

# Quick pass (1K fixtures only)
python generate_test_data.py --quick

# Full sweep (1K / 100K / 1M)
python generate_test_data.py
```

### Run All Tests
```bash
# Wrapper around tests/ci/run_baseline_qa.py
python run_all_tests.py
```

### Run Specific Category
```bash
# Options: ply, pipeline, sorting, runtime, module, qa
python run_all_tests.py --category pipeline
```

### Run Demo Scene
```bash
godot gaussian_splat_demo.tscn
```

## Success Metrics

The test suite validates these success criteria:

| Metric | Target | Test |
|--------|--------|------|
| Integration Tests | 15/15 pass | `test_phase4_integration.gd` |
| 1K Splats FPS | > 60 | Performance benchmark |
| 100K Splats FPS | > 60 | Performance benchmark |
| 1M Splats FPS | > 60 | Performance benchmark |
| Visual Quality | SSIM > 0.95 | Visual validation |
| Memory Leaks | None | Stress tests |
| 10M Splat Handling | No crash | Stress tests |

## CI/CD Integration

### GitHub Actions
The `.github/workflows/test_phase4.yml` workflow runs:
- Windows and Linux tests in parallel
- Generates test reports
- Uploads artifacts (results, screenshots)
- Comments results on PRs

### Local CI
```bash
# Run quick validation
python run_all_tests.py --quick

# Full validation before commit
python run_all_tests.py
```

## Test Reports

Reports are generated in JSON format:
- `test_results.json` - Overall test summary
- `benchmark_results.json` - Performance metrics
- `visual_quality_results.json` - Visual validation results
- `test_output/` - Generated screenshots

## Performance Baselines

Default performance targets (RTX 3060 equivalent):

| Splat Count | Avg Frame Time | P95 Frame Time | Target FPS |
|-------------|---------------|----------------|------------|
| 1K | < 1.0 ms | < 2.0 ms | 60+ |
| 100K | < 8.0 ms | < 12.0 ms | 60+ |
| 1M | < 16.67 ms | < 20.0 ms | 60+ |

## Troubleshooting

### Tests Skip
- Ensure Godot is built with `tests=yes`
- Check RenderingDevice availability
- Verify test data exists

### Performance Issues
- Close background applications
- Ensure GPU drivers are updated
- Check thermal throttling

### Visual Test Failures
- Generate reference images first
- Ensure consistent rendering settings
- Check display color profile

## Development

### Adding New Tests
1. Add test function to appropriate script
2. Update success metrics if needed
3. Run locally to verify
4. Update CI configuration if necessary

### Updating Baselines
```bash
python benchmark_performance.py --update-baseline
```

## Directory Structure
```
test_data/
├── fixtures/                    # Committed minimal test fixtures
│   └── basic_splat.ply          # Basic splat with .import metadata
├── single_gaussian.ply          # Committed: single splat fixture
├── grid_gaussians.ply           # Committed: grid arrangement fixture
├── depth_stack_gaussians.ply    # Committed: depth sorting fixture
├── generate_test_data.py        # PLY file generator (creates larger files on-demand)
├── test_phase4_integration.gd   # Integration tests
├── benchmark_performance.py     # Performance benchmarks
├── validate_visual_quality.py   # Visual validation
├── run_all_tests.py             # Wrapper for tests/ci/run_baseline_qa.py
├── gaussian_splat_demo.tscn     # Interactive demo
├── demo_controller.gd           # Demo logic
├── README.md                    # This file
├── visual_quality_reference.json # Stored preview baseline
└── *.json                       # Test results and reports
```

## Contact

For issues or questions about the test suite, please refer to the main project documentation or open an issue in the repository.
