# Baseline QA Test Coverage

_Note: coverage metrics may be stale; rerun `python tests/ci/run_baseline_qa.py` to validate. Last reviewed 2025-11-27._

This document outlines the comprehensive test coverage provided by the Baseline QA automation system for Issue #124.

## Test Categories

### 1. PLY Loader Tests (`test_ply_loader_ci.gd`)

**Scope:** Core PLY file loading and saving functionality

**Test Cases:**
- **Basic PLY Loading**: Tests loading existing PLY files with validation of splat count and AABB
- **PLY Saving**: Tests creating and saving Gaussian data to PLY format
- **Roundtrip Validation**: Tests loading saved PLY files to verify data integrity
- **Error Handling**: Tests proper error codes for invalid/missing files

**Coverage Areas:**
- `GaussianData` class instantiation
- File I/O operations (`load_from_file`, `save_to_file`)
- Data structure validation (positions, scales, rotations, opacities)
- Error code validation (`ERR_FILE_NOT_FOUND`, etc.)
- User directory file operations

**Success Criteria:**
- All PLY operations return proper error codes
- Data integrity maintained through save/load cycles
- Reasonable data ranges validated (count > 0, valid AABB)

### 2. PLY Pipeline Tests (`test_ply_pipeline_ci.gd`)

**Scope:** Complete PLY data pipeline integration

**Test Cases:**
- **GaussianSplatAsset Loading**: Tests asset-based PLY loading with array validation
- **PLYLoader Integration**: Tests PLYLoader workflow to GaussianData
- **Renderer Integration**: Tests complete pipeline from PLY to renderer
- **Data Consistency**: Tests data integrity across different pipeline components

**Coverage Areas:**
- `GaussianSplatAsset` functionality
- `PLYLoader` class operations
- `GaussianSplatRenderer` integration
- Array size validation (positions, colors, scales, rotations)
- Cross-component data consistency
- Memory management and object lifecycle

**Success Criteria:**
- All pipeline components integrate without data loss
- Array sizes match expected patterns
- Renderer accepts and processes PLY-derived data
- Data consistency maintained across transformations

### 3. GPU Sorting Tests (`test_gpu_sorting_ci.gd`)

**Scope:** GPU sorting functionality and performance validation

**Test Cases:**
- **Renderer Initialization**: Tests GPU context and RenderingDevice availability
- **Sorting Method Configuration**: Tests bitonic sorting method setup
- **Multi-Scale Dataset Testing**: Tests small (1K), medium (10K), and large (100K) datasets
- **Performance Validation**: Tests sorting performance against target thresholds

**Coverage Areas:**
- `GaussianSplatRenderer` initialization
- GPU context handling in headless environments
- Bitonic sorting algorithm performance
- Test data generation and validation
- Performance metrics collection
- Graceful degradation for headless CI

**Performance Targets:**
- Small datasets (1K): < 5ms (CI-adjusted)
- Medium datasets (10K): < 10ms (CI-adjusted)
- Large datasets (100K): < 20ms (CI-adjusted)
- Throughput measurement in M splats/second

**Success Criteria:**
- Renderer initializes properly (or gracefully handles headless mode)
- Sorting completes for all dataset sizes
- Performance within acceptable ranges for CI environment
- Proper error handling for GPU unavailability

### 4. Module Test Suite (`run_module_tests.py`)

**Scope:** Godot doctest coverage compiled into the module (`tests=yes` builds only)

**Test Cases:**
- **Data packing + layout**: `test_gaussian_data.h` memory layout and packing validation
- **GPU sorting**: `test_gpu_sorting.h` correctness and performance assertions
- **Pipeline integration**: `test_renderer_pipeline.h`, `test_phase1_integration.h`
- **Deterministic render validation**: debug projection output compared against a golden gradient

**Coverage Areas:**
- Core data layout and packing invariants
- GPU sorting correctness and metrics
- Render pipeline wiring and final output copy paths
- Deterministic render output checks for regression detection

**Success Criteria:**
- Godot test runner reports all Gaussian Splatting cases passing
- Integration suite runs when tests are enabled in the binary
- Module tests are skipped (not failed) on binaries without `tests=yes`

## CI Integration

### Automation Framework

**Test Runner:** `run_baseline_qa.py`
- Orchestrates all test execution
- Provides comprehensive error reporting
- Generates JSON results for CI consumption
- Implements timeout handling and exception recovery

**CI Workflow:** `baseline_qa.yml`
- Triggered on pushes to main branches and PRs
- Runs on Ubuntu with virtual display
- Includes both fast (pre-built binary) and thorough (compiled module) variants
- Generates GitHub Actions summary and PR comments

### Error Reporting

**Actionable Failure Analysis:**
- Timeout detection and guidance
- GPU context availability checking
- Module dependency validation
- Functional test failure classification

**Output Formats:**
- Console output with status indicators
- JSON results for programmatic analysis
- GitHub Actions summary with metrics table
- PR comments with test status overview

## Test Data Requirements

### Generated Test Data
- PLY files with known splat counts and positions
- Deterministic data for consistency validation
- Various dataset sizes for performance testing

### Expected Files
- `res://test_data/test.ply` (optional, tests handle absence gracefully)
- `res://tests/fixtures/test_splats.ply` (optional, fallback to generated data)
- User directory access for temporary files

## Headless Mode Compatibility

### GPU Context Handling
- Tests detect GPU unavailability in CI
- Graceful degradation to CPU-only validation
- Performance tests skip GPU-specific metrics when appropriate
- Clear messaging about expected limitations

### File System Access
- Uses Godot's user directory for temporary files
- Validates file creation and access permissions
- Handles platform-specific path differences

## Maintenance Guidelines

### Adding New Tests
1. Follow the established test structure with `run_test()` methodology
2. Include proper error handling and timeout considerations
3. Generate JSON-compatible results for CI consumption
4. Document new test coverage in this file

### Performance Threshold Updates
- Monitor CI environment performance over time
- Adjust thresholds based on realistic CI hardware capabilities
- Maintain separate targets for development vs CI environments

### CI Configuration Updates
- Update workflow when new test categories are added
- Ensure artifact collection captures new result files
- Maintain compatibility with both fast and thorough test variants

## Known Limitations

### CI Environment Constraints
- Limited GPU access in standard CI runners
- Virtual display required for Godot initialization
- Network and disk I/O performance variations

### Test Scope Boundaries
- Does not test actual rendering output (visual validation)
- Limited to functional and performance validation
- Does not test platform-specific GPU drivers

## Success Metrics

### Overall Goals
- **Test Reliability**: > 95% pass rate on clean commits
- **Execution Time**: < 10 minutes for fast tests, < 45 minutes for thorough tests
- **Coverage Completeness**: All core PLY and GPU sorting paths validated
- **Error Detection**: Clear identification of regressions in asset loading and sorting

### Quality Gates
- Zero test failures on main branch
- Performance regression detection (> 50% slowdown triggers investigation)
- Proper error handling for all failure modes
- Comprehensive reporting for debugging support
