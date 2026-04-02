# Gaussian Splatting Integration Tests

_Last updated: 2025-11-27._

Comprehensive test suite for validating the complete Gaussian Splatting pipeline integration, LOD system, performance benchmarks, and memory management.

## 📋 Test Categories

### 1. Module Integration Tests (`test_integration.cpp`)
Tests runtime wiring between components:
- **Component Instantiation**: Validates all components can be created
- **Reference Counting**: Verifies proper buffer ownership and cleanup
- **Pipeline Execution**: Tests full rendering pipeline with 100K splats
- **Buffer Cycling**: Validates triple buffering with stall detection
- **API Consistency**: Ensures all APIs follow Godot patterns
- **Error Recovery**: Tests graceful handling of error conditions

### 2. LOD System Tests (`test_lod_system.cpp`)
Validates Level-of-Detail functionality:
- **HierarchicalSplatStructure**: Spatial hierarchy builds, queries, and parallel-build fallback
- **Renderer LOD Culling**: Bias and distance limits affect live culling as expected
- **Node Quality Presets**: Neutral node-facing quality and streaming configs stay coherent
- **Hierarchy Scalability**: Large hierarchy builds and queries stay on the accepted live path

### 3. Performance Benchmarks (`performance_benchmark.cpp`)
Stress tests with varying splat counts:
- **100K Splats**: Target 400+ FPS on RTX 3090
- **1M Splats**: Target 150+ FPS on RTX 3090
- **10M Splats**: Target 45+ FPS on RTX 3090
- **Metrics Tracked**:
  - Frame time percentiles (P50, P95, P99)
  - GPU memory usage (peak and average)
  - Sort time with/without async compute
  - Bandwidth utilization
  - Stall percentages

### 4. Headless Build Tests (`test_headless_build.py`)
Validates module without GUI:
- **Module Registration**: Confirms module loads in engine
- **GPU Initialization**: Tests RenderingDevice creation
- **Memory Leak Detection**: Monitors for memory growth
- **Component Lifecycle**: Tests proper init/cleanup sequences

## 🚀 Quick Start

### Basic Test Run
```bash
# Run core integration tests only
run_integration_tests.bat

# Run with performance benchmarks
run_integration_tests.bat --benchmarks

# Full test suite including stress tests
run_integration_tests.bat --benchmarks --stress --heavy
```

### Python Test Runner
```bash
# Run specific test suite
python run_integration_tests.py

# Include benchmarks
python run_integration_tests.py --benchmarks

# Run heavy tests (10M+ splats)
python run_integration_tests.py --benchmarks --heavy

# Run stress tests
python run_integration_tests.py --stress
```

### Headless Testing
```bash
# Run headless module tests
python test_headless_build.py

# Or via batch script
run_integration_tests.bat --headless
```

## 🔧 Test Configuration

### Performance Targets

| Component | Operation | Target | Threshold |
|-----------|----------|--------|-----------|
| BitonicSort | 100K splats | < 10ms | < 20ms |
| BitonicSort | 1M splats | < 50ms | < 100ms |
| Memory Stream | Buffer switch | 0 stalls | < 5% frames |
| LOD System | Frustum cull 100K | < 1ms | < 2ms |
| LOD System | Transition blend | > 95% smooth | > 90% |
| Full Pipeline | 100K @ 1080p | > 200 FPS | > 120 FPS |

### Memory Budgets

| Test Scenario | GPU Memory | CPU Memory | Notes |
|---------------|------------|------------|-------|
| 100K splats | < 50 MB | < 100 MB | Basic scene |
| 1M splats | < 500 MB | < 1 GB | Medium scene |
| 10M splats | < 2 GB | < 4 GB | Large scene |
| Stress test | < 4 GB | < 8 GB | Maximum load |

## 📊 Metrics Collection

### Frame Time Analysis
```cpp
// Collected metrics per frame
struct FrameMetrics {
    float frame_time_ms;      // Total frame time
    float upload_time_ms;     // Data upload to GPU
    float sort_time_ms;       // Sorting operation
    float render_time_ms;     // Tile rendering
    float cull_time_ms;       // Frustum culling
    uint32_t visible_splats;  // Rendered count
};
```

### Sorting Performance
```cpp
// From BitonicSort::get_metrics()
struct SortingMetrics {
    float last_sort_time_ms;
    float avg_sort_time_ms;
    uint32_t async_sorts;      // Async compute usage
    float async_speedup;       // Performance gain
    float bandwidth_utilization;
};
```

### Memory Stream Statistics
```cpp
// From GaussianMemoryStream::get_stats()
struct StreamingStats {
    uint32_t stalls;           // Pipeline stalls
    uint32_t buffer_switches;  // Triple buffer cycles
    uint32_t pool_hits;        // Memory pool efficiency
    float stall_percentage;    // Stall rate
};
```

##  Test Implementation Details

### Integration Test Flow
1. **Initialize RenderingDevice** - Create GPU context
2. **Generate Test Data** - Create procedural Gaussians
3. **Setup Memory Stream** - Initialize triple buffering
4. **Initialize Sorter** - Setup BitonicSort with async fallback
5. **Execute Pipeline** - Upload → Sort → Render
6. **Collect Metrics** - Frame times, stalls, memory
7. **Validate Results** - Check against thresholds

### LOD Test Scenarios
1. **Static Camera** - Validate stable LOD selection
2. **Parallel Build Fallback** - Validate large-scene hierarchy build stays subdivided
3. **Renderer Culling** - Validate distance and frustum controls affect live visibility
4. **Node Presets** - Validate neutral node-facing quality and streaming config surfaces

### Async Compute Validation
```cpp
// Test async compute availability and fallback
Ref<AsyncComputePipeline> async_pipeline;
async_pipeline.instantiate();
bool async_available = async_pipeline->initialize(rd) == OK;

if (async_available) {
    sorter->set_async_pipeline(async_pipeline.ptr());
    // Run with async compute
} else {
    // Fallback to single queue
}
```

## 📈 Performance Baselines

### RTX 3090 Expected Performance
| Splat Count | Sort Time | Render Time | Total Frame | FPS |
|-------------|-----------|-------------|-------------|-----|
| 100K | 5ms | 2ms | 8ms | 125 |
| 500K | 25ms | 8ms | 35ms | 28 |
| 1M | 45ms | 15ms | 65ms | 15 |
| 5M | 150ms | 60ms | 220ms | 4.5 |
| 10M | 280ms | 110ms | 400ms | 2.5 |

### Memory Pool Efficiency
- **Target**: > 80% pool hit rate
- **Acceptable**: > 60% pool hit rate
- **Defrag Trigger**: < 40% efficiency

## 🐛 Debugging Failed Tests

### Common Issues and Solutions

1. **GPU Initialization Failure**
   - Check Vulkan drivers are installed
   - Verify GPU supports Vulkan 1.3+
   - Try `--rendering-driver opengl3` fallback

2. **Memory Stream Stalls**
   - Increase buffer count (triple → quad)
   - Reduce upload size per frame
   - Check GPU transfer queue availability

3. **Sort Performance Issues**
   - Verify power-of-two padding
   - Check async compute availability
   - Profile with RenderDoc/NSight

4. **LOD Transition Artifacts**
   - Increase hysteresis threshold
   - Smooth blend weights over more frames
   - Validate seed consistency

##  Test Output Analysis

### JSON Report Format
```json
{
  "timestamp": "2024-12-20 15:30:00",
  "test_suites": {
    "integration": {
      "passed": true,
      "tests": [...],
      "duration_ms": 1234
    },
    "performance": {
      "benchmarks": [{
        "name": "100K_splats",
        "avg_fps": 245.6,
        "avg_frame_time_ms": 4.07,
        "sort_time_ms": 2.3,
        "stall_percentage": 0.0
      }]
    }
  },
  "summary": {
    "total_tests": 25,
    "passed": 24,
    "failed": 1,
    "success_rate": 96.0
  }
}
```

### Log File Locations
```
modules/gaussian_splatting/tests/
├── test_results/
│   ├── cpp_tests_TIMESTAMP.log
│   ├── integration_TIMESTAMP.log
│   ├── bench_100k_TIMESTAMP.log
│   └── headless_TIMESTAMP.log
├── integration_report_TIMESTAMP.json
└── integration_report_TIMESTAMP.md
```

## 🚦 CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Integration Tests
  run: |
    cd modules/gaussian_splatting/tests
    python run_integration_tests.py --benchmarks

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: modules/gaussian_splatting/tests/test_results/
```

## 📝 Adding New Tests

### C++ Test Template
```cpp
TEST_CASE("[Integration] Your test name") {
    SKIP_IF(!is_rendering_device_available(), "GPU required");

    // Setup
    Ref<YourComponent> component;
    component.instantiate();

    // Execute
    Error err = component->initialize();

    // Verify
    CHECK(err == OK);
    CHECK(component->get_metric() < threshold);

    // Cleanup
    component->cleanup();
}
```

### Python Benchmark Template
```python
def test_your_benchmark(self) -> Dict:
    result = {
        "name": "your_benchmark",
        "passed": False,
        "metrics": {}
    }

    # Setup test environment
    # Execute benchmark
    # Collect metrics
    # Validate against thresholds

    return result
```

## 🤝 Contributing

1. Add tests for new features in appropriate test files
2. Update performance baselines when optimizing
3. Document any new metrics or thresholds
4. Ensure tests pass before submitting PR
5. Include test results in PR description

## 📚 References

- [Godot Testing Documentation](https://docs.godotengine.org/en/stable/development/cpp/unit_testing.html)
- [GPU Profiling Best Practices](https://developer.nvidia.com/nsight-graphics)
- [Vulkan Validation Layers](https://vulkan.lunarg.com/doc/sdk/latest/windows/validation_layers.html)
