#ifndef TEST_GPU_INTEGRATION_H
#define TEST_GPU_INTEGRATION_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"

class RenderingDevice;

// GPU Integration Validation Test Suite
//
// Purpose: Validate the ACTUAL GPU implementation, not future promises
// Documents what really works vs what are stubs/fallbacks
//
// Key findings:
// - Radix sort is the only supported backend
// - Async compute ALWAYS returns false (no separate queues)
// - All GPU work runs on graphics queue with CPU sync
// - Tile renderer composite path produces visible splats
//
class TestGPUIntegration : public RefCounted {
    GDCLASS(TestGPUIntegration, RefCounted);

private:
    RenderingDevice *rd;

protected:
    static void _bind_methods();

public:
    TestGPUIntegration();
    ~TestGPUIntegration();

    // Initialize test environment
    Error initialize();
    void cleanup();

    // Main test runner
    void run_all_tests();

    // Individual component tests
    void test_radix_sort();          // Validates radix sorter path on GPU
    void test_memory_stream();       // Tests triple buffering and pool diagnostics
    void test_async_fallback();      // Verifies async always returns false
    void test_tile_renderer_smoke(); // Verifies tile rasterization outputs visible splats
    void test_buffer_ownership();    // Validates reference counting
    void test_performance_metrics(); // Tests GPU timestamp collection
    void test_buffer_manager_integration(); // Tests GPUBufferManager upload and rendering integration
    void test_resolve_debug_toggles(); // Verifies resolve debug mode toggles exclusivity

    // Print validation summary
    void print_summary();

    // Utility functions
    uint32_t next_power_of_two(uint32_t n) const {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n++;
        return n;
    }
};

#endif // TEST_GPU_INTEGRATION_H
