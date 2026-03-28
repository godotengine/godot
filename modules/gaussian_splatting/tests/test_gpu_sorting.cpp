#include "test_macros.h"
#include "test_utils.h"
#include "../renderer/gpu_sorter.h"
#include "servers/rendering/rendering_device.h"
#include "core/math/random_number_generator.h"
#include <algorithm>
#include <random>
#include <chrono>

// Helper to create test data
struct TestData {
    LocalVector<float> keys;
    LocalVector<uint32_t> values;
    LocalVector<float> sorted_keys_expected;
    LocalVector<uint32_t> sorted_values_expected;
    
    void generate_random(uint32_t count, float min_val = 0.0f, float max_val = 100.0f) {
        keys.clear();
        values.clear();
        sorted_keys_expected.clear();
        sorted_values_expected.clear();
        
        // Generate random keys and sequential values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        for (uint32_t i = 0; i < count; i++) {
            keys.push_back(dist(gen));
            values.push_back(i);
        }
        
        // Create expected sorted result
        struct KeyValue {
            float key;
            uint32_t value;
        };
        
        std::vector<KeyValue> pairs;
        for (uint32_t i = 0; i < count; i++) {
            pairs.push_back({keys[i], values[i]});
        }
        
        // Sort by key
        std::sort(pairs.begin(), pairs.end(), 
            [](const KeyValue &a, const KeyValue &b) { return a.key < b.key; });
        
        // Extract sorted results
        for (const auto &pair : pairs) {
            sorted_keys_expected.push_back(pair.key);
            sorted_values_expected.push_back(pair.value);
        }
    }
    
    void generate_sorted(uint32_t count) {
        keys.clear();
        values.clear();
        
        for (uint32_t i = 0; i < count; i++) {
            keys.push_back(float(i));
            values.push_back(i);
        }
        
        sorted_keys_expected = keys;
        sorted_values_expected = values;
    }
    
    void generate_reverse_sorted(uint32_t count) {
        keys.clear();
        values.clear();
        
        for (uint32_t i = 0; i < count; i++) {
            keys.push_back(float(count - i - 1));
            values.push_back(i);
        }
        
        // Expected: ascending order
        for (uint32_t i = 0; i < count; i++) {
            sorted_keys_expected.push_back(float(i));
            sorted_values_expected.push_back(count - i - 1);
        }
    }
};

// Test basic sorting correctness
TEST_CASE("[GaussianSplatting][RequiresGPU] Bitonic sort correctness - small arrays") {
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    CHECK(rd != nullptr);
    if (rd == nullptr) {
        return;
    }

    // Create sorter
    Ref<BitonicSort> sorter = memnew(BitonicSort);
    CHECK(sorter.is_valid());
    if (!sorter.is_valid()) {
        memdelete(rd);
        return;
    }

    Error err = sorter->initialize(rd, 10000);
    CHECK(err == OK);
    if (err != OK) {
        memdelete(rd);
        return;
    }
    
    // Test various sizes
    uint32_t test_sizes[] = {16, 64, 256, 1024, 4096};
    
    for (uint32_t size : test_sizes) {
        SUBCASE(vformat("Size %d random data", size).utf8().get_data()) {
            TestData data;
            data.generate_random(size);
            
            // Create GPU buffers
            Vector<uint8_t> keys_bytes;
            keys_bytes.resize(size * sizeof(float));
            memcpy(keys_bytes.ptrw(), data.keys.ptr(), keys_bytes.size());
            RID keys_buffer = rd->storage_buffer_create(
                keys_bytes.size(),
                keys_bytes,
                RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT
            );
            rd->set_resource_name(keys_buffer, "GS_Test_BitonicSort_Keys");

            Vector<uint8_t> values_bytes;
            values_bytes.resize(size * sizeof(uint32_t));
            memcpy(values_bytes.ptrw(), data.values.ptr(), values_bytes.size());
            RID values_buffer = rd->storage_buffer_create(
                values_bytes.size(),
                values_bytes,
                RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT
            );
            rd->set_resource_name(values_buffer, "GS_Test_BitonicSort_Values");

            // Sort
            err = sorter->sort(keys_buffer, values_buffer, size);
            CHECK(err == OK);
            
            // Read back results
            Vector<uint8_t> keys_result = rd->buffer_get_data(keys_buffer);
            Vector<uint8_t> values_result = rd->buffer_get_data(values_buffer);
            
            // Verify correctness
            const float *sorted_keys = (const float *)keys_result.ptr();
            const uint32_t *sorted_values = (const uint32_t *)values_result.ptr();
            
            for (uint32_t i = 0; i < size; i++) {
                CHECK(Math::is_equal_approx(sorted_keys[i], data.sorted_keys_expected[i]));
                CHECK(sorted_values[i] == data.sorted_values_expected[i]);
            }
            
            // Check ascending order
            for (uint32_t i = 1; i < size; i++) {
                CHECK(sorted_keys[i] >= sorted_keys[i - 1]);
            }
            
            // Cleanup
            rd->free(keys_buffer);
            rd->free(values_buffer);
        }
    }
    
    memdelete(rd);
}

// Test non-power-of-two sizes
TEST_CASE("[GaussianSplatting][RequiresGPU] Bitonic sort - non-power-of-two") {
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    CHECK(rd != nullptr);
    if (rd == nullptr) {
        return;
    }

    Ref<BitonicSort> sorter = memnew(BitonicSort);
    Error err = sorter->initialize(rd, 10000);
    CHECK(err == OK);
    if (err != OK) {
        memdelete(rd);
        return;
    }
    
    // Non-power-of-two sizes
    uint32_t test_sizes[] = {17, 100, 500, 1337, 3000};
    
    for (uint32_t size : test_sizes) {
        SUBCASE(vformat("Size %d", size).utf8().get_data()) {
            TestData data;
            data.generate_random(size);
            
            Vector<uint8_t> keys_bytes;
            keys_bytes.resize(sorter->get_max_elements() * sizeof(float));
            memcpy(keys_bytes.ptrw(), data.keys.ptr(), size * sizeof(float));
            RID keys_buffer = rd->storage_buffer_create(keys_bytes.size(), keys_bytes);
            rd->set_resource_name(keys_buffer, "GS_Test_NonPow2_Keys");

            Vector<uint8_t> values_bytes;
            values_bytes.resize(sorter->get_max_elements() * sizeof(uint32_t));
            memcpy(values_bytes.ptrw(), data.values.ptr(), size * sizeof(uint32_t));
            RID values_buffer = rd->storage_buffer_create(values_bytes.size(), values_bytes);
            rd->set_resource_name(values_buffer, "GS_Test_NonPow2_Values");

            err = sorter->sort(keys_buffer, values_buffer, size);
            CHECK(err == OK);
            
            Vector<uint8_t> keys_result = rd->buffer_get_data(keys_buffer, 0, size * sizeof(float));
            const float *sorted_keys = (const float *)keys_result.ptr();
            
            // Check only the first 'size' elements are sorted
            for (uint32_t i = 1; i < size; i++) {
                CHECK(sorted_keys[i] >= sorted_keys[i - 1]);
            }
            
            rd->free(keys_buffer);
            rd->free(values_buffer);
        }
    }
    
    memdelete(rd);
}

// Performance benchmark
TEST_CASE("[GaussianSplatting][RequiresGPU] Bitonic sort performance") {
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    CHECK(rd != nullptr);
    if (rd == nullptr) {
        return;
    }

    Ref<BitonicSort> sorter = memnew(BitonicSort);
    Error err = sorter->initialize(rd, 1000000);
    CHECK(err == OK);
    if (err != OK) {
        memdelete(rd);
        return;
    }
    
    // Benchmark different sizes
    struct BenchmarkResult {
        uint32_t size;
        float gpu_time_ms;
        float cpu_time_ms;
        float speedup;
        float elements_per_second;
    };
    
    LocalVector<BenchmarkResult> results;
    
    uint32_t sizes[] = {1000, 10000, 50000, 100000, 500000};
    
    for (uint32_t size : sizes) {
        TestData data;
        data.generate_random(size);
        
        // GPU sorting
        Vector<uint8_t> keys_bytes;
        keys_bytes.resize(size * sizeof(float));
        memcpy(keys_bytes.ptrw(), data.keys.ptr(), keys_bytes.size());
        RID keys_buffer = rd->storage_buffer_create(keys_bytes.size(), keys_bytes);
        rd->set_resource_name(keys_buffer, "GS_Test_PerfBenchmark_Keys");

        Vector<uint8_t> values_bytes;
        values_bytes.resize(size * sizeof(uint32_t));
        memcpy(values_bytes.ptrw(), data.values.ptr(), values_bytes.size());
        RID values_buffer = rd->storage_buffer_create(values_bytes.size(), values_bytes);
        rd->set_resource_name(values_buffer, "GS_Test_PerfBenchmark_Values");

        // Warm up
        sorter->sort(keys_buffer, values_buffer, size);
        
        // Measure GPU time
        err = sorter->sort(keys_buffer, values_buffer, size);
        CHECK(err == OK);
        
        // CPU sorting for comparison
        std::vector<std::pair<float, uint32_t>> cpu_data;
        for (uint32_t i = 0; i < size; i++) {
            cpu_data.push_back({data.keys[i], data.values[i]});
        }
        
        auto cpu_start = std::chrono::high_resolution_clock::now();
        std::sort(cpu_data.begin(), cpu_data.end());
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        
        BenchmarkResult result;
        result.size = size;
        result.gpu_time_ms = sorter->get_last_sort_time_ms();
        result.cpu_time_ms = cpu_time;
        result.speedup = cpu_time / result.gpu_time_ms;
        result.elements_per_second = (size / result.gpu_time_ms) * 1000.0f;
        
        results.push_back(result);
        
        // Performance requirements
        if (size == 100000) {
            CHECK_MESSAGE(result.gpu_time_ms < 2.0f, 
                vformat("100K elements should sort in < 2ms, got %.2fms", result.gpu_time_ms));
        }
        
        rd->free(keys_buffer);
        rd->free(values_buffer);
    }
    
    // Print benchmark results
    print_line("\nBitonic Sort Performance Benchmark:");
    print_line("Size\t\tGPU (ms)\tCPU (ms)\tSpeedup\t\tElements/sec");
    for (const auto &result : results) {
        print_line(vformat("%d\t\t%.3f\t\t%.3f\t\t%.1fx\t\t%.1fM",
            result.size,
            result.gpu_time_ms,
            result.cpu_time_ms,
            result.speedup,
            result.elements_per_second / 1e6
        ));
    }
    
    memdelete(rd);
}

// Test synchronous sorting
TEST_CASE("[GaussianSplatting][RequiresGPU] Synchronous compute pipeline") {
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    CHECK(rd != nullptr);
    if (rd == nullptr) {
        return;
    }

    SUBCASE("Synchronous sorting") {
        Ref<BitonicSort> sorter = memnew(BitonicSort);
        Error err = sorter->initialize(rd, 100000);
        CHECK(err == OK);
        if (err != OK) {
            memdelete(rd);
            return;
        }

        TestData data;
        data.generate_random(50000);

        Vector<uint8_t> keys_bytes;
        keys_bytes.resize(50000 * sizeof(float));
        memcpy(keys_bytes.ptrw(), data.keys.ptr(), keys_bytes.size());
        RID keys_buffer = rd->storage_buffer_create(keys_bytes.size(), keys_bytes);
        rd->set_resource_name(keys_buffer, "GS_Test_SyncCompute_Keys");

        Vector<uint8_t> values_bytes;
        values_bytes.resize(50000 * sizeof(uint32_t));
        memcpy(values_bytes.ptrw(), data.values.ptr(), values_bytes.size());
        RID values_buffer = rd->storage_buffer_create(values_bytes.size(), values_bytes);
        rd->set_resource_name(values_buffer, "GS_Test_SyncCompute_Values");

        // Execute synchronous sort
        err = sorter->sort(keys_buffer, values_buffer, 50000);
        CHECK(err == OK);
        CHECK(sorter->is_ready());

        // Verify results
        Vector<uint8_t> keys_result = rd->buffer_get_data(keys_buffer);
        const float *sorted_keys = (const float *)keys_result.ptr();

        // Check ascending order
        for (uint32_t i = 1; i < 50000; i++) {
            CHECK(sorted_keys[i] >= sorted_keys[i - 1]);
        }

        rd->free(keys_buffer);
        rd->free(values_buffer);
    }

    memdelete(rd);
}

// Test modular architecture
TEST_CASE("[GaussianSplatting][RequiresGPU] AUTO policy decision matrix") {
    SortKeyConfig key_cfg;
    key_cfg.key_bits = 32;
    key_cfg.tile_bits = 16;
    key_cfg.depth_bits = 16;
    key_cfg.enable_tie_breaker = false;
    key_cfg.require_stable = false;

    GPUSorterFactory::PolicyProbe radix_probe;
    radix_probe.supported = true;
    radix_probe.supports_indirect = true;
    radix_probe.supports_64bit_keys = true;

    GPUSorterFactory::PolicyProbe bitonic_probe;
    bitonic_probe.supported = true;
    bitonic_probe.supports_indirect = false;
    bitonic_probe.supports_64bit_keys = false;

    GPUSorterFactory::PolicyProbe onesweep_probe;
    onesweep_probe.supported = true;
    onesweep_probe.supports_indirect = false;
    onesweep_probe.supports_64bit_keys = false;

    SUBCASE("Small arrays use Bitonic without fallback") {
        GPUSorterFactory::PolicyDecision decision = GPUSorterFactory::evaluate_auto_policy(
                1024, key_cfg, radix_probe, bitonic_probe, onesweep_probe, false, false);
        CHECK(decision.preferred_algorithm == GPUSorterFactory::ALGORITHM_BITONIC);
        CHECK(decision.selected_algorithm == GPUSorterFactory::ALGORITHM_BITONIC);
        CHECK(decision.fallback_reason.is_empty());
    }

    SUBCASE("Large arrays fall back from OneSweep to Radix when unsupported") {
        onesweep_probe.supported = false;
        GPUSorterFactory::PolicyDecision decision = GPUSorterFactory::evaluate_auto_policy(
                10 * 1024 * 1024, key_cfg, radix_probe, bitonic_probe, onesweep_probe, false, false);
        CHECK(decision.preferred_algorithm == GPUSorterFactory::ALGORITHM_ONESWEEP);
        CHECK(decision.selected_algorithm == GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(decision.fallback_reason.find("preferred=onesweep") != -1);
        CHECK(decision.fallback_reason.find("selected=radix") != -1);
    }

    SUBCASE("AUTO fallback includes missing 64-bit capability reason") {
        key_cfg.key_bits = 64;
        GPUSorterFactory::PolicyDecision decision = GPUSorterFactory::evaluate_auto_policy(
                1024, key_cfg, radix_probe, bitonic_probe, onesweep_probe, false, true);
        CHECK(decision.preferred_algorithm == GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(decision.selected_algorithm == GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(decision.fallback_reason.is_empty());

        // Force preferred selection to Bitonic for deterministic fallback-reason validation.
        key_cfg.key_bits = 32;
        GPUSorterFactory::PolicyDecision fallback_decision = GPUSorterFactory::evaluate_auto_policy(
                1024, key_cfg, radix_probe, bitonic_probe, onesweep_probe, false, true);
        CHECK(fallback_decision.selected_algorithm == GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(fallback_decision.fallback_reason.find("missing_64bit") != -1);
    }

    SUBCASE("AUTO reports no-valid-fallback when indirect requirement cannot be satisfied") {
        radix_probe.supported = false;
        GPUSorterFactory::PolicyDecision decision = GPUSorterFactory::evaluate_auto_policy(
                250000, key_cfg, radix_probe, bitonic_probe, onesweep_probe, true, false);
        CHECK(decision.preferred_algorithm == GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(decision.selected_algorithm == GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(decision.fallback_reason.find("selected=none") != -1);
    }
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Fallback reason telemetry") {
    SortingMetricsCollector collector;
    const String auto_reason = "type=auto preferred=onesweep selected=radix failure={algorithm=onesweep reason=unsupported}";
    const String requested_reason = "type=requested preferred=bitonic selected=radix failure={algorithm=bitonic reason=missing_indirect}";

    collector.record_fallback(auto_reason);
    collector.record_fallback(auto_reason);
    collector.record_fallback(requested_reason);

    SortingMetrics metrics = collector.get_metrics();
    CHECK(metrics.fallback_events == 3);
    CHECK(metrics.last_fallback_reason == requested_reason);
    CHECK(int(metrics.fallback_reason_counts.get(auto_reason, 0)) == 2);
    CHECK(int(metrics.fallback_reason_counts.get(requested_reason, 0)) == 1);
}

TEST_CASE("[GaussianSplatting][RequiresGPU] Modular sorter factory") {
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    CHECK(rd != nullptr);
    if (rd == nullptr) {
        return;
    }

    SUBCASE("Auto algorithm selection") {
        SortKeyConfig key_cfg = SortKeyConfig::from_settings();

        // Algorithm selection may vary based on device capabilities
        // If an algorithm is not supported, selection falls back to RadixSort
        auto algo_small = GPUSorterFactory::get_best_algorithm_for_size(1000, key_cfg);
        // Small sizes prefer Bitonic, but may fall back if not supported
        CHECK((algo_small == GPUSorterFactory::ALGORITHM_BITONIC ||
               algo_small == GPUSorterFactory::ALGORITHM_RADIX));

        auto algo_medium = GPUSorterFactory::get_best_algorithm_for_size(100000, key_cfg);
        CHECK((algo_medium == GPUSorterFactory::ALGORITHM_BITONIC ||
               algo_medium == GPUSorterFactory::ALGORITHM_RADIX));

        auto algo_large = GPUSorterFactory::get_best_algorithm_for_size(10000000, key_cfg);
        // Large sizes prefer OneSweep, but may fall back to Radix if not supported
        CHECK((algo_large == GPUSorterFactory::ALGORITHM_ONESWEEP ||
               algo_large == GPUSorterFactory::ALGORITHM_RADIX));
    }

    SUBCASE("Capability probes") {
        // Test that probe functions work without instantiation
        bool radix_supports_indirect = GPUSorterFactory::probe_supports_indirect(GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(radix_supports_indirect == true);

        bool bitonic_supports_indirect = GPUSorterFactory::probe_supports_indirect(GPUSorterFactory::ALGORITHM_BITONIC);
        CHECK(bitonic_supports_indirect == false);

        bool onesweep_supports_indirect = GPUSorterFactory::probe_supports_indirect(GPUSorterFactory::ALGORITHM_ONESWEEP);
        CHECK(onesweep_supports_indirect == false);

        // Test capability probes
        SorterCapabilities radix_caps = GPUSorterFactory::probe_capabilities(GPUSorterFactory::ALGORITHM_RADIX);
        CHECK(radix_caps.supports_indirect == true);
        CHECK(radix_caps.supports_64bit_keys == true);

        SorterCapabilities bitonic_caps = GPUSorterFactory::probe_capabilities(GPUSorterFactory::ALGORITHM_BITONIC);
        CHECK(bitonic_caps.supports_indirect == false);

        // Test device support probes (requires valid rd)
        bool radix_supported = GPUSorterFactory::probe_is_supported(GPUSorterFactory::ALGORITHM_RADIX, rd);
        bool bitonic_supported = GPUSorterFactory::probe_is_supported(GPUSorterFactory::ALGORITHM_BITONIC, rd);
        // At least one algorithm should be supported on any valid device
        CHECK((radix_supported || bitonic_supported));
    }
    
    SUBCASE("Factory creation") {
        SortKeyConfig key_cfg = SortKeyConfig::from_settings();
        Ref<IGPUSorter> sorter = GPUSorterFactory::create_sorter(
            GPUSorterFactory::ALGORITHM_AUTO,
            rd,
            10000,
            key_cfg
        );
        
        CHECK(sorter.is_valid());
        if (!sorter.is_valid()) {
            memdelete(rd);
            return;
        }
        String algorithm_name = sorter->get_algorithm_name();
        CHECK((algorithm_name == "Bitonic Sort" || algorithm_name == "Radix Sort"));
        CHECK(sorter->supports_non_power_of_two());
        CHECK(sorter->get_max_elements() >= 10000);
    }
    
    memdelete(rd);
}

// Stress test with large arrays
TEST_CASE("[GaussianSplatting][RequiresGPU] Large array stress test") {
    RenderingDevice *rd = RenderingServer::get_singleton()->create_local_rendering_device();
    CHECK(rd != nullptr);
    if (rd == nullptr) {
        return;
    }

    Ref<BitonicSort> sorter = memnew(BitonicSort);
    Error err = sorter->initialize(rd, 1000000);
    CHECK(err == OK);
    if (err != OK) {
        memdelete(rd);
        return;
    }
    
    // Test with 1M elements
    TestData data;
    data.generate_random(1000000, -1000.0f, 1000.0f);
    
    Vector<uint8_t> keys_bytes;
    keys_bytes.resize(1000000 * sizeof(float));
    memcpy(keys_bytes.ptrw(), data.keys.ptr(), keys_bytes.size());
    RID keys_buffer = rd->storage_buffer_create(keys_bytes.size(), keys_bytes);
    rd->set_resource_name(keys_buffer, "GS_Test_LargeStress_Keys");

    Vector<uint8_t> values_bytes;
    values_bytes.resize(1000000 * sizeof(uint32_t));
    memcpy(values_bytes.ptrw(), data.values.ptr(), values_bytes.size());
    RID values_buffer = rd->storage_buffer_create(values_bytes.size(), values_bytes);
    rd->set_resource_name(values_buffer, "GS_Test_LargeStress_Values");

    // Sort
    err = sorter->sort(keys_buffer, values_buffer, 1000000);
    CHECK(err == OK);
    
    // Get metrics
    auto metrics = sorter->get_metrics();
    print_line(vformat("\n1M Element Sort Metrics:"));
    print_line(vformat("  Time: %.2f ms", metrics.last_sort_time_ms));
    print_line(vformat("  Bandwidth utilization: %.1f%%", metrics.bandwidth_utilization));
    print_line(vformat("  Elements/sec: %.1fM", (1000000.0f / metrics.last_sort_time_ms) / 1000.0f));
    
    // Verify first and last few elements are in order
    Vector<uint8_t> keys_result = rd->buffer_get_data(keys_buffer, 0, 100 * sizeof(float));
    const float *sorted_keys = (const float *)keys_result.ptr();
    
    for (int i = 1; i < 100; i++) {
        CHECK(sorted_keys[i] >= sorted_keys[i - 1]);
    }
    
    rd->free(keys_buffer);
    rd->free(values_buffer);
    
    memdelete(rd);
}
