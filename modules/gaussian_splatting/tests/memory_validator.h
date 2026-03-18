/*
 * Memory Validation System for Gaussian Splatting
 * Detects memory leaks, tracks allocations, and validates cleanup.
 */

#ifndef MEMORY_VALIDATOR_H
#define MEMORY_VALIDATOR_H

#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include <atomic>
#include <mutex>

class RenderingDevice;

// Memory allocation tracking
struct MemoryAllocation {
    void *address = nullptr;
    size_t size = 0;
    String type;  // CPU, GPU, Buffer, etc.
    String source_file;
    int source_line = 0;
    uint64_t timestamp = 0;
    uint32_t thread_id = 0;
    bool is_gpu = false;
    RID gpu_rid;  // For GPU resources

    String to_string() const;
};

// Memory statistics
struct MemoryStats {
    // Current state
    size_t current_cpu_bytes = 0;
    size_t current_gpu_bytes = 0;
    uint32_t current_allocations = 0;
    uint32_t current_gpu_resources = 0;

    // Peak values
    size_t peak_cpu_bytes = 0;
    size_t peak_gpu_bytes = 0;
    uint32_t peak_allocations = 0;
    uint32_t peak_gpu_resources = 0;

    // Totals
    uint64_t total_allocated_bytes = 0;
    uint64_t total_freed_bytes = 0;
    uint32_t total_allocations = 0;
    uint32_t total_deallocations = 0;

    // Leaks
    uint32_t leaked_allocations = 0;
    size_t leaked_bytes = 0;
    LocalVector<MemoryAllocation> leak_details;

    // Fragmentation
    float fragmentation_ratio = 0.0f;
    uint32_t fragmented_blocks = 0;

    Dictionary to_dict() const;
    void print_summary() const;
};

// GPU memory tracking
class GPUMemoryTracker {
private:
    RenderingDevice *rd = nullptr;
    HashMap<RID, MemoryAllocation> gpu_allocations;
    std::mutex mutex;

    // Memory pools tracking
    struct PoolInfo {
        size_t allocated = 0;
        size_t used = 0;
        uint32_t blocks = 0;
    };
    HashMap<String, PoolInfo> memory_pools;

public:
    GPUMemoryTracker(RenderingDevice *p_rd);

    // Track GPU resource allocation/deallocation
    void track_buffer_creation(RID rid, size_t size, const String &name);
    void track_texture_creation(RID rid, size_t size, const String &name);
    void track_shader_creation(RID rid, const String &name);
    void track_resource_free(RID rid);

    // Memory pool tracking
    void track_pool_allocation(const String &pool_name, size_t size);
    void track_pool_free(const String &pool_name, size_t size);

    // Query current state
    size_t get_total_gpu_memory() const;
    uint32_t get_resource_count() const;
    LocalVector<MemoryAllocation> get_active_allocations() const;

    // Validation
    bool validate_no_leaks();
    LocalVector<MemoryAllocation> find_leaks();
    void print_leak_report();

    // Accessor
    RenderingDevice *get_rd() const { return rd; }
};

// Main memory validator
class MemoryValidator : public RefCounted {
    GDCLASS(MemoryValidator, RefCounted);

protected:
    static void _bind_methods();

private:
    static MemoryValidator *singleton;

    // CPU memory tracking
    HashMap<void *, MemoryAllocation> cpu_allocations;
    std::mutex cpu_mutex;

    // GPU tracking
    GPUMemoryTracker *gpu_tracker = nullptr;

    // Statistics
    MemoryStats stats;
    std::atomic<size_t> total_cpu_bytes{0};
    std::atomic<size_t> total_gpu_bytes{0};
    std::atomic<uint32_t> allocation_counter{0};

    // Configuration
    bool tracking_enabled = true;
    bool strict_mode = false;  // Fail on any leak
    size_t max_memory_threshold = 0;  // 0 = no limit
    bool track_stack_traces = false;

    // Leak detection
    struct LeakPattern {
        String type;
        size_t min_size;
        size_t max_size;
        uint32_t count;
    };
    LocalVector<LeakPattern> detected_patterns;

    // Helper methods
    void update_peak_stats();
    void analyze_leak_patterns();
    String get_stack_trace() const;

public:
    MemoryValidator();
    ~MemoryValidator();

    static MemoryValidator *get_singleton() { return singleton; }

    // Initialize with rendering device for GPU tracking
    Error initialize(RenderingDevice *p_rd);

    // CPU memory tracking
    void track_allocation(void *ptr, size_t size, const String &type = "Generic");
    void track_allocation_with_source(void *ptr, size_t size, const String &file, int line);
    void track_deallocation(void *ptr);
    void track_reallocation(void *old_ptr, void *new_ptr, size_t new_size);

    // Validation methods
    bool validate_no_leaks();
    bool validate_memory_usage(size_t max_bytes);
    bool validate_allocation_count(uint32_t max_count);
    bool validate_no_fragmentation(float max_ratio = 0.3f);

    // Stress testing
    void stress_test_allocation_patterns(uint32_t iterations);
    void stress_test_fragmentation(uint32_t iterations);
    void stress_test_concurrent_allocations(uint32_t threads, uint32_t allocations_per_thread);
    void stress_test_gpu_allocations(uint32_t iterations);

    // Defragmentation testing
    void test_defragmentation_simple();
    void test_defragmentation_complex();
    float measure_fragmentation() const;

    // Memory pressure simulation
    void simulate_memory_pressure(size_t target_bytes);
    void simulate_oom_conditions();

    // Reporting
    MemoryStats get_stats() const { return stats; }
    void print_report() const;
    void save_report(const String &filepath) const;
    Dictionary get_report_dict() const;

    // Leak detection and analysis
    LocalVector<MemoryAllocation> find_leaks() const;
    void print_leak_details() const;
    void analyze_allocation_patterns();

    // Configuration
    void set_tracking_enabled(bool enabled) { tracking_enabled = enabled; }
    void set_strict_mode(bool strict) { strict_mode = strict; }
    void set_max_memory_threshold(size_t bytes) { max_memory_threshold = bytes; }
    void set_track_stack_traces(bool track) { track_stack_traces = track; }

    // Reset/Clear
    void reset();
    void clear_stats();

    // Snapshot functionality
    void take_snapshot(const String &name);
    void compare_snapshots(const String &before, const String &after);
    LocalVector<MemoryAllocation> get_allocations_between_snapshots(const String &before, const String &after);

    // GPU specific
    GPUMemoryTracker *get_gpu_tracker() { return gpu_tracker; }
};

// RAII helper for scoped memory tracking
class ScopedMemoryTracker {
private:
    String scope_name;
    MemoryStats initial_stats;
    uint64_t start_time;

public:
    ScopedMemoryTracker(const String &name);
    ~ScopedMemoryTracker();

    MemoryStats get_delta() const;
    void print_summary() const;
};

// Memory allocation hooks
class MemoryHooks {
public:
    static void *tracked_malloc(size_t size, const char *file = nullptr, int line = 0);
    static void tracked_free(void *ptr);
    static void *tracked_realloc(void *ptr, size_t new_size, const char *file = nullptr, int line = 0);

    static void install_hooks();
    static void uninstall_hooks();
};

// Macros for easy tracking
#ifdef DEBUG_ENABLED
    #define TRACK_ALLOC(ptr, size) \
        MemoryValidator::get_singleton()->track_allocation_with_source(ptr, size, __FILE__, __LINE__)
    #define TRACK_FREE(ptr) \
        MemoryValidator::get_singleton()->track_deallocation(ptr)
    #define SCOPED_MEMORY_TRACK(name) \
        ScopedMemoryTracker _memory_tracker(name)
#else
    #define TRACK_ALLOC(ptr, size)
    #define TRACK_FREE(ptr)
    #define SCOPED_MEMORY_TRACK(name)
#endif

// Test utilities
class MemoryValidationTests {
public:
    static bool test_no_leaks_after_init();
    static bool test_streaming_memory_lifecycle();
    static bool test_gpu_resource_cleanup();
    static bool test_concurrent_allocations();
    static bool test_memory_pools();
    static bool test_defragmentation();
    static bool test_oom_handling();
    static bool test_leak_detection();

    static void run_all_tests();
    static void run_stress_tests();
};

#endif // MEMORY_VALIDATOR_H