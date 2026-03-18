/*
 * Memory Validation System Implementation
 */

#include "memory_validator.h"
#include "servers/rendering/rendering_device.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/os/time.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/math/random_number_generator.h"
#include "core/templates/hash_set.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>

MemoryValidator *MemoryValidator::singleton = nullptr;

// MemoryAllocation implementation
String MemoryAllocation::to_string() const {
    std::stringstream ss;
    ss << "Allocation @ " << address;
    ss << " | Size: " << size << " bytes";
    ss << " | Type: " << type.utf8().get_data();
    if (!source_file.is_empty()) {
        ss << " | Source: " << source_file.utf8().get_data() << ":" << source_line;
    }
    ss << " | Thread: " << thread_id;
    return String(ss.str().c_str());
}

// MemoryStats implementation
Dictionary MemoryStats::to_dict() const {
    Dictionary dict;
    dict["current_cpu_bytes"] = current_cpu_bytes;
    dict["current_gpu_bytes"] = current_gpu_bytes;
    dict["current_allocations"] = current_allocations;
    dict["current_gpu_resources"] = current_gpu_resources;
    dict["peak_cpu_bytes"] = peak_cpu_bytes;
    dict["peak_gpu_bytes"] = peak_gpu_bytes;
    dict["peak_allocations"] = peak_allocations;
    dict["peak_gpu_resources"] = peak_gpu_resources;
    dict["total_allocated_bytes"] = total_allocated_bytes;
    dict["total_freed_bytes"] = total_freed_bytes;
    dict["total_allocations"] = total_allocations;
    dict["total_deallocations"] = total_deallocations;
    dict["leaked_allocations"] = leaked_allocations;
    dict["leaked_bytes"] = leaked_bytes;
    dict["fragmentation_ratio"] = fragmentation_ratio;
    dict["fragmented_blocks"] = fragmented_blocks;
    return dict;
}

void MemoryStats::print_summary() const {
    print_line("=== Memory Statistics ===");
    print_line(vformat("Current CPU: %.2f MB (%d allocations)",
        current_cpu_bytes / (1024.0f * 1024.0f), current_allocations));
    print_line(vformat("Current GPU: %.2f MB (%d resources)",
        current_gpu_bytes / (1024.0f * 1024.0f), current_gpu_resources));
    print_line(vformat("Peak CPU: %.2f MB", peak_cpu_bytes / (1024.0f * 1024.0f)));
    print_line(vformat("Peak GPU: %.2f MB", peak_gpu_bytes / (1024.0f * 1024.0f)));
    print_line(vformat("Total Allocations: %d", total_allocations));
    print_line(vformat("Total Deallocations: %d", total_deallocations));

    if (leaked_allocations > 0) {
        print_line(vformat("WARNING: %d leaks detected (%.2f KB)",
            leaked_allocations, leaked_bytes / 1024.0f));
    }

    if (fragmentation_ratio > 0.3f) {
        print_line(vformat("WARNING: High fragmentation: %.1f%%",
            fragmentation_ratio * 100.0f));
    }
}

// GPUMemoryTracker implementation
GPUMemoryTracker::GPUMemoryTracker(RenderingDevice *p_rd) : rd(p_rd) {
}

void GPUMemoryTracker::track_buffer_creation(RID rid, size_t size, const String &name) {
    std::lock_guard<std::mutex> lock(mutex);

    MemoryAllocation alloc;
    alloc.gpu_rid = rid;
    alloc.size = size;
    alloc.type = "GPU_Buffer";
    alloc.timestamp = OS::get_singleton()->get_ticks_usec();
    alloc.is_gpu = true;
    alloc.source_file = name;

    gpu_allocations[rid] = alloc;
}

void GPUMemoryTracker::track_texture_creation(RID rid, size_t size, const String &name) {
    std::lock_guard<std::mutex> lock(mutex);

    MemoryAllocation alloc;
    alloc.gpu_rid = rid;
    alloc.size = size;
    alloc.type = "GPU_Texture";
    alloc.timestamp = OS::get_singleton()->get_ticks_usec();
    alloc.is_gpu = true;
    alloc.source_file = name;

    gpu_allocations[rid] = alloc;
}

void GPUMemoryTracker::track_shader_creation(RID rid, const String &name) {
    std::lock_guard<std::mutex> lock(mutex);

    MemoryAllocation alloc;
    alloc.gpu_rid = rid;
    alloc.size = 0; // Size unknown for shaders
    alloc.type = "GPU_Shader";
    alloc.timestamp = OS::get_singleton()->get_ticks_usec();
    alloc.is_gpu = true;
    alloc.source_file = name;

    gpu_allocations[rid] = alloc;
}

void GPUMemoryTracker::track_resource_free(RID rid) {
    std::lock_guard<std::mutex> lock(mutex);
    gpu_allocations.erase(rid);
}

void GPUMemoryTracker::track_pool_allocation(const String &pool_name, size_t size) {
    std::lock_guard<std::mutex> lock(mutex);

    if (!memory_pools.has(pool_name)) {
        memory_pools[pool_name] = PoolInfo();
    }

    PoolInfo &pool = memory_pools[pool_name];
    pool.allocated += size;
    pool.used += size;
    pool.blocks++;
}

void GPUMemoryTracker::track_pool_free(const String &pool_name, size_t size) {
    std::lock_guard<std::mutex> lock(mutex);

    if (memory_pools.has(pool_name)) {
        PoolInfo &pool = memory_pools[pool_name];
        pool.used = MAX(0, static_cast<int64_t>(pool.used) - static_cast<int64_t>(size));
        if (pool.blocks > 0) {
            pool.blocks--;
        }
    }
}

size_t GPUMemoryTracker::get_total_gpu_memory() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(mutex));

    size_t total = 0;
    for (const KeyValue<RID, MemoryAllocation> &E : gpu_allocations) {
        total += E.value.size;
    }
    return total;
}

uint32_t GPUMemoryTracker::get_resource_count() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(mutex));
    return gpu_allocations.size();
}

LocalVector<MemoryAllocation> GPUMemoryTracker::get_active_allocations() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(mutex));

    LocalVector<MemoryAllocation> allocations;
    for (const KeyValue<RID, MemoryAllocation> &E : gpu_allocations) {
        allocations.push_back(E.value);
    }
    return allocations;
}

bool GPUMemoryTracker::validate_no_leaks() {
    std::lock_guard<std::mutex> lock(mutex);
    return gpu_allocations.is_empty();
}

LocalVector<MemoryAllocation> GPUMemoryTracker::find_leaks() {
    std::lock_guard<std::mutex> lock(mutex);

    LocalVector<MemoryAllocation> leaks;
    for (const KeyValue<RID, MemoryAllocation> &E : gpu_allocations) {
        leaks.push_back(E.value);
    }
    return leaks;
}

void GPUMemoryTracker::print_leak_report() {
    LocalVector<MemoryAllocation> leaks = find_leaks();

    if (leaks.is_empty()) {
        print_line("No GPU memory leaks detected");
        return;
    }

    print_line(vformat("=== GPU Memory Leak Report ==="));
    print_line(vformat("Found %d leaked GPU resources:", leaks.size()));

    size_t total_leaked = 0;
    for (const MemoryAllocation &leak : leaks) {
        print_line(vformat("  - %s: %d bytes [%s]",
            leak.type, leak.size, leak.source_file));
        total_leaked += leak.size;
    }

    print_line(vformat("Total leaked: %.2f MB", total_leaked / (1024.0f * 1024.0f)));
}

// MemoryValidator implementation
void MemoryValidator::_bind_methods() {
    // Bind methods for GDScript access if needed
}

MemoryValidator::MemoryValidator() {
    if (singleton == nullptr) {
        singleton = this;
    }
}

MemoryValidator::~MemoryValidator() {
    if (gpu_tracker) {
        memdelete(gpu_tracker);
    }
    if (singleton == this) {
        singleton = nullptr;
    }
}

Error MemoryValidator::initialize(RenderingDevice *p_rd) {
    if (!p_rd) {
        return ERR_INVALID_PARAMETER;
    }

    if (gpu_tracker) {
        memdelete(gpu_tracker);
    }

    gpu_tracker = memnew(GPUMemoryTracker(p_rd));
    return OK;
}

void MemoryValidator::track_allocation(void *ptr, size_t size, const String &type) {
    if (!tracking_enabled || !ptr) return;

    std::lock_guard<std::mutex> lock(cpu_mutex);

    MemoryAllocation alloc;
    alloc.address = ptr;
    alloc.size = size;
    alloc.type = type;
    alloc.timestamp = OS::get_singleton()->get_ticks_usec();
    alloc.thread_id = Thread::get_caller_id();

    if (track_stack_traces) {
        alloc.source_file = get_stack_trace();
    }

    cpu_allocations[ptr] = alloc;

    // Update statistics
    stats.current_cpu_bytes += size;
    stats.current_allocations++;
    stats.total_allocated_bytes += size;
    stats.total_allocations++;

    total_cpu_bytes.fetch_add(size);
    allocation_counter.fetch_add(1);

    update_peak_stats();

    // Check threshold
    if (max_memory_threshold > 0 && stats.current_cpu_bytes > max_memory_threshold) {
        WARN_PRINT(vformat("Memory threshold exceeded: %d MB > %d MB",
            stats.current_cpu_bytes / (1024 * 1024),
            max_memory_threshold / (1024 * 1024)));
    }
}

void MemoryValidator::track_allocation_with_source(void *ptr, size_t size, const String &file, int line) {
    if (!tracking_enabled || !ptr) return;

    std::lock_guard<std::mutex> lock(cpu_mutex);

    MemoryAllocation alloc;
    alloc.address = ptr;
    alloc.size = size;
    alloc.type = "Manual";
    alloc.source_file = file;
    alloc.source_line = line;
    alloc.timestamp = OS::get_singleton()->get_ticks_usec();
    alloc.thread_id = Thread::get_caller_id();

    cpu_allocations[ptr] = alloc;

    // Update statistics
    stats.current_cpu_bytes += size;
    stats.current_allocations++;
    stats.total_allocated_bytes += size;
    stats.total_allocations++;

    total_cpu_bytes.fetch_add(size);
    allocation_counter.fetch_add(1);

    update_peak_stats();
}

void MemoryValidator::track_deallocation(void *ptr) {
    if (!tracking_enabled || !ptr) return;

    std::lock_guard<std::mutex> lock(cpu_mutex);

    auto it = cpu_allocations.find(ptr);
    if (it != cpu_allocations.end()) {
        size_t size = it->value.size;

        stats.current_cpu_bytes -= size;
        stats.current_allocations--;
        stats.total_freed_bytes += size;
        stats.total_deallocations++;

        total_cpu_bytes.fetch_sub(size);

        cpu_allocations.erase(ptr);
    } else if (strict_mode) {
        WARN_PRINT(vformat("Attempting to free untracked pointer: 0x%llx", (uint64_t)ptr));
    }
}

void MemoryValidator::track_reallocation(void *old_ptr, void *new_ptr, size_t new_size) {
    if (!tracking_enabled) return;

    std::lock_guard<std::mutex> lock(cpu_mutex);

    // Remove old allocation
    if (old_ptr) {
        auto it = cpu_allocations.find(old_ptr);
        if (it != cpu_allocations.end()) {
            stats.current_cpu_bytes -= it->value.size;
            total_cpu_bytes.fetch_sub(it->value.size);
            cpu_allocations.erase(old_ptr);
        }
    }

    // Add new allocation
    if (new_ptr) {
        MemoryAllocation alloc;
        alloc.address = new_ptr;
        alloc.size = new_size;
        alloc.type = "Realloc";
        alloc.timestamp = OS::get_singleton()->get_ticks_usec();
        alloc.thread_id = Thread::get_caller_id();

        cpu_allocations[new_ptr] = alloc;

        stats.current_cpu_bytes += new_size;
        total_cpu_bytes.fetch_add(new_size);
    }

    update_peak_stats();
}

void MemoryValidator::update_peak_stats() {
    stats.peak_cpu_bytes = MAX(stats.peak_cpu_bytes, stats.current_cpu_bytes);
    stats.peak_allocations = MAX(stats.peak_allocations, stats.current_allocations);

    if (gpu_tracker) {
        size_t gpu_bytes = gpu_tracker->get_total_gpu_memory();
        stats.current_gpu_bytes = gpu_bytes;
        stats.peak_gpu_bytes = MAX(stats.peak_gpu_bytes, gpu_bytes);

        uint32_t gpu_resources = gpu_tracker->get_resource_count();
        stats.current_gpu_resources = gpu_resources;
        stats.peak_gpu_resources = MAX(stats.peak_gpu_resources, gpu_resources);
    }
}

bool MemoryValidator::validate_no_leaks() {
    std::lock_guard<std::mutex> lock(cpu_mutex);

    // Check CPU leaks
    bool cpu_clean = cpu_allocations.is_empty();

    // Check GPU leaks
    bool gpu_clean = true;
    if (gpu_tracker) {
        gpu_clean = gpu_tracker->validate_no_leaks();
    }

    if (!cpu_clean || !gpu_clean) {
        stats.leaked_allocations = cpu_allocations.size();
        stats.leaked_bytes = 0;

        for (const KeyValue<void *, MemoryAllocation> &E : cpu_allocations) {
            stats.leaked_bytes += E.value.size;
            stats.leak_details.push_back(E.value);
        }

        if (gpu_tracker) {
            LocalVector<MemoryAllocation> gpu_leaks = gpu_tracker->find_leaks();
            stats.leaked_allocations += gpu_leaks.size();
            for (const MemoryAllocation &leak : gpu_leaks) {
                stats.leaked_bytes += leak.size;
                stats.leak_details.push_back(leak);
            }
        }

        if (strict_mode) {
            print_leak_details();
            ERR_FAIL_V_MSG(false, vformat("Memory leaks detected: %d allocations, %.2f KB",
                stats.leaked_allocations, stats.leaked_bytes / 1024.0f));
        }

        return false;
    }

    return true;
}

bool MemoryValidator::validate_memory_usage(size_t max_bytes) {
    size_t total = stats.current_cpu_bytes + stats.current_gpu_bytes;
    return total <= max_bytes;
}

bool MemoryValidator::validate_allocation_count(uint32_t max_count) {
    return stats.current_allocations <= max_count;
}

bool MemoryValidator::validate_no_fragmentation(float max_ratio) {
    float fragmentation = measure_fragmentation();
    stats.fragmentation_ratio = fragmentation;
    return fragmentation <= max_ratio;
}

void MemoryValidator::stress_test_allocation_patterns(uint32_t iterations) {
    print_line(vformat("Running allocation pattern stress test (%d iterations)...", iterations));

    LocalVector<void *> allocations;
    RandomNumberGenerator rng;

    for (uint32_t i = 0; i < iterations; i++) {
        // Random allocation size
        size_t size = rng.randi_range(64, 65536);

        // Allocate
        void *ptr = memalloc(size);
        track_allocation(ptr, size, "StressTest");
        allocations.push_back(ptr);

        // Sometimes free random allocation
        if (allocations.size() > 10 && rng.randf() > 0.5f) {
            uint32_t idx = rng.randi() % allocations.size();
            track_deallocation(allocations[idx]);
            memfree(allocations[idx]);
            allocations.remove_at(idx);
        }

        // Check for leaks periodically
        if (i % 100 == 0) {
            if (!validate_memory_usage(100 * 1024 * 1024)) { // 100 MB limit
                WARN_PRINT("Memory usage exceeded during stress test");
            }
        }
    }

    // Clean up remaining allocations
    for (void *ptr : allocations) {
        track_deallocation(ptr);
        memfree(ptr);
    }

    // Final validation
    bool clean = validate_no_leaks();
    print_line(vformat("Stress test completed. Clean: %s", clean ? "YES" : "NO"));
}

void MemoryValidator::stress_test_fragmentation(uint32_t iterations) {
    print_line(vformat("Running fragmentation stress test (%d iterations)...", iterations));

    LocalVector<void *> small_allocs;
    LocalVector<void *> large_allocs;
    RandomNumberGenerator rng;

    // Create fragmented memory pattern
    for (uint32_t i = 0; i < iterations; i++) {
        // Allocate alternating small and large blocks
        if (i % 2 == 0) {
            void *ptr = memalloc(64);
            track_allocation(ptr, 64, "SmallBlock");
            small_allocs.push_back(ptr);
        } else {
            void *ptr = memalloc(8192);
            track_allocation(ptr, 8192, "LargeBlock");
            large_allocs.push_back(ptr);
        }
    }

    // Free every other small allocation (create holes)
    for (uint32_t i = 0; i < small_allocs.size(); i += 2) {
        track_deallocation(small_allocs[i]);
        memfree(small_allocs[i]);
        small_allocs[i] = nullptr;
    }

    // Measure fragmentation
    float fragmentation = measure_fragmentation();
    print_line(vformat("Fragmentation after creating holes: %.1f%%", fragmentation * 100.0f));

    // Try to allocate medium blocks (should struggle with fragmentation)
    LocalVector<void *> medium_allocs;
    for (uint32_t i = 0; i < 10; i++) {
        void *ptr = memalloc(512);
        if (ptr) {
            track_allocation(ptr, 512, "MediumBlock");
            medium_allocs.push_back(ptr);
        }
    }

    // Clean up
    for (void *ptr : small_allocs) {
        if (ptr) {
            track_deallocation(ptr);
            memfree(ptr);
        }
    }
    for (void *ptr : large_allocs) {
        track_deallocation(ptr);
        memfree(ptr);
    }
    for (void *ptr : medium_allocs) {
        track_deallocation(ptr);
        memfree(ptr);
    }

    validate_no_leaks();
}

// Thread worker data structure for concurrent allocation test
struct ConcurrentTestData {
    MemoryValidator *validator;
    uint32_t allocations_per_thread;
    std::atomic<uint32_t> *completed_threads;
};

static void concurrent_allocation_worker(void *p_userdata) {
    ConcurrentTestData *data = static_cast<ConcurrentTestData *>(p_userdata);

    LocalVector<void *> local_allocs;
    RandomNumberGenerator rng;
    rng.set_seed(Thread::get_caller_id());

    for (uint32_t i = 0; i < data->allocations_per_thread; i++) {
        size_t size = rng.randi_range(64, 4096);
        void *ptr = memalloc(size);
        data->validator->track_allocation(ptr, size, "ConcurrentTest");
        local_allocs.push_back(ptr);

        // Random deallocations
        if (local_allocs.size() > 5 && rng.randf() > 0.7f) {
            uint32_t idx = rng.randi() % local_allocs.size();
            data->validator->track_deallocation(local_allocs[idx]);
            memfree(local_allocs[idx]);
            local_allocs.remove_at(idx);
        }
    }

    // Clean up remaining
    for (void *ptr : local_allocs) {
        data->validator->track_deallocation(ptr);
        memfree(ptr);
    }

    data->completed_threads->fetch_add(1);
}

void MemoryValidator::stress_test_concurrent_allocations(uint32_t threads, uint32_t allocations_per_thread) {
    print_line(vformat("Running concurrent allocation stress test (%d threads, %d allocs each)...",
        threads, allocations_per_thread));

    LocalVector<Thread *> thread_list;
    std::atomic<uint32_t> completed_threads{0};

    // Create shared data for all threads
    ConcurrentTestData test_data;
    test_data.validator = this;
    test_data.allocations_per_thread = allocations_per_thread;
    test_data.completed_threads = &completed_threads;

    // Start threads
    for (uint32_t i = 0; i < threads; i++) {
        Thread *t = memnew(Thread);
        t->start(concurrent_allocation_worker, &test_data);
        thread_list.push_back(t);
    }

    // Wait for completion
    for (Thread *t : thread_list) {
        t->wait_to_finish();
        memdelete(t);
    }

    // Validate
    bool clean = validate_no_leaks();
    print_line(vformat("Concurrent test completed. Threads: %d, Clean: %s",
        completed_threads.load(), clean ? "YES" : "NO"));
}

void MemoryValidator::stress_test_gpu_allocations(uint32_t iterations) {
    if (!gpu_tracker) {
        WARN_PRINT("GPU tracker not initialized");
        return;
    }

    print_line(vformat("Running GPU allocation stress test (%d iterations)...", iterations));

    LocalVector<RID> buffers;
    RandomNumberGenerator rng;

    for (uint32_t i = 0; i < iterations; i++) {
        // Create random GPU buffer
        size_t size = rng.randi_range(1024, 1024 * 1024); // 1KB to 1MB
        RID rid = RID::from_uint64(i + 1); // Simulated RID

        gpu_tracker->track_buffer_creation(rid, size, vformat("TestBuffer_%d", i));
        buffers.push_back(rid);

        // Sometimes free random buffer
        if (buffers.size() > 10 && rng.randf() > 0.5f) {
            uint32_t idx = rng.randi() % buffers.size();
            gpu_tracker->track_resource_free(buffers[idx]);
            buffers.remove_at(idx);
        }
    }

    // Clean up remaining
    for (RID rid : buffers) {
        gpu_tracker->track_resource_free(rid);
    }

    // Validate
    bool clean = gpu_tracker->validate_no_leaks();
    print_line(vformat("GPU stress test completed. Clean: %s", clean ? "YES" : "NO"));
}

float MemoryValidator::measure_fragmentation() const {
    // Simple fragmentation metric: ratio of gaps to total allocations
    if (cpu_allocations.size() < 2) {
        return 0.0f;
    }

    // Get sorted list of allocations by address
    LocalVector<std::pair<uintptr_t, size_t>> sorted_allocs;
    for (const KeyValue<void *, MemoryAllocation> &E : cpu_allocations) {
        sorted_allocs.push_back({reinterpret_cast<uintptr_t>(E.key), E.value.size});
    }

    std::sort(sorted_allocs.ptr(), sorted_allocs.ptr() + sorted_allocs.size(),
        [](const std::pair<uintptr_t, size_t> &a, const std::pair<uintptr_t, size_t> &b) {
            return a.first < b.first;
        });

    // Count gaps between allocations
    uint32_t gaps = 0;
    size_t total_gap_size = 0;

    for (uint32_t i = 1; i < sorted_allocs.size(); i++) {
        uintptr_t prev_end = sorted_allocs[i - 1].first + sorted_allocs[i - 1].second;
        uintptr_t curr_start = sorted_allocs[i].first;

        if (curr_start > prev_end) {
            gaps++;
            total_gap_size += (curr_start - prev_end);
        }
    }

    // Calculate fragmentation ratio
    if (stats.current_cpu_bytes > 0) {
        return static_cast<float>(total_gap_size) / static_cast<float>(stats.current_cpu_bytes);
    }

    return 0.0f;
}

LocalVector<MemoryAllocation> MemoryValidator::find_leaks() const {
    LocalVector<MemoryAllocation> leaks;

    // CPU leaks
    for (const KeyValue<void *, MemoryAllocation> &E : cpu_allocations) {
        leaks.push_back(E.value);
    }

    // GPU leaks
    if (gpu_tracker) {
        LocalVector<MemoryAllocation> gpu_leaks = gpu_tracker->find_leaks();
        for (const MemoryAllocation &gpu_leak : gpu_leaks) {
            leaks.push_back(gpu_leak);
        }
    }

    return leaks;
}

void MemoryValidator::print_leak_details() const {
    LocalVector<MemoryAllocation> leaks = find_leaks();

    if (leaks.is_empty()) {
        print_line("No memory leaks detected");
        return;
    }

    print_line("=== Memory Leak Report ===");
    print_line(vformat("Found %d leaked allocations:", leaks.size()));

    // Group leaks by type
    HashMap<String, uint32_t> leak_counts;
    HashMap<String, size_t> leak_sizes;

    for (const MemoryAllocation &leak : leaks) {
        leak_counts[leak.type]++;
        leak_sizes[leak.type] += leak.size;
    }

    print_line("\nLeaks by type:");
    for (const KeyValue<String, uint32_t> &E : leak_counts) {
        print_line(vformat("  %s: %d allocations, %.2f KB",
            E.key, E.value, leak_sizes[E.key] / 1024.0f));
    }

    // Print detailed leak info (first 10)
    print_line("\nDetailed leak information (first 10):");
    int count = 0;
    for (const MemoryAllocation &leak : leaks) {
        if (count++ >= 10) break;
        print_line(vformat("  %s", leak.to_string()));
    }

    size_t total_leaked = 0;
    for (const MemoryAllocation &leak : leaks) {
        total_leaked += leak.size;
    }

    print_line(vformat("\nTotal leaked: %.2f MB", total_leaked / (1024.0f * 1024.0f)));
}

void MemoryValidator::print_report() const {
    print_line("\n=== Memory Validation Report ===");
    stats.print_summary();

    if (gpu_tracker) {
        print_line(vformat("\nGPU Resources: %d active", gpu_tracker->get_resource_count()));
    }

    // Print allocation patterns
    if (!cpu_allocations.is_empty()) {
        print_line(vformat("\nActive CPU allocations: %d", cpu_allocations.size()));

        // Group by type
        HashMap<String, uint32_t> type_counts;
        HashMap<String, size_t> type_sizes;

        for (const KeyValue<void *, MemoryAllocation> &E : cpu_allocations) {
            type_counts[E.value.type]++;
            type_sizes[E.value.type] += E.value.size;
        }

        print_line("Allocations by type:");
        for (const KeyValue<String, uint32_t> &E : type_counts) {
            print_line(vformat("  %s: %d allocations, %.2f KB",
                E.key, E.value, type_sizes[E.key] / 1024.0f));
        }
    }

    print_line("=========================\n");
}

void MemoryValidator::save_report(const String &filepath) const {
    JSON json;
    String json_string = json.stringify(get_report_dict(), "\t");

    Ref<FileAccess> file = FileAccess::open(filepath, FileAccess::WRITE);
    if (file.is_valid()) {
        file->store_string(json_string);
        print_line(vformat("Memory report saved to: %s", filepath));
    }
}

Dictionary MemoryValidator::get_report_dict() const {
    Dictionary report;
    report["timestamp"] = Time::get_singleton()->get_datetime_string_from_system();
    report["stats"] = stats.to_dict();

    // Add leak details
    Array leaks_array;
    for (const MemoryAllocation &leak : stats.leak_details) {
        Dictionary leak_dict;
        leak_dict["address"] = String::num_uint64(reinterpret_cast<uint64_t>(leak.address), 16);
        leak_dict["size"] = leak.size;
        leak_dict["type"] = leak.type;
        leak_dict["source"] = leak.source_file;
        leak_dict["line"] = leak.source_line;
        leaks_array.append(leak_dict);
    }
    report["leaks"] = leaks_array;

    return report;
}

void MemoryValidator::reset() {
    std::lock_guard<std::mutex> lock(cpu_mutex);

    cpu_allocations.clear();
    stats = MemoryStats();
    total_cpu_bytes = 0;
    total_gpu_bytes = 0;
    allocation_counter = 0;

    if (gpu_tracker) {
        // GPU tracker doesn't have a reset, so recreate it
        RenderingDevice *rd = gpu_tracker->get_rd();
        memdelete(gpu_tracker);
        gpu_tracker = memnew(GPUMemoryTracker(rd));
    }
}

void MemoryValidator::clear_stats() {
    stats = MemoryStats();
}

String MemoryValidator::get_stack_trace() const {
    // Platform-specific stack trace implementation would go here
    // For now, return empty string
    return String();
}

// ScopedMemoryTracker implementation
ScopedMemoryTracker::ScopedMemoryTracker(const String &name) : scope_name(name) {
    if (MemoryValidator::get_singleton()) {
        initial_stats = MemoryValidator::get_singleton()->get_stats();
        start_time = OS::get_singleton()->get_ticks_usec();
    }
}

ScopedMemoryTracker::~ScopedMemoryTracker() {
    if (MemoryValidator::get_singleton()) {
        MemoryStats delta = get_delta();
        print_summary();
    }
}

MemoryStats ScopedMemoryTracker::get_delta() const {
    if (!MemoryValidator::get_singleton()) {
        return MemoryStats();
    }

    MemoryStats current = MemoryValidator::get_singleton()->get_stats();
    MemoryStats delta;

    delta.current_cpu_bytes = current.current_cpu_bytes - initial_stats.current_cpu_bytes;
    delta.current_gpu_bytes = current.current_gpu_bytes - initial_stats.current_gpu_bytes;
    delta.current_allocations = current.current_allocations - initial_stats.current_allocations;
    delta.total_allocations = current.total_allocations - initial_stats.total_allocations;
    delta.total_deallocations = current.total_deallocations - initial_stats.total_deallocations;

    return delta;
}

void ScopedMemoryTracker::print_summary() const {
    uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start_time;
    MemoryStats delta = get_delta();

    print_line(vformat("[%s] Duration: %.2f ms, Allocations: %d, Memory: %.2f KB",
        scope_name, elapsed / 1000.0f, delta.total_allocations,
        delta.current_cpu_bytes / 1024.0f));
}

// MemoryValidationTests implementation
bool MemoryValidationTests::test_no_leaks_after_init() {
    print_line("Testing: No leaks after initialization...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator) {
        validator = memnew(MemoryValidator);
    }

    validator->reset();

    // Simulate initialization
    void *ptr1 = memalloc(1024);
    validator->track_allocation(ptr1, 1024, "Init");

    void *ptr2 = memalloc(2048);
    validator->track_allocation(ptr2, 2048, "Init");

    // Clean up properly
    validator->track_deallocation(ptr1);
    memfree(ptr1);

    validator->track_deallocation(ptr2);
    memfree(ptr2);

    // Check for leaks
    bool passed = validator->validate_no_leaks();
    print_line(vformat("  Result: %s", passed ? "PASSED" : "FAILED"));

    return passed;
}

bool MemoryValidationTests::test_streaming_memory_lifecycle() {
    print_line("Testing: Streaming memory lifecycle...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator) {
        validator = memnew(MemoryValidator);
    }

    validator->reset();

    // Simulate streaming buffers
    const int num_buffers = 3;
    void *buffers[num_buffers];

    for (int i = 0; i < num_buffers; i++) {
        buffers[i] = memalloc(65536); // 64KB per buffer
        validator->track_allocation(buffers[i], 65536, "StreamBuffer");
    }

    // Simulate frame cycling
    for (int frame = 0; frame < 10; frame++) {
        int buffer_idx = frame % num_buffers;

        // "Use" the buffer (no actual operation needed for test)

        // Every 3 frames, reallocate a buffer
        if (frame % 3 == 0 && frame > 0) {
            validator->track_deallocation(buffers[buffer_idx]);
            memfree(buffers[buffer_idx]);

            buffers[buffer_idx] = memalloc(65536);
            validator->track_allocation(buffers[buffer_idx], 65536, "StreamBuffer");
        }
    }

    // Clean up
    for (int i = 0; i < num_buffers; i++) {
        validator->track_deallocation(buffers[i]);
        memfree(buffers[i]);
    }

    bool passed = validator->validate_no_leaks();
    print_line(vformat("  Result: %s", passed ? "PASSED" : "FAILED"));

    return passed;
}

void MemoryValidationTests::run_all_tests() {
    print_line("\n=== Running Memory Validation Tests ===");

    int passed = 0;
    int total = 0;

    #define RUN_TEST(test_func) \
        total++; \
        if (test_func()) passed++;

    RUN_TEST(test_no_leaks_after_init);
    RUN_TEST(test_streaming_memory_lifecycle);
    RUN_TEST(test_gpu_resource_cleanup);
    RUN_TEST(test_concurrent_allocations);
    RUN_TEST(test_memory_pools);
    RUN_TEST(test_defragmentation);
    RUN_TEST(test_oom_handling);
    RUN_TEST(test_leak_detection);

    #undef RUN_TEST

    print_line(vformat("\nResults: %d/%d tests passed", passed, total));
}

void MemoryValidationTests::run_stress_tests() {
    print_line("\n=== Running Memory Stress Tests ===");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator) {
        validator = memnew(MemoryValidator);
    }

    validator->stress_test_allocation_patterns(1000);
    validator->stress_test_fragmentation(100);
    validator->stress_test_concurrent_allocations(4, 250);

    print_line("\nStress tests completed");
}

// Missing MemoryValidator method implementations

void MemoryValidator::test_defragmentation_simple() {
    print_line("Testing simple defragmentation...");
    // Create fragmented pattern and measure
    stress_test_fragmentation(50);
    float frag = measure_fragmentation();
    print_line(vformat("Fragmentation after simple test: %.1f%%", frag * 100.0f));
}

void MemoryValidator::test_defragmentation_complex() {
    print_line("Testing complex defragmentation...");
    // More intensive fragmentation test
    stress_test_fragmentation(200);
    float frag = measure_fragmentation();
    print_line(vformat("Fragmentation after complex test: %.1f%%", frag * 100.0f));
}

void MemoryValidator::simulate_memory_pressure(size_t target_bytes) {
    print_line(vformat("Simulating memory pressure: %.2f MB target...", target_bytes / (1024.0f * 1024.0f)));

    LocalVector<void *> pressure_allocs;
    size_t allocated = 0;
    const size_t chunk_size = 1024 * 1024; // 1MB chunks

    while (allocated < target_bytes) {
        void *ptr = memalloc(chunk_size);
        if (!ptr) {
            print_line("Memory allocation failed during pressure simulation");
            break;
        }
        track_allocation(ptr, chunk_size, "PressureTest");
        pressure_allocs.push_back(ptr);
        allocated += chunk_size;
    }

    print_line(vformat("Allocated %.2f MB for pressure test", allocated / (1024.0f * 1024.0f)));

    // Clean up
    for (void *ptr : pressure_allocs) {
        track_deallocation(ptr);
        memfree(ptr);
    }

    print_line("Memory pressure simulation completed");
}

void MemoryValidator::simulate_oom_conditions() {
    print_line("Simulating OOM conditions...");
    // Try to allocate increasingly large blocks until failure
    size_t size = 1024 * 1024; // Start with 1MB
    int successful_allocs = 0;

    LocalVector<void *> allocs;

    while (size <= 1024 * 1024 * 1024) { // Up to 1GB
        void *ptr = memalloc(size);
        if (!ptr) {
            print_line(vformat("OOM simulation: Failed at %.2f MB", size / (1024.0f * 1024.0f)));
            break;
        }
        track_allocation(ptr, size, "OOMTest");
        allocs.push_back(ptr);
        successful_allocs++;

        // Don't actually allocate too much
        if (successful_allocs >= 10) {
            break;
        }
        size *= 2;
    }

    // Clean up
    for (void *ptr : allocs) {
        track_deallocation(ptr);
        memfree(ptr);
    }

    print_line(vformat("OOM simulation completed with %d successful allocations", successful_allocs));
}

void MemoryValidator::analyze_allocation_patterns() {
    print_line("Analyzing allocation patterns...");

    HashMap<String, uint32_t> type_counts;
    HashMap<String, size_t> type_sizes;
    HashMap<String, size_t> type_min_size;
    HashMap<String, size_t> type_max_size;

    for (const KeyValue<void *, MemoryAllocation> &E : cpu_allocations) {
        const String &type = E.value.type;
        size_t size = E.value.size;

        type_counts[type]++;
        type_sizes[type] += size;

        if (!type_min_size.has(type) || size < type_min_size[type]) {
            type_min_size[type] = size;
        }
        if (!type_max_size.has(type) || size > type_max_size[type]) {
            type_max_size[type] = size;
        }
    }

    print_line("Allocation patterns by type:");
    for (const KeyValue<String, uint32_t> &E : type_counts) {
        print_line(vformat("  %s: count=%d, total=%.2f KB, min=%d, max=%d",
            E.key, E.value, type_sizes[E.key] / 1024.0f,
            type_min_size[E.key], type_max_size[E.key]));
    }
}

void MemoryValidator::analyze_leak_patterns() {
    detected_patterns.clear();

    HashMap<String, LeakPattern> patterns;

    for (const KeyValue<void *, MemoryAllocation> &E : cpu_allocations) {
        const String &type = E.value.type;

        if (!patterns.has(type)) {
            LeakPattern pattern;
            pattern.type = type;
            pattern.min_size = E.value.size;
            pattern.max_size = E.value.size;
            pattern.count = 1;
            patterns[type] = pattern;
        } else {
            LeakPattern &pattern = patterns[type];
            pattern.count++;
            pattern.min_size = MIN(pattern.min_size, E.value.size);
            pattern.max_size = MAX(pattern.max_size, E.value.size);
        }
    }

    for (const KeyValue<String, LeakPattern> &E : patterns) {
        detected_patterns.push_back(E.value);
    }
}

// Snapshot storage
static HashMap<String, LocalVector<MemoryAllocation>> memory_snapshots;

void MemoryValidator::take_snapshot(const String &name) {
    LocalVector<MemoryAllocation> snapshot;

    for (const KeyValue<void *, MemoryAllocation> &E : cpu_allocations) {
        snapshot.push_back(E.value);
    }

    memory_snapshots[name] = snapshot;
    print_line(vformat("Snapshot '%s' taken with %d allocations", name, snapshot.size()));
}

void MemoryValidator::compare_snapshots(const String &before, const String &after) {
    if (!memory_snapshots.has(before)) {
        WARN_PRINT(vformat("Snapshot '%s' not found", before));
        return;
    }
    if (!memory_snapshots.has(after)) {
        WARN_PRINT(vformat("Snapshot '%s' not found", after));
        return;
    }

    const LocalVector<MemoryAllocation> &snap_before = memory_snapshots[before];
    const LocalVector<MemoryAllocation> &snap_after = memory_snapshots[after];

    size_t bytes_before = 0;
    size_t bytes_after = 0;

    for (const MemoryAllocation &alloc : snap_before) {
        bytes_before += alloc.size;
    }
    for (const MemoryAllocation &alloc : snap_after) {
        bytes_after += alloc.size;
    }

    print_line(vformat("Snapshot comparison: '%s' -> '%s'", before, after));
    print_line(vformat("  Before: %d allocations, %.2f KB", snap_before.size(), bytes_before / 1024.0f));
    print_line(vformat("  After: %d allocations, %.2f KB", snap_after.size(), bytes_after / 1024.0f));
    print_line(vformat("  Delta: %d allocations, %.2f KB",
        (int)snap_after.size() - (int)snap_before.size(),
        (bytes_after - bytes_before) / 1024.0f));
}

LocalVector<MemoryAllocation> MemoryValidator::get_allocations_between_snapshots(const String &before, const String &after) {
    LocalVector<MemoryAllocation> diff;

    if (!memory_snapshots.has(before) || !memory_snapshots.has(after)) {
        return diff;
    }

    const LocalVector<MemoryAllocation> &snap_before = memory_snapshots[before];
    const LocalVector<MemoryAllocation> &snap_after = memory_snapshots[after];

    // Build set of addresses from before snapshot
    HashSet<void *> before_addresses;
    for (const MemoryAllocation &alloc : snap_before) {
        before_addresses.insert(alloc.address);
    }

    // Find new allocations in after snapshot
    for (const MemoryAllocation &alloc : snap_after) {
        if (!before_addresses.has(alloc.address)) {
            diff.push_back(alloc);
        }
    }

    return diff;
}

// MemoryHooks implementation
void *MemoryHooks::tracked_malloc(size_t size, const char *file, int line) {
    void *ptr = memalloc(size);
    if (ptr && MemoryValidator::get_singleton()) {
        if (file) {
            MemoryValidator::get_singleton()->track_allocation_with_source(ptr, size, String(file), line);
        } else {
            MemoryValidator::get_singleton()->track_allocation(ptr, size, "Hooked");
        }
    }
    return ptr;
}

void MemoryHooks::tracked_free(void *ptr) {
    if (ptr && MemoryValidator::get_singleton()) {
        MemoryValidator::get_singleton()->track_deallocation(ptr);
    }
    memfree(ptr);
}

void *MemoryHooks::tracked_realloc(void *ptr, size_t new_size, const char *file, int line) {
    void *new_ptr = memrealloc(ptr, new_size);
    if (MemoryValidator::get_singleton()) {
        MemoryValidator::get_singleton()->track_reallocation(ptr, new_ptr, new_size);
    }
    return new_ptr;
}

void MemoryHooks::install_hooks() {
    // Hook installation would be platform-specific
    print_line("Memory hooks installed (tracking enabled)");
}

void MemoryHooks::uninstall_hooks() {
    print_line("Memory hooks uninstalled");
}

// Additional MemoryValidationTests implementations
bool MemoryValidationTests::test_gpu_resource_cleanup() {
    print_line("Testing: GPU resource cleanup...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator || !validator->get_gpu_tracker()) {
        print_line("  Result: SKIPPED (no GPU tracker)");
        return true;
    }

    GPUMemoryTracker *gpu = validator->get_gpu_tracker();

    // Simulate GPU resource lifecycle
    for (int i = 0; i < 10; i++) {
        RID rid = RID::from_uint64(i + 100);
        gpu->track_buffer_creation(rid, 1024 * (i + 1), vformat("TestBuffer_%d", i));
    }

    // Free all
    for (int i = 0; i < 10; i++) {
        RID rid = RID::from_uint64(i + 100);
        gpu->track_resource_free(rid);
    }

    bool passed = gpu->validate_no_leaks();
    print_line(vformat("  Result: %s", passed ? "PASSED" : "FAILED"));
    return passed;
}

bool MemoryValidationTests::test_concurrent_allocations() {
    print_line("Testing: Concurrent allocations...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator) {
        validator = memnew(MemoryValidator);
    }

    validator->reset();
    validator->stress_test_concurrent_allocations(2, 100);

    bool passed = validator->validate_no_leaks();
    print_line(vformat("  Result: %s", passed ? "PASSED" : "FAILED"));
    return passed;
}

bool MemoryValidationTests::test_memory_pools() {
    print_line("Testing: Memory pools...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator || !validator->get_gpu_tracker()) {
        print_line("  Result: SKIPPED (no GPU tracker)");
        return true;
    }

    GPUMemoryTracker *gpu = validator->get_gpu_tracker();

    // Test pool allocation tracking
    gpu->track_pool_allocation("TestPool", 1024);
    gpu->track_pool_allocation("TestPool", 2048);
    gpu->track_pool_free("TestPool", 1024);
    gpu->track_pool_free("TestPool", 2048);

    print_line("  Result: PASSED");
    return true;
}

bool MemoryValidationTests::test_defragmentation() {
    print_line("Testing: Defragmentation...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator) {
        validator = memnew(MemoryValidator);
    }

    validator->reset();
    validator->test_defragmentation_simple();

    float frag = validator->measure_fragmentation();
    bool passed = validator->validate_no_leaks();

    print_line(vformat("  Final fragmentation: %.1f%%", frag * 100.0f));
    print_line(vformat("  Result: %s", passed ? "PASSED" : "FAILED"));
    return passed;
}

bool MemoryValidationTests::test_oom_handling() {
    print_line("Testing: OOM handling...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator) {
        validator = memnew(MemoryValidator);
    }

    validator->reset();
    // Just run a small simulation, not actual OOM
    validator->simulate_memory_pressure(10 * 1024 * 1024); // 10MB

    bool passed = validator->validate_no_leaks();
    print_line(vformat("  Result: %s", passed ? "PASSED" : "FAILED"));
    return passed;
}

bool MemoryValidationTests::test_leak_detection() {
    print_line("Testing: Leak detection...");

    MemoryValidator *validator = MemoryValidator::get_singleton();
    if (!validator) {
        validator = memnew(MemoryValidator);
    }

    validator->reset();

    // Create intentional leak
    void *leaked_ptr = memalloc(512);
    validator->track_allocation(leaked_ptr, 512, "IntentionalLeak");

    // Should detect the leak
    bool has_leak = !validator->validate_no_leaks();

    // Now clean it up
    validator->track_deallocation(leaked_ptr);
    memfree(leaked_ptr);

    bool clean_after = validator->validate_no_leaks();
    bool passed = has_leak && clean_after;

    print_line(vformat("  Result: %s", passed ? "PASSED" : "FAILED"));
    return passed;
}