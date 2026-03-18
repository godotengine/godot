/**
 * @file gaussian_splat_manager.h
 * @brief Global singleton manager for Gaussian Splatting resources and configuration.
 *
 * The GaussianSplatManager coordinates engine-wide Gaussian Splatting state including
 * GPU buffer registration, performance monitoring, sorting configuration, and
 * RenderingDevice lifecycle management for multi-threaded rendering.
 */

#ifndef GAUSSIAN_SPLAT_MANAGER_H
#define GAUSSIAN_SPLAT_MANAGER_H

#include <atomic>

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/math/transform_3d.h"
#include "core/os/mutex.h"
#include "core/os/semaphore.h"
#include "core/os/safe_binary_mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/safe_refcount.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "servers/rendering_server.h"
#include "gaussian_data.h"
#include "gaussian_splat_asset.h"

class RenderingDevice;
class GaussianSplatNode3D;

/**
 * @class GaussianSplatManager
 * @brief Singleton manager for Gaussian Splatting engine integration.
 *
 * GaussianSplatManager serves as the central coordination point for all Gaussian
 * Splatting operations in the engine. It handles:
 *
 * - **GPU Buffer Management**: Registration and lifecycle of GPU buffers shared
 *   across multiple GaussianSplatNode3D instances.
 * - **Device Management**: Creation and binding of RenderingDevice instances for
 *   multi-threaded GPU submission.
 * - **Performance Monitoring**: Tracking of global statistics (total splats, memory).
 * - **Sorting Configuration**: Project-setting-driven thresholds for GPU sorting
 *   algorithm selection.
 *
 * Access the singleton via GaussianSplatManager::get_singleton().
 *
 * @note This class manages thread-local RenderingDevice instances. The shared
 *       submission device is protected by submission_mutex for thread-safe access.
 */
// Lock ordering (acquire in this order to prevent deadlocks):
//
//   Level 1. submission_mutex           (outermost — serializes GPU submission)
//   Level 2. resource_maps_mutex        (protects gaussian_buffers / buffer_lookup / dynamic_asset_cache)
//   Level 3. active_nodes_mutex         (protects the active_nodes set)
//   Level 4. local_device_destroy_dispatch_mutex  (innermost — device teardown only)
//
// Rules:
//   - Never acquire a lower-numbered (outer) lock while holding a higher-numbered (inner) one.
//   - In practice no method currently nests two of these locks in the same scope;
//     the ordering is documented here so that future changes preserve this invariant.
//   - The destructor acquires them sequentially (not nested):
//     active_nodes_mutex -> release -> resource_maps_mutex -> release ->
//     local_device_destroy_dispatch_mutex -> release.
//
// Validated at runtime in DEV_ENABLED builds via thread_local lock-level tracking
// (see _gs_lock_level_guard in the .cpp file).

class GaussianSplatManager : public Object {
    GDCLASS(GaussianSplatManager, Object);

public:
    /** @brief Mutex tag for SafeBinaryMutex identification in debugging. */
    static constexpr int SUBMISSION_MUTEX_TAG = 0x47534c54; // 'GSLT'

    // Lock hierarchy levels used by the DEV_ENABLED ordering assertions.
    // Public so that external debug tooling can inspect them if needed.
    static constexpr uint32_t LOCK_LEVEL_SUBMISSION = 1;
    static constexpr uint32_t LOCK_LEVEL_RESOURCE_MAPS = 2;
    static constexpr uint32_t LOCK_LEVEL_ACTIVE_NODES = 3;
    static constexpr uint32_t LOCK_LEVEL_DEVICE_DESTROY = 4;

private:
    static GaussianSplatManager *singleton;

    struct BufferEntry {
        RID gpu_buffer;
        uint32_t ref_count = 0;
        uint64_t gaussian_count = 0;
        uint64_t memory_usage = 0;
    };

    struct DynamicAssetEntry {
        RID gaussian_buffer;
        uint32_t gaussian_count = 0;
        uint32_t ref_count = 0;
        uint64_t memory_usage = 0;
    };

    HashMap<ObjectID, BufferEntry> gaussian_buffers;
    HashMap<RID, ObjectID> buffer_lookup;
    HashMap<RID, DynamicAssetEntry> dynamic_asset_cache;
    mutable Mutex resource_maps_mutex;
    HashSet<ObjectID> active_nodes;
    Mutex active_nodes_mutex;
    SafeFlag main_thread_dispatch_pending;

    // Lazily-created RenderingDevice instances bound on the render thread.
    // The shared submission device owns the compute queue when async mode is enabled.
    // Primary device copied from the render thread for callers that need a short-lived per-thread RD.
    std::atomic<RenderingDevice *> primary_local_device{ nullptr };
    // Shared compute submission device used when multiple threads queue work asynchronously.
    std::atomic<RenderingDevice *> shared_submission_device{ nullptr };
    SafeFlag primary_device_request_pending;
    SafeFlag shared_device_request_pending;
    bool primary_device_render_thread_bound = false;
    bool shared_device_render_thread_bound = false;
    mutable Mutex local_device_destroy_dispatch_mutex;
    mutable Semaphore local_device_destroy_dispatch_semaphore;
    std::atomic<uint64_t> local_device_destroy_next_request_id{ 1 };
    std::atomic<uint64_t> local_device_destroy_completed_request_id{ 0 };
    RenderingDevice *local_device_destroy_pending_primary = nullptr;
    RenderingDevice *local_device_destroy_pending_shared = nullptr;
    // Serializes GPU submissions when multiple threads share the submission queue.
    // Guards access to the shared submission device so only one thread records at a time.
    static SafeBinaryMutex<SUBMISSION_MUTEX_TAG> submission_mutex;
    friend SafeBinaryMutex<SUBMISSION_MUTEX_TAG> &_get_submission_mutex();

    // Create or fetch the shared submission device; caller must hold submission_mutex.
    RenderingDevice *_ensure_shared_submission_device_locked();
    void _release_registered_resources();
    void _destroy_local_devices();
    void _ensure_render_thread_binding(RenderingDevice *p_device, bool &r_bound_flag);
    void _connect_frame_callbacks();
    void _disconnect_frame_callbacks();
    void _request_primary_local_device();
    void _request_shared_local_device();
    void _create_local_device_on_render_thread(bool p_primary_device);
    bool _dispatch_local_device_destroy_on_render_thread(RenderingDevice *p_primary, RenderingDevice *p_shared);
    static void _destroy_local_devices_immediate(RenderingDevice *p_primary, RenderingDevice *p_shared);
    void _destroy_local_devices_on_render_thread(uint64_t p_request_id);
    void _on_frame_pre_draw();
    void _process_active_nodes_main_thread();
    bool frame_callbacks_connected = false;

    // Performance tracking
    uint64_t total_gaussians = 0;
    uint64_t total_memory_usage = 0;
    uint64_t last_reported_gaussians = 0;
    uint64_t last_reported_memory = 0;
    float last_frame_time_ms = 0.0f;

    // Module configuration
    bool gpu_sorting_enabled = true;
    bool shared_submission_device_enabled = false;
    int max_gpu_buffer_count = 128;

    // RenderDoc compatibility mode - when enabled, skips creating local devices
    // which cause crashes when RenderDoc hooks Vulkan (multiple devices not supported)
    bool renderdoc_compatibility_mode = false;
    static bool _detect_renderdoc();
    // Sorting heuristics exposed through project settings
    uint32_t sorting_bitonic_max = 131072;
    uint32_t sorting_radix_max = 1500000;
    uint32_t sorting_onesweep_max = 3000000;
    uint32_t sorting_hybrid_trigger = 3000000;
    uint32_t sorting_hybrid_batch = 1500000;
    uint32_t sorting_history_size = 120;
    uint32_t sorting_log_interval = 60;
    float sorting_target_ms = 2.0f;
    bool sorting_log_metrics = true;
    int sorting_force_algorithm = 0; // 0=auto, 1=radix, 2=bitonic, 3=onesweep
    bool sorting_force_cpu_sort = false;

protected:
    static void _bind_methods();
    void _recalculate_totals();
    void _recalculate_totals_unlocked();

public:
    /**
     * @struct SharedDynamicAssetHandle
     * @brief Handle for shared GPU resources associated with a dynamic asset.
     *
     * Returned by acquire_dynamic_asset() to provide access to GPU buffers
     * that may be shared across multiple renderer instances.
     */
    struct SharedDynamicAssetHandle {
        RID asset_rid;              ///< Asset resource ID for cache lookup.
        RID gaussian_buffer;        ///< GPU buffer containing Gaussian data.
        uint32_t gaussian_count = 0;///< Number of Gaussians in the buffer.

        /** @brief Returns true if this handle references valid GPU resources. */
        bool is_valid() const { return asset_rid.is_valid(); }
    };

    /**
     * @class ScopedSubmissionLock
     * @brief RAII guard for thread-safe GPU submission.
     *
     * Acquires the submission mutex and provides access to a submission-capable
     * RenderingDevice. The lock is released when the guard goes out of scope.
     *
     * @note When shared_submission_device is enabled, returns the shared device.
     *       When disabled, returns the primary device without mutex acquisition.
     *
     * Usage:
     * @code
     * auto lock = GaussianSplatManager::get_singleton()->acquire_submission_lock();
     * if (lock.is_valid()) {
     *     RenderingDevice *rd = lock.get_rendering_device();
     *     // Perform GPU operations...
     * }
     * @endcode
     */
    class ScopedSubmissionLock {
        // GaussianSplatManager is the sole factory for ScopedSubmissionLock instances
        // via acquire_submission_lock(). The private constructor enforces this contract.
        friend class GaussianSplatManager;

        const SafeBinaryMutex<SUBMISSION_MUTEX_TAG> *mutex = nullptr;
        RenderingDevice *rendering_device = nullptr;
        bool locked = false;

#ifdef DEV_ENABLED
        // Lock-ordering bookkeeping: the lock level held before we acquired
        // submission_mutex so that _release() can restore it.
        uint32_t previous_lock_level = 0;
#endif

        explicit ScopedSubmissionLock(GaussianSplatManager &p_manager);

        void _release();

    public:
        ScopedSubmissionLock() = default;
        ScopedSubmissionLock(const ScopedSubmissionLock &) = delete;
        ScopedSubmissionLock &operator=(const ScopedSubmissionLock &) = delete;

        ScopedSubmissionLock(ScopedSubmissionLock &&p_other) noexcept;
        ScopedSubmissionLock &operator=(ScopedSubmissionLock &&p_other) noexcept;

        ~ScopedSubmissionLock();

        RenderingDevice *get_rendering_device() const { return rendering_device; }
        RenderingDevice *operator->() const { return rendering_device; }
        bool is_valid() const { return rendering_device != nullptr; }
    };

    GaussianSplatManager();
    ~GaussianSplatManager();

    /** @brief Returns the global singleton instance. */
    static GaussianSplatManager *get_singleton() { return singleton; }

    /// @name Device Management
    /// @{

    /**
     * @brief Returns the main RenderingDevice for GPU operations.
     * @return RenderingDevice pointer suitable for operations visible in the final render.
     *
     * @note Returns the engine's main RenderingDevice (not a local device) to ensure
     *       GPU resources are compatible with the main rendering pipeline.
     */
    RenderingDevice *get_primary_rendering_device();

    /**
     * @brief Returns the shared submission device for multi-threaded work.
     * @return RenderingDevice pointer, or nullptr if shared device is disabled.
     * @note Prefer using acquire_submission_lock() for thread-safe access.
     */
    RenderingDevice *get_shared_submission_device();

    /**
     * @brief Acquires the submission mutex and returns an RAII lock guard.
     * @return ScopedSubmissionLock that provides access to the submission device.
     */
    ScopedSubmissionLock acquire_submission_lock();

    /**
     * @brief Returns a submission-capable device while holding the lock guard.
     * @param p_device Preferred device, or nullptr to use the shared device.
     * @param r_lock Lock guard that must remain valid during device usage.
     * @return RenderingDevice pointer for GPU submission.
     */
    RenderingDevice *acquire_submission_device(RenderingDevice *p_device, ScopedSubmissionLock &r_lock);

    /**
     * @brief Checks if the given device is the shared submission device.
     * @param p_device Device pointer to check.
     * @return True if p_device matches the shared submission device.
     */
    bool is_shared_submission_device(RenderingDevice *p_device) const;

    /**
     * @brief Ensures thread-local RenderingDevice instances are bound on the render thread.
     * @note Called internally by the rendering pipeline during frame setup.
     */
    void ensure_render_thread_local_devices_current();

    /// @}

    /// @name Resource Management
    /// @{

    /**
     * @brief Registers a GaussianData resource and creates its GPU buffer.
     * @param p_gaussian_data Source data to upload.
     * @param p_rd RenderingDevice to use, or nullptr for default.
     * @return RID of the registered GPU buffer.
     */
    RID register_gaussian_buffer(const Ref<::GaussianData> &p_gaussian_data, RenderingDevice *p_rd = nullptr);

    /**
     * @brief Unregisters a GPU buffer and decrements its reference count.
     * @param p_buffer Buffer RID to unregister.
     */
    void unregister_gaussian_buffer(RID p_buffer);

    /**
     * @brief Retrieves the GPU buffer RID for a registered GaussianData resource.
     * @param p_gaussian_data Object ID of the GaussianData resource.
     * @return GPU buffer RID, or an invalid RID if not registered.
     */
    RID get_gpu_buffer(RID p_gaussian_data) const;

    /**
     * @brief Acquires a shared dynamic asset handle with GPU buffers.
     * @param p_asset Asset to acquire.
     * @param p_gaussian_data Associated Gaussian data.
     * @param p_rd RenderingDevice to use.
     * @return Handle containing GPU buffer RIDs.
     */
    SharedDynamicAssetHandle acquire_dynamic_asset(const Ref<GaussianSplatAsset> &p_asset,
            const Ref<::GaussianData> &p_gaussian_data, RenderingDevice *p_rd = nullptr);

    /**
     * @brief Releases a previously acquired dynamic asset handle.
     * @param p_handle Handle returned by acquire_dynamic_asset().
     */
    void release_dynamic_asset(const SharedDynamicAssetHandle &p_handle);

    /// @}

    /// @name Performance Monitoring
    /// @{

    /**
     * @brief Updates global performance statistics.
     * @param p_gaussian_count Total active Gaussian count.
     * @param p_memory_usage Total GPU memory usage in bytes.
     */
    void update_stats(uint64_t p_gaussian_count, uint64_t p_memory_usage);

    /**
     * @brief Returns global statistics as a Dictionary.
     * @return Dictionary with keys: "total_gaussians", "total_memory_usage", etc.
     */
    Dictionary get_global_stats() const;

    /// @}

    /// @name Configuration
    /// @{

    /**
     * @brief Enables or disables GPU-based sorting.
     * @param p_enabled When false, falls back to CPU sorting.
     */
    void set_gpu_sorting_enabled(bool p_enabled);

    /** @brief Returns true if GPU sorting is enabled. */
    bool is_gpu_sorting_enabled() const { return gpu_sorting_enabled; }

    /**
     * @brief Enables or disables the shared submission device.
     * @param p_enabled When true, uses a shared device for multi-threaded submission.
     */
    void set_shared_submission_device_enabled(bool p_enabled);

    /** @brief Returns true if the shared submission device is enabled. */
    bool is_shared_submission_device_enabled() const { return shared_submission_device_enabled; }

    /**
     * @brief Returns sorting algorithm configuration thresholds.
     * @return Dictionary with sorting threshold values.
     */
    Dictionary get_sorting_config() const;

    /// @}

    /// @name Sorting Thresholds
    /// @brief Splat count thresholds for sorting algorithm selection.
    /// @{

    uint32_t get_sorting_bitonic_max() const { return sorting_bitonic_max; }
    uint32_t get_sorting_radix_max() const { return sorting_radix_max; }
    uint32_t get_sorting_onesweep_max() const { return sorting_onesweep_max; }
    uint32_t get_sorting_hybrid_trigger() const { return sorting_hybrid_trigger; }
    uint32_t get_sorting_hybrid_batch() const { return sorting_hybrid_batch; }
    uint32_t get_sorting_history_size() const { return sorting_history_size; }
    uint32_t get_sorting_log_interval() const { return sorting_log_interval; }
    float get_sorting_target_ms() const { return sorting_target_ms; }
    bool is_sorting_log_metrics_enabled() const { return sorting_log_metrics; }
    bool is_sorting_force_cpu_sort_enabled() const { return sorting_force_cpu_sort; }

    /// @}

    /// @name Engine Integration
    /// @{

    /**
     * @brief Initializes the Gaussian Splatting module.
     *
     * Called during engine startup. Sets up project settings, connects to
     * RenderingServer signals, and prepares the GPU buffer registry.
     */
    void initialize_module();

    /**
     * @brief Finalizes the module and releases all GPU resources.
     *
     * Called during engine shutdown. Releases all registered GPU buffers
     * and destroys local RenderingDevice instances.
     */
    void finalize_module();

    /**
     * @brief Registers a GaussianSplatNode3D for per-frame processing.
     * @param p_node Node to register.
     * @note Called automatically when nodes enter the scene tree.
     */
    void register_node(GaussianSplatNode3D *p_node);

    /**
     * @brief Unregisters a GaussianSplatNode3D.
     * @param p_node Node to unregister.
     * @note Called automatically when nodes exit the scene tree.
     */
    void unregister_node(GaussianSplatNode3D *p_node);

    /// @}
};

#endif // GAUSSIAN_SPLAT_MANAGER_H
