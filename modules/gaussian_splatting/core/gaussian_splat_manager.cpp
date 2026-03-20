#include "gaussian_splat_manager.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_method_pointer.h"
#include "core/object/object.h"
#include "core/os/os.h"
#include "core/variant/variant.h"
#include "gaussian_data.h"
#include "gs_project_settings.h"
#include "../nodes/gaussian_splat_node_3d.h"
#include "../logger/gs_logger.h"
#include "../interfaces/sync_policy.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"
#include "core/templates/local_vector.h"
#include "core/os/safe_binary_mutex.h"
#include <utility>

#ifdef WINDOWS_ENABLED
#include <windows.h>
#endif

namespace {

// Project settings helpers: delegates to gs_project_settings.h (gs::settings namespace).
static uint32_t _get_uint_setting(ProjectSettings *ps, const StringName &name, uint32_t fallback) {
    return gs::settings::get_uint(ps, name, fallback);
}
static float _get_float_setting(ProjectSettings *ps, const StringName &name, float fallback) {
    return gs::settings::get_float(ps, name, fallback);
}
static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
    return gs::settings::get_bool(ps, name, fallback);
}
static int _get_int_setting(ProjectSettings *ps, const StringName &name, int fallback) {
    return static_cast<int>(gs::settings::get_uint(ps, name, static_cast<uint32_t>(fallback)));
}
static bool _is_data_log_enabled() {
    return gs::settings::is_data_log_enabled();
}

} // namespace

// ---------------------------------------------------------------------------
// Lock-ordering debug assertions (DEV_ENABLED builds only).
//
// Each thread tracks the highest lock level it currently holds. When acquiring
// a lock, we assert that its level is strictly greater than the current held
// level — i.e. we only allow acquiring locks in increasing level order.
//
// The guard is RAII: the constructor validates and raises the level, and the
// destructor restores the previous level.
// ---------------------------------------------------------------------------
#ifdef DEV_ENABLED

// Highest lock-hierarchy level held by the current thread (0 = none).
thread_local static uint32_t _gs_held_lock_level = 0;

struct _GSLockLevelGuard {
    uint32_t previous_level;
    uint32_t this_level;

    explicit _GSLockLevelGuard(uint32_t p_level, const char *p_lock_name) :
            previous_level(_gs_held_lock_level), this_level(p_level) {
        if (p_level <= _gs_held_lock_level) {
            ERR_PRINT(vformat("[GaussianSplatManager] Lock ordering violation: acquiring %s (level %d) "
                    "while already holding a lock at level %d. "
                    "See lock ordering comment in gaussian_splat_manager.h.",
                    p_lock_name, p_level, _gs_held_lock_level));
            DEV_ASSERT(p_level > _gs_held_lock_level);
        }
        _gs_held_lock_level = p_level;
    }

    ~_GSLockLevelGuard() {
        _gs_held_lock_level = previous_level;
    }
};

#define _GS_LOCK_ORDER_GUARD_CONCAT_INNER(a, b) a##b
#define _GS_LOCK_ORDER_GUARD_CONCAT(a, b) _GS_LOCK_ORDER_GUARD_CONCAT_INNER(a, b)
#define GS_LOCK_ORDER_GUARD(level, name) \
    _GSLockLevelGuard _GS_LOCK_ORDER_GUARD_CONCAT(_gs_lock_guard_, __LINE__)(level, name)

#else

#define GS_LOCK_ORDER_GUARD(level, name) ((void)0)

#endif // DEV_ENABLED

SafeBinaryMutex<GaussianSplatManager::SUBMISSION_MUTEX_TAG> &_get_submission_mutex() {
    return GaussianSplatManager::submission_mutex;
}

template <>
thread_local SafeBinaryMutex<GaussianSplatManager::SUBMISSION_MUTEX_TAG>::TLSData
        SafeBinaryMutex<GaussianSplatManager::SUBMISSION_MUTEX_TAG>::tls_data(_get_submission_mutex());

SafeBinaryMutex<GaussianSplatManager::SUBMISSION_MUTEX_TAG> GaussianSplatManager::submission_mutex;

GaussianSplatManager *GaussianSplatManager::singleton = nullptr;

bool GaussianSplatManager::_detect_renderdoc() {
    // RenderDoc detection via environment variables
    // RenderDoc sets RENDERDOC_* vars when hooking an application
    OS *os = OS::get_singleton();
    if (!os) {
        return false;
    }

    // Check for RenderDoc environment variables
    if (os->has_environment("RENDERDOC_CAPFILE") ||
        os->has_environment("RENDERDOC_CAPOPTS") ||
        os->has_environment("RENDERDOC_DEBUG_LOG_FILE") ||
        os->has_environment("ENABLE_VULKAN_RENDERDOC_CAPTURE")) {
        return true;
    }

#ifdef WINDOWS_ENABLED
    // On Windows, also check if RenderDoc's DLL is loaded
    if (GetModuleHandleW(L"renderdoc.dll") != nullptr) {
        return true;
    }
#endif

    return false;
}

GaussianSplatManager::ScopedSubmissionLock::ScopedSubmissionLock(GaussianSplatManager &p_manager) {
    if (p_manager.shared_submission_device_enabled) {
        mutex = &GaussianSplatManager::submission_mutex;
        if (mutex) {
#ifdef DEV_ENABLED
            previous_lock_level = _gs_held_lock_level;
            if (LOCK_LEVEL_SUBMISSION <= _gs_held_lock_level) {
                ERR_PRINT(vformat("[GaussianSplatManager] Lock ordering violation: acquiring submission_mutex (level %d) "
                        "while already holding a lock at level %d. "
                        "See lock ordering comment in gaussian_splat_manager.h.",
                        LOCK_LEVEL_SUBMISSION, _gs_held_lock_level));
                DEV_ASSERT(LOCK_LEVEL_SUBMISSION > _gs_held_lock_level);
            }
            _gs_held_lock_level = LOCK_LEVEL_SUBMISSION;
#endif
            mutex->lock();
            locked = true;
        }
        rendering_device = p_manager._ensure_shared_submission_device_locked();
        if (!rendering_device) {
            rendering_device = p_manager.get_primary_rendering_device();
            if (!rendering_device) {
                WARN_PRINT_ONCE("[GaussianSplatManager] Unable to provide submission RenderingDevice");
            }
        }
    } else {
        rendering_device = p_manager.get_primary_rendering_device();
        if (!rendering_device) {
            WARN_PRINT_ONCE("[GaussianSplatManager] Primary RenderingDevice unavailable");
        }
    }
}

GaussianSplatManager::ScopedSubmissionLock::ScopedSubmissionLock(ScopedSubmissionLock &&p_other) noexcept {
    mutex = p_other.mutex;
    rendering_device = p_other.rendering_device;
    locked = p_other.locked;
#ifdef DEV_ENABLED
    previous_lock_level = p_other.previous_lock_level;
    p_other.previous_lock_level = 0;
#endif
    p_other.mutex = nullptr;
    p_other.rendering_device = nullptr;
    p_other.locked = false;
}

GaussianSplatManager::ScopedSubmissionLock &GaussianSplatManager::ScopedSubmissionLock::operator=(ScopedSubmissionLock &&p_other) noexcept {
    if (this == &p_other) {
        return *this;
    }

    _release();

    mutex = p_other.mutex;
    rendering_device = p_other.rendering_device;
    locked = p_other.locked;
#ifdef DEV_ENABLED
    previous_lock_level = p_other.previous_lock_level;
    p_other.previous_lock_level = 0;
#endif

    p_other.mutex = nullptr;
    p_other.rendering_device = nullptr;
    p_other.locked = false;

    return *this;
}

GaussianSplatManager::ScopedSubmissionLock::~ScopedSubmissionLock() {
    _release();
}

void GaussianSplatManager::ScopedSubmissionLock::_release() {
    if (locked && mutex) {
        mutex->unlock();
#ifdef DEV_ENABLED
        _gs_held_lock_level = previous_lock_level;
        previous_lock_level = 0;
#endif
    }
    mutex = nullptr;
    rendering_device = nullptr;
    locked = false;
}

GaussianSplatManager::GaussianSplatManager() {
    ERR_FAIL_COND_MSG(singleton != nullptr, "GaussianSplatManager singleton already exists.");
    singleton = this;

    main_thread_dispatch_pending.clear();

    {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
        MutexLock resource_lock(resource_maps_mutex);
        gaussian_buffer_owner_devices.clear();
        dynamic_asset_owner_devices.clear();
    }

    // Check for RenderDoc before creating local devices
    // RenderDoc does NOT support multiple Vulkan devices in a single instance
    // See: https://github.com/baldurk/renderdoc/issues/2961
    renderdoc_compatibility_mode = _detect_renderdoc();

    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (ps && ps->has_setting("rendering/gaussian_splatting/renderdoc_compatibility")) {
        // Allow manual override via project setting
        renderdoc_compatibility_mode = ps->get_setting("rendering/gaussian_splatting/renderdoc_compatibility");
    }

    if (renderdoc_compatibility_mode) {
        if (_is_data_log_enabled()) {
            GS_LOG_INFO_DEFAULT("[GaussianSplatManager] RenderDoc detected - using main RenderingDevice only (no local devices)");
        }
        primary_device_render_thread_bound = false;
        shared_device_render_thread_bound = false;
    } else {
        // Create a local device for primary operations
        RenderingServer *rs = RenderingServer::get_singleton();
        if (rs) {
            _request_primary_local_device();
            _request_shared_local_device();
            primary_device_render_thread_bound = false;
            shared_device_render_thread_bound = false;
        } else {
            WARN_PRINT_ONCE("[GaussianSplatManager] RenderingServer unavailable; local devices will be lazily created");
            primary_device_render_thread_bound = false;
            shared_device_render_thread_bound = false;
        }
    }

    if (!ps) {
        return;
    }

    // Initialize module configuration from project settings
    if (ps->has_setting("rendering/gaussian_splatting/gpu_sorting_enabled")) {
        gpu_sorting_enabled = ps->get_setting("rendering/gaussian_splatting/gpu_sorting_enabled");
    }
    if (ps->has_setting("rendering/gaussian_splatting/shared_submission_device_enabled")) {
        shared_submission_device_enabled = ps->get_setting("rendering/gaussian_splatting/shared_submission_device_enabled");
    }

    sorting_bitonic_max = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/bitonic_max_elements", sorting_bitonic_max);
    sorting_radix_max = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/radix_max_elements", sorting_radix_max);
    sorting_onesweep_max = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/onesweep_max_elements", sorting_onesweep_max);
    sorting_hybrid_trigger = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/hybrid_trigger_elements", sorting_hybrid_trigger);
    sorting_hybrid_batch = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/hybrid_batch_size", sorting_hybrid_batch);
    sorting_history_size = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/history_size", sorting_history_size);
    sorting_log_interval = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/log_interval_frames", sorting_log_interval);
    sorting_target_ms = MAX(0.0f, _get_float_setting(ps, "rendering/gaussian_splatting/sorting/target_sort_time_ms", sorting_target_ms));
    sorting_log_metrics = _get_bool_setting(ps, "rendering/gaussian_splatting/sorting/log_metrics", sorting_log_metrics);
    sorting_force_algorithm = CLAMP(_get_int_setting(ps, "rendering/gaussian_splatting/sorting/force_algorithm", sorting_force_algorithm), 0, 3);
    sorting_force_cpu_sort = _get_bool_setting(ps, "rendering/gaussian_splatting/sorting/force_cpu_sort", sorting_force_cpu_sort);
}

void GaussianSplatManager::_release_registered_resources() {
    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);

    if (registered_resources_released && gaussian_buffers.is_empty() && dynamic_asset_cache.is_empty()) {
        return;
    }

    HashMap<ObjectID, BufferEntry> remaining_gaussian_buffers;
    HashMap<RID, ObjectID> remaining_buffer_lookup;
    HashMap<ObjectID, RenderingDevice *> remaining_gaussian_owner_devices;
    HashMap<RID, DynamicAssetEntry> remaining_dynamic_asset_cache;
    HashMap<RID, RenderingDevice *> remaining_dynamic_owner_devices;

    for (const KeyValue<ObjectID, BufferEntry> &E : gaussian_buffers) {
        if (!E.value.gpu_buffer.is_valid()) {
            continue;
        }
        RenderingDevice *owner_device = nullptr;
        if (RenderingDevice *const *owner_ptr = gaussian_buffer_owner_devices.getptr(E.key)) {
            owner_device = *owner_ptr;
        }
        if (!owner_device) {
            ERR_PRINT(vformat("[GaussianSplatManager] Missing owner device for registered buffer %s during shutdown",
                    String::num_uint64(E.value.gpu_buffer.get_id())));
            remaining_gaussian_buffers.insert(E.key, E.value);
            remaining_buffer_lookup.insert(E.value.gpu_buffer, E.key);
            continue;
        }
        owner_device->free(E.value.gpu_buffer);
    }
    for (const KeyValue<ObjectID, BufferEntry> &E : remaining_gaussian_buffers) {
        if (RenderingDevice *const *owner_ptr = gaussian_buffer_owner_devices.getptr(E.key)) {
            remaining_gaussian_owner_devices.insert(E.key, *owner_ptr);
        }
    }
    gaussian_buffers = remaining_gaussian_buffers;
    buffer_lookup = remaining_buffer_lookup;
    gaussian_buffer_owner_devices = remaining_gaussian_owner_devices;

    for (KeyValue<RID, DynamicAssetEntry> &E : dynamic_asset_cache) {
        DynamicAssetEntry &entry = E.value;
        if (!entry.gaussian_buffer.is_valid()) {
            continue;
        }
        RenderingDevice *owner_device = nullptr;
        if (RenderingDevice *const *owner_ptr = dynamic_asset_owner_devices.getptr(E.key)) {
            owner_device = *owner_ptr;
        }
        if (!owner_device) {
            ERR_PRINT(vformat("[GaussianSplatManager] Missing owner device for dynamic asset buffer %s during shutdown",
                    String::num_uint64(entry.gaussian_buffer.get_id())));
            remaining_dynamic_asset_cache.insert(E.key, E.value);
            continue;
        }
        owner_device->free(entry.gaussian_buffer);
    }
    for (const KeyValue<RID, DynamicAssetEntry> &E : remaining_dynamic_asset_cache) {
        if (RenderingDevice *const *owner_ptr = dynamic_asset_owner_devices.getptr(E.key)) {
            remaining_dynamic_owner_devices.insert(E.key, *owner_ptr);
        }
    }
    dynamic_asset_cache = remaining_dynamic_asset_cache;
    dynamic_asset_owner_devices = remaining_dynamic_owner_devices;

    if (!gaussian_buffers.is_empty() || !dynamic_asset_cache.is_empty()) {
        GS_LOG_WARN_DEFAULT(vformat("[GaussianSplatManager] Deferred shutdown release for %d registered buffers and %d dynamic assets due to missing owner metadata",
                gaussian_buffers.size(), dynamic_asset_cache.size()));
    }

    registered_resources_released = gaussian_buffers.is_empty() && dynamic_asset_cache.is_empty();
    _recalculate_totals_unlocked();
}

bool GaussianSplatManager::_dispatch_local_device_destroy_on_render_thread(RenderingDevice *p_primary, RenderingDevice *p_shared) {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs || rs->is_on_render_thread() || !rs->is_render_loop_enabled()) {
        return false;
    }

    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_DEVICE_DESTROY, "local_device_destroy_request_mutex");
    MutexLock request_lock(local_device_destroy_request_mutex);
    {
        MutexLock pending_lock(local_device_destroy_pending_mutex);
        local_device_destroy_pending_primary = p_primary;
        local_device_destroy_pending_shared = p_shared;
    }

    bool dispatch_submitted = false;
    uint64_t request_id = 0;
    const bool completed = local_device_destroy_dispatcher.dispatch_call_on_render_thread_blocking(
            callable_mp(this, &GaussianSplatManager::_destroy_local_devices_on_render_thread),
            &dispatch_submitted,
            true,
            &request_id,
            "[GaussianSplatManager] Local-device destroy dispatch");
    if (!completed && dispatch_submitted) {
        WARN_PRINT(vformat("[GaussianSplatManager] Local-device destroy request %d timed out after render-thread submission; deferring to render-thread callback to avoid unsafe double destroy",
                uint64_t(request_id)));
        return true;
    }
    return completed;
}

void GaussianSplatManager::_destroy_local_devices_immediate(RenderingDevice *p_primary, RenderingDevice *p_shared) {
    if (p_primary) {
        p_primary->make_current();
        memdelete(p_primary);
    }
    if (p_shared && p_shared != p_primary) {
        p_shared->make_current();
        memdelete(p_shared);
    }
}

void GaussianSplatManager::_destroy_local_devices_on_render_thread(uint64_t p_request_id) {
    RenderingDevice *pending_primary = nullptr;
    RenderingDevice *pending_shared = nullptr;
    {
        MutexLock pending_lock(local_device_destroy_pending_mutex);
        pending_primary = local_device_destroy_pending_primary;
        pending_shared = local_device_destroy_pending_shared;
        local_device_destroy_pending_primary = nullptr;
        local_device_destroy_pending_shared = nullptr;
    }

    _destroy_local_devices_immediate(pending_primary, pending_shared);
    local_device_destroy_dispatcher.notify_completed(p_request_id);
}

void GaussianSplatManager::_destroy_local_devices() {
    RenderingDevice *primary = primary_local_device.exchange(nullptr, std::memory_order_acq_rel);
    RenderingDevice *shared = shared_submission_device.exchange(nullptr, std::memory_order_acq_rel);
    primary_device_render_thread_bound = false;
    shared_device_render_thread_bound = false;

    if (!primary && !shared) {
        return;
    }
    if (_dispatch_local_device_destroy_on_render_thread(primary, shared)) {
        return;
    }

    _destroy_local_devices_immediate(primary, shared);
}

void GaussianSplatManager::_ensure_render_thread_binding(RenderingDevice *p_device, bool &r_bound_flag) {
    if (!p_device || r_bound_flag) {
        return;
    }

    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs || !rs->is_on_render_thread()) {
        return;
    }

    p_device->make_current();
    r_bound_flag = true;
}

void GaussianSplatManager::_connect_frame_callbacks() {
    if (frame_callbacks_connected) {
        return;
    }

    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        return;
    }

    Callable callback = callable_mp(this, &GaussianSplatManager::_on_frame_pre_draw);
    if (!rs->is_connected(SNAME("frame_pre_draw"), callback)) {
        rs->connect(SNAME("frame_pre_draw"), callback);
    }
    frame_callbacks_connected = true;
}

void GaussianSplatManager::_disconnect_frame_callbacks() {
    if (!frame_callbacks_connected) {
        return;
    }

    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs) {
        Callable callback = callable_mp(this, &GaussianSplatManager::_on_frame_pre_draw);
        if (rs->is_connected(SNAME("frame_pre_draw"), callback)) {
            rs->disconnect(SNAME("frame_pre_draw"), callback);
        }
    }

    frame_callbacks_connected = false;
}

void GaussianSplatManager::_request_primary_local_device() {
    // Skip local device creation in RenderDoc compatibility mode
    if (renderdoc_compatibility_mode) {
        return;
    }
    if (primary_local_device.load(std::memory_order_acquire)) {
        return;
    }
    if (primary_device_request_pending.is_set()) {
        return;
    }

    primary_device_request_pending.set();
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        WARN_PRINT_ONCE("[GaussianSplatManager] RenderingServer unavailable; local devices will be lazily created");
        primary_device_request_pending.clear();
        return;
    }

    Callable callable = callable_mp(this, &GaussianSplatManager::_create_local_device_on_render_thread).bind(true);
    rs->call_on_render_thread(callable);
}

void GaussianSplatManager::_request_shared_local_device() {
    // Skip local device creation in RenderDoc compatibility mode
    if (renderdoc_compatibility_mode) {
        return;
    }
    if (shared_submission_device.load(std::memory_order_acquire)) {
        return;
    }
    if (shared_device_request_pending.is_set()) {
        return;
    }

    shared_device_request_pending.set();
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs) {
        WARN_PRINT_ONCE("[GaussianSplatManager] RenderingServer unavailable; cannot provide shared submission device");
        shared_device_request_pending.clear();
        return;
    }

    Callable callable = callable_mp(this, &GaussianSplatManager::_create_local_device_on_render_thread).bind(false);
    rs->call_on_render_thread(callable);
}

void GaussianSplatManager::_create_local_device_on_render_thread(bool p_primary_device) {
    // Skip in RenderDoc mode - multiple Vulkan devices are not supported
    if (renderdoc_compatibility_mode) {
        if (p_primary_device) {
            primary_device_request_pending.clear();
        } else {
            shared_device_request_pending.clear();
        }
        return;
    }

    if (p_primary_device && primary_local_device.load(std::memory_order_acquire)) {
        primary_device_request_pending.clear();
        return;
    }
    if (!p_primary_device && shared_submission_device.load(std::memory_order_acquire)) {
        shared_device_request_pending.clear();
        return;
    }

    RenderingServer *rs = RenderingServer::get_singleton();
    RenderingDevice *created_device = nullptr;
    if (rs) {
        created_device = rs->create_local_rendering_device();
    }

    if (!created_device) {
        if (p_primary_device) {
            WARN_PRINT_ONCE("[GaussianSplatManager] Failed to create primary local RenderingDevice");
            primary_device_render_thread_bound = false;
            primary_device_request_pending.clear();
        } else {
            WARN_PRINT_ONCE("[GaussianSplatManager] Failed to create shared local RenderingDevice for submissions");
            shared_device_render_thread_bound = false;
            shared_device_request_pending.clear();
        }
        return;
    }

    if (p_primary_device) {
        RenderingDevice *previous = primary_local_device.exchange(created_device, std::memory_order_acq_rel);
        if (previous && previous != created_device) {
            memdelete(previous);
        }
        primary_device_render_thread_bound = rs ? rs->is_on_render_thread() : false;
        _ensure_render_thread_binding(created_device, primary_device_render_thread_bound);
        primary_device_request_pending.clear();
    } else {
        RenderingDevice *previous = shared_submission_device.exchange(created_device, std::memory_order_acq_rel);
        if (previous && previous != created_device) {
            memdelete(previous);
        }
        shared_device_render_thread_bound = rs ? rs->is_on_render_thread() : false;
        _ensure_render_thread_binding(created_device, shared_device_render_thread_bound);
        shared_device_request_pending.clear();
    }
}

void GaussianSplatManager::_on_frame_pre_draw() {
    bool should_schedule = false;

    {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        if (!main_thread_dispatch_pending.is_set()) {
            main_thread_dispatch_pending.set();
            should_schedule = true;
        }
    }

    if (should_schedule) {
        callable_mp(this, &GaussianSplatManager::_process_active_nodes_main_thread).call_deferred();
    }
}

void GaussianSplatManager::_process_active_nodes_main_thread() {
    LocalVector<ObjectID> nodes_to_process;
    bool disconnect_callbacks = false;

    {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        main_thread_dispatch_pending.clear();

        if (active_nodes.is_empty()) {
            disconnect_callbacks = true;
        } else {
            nodes_to_process.reserve(active_nodes.size());
            for (const ObjectID &id : active_nodes) {
                nodes_to_process.push_back(id);
            }
        }
    }

    if (disconnect_callbacks) {
        _disconnect_frame_callbacks();
        return;
    }

    LocalVector<ObjectID> stale_nodes;
    for (const ObjectID &id : nodes_to_process) {
        Object *obj = ObjectDB::get_instance(id);
        GaussianSplatNode3D *node = Object::cast_to<GaussianSplatNode3D>(obj);
        if (!node) {
            stale_nodes.push_back(id);
            continue;
        }

        node->process_gaussian_render();
    }

    if (!stale_nodes.is_empty()) {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        for (const ObjectID &id : stale_nodes) {
            active_nodes.erase(id);
        }
        disconnect_callbacks = active_nodes.is_empty();
    } else {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        disconnect_callbacks = active_nodes.is_empty();
    }

    if (disconnect_callbacks) {
        _disconnect_frame_callbacks();
    }
}

GaussianSplatManager::~GaussianSplatManager() {
    _disconnect_frame_callbacks();
    {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        active_nodes.clear();
    }
    // Sequential: resource_maps_mutex (level 2) acquired/released inside,
    // then local_device_destroy_dispatch_mutex (level 4) acquired/released inside.
    // No nesting — each lock scope is independent.
    _release_registered_resources();
    _destroy_local_devices();

    singleton = nullptr;
}

void GaussianSplatManager::_bind_methods() {
    // Bind methods for GDScript access
    ClassDB::bind_method(D_METHOD("get_global_stats"), &GaussianSplatManager::get_global_stats);
    ClassDB::bind_method(D_METHOD("set_gpu_sorting_enabled", "enabled"), &GaussianSplatManager::set_gpu_sorting_enabled);
    ClassDB::bind_method(D_METHOD("is_gpu_sorting_enabled"), &GaussianSplatManager::is_gpu_sorting_enabled);
    ClassDB::bind_method(D_METHOD("set_shared_submission_device_enabled", "enabled"), &GaussianSplatManager::set_shared_submission_device_enabled);
    ClassDB::bind_method(D_METHOD("is_shared_submission_device_enabled"), &GaussianSplatManager::is_shared_submission_device_enabled);
    ClassDB::bind_method(D_METHOD("get_sorting_config"), &GaussianSplatManager::get_sorting_config);

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gpu_sorting_enabled"), "set_gpu_sorting_enabled", "is_gpu_sorting_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shared_submission_device_enabled"),
            "set_shared_submission_device_enabled", "is_shared_submission_device_enabled");
}

void GaussianSplatManager::_recalculate_totals() {
    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);
    _recalculate_totals_unlocked();
}

void GaussianSplatManager::_recalculate_totals_unlocked() {
    total_gaussians = 0;
    total_memory_usage = 0;

    for (const KeyValue<ObjectID, BufferEntry> &E : gaussian_buffers) {
        total_gaussians += E.value.gaussian_count;
        total_memory_usage += E.value.memory_usage;
    }

    for (const KeyValue<RID, DynamicAssetEntry> &E : dynamic_asset_cache) {
        total_gaussians += E.value.gaussian_count;
        total_memory_usage += E.value.memory_usage;
    }
}

RID GaussianSplatManager::register_gaussian_buffer(const Ref<::GaussianData> &p_gaussian_data, RenderingDevice *p_rd) {
    ERR_FAIL_COND_V_MSG(p_gaussian_data.is_null(), RID(), "GaussianData must be valid to register a GPU buffer");

    RenderingDevice *owner_device = p_rd ? p_rd : get_primary_rendering_device();
    ERR_FAIL_NULL_V_MSG(owner_device, RID(), "RenderingDevice unavailable; cannot allocate gaussian buffer");

    ObjectID data_id = p_gaussian_data->get_instance_id();
    ERR_FAIL_COND_V_MSG(data_id == ObjectID(), RID(), "GaussianData instance must be registered in ObjectDB");

    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);

    BufferEntry *existing = gaussian_buffers.getptr(data_id);
    if (existing) {
        RenderingDevice *existing_owner = nullptr;
        if (RenderingDevice *const *owner_ptr = gaussian_buffer_owner_devices.getptr(data_id)) {
            existing_owner = *owner_ptr;
        }
        ERR_FAIL_NULL_V_MSG(existing_owner, RID(),
                "[GaussianSplatManager] Missing owner device for existing gaussian buffer");
        ERR_FAIL_COND_V_MSG(existing_owner != owner_device, RID(),
                "[GaussianSplatManager] Gaussian buffer owner mismatch for shared GaussianData");
        existing->ref_count++;
        return existing->gpu_buffer;
    }

    RID gpu_buffer = p_gaussian_data->create_gpu_buffer(owner_device);
    if (!gpu_buffer.is_valid()) {
        GS_LOG_WARN_DEFAULT("[GaussianSplatManager] Failed to allocate GPU buffer for gaussian dataset");
        return RID();
    }

    BufferEntry entry;
    entry.gpu_buffer = gpu_buffer;
    entry.ref_count = 1;
    entry.gaussian_count = p_gaussian_data->get_count();
    entry.memory_usage = (uint64_t)p_gaussian_data->get_memory_usage();

    gaussian_buffers[data_id] = entry;
    buffer_lookup[gpu_buffer] = data_id;
    gaussian_buffer_owner_devices.insert(data_id, owner_device);
    registered_resources_released = false;

    _recalculate_totals_unlocked();

    return gpu_buffer;
}

void GaussianSplatManager::unregister_gaussian_buffer(RID p_buffer) {
    if (!p_buffer.is_valid()) {
        return;
    }

    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);

    ObjectID *data_id_ptr = buffer_lookup.getptr(p_buffer);
    if (!data_id_ptr) {
        return;
    }

    ObjectID data_id = *data_id_ptr;
    BufferEntry *entry = gaussian_buffers.getptr(data_id);
    if (!entry) {
        buffer_lookup.erase(p_buffer);
        return;
    }

    ERR_FAIL_COND(entry->ref_count == 0);

    RenderingDevice *owner_device = nullptr;
    if (entry->ref_count == 1 && entry->gpu_buffer.is_valid()) {
        if (RenderingDevice *const *owner_ptr = gaussian_buffer_owner_devices.getptr(data_id)) {
            owner_device = *owner_ptr;
        }
        ERR_FAIL_NULL_MSG(owner_device, "[GaussianSplatManager] Missing owner device for gaussian buffer free");
    }

    if (--entry->ref_count > 0) {
        return;
    }

    if (entry->gpu_buffer.is_valid()) {
        owner_device->free(entry->gpu_buffer);
    }

    gaussian_buffers.erase(data_id);
    gaussian_buffer_owner_devices.erase(data_id);
    buffer_lookup.erase(p_buffer);
    _recalculate_totals_unlocked();
}

RID GaussianSplatManager::get_gpu_buffer(RID p_gaussian_data) const {
    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);
    if (buffer_lookup.has(p_gaussian_data)) {
        return p_gaussian_data;
    }
    return RID();
}

GaussianSplatManager::SharedDynamicAssetHandle GaussianSplatManager::acquire_dynamic_asset(
        const Ref<GaussianSplatAsset> &p_asset, const Ref<::GaussianData> &p_gaussian_data, RenderingDevice *p_rd) {
    SharedDynamicAssetHandle handle;

    ERR_FAIL_COND_V(p_asset.is_null(), handle);
    ERR_FAIL_COND_V(p_gaussian_data.is_null(), handle);

    RID asset_rid = p_asset->get_rid();
    if (!asset_rid.is_valid()) {
        asset_rid = RID::from_uint64((uint64_t)p_asset->get_instance_id());
    }
    ERR_FAIL_COND_V_MSG(!asset_rid.is_valid(), handle, "GaussianSplatAsset must provide a valid RID or instance identifier");

    RenderingDevice *owner_device = p_rd ? p_rd : get_primary_rendering_device();
    ERR_FAIL_NULL_V_MSG(owner_device, handle, "RenderingDevice unavailable for dynamic asset buffer allocation");

    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);

    DynamicAssetEntry *entry = dynamic_asset_cache.getptr(asset_rid);
    if (!entry) {
        DynamicAssetEntry new_entry;
        dynamic_asset_cache.insert(asset_rid, new_entry);
        entry = dynamic_asset_cache.getptr(asset_rid);
    }

    ERR_FAIL_NULL_V(entry, handle);

    RenderingDevice *tracked_owner_device = nullptr;
    if (RenderingDevice *const *owner_ptr = dynamic_asset_owner_devices.getptr(asset_rid)) {
        tracked_owner_device = *owner_ptr;
    }
    if (entry->gaussian_buffer.is_valid()) {
        ERR_FAIL_NULL_V_MSG(tracked_owner_device, handle,
                "[GaussianSplatManager] Missing owner device for existing dynamic asset buffer");
    }

    uint32_t desired_count = p_gaussian_data->get_count();
    const bool owner_changed = entry->gaussian_buffer.is_valid() && tracked_owner_device != owner_device;
    ERR_FAIL_COND_V_MSG(owner_changed && entry->ref_count > 0, handle,
            "[GaussianSplatManager] Cannot migrate dynamic asset buffer owner while handles are still in use");
    bool needs_upload = !entry->gaussian_buffer.is_valid() || entry->gaussian_count != desired_count || owner_changed;

    if (needs_upload) {
        if (entry->gaussian_buffer.is_valid()) {
            tracked_owner_device->free(entry->gaussian_buffer);
            entry->gaussian_buffer = RID();
            dynamic_asset_owner_devices.erase(asset_rid);
        }
        entry->gaussian_buffer = p_gaussian_data->create_gpu_buffer(owner_device);
        entry->gaussian_count = desired_count;
        entry->memory_usage = (uint64_t)p_gaussian_data->get_memory_usage();
        if (!entry->gaussian_buffer.is_valid()) {
            entry->gaussian_count = 0;
            entry->memory_usage = 0;
            dynamic_asset_owner_devices.erase(asset_rid);
        } else {
            dynamic_asset_owner_devices.insert(asset_rid, owner_device);
            registered_resources_released = false;
        }
    }

    if (!entry->gaussian_buffer.is_valid()) {
        return handle;
    }

    entry->ref_count++;

    handle.asset_rid = asset_rid;
    handle.gaussian_buffer = entry->gaussian_buffer;
    handle.gaussian_count = entry->gaussian_count;

    _recalculate_totals_unlocked();

    return handle;
}

void GaussianSplatManager::release_dynamic_asset(const SharedDynamicAssetHandle &p_handle) {
    if (!p_handle.is_valid()) {
        return;
    }

    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);

    DynamicAssetEntry *entry = dynamic_asset_cache.getptr(p_handle.asset_rid);
    if (!entry) {
        return;
    }

    ERR_FAIL_COND(entry->ref_count == 0);

    RenderingDevice *owner_device = nullptr;
    if (entry->ref_count == 1 && entry->gaussian_buffer.is_valid()) {
        if (RenderingDevice *const *owner_ptr = dynamic_asset_owner_devices.getptr(p_handle.asset_rid)) {
            owner_device = *owner_ptr;
        }
        ERR_FAIL_NULL_MSG(owner_device, "[GaussianSplatManager] Missing owner device for dynamic asset buffer free");
    }

    if (--entry->ref_count > 0) {
        return;
    }

    if (entry->gaussian_buffer.is_valid()) {
        owner_device->free(entry->gaussian_buffer);
    }

    dynamic_asset_cache.erase(p_handle.asset_rid);
    dynamic_asset_owner_devices.erase(p_handle.asset_rid);
    _recalculate_totals_unlocked();
}

void GaussianSplatManager::update_stats(uint64_t p_gaussian_count, uint64_t p_memory_usage) {
    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);
    last_reported_gaussians = p_gaussian_count;
    last_reported_memory = p_memory_usage;
    _recalculate_totals_unlocked();
}

Dictionary GaussianSplatManager::get_global_stats() const {
    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_RESOURCE_MAPS, "resource_maps_mutex");
    MutexLock resource_lock(resource_maps_mutex);

    Dictionary stats;
    stats["total_gaussians"] = total_gaussians;
    stats["total_memory_mb"] = total_memory_usage / (1024.0 * 1024.0);
    stats["buffer_count"] = gaussian_buffers.size();
    stats["gpu_sorting_enabled"] = gpu_sorting_enabled;
    stats["shared_submission_device_enabled"] = shared_submission_device_enabled;
    stats["sorting"] = get_sorting_config();
    stats["reported_gaussians"] = last_reported_gaussians;
    stats["reported_memory_mb"] = last_reported_memory / (1024.0 * 1024.0);
    return stats;
}

void GaussianSplatManager::set_gpu_sorting_enabled(bool p_enabled) {
    gpu_sorting_enabled = p_enabled;
}

void GaussianSplatManager::set_shared_submission_device_enabled(bool p_enabled) {
    shared_submission_device_enabled = p_enabled;
}

Dictionary GaussianSplatManager::get_sorting_config() const {
    Dictionary cfg;
    cfg["bitonic_max_elements"] = sorting_bitonic_max;
    cfg["radix_max_elements"] = sorting_radix_max;
    cfg["onesweep_max_elements"] = sorting_onesweep_max;
    cfg["hybrid_trigger_elements"] = sorting_hybrid_trigger;
    cfg["hybrid_batch_size"] = sorting_hybrid_batch;
    cfg["history_size"] = sorting_history_size;
    cfg["log_interval_frames"] = sorting_log_interval;
    cfg["target_sort_time_ms"] = sorting_target_ms;
    cfg["log_metrics"] = sorting_log_metrics;
    cfg["force_algorithm"] = sorting_force_algorithm;
    cfg["force_cpu_sort"] = sorting_force_cpu_sort;
    return cfg;
}

void GaussianSplatManager::initialize_module() {
    // Register project settings
    GLOBAL_DEF("rendering/gaussian_splatting/gpu_sorting_enabled", true);
    GLOBAL_DEF("rendering/gaussian_splatting/shared_submission_device_enabled", false);
    // Scene composite depth policy: 0=strict (skip frame if depth contract is missing), 1=relaxed (allow no-depth blend fallback).
    GLOBAL_DEF("rendering/gaussian_splatting/composite/scene_depth_policy", 0);
    GLOBAL_DEF("rendering/gaussian_splatting/streaming/enabled", true);
    // Chunk-level frustum culling for streaming (FlashGS/LiteGS/H3DGS technique)
    // Culls entire chunks before loading, reducing GPU resource waste on off-screen chunks
    GLOBAL_DEF("rendering/gaussian_splatting/streaming/chunk_frustum_culling_enabled", true);
    GLOBAL_DEF("rendering/gaussian_splatting/streaming/chunk_frustum_padding", 1.5f);
    // Recovery policy when culling repeatedly reports zero visible chunks:
    // 0=startup-only guard, 1=persistent bounded recovery.
    GLOBAL_DEF("rendering/gaussian_splatting/streaming/zero_visible_recovery_mode", 1);
    GLOBAL_DEF("rendering/gaussian_splatting/streaming/zero_visible_recovery_trigger_frames", 16);
    GLOBAL_DEF("rendering/gaussian_splatting/streaming/zero_visible_recovery_cooldown_frames", 30);
    GLOBAL_DEF("rendering/gaussian_splatting/streaming/zero_visible_recovery_log_interval_frames", 120);
    GLOBAL_DEF("rendering/gaussian_splatting/max_gpu_buffer_count", 128);
    // Cache PLY loads into a sibling .gsplatworld file for faster subsequent loads.
    GLOBAL_DEF("rendering/gaussian_splatting/import/use_gsplatworld_cache", true);
    // Disable by default to avoid heavy gzip decompression cost on large cache-hit loads.
    GLOBAL_DEF("rendering/gaussian_splatting/import/gsplatworld_compression_enabled", false);
    GLOBAL_DEF("rendering/gaussian_splatting/editor/hot_reload_enabled", true);
    GLOBAL_DEF("rendering/gaussian_splatting/editor/hot_reload_poll_interval_sec", 1.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_enabled", false);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_direction_x", 1.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_direction_y", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_direction_z", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_strength", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_frequency", 1.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_spatial_frequency", 0.1f);
    GLOBAL_DEF("rendering/gaussian_splatting/animation/wind_time_scale", 1.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/max_effectors", 1);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/sphere_effector_enabled", false);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/sphere_effector_center_x", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/sphere_effector_center_y", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/sphere_effector_center_z", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/sphere_effector_radius", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/sphere_effector_strength", 0.0f);
    GLOBAL_DEF("rendering/gaussian_splatting/effects/sphere_effector_falloff", 2.0f);

    // Culling and LOD settings from PR #146
    GLOBAL_DEF("rendering/gaussian_splatting/culling/octree_max_depth", 8);
    GLOBAL_DEF("rendering/gaussian_splatting/culling/min_gaussians_per_leaf", 32);

    // Cluster-level coarse culling (LiteGS-style) - groups splats into clusters for fast AABB testing
    GLOBAL_DEF("rendering/gaussian_splatting/culling/cluster_culling_enabled", true);
    GLOBAL_DEF("rendering/gaussian_splatting/culling/cluster_target_size", 128);       // Splats per cluster (32-256)
    GLOBAL_DEF("rendering/gaussian_splatting/culling/cluster_frustum_slack", 2.0f);    // AABB expansion factor
    // Projection anti-aliasing floor (adds minimum covariance variance in tile binning).
    // Lower values are sharper but can re-introduce subpixel holes; 0.35 balances sharpness/stability.
    GLOBAL_DEF("rendering/gaussian_splatting/rasterization/low_pass_filter", 0.35f);

	// Opacity-aware bounding (FlashGS optimization) - reduces tile-Gaussian pairs by ~94%
	// When enabled, splat radii are calculated based on opacity: r = sqrt(2 * ln(alpha/tau) * lambda)
	GLOBAL_DEF("rendering/gaussian_splatting/culling/opacity_aware_bounds", true);     // Enable opacity-aware bounding
	GLOBAL_DEF("rendering/gaussian_splatting/culling/visibility_threshold", 0.01f);    // tau: minimum visible contribution

	// EXPERIMENTAL: overflow auto-tuner (disabled by default due to splat decay bug).
	// Enable only for testing until the feedback loop is fixed.
	GLOBAL_DEF("rendering/gaussian_splatting/cull/overflow_autotune_enabled", false);

    // LOD + debug overlay settings now register in their owning config managers.
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_frame_logging", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_frame_logging_verbose", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/frame_log_frequency", 300);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_mainloop_probes", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_all_debug", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_sort_path_logs", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_tile_logs", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_tile_pipeline_logs", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_tile_dispatch_logs", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_gpu_counter_logs", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_binning_counters", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_cull_counters", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_cull_guardrails", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/cull_guardrail_position_epsilon", 0.05f);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/cull_guardrail_rotation_epsilon", 0.01f);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/cull_guardrail_drop_ratio", 0.75f);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/cull_guardrail_min_visible", 256);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_autotune_logs", false);
	GLOBAL_DEF("rendering/gaussian_splatting/debug/enable_data_logging", false);

	// Logging configuration (quiet by default).
	GLOBAL_DEF("rendering/gaussian_splatting/logging/verbosity", "silent");
	GLOBAL_DEF("rendering/gaussian_splatting/logging/rate_limit_ms", 1000);

	// Production diagnostics contract + perf gates.
	GLOBAL_DEF("rendering/gaussian_splatting/diagnostics/validate_production_metrics", true);
	GLOBAL_DEF("rendering/gaussian_splatting/diagnostics/summary_interval_frames", 600);
	GLOBAL_DEF("rendering/gaussian_splatting/diagnostics/summary_history_size", 60);
	GLOBAL_DEF("rendering/gaussian_splatting/diagnostics/perf_gate_enabled", false);
	GLOBAL_DEF("rendering/gaussian_splatting/diagnostics/perf_gate_splat_threshold", 100000);
	GLOBAL_DEF("rendering/gaussian_splatting/diagnostics/perf_gate_budget_ms", 16.0f);

	// Quality tier presets (hardware-level defaults).
	GLOBAL_DEF("rendering/gaussian_splatting/quality/tier_preset", "custom");
	GLOBAL_DEF("rendering/gaussian_splatting/quality/tier_apply_pipeline_toggles", true);
	GLOBAL_DEF("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", true);

    // Pipeline feature + GPU sorting settings now register in their owning config managers.

	// Predictive prefetch settings for streaming
	// When enabled, uses camera velocity to predict future position and preload chunks
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/predictive_prefetch_enabled", true);
	// How far ahead (in world units) to look for chunks to prefetch based on camera velocity
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/prefetch_lookahead_distance", 10.0f);
	// Async chunk packing (background CPU packing)
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/async_pack_enabled", true);
	// Worker threads for async chunk packing (minimum 1).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/pack_worker_threads", 2);
	// Max async pack jobs queued/processing at once (0 = unlimited).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_pack_jobs_in_flight", 4);
	// Limit chunk uploads per frame to avoid stalls (0 = unlimited).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame", 16);
	// Optional prefetch queue budget per frame (0 = disabled).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_prefetch_loads_per_frame", 6);
	// Max visible chunks scanned per frame during load candidate selection (0 = unbounded).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_visible_chunk_scan_per_frame", 4096);
	// Max chunks scanned per frame during predictive prefetch candidate selection (0 = unbounded).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_prefetch_chunk_scan_per_frame", 4096);
	// Queue-pressure throttle for candidate scan budgets.
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/queue_pressure_candidate_scan_throttle_enabled", true);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/queue_pressure_candidate_scan_throttle_min_queue_depth", 1);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/queue_pressure_visible_scan_cap", 1024);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/queue_pressure_prefetch_scan_cap", 1024);
	// Cap total chunk upload bandwidth per frame in MB (0 = unlimited).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_upload_mb_per_frame", 128);
	// Cap chunk upload size per slice in MB (0 = unlimited).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_upload_mb_per_slice", 16);
	// Optional bandwidth cap for upload queue in MB/s (0 = disabled).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_upload_mb_per_second", 0);
	// Min frames a chunk must remain resident before eviction.
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/eviction_hysteresis_frames", 5);
	// Max chunks evicted per frame (0 = unlimited).
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_evictions_per_frame", 4);
	// Async IO for gsplatworld sources.
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/async_io_enabled", false);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/io_source_path", String());

	// VRAM budget auto-regulation settings for streaming system (H3DGS-style).
	// Enables graceful degradation when approaching memory limits.
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/vram_budget_mb", 12288);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/auto_regulate_enabled", true);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/vram_warning_threshold_percent", 85);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/min_chunks_in_vram", 4);
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/max_chunks_in_vram", 128);
	// Step size for budget adjustments (1% as recommended by Virtual Memory 3DGS paper)
	GLOBAL_DEF("rendering/gaussian_splatting/streaming/regulation_step_percent", 1.0f);

	// Sorting configuration settings from master
	GLOBAL_DEF("rendering/gaussian_splatting/sorting/bitonic_max_elements", (int)sorting_bitonic_max);
	GLOBAL_DEF("rendering/gaussian_splatting/sorting/radix_max_elements", (int)sorting_radix_max);
	GLOBAL_DEF("rendering/gaussian_splatting/sorting/onesweep_max_elements", (int)sorting_onesweep_max);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/hybrid_trigger_elements", (int)sorting_hybrid_trigger);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/hybrid_batch_size", (int)sorting_hybrid_batch);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/history_size", (int)sorting_history_size);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/log_interval_frames", (int)sorting_log_interval);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/target_sort_time_ms", sorting_target_ms);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/log_metrics", sorting_log_metrics);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/force_algorithm", 0);
    GLOBAL_DEF("rendering/gaussian_splatting/sorting/force_cpu_sort", false);

    // Note: set_custom_property_info signature has changed in Godot 4
    // The property info is now set through GLOBAL_DEF
}

void GaussianSplatManager::finalize_module() {
    // Cleanup module resources
    _release_registered_resources();
    {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        active_nodes.clear();
    }
    _disconnect_frame_callbacks();
}

RenderingDevice *GaussianSplatManager::get_primary_rendering_device() {
    // CRITICAL FIX: Always return the MAIN RenderingDevice, not a separate local device.
    // Resources created on a local device CANNOT be used by the main rendering pipeline,
    // which causes cross-device errors and prevents splats from being visible on screen.
    // The main RenderingDevice is the one Godot uses for presentation - all our GPU
    // resources must be created on this device to be usable by TileRenderer's draw calls.
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs) {
        RenderingDevice *main_device = rs->get_rendering_device();
        if (main_device) {
            return main_device;
        }
    }

    // Fallback to legacy behavior only if main device unavailable (shouldn't happen)
    RenderingDevice *device = primary_local_device.load(std::memory_order_acquire);
    if (!device) {
        _request_primary_local_device();
        device = primary_local_device.load(std::memory_order_acquire);
    }
    return device;
}

RenderingDevice *GaussianSplatManager::get_shared_submission_device() {
    if (!shared_submission_device_enabled) {
        RenderingDevice *primary = get_primary_rendering_device();
        if (!primary) {
            WARN_PRINT_ONCE("[GaussianSplatManager] Primary RenderingDevice unavailable");
        }
        return primary;
    }

    GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_SUBMISSION, "submission_mutex");
    MutexLock<SafeBinaryMutex<SUBMISSION_MUTEX_TAG>> lock(submission_mutex);
    RenderingDevice *shared = _ensure_shared_submission_device_locked();
    if (!shared) {
        shared = get_primary_rendering_device();
        if (!shared) {
            WARN_PRINT_ONCE("[GaussianSplatManager] Unable to provide submission RenderingDevice");
        }
    }
    return shared;
}

RenderingDevice *GaussianSplatManager::_ensure_shared_submission_device_locked() {
    RenderingDevice *shared = shared_submission_device.load(std::memory_order_acquire);
    if (shared) {
        return shared;
    }

    if (!shared_device_request_pending.is_set()) {
        _request_shared_local_device();
    }

    return shared_submission_device.load(std::memory_order_acquire);
}

GaussianSplatManager::ScopedSubmissionLock GaussianSplatManager::acquire_submission_lock() {
    return ScopedSubmissionLock(*this);
}

RenderingDevice *GaussianSplatManager::acquire_submission_device(RenderingDevice *p_device, ScopedSubmissionLock &r_lock) {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (rs && rs->is_on_render_thread()) {
        ensure_render_thread_local_devices_current();
    }

    if (p_device && !is_shared_submission_device(p_device)) {
        return p_device;
    }

    ScopedSubmissionLock lock(*this);
    RenderingDevice *device = lock.get_rendering_device();
    if (device) {
        r_lock = std::move(lock);
    }
    return device;
}

bool GaussianSplatManager::is_shared_submission_device(RenderingDevice *p_device) const {
    return p_device != nullptr && p_device == shared_submission_device.load(std::memory_order_acquire);
}

void GaussianSplatManager::ensure_render_thread_local_devices_current() {
    RenderingServer *rs = RenderingServer::get_singleton();
    if (!rs || !rs->is_on_render_thread()) {
        return;
    }

    RenderingDevice *primary = primary_local_device.load(std::memory_order_acquire);
    _ensure_render_thread_binding(primary, primary_device_render_thread_bound);

    RenderingDevice *shared = shared_submission_device.load(std::memory_order_acquire);
    if (shared && shared != primary) {
        _ensure_render_thread_binding(shared, shared_device_render_thread_bound);
    } else if (shared) {
        shared_device_render_thread_bound = primary_device_render_thread_bound;
    }
}

void GaussianSplatManager::register_node(GaussianSplatNode3D *p_node) {
    ERR_FAIL_NULL(p_node);

    ObjectID node_id = p_node->get_instance_id();
    if (node_id == ObjectID()) {
        return;
    }

    bool should_connect = false;
    {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        should_connect = active_nodes.is_empty();
        active_nodes.insert(node_id);
    }

    if (should_connect) {
        _connect_frame_callbacks();
    }
}

void GaussianSplatManager::unregister_node(GaussianSplatNode3D *p_node) {
    if (!p_node) {
        return;
    }

    ObjectID node_id = p_node->get_instance_id();
    if (node_id == ObjectID()) {
        return;
    }

    bool should_disconnect = false;
    {
        GS_LOCK_ORDER_GUARD(GaussianSplatManager::LOCK_LEVEL_ACTIVE_NODES, "active_nodes_mutex");
        MutexLock lock(active_nodes_mutex);
        active_nodes.erase(node_id);
        should_disconnect = active_nodes.is_empty();
    }

    if (should_disconnect) {
        _disconnect_frame_callbacks();
    }
}
