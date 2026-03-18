#ifndef GS_SYNC_POLICY_H
#define GS_SYNC_POLICY_H

#include "core/object/ref_counted.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"

// Helper functions to safely submit/sync RenderingDevice
// Only the main/local RenderingDevice can call submit() and sync()
namespace gs_device_utils {

inline bool is_local_device(RenderingDevice *p_device) {
    if (!p_device) {
        return false;
    }
    // Local devices are NOT the main instance and CAN call submit/sync
    return !p_device->is_main_rendering_device();
}

// Safe submit - only calls submit() on local (non-main) devices
inline void safe_submit(RenderingDevice *p_device) {
    if (p_device && is_local_device(p_device)) {
        p_device->submit();
    }
    // Main device: submit is handled automatically by Godot's frame
}

// Safe sync - only calls sync() on local (non-main) devices
inline void safe_sync(RenderingDevice *p_device) {
    if (p_device && is_local_device(p_device)) {
        p_device->sync();
    }
    // Main device: sync is handled automatically by Godot's frame
}

// Safe submit and sync combo
inline void safe_submit_and_sync(RenderingDevice *p_device) {
    if (p_device && is_local_device(p_device)) {
        p_device->submit();
        p_device->sync();
    }
    // Main device: handled automatically by Godot's frame
}

} // namespace gs_device_utils

namespace gs_sort_policy {

enum class ReadbackMode {
    STRICT_ASYNC,
    ASYNC_WITH_SYNC_BOOTSTRAP,
    STRICT_SYNC,
    DEBUG_VALIDATION,
};

struct ReadbackPolicy {
    ReadbackMode mode = ReadbackMode::STRICT_ASYNC;
    bool allow_sync_readback = false;
    bool allow_sync_sort_fallback = false;
    bool allow_sync_bootstrap = false;
    bool allow_sync_pending_readback = false;
    bool allow_sync_enqueue_fallback = false;
};

inline const char *mode_name(ReadbackMode p_mode) {
    switch (p_mode) {
        case ReadbackMode::STRICT_ASYNC:
            return "strict_async";
        case ReadbackMode::ASYNC_WITH_SYNC_BOOTSTRAP:
            return "async_with_sync_bootstrap";
        case ReadbackMode::STRICT_SYNC:
            return "strict_sync";
        case ReadbackMode::DEBUG_VALIDATION:
            return "debug_validation";
        default:
            return "strict_async";
    }
}

inline ReadbackPolicy make_policy(ReadbackMode p_mode) {
    ReadbackPolicy policy;
    policy.mode = p_mode;
    switch (p_mode) {
        case ReadbackMode::STRICT_ASYNC: {
            break;
        }
        case ReadbackMode::ASYNC_WITH_SYNC_BOOTSTRAP: {
            policy.allow_sync_bootstrap = true;
            // Keep startup correctness with a single bootstrap sample, but
            // preserve strict async behavior in steady-state to avoid recurring
            // CPU/GPU stalls from pending/enqueue sync readbacks.
            policy.allow_sync_pending_readback = false;
            policy.allow_sync_enqueue_fallback = false;
            break;
        }
        case ReadbackMode::STRICT_SYNC:
        case ReadbackMode::DEBUG_VALIDATION: {
            policy.allow_sync_readback = true;
            policy.allow_sync_sort_fallback = true;
            policy.allow_sync_bootstrap = true;
            policy.allow_sync_pending_readback = true;
            policy.allow_sync_enqueue_fallback = true;
            break;
        }
        default:
            break;
    }
    return policy;
}

inline ReadbackPolicy resolve_readback_policy(bool p_debug_sync_requested, bool p_preserve_gpu_timestamps) {
    if (p_debug_sync_requested && !p_preserve_gpu_timestamps) {
        return make_policy(ReadbackMode::DEBUG_VALIDATION);
    }
    if (p_preserve_gpu_timestamps) {
        return make_policy(ReadbackMode::STRICT_ASYNC);
    }
    return make_policy(ReadbackMode::ASYNC_WITH_SYNC_BOOTSTRAP);
}

} // namespace gs_sort_policy

// Pure abstract interface - implementations should inherit from RefCounted
class ISyncPolicy {
public:
    virtual ~ISyncPolicy() = default;
    virtual bool sync(RenderingDevice *p_device, const char *p_context = nullptr) = 0;
};

class CoarseSyncPolicy : public RefCounted, public ISyncPolicy {
    GDCLASS(CoarseSyncPolicy, RefCounted);

protected:
    static void _bind_methods() {}

public:
    bool sync(RenderingDevice *p_device, const char *p_context = nullptr) override {
        if (p_device == nullptr) {
            return false;
        }

        // Use safe variants to avoid errors on main device
        gs_device_utils::safe_submit_and_sync(p_device);
        return true;
    }
};

#endif // GS_SYNC_POLICY_H
