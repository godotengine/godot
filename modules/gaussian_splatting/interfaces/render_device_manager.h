#ifndef GS_RENDER_DEVICE_MANAGER_H
#define GS_RENDER_DEVICE_MANAGER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/string/ustring.h"
#include "servers/rendering/rendering_device.h"
#include "../core/gaussian_splat_manager.h"

// Context struct holding device references for render operations
struct DeviceContext {
    RenderingDevice *main_device = nullptr;      // Primary RD (singleton)
    RenderingDevice *submission_device = nullptr; // Device for command submission
    RenderingDevice *resource_device = nullptr;   // Device that owns resources
};

// Pure abstract interface for RenderingDevice management
// Manages device lifecycle, resource ownership tracking, and cross-device sync
class IRenderDeviceManager {
public:
    virtual ~IRenderDeviceManager() = default;

    // Device access
    virtual DeviceContext get_context() const = 0;
    virtual RenderingDevice *get_main_device() const = 0;
    virtual RenderingDevice *get_submission_device() const = 0;
    virtual RenderingDevice *acquire_rendering_device() = 0;
    virtual bool ensure_rendering_device(const char *p_context) = 0;
    virtual bool ensure_submission_device(const char *p_context) = 0;

    // Resource ownership tracking
    virtual void track_resource(const RID &p_rid, RenderingDevice *p_device, bool p_owned = true, const char *p_label = nullptr) = 0;
    virtual RenderingDevice *get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback = nullptr) const = 0;
    virtual void forget_resource(const RID &p_rid) = 0;
    virtual void free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid) = 0;

    // Texture-specific tracking (textures have special handling)
    virtual void track_texture(const RID &p_texture, RenderingDevice *p_device) = 0;
    virtual RenderingDevice *get_texture_owner(const RID &p_texture) const = 0;

    // Command submission
    virtual void submit_and_sync(RenderingDevice *p_device) = 0;

    // Diagnostics
    virtual void push_texture_trace(const String &p_action, const RID &p_texture, RenderingDevice *p_device) = 0;
    virtual void push_cross_device_operation(const String &p_context, RenderingDevice *p_source, RenderingDevice *p_target) = 0;
};

// Concrete implementation of IRenderDeviceManager
class RenderDeviceManager : public RefCounted, public IRenderDeviceManager {
    GDCLASS(RenderDeviceManager, RefCounted);

public:
    RenderDeviceManager();
    ~RenderDeviceManager();

    // Initialize with optional primary device (if null, acquires from singleton)
    Error initialize(RenderingDevice *p_primary_device = nullptr);
    void shutdown();

    // IRenderDeviceManager implementation
    DeviceContext get_context() const override;
    RenderingDevice *get_main_device() const override;
    RenderingDevice *get_submission_device() const override;
    RenderingDevice *acquire_rendering_device() override;
    bool ensure_rendering_device(const char *p_context) override;
    bool ensure_submission_device(const char *p_context) override;

    void track_resource(const RID &p_rid, RenderingDevice *p_device, bool p_owned = true, const char *p_label = nullptr) override;
    RenderingDevice *get_resource_owner(const RID &p_rid, RenderingDevice *p_fallback = nullptr) const override;
    void forget_resource(const RID &p_rid) override;
    void free_owned_resource(RenderingDevice *p_fallback_device, RID &p_rid) override;

    void track_texture(const RID &p_texture, RenderingDevice *p_device) override;
    RenderingDevice *get_texture_owner(const RID &p_texture) const override;

    void submit_and_sync(RenderingDevice *p_device) override;

    void push_texture_trace(const String &p_action, const RID &p_texture, RenderingDevice *p_device) override;
    void push_cross_device_operation(const String &p_context, RenderingDevice *p_source, RenderingDevice *p_target) override;

    // Diagnostics access
    struct TextureTraceEntry {
        uint64_t timestamp_usec = 0;
        String action;
        uint64_t texture_rid = 0;
        uint64_t device_instance_id = 0;
#ifdef DEBUG_ENABLED
        uint64_t device_pointer_debug = 0;
#endif
    };

    struct CrossDeviceOperation {
        uint64_t timestamp_usec = 0;
        String context;
        uint64_t source_device_instance_id = 0;
        uint64_t target_device_instance_id = 0;
#ifdef DEBUG_ENABLED
        uint64_t source_device_pointer_debug = 0;
        uint64_t target_device_pointer_debug = 0;
#endif
    };

    const Vector<TextureTraceEntry> &get_texture_trace() const { return texture_trace; }
    const Vector<CrossDeviceOperation> &get_cross_device_operations() const { return cross_device_ops; }
    void clear_diagnostics();

    // Resource statistics (Phase C addition)
    uint32_t get_tracked_resource_count() const { return resource_owner_map.size(); }
    uint32_t get_tracked_texture_count() const { return texture_owner_map.size(); }

protected:
    static void _bind_methods();

private:
	// Device references
	RenderingDevice *main_rd = nullptr;
	RenderingDevice *local_rd = nullptr;
	RenderingDevice *submission_rd = nullptr;
	bool owns_local_rd = false;
	mutable bool reported_missing_device = false;
	mutable bool reported_missing_submission = false;

	// Resource tracking maps
	enum ResourceTypeFlags : uint8_t {
		RESOURCE_TYPE_NONE = 0,
		RESOURCE_TYPE_TEXTURE = 1 << 0,
		RESOURCE_TYPE_FRAMEBUFFER = 1 << 1,
		RESOURCE_TYPE_UNIFORM_SET = 1 << 2,
		RESOURCE_TYPE_RENDER_PIPELINE = 1 << 3,
		RESOURCE_TYPE_COMPUTE_PIPELINE = 1 << 4,
		RESOURCE_TYPE_BUFFER = 1 << 5,
	};

	HashMap<uint64_t, RenderingDevice *> resource_owner_map;
	HashMap<uint64_t, uint64_t> resource_owner_instance_id_map;
	HashMap<uint64_t, bool> resource_ownership_map;
	HashMap<uint64_t, String> resource_label_map;
	HashMap<uint64_t, uint8_t> resource_type_map;
	HashMap<uint64_t, RenderingDevice *> texture_owner_map;

    // Diagnostics
    Vector<TextureTraceEntry> texture_trace;
    Vector<CrossDeviceOperation> cross_device_ops;
    static constexpr int MAX_TRACE_ENTRIES = 1000;

    // Internal helpers
    RenderingDevice *_acquire_manager_device();
    bool _is_main_rendering_device(RenderingDevice *p_device) const;

public:
    // Extended helpers for cross-device operations (used by god class)
    RenderingDevice *acquire_submission_device_for(RenderingDevice *p_device, GaussianSplatManager::ScopedSubmissionLock &r_lock) const;
    void synchronize_tile_submission(RenderingDevice *p_device, const char *p_context);
};

#endif // GS_RENDER_DEVICE_MANAGER_H
