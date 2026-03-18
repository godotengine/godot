#ifndef GS_CULLER_INTERFACES_H
#define GS_CULLER_INTERFACES_H

#include "core/math/plane.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2i.h"
#include "core/templates/local_vector.h"
#include "servers/rendering/rendering_device.h"

// Culling parameters passed to the culler
struct CullParams {
    // Transforms
    Transform3D world_to_camera_transform;  // World-to-camera transform (view matrix)
    Transform3D camera_to_world_transform;  // Camera-to-world transform (camera's world pose)
    Projection projection;
    Size2i viewport_size;

    // Frustum planes (extracted from projection)
    Vector<Plane> frustum_planes;

    // Camera info
    Vector3 camera_position;
    float pixel_scale_y = 1.0f;
    bool orthographic = false;

    // Culling thresholds
    float near_tolerance = 0.01f;
    float far_tolerance = 1000.0f;
    float min_screen_size = 0.5f;
    float max_distance_sq = 10000.0f;
    float importance_threshold = 0.0f;
    float radius_multiplier = 1.0f;
    float frustum_plane_slack = 1.0f;
    float tiny_splat_screen_radius = 0.5f;

    // Limits
    uint32_t max_visible = UINT32_MAX;
    bool frustum_culling_enabled = true;

    // Readback controls (GPU cullers can ignore unneeded buffers)
    bool readback_indices = true;
    bool readback_distances = true;
    bool readback_importance = true;
};

// Culling statistics/counters
struct CullCounters {
    uint32_t visible_count = 0;
    uint32_t frustum_culled = 0;
    uint32_t distance_culled = 0;
    uint32_t screen_culled = 0;
    uint32_t importance_culled = 0;
    uint32_t clipped_count = 0;
    uint32_t near_clamped = 0;
    uint32_t behind_culled = 0;
    uint32_t total_input = 0;
    float culling_time_ms = 0.0f;
    bool used_hierarchical = false;
};

// Result of a culling operation
struct CullResult {
    LocalVector<uint32_t> visible_indices;
    LocalVector<float> distances_sq;
    LocalVector<float> importance_weights;
    CullCounters counters;
    bool success = false;
};

// Input buffer description for GPU culling
struct CullInputBuffers {
    RID gaussian_buffer;              // Source gaussian data buffer
    RenderingDevice *buffer_device = nullptr;
    uint32_t total_splat_count = 0;
};

// Pure abstract interface for culling implementations
class ICuller {
public:
    virtual ~ICuller() = default;

    // Initialize the culler with a rendering device
    virtual Error initialize(RenderingDevice *p_device) = 0;

    // Release all resources
    virtual void shutdown() = 0;

    // Check if the culler is ready to use
    virtual bool is_ready() const = 0;

    // Perform culling operation
    // For GPU cullers: dispatches compute shader and reads back results
    // For CPU cullers: performs culling on CPU
    virtual CullResult cull(const CullParams &p_params, const CullInputBuffers &p_input) = 0;

    // Get the name of this culler implementation
    virtual String get_name() const = 0;

    // Check if this is a GPU-based culler
    virtual bool is_gpu_based() const = 0;
};

// Interface for hierarchical/chunk-based culling
struct ChunkBounds {
    Vector3 center;
    float radius = 0.0f;
    AABB aabb;
    LocalVector<uint32_t> splat_indices;
};

class IHierarchicalCuller : public ICuller {
public:
    // Set the chunk hierarchy for hierarchical culling
    virtual void set_chunks(const LocalVector<ChunkBounds> &p_chunks) = 0;

    // Get visible chunk indices after culling
    virtual const LocalVector<uint32_t> &get_visible_chunks() const = 0;
};

#endif // GS_CULLER_INTERFACES_H
