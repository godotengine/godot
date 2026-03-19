/**
 * @file gaussian_data.h
 * @brief Core data structures for Gaussian Splatting representation.
 *
 * This file defines the primary data types used throughout the Gaussian Splatting
 * module: the GPU-aligned Gaussian struct for rendering, the GaussianData namespace
 * structures for CPU-side manipulation, and the GaussianData Resource class for
 * Godot integration.
 */

#ifndef GAUSSIAN_DATA_H
#define GAUSSIAN_DATA_H

#include <cstddef>

#include "core/io/resource.h"
#include "core/os/mutex.h"
#include "core/os/rw_lock.h"
#include "core/variant/variant.h"
#include "core/math/plane.h"
#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/math/quaternion.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/color.h"
#include "core/math/aabb.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "servers/rendering_server.h"
#include "../animation/animation_state_machine.h"
#include "../persistence/incremental_saver.h"
#include "../resources/color_grading_resource.h"
#include <cstdint>
#include <cstring>

namespace GaussianSplatting {

/**
 * @struct PainterlyMetadata
 * @brief Per-splat metadata for painterly rendering effects.
 *
 * Stores parameters that control stylized rendering such as brush stroke
 * appearance, temporal stability, and noise-based variation.
 */
struct PainterlyMetadata {
    Vector2 jitter;
    Vector2 blue_noise;
    float stroke_scale;
    float stroke_angle;
    uint32_t temporal_seed;
    float stability;

    PainterlyMetadata()
        : jitter(Vector2()),
          blue_noise(Vector2()),
          stroke_scale(1.0f),
          stroke_angle(0.0f),
          temporal_seed(0),
          stability(1.0f) {}
};

/**
 * @struct GaussianData
 * @brief CPU-side representation of a single Gaussian splat.
 *
 * This structure holds the full state of a Gaussian splat for CPU manipulation
 * and editing. It is used during asset loading, editor painting, and animation.
 * For GPU rendering, data is packed into the Gaussian struct which has stricter
 * alignment requirements.
 *
 * @note The covariance array stores the upper triangle of the 3x3 covariance
 *       matrix in row-major order: [xx, xy, xz, yy, yz, zz]. This is used for
 *       CPU-side operations only; the GPU Gaussian struct computes covariance
 *       from rotation and scale at runtime rather than storing it directly.
 */
struct GaussianData {
    Vector3 position;       ///< World-space center position.
    Color color;            ///< RGBA color with alpha as opacity.
    Quaternion rotation;    ///< Orientation quaternion.
    Vector3 scale;          ///< Per-axis scale factors.
    Vector3 normal;         ///< Surface normal for 2D/surfel mode.
    float area;             ///< Projected area hint for culling.
    float covariance[6] = {};    ///< Upper-triangle 3D covariance matrix.
    uint32_t index;         ///< Original index in source data.
    float importance;       ///< LOD importance weight (0-1).
    PainterlyMetadata painterly; ///< Painterly rendering metadata.

    GaussianData()
        : position(Vector3()),
          color(Color(1, 1, 1, 1)),
          rotation(Quaternion()),
          scale(Vector3(1, 1, 1)),
          normal(Vector3(0, 1, 0)),
          area(1.0f),
          index(0),
          importance(1.0f) {
    }

    /**
     * @brief Computes the bounding radius of this Gaussian.
     * @return Radius encompassing approximately 99.7% of the Gaussian distribution (3 sigma).
     *
     * Uses the maximum absolute scale component to compute a conservative bounding
     * sphere radius for frustum culling and spatial queries.
     */
    float compute_radius() const {
        // Use absolute values to handle negative scales
        float max_scale = MAX(MAX(Math::abs(scale.x), Math::abs(scale.y)), Math::abs(scale.z));
        if (max_scale <= 0.0f) {
            max_scale = 1.0f;
        }
        return max_scale * 3.0f;
    }
};

} // namespace GaussianSplatting

/**
 * @struct Gaussian
 * @brief GPU-aligned structure for efficient splat rendering.
 *
 * This struct is designed for direct GPU upload with std430 layout compatibility.
 * It packs all per-splat data needed by the tile-based rasterizer including
 * position, orientation, spherical harmonics, and painterly extensions.
 *
 * @note Must remain 16-byte aligned (144 bytes total). Do not reorder fields
 *       without updating the corresponding GLSL struct definition.
 */
struct Gaussian {
    Vector3 position;
    float opacity;

    Vector3 scale;
    float area;

    Quaternion rotation;

    // Spherical harmonics coefficients
    Color sh_dc;        // DC term
    Vector3 sh_1[3];    // First-order SH

    // 2D Gaussian support (surfels)
    Vector3 normal;
    float stroke_age;

    // Padding for alignment
    float _padding; // Ensures brush_axes is 8-byte aligned

    // Painterly rendering extensions (packed for GPU alignment)
    Vector2 brush_axes;
    uint32_t painterly_meta; // lower 16 bits: palette id, upper 16 bits: painterly flags / brush override ids

    // Final padding to reach 144 bytes (16-byte aligned)
    float _padding2[3]; // 12 bytes to reach 144 total
};

static_assert(sizeof(Gaussian) % 16 == 0, "Gaussian struct must remain 16-byte aligned for GPU uploads");
static_assert(sizeof(Gaussian) == 144, "Gaussian struct must be exactly 144 bytes for GPU compatibility");
static_assert(offsetof(Gaussian, brush_axes) % 8 == 0, "Gaussian::brush_axes must stay 8-byte aligned for std430 layout");

/**
 * @brief Packs palette ID and painterly flags into a single uint32.
 * @param palette_id 16-bit palette index for color lookup.
 * @param flags 16-bit bitfield for painterly rendering options or brush override IDs.
 * @return Packed metadata value for Gaussian::painterly_meta.
 */
constexpr uint32_t gaussian_pack_painterly_meta(uint16_t palette_id, uint16_t flags = 0) {
    return (uint32_t(flags) << 16) | uint32_t(palette_id);
}

/**
 * @brief Updates the palette ID in a packed painterly metadata value.
 * @param meta Existing packed metadata.
 * @param palette_id New 16-bit palette index.
 * @return Updated metadata with new palette ID.
 */
constexpr uint32_t gaussian_set_palette_id(uint32_t meta, uint16_t palette_id) {
    return (meta & 0xFFFF0000u) | uint32_t(palette_id);
}

/**
 * @brief Updates the painterly flags in a packed metadata value.
 * @param meta Existing packed metadata.
 * @param flags New 16-bit painterly flags bitfield or brush override ID.
 * @return Updated metadata with new flags.
 */
constexpr uint32_t gaussian_set_painterly_flags(uint32_t meta, uint16_t flags) {
    return (meta & 0x0000FFFFu) | (uint32_t(flags) << 16);
}

/**
 * @brief Extracts the palette ID from a packed painterly metadata value.
 * @param meta Packed metadata from Gaussian::painterly_meta.
 * @return 16-bit palette index.
 */
constexpr uint16_t gaussian_get_palette_id(uint32_t meta) {
    return uint16_t(meta & 0xFFFFu);
}

/**
 * @brief Extracts the painterly flags from a packed metadata value.
 * @param meta Packed metadata from Gaussian::painterly_meta.
 * @return 16-bit painterly flags bitfield or brush override ID.
 */
constexpr uint16_t gaussian_get_painterly_flags(uint32_t meta) {
    return uint16_t((meta >> 16) & 0xFFFFu);
}

/**
 * @brief Updates the brush override ID stored in the painterly flags lane.
 * @param meta Existing packed metadata.
 * @param override_id New 16-bit brush override ID.
 * @return Updated metadata with new brush override ID.
 */
constexpr uint32_t gaussian_set_brush_override_id(uint32_t meta, uint16_t override_id) {
    return gaussian_set_painterly_flags(meta, override_id);
}

/**
 * @brief Extracts the brush override ID stored in the painterly flags lane.
 * @param meta Packed metadata from Gaussian::painterly_meta.
 * @return 16-bit brush override ID.
 */
constexpr uint16_t gaussian_get_brush_override_id(uint32_t meta) {
    return gaussian_get_painterly_flags(meta);
}

/**
 * @class GaussianData
 * @brief Resource class for storing and managing Gaussian splat data.
 *
 * GaussianData is the primary data container for Gaussian Splatting assets in Godot.
 * It manages the storage, loading, and GPU upload of splat data, and provides
 * spatial query acceleration via an internal octree structure.
 *
 * Key features:
 * - Load/save PLY and SPLAT file formats
 * - Spherical harmonics support (degree 0-3)
 * - Runtime modification overlay for non-destructive editing
 * - Hot render payload separated from cold editor/runtime overlay metadata
 * - Animation system integration for animated splats
 * - Octree-based spatial queries for frustum culling
 *
 * @note Thread safety: Gaussian payload, SH metadata, and brush history mutations
 *       are serialized by `data_rwlock` (RWLock — readers may run concurrently,
 *       writers are exclusive). Animation caches are protected by a separate
 *       `animation_cache_mutex`. Async streaming pack jobs must read chunk data
 *       through capture_chunk_snapshot() only, which captures under the read lock.
 */
class GaussianData : public Resource {
    GDCLASS(GaussianData, Resource);

private:
    LocalVector<Gaussian> gaussians;
    uint32_t sh_degree = 0;
    uint32_t sh_first_order_count = 0;
    uint32_t sh_high_order_count = 0;
    uint32_t sh_high_order_capacity = 0; // Slack-factored capacity to avoid reallocation thrashing
    LocalVector<Vector3> sh_high_order_coefficients;
    mutable RWLock data_rwlock;
    bool is_2d_mode = false;

    struct BrushStroke {
        Vector3 center;
        float radius = 0.0f;
        Color color;
        float opacity = 1.0f;
        float hardness = 1.0f;
        uint64_t timestamp_us = 0;
    };

    // Cold/editor-side state intentionally split from hot render payload.
    struct EditMetadataState {
        LocalVector<Vector3> runtime_positions;      // Modified positions overlay
        LocalVector<Color> runtime_colors;           // Modified colors overlay
        LocalVector<float> runtime_opacities;        // Modified opacities overlay
        LocalVector<bool> runtime_position_flags;    // Tracks which positions were overridden
        LocalVector<bool> runtime_color_flags;       // Tracks which colors were overridden
        LocalVector<bool> runtime_opacity_flags;     // Tracks which opacities were overridden
        LocalVector<bool> modified_flags;            // Per-splat modification tracking
        bool has_runtime_modifications = false;      // Global modification flag
        Vector<BrushStroke> recorded_brush_strokes;
    };
    EditMetadataState edit_state;

    static Dictionary _brush_stroke_to_dict(const BrushStroke &p_stroke);
    static BrushStroke _brush_stroke_from_dict(const Dictionary &p_dict);

    // Animation system integration (v0.6.0)
    Ref<GaussianSplatting::GaussianAnimationStateMachine> animation_state_machine;
    Ref<GaussianSplatting::GaussianIncrementalSaver> incremental_saver;
    bool animation_enabled = true;
    mutable Mutex animation_cache_mutex; ///< Protects all animation cache fields below.
    mutable LocalVector<Vector3> animated_positions_cache;
    mutable LocalVector<Color> animated_colors_cache;
    mutable LocalVector<float> animated_opacities_cache;
    mutable LocalVector<bool> animated_positions_valid_cache;
    mutable LocalVector<bool> animated_colors_valid_cache;
    mutable LocalVector<bool> animated_opacities_valid_cache;
    mutable float last_animation_time = -1.0f;
    mutable bool animation_cache_dirty = true;

    // Cached RenderingDevice singleton to prevent recreation
    static RenderingDevice *cached_rd;

    // Color grading baking system
    struct ColorGradingBakeInfo {
        bool is_baked = false;
        Ref<class ColorGradingResource> applied_grading;
        LocalVector<Color> original_sh_dc;  // Backup of DC coefficients
    };
    ColorGradingBakeInfo bake_info;

    Color apply_color_grading_cpu(const Color &p_color, const Ref<class ColorGradingResource> &p_grading);

    // Spatial acceleration structure
    struct OctreeNode {
        AABB bounds;
        LocalVector<uint32_t> indices;
        uint32_t children[8];  // Changed from uint8_t to support >255 nodes
        uint8_t level;
    };
    LocalVector<OctreeNode> octree;
    mutable bool octree_dirty = true;

    // Private octree helper
    void _subdivide_octree_node(uint32_t node_idx, int max_depth, uint32_t min_gaussians = 32);
    void _on_gaussian_storage_changed();
    void _on_gaussian_storage_changed_locked();
    void _set_spherical_harmonics_locked(int p_index, const float *p_coeffs, int p_count);
    bool _clear_runtime_modifications_locked();
    void _clear_brush_strokes_locked();
    void _set_runtime_position_locked(int p_idx, const Vector3& p_pos);
    void _set_runtime_color_locked(int p_idx, const Color& p_col);
    void _set_runtime_opacity_locked(int p_idx, float p_opacity);
    static bool _is_finite_and_bounded(float p_value, float p_abs_max);
    static bool _is_finite_vector2(const Vector2 &p_value, float p_abs_max);
    static bool _is_finite_vector3(const Vector3 &p_value, float p_abs_max);
    static bool _is_finite_quaternion(const Quaternion &p_value, float p_abs_max);
    static bool _is_finite_color(const Color &p_value, float p_abs_max);
    bool _validate_gpu_payload_locked(String *r_error_message = nullptr) const;

protected:
    static void _bind_methods();

public:
    GaussianData();
    ~GaussianData();

    /// @name Core Data Management
    /// @{

    /**
     * @brief Resizes the internal Gaussian storage.
     * @param p_count New number of Gaussians to allocate.
     *
     * @warning All entries are default-initialized. Existing data is NOT preserved.
     */
    void resize(int p_count);

    /**
     * @brief Replaces all Gaussian data.
     * @param p_gaussians Source data to copy.
     */
    void set_gaussians(const LocalVector<Gaussian> &p_gaussians);

    /// @overload
    void set_gaussians(const Vector<Gaussian> &p_gaussians);

    /**
     * @brief Replaces Gaussian payload and SH metadata in one bulk operation.
     * @param p_gaussians Source Gaussian storage.
     * @param p_sh_high_order_coefficients High-order SH coefficients (flattened per splat).
     * @param p_sh_first_order_count Number of first-order SH vectors per splat (0-3).
     * @param p_sh_high_order_count Number of high-order SH vectors per splat.
     * @param p_is_2d_mode Whether the dataset should be flagged as 2D mode.
     *
     * This bypasses per-splat SH setter loops and is intended for large pre-baked
     * payloads (for example gsplatworld cache loads).
     */
    void set_gaussian_payload(const LocalVector<Gaussian> &p_gaussians,
            const LocalVector<Vector3> &p_sh_high_order_coefficients,
            uint32_t p_sh_first_order_count,
            uint32_t p_sh_high_order_count,
            bool p_is_2d_mode);

    /**
     * @brief Sets a single Gaussian at the specified index.
     * @param p_index Zero-based index (must be < get_count()).
     * @param p_gaussian Data to assign.
     */
    void set_gaussian(int p_index, const Gaussian &p_gaussian);

    /**
     * @brief Retrieves a copy of a Gaussian at the specified index.
     * @param p_index Zero-based index (must be < get_count()).
     * @return Copy of the Gaussian data.
     */
    Gaussian get_gaussian(int p_index) const;

    /**
     * @brief Returns a pointer to the raw Gaussian array.
     * @return Pointer to contiguous Gaussian storage, or nullptr if empty.
     */
    const Gaussian *get_gaussians() const;

    /// @}

    /// @name Batch Operations
    /// @brief High-performance bulk setters for updating multiple properties at once.
    /// @{

    /**
     * @brief Sets positions for all Gaussians.
     * @param p_positions Array of 3D positions (must match get_count()).
     */
    void set_positions(const PackedVector3Array &p_positions);

    /**
     * @brief Sets scales for all Gaussians.
     * @param p_scales Array of per-axis scale vectors (must match get_count()).
     */
    void set_scales(const PackedVector3Array &p_scales);

    /**
     * @brief Sets rotations for all Gaussians.
     * @param p_rotations Array of orientation quaternions (must match get_count()).
     */
    void set_rotations(const TypedArray<Quaternion> &p_rotations);

    /**
     * @brief Sets opacities for all Gaussians.
     * @param p_opacities Array of opacity values in [0, 1] (must match get_count()).
     */
    void set_opacities(const PackedFloat32Array &p_opacities);

    /**
     * @brief Sets spherical harmonics coefficients for all Gaussians.
     * @param p_sh_data Packed float array with SH data for all splats.
     *
     * The array layout depends on the SH degree being used (set via the file loader).
     */
    void set_spherical_harmonics(const PackedFloat32Array &p_sh_data);

    /// @overload Set SH for a single Gaussian.
    void set_spherical_harmonics(int p_index, const float *p_coeffs, int p_count);

    /**
     * @brief Sets palette IDs for painterly rendering.
     * @param p_palette_ids Array of palette indices (must match get_count()).
     */
    void set_palette_ids(const PackedInt32Array &p_palette_ids);

    /**
     * @brief Sets painterly flags for stylized rendering.
     * @param p_flags Array of flag bitfields (must match get_count()).
     */
    void set_painterly_flags(const PackedInt32Array &p_flags);

    /**
     * @brief Gets brush override IDs stored in the painterly flags lane.
     * @return Array of 16-bit brush override IDs, one per splat.
     */
    PackedInt32Array get_brush_override_ids() const;

    /**
     * @brief Gets sanitized brush override IDs stored in the painterly flags lane.
     * @return Array of clamped 16-bit brush override IDs, one per splat.
     */
    PackedInt32Array get_brush_override_ids_buffer() const { return get_brush_override_ids(); }

    /**
     * @brief Sets brush override IDs using the painterly flags lane.
     * @param p_override_ids Array of override IDs (must match get_count()).
     */
    void set_brush_override_ids(const PackedInt32Array &p_override_ids);

    /**
     * @brief Sets brush axis vectors for painterly stroke orientation.
     * @param p_brush_axes Array of 2D axis vectors (must match get_count()).
     */
    void set_brush_axes(const PackedVector2Array &p_brush_axes);

    /**
     * @brief Sets stroke ages for painterly rendering animation.
     * @param p_stroke_ages Array of age values (must match get_count()).
     */
    void set_stroke_ages(const PackedFloat32Array &p_stroke_ages);

    /// @}

    /// @name 2D Gaussian (Surfel) Support
    /// @{

    /**
     * @brief Enables or disables 2D Gaussian (surfel) mode.
     * @param p_enabled When true, splats use normals for disc-like rendering.
     */
    void set_2d_mode(bool p_enabled);
    bool get_2d_mode() const { return is_2d_mode; }

    /**
     * @brief Sets surface normals for 2D Gaussian (surfel) rendering.
     * @param p_normals Array of unit normal vectors (must match get_count()).
     */
    void set_normals(const PackedVector3Array &p_normals);

    /// @}

    /// @name File I/O
    /// @{

    /**
     * @brief Loads Gaussian data from a PLY or SPLAT file.
     * @param p_path Absolute or res:// path to the file.
     * @return OK on success, or an error code.
     */
    Error load_from_file(const String &p_path);
    Error populate_from_asset(const Ref<class GaussianSplatAsset> &p_asset);

    /**
     * @brief Saves current Gaussian data to a PLY file.
     * @param p_path Destination path.
     * @return OK on success, or an error code.
     */
    Error save_to_file(const String &p_path) const;

    /// @}

    /// @name Spatial Queries
    /// @{

    /**
     * @brief Builds or rebuilds the internal octree for spatial queries.
     * @param p_max_depth Maximum octree subdivision depth (default 8).
     * @param p_min_gaussians Minimum splats per node before subdivision stops.
     */
    void build_octree(int p_max_depth = 8, uint32_t p_min_gaussians = 32);

    /**
     * @brief Queries the octree for Gaussians intersecting the given bounds.
     * @param p_bounds Axis-aligned bounding box to query.
     * @return Array of Gaussian indices that may intersect the bounds.
     */
    TypedArray<int> query_octree(const AABB &p_bounds) const;

    /**
     * @brief Gathers indices of Gaussians inside a frustum.
     * @param p_planes Frustum planes (typically 6 planes from a camera projection).
     * @param[out] r_indices Output vector receiving indices of visible Gaussians.
     */
    void gather_frustum_indices(const Vector<Plane> &p_planes, LocalVector<uint32_t> &r_indices) const;

    /** @brief Direct access to internal storage for performance-critical code. */
    const LocalVector<Gaussian> &get_gaussian_storage() const { return gaussians; }

    /**
     * @brief Captures a coherent chunk snapshot for async pack jobs.
     *
     * This is the only supported async pack read path. The snapshot captures
     * Gaussian payload and SH data under a single lock so worker threads never
     * read mutable storage directly.
     *
     * @param p_start First Gaussian index in the chunk.
     * @param p_count Number of Gaussians to capture.
     * @param[out] r_gaussians Snapshot of Gaussian payload.
     * @param[out] r_sh_high_order Snapshot of high-order SH coefficients for the chunk.
     * @param[out] r_sh_first_order_count First-order SH coefficient count.
     * @param[out] r_sh_high_order_count High-order SH coefficient count.
     * @return True when a valid snapshot was captured.
     */
    bool capture_chunk_snapshot(uint32_t p_start, uint32_t p_count,
            LocalVector<Gaussian> &r_gaussians,
            LocalVector<Vector3> &r_sh_high_order,
            uint32_t &r_sh_first_order_count,
            uint32_t &r_sh_high_order_count) const;

    bool capture_indexed_chunk_snapshot(const uint32_t *p_indices, uint32_t p_count,
            LocalVector<Gaussian> &r_gaussians,
            LocalVector<Vector3> &r_sh_high_order,
            uint32_t &r_sh_first_order_count,
            uint32_t &r_sh_high_order_count) const;

    /// @}

    /// @name Spherical Harmonics
    /// @{

    /** @brief Returns the spherical harmonics degree (0-3). */
    uint32_t get_sh_degree() const { return sh_degree; }

    /** @brief Returns the count of first-order SH coefficients per Gaussian. */
    uint32_t get_sh_first_order_count() const { return sh_first_order_count; }

    /** @brief Returns the count of higher-order SH coefficients per Gaussian. */
    uint32_t get_sh_high_order_count() const { return sh_high_order_count; }

    /**
     * @brief Returns a pointer to the high-order SH coefficient storage.
     * @return Pointer to Vector3 array, or nullptr if no high-order SH data.
     */
    const Vector3 *get_sh_high_order_coefficients_ptr() const;

    /**
     * @brief Retrieves all SH coefficients for a single Gaussian.
     * @param p_index Gaussian index (must be < get_count()).
     * @return Packed array of SH coefficients.
     */
    PackedFloat32Array get_spherical_harmonics(int p_index) const;

    /** @brief Returns true if higher-order SH coefficients are available. */
    bool has_full_sh() const;

    /// @}

    /// @name Statistics
    /// @{

    /** @brief Returns the number of Gaussians in this resource. */
    int get_count() const { return gaussians.size(); }

    /** @brief Returns the axis-aligned bounding box of all Gaussians. */
    AABB get_aabb() const;

    /** @brief Recomputes and returns the AABB (use when data has changed). */
    AABB compute_aabb() const;

    /** @brief Estimates memory usage in bytes. */
    float get_memory_usage() const;

    /// @}

    /// @name GPU Buffer Management
    /// @{

    /**
     * @brief Creates a GPU buffer containing all Gaussian data.
     * @param p_rd RenderingDevice to use, or nullptr for the default device.
     * @return RID of the created storage buffer.
     *
     * @note The caller is responsible for freeing the returned buffer.
     */
    RID create_gpu_buffer(RenderingDevice *p_rd = nullptr) const;

    /**
     * @brief Updates an existing GPU buffer with current data.
     * @param p_buffer Buffer RID previously created with create_gpu_buffer().
     * @param p_rd RenderingDevice to use, or nullptr for the default device.
     */
    void update_gpu_buffer(RID p_buffer, RenderingDevice *p_rd = nullptr) const;

    /**
     * @brief Validates that the current payload is finite and within GPU-safe ranges.
     * @param r_error_message Optional detailed reason when validation fails.
     * @return OK if payload is upload-safe, ERR_INVALID_DATA otherwise.
     */
    Error validate_gpu_payload(String *r_error_message = nullptr) const;

    /// @}

    /// @name Runtime Modification API
    /// @brief Non-destructive overlay system for editor painting and live editing.
    /// @{

    /**
     * @brief Sets a temporary position override for a Gaussian.
     * @param p_idx Gaussian index.
     * @param p_pos New position (applied as overlay, not committed to base data).
     */
    void set_runtime_position(int p_idx, const Vector3& p_pos);
    /**
     * @brief Sets a temporary color override for a Gaussian.
     * @param p_idx Gaussian index.
     * @param p_col New color (applied as overlay, not committed to base data).
     */
    void set_runtime_color(int p_idx, const Color& p_col);

    /**
     * @brief Sets a temporary opacity override for a Gaussian.
     * @param p_idx Gaussian index.
     * @param p_opacity New opacity in [0, 1] (applied as overlay).
     */
    void set_runtime_opacity(int p_idx, float p_opacity);

    /**
     * @brief Applies a color to a range of Gaussians.
     * @param p_start Starting index.
     * @param p_count Number of Gaussians to modify.
     * @param p_col Color to apply.
     */
    void apply_color_range(int p_start, int p_count, const Color& p_col);

    /**
     * @brief Marks a range of Gaussians as requiring GPU re-upload.
     * @param p_start Starting index.
     * @param p_count Number of Gaussians to mark dirty.
     */
    void mark_range_dirty(int p_start, int p_count);

    /** @brief Commits all runtime modifications to the base Gaussian data. */
    void commit_runtime_changes();

    /** @brief Discards all runtime modifications, reverting to base data. */
    void revert_runtime_changes();

    /**
     * @brief Applies a paint brush stroke to Gaussians within a radius.
     * @param p_center World-space center of the brush.
     * @param p_radius Brush radius in world units.
     * @param p_color Stroke color.
     * @param p_opacity Stroke opacity in [0, 1].
     * @param p_hardness Edge hardness in [0, 1] (1 = hard edge, 0 = soft falloff).
     */
    void apply_brush_stroke(const Vector3 &p_center, float p_radius, const Color &p_color, float p_opacity, float p_hardness);

    /**
     * @brief Returns all recorded brush strokes as an Array of Dictionaries.
     * @return Array of brush stroke data for serialization/undo.
     */
    Array get_brush_strokes() const;

    /** @brief Clears all recorded brush strokes. */
    void clear_brush_strokes();

    /**
     * @brief Restores brush strokes from a previously saved Array.
     * @param p_strokes Array of brush stroke Dictionaries.
     */
    void set_brush_strokes(const Array &p_strokes);

    /**
     * @brief Captures the current state of Gaussians within a brush radius for undo.
     * @param p_center World-space center of the brush.
     * @param p_radius Brush radius in world units.
     * @return Dictionary with keys: "indices", "colors", "opacities" for affected splats.
     */
    Dictionary capture_brush_affected_state(const Vector3 &p_center, float p_radius) const;

    /**
     * @brief Restores previously captured brush stroke state (for undo).
     * @param p_saved_state Dictionary produced by capture_brush_affected_state().
     */
    void restore_brush_stroke(const Dictionary &p_saved_state);

    /** @brief Returns true if there are uncommitted runtime modifications. */
    bool has_modifications() const { return edit_state.has_runtime_modifications; }

    /// @}

    /// @name Animation System
    /// @brief Integration with GaussianAnimationStateMachine for animated splats.
    /// @{

    /**
     * @brief Sets the animation state machine for animated splats.
     * @param p_animation Animation state machine resource.
     */
    void set_animation_state_machine(const Ref<GaussianSplatting::GaussianAnimationStateMachine>& p_animation);

    /** @brief Returns the current animation state machine. */
    Ref<GaussianSplatting::GaussianAnimationStateMachine> get_animation_state_machine() const { return animation_state_machine; }

    /** @brief Returns true if an animation state machine is assigned. */
    bool has_animation() const { return animation_state_machine.is_valid(); }

    /**
     * @brief Sets the incremental saver for progressive save operations.
     * @param p_saver Incremental saver resource.
     */
    void set_incremental_saver(const Ref<GaussianSplatting::GaussianIncrementalSaver>& p_saver);

    /** @brief Returns the current incremental saver. */
    Ref<GaussianSplatting::GaussianIncrementalSaver> get_incremental_saver() const { return incremental_saver; }

    /**
     * @brief Updates animation state by the given delta time.
     * @param p_delta Time in seconds since the last update.
     */
    void update_animation(float p_delta);

    /**
     * @brief Applies animation state at a specific time.
     * @param p_time Absolute animation time in seconds.
     */
    void apply_animation_at_time(float p_time);

    /** @brief Enables or disables animation playback. */
    void set_animation_enabled(bool p_enabled) { animation_enabled = p_enabled; }

    /** @brief Returns true if animation playback is enabled. */
    bool is_animation_enabled() const { return animation_enabled; }

    /**
     * @brief Gets the animated position for a Gaussian.
     * @param p_index Gaussian index.
     * @param p_time Animation time, or -1 to use the current cached time.
     * @return Animated position, or base position if no animation is active.
     */
    Vector3 get_animated_position(int p_index, float p_time = -1.0f) const;
    /**
     * @brief Gets the animated color for a Gaussian.
     * @param p_index Gaussian index.
     * @param p_time Animation time, or -1 to use the current cached time.
     * @return Animated color, or base color if no animation is active.
     */
    Color get_animated_color(int p_index, float p_time = -1.0f) const;

    /**
     * @brief Gets the animated opacity for a Gaussian.
     * @param p_index Gaussian index.
     * @param p_time Animation time, or -1 to use the current cached time.
     * @return Animated opacity, or base opacity if no animation is active.
     */
    float get_animated_opacity(int p_index, float p_time = -1.0f) const;

    /**
     * @brief Gets the animated scale for a Gaussian.
     * @param p_index Gaussian index.
     * @param p_time Animation time, or -1 to use the current cached time.
     * @return Animated scale, or base scale if no animation is active.
     */
    Vector3 get_animated_scale(int p_index, float p_time = -1.0f) const;

    /**
     * @brief Gets the animated rotation for a Gaussian.
     * @param p_index Gaussian index.
     * @param p_time Animation time, or -1 to use the current cached time.
     * @return Animated rotation, or base rotation if no animation is active.
     */
    Quaternion get_animated_rotation(int p_index, float p_time = -1.0f) const;

    /// @}

    /// @name Color Grading Baking
    /// @{

    /**
     * @brief Bakes color grading into the Gaussian SH DC coefficients.
     * @param p_grading Color grading settings to bake.
     * @return OK on success, error code on failure.
     *
     * This permanently modifies the base color (SH DC coefficients) of each Gaussian
     * by applying the specified color grading. The original colors are backed up
     * and can be restored via restore_original_colors().
     *
     * @note Baking color grading has zero runtime cost - the modified colors are
     *       "frozen" into the splat data. Real-time color grading in the shader
     *       has a small cost (~0.2ms for 1M splats).
     */
    Error bake_color_grading(const Ref<class ColorGradingResource> &p_grading);

    /**
     * @brief Restores the original colors before any color grading was baked.
     *
     * This reverts all Gaussian SH DC coefficients to their state before the first
     * bake_color_grading() call. Does nothing if no baking has been applied.
     */
    void restore_original_colors();

    /** @brief Returns true if color grading has been baked into the data. */
    bool is_color_grading_baked() const { return bake_info.is_baked; }

    /** @brief Returns the color grading settings that were baked, or null if none. */
    Ref<class ColorGradingResource> get_baked_grading() const { return bake_info.applied_grading; }

    /// @}

    /**
     * @brief Returns a cached RenderingDevice singleton.
     * @return RenderingDevice pointer, or nullptr if unavailable.
     */
    static RenderingDevice* get_rendering_device();
};

#endif // GAUSSIAN_DATA_H
