#ifndef GS_RENDERER_INTERFACES_H
#define GS_RENDERER_INTERFACES_H

/**
 * @file renderer_interfaces.h
 * @brief Segregated interfaces for Gaussian Splat rendering.
 *
 * Provides dependency inversion for renderer consumers (nodes, containers).
 * The monolithic IRenderer is split into focused role interfaces following
 * the Interface Segregation Principle — consumers should depend only on the
 * methods they actually use.
 *
 * Sub-interfaces:
 *   IRendererLifecycle — init, shutdown, data binding
 *   IRendererConfig    — camera, LOD, painterly, and global settings
 *   IRendererDebug     — debug overlays, stats, diagnostics
 *   IRendererPipeline  — render_for_view, get_output, queries
 *
 * IRenderer inherits all four for backward compatibility.
 */

#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"

// Forward declarations
class GaussianData;
class GaussianSplatAsset;
class GaussianStreamingSystem;
class ColorGradingResource;
struct StaticChunk;

// =============================================================================
// IRendererLifecycle — initialization, shutdown, data binding
// =============================================================================

/**
 * @class IRendererLifecycle
 * @brief Lifecycle and data-binding surface.
 *
 * Consumers that only need low-level setup or data binding (for example tests,
 * focused tools, editor preview helpers, or import/runtime plumbing) depend on
 * this interface alone. High-level scene workflows should prefer the
 * node/director submission paths instead of binding raw data here directly.
 */
class IRendererLifecycle {
public:
	virtual ~IRendererLifecycle() = default;

	/// Initialize GPU resources. Must be called before rendering.
	virtual void initialize() = 0;

	/// Bind raw GaussianData directly. Low-level API, not the canonical scene path.
	virtual Error set_gaussian_data(const Ref<GaussianData> &p_data) = 0;

	/// Get the current Gaussian data.
	virtual Ref<GaussianData> get_gaussian_data() const = 0;

	/// Set the Gaussian asset (preferred asset-backed alternative to raw data binding).
	virtual void set_gaussian_asset(const Ref<GaussianSplatAsset> &p_asset) = 0;

	/// Set static chunk data for world rendering.
	virtual void set_static_chunks(const Vector<StaticChunk> &p_chunks) = 0;
};

// =============================================================================
// IRendererConfig — camera state, LOD, painterly, and global settings
// =============================================================================

/**
 * @class IRendererConfig
 * @brief Configuration surface for camera, LOD, painterly, and global settings.
 *
 * Consumers that configure the renderer (e.g. property panels, nodes that
 * forward inspector properties) depend on this interface.
 */
class IRendererConfig {
public:
	virtual ~IRendererConfig() = default;

	// -- Camera State ---------------------------------------------------------

	/// Set camera world transform for view-dependent rendering.
	virtual void set_camera_transform(const Transform3D &p_transform) = 0;

	/// Get current camera transform.
	virtual Transform3D get_camera_transform() const = 0;

	/// Set camera projection matrix.
	virtual void set_camera_projection(const Projection &p_projection) = 0;

	// -- LOD & Culling --------------------------------------------------------

	virtual void set_lod_enabled(bool p_enabled) = 0;
	virtual bool get_lod_enabled() const = 0;

	virtual void set_lod_bias(float p_bias) = 0;
	virtual float get_lod_bias() const = 0;

	virtual void set_lod_max_distance(float p_distance) = 0;
	virtual float get_lod_max_distance() const = 0;

	virtual void set_frustum_culling(bool p_enabled) = 0;
	virtual bool get_frustum_culling() const = 0;

	virtual void set_max_splats(int p_count) = 0;
	virtual int get_max_splats() const = 0;

	virtual void set_async_upload_enabled(bool p_enabled) = 0;
	virtual bool get_async_upload_enabled() const = 0;

	// -- Painterly Configuration ----------------------------------------------

	virtual void set_painterly_enabled(bool p_enabled) = 0;
	virtual bool get_painterly_enabled() const = 0;

	virtual void set_painterly_edge_threshold(float p_threshold) = 0;
	virtual float get_painterly_edge_threshold() const = 0;

	virtual void set_painterly_stroke_opacity(float p_opacity) = 0;
	virtual float get_painterly_stroke_opacity() const = 0;

	virtual void set_painterly_stroke_length(float p_length) = 0;
	virtual float get_painterly_stroke_length() const = 0;

	virtual void set_painterly_gamma(float p_gamma) = 0;
	virtual float get_painterly_gamma() const = 0;

	// -- Global Configuration -------------------------------------------------

	virtual void set_opacity_multiplier(float p_opacity) = 0;
	virtual float get_opacity_multiplier() const = 0;

	virtual void set_color_grading(const Ref<ColorGradingResource> &p_grading) = 0;
};

// =============================================================================
// IRendererDebug — debug overlays, statistics, diagnostics
// =============================================================================

/**
 * @class IRendererDebug
 * @brief Debug and diagnostics surface.
 *
 * Editor tools and performance HUDs depend on this interface to query
 * statistics and toggle debug overlays without pulling in the full
 * rendering API.
 */
class IRendererDebug {
public:
	virtual ~IRendererDebug() = default;

	// -- Debug Overlay Controls -----------------------------------------------

	virtual void set_debug_show_tile_grid(bool p_enabled) = 0;
	virtual bool is_debug_show_tile_grid() const = 0;

	virtual void set_debug_show_density_heatmap(bool p_enabled) = 0;
	virtual bool is_debug_show_density_heatmap() const = 0;

	virtual void set_debug_show_performance_hud(bool p_enabled) = 0;
	virtual bool is_debug_show_performance_hud() const = 0;

	virtual void set_debug_show_residency_hud(bool p_enabled) = 0;
	virtual bool is_debug_show_residency_hud() const = 0;

	virtual void set_debug_overlay_opacity(float p_opacity) = 0;
	virtual float get_debug_overlay_opacity() const = 0;

	virtual void set_debug_preview_mode_int(int p_mode) = 0;
	virtual int get_debug_preview_mode_int() const = 0;

	// -- Statistics & Queries -------------------------------------------------

	/// Get comprehensive render statistics as a Dictionary.
	virtual Dictionary get_render_stats() const = 0;

	/// Get count of visible splats after culling.
	virtual uint32_t get_visible_splat_count() const = 0;

	/// Check if renderer has valid content to display.
	virtual bool has_rendered_content() const = 0;
};

// =============================================================================
// IRendererPipeline — frame rendering and output retrieval
// =============================================================================

/**
 * @class IRendererPipeline
 * @brief Rendering execution surface.
 *
 * Consumers that drive the actual render loop (viewport integration,
 * standalone render paths) depend on this interface.
 */
class IRendererPipeline {
public:
	virtual ~IRendererPipeline() = default;

	/// Render to a viewport target (standalone rendering path).
	virtual bool render_for_view(
			const Transform3D &p_world_to_camera_transform,
			const Projection &p_cam_projection,
			RID p_render_target,
			const Size2i &p_viewport_size) = 0;

	/// Get the final rendered texture RID.
	virtual RID get_final_texture() const = 0;
};

// =============================================================================
// IRenderer — composite interface (backward-compatible)
// =============================================================================

/**
 * @class IRenderer
 * @brief Composite interface that inherits all focused role interfaces.
 *
 * Existing code that depends on IRenderer continues to work unchanged.
 * New code should prefer the narrowest sub-interface that covers its needs:
 *   - IRendererLifecycle for init/data binding
 *   - IRendererConfig    for camera/LOD/painterly settings
 *   - IRendererDebug     for overlays, stats, diagnostics
 *   - IRendererPipeline  for render_for_view / get_final_texture
 *
 * Direct `set_gaussian_data()` binding is the low-level raw-data path used by
 * tests, tools, editor preview, and internal runtime plumbing. Production scene
 * code usually reaches the renderer through nodes, world submissions, and the
 * scene director instead.
 *
 * ## Usage
 * @code
 * IRenderer *renderer = get_renderer();
 * renderer->initialize();
 * renderer->set_gaussian_data(my_data); // low-level raw-data binding
 * renderer->set_camera_transform(camera_xform);
 * @endcode
 */
class IRenderer : public IRendererLifecycle,
				  public IRendererConfig,
				  public IRendererDebug,
				  public IRendererPipeline {
public:
	~IRenderer() override = default;
};

/**
 * @class IRendererProvider
 * @brief Interface for objects that provide renderer access.
 *
 * Used by systems that need to obtain a renderer instance without
 * knowing the concrete type.
 */
class IRendererProvider {
public:
	virtual ~IRendererProvider() = default;

	/// Get the renderer instance (may be null if not initialized).
	virtual IRenderer *get_renderer() const = 0;

	/// Check if a renderer is available.
	virtual bool has_renderer() const = 0;
};

#endif // GS_RENDERER_INTERFACES_H
