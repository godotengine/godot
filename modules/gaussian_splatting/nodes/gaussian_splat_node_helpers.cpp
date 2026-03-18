#include "gaussian_splat_node_helpers.h"

#include "gaussian_splat_node_3d.h"
#include "../core/gaussian_splat_settings_manager.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../core/gaussian_streaming.h"
#include "../core/quality_tier_config.h"
#include "../io/io_settings_utils.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../renderer/gpu_debug_utils.h"
#include "../logger/gs_logger.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/object/object.h"
#include "core/object/callable_method_pointer.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "../lod/lod_config.h"
#include "scene/3d/camera_3d.h"
#include "scene/main/node.h"
#include "scene/main/viewport.h"
#include "servers/rendering_server.h"

#include <cmath>

namespace {

Mutex g_renderer_settings_owner_mutex;
HashMap<ObjectID, ObjectID> g_renderer_settings_owner_lookup;

static bool _settings_owner_is_live(ObjectID p_owner_id) {
    if (p_owner_id == ObjectID()) {
        return false;
    }
    Object *owner_obj = ObjectDB::get_instance(p_owner_id);
    return Object::cast_to<GaussianSplatNode3D>(owner_obj) != nullptr;
}

static bool _claim_renderer_settings_owner(ObjectID p_renderer_id, ObjectID p_owner_id) {
    if (p_renderer_id == ObjectID() || p_owner_id == ObjectID()) {
        return false;
    }

    MutexLock lock(g_renderer_settings_owner_mutex);
    ObjectID *existing_owner = g_renderer_settings_owner_lookup.getptr(p_renderer_id);
    if (!existing_owner) {
        g_renderer_settings_owner_lookup.insert(p_renderer_id, p_owner_id);
        return true;
    }
    if (*existing_owner == p_owner_id) {
        return true;
    }
    if (!_settings_owner_is_live(*existing_owner)) {
        *existing_owner = p_owner_id;
        return true;
    }
    return false;
}

static void _release_renderer_settings_owner(ObjectID p_renderer_id, ObjectID p_owner_id) {
    if (p_renderer_id == ObjectID()) {
        return;
    }

    MutexLock lock(g_renderer_settings_owner_mutex);
    ObjectID *existing_owner = g_renderer_settings_owner_lookup.getptr(p_renderer_id);
    if (!existing_owner) {
        return;
    }
    if (*existing_owner == p_owner_id || !_settings_owner_is_live(*existing_owner)) {
        g_renderer_settings_owner_lookup.erase(p_renderer_id);
    }
}

static GaussianSplatRenderer::DebugPreviewMode _node_debug_draw_to_renderer_preview_mode(GaussianSplatNode3D::DebugDrawMode p_mode) {
    switch (p_mode) {
        case GaussianSplatNode3D::DEBUG_DRAW_OFF:
            return GaussianSplatRenderer::DEBUG_PREVIEW_OFF;
        case GaussianSplatNode3D::DEBUG_DRAW_WIREFRAME:
            return GaussianSplatRenderer::DEBUG_PREVIEW_WIREFRAME;
        case GaussianSplatNode3D::DEBUG_DRAW_POINTS:
            return GaussianSplatRenderer::DEBUG_PREVIEW_POINTS;
        case GaussianSplatNode3D::DEBUG_DRAW_HEATMAP:
            return GaussianSplatRenderer::DEBUG_PREVIEW_HEATMAP;
        default:
            return GaussianSplatRenderer::DEBUG_PREVIEW_OFF;
    }
}

} // namespace

void GaussianSplatNodeAssetHelper::load_asset() {
    if (GaussianSplatting::is_debug_frame_logging_enabled()) {
        GS_LOG_STREAMING_DEBUG(vformat("[ASSET-HELPER] load_asset helper, ply=%s", owner.ply_file_path));
    }
    if (owner.ply_file_path.is_empty()) {
        if (GaussianSplatting::is_debug_frame_logging_enabled()) {
            GS_LOG_STREAMING_DEBUG(vformat("[ASSET-HELPER] load_asset called, ply_path=%s", owner.ply_file_path));
        }
        return;
    }

    owner.asset_loading = true;

    Ref<GaussianSplatAsset> new_asset;
    new_asset.instantiate();

    Error load_error = new_asset->load_from_file(owner.ply_file_path);
    if (GaussianSplatting::is_debug_frame_logging_enabled()) {
        GS_LOG_STREAMING_DEBUG(vformat("[ASSET-HELPER] load_from_file returned err=%d count=%d",
                (int)load_error, new_asset->get_splat_count()));
    }

    if (load_error == OK && new_asset->get_splat_count() > 0) {
        owner.set_splat_asset(new_asset);
        if (GaussianSplatting::is_debug_frame_logging_enabled()) {
            GS_LOG_STREAMING_DEBUG(vformat("[ASSET-HELPER] Load success, splat_count=%d", new_asset->get_splat_count()));
        }
        owner.emit_signal("asset_loaded");
    } else {
        String error_message = vformat("Failed to load PLY file: %s (Error %d)", owner.ply_file_path, (int)load_error);
        owner.emit_signal("asset_loading_failed", error_message);
    }

    owner.asset_loading = false;
}

void GaussianSplatNodeAssetHelper::update_asset() {
    if (!owner.splat_asset.is_valid()) {
        return;
    }

    owner.total_splat_count = owner.splat_asset->get_splat_count();

    // Read import settings from asset metadata
    Dictionary import_metadata = owner.splat_asset->get_import_metadata();
    if (import_metadata.has(StringName("enable_lod"))) {
        owner.asset_lod_enabled = (bool)import_metadata[StringName("enable_lod")];
    } else {
        owner.asset_lod_enabled = true; // Default to enabled
    }
    if (import_metadata.has(StringName("optimize_for_gpu"))) {
        owner.asset_optimize_for_gpu = (bool)import_metadata[StringName("optimize_for_gpu")];
    } else {
        owner.asset_optimize_for_gpu = true; // Default to enabled
    }

    if (GaussianSplatting::is_debug_resource_logging_enabled()) {
        GS_LOG_STREAMING_DEBUG(vformat("[ASSET-HELPER] update_asset: enable_lod=%s, optimize_for_gpu=%s",
                owner.asset_lod_enabled ? "true" : "false",
                owner.asset_optimize_for_gpu ? "true" : "false"));
    }

    const bool bounds_dirty = import_metadata.has(StringName("bounds_dirty")) && (bool)import_metadata[StringName("bounds_dirty")];

    bool used_cached_bounds = false;
    if (!bounds_dirty && import_metadata.has(StringName("bounds"))) {
        const Variant bounds_var = import_metadata[StringName("bounds")];
        if (bounds_var.get_type() == Variant::AABB) {
            AABB cached_bounds = bounds_var;
            if (cached_bounds.size != Vector3()) {
                owner.local_aabb = cached_bounds;
                used_cached_bounds = true;
            }
        }
    }

    PackedFloat32Array positions = owner.splat_asset->get_positions();
    PackedFloat32Array scales = owner.splat_asset->get_scales();
    bool recomputed_bounds = false;
    if (!used_cached_bounds && positions.size() >= 3) {
        Vector3 min_pos = Vector3(INFINITY, INFINITY, INFINITY);
        Vector3 max_pos = Vector3(-INFINITY, -INFINITY, -INFINITY);

        // Compute max scale extent to inflate bounds conservatively.
        // This prevents early culling of splats with large scales.
        float max_scale_extent = 0.0f;
        const int splat_count = positions.size() / 3;
        const bool has_scales = scales.size() >= splat_count * 3;

        for (int i = 0; i < splat_count; i++) {
            Vector3 pos(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            min_pos = min_pos.min(pos);
            max_pos = max_pos.max(pos);

            if (has_scales) {
                // Conservative: use max component of scale as radius extent.
                float sx = std::abs(scales[i * 3]);
                float sy = std::abs(scales[i * 3 + 1]);
                float sz = std::abs(scales[i * 3 + 2]);
                float local_max = MAX(sx, MAX(sy, sz));
                max_scale_extent = MAX(max_scale_extent, local_max);
            }
        }

        // Inflate bounds by the maximum splat scale (conservative, avoids per-splat rotation).
        // Multiply by a factor to account for Gaussian tails visible beyond 1-sigma.
        const float scale_multiplier = 3.0f; // ~3-sigma covers 99.7% of Gaussian
        Vector3 inflation(max_scale_extent * scale_multiplier, max_scale_extent * scale_multiplier, max_scale_extent * scale_multiplier);
        min_pos -= inflation;
        max_pos += inflation;

        owner.local_aabb = AABB(min_pos, max_pos - min_pos);
        recomputed_bounds = true;
    }

    // If we recomputed bounds from raw buffers, refresh cached metadata and
    // clear bounds_dirty so later asset refreshes can use the fast cached path.
    if (recomputed_bounds && (bounds_dirty || !import_metadata.has(StringName("bounds")))) {
        import_metadata[StringName("bounds")] = owner.local_aabb;
        import_metadata[StringName("bounds_dirty")] = false;
        owner.splat_asset->set_import_metadata(import_metadata);
    }

    double total_bytes = 0.0;
    total_bytes += double(positions.size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_colors().size()) * sizeof(Color);
    total_bytes += double(owner.splat_asset->get_scales().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_rotations().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_sh_dc_coefficients().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_sh_first_order_coefficients().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_sh_high_order_coefficients().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_opacity_logits().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_palette_ids().size()) * sizeof(int32_t);
    total_bytes += double(owner.splat_asset->get_painterly_flags().size()) * sizeof(int32_t);
    total_bytes += double(owner.splat_asset->get_normals().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_brush_axes().size()) * sizeof(float);
    total_bytes += double(owner.splat_asset->get_stroke_ages().size()) * sizeof(float);
    owner.gpu_memory_mb = total_bytes / (1024.0 * 1024.0);

    owner._update_bounds();
    owner._update_quality_settings();
    owner._apply_painterly_settings();
    owner._mark_render_state_dirty();
    owner._ensure_gaussian_base();
    owner._sync_gaussian_storage();
    owner._update_render_instance();
    owner._register_shared_renderer();
}

void GaussianSplatNodeAssetHelper::clear_asset() {
    owner.visible_splat_count = 0;
    owner.total_splat_count = 0;
    owner.gpu_memory_mb = 0.0f;
    owner.asset_lod_enabled = true;
    owner.asset_optimize_for_gpu = true;

    owner.renderer_helper.release_renderer_settings_ownership();
    owner._unregister_shared_renderer();
    owner.renderer_data.unref();
    owner.runtime_asset.unref();
    owner._mark_render_state_dirty();
    owner._apply_renderer_settings();

    if (owner.render_instance.is_valid()) {
        RenderingServer *rs = RS::get_singleton();
        if (rs) {
            rs->instance_set_visible(owner.render_instance, false);
        }
    }

    owner._release_gaussian_base();
}

Viewport *GaussianSplatNodeViewportHelper::find_editor_scene_viewport() const {
    Engine *engine = Engine::get_singleton();
    if (!engine || !engine->is_editor_hint()) {
        return owner.get_viewport();
    }

    Viewport *viewport = owner.get_viewport();
    Viewport *best_with_camera = nullptr;
    int32_t best_area = -1;

    for (Viewport *current = viewport; current; current = Object::cast_to<Viewport>(current->get_parent())) {
        Size2i size = current->get_visible_rect().size;
        const int32_t area = size.x * size.y;
        const bool has_camera = current->get_camera_3d() != nullptr;

#ifdef DEBUG_ENABLED
        String candidate = "[GaussianSplatNode3D] Viewport candidate '" + String(current->get_name()) + "' size=" +
                itos(size.x) + "x" + itos(size.y) + " camera=" + (has_camera ? "true" : "false") +
                " area=" + itos(area);
        GS_LOG_RENDERER_DEBUG(candidate);
#endif

        if (has_camera && area > best_area) {
            best_area = area;
            best_with_camera = current;
        }
    }

    if (best_with_camera) {
#ifdef DEBUG_ENABLED
        Size2i size = best_with_camera->get_visible_rect().size;
        String selected = "[GaussianSplatNode3D] Selected scene viewport '" + String(best_with_camera->get_name()) +
                "' size=" + itos(size.x) + "x" + itos(size.y);
        GS_LOG_RENDERER_DEBUG(selected);
#endif
        return best_with_camera;
    }

    Viewport *largest_viewport = viewport;
    best_area = -1;
    for (Viewport *current = viewport; current; current = Object::cast_to<Viewport>(current->get_parent())) {
        Size2i size = current->get_visible_rect().size;
        const int32_t area = size.x * size.y;
        if (area > best_area) {
            best_area = area;
            largest_viewport = current;
        }
    }

    return largest_viewport ? largest_viewport : viewport;
}

void GaussianSplatNodeViewportHelper::update_cached_render_target(Viewport *p_viewport) {
    if (p_viewport != owner.observed_viewport) {
        disconnect_viewport_observers();

        if (!p_viewport) {
            return;
        }

        connect_viewport_observers(p_viewport);
    } else if (!p_viewport) {
        disconnect_viewport_observers();
        return;
    }

    if (!p_viewport) {
        return;
    }

    ensure_viewport_texture_binding(p_viewport);

    GaussianSplatNode3D::ViewportTextureState previous_state = owner.viewport_texture_state;
    bool ready = acquire_viewport_render_target(p_viewport);

    if (ready) {
#ifdef DEBUG_ENABLED
        if (owner.cached_viewport_render_target.is_valid() &&
                previous_state != GaussianSplatNode3D::ViewportTextureState::READY) {
            GS_LOG_RENDERER_DEBUG(vformat("[GaussianSplatNode3D] Cached viewport render target RID (rid=%s).",
                    itos(owner.cached_viewport_render_target.get_id())));
        }
#endif
        if (previous_state != GaussianSplatNode3D::ViewportTextureState::READY) {
            owner.viewport_texture_missing_reported = false;
            owner._queue_first_frame_render();
        }
    } else {
        if (owner.viewport_texture_state == GaussianSplatNode3D::ViewportTextureState::WAITING_FOR_TEXTURE) {
#ifdef DEBUG_ENABLED
            if (previous_state == GaussianSplatNode3D::ViewportTextureState::READY) {
                GS_LOG_RENDERER_DEBUG("[GaussianSplatNode3D] Viewport render target lost; waiting for recreation.");
            }
#endif
            queue_viewport_bootstrap();
            if (!owner.viewport_texture_missing_reported) {
                GS_LOG_RENDERER_WARN(vformat("[GaussianSplatNode3D] Viewport render target unavailable for node '%s'. Waiting for first frame.",
                        owner.get_name()));
                owner.viewport_texture_missing_reported = true;
            }
        }
    }
}

bool GaussianSplatNodeViewportHelper::acquire_viewport_render_target(Viewport *p_viewport) {
    if (!p_viewport) {
        owner.cached_viewport_id = ObjectID();
        owner.cached_viewport_render_target = RID();
        owner.cached_viewport_render_texture = RID();
        owner.cached_viewport_size = Vector2i();
        owner.viewport_texture_state = GaussianSplatNode3D::ViewportTextureState::INACTIVE;
        owner.viewport_texture_missing_reported = false;
        owner.first_frame_render_deferred = false;
        return false;
    }

    owner.cached_viewport_id = p_viewport->get_instance_id();

    Size2i viewport_size = p_viewport->get_visible_rect().size;
    if (viewport_size.x <= 0 || viewport_size.y <= 0) {
        viewport_size = Size2i(1, 1);
    }
    owner.cached_viewport_size = Vector2i(viewport_size.x, viewport_size.y);

    RID render_target;
    RID render_texture;
    RenderingServer *rs = RS::get_singleton();
    if (rs) {
        RID viewport_rid = p_viewport->get_viewport_rid();
        if (viewport_rid.is_valid()) {
            render_target = rs->viewport_get_render_target(viewport_rid);
            render_texture = rs->viewport_get_texture(viewport_rid);
        }
    }

#ifdef DEBUG_ENABLED
    Node *parent = p_viewport->get_parent();
    String bind_message = "[GaussianSplatNode3D] Binding viewport '" + String(p_viewport->get_name()) + "' (parent='" +
            String(parent ? parent->get_name() : StringName("<null>")) + "') size=" + itos(viewport_size.x) + "x" +
            itos(viewport_size.y) + " camera=" + (p_viewport->get_camera_3d() ? "true" : "false") +
            " target_rid=" + itos(render_target.is_valid() ? int64_t(render_target.get_id()) : int64_t(-1));
    GS_LOG_RENDERER_DEBUG(bind_message);
#endif

    if (!render_target.is_valid() && owner.cached_viewport_texture.is_valid()) {
        render_texture = owner.cached_viewport_texture->get_rid();
    }

    if (render_target.is_valid()) {
        owner.cached_viewport_render_target = render_target;
        owner.cached_viewport_render_texture = render_texture;
        owner.viewport_texture_state = GaussianSplatNode3D::ViewportTextureState::READY;
        owner.viewport_texture_missing_reported = false;
        return true;
    }

    owner.cached_viewport_render_target = RID();
    owner.cached_viewport_render_texture = RID();
    owner.viewport_texture_state = GaussianSplatNode3D::ViewportTextureState::WAITING_FOR_TEXTURE;
    owner.first_frame_render_deferred = false;
    return false;
}

void GaussianSplatNodeViewportHelper::connect_viewport_observers(Viewport *p_viewport) {
    if (!p_viewport) {
        return;
    }

    owner.observed_viewport = p_viewport;
    owner.viewport_texture_state = GaussianSplatNode3D::ViewportTextureState::WAITING_FOR_TEXTURE;
    owner.viewport_texture_missing_reported = false;
    owner.first_frame_render_deferred = false;

    Callable size_changed_callable = callable_mp(&owner, &GaussianSplatNode3D::_on_viewport_size_changed);
    if (!p_viewport->is_connected("size_changed", size_changed_callable)) {
        p_viewport->connect("size_changed", size_changed_callable, Object::CONNECT_REFERENCE_COUNTED);
    }

    Callable exited_callable = callable_mp(&owner, &GaussianSplatNode3D::_on_observed_viewport_exited);
    if (!p_viewport->is_connected(SceneStringName(tree_exited), exited_callable)) {
        p_viewport->connect(SceneStringName(tree_exited), exited_callable, Object::CONNECT_REFERENCE_COUNTED);
    }

    ensure_viewport_texture_binding(p_viewport);
}

void GaussianSplatNodeViewportHelper::disconnect_viewport_observers() {
    if (owner.observed_viewport) {
        Callable size_changed_callable = callable_mp(&owner, &GaussianSplatNode3D::_on_viewport_size_changed);
        if (owner.observed_viewport->is_connected("size_changed", size_changed_callable)) {
            owner.observed_viewport->disconnect("size_changed", size_changed_callable);
        }

        Callable exited_callable = callable_mp(&owner, &GaussianSplatNode3D::_on_observed_viewport_exited);
        if (owner.observed_viewport->is_connected(SceneStringName(tree_exited), exited_callable)) {
            owner.observed_viewport->disconnect(SceneStringName(tree_exited), exited_callable);
        }
    }

    if (owner.cached_viewport_texture.is_valid()) {
        Callable texture_ready_callable = callable_mp(&owner, &GaussianSplatNode3D::_on_viewport_texture_ready);
        if (owner.cached_viewport_texture->is_connected("changed", texture_ready_callable)) {
            owner.cached_viewport_texture->disconnect("changed", texture_ready_callable);
        }
    }

    owner.cached_viewport_texture.unref();
    owner.observed_viewport = nullptr;
    owner.cached_viewport_id = ObjectID();
    owner.cached_viewport_render_target = RID();
    owner.cached_viewport_render_texture = RID();
    owner.cached_viewport_size = Vector2i();
    owner.viewport_texture_state = GaussianSplatNode3D::ViewportTextureState::INACTIVE;
    owner.viewport_bootstrap_deferred = false;
    owner.first_frame_render_deferred = false;
    owner.viewport_texture_missing_reported = false;
}

void GaussianSplatNodeViewportHelper::ensure_viewport_texture_binding(Viewport *p_viewport) {
    Ref<ViewportTexture> viewport_texture = p_viewport ? p_viewport->get_texture() : Ref<ViewportTexture>();
    if (viewport_texture == owner.cached_viewport_texture) {
        return;
    }

    Callable texture_ready_callable = callable_mp(&owner, &GaussianSplatNode3D::_on_viewport_texture_ready);

    if (owner.cached_viewport_texture.is_valid()) {
        if (owner.cached_viewport_texture->is_connected("changed", texture_ready_callable)) {
            owner.cached_viewport_texture->disconnect("changed", texture_ready_callable);
        }
    }

    owner.cached_viewport_texture = viewport_texture;

    if (owner.cached_viewport_texture.is_valid() &&
            !owner.cached_viewport_texture->is_connected("changed", texture_ready_callable)) {
        owner.cached_viewport_texture->connect("changed", texture_ready_callable, Object::CONNECT_REFERENCE_COUNTED);
    }
}

void GaussianSplatNodeViewportHelper::queue_viewport_bootstrap() {
    if (owner.viewport_bootstrap_deferred || !owner.is_inside_tree()) {
        return;
    }

    owner.viewport_bootstrap_deferred = true;
    callable_mp(&owner, &GaussianSplatNode3D::_deferred_viewport_bootstrap).call_deferred();
}

void GaussianSplatNodeViewportHelper::deferred_viewport_bootstrap() {
    if (!owner.is_inside_tree()) {
        owner.viewport_bootstrap_deferred = false;
        return;
    }

    Viewport *viewport = Engine::get_singleton()->is_editor_hint()
            ? find_editor_scene_viewport()
            : owner.get_viewport();
    update_cached_render_target(viewport);

    owner.viewport_bootstrap_deferred = false;
}

void GaussianSplatNodeViewportHelper::on_viewport_texture_ready() {
    if (!owner.is_inside_tree()) {
        return;
    }

    // Guard against double-fire: if the signal fires twice and the first
    // invocation already transitioned the state to READY, the second call
    // is a no-op.  This prevents redundant first-frame renders and avoids
    // re-entrant viewport bootstrap queues.
    if (owner.viewport_texture_state == GaussianSplatNode3D::ViewportTextureState::READY &&
            owner.cached_viewport_render_target.is_valid()) {
        return;
    }

    Viewport *viewport = owner.observed_viewport ? owner.observed_viewport : owner.get_viewport();
    if (!viewport) {
        return;
    }

    ensure_viewport_texture_binding(viewport);

    GaussianSplatNode3D::ViewportTextureState previous_state = owner.viewport_texture_state;
    bool ready = acquire_viewport_render_target(viewport);

    if (ready) {
        if (previous_state != GaussianSplatNode3D::ViewportTextureState::READY) {
            owner._queue_first_frame_render();
        }
    } else {
        queue_viewport_bootstrap();
    }
}

void GaussianSplatNodeViewportHelper::on_viewport_size_changed() {
    if (!owner.is_inside_tree()) {
        return;
    }

    if (!owner.observed_viewport) {
        return;
    }

    Size2i viewport_size = owner.observed_viewport->get_visible_rect().size;
    if (viewport_size.x <= 0 || viewport_size.y <= 0) {
        viewport_size = Size2i(1, 1);
    }

    owner.cached_viewport_size = Vector2i(viewport_size.x, viewport_size.y);
    owner.cached_viewport_render_target = RID();
    owner.cached_viewport_render_texture = RID();
    owner.viewport_texture_state = GaussianSplatNode3D::ViewportTextureState::WAITING_FOR_TEXTURE;
    owner.viewport_texture_missing_reported = false;
    owner.first_frame_render_deferred = false;

    ensure_viewport_texture_binding(owner.observed_viewport);
    queue_viewport_bootstrap();
}

void GaussianSplatNodeViewportHelper::on_observed_viewport_exited() {
    disconnect_viewport_observers();
}

void GaussianSplatNodeDebugHelper::apply_renderer_debug_settings() {
    if (!owner.renderer.is_valid()) {
        return;
    }
    if (!owner.renderer_helper.can_apply_renderer_settings()) {
        return;
    }

    owner.renderer->set_debug_show_tile_grid(owner.show_tile_grid);
    owner.renderer->set_debug_show_density_heatmap(owner.show_density_heatmap);
    owner.renderer->set_debug_show_performance_hud(owner.show_performance_hud);
    owner.renderer->set_debug_show_residency_hud(owner.show_residency_hud);
    owner.renderer->set_debug_overlay_opacity(owner.debug_overlay_opacity);
    owner.renderer->set_debug_preview_mode(_node_debug_draw_to_renderer_preview_mode(owner.debug_draw_mode));
    if (owner.runtime_preview_enabled) {
        owner.runtime_preview_restore_mode = owner.renderer->get_debug_preview_mode();
        owner.renderer->set_debug_preview_mode(GaussianSplatRenderer::DEBUG_PREVIEW_RUNTIME_MODIFICATIONS);
    }

    owner._update_debug_hud_visibility();
}

void GaussianSplatNodeDebugHelper::set_show_tile_grid(bool p_show) {
    if (owner.show_tile_grid == p_show) {
        return;
    }

    owner.show_tile_grid = p_show;

    GaussianSplatSettingsManager::set_debug_show_tile_grid(owner.show_tile_grid);

    if (owner.renderer.is_valid() && owner.renderer_helper.can_apply_renderer_settings()) {
        owner.renderer->set_debug_show_tile_grid(owner.show_tile_grid);
    }
}

void GaussianSplatNodeDebugHelper::set_show_density_heatmap(bool p_show) {
    if (owner.show_density_heatmap == p_show) {
        return;
    }

    owner.show_density_heatmap = p_show;

    GaussianSplatSettingsManager::set_debug_show_density_heatmap(owner.show_density_heatmap);

    if (owner.renderer.is_valid() && owner.renderer_helper.can_apply_renderer_settings()) {
        owner.renderer->set_debug_show_density_heatmap(owner.show_density_heatmap);
    }
}

void GaussianSplatNodeDebugHelper::set_show_performance_hud(bool p_show) {
    if (owner.show_performance_hud == p_show) {
        return;
    }

    owner.show_performance_hud = p_show;

    GaussianSplatSettingsManager::set_debug_show_performance_hud(owner.show_performance_hud);

    if (owner.renderer.is_valid() && owner.renderer_helper.can_apply_renderer_settings()) {
        owner.renderer->set_debug_show_performance_hud(owner.show_performance_hud);
    }

    if (owner.show_performance_overlay) {
        owner.update_gizmos();
    }

    owner._update_debug_hud_visibility();
}

void GaussianSplatNodeDebugHelper::set_show_lod_spheres(bool p_show) {
    if (owner.show_lod_spheres == p_show) {
        return;
    }

    owner.show_lod_spheres = p_show;
    owner.update_gizmos();
}

void GaussianSplatNodeDebugHelper::set_show_performance_overlay(bool p_show) {
    if (owner.show_performance_overlay == p_show) {
        return;
    }

    owner.show_performance_overlay = p_show;
    owner.update_gizmos();
}

void GaussianSplatNodeDebugHelper::set_debug_overlay_opacity(float p_opacity) {
    float clamped = CLAMP(p_opacity, 0.0f, 1.0f);
    if (Math::is_equal_approx(owner.debug_overlay_opacity, clamped)) {
        return;
    }

    owner.debug_overlay_opacity = clamped;

    if (owner.renderer.is_valid() && owner.renderer_helper.can_apply_renderer_settings()) {
        owner.renderer->set_debug_overlay_opacity(owner.debug_overlay_opacity);
    }
}

void GaussianSplatNodeDebugHelper::set_debug_draw_mode(int p_mode) {
    const auto mode = static_cast<GaussianSplatNode3D::DebugDrawMode>(p_mode);
    if (owner.debug_draw_mode == mode) {
        return;
    }

    owner.debug_draw_mode = mode;

    if (owner.renderer.is_valid() && owner.renderer_helper.can_apply_renderer_settings()) {
        if (owner.runtime_preview_enabled) {
            owner.runtime_preview_restore_mode = _node_debug_draw_to_renderer_preview_mode(owner.debug_draw_mode);
            owner.renderer->set_debug_preview_mode(GaussianSplatRenderer::DEBUG_PREVIEW_RUNTIME_MODIFICATIONS);
        } else {
            owner.renderer->set_debug_preview_mode(_node_debug_draw_to_renderer_preview_mode(owner.debug_draw_mode));
        }
    }

    owner.update_gizmos();
}

void GaussianSplatNodeDebugHelper::set_runtime_preview_enabled(bool p_enabled) {
    if (owner.runtime_preview_enabled == p_enabled) {
        return;
    }

    owner.runtime_preview_enabled = p_enabled;

    if (owner.renderer.is_valid() && owner.renderer_helper.can_apply_renderer_settings()) {
        if (owner.runtime_preview_enabled) {
            owner.runtime_preview_restore_mode = owner.renderer->get_debug_preview_mode();
            owner.renderer->set_debug_preview_mode(GaussianSplatRenderer::DEBUG_PREVIEW_RUNTIME_MODIFICATIONS);
        } else {
            owner.renderer->set_debug_preview_mode(static_cast<GaussianSplatRenderer::DebugPreviewMode>(owner.runtime_preview_restore_mode));
        }
    }

    owner.update_gizmos();
}

void GaussianSplatNodeDebugHelper::set_show_residency_hud(bool p_show) {
    if (owner.show_residency_hud == p_show) {
        return;
    }

    owner.show_residency_hud = p_show;

    GaussianSplatSettingsManager::set_debug_show_residency_hud(owner.show_residency_hud);

    if (owner.renderer.is_valid() && owner.renderer_helper.can_apply_renderer_settings()) {
        owner.renderer->set_debug_show_residency_hud(owner.show_residency_hud);
    }

    owner._update_debug_hud_visibility();
}

void GaussianSplatNodeQualityHelper::apply_quality_preset() {
    Dictionary config;
    fill_preset_config(owner.quality_preset, config);

    auto get_float = [&](const char *p_key, float p_default) -> float {
        Variant value = config.get(p_key, Variant(p_default));
        return (float)(double)value;
    };

    auto get_int = [&](const char *p_key, int p_default) -> int {
        Variant value = config.get(p_key, Variant(p_default));
        if (value.get_type() == Variant::FLOAT) {
            return (int)((double)value);
        }
        return (int)value;
    };

    owner.lod_bias = CLAMP(get_float("lod_bias", owner.lod_bias), 0.1f, 4.0f);
    owner.max_splat_count = MAX(1000, get_int("max_splat_count", owner.max_splat_count));

    update_quality_settings();
    owner._update_instance_params_in_director();
}

void GaussianSplatNodeQualityHelper::update_quality_settings() {
    Dictionary config;
    fill_preset_config(owner.quality_preset, config);

    auto get_float = [&](const char *p_key, float p_default) -> float {
        Variant value = config.get(p_key, Variant(p_default));
        return (float)(double)value;
    };

    auto get_int = [&](const char *p_key, int p_default) -> int {
        Variant value = config.get(p_key, Variant(p_default));
        if (value.get_type() == Variant::FLOAT) {
            return (int)((double)value);
        }
        return (int)value;
    };

    auto get_bool = [&](const char *p_key, bool p_default) -> bool {
        Variant value = config.get(p_key, Variant(p_default));
        return (bool)value;
    };

    owner.lod_bias = get_float("lod_bias", owner.lod_bias);
    owner.max_splat_count = MAX(1000, get_int("max_splat_count", owner.max_splat_count));

    const float min_budget_fraction = CLAMP(get_float("min_budget_fraction", 0.1f), 0.0f, 1.0f);
    const float importance_threshold = MAX(0.0f, get_float("importance_threshold", 0.1f));
    const float size_cull_threshold = MAX(0.0f, get_float("size_cull_threshold", 1.0f));
    const bool smooth_transitions = get_bool("smooth_transitions", true);
    const float transition_time = MAX(0.0f, get_float("transition_time", 0.25f));
    const float target_fps = MAX(1.0f, get_float("target_fps", 60.0f));
    const float quality_rate = MAX(0.0f, get_float("quality_rate", 0.18f));
    const bool temporal_coherence = get_bool("temporal_coherence", true);

    const int max_gpu_memory_mb = MAX(64, get_int("max_gpu_memory_mb", 512));
    const int target_gpu_memory_mb = MAX(32, get_int("target_gpu_memory_mb", 384));
    const float load_ahead_factor = MAX(0.0f, get_float("load_ahead_factor", 0.25f));
    const float unload_factor = MAX(0.1f, get_float("unload_factor", 1.05f));
    const int max_concurrent_loads = MAX(1, get_int("max_concurrent_loads", 2));
    const bool predictive_loading = get_bool("predictive_loading", true);
    const float prediction_time = MAX(0.0f, get_float("prediction_time", 0.5f));
    const int lod_level_count = CLAMP(get_int("lod_levels", 4), 1, 4);
    const float lod_distance_multiplier = MAX(1.0f, get_float("lod_multiplier", 1.8f));
    const bool adaptive_quality = get_bool("adaptive_quality", true);
    const int stream_budget_ms = MAX(1, get_int("stream_budget_ms", 2));
    const bool async_loading = get_bool("async_loading", true);
    const bool compression = get_bool("compression", true);

    int effective_max_splats = owner.max_splat_count;
    int effective_max_gpu_mb = max_gpu_memory_mb;
    int effective_target_gpu_mb = target_gpu_memory_mb;
    float effective_load_ahead = load_ahead_factor;
    float effective_unload = unload_factor;
    int effective_concurrent_loads = max_concurrent_loads;
    int effective_stream_budget_ms = stream_budget_ms;
    apply_quality_tier_limits(effective_max_splats, effective_max_gpu_mb, effective_target_gpu_mb,
            effective_load_ahead, effective_unload, effective_concurrent_loads, effective_stream_budget_ms);

    const float lod0_ratio = CLAMP(get_float("lod0_ratio", 0.25f), 0.0f, 1.0f);
    const float lod1_ratio = CLAMP(get_float("lod1_ratio", 0.5f), lod0_ratio, 1.0f);
    const float lod2_ratio = CLAMP(get_float("lod2_ratio", 0.75f), lod1_ratio, 1.0f);
    const float lod3_ratio = CLAMP(get_float("lod3_ratio", 1.0f), lod2_ratio, 1.0f);

    const float effective_distance = owner.max_render_distance > 0.0f ? owner.max_render_distance : 1000000.0f;

    auto compute_distance = [&](float p_ratio, float p_previous) {
        float value = effective_distance * p_ratio;
        return MAX(p_previous, value);
    };

    const float lod0_distance = compute_distance(lod0_ratio, 0.0f);
    const float lod1_distance = compute_distance(lod1_ratio, lod0_distance);
    const float lod2_distance = compute_distance(lod2_ratio, lod1_distance);
    float lod3_distance = compute_distance(lod3_ratio, lod2_distance);
    lod3_distance = MAX(lod3_distance, effective_distance);

    owner.max_splat_count = MAX(1000, effective_max_splats);
    const uint32_t max_budget = MAX(1, owner.max_splat_count);
    uint32_t min_budget = (uint32_t)(max_budget * min_budget_fraction);
    min_budget = MAX((uint32_t)1000, min_budget);
    min_budget = MIN(max_budget, min_budget);

    apply_quality_lod_config(lod0_distance, lod1_distance, lod2_distance, lod3_distance,
            effective_distance, max_budget, min_budget, importance_threshold, size_cull_threshold,
            smooth_transitions, transition_time, target_fps, quality_rate, temporal_coherence);
    apply_streaming_config_values(effective_max_gpu_mb, effective_target_gpu_mb, effective_distance,
            effective_load_ahead, effective_unload, effective_concurrent_loads, predictive_loading,
            prediction_time, lod_level_count, lod_distance_multiplier, adaptive_quality,
            effective_stream_budget_ms, async_loading, compression);

    if (owner.renderer.is_valid()) {
        owner._apply_renderer_settings();
    }
}

void GaussianSplatNodeQualityHelper::apply_quality_tier_limits(int &effective_max_splats, int &effective_max_gpu_mb,
        int &effective_target_gpu_mb, float &effective_load_ahead, float &effective_unload,
        int &effective_concurrent_loads, int &effective_stream_budget_ms) const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    const String tier_preset = ps->get_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
    const bool apply_tier_budgets = ps->get_setting("rendering/gaussian_splatting/quality/tier_apply_streaming_budgets", true);
    if (!apply_tier_budgets) {
        return;
    }

    QualityTierConfig tier_config;
    if (!get_quality_tier_config(tier_preset, tier_config)) {
        return;
    }

    effective_max_splats = MIN(effective_max_splats, (int)tier_config.max_splats);
    effective_max_gpu_mb = MIN(effective_max_gpu_mb, (int)tier_config.max_gpu_memory_mb);
    effective_target_gpu_mb = MIN(effective_target_gpu_mb, (int)tier_config.target_gpu_memory_mb);
    effective_load_ahead = MIN(effective_load_ahead, tier_config.load_ahead_factor);
    effective_unload = MIN(effective_unload, tier_config.unload_factor);
    effective_concurrent_loads = MIN(effective_concurrent_loads, (int)tier_config.max_concurrent_loads);
    effective_stream_budget_ms = MIN(effective_stream_budget_ms, (int)tier_config.stream_budget_ms);
}

void GaussianSplatNodeQualityHelper::apply_quality_lod_config(float lod0_distance, float lod1_distance, float lod2_distance, float lod3_distance,
        float effective_distance, uint32_t max_budget, uint32_t min_budget, float importance_threshold, float size_cull_threshold,
        bool smooth_transitions, float transition_time, float target_fps, float quality_rate,
        bool temporal_coherence) {
    owner.lod_config.lod_bias = owner.lod_bias;
    owner.lod_config.lod0_distance = lod0_distance;
    owner.lod_config.lod1_distance = lod1_distance;
    owner.lod_config.lod2_distance = lod2_distance;
    owner.lod_config.lod3_distance = lod3_distance;
    owner.lod_config.cull_distance = MAX(lod3_distance, effective_distance);
    owner.lod_config.max_splats_per_frame = max_budget;
    owner.lod_config.min_splats_per_frame = min_budget;
    owner.lod_config.importance_threshold = importance_threshold;
    owner.lod_config.size_cull_threshold = size_cull_threshold;
    owner.lod_config.smooth_transitions = smooth_transitions;
    owner.lod_config.transition_time = transition_time;
    owner.lod_config.target_framerate = target_fps;
    owner.lod_config.quality_adjustment_rate = quality_rate;
    owner.lod_config.enable_temporal_coherence = temporal_coherence;
    owner.lod_config.enable_painterly_mode = owner.enable_painterly;
}

void GaussianSplatNodeQualityHelper::apply_streaming_config_values(int effective_max_gpu_mb, int effective_target_gpu_mb,
        float effective_distance, float effective_load_ahead, float effective_unload, int effective_concurrent_loads,
        bool predictive_loading, float prediction_time, int lod_level_count, float lod_distance_multiplier,
        bool adaptive_quality, int effective_stream_budget_ms, bool async_loading, bool compression) {
    owner.streaming_config.max_gpu_memory = (uint64_t)effective_max_gpu_mb * 1024 * 1024;
    owner.streaming_config.target_gpu_memory = MIN(owner.streaming_config.max_gpu_memory, (uint64_t)effective_target_gpu_mb * 1024 * 1024);
    owner.streaming_config.max_cpu_memory = owner.streaming_config.max_gpu_memory * 2;
    owner.streaming_config.load_ahead_distance = MAX(0.0f, effective_distance * effective_load_ahead);
    float computed_unload_distance = effective_distance * effective_unload;
    owner.streaming_config.unload_distance = MAX(owner.streaming_config.load_ahead_distance, computed_unload_distance);
    if (owner.streaming_config.unload_distance <= owner.streaming_config.load_ahead_distance) {
        owner.streaming_config.unload_distance = owner.streaming_config.load_ahead_distance + 1.0f;
    }
    owner.streaming_config.max_concurrent_loads = effective_concurrent_loads;
    owner.streaming_config.enable_predictive_loading = predictive_loading;
    owner.streaming_config.prediction_time = prediction_time;
    owner.streaming_config.num_lod_levels = lod_level_count;
    owner.streaming_config.lod_distance_multiplier = lod_distance_multiplier;
    owner.streaming_config.enable_adaptive_quality = adaptive_quality;
    owner.streaming_config.enable_painterly_mode = owner.enable_painterly;
    owner.streaming_config.painterly_seed = owner.painterly_seed;
    const float safe_blend = MAX(owner.temporal_blend, 0.01f);
    owner.streaming_config.painterly_transition_rate = 1.0f / safe_blend;
    owner.streaming_config.painterly_hold_strength = owner.edge_threshold;
    owner.streaming_config.stream_budget_ms = effective_stream_budget_ms;
    owner.streaming_config.enable_async_loading = async_loading;
    owner.streaming_config.enable_compression = compression;
}

void GaussianSplatNodeQualityHelper::fill_preset_config(int p_preset, Dictionary &config) const {
    const auto preset = static_cast<GaussianSplatNode3D::QualityPreset>(p_preset);
    switch (preset) {
        case GaussianSplatNode3D::QUALITY_PERFORMANCE:
            config["lod_bias"] = 2.0f;
            config["max_splat_count"] = 200000;
            config["min_budget_fraction"] = 0.15f;
            config["importance_threshold"] = 0.2f;
            config["size_cull_threshold"] = 1.5f;
            config["smooth_transitions"] = false;
            config["transition_time"] = 0.1f;
            config["target_fps"] = 75.0f;
            config["quality_rate"] = 0.25f;
            config["temporal_coherence"] = false;
            config["max_gpu_memory_mb"] = 256;
            config["target_gpu_memory_mb"] = 192;
            config["load_ahead_factor"] = 0.15f;
            config["unload_factor"] = 0.95f;
            config["max_concurrent_loads"] = 1;
            config["predictive_loading"] = false;
            config["prediction_time"] = 0.25f;
            config["lod_levels"] = 3;
            config["lod_multiplier"] = 1.6f;
            config["adaptive_quality"] = false;
            config["stream_budget_ms"] = 1;
            config["async_loading"] = false;
            config["compression"] = true;
            config["lod0_ratio"] = 0.15f;
            config["lod1_ratio"] = 0.35f;
            config["lod2_ratio"] = 0.6f;
            config["lod3_ratio"] = 1.0f;
            break;

        case GaussianSplatNode3D::QUALITY_BALANCED:
            config["lod_bias"] = 1.0f;
            config["max_splat_count"] = 500000;
            config["min_budget_fraction"] = 0.1f;
            config["importance_threshold"] = 0.12f;
            config["size_cull_threshold"] = 1.0f;
            config["smooth_transitions"] = true;
            config["transition_time"] = 0.25f;
            config["target_fps"] = 60.0f;
            config["quality_rate"] = 0.18f;
            config["temporal_coherence"] = true;
            config["max_gpu_memory_mb"] = 512;
            config["target_gpu_memory_mb"] = 384;
            config["load_ahead_factor"] = 0.25f;
            config["unload_factor"] = 1.05f;
            config["max_concurrent_loads"] = 2;
            config["predictive_loading"] = true;
            config["prediction_time"] = 0.5f;
            config["lod_levels"] = 4;
            config["lod_multiplier"] = 1.8f;
            config["adaptive_quality"] = true;
            config["stream_budget_ms"] = 2;
            config["async_loading"] = true;
            config["compression"] = true;
            config["lod0_ratio"] = 0.25f;
            config["lod1_ratio"] = 0.5f;
            config["lod2_ratio"] = 0.75f;
            config["lod3_ratio"] = 1.0f;
            break;

        case GaussianSplatNode3D::QUALITY_QUALITY:
            config["lod_bias"] = 0.75f;
            config["max_splat_count"] = 1000000;
            config["min_budget_fraction"] = 0.05f;
            config["importance_threshold"] = 0.08f;
            config["size_cull_threshold"] = 0.6f;
            config["smooth_transitions"] = true;
            config["transition_time"] = 0.35f;
            config["target_fps"] = 60.0f;
            config["quality_rate"] = 0.1f;
            config["temporal_coherence"] = true;
            config["max_gpu_memory_mb"] = 768;
            config["target_gpu_memory_mb"] = 640;
            config["load_ahead_factor"] = 0.35f;
            config["unload_factor"] = 1.2f;
            config["max_concurrent_loads"] = 3;
            config["predictive_loading"] = true;
            config["prediction_time"] = 0.75f;
            config["lod_levels"] = 4;
            config["lod_multiplier"] = 2.0f;
            config["adaptive_quality"] = true;
            config["stream_budget_ms"] = 3;
            config["async_loading"] = true;
            config["compression"] = false;
            config["lod0_ratio"] = 0.35f;
            config["lod1_ratio"] = 0.65f;
            config["lod2_ratio"] = 0.9f;
            config["lod3_ratio"] = 1.0f;
            break;

        case GaussianSplatNode3D::QUALITY_CUSTOM:
        default:
            config["lod_bias"] = owner.lod_bias;
            config["max_splat_count"] = owner.max_splat_count;
            config["min_budget_fraction"] = 0.1f;
            config["importance_threshold"] = 0.1f;
            config["size_cull_threshold"] = 1.0f;
            config["smooth_transitions"] = true;
            config["transition_time"] = 0.25f;
            config["target_fps"] = 60.0f;
            config["quality_rate"] = 0.18f;
            config["temporal_coherence"] = true;
            config["max_gpu_memory_mb"] = 512;
            config["target_gpu_memory_mb"] = 384;
            config["load_ahead_factor"] = 0.25f;
            config["unload_factor"] = 1.05f;
            config["max_concurrent_loads"] = 2;
            config["predictive_loading"] = true;
            config["prediction_time"] = 0.5f;
            config["lod_levels"] = 4;
            config["lod_multiplier"] = 1.8f;
            config["adaptive_quality"] = true;
            config["stream_budget_ms"] = 2;
            config["async_loading"] = true;
            config["compression"] = true;
            config["lod0_ratio"] = 0.25f;
            config["lod1_ratio"] = 0.5f;
            config["lod2_ratio"] = 0.75f;
            config["lod3_ratio"] = 1.0f;
            break;
    }
}

void GaussianSplatNodeVisibilityHelper::clear_parent_visibility_tracking() {
    if (owner.parent_visibility_target) {
        Callable callable_with_bool = callable_mp(&owner, &GaussianSplatNode3D::_on_parent_visibility_changed_with_bool);
        if (owner.parent_visibility_target->is_connected("visibility_changed", callable_with_bool)) {
            owner.parent_visibility_target->disconnect("visibility_changed", callable_with_bool);
        }

        Callable callable_no_arg = callable_mp(&owner, &GaussianSplatNode3D::_on_parent_visibility_changed);
        if (owner.parent_visibility_target->is_connected("visibility_changed", callable_no_arg)) {
            owner.parent_visibility_target->disconnect("visibility_changed", callable_no_arg);
        }
    }

    owner.parent_visibility_target = nullptr;
    owner.parent_visible = true;
}

void GaussianSplatNodeVisibilityHelper::update_parent_visibility_tracking() {
    if (owner.update_mode != GaussianSplatNode3D::UPDATE_MODE_WHEN_PARENT_VISIBLE) {
        clear_parent_visibility_tracking();
        return;
    }

    if (!owner.is_inside_tree()) {
        clear_parent_visibility_tracking();
        owner.parent_visible = true;
        return;
    }

    Node *parent = owner.get_parent();
    if (owner.parent_visibility_target == parent && owner.parent_visibility_target) {
        update_parent_visibility_state();
        return;
    }

    clear_parent_visibility_tracking();

    if (!parent) {
        owner.parent_visible = true;
        return;
    }

    if (!parent->has_signal("visibility_changed")) {
        update_parent_visibility_state();
        return;
    }

    Error err = parent->connect("visibility_changed", callable_mp(&owner, &GaussianSplatNode3D::_on_parent_visibility_changed_with_bool));
    if (err != OK) {
        err = parent->connect("visibility_changed", callable_mp(&owner, &GaussianSplatNode3D::_on_parent_visibility_changed));
    }

    if (err == OK) {
        owner.parent_visibility_target = parent;
    }

    update_parent_visibility_state();
}

void GaussianSplatNodeVisibilityHelper::update_parent_visibility_state() {
    bool new_parent_visible = true;

    Node *parent = owner.get_parent();
    if (parent) {
        if (Node3D *parent_node3d = Object::cast_to<Node3D>(parent)) {
            new_parent_visible = parent_node3d->is_visible_in_tree();
        }
    }

    owner.parent_visible = new_parent_visible;
}

void GaussianSplatNodeVisibilityHelper::on_parent_visibility_changed_with_bool(bool p_visible) {
    (void)p_visible;
    on_parent_visibility_changed();
}

void GaussianSplatNodeVisibilityHelper::on_parent_visibility_changed() {
    update_parent_visibility_state();

    if (owner.update_mode == GaussianSplatNode3D::UPDATE_MODE_WHEN_PARENT_VISIBLE) {
        owner._update_visibility();
    }
}

bool GaussianSplatNodeRendererHelper::can_apply_renderer_settings() const {
    if (!owner.renderer.is_valid()) {
        return false;
    }
    if (!owner.is_inside_tree() || !owner.is_inside_world()) {
        return false;
    }

    const bool has_local_source_data = owner.renderer_data.is_valid() || owner.splat_asset.is_valid() || owner.runtime_asset.is_valid();
    if (!has_local_source_data) {
        return false;
    }

    const ObjectID renderer_id = owner.renderer->get_instance_id();
    const ObjectID node_id = owner.get_instance_id();
    if (renderer_id == ObjectID() || node_id == ObjectID()) {
        return false;
    }

    const auto &scene_state = owner.renderer->get_scene_state();
    if (scene_state.gaussian_data.is_valid()) {
        const bool owns_scene_data = owner.renderer_data.is_valid() && scene_state.gaussian_data == owner.renderer_data;
        const bool owns_scene_asset = scene_state.active_asset.is_valid() &&
                ((owner.splat_asset.is_valid() && scene_state.active_asset == owner.splat_asset) ||
                        (owner.runtime_asset.is_valid() && scene_state.active_asset == owner.runtime_asset));
        if (!owns_scene_data && !owns_scene_asset) {
            _release_renderer_settings_owner(renderer_id, node_id);
            return false;
        }
    }

    return _claim_renderer_settings_owner(renderer_id, node_id);
}

void GaussianSplatNodeRendererHelper::release_renderer_settings_ownership() {
    if (!owner.renderer.is_valid()) {
        return;
    }
    const ObjectID renderer_id = owner.renderer->get_instance_id();
    const ObjectID node_id = owner.get_instance_id();
    _release_renderer_settings_owner(renderer_id, node_id);
}

void GaussianSplatNodeRendererHelper::ensure_renderer() {
    owner._ensure_gaussian_base();

    if (!owner.is_inside_world()) {
        release_renderer_settings_ownership();
        return;
    }
    if (!owner.renderer.is_valid()) {
        GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
        if (director) {
            owner.renderer = director->get_shared_renderer(owner.get_world_3d().ptr());
            if (owner.renderer.is_valid()) {
                apply_renderer_settings();
            }
        }
    }

    owner._sync_gaussian_storage();
}

void GaussianSplatNodeRendererHelper::apply_renderer_settings() {
    if (!owner.renderer.is_valid()) {
        return;
    }
    if (!can_apply_renderer_settings()) {
        return;
    }

    uint32_t local_splat_count = 0;
    if (owner.renderer_data.is_valid()) {
        local_splat_count = owner.renderer_data->get_count();
    } else if (owner.splat_asset.is_valid()) {
        local_splat_count = owner.splat_asset->get_splat_count();
    }

    uint32_t renderer_splat_count = 0;
    const auto &scene_state = owner.renderer->get_scene_state();
    if (scene_state.gaussian_data.is_valid()) {
        renderer_splat_count = scene_state.gaussian_data->get_count();
    }

    const bool renderer_shared = renderer_splat_count > 0 && renderer_splat_count != local_splat_count;
    if (!renderer_shared) {
        owner.renderer->set_max_splats(owner.max_splat_count);
    }
    // Use asset-level LOD setting from import metadata (defaults to true)
    owner.renderer->set_lod_enabled(owner.asset_lod_enabled);
    owner.renderer->set_lod_bias(owner.lod_bias);
    owner.renderer->set_lod_max_distance(owner.max_render_distance);
    owner.renderer->set_frustum_culling(owner.use_frustum_culling);
    // Use asset-level GPU optimization setting from import metadata (defaults to true)
    owner.renderer->set_async_upload_enabled(owner.asset_optimize_for_gpu);
    owner.renderer->set_painterly_enabled(owner.enable_painterly);
    owner.renderer->set_painterly_edge_threshold(owner.edge_threshold);
    owner.renderer->set_painterly_stroke_opacity(owner.stroke_opacity);
    owner.renderer->set_painterly_stroke_length(owner.stroke_width);
    owner.renderer->set_painterly_gamma(MAX(owner.temporal_blend, 0.01f));
    owner.renderer->set_opacity_multiplier(owner.opacity);
    owner.renderer->set_color_grading(owner.color_grading);
    {
        GaussianStreamingSystem::ConfigOverrides overrides;

        overrides.override_prefetch = true;
        overrides.predictive_prefetch_enabled = owner.streaming_config.enable_predictive_loading;
        overrides.prefetch_lookahead_distance = owner.streaming_config.load_ahead_distance;

        VRAMBudgetConfig vram_config = VRAMBudgetConfig::load_from_project_settings();
        uint32_t budget_mb = uint32_t(owner.streaming_config.max_gpu_memory / (1024 * 1024));
        vram_config.budget_mb = MAX(64u, budget_mb);
        vram_config.min_chunks = MIN(vram_config.min_chunks, vram_config.max_chunks);
        overrides.override_vram_budget = true;
        overrides.vram_budget_config = vram_config;

        LODConfig lod_override;
        lod_override.load_from_project_settings();
        lod_override.enabled = owner.streaming_config.enable_adaptive_quality;
        lod_override.num_levels = CLAMP((int)owner.streaming_config.num_lod_levels, 2, 8);
        lod_override.base_threshold = MAX(1.0f, owner.lod_config.lod0_distance);
        lod_override.max_distance = MAX(lod_override.base_threshold * 2.0f, owner.lod_config.cull_distance);
        overrides.override_lod_config = true;
        overrides.lod_config = lod_override;

        owner.renderer->set_streaming_config_overrides(overrides);
    }
    owner.debug_helper.apply_renderer_debug_settings();
}

void GaussianSplatNodeRendererHelper::upload_asset_to_renderer() {
#ifndef GS_SILENCE_LOGS
    // DEBUG: Track upload calls
    static int upload_call_count = 0;
    upload_call_count++;
    if (GaussianSplattingIO::is_data_log_enabled()) {
        GS_LOG_RENDERER_DEBUG(vformat("[UPLOAD_ASSET #%d] Called, renderer.is_valid=%s, splat_asset.is_valid=%s, procedural_data=%s",
            upload_call_count,
            owner.renderer.is_valid() ? "true" : "false",
            owner.splat_asset.is_valid() ? "true" : "false",
            owner.renderer_data.is_valid() ? "true" : "false"));
    }
#endif

    if (owner.splat_asset.is_null() && owner.renderer_data.is_null()) {
        release_renderer_settings_ownership();
        owner._unregister_shared_renderer();
        owner.renderer_data.unref();
        owner.visible_splat_count = 0;
        owner.total_splat_count = 0;
        owner.gpu_memory_mb = 0.0f;
        return;
    }

    if (owner.renderer_data.is_valid()) {
        owner.total_splat_count = owner.renderer_data->get_count();
        owner.gpu_memory_mb = owner.renderer_data->get_memory_usage() / (1024.0f * 1024.0f);
    } else if (owner.splat_asset.is_valid()) {
        owner.total_splat_count = owner.splat_asset->get_splat_count();
    }

    owner._register_shared_renderer();
    owner._sync_gaussian_storage();
}

void GaussianSplatNodeRendererHelper::mark_render_state_dirty() {
    owner.render_state_dirty = true;
    if (owner.renderer.is_valid()) {
        owner.renderer->invalidate_cached_render();
    }
}
