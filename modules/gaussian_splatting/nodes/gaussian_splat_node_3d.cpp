#include "gaussian_splat_node_3d.h"
#include "../core/gs_project_settings.h"
#include "../core/gaussian_data.h"
#include "../core/gaussian_splat_manager.h"
#include "../core/gaussian_splat_settings_manager.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../core/quality_tier_config.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "../logger/gs_debug_trace.h"
#include "../renderer/gpu_debug_utils.h"
#include "../resources/color_grading_resource.h"
#include "gaussian_splat_debug_hud.h"
#include "core/error/error_macros.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "core/io/resource_loader.h"
#include "core/object/callable_method_pointer.h"
#include "core/object/object.h"
#include "core/templates/list.h"
#include "scene/3d/camera_3d.h"
#include "scene/main/node.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/viewport.h"
#include "scene/resources/3d/world_3d.h"
#include "servers/rendering/renderer_rd/storage_rd/gaussian_splat_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering_server.h"
#include <cstdint>

#include "../logger/gs_logger.h"

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
namespace {
static bool _is_frame_log_enabled() { return gs::settings::is_frame_log_enabled(); }

static bool _is_multi_instance_shared_renderer_active(const Ref<GaussianSplatRenderer> &p_renderer) {
    if (!p_renderer.is_valid()) {
        return false;
    }
    GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
    return director && director->get_instance_count_for_renderer(p_renderer.ptr()) > 1u;
}
} // namespace

void GaussianSplatNode3D::_bind_methods() {
    // Asset management
    ADD_GROUP("Asset", "");
    ClassDB::bind_method(D_METHOD("set_ply_file_path", "path"), &GaussianSplatNode3D::set_ply_file_path);
    ClassDB::bind_method(D_METHOD("get_ply_file_path"), &GaussianSplatNode3D::get_ply_file_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "ply_file_path", PROPERTY_HINT_FILE, "*.ply,*.spz"), "set_ply_file_path", "get_ply_file_path");

    ClassDB::bind_method(D_METHOD("set_splat_asset", "asset"), &GaussianSplatNode3D::set_splat_asset);
    ClassDB::bind_method(D_METHOD("get_splat_asset"), &GaussianSplatNode3D::get_splat_asset);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "splat_asset", PROPERTY_HINT_RESOURCE_TYPE, "GaussianSplatAsset"), "set_splat_asset", "get_splat_asset");

    ClassDB::bind_method(D_METHOD("set_auto_load", "enabled"), &GaussianSplatNode3D::set_auto_load);
    ClassDB::bind_method(D_METHOD("is_auto_load_enabled"), &GaussianSplatNode3D::is_auto_load_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_load"), "set_auto_load", "is_auto_load_enabled");

    ClassDB::bind_method(D_METHOD("reload_asset"), &GaussianSplatNode3D::reload_asset);
    ClassDB::bind_method(D_METHOD("is_asset_loading"), &GaussianSplatNode3D::is_asset_loading);

    ClassDB::bind_method(D_METHOD("set_splat_data", "positions", "colors", "scales", "opacities", "rotations",
                               "spherical_harmonics", "palette_ids", "painterly_flags", "normals", "brush_axes",
                               "stroke_ages", "is_2d_mode"),
            &GaussianSplatNode3D::set_splat_data,
            DEFVAL(PackedColorArray()),
            DEFVAL(PackedVector3Array()),
            DEFVAL(PackedFloat32Array()),
            DEFVAL(TypedArray<Quaternion>()),
            DEFVAL(PackedFloat32Array()),
            DEFVAL(PackedInt32Array()),
            DEFVAL(PackedInt32Array()),
            DEFVAL(PackedVector3Array()),
            DEFVAL(PackedVector2Array()),
            DEFVAL(PackedFloat32Array()),
            DEFVAL(false));

    // Quality settings
    ADD_GROUP("Quality", "quality/");
    ClassDB::bind_method(D_METHOD("set_quality_preset", "preset"), &GaussianSplatNode3D::set_quality_preset);
    ClassDB::bind_method(D_METHOD("get_quality_preset"), &GaussianSplatNode3D::get_quality_preset);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "quality/preset", PROPERTY_HINT_ENUM, "Performance,Balanced,Quality,Custom"), "set_quality_preset", "get_quality_preset");

    ClassDB::bind_method(D_METHOD("set_lod_bias", "bias"), &GaussianSplatNode3D::set_lod_bias);
    ClassDB::bind_method(D_METHOD("get_lod_bias"), &GaussianSplatNode3D::get_lod_bias);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "quality/lod_bias", PROPERTY_HINT_RANGE, "0.1,4.0,0.1"), "set_lod_bias", "get_lod_bias");

    ClassDB::bind_method(D_METHOD("set_max_render_distance", "distance"), &GaussianSplatNode3D::set_max_render_distance);
    ClassDB::bind_method(D_METHOD("get_max_render_distance"), &GaussianSplatNode3D::get_max_render_distance);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "quality/max_render_distance", PROPERTY_HINT_RANGE, "0.0,10000.0,1.0,or_greater,suffix:m"), "set_max_render_distance", "get_max_render_distance");

    ClassDB::bind_method(D_METHOD("set_max_splat_count", "count"), &GaussianSplatNode3D::set_max_splat_count);
    ClassDB::bind_method(D_METHOD("get_max_splat_count"), &GaussianSplatNode3D::get_max_splat_count);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "quality/max_splat_count", PROPERTY_HINT_RANGE, "1000,10000000,1000"), "set_max_splat_count", "get_max_splat_count");

    // Painterly settings
    ADD_GROUP("Painterly", "painterly/");
    ClassDB::bind_method(D_METHOD("set_enable_painterly", "enabled"), &GaussianSplatNode3D::set_enable_painterly);
    ClassDB::bind_method(D_METHOD("is_painterly_enabled"), &GaussianSplatNode3D::is_painterly_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "painterly/enabled"), "set_enable_painterly", "is_painterly_enabled");

    ClassDB::bind_method(D_METHOD("set_edge_threshold", "threshold"), &GaussianSplatNode3D::set_edge_threshold);
    ClassDB::bind_method(D_METHOD("get_edge_threshold"), &GaussianSplatNode3D::get_edge_threshold);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/edge_threshold", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_edge_threshold", "get_edge_threshold");

    ClassDB::bind_method(D_METHOD("set_stroke_opacity", "opacity"), &GaussianSplatNode3D::set_stroke_opacity);
    ClassDB::bind_method(D_METHOD("get_stroke_opacity"), &GaussianSplatNode3D::get_stroke_opacity);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/stroke_opacity", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_stroke_opacity", "get_stroke_opacity");

    ClassDB::bind_method(D_METHOD("set_stroke_width", "width"), &GaussianSplatNode3D::set_stroke_width);
    ClassDB::bind_method(D_METHOD("get_stroke_width"), &GaussianSplatNode3D::get_stroke_width);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/stroke_width", PROPERTY_HINT_RANGE, "0.1,5.0,0.1"), "set_stroke_width", "get_stroke_width");

    // Legacy method retained for existing scenes/scripts.
    ClassDB::bind_method(D_METHOD("set_color_variation", "variation"), &GaussianSplatNode3D::set_color_variation);
    ClassDB::bind_method(D_METHOD("get_color_variation"), &GaussianSplatNode3D::get_color_variation);

    ClassDB::bind_method(D_METHOD("set_temporal_blend", "blend"), &GaussianSplatNode3D::set_temporal_blend);
    ClassDB::bind_method(D_METHOD("get_temporal_blend"), &GaussianSplatNode3D::get_temporal_blend);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "painterly/temporal_blend", PROPERTY_HINT_RANGE, "0.01,1.0,0.01"), "set_temporal_blend", "get_temporal_blend");

    ClassDB::bind_method(D_METHOD("set_painterly_seed", "seed"), &GaussianSplatNode3D::set_painterly_seed);
    ClassDB::bind_method(D_METHOD("get_painterly_seed"), &GaussianSplatNode3D::get_painterly_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "painterly/seed", PROPERTY_HINT_RANGE, "0,65535,1"), "set_painterly_seed", "get_painterly_seed");

    // Rendering settings
    ADD_GROUP("Rendering", "rendering/");
    ClassDB::bind_method(D_METHOD("set_update_mode", "mode"), &GaussianSplatNode3D::set_update_mode);
    ClassDB::bind_method(D_METHOD("get_update_mode"), &GaussianSplatNode3D::get_update_mode);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "rendering/update_mode", PROPERTY_HINT_ENUM, "Always,When Visible,When Parent Visible,Manual"), "set_update_mode", "get_update_mode");

    ClassDB::bind_method(D_METHOD("set_cast_shadow", "enabled"), &GaussianSplatNode3D::set_cast_shadow);
    ClassDB::bind_method(D_METHOD("get_cast_shadow"), &GaussianSplatNode3D::get_cast_shadow);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rendering/cast_shadow"), "set_cast_shadow", "get_cast_shadow");

    ClassDB::bind_method(D_METHOD("set_use_frustum_culling", "enabled"), &GaussianSplatNode3D::set_use_frustum_culling);
    ClassDB::bind_method(D_METHOD("is_frustum_culling_enabled"), &GaussianSplatNode3D::is_frustum_culling_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rendering/frustum_culling"), "set_use_frustum_culling", "is_frustum_culling_enabled");

    // Legacy method retained for existing scenes/scripts.
    ClassDB::bind_method(D_METHOD("set_use_occlusion_culling", "enabled"), &GaussianSplatNode3D::set_use_occlusion_culling);
    ClassDB::bind_method(D_METHOD("is_occlusion_culling_enabled"), &GaussianSplatNode3D::is_occlusion_culling_enabled);

    // Legacy render-thread path deprecated; no longer exposed as a property.

    ClassDB::bind_method(D_METHOD("set_opacity", "opacity"), &GaussianSplatNode3D::set_opacity);
    ClassDB::bind_method(D_METHOD("get_opacity"), &GaussianSplatNode3D::get_opacity);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rendering/opacity", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_opacity", "get_opacity");

    ClassDB::bind_method(D_METHOD("set_wind_override_enabled", "enabled"), &GaussianSplatNode3D::set_wind_override_enabled);
    ClassDB::bind_method(D_METHOD("is_wind_override_enabled"), &GaussianSplatNode3D::is_wind_override_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rendering/wind_override_enabled"), "set_wind_override_enabled", "is_wind_override_enabled");

    ClassDB::bind_method(D_METHOD("set_wind_enabled", "enabled"), &GaussianSplatNode3D::set_wind_enabled);
    ClassDB::bind_method(D_METHOD("is_wind_enabled"), &GaussianSplatNode3D::is_wind_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "rendering/wind_enabled"), "set_wind_enabled", "is_wind_enabled");

    ClassDB::bind_method(D_METHOD("set_wind_strength", "strength"), &GaussianSplatNode3D::set_wind_strength);
    ClassDB::bind_method(D_METHOD("get_wind_strength"), &GaussianSplatNode3D::get_wind_strength);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rendering/wind_strength", PROPERTY_HINT_RANGE, "0.0,10.0,0.01,or_greater"), "set_wind_strength", "get_wind_strength");

    ClassDB::bind_method(D_METHOD("set_wind_direction", "direction"), &GaussianSplatNode3D::set_wind_direction);
    ClassDB::bind_method(D_METHOD("get_wind_direction"), &GaussianSplatNode3D::get_wind_direction);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "rendering/wind_direction"), "set_wind_direction", "get_wind_direction");

    ClassDB::bind_method(D_METHOD("set_wind_frequency", "frequency"), &GaussianSplatNode3D::set_wind_frequency);
    ClassDB::bind_method(D_METHOD("get_wind_frequency"), &GaussianSplatNode3D::get_wind_frequency);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rendering/wind_frequency", PROPERTY_HINT_RANGE, "0.0,8.0,0.01,or_greater"), "set_wind_frequency", "get_wind_frequency");

    ClassDB::bind_method(D_METHOD("set_color_grading", "grading"), &GaussianSplatNode3D::set_color_grading);
    ClassDB::bind_method(D_METHOD("get_color_grading"), &GaussianSplatNode3D::get_color_grading);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "rendering/color_grading", PROPERTY_HINT_RESOURCE_TYPE, "ColorGradingResource"), "set_color_grading", "get_color_grading");

    ClassDB::bind_method(D_METHOD("bake_color_grading"), &GaussianSplatNode3D::bake_color_grading);
    ClassDB::bind_method(D_METHOD("bake_color_grading_snapshot", "grading_snapshot"), &GaussianSplatNode3D::bake_color_grading_snapshot);
    ClassDB::bind_method(D_METHOD("restore_color_grading"), &GaussianSplatNode3D::restore_color_grading);
    ClassDB::bind_method(D_METHOD("is_color_grading_baked"), &GaussianSplatNode3D::is_color_grading_baked);

    // Performance monitoring
    ClassDB::bind_method(D_METHOD("get_visible_splat_count"), &GaussianSplatNode3D::get_visible_splat_count);
    ClassDB::bind_method(D_METHOD("get_total_splat_count"), &GaussianSplatNode3D::get_total_splat_count);
    ClassDB::bind_method(D_METHOD("get_last_update_time_ms"), &GaussianSplatNode3D::get_last_update_time_ms);
    ClassDB::bind_method(D_METHOD("get_gpu_memory_mb"), &GaussianSplatNode3D::get_gpu_memory_mb);
    ClassDB::bind_method(D_METHOD("get_statistics"), &GaussianSplatNode3D::get_statistics);
    ClassDB::bind_method(D_METHOD("get_configuration_warnings"), &GaussianSplatNode3D::get_configuration_warnings);

    // Editor helpers
    ADD_GROUP("Debug", "debug/");
    ClassDB::bind_method(D_METHOD("set_preview_enabled", "enabled"), &GaussianSplatNode3D::set_preview_enabled);
    ClassDB::bind_method(D_METHOD("is_preview_enabled"), &GaussianSplatNode3D::is_preview_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/preview_enabled"), "set_preview_enabled", "is_preview_enabled");

    ClassDB::bind_method(D_METHOD("set_show_bounds", "show"), &GaussianSplatNode3D::set_show_bounds);
    ClassDB::bind_method(D_METHOD("is_showing_bounds"), &GaussianSplatNode3D::is_showing_bounds);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_bounds"), "set_show_bounds", "is_showing_bounds");

    ClassDB::bind_method(D_METHOD("set_show_statistics", "show"), &GaussianSplatNode3D::set_show_statistics);
    ClassDB::bind_method(D_METHOD("is_showing_statistics"), &GaussianSplatNode3D::is_showing_statistics);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_statistics"), "set_show_statistics", "is_showing_statistics");

    // Debug overlay methods from PR #145
    ClassDB::bind_method(D_METHOD("set_show_tile_grid", "show"), &GaussianSplatNode3D::set_show_tile_grid);
    ClassDB::bind_method(D_METHOD("is_showing_tile_grid"), &GaussianSplatNode3D::is_showing_tile_grid);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_tile_grid"), "set_show_tile_grid", "is_showing_tile_grid");

    ClassDB::bind_method(D_METHOD("set_show_density_heatmap", "show"), &GaussianSplatNode3D::set_show_density_heatmap);
    ClassDB::bind_method(D_METHOD("is_showing_density_heatmap"), &GaussianSplatNode3D::is_showing_density_heatmap);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_density_heatmap"), "set_show_density_heatmap", "is_showing_density_heatmap");

    ClassDB::bind_method(D_METHOD("set_show_performance_hud", "show"), &GaussianSplatNode3D::set_show_performance_hud);
    ClassDB::bind_method(D_METHOD("is_showing_performance_hud"), &GaussianSplatNode3D::is_showing_performance_hud);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_performance_hud"), "set_show_performance_hud", "is_showing_performance_hud");

    ClassDB::bind_method(D_METHOD("set_show_lod_spheres", "show"), &GaussianSplatNode3D::set_show_lod_spheres);
    ClassDB::bind_method(D_METHOD("is_showing_lod_spheres"), &GaussianSplatNode3D::is_showing_lod_spheres);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_lod_spheres"), "set_show_lod_spheres", "is_showing_lod_spheres");

    ClassDB::bind_method(D_METHOD("set_show_performance_overlay", "show"), &GaussianSplatNode3D::set_show_performance_overlay);
    ClassDB::bind_method(D_METHOD("is_showing_performance_overlay"), &GaussianSplatNode3D::is_showing_performance_overlay);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_performance_overlay"), "set_show_performance_overlay", "is_showing_performance_overlay");

    ClassDB::bind_method(D_METHOD("set_debug_overlay_opacity", "opacity"), &GaussianSplatNode3D::set_debug_overlay_opacity);
    ClassDB::bind_method(D_METHOD("get_debug_overlay_opacity"), &GaussianSplatNode3D::get_debug_overlay_opacity);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "debug/overlay_opacity", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_debug_overlay_opacity", "get_debug_overlay_opacity");

    ClassDB::bind_method(D_METHOD("set_debug_draw_mode", "mode"), &GaussianSplatNode3D::set_debug_draw_mode);
    ClassDB::bind_method(D_METHOD("get_debug_draw_mode"), &GaussianSplatNode3D::get_debug_draw_mode);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "debug/debug_draw_mode", PROPERTY_HINT_ENUM, "Off,Wireframe,Points,Heatmap"), "set_debug_draw_mode", "get_debug_draw_mode");

    ClassDB::bind_method(D_METHOD("set_runtime_preview_enabled", "enabled"), &GaussianSplatNode3D::set_runtime_preview_enabled);
    ClassDB::bind_method(D_METHOD("is_runtime_preview_enabled"), &GaussianSplatNode3D::is_runtime_preview_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/runtime_preview"), "set_runtime_preview_enabled", "is_runtime_preview_enabled");

    ClassDB::bind_method(D_METHOD("set_show_residency_hud", "show"), &GaussianSplatNode3D::set_show_residency_hud);
    ClassDB::bind_method(D_METHOD("is_showing_residency_hud"), &GaussianSplatNode3D::is_showing_residency_hud);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug/show_residency_hud"), "set_show_residency_hud", "is_showing_residency_hud");

    // Get renderer method from master
    ClassDB::bind_method(D_METHOD("get_renderer"), &GaussianSplatNode3D::get_renderer);

    // Manual update
    ClassDB::bind_method(D_METHOD("update_splats"), &GaussianSplatNode3D::update_splats);
    ClassDB::bind_method(D_METHOD("force_update"), &GaussianSplatNode3D::force_update);

    // Enums
    BIND_ENUM_CONSTANT(QUALITY_PERFORMANCE);
    BIND_ENUM_CONSTANT(QUALITY_BALANCED);
    BIND_ENUM_CONSTANT(QUALITY_QUALITY);
    BIND_ENUM_CONSTANT(QUALITY_CUSTOM);

    BIND_ENUM_CONSTANT(UPDATE_MODE_ALWAYS);
    BIND_ENUM_CONSTANT(UPDATE_MODE_WHEN_VISIBLE);
    BIND_ENUM_CONSTANT(UPDATE_MODE_WHEN_PARENT_VISIBLE);
    BIND_ENUM_CONSTANT(UPDATE_MODE_MANUAL);

    BIND_ENUM_CONSTANT(DEBUG_DRAW_OFF);
    BIND_ENUM_CONSTANT(DEBUG_DRAW_WIREFRAME);
    BIND_ENUM_CONSTANT(DEBUG_DRAW_POINTS);
    BIND_ENUM_CONSTANT(DEBUG_DRAW_HEATMAP);

    // Signals
    ADD_SIGNAL(MethodInfo("asset_loaded"));
    ADD_SIGNAL(MethodInfo("asset_loading_failed", PropertyInfo(Variant::STRING, "error")));
    ADD_SIGNAL(MethodInfo("viewport_visibility_changed", PropertyInfo(Variant::BOOL, "visible")));
}

GaussianSplatNode3D::GaussianSplatNode3D() :
        asset_helper(*this),
        viewport_helper(*this),
        debug_helper(*this),
        quality_helper(*this),
        visibility_helper(*this),
        renderer_helper(*this) {
    set_notify_transform(true);

    // Initialize default bounds
    local_aabb = AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
    world_aabb = local_aabb;
    _ensure_gaussian_base();

    GaussianSplatSettingsManager::DebugOverlaySettings debug_settings =
            GaussianSplatSettingsManager::load_debug_overlay_settings();
    show_tile_grid = debug_settings.show_tile_grid;
    show_density_heatmap = debug_settings.show_density_heatmap;
    show_performance_hud = debug_settings.show_performance_hud;
    show_residency_hud = debug_settings.show_residency_hud;

    // Apply default quality preset
    _apply_quality_preset();
}

GaussianSplatNode3D::~GaussianSplatNode3D() {
    _disconnect_viewport_observers();
    _clear_asset();
    _release_gaussian_base();
    _clear_parent_visibility_tracking();

    if (render_instance.is_valid()) {
        RS::get_singleton()->free(render_instance);
    }
}

void GaussianSplatNode3D::_notification_enter_tree() {
    // When GaussianSplatManager exists, it drives updates via frame_pre_draw.
    // Only enable NOTIFICATION_PROCESS as fallback when manager is unavailable.
    GaussianSplatManager *manager = GaussianSplatManager::get_singleton();
    if (manager) {
        set_update_mode(update_mode);
        // Manager handles updates - disable node processing to avoid double updates.
        set_process(false);
        manager->register_node(this);
    } else {
        // Fallback: no manager, enable process-based updates.
        set_update_mode(update_mode);
    }

    if (auto_load && !ply_file_path.is_empty() && splat_asset.is_null()) {
        _load_asset();
    }
    _update_bounds();
    _update_visibility();
    Viewport *initial_viewport = Engine::get_singleton()->is_editor_hint() ? _find_editor_scene_viewport() : get_viewport();
    _update_cached_render_target(initial_viewport);

    // Ensure renderer exists and has correct instance transform before first render.
    // This is critical for editor viewport rendering which may happen before
    // NOTIFICATION_PROCESS fires.
    _ensure_renderer();
    _update_render_instance();
    _register_shared_renderer();
    _update_debug_hud_visibility();
}

void GaussianSplatNode3D::_notification_enter_world() {
    _ensure_renderer();
    _update_render_instance();
    _register_shared_renderer();
}

void GaussianSplatNode3D::_notification_exit_tree() {
    if (debug_hud_control) {
        debug_hud_control->set_visible(false);
        debug_hud_control->set_process(false);
    }
    visible_in_viewport = false;
    _update_visibility();
    _clear_parent_visibility_tracking();
    _update_cached_render_target(nullptr);
    _unregister_shared_renderer();

    // Release GPU storage when detached to avoid keeping allocations alive.
    // Storage will be reallocated on re-enter if needed.
    _release_gaussian_base();

    if (render_instance.is_valid()) {
        RS::get_singleton()->free(render_instance);
        render_instance = RID();
    }

    if (GaussianSplatManager *manager = GaussianSplatManager::get_singleton()) {
        manager->unregister_node(this);
    }
}

void GaussianSplatNode3D::_notification_process() {
    if (OS::get_singleton()->has_feature("headless")) {
        process_gaussian_render();
        return;
    }

    // Only process here if GaussianSplatManager is unavailable.
    // When the manager exists, it calls process_gaussian_render() from frame_pre_draw.
    if (!GaussianSplatManager::get_singleton()) {
        process_gaussian_render();
    }
}

#ifdef TOOLS_ENABLED
void GaussianSplatNode3D::_notification_editor_post_save() {
    // Restore editor state after the scene is saved and resources are reloaded.
    if (!ply_file_path.is_empty() && splat_asset.is_null()) {
        _load_asset();
    }
}
#endif

void GaussianSplatNode3D::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            _notification_enter_tree();
        } break;

        case NOTIFICATION_ENTER_WORLD: {
            _notification_enter_world();
        } break;

        case NOTIFICATION_READY: {
        } break;

        case NOTIFICATION_EXIT_TREE: {
            _notification_exit_tree();
        } break;

        case NOTIFICATION_TRANSFORM_CHANGED: {
            _on_transform_changed();
        } break;

        case NOTIFICATION_VISIBILITY_CHANGED: {
            _update_visibility();
        } break;

        case NOTIFICATION_PROCESS: {
            _notification_process();
        } break;

#ifdef TOOLS_ENABLED
        case NOTIFICATION_EDITOR_PRE_SAVE: {
            // Store editor state before saving
        } break;

        case NOTIFICATION_EDITOR_POST_SAVE: {
            _notification_editor_post_save();
        } break;
#endif
    }
}

void GaussianSplatNode3D::_validate_property(PropertyInfo &p_property) const {
    if (_is_multi_instance_shared_renderer_active(renderer)) {
        if (p_property.name.begins_with("painterly/") || p_property.name == "rendering/color_grading") {
            p_property.usage = PROPERTY_USAGE_NO_EDITOR;
            return;
        }
    }

    // Hide painterly settings if disabled
    if (!enable_painterly) {
        if (p_property.name.begins_with("painterly/") && p_property.name != "painterly/enabled") {
            p_property.usage = PROPERTY_USAGE_NO_EDITOR;
        }
    }
    // Hide per-instance wind controls unless override is enabled.
    if (!wind_override_enabled) {
        if (p_property.name == "rendering/wind_enabled" ||
                p_property.name == "rendering/wind_strength" ||
                p_property.name == "rendering/wind_direction" ||
                p_property.name == "rendering/wind_frequency") {
            p_property.usage = PROPERTY_USAGE_NO_EDITOR;
        }
    }

    // Hide custom quality settings if not in custom mode
    if (quality_preset != QUALITY_CUSTOM) {
        if (p_property.name == "quality/lod_bias" ||
            p_property.name == "quality/max_splat_count") {
            p_property.usage = PROPERTY_USAGE_NO_EDITOR;
        }
    }

    // Hide debug options in release builds
#ifndef DEBUG_ENABLED
    if (p_property.name.begins_with("debug/")) {
        p_property.usage = PROPERTY_USAGE_NO_EDITOR;
    }
#endif
}

bool GaussianSplatNode3D::_set(const StringName &p_name, const Variant &p_value) {
    // Legacy serialized property compatibility after property unbinding.
    if (p_name == StringName("painterly/color_variation")) {
        set_color_variation((float)p_value);
        return true;
    }
    if (p_name == StringName("rendering/occlusion_culling")) {
        set_use_occlusion_culling((bool)p_value);
        return true;
    }
    return false;
}

bool GaussianSplatNode3D::_get(const StringName &p_name, Variant &r_ret) const {
    if (p_name == StringName("painterly/color_variation")) {
        r_ret = color_variation;
        return true;
    }
    if (p_name == StringName("rendering/occlusion_culling")) {
        r_ret = use_occlusion_culling;
        return true;
    }
    if (p_name == StringName("stats/visible_splats")) {
        r_ret = (int64_t)visible_splat_count;
        return true;
    }
    if (p_name == StringName("stats/total_splats")) {
        r_ret = (int64_t)total_splat_count;
        return true;
    }
    if (p_name == StringName("stats/update_time_ms")) {
        r_ret = last_update_time_ms;
        return true;
    }
    if (p_name == StringName("stats/gpu_memory_mb")) {
        r_ret = gpu_memory_mb;
        return true;
    }
    return false;
}

void GaussianSplatNode3D::_get_property_list(List<PropertyInfo> *p_list) const {
    // Add performance statistics as read-only properties
    if (show_statistics && splat_asset.is_valid()) {
        p_list->push_back(PropertyInfo(Variant::INT, "stats/visible_splats", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
        p_list->push_back(PropertyInfo(Variant::INT, "stats/total_splats", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
        p_list->push_back(PropertyInfo(Variant::FLOAT, "stats/update_time_ms", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
        p_list->push_back(PropertyInfo(Variant::FLOAT, "stats/gpu_memory_mb", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
    }
}

void GaussianSplatNode3D::set_ply_file_path(const String &p_path) {
    if (ply_file_path == p_path) {
        return;
    }

    ply_file_path = p_path;

    if (auto_load && is_inside_tree()) {
        _load_asset();
    }

    notify_property_list_changed();
    update_configuration_warnings();
}

void GaussianSplatNode3D::set_splat_asset(const Ref<GaussianSplatAsset> &p_asset) {
    if (splat_asset == p_asset) {
        return;
    }

    if (splat_asset.is_valid() && splat_asset->is_connected("changed", callable_mp(this, &GaussianSplatNode3D::_on_asset_changed))) {
        splat_asset->disconnect("changed", callable_mp(this, &GaussianSplatNode3D::_on_asset_changed));
    }

    _clear_asset();
    splat_asset = p_asset;

    if (splat_asset.is_valid()) {
        if (!splat_asset->is_connected("changed", callable_mp(this, &GaussianSplatNode3D::_on_asset_changed))) {
            splat_asset->connect("changed", callable_mp(this, &GaussianSplatNode3D::_on_asset_changed));
        }
        _update_asset();
    }

    notify_property_list_changed();
    update_configuration_warnings();
}

void GaussianSplatNode3D::set_auto_load(bool p_enabled) {
    auto_load = p_enabled;

    if (auto_load && is_inside_tree() && !ply_file_path.is_empty() && splat_asset.is_null()) {
        _load_asset();
    }
}

void GaussianSplatNode3D::reload_asset() {
    _load_asset();
}

void GaussianSplatNode3D::set_splat_data(const PackedVector3Array &p_positions,
        const PackedColorArray &p_colors,
        const PackedVector3Array &p_scales,
        const PackedFloat32Array &p_opacities,
        const TypedArray<Quaternion> &p_rotations,
        const PackedFloat32Array &p_spherical_harmonics,
        const PackedInt32Array &p_palette_ids,
        const PackedInt32Array &p_painterly_flags,
        const PackedVector3Array &p_normals,
        const PackedVector2Array &p_brush_axes,
        const PackedFloat32Array &p_stroke_ages,
        bool p_is_2d_mode) {
    int splat_count = p_positions.size();
    if (!_validate_splat_data_inputs(splat_count, p_positions, p_colors, p_scales, p_opacities, p_rotations,
                p_spherical_harmonics, p_palette_ids, p_painterly_flags, p_normals, p_brush_axes, p_stroke_ages)) {
        return;
    }

    if (!_ensure_renderer_for_manual_data()) {
        return;
    }

    _reset_manual_splat_state();
    _ensure_renderer_data_for_splats(splat_count, p_positions);
    _apply_optional_splat_arrays(splat_count, p_colors, p_scales, p_opacities, p_rotations,
            p_spherical_harmonics, p_palette_ids, p_painterly_flags, p_normals, p_brush_axes, p_stroke_ages);
    renderer_data->set_2d_mode(p_is_2d_mode);
    _populate_runtime_asset_from_renderer_data();
    _compute_manual_splat_bounds(splat_count, p_positions, p_scales);
    _finalize_manual_splat_setup(splat_count);
}

bool GaussianSplatNode3D::_ensure_renderer_for_manual_data() {
    _ensure_renderer();
    if (!renderer.is_valid()) {
        GS_LOG_RENDERER_ERROR("[GaussianSplatNode3D] Unable to initialize shared GaussianSplatRenderer for manual splat data.");
        return false;
    }
    return true;
}

bool GaussianSplatNode3D::_validate_splat_data_inputs(int splat_count,
        const PackedVector3Array &p_positions,
        const PackedColorArray &p_colors,
        const PackedVector3Array &p_scales,
        const PackedFloat32Array &p_opacities,
        const TypedArray<Quaternion> &p_rotations,
        const PackedFloat32Array &p_spherical_harmonics,
        const PackedInt32Array &p_palette_ids,
        const PackedInt32Array &p_painterly_flags,
        const PackedVector3Array &p_normals,
        const PackedVector2Array &p_brush_axes,
        const PackedFloat32Array &p_stroke_ages) const {
    (void)p_positions;

    ERR_FAIL_COND_V_MSG(splat_count <= 0, false,
            "[GaussianSplatNode3D] Positions array must contain at least one splat.");

    if (!p_scales.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_scales.size() != splat_count, false,
                "[GaussianSplatNode3D] Scale array must match positions length.");
    }
    if (p_rotations.size() > 0) {
        ERR_FAIL_COND_V_MSG(p_rotations.size() != splat_count, false,
                "[GaussianSplatNode3D] Rotation array must match positions length.");
    }
    if (!p_opacities.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_opacities.size() != splat_count, false,
                "[GaussianSplatNode3D] Opacity array must match positions length.");
    }
    if (p_spherical_harmonics.is_empty() && !p_colors.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_colors.size() != splat_count, false,
                "[GaussianSplatNode3D] Color array must match positions length.");
    }
    if (!p_palette_ids.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_palette_ids.size() != splat_count, false,
                "[GaussianSplatNode3D] Palette id array must match positions length.");
    }
    if (!p_painterly_flags.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_painterly_flags.size() != splat_count, false,
                "[GaussianSplatNode3D] Painterly flags array must match positions length.");
    }
    if (!p_normals.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_normals.size() != splat_count, false,
                "[GaussianSplatNode3D] Normal array must match positions length.");
    }
    if (!p_brush_axes.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_brush_axes.size() != splat_count, false,
                "[GaussianSplatNode3D] Brush axes array must match positions length.");
    }
    if (!p_stroke_ages.is_empty()) {
        ERR_FAIL_COND_V_MSG(p_stroke_ages.size() != splat_count, false,
                "[GaussianSplatNode3D] Stroke ages array must match positions length.");
    }

    return true;
}

void GaussianSplatNode3D::_reset_manual_splat_state() {
    if (splat_asset.is_valid()) {
        const Callable changed_callable = callable_mp(this, &GaussianSplatNode3D::_on_asset_changed);
        if (splat_asset->is_connected("changed", changed_callable)) {
            splat_asset->disconnect("changed", changed_callable);
        }
    }
    splat_asset.unref();
    asset_loading = false;
    ply_file_path = String();
}

void GaussianSplatNode3D::_ensure_renderer_data_for_splats(int splat_count, const PackedVector3Array &p_positions) {
    if (renderer_data.is_null()) {
        renderer_data.instantiate();
    }

    renderer_data->resize(splat_count);
    renderer_data->set_positions(p_positions);
}

void GaussianSplatNode3D::_apply_optional_splat_arrays(int splat_count,
        const PackedColorArray &p_colors,
        const PackedVector3Array &p_scales,
        const PackedFloat32Array &p_opacities,
        const TypedArray<Quaternion> &p_rotations,
        const PackedFloat32Array &p_spherical_harmonics,
        const PackedInt32Array &p_palette_ids,
        const PackedInt32Array &p_painterly_flags,
        const PackedVector3Array &p_normals,
        const PackedVector2Array &p_brush_axes,
        const PackedFloat32Array &p_stroke_ages) {
    if (!p_scales.is_empty()) {
        renderer_data->set_scales(p_scales);
    }

    if (p_rotations.size() > 0) {
        renderer_data->set_rotations(p_rotations);
    }

    if (!p_opacities.is_empty()) {
        renderer_data->set_opacities(p_opacities);
    } else if (p_colors.size() == splat_count) {
        PackedFloat32Array opacity_from_color;
        opacity_from_color.resize(splat_count);
        float *opacity_ptr = opacity_from_color.ptrw();
        for (int i = 0; i < splat_count; i++) {
            opacity_ptr[i] = CLAMP(p_colors[i].a, 0.0f, 1.0f);
        }
        renderer_data->set_opacities(opacity_from_color);
    }

    if (!p_spherical_harmonics.is_empty()) {
        renderer_data->set_spherical_harmonics(p_spherical_harmonics);
    } else if (!p_colors.is_empty()) {
        PackedFloat32Array sh_dc;
        sh_dc.resize(splat_count * 3);
        float *sh_ptr = sh_dc.ptrw();
        for (int i = 0; i < splat_count; i++) {
            const Color color = p_colors[i];
            sh_ptr[i * 3 + 0] = color.r;
            sh_ptr[i * 3 + 1] = color.g;
            sh_ptr[i * 3 + 2] = color.b;
        }
        renderer_data->set_spherical_harmonics(sh_dc);
    }

    if (!p_palette_ids.is_empty()) {
        renderer_data->set_palette_ids(p_palette_ids);
    }

    if (!p_painterly_flags.is_empty()) {
        renderer_data->set_painterly_flags(p_painterly_flags);
    }

    if (!p_normals.is_empty()) {
        renderer_data->set_normals(p_normals);
    }

    if (!p_brush_axes.is_empty()) {
        renderer_data->set_brush_axes(p_brush_axes);
    }

    if (!p_stroke_ages.is_empty()) {
        renderer_data->set_stroke_ages(p_stroke_ages);
    }
}

void GaussianSplatNode3D::_populate_runtime_asset_from_renderer_data() {
    if (runtime_asset.is_null()) {
        runtime_asset.instantiate();
        runtime_asset->set_asset_type(GaussianSplatAsset::ASSET_TYPE_DYNAMIC);
    }
    Error asset_err = runtime_asset->populate_from_gaussian_data(renderer_data);
    if (asset_err != OK) {
        GS_LOG_WARN_DEFAULT(vformat("[GaussianSplatNode3D] Failed to build runtime asset from splat data (err=%d).", asset_err));
    }
}

void GaussianSplatNode3D::_compute_manual_splat_bounds(int splat_count,
        const PackedVector3Array &p_positions,
        const PackedVector3Array &p_scales) {
    Vector3 min_pos = Vector3(INFINITY, INFINITY, INFINITY);
    Vector3 max_pos = Vector3(-INFINITY, -INFINITY, -INFINITY);
    float max_scale_extent = 0.0f;
    const bool has_scales = !p_scales.is_empty() && p_scales.size() == splat_count;

    for (int i = 0; i < splat_count; i++) {
        const Vector3 pos = p_positions[i];
        min_pos = min_pos.min(pos);
        max_pos = max_pos.max(pos);

        if (has_scales) {
            const Vector3 &scale = p_scales[i];
            float local_max = MAX(Math::abs(scale.x), MAX(Math::abs(scale.y), Math::abs(scale.z)));
            max_scale_extent = MAX(max_scale_extent, local_max);
        }
    }

    if (splat_count > 0) {
        const float scale_multiplier = 3.0f;
        Vector3 inflation(max_scale_extent * scale_multiplier, max_scale_extent * scale_multiplier, max_scale_extent * scale_multiplier);
        min_pos -= inflation;
        max_pos += inflation;

        Vector3 size = max_pos - min_pos;
        local_aabb = AABB(min_pos, size);
    } else {
        local_aabb = AABB();
    }
}

void GaussianSplatNode3D::_finalize_manual_splat_setup(int splat_count) {
    total_splat_count = splat_count;
    visible_splat_count = 0;
    gpu_memory_mb = renderer_data->get_memory_usage() / (1024.0f * 1024.0f);

    bounds_dirty = true;
    _update_bounds();
    _update_render_instance();
    _sync_gaussian_storage();
    _register_shared_renderer();
    _mark_render_state_dirty();
    update_gizmos();
    update_configuration_warnings();
}

void GaussianSplatNode3D::set_quality_preset(QualityPreset p_preset) {
    if (quality_preset == p_preset) {
        return;
    }

    quality_preset = p_preset;
    _apply_quality_preset();
    _update_quality_settings();
    _mark_render_state_dirty();
    _update_instance_params_in_director();

    notify_property_list_changed();
}

void GaussianSplatNode3D::set_lod_bias(float p_bias) {
    lod_bias = CLAMP(p_bias, 0.1f, 4.0f);
    _update_quality_settings();
    _mark_render_state_dirty();
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_max_render_distance(float p_distance) {
    max_render_distance = MAX(0.0f, p_distance);
    _update_visibility();
    _update_quality_settings();
    _mark_render_state_dirty();
}

void GaussianSplatNode3D::set_max_splat_count(int p_count) {
    max_splat_count = MAX(1000, p_count);
    _update_quality_settings();
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_enable_painterly(bool p_enabled) {
    if (enable_painterly == p_enabled) {
        return;
    }

    enable_painterly = p_enabled;
    _apply_painterly_settings();
    _update_quality_settings();
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
    notify_property_list_changed();
}

void GaussianSplatNode3D::set_edge_threshold(float p_threshold) {
    if (_is_multi_instance_shared_renderer_active(renderer)) {
        return;
    }
    edge_threshold = CLAMP(p_threshold, 0.0f, 1.0f);
    _apply_painterly_settings();
    _update_quality_settings();
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
}

void GaussianSplatNode3D::set_stroke_opacity(float p_opacity) {
    if (_is_multi_instance_shared_renderer_active(renderer)) {
        return;
    }
    stroke_opacity = CLAMP(p_opacity, 0.0f, 1.0f);
    _apply_painterly_settings();
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
}

void GaussianSplatNode3D::set_stroke_width(float p_width) {
    if (_is_multi_instance_shared_renderer_active(renderer)) {
        return;
    }
    stroke_width = CLAMP(p_width, 0.1f, 5.0f);
    _apply_painterly_settings();
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
}

void GaussianSplatNode3D::set_color_variation(float p_variation) {
    // Compatibility-only placeholder: renderer currently has no color-variation control.
    color_variation = CLAMP(p_variation, 0.0f, 0.5f);
}

void GaussianSplatNode3D::set_temporal_blend(float p_blend) {
    if (_is_multi_instance_shared_renderer_active(renderer)) {
        return;
    }
    temporal_blend = CLAMP(p_blend, 0.01f, 1.0f);
    _apply_painterly_settings();
    _update_quality_settings();
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
}

void GaussianSplatNode3D::set_painterly_seed(uint32_t p_seed) {
    if (_is_multi_instance_shared_renderer_active(renderer)) {
        return;
    }
    painterly_seed = MIN(p_seed, (uint32_t)65535);
    _apply_painterly_settings();
    _update_quality_settings();
    _mark_render_state_dirty();
}

void GaussianSplatNode3D::set_update_mode(ViewportUpdateMode p_mode) {
    ViewportUpdateMode previous_mode = update_mode;
    update_mode = p_mode;

    if (update_mode == UPDATE_MODE_WHEN_PARENT_VISIBLE) {
        if (is_inside_tree()) {
            _update_parent_visibility_tracking();
        } else {
            parent_visible = true;
        }
    } else if (previous_mode == UPDATE_MODE_WHEN_PARENT_VISIBLE) {
        _clear_parent_visibility_tracking();
    }

    // Enable or disable process based on update mode.
    // Only enable if manager is unavailable (manager handles updates via frame_pre_draw).
    if (is_inside_tree() && !GaussianSplatManager::get_singleton()) {
        set_process(update_mode != UPDATE_MODE_MANUAL);
    }
}

void GaussianSplatNode3D::set_cast_shadow(bool p_enabled) {
    if (cast_shadow == p_enabled) {
        return;
    }

    cast_shadow = p_enabled;
    if (render_instance.is_valid()) {
        if (RenderingServer *rs = RS::get_singleton()) {
            rs->instance_geometry_set_cast_shadows_setting(render_instance,
                    cast_shadow ? RS::SHADOW_CASTING_SETTING_ON : RS::SHADOW_CASTING_SETTING_OFF);
        }
    }
    _sync_gaussian_storage();
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_use_frustum_culling(bool p_enabled) {
    use_frustum_culling = p_enabled;
    _update_visibility();
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
}

void GaussianSplatNode3D::set_use_occlusion_culling(bool p_enabled) {
    // Compatibility-only placeholder: renderer currently has no dedicated occlusion-culling toggle.
    use_occlusion_culling = p_enabled;
}

void GaussianSplatNode3D::set_opacity(float p_opacity) {
    opacity = CLAMP(p_opacity, 0.0f, 1.0f);
    if (renderer.is_valid()) {
        _apply_renderer_settings();
    }
    _mark_render_state_dirty();
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_wind_override_enabled(bool p_enabled) {
    if (wind_override_enabled == p_enabled) {
        return;
    }
    wind_override_enabled = p_enabled;
    notify_property_list_changed();
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_wind_enabled(bool p_enabled) {
    if (wind_enabled == p_enabled) {
        return;
    }
    wind_enabled = p_enabled;
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_wind_strength(float p_strength) {
    const float clamped_strength = MAX(0.0f, p_strength);
    if (Math::is_equal_approx(wind_strength, clamped_strength)) {
        return;
    }
    wind_strength = clamped_strength;
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_wind_direction(const Vector3 &p_direction) {
    if (wind_direction.is_equal_approx(p_direction)) {
        return;
    }
    wind_direction = p_direction;
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_wind_frequency(float p_frequency) {
    const float clamped_frequency = MAX(0.0f, p_frequency);
    if (Math::is_equal_approx(wind_frequency, clamped_frequency)) {
        return;
    }
    wind_frequency = clamped_frequency;
    _update_instance_params_in_director();
}

void GaussianSplatNode3D::set_preview_enabled(bool p_enabled) {
    preview_enabled = p_enabled;
    _update_visibility();
    update_gizmos();
}

void GaussianSplatNode3D::set_show_bounds(bool p_show) {
    show_bounds = p_show;
    update_gizmos();
}

void GaussianSplatNode3D::set_show_statistics(bool p_show) {
    show_statistics = p_show;
    last_stats_inspector_refresh_usec = 0;
    notify_property_list_changed();
    update_gizmos();
}

void GaussianSplatNode3D::set_show_tile_grid(bool p_show) {
    debug_helper.set_show_tile_grid(p_show);
}

void GaussianSplatNode3D::set_show_density_heatmap(bool p_show) {
    debug_helper.set_show_density_heatmap(p_show);
}

void GaussianSplatNode3D::set_show_performance_hud(bool p_show) {
    debug_helper.set_show_performance_hud(p_show);
}

void GaussianSplatNode3D::set_show_lod_spheres(bool p_show) {
    debug_helper.set_show_lod_spheres(p_show);
}

void GaussianSplatNode3D::set_show_performance_overlay(bool p_show) {
    debug_helper.set_show_performance_overlay(p_show);
}

void GaussianSplatNode3D::set_debug_overlay_opacity(float p_opacity) {
    debug_helper.set_debug_overlay_opacity(p_opacity);
}

void GaussianSplatNode3D::set_debug_draw_mode(DebugDrawMode p_mode) {
    debug_helper.set_debug_draw_mode(static_cast<int>(p_mode));
}

void GaussianSplatNode3D::set_runtime_preview_enabled(bool p_enabled) {
    debug_helper.set_runtime_preview_enabled(p_enabled);
}

void GaussianSplatNode3D::set_show_residency_hud(bool p_show) {
    debug_helper.set_show_residency_hud(p_show);
}

Dictionary GaussianSplatNode3D::get_statistics() const {
    Dictionary stats;
    stats["update_time_ms"] = last_update_time_ms;
    stats["gpu_memory_mb"] = gpu_memory_mb;
    stats["bounds"] = local_aabb;
    stats["debug_draw_mode"] = debug_draw_mode;
    stats["debug_preview_mode"] = debug_draw_mode;
    stats["show_lod_spheres"] = show_lod_spheres;
    stats["show_performance_overlay"] = show_performance_overlay;
    stats["preview_enabled"] = preview_enabled;

    if (renderer.is_valid()) {
        Dictionary render_stats = renderer->get_render_stats();
        if (render_stats.has("visible_splats")) {
            stats["renderer_visible_splats"] = render_stats["visible_splats"];
        }
        if (render_stats.has("total_splats")) {
            stats["renderer_total_splats"] = render_stats["total_splats"];
        }
        Array keys = render_stats.keys();
        for (int i = 0; i < keys.size(); i++) {
            const Variant &key = keys[i];
            stats[key] = render_stats[key];
        }
    }

    // Keep node-level counters authoritative for this API even when a shared renderer
    // reports aggregate world metrics.
    stats["visible_splats"] = visible_splat_count;
    stats["total_splats"] = total_splat_count;

    return stats;
}

AABB GaussianSplatNode3D::get_aabb() const {
    return local_aabb;
}

void GaussianSplatNode3D::update_splats() {
    static int update_call_count = 0;
    update_call_count++;

    if (!is_inside_tree()) {
#ifndef GS_SILENCE_LOGS
        if (_is_frame_log_enabled() && update_call_count <= 10) {
            GS_LOG_RENDERER_DEBUG("[update_splats] Not in tree, returning");
        }
#endif
        return;
    }

    uint64_t start_time = OS::get_singleton()->get_ticks_usec();
    RenderingServer *rs = RS::get_singleton();

    _update_visibility();
    _update_render_instance();

    const int asset_splat_count = splat_asset.is_valid() ? splat_asset->get_splat_count() : 0;
    const int procedural_splat_count = renderer_data.is_valid() ? renderer_data->get_count() : 0;

    _log_update_splats_call(update_call_count, asset_splat_count, procedural_splat_count);
    if (_handle_empty_splat_frame(rs, start_time, asset_splat_count, procedural_splat_count)) {
        return;
    }

    _ensure_renderer();
    _apply_renderer_settings();

    // DEBUG: Check render_state_dirty before upload
#ifndef GS_SILENCE_LOGS
    if (_is_frame_log_enabled() && update_call_count <= 10) {
        GS_LOG_RENDERER_DEBUG(vformat("[update_splats] render_state_dirty=%s before upload check", render_state_dirty ? "true" : "false"));
    }
#endif

    if (render_state_dirty) {
        _upload_asset_to_renderer();
        render_state_dirty = false;
    }

    _sync_renderer_splat_counts(asset_splat_count, procedural_splat_count);
    _update_renderer_gpu_memory();
    _update_viewport_render_state(rs, update_call_count);
    _apply_render_instance_state(rs);
    _sync_gaussian_storage();
    _finalize_update_splats(start_time);
}

void GaussianSplatNode3D::_log_update_splats_call(int update_call_count, int asset_splat_count, int procedural_splat_count) const {
#ifndef GS_SILENCE_LOGS
    if (_is_frame_log_enabled() && update_call_count <= 10) {
        GS_LOG_RENDERER_DEBUG(vformat("[update_splats] Call #%d: asset_splats=%d, procedural_splats=%d",
            update_call_count, asset_splat_count, procedural_splat_count));
    }
#else
    (void)update_call_count;
    (void)asset_splat_count;
    (void)procedural_splat_count;
#endif
}

bool GaussianSplatNode3D::_handle_empty_splat_frame(RenderingServer *rs, uint64_t start_time,
        int asset_splat_count, int procedural_splat_count) {
    if (asset_splat_count != 0 || procedural_splat_count != 0) {
        return false;
    }

    visible_splat_count = 0;
    total_splat_count = 0;

    if (rs && render_instance.is_valid()) {
        rs->instance_set_visible(render_instance, false);
    }

    _finalize_update_splats(start_time);
    return true;
}

void GaussianSplatNode3D::_sync_renderer_splat_counts(int asset_splat_count, int procedural_splat_count) {
    if (!renderer.is_valid()) {
        return;
    }

    const int expected_total = asset_splat_count > 0 ? asset_splat_count : procedural_splat_count;
    total_splat_count = expected_total > 0 ? (uint32_t)expected_total : 0;

    Dictionary stats = renderer->get_render_stats();
    uint32_t renderer_visible = 0;
    if (stats.has("visible_splats")) {
        renderer_visible = (uint32_t)(int64_t)stats["visible_splats"];
    } else {
        renderer_visible = renderer->get_visible_splat_count();
    }

    if (total_splat_count == 0) {
        visible_splat_count = 0;
    } else {
        visible_splat_count = MIN(renderer_visible, total_splat_count);
    }
}

void GaussianSplatNode3D::_update_renderer_gpu_memory() {
    if (renderer_data.is_valid()) {
        gpu_memory_mb = renderer_data->get_memory_usage() / (1024.0f * 1024.0f);
    }
}

void GaussianSplatNode3D::_update_viewport_render_state(RenderingServer *rs, int update_call_count) {
    (void)update_call_count;
    if (!renderer.is_valid()) {
        return;
    }

    Viewport *viewport = Engine::get_singleton()->is_editor_hint() ? _find_editor_scene_viewport() : get_viewport();
    _update_cached_render_target(viewport);

    if (!viewport || !rs) {
        return;
    }

    Size2i viewport_size = Size2i(cached_viewport_size.x, cached_viewport_size.y);
    if (viewport_size.x <= 16 || viewport_size.y <= 16) {
        Size2i fallback_size = viewport->get_visible_rect().size;
        if (fallback_size.x <= 16 || fallback_size.y <= 16) {
            if (DisplayServer::get_singleton()) {
                fallback_size = DisplayServer::get_singleton()->window_get_size();
            }
            if (fallback_size.x <= 16 || fallback_size.y <= 16) {
                fallback_size = Size2i(1280, 720);
            }
        }
        viewport_size = fallback_size;
        cached_viewport_size = Vector2i(fallback_size.x, fallback_size.y);
    }

    if (!cached_viewport_render_target.is_valid()) {
        RID viewport_rid = viewport->get_viewport_rid();
#ifndef GS_SILENCE_LOGS
        if (_is_frame_log_enabled() && update_call_count <= 10) {
            GS_LOG_RENDERER_DEBUG(vformat("[update_splats] viewport_rid valid=%s", viewport_rid.is_valid() ? "true" : "false"));
        }
#endif
        if (viewport_rid.is_valid()) {
            cached_viewport_render_target = rs->viewport_get_render_target(viewport_rid);
            cached_viewport_render_texture = rs->viewport_get_texture(viewport_rid);
#ifndef GS_SILENCE_LOGS
            if (_is_frame_log_enabled() && update_call_count <= 10) {
                GS_LOG_RENDERER_DEBUG(vformat("[update_splats] Got render_target valid=%s, texture valid=%s",
                    cached_viewport_render_target.is_valid() ? "true" : "false",
                    cached_viewport_render_texture.is_valid() ? "true" : "false"));
            }
#endif
            if (cached_viewport_render_target.is_valid()) {
                viewport_texture_state = ViewportTextureState::READY;
            }
        }
    }

#ifndef GS_SILENCE_LOGS
    if (_is_frame_log_enabled() && update_call_count <= 10) {
        GS_LOG_RENDERER_DEBUG(vformat("[update_splats] viewport_texture_state=%d (READY=%d), cached_render_target valid=%s",
            (int)viewport_texture_state, (int)ViewportTextureState::READY,
            cached_viewport_render_target.is_valid() ? "true" : "false"));
    }
#endif

    // Always update the renderer's camera even when the viewport texture
    // is not ready yet.  Without this, viewport size changes or bootstrap
    // delays leave the renderer with stale camera data, causing the GPU
    // culler to use an outdated frustum and cull all splats.
    Camera3D *camera = viewport->get_camera_3d();
    Transform3D camera_to_world_transform = camera ? camera->get_camera_transform() : get_global_transform();
    Projection camera_projection;
    if (camera) {
        camera_projection = camera->get_camera_projection();
    } else {
        float aspect = viewport_size.y > 0 ? viewport_size.x / (float)viewport_size.y : 1.0f;
        float far_plane = max_render_distance > 0.0f ? max_render_distance : 1000.0f;
        camera_projection.set_perspective(60.0f, aspect, 0.1f, far_plane);
    }

    renderer->set_camera_transform(camera_to_world_transform);
    renderer->set_camera_projection(camera_projection);

    if (viewport_texture_state != ViewportTextureState::READY || !cached_viewport_render_target.is_valid()) {
#ifndef GS_SILENCE_LOGS
        if (_is_frame_log_enabled() && update_call_count <= 10) {
            GS_LOG_RENDERER_DEBUG("[update_splats] Queueing viewport bootstrap...");
        }
#endif
        _queue_viewport_bootstrap();
        return;
    }

    if (OS::get_singleton()->has_feature("headless")) {
        renderer->tick_streaming_only(camera_to_world_transform, camera_projection);
    }
}

void GaussianSplatNode3D::_apply_render_instance_state(RenderingServer *rs) {
    if (rs && render_instance.is_valid()) {
        rs->instance_set_visible(render_instance, parent_visible && visible_in_viewport && is_visible_in_tree());
        rs->instance_set_custom_aabb(render_instance, local_aabb);
        rs->instance_set_transform(render_instance, get_global_transform());
    }
}

void GaussianSplatNode3D::_finalize_update_splats(uint64_t start_time) {
    const uint64_t now_usec = OS::get_singleton()->get_ticks_usec();
    last_update_time_ms = (now_usec - start_time) / 1000.0f;

#ifdef TOOLS_ENABLED
    if (Engine::get_singleton()->is_editor_hint() && show_statistics && splat_asset.is_valid()) {
        static constexpr uint64_t STATS_INSPECTOR_REFRESH_INTERVAL_USEC = 250000;
        if (last_stats_inspector_refresh_usec == 0 ||
                now_usec - last_stats_inspector_refresh_usec >= STATS_INSPECTOR_REFRESH_INTERVAL_USEC) {
            last_stats_inspector_refresh_usec = now_usec;
            notify_property_list_changed();
        }
    }
#endif
}

void GaussianSplatNode3D::process_gaussian_render() {
    if (!is_inside_tree()) {
        return;
    }

    _update_visibility();

    if (renderer.is_valid()) {
        const bool shared_renderer_multi_instance = _is_multi_instance_shared_renderer_active(renderer);
        if (shared_renderer_multi_instance_state != shared_renderer_multi_instance) {
            shared_renderer_multi_instance_state = shared_renderer_multi_instance;
            _apply_renderer_settings();
            notify_property_list_changed();
        }
    } else {
        shared_renderer_multi_instance_state = false;
    }

    bool should_update = false;
    switch (update_mode) {
        case UPDATE_MODE_ALWAYS:
            should_update = true;
            break;
        case UPDATE_MODE_WHEN_VISIBLE:
            should_update = visible_in_viewport;
            break;
        case UPDATE_MODE_WHEN_PARENT_VISIBLE:
            should_update = parent_visible && is_visible_in_tree();
            break;
        case UPDATE_MODE_MANUAL:
            should_update = false;
            break;
    }

    if (should_update) {
        update_splats();
        return;
    }

    RenderingServer *rs = RS::get_singleton();
    if (rs && render_instance.is_valid()) {
        rs->instance_set_visible(render_instance, parent_visible && visible_in_viewport && is_visible_in_tree());
    }
}

void GaussianSplatNode3D::force_update() {
    update_splats();
    emit_signal("viewport_visibility_changed", visible_in_viewport);
}

PackedStringArray GaussianSplatNode3D::get_configuration_warnings() const {
    PackedStringArray warnings = Node3D::get_configuration_warnings();

    if (ply_file_path.is_empty() && splat_asset.is_null() && renderer_data.is_null() && runtime_asset.is_null()) {
        warnings.push_back("No Gaussian splat file or asset assigned. Set a file path (.ply or .spz) or assign a GaussianSplatAsset.");
    }

    if (!ply_file_path.is_empty() && !FileAccess::exists(ply_file_path)) {
        warnings.push_back(vformat("Gaussian splat file not found: %s", ply_file_path));
    }

    if (max_render_distance <= 0.0f) {
        warnings.push_back("Max render distance is 0. Splats will not be visible.");
    }

    const Vector3 scale = get_global_transform().basis.get_scale();
    const float sx = Math::abs(scale.x);
    const float sy = Math::abs(scale.y);
    const float sz = Math::abs(scale.z);
    if (!Math::is_equal_approx(sx, sy) || !Math::is_equal_approx(sx, sz) || !Math::is_equal_approx(sy, sz)) {
        warnings.push_back("Non-uniform scale detected. Instance pipeline uses uniform scale for splats.");
    }

    return warnings;
}

void GaussianSplatNode3D::_load_asset() {
    if (_is_frame_log_enabled()) {
        GS_LOG_RENDERER_DEBUG(vformat("[NODE] _load_asset called, ply_file_path=%s auto_load=%s",
                ply_file_path, auto_load ? "true" : "false"));
    }
    asset_helper.load_asset();
}

void GaussianSplatNode3D::_update_asset() {
    asset_helper.update_asset();
    update_gizmos();
}

void GaussianSplatNode3D::_clear_asset() {
    asset_helper.clear_asset();
}

void GaussianSplatNode3D::_update_bounds() {
    if (!is_inside_tree()) {
        return;
    }

    Transform3D global_transform = get_global_transform();
    world_aabb = global_transform.xform(local_aabb);
    bounds_dirty = false;
    _ensure_gaussian_base();
    _sync_gaussian_storage();
}

void GaussianSplatNode3D::_update_visibility() {
    const bool previous_visibility = visible_in_viewport;
    bool new_visibility = false;

    // In editor, preview_enabled controls actual rendering visibility (not just gizmos).
    // At runtime, preview_enabled is ignored so splats always render when visible.
#ifdef TOOLS_ENABLED
    const bool preview_blocked = Engine::get_singleton()->is_editor_hint() && !preview_enabled;
#else
    const bool preview_blocked = false;
#endif

    if (is_inside_tree() && parent_visible && is_visible_in_tree() && !preview_blocked) {
        if (bounds_dirty) {
            _update_bounds();
        }

        // CPU-side distance and frustum culling is intentionally disabled.
        // viewport->get_camera_3d() can return a camera that doesn't match
        // the rendering pipeline's actual camera (wrong in editor, potentially
        // stale at runtime), causing splats to be incorrectly hidden.  When
        // visible_in_viewport is false the render instance is hidden and the
        // GPU cull shader never runs, so it cannot correct the mistake.
        // The GPU cull shader already handles frustum culling with the
        // correct camera from the rendering pipeline.
        new_visibility = true;
    }

    visible_in_viewport = new_visibility;

    if (render_instance.is_valid()) {
        if (RenderingServer *rs = RS::get_singleton()) {
            rs->instance_set_visible(render_instance, parent_visible && visible_in_viewport && is_visible_in_tree());
        }
    }

    if (previous_visibility != visible_in_viewport) {
        _update_instance_params_in_director();
        emit_signal("viewport_visibility_changed", visible_in_viewport);
    }
}

void GaussianSplatNode3D::_update_quality_settings() {
    quality_helper.update_quality_settings();
}

void GaussianSplatNode3D::_apply_quality_tier_limits(int &effective_max_splats, int &effective_max_gpu_mb,
        int &effective_target_gpu_mb, float &effective_load_ahead, float &effective_unload,
        int &effective_concurrent_loads, int &effective_stream_budget_ms) const {
    quality_helper.apply_quality_tier_limits(effective_max_splats, effective_max_gpu_mb, effective_target_gpu_mb,
            effective_load_ahead, effective_unload, effective_concurrent_loads, effective_stream_budget_ms);
}

void GaussianSplatNode3D::_apply_quality_lod_config(float lod0_distance, float lod1_distance, float lod2_distance, float lod3_distance,
        float effective_distance, uint32_t max_budget, uint32_t min_budget, float importance_threshold, float size_cull_threshold,
        bool smooth_transitions, float transition_time, float target_fps, float quality_rate,
        bool temporal_coherence) {
    quality_helper.apply_quality_lod_config(lod0_distance, lod1_distance, lod2_distance, lod3_distance,
            effective_distance, max_budget, min_budget, importance_threshold, size_cull_threshold,
            smooth_transitions, transition_time, target_fps, quality_rate, temporal_coherence);
}

void GaussianSplatNode3D::_apply_streaming_config_values(int effective_max_gpu_mb, int effective_target_gpu_mb,
        float effective_distance, float effective_load_ahead, float effective_unload, int effective_concurrent_loads,
        bool predictive_loading, float prediction_time, int lod_level_count, float lod_distance_multiplier,
        bool adaptive_quality, int effective_stream_budget_ms, bool async_loading, bool compression) {
    quality_helper.apply_streaming_config_values(effective_max_gpu_mb, effective_target_gpu_mb, effective_distance,
            effective_load_ahead, effective_unload, effective_concurrent_loads, predictive_loading,
            prediction_time, lod_level_count, lod_distance_multiplier, adaptive_quality,
            effective_stream_budget_ms, async_loading, compression);
}

void GaussianSplatNode3D::_apply_painterly_settings() {
    if (!enable_painterly || !splat_asset.is_valid()) {
        return;
    }

    // Configure painterly manager
    GaussianSplatting::PainterlyManager::Settings settings;
    settings.base_seed = painterly_seed;
    const float safe_blend = MAX(temporal_blend, 0.01f);
    settings.blend_rate = 1.0f / safe_blend;
    settings.hold_strength = edge_threshold;
    settings.jitter_amplitude = stroke_width * 0.1f;

    painterly_manager.configure(settings);
}

void GaussianSplatNode3D::_ensure_renderer() {
    renderer_helper.ensure_renderer();
}

void GaussianSplatNode3D::_ensure_debug_hud_control() {
    if (debug_hud_control) {
        return;
    }
    if (!is_inside_tree()) {
        return;
    }

    if (!debug_hud_layer) {
        debug_hud_layer = memnew(CanvasLayer);
        debug_hud_layer->set_name("GaussianSplatDebugHUDLayer");
        add_child(debug_hud_layer);
    }

    debug_hud_control = memnew(GaussianSplatDebugHUD);
    debug_hud_control->set_name("GaussianSplatDebugHUD");
    debug_hud_control->set_splat_node(this);
    debug_hud_control->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
    debug_hud_layer->add_child(debug_hud_control);
}

void GaussianSplatNode3D::_update_debug_hud_visibility() {
    bool should_show_hud = show_performance_hud || show_residency_hud;
    if (renderer.is_valid()) {
        should_show_hud = renderer->is_debug_show_performance_hud() || renderer->is_debug_show_residency_hud();
    }
    if (!should_show_hud) {
        if (debug_hud_control) {
            debug_hud_control->set_visible(false);
            debug_hud_control->set_process(false);
        }
        return;
    }

    _ensure_debug_hud_control();
    if (!debug_hud_control) {
        return;
    }

    debug_hud_control->set_visible(true);
    debug_hud_control->set_process(true);
    debug_hud_control->set_splat_node(this);
    debug_hud_control->refresh_stats();
}

void GaussianSplatNode3D::_apply_renderer_settings() {
    renderer_helper.apply_renderer_settings();
}

void GaussianSplatNode3D::_update_render_instance() {
    RenderingServer *rs = RS::get_singleton();
    if (!rs) {
        return;
    }

    if (!render_instance.is_valid()) {
        render_instance = rs->instance_create();
        rs->instance_attach_object_instance_id(render_instance, get_instance_id());
    }

    if (is_inside_tree() && is_inside_world()) {
        Ref<World3D> world = get_world_3d();
        rs->instance_set_scenario(render_instance, world->get_scenario());

        Transform3D xform = get_global_transform();
        rs->instance_set_transform(render_instance, xform);
        rs->instance_set_visible(render_instance, parent_visible && visible_in_viewport && is_visible_in_tree());

        // Instance pipeline uses per-instance transforms; no global instance transform is required.
    }

    rs->instance_set_custom_aabb(render_instance, local_aabb);
    rs->instance_geometry_set_cast_shadows_setting(render_instance,
            cast_shadow ? RS::SHADOW_CASTING_SETTING_ON : RS::SHADOW_CASTING_SETTING_OFF);

    _ensure_gaussian_base();
    _set_instance_base(gaussian_base);
    _sync_gaussian_storage();
}

void GaussianSplatNode3D::_ensure_gaussian_base() {
    if (gaussian_base.is_valid()) {
        return;
    }

    RendererRD::GaussianSplatStorage *storage = RendererRD::GaussianSplatStorage::get_singleton();
    if (!storage) {
        return;
    }

	gaussian_base = storage->gaussian_allocate();
	storage->gaussian_initialize(gaussian_base);
#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED
	storage->gaussian_set_renderer(gaussian_base, renderer);
#endif
	storage->gaussian_set_aabb(gaussian_base, world_aabb);
	storage->gaussian_set_casts_shadow(gaussian_base, cast_shadow);
	_set_instance_base(gaussian_base);
}

void GaussianSplatNode3D::_release_gaussian_base() {
    if (!gaussian_base.is_valid()) {
        return;
    }

    _set_instance_base(RID());

    RendererRD::GaussianSplatStorage *storage = RendererRD::GaussianSplatStorage::get_singleton();
	if (storage) {
#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED
		storage->gaussian_set_renderer(gaussian_base, Ref<GaussianSplatRenderer>());
#endif
		storage->gaussian_set_aabb(gaussian_base, AABB());
		storage->gaussian_set_casts_shadow(gaussian_base, false);
		storage->gaussian_free(gaussian_base);
	}

    gaussian_base = RID();
}

void GaussianSplatNode3D::_sync_gaussian_storage() {
    if (!gaussian_base.is_valid()) {
        return;
    }

    RendererRD::GaussianSplatStorage *storage = RendererRD::GaussianSplatStorage::get_singleton();
    if (!storage) {
        return;
    }

#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED
	storage->gaussian_set_renderer(gaussian_base, renderer);
#endif
	storage->gaussian_set_aabb(gaussian_base, world_aabb);
	storage->gaussian_set_casts_shadow(gaussian_base, cast_shadow);
}

Viewport *GaussianSplatNode3D::_find_editor_scene_viewport() const {
    return viewport_helper.find_editor_scene_viewport();
}

void GaussianSplatNode3D::_update_cached_render_target(Viewport *p_viewport) {
    viewport_helper.update_cached_render_target(p_viewport);
}

bool GaussianSplatNode3D::_acquire_viewport_render_target(Viewport *p_viewport) {
    return viewport_helper.acquire_viewport_render_target(p_viewport);
}

void GaussianSplatNode3D::_connect_viewport_observers(Viewport *p_viewport) {
    viewport_helper.connect_viewport_observers(p_viewport);
}

void GaussianSplatNode3D::_disconnect_viewport_observers() {
    viewport_helper.disconnect_viewport_observers();
}

void GaussianSplatNode3D::_ensure_viewport_texture_binding(Viewport *p_viewport) {
    viewport_helper.ensure_viewport_texture_binding(p_viewport);
}

void GaussianSplatNode3D::_queue_viewport_bootstrap() {
    viewport_helper.queue_viewport_bootstrap();
}

void GaussianSplatNode3D::_deferred_viewport_bootstrap() {
    viewport_helper.deferred_viewport_bootstrap();
}

void GaussianSplatNode3D::_queue_first_frame_render() {
    if (first_frame_render_deferred || !is_inside_tree()) {
        return;
    }

    first_frame_render_deferred = true;
    callable_mp(this, &GaussianSplatNode3D::_dispatch_first_frame_render).call_deferred();
}

void GaussianSplatNode3D::_dispatch_first_frame_render() {
    first_frame_render_deferred = false;

    if (!is_inside_tree()) {
        return;
    }

    if (viewport_texture_state != ViewportTextureState::READY || !cached_viewport_render_target.is_valid()) {
        _queue_viewport_bootstrap();
        return;
    }

    _mark_render_state_dirty();
    force_update();
}

void GaussianSplatNode3D::_on_viewport_texture_ready() {
    viewport_helper.on_viewport_texture_ready();
}

void GaussianSplatNode3D::_on_viewport_size_changed() {
    viewport_helper.on_viewport_size_changed();
}

void GaussianSplatNode3D::_on_observed_viewport_exited() {
    viewport_helper.on_observed_viewport_exited();
}

void GaussianSplatNode3D::_set_instance_base(const RID &p_base) {
    if (!render_instance.is_valid()) {
        return;
    }

    RenderingServer *rs = RS::get_singleton();
    if (!rs) {
        return;
    }

    rs->instance_set_base(render_instance, p_base);
}

void GaussianSplatNode3D::_upload_asset_to_renderer() {
    renderer_helper.upload_asset_to_renderer();
}

void GaussianSplatNode3D::_mark_render_state_dirty() {
    renderer_helper.mark_render_state_dirty();
}

bool GaussianSplatNode3D::_resolve_is_2d_mode() const {
    if (renderer_data.is_valid()) {
        return renderer_data->get_2d_mode();
    }
    if (splat_asset.is_valid()) {
        Dictionary import_metadata = splat_asset->get_import_metadata();
        if (import_metadata.has(StringName("gaussian_2d_mode"))) {
            return (bool)import_metadata[StringName("gaussian_2d_mode")];
        }
    }
    return false;
}

uint32_t GaussianSplatNode3D::_get_instance_flags() const {
    return _resolve_is_2d_mode() ? 1u : 0u;
}

float GaussianSplatNode3D::_get_instance_wind_intensity() const {
    if (!wind_override_enabled) {
        return 1.0f;
    }
    return MAX(0.0f, wind_strength);
}

uint32_t GaussianSplatNode3D::_get_instance_wind_mode() const {
    if (!wind_override_enabled) {
        return GaussianSplatSceneDirector::INSTANCE_WIND_INHERIT;
    }
    return wind_enabled ?
            GaussianSplatSceneDirector::INSTANCE_WIND_FORCE_ENABLED :
            GaussianSplatSceneDirector::INSTANCE_WIND_FORCE_DISABLED;
}

Vector3 GaussianSplatNode3D::_get_instance_wind_direction() const {
    if (!wind_override_enabled) {
        return Vector3();
    }
    return wind_direction;
}

float GaussianSplatNode3D::_get_instance_wind_frequency() const {
    if (!wind_override_enabled) {
        return 1.0f;
    }
    return MAX(0.0f, wind_frequency);
}

void GaussianSplatNode3D::_register_instance_in_director() {
    const bool in_tree = is_inside_tree();
    const bool in_world = is_inside_world();
    const bool has_asset = splat_asset.is_valid();
    const bool has_runtime_asset = runtime_asset.is_valid();
    if (GaussianSplatting::debug_trace_is_enabled()) {
        GaussianSplatting::debug_trace_record_event("instance_reg",
                vformat("node=%s in_tree=%s in_world=%s has_asset=%s runtime_asset=%s",
                        get_name(), in_tree ? "Y" : "N",
                        in_world ? "Y" : "N", has_asset ? "Y" : "N",
                        has_runtime_asset ? "Y" : "N"),
                false);
    }
    if (!in_tree || !in_world) {
        return;
    }
    Ref<GaussianSplatAsset> asset = splat_asset;
    if (asset.is_null() && renderer_data.is_valid()) {
        if (runtime_asset.is_null()) {
            runtime_asset.instantiate();
            runtime_asset->set_asset_type(GaussianSplatAsset::ASSET_TYPE_DYNAMIC);
        }
        Error asset_err = runtime_asset->populate_from_gaussian_data(renderer_data);
        if (asset_err != OK) {
            GS_LOG_WARN_DEFAULT(vformat("[GaussianSplatNode3D] Failed to build runtime asset for instance pipeline (err=%d).", asset_err));
        }
        asset = runtime_asset;
    } else if (asset.is_null()) {
        asset = runtime_asset;
    }
    if (asset.is_null()) {
        WARN_PRINT_ONCE("[GaussianSplatNode3D] Instance pipeline requires a GaussianSplatAsset; skipping instance registration.");
        return;
    }
    GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
    if (!director) {
        return;
    }
    director->register_instance(get_instance_id(), asset, get_global_transform(),
            opacity, lod_bias, _get_instance_flags(), cast_shadow,
            _get_instance_wind_intensity(), _get_instance_wind_mode(),
            _get_instance_wind_direction(), _get_instance_wind_frequency(),
            parent_visible && visible_in_viewport && is_visible_in_tree());
}

void GaussianSplatNode3D::_unregister_instance_in_director() {
    if (GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton()) {
        director->unregister_instance(get_instance_id());
    }
}

void GaussianSplatNode3D::_update_instance_transform_in_director() {
    if (!is_inside_tree() || !is_inside_world()) {
        return;
    }
    if (GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton()) {
        director->update_instance_transform(get_instance_id(), get_global_transform());
    }
}

void GaussianSplatNode3D::_update_instance_params_in_director() {
    if (!is_inside_tree() || !is_inside_world()) {
        return;
    }
    if (GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton()) {
        director->update_instance_params(get_instance_id(), opacity, lod_bias, _get_instance_flags(), cast_shadow,
                _get_instance_wind_intensity(), _get_instance_wind_mode(),
                _get_instance_wind_direction(), _get_instance_wind_frequency(),
                parent_visible && visible_in_viewport && is_visible_in_tree());
    }
    if (renderer.is_valid()) {
        renderer->invalidate_cached_render();
    }
}

void GaussianSplatNode3D::_register_shared_renderer() {
    const bool debug_logs_enabled = GaussianSplatting::is_debug_frame_logging_enabled();
    if (debug_logs_enabled) {
        GS_LOG_RENDERER_DEBUG(vformat("[NODE-REG] _register_shared_renderer: in_tree=%s in_world=%s",
                is_inside_tree() ? "yes" : "no", is_inside_world() ? "yes" : "no"));
    }
    if (!is_inside_tree() || !is_inside_world()) {
        return;
    }
    if (splat_asset.is_null() && renderer_data.is_null() && runtime_asset.is_null()) {
        if (debug_logs_enabled) {
            GS_LOG_RENDERER_DEBUG("[NODE-REG] Early return: no data");
        }
        return;
    }
    _register_instance_in_director();

}

void GaussianSplatNode3D::_unregister_shared_renderer() {
    _unregister_instance_in_director();
}

void GaussianSplatNode3D::_update_shared_transform() {
    _update_instance_transform_in_director();
}

void GaussianSplatNode3D::_on_asset_changed() {
    _update_asset();
    _register_shared_renderer();
    _mark_render_state_dirty();
    update_configuration_warnings();
}

void GaussianSplatNode3D::_on_color_grading_changed() {
    if (renderer.is_valid()) {
        if (_is_multi_instance_shared_renderer_active(renderer)) {
            renderer->set_color_grading(Ref<ColorGradingResource>());
        } else {
            renderer->set_color_grading(color_grading);
        }
        renderer->invalidate_cached_render();
    }
}

void GaussianSplatNode3D::_on_transform_changed() {
    bounds_dirty = true;
    _update_bounds();
    _update_render_instance();
    _update_shared_transform();
}

void GaussianSplatNode3D::_clear_parent_visibility_tracking() {
    visibility_helper.clear_parent_visibility_tracking();
}

void GaussianSplatNode3D::_update_parent_visibility_tracking() {
    visibility_helper.update_parent_visibility_tracking();
}

void GaussianSplatNode3D::_update_parent_visibility_state() {
    visibility_helper.update_parent_visibility_state();
}

void GaussianSplatNode3D::_on_parent_visibility_changed_with_bool(bool p_visible) {
    visibility_helper.on_parent_visibility_changed_with_bool(p_visible);
}

void GaussianSplatNode3D::_on_parent_visibility_changed() {
    visibility_helper.on_parent_visibility_changed();
}

void GaussianSplatNode3D::_apply_quality_preset() {
    quality_helper.apply_quality_preset();
}

Dictionary GaussianSplatNode3D::_get_preset_config(QualityPreset p_preset) const {
    Dictionary config;

    _fill_preset_config(p_preset, config);

    return config;
}

void GaussianSplatNode3D::_fill_preset_config(QualityPreset p_preset, Dictionary &config) const {
    quality_helper.fill_preset_config(p_preset, config);
}

Ref<GaussianSplatRenderer> GaussianSplatNode3D::get_renderer() {
    _ensure_renderer();
    return renderer;
}

// ============================================================================
// Color Grading
// ============================================================================

void GaussianSplatNode3D::set_color_grading(const Ref<ColorGradingResource> &p_grading) {
    const Callable grading_changed = callable_mp(this, &GaussianSplatNode3D::_on_color_grading_changed);
    if (color_grading.is_valid() && color_grading->is_connected("changed", grading_changed)) {
        color_grading->disconnect("changed", grading_changed);
    }

    color_grading = p_grading;
    if (color_grading.is_valid() && !color_grading->is_connected("changed", grading_changed)) {
        color_grading->connect("changed", grading_changed);
    }

    if (renderer.is_valid()) {
        if (_is_multi_instance_shared_renderer_active(renderer)) {
            renderer->set_color_grading(Ref<ColorGradingResource>());
        } else {
            renderer->set_color_grading(color_grading);
        }
        renderer->invalidate_cached_render();
    }

    _mark_render_state_dirty();
    notify_property_list_changed();
}

Error GaussianSplatNode3D::bake_color_grading() {
    return bake_color_grading_snapshot(color_grading);
}

Error GaussianSplatNode3D::bake_color_grading_snapshot(const Ref<ColorGradingResource> &p_grading_snapshot) {
    if (!p_grading_snapshot.is_valid()) {
        ERR_PRINT("Cannot bake color grading: no ColorGradingResource assigned");
        return ERR_UNCONFIGURED;
    }

    if (!renderer_data.is_valid()) {
        ERR_PRINT("Cannot bake color grading: no gaussian data loaded");
        return ERR_UNCONFIGURED;
    }

    Error err = renderer_data->bake_color_grading(p_grading_snapshot);
    if (err != OK) {
        ERR_PRINT("Failed to bake color grading");
        return err;
    }

    // Disable color grading to prevent double application
    // (grading is now baked into the SH DC coefficients)
    if (color_grading.is_valid()) {
        color_grading->set_enabled(false);
    }

    render_state_dirty = true;
    return OK;
}

void GaussianSplatNode3D::restore_color_grading() {
    if (!renderer_data.is_valid()) {
        ERR_PRINT("Cannot restore color grading: no gaussian data loaded");
        return;
    }

    renderer_data->restore_original_colors();

    // Re-enable color grading so it can be applied in real-time again
    if (color_grading.is_valid()) {
        color_grading->set_enabled(true);
    }

    render_state_dirty = true;
}

bool GaussianSplatNode3D::is_color_grading_baked() const {
    if (!renderer_data.is_valid()) {
        return false;
    }
    return renderer_data->is_color_grading_baked();
}

#ifdef TOOLS_ENABLED
bool GaussianSplatNode3D::_can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
    Dictionary d = p_data;
    if (!d.has("type") || !d.has("files")) {
        return false;
    }

    if (String(d["type"]) != "files") {
        return false;
    }

    Vector<String> files = d["files"];
    if (files.size() != 1) {
        return false;
    }

    const String file_lower = files[0].to_lower();
    return file_lower.ends_with(".ply") || file_lower.ends_with(".spz");
}

void GaussianSplatNode3D::_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
    Dictionary d = p_data;
    Vector<String> files = d["files"];

    if (!files.is_empty()) {
        const String file_lower = files[0].to_lower();
        if (file_lower.ends_with(".ply") || file_lower.ends_with(".spz")) {
            set_ply_file_path(files[0]);
        }
    }
}
#endif
