#include "gaussian_splat_settings_manager.h"
#include "gs_project_settings.h"

#include "core/config/engine.h"
#include "core/math/math_funcs.h"

namespace {

constexpr const char *DEBUG_SHOW_TILE_GRID_PATH = "rendering/gaussian_splatting/debug/show_tile_grid";
constexpr const char *DEBUG_SHOW_DENSITY_HEATMAP_PATH = "rendering/gaussian_splatting/debug/show_density_heatmap";
constexpr const char *DEBUG_SHOW_PERFORMANCE_HUD_PATH = "rendering/gaussian_splatting/debug/show_performance_hud";
constexpr const char *DEBUG_SHOW_RESIDENCY_HUD_PATH = "rendering/gaussian_splatting/debug/show_residency_hud";

} // namespace

bool GaussianSplatSettingsManager::_get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
    return gs::settings::get_bool(ps, name, fallback);
}

void GaussianSplatSettingsManager::_persist_if_editor(const StringName &name, bool value) {
#ifdef TOOLS_ENABLED
    Engine *engine = Engine::get_singleton();
    if (!engine || !engine->is_editor_hint()) {
        return;
    }
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        ps->set_setting(name, value);
        ps->save();
    }
#else
    (void)name;
    (void)value;
#endif
}

void GaussianSplatSettingsManager::register_debug_overlay_project_settings() {
    GLOBAL_DEF(String(DEBUG_SHOW_TILE_GRID_PATH), false);
    GLOBAL_DEF(String(DEBUG_SHOW_DENSITY_HEATMAP_PATH), false);
    GLOBAL_DEF(String(DEBUG_SHOW_PERFORMANCE_HUD_PATH), false);
    GLOBAL_DEF(String(DEBUG_SHOW_RESIDENCY_HUD_PATH), false);
}

GaussianSplatSettingsManager::DebugOverlaySettings GaussianSplatSettingsManager::load_debug_overlay_settings() {
    DebugOverlaySettings settings;
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return settings;
    }

    settings.show_tile_grid = _get_bool_setting(ps, StringName(DEBUG_SHOW_TILE_GRID_PATH), settings.show_tile_grid);
    settings.show_density_heatmap = _get_bool_setting(ps, StringName(DEBUG_SHOW_DENSITY_HEATMAP_PATH), settings.show_density_heatmap);
    settings.show_performance_hud = _get_bool_setting(ps, StringName(DEBUG_SHOW_PERFORMANCE_HUD_PATH), settings.show_performance_hud);
    settings.show_residency_hud = _get_bool_setting(ps, StringName(DEBUG_SHOW_RESIDENCY_HUD_PATH), settings.show_residency_hud);

    return settings;
}

bool GaussianSplatSettingsManager::get_debug_show_tile_grid() {
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return _get_bool_setting(ps, StringName(DEBUG_SHOW_TILE_GRID_PATH), false);
    }
    return false;
}

bool GaussianSplatSettingsManager::get_debug_show_density_heatmap() {
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return _get_bool_setting(ps, StringName(DEBUG_SHOW_DENSITY_HEATMAP_PATH), false);
    }
    return false;
}

bool GaussianSplatSettingsManager::get_debug_show_performance_hud() {
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return _get_bool_setting(ps, StringName(DEBUG_SHOW_PERFORMANCE_HUD_PATH), false);
    }
    return false;
}

bool GaussianSplatSettingsManager::get_debug_show_residency_hud() {
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return _get_bool_setting(ps, StringName(DEBUG_SHOW_RESIDENCY_HUD_PATH), false);
    }
    return false;
}

void GaussianSplatSettingsManager::set_debug_show_tile_grid(bool p_enabled) {
    _persist_if_editor(StringName(DEBUG_SHOW_TILE_GRID_PATH), p_enabled);
}

void GaussianSplatSettingsManager::set_debug_show_density_heatmap(bool p_enabled) {
    _persist_if_editor(StringName(DEBUG_SHOW_DENSITY_HEATMAP_PATH), p_enabled);
}

void GaussianSplatSettingsManager::set_debug_show_performance_hud(bool p_enabled) {
    _persist_if_editor(StringName(DEBUG_SHOW_PERFORMANCE_HUD_PATH), p_enabled);
}

void GaussianSplatSettingsManager::set_debug_show_residency_hud(bool p_enabled) {
    _persist_if_editor(StringName(DEBUG_SHOW_RESIDENCY_HUD_PATH), p_enabled);
}

void initialize_gaussian_splat_settings() {
    GaussianSplatSettingsManager::register_debug_overlay_project_settings();
}
