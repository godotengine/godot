#ifndef GAUSSIAN_SPLAT_SETTINGS_MANAGER_H
#define GAUSSIAN_SPLAT_SETTINGS_MANAGER_H

#include "core/config/project_settings.h"
#include "core/string/string_name.h"

class GaussianSplatSettingsManager {
public:
    struct DebugOverlaySettings {
        bool show_tile_grid = false;
        bool show_density_heatmap = false;
        bool show_performance_hud = false;
        bool show_residency_hud = false;
    };

    static void register_debug_overlay_project_settings();
    static DebugOverlaySettings load_debug_overlay_settings();

    static bool get_debug_show_tile_grid();
    static bool get_debug_show_density_heatmap();
    static bool get_debug_show_performance_hud();
    static bool get_debug_show_residency_hud();

    static void set_debug_show_tile_grid(bool p_enabled);
    static void set_debug_show_density_heatmap(bool p_enabled);
    static void set_debug_show_performance_hud(bool p_enabled);
    static void set_debug_show_residency_hud(bool p_enabled);

private:
    static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback);
    static void _persist_if_editor(const StringName &name, bool value);
};

void initialize_gaussian_splat_settings();

#endif // GAUSSIAN_SPLAT_SETTINGS_MANAGER_H
