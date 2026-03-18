#include "gaussian_import_preset.h"

#include "core/math/math_funcs.h"
#include "core/string/string_name.h"
#include "core/string/translation.h"

static Vector<GaussianImportPresetDefinition> &_gaussian_import_presets() {
    static Vector<GaussianImportPresetDefinition> presets;
    if (presets.is_empty()) {
        GaussianImportPresetDefinition mobile;
        mobile.id = "mobile";
        mobile.display_name = TTR("Mobile");
        mobile.max_splats = 250000;
        mobile.density_multiplier = 0.4;
        mobile.enable_lod = true;
        mobile.optimize_for_gpu = true;
        mobile.quantize_positions = true;
        mobile.quantize_colors = true;
        mobile.quantize_scales = true;
        mobile.quantize_rotations = true;
        mobile.pack_opacity = true;
        mobile.thumbnail_style = 1; // density
        mobile.include_statistics = true;
        mobile.include_memory_estimate = true;
        mobile.default_thumbnail_size = 96;
        mobile.default_asset_type = 0;
        presets.push_back(mobile);

        GaussianImportPresetDefinition desktop;
        desktop.id = "desktop";
        desktop.display_name = TTR("Desktop");
        desktop.max_splats = 750000;
        desktop.density_multiplier = 0.7;
        desktop.enable_lod = true;
        desktop.optimize_for_gpu = true;
        desktop.quantize_positions = true;
        desktop.quantize_colors = false;
        desktop.quantize_scales = true;
        desktop.quantize_rotations = false;
        desktop.pack_opacity = false;
        desktop.thumbnail_style = 0; // color
        desktop.include_statistics = true;
        desktop.include_memory_estimate = true;
        desktop.default_thumbnail_size = 128;
        desktop.default_asset_type = 0;
        presets.push_back(desktop);

        GaussianImportPresetDefinition high;
        high.id = "high";
        high.display_name = TTR("High Quality");
        high.max_splats = 1000000;
        high.density_multiplier = 1.0;
        high.enable_lod = true;
        high.optimize_for_gpu = true;
        high.quantize_positions = false;
        high.quantize_colors = false;
        high.quantize_scales = false;
        high.quantize_rotations = false;
        high.pack_opacity = false;
        high.thumbnail_style = 0;
        high.include_statistics = true;
        high.include_memory_estimate = true;
        high.default_thumbnail_size = 160;
        high.default_asset_type = 0;
        presets.push_back(high);

        GaussianImportPresetDefinition ultra;
        ultra.id = "ultra";
        ultra.display_name = TTR("Ultra Quality");
        ultra.max_splats = 0;
        ultra.density_multiplier = 1.0;
        ultra.enable_lod = true;
        ultra.optimize_for_gpu = true;
        ultra.quantize_positions = false;
        ultra.quantize_colors = false;
        ultra.quantize_scales = false;
        ultra.quantize_rotations = false;
        ultra.pack_opacity = false;
        ultra.thumbnail_style = 2; // normals style
        ultra.include_statistics = true;
        ultra.include_memory_estimate = true;
        ultra.default_thumbnail_size = 192;
        ultra.default_asset_type = 0;
        presets.push_back(ultra);

        GaussianImportPresetDefinition development;
        development.id = "development";
        development.display_name = TTR("Development");
        development.max_splats = 0;
        development.density_multiplier = 1.0;
        development.enable_lod = false;
        development.optimize_for_gpu = false;
        development.quantize_positions = false;
        development.quantize_colors = false;
        development.quantize_scales = false;
        development.quantize_rotations = false;
        development.pack_opacity = false;
        development.thumbnail_style = 3; // heatmap
        development.include_statistics = true;
        development.include_memory_estimate = true;
        development.default_thumbnail_size = 128;
        development.default_asset_type = 1; // dynamic asset for iteration
        presets.push_back(development);
    }
    return presets;
}

const Vector<GaussianImportPresetDefinition> &gaussian_get_import_presets() {
    return _gaussian_import_presets();
}

int gaussian_find_import_preset_index(const String &p_id) {
    const Vector<GaussianImportPresetDefinition> &presets = _gaussian_import_presets();
    String lower = p_id.to_lower();
    for (int i = 0; i < presets.size(); i++) {
        if (presets[i].id == lower) {
            return i;
        }
    }
    return -1;
}

const GaussianImportPresetDefinition &gaussian_get_import_preset_by_index(int p_index) {
    const Vector<GaussianImportPresetDefinition> &presets = _gaussian_import_presets();
    int clamped = CLAMP(p_index, 0, presets.size() - 1);
    return presets[clamped];
}

const GaussianImportPresetDefinition &gaussian_get_import_preset_by_name(const String &p_name) {
    int idx = gaussian_find_import_preset_index(p_name);
    if (idx < 0) {
        return gaussian_get_import_preset_by_index(1); // fallback to desktop/high
    }
    return gaussian_get_import_preset_by_index(idx);
}
