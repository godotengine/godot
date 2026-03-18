#ifndef GAUSSIAN_IMPORT_PRESET_H
#define GAUSSIAN_IMPORT_PRESET_H

#include "core/string/ustring.h"
#include "core/templates/vector.h"

struct GaussianImportPresetDefinition {
    String id;
    String display_name;
    int max_splats = 0;
    double density_multiplier = 1.0;
    bool enable_lod = true;
    bool optimize_for_gpu = true;
    bool quantize_positions = false;
    bool quantize_colors = false;
    bool quantize_scales = false;
    bool quantize_rotations = false;
    bool pack_opacity = false;
    int thumbnail_style = 0;
    bool include_statistics = true;
    bool include_memory_estimate = true;
    int default_thumbnail_size = 128;
    int default_asset_type = 0; // 0 = static, 1 = dynamic
};

const Vector<GaussianImportPresetDefinition> &gaussian_get_import_presets();
int gaussian_find_import_preset_index(const String &p_id);
const GaussianImportPresetDefinition &gaussian_get_import_preset_by_index(int p_index);
const GaussianImportPresetDefinition &gaussian_get_import_preset_by_name(const String &p_name);

#endif // GAUSSIAN_IMPORT_PRESET_H
