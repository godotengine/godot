#ifndef PIPELINE_FEATURE_SET_H
#define PIPELINE_FEATURE_SET_H

#include <cstdint>

#include "core/config/project_settings.h"
#include "core/string/ustring.h"
#include "core/variant/dictionary.h"

class RenderingDevice;

struct PipelineFeatureSet {
    // Packed payloads store the projected global index in 16 bits.
    static constexpr uint32_t PACKED_STAGE_MAX_TOTAL_SPLATS = 65535u;

    bool enable_two_stage_sort = false;
    bool enable_packed_stage_data = false;
    bool enable_tighter_bounds = false;
    bool enable_fast_raster = false;
    bool enable_sh_amortization = false;
    bool enable_all_experimental = false;
    int sh_amortization_divisor = 10;
    bool disable_sh_amortization_on_visibility_change = true;
    float sh_amortization_visibility_threshold = 0.25f;

    static const String SECTION_PATH;
    static const String ENABLE_TWO_STAGE_SORT_PATH;
    static const String ENABLE_PACKED_STAGE_DATA_PATH;
    static const String ENABLE_TIGHTER_BOUNDS_PATH;
    static const String ENABLE_FAST_RASTER_PATH;
    static const String ENABLE_SH_AMORTIZATION_PATH;
    static const String SH_AMORTIZATION_DIVISOR_PATH;
    static const String DISABLE_SH_AMORTIZATION_VISIBILITY_PATH;
    static const String SH_AMORTIZATION_VISIBILITY_THRESHOLD_PATH;
    static const String ENABLE_ALL_EXPERIMENTAL_PATH;

    void load_from_project_settings();
    void save_to_project_settings() const;
    void reset_to_defaults();

    bool validate(uint32_t p_total_gaussians = 0) const;
    String get_validation_errors(uint32_t p_total_gaussians = 0) const;

    PipelineFeatureSet get_effective(RenderingDevice *p_device,
            bool p_compute_raster_enabled,
            bool p_global_sort_enabled,
            String *r_warnings = nullptr) const;

    Dictionary get_effective_config_snapshot() const;

    void print_config_summary() const;

    Dictionary loaded_provenance_snapshot;
    mutable Dictionary effective_provenance_snapshot;
    mutable bool effective_provenance_snapshot_valid = false;
};

extern PipelineFeatureSet g_pipeline_feature_set;

void initialize_pipeline_feature_set();

#endif // PIPELINE_FEATURE_SET_H
