#include "pipeline_feature_set.h"

#include "servers/rendering/rendering_device.h"
#include "../core/effective_config_snapshot.h"
#include "../core/quality_tier_config.h"
#include "../logger/gs_logger.h"

const String PipelineFeatureSet::SECTION_PATH = "rendering/gaussian_splatting/pipeline/";
const String PipelineFeatureSet::ENABLE_TWO_STAGE_SORT_PATH = SECTION_PATH + "enable_two_stage_sort";
const String PipelineFeatureSet::ENABLE_PACKED_STAGE_DATA_PATH = SECTION_PATH + "enable_packed_stage_data";
const String PipelineFeatureSet::ENABLE_TIGHTER_BOUNDS_PATH = SECTION_PATH + "enable_tighter_bounds";
const String PipelineFeatureSet::ENABLE_FAST_RASTER_PATH = SECTION_PATH + "enable_fast_raster";
const String PipelineFeatureSet::ENABLE_SH_AMORTIZATION_PATH = SECTION_PATH + "enable_sh_amortization";
const String PipelineFeatureSet::SH_AMORTIZATION_DIVISOR_PATH = SECTION_PATH + "sh_amortization_divisor";
const String PipelineFeatureSet::DISABLE_SH_AMORTIZATION_VISIBILITY_PATH = SECTION_PATH + "sh_amortization_disable_on_visibility_change";
const String PipelineFeatureSet::SH_AMORTIZATION_VISIBILITY_THRESHOLD_PATH = SECTION_PATH + "sh_amortization_visibility_threshold";
const String PipelineFeatureSet::ENABLE_ALL_EXPERIMENTAL_PATH = SECTION_PATH + "enable_all_experimental";

PipelineFeatureSet g_pipeline_feature_set;

static void _describe_project_setting_source(ProjectSettings *p_ps, const String &p_path,
        String &r_source, String &r_source_label) {
    if (p_ps != nullptr && p_ps->has_setting(p_path) && !p_ps->is_builtin_setting(p_path)) {
        r_source = "project_override";
        r_source_label = "project override";
        return;
    }
    r_source = "code_default";
    r_source_label = "code default";
}

static void _register_pipeline_project_settings() {
    GLOBAL_DEF(PipelineFeatureSet::ENABLE_TWO_STAGE_SORT_PATH, g_pipeline_feature_set.enable_two_stage_sort);
    GLOBAL_DEF(PipelineFeatureSet::ENABLE_PACKED_STAGE_DATA_PATH, g_pipeline_feature_set.enable_packed_stage_data);
    GLOBAL_DEF(PipelineFeatureSet::ENABLE_TIGHTER_BOUNDS_PATH, g_pipeline_feature_set.enable_tighter_bounds);
    GLOBAL_DEF(PipelineFeatureSet::ENABLE_FAST_RASTER_PATH, g_pipeline_feature_set.enable_fast_raster);
    GLOBAL_DEF(PipelineFeatureSet::ENABLE_SH_AMORTIZATION_PATH, g_pipeline_feature_set.enable_sh_amortization);
    GLOBAL_DEF(PipelineFeatureSet::SH_AMORTIZATION_DIVISOR_PATH, g_pipeline_feature_set.sh_amortization_divisor);
    GLOBAL_DEF(PipelineFeatureSet::DISABLE_SH_AMORTIZATION_VISIBILITY_PATH,
            g_pipeline_feature_set.disable_sh_amortization_on_visibility_change);
    GLOBAL_DEF(PipelineFeatureSet::SH_AMORTIZATION_VISIBILITY_THRESHOLD_PATH,
            g_pipeline_feature_set.sh_amortization_visibility_threshold);
    GLOBAL_DEF(PipelineFeatureSet::ENABLE_ALL_EXPERIMENTAL_PATH, g_pipeline_feature_set.enable_all_experimental);
}

void PipelineFeatureSet::load_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    enable_two_stage_sort = ps->get_setting(ENABLE_TWO_STAGE_SORT_PATH, false);
    enable_packed_stage_data = ps->get_setting(ENABLE_PACKED_STAGE_DATA_PATH, false);
    enable_tighter_bounds = ps->get_setting(ENABLE_TIGHTER_BOUNDS_PATH, false);
    enable_fast_raster = ps->get_setting(ENABLE_FAST_RASTER_PATH, false);
    enable_sh_amortization = ps->get_setting(ENABLE_SH_AMORTIZATION_PATH, false);
    sh_amortization_divisor = ps->get_setting(SH_AMORTIZATION_DIVISOR_PATH, sh_amortization_divisor);
    disable_sh_amortization_on_visibility_change = ps->get_setting(DISABLE_SH_AMORTIZATION_VISIBILITY_PATH,
            disable_sh_amortization_on_visibility_change);
    sh_amortization_visibility_threshold = ps->get_setting(SH_AMORTIZATION_VISIBILITY_THRESHOLD_PATH,
            sh_amortization_visibility_threshold);
    enable_all_experimental = ps->get_setting(ENABLE_ALL_EXPERIMENTAL_PATH, false);

    const String tier_preset = ps->get_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
    const bool apply_tier_toggles = ps->get_setting("rendering/gaussian_splatting/quality/tier_apply_pipeline_toggles", true);

    String two_stage_sort_source;
    String two_stage_sort_source_label;
    String packed_stage_source;
    String packed_stage_source_label;
    String tighter_bounds_source;
    String tighter_bounds_source_label;
    String fast_raster_source;
    String fast_raster_source_label;
    String sh_amortization_source;
    String sh_amortization_source_label;
    String sh_amortization_divisor_source;
    String sh_amortization_divisor_source_label;

    _describe_project_setting_source(ps, ENABLE_TWO_STAGE_SORT_PATH, two_stage_sort_source, two_stage_sort_source_label);
    _describe_project_setting_source(ps, ENABLE_PACKED_STAGE_DATA_PATH, packed_stage_source, packed_stage_source_label);
    _describe_project_setting_source(ps, ENABLE_TIGHTER_BOUNDS_PATH, tighter_bounds_source, tighter_bounds_source_label);
    _describe_project_setting_source(ps, ENABLE_FAST_RASTER_PATH, fast_raster_source, fast_raster_source_label);
    _describe_project_setting_source(ps, ENABLE_SH_AMORTIZATION_PATH, sh_amortization_source, sh_amortization_source_label);
    _describe_project_setting_source(ps, SH_AMORTIZATION_DIVISOR_PATH, sh_amortization_divisor_source, sh_amortization_divisor_source_label);

    if (apply_tier_toggles) {
        QualityTierConfig tier_config;
        if (get_quality_tier_config(tier_preset, tier_config)) {
            enable_packed_stage_data = tier_config.enable_packed_stage_data;
            enable_tighter_bounds = tier_config.enable_tighter_bounds;
            enable_fast_raster = tier_config.enable_fast_raster;
            enable_sh_amortization = tier_config.enable_sh_amortization;
            sh_amortization_divisor = tier_config.sh_amortization_divisor;
            packed_stage_source = "tier_preset";
            packed_stage_source_label = vformat("tier preset '%s'", tier_preset);
            tighter_bounds_source = "tier_preset";
            tighter_bounds_source_label = packed_stage_source_label;
            fast_raster_source = "tier_preset";
            fast_raster_source_label = packed_stage_source_label;
            sh_amortization_source = "tier_preset";
            sh_amortization_source_label = packed_stage_source_label;
            sh_amortization_divisor_source = "tier_preset";
            sh_amortization_divisor_source_label = packed_stage_source_label;
            GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] Applying quality tier preset: %s", tier_config.name));
        }
    }

    Dictionary snapshot;
    GaussianEffectiveConfig::set_entry(snapshot, StringName("pipeline_two_stage_sort"),
            enable_two_stage_sort, two_stage_sort_source, two_stage_sort_source_label);
    GaussianEffectiveConfig::set_entry(snapshot, StringName("pipeline_packed_stage_data"),
            enable_packed_stage_data, packed_stage_source, packed_stage_source_label);
    GaussianEffectiveConfig::set_entry(snapshot, StringName("pipeline_tighter_bounds"),
            enable_tighter_bounds, tighter_bounds_source, tighter_bounds_source_label);
    GaussianEffectiveConfig::set_entry(snapshot, StringName("pipeline_fast_raster"),
            enable_fast_raster, fast_raster_source, fast_raster_source_label);
    GaussianEffectiveConfig::set_entry(snapshot, StringName("pipeline_sh_amortization"),
            enable_sh_amortization, sh_amortization_source, sh_amortization_source_label);
    GaussianEffectiveConfig::set_entry(snapshot, StringName("pipeline_sh_amortization_divisor"),
            int64_t(sh_amortization_divisor), sh_amortization_divisor_source, sh_amortization_divisor_source_label);
    loaded_provenance_snapshot = snapshot;
    effective_provenance_snapshot = Dictionary();
    effective_provenance_snapshot_valid = false;

    if (enable_all_experimental || enable_two_stage_sort || enable_packed_stage_data ||
            enable_tighter_bounds || enable_fast_raster || enable_sh_amortization) {
        print_config_summary();
    }
}

void PipelineFeatureSet::save_to_project_settings() const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    ps->set_setting(ENABLE_TWO_STAGE_SORT_PATH, enable_two_stage_sort);
    ps->set_setting(ENABLE_PACKED_STAGE_DATA_PATH, enable_packed_stage_data);
    ps->set_setting(ENABLE_TIGHTER_BOUNDS_PATH, enable_tighter_bounds);
    ps->set_setting(ENABLE_FAST_RASTER_PATH, enable_fast_raster);
    ps->set_setting(ENABLE_SH_AMORTIZATION_PATH, enable_sh_amortization);
    ps->set_setting(SH_AMORTIZATION_DIVISOR_PATH, sh_amortization_divisor);
    ps->set_setting(DISABLE_SH_AMORTIZATION_VISIBILITY_PATH, disable_sh_amortization_on_visibility_change);
    ps->set_setting(SH_AMORTIZATION_VISIBILITY_THRESHOLD_PATH, sh_amortization_visibility_threshold);
    ps->set_setting(ENABLE_ALL_EXPERIMENTAL_PATH, enable_all_experimental);

    ps->save();

    GS_LOG_INFO_DEFAULT("[Pipeline Feature Set] Configuration saved to project settings");
}

void PipelineFeatureSet::reset_to_defaults() {
    enable_two_stage_sort = false;
    enable_packed_stage_data = false;
    enable_tighter_bounds = false;
    enable_fast_raster = false;
    enable_sh_amortization = false;
    sh_amortization_divisor = 10;
    disable_sh_amortization_on_visibility_change = true;
    sh_amortization_visibility_threshold = 0.25f;
    enable_all_experimental = false;

    GS_LOG_INFO_DEFAULT("[Pipeline Feature Set] Reset to default configuration");
}

bool PipelineFeatureSet::validate(uint32_t p_total_gaussians) const {
    return get_validation_errors(p_total_gaussians).is_empty();
}

String PipelineFeatureSet::get_validation_errors(uint32_t p_total_gaussians) const {
    PackedStringArray errors;
    const bool packed_stage_requested = enable_all_experimental || enable_packed_stage_data;
    const bool sh_amortization_requested = enable_all_experimental || enable_sh_amortization;

    if (packed_stage_requested && p_total_gaussians > PACKED_STAGE_MAX_TOTAL_SPLATS) {
        errors.push_back(vformat(
                "Packed stage data requires <= %d total splats, got %d.",
                int(PACKED_STAGE_MAX_TOTAL_SPLATS),
                int(p_total_gaussians)));
    }

    if (sh_amortization_requested && sh_amortization_divisor <= 1) {
        errors.push_back("SH amortization divisor must be > 1.");
    }

    if (sh_amortization_requested && disable_sh_amortization_on_visibility_change) {
        if (!Math::is_finite(sh_amortization_visibility_threshold)) {
            errors.push_back("SH amortization visibility threshold must be finite.");
        } else {
            if (sh_amortization_visibility_threshold < 0.0f) {
                errors.push_back("SH amortization visibility threshold must be >= 0.");
            }
            if (sh_amortization_visibility_threshold > 1.0f) {
                errors.push_back("SH amortization visibility threshold must be <= 1.");
            }
        }
    }

    return String("\n").join(errors);
}

PipelineFeatureSet PipelineFeatureSet::get_effective(RenderingDevice *p_device,
        bool p_compute_raster_enabled,
        bool p_global_sort_enabled,
        String *r_warnings) const {
    PipelineFeatureSet effective = *this;
    Dictionary provenance_snapshot = loaded_provenance_snapshot.duplicate(true);

    if (enable_all_experimental) {
        effective.enable_two_stage_sort = true;
        effective.enable_packed_stage_data = true;
        effective.enable_tighter_bounds = true;
        effective.enable_fast_raster = true;
        effective.enable_sh_amortization = true;
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_two_stage_sort"),
                true, "project_override", "project override");
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_packed_stage_data"),
                true, "project_override", "project override");
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_tighter_bounds"),
                true, "project_override", "project override");
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_fast_raster"),
                true, "project_override", "project override");
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_sh_amortization"),
                true, "project_override", "project override");
    }

    auto warn = [&](const String &p_msg) {
        if (r_warnings) {
            *r_warnings += p_msg + "\n";
        }
    };

    if (effective.enable_two_stage_sort && !p_global_sort_enabled) {
        warn("Two-stage sort requires global composite sort; disabling feature.");
        effective.enable_two_stage_sort = false;
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_two_stage_sort"),
                false, "runtime_requirement", "disabled by runtime requirement");
    }

    if (!p_compute_raster_enabled) {
        if (effective.enable_fast_raster) {
            warn("Fast raster path requires compute raster; disabling feature.");
            effective.enable_fast_raster = false;
            GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_fast_raster"),
                    false, "runtime_requirement", "disabled by runtime requirement");
        }
    }

    if (effective.enable_sh_amortization && effective.sh_amortization_divisor <= 1) {
        warn("SH amortization requires divisor > 1; disabling feature.");
        effective.enable_sh_amortization = false;
        effective.sh_amortization_divisor = 1;
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_sh_amortization"),
                false, "invalid_setting", "disabled by invalid setting");
        GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_sh_amortization_divisor"),
                int64_t(1), "invalid_setting", "disabled by invalid setting");
    }
    if (!effective.enable_sh_amortization) {
        effective.sh_amortization_divisor = 1;
        Dictionary divisor_entry = GaussianEffectiveConfig::get_entry(provenance_snapshot, StringName("pipeline_sh_amortization_divisor"));
        if (divisor_entry.is_empty()) {
            GaussianEffectiveConfig::set_entry(provenance_snapshot, StringName("pipeline_sh_amortization_divisor"),
                    int64_t(1), "project_setting", "project setting");
        } else {
            divisor_entry[StringName("value")] = int64_t(1);
            divisor_entry[StringName("display_value")] = String("1");
            provenance_snapshot[StringName("pipeline_sh_amortization_divisor")] = divisor_entry;
        }
    }
    if (effective.enable_sh_amortization && effective.disable_sh_amortization_on_visibility_change) {
        if (!Math::is_finite(effective.sh_amortization_visibility_threshold)) {
            warn("SH amortization visibility threshold must be finite; resetting to 0.25.");
            effective.sh_amortization_visibility_threshold = 0.25f;
        } else if (effective.sh_amortization_visibility_threshold < 0.0f) {
            warn("SH amortization visibility threshold < 0; clamping to 0.");
            effective.sh_amortization_visibility_threshold = 0.0f;
        } else if (effective.sh_amortization_visibility_threshold > 1.0f) {
            warn("SH amortization visibility threshold > 1; clamping to 1.");
            effective.sh_amortization_visibility_threshold = 1.0f;
        }
    }

    if (!p_device) {
        warn("No RenderingDevice available to validate pipeline feature capabilities.");
        effective_provenance_snapshot = provenance_snapshot;
        effective_provenance_snapshot_valid = true;
        return effective;
    }

    uint64_t subgroup_ops = p_device->limit_get(RenderingDevice::LIMIT_SUBGROUP_OPERATIONS);
    uint64_t subgroup_stages = p_device->limit_get(RenderingDevice::LIMIT_SUBGROUP_IN_SHADERS);
    bool has_basic = (subgroup_ops & RenderingDevice::SUBGROUP_BASIC_BIT) != 0;
    bool has_ballot = (subgroup_ops & RenderingDevice::SUBGROUP_BALLOT_BIT) != 0;
    bool has_compute = (subgroup_stages & RenderingDevice::SHADER_STAGE_COMPUTE_BIT) != 0;
    bool subgroups_available = has_basic && has_ballot && has_compute;

    if (!subgroups_available && effective.enable_fast_raster) {
        warn("Fast raster path requested but subgroup operations are unavailable; expect reduced gains.");
    }

    effective_provenance_snapshot = provenance_snapshot;
    effective_provenance_snapshot_valid = true;

    return effective;
}

Dictionary PipelineFeatureSet::get_effective_config_snapshot() const {
	if (effective_provenance_snapshot_valid) {
		return effective_provenance_snapshot.duplicate(true);
	}
	Dictionary snapshot = loaded_provenance_snapshot.duplicate(true);
	GaussianEffectiveConfig::mark_snapshot_limited(snapshot, "runtime capability validation pending");
	return snapshot;
}

void PipelineFeatureSet::print_config_summary() const {
    GS_LOG_INFO_DEFAULT("[Pipeline Feature Set] ========== Configuration Summary ==========");
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] enable_all_experimental: %s", enable_all_experimental ? "enabled" : "disabled"));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] two_stage_sort: %s", enable_two_stage_sort ? "enabled" : "disabled"));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] packed_stage_data: %s", enable_packed_stage_data ? "enabled" : "disabled"));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] tighter_bounds: %s", enable_tighter_bounds ? "enabled" : "disabled"));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] fast_raster: %s", enable_fast_raster ? "enabled" : "disabled"));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] sh_amortization: %s", enable_sh_amortization ? "enabled" : "disabled"));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] sh_amortization_divisor: %d", sh_amortization_divisor));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] sh_amortization_visibility_threshold: %.3f",
            sh_amortization_visibility_threshold));
    GS_LOG_INFO_DEFAULT(vformat("[Pipeline Feature Set] sh_amortization_disable_on_visibility_change: %s",
            disable_sh_amortization_on_visibility_change ? "enabled" : "disabled"));
    GS_LOG_INFO_DEFAULT("[Pipeline Feature Set] ================================================");
}

void initialize_pipeline_feature_set() {
    _register_pipeline_project_settings();
    g_pipeline_feature_set.load_from_project_settings();

    if (!g_pipeline_feature_set.validate()) {
        GS_LOG_WARN_DEFAULT("[Pipeline Feature Set] Invalid configuration detected:");
        GS_LOG_WARN_DEFAULT(g_pipeline_feature_set.get_validation_errors());
        GS_LOG_INFO_DEFAULT("[Pipeline Feature Set] Resetting to defaults...");
        g_pipeline_feature_set.reset_to_defaults();
        g_pipeline_feature_set.save_to_project_settings();
    }
}
