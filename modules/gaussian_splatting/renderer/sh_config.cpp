#include "sh_config.h"
#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "../core/gs_project_settings.h"
#include "../core/quality_tier_config.h"
#include "../logger/gs_logger.h"

// Project settings paths
const String SHConfig::BANDS_PATH = SH_CONFIG_BANDS_PATH;
const String SHConfig::DC_LOGIT_PATH = SH_CONFIG_DC_LOGIT_PATH;
const String SHConfig::PROGRESSIVE_PATH = SH_CONFIG_PROGRESSIVE_PATH;

// Global instance
SHConfig g_sh_config;

void SHConfig::load_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    // Load SH band level with sentinel-based tier seeding.
    // -1 means "not explicitly set by user" -- check active tier for a recommendation.
    int raw_band_value = gs::settings::get_int(ps, BANDS_PATH, -1);
    int band_value = raw_band_value;
    if (band_value < 0) {
        // Sentinel: user never set this. Check tier.
        band_value = static_cast<int>(SH_BAND_3); // Code default.
        const String tier_preset = ps->get_setting("rendering/gaussian_splatting/quality/tier_preset", "custom");
        QualityTierConfig tier_config;
        if (get_quality_tier_config(tier_preset, tier_config) && tier_config.sh_bands >= 0) {
            band_value = tier_config.sh_bands;
        }
    }
    sh_bands = static_cast<SHBandLevel>(CLAMP(band_value, 0, static_cast<int>(SH_BAND_MAX)));

    // Write resolved value back so any consumer reading ProjectSettings
    // directly never sees the sentinel -1.
    if (raw_band_value < 0) {
        ps->set_setting(BANDS_PATH, static_cast<int>(sh_bands));
    }

    // Load progressive loading setting
    progressive_load = ps->get_setting(PROGRESSIVE_PATH, false);

    // Load DC encoding mode
    dc_is_logit = ps->get_setting(DC_LOGIT_PATH, false);

    if (GS_LOG_ENABLED(gs_logger::Category::STREAMING, gs_logger::Level::INFO)) {
        print_config_summary();
    }
}

void SHConfig::save_to_project_settings() const {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    ps->set_setting(BANDS_PATH, static_cast<int>(sh_bands));
    ps->set_setting(DC_LOGIT_PATH, dc_is_logit);
    ps->set_setting(PROGRESSIVE_PATH, progressive_load);

    ps->save();

    GS_LOG_STREAMING_INFO(String("[SH Config] Configuration saved to project settings"));
}

void SHConfig::reset_to_defaults() {
    sh_bands = SH_BAND_3;
    progressive_load = false;
    dc_is_logit = false;

    GS_LOG_STREAMING_INFO(String("[SH Config] Reset to default configuration (SH3, progressive disabled)"));
}

bool SHConfig::validate() const {
    return sh_bands >= SH_BAND_0 && sh_bands <= SH_BAND_MAX;
}

int SHConfig::get_coefficient_count(SHBandLevel level) {
    // Number of SH coefficients per channel for each band level
    // SH0: 1 coefficient (DC)
    // SH1: 4 coefficients (1 DC + 3 first-order)
    // SH2: 9 coefficients (1 DC + 3 first + 5 second)
    // SH3: 16 coefficients (1 DC + 3 first + 5 second + 7 third)
    switch (level) {
        case SH_BAND_0: return 1;
        case SH_BAND_1: return 4;
        case SH_BAND_2: return 9;
        case SH_BAND_3: return 16;
        default: return 16;
    }
}

int SHConfig::get_float_count(SHBandLevel level) {
    // Each coefficient has 3 color channels (RGB)
    return get_coefficient_count(level) * 3;
}

const char* SHConfig::get_band_name(SHBandLevel level) {
    switch (level) {
        case SH_BAND_0: return "SH0 (DC only)";
        case SH_BAND_1: return "SH1 (1st order)";
        case SH_BAND_2: return "SH2 (2nd order)";
        case SH_BAND_3: return "SH3 (3rd order)";
        default: return "Unknown";
    }
}

float SHConfig::get_memory_multiplier(SHBandLevel level) {
    // Memory multiplier relative to full SH3
    // SH3 = 48 floats, SH0 = 3 floats
    float sh3_floats = static_cast<float>(get_float_count(SH_BAND_3));
    float level_floats = static_cast<float>(get_float_count(level));
    return level_floats / sh3_floats;
}

void SHConfig::print_config_summary() const {
    GS_LOG_STREAMING_INFO(String("[SH Config] ========== Configuration Summary =========="));
    GS_LOG_STREAMING_INFO(vformat("[SH Config] SH Band Level: %s (%d coefficients, %d floats per splat)",
            get_band_name(sh_bands), get_coefficient_count(sh_bands), get_float_count(sh_bands)));
    GS_LOG_STREAMING_INFO(vformat("[SH Config] Memory Usage: %.1f%% of full SH3",
            get_memory_multiplier(sh_bands) * 100.0f));
    GS_LOG_STREAMING_INFO(vformat("[SH Config] Progressive Loading: %s",
            progressive_load ? "enabled (SH0 first, then higher bands)" : "disabled"));
    GS_LOG_STREAMING_INFO(vformat("[SH Config] DC Logit Decode: %s",
            dc_is_logit ? "enabled (sigmoid)" : "disabled (linear)"));
    GS_LOG_STREAMING_INFO(String("[SH Config] ================================================"));
}

void initialize_sh_config() {
    // Register project settings with proper hints
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    // SH bands setting with enum hint.
    // Default is -1 (sentinel): "use tier recommendation or code-default SH3".
    // Values 0-3 mean the user explicitly chose a band level.
    if (!ps->has_setting(SHConfig::BANDS_PATH)) {
        ps->set_setting(SHConfig::BANDS_PATH, -1);
    }
    ps->set_initial_value(SHConfig::BANDS_PATH, -1);
    ps->set_custom_property_info(PropertyInfo(
        Variant::INT,
        SHConfig::BANDS_PATH,
        PROPERTY_HINT_ENUM,
        "Auto (Tier Default):-1,SH0 (DC Only):0,SH1 (1st Order):1,SH2 (2nd Order):2,SH3 (3rd Order):3"
    ));

    // DC logit decode toggle
    if (!ps->has_setting(SHConfig::DC_LOGIT_PATH)) {
        ps->set_setting(SHConfig::DC_LOGIT_PATH, false);
    }
    ps->set_initial_value(SHConfig::DC_LOGIT_PATH, false);
    ps->set_custom_property_info(PropertyInfo(
        Variant::BOOL,
        SHConfig::DC_LOGIT_PATH,
        PROPERTY_HINT_NONE,
        ""
    ));

    // Progressive loading setting
    if (!ps->has_setting(SHConfig::PROGRESSIVE_PATH)) {
        ps->set_setting(SHConfig::PROGRESSIVE_PATH, false);
    }
    ps->set_initial_value(SHConfig::PROGRESSIVE_PATH, false);
    ps->set_custom_property_info(PropertyInfo(
        Variant::BOOL,
        SHConfig::PROGRESSIVE_PATH,
        PROPERTY_HINT_NONE,
        ""
    ));

    // Load current settings
    g_sh_config.load_from_project_settings();

    if (!g_sh_config.validate()) {
        GS_LOG_STREAMING_WARN(String("[SH Config] Invalid configuration detected, resetting to defaults"));
        g_sh_config.reset_to_defaults();
        g_sh_config.save_to_project_settings();
    }
}
