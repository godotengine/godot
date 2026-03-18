// Render configuration state definitions for GaussianSplatRenderer.
// Included inside the class declaration to keep config types grouped.
//
// See also: render_types/render_config_types.h for standalone types.

    struct RenderConfig {
        RenderMode render_mode = MODE_3D;
        float opacity_multiplier = 1.0f;
        Ref<class ColorGradingResource> color_grading;
    };

    struct CullingConfig {
        bool solid_coverage_enabled = false;        // Disabled by default - exposes sorting bugs when enabled
        float solid_coverage_alpha_floor = 0.95f;   // Minimum alpha when solid coverage is enabled
    };

    struct PainterlyConfig {
        bool enabled = false;
        bool enable_strokes = true;
        bool low_end_mode = false;
        float internal_scale = 1.0f;
        float edge_threshold = 0.25f;
        float edge_intensity = 1.5f;
        float stroke_length = 32.0f;
        float stroke_opacity = 0.8f;
        float gamma = 2.2f;
    };
