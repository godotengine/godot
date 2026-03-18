// Interactive state definitions for GaussianSplatRenderer.
// Included inside the class declaration to keep interactive types grouped.
//
// See also: render_types/render_config_types.h for standalone StateUniformData.

    // Import StateUniformData from namespace for class-internal use
    using StateUniformData = GaussianRenderConfig::StateUniformData;

    struct InteractiveStateConfig {
        StateUniformData uniform_data;
        InteractiveState current_state = STATE_NORMAL;
        HashMap<InteractiveState, RID> state_shaders;
        bool state_dirty = false;
        Color highlight_color = Color(1.2, 1.2, 0.8, 1.0);
        Color outline_color = Color(1.0, 0.5, 0.0, 1.0);
        float outline_width = 2.0;
        bool highlight_enabled = false;
        bool outline_enabled = false;
    };
