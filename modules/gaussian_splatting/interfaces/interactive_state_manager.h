#ifndef GS_INTERACTIVE_STATE_MANAGER_H
#define GS_INTERACTIVE_STATE_MANAGER_H

#include "interactive_state_interfaces.h"
#include "core/object/ref_counted.h"
#include "core/object/object_id.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"

class GaussianSplatRenderer;

// Concrete implementation of IInteractiveStateManager
// Extracted from GaussianSplatRenderer god class (Phase 8 refactoring)
class InteractiveStateManager : public RefCounted, public IInteractiveStateManager {
    GDCLASS(InteractiveStateManager, RefCounted);

public:
    InteractiveStateManager();
    ~InteractiveStateManager();

    // IInteractiveStateManager interface - Lifecycle
    Error initialize(RenderingDevice *p_device) override;
    void shutdown() override;
    bool is_initialized() const override;

    // IInteractiveStateManager interface - Mode management
    void set_mode(InteractiveMode p_mode) override;
    InteractiveMode get_mode() const override { return current_mode; }

    // IInteractiveStateManager interface - Hover state
    void set_hover_index(uint32_t p_index) override;
    void clear_hover() override;
    uint32_t get_hover_index() const override { return hover_index; }
    bool has_hover() const override { return hover_index != UINT32_MAX; }

    // IInteractiveStateManager interface - Selection management
    void set_selection(const LocalVector<uint32_t> &p_indices) override;
    void add_to_selection(uint32_t p_index) override;
    void remove_from_selection(uint32_t p_index) override;
    void toggle_selection(uint32_t p_index) override;
    void clear_selection() override;
    const LocalVector<uint32_t> &get_selection() const override { return selected_indices; }
    bool is_selected(uint32_t p_index) const override;
    bool has_selection() const override { return !selected_indices.is_empty(); }
    uint32_t get_selection_count() const override { return selected_indices.size(); }

    // IInteractiveStateManager interface - Primary selection
    void set_primary_selection(uint32_t p_index) override;
    uint32_t get_primary_selection() const override { return primary_selection; }

    // IInteractiveStateManager interface - GPU picking
    PickResult pick(const PickParams &p_params, RID p_depth_texture) override;

    // IInteractiveStateManager interface - GPU uniform access
    RID get_uniform_set() const override { return uniform_set; }
    RID get_uniform_buffer() const { return uniform_buffer; }
    void update_gpu_state() override;

    // IInteractiveStateManager interface - Visual configuration
    void set_hover_color(const Color &p_color) override;
    void set_selection_color(const Color &p_color) override;
    void set_disabled_color(const Color &p_color) override;
    void set_highlight_intensity(float p_intensity) override;

    // Legacy API from god class - for backwards compatibility during transition
    void set_highlight_enabled(bool p_enabled);
    void set_outline_enabled(bool p_enabled);
    bool is_highlight_enabled() const { return highlight_enabled; }
    bool is_outline_enabled() const { return outline_enabled; }
    void set_highlight_color(const Color &p_color);
    void set_outline_color(const Color &p_color);
    void set_outline_width(float p_width);
    Color get_highlight_color() const { return highlight_color; }
    Color get_outline_color() const { return outline_color; }
    float get_outline_width() const { return outline_width; }

    // IInteractiveStateManager interface - State snapshot
    SelectionState get_state() const override;
    void restore_state(const SelectionState &p_state) override;

    // IInteractiveStateManager interface - Implementation info
    String get_name() const override { return "InteractiveStateManager"; }

    // State validation and shader binding (from god class)
    bool validate_state_transition(InteractiveMode p_from, InteractiveMode p_to) const;
    void bind_state_shader(InteractiveMode p_state, HashMap<int, RID> &p_state_shaders);
    void ensure_state_shader_cache(HashMap<int, RID> &p_state_shaders, RID p_gaussian_shader);

    // Legacy renderer helpers (god class extraction)
    bool apply_renderer_state(GaussianSplatRenderer *p_renderer, int p_state);
    void initialize_renderer_state_shaders(GaussianSplatRenderer *p_renderer);
    void ensure_renderer_state_shader_cache(GaussianSplatRenderer *p_renderer);
    void bind_renderer_state_shader(GaussianSplatRenderer *p_renderer, int p_state);
    void update_renderer_state_uniforms(GaussianSplatRenderer *p_renderer);
    RID ensure_state_uniform_buffer(GaussianSplatRenderer *p_renderer, RenderingDevice *p_device);
    bool validate_renderer_state_transition(GaussianSplatRenderer *p_renderer, int p_from, int p_to) const;
    void notify_renderer_state_change(GaussianSplatRenderer *p_renderer);
    void enable_renderer_highlight_effect(GaussianSplatRenderer *p_renderer, const Color &p_color);
    void enable_renderer_outline_effect(GaussianSplatRenderer *p_renderer, const Color &p_color, float p_width);
    void remove_renderer_visual_effects(GaussianSplatRenderer *p_renderer);

protected:
    static void _bind_methods();

private:
    RenderingDevice *rd = nullptr;
    ObjectID rd_instance_id;
    bool initialized = false;

    // State
    InteractiveMode current_mode = InteractiveMode::NORMAL;
    uint32_t hover_index = UINT32_MAX;
    uint32_t primary_selection = UINT32_MAX;
    LocalVector<uint32_t> selected_indices;
    HashSet<uint32_t> selected_set;  // For O(1) lookup

    // GPU resources
    RID uniform_buffer;
    RID uniform_set;
    RID selection_buffer;  // For large selection sets
    RID state_uniform_buffer;
    RenderingDevice *state_uniform_device = nullptr;
    ObjectID state_uniform_device_id;
    InteractiveUniformData uniform_data;
    bool gpu_dirty = true;

    // Visual configuration (from god class)
    Color hover_color = Color(1.0f, 0.8f, 0.0f, 1.0f);
    Color selection_color = Color(0.2f, 0.6f, 1.0f, 1.0f);
    Color disabled_color = Color(0.5f, 0.5f, 0.5f, 0.5f);
    float highlight_intensity = 1.0f;

    // Legacy effect toggles from god class
    Color highlight_color = Color(1.2f, 1.2f, 0.8f, 1.0f);
    Color outline_color = Color(1.0f, 0.5f, 0.0f, 1.0f);
    float outline_width = 2.0f;
    bool highlight_enabled = false;
    bool outline_enabled = false;

    void _mark_dirty();
    void _rebuild_selection_set();
    Error _create_gpu_resources();
    void _free_gpu_resources();
    void _update_uniform_buffer_data();
};

#endif // GS_INTERACTIVE_STATE_MANAGER_H
