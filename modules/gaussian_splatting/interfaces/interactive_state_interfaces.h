#ifndef GS_INTERACTIVE_STATE_INTERFACES_H
#define GS_INTERACTIVE_STATE_INTERFACES_H

#include "core/templates/local_vector.h"
#include "core/templates/rid.h"
#include "core/variant/dictionary.h"
#include "servers/rendering/rendering_device.h"

// Interactive state for visual feedback
enum class InteractiveMode {
    NORMAL = 0,
    HOVERED = 1,
    SELECTED = 2,
    DISABLED = 3
};

// Selection operation types
enum class SelectionOp {
    SET,      // Replace selection
    ADD,      // Add to selection
    REMOVE,   // Remove from selection
    TOGGLE    // Toggle selection state
};

// Selection parameters for GPU-based picking
struct PickParams {
    Vector2 screen_position = Vector2();
    float pick_radius = 5.0f;  // In pixels
    bool depth_test = true;
    uint32_t layer_mask = 0xFFFFFFFF;
};

// Result of a pick operation
struct PickResult {
    int32_t gaussian_index = -1;
    float depth = INFINITY;
    float distance_to_cursor = INFINITY;
    bool hit = false;
};

// Selection state for a group of gaussians
struct SelectionState {
    LocalVector<uint32_t> selected_indices;
    uint32_t hover_index = UINT32_MAX;
    uint32_t primary_selection = UINT32_MAX;  // For multi-select operations
    bool has_selection = false;
    bool has_hover = false;
};

// GPU uniform data for interactive state shader
struct InteractiveUniformData {
    uint32_t hover_index = UINT32_MAX;
    uint32_t selection_count = 0;
    uint32_t mode = 0;
    float highlight_intensity = 1.0f;
    // Color overrides
    float hover_color[4] = { 1.0f, 0.8f, 0.0f, 1.0f };    // Yellow
    float select_color[4] = { 0.2f, 0.6f, 1.0f, 1.0f };   // Blue
    float disabled_color[4] = { 0.5f, 0.5f, 0.5f, 0.5f }; // Gray
};

// Pure abstract interface for interactive state management
class IInteractiveStateManager {
public:
    virtual ~IInteractiveStateManager() = default;

    // Lifecycle
    virtual Error initialize(RenderingDevice *p_device) = 0;
    virtual void shutdown() = 0;
    virtual bool is_initialized() const = 0;

    // Mode management
    virtual void set_mode(InteractiveMode p_mode) = 0;
    virtual InteractiveMode get_mode() const = 0;

    // Hover state
    virtual void set_hover_index(uint32_t p_index) = 0;
    virtual void clear_hover() = 0;
    virtual uint32_t get_hover_index() const = 0;
    virtual bool has_hover() const = 0;

    // Selection management
    virtual void set_selection(const LocalVector<uint32_t> &p_indices) = 0;
    virtual void add_to_selection(uint32_t p_index) = 0;
    virtual void remove_from_selection(uint32_t p_index) = 0;
    virtual void toggle_selection(uint32_t p_index) = 0;
    virtual void clear_selection() = 0;
    virtual const LocalVector<uint32_t> &get_selection() const = 0;
    virtual bool is_selected(uint32_t p_index) const = 0;
    virtual bool has_selection() const = 0;
    virtual uint32_t get_selection_count() const = 0;

    // Primary selection (for multi-select operations)
    virtual void set_primary_selection(uint32_t p_index) = 0;
    virtual uint32_t get_primary_selection() const = 0;

    // GPU picking
    virtual PickResult pick(const PickParams &p_params, RID p_depth_texture) = 0;

    // GPU uniform access
    virtual RID get_uniform_set() const = 0;
    virtual void update_gpu_state() = 0;

    // Visual configuration
    virtual void set_hover_color(const Color &p_color) = 0;
    virtual void set_selection_color(const Color &p_color) = 0;
    virtual void set_disabled_color(const Color &p_color) = 0;
    virtual void set_highlight_intensity(float p_intensity) = 0;

    // Snapshot for serialization
    virtual SelectionState get_state() const = 0;
    virtual void restore_state(const SelectionState &p_state) = 0;

    // Implementation info
    virtual String get_name() const = 0;
};

#endif // GS_INTERACTIVE_STATE_INTERFACES_H
