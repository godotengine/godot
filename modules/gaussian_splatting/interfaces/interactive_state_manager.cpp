#include "interactive_state_manager.h"
#include "../logger/gs_logger.h"
#include "../renderer/gaussian_splat_renderer.h"
#include "core/object/object.h"

namespace {

static RenderingDevice *_resolve_device(RenderingDevice *p_fallback, const ObjectID &p_id) {
	if (p_id.is_valid()) {
		if (Object *obj = ObjectDB::get_instance(p_id)) {
			if (RenderingDevice *resolved = Object::cast_to<RenderingDevice>(obj)) {
				return resolved;
			}
		}
		// Owner was tracked but no longer resolves: do not fall back to a potentially stale device.
		return nullptr;
	}
	return p_fallback;
}

static void _free_buffer_if_owned(RenderingDevice *p_device, RID &p_buffer) {
	if (!p_buffer.is_valid()) {
		return;
	}
	if (p_device) {
		p_device->free(p_buffer);
	}
	p_buffer = RID();
}

} // namespace

void InteractiveStateManager::_bind_methods() {
    // Bind methods for script access if needed
}

InteractiveStateManager::InteractiveStateManager() {
}

InteractiveStateManager::~InteractiveStateManager() {
    shutdown();
}

Error InteractiveStateManager::initialize(RenderingDevice *p_device) {
    if (!p_device) {
        return ERR_INVALID_PARAMETER;
    }

    rd = p_device;
    rd_instance_id = p_device->get_instance_id();
    Error err = _create_gpu_resources();
    if (err != OK) {
        return err;
    }

    initialized = true;
    return OK;
}

void InteractiveStateManager::shutdown() {
    _free_gpu_resources();
    selected_indices.clear();
    selected_set.clear();
    hover_index = UINT32_MAX;
    primary_selection = UINT32_MAX;
    current_mode = InteractiveMode::NORMAL;
    initialized = false;
    rd = nullptr;
    rd_instance_id = ObjectID();
}

bool InteractiveStateManager::is_initialized() const {
    return initialized && rd != nullptr;
}

void InteractiveStateManager::set_mode(InteractiveMode p_mode) {
    if (current_mode != p_mode) {
        // Validate transition
        if (!validate_state_transition(current_mode, p_mode)) {
            return;
        }
        current_mode = p_mode;
        _mark_dirty();
    }
}

void InteractiveStateManager::set_hover_index(uint32_t p_index) {
    if (hover_index != p_index) {
        hover_index = p_index;
        _mark_dirty();
    }
}

void InteractiveStateManager::clear_hover() {
    if (hover_index != UINT32_MAX) {
        hover_index = UINT32_MAX;
        _mark_dirty();
    }
}

void InteractiveStateManager::set_selection(const LocalVector<uint32_t> &p_indices) {
    selected_indices = p_indices;
    _rebuild_selection_set();
    if (!selected_indices.is_empty() && primary_selection == UINT32_MAX) {
        primary_selection = selected_indices[0];
    }
    _mark_dirty();
}

void InteractiveStateManager::add_to_selection(uint32_t p_index) {
    if (!selected_set.has(p_index)) {
        selected_indices.push_back(p_index);
        selected_set.insert(p_index);
        if (primary_selection == UINT32_MAX) {
            primary_selection = p_index;
        }
        _mark_dirty();
    }
}

void InteractiveStateManager::remove_from_selection(uint32_t p_index) {
    if (selected_set.has(p_index)) {
        selected_set.erase(p_index);
        // Remove from vector (O(n) but selections are typically small)
        for (uint32_t i = 0; i < selected_indices.size(); i++) {
            if (selected_indices[i] == p_index) {
                selected_indices.remove_at(i);
                break;
            }
        }
        if (primary_selection == p_index) {
            primary_selection = selected_indices.is_empty() ? UINT32_MAX : selected_indices[0];
        }
        _mark_dirty();
    }
}

void InteractiveStateManager::toggle_selection(uint32_t p_index) {
    if (is_selected(p_index)) {
        remove_from_selection(p_index);
    } else {
        add_to_selection(p_index);
    }
}

void InteractiveStateManager::clear_selection() {
    if (!selected_indices.is_empty()) {
        selected_indices.clear();
        selected_set.clear();
        primary_selection = UINT32_MAX;
        _mark_dirty();
    }
}

bool InteractiveStateManager::is_selected(uint32_t p_index) const {
    return selected_set.has(p_index);
}

void InteractiveStateManager::set_primary_selection(uint32_t p_index) {
    if (primary_selection != p_index) {
        primary_selection = p_index;
        // Ensure primary is in selection
        if (p_index != UINT32_MAX && !is_selected(p_index)) {
            add_to_selection(p_index);
        }
    }
}

PickResult InteractiveStateManager::pick(const PickParams &p_params, RID p_depth_texture) {
    PickResult result;

    // GPU picking would be implemented here.
    // For now, return no hit - the actual picking logic would require:
    // 1. Reading back the gaussian ID from a picking buffer
    // 2. Or doing CPU raycast against gaussian positions

    return result;
}

void InteractiveStateManager::update_gpu_state() {
    if (!is_initialized() || !gpu_dirty) {
        return;
    }

    _update_uniform_buffer_data();
    gpu_dirty = false;
}

void InteractiveStateManager::_update_uniform_buffer_data() {
    // Update uniform data structure
    uniform_data.hover_index = hover_index;
    uniform_data.selection_count = selected_indices.size();
    uniform_data.mode = static_cast<uint32_t>(current_mode);

    // Use highlight_intensity for overall strength
    uniform_data.highlight_intensity = highlight_enabled ? highlight_intensity : 0.0f;

    // Hover color (yellow by default)
    uniform_data.hover_color[0] = hover_color.r;
    uniform_data.hover_color[1] = hover_color.g;
    uniform_data.hover_color[2] = hover_color.b;
    uniform_data.hover_color[3] = hover_color.a;

    // Selection color (blue by default)
    uniform_data.select_color[0] = selection_color.r;
    uniform_data.select_color[1] = selection_color.g;
    uniform_data.select_color[2] = selection_color.b;
    uniform_data.select_color[3] = selection_color.a;

    // Disabled color (gray by default)
    uniform_data.disabled_color[0] = disabled_color.r;
    uniform_data.disabled_color[1] = disabled_color.g;
    uniform_data.disabled_color[2] = disabled_color.b;
    uniform_data.disabled_color[3] = disabled_color.a;

    // Upload to GPU
    if (uniform_buffer.is_valid() && rd) {
        Vector<uint8_t> data;
        data.resize(sizeof(InteractiveUniformData));
        memcpy(data.ptrw(), &uniform_data, sizeof(InteractiveUniformData));
        rd->buffer_update(uniform_buffer, 0, sizeof(InteractiveUniformData), data.ptr());
    }
}

void InteractiveStateManager::set_hover_color(const Color &p_color) {
    hover_color = p_color;
    _mark_dirty();
}

void InteractiveStateManager::set_selection_color(const Color &p_color) {
    selection_color = p_color;
    _mark_dirty();
}

void InteractiveStateManager::set_disabled_color(const Color &p_color) {
    disabled_color = p_color;
    _mark_dirty();
}

void InteractiveStateManager::set_highlight_intensity(float p_intensity) {
    highlight_intensity = CLAMP(p_intensity, 0.0f, 2.0f);
    _mark_dirty();
}

// Legacy API from god class
void InteractiveStateManager::set_highlight_enabled(bool p_enabled) {
    if (highlight_enabled != p_enabled) {
        highlight_enabled = p_enabled;
        _mark_dirty();
    }
}

void InteractiveStateManager::set_outline_enabled(bool p_enabled) {
    if (outline_enabled != p_enabled) {
        outline_enabled = p_enabled;
        _mark_dirty();
    }
}

void InteractiveStateManager::set_highlight_color(const Color &p_color) {
    highlight_color = p_color;
    _mark_dirty();
}

void InteractiveStateManager::set_outline_color(const Color &p_color) {
    outline_color = p_color;
    _mark_dirty();
}

void InteractiveStateManager::set_outline_width(float p_width) {
    outline_width = MAX(0.0f, p_width);
    _mark_dirty();
}

SelectionState InteractiveStateManager::get_state() const {
    SelectionState state;
    state.selected_indices = selected_indices;
    state.hover_index = hover_index;
    state.primary_selection = primary_selection;
    state.has_selection = !selected_indices.is_empty();
    state.has_hover = hover_index != UINT32_MAX;
    return state;
}

void InteractiveStateManager::restore_state(const SelectionState &p_state) {
    selected_indices = p_state.selected_indices;
    _rebuild_selection_set();
    hover_index = p_state.hover_index;
    primary_selection = p_state.primary_selection;
    _mark_dirty();
}

// State validation (extracted from god class)
bool InteractiveStateManager::validate_state_transition(InteractiveMode p_from, InteractiveMode p_to) const {
    // Validate state transitions to ensure consistency.
    // For example, disabled splats can't be hovered or selected.
    if (p_from == InteractiveMode::DISABLED &&
        (p_to == InteractiveMode::HOVERED || p_to == InteractiveMode::SELECTED)) {
        GS_LOG_WARN_DEFAULT("Invalid state transition: Cannot hover or select disabled splats");
        return false;
    }
    return true;
}

// Shader binding (extracted from god class)
void InteractiveStateManager::bind_state_shader(InteractiveMode p_state, HashMap<int, RID> &p_state_shaders) {
    int state_key = static_cast<int>(p_state);
    if (!p_state_shaders.has(state_key)) {
        return;
    }

    RID shader = p_state_shaders[state_key];
    if (shader.is_valid()) {
        // The actual binding happens in the render loop
        // Here we just mark that we need to update
        _mark_dirty();
    }
}

void InteractiveStateManager::ensure_state_shader_cache(HashMap<int, RID> &p_state_shaders, RID p_gaussian_shader) {
    // Populate shader map with the base gaussian shader once it's compiled.
    if (!p_gaussian_shader.is_valid()) {
        return;
    }

    // All states use the same shader for now (differentiated by uniforms)
    p_state_shaders.insert(static_cast<int>(InteractiveMode::NORMAL), p_gaussian_shader);
    p_state_shaders.insert(static_cast<int>(InteractiveMode::HOVERED), p_gaussian_shader);
    p_state_shaders.insert(static_cast<int>(InteractiveMode::SELECTED), p_gaussian_shader);
    p_state_shaders.insert(static_cast<int>(InteractiveMode::DISABLED), p_gaussian_shader);
}

void InteractiveStateManager::_mark_dirty() {
    gpu_dirty = true;
}

void InteractiveStateManager::_rebuild_selection_set() {
    selected_set.clear();
    for (uint32_t idx : selected_indices) {
        selected_set.insert(idx);
    }
}

Error InteractiveStateManager::_create_gpu_resources() {
    if (!rd) {
        return ERR_UNCONFIGURED;
    }

    // Create uniform buffer for interactive state
    uniform_buffer = rd->uniform_buffer_create(sizeof(InteractiveUniformData));
    if (!uniform_buffer.is_valid()) {
        return ERR_CANT_CREATE;
    }
    rd->set_resource_name(uniform_buffer, "GS_InteractiveStateManager_UniformBuffer");

    // Initialize with default data
    _update_uniform_buffer_data();

    // Note: uniform_set creation would require a shader reference,
    // which would be provided during full integration with the renderer.
    // For now, the buffer is created but the set is not.

    gpu_dirty = true;
    return OK;
}

void InteractiveStateManager::_free_gpu_resources() {
    RenderingDevice *uniform_device = _resolve_device(rd, rd_instance_id);
    RenderingDevice *state_device = _resolve_device(state_uniform_device, state_uniform_device_id);

    _free_buffer_if_owned(state_device, state_uniform_buffer);
    _free_buffer_if_owned(uniform_device, uniform_buffer);
    _free_buffer_if_owned(uniform_device, selection_buffer);

    state_uniform_device = nullptr;
    state_uniform_device_id = ObjectID();
    uniform_set = RID();
}

bool InteractiveStateManager::apply_renderer_state(GaussianSplatRenderer *p_renderer, int p_state) {
    if (!p_renderer) {
        return false;
    }

    auto &interactive_state = p_renderer->get_interactive_state_config();
    if (!validate_renderer_state_transition(p_renderer, interactive_state.current_state, p_state)) {
        return false;
    }

    interactive_state.current_state = static_cast<GaussianSplatRenderer::InteractiveState>(p_state);
    set_mode(static_cast<InteractiveMode>(p_state));
    bind_renderer_state_shader(p_renderer, p_state);
    notify_renderer_state_change(p_renderer);

    switch (interactive_state.current_state) {
        case GaussianSplatRenderer::STATE_HOVERED:
            enable_renderer_highlight_effect(p_renderer, Color(1.1f, 1.1f, 0.9f, 1.0f));
            break;
        case GaussianSplatRenderer::STATE_SELECTED:
            enable_renderer_outline_effect(p_renderer, Color(1.0f, 0.7f, 0.0f, 1.0f), 2.5f);
            break;
        case GaussianSplatRenderer::STATE_DISABLED:
            enable_renderer_highlight_effect(p_renderer, Color(0.5f, 0.5f, 0.5f, 0.5f));
            break;
        case GaussianSplatRenderer::STATE_NORMAL:
        default:
            remove_renderer_visual_effects(p_renderer);
            break;
    }

    return true;
}

void InteractiveStateManager::initialize_renderer_state_shaders(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }

    auto &interactive_state = p_renderer->get_interactive_state_config();
    interactive_state.state_shaders.clear();
    ensure_renderer_state_shader_cache(p_renderer);
}

void InteractiveStateManager::ensure_renderer_state_shader_cache(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }

    auto &pipeline_state = p_renderer->get_pipeline_state();
    if (!pipeline_state.gaussian_shader.is_valid()) {
        return;
    }

    auto &interactive_state = p_renderer->get_interactive_state_config();
    interactive_state.state_shaders.insert(GaussianSplatRenderer::STATE_NORMAL, pipeline_state.gaussian_shader);
    interactive_state.state_shaders.insert(GaussianSplatRenderer::STATE_HOVERED, pipeline_state.gaussian_shader);
    interactive_state.state_shaders.insert(GaussianSplatRenderer::STATE_SELECTED, pipeline_state.gaussian_shader);
    interactive_state.state_shaders.insert(GaussianSplatRenderer::STATE_DISABLED, pipeline_state.gaussian_shader);
}

void InteractiveStateManager::bind_renderer_state_shader(GaussianSplatRenderer *p_renderer, int p_state) {
    if (!p_renderer) {
        return;
    }
    if (!p_renderer->ensure_rendering_device("bind_renderer_state_shader")) {
        return;
    }

    ensure_renderer_state_shader_cache(p_renderer);

    GaussianSplatRenderer::InteractiveState state = static_cast<GaussianSplatRenderer::InteractiveState>(p_state);
    auto &interactive_state = p_renderer->get_interactive_state_config();
    if (!interactive_state.state_shaders.has(state)) {
        return;
    }

    RID shader = interactive_state.state_shaders[state];
    if (shader.is_valid()) {
        interactive_state.state_dirty = true;
    }
}

void InteractiveStateManager::update_renderer_state_uniforms(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }
    if (!p_renderer->ensure_rendering_device("update_renderer_state_uniforms") ||
            !state_uniform_buffer.is_valid() || !state_uniform_device) {
        return;
    }

    auto &interactive_state = p_renderer->get_interactive_state_config();
    interactive_state.uniform_data.highlight_strength = interactive_state.highlight_enabled ? 1.0f : 0.0f;
    interactive_state.uniform_data.outline_width = interactive_state.outline_enabled ? interactive_state.outline_width : 0.0f;
    interactive_state.uniform_data.state = static_cast<float>(interactive_state.current_state);
    interactive_state.uniform_data.reserved = 0.0f;
    interactive_state.uniform_data.highlight_color = interactive_state.highlight_color;
    interactive_state.uniform_data.outline_color = interactive_state.outline_color;

    state_uniform_device->buffer_update(state_uniform_buffer, 0,
            sizeof(interactive_state.uniform_data), &interactive_state.uniform_data);

    interactive_state.state_dirty = false;
}

RID InteractiveStateManager::ensure_state_uniform_buffer(GaussianSplatRenderer *p_renderer, RenderingDevice *p_device) {
    if (!p_renderer || !p_device) {
        return RID();
    }

    if (state_uniform_buffer.is_valid() && state_uniform_device && state_uniform_device != p_device) {
        RenderingDevice *resolved_state_device = _resolve_device(state_uniform_device, state_uniform_device_id);
        _free_buffer_if_owned(resolved_state_device, state_uniform_buffer);
        state_uniform_device = nullptr;
        state_uniform_device_id = ObjectID();
    } else if (state_uniform_buffer.is_valid() && !state_uniform_device) {
        state_uniform_device = p_device;
        state_uniform_device_id = p_device->get_instance_id();
    }

    if (!state_uniform_buffer.is_valid()) {
        auto &interactive_state = p_renderer->get_interactive_state_config();
        Vector<uint8_t> state_data;
        state_data.resize(sizeof(interactive_state.uniform_data));
        memcpy(state_data.ptrw(), &interactive_state.uniform_data, sizeof(interactive_state.uniform_data));
        state_uniform_buffer = p_device->uniform_buffer_create(state_data.size(), state_data);
        if (state_uniform_buffer.is_valid()) {
            p_device->set_resource_name(state_uniform_buffer, "GS_InteractiveStateManager_StateUniformBuffer");
        }
        state_uniform_device = p_device;
        state_uniform_device_id = p_device->get_instance_id();
    }

    if (state_uniform_buffer.is_valid() && state_uniform_device) {
        auto &interactive_state = p_renderer->get_interactive_state_config();
        if (interactive_state.state_dirty) {
            state_uniform_device->buffer_update(state_uniform_buffer, 0,
                    sizeof(interactive_state.uniform_data), &interactive_state.uniform_data);
            interactive_state.state_dirty = false;
        }
    }

    return state_uniform_buffer;
}

bool InteractiveStateManager::validate_renderer_state_transition(GaussianSplatRenderer *p_renderer, int p_from, int p_to) const {
    if (!p_renderer) {
        return false;
    }

    return validate_state_transition(static_cast<InteractiveMode>(p_from), static_cast<InteractiveMode>(p_to));
}

void InteractiveStateManager::notify_renderer_state_change(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }

    static const char *state_names[] = { "NORMAL", "HOVERED", "SELECTED", "DISABLED" };
    auto &interactive_state = p_renderer->get_interactive_state_config();
    if (interactive_state.current_state <= GaussianSplatRenderer::STATE_DISABLED) {
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianSplat] State changed to: %s",
                state_names[interactive_state.current_state]));
    }
}

void InteractiveStateManager::enable_renderer_highlight_effect(GaussianSplatRenderer *p_renderer, const Color &p_color) {
    if (!p_renderer) {
        return;
    }

    set_highlight_color(p_color);
    set_highlight_enabled(true);

    auto &interactive_state = p_renderer->get_interactive_state_config();
    interactive_state.highlight_color = p_color;
    interactive_state.highlight_enabled = true;
    interactive_state.state_dirty = true;
    update_renderer_state_uniforms(p_renderer);
}

void InteractiveStateManager::enable_renderer_outline_effect(GaussianSplatRenderer *p_renderer, const Color &p_color, float p_width) {
    if (!p_renderer) {
        return;
    }

    set_outline_color(p_color);
    set_outline_width(p_width);
    set_outline_enabled(true);

    auto &interactive_state = p_renderer->get_interactive_state_config();
    interactive_state.outline_color = p_color;
    interactive_state.outline_width = MAX(0.0f, p_width);
    interactive_state.outline_enabled = true;
    interactive_state.state_dirty = true;
    update_renderer_state_uniforms(p_renderer);
}

void InteractiveStateManager::remove_renderer_visual_effects(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }

    set_highlight_enabled(false);
    set_outline_enabled(false);

    auto &interactive_state = p_renderer->get_interactive_state_config();
    interactive_state.highlight_enabled = false;
    interactive_state.outline_enabled = false;
    interactive_state.state_dirty = true;
    update_renderer_state_uniforms(p_renderer);
}

void GaussianSplatRenderer::enable_highlight_effect(const Color &p_color) {
    if (subsystem_state.interactive_state_manager.is_valid()) {
        subsystem_state.interactive_state_manager->enable_renderer_highlight_effect(this, p_color);
    }
}

void GaussianSplatRenderer::enable_outline_effect(const Color &p_color, float p_width) {
    if (subsystem_state.interactive_state_manager.is_valid()) {
        subsystem_state.interactive_state_manager->enable_renderer_outline_effect(this, p_color, p_width);
    }
}

void GaussianSplatRenderer::remove_visual_effects() {
    if (subsystem_state.interactive_state_manager.is_valid()) {
        subsystem_state.interactive_state_manager->remove_renderer_visual_effects(this);
    }
}
