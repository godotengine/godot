/**
 * gaussian_data_animation.cpp -- Companion .cpp for gaussian_data.h
 *
 * Contains the animation system integration methods of GaussianData:
 *   - Animation state machine / incremental saver wiring
 *   - Per-frame animation update and cache management
 *   - Per-splat animated property accessors (position, color, opacity,
 *     scale, rotation)
 */

#include "gaussian_data.h"

// ---------------------------------------------------------------------------
// Animation state-machine & incremental-saver wiring
// ---------------------------------------------------------------------------

void GaussianData::set_animation_state_machine(const Ref<GaussianSplatting::GaussianAnimationStateMachine>& p_animation) {
    animation_state_machine = p_animation;
    if (animation_state_machine.is_valid()) {
        animation_state_machine->set_splat_count(gaussians.size());
        if (incremental_saver.is_valid()) {
            animation_state_machine->set_incremental_saver(incremental_saver);
        }
    }
    {
        MutexLock anim_lock(animation_cache_mutex);
        animation_cache_dirty = true;
    }
}

void GaussianData::set_incremental_saver(const Ref<GaussianSplatting::GaussianIncrementalSaver>& p_saver) {
    incremental_saver = p_saver;
    if (incremental_saver.is_valid() && animation_state_machine.is_valid()) {
        animation_state_machine->set_incremental_saver(incremental_saver);
    }
}

// ---------------------------------------------------------------------------
// Per-frame animation update
// ---------------------------------------------------------------------------

void GaussianData::update_animation(float p_delta) {
    if (!animation_state_machine.is_valid() || !animation_enabled) {
        return;
    }

    animation_state_machine->update(p_delta);
    {
        MutexLock anim_lock(animation_cache_mutex);
        animation_cache_dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Bulk apply: sample every track and write results into the gaussian array
// ---------------------------------------------------------------------------

void GaussianData::apply_animation_at_time(float p_time) {
    if (!animation_state_machine.is_valid() || !animation_enabled) {
        return;
    }
    RWLockWrite lock(data_rwlock);
    MutexLock anim_lock(animation_cache_mutex);

    bool animate_positions = animation_state_machine->has_active_track(GaussianSplatting::ANIMATION_PROPERTY_POSITION);
    bool animate_colors = animation_state_machine->has_active_track(GaussianSplatting::ANIMATION_PROPERTY_COLOR);
    bool animate_opacities = animation_state_machine->has_active_track(GaussianSplatting::ANIMATION_PROPERTY_OPACITY);

    bool needs_position_cache = animate_positions && animated_positions_cache.size() != gaussians.size();
    bool needs_color_cache = animate_colors && animated_colors_cache.size() != gaussians.size();
    bool needs_opacity_cache = animate_opacities && animated_opacities_cache.size() != gaussians.size();

    // Update cache if needed
    if (animation_cache_dirty || p_time != last_animation_time || needs_position_cache || needs_color_cache || needs_opacity_cache) {
        if (animate_positions) {
            animated_positions_cache.resize(gaussians.size());
            animated_positions_valid_cache.resize(gaussians.size());
            for (uint32_t i = 0; i < gaussians.size(); i++) {
                Vector3 sampled_position;
                if (animation_state_machine->try_sample_position(i, p_time, sampled_position)) {
                    animated_positions_cache[i] = sampled_position;
                    animated_positions_valid_cache[i] = true;
                } else {
                    animated_positions_valid_cache[i] = false;
                }
            }
        } else {
            animated_positions_cache.clear();
            animated_positions_valid_cache.clear();
        }

        if (animate_colors) {
            animated_colors_cache.resize(gaussians.size());
            animated_colors_valid_cache.resize(gaussians.size());
            for (uint32_t i = 0; i < gaussians.size(); i++) {
                Color sampled_color;
                if (animation_state_machine->try_sample_color(i, p_time, sampled_color)) {
                    animated_colors_cache[i] = sampled_color;
                    animated_colors_valid_cache[i] = true;
                } else {
                    animated_colors_valid_cache[i] = false;
                }
            }
        } else {
            animated_colors_cache.clear();
            animated_colors_valid_cache.clear();
        }

        if (animate_opacities) {
            animated_opacities_cache.resize(gaussians.size());
            animated_opacities_valid_cache.resize(gaussians.size());
            for (uint32_t i = 0; i < gaussians.size(); i++) {
                float sampled_opacity;
                if (animation_state_machine->try_sample_opacity(i, p_time, sampled_opacity)) {
                    animated_opacities_cache[i] = sampled_opacity;
                    animated_opacities_valid_cache[i] = true;
                } else {
                    animated_opacities_valid_cache[i] = false;
                }
            }
        } else {
            animated_opacities_cache.clear();
            animated_opacities_valid_cache.clear();
        }

        last_animation_time = p_time;
        animation_cache_dirty = false;
    } else {
        if (!animate_positions) {
            animated_positions_cache.clear();
            animated_positions_valid_cache.clear();
        }
        if (!animate_colors) {
            animated_colors_cache.clear();
            animated_colors_valid_cache.clear();
        }
        if (!animate_opacities) {
            animated_opacities_cache.clear();
            animated_opacities_valid_cache.clear();
        }
    }

    // Apply animated values to gaussians
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        // Check if there are animated values for this splat
        if (animate_positions && i < animated_positions_cache.size() && i < animated_positions_valid_cache.size() && animated_positions_valid_cache[i]) {
            gaussians[i].position = animated_positions_cache[i];
        }

        if (animate_colors && i < animated_colors_cache.size() && i < animated_colors_valid_cache.size() && animated_colors_valid_cache[i]) {
            gaussians[i].sh_dc = animated_colors_cache[i];
        }

        if (animate_opacities && i < animated_opacities_cache.size() && i < animated_opacities_valid_cache.size() && animated_opacities_valid_cache[i]) {
            gaussians[i].opacity = animated_opacities_cache[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Per-splat animated property accessors
// ---------------------------------------------------------------------------

Vector3 GaussianData::get_animated_position(int p_index, float p_time) const {
    if (p_index < 0 || (uint32_t)p_index >= gaussians.size()) {
        return Vector3();
    }

    if (!animation_state_machine.is_valid() || !animation_enabled) {
        return gaussians[p_index].position;
    }

    Vector3 animated_pos;
    if (animation_state_machine->try_sample_position(p_index, p_time, animated_pos)) {
        return animated_pos;
    }

    return gaussians[p_index].position;
}

Color GaussianData::get_animated_color(int p_index, float p_time) const {
    if (p_index < 0 || (uint32_t)p_index >= gaussians.size()) {
        return Color();
    }

    if (!animation_state_machine.is_valid() || !animation_enabled) {
        return gaussians[p_index].sh_dc;
    }

    Color animated_color;
    if (animation_state_machine->try_sample_color(p_index, p_time, animated_color)) {
        return animated_color;
    }

    return gaussians[p_index].sh_dc;
}

float GaussianData::get_animated_opacity(int p_index, float p_time) const {
    if (p_index < 0 || (uint32_t)p_index >= gaussians.size()) {
        return 1.0f;
    }

    if (!animation_state_machine.is_valid() || !animation_enabled) {
        return gaussians[p_index].opacity;
    }

    float animated_opacity;
    if (animation_state_machine->try_sample_opacity(p_index, p_time, animated_opacity)) {
        return animated_opacity;
    }

    return gaussians[p_index].opacity;
}

Vector3 GaussianData::get_animated_scale(int p_index, float p_time) const {
    if (p_index < 0 || (uint32_t)p_index >= gaussians.size()) {
        return Vector3(1, 1, 1);
    }

    if (!animation_state_machine.is_valid() || !animation_enabled) {
        return gaussians[p_index].scale;
    }

    Vector3 animated_scale;
    if (animation_state_machine->try_sample_scale(p_index, p_time, animated_scale)) {
        return animated_scale;
    }

    return gaussians[p_index].scale;
}

Quaternion GaussianData::get_animated_rotation(int p_index, float p_time) const {
    if (p_index < 0 || (uint32_t)p_index >= gaussians.size()) {
        return Quaternion();
    }

    if (!animation_state_machine.is_valid() || !animation_enabled) {
        return gaussians[p_index].rotation;
    }

    Quaternion animated_rotation;
    if (animation_state_machine->try_sample_rotation(p_index, p_time, animated_rotation)) {
        return animated_rotation;
    }

    return gaussians[p_index].rotation;
}
