/**
 * @file gaussian_data_edits.cpp
 * @brief Companion .cpp for gaussian_data.h containing runtime modification
 *        and editing methods.
 *
 * This file holds the GaussianData methods responsible for runtime splat
 * editing: position / color / opacity overlays, bulk range operations,
 * commit / revert lifecycle, brush-stroke recording, and undo-capable
 * capture / restore helpers.  Splitting these out keeps the main
 * gaussian_data.cpp focused on data management, loading, and GPU upload.
 */

#include "gaussian_data.h"
#include "core/math/math_funcs.h"
#include "core/os/time.h"

// ---------------------------------------------------------------------------
// Internal helpers (lock must already be held by the caller).
// ---------------------------------------------------------------------------

bool GaussianData::_clear_runtime_modifications_locked() {
    const bool had_runtime_modifications = edit_state.has_runtime_modifications ||
            !edit_state.runtime_positions.is_empty() ||
            !edit_state.runtime_colors.is_empty() ||
            !edit_state.runtime_opacities.is_empty() ||
            !edit_state.runtime_position_flags.is_empty() ||
            !edit_state.runtime_color_flags.is_empty() ||
            !edit_state.runtime_opacity_flags.is_empty() ||
            !edit_state.modified_flags.is_empty();
    edit_state.runtime_positions.clear();
    edit_state.runtime_colors.clear();
    edit_state.runtime_opacities.clear();
    edit_state.runtime_position_flags.clear();
    edit_state.runtime_color_flags.clear();
    edit_state.runtime_opacity_flags.clear();
    edit_state.modified_flags.clear();
    edit_state.has_runtime_modifications = false;
    return had_runtime_modifications;
}

void GaussianData::_clear_brush_strokes_locked() {
    edit_state.recorded_brush_strokes.clear();
}

void GaussianData::_set_runtime_position_locked(int p_idx, const Vector3& p_pos) {
    const int count = gaussians.size();
    if (edit_state.runtime_positions.size() != count) {
        edit_state.runtime_positions.resize(count);
    }
    if (edit_state.runtime_position_flags.size() != count) {
        edit_state.runtime_position_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.runtime_position_flags[i] = false;
        }
    }
    if (edit_state.modified_flags.size() != count) {
        edit_state.modified_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.modified_flags[i] = false;
        }
    }
    edit_state.runtime_positions[p_idx] = p_pos;
    edit_state.runtime_position_flags[p_idx] = true;
    edit_state.modified_flags[p_idx] = true;
    edit_state.has_runtime_modifications = true;
}

void GaussianData::_set_runtime_color_locked(int p_idx, const Color& p_col) {
    const int count = gaussians.size();
    if (edit_state.runtime_colors.size() != count) {
        edit_state.runtime_colors.resize(count);
    }
    if (edit_state.runtime_color_flags.size() != count) {
        edit_state.runtime_color_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.runtime_color_flags[i] = false;
        }
    }
    if (edit_state.modified_flags.size() != count) {
        edit_state.modified_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.modified_flags[i] = false;
        }
    }
    edit_state.runtime_colors[p_idx] = p_col;
    edit_state.runtime_color_flags[p_idx] = true;
    edit_state.modified_flags[p_idx] = true;
    edit_state.has_runtime_modifications = true;
}

void GaussianData::_set_runtime_opacity_locked(int p_idx, float p_opacity) {
    const int count = gaussians.size();
    if (edit_state.runtime_opacities.size() != count) {
        edit_state.runtime_opacities.resize(count);
    }
    if (edit_state.runtime_opacity_flags.size() != count) {
        edit_state.runtime_opacity_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.runtime_opacity_flags[i] = false;
        }
    }
    if (edit_state.modified_flags.size() != count) {
        edit_state.modified_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.modified_flags[i] = false;
        }
    }
    edit_state.runtime_opacities[p_idx] = CLAMP(p_opacity, 0.0f, 1.0f);
    edit_state.runtime_opacity_flags[p_idx] = true;
    edit_state.modified_flags[p_idx] = true;
    edit_state.has_runtime_modifications = true;
}

// Public thread-safe wrappers that acquire the lock.
void GaussianData::set_runtime_position(int p_idx, const Vector3& p_pos) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_INDEX(p_idx, (int)gaussians.size());
    _set_runtime_position_locked(p_idx, p_pos);
}

void GaussianData::set_runtime_color(int p_idx, const Color& p_col) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_INDEX(p_idx, (int)gaussians.size());
    _set_runtime_color_locked(p_idx, p_col);
}

void GaussianData::set_runtime_opacity(int p_idx, float p_opacity) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_INDEX(p_idx, (int)gaussians.size());
    _set_runtime_opacity_locked(p_idx, p_opacity);
}

// Bulk overlay operations for applying edits across a range of splats.
void GaussianData::apply_color_range(int p_start, int p_count, const Color& p_col) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_INDEX(p_start, (int)gaussians.size());
    ERR_FAIL_COND(p_start + p_count > (int)gaussians.size());
    const int count = gaussians.size();
    if (edit_state.runtime_colors.size() != count) {
        edit_state.runtime_colors.resize(count);
    }
    if (edit_state.runtime_color_flags.size() != count) {
        edit_state.runtime_color_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.runtime_color_flags[i] = false;
        }
    }
    if (edit_state.modified_flags.size() != count) {
        edit_state.modified_flags.resize(count);
        for (int i = 0; i < count; i++) {
            edit_state.modified_flags[i] = false;
        }
    }
    const int end = p_start + p_count;
    for (int i = p_start; i < end; i++) {
        edit_state.runtime_colors[i] = p_col;
        edit_state.runtime_color_flags[i] = true;
        edit_state.modified_flags[i] = true;
    }
    edit_state.has_runtime_modifications = true;
}

void GaussianData::mark_range_dirty(int p_start, int p_count) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_INDEX(p_start, (int)gaussians.size());
    ERR_FAIL_COND(p_start + p_count > (int)gaussians.size());
    if (edit_state.modified_flags.size() != gaussians.size()) {
        edit_state.modified_flags.resize(gaussians.size());
        for (int i = 0; i < (int)edit_state.modified_flags.size(); i++) {
            edit_state.modified_flags[i] = false;
        }
    }
    const int end = p_start + p_count;
    for (int i = p_start; i < end; i++) {
        edit_state.modified_flags[i] = true;
    }
    edit_state.has_runtime_modifications = true;
}

// State management helpers to commit or discard runtime edits.
void GaussianData::commit_runtime_changes() {
    bool committed_changes = false;
    {
        RWLockWrite lock(data_rwlock);
        if (!edit_state.has_runtime_modifications) {
            return;
        }

        for (int i = 0; i < (int)edit_state.modified_flags.size(); i++) {
            if (edit_state.modified_flags[i]) {
                Gaussian original = gaussians[i];
                bool changed = false;
                if (i < (int)edit_state.runtime_positions.size() &&
                    i < (int)edit_state.runtime_position_flags.size() &&
                    edit_state.runtime_position_flags[i]) {
                    gaussians[i].position = edit_state.runtime_positions[i];
                    changed = true;
                }
                if (i < (int)edit_state.runtime_colors.size() &&
                    i < (int)edit_state.runtime_color_flags.size() &&
                    edit_state.runtime_color_flags[i]) {
                    gaussians[i].sh_dc = edit_state.runtime_colors[i];
                    changed = true;
                }
                if (i < (int)edit_state.runtime_opacities.size() &&
                    i < (int)edit_state.runtime_opacity_flags.size() &&
                    edit_state.runtime_opacity_flags[i]) {
                    gaussians[i].opacity = edit_state.runtime_opacities[i];
                    changed = true;
                }
                if (changed && incremental_saver.is_valid()) {
                    incremental_saver->record_splat_change(i, original, gaussians[i]);
                }
                committed_changes = committed_changes || changed;
            }
        }

        _clear_runtime_modifications_locked();
        if (committed_changes) {
            octree_dirty = true;
            {
                MutexLock anim_lock(animation_cache_mutex);
                animation_cache_dirty = true;
            }
        }
    }

    if (committed_changes) {
        emit_changed();
    }
}

void GaussianData::revert_runtime_changes() {
    bool had_runtime_modifications = false;
    {
        RWLockWrite lock(data_rwlock);
        had_runtime_modifications = _clear_runtime_modifications_locked();
    }

    if (had_runtime_modifications) {
        emit_changed();
    }
}

Dictionary GaussianData::_brush_stroke_to_dict(const BrushStroke &p_stroke) {
    Dictionary dict;
    dict["center"] = p_stroke.center;
    dict["radius"] = p_stroke.radius;
    dict["color"] = p_stroke.color;
    dict["opacity"] = p_stroke.opacity;
    dict["hardness"] = p_stroke.hardness;
    dict["timestamp_us"] = (int64_t)p_stroke.timestamp_us;
    return dict;
}

GaussianData::BrushStroke GaussianData::_brush_stroke_from_dict(const Dictionary &p_dict) {
    BrushStroke stroke;
    if (p_dict.has("center")) {
        stroke.center = p_dict["center"];
    }
    if (p_dict.has("radius")) {
        stroke.radius = MAX(0.0f, float(p_dict["radius"]));
    }
    if (p_dict.has("color")) {
        stroke.color = p_dict["color"];
    }
    if (p_dict.has("opacity")) {
        stroke.opacity = CLAMP(float(p_dict["opacity"]), 0.0f, 1.0f);
    }
    if (p_dict.has("hardness")) {
        stroke.hardness = MAX(0.01f, float(p_dict["hardness"]));
    }
    if (p_dict.has("timestamp_us")) {
        stroke.timestamp_us = (uint64_t)MAX(int64_t(0), int64_t(p_dict["timestamp_us"]));
    }
    return stroke;
}

void GaussianData::apply_brush_stroke(const Vector3 &p_center, float p_radius, const Color &p_color, float p_opacity, float p_hardness) {
    {
        RWLockWrite lock(data_rwlock);
        ERR_FAIL_COND(gaussians.is_empty());
        ERR_FAIL_COND_MSG(p_radius <= 0.0f, "Brush radius must be greater than zero.");

        BrushStroke stroke;
        stroke.center = p_center;
        stroke.radius = MAX(0.001f, p_radius);
        stroke.color = p_color;
        stroke.opacity = CLAMP(p_opacity, 0.0f, 1.0f);
        stroke.hardness = MAX(0.01f, p_hardness);
        if (Time *time_singleton = Time::get_singleton()) {
            stroke.timestamp_us = time_singleton->get_ticks_usec();
        }

        edit_state.recorded_brush_strokes.push_back(stroke);
        if (edit_state.recorded_brush_strokes.size() > 2048) {
            edit_state.recorded_brush_strokes.remove_at(0);
        }

        const float radius_sq = stroke.radius * stroke.radius;
        const Color target_color = stroke.color;
        const float target_alpha = CLAMP(target_color.a, 0.0f, 1.0f);

        for (int i = 0; i < (int)gaussians.size(); i++) {
            const Vector3 &pos = gaussians[i].position;
            float distance_sq = p_center.distance_squared_to(pos);
            if (distance_sq > radius_sq) {
                continue;
            }

            float distance = Math::sqrt(distance_sq);
            float normalized = distance / stroke.radius;
            float falloff = 1.0f - CLAMP(normalized, 0.0f, 1.0f);
            falloff = Math::pow(falloff, stroke.hardness);
            float strength = falloff * stroke.opacity;
            if (strength <= 0.0001f) {
                continue;
            }

            Color base_color = gaussians[i].sh_dc;
            if (i < (int)edit_state.runtime_colors.size() && i < (int)edit_state.runtime_color_flags.size() && edit_state.runtime_color_flags[i]) {
                base_color = edit_state.runtime_colors[i];
            }
            Color blended = base_color.lerp(target_color, strength);
            _set_runtime_color_locked(i, blended);

            float base_opacity = gaussians[i].opacity;
            if (i < (int)edit_state.runtime_opacities.size() && i < (int)edit_state.runtime_opacity_flags.size() && edit_state.runtime_opacity_flags[i]) {
                base_opacity = edit_state.runtime_opacities[i];
            }
            float new_opacity = Math::lerp(base_opacity, target_alpha, strength);
            _set_runtime_opacity_locked(i, new_opacity);
        }
    }

    emit_changed();
}

Array GaussianData::get_brush_strokes() const {
    RWLockRead lock(data_rwlock);
    Array result;
    result.resize(edit_state.recorded_brush_strokes.size());
    for (int i = 0; i < edit_state.recorded_brush_strokes.size(); i++) {
        result[i] = _brush_stroke_to_dict(edit_state.recorded_brush_strokes[i]);
    }
    return result;
}

void GaussianData::clear_brush_strokes() {
    RWLockWrite lock(data_rwlock);
    _clear_brush_strokes_locked();
}

void GaussianData::set_brush_strokes(const Array &p_strokes) {
    RWLockWrite lock(data_rwlock);
    _clear_brush_strokes_locked();
    for (int i = 0; i < p_strokes.size(); i++) {
        Dictionary dict = p_strokes[i];
        edit_state.recorded_brush_strokes.push_back(_brush_stroke_from_dict(dict));
    }
}

Dictionary GaussianData::capture_brush_affected_state(const Vector3 &p_center, float p_radius) const {
    MutexLock lock(sh_mutex);
    Dictionary state;
    ERR_FAIL_COND_V(gaussians.is_empty(), state);
    ERR_FAIL_COND_V(p_radius <= 0.0f, state);

    PackedInt32Array indices;
    PackedColorArray colors;
    PackedFloat32Array opacities;

    const float radius_sq = p_radius * p_radius;
    const int count = gaussians.size();

    for (int i = 0; i < count; i++) {
        float distance_sq = p_center.distance_squared_to(gaussians[i].position);
        if (distance_sq > radius_sq) {
            continue;
        }

        indices.push_back(i);

        // Capture the effective color (runtime overlay if present, else base).
        if (i < (int)edit_state.runtime_colors.size() &&
                i < (int)edit_state.runtime_color_flags.size() &&
                edit_state.runtime_color_flags[i]) {
            colors.push_back(edit_state.runtime_colors[i]);
        } else {
            colors.push_back(gaussians[i].sh_dc);
        }

        // Capture the effective opacity.
        if (i < (int)edit_state.runtime_opacities.size() &&
                i < (int)edit_state.runtime_opacity_flags.size() &&
                edit_state.runtime_opacity_flags[i]) {
            opacities.push_back(edit_state.runtime_opacities[i]);
        } else {
            opacities.push_back(gaussians[i].opacity);
        }
    }

    state["indices"] = indices;
    state["colors"] = colors;
    state["opacities"] = opacities;
    return state;
}

void GaussianData::restore_brush_stroke(const Dictionary &p_saved_state) {
    ERR_FAIL_COND(!p_saved_state.has("indices"));
    ERR_FAIL_COND(!p_saved_state.has("colors"));
    ERR_FAIL_COND(!p_saved_state.has("opacities"));

    PackedInt32Array indices = p_saved_state["indices"];
    PackedColorArray colors = p_saved_state["colors"];
    PackedFloat32Array opacities = p_saved_state["opacities"];

    ERR_FAIL_COND(indices.size() != colors.size());
    ERR_FAIL_COND(indices.size() != opacities.size());

    {
        MutexLock lock(sh_mutex);
        ERR_FAIL_COND(gaussians.is_empty());

        const int n = indices.size();
        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            ERR_FAIL_INDEX(idx, (int)gaussians.size());

            _set_runtime_color_locked(idx, colors[i]);
            _set_runtime_opacity_locked(idx, opacities[i]);
        }
    }

    emit_changed();
}
