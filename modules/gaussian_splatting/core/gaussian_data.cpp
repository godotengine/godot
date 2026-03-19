/**
 * gaussian_data.cpp — Core data management for GaussianData resource.
 *
 * Contains constructors, _bind_methods, batch setters, SH operations,
 * chunk snapshots, and basic statistics (get_aabb, get_memory_usage).
 *
 * Companion .cpp files hold extracted subsystem implementations:
 *   gaussian_data_edits.cpp        — Runtime modification overlay / brush strokes
 *   gaussian_data_io.cpp           — File I/O (PLY load/save, asset population)
 *   gaussian_data_octree.cpp       — Spatial acceleration (octree, frustum gather)
 *   gaussian_data_gpu.cpp          — GPU buffer management / payload validation
 *   gaussian_data_animation.cpp    — Animation state machine integration
 *   gaussian_data_color_grading.cpp — Color grading bake/restore
 */

#include "gaussian_data.h"
#include "gs_project_settings.h"
#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/templates/hash_set.h"
#include "core/templates/sort_array.h"
#include "core/math/math_funcs.h"
#include "core/templates/span.h"
#include "core/os/time.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"
#include "../io/gaussian_data_loader.h"
#include "gaussian_splat_asset.h"
#include "../persistence/incremental_saver.h"
#include "../renderer/gaussian_gpu_layout.h"
#include "../logger/gs_logger.h"
#include "../interfaces/sync_policy.h"
#include "../resources/color_grading_resource.h"
#include <algorithm>
#include <cmath>

// Static member initialization
RenderingDevice* GaussianData::cached_rd = nullptr;

// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).

namespace {
template <typename T, typename Container>
void copy_local_vector(LocalVector<T> &r_target, const Container &p_source) {
    uint32_t count = p_source.size();
    r_target.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        r_target[i] = p_source[i];
    }
}
} // namespace

RenderingDevice* GaussianData::get_rendering_device() {
    // DEPRECATED: This returns the main instance which cannot submit/sync
    // Use GaussianSplatManager::get_primary_rendering_device() instead
    // Keeping for backward compatibility only
    WARN_PRINT_ONCE("GaussianData::get_rendering_device() is deprecated - returns main instance");
    if (!cached_rd) {
        cached_rd = RenderingDevice::get_singleton();
        if (!cached_rd) {
            GS_LOG_ERROR_DEFAULT("Failed to get RenderingDevice singleton");
        }
    }
    return cached_rd;
}

GaussianData::GaussianData() {
    // Initialize with default values
    sh_degree = 0;
    sh_first_order_count = 0;
    sh_high_order_count = 0;
    sh_high_order_capacity = 0;
    is_2d_mode = false;
}

GaussianData::~GaussianData() {
    // Clean up resources
    gaussians.clear();
    octree.clear();
}

void GaussianData::_bind_methods() {
    // Core data management
    ClassDB::bind_method(D_METHOD("resize", "count"), &GaussianData::resize);
    // Note: Can't bind methods that return custom structs directly
    // ClassDB::bind_method(D_METHOD("set_gaussian", "index", "gaussian"), &GaussianData::set_gaussian);
    // ClassDB::bind_method(D_METHOD("get_gaussian", "index"), &GaussianData::get_gaussian);

    // Batch operations
    ClassDB::bind_method(D_METHOD("set_positions", "positions"), &GaussianData::set_positions);
    ClassDB::bind_method(D_METHOD("set_scales", "scales"), &GaussianData::set_scales);
    ClassDB::bind_method(D_METHOD("set_rotations", "rotations"), &GaussianData::set_rotations);
    ClassDB::bind_method(D_METHOD("set_opacities", "opacities"), &GaussianData::set_opacities);
    ClassDB::bind_method(D_METHOD("set_spherical_harmonics", "sh_data"), static_cast<void (GaussianData::*)(const PackedFloat32Array &)>(&GaussianData::set_spherical_harmonics));
    ClassDB::bind_method(D_METHOD("get_spherical_harmonics", "splat_idx"), &GaussianData::get_spherical_harmonics);
    ClassDB::bind_method(D_METHOD("has_full_sh"), &GaussianData::has_full_sh);
    ClassDB::bind_method(D_METHOD("get_sh_degree"), &GaussianData::get_sh_degree);
    ClassDB::bind_method(D_METHOD("set_palette_ids", "palette_ids"), &GaussianData::set_palette_ids);
    ClassDB::bind_method(D_METHOD("set_painterly_flags", "painterly_flags"), &GaussianData::set_painterly_flags);
    ClassDB::bind_method(D_METHOD("get_brush_override_ids"), &GaussianData::get_brush_override_ids);
    ClassDB::bind_method(D_METHOD("get_brush_override_ids_buffer"), &GaussianData::get_brush_override_ids_buffer);
    ClassDB::bind_method(D_METHOD("set_brush_override_ids", "brush_override_ids"), &GaussianData::set_brush_override_ids);
    ClassDB::bind_method(D_METHOD("set_brush_axes", "brush_axes"), &GaussianData::set_brush_axes);
    ClassDB::bind_method(D_METHOD("set_stroke_ages", "stroke_ages"), &GaussianData::set_stroke_ages);

    // 2D mode
    ClassDB::bind_method(D_METHOD("set_2d_mode", "enabled"), &GaussianData::set_2d_mode);
    ClassDB::bind_method(D_METHOD("get_2d_mode"), &GaussianData::get_2d_mode);
    ClassDB::bind_method(D_METHOD("set_normals", "normals"), &GaussianData::set_normals);

    // File I/O
    ClassDB::bind_method(D_METHOD("load_from_file", "path"), &GaussianData::load_from_file);
    ClassDB::bind_method(D_METHOD("save_to_file", "path"), &GaussianData::save_to_file);

    // Spatial queries
    ClassDB::bind_method(D_METHOD("build_octree", "max_depth", "min_gaussians"), &GaussianData::build_octree, DEFVAL(8), DEFVAL(32));
    ClassDB::bind_method(D_METHOD("query_octree", "bounds"), &GaussianData::query_octree);

    // Statistics
    ClassDB::bind_method(D_METHOD("get_count"), &GaussianData::get_count);
    ClassDB::bind_method(D_METHOD("get_aabb"), &GaussianData::get_aabb);
    ClassDB::bind_method(D_METHOD("get_memory_usage"), &GaussianData::get_memory_usage);

    // Animation system (v0.6.0)
    ClassDB::bind_method(D_METHOD("set_animation_state_machine", "animation"), &GaussianData::set_animation_state_machine);
    ClassDB::bind_method(D_METHOD("get_animation_state_machine"), &GaussianData::get_animation_state_machine);
    ClassDB::bind_method(D_METHOD("has_animation"), &GaussianData::has_animation);
    ClassDB::bind_method(D_METHOD("update_animation", "delta"), &GaussianData::update_animation);
    ClassDB::bind_method(D_METHOD("apply_animation_at_time", "time"), &GaussianData::apply_animation_at_time);
    ClassDB::bind_method(D_METHOD("set_animation_enabled", "enabled"), &GaussianData::set_animation_enabled);
    ClassDB::bind_method(D_METHOD("is_animation_enabled"), &GaussianData::is_animation_enabled);
    ClassDB::bind_method(D_METHOD("set_incremental_saver", "saver"), &GaussianData::set_incremental_saver);
    ClassDB::bind_method(D_METHOD("get_incremental_saver"), &GaussianData::get_incremental_saver);
    ClassDB::bind_method(D_METHOD("get_animated_position", "index", "time"), &GaussianData::get_animated_position, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("get_animated_color", "index", "time"), &GaussianData::get_animated_color, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("get_animated_opacity", "index", "time"), &GaussianData::get_animated_opacity, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("get_animated_scale", "index", "time"), &GaussianData::get_animated_scale, DEFVAL(-1.0f));
    ClassDB::bind_method(D_METHOD("get_animated_rotation", "index", "time"), &GaussianData::get_animated_rotation, DEFVAL(-1.0f));

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "2d_mode"), "set_2d_mode", "get_2d_mode");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "count", PROPERTY_HINT_RANGE, "0,10000000,1"), "", "get_count");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "animation_enabled"), "set_animation_enabled", "is_animation_enabled");

    // Runtime painting utilities
    ClassDB::bind_method(D_METHOD("set_runtime_position", "index", "position"), &GaussianData::set_runtime_position);
    ClassDB::bind_method(D_METHOD("set_runtime_color", "index", "color"), &GaussianData::set_runtime_color);
    ClassDB::bind_method(D_METHOD("set_runtime_opacity", "index", "opacity"), &GaussianData::set_runtime_opacity);
    ClassDB::bind_method(D_METHOD("apply_color_range", "start", "count", "color"), &GaussianData::apply_color_range);
    ClassDB::bind_method(D_METHOD("mark_range_dirty", "start", "count"), &GaussianData::mark_range_dirty);
    ClassDB::bind_method(D_METHOD("commit_runtime_changes"), &GaussianData::commit_runtime_changes);
    ClassDB::bind_method(D_METHOD("revert_runtime_changes"), &GaussianData::revert_runtime_changes);
    ClassDB::bind_method(D_METHOD("apply_brush_stroke", "center", "radius", "color", "opacity", "hardness"), &GaussianData::apply_brush_stroke);
    ClassDB::bind_method(D_METHOD("get_brush_strokes"), &GaussianData::get_brush_strokes);
    ClassDB::bind_method(D_METHOD("clear_brush_strokes"), &GaussianData::clear_brush_strokes);
    ClassDB::bind_method(D_METHOD("set_brush_strokes", "strokes"), &GaussianData::set_brush_strokes);
    ClassDB::bind_method(D_METHOD("capture_brush_affected_state", "center", "radius"), &GaussianData::capture_brush_affected_state);
    ClassDB::bind_method(D_METHOD("restore_brush_stroke", "saved_state"), &GaussianData::restore_brush_stroke);
}

void GaussianData::_on_gaussian_storage_changed() {
    RWLockWrite lock(data_rwlock);
    _on_gaussian_storage_changed_locked();
}

void GaussianData::_on_gaussian_storage_changed_locked() {
    octree.clear();
    octree_dirty = true;

    _clear_runtime_modifications_locked();
    _clear_brush_strokes_locked();

    {
        MutexLock anim_lock(animation_cache_mutex);
        animated_positions_cache.clear();
        animated_colors_cache.clear();
        animated_opacities_cache.clear();
        animated_positions_valid_cache.clear();
        animated_colors_valid_cache.clear();
        animated_opacities_valid_cache.clear();
        last_animation_time = -1.0f;
        animation_cache_dirty = true;
    }

#ifndef GS_SILENCE_LOGS
    // DEBUG: Log state BEFORE calculating SH count
    if (gs::settings::is_data_log_enabled()) {
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianData::_on_storage_changed] Called with %d gaussians", gaussians.size()));
        if (gaussians.size() > 0) {
            const Gaussian &sample = gaussians[0];
            GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianData::_on_storage_changed] BEFORE calc: gaussian[0].sh_1[0] = (%f, %f, %f)",
                sample.sh_1[0].x, sample.sh_1[0].y, sample.sh_1[0].z));
            GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianData::_on_storage_changed] BEFORE calc: gaussian[0].sh_1[1] = (%f, %f, %f)",
                sample.sh_1[1].x, sample.sh_1[1].y, sample.sh_1[1].z));
        }
    }
#endif

    uint32_t max_first_order = 0;
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        const Gaussian &g = gaussians[i];
        for (uint32_t j = 0; j < 3; j++) {
            if (!g.sh_1[j].is_zero_approx()) {
                max_first_order = MAX(max_first_order, j + 1);
            }
        }
    }

    sh_first_order_count = max_first_order;
    sh_high_order_count = 0;
    sh_high_order_capacity = 0;
    sh_high_order_coefficients.clear();

    uint32_t total_vectors = 1 + sh_first_order_count + sh_high_order_count;
    uint32_t degree = 0;
    while ((degree + 1) * (degree + 1) <= total_vectors) {
        degree++;
    }
    sh_degree = degree > 0 ? degree - 1 : 0;

#ifndef GS_SILENCE_LOGS
    // DEBUG: Log calculated SH count
    if (gs::settings::is_data_log_enabled()) {
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianData::_on_storage_changed] Calculated sh_first_order_count = %d", sh_first_order_count));
        if (sh_first_order_count == 0) {
            GS_LOG_WARN_DEFAULT("[GaussianData::_on_storage_changed] WARNING: sh_first_order_count is ZERO - no SH data will be sent to GPU!");
        }
    }
#endif

    if (animation_state_machine.is_valid()) {
        animation_state_machine->set_splat_count(gaussians.size());
    }
}

void GaussianData::set_gaussians(const LocalVector<Gaussian> &p_gaussians) {
    RWLockWrite lock(data_rwlock);
    copy_local_vector(gaussians, p_gaussians);
    _on_gaussian_storage_changed_locked();
}

void GaussianData::set_gaussians(const Vector<Gaussian> &p_gaussians) {
    RWLockWrite lock(data_rwlock);
    copy_local_vector(gaussians, p_gaussians);
    _on_gaussian_storage_changed_locked();
}

void GaussianData::set_gaussian_payload(const LocalVector<Gaussian> &p_gaussians,
        const LocalVector<Vector3> &p_sh_high_order_coefficients,
        uint32_t p_sh_first_order_count,
        uint32_t p_sh_high_order_count,
        bool p_is_2d_mode) {
    RWLockWrite lock(data_rwlock);

    copy_local_vector(gaussians, p_gaussians);

    _clear_runtime_modifications_locked();
    _clear_brush_strokes_locked();
    octree.clear();
    octree_dirty = true;

    {
        MutexLock anim_lock(animation_cache_mutex);
        animated_positions_cache.clear();
        animated_colors_cache.clear();
        animated_opacities_cache.clear();
        animated_positions_valid_cache.clear();
        animated_colors_valid_cache.clear();
        animated_opacities_valid_cache.clear();
        last_animation_time = -1.0f;
        animation_cache_dirty = true;
    }

    sh_first_order_count = MIN<uint32_t>(p_sh_first_order_count, 3u);
    sh_high_order_count = p_sh_high_order_count;
    sh_high_order_capacity = p_sh_high_order_count;
    sh_high_order_coefficients.clear();
    if (sh_high_order_count > 0) {
        const uint64_t expected = uint64_t(gaussians.size()) * uint64_t(sh_high_order_count);
        if (expected == uint64_t(p_sh_high_order_coefficients.size())) {
            copy_local_vector(sh_high_order_coefficients, p_sh_high_order_coefficients);
        } else {
            sh_high_order_count = 0;
            sh_high_order_capacity = 0;
            if (gs::settings::is_data_log_enabled()) {
                GS_LOG_STREAMING_WARN(vformat(
                        "[GaussianData] Ignoring mismatched high-order SH payload: expected=%d got=%d",
                        int(expected), int(p_sh_high_order_coefficients.size())));
            }
        }
    }

    uint32_t total_vectors = 1 + sh_first_order_count + sh_high_order_count;
    uint32_t degree = 0;
    while (((degree + 1) * (degree + 1)) <= total_vectors) {
        degree++;
    }
    sh_degree = degree > 0 ? degree - 1 : 0;
    is_2d_mode = p_is_2d_mode;

    if (animation_state_machine.is_valid()) {
        animation_state_machine->set_splat_count(gaussians.size());
    }
}

const Gaussian *GaussianData::get_gaussians() const {
    return gaussians.is_empty() ? nullptr : gaussians.ptr();
}

void GaussianData::resize(int p_count) {
    ERR_FAIL_COND(p_count < 0);
    RWLockWrite lock(data_rwlock);
    gaussians.resize(p_count);

    // Resize SH high-order coefficients (already holding data_rwlock write, no inner lock needed)
    if (sh_high_order_count > 0) {
        size_t expected_size = (size_t)p_count * sh_high_order_count;
        size_t previous_size = sh_high_order_coefficients.size();
        // Use slack factor: only reallocate when growing past capacity or shrinking below 50%.
        if (expected_size > previous_size || (previous_size > 0 && expected_size < previous_size / 2)) {
            size_t alloc_size = MAX(expected_size, previous_size);
            sh_high_order_coefficients.resize(alloc_size);
            for (size_t i = previous_size; i < alloc_size; i++) {
                sh_high_order_coefficients[i] = Vector3();
            }
        }
        sh_high_order_capacity = sh_high_order_count;
    } else {
        sh_high_order_coefficients.clear();
        sh_high_order_capacity = 0;
    }

    // Initialize new Gaussians with default values
    for (int i = 0; i < p_count; i++) {
        Gaussian &g = gaussians[i];
        g.position = Vector3();
        g.scale = Vector3(1, 1, 1);
        g.rotation = Quaternion();
        g.opacity = 1.0f;
        g.sh_dc = Color(1, 1, 1, 1);
        g.normal = Vector3(0, 1, 0);
        g.area = 1.0f;
        g.brush_axes = Vector2(1.0f, 1.0f);
        g.stroke_age = 0.0f;
        g.painterly_meta = gaussian_pack_painterly_meta(0);

        // Initialize SH coefficients
        for (int j = 0; j < 3; j++) {
            g.sh_1[j] = Vector3();
        }
    }

    _on_gaussian_storage_changed_locked();
}

void GaussianData::set_gaussian(int p_index, const Gaussian &p_gaussian) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_INDEX(p_index, (int)gaussians.size());
    gaussians[p_index] = p_gaussian;
    octree_dirty = true;
    {
        MutexLock anim_lock(animation_cache_mutex);
        animation_cache_dirty = true;
    }
}

Gaussian GaussianData::get_gaussian(int p_index) const {
    ERR_FAIL_INDEX_V(p_index, (int)gaussians.size(), Gaussian());
    return gaussians[p_index];
}

const Vector3 *GaussianData::get_sh_high_order_coefficients_ptr() const {
    return sh_high_order_coefficients.is_empty() ? nullptr : sh_high_order_coefficients.ptr();
}

bool GaussianData::capture_chunk_snapshot(uint32_t p_start, uint32_t p_count,
        LocalVector<Gaussian> &r_gaussians,
        LocalVector<Vector3> &r_sh_high_order,
        uint32_t &r_sh_first_order_count,
        uint32_t &r_sh_high_order_count) const {
    RWLockRead lock(data_rwlock);

#ifdef DEBUG_ENABLED
    const uint32_t gaussians_size_before_copy = gaussians.size();
    const uint32_t sh_coeff_size_before_copy = sh_high_order_coefficients.size();
#endif

    r_gaussians.clear();
    r_sh_high_order.clear();
    r_sh_first_order_count = sh_first_order_count;
    r_sh_high_order_count = sh_high_order_count;

    if (p_count == 0) {
        return true;
    }

    const uint64_t gaussian_count = static_cast<uint64_t>(gaussians.size());
    const uint64_t start = static_cast<uint64_t>(p_start);
    const uint64_t count = static_cast<uint64_t>(p_count);
    if (start + count > gaussian_count) {
        r_sh_first_order_count = 0;
        r_sh_high_order_count = 0;
        return false;
    }

    r_gaussians.resize(p_count);
    Gaussian *gaussian_dst = r_gaussians.ptr();
    for (uint32_t i = 0; i < p_count; i++) {
        gaussian_dst[i] = gaussians[p_start + i];
    }

#ifdef DEBUG_ENABLED
    ERR_FAIL_COND_V_MSG(gaussians_size_before_copy != gaussians.size(), false,
            "[GaussianData] Snapshot failed: gaussian storage changed while capture lock was held.");
#endif

    if (r_sh_high_order_count == 0) {
        return true;
    }

    const uint64_t high_order_count = static_cast<uint64_t>(r_sh_high_order_count);
    const uint64_t coeff_offset = start * high_order_count;
    const uint64_t coeff_count = count * high_order_count;
    const uint64_t coeff_end = coeff_offset + coeff_count;
    const uint64_t coeff_storage_size = static_cast<uint64_t>(sh_high_order_coefficients.size());
    if (coeff_count > UINT32_MAX || coeff_end > coeff_storage_size) {
        r_sh_high_order_count = 0;
        r_sh_high_order.clear();
        return false;
    }

    r_sh_high_order.resize(static_cast<uint32_t>(coeff_count));
    Vector3 *coeff_dst = r_sh_high_order.ptr();
    for (uint32_t i = 0; i < static_cast<uint32_t>(coeff_count); i++) {
        coeff_dst[i] = sh_high_order_coefficients[coeff_offset + i];
    }

#ifdef DEBUG_ENABLED
    ERR_FAIL_COND_V_MSG(sh_coeff_size_before_copy != sh_high_order_coefficients.size(), false,
            "[GaussianData] Snapshot failed: SH coefficient storage changed while capture lock was held.");
#endif

    return true;
}

bool GaussianData::capture_indexed_chunk_snapshot(const uint32_t *p_indices, uint32_t p_count,
        LocalVector<Gaussian> &r_gaussians,
        LocalVector<Vector3> &r_sh_high_order,
        uint32_t &r_sh_first_order_count,
        uint32_t &r_sh_high_order_count) const {
    RWLockRead lock(data_rwlock);

#ifdef DEBUG_ENABLED
    const uint32_t gaussians_size_before_copy = gaussians.size();
    const uint32_t sh_coeff_size_before_copy = sh_high_order_coefficients.size();
#endif

    r_gaussians.clear();
    r_sh_high_order.clear();
    r_sh_first_order_count = sh_first_order_count;
    r_sh_high_order_count = sh_high_order_count;

    if (p_count == 0) {
        return true;
    }
    if (p_indices == nullptr) {
        r_sh_first_order_count = 0;
        r_sh_high_order_count = 0;
        return false;
    }

    const uint32_t gaussian_count = static_cast<uint32_t>(gaussians.size());
    r_gaussians.resize(p_count);
    Gaussian *gaussian_dst = r_gaussians.ptr();
    for (uint32_t i = 0; i < p_count; i++) {
        const uint32_t source_index = p_indices[i];
        if (source_index >= gaussian_count) {
            r_gaussians.clear();
            r_sh_high_order.clear();
            r_sh_first_order_count = 0;
            r_sh_high_order_count = 0;
            return false;
        }
        gaussian_dst[i] = gaussians[source_index];
    }

#ifdef DEBUG_ENABLED
    ERR_FAIL_COND_V_MSG(gaussians_size_before_copy != gaussians.size(), false,
            "[GaussianData] Indexed snapshot failed: gaussian storage changed while capture lock was held.");
#endif

    if (r_sh_high_order_count == 0) {
        return true;
    }

    const uint64_t high_order_count = static_cast<uint64_t>(r_sh_high_order_count);
    const uint64_t coeff_count = static_cast<uint64_t>(p_count) * high_order_count;
    const uint64_t coeff_storage_size = static_cast<uint64_t>(sh_high_order_coefficients.size());
    if (coeff_count > UINT32_MAX) {
        r_sh_high_order_count = 0;
        r_sh_high_order.clear();
        return false;
    }

    r_sh_high_order.resize(static_cast<uint32_t>(coeff_count));
    Vector3 *coeff_dst = r_sh_high_order.ptr();
    uint64_t write_offset = 0;
    for (uint32_t i = 0; i < p_count; i++) {
        const uint64_t coeff_offset = static_cast<uint64_t>(p_indices[i]) * high_order_count;
        if (coeff_offset + high_order_count > coeff_storage_size) {
            r_sh_high_order_count = 0;
            r_sh_high_order.clear();
            return false;
        }
        for (uint32_t c = 0; c < r_sh_high_order_count; c++) {
            coeff_dst[write_offset++] = sh_high_order_coefficients[coeff_offset + c];
        }
    }

#ifdef DEBUG_ENABLED
    ERR_FAIL_COND_V_MSG(sh_coeff_size_before_copy != sh_high_order_coefficients.size(), false,
            "[GaussianData] Indexed snapshot failed: SH coefficient storage changed while capture lock was held.");
#endif

    return true;
}

void GaussianData::set_positions(const PackedVector3Array &p_positions) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_positions.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].position = p_positions[i];
    }
}

void GaussianData::set_scales(const PackedVector3Array &p_scales) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_scales.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].scale = p_scales[i];
    }
}

void GaussianData::set_rotations(const TypedArray<Quaternion> &p_rotations) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_rotations.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].rotation = p_rotations[i];
    }
}

void GaussianData::set_opacities(const PackedFloat32Array &p_opacities) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_opacities.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].opacity = p_opacities[i];
    }
}

void GaussianData::set_spherical_harmonics(const PackedFloat32Array &p_sh_data) {
    RWLockWrite lock(data_rwlock);

    uint32_t gaussian_count = gaussians.size();
    ERR_FAIL_COND_MSG(gaussian_count == 0, "No gaussians available for SH assignment");

    int total_components = p_sh_data.size();
    ERR_FAIL_COND_MSG(total_components % (int)gaussian_count != 0,
            "Spherical harmonics array size does not match gaussian count");

    int floats_per_gaussian = total_components / (int)gaussian_count;
    ERR_FAIL_COND_MSG(floats_per_gaussian < 3,
            "Spherical harmonics data must contain at least DC coefficients");
    ERR_FAIL_COND_MSG(floats_per_gaussian % 3 != 0,
            "Spherical harmonics coefficients must be RGB triplets");

    sh_first_order_count = 0;
    sh_high_order_count = 0;
    sh_high_order_capacity = 0;
    sh_degree = 0;
    sh_high_order_coefficients.clear();

    const float *data_ptr = p_sh_data.ptr();
    for (uint32_t i = 0; i < gaussian_count; i++) {
        _set_spherical_harmonics_locked(i, data_ptr + i * floats_per_gaussian, floats_per_gaussian);
    }
}

void GaussianData::set_palette_ids(const PackedInt32Array &p_palette_ids) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_palette_ids.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        int32_t value = p_palette_ids[i];
        value = CLAMP(value, 0, 65535);
        Gaussian &g = gaussians[i];
        g.painterly_meta = gaussian_set_palette_id(g.painterly_meta, (uint16_t)value);
    }
}

void GaussianData::set_painterly_flags(const PackedInt32Array &p_flags) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_flags.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        int32_t value = p_flags[i];
        value = CLAMP(value, 0, 65535);
        Gaussian &g = gaussians[i];
        g.painterly_meta = gaussian_set_painterly_flags(g.painterly_meta, (uint16_t)value);
    }
}

PackedInt32Array GaussianData::get_brush_override_ids() const {
    RWLockRead lock(data_rwlock);

    PackedInt32Array result;
    result.resize(gaussians.size());
    int32_t *write = result.ptrw();
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        write[i] = gaussian_get_brush_override_id(gaussians[i].painterly_meta);
    }
    return result;
}

void GaussianData::set_brush_override_ids(const PackedInt32Array &p_override_ids) {
    set_painterly_flags(p_override_ids);
}

PackedFloat32Array GaussianData::get_spherical_harmonics(int p_index) const {
    ERR_FAIL_INDEX_V(p_index, (int)gaussians.size(), PackedFloat32Array());

    RWLockRead lock(data_rwlock);

    uint32_t vector_count = 1 + sh_first_order_count + sh_high_order_count;
    PackedFloat32Array result;
    if (vector_count == 0) {
        return result;
    }

    result.resize(vector_count * 3);
    float *write = result.ptrw();
    int offset = 0;

    const Gaussian &g = gaussians[p_index];
    write[offset++] = g.sh_dc.r;
    write[offset++] = g.sh_dc.g;
    write[offset++] = g.sh_dc.b;

    for (uint32_t i = 0; i < sh_first_order_count; i++) {
        const Vector3 &coeff = g.sh_1[i];
        write[offset++] = coeff.x;
        write[offset++] = coeff.y;
        write[offset++] = coeff.z;
    }

    if (sh_high_order_count > 0 && !sh_high_order_coefficients.is_empty()) {
        size_t base = (size_t)p_index * sh_high_order_count;
        for (uint32_t i = 0; i < sh_high_order_count; i++) {
            const Vector3 &coeff = sh_high_order_coefficients[base + i];
            write[offset++] = coeff.x;
            write[offset++] = coeff.y;
            write[offset++] = coeff.z;
        }
    }

    return result;
}

bool GaussianData::has_full_sh() const {
    RWLockRead lock(data_rwlock);
    const uint32_t full_high_order = 12;
    return sh_high_order_count >= full_high_order;
}

void GaussianData::_set_spherical_harmonics_locked(int p_index, const float *p_coeffs, int p_count) {
    ERR_FAIL_INDEX(p_index, (int)gaussians.size());
    ERR_FAIL_COND_MSG(p_coeffs == nullptr, "Spherical harmonics pointer must be valid");
    ERR_FAIL_COND_MSG(p_count < 3, "Spherical harmonics data must include DC coefficients");
    ERR_FAIL_COND_MSG(p_count % 3 != 0, "Spherical harmonics data must be in RGB triplets");

    const uint32_t max_first_order = 3;
    int triplet_count = p_count / 3;
    int extra_triplets = triplet_count - 1;
    if (extra_triplets < 0) {
        extra_triplets = 0;
    }
    uint32_t first_order = (uint32_t)((extra_triplets < (int)max_first_order) ? extra_triplets : (int)max_first_order);
    uint32_t high_order = extra_triplets > (int)first_order ? (uint32_t)extra_triplets - first_order : 0;

    Gaussian &g = gaussians[p_index];
    g.sh_dc = Color(p_coeffs[0], p_coeffs[1], p_coeffs[2], 1.0f);

    for (uint32_t i = 0; i < max_first_order; i++) {
        if (i < first_order) {
            uint32_t base = 3 + i * 3;
            g.sh_1[i] = Vector3(p_coeffs[base + 0], p_coeffs[base + 1], p_coeffs[base + 2]);
        } else {
            g.sh_1[i] = Vector3();
        }
    }

    if (first_order > sh_first_order_count) {
        sh_first_order_count = first_order;
    }

    uint32_t required_high_order = sh_high_order_count > high_order ? sh_high_order_count : high_order;

    // Overflow check: prevent allocation overflow for large datasets.
    ERR_FAIL_COND(gaussians.size() * required_high_order > INT_MAX / sizeof(Vector3));

    if (required_high_order != sh_high_order_count) {
        // Per-splat coefficient count changed — must re-layout storage.
        // Apply slack factor: only reallocate when new size exceeds current capacity
        // or falls below 50% of capacity, to avoid reallocation thrashing.
        size_t new_logical_size = (size_t)gaussians.size() * required_high_order;
        size_t current_capacity = sh_high_order_coefficients.size();
        bool needs_realloc = (new_logical_size > current_capacity) ||
                (current_capacity > 0 && new_logical_size < current_capacity / 2);
        // Layout always changes when per-splat count changes, so we must
        // re-layout even if capacity is sufficient.
        LocalVector<Vector3> new_storage;
        size_t alloc_size = needs_realloc
                ? MAX(new_logical_size, current_capacity)
                : current_capacity;
        new_storage.resize(alloc_size);
        for (size_t i = 0; i < alloc_size; i++) {
            new_storage[i] = Vector3();
        }
        for (uint32_t idx = 0; idx < gaussians.size(); idx++) {
            if (sh_high_order_count == 0) {
                continue;
            }
            size_t old_base = (size_t)idx * sh_high_order_count;
            size_t new_base = (size_t)idx * required_high_order;
            for (uint32_t j = 0; j < sh_high_order_count; j++) {
                new_storage[new_base + j] = sh_high_order_coefficients[old_base + j];
            }
        }
        sh_high_order_coefficients = new_storage;
        sh_high_order_count = required_high_order;
        sh_high_order_capacity = (uint32_t)(alloc_size / MAX((size_t)gaussians.size(), (size_t)1));
    } else if (sh_high_order_count > 0) {
        size_t expected_size = (size_t)gaussians.size() * sh_high_order_count;
        if (sh_high_order_coefficients.size() < expected_size) {
            // Only grow — use slack factor to avoid repeated resizes.
            size_t alloc_size = MAX(expected_size, (size_t)sh_high_order_coefficients.size());
            size_t previous_size = sh_high_order_coefficients.size();
            sh_high_order_coefficients.resize(alloc_size);
            for (size_t i = previous_size; i < alloc_size; i++) {
                sh_high_order_coefficients[i] = Vector3();
            }
            sh_high_order_capacity = (uint32_t)(alloc_size / MAX((size_t)gaussians.size(), (size_t)1));
        }
    }

    if (sh_high_order_count > 0) {
        size_t base = (size_t)p_index * sh_high_order_count;
        for (uint32_t i = 0; i < sh_high_order_count; i++) {
            if (i < high_order) {
                uint32_t offset = 3 + (first_order + i) * 3;
                sh_high_order_coefficients[base + i] = Vector3(
                        p_coeffs[offset + 0],
                        p_coeffs[offset + 1],
                        p_coeffs[offset + 2]);
            } else {
                sh_high_order_coefficients[base + i] = Vector3();
            }
        }
    }

    uint32_t total_vectors = 1 + sh_first_order_count + sh_high_order_count;
    uint32_t degree = 0;
    while (((degree + 1) * (degree + 1)) <= total_vectors) {
        degree++;
    }
    sh_degree = degree > 0 ? degree - 1 : 0;

    octree_dirty = true;

#ifndef GS_SILENCE_LOGS
    // DEBUG: Log first gaussian SH data after setting
    if (gs::settings::is_data_log_enabled() && p_index == 0) {
        const Gaussian &verify = gaussians[p_index];
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianData::set_SH] After set_SH: sh_1[0] = (%f, %f, %f)",
            verify.sh_1[0].x, verify.sh_1[0].y, verify.sh_1[0].z));
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianData::set_SH] After set_SH: sh_1[1] = (%f, %f, %f)",
            verify.sh_1[1].x, verify.sh_1[1].y, verify.sh_1[1].z));
        GS_LOG_DEBUG(gs_logger::Category::GENERAL, vformat("[GaussianData::set_SH] first_order=%d, high_order=%d, sh_first_order_count=%d",
            first_order, high_order, sh_first_order_count));
    }
#endif
}

void GaussianData::set_spherical_harmonics(int p_index, const float *p_coeffs, int p_count) {
    RWLockWrite lock(data_rwlock);
    _set_spherical_harmonics_locked(p_index, p_coeffs, p_count);
}

void GaussianData::set_brush_axes(const PackedVector2Array &p_brush_axes) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_brush_axes.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].brush_axes = p_brush_axes[i];
    }
}

void GaussianData::set_stroke_ages(const PackedFloat32Array &p_stroke_ages) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_stroke_ages.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].stroke_age = p_stroke_ages[i];
    }
}

void GaussianData::set_2d_mode(bool p_enabled) {
    RWLockWrite lock(data_rwlock);
    is_2d_mode = p_enabled;
}

void GaussianData::set_normals(const PackedVector3Array &p_normals) {
    RWLockWrite lock(data_rwlock);
    ERR_FAIL_COND(p_normals.size() != (int)gaussians.size());
    for (uint32_t i = 0; i < gaussians.size(); i++) {
        gaussians[i].normal = p_normals[i];
    }
}


AABB GaussianData::get_aabb() const {
    if (gaussians.is_empty()) {
        return AABB();
    }

    // Initialize with first gaussian including its scale extent
    const Vector3 &first_pos = gaussians[0].position;
    const Vector3 &first_scale = gaussians[0].scale;
    Vector3 first_extent = Vector3(Math::abs(first_scale.x), Math::abs(first_scale.y), Math::abs(first_scale.z)) * 3.0f;
    Vector3 min_pos = first_pos - first_extent;
    Vector3 max_pos = first_pos + first_extent;

    for (uint32_t i = 1; i < gaussians.size(); i++) {
        const Vector3 &pos = gaussians[i].position;
        const Vector3 &scale = gaussians[i].scale;

        // Include scale in bounds calculation (use abs to handle negative scales)
        Vector3 extent = Vector3(Math::abs(scale.x), Math::abs(scale.y), Math::abs(scale.z)) * 3.0f; // 3 sigma coverage
        min_pos = min_pos.min(pos - extent);
        max_pos = max_pos.max(pos + extent);
    }

    return AABB(min_pos, max_pos - min_pos);
}

AABB GaussianData::compute_aabb() const {
    return get_aabb();
}

float GaussianData::get_memory_usage() const {
    // Calculate memory usage in bytes
    float base_size = sizeof(Gaussian) * gaussians.size();
    float octree_size = sizeof(OctreeNode) * octree.size();
    return base_size + octree_size;
}
