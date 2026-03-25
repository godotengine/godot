#include "painterly_manager.h"
#include "core/math/math_funcs.h"

namespace GaussianSplatting {

namespace {
static const Vector2 BLUE_NOISE_TABLE[] = {
    Vector2(0.0312f, 0.5625f), Vector2(0.7812f, 0.1562f),
    Vector2(0.4062f, 0.9062f), Vector2(0.9687f, 0.4687f),
    Vector2(0.2187f, 0.0312f), Vector2(0.5937f, 0.2812f),
    Vector2(0.3437f, 0.7187f), Vector2(0.7187f, 0.9687f),
    Vector2(0.0937f, 0.4062f), Vector2(0.8437f, 0.8437f),
    Vector2(0.4687f, 0.0937f), Vector2(0.9062f, 0.5312f),
    Vector2(0.1562f, 0.7812f), Vector2(0.5312f, 0.0312f),
    Vector2(0.2812f, 0.4687f), Vector2(0.9687f, 0.2187f)
};
}

PainterlyManager::PainterlyManager() {
    frame_counter = 0;
}

void PainterlyManager::configure(const Settings &p_settings) {
    settings = p_settings;
    frame_counter = 0;
    history.clear();
}

void PainterlyManager::ensure_metadata_for_level(Vector<GaussianData> &splats, uint32_t lod_level) {
    for (uint32_t i = 0; i < splats.size(); i++) {
        GaussianData &splat = splats.write[i];
        PainterlyMetadata &meta = splat.painterly;
        if (meta.temporal_seed == 0) {
            meta = generate_metadata(splat, lod_level);
        } else {
            uint32_t combined = hash_combine(meta.temporal_seed, lod_level * 2654435761u);
            meta.temporal_seed = combined != 0 ? combined : 1;
        }
    }
}

void PainterlyManager::apply_temporal_smoothing(AdaptiveLODSystem::LODSelection &selection, float delta_time) {
    if (selection.visible_indices.is_empty()) {
        return;
    }

    frame_counter++;

    uint32_t target_size = selection.visible_indices.size();
    selection.painterly_metadata.resize(target_size);
    selection.painterly_seeds.resize(target_size);
    selection.painterly_prev_seeds.resize(target_size);
    selection.painterly_blend_weights.resize(target_size);

    float step = settings.blend_rate * delta_time;
    const float blend_step = MIN(step, 1.0f);
    float hold = settings.hold_strength;

    for (uint32_t i = 0; i < target_size; i++) {
        uint32_t idx = selection.visible_indices[i];
        const PainterlyMetadata &current_meta = selection.painterly_metadata[i];
        uint32_t current_seed = current_meta.temporal_seed;
        uint8_t lod_level = selection.lod_levels[i];

        HistoryEntry &entry = history[idx];
        if (!entry.initialized) {
            entry.initialized = true;
            entry.active_seed = current_seed != 0 ? current_seed : 1;
            entry.previous_seed = entry.active_seed;
            entry.active_metadata = current_meta;
            entry.previous_metadata = current_meta;
            entry.blend = 1.0f;
            entry.last_lod = lod_level;
        } else {
            if (entry.active_seed != current_seed && current_seed != 0) {
                entry.previous_seed = entry.active_seed;
                entry.previous_metadata = entry.active_metadata;
                entry.active_seed = current_seed;
                entry.active_metadata = current_meta;
                entry.blend = 0.0f;
            } else {
                // Keep metadata in sync when seed stays the same.
                entry.active_metadata = blend_metadata(entry.active_metadata, current_meta, blend_step);
                entry.previous_metadata = entry.active_metadata;
                entry.previous_seed = entry.active_seed;
            }

            if (lod_level > entry.last_lod) {
                entry.blend = MIN(entry.blend, 1.0f - hold);
            }

            entry.last_lod = lod_level;
        }

        if (entry.blend < 1.0f) {
            entry.blend = MIN(1.0f, entry.blend + step);
        }

        PainterlyMetadata final_meta;
        if (entry.previous_seed == entry.active_seed || entry.blend >= 0.999f) {
            final_meta = entry.active_metadata;
            entry.previous_seed = entry.active_seed;
            entry.previous_metadata = entry.active_metadata;
            entry.blend = 1.0f;
        } else {
            final_meta = blend_metadata(entry.previous_metadata, entry.active_metadata, entry.blend);
        }

        selection.painterly_metadata[i] = final_meta;
        selection.painterly_seeds[i] = entry.active_seed;
        selection.painterly_prev_seeds[i] = entry.previous_seed;
        selection.painterly_blend_weights[i] = entry.blend;
    }
}

PainterlyMetadata PainterlyManager::generate_metadata(const GaussianData &splat, uint32_t lod_level) const {
    uint32_t seed = hash_combine(settings.base_seed, splat.index + 1);
    seed = hash_combine(seed, lod_level + 1);

    uint32_t noise_hash = hash_combine(seed, 0x9E3779B9u);
    uint32_t jitter_hash = hash_combine(seed, 0x7F4A7C15u);
    uint32_t angle_hash = hash_combine(seed, 0xA511E9B3u);
    uint32_t scale_hash = hash_combine(seed, 0x6C8E9CF5u);

    PainterlyMetadata meta;
    meta.temporal_seed = seed != 0 ? seed : 1;
    meta.blue_noise = pick_blue_noise(noise_hash);
    float jitter_offset_x = hash_to_unit_float(jitter_hash) - 0.5f;
    float jitter_offset_y = hash_to_unit_float(hash_combine(jitter_hash, 0x51633u)) - 0.5f;
    Vector2 blue_noise_offset = meta.blue_noise - Vector2(0.5f, 0.5f);
    meta.jitter = (blue_noise_offset * 0.5f + Vector2(jitter_offset_x, jitter_offset_y)) * settings.jitter_amplitude;
    meta.stroke_angle = hash_to_unit_float(angle_hash) * Math::TAU - Math::PI;
    meta.stroke_scale = 0.6f + 0.8f * hash_to_unit_float(scale_hash);
    meta.stability = 1.0f;

    // Slightly bias jitter direction by splat normal for stability.
    Vector3 normal = splat.normal.normalized();
    if (!normal.is_zero_approx()) {
        float normal_factor = Math::abs(normal.dot(Vector3(0, 1, 0)));
        meta.jitter *= Math::lerp(0.7f, 1.0f, normal_factor);
    }

    return meta;
}

PainterlyMetadata PainterlyManager::blend_metadata(const PainterlyMetadata &from, const PainterlyMetadata &to, float t) const {
    float weight_to = CLAMP(t, 0.0f, 1.0f);
    float weight_from = 1.0f - weight_to;

    PainterlyMetadata result;

    if (weight_to == 0.0f) {
        result.temporal_seed = to.temporal_seed != 0 ? to.temporal_seed : from.temporal_seed;
        result.jitter = from.jitter;
        result.blue_noise = from.blue_noise;
        result.stroke_scale = from.stroke_scale;
        result.stroke_angle = from.stroke_angle;
        result.stability = from.stability;
        return result;
    }

    if (weight_to == 1.0f) {
        result.temporal_seed = to.temporal_seed != 0 ? to.temporal_seed : from.temporal_seed;
        result.jitter = to.jitter;
        result.blue_noise = to.blue_noise;
        result.stroke_scale = to.stroke_scale;
        result.stroke_angle = to.stroke_angle;
        result.stability = to.stability;
        return result;
    }

    result.temporal_seed = to.temporal_seed != 0 ? to.temporal_seed : from.temporal_seed;
    result.jitter = from.jitter * weight_from + to.jitter * weight_to;
    result.blue_noise = from.blue_noise * weight_from + to.blue_noise * weight_to;
    result.stroke_scale = Math::lerp(from.stroke_scale, to.stroke_scale, weight_to);

    Vector2 dir_from(Math::cos(from.stroke_angle), Math::sin(from.stroke_angle));
    Vector2 dir_to(Math::cos(to.stroke_angle), Math::sin(to.stroke_angle));
    Vector2 dir_mix = dir_from * weight_from + dir_to * weight_to;
    if (dir_mix.length_squared() > 0.0001f) {
        result.stroke_angle = Math::atan2(dir_mix.y, dir_mix.x);
    } else {
        result.stroke_angle = to.stroke_angle;
    }

    result.stability = Math::lerp(from.stability, to.stability, weight_to);
    return result;
}

uint32_t PainterlyManager::hash_combine(uint32_t seed, uint32_t value) {
    seed ^= value + 0x9e3779b9u + (seed << 6) + (seed >> 2);
    return seed;
}

float PainterlyManager::hash_to_unit_float(uint32_t value) {
    return (value & 0x00FFFFFFu) / 16777215.0f;
}

Vector2 PainterlyManager::pick_blue_noise(uint32_t hash) {
    constexpr uint32_t sample_count = sizeof(BLUE_NOISE_TABLE) / sizeof(Vector2);
    uint32_t index = hash % sample_count;
    return BLUE_NOISE_TABLE[index];
}

} // namespace GaussianSplatting
