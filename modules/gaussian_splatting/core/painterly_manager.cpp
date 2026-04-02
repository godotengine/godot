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
}

void PainterlyManager::configure(const Settings &p_settings) {
    settings = p_settings;
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
