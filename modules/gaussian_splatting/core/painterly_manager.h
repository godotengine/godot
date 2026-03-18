#ifndef PAINTERLY_MANAGER_H
#define PAINTERLY_MANAGER_H

#include "core/math/math_funcs.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "../lod/adaptive_lod_system.h"
#include "gaussian_data.h"

namespace GaussianSplatting {

class PainterlyManager {
public:
    struct Settings {
        uint32_t base_seed = 1337;
        float blend_rate = 4.0f;           // How quickly transitions reach the new seed.
        float hold_strength = 0.2f;        // Portion of previous seed preserved when degrading LOD.
        float jitter_amplitude = 0.35f;    // Maximum painterly jitter strength in UV units.
    };

    PainterlyManager();

    void configure(const Settings &p_settings);

    void ensure_metadata_for_level(Vector<GaussianData> &splats, uint32_t lod_level);

    void apply_temporal_smoothing(AdaptiveLODSystem::LODSelection &selection, float delta_time);

    const Settings &get_settings() const { return settings; }

private:
    Settings settings;
    uint64_t frame_counter;

    struct HistoryEntry {
        PainterlyMetadata active_metadata;
        PainterlyMetadata previous_metadata;
        uint32_t active_seed;
        uint32_t previous_seed;
        uint8_t last_lod;
        float blend;
        bool initialized;

        HistoryEntry()
            : active_metadata(),
              previous_metadata(),
              active_seed(0),
              previous_seed(0),
              last_lod(0),
              blend(1.0f),
              initialized(false) {}
    };

    HashMap<uint32_t, HistoryEntry> history;

    PainterlyMetadata generate_metadata(const GaussianData &splat, uint32_t lod_level) const;
    PainterlyMetadata blend_metadata(const PainterlyMetadata &from, const PainterlyMetadata &to, float t) const;

    static uint32_t hash_combine(uint32_t seed, uint32_t value);
    static float hash_to_unit_float(uint32_t value);
    static Vector2 pick_blue_noise(uint32_t hash);
};

} // namespace GaussianSplatting

#endif // PAINTERLY_MANAGER_H
