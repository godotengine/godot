#ifndef STREAMING_CONFIG_OVERRIDES_H
#define STREAMING_CONFIG_OVERRIDES_H

#include "core/string/ustring.h"
#include "streaming_vram_regulator.h"
#include "../lod/lod_config.h"

namespace GaussianStreamingTypes {

struct ConfigOverrides {
    bool override_chunk_culling = false;
    bool chunk_frustum_culling_enabled = true;
    float chunk_frustum_padding = 1.5f;

    bool override_prefetch = false;
    bool predictive_prefetch_enabled = true;
    float prefetch_lookahead_distance = 10.0f;

    bool override_vram_budget = false;
    VRAMBudgetConfig vram_budget_config;

    bool override_lod_config = false;
    LODConfig lod_config;

    bool override_lod_blend = false;
    LODBlendConfig lod_blend_config;

    bool override_streaming_tuning = false;
    uint32_t max_chunk_loads_per_frame = 0;

    bool override_io_source = false;
    String io_source_path;

    bool has_any_override() const {
        return override_chunk_culling || override_prefetch || override_vram_budget ||
                override_lod_config || override_lod_blend || override_streaming_tuning ||
                override_io_source;
    }
};

} // namespace GaussianStreamingTypes

#endif // STREAMING_CONFIG_OVERRIDES_H
