// gs_culling_utils.glsl — Distance culling, hashing, and overlap filtering for tile binning.
//
// Requires: gs_render_params.glsl (for params.distance_cull_config,
// gs_get_overlap_keep_ratio()) to be included before this file.

#ifndef GS_CULLING_UTILS_GLSL_INCLUDED
#define GS_CULLING_UTILS_GLSL_INCLUDED

// Hash a 32-bit value for deterministic culling randomness.
uint gs_hash_u32(uint v) {
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

// Returns true if splat should be culled based on distance.
// p_stable_splat_key must be stable across camera-motion-induced sort order changes.
bool gs_should_distance_cull(uint p_stable_splat_key, float world_distance) {
    if (params.distance_cull_config.z < 0.5) {
        return false; // Disabled
    }

    float start_dist = max(params.distance_cull_config.x, 0.0);
    float max_rate = clamp(params.distance_cull_config.y, 0.0, 1.0);

    if (world_distance < start_dist) {
        return false;
    }

    // Smooth quadratic ramp from 0% to max_rate.
    float denom = max(start_dist, 0.001);
    float t = clamp((world_distance - start_dist) / denom, 0.0, 1.0);
    float cull_probability = t * t * max_rate;

    // Deterministic hash for temporal stability (splat index only, no frame_id).
    // Using frame_id causes different splats to be culled each frame = flickering.
    uint hash = p_stable_splat_key ^ (p_stable_splat_key >> 16u);
    hash *= 0x85ebca6bu;
    hash ^= (hash >> 13u);
    hash *= 0xc2b2ae35u;
    float rand = float(hash & 0xFFFFu) / 65535.0;

    return rand < cull_probability;
}

// Decide whether to keep an overlap record for diagnostics or coverage sampling.
bool gs_keep_overlap_record(uint gaussian_idx, uint instance_id, uint tile_idx) {
    float keep_ratio = gs_get_overlap_keep_ratio();
    if (keep_ratio >= 0.9999) {
        return true;
    }
    if (keep_ratio <= 0.0) {
        return false;
    }

    // Deterministic per (gaussian, instance, tile) so COUNT and EMIT stay in lockstep.
    uint seed = gs_hash_u32(gaussian_idx ^ (instance_id * 0x9e3779b9u));
    seed = gs_hash_u32(seed ^ (tile_idx * 0x85ebca6bu));
    float rand = float(seed & 0x00FFFFFFu) * (1.0 / 16777215.0);
    return rand < keep_ratio;
}

#endif // GS_CULLING_UTILS_GLSL_INCLUDED
