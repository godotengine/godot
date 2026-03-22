#include "streaming_eviction_controller.h"

#include "gaussian_streaming.h"
#include "gs_project_settings.h"
#include "core/config/project_settings.h"
#include "../logger/gs_logger.h"

#include <algorithm>

void StreamingEvictionController::load_streaming_tuning_config_from_project_settings() {
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }

    eviction_hysteresis_frames = gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/eviction_hysteresis_frames",
            eviction_hysteresis_frames);
    max_evictions_per_frame = gs::settings::get_uint(ps,
            "rendering/gaussian_splatting/streaming/max_evictions_per_frame",
            max_evictions_per_frame);
}

void StreamingEvictionController::reset_per_frame_counters() {
    chunks_evicted_this_frame = 0;
    visible_chunks_evicted_this_frame = 0;
}

void StreamingEvictionController::touch_chunk_use(uint64_t &r_last_used_frame) {
    r_last_used_frame = ++chunk_load_counter;
}

void StreamingEvictionController::record_eviction_result(EvictionResult p_result) {
    if (p_result == EvictionResult::EvictedNonVisible || p_result == EvictionResult::EvictedVisible) {
        chunks_evicted_this_frame++;
        if (p_result == EvictionResult::EvictedVisible) {
            visible_chunks_evicted_this_frame++;
        }
    }
}

void StreamingEvictionController::record_total_eviction() {
    chunks_evicted_this_frame++;
}

StreamingEvictionController::EvictionResult StreamingEvictionController::evict_least_recently_used(
        GaussianStreamingSystem &system, bool p_allow_visible_eviction) {
    if (cached_eviction_frame != system.total_frame_count) {
        cached_visible_chunks.clear();
        cached_nonvisible_chunks.clear();
        for (uint32_t i = 0; i < system.chunks.size(); i++) {
            const GaussianStreamingSystem::StreamingChunk &chunk = system.chunks[i];
            if (!chunk.is_loaded) {
                continue;
            }
            if (eviction_hysteresis_frames > 0 &&
                    system.total_frame_count - chunk.last_loaded_frame < eviction_hysteresis_frames) {
                continue;
            }
            if (chunk.is_visible) {
                cached_visible_chunks.push_back(i);
            } else {
                cached_nonvisible_chunks.push_back(i);
            }
        }
        cached_eviction_frame = system.total_frame_count;
    }

    uint32_t best_nonvis_idx = UINT32_MAX;
    uint64_t best_nonvis_frame = UINT64_MAX;
    float best_nonvis_distance = -1.0f;

    uint32_t best_vis_idx = UINT32_MAX;
    uint64_t best_vis_frame = UINT64_MAX;
    float best_vis_distance = -1.0f;

    for (uint32_t i : cached_nonvisible_chunks) {
        const GaussianStreamingSystem::StreamingChunk &chunk = system.chunks[i];
        if (!chunk.is_loaded) {
            continue;
        }
        const uint64_t used_frame = chunk.last_used_frame;
        const float dist = chunk.distance;
        if (used_frame < best_nonvis_frame ||
                (used_frame == best_nonvis_frame && dist > best_nonvis_distance)) {
            best_nonvis_frame = used_frame;
            best_nonvis_distance = dist;
            best_nonvis_idx = i;
        }
    }

    for (uint32_t i : cached_visible_chunks) {
        const GaussianStreamingSystem::StreamingChunk &chunk = system.chunks[i];
        if (!chunk.is_loaded) {
            continue;
        }
        const uint64_t used_frame = chunk.last_used_frame;
        const float dist = chunk.distance;
        if (used_frame < best_vis_frame ||
                (used_frame == best_vis_frame && dist > best_vis_distance)) {
            best_vis_frame = used_frame;
            best_vis_distance = dist;
            best_vis_idx = i;
        }
    }

    if (best_nonvis_idx != UINT32_MAX) {
        system._unload_chunk(best_nonvis_idx);
        return EvictionResult::EvictedNonVisible;
    }

    if (best_vis_idx != UINT32_MAX) {
        if (!p_allow_visible_eviction) {
            if (system.total_frame_count - last_stabilize_log_frame >= 300) {
                GS_LOG_STREAMING_DEBUG(vformat("[STREAM-STABLE] All %d loaded chunks are visible - stabilizing (not evicting)",
                        system.budget.loaded_chunks_count));
                last_stabilize_log_frame = system.total_frame_count;
            }
            return EvictionResult::SkippedAllVisible;
        }

        system._unload_chunk(best_vis_idx);
        return EvictionResult::EvictedVisible;
    }

    return EvictionResult::NoEviction;
}

bool StreamingEvictionController::evict_non_primary_lru(GaussianStreamingSystem &system) {
    if (max_evictions_per_frame > 0 && chunks_evicted_this_frame >= max_evictions_per_frame) {
        return false;
    }

    if (cached_non_primary_lru_frame != system.total_frame_count) {
        cached_non_primary_lru_candidates.clear();
        for (uint32_t asset_id : system.asset_registry.atlas_asset_order) {
            if (asset_id == GaussianStreamingSystem::PRIMARY_ASSET_ID) {
                continue;
            }
            GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(asset_id);
            if (!asset) {
                continue;
            }
            LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
            for (uint32_t chunk_id = 0; chunk_id < asset_chunks.size(); chunk_id++) {
                const GaussianStreamingSystem::StreamingChunk &chunk = asset_chunks[chunk_id];
                if (!chunk.is_loaded) {
                    continue;
                }
                NonPrimaryEvictionCandidate candidate;
                candidate.asset_id = asset_id;
                candidate.chunk_id = chunk_id;
                candidate.last_used_frame = chunk.last_used_frame;
                candidate.distance = chunk.distance;
                cached_non_primary_lru_candidates.push_back(candidate);
            }
        }
        if (!cached_non_primary_lru_candidates.is_empty()) {
            NonPrimaryEvictionCandidate *candidate_ptr = cached_non_primary_lru_candidates.ptr();
            std::sort(candidate_ptr, candidate_ptr + cached_non_primary_lru_candidates.size(),
                    [](const NonPrimaryEvictionCandidate &a, const NonPrimaryEvictionCandidate &b) {
                        if (a.last_used_frame != b.last_used_frame) {
                            return a.last_used_frame < b.last_used_frame;
                        }
                        if (a.distance != b.distance) {
                            return a.distance > b.distance;
                        }
                        if (a.asset_id != b.asset_id) {
                            return a.asset_id < b.asset_id;
                        }
                        return a.chunk_id < b.chunk_id;
                    });
        }
        cached_non_primary_lru_cursor = 0;
        cached_non_primary_lru_frame = system.total_frame_count;
        system.scheduler.last_non_primary_scan_count = cached_non_primary_lru_candidates.size();
    }

    while (cached_non_primary_lru_cursor < cached_non_primary_lru_candidates.size()) {
        const NonPrimaryEvictionCandidate &candidate = cached_non_primary_lru_candidates[cached_non_primary_lru_cursor++];
        GaussianStreamingSystem::AtlasAssetState *asset = system._get_asset_state(candidate.asset_id);
        if (!asset) {
            continue;
        }
        LocalVector<GaussianStreamingSystem::StreamingChunk> &asset_chunks = system._get_asset_chunks(*asset);
        if (candidate.chunk_id >= asset_chunks.size()) {
            continue;
        }
        if (!asset_chunks[candidate.chunk_id].is_loaded) {
            continue;
        }

        const bool was_visible = asset_chunks[candidate.chunk_id].is_visible;
        system._unload_chunk(candidate.asset_id, candidate.chunk_id);
        record_eviction_result(was_visible ? EvictionResult::EvictedVisible : EvictionResult::EvictedNonVisible);
        return true;
    }

    return false;
}

bool StreamingEvictionController::ensure_atlas_slot_available(GaussianStreamingSystem &system, uint32_t requesting_asset_id) {
    (void)requesting_asset_id;
    if (system.atlas_allocator.has_free_slots()) {
        return true;
    }

    if (evict_non_primary_lru(system)) {
        return system.atlas_allocator.has_free_slots();
    }

    return false;
}
