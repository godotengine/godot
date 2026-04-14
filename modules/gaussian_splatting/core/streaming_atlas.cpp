/**************************************************************************/
/* streaming_atlas.cpp                                                    */
/*                                                                        */
/* GaussianAtlasAllocator method implementations and                      */
/* GaussianStreamingSystem residency helpers.                             */
/**************************************************************************/

#include "streaming_atlas.h"

#include "gaussian_streaming.h"
#include "../logger/gs_debug_trace.h"
#include <cstdint>

void GaussianAtlasAllocator::reset(uint32_t p_slot_count) {
	capacity = p_slot_count;
	free_slots.clear();
	slot_map.clear();
	if (capacity == 0) {
		return;
	}
	free_slots.reserve(capacity);
	for (int32_t i = static_cast<int32_t>(capacity) - 1; i >= 0; i--) {
		free_slots.push_back(static_cast<uint32_t>(i));
	}
}

bool GaussianAtlasAllocator::allocate_slot(uint64_t p_chunk_key, uint32_t &r_slot) {
	if (const uint32_t *slot = slot_map.getptr(p_chunk_key)) {
		r_slot = *slot;
		return true;
	}
	if (free_slots.is_empty()) {
		return false;
	}
	r_slot = free_slots[free_slots.size() - 1];
	free_slots.resize(free_slots.size() - 1);
	slot_map[p_chunk_key] = r_slot;
	return true;
}

void GaussianAtlasAllocator::release_slot(uint64_t p_chunk_key) {
	const uint32_t *slot = slot_map.getptr(p_chunk_key);
	if (!slot) {
		return;
	}
	free_slots.push_back(*slot);
	slot_map.erase(p_chunk_key);
}

bool GaussianAtlasAllocator::get_slot(uint64_t p_chunk_key, uint32_t &r_slot) const {
	if (const uint32_t *slot = slot_map.getptr(p_chunk_key)) {
		r_slot = *slot;
		return true;
	}
	return false;
}

void GaussianAtlasAllocator::clear() {
	capacity = 0;
	free_slots.clear();
	slot_map.clear();
}

void GaussianStreamingSystem::_apply_requested_residency(bool can_async_pack) {
	if (!asset_registry.request_pending) {
		return;
	}
	const bool trace_enabled = GaussianSplatting::debug_trace_is_enabled();
	if (trace_enabled) {
		GaussianSplatting::debug_trace_record_event("streaming",
				vformat("ApplyResidency START: atlas_asset_order=%d", asset_registry.atlas_asset_order.size()),
				false);
	}

    bool has_deferred_requested_chunks = false;
    for (uint32_t asset_id : asset_registry.atlas_asset_order) {
        AtlasAssetState *asset = _get_asset_state(asset_id);
        if (!asset || !asset->data.is_valid()) {
            if (trace_enabled) {
                GaussianSplatting::debug_trace_record_event("streaming",
						vformat("ApplyResidency SKIP INVALID asset_id=%d", asset_id),
						true);
			}
			continue;
		}

        LocalVector<StreamingChunk> &asset_chunks = _get_asset_chunks(*asset);
        if (trace_enabled) {
            GaussianSplatting::debug_trace_record_event("streaming",
					vformat("ApplyResidency asset_id=%d chunks=%d requested_chunks=%d",
					asset_id, asset_chunks.size(), asset->requested_chunks.size()),
				false);
        }
        if (asset_id != PRIMARY_ASSET_ID) {
            _evict_unrequested_chunks(asset_id, *asset, asset_chunks);
        }
        const bool deferred_chunks = _load_requested_chunks(asset_id, *asset, asset_chunks, trace_enabled, can_async_pack);
        has_deferred_requested_chunks = has_deferred_requested_chunks || deferred_chunks;
    }

	if (trace_enabled) {
		GaussianSplatting::debug_trace_record_event("streaming",
				vformat("ApplyResidency END: loaded_chunks_count=%d vram=%s", budget.loaded_chunks_count, String::num_uint64(_get_total_vram_usage_bytes())),
				false);
	}
	asset_registry.request_pending = has_deferred_requested_chunks;
	asset_registry.request_collection_active = false;
}

void GaussianStreamingSystem::_evict_unrequested_chunks(uint32_t asset_id, AtlasAssetState &asset,
		LocalVector<StreamingChunk> &asset_chunks) {
	for (uint32_t i = 0; i < asset_chunks.size(); i++) {
		StreamingChunk &chunk = asset_chunks[i];
		const bool requested = _is_requested_chunk_in_current_generation(asset, i);
		if (requested) {
			continue;
		}

		if (chunk.upload_pending) {
			upload_pipeline.cancel_chunk_jobs(*this, asset_id, i, chunk.buffer_slot);
		}
		if (chunk.is_loaded) {
			_unload_chunk(asset_id, i);
			eviction_controller.record_total_eviction();
		}
	}
}

bool GaussianStreamingSystem::_load_requested_chunks(uint32_t asset_id, AtlasAssetState &asset,
		LocalVector<StreamingChunk> &asset_chunks, bool trace_enabled, bool can_async_pack) {
	int chunks_processed = 0;
	int chunks_already_loaded = 0;
	int chunks_queued = 0;
	bool has_deferred_chunks = false;

	for (uint32_t chunk_id : asset.requested_chunks) {
		if (chunk_id >= asset_chunks.size()) {
			continue;
		}
		chunks_processed++;
        const GaussianStreamingSystem::RequestedChunkState *request_state =
                asset.requested_chunk_state.getptr(chunk_id);
        if (request_state &&
                request_state->stamp == asset_registry.request_generation &&
                request_state->request_result == GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_FAILED) {
            continue;
        }
        StreamingChunk &chunk = asset_chunks[chunk_id];
        if (chunk.is_loaded || chunk.upload_pending) {
            chunks_already_loaded++;
            if (chunk.is_loaded) {
                chunk.explicit_request_generation = 0;
                eviction_controller.touch_chunk_use(chunk.last_used_frame);
                _update_requested_chunk_state(asset, chunk_id,
                        GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_SATISFIED,
                        GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_SATISFIED);
            } else {
                chunk.explicit_request_generation = request_state ? request_state->stamp : asset_registry.request_generation;
                _update_requested_chunk_state(asset, chunk_id,
                        GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_QUEUED,
                        GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_QUEUED);
            }
            continue;
        }
        const bool queued = _enqueue_chunk_load_request(asset_id, chunk_id, can_async_pack, !can_async_pack);

        if (queued) {
            chunks_queued++;
            chunk.explicit_request_generation = request_state ? request_state->stamp : asset_registry.request_generation;
            _update_requested_chunk_state(asset, chunk_id,
                    GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_QUEUED,
                    GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_QUEUED);
            if (!can_async_pack) {
                has_deferred_chunks = true;
            }
        }
        if (!queued && !chunk.is_loaded && !chunk.upload_pending) {
            if (trace_enabled) {
                GaussianSplatting::debug_trace_record_event("streaming",
                        vformat("ApplyResidency DEFER chunk_id=%d", chunk_id),
                        true);
            }
            _update_requested_chunk_state(asset, chunk_id,
                    GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_DEFERRED,
                    GaussianStreamingTypes::RESIDENCY_REQUEST_STATE_DEFERRED,
                    ERR_BUSY);
            has_deferred_chunks = true;
        }
        if (!can_async_pack && !chunk.is_loaded && !chunk.upload_pending) {
            has_deferred_chunks = true;
        }
	}

	if (trace_enabled) {
		GaussianSplatting::debug_trace_record_event("streaming",
				vformat("ApplyResidency asset_id=%d processed=%d already_loaded=%d queued=%d",
						asset_id, chunks_processed, chunks_already_loaded, chunks_queued),
				false);
	}
	return has_deferred_chunks;
}
