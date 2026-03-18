#include "gaussian_streaming.h"
#include "../renderer/quantization_config.h"
#include "../logger/gs_logger.h"
#include "core/templates/span.h"
#include "servers/rendering/rendering_device.h"

// ==============================================================================
// ChunkQuantizationInfo Implementation (Unity Technique)
// ==============================================================================

void ChunkQuantizationInfo::compute_from_gaussians(const LocalVector<Gaussian> &gaussians,
                                                     uint32_t start_idx, uint32_t count,
                                                     uint32_t pos_bits, uint32_t sc_bits,
                                                     bool quantize_scale) {
    if (count == 0 || start_idx + count > gaussians.size()) {
        clear();
        return;
    }

    position_bits = pos_bits;
    scale_bits = sc_bits;
    scales_quantized = quantize_scale;

    // Initialize bounds with first element
    const Gaussian &first = gaussians[start_idx];
    position_min = first.position;
    position_max = first.position;
    scale_min = first.scale;
    scale_max = first.scale;

    // Compute min/max bounds for all Gaussians in the chunk
    for (uint32_t i = 1; i < count; i++) {
        const Gaussian &g = gaussians[start_idx + i];

        position_min.x = MIN(position_min.x, g.position.x);
        position_min.y = MIN(position_min.y, g.position.y);
        position_min.z = MIN(position_min.z, g.position.z);

        position_max.x = MAX(position_max.x, g.position.x);
        position_max.y = MAX(position_max.y, g.position.y);
        position_max.z = MAX(position_max.z, g.position.z);

        if (quantize_scale) {
            scale_min.x = MIN(scale_min.x, g.scale.x);
            scale_min.y = MIN(scale_min.y, g.scale.y);
            scale_min.z = MIN(scale_min.z, g.scale.z);

            scale_max.x = MAX(scale_max.x, g.scale.x);
            scale_max.y = MAX(scale_max.y, g.scale.y);
            scale_max.z = MAX(scale_max.z, g.scale.z);
        }
    }

    // Compute ranges (add small epsilon to avoid division by zero)
    const float epsilon = 1e-6f;
    position_range = position_max - position_min;
    position_range.x = MAX(position_range.x, epsilon);
    position_range.y = MAX(position_range.y, epsilon);
    position_range.z = MAX(position_range.z, epsilon);

    if (quantize_scale) {
        scale_range = scale_max - scale_min;
        scale_range.x = MAX(scale_range.x, epsilon);
        scale_range.y = MAX(scale_range.y, epsilon);
        scale_range.z = MAX(scale_range.z, epsilon);
    } else {
        scale_range = Vector3();
    }
}

void ChunkQuantizationInfo::quantize_position(const Vector3 &pos, uint32_t &out_x, uint32_t &out_y, uint32_t &out_z) const {
    uint32_t max_val = (1u << position_bits) - 1;

    // Normalize to [0, 1] then scale to [0, max_val]
    float nx = (pos.x - position_min.x) / position_range.x;
    float ny = (pos.y - position_min.y) / position_range.y;
    float nz = (pos.z - position_min.z) / position_range.z;

    // Clamp and quantize
    out_x = uint32_t(CLAMP(nx * float(max_val) + 0.5f, 0.0f, float(max_val)));
    out_y = uint32_t(CLAMP(ny * float(max_val) + 0.5f, 0.0f, float(max_val)));
    out_z = uint32_t(CLAMP(nz * float(max_val) + 0.5f, 0.0f, float(max_val)));
}

void ChunkQuantizationInfo::quantize_scale(const Vector3 &scale, uint32_t &out_x, uint32_t &out_y, uint32_t &out_z) const {
    if (!scales_quantized) {
        out_x = out_y = out_z = 0;
        return;
    }

    uint32_t max_val = (1u << scale_bits) - 1;

    // Normalize to [0, 1] then scale to [0, max_val]
    float nx = (scale.x - scale_min.x) / scale_range.x;
    float ny = (scale.y - scale_min.y) / scale_range.y;
    float nz = (scale.z - scale_min.z) / scale_range.z;

    // Clamp and quantize
    out_x = uint32_t(CLAMP(nx * float(max_val) + 0.5f, 0.0f, float(max_val)));
    out_y = uint32_t(CLAMP(ny * float(max_val) + 0.5f, 0.0f, float(max_val)));
    out_z = uint32_t(CLAMP(nz * float(max_val) + 0.5f, 0.0f, float(max_val)));
}

Vector3 ChunkQuantizationInfo::dequantize_position(uint32_t x, uint32_t y, uint32_t z) const {
    uint32_t max_val = (1u << position_bits) - 1;
    float inv_max = 1.0f / float(max_val);

    return Vector3(
        position_min.x + float(x) * inv_max * position_range.x,
        position_min.y + float(y) * inv_max * position_range.y,
        position_min.z + float(z) * inv_max * position_range.z
    );
}

Vector3 ChunkQuantizationInfo::dequantize_scale(uint32_t x, uint32_t y, uint32_t z) const {
    if (!scales_quantized) {
        return Vector3();
    }

    uint32_t max_val = (1u << scale_bits) - 1;
    float inv_max = 1.0f / float(max_val);

    return Vector3(
        scale_min.x + float(x) * inv_max * scale_range.x,
        scale_min.y + float(y) * inv_max * scale_range.y,
        scale_min.z + float(z) * inv_max * scale_range.z
    );
}

void ChunkQuantizationInfo::clear() {
    position_min = Vector3();
    position_max = Vector3();
    scale_min = Vector3();
    scale_max = Vector3();
    position_range = Vector3();
    scale_range = Vector3();
    position_bits = 16;
    scale_bits = 12;
    scales_quantized = false;
}

float ChunkQuantizationInfo::get_max_position_error() const {
    // Maximum quantization error is half the step size
    uint32_t max_val = (1u << position_bits) - 1;
    float inv_max = 1.0f / float(max_val);
    float max_err = 0.0f;
    max_err = MAX(max_err, position_range.x * inv_max * 0.5f);
    max_err = MAX(max_err, position_range.y * inv_max * 0.5f);
    max_err = MAX(max_err, position_range.z * inv_max * 0.5f);
    return max_err;
}

float ChunkQuantizationInfo::get_max_scale_error() const {
    if (!scales_quantized) {
        return 0.0f;
    }

    uint32_t max_val = (1u << scale_bits) - 1;
    float inv_max = 1.0f / float(max_val);
    float max_err = 0.0f;
    max_err = MAX(max_err, scale_range.x * inv_max * 0.5f);
    max_err = MAX(max_err, scale_range.y * inv_max * 0.5f);
    max_err = MAX(max_err, scale_range.z * inv_max * 0.5f);
    return max_err;
}

// ==============================================================================
// GaussianStreamingSystem Quantization Methods
// ==============================================================================

void GaussianStreamingSystem::_load_quantization_config_from_project_settings() {
    per_chunk_quantization_enabled = g_quantization_config.per_chunk_quantization;
    quantization_position_bits = g_quantization_config.position_bits;
    quantization_scale_bits = g_quantization_config.scale_bits;
    quantization_scales_enabled = g_quantization_config.quantize_scales;
}

void GaussianStreamingSystem::_compute_chunk_quantization(uint32_t chunk_idx) {
    _compute_chunk_quantization(PRIMARY_ASSET_ID, chunk_idx);
}

void GaussianStreamingSystem::_compute_chunk_quantization(uint32_t asset_id, uint32_t chunk_idx) {
    AtlasAssetState *asset = _get_asset_state(asset_id);
    if (!asset || !asset->data.is_valid()) {
        return;
    }

    LocalVector<StreamingChunk> &asset_chunks = _get_asset_chunks(*asset);
    if (chunk_idx >= asset_chunks.size()) {
        return;
    }

    StreamingChunk &chunk = asset_chunks[chunk_idx];
    if (chunk.quantization_computed) {
        return; // Already computed
    }

    if (chunk.source_index_remapped && asset_id == PRIMARY_ASSET_ID) {
        LocalVector<Gaussian> remapped_chunk_gaussians;
        remapped_chunk_gaussians.resize(chunk.count);
        for (uint32_t i = 0; i < chunk.count; i++) {
            uint32_t source_index = 0;
            if (!_resolve_primary_chunk_source_index(chunk, i, source_index)) {
                chunk.quantization.clear();
                chunk.quantization_computed = true;
                WARN_PRINT_ONCE("[Quantization] Spatial chunk source index resolve failed; using cleared bounds.");
                return;
            }
            remapped_chunk_gaussians[i] = asset->data->get_gaussian(source_index);
        }
        chunk.quantization.compute_from_gaussians(
                remapped_chunk_gaussians,
                0,
                chunk.count,
                quantization_position_bits,
                quantization_scale_bits,
                quantization_scales_enabled);
    } else {
        const LocalVector<Gaussian> &gaussians = asset->data->get_gaussian_storage();
        chunk.quantization.compute_from_gaussians(
            gaussians,
            chunk.start_idx,
            chunk.count,
            quantization_position_bits,
            quantization_scale_bits,
            quantization_scales_enabled
        );
    }
    chunk.quantization_computed = true;

    // Log quantization stats for first few chunks
    static uint32_t logged_chunks = 0;
    if (logged_chunks < 3) {
        float max_pos_err = chunk.quantization.get_max_position_error();
        GS_LOG_STREAMING_DEBUG(vformat("[Quantization] Asset %d Chunk %d: pos_range=(%.3f, %.3f, %.3f), max_err=%.6f",
            asset_id,
            chunk_idx,
            chunk.quantization.position_range.x,
            chunk.quantization.position_range.y,
            chunk.quantization.position_range.z,
            max_pos_err));
        logged_chunks++;
    }
}

bool GaussianStreamingSystem::_release_quantization_buffer(RenderingDevice *p_rd, const char *p_context, bool p_allow_deferred_release) {
    if (!quantization_buffer.is_valid()) {
        bool changed_state = false;
        if (quantization_buffer_size != 0) {
            WARN_PRINT(vformat("[Quantization] %s: quantization buffer size (%d) was non-zero with invalid RID; resetting state.",
                    p_context ? p_context : "quantization_buffer_release",
                    quantization_buffer_size));
            changed_state = true;
        }
        quantization_buffer_size = 0;
        quantization_release_deferred_logged = false;
        return changed_state;
    }

    if (!p_rd) {
        if (p_allow_deferred_release) {
            if (!quantization_release_deferred_logged) {
                WARN_PRINT(vformat("[Quantization] %s: deferred quantization buffer free because RenderingDevice is null (rid=%s size=%d).",
                        p_context ? p_context : "quantization_buffer_release",
                        String::num_uint64(static_cast<uint64_t>(quantization_buffer.get_id())),
                        quantization_buffer_size));
                quantization_release_deferred_logged = true;
            }
            return false;
        }
        WARN_PRINT(vformat("[Quantization] %s: RenderingDevice is null while releasing quantization buffer RID %s; dropping handle.",
                p_context ? p_context : "quantization_buffer_release",
                String::num_uint64(static_cast<uint64_t>(quantization_buffer.get_id()))));
    } else {
        p_rd->free(quantization_buffer);
    }

    quantization_buffer = RID();
    quantization_buffer_size = 0;
    quantization_release_deferred_logged = false;
    return true;
}

bool GaussianStreamingSystem::_upload_quantization_buffer(RenderingDevice *p_rd) {
    if (!per_chunk_quantization_enabled) {
        return false;
    }

    const bool rebuild_cache = !quantization_cpu_cache_valid || quantization_dirty;
    if (rebuild_cache) {
        uint32_t total_chunks = 0;
        for (uint32_t asset_id : atlas_asset_order) {
            AtlasAssetState *asset = _get_asset_state(asset_id);
            if (!asset) {
                continue;
            }
            total_chunks += _get_asset_chunks(*asset).size();
        }

        quantization_gpu_data.clear();
        quantization_gpu_data.reserve(total_chunks);

        for (uint32_t asset_id : atlas_asset_order) {
            AtlasAssetState *asset = _get_asset_state(asset_id);
            if (!asset) {
                continue;
            }

            LocalVector<StreamingChunk> &asset_chunks = _get_asset_chunks(*asset);
            asset->quant_base = quantization_gpu_data.size();
            asset->quant_count = asset_chunks.size();

            for (uint32_t i = 0; i < asset_chunks.size(); i++) {
                if (!asset_chunks[i].quantization_computed) {
                    // Quantization is expected to be computed during chunk creation.
                    // Avoid per-frame recompute; fallback to cleared bounds if missing.
                    WARN_PRINT_ONCE("[Quantization] Missing precomputed chunk quantization; using cleared bounds. Reinitialize to avoid this.");
                    asset_chunks[i].quantization.clear();
                    asset_chunks[i].quantization_computed = true;
                }
                quantization_gpu_data.push_back(_create_gpu_quantization_data(
                    asset_chunks[i].quantization,
                    asset_chunks[i].start_idx,
                    asset_chunks[i].count));
            }
        }

        quantization_cpu_cache_valid = true;
        quantization_dirty = false;
    }

    const uint32_t buffer_size = uint32_t(quantization_gpu_data.size()) * sizeof(ChunkQuantizationGPU);
    bool quant_resource_changed = false;
    if (buffer_size == 0) {
        return _release_quantization_buffer(p_rd, "_upload_quantization_buffer(empty)", true);
    }

    if (!p_rd) {
        return false;
    }

    Span<const ChunkQuantizationGPU> quant_span(quantization_gpu_data.ptr(), quantization_gpu_data.size());
    const bool needs_recreate = !quantization_buffer.is_valid() || quantization_buffer_size != buffer_size;
    if (needs_recreate) {
        if (quantization_buffer.is_valid() || quantization_buffer_size != 0) {
            const bool released = _release_quantization_buffer(p_rd, "_upload_quantization_buffer(recreate)", true);
            const bool has_stale_quant_state = quantization_buffer.is_valid() || quantization_buffer_size != 0;
            if (!released && has_stale_quant_state) {
                return false;
            }
        }
        quantization_buffer = p_rd->storage_buffer_create(buffer_size, quant_span.reinterpret<uint8_t>());
        p_rd->set_resource_name(quantization_buffer, "GS_Streaming_QuantizationBuffer");
        quantization_buffer_size = buffer_size;
        quantization_release_deferred_logged = false;
        quant_resource_changed = true;
        GS_LOG_STREAMING_INFO(vformat("[Quantization] Created GPU buffer for %d chunks (%d bytes)",
            quantization_gpu_data.size(), buffer_size));
    } else {
        p_rd->buffer_update(quantization_buffer, 0, buffer_size, quant_span.reinterpret<uint8_t>().ptr());
    }
    return quant_resource_changed;
}

ChunkQuantizationGPU GaussianStreamingSystem::_create_gpu_quantization_data(
    const ChunkQuantizationInfo &info, uint32_t start_idx, uint32_t count) const {

    ChunkQuantizationGPU gpu_data;

    gpu_data.position_min[0] = info.position_min.x;
    gpu_data.position_min[1] = info.position_min.y;
    gpu_data.position_min[2] = info.position_min.z;
    gpu_data.position_bits = info.position_bits;

    gpu_data.position_range[0] = info.position_range.x;
    gpu_data.position_range[1] = info.position_range.y;
    gpu_data.position_range[2] = info.position_range.z;
    gpu_data.scale_bits = info.scales_quantized ? info.scale_bits : 0;

    gpu_data.scale_min[0] = info.scale_min.x;
    gpu_data.scale_min[1] = info.scale_min.y;
    gpu_data.scale_min[2] = info.scale_min.z;
    gpu_data.start_index = start_idx;

    gpu_data.scale_range[0] = info.scale_range.x;
    gpu_data.scale_range[1] = info.scale_range.y;
    gpu_data.scale_range[2] = info.scale_range.z;
    gpu_data.count = count;

    return gpu_data;
}

Dictionary GaussianStreamingSystem::get_quantization_stats() const {
    Dictionary stats;
    stats["enabled"] = per_chunk_quantization_enabled;
    stats["position_bits"] = quantization_position_bits;
    stats["scale_bits"] = quantization_scale_bits;
    stats["scales_quantized"] = quantization_scales_enabled;

    if (per_chunk_quantization_enabled && !atlas_asset_order.is_empty()) {
        // Calculate average and max errors across all chunks
        float total_pos_error = 0.0f;
        float max_pos_error = 0.0f;
        uint32_t computed_count = 0;

        for (uint32_t asset_id : atlas_asset_order) {
            const AtlasAssetState *asset = _get_asset_state(asset_id);
            if (!asset) {
                continue;
            }
            const LocalVector<StreamingChunk> &asset_chunks = _get_asset_chunks(*asset);
            for (const StreamingChunk &chunk : asset_chunks) {
                if (chunk.quantization_computed) {
                    float err = chunk.quantization.get_max_position_error();
                    total_pos_error += err;
                    max_pos_error = MAX(max_pos_error, err);
                    computed_count++;
                }
            }
        }

        if (computed_count > 0) {
            stats["avg_position_error"] = total_pos_error / float(computed_count);
            stats["max_position_error"] = max_pos_error;
            stats["chunks_computed"] = computed_count;
        }

        // Calculate compression ratio
        float pos_compression = g_quantization_config.get_position_compression_ratio();
        float total_compression = g_quantization_config.get_total_compression_ratio();
        stats["position_compression_ratio"] = pos_compression;
        stats["total_compression_ratio"] = total_compression;
    }

    return stats;
}
