#include "resident_instance_contract_publisher.h"

#include "gaussian_gpu_layout.h"
#include "gaussian_splat_renderer.h"
#include "gpu_sorting_config.h"
#include "instance_pipeline_contract.h"
#include "quantization_config.h"
#include "../core/gaussian_splat_scene_director.h"
#include "../interfaces/gpu_sorting_pipeline.h"
#include "servers/rendering/rendering_device.h"

#include <algorithm>
#include <cstring>
#include <cstdint>

namespace {

using GaussianSplatting::InstancePipelineContract::InvariantViolationReason;
constexpr uint32_t kPrimaryResidentAssetId = 0u;

struct ResidentChunkDescriptor {
	AABB bounds;
	Vector<uint32_t> source_indices;
	uint32_t start_idx = 0;
	uint32_t count = 0;
	bool source_index_remapped = false;
};

struct ResidentAssetDescriptor {
	uint32_t submission_asset_id = 0;
	uint32_t dense_asset_id = 0;
	Ref<GaussianData> data;
	Vector<GaussianSplatRenderer::StaticChunk> static_chunks;
};

static uint64_t _mix_generation(uint64_t p_accum, uint64_t p_value) {
	uint64_t x = p_accum + 0x9e3779b97f4a7c15ULL;
	x ^= p_value + (x << 6) + (x >> 2);
	return x;
}

static AABB _compute_contiguous_chunk_bounds(const LocalVector<Gaussian> &p_gaussians, uint32_t p_start, uint32_t p_count) {
	if (p_count == 0 || p_start >= p_gaussians.size()) {
		return AABB();
	}

	Vector3 min_pos = p_gaussians[p_start].position;
	Vector3 max_pos = p_gaussians[p_start].position;
	const uint32_t end = MIN<uint32_t>(p_start + p_count, p_gaussians.size());
	for (uint32_t i = p_start + 1; i < end; i++) {
		const Vector3 &pos = p_gaussians[i].position;
		min_pos.x = MIN(min_pos.x, pos.x);
		min_pos.y = MIN(min_pos.y, pos.y);
		min_pos.z = MIN(min_pos.z, pos.z);
		max_pos.x = MAX(max_pos.x, pos.x);
		max_pos.y = MAX(max_pos.y, pos.y);
		max_pos.z = MAX(max_pos.z, pos.z);
	}
	Vector3 size = max_pos - min_pos;
	if (size == Vector3()) {
		size = Vector3(0.001f, 0.001f, 0.001f);
	}
	return AABB(min_pos, size);
}

template <typename T>
static bool _upload_typed_storage_buffer(GaussianSplatRenderer *p_renderer, RenderingDevice *p_rd, RID &r_buffer,
		uint32_t &r_buffer_size, const char *p_label, const Vector<T> &p_data) {
	if (p_data.is_empty()) {
		p_renderer->free_owned_resource(p_rd, r_buffer);
		r_buffer_size = 0;
		return false;
	}

	const uint64_t required_size_u64 = uint64_t(p_data.size()) * sizeof(T);
	if (required_size_u64 > uint64_t(UINT32_MAX)) {
		return false;
	}
	const uint32_t required_size = uint32_t(required_size_u64);
	Vector<uint8_t> upload_bytes;
	upload_bytes.resize(required_size);
	memcpy(upload_bytes.ptrw(), p_data.ptr(), required_size);

	if (!r_buffer.is_valid() || r_buffer_size != required_size) {
		p_renderer->free_owned_resource(p_rd, r_buffer);
		r_buffer = p_rd->storage_buffer_create(required_size, upload_bytes);
		if (!r_buffer.is_valid()) {
			r_buffer_size = 0;
			return false;
		}
		p_rd->set_resource_name(r_buffer, p_label);
		p_renderer->track_resource_owner(r_buffer, p_rd);
		r_buffer_size = required_size;
		return true;
	}

	p_rd->buffer_update(r_buffer, 0, required_size, upload_bytes.ptr());
	return true;
}

static void _append_chunk_descriptors_for_asset(const ResidentAssetDescriptor &p_asset,
		LocalVector<ResidentChunkDescriptor> &r_chunks) {
	const uint32_t total_count = p_asset.data.is_valid() ? uint32_t(MAX(0, p_asset.data->get_count())) : 0u;
	if (total_count == 0) {
		return;
	}

	if (!p_asset.static_chunks.is_empty()) {
		for (int chunk_idx = 0; chunk_idx < p_asset.static_chunks.size(); chunk_idx++) {
			const GaussianSplatRenderer::StaticChunk &chunk = p_asset.static_chunks[chunk_idx];
			if (chunk.indices.is_empty()) {
				continue;
			}
			uint32_t offset = 0;
			while (offset < uint32_t(chunk.indices.size())) {
				const uint32_t split_count = MIN<uint32_t>(GaussianStreamingSystem::CHUNK_SIZE,
						uint32_t(chunk.indices.size()) - offset);
				ResidentChunkDescriptor descriptor;
				descriptor.bounds = chunk.bounds;
				descriptor.count = split_count;
				descriptor.source_index_remapped = true;
				descriptor.source_indices.resize(split_count);
				for (uint32_t local_idx = 0; local_idx < split_count; local_idx++) {
					descriptor.source_indices.write[local_idx] = chunk.indices[offset + local_idx];
				}
				r_chunks.push_back(descriptor);
				offset += split_count;
			}
		}
		return;
	}

	const LocalVector<Gaussian> &gaussians = p_asset.data->get_gaussian_storage();
	for (uint32_t start = 0; start < total_count; start += GaussianStreamingSystem::CHUNK_SIZE) {
		const uint32_t count = MIN<uint32_t>(GaussianStreamingSystem::CHUNK_SIZE, total_count - start);
		ResidentChunkDescriptor descriptor;
		descriptor.start_idx = start;
		descriptor.count = count;
		descriptor.bounds = _compute_contiguous_chunk_bounds(gaussians, start, count);
		r_chunks.push_back(descriptor);
	}
}

} // namespace

namespace ResidentInstanceContractPublisher {

bool publish(GaussianSplatRenderer *p_renderer, bool p_allow_primary_fallback_instance, String *r_reason) {
	ERR_FAIL_NULL_V(p_renderer, false);

	if (r_reason) {
		*r_reason = String();
	}

	if (!p_renderer->ensure_rendering_device("resident_instance_contract_publish")) {
		if (r_reason) {
			*r_reason = "rendering_device_unavailable";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	if (g_quantization_config.per_chunk_quantization) {
		if (r_reason) {
			*r_reason = "resident_quantization_unsupported";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	RenderingDevice *rd = p_renderer->get_device_state().rd;
	if (rd == nullptr) {
		if (r_reason) {
			*r_reason = "rendering_device_unavailable";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	const Ref<GaussianData> primary_data = p_renderer->get_scene_state().gaussian_data;
	const bool has_primary_data = primary_data.is_valid() && primary_data->get_count() > 0;

	LocalVector<ResidentAssetDescriptor> assets;
	assets.reserve(8);

	ResidentAssetDescriptor primary_asset;
	primary_asset.submission_asset_id = kPrimaryResidentAssetId;
	primary_asset.dense_asset_id = 0;
	primary_asset.data = primary_data;
	if (has_primary_data) {
		primary_asset.static_chunks = p_renderer->get_static_chunks();
	}
	assets.push_back(primary_asset);

	GaussianSplatSceneDirector *director = GaussianSplatSceneDirector::get_singleton();
	LocalVector<InstanceAssetRegistration> instance_assets;
	if (director != nullptr) {
		director->collect_instance_assets_for_renderer(p_renderer, instance_assets,
				p_renderer->is_shadow_instance_filter_enabled());
	}
	std::sort(instance_assets.ptr(), instance_assets.ptr() + instance_assets.size(),
			[](const InstanceAssetRegistration &a, const InstanceAssetRegistration &b) {
				return a.asset_id < b.asset_id;
			});

	uint32_t next_dense_id = 1;
	for (const InstanceAssetRegistration &entry : instance_assets) {
		if (entry.asset_id == 0 || entry.data.is_null()) {
			continue;
		}
		ResidentAssetDescriptor asset;
		asset.submission_asset_id = entry.asset_id;
		asset.dense_asset_id = next_dense_id++;
		asset.data = entry.data;
		assets.push_back(asset);
	}

	LocalVector<InstanceDataGPU> instances;
	if (director != nullptr) {
		director->build_instance_buffer_for_renderer(p_renderer, instances,
				p_renderer->is_shadow_instance_filter_enabled());
	}
	if (instances.is_empty() && p_allow_primary_fallback_instance && has_primary_data) {
		InstanceDataGPU fallback_instance = {};
		fallback_instance.rotation[3] = 1.0f;
		fallback_instance.inv_rotation[3] = 1.0f;
		fallback_instance.translation_scale[3] = 1.0f;
		fallback_instance.params[0] = 1.0f;
		fallback_instance.params[1] = 1.0f;
		fallback_instance.params[2] = 1.0f;
		fallback_instance.wind_params[3] = 1.0f;
		fallback_instance.ids[0] = kPrimaryResidentAssetId;
		fallback_instance.ids[1] = GS_INSTANCE_FLAG_ROTATION_IDENTITY |
				GS_INSTANCE_FLAG_SCALE_IDENTITY |
				GS_INSTANCE_FLAG_TRANSLATION_ZERO;
		instances.push_back(fallback_instance);
	}

	if (instances.is_empty()) {
		if (r_reason) {
			*r_reason = "resident_no_instances";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	Vector<AssetMetaGPU> asset_meta_cpu;
	asset_meta_cpu.resize(next_dense_id);
	Vector<AssetChunkIndexGPU> asset_chunk_index_cpu;
	Vector<ChunkMetaGPU> chunk_meta_cpu;
	Vector<PackedGaussian> atlas_gaussian_cpu;

	uint32_t max_chunk_count_per_asset = 0;
	uint32_t max_chunk_splats = 0;
	uint64_t source_generation = 0x6a09e667f3bcc909ULL;
	source_generation = _mix_generation(source_generation, has_primary_data ? uint64_t(primary_data->get_instance_id()) : 0ULL);
	source_generation = _mix_generation(source_generation, uint64_t(p_renderer->get_static_chunks().size()));
	if (director != nullptr) {
		source_generation = _mix_generation(source_generation, director->get_instance_generation_for_renderer(p_renderer));
	}

	for (uint32_t asset_index = 0; asset_index < assets.size(); asset_index++) {
		const ResidentAssetDescriptor &asset = assets[asset_index];
		asset_meta_cpu.write[asset.dense_asset_id] = AssetMetaGPU();
		source_generation = _mix_generation(source_generation, asset.submission_asset_id);
		source_generation = _mix_generation(source_generation, asset.data.is_valid() ? uint64_t(asset.data->get_instance_id()) : 0ULL);

		if (asset.data.is_null() || asset.data->get_count() <= 0) {
			continue;
		}

		LocalVector<ResidentChunkDescriptor> chunk_descriptors;
		_append_chunk_descriptors_for_asset(asset, chunk_descriptors);
		if (chunk_descriptors.is_empty()) {
			continue;
		}

		AssetMetaGPU asset_meta = {};
		asset_meta.lod_count = 1;
		asset_meta.sh_degree = asset.data->get_sh_degree();
		asset_meta.flags = asset.data->get_2d_mode() ? GS_INSTANCE_FLAG_IS_2D : 0u;
		const AABB asset_bounds = asset.data->get_aabb();
		const Vector3 asset_center = asset_bounds.get_center();
		const Vector3 asset_half = asset_bounds.size * 0.5f;
		asset_meta.bounds_center_local[0] = asset_center.x;
		asset_meta.bounds_center_local[1] = asset_center.y;
		asset_meta.bounds_center_local[2] = asset_center.z;
		asset_meta.bounds_radius_local = asset_half.length();
		asset_meta.chunk_index_base = asset_chunk_index_cpu.size();
		asset_meta.chunk_index_count = chunk_descriptors.size();
		asset_meta.lod_ranges[0].base = asset_meta.chunk_index_base;
		asset_meta.lod_ranges[0].count = asset_meta.chunk_index_count;

		max_chunk_count_per_asset = MAX<uint32_t>(max_chunk_count_per_asset, chunk_descriptors.size());

		for (uint32_t chunk_index = 0; chunk_index < chunk_descriptors.size(); chunk_index++) {
			const ResidentChunkDescriptor &descriptor = chunk_descriptors[chunk_index];
			LocalVector<Gaussian> gaussian_snapshot;
			LocalVector<Vector3> sh_high_order_snapshot;
			uint32_t sh_first_order = 0;
			uint32_t sh_high_order = 0;
			bool captured = false;
			if (descriptor.source_index_remapped) {
				captured = asset.data->capture_indexed_chunk_snapshot(descriptor.source_indices.ptr(), descriptor.count,
						gaussian_snapshot, sh_high_order_snapshot, sh_first_order, sh_high_order);
			} else {
				captured = asset.data->capture_chunk_snapshot(descriptor.start_idx, descriptor.count,
						gaussian_snapshot, sh_high_order_snapshot, sh_first_order, sh_high_order);
			}
			if (!captured || gaussian_snapshot.size() != descriptor.count) {
				if (r_reason) {
					*r_reason = "resident_chunk_snapshot_failed";
				}
				p_renderer->clear_instance_pipeline_buffers();
				return false;
			}

			SHCompressionMetrics sh_metrics;
			Vector<PackedGaussian> packed_chunk;
			const Vector3 *sh_coeffs = sh_high_order_snapshot.is_empty() ? nullptr : sh_high_order_snapshot.ptr();
			pack_gaussians_range(gaussian_snapshot, 0, descriptor.count, packed_chunk, sh_metrics, sh_coeffs,
					sh_first_order, sh_high_order);

			const uint32_t atlas_base = atlas_gaussian_cpu.size();
			for (int i = 0; i < packed_chunk.size(); i++) {
				atlas_gaussian_cpu.push_back(packed_chunk[i]);
			}

			ChunkMetaGPU chunk_meta = {};
			chunk_meta.atlas_base = atlas_base;
			chunk_meta.splat_count = descriptor.count;
			const Vector3 chunk_center = descriptor.bounds.get_center();
			const Vector3 chunk_half = descriptor.bounds.size * 0.5f;
			chunk_meta.bounds_center_local[0] = chunk_center.x;
			chunk_meta.bounds_center_local[1] = chunk_center.y;
			chunk_meta.bounds_center_local[2] = chunk_center.z;
			chunk_meta.bounds_radius_local = chunk_half.length();
			chunk_meta.asset_id = asset.dense_asset_id;
			chunk_meta.lod_level = 0;
			chunk_meta.flags = asset_meta.flags;
			chunk_meta.sh_limit = CLAMP(asset.data->get_sh_degree(), 0u, 3u);
			chunk_meta_cpu.push_back(chunk_meta);

			AssetChunkIndexGPU chunk_index_gpu = {};
			chunk_index_gpu.chunk_id = chunk_meta_cpu.size() - 1;
			asset_chunk_index_cpu.push_back(chunk_index_gpu);
			max_chunk_splats = MAX(max_chunk_splats, descriptor.count);
		}

		asset_meta_cpu.write[asset.dense_asset_id] = asset_meta;
	}

	if (atlas_gaussian_cpu.is_empty()) {
		if (r_reason) {
			*r_reason = "resident_atlas_empty";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	GaussianSplatRenderer::ResourceState &resource_state = p_renderer->get_resource_state();
	if (!_upload_typed_storage_buffer(p_renderer, rd, resource_state.resident_atlas_gaussian_buffer,
				resource_state.resident_atlas_gaussian_buffer_size, "GS_ResidentAtlasGaussians", atlas_gaussian_cpu) ||
			!_upload_typed_storage_buffer(p_renderer, rd, resource_state.resident_asset_meta_buffer,
				resource_state.resident_asset_meta_buffer_size, "GS_ResidentAssetMeta", asset_meta_cpu) ||
			!_upload_typed_storage_buffer(p_renderer, rd, resource_state.resident_chunk_meta_buffer,
				resource_state.resident_chunk_meta_buffer_size, "GS_ResidentChunkMeta", chunk_meta_cpu) ||
			!_upload_typed_storage_buffer(p_renderer, rd, resource_state.resident_asset_chunk_index_buffer,
				resource_state.resident_asset_chunk_index_buffer_size, "GS_ResidentAssetChunkIndex", asset_chunk_index_cpu)) {
		if (r_reason) {
			*r_reason = "resident_dataset_upload_failed";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	const uint32_t instance_count = instances.size();
	GaussianRenderPipeline::InstancePipelineBuffers buffers;
	buffers.atlas_gaussian_buffer = resource_state.resident_atlas_gaussian_buffer;
	buffers.atlas_gaussian_count = atlas_gaussian_cpu.size();
	buffers.asset_meta_buffer = resource_state.resident_asset_meta_buffer;
	buffers.chunk_meta_buffer = resource_state.resident_chunk_meta_buffer;
	buffers.asset_chunk_index_buffer = resource_state.resident_asset_chunk_index_buffer;
	buffers.quantization_required = false;
	buffers.quantization_buffer = RID();
	buffers.dispatch_chunk_count = MAX<uint32_t>(1u, max_chunk_count_per_asset);
	buffers.max_chunk_splats = MAX<uint32_t>(1u, max_chunk_splats);

	uint64_t max_visible_splats_u64 = atlas_gaussian_cpu.size();
	const int configured_max_splats = p_renderer->get_performance_settings().max_splats;
	if (configured_max_splats > 0) {
		max_visible_splats_u64 = MIN<uint64_t>(max_visible_splats_u64, uint64_t(configured_max_splats));
	}
	const uint64_t instance_requirement = uint64_t(instance_count) * uint64_t(buffers.dispatch_chunk_count) *
			uint64_t(buffers.max_chunk_splats);
	max_visible_splats_u64 = MAX<uint64_t>(max_visible_splats_u64, instance_requirement);
	const uint64_t sort_cap = g_gpu_sorting_config.max_sort_elements > 0
			? uint64_t(g_gpu_sorting_config.max_sort_elements)
			: uint64_t(UINT32_MAX);
	buffers.max_visible_splats = uint32_t(MIN<uint64_t>(max_visible_splats_u64, sort_cap));
	uint64_t max_visible_chunks_u64 = uint64_t(instance_count) * uint64_t(buffers.dispatch_chunk_count);
	buffers.max_visible_chunks = uint32_t(MIN<uint64_t>(max_visible_chunks_u64,
			uint64_t(MAX<uint32_t>(1u, buffers.max_visible_splats))));

	auto ensure_owner = [&](RID &r_buffer, uint32_t *r_capacity) {
		if (r_buffer.is_valid() && p_renderer->get_resource_owner(r_buffer, rd) != rd) {
			p_renderer->free_owned_resource(rd, r_buffer);
			if (r_capacity != nullptr) {
				*r_capacity = 0;
			}
		}
	};
	ensure_owner(resource_state.instance_visible_chunk_buffer, &resource_state.instance_visible_chunk_capacity);
	ensure_owner(resource_state.instance_splat_ref_buffer, &resource_state.instance_splat_ref_capacity);
	ensure_owner(resource_state.instance_counter_buffer, nullptr);
	ensure_owner(resource_state.instance_chunk_dispatch_buffer, nullptr);
	ensure_owner(resource_state.instance_indirect_count_buffer, nullptr);
	ensure_owner(resource_state.instance_count_buffer, nullptr);

	if (!resource_state.instance_visible_chunk_buffer.is_valid() ||
			resource_state.instance_visible_chunk_capacity < buffers.max_visible_chunks) {
		p_renderer->free_owned_resource(rd, resource_state.instance_visible_chunk_buffer);
		const uint32_t buffer_size = MAX<uint32_t>(1u, buffers.max_visible_chunks) * sizeof(VisibleChunkRefGPU);
		resource_state.instance_visible_chunk_buffer = rd->storage_buffer_create(buffer_size);
		if (!resource_state.instance_visible_chunk_buffer.is_valid()) {
			if (r_reason) {
				*r_reason = "resident_visible_chunk_buffer_failed";
			}
			p_renderer->clear_instance_pipeline_buffers();
			return false;
		}
		rd->set_resource_name(resource_state.instance_visible_chunk_buffer, "GS_InstanceVisibleChunks");
		p_renderer->track_resource_owner(resource_state.instance_visible_chunk_buffer, rd);
		resource_state.instance_visible_chunk_capacity = MAX<uint32_t>(1u, buffers.max_visible_chunks);
	}
	buffers.visible_chunk_buffer = resource_state.instance_visible_chunk_buffer;

	if (!resource_state.instance_splat_ref_buffer.is_valid() ||
			resource_state.instance_splat_ref_capacity < buffers.max_visible_splats) {
		p_renderer->free_owned_resource(rd, resource_state.instance_splat_ref_buffer);
		const uint32_t buffer_size = MAX<uint32_t>(1u, buffers.max_visible_splats) * sizeof(SplatRefGPU);
		resource_state.instance_splat_ref_buffer = rd->storage_buffer_create(buffer_size);
		if (!resource_state.instance_splat_ref_buffer.is_valid()) {
			if (r_reason) {
				*r_reason = "resident_splat_ref_buffer_failed";
			}
			p_renderer->clear_instance_pipeline_buffers();
			return false;
		}
		rd->set_resource_name(resource_state.instance_splat_ref_buffer, "GS_InstanceSplatRefs");
		p_renderer->track_resource_owner(resource_state.instance_splat_ref_buffer, rd);
		resource_state.instance_splat_ref_capacity = MAX<uint32_t>(1u, buffers.max_visible_splats);
	}
	buffers.splat_ref_buffer = resource_state.instance_splat_ref_buffer;

	if (!resource_state.instance_counter_buffer.is_valid()) {
		resource_state.instance_counter_buffer = rd->storage_buffer_create(sizeof(uint32_t) * 2);
		if (!resource_state.instance_counter_buffer.is_valid()) {
			if (r_reason) {
				*r_reason = "resident_counter_buffer_failed";
			}
			p_renderer->clear_instance_pipeline_buffers();
			return false;
		}
		rd->set_resource_name(resource_state.instance_counter_buffer, "GS_InstanceCounters");
		p_renderer->track_resource_owner(resource_state.instance_counter_buffer, rd);
	}
	buffers.counter_buffer = resource_state.instance_counter_buffer;

	if (!resource_state.instance_chunk_dispatch_buffer.is_valid()) {
		resource_state.instance_chunk_dispatch_buffer = rd->storage_buffer_create(
				sizeof(uint32_t) * 3, Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
		if (!resource_state.instance_chunk_dispatch_buffer.is_valid()) {
			if (r_reason) {
				*r_reason = "resident_chunk_dispatch_buffer_failed";
			}
			p_renderer->clear_instance_pipeline_buffers();
			return false;
		}
		rd->set_resource_name(resource_state.instance_chunk_dispatch_buffer, "GS_InstanceChunkDispatch");
		p_renderer->track_resource_owner(resource_state.instance_chunk_dispatch_buffer, rd);
	}
	buffers.chunk_dispatch_buffer = resource_state.instance_chunk_dispatch_buffer;

	if (!resource_state.instance_indirect_count_buffer.is_valid()) {
		resource_state.instance_indirect_count_buffer = rd->storage_buffer_create(
				sizeof(GaussianSplatting::IndirectDispatchLayout), Vector<uint8_t>(),
				RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
		if (!resource_state.instance_indirect_count_buffer.is_valid()) {
			if (r_reason) {
				*r_reason = "resident_indirect_count_buffer_failed";
			}
			p_renderer->clear_instance_pipeline_buffers();
			return false;
		}
		rd->set_resource_name(resource_state.instance_indirect_count_buffer, "GS_InstanceIndirectCount");
		p_renderer->track_resource_owner(resource_state.instance_indirect_count_buffer, rd);
	}
	buffers.indirect_count_buffer = resource_state.instance_indirect_count_buffer;

	if (!resource_state.instance_count_buffer.is_valid()) {
		resource_state.instance_count_buffer = rd->storage_buffer_create(
				sizeof(GaussianSplatting::IndirectDispatchLayout));
		if (!resource_state.instance_count_buffer.is_valid()) {
			if (r_reason) {
				*r_reason = "resident_instance_count_buffer_failed";
			}
			p_renderer->clear_instance_pipeline_buffers();
			return false;
		}
		rd->set_resource_name(resource_state.instance_count_buffer, "GS_InstanceCount");
		p_renderer->track_resource_owner(resource_state.instance_count_buffer, rd);
	}
	buffers.instance_count_buffer = resource_state.instance_count_buffer;

	Ref<GPUSortingPipeline> sorting_pipeline = p_renderer->get_subsystem_state().sorting_pipeline;
	if (sorting_pipeline.is_null()) {
		if (r_reason) {
			*r_reason = "resident_sorting_pipeline_unavailable";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}
	const uint32_t required_sort_capacity = MAX<uint32_t>(1u, buffers.max_visible_splats);
	if (required_sort_capacity > sorting_pipeline->get_max_elements()) {
		sorting_pipeline->rebuild_sorter(required_sort_capacity);
	}
	sorting_pipeline->ensure_buffers(required_sort_capacity);
	const SortBufferHandles sort_handles = sorting_pipeline->get_buffer_handles();
	if (!sort_handles.valid) {
		if (r_reason) {
			*r_reason = "resident_sort_buffers_unavailable";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}
	buffers.sort_key_buffer = sort_handles.keys_buffer;
	buffers.sort_value_buffer = sort_handles.indices_buffer;
	if (sort_handles.capacity > 0 && buffers.max_visible_splats > sort_handles.capacity) {
		buffers.max_visible_splats = sort_handles.capacity;
		buffers.max_visible_chunks = MIN<uint32_t>(buffers.max_visible_chunks, buffers.max_visible_splats);
	}

	GaussianRenderPipeline::PublishedInstanceAssetRemap remap;
	remap.asset_to_dense_id.insert(kPrimaryResidentAssetId, 0u);
	for (const ResidentAssetDescriptor &asset : assets) {
		remap.asset_to_dense_id.insert(asset.submission_asset_id, asset.dense_asset_id);
	}
	remap.generation = source_generation == 0 ? 1u : source_generation;
	remap.valid = true;

	p_renderer->publish_instance_pipeline_contract(buffers, remap,
			GaussianRenderPipeline::InstanceBackendPolicy::RESIDENT, source_generation, "atlas_emulation");
	resource_state.instance_pipeline_content_generation = source_generation;

	if (!p_renderer->update_instance_buffer(instances)) {
		if (r_reason) {
			*r_reason = "resident_instance_upload_failed";
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	const GaussianRenderPipeline::InstancePipelineBuffers &published_buffers = p_renderer->get_instance_pipeline_buffers();
	InvariantViolationReason violation_reason = InvariantViolationReason::NONE;
	if (!GaussianSplatting::InstancePipelineContract::has_atlas_buffers(published_buffers)) {
		violation_reason = GaussianSplatting::InstancePipelineContract::first_atlas_violation(published_buffers);
	} else if (!GaussianSplatting::InstancePipelineContract::has_cull_buffers(published_buffers)) {
		violation_reason = GaussianSplatting::InstancePipelineContract::first_cull_violation(published_buffers);
	} else if (!GaussianSplatting::InstancePipelineContract::has_sort_buffers(published_buffers)) {
		violation_reason = GaussianSplatting::InstancePipelineContract::first_sort_violation(published_buffers);
	} else if (!GaussianSplatting::InstancePipelineContract::has_raster_buffers(published_buffers)) {
		violation_reason = GaussianSplatting::InstancePipelineContract::first_raster_violation(published_buffers);
	}
	if (violation_reason != InvariantViolationReason::NONE) {
		if (r_reason) {
			*r_reason = GaussianSplatting::InstancePipelineContract::get_violation_reason_name(violation_reason);
		}
		p_renderer->clear_instance_pipeline_buffers();
		return false;
	}

	return true;
}

} // namespace ResidentInstanceContractPublisher
