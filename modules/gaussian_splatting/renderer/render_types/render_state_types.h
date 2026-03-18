/**
 * @file render_state_types.h
 * @brief Shared renderer state type definitions for orchestrators.
 *
 * This header contains state/data structs used by renderer orchestrators so
 * they can depend on narrow contracts instead of the full renderer header.
 */

#ifndef GAUSSIAN_RENDER_STATE_TYPES_H
#define GAUSSIAN_RENDER_STATE_TYPES_H

#include "core/math/transform_3d.h"
#include "core/object/object_id.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid.h"
#include "core/templates/vector.h"
#include "../../core/gaussian_data.h"
#include "../../core/gaussian_splat_asset.h"
#include "../../core/gaussian_splat_manager.h"
#include "../../core/gaussian_streaming.h"
#include "../gaussian_gpu_layout.h"
#include "../gpu_memory_stream.h"
#include <cstdint>

class IGPUSorter;

namespace GaussianRenderState {

enum class IndexDomain : uint8_t {
	UNKNOWN = 0,
	CHUNK_REF,
	SPLAT_REF,
	GAUSSIAN_GLOBAL,
};

inline const char *index_domain_to_string(IndexDomain p_domain) {
	switch (p_domain) {
		case IndexDomain::CHUNK_REF:
			return "chunk_ref";
		case IndexDomain::SPLAT_REF:
			return "splat_ref";
		case IndexDomain::GAUSSIAN_GLOBAL:
			return "gaussian_global";
		case IndexDomain::UNKNOWN:
		default:
			return "unknown";
	}
}

struct SceneState {
	Ref<::GaussianData> gaussian_data;
	Ref<GaussianSplatAsset> active_asset;
};

struct CullStageOutput {
	bool has_visible = false;
	uint32_t visible_count = 0;
	uint32_t candidate_count = 0;
	float cull_time_ms = 0.0f;
	IndexDomain visible_domain = IndexDomain::UNKNOWN;
};

struct SortStageSummary {
	uint32_t sorted_count = 0;
	float sort_time_ms = 0.0f;
	IndexDomain input_domain = IndexDomain::UNKNOWN;
	IndexDomain output_domain = IndexDomain::UNKNOWN;
};

struct StreamingState {
	Ref<GaussianMemoryStream> memory_stream;
	Ref<GaussianStreamingSystem> current_streaming_system;
	uint32_t streaming_gpu_splat_count = 0;
	ObjectID registered_gaussian_data_id = ObjectID();
	GaussianSplatManager::SharedDynamicAssetHandle shared_dynamic_asset_handle;
	RID current_stream_gpu_buffer;
	RID registered_gaussian_buffer;
	RID gpu_gaussian_cache_buffer;
	LocalVector<Gaussian> cached_streamed_gaussians;
	LocalVector<uint32_t> cached_streamed_indices;
	LocalVector<uint32_t> cached_streamed_source_indices;
	LocalVector<uint8_t> cached_streamed_sh_limits;
	HashMap<uint32_t, uint32_t> cached_streamed_index_lookup;
	LocalVector<PackedGaussian> gpu_gaussian_cache;
	uint32_t gpu_gaussian_cache_start = 0;
	uint32_t gpu_gaussian_cache_count = 0;
	uint64_t gpu_gaussian_cache_frame = 0;
	bool gpu_gaussian_cache_valid = false;
	uint64_t streamed_indices_generation = 0;
	uint32_t streaming_gpu_total_capacity = 0;
	bool streamed_indices_are_local = false;
	bool cached_streamed_indices_valid = false;
	bool use_streamed_data = false;
	uint64_t last_streaming_init_attempt_frame = 0;

	uint64_t last_rebuild_frame = 0;
	uint64_t last_throttle_log_frame = 0;
	uint64_t last_perf_log_frame = 0;
	uint32_t perf_log_counter = 0;

	static constexpr uint32_t SMALL_CHANGE_THRESHOLD = 8;
	static constexpr uint64_t MIN_FRAMES_BETWEEN_SMALL_REBUILDS = 30;
	static constexpr uint64_t LOG_THROTTLE_FRAMES = 300;
};

struct SortingState {
	Ref<IGPUSorter> gpu_sorter;
	uint32_t local_sort_buffer_capacity = 0;
	uint32_t sort_buffer_capacity = 0;
	uint32_t culled_position_capacity = 0;
	bool sorter_needs_rebuild = true;
	Vector<uint8_t> sort_key_bytes;
	Vector<uint8_t> sort_index_bytes;
	Vector<uint8_t> culled_position_bytes;
	bool sort_keys_external = false;
	bool sort_indices_external = false;
	bool sort_buffers_pipeline_managed = false;
	uint64_t current_sort_timeline_value = 0;
	uint64_t last_sort_submission_value = 0;
	bool sorting_in_progress = false;
	bool sorting_initialized = false;
	uint32_t sorted_splat_count = 0;
	uint64_t last_cull_indices_signature = 0;
	bool last_cull_indices_signature_valid = false;
	String active_sort_algorithm = "uninitialized";
	String sort_switch_reason = "uninitialized";
	bool override_force_cpu = false;
	bool override_force_algorithm = false;
	String override_forced_algorithm = "auto";
	Transform3D last_sort_world_to_camera_transform;
	bool last_sort_transform_valid = false;
	uint32_t sorter_init_failure_count = 0;
	uint64_t last_sorter_init_failure_frame = 0;
	static constexpr uint32_t kSorterInitBackoffFrames = 60;
	static constexpr uint32_t kSorterInitMaxFailures = 5;
};

} // namespace GaussianRenderState

#endif // GAUSSIAN_RENDER_STATE_TYPES_H
