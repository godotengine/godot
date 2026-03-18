#ifndef TILE_RENDER_RESOURCES_H
#define TILE_RENDER_RESOURCES_H

#include "tile_render_types.h"
#include "gpu_sorter.h"

#include <memory>

class TileBinningShaderRD;
class TilePrefixScanShaderRD;
class TileRasterizerShaderRD;
class TileRasterizerComputeShaderRD;
class TileResolveShaderRD;

class GPUPerformanceMonitor;
class TileRenderer;

namespace GaussianSplatting {

struct TileRenderTargets {
	explicit TileRenderTargets(TileRenderer &p_owner) : owner(p_owner) {}

	void destroy_output_textures();
	void destroy_resolve_textures();
	void create_output_textures(const Vector2i &p_size, RD::DataFormat p_format);
	void ensure_resolve_resources(const Vector2i &p_size, RD::DataFormat p_format);

	RID get_output_texture(bool p_prefer_raw, bool p_force_resolved) const;
	RID get_depth_texture(bool p_prefer_raw, bool p_force_resolved) const;
	RenderingDevice *get_output_texture_owner(bool p_prefer_raw, bool p_force_resolved) const;
	RenderingDevice *get_depth_texture_owner(bool p_prefer_raw, bool p_force_resolved) const;

	TileRenderer &owner;
	RID output_texture;
	RID depth_texture;
	RID normal_texture;
	RID output_texture_external;
	RID depth_texture_external;
	RID resolved_texture;
	RID resolved_depth_texture;
	RID resolved_texture_external;
	RID resolved_depth_texture_external;
	RID tile_framebuffer;
	RD::FramebufferFormatID tile_framebuffer_format = RD::INVALID_ID;
	RenderingDevice *tile_framebuffer_owner = nullptr;
	BufferOwnership output_texture_owner;
	RenderingDevice *output_texture_local_owner = nullptr;
	bool output_texture_external_owned = false;
	BufferOwnership depth_texture_owner;
	RenderingDevice *depth_texture_local_owner = nullptr;
	bool depth_texture_external_owned = false;
	BufferOwnership normal_texture_owner;
	RenderingDevice *normal_texture_local_owner = nullptr;
	BufferOwnership resolved_texture_owner;
	RenderingDevice *resolved_texture_local_owner = nullptr;
	BufferOwnership resolved_depth_texture_owner;
	RenderingDevice *resolved_depth_texture_local_owner = nullptr;
	bool depth_texture_copy_compatible = false;
};

struct TileDeviceContext {
	RenderingDevice *resource_rd = nullptr;
	RenderingDevice *submission_rd = nullptr;
	bool warned_missing_submission_device = false;
	GPUPerformanceMonitor *performance_monitor = nullptr;
};

struct TileShaderResources {
	explicit TileShaderResources(TileRenderer &p_owner);
	~TileShaderResources();

	void release(RenderingDevice *p_device, RenderingDevice *p_pipeline_owner);
	void reset_state();

	TileRenderer &owner;
	RID tile_binning_shader;
	RID tile_binning_pipeline;
	RID tile_binning_count_shader;
	RID tile_binning_count_pipeline;
	RID tile_prefix_shader;
	RID tile_prefix_shader_pass2;
	RID tile_prefix_shader_pass3;
	RID tile_prefix_pipeline_pass1;
	RID tile_prefix_pipeline_pass2;
	RID tile_prefix_pipeline_pass3;
	RID tile_raster_shader;
	RID tile_raster_pipeline;
	RID tile_raster_compute_shader;
	RID tile_raster_compute_pipeline;
	RID tile_resolve_shader;
	RID tile_resolve_pipeline;
	RenderingDevice *shader_device = nullptr;
	uint64_t shader_device_instance = 0;
	std::unique_ptr<TileBinningShaderRD> tile_binning_shader_source;
	std::unique_ptr<TilePrefixScanShaderRD> tile_prefix_shader_source;
	std::unique_ptr<TileRasterizerShaderRD> tile_raster_shader_source;
	std::unique_ptr<TileRasterizerComputeShaderRD> tile_raster_compute_shader_source;
	std::unique_ptr<TileResolveShaderRD> tile_resolve_shader_source;
	bool tile_binning_shader_initialized = false;
	bool tile_prefix_shader_initialized = false;
	bool tile_raster_shader_initialized = false;
	bool tile_raster_compute_shader_initialized = false;
	bool tile_resolve_shader_initialized = false;
	bool subgroups_available = false;
	bool resolve_pipeline_initialized = false;
	bool quantized_storage_enabled = false;
	uint64_t shader_defines_hash = 0;
	RD::DataFormat resolve_pipeline_format = RD::DATA_FORMAT_MAX;
};

struct TileGlobalSortResources {
	explicit TileGlobalSortResources(TileRenderer &p_owner) : owner(p_owner) {}

	void release(RenderingDevice *p_default_device);
	void reset_state(bool p_clear_sorter);
	void ensure_resources(uint32_t p_visible_count);

	TileRenderer &owner;
	Ref<IGPUSorter> sorter;
	bool sorter_available = true;
	bool sorter_missing_logged = false;
	uint64_t sorter_device_id = 0;
	uint32_t capacity = 0;
	SortKeyConfig key_config;
	RID keys_buffer;
	RID values_buffer;
	RID tile_counts_buffers[2];
	uint32_t tile_counts_index = 0;
	bool tile_counts_ready[2] = { false, false };
	uint32_t tile_counts_bytes = 0;
	RID tile_ranges_buffer;
	RID prefix_total_buffer;
	RID indirect_dispatch_buffer;
	RID wg_sums_buffer;
	RID wg_offsets_buffer;
	BufferOwnership buffer_owner;
	uint32_t tile_buffer_tiles = 0;

	RID get_tile_counts_buffer() const {
		return tile_counts_buffers[tile_counts_index];
	}

	void advance_tile_counts_buffer();
	void mark_tile_counts_dirty();
	bool ensure_tile_counts_ready(RenderingDevice *p_device);
	void prepare_next_tile_counts_buffer(RenderingDevice *p_device);
};

struct TileUniformBuffers {
	explicit TileUniformBuffers(TileRenderer &p_owner) : owner(p_owner) {}

	void release(RenderingDevice *p_default_device);
	void reset_state();
	bool ensure_param_buffer(RenderingDevice *p_device);
	bool ensure_prefix_param_buffer(RenderingDevice *p_device, uint32_t p_size);
	RID get_default_state_uniform(RenderingDevice *p_device);

	TileRenderer &owner;
	RID param_uniform_buffer;
	BufferOwnership param_uniform_buffer_owner;
	RID prefix_param_uniform_buffer;
	BufferOwnership prefix_param_owner;
	RID default_state_uniform_buffer;
	RenderingDevice *default_state_uniform_owner = nullptr;
};

struct TileProjectionBuffers {
	explicit TileProjectionBuffers(TileRenderer &p_owner) : owner(p_owner) {}

	void release(RenderingDevice *p_default_device);
	void reset_state();
	void ensure_projection_buffer(uint32_t p_visible_count);

	TileRenderer &owner;
	RID projection_buffer;
	BufferOwnership projection_buffer_owner;
	uint32_t projection_buffer_size = 0;
	uint64_t sorter_device_id = 0;
	bool sorter_available = true;
	bool sorter_missing_logged = false;
};

struct TileSHCacheBuffers {
	explicit TileSHCacheBuffers(TileRenderer &p_owner) : owner(p_owner) {}

	void release(RenderingDevice *p_default_device);
	void reset_state();
	bool ensure_color_cache(uint32_t p_total_gaussians);

	TileRenderer &owner;
	RID sh_color_cache;
	BufferOwnership sh_color_cache_owner;
	uint32_t sh_color_cache_size = 0;
};

struct TileSubpixelHistoryBuffers {
	explicit TileSubpixelHistoryBuffers(TileRenderer &p_owner) : owner(p_owner) {}

	void release(RenderingDevice *p_default_device);
	void reset_state();
	bool ensure_history_buffer(uint32_t p_total_gaussians);

	TileRenderer &owner;
	RID subpixel_history_buffer;
	BufferOwnership subpixel_history_owner;
	uint32_t subpixel_history_size = 0;
};

struct TileSubpixelVisibilityBuffers {
	explicit TileSubpixelVisibilityBuffers(TileRenderer &p_owner) : owner(p_owner) {}

	void release(RenderingDevice *p_default_device);
	void reset_state();
	bool ensure_visibility_buffer(uint32_t p_visible_count);

	TileRenderer &owner;
	RID subpixel_visibility_buffer;
	BufferOwnership subpixel_visibility_owner;
	uint32_t subpixel_visibility_size = 0;
};

} // namespace GaussianSplatting

#endif // TILE_RENDER_RESOURCES_H
