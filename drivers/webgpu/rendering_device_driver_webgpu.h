/**************************************************************************/
/*  rendering_device_driver_webgpu.h                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#ifdef WEBGPU_ENABLED

#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/safe_refcount.h"
#include "drivers/webgpu/spirv_preprocess.h"
#include "drivers/webgpu/webgpu_objects.h"
#include "servers/rendering/rendering_device_driver.h"

#include <webgpu/webgpu.h>

class RenderingContextDriverWebGPU;
class RenderingShaderContainerFormatWebGPU;

class RenderingDeviceDriverWebGPU : public RenderingDeviceDriver {
	RenderingContextDriverWebGPU *context_driver = nullptr;
	WGPUDevice device = nullptr;
	WGPUQueue queue = nullptr;

	uint32_t frame_count = 1;
	uint32_t frame_index = 0;
	uint32_t frames_drawn = 0;

	// Per-frame performance counters (logged once per second in WEBGPU_VERBOSE builds).
	struct PerfCounters {
		uint32_t draw_calls = 0;
		uint32_t set_bind_group_calls = 0;
		uint32_t push_constant_writes = 0;
		uint32_t push_constant_skipped = 0;
		uint32_t render_passes = 0;
		uint32_t bind_group_cache_misses = 0;
		uint32_t set_pipeline_calls = 0;
		uint32_t set_vertex_buffer_calls = 0;
		uint32_t gap_bind_group_calls = 0;
		uint32_t first_instance_draws = 0;
		uint32_t ring_overflows = 0;
		double last_log_time = 0;
		uint32_t frames_since_log = 0;
		void reset() {
			draw_calls = 0;
			set_bind_group_calls = 0;
			push_constant_writes = 0;
			push_constant_skipped = 0;
			render_passes = 0;
			bind_group_cache_misses = 0;
			set_pipeline_calls = 0;
			set_vertex_buffer_calls = 0;
			gap_bind_group_calls = 0;
			first_instance_draws = 0;
			ring_overflows = 0;
		}
	} perf;

	Capabilities capabilities;
	MultiviewCapabilities multiview_capabilities;
	FragmentShadingRateCapabilities fsr_capabilities;
	FragmentDensityMapCapabilities fdm_capabilities;
	WGPULimits device_limits = WGPU_LIMITS_INIT;
	bool has_texture_formats_tier1 = false;
	bool has_core_features = false;
	bool has_texture_formats_tier2 = false;
	bool has_rw_storage_textures = false; // readonly-and-readwrite-storage-textures

	bool float32_filterable_supported = false;
	bool float32_blendable_supported = false;
	// Optional texture-compression features (BC, ETC2, ASTC). Requested by the JS
	// shell at device creation; the driver must only report the corresponding
	// DataFormats as supported when the feature is actually enabled, otherwise
	// texture creation will fail at runtime.
	bool has_texture_compression_bc = false;
	bool has_texture_compression_etc2 = false;
	bool has_texture_compression_astc = false;

	RenderingShaderContainerFormatWebGPU *shader_container_format = nullptr;

	// --- Push Constant Emulation ---
	// Ring buffer for push constant data. Each slot is 256-byte aligned.
	WGPUBuffer push_constant_ring_buffer = nullptr;
	static constexpr uint32_t PUSH_CONSTANT_RING_SIZE = 512 * 1024; // 512KB = 2048 draw calls at 256B/slot
	static constexpr uint32_t PUSH_CONSTANT_SLOT_ALIGNMENT = 256;
	// Binding slot used for the PC ring buffer inside the push-constant group.
	static constexpr uint32_t PUSH_CONSTANT_RING_BINDING = spirv_preprocess::PUSH_CONSTANT_BINDING;
	uint32_t push_constant_ring_offset = 0;

	// When the ring fills up mid-segment, recording continues in a new "page": the page's data
	// goes to push_constant_pages[page - 1] and an in-encoder copy moves it into the ring, so
	// every draw sees its own data in submission order without submitting mid-recording.
	uint32_t push_constant_page = 0;
	LocalVector<WGPUBuffer> push_constant_pages; // CopySrc | CopyDst, PUSH_CONSTANT_RING_SIZE each.

	// CPU-side shadow of all pages (page N at N * PUSH_CONSTANT_RING_SIZE) for batched writes.
	// Accumulated during command recording, flushed once before queue submit.
	LocalVector<uint8_t> push_constant_shadow;
	uint64_t push_constant_shadow_dirty_start = UINT64_MAX;
	uint64_t push_constant_shadow_dirty_end = 0;
	void _flush_push_constant_shadow();

	// Universal push constant bind group layout: group N, binding PUSH_CONSTANT_RING_BINDING,
	// ReadOnlyStorage buffer with hasDynamicOffset=true. Created once at initialize().
	// All shaders use var<storage, read> for the PC ring buffer — var<uniform> would require
	// std140 layout which is incompatible with push constant structs containing arrays.
	WGPUBindGroupLayout push_constant_bind_group_layout = nullptr;
	// Bind group backed by the ring buffer (reused every frame with dynamic offsets).
	WGPUBindGroup push_constant_bind_group = nullptr;

	// Empty bind group for filling pipeline layout gaps.
	// Firefox/wgpu requires ALL bind group slots to be set before draw calls.
	WGPUBindGroupLayout empty_bind_group_layout = nullptr;
	WGPUBindGroup empty_bind_group = nullptr;

	// --- Fallback Textures ---
	// Small float texture used when a depth-format fallback texture is bound
	// to a BGL entry that expects Float sample type (depth textures can't be
	// sampled as Float in WebGPU, unlike Vulkan).
	WGPUTexture fallback_float_texture = nullptr;
	WGPUTextureView fallback_float_texture_view = nullptr;
	// Cube variant (6 layers) for BGL entries expecting TextureViewDimension::Cube.
	WGPUTexture fallback_cube_texture = nullptr;
	WGPUTextureView fallback_cube_texture_view = nullptr;
	// Multisampled (4x) variant for BGL entries expecting multisampled=true.
	// Used when a depth-format MSAA texture can't be sampled as UnfilterableFloat
	// (e.g. ResolveRasterShaderRD binds a depth MSAA texture into a float MSAA slot).
	WGPUTexture fallback_ms_texture = nullptr;
	WGPUTextureView fallback_ms_texture_view = nullptr;

	// --- Dummy Samplers for BGL Rebinding ---
	// When a bind group must be re-created with a different BGL (e.g., because
	// the original BGL has a Comparison sampler but the target has Filtering),
	// these dummy samplers are used as substitutes.
	WGPUSampler dummy_filtering_sampler = nullptr;
	WGPUSampler dummy_comparison_sampler = nullptr;

	// Bindings no stage reads get layout entries guessed from Godot's uniform types, which the bound resource need
	// not match. WebGPU validates them anyway, so they bind 1x1 dummies matching the guess, keyed by layout shape.
	HashMap<uint64_t, Pair<WGPUTexture, WGPUTextureView>> unused_binding_views;
	WGPUTextureView _unused_binding_view(const WGPUBindGroupLayoutEntry &p_layout);
	void _fit_unused_bindings(LocalVector<WGPUBindGroupEntry> &r_entries, const WGShader *p_shader, uint32_t p_set_idx);

	// --- BGL Rebinding Helper ---
	WGPUBindGroup _get_compatible_bind_group(WGUniformSet *p_us, WGShader *p_target_shader, uint32_t p_set_idx);

	// --- Internal Helpers ---
	void _check_capabilities();
	WGPUTextureFormat _data_format_to_wgpu(DataFormat p_format) const;
	DataFormat _wgpu_to_data_format(WGPUTextureFormat p_format) const;
	// Promotes R8/RG8/R16/RG16 formats to their 32-bit equivalents when a texture
	// or render target needs STORAGE usage (base WebGPU does not support those as
	// storage texel formats). With texture-formats-tier1, these formats are valid
	// storage formats natively and promotion is skipped.
	WGPUTextureFormat _promote_storage_format(WGPUTextureFormat p_format) const;
	WGPUBufferUsage _buffer_usage_to_wgpu(BitField<BufferUsageBits> p_usage) const;
	WGPUTextureUsage _texture_usage_to_wgpu(BitField<TextureUsageBits> p_usage) const;
	WGPUTextureDimension _texture_type_to_dimension(TextureType p_type) const;
	WGPUTextureViewDimension _texture_type_to_view_dimension(TextureType p_type) const;

	void _flush_push_constants(WGCommandBuffer *p_cmd_buf, WGShader *p_shader);
	void _restart_pass_after_push_constant_page(WGCommandBuffer *p_cmd_buf, bool p_compute);
	void _rebind_render_state(WGCommandBuffer *p_cmd_buf);

	// Region clears, as in the Metal driver: WebGPU's loadOp=Clear always covers the whole attachment,
	// so partial clears draw a viewported triangle that writes the clear values (colors arrive through
	// the push constant ring, depth through the viewport depth range, stencil as the reference).
	static constexpr uint32_t CLEAR_MAX_COLORS = WGCommandBuffer::MAX_PUSH_CONSTANT_SIZE / (4 * sizeof(float));
	struct ClearPipelineKey {
		WGPUTextureFormat color_formats[CLEAR_MAX_COLORS] = {}; // Undefined for unused slots.
		WGPUTextureFormat depth_stencil_format = WGPUTextureFormat_Undefined;
		uint32_t color_count = 0;
		uint32_t color_write_mask = 0; // Bit i: colour slot i is cleared.
		uint32_t depth_stencil_write = 0; // Bit 0: depth, bit 1: stencil.
		uint32_t sample_count = 1;

		bool operator==(const ClearPipelineKey &p_other) const { return memcmp(this, &p_other, sizeof(ClearPipelineKey)) == 0; }
		uint32_t hash() const { return hash_murmur3_buffer(this, sizeof(ClearPipelineKey)); }
	};
	HashMap<ClearPipelineKey, WGPURenderPipeline> clear_pipelines;
	WGPUPipelineLayout clear_pipeline_layout = nullptr;
	WGShader clear_shader; // Only routes the clear colors through _flush_push_constants at group 0.
	WGPURenderPipeline _get_clear_pipeline(const ClearPipelineKey &p_key);
	void _render_clear(WGCommandBuffer *p_cmd_buf, VectorView<AttachmentClear> p_clears, VectorView<Rect2i> p_rects);
	void _copy_buffer_to_texture(WGCommandBuffer *p_cmd_buf, WGBuffer *p_src, WGTexture *p_dst, const BufferTextureCopyRegion &p_region, uint32_t p_depth_or_layers);
	// Native Dawn parses and validates WGSL on module creation, which dominates loading many shader variants.
	Mutex stage_module_mutex;
	WGPUShaderModule _stage_module(WGShader *p_shader, ShaderStage p_stage);
	WGPUShaderModule _create_module_with_spec_constants(const PackedByteArray &p_spirv, VectorView<PipelineSpecializationConstant> p_constants, ShaderStage p_stage, const WGShader *p_shader);
	PipelineID _render_pipeline_create_now(ShaderID p_shader, VertexFormatID p_vertex_format, RenderPrimitive p_render_primitive, PipelineRasterizationState p_rasterization_state, PipelineMultisampleState p_multisample_state, PipelineDepthStencilState p_depth_stencil_state, PipelineColorBlendState p_blend_state, VectorView<int32_t> p_color_attachments, BitField<PipelineDynamicStateFlags> p_dynamic_state, RenderPassID p_render_pass, uint32_t p_render_subpass, VectorView<PipelineSpecializationConstant> p_specialization_constants);
	PipelineID _compute_pipeline_create_now(ShaderID p_shader, VectorView<PipelineSpecializationConstant> p_specialization_constants);
	void _create_deferred_pipeline(WGPipelineWrapper *p_pw);

	// Bindings rewritten by _postprocess_wgsl that the bind group layouts must mirror.
	struct WGSLRewrites {
		String error;
		HashSet<uint32_t> read_storage_to_sampled;
		HashMap<uint32_t, WGPUTextureFormat> read_storage_formats;
	};
	// Applies the WebGPU-compatibility text rewrites to Tint's WGSL for one stage.
	CharString _postprocess_wgsl(const char *p_wgsl, ShaderStage p_stage, const HashSet<uint32_t> &p_vertex_storage_buffers, WGSLRewrites &r_rewrites) const;
	SafeNumeric<uint64_t> shader_uid_counter;

public:
	RenderingDeviceDriverWebGPU(RenderingContextDriverWebGPU *p_context_driver);
	virtual ~RenderingDeviceDriverWebGPU() override;

	// -----------------------------------------------------------------------
	// GENERIC
	// -----------------------------------------------------------------------

	/// Initialize the WebGPU driver: import the device from JS, create push
	/// constant ring buffer, bind group layouts, fallback textures, and samplers.
	virtual Error initialize(uint32_t p_device_index, uint32_t p_frame_count) override final;

	// -----------------------------------------------------------------------
	// BUFFERS
	// -----------------------------------------------------------------------

	/// Create a GPU buffer. WebGPU sizes are 4-byte aligned internally.
	virtual BufferID buffer_create(uint64_t p_size, BitField<BufferUsageBits> p_usage, MemoryAllocationType p_allocation_type, uint64_t p_frames_drawn) override final;
	/// Create a GPU buffer with initial data using mappedAtCreation (zero staging overhead).
	virtual BufferID buffer_create_with_data(uint64_t p_size, BitField<BufferUsageBits> p_usage, MemoryAllocationType p_allocation_type, const uint8_t *p_data, uint64_t p_data_size) override final;
	/// Stub — WebGPU has no texel buffer views. Always returns true.
	virtual bool buffer_set_texel_format(BufferID p_buffer, DataFormat p_format) override final;
	/// Release a GPU buffer and its shadow CPU memory.
	virtual void buffer_free(BufferID p_buffer) override final;
	/// Return the aligned allocation size of a buffer.
	virtual uint64_t buffer_get_allocation_size(BufferID p_buffer) override final;
	/// Map a buffer for CPU access. Uses a shadow CPU buffer since WebGPU
	/// buffer mapping is async. For readback buffers, the shadow is populated
	/// by async wgpuBufferMapAsync callbacks between frames.
	virtual uint8_t *buffer_map(BufferID p_buffer) override final;
	/// Unmap a buffer. If the shadow was modified, writes data to the GPU buffer.
	virtual void buffer_unmap(BufferID p_buffer) override final;
	/// Persistent map for streaming uploads — returns shadow CPU buffer.
	virtual uint8_t *buffer_persistent_map_advance(BufferID p_buffer, uint64_t p_frames_drawn) override final;
	/// Return dynamic offsets for buffers. Stub — returns 0.
	virtual uint64_t buffer_get_dynamic_offsets(Span<BufferID> p_buffers) override final;
	/// Flush a buffer's shadow CPU data to the GPU via wgpuQueueWriteBuffer.
	virtual void buffer_flush(BufferID p_buffer) override final;
	/// Direct queue write bypassing staging buffers (for skeleton/bone updates).
	virtual void buffer_write_direct(BufferID p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) override final;
	/// WebGPU: initiate async buffer map so it completes by next frame.
	virtual void buffer_initiate_async_map(BufferID p_buffer) override final;
	virtual uint32_t texture_get_gpu_pixel_size(TextureID p_texture) override final;
	virtual void texture_readback_convert(TextureID p_texture,
			const uint8_t *p_src, uint32_t p_src_pitch,
			uint8_t *p_dst, uint32_t p_dst_pitch,
			uint32_t p_width, uint32_t p_height) override final;
	virtual void texture_upload_convert(TextureID p_texture,
			const uint8_t *p_src, uint32_t p_src_pitch,
			uint8_t *p_dst, uint32_t p_dst_pitch,
			uint32_t p_width, uint32_t p_height) override final;
	/// Stub — WebGPU has no buffer device addresses. Returns 0.
	virtual uint64_t buffer_get_device_address(BufferID p_buffer) override final;
	/// WebGPU-specific: direct buffer readback using persistent staging buffer cache.
	/// Always handles the request on web: r_data stays empty until the first async readback
	/// lands (1-frame latency), then holds the most recent completed readback.
	/// Bypasses the default staging buffer path which can't handle WebGPU's async-only map.
	virtual bool buffer_get_data_direct(BufferID p_buffer, uint64_t p_offset, uint64_t p_size, Vector<uint8_t> &r_data) override final;

	/// Persistent readback entry for async buffer/texture map operations.
	/// Heap-allocated so that pointers remain stable across HashMap rehashes
	/// and can safely be passed to async WebGPU map callbacks.
	struct ReadbackEntry {
		WGPUBuffer staging = nullptr; ///< Persistent staging buffer (CopyDst | MapRead).
		uint8_t *shadow = nullptr; ///< CPU-side shadow buffer.
		uint64_t size = 0; ///< Buffer size in bytes.
		bool map_complete = false; ///< Set by async map callback.
		bool map_pending = false; ///< True while mapAsync is in flight (not yet completed).
		bool has_data = false; ///< True after first successful readback.
		bool cancelled = false; ///< Source freed while map pending; callback will clean up.
	};
	HashMap<uint64_t, ReadbackEntry *> _readback_cache; ///< Keyed by source buffer/texture pointer.
	/// Async map callback — copies GPU data to shadow buffer.
	static void _readback_map_cb(WGPUMapAsyncStatus p_status, WGPUStringView p_message, void *p_userdata1, void *p_userdata2);

	// -----------------------------------------------------------------------
	// TEXTURES
	// -----------------------------------------------------------------------

	/// Create a GPU texture. Formats mapped via pixel_formats_webgpu.h.
	/// 3-component formats (RGB8, etc.) are promoted to RGBA equivalents.
	virtual TextureID texture_create(const TextureFormat &p_format, const TextureView &p_view) override final;
	/// Stub — not applicable for WebGPU web exports (no native texture interop).
	virtual TextureID texture_create_from_extension(uint64_t p_native_texture, TextureType p_type, DataFormat p_format, uint32_t p_array_layers, bool p_depth_stencil, uint32_t p_mipmaps) override final;
	/// Create a shared texture view from an existing texture.
	virtual TextureID texture_create_shared(TextureID p_original_texture, const TextureView &p_view) override final;
	/// Create a shared texture view for a specific slice (layer/mipmap range).
	virtual TextureID texture_create_shared_from_slice(TextureID p_original_texture, const TextureView &p_view, TextureSliceType p_slice_type, uint32_t p_layer, uint32_t p_layers, uint32_t p_mipmap, uint32_t p_mipmaps) override final;
	/// Release a texture and its default view.
	virtual void texture_free(TextureID p_texture) override final;
	/// Return the GPU allocation size of a texture.
	virtual uint64_t texture_get_allocation_size(TextureID p_texture) override final;
	/// Get the copyable layout (row pitch, size) for a texture subresource.
	/// WebGPU requires 256-byte row alignment for buffer↔texture copies.
	virtual void texture_get_copyable_layout(TextureID p_texture, const TextureSubresource &p_subresource, TextureCopyableLayout *r_layout) override final;
	/// Read texture data back to CPU. Uses same persistent readback cache as buffer_get_data_direct.
	/// Returns zeros on first call (1-frame latency), valid data on subsequent calls.
	virtual Vector<uint8_t> texture_get_data(TextureID p_texture, uint32_t p_layer) override final;
	/// Query which texture usage bits are supported for a given format.
	virtual BitField<TextureUsageBits> texture_get_usages_supported_by_format(DataFormat p_format, bool p_cpu_readable) override final;
	virtual bool texture_is_format_supported_for_samples(DataFormat p_format, TextureSamples p_samples) override final;
	/// Check if a texture can create a shared view with a different format.
	virtual bool texture_can_make_shared_with_format(TextureID p_texture, DataFormat p_format, bool &r_raw_reinterpretation) override final;

	// -----------------------------------------------------------------------
	// SAMPLERS
	// -----------------------------------------------------------------------

	/// Create a sampler with the specified filter, address, LOD, and compare modes.
	virtual SamplerID sampler_create(const SamplerState &p_state) override final;
	/// Release a sampler.
	virtual void sampler_free(SamplerID p_sampler) override final;
	/// Check if a format supports the specified filter mode.
	virtual bool sampler_is_format_supported_for_filter(DataFormat p_format, SamplerFilter p_filter) override final;

	// -----------------------------------------------------------------------
	// VERTEX FORMAT
	// -----------------------------------------------------------------------

	/// Create a vertex format descriptor from vertex attributes and bindings.
	virtual VertexFormatID vertex_format_create(Span<VertexAttribute> p_vertex_attribs, const VertexAttributeBindingsMap &p_vertex_bindings) override final;
	/// Release a vertex format descriptor.
	virtual void vertex_format_free(VertexFormatID p_vertex_format) override final;

	// -----------------------------------------------------------------------
	// BARRIERS
	// -----------------------------------------------------------------------

	/// No-op — WebGPU handles synchronization automatically via the command encoder.
	/// Cross-pass barriers are handled by automatic encoder splitting.
	virtual void command_pipeline_barrier(
			CommandBufferID p_cmd_buffer,
			BitField<PipelineStageBits> p_src_stages,
			BitField<PipelineStageBits> p_dst_stages,
			VectorView<MemoryAccessBarrier> p_memory_barriers,
			VectorView<BufferBarrier> p_buffer_barriers,
			VectorView<TextureBarrier> p_texture_barriers,
			VectorView<AccelerationStructureBarrier> p_acceleration_structure_barriers) override final;

	// -----------------------------------------------------------------------
	// FENCES
	// -----------------------------------------------------------------------

	/// Create a fence for GPU→CPU synchronization.
	virtual FenceID fence_create() override final;
	/// Wait for a fence to be signaled. Processes pending WebGPU callbacks.
	virtual Error fence_wait(FenceID p_fence) override final;
	/// Release a fence.
	virtual void fence_free(FenceID p_fence) override final;

	// -----------------------------------------------------------------------
	// SEMAPHORES
	// -----------------------------------------------------------------------

	/// Create a semaphore. Stub — WebGPU has no explicit semaphores.
	virtual SemaphoreID semaphore_create() override final;
	/// Release a semaphore.
	virtual void semaphore_free(SemaphoreID p_semaphore) override final;

	// -----------------------------------------------------------------------
	// COMMAND BUFFERS
	// -----------------------------------------------------------------------

	/// Return the single command queue family (WebGPU has one universal queue).
	virtual CommandQueueFamilyID command_queue_family_get(BitField<CommandQueueFamilyBits> p_cmd_queue_family_bits, RenderingContextDriver::SurfaceID p_surface = 0) override final;
	/// Create a command queue wrapper.
	virtual CommandQueueID command_queue_create(CommandQueueFamilyID p_cmd_queue_family, bool p_identify_as_main_queue = false) override final;
	/// Submit command buffers to the GPU queue and present swap chains.
	virtual Error command_queue_execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_semaphores, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_semaphores, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains) override final;
	/// Release a command queue.
	virtual void command_queue_free(CommandQueueID p_cmd_queue) override final;

	/// Create a command pool. Stub — WebGPU doesn't have command pools.
	virtual CommandPoolID command_pool_create(CommandQueueFamilyID p_cmd_queue_family, CommandBufferType p_cmd_buffer_type) override final;
	/// Reset a command pool. Returns true on success.
	virtual bool command_pool_reset(CommandPoolID p_cmd_pool) override final;
	/// Release a command pool.
	virtual void command_pool_free(CommandPoolID p_cmd_pool) override final;

	/// Create a command buffer with a fresh WGPUCommandEncoder.
	virtual CommandBufferID command_buffer_create(CommandPoolID p_cmd_pool) override final;
	/// Begin recording commands into a command buffer.
	virtual bool command_buffer_begin(CommandBufferID p_cmd_buffer) override final;
	/// Begin a secondary command buffer. Stub — WebGPU has no secondary command buffers.
	virtual bool command_buffer_begin_secondary(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, uint32_t p_subpass, FramebufferID p_framebuffer) override final;
	/// End recording and finish the command encoder.
	virtual void command_buffer_end(CommandBufferID p_cmd_buffer) override final;
	/// Execute secondary command buffers. No-op — WebGPU has no secondary command buffers.
	virtual void command_buffer_execute_secondary(CommandBufferID p_cmd_buffer, VectorView<CommandBufferID> p_secondary_cmd_buffers) override final;

	// -----------------------------------------------------------------------
	// SWAP CHAIN
	// -----------------------------------------------------------------------

	/// Create a swap chain for a browser canvas surface.
	virtual SwapChainID swap_chain_create(RenderingContextDriver::SurfaceID p_surface) override final;
	/// Resize the swap chain. Configures the WGPUSurface with the new dimensions.
	virtual Error swap_chain_resize(CommandQueueID p_cmd_queue, SwapChainID p_swap_chain, uint32_t p_desired_framebuffer_count) override final;
	/// Acquire the next framebuffer from the swap chain for rendering.
	virtual FramebufferID swap_chain_acquire_framebuffer(CommandQueueID p_cmd_queue, SwapChainID p_swap_chain, bool &r_resize_required) override final;
	/// Get the render pass associated with a swap chain.
	virtual RenderPassID swap_chain_get_render_pass(SwapChainID p_swap_chain) override final;
	/// Get the pixel format of a swap chain (always BGRA8Unorm for WebGPU).
	virtual DataFormat swap_chain_get_format(SwapChainID p_swap_chain) override final;
	/// Release a swap chain and its associated render pass.
	virtual void swap_chain_free(SwapChainID p_swap_chain) override final;

	// -----------------------------------------------------------------------
	// FRAMEBUFFER
	// -----------------------------------------------------------------------

	/// Create a framebuffer from a set of texture attachments.
	virtual FramebufferID framebuffer_create(RenderPassID p_render_pass, VectorView<TextureID> p_attachments, uint32_t p_width, uint32_t p_height) override final;
	/// Release a framebuffer.
	virtual void framebuffer_free(FramebufferID p_framebuffer) override final;

	// -----------------------------------------------------------------------
	// SHADER
	// -----------------------------------------------------------------------

	/// Create a shader from a container (SPIR-V). Translates SPIR-V → WGSL via Tint
	/// at runtime, splits combined image-samplers, and remaps push constants to storage buffer.
	virtual ShaderID shader_create_from_container(const Ref<RenderingShaderContainer> &p_shader_container, const Vector<ImmutableSampler> &p_immutable_samplers) override final;
	/// Get the layout hash for a shader (used for pipeline cache keying).
	virtual uint32_t shader_get_layout_hash(ShaderID p_shader) override final;
	/// Release a shader and all its resources (modules, layouts, bind groups).
	virtual void shader_free(ShaderID p_shader) override final;
	/// Release shader modules but keep the shader object alive (for pipeline reuse).
	virtual void shader_destroy_modules(ShaderID p_shader) override final;

	// -----------------------------------------------------------------------
	// UNIFORM SET
	// -----------------------------------------------------------------------

	/// Create a uniform set (bind group) from a list of bound uniforms.
	/// Handles combined image-sampler splitting and BGL compatibility adaptation.
	virtual UniformSetID uniform_set_create(VectorView<BoundUniform> p_uniforms, ShaderID p_shader, uint32_t p_set_index, int p_linear_pool_index) override final;
	/// Release a uniform set and its cached rebind variants.
	virtual void uniform_set_free(UniformSetID p_uniform_set) override final;
	/// Get dynamic offsets for uniform sets. Returns 0 (not used in WebGPU).
	virtual uint32_t uniform_sets_get_dynamic_offsets(VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count) const override final;
	/// Prepare a uniform set for use with a specific shader. Adapts bind group
	/// layout if the target shader uses a different BGL than the source.
	virtual void command_uniform_set_prepare_for_use(CommandBufferID p_cmd_buffer, UniformSetID p_uniform_set, ShaderID p_shader, uint32_t p_set_index) override final;

	// -----------------------------------------------------------------------
	// TRANSFER
	// -----------------------------------------------------------------------

	/// Clear a buffer region to zero. Size is rounded up to 4-byte alignment.
	virtual void command_clear_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, uint64_t p_offset, uint64_t p_size) override final;
	/// Copy data between GPU buffers. Handles shadow buffer fallback for CPU staging.
	virtual void command_copy_buffer(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, BufferID p_dst_buffer, VectorView<BufferCopyRegion> p_regions) override final;
	/// Copy data between GPU textures.
	virtual void command_copy_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<TextureCopyRegion> p_regions) override final;
	/// Resolve MSAA color through a render pass, with a copy fallback for non-renderable destinations.
	virtual void command_resolve_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) override final;
	/// Clear a color texture to a solid color via a single-attachment render pass.
	virtual void command_clear_color_texture(CommandBufferID p_cmd_buffer, TextureID p_texture, TextureLayout p_texture_layout, const Color &p_color, const TextureSubresourceRange &p_subresources) override final;
	/// Clear a depth/stencil texture via a depth-only render pass.
	virtual void command_clear_depth_stencil_texture(CommandBufferID p_cmd_buffer, TextureID p_texture, TextureLayout p_texture_layout, float p_depth, uint8_t p_stencil, const TextureSubresourceRange &p_subresources) override final;
	/// Copy data from a buffer to a texture.
	virtual void command_copy_buffer_to_texture(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<BufferTextureCopyRegion> p_regions) override final;
	/// Direct CPU->GPU multi-layer texture write via wgpuQueueWriteTexture.
	/// Skips the transfer-worker GPU staging buffer and command encoder
	/// entirely — saves N bytes of peak VRAM per Texture2DArray upload (e.g.
	/// ~300 MB for the 1024x1024 RGBA x 75 layers tier on Shiny Gen) and
	/// removes queue serialization tied to the wasted wgpuDeviceCreateBuffer.
	virtual void texture_initialize_direct_layered(TextureID p_dst_texture, TextureLayout p_dst_layout, const uint8_t *p_cpu_data, uint64_t p_total_size, uint32_t p_aligned_bpr, uint32_t p_rows_per_image, uint32_t p_width, uint32_t p_height, uint32_t p_layer_count, uint32_t p_base_layer, uint32_t p_mip_level) override final;
	/// Copy data from a texture to a buffer.
	virtual void command_copy_texture_to_buffer(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, BufferID p_dst_buffer, VectorView<BufferTextureCopyRegion> p_regions) override final;

	// -----------------------------------------------------------------------
	// PIPELINE
	// -----------------------------------------------------------------------

	/// Release a render or compute pipeline.
	virtual void pipeline_free(PipelineID p_pipeline) override final;
	/// Write push constant data. Emulated via a ring buffer (WebGPU has no push constants).
	virtual void command_bind_push_constants(CommandBufferID p_cmd_buffer, ShaderID p_shader, uint32_t p_first_index, VectorView<uint32_t> p_data) override final;
	/// Stub — WebGPU has no pipeline cache. Returns true.
	virtual bool pipeline_cache_create(const Vector<uint8_t> &p_data) override final;
	/// Stub — no pipeline cache to free.
	virtual void pipeline_cache_free() override final;
	/// Stub — returns 0 (no pipeline cache).
	virtual size_t pipeline_cache_query_size() override final;
	/// Stub — returns empty vector (no pipeline cache).
	virtual Vector<uint8_t> pipeline_cache_serialize() override final;

	// -----------------------------------------------------------------------
	// RENDERING
	// -----------------------------------------------------------------------

	/// Create a render pass descriptor from attachments and subpasses.
	virtual RenderPassID render_pass_create(VectorView<Attachment> p_attachments, VectorView<Subpass> p_subpasses, VectorView<SubpassDependency> p_subpass_dependencies, uint32_t p_view_count, AttachmentReference p_fragment_density_map_attachment) override final;
	/// Release a render pass descriptor.
	virtual void render_pass_free(RenderPassID p_render_pass) override final;

	/// Begin a render pass. Handles automatic command encoder splitting when
	/// a texture transitions from write (previous pass) to read (this pass).
	virtual void command_begin_render_pass(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, FramebufferID p_framebuffer, CommandBufferType p_cmd_buffer_type, const Rect2i &p_rect, VectorView<RenderPassClearValue> p_clear_values) override final;
	/// End the current render pass.
	virtual void command_end_render_pass(CommandBufferID p_cmd_buffer) override final;
	/// Transition to the next subpass within a render pass.
	virtual void command_next_render_subpass(CommandBufferID p_cmd_buffer, CommandBufferType p_cmd_buffer_type) override final;
	/// Set viewport(s) for the current render pass.
	virtual void command_render_set_viewport(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_viewports) override final;
	/// Set scissor rect(s) for the current render pass.
	virtual void command_render_set_scissor(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_scissors) override final;
	/// Clear rects of the current subpass's attachments with a clear draw (WebGPU has no mid-pass clear).
	virtual void command_render_clear_attachments(CommandBufferID p_cmd_buffer, VectorView<AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) override final;

	/// Bind a render pipeline for subsequent draw calls.
	virtual void command_bind_render_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) override final;
	/// Bind uniform sets (bind groups) for a render pipeline. Handles BGL
	/// compatibility adaptation and push constant group merging.
	virtual void command_bind_render_uniform_sets(CommandBufferID p_cmd_buffer, VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) override final;

	/// Issue a non-indexed draw call.
	virtual void command_render_draw(CommandBufferID p_cmd_buffer, uint32_t p_vertex_count, uint32_t p_instance_count, uint32_t p_base_vertex, uint32_t p_first_instance) override final;
	/// Issue an indexed draw call.
	virtual void command_render_draw_indexed(CommandBufferID p_cmd_buffer, uint32_t p_index_count, uint32_t p_instance_count, uint32_t p_first_index, int32_t p_vertex_offset, uint32_t p_first_instance) override final;
	/// Issue an indirect indexed draw call.
	virtual void command_render_draw_indexed_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) override final;
	/// Issue an indirect indexed draw call with draw count from a buffer.
	/// Falls back to max_draw_count (async count readback not yet implemented).
	virtual void command_render_draw_indexed_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) override final;
	/// Issue an indirect non-indexed draw call.
	virtual void command_render_draw_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) override final;
	/// Issue an indirect non-indexed draw call with draw count from a buffer.
	virtual void command_render_draw_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) override final;

	/// Bind vertex buffers for the current render pass.
	virtual void command_render_bind_vertex_buffers(CommandBufferID p_cmd_buffer, uint32_t p_binding_count, const BufferID *p_buffers, const uint64_t *p_offsets, uint64_t p_dynamic_offsets) override final;
	/// Bind an index buffer for the current render pass.
	virtual void command_render_bind_index_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, IndexBufferFormat p_format, uint64_t p_offset) override final;

	/// Set blend constants for the current render pass.
	virtual void command_render_set_blend_constants(CommandBufferID p_cmd_buffer, const Color &p_constants) override final;
	/// Set line width. Stub — WebGPU only supports 1.0 line width.
	virtual void command_render_set_line_width(CommandBufferID p_cmd_buffer, float p_width) override final;

	/// Create a render pipeline. Handles SPIR-V → WGSL translation for specialization
	/// constants and alpha-strip for BGRA8Unorm swap chain compatibility.
	virtual PipelineID render_pipeline_create(
			ShaderID p_shader,
			VertexFormatID p_vertex_format,
			RenderPrimitive p_render_primitive,
			PipelineRasterizationState p_rasterization_state,
			PipelineMultisampleState p_multisample_state,
			PipelineDepthStencilState p_depth_stencil_state,
			PipelineColorBlendState p_blend_state,
			VectorView<int32_t> p_color_attachments,
			BitField<PipelineDynamicStateFlags> p_dynamic_state,
			RenderPassID p_render_pass,
			uint32_t p_render_subpass,
			VectorView<PipelineSpecializationConstant> p_specialization_constants) override final;

	// -----------------------------------------------------------------------
	// COMPUTE
	// -----------------------------------------------------------------------

	/// Bind a compute pipeline. Ends any active render pass and begins a compute pass.
	virtual void command_bind_compute_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) override final;
	/// Bind uniform sets (bind groups) for a compute pipeline.
	virtual void command_bind_compute_uniform_sets(CommandBufferID p_cmd_buffer, VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) override final;
	/// Dispatch compute workgroups.
	virtual void command_compute_dispatch(CommandBufferID p_cmd_buffer, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) override final;
	/// Dispatch compute workgroups with parameters from a GPU buffer.
	virtual void command_compute_dispatch_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset) override final;
	/// Create a compute pipeline with optional specialization constants.
	virtual PipelineID compute_pipeline_create(ShaderID p_shader, VectorView<PipelineSpecializationConstant> p_specialization_constants) override final;

	// -----------------------------------------------------------------------
	// QUERIES
	// -----------------------------------------------------------------------

	/// Timestamps are unsupported (as on Metal): dummy pool, results are zeros.
	virtual QueryPoolID timestamp_query_pool_create(uint32_t p_query_count) override final;
	virtual void timestamp_query_pool_free(QueryPoolID p_pool_id) override final;
	virtual void timestamp_query_pool_get_results(QueryPoolID p_pool_id, uint32_t p_query_count, uint64_t *r_results) override final;
	virtual uint64_t timestamp_query_result_to_time(uint64_t p_result) override final;
	virtual void command_timestamp_query_pool_reset(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_query_count) override final;
	virtual void command_timestamp_write(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_index) override final;

	// -----------------------------------------------------------------------
	// LABELS & DEBUG
	// -----------------------------------------------------------------------

	/// Begin a debug label group. No-op — WebGPU has no debug label API.
	virtual void command_begin_label(CommandBufferID p_cmd_buffer, const char *p_label_name, const Color &p_color) override final;
	/// End a debug label group. No-op.
	virtual void command_end_label(CommandBufferID p_cmd_buffer) override final;
	/// Insert a breadcrumb for GPU crash debugging. No-op.
	virtual void command_insert_breadcrumb(CommandBufferID p_cmd_buffer, uint32_t p_data) override final;

	// -----------------------------------------------------------------------
	// SUBMISSION
	// -----------------------------------------------------------------------

	/// Begin a new frame segment. Resets the push constant ring (and logs perf counters when verbose).
	virtual void begin_segment(uint32_t p_frame_index, uint32_t p_frames_drawn) override final;
	/// End a frame segment. Flushes accumulated push constant data to the GPU.
	virtual void end_segment() override final;

	// Godot 4.7 interfaces for features unavailable on WebGPU.
	virtual ColorSpace swap_chain_get_color_space(SwapChainID p_swap_chain) override final { return COLOR_SPACE_REC709_NONLINEAR_SRGB; }
	virtual bool swap_chain_get_hdr_output_supported(SwapChainID p_swap_chain) override final { return false; }
	virtual AccelerationStructureID blas_create(VectorView<AccelerationStructureGeometry> p_geometries, BitField<AccelerationStructureFlagBits> p_flags) override final { ERR_FAIL_V_MSG(AccelerationStructureID(), "WebGPU does not support ray tracing."); }
	virtual AccelerationStructureID tlas_create(uint32_t p_max_instance_count, BitField<AccelerationStructureFlagBits> p_flags) override final { ERR_FAIL_V_MSG(AccelerationStructureID(), "WebGPU does not support ray tracing."); }
	virtual void acceleration_structure_instance_write(uint8_t *r_driver_instance, const AccelerationStructureInstance &p_instance) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }
	virtual void acceleration_structure_free(AccelerationStructureID p_acceleration_structure) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }
	virtual uint32_t acceleration_structure_get_scratch_size_bytes(AccelerationStructureID p_acceleration_structure) override final { ERR_FAIL_V_MSG(uint32_t(), "WebGPU does not support ray tracing."); }
	virtual RaytracingPipelineID raytracing_pipeline_create(VectorView<PipelineShader> p_shaders, VectorView<uint32_t> p_raygen_shader_indices, VectorView<uint32_t> p_miss_shader_indices, VectorView<HitGroup> p_hit_groups, uint32_t p_max_trace_recursion_depth, ShaderID p_layout_defining_shader) override final { ERR_FAIL_V_MSG(RaytracingPipelineID(), "WebGPU does not support ray tracing."); }
	virtual void raytracing_pipeline_free(RaytracingPipelineID p_pipeline) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }
	virtual bool raytracing_pipeline_get_shader_group_handles(RaytracingPipelineID p_pipeline, uint32_t p_group_index_offset, VectorView<uint32_t> p_group_indices, uint8_t *r_data, uint32_t p_data_stride_bytes) override final { ERR_FAIL_V_MSG(bool(), "WebGPU does not support ray tracing."); }
	virtual void command_build_blas(CommandBufferID p_cmd_buffer, AccelerationStructureID p_acceleration_structure, BufferID p_scratch_buffer) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }
	virtual void command_build_tlas(CommandBufferID p_cmd_buffer, AccelerationStructureID p_acceleration_structure, BufferID p_scratch_buffer, BufferID p_instance_buffer, uint32_t p_instance_offset, uint32_t p_instance_count) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }
	virtual void command_bind_raytracing_pipeline(CommandBufferID p_cmd_buffer, RaytracingPipelineID p_pipeline) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }
	virtual void command_bind_raytracing_uniform_set(CommandBufferID p_cmd_buffer, UniformSetID p_uniform_set, ShaderID p_shader, uint32_t p_set_index) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }
	virtual void command_trace_rays(CommandBufferID p_cmd_buffer, const ShaderBindingTable &p_raygen_sbt, const ShaderBindingTable &p_miss_sbt, const ShaderBindingTable &p_hit_sbt, uint32_t p_width, uint32_t p_height, uint32_t p_depth) override final { ERR_FAIL_MSG("WebGPU does not support ray tracing."); }

	// -----------------------------------------------------------------------
	// MISC
	// -----------------------------------------------------------------------

	/// Set a debug name on a GPU resource. Stub — resource labeling not yet implemented.
	virtual void set_object_name(ObjectType p_type, ID p_driver_id, const String &p_name) override final;
	/// Get the underlying WGPU handle (WGPUDevice, WGPUTexture, ...) for a resource.
	virtual uint64_t get_resource_native_handle(DriverResource p_type, ID p_driver_id) override final;
	/// Get total GPU memory used. Returns 0 (not tracked in WebGPU).
	virtual uint64_t get_total_memory_used() override final;
	/// Get lazily-allocated GPU memory. Returns 0 (not applicable to WebGPU).
	virtual uint64_t get_lazily_memory_used() override final;
	/// Query a device limit (max texture size, max uniform buffer size, etc.).
	virtual uint64_t limit_get(Limit p_limit) override final;
	/// Query an API trait (e.g., texture data row pitch step).
	virtual uint64_t api_trait_get(ApiTrait p_trait) override final;
	/// Check if a feature is supported (multiview, VRS, etc.).
	virtual bool has_feature(Features p_feature) override final;
	virtual DriverWorkarounds get_driver_workarounds() const override final;
	/// Get multiview capabilities. Not supported in WebGPU.
	virtual const MultiviewCapabilities &get_multiview_capabilities() override final;
	/// Get fragment shading rate capabilities. Not supported in WebGPU.
	virtual const FragmentShadingRateCapabilities &get_fragment_shading_rate_capabilities() override final;
	/// Get fragment density map capabilities. Not supported in WebGPU.
	virtual const FragmentDensityMapCapabilities &get_fragment_density_map_capabilities() override final;
	/// Return "WebGPU" as the API name.
	virtual String get_api_name() const override final;
	/// Return "1.0" as the API version.
	virtual String get_api_version() const override final;
	/// Return a unique identifier for pipeline cache (empty — no cache in WebGPU).
	virtual String get_pipeline_cache_uuid() const override final;
	/// Get device capabilities (limits, features, format support).
	virtual const Capabilities &get_capabilities() const override final;
	/// Get the shader container format (SPIR-V with WebGPU-specific metadata).
	virtual const RenderingShaderContainerFormat &get_shader_container_format() const override final;
};

#endif // WEBGPU_ENABLED
