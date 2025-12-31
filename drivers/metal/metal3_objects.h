/**************************************************************************/
/*  metal3_objects.h                                                      */
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

/**************************************************************************/
/*                                                                        */
/* Portions of this code were derived from MoltenVK.                      */
/*                                                                        */
/* Copyright (c) 2015-2023 The Brenwill Workshop Ltd.                     */
/* (http://www.brenwill.com)                                              */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#import "metal_objects_shared.h"

#include "servers/rendering/rendering_device_driver.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <simd/simd.h>
#import <zlib.h>
#import <initializer_list>
#import <optional>

namespace MTL3 {

enum class MDCommandBufferStateType {
	None,
	Render,
	Compute,
	Blit,
};

// These types are defined in the global namespace (metal_objects_shared.h / rendering_device_driver_metal.h)
using ::MDAttachment;
using ::MDAttachmentType;
using ::MDCommandBufferBase;
using ::MDFrameBuffer;
using ::MDRenderPass;
using ::MDRingBuffer;
using ::MDSubpass;

using ::DynamicOffsetLayout;
using ::MDComputePipeline;
using ::MDComputeShader;
using ::MDLibrary;
using ::MDPipeline;
using ::MDPipelineType;
using ::MDRenderPipeline;
using ::MDRenderShader;
using ::MDShader;
using ::MDUniformSet;
using ::ShaderCacheEntry;
using ::ShaderLoadStrategy;
using ::UniformInfo;
using ::UniformSet;

using RDM = ::RenderingDeviceDriverMetal;

struct ResourceUsageEntry {
	StageResourceUsage usage = ResourceUnused;
	uint32_t unused = 0;

	ResourceUsageEntry() {}
	ResourceUsageEntry(StageResourceUsage p_usage) :
			usage(p_usage) {}
};

} // namespace MTL3

template <>
struct is_zero_constructible<MTL3::ResourceUsageEntry> : std::true_type {};

namespace MTL3 {

/*! Track the cumulative usage for a resource during a render or compute pass */
typedef HashMap<MTLResourceUnsafe, ResourceUsageEntry> ResourceToStageUsage;

/*! Track resource and ensure they are resident prior to dispatch or draw commands.
 *
 * The primary purpose of this data structure is to track all the resources that must be made resident prior
 * to issuing the next dispatch or draw command. It aggregates all resources used from argument buffers.
 *
 * As an optimization, this data structure also tracks previous usage for resources, so that
 * it may avoid binding them again in later commands if the resource is already resident and its usage flagged.
 */
struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) ResourceTracker {
	// A constant specifying how many iterations a resource can remain in
	// the _previous HashSet before it will be removed permanently.
	//
	// Keeping them in the _previous HashMap reduces churn if resources are regularly
	// bound. 256 is arbitrary, but if an object remains unused for 256 encoders,
	// it will be released.
	static constexpr uint32_t RESOURCE_UNUSED_CLEANUP_COUNT = 256;

	// Used as a scratch buffer to periodically clean up resources from _previous.
	ResourceVector _scratch;
	// Tracks all resources and their prior usage for the duration of the encoder.
	ResourceToStageUsage _previous;
	// Tracks resources for the current command that must be made resident
	ResourceUsageMap _current;

	void merge_from(const ResourceUsageMap &p_from);
	void encode(id<MTLRenderCommandEncoder> __unsafe_unretained p_enc);
	void encode(id<MTLComputeCommandEncoder> __unsafe_unretained p_enc);
	void reset();
};

struct BindingCache {
	struct BufferBinding {
		id<MTLBuffer> __unsafe_unretained buffer = nil;
		NSUInteger offset = 0;

		bool operator!=(const BufferBinding &p_other) const {
			return buffer != p_other.buffer || offset != p_other.offset;
		}
	};

	LocalVector<id<MTLTexture> __unsafe_unretained> textures;
	LocalVector<id<MTLSamplerState> __unsafe_unretained> samplers;
	LocalVector<BufferBinding> buffers;

	_FORCE_INLINE_ void clear() {
		textures.clear();
		samplers.clear();
		buffers.clear();
	}

private:
	template <typename T>
	_FORCE_INLINE_ void ensure_size(LocalVector<T> &p_vec, uint32_t p_required) {
		if (p_vec.size() < p_required) {
			p_vec.resize_initialized(p_required);
		}
	}

public:
	_FORCE_INLINE_ bool update(NSRange p_range, id<MTLTexture> __unsafe_unretained const *p_values) {
		if (p_range.length == 0) {
			return false;
		}
		uint32_t required = (uint32_t)(p_range.location + p_range.length);
		ensure_size(textures, required);
		bool changed = false;
		for (NSUInteger i = 0; i < p_range.length; ++i) {
			uint32_t slot = (uint32_t)(p_range.location + i);
			id<MTLTexture> value = p_values[i];
			if (textures[slot] != value) {
				textures[slot] = value;
				changed = true;
			}
		}
		return changed;
	}

	_FORCE_INLINE_ bool update(NSRange p_range, id<MTLSamplerState> __unsafe_unretained const *p_values) {
		if (p_range.length == 0) {
			return false;
		}
		uint32_t required = (uint32_t)(p_range.location + p_range.length);
		ensure_size(samplers, required);
		bool changed = false;
		for (NSUInteger i = 0; i < p_range.length; ++i) {
			uint32_t slot = (uint32_t)(p_range.location + i);
			id<MTLSamplerState> __unsafe_unretained value = p_values[i];
			if (samplers[slot] != value) {
				samplers[slot] = value;
				changed = true;
			}
		}
		return changed;
	}

	_FORCE_INLINE_ bool update(NSRange p_range, id<MTLBuffer> __unsafe_unretained const *p_values, const NSUInteger *p_offsets) {
		if (p_range.length == 0) {
			return false;
		}
		uint32_t required = (uint32_t)(p_range.location + p_range.length);
		ensure_size(buffers, required);
		BufferBinding *buffers_ptr = buffers.ptr() + p_range.location;
		bool changed = false;
		for (NSUInteger i = 0; i < p_range.length; ++i) {
			BufferBinding &binding = *buffers_ptr;
			BufferBinding new_binding = {
				.buffer = p_values[i],
				.offset = p_offsets[i],
			};
			if (binding != new_binding) {
				binding = new_binding;
				changed = true;
			}
			++buffers_ptr;
		}
		return changed;
	}

	_FORCE_INLINE_ bool update(id<MTLBuffer> __unsafe_unretained p_buffer, NSUInteger p_offset, uint32_t p_index) {
		uint32_t required = p_index + 1;
		ensure_size(buffers, required);
		BufferBinding &binding = buffers.ptr()[p_index];
		BufferBinding new_binding = {
			.buffer = p_buffer,
			.offset = p_offset,
		};
		if (binding != new_binding) {
			binding = new_binding;
			return true;
		}
		return false;
	}
};

// A type used to encode resources directly to a MTLCommandEncoder
struct DirectEncoder {
	id<MTLCommandEncoder> __unsafe_unretained encoder;
	BindingCache &cache;
	enum Mode {
		RENDER,
		COMPUTE
	};
	Mode mode;

	void set(id<MTLBuffer> __unsafe_unretained *p_buffers, const NSUInteger *p_offsets, NSRange p_range);
	void set(id<MTLBuffer> __unsafe_unretained p_buffer, const NSUInteger p_offset, uint32_t p_index);
	void set(id<MTLTexture> __unsafe_unretained *p_textures, NSRange p_range);
	void set(id<MTLSamplerState> __unsafe_unretained *p_samplers, NSRange p_range);

	DirectEncoder(id<MTLCommandEncoder> __unsafe_unretained p_encoder, BindingCache &p_cache) :
			encoder(p_encoder), cache(p_cache) {
		if ([p_encoder conformsToProtocol:@protocol(MTLRenderCommandEncoder)]) {
			mode = RENDER;
		} else {
			mode = COMPUTE;
		}
	}
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDCommandBuffer : public MDCommandBufferBase {
	friend class MDUniformSet;

private:
#pragma mark - Common State

	// From RenderingDevice
	static constexpr uint32_t MAX_PUSH_CONSTANT_SIZE = 128;

	uint8_t push_constant_data[MAX_PUSH_CONSTANT_SIZE];
	uint32_t push_constant_data_len = 0;
	uint32_t push_constant_binding = UINT32_MAX;

	BindingCache binding_cache;

#pragma mark - Argument Buffer Ring Allocator

	using Alloc = MDRingBuffer::Allocation;

	// Used for argument buffers that contain dynamic uniforms.
	MDRingBuffer _scratch;

	/// Allocates from the ring buffer for dynamic argument buffers.
	Alloc allocate_arg_buffer(uint32_t p_size);

	struct {
		id rs = nil; // id<MTLResidencySet>, but untyped for API availability.
	} _frame_state;

#pragma mark - Synchronization

	enum {
		STAGE_RENDER,
		STAGE_COMPUTE,
		STAGE_BLIT,
		STAGE_MAX,
	};
	bool use_barriers = false;
	MTLStages pending_after_stages[STAGE_MAX] = { 0, 0, 0 };
	MTLStages pending_before_queue_stages[STAGE_MAX] = { 0, 0, 0 };
	void _encode_barrier(id<MTLCommandEncoder> p_enc);

	void reset();

	::RenderingDeviceDriverMetal *device_driver = nullptr;
	id<MTLCommandQueue> queue = nil;
	id<MTLCommandBuffer> commandBuffer = nil;
	bool state_begin = false;

	_FORCE_INLINE_ id<MTLCommandBuffer> command_buffer() {
		DEV_ASSERT(state_begin);
		if (commandBuffer == nil) {
			commandBuffer = queue.commandBuffer;
			if (use_barriers) {
				GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability");
				[commandBuffer useResidencySet:_frame_state.rs];
				GODOT_CLANG_WARNING_POP;
			}
		}
		return commandBuffer;
	}

	void _end_compute_dispatch();
	void _end_blit();
	id<MTLBlitCommandEncoder> _ensure_blit_encoder();

	enum class CopySource {
		Buffer,
		Texture,
	};
	void _copy_texture_buffer(CopySource p_source,
			RDD::TextureID p_texture,
			RDD::BufferID p_buffer,
			VectorView<RDD::BufferTextureCopyRegion> p_regions);

#pragma mark - Render

	void _render_set_dirty_state();
	void _render_bind_uniform_sets();
	void _bind_uniforms_argument_buffers(MDUniformSet *p_set, MDShader *p_shader, uint32_t p_set_index, uint32_t p_dynamic_offsets);
	void _bind_uniforms_direct(MDUniformSet *p_set, MDShader *p_shader, DirectEncoder p_enc, uint32_t p_set_index, uint32_t p_dynamic_offsets);

	void _populate_vertices(simd::float4 *p_vertices, Size2i p_fb_size, VectorView<Rect2i> p_rects);
	uint32_t _populate_vertices(simd::float4 *p_vertices, uint32_t p_index, Rect2i const &p_rect, Size2i p_fb_size);
	void _end_render_pass();
	void _render_clear_render_area();

#pragma mark - Compute

	void _compute_set_dirty_state();
	void _compute_bind_uniform_sets();
	void _bind_uniforms_argument_buffers_compute(MDUniformSet *p_set, MDShader *p_shader, uint32_t p_set_index, uint32_t p_dynamic_offsets);

public:
	MDCommandBufferStateType type = MDCommandBufferStateType::None;

	struct RenderState {
		MDRenderPass *pass = nullptr;
		MDFrameBuffer *frameBuffer = nullptr;
		MDRenderPipeline *pipeline = nullptr;
		LocalVector<RDD::RenderPassClearValue> clear_values;
		LocalVector<MTLViewport> viewports;
		LocalVector<MTLScissorRect> scissors;
		std::optional<Color> blend_constants;
		uint32_t current_subpass = UINT32_MAX;
		Rect2i render_area = {};
		bool is_rendering_entire_area = false;
		MTLRenderPassDescriptor *desc = nil;
		id<MTLRenderCommandEncoder> encoder = nil;
		id<MTLBuffer> __unsafe_unretained index_buffer = nil; // Buffer is owned by RDD.
		MTLIndexType index_type = MTLIndexTypeUInt16;
		uint32_t index_offset = 0;
		LocalVector<id<MTLBuffer> __unsafe_unretained> vertex_buffers;
		LocalVector<NSUInteger> vertex_offsets;
		ResourceTracker resource_tracker;
		// clang-format off
		enum DirtyFlag: uint16_t {
			DIRTY_NONE     = 0,
			DIRTY_PIPELINE = 1 << 0, //! pipeline state
			DIRTY_UNIFORMS = 1 << 1, //! uniform sets
			DIRTY_PUSH     = 1 << 2, //! push constants
			DIRTY_DEPTH    = 1 << 3, //! depth / stencil state
			DIRTY_VERTEX   = 1 << 4, //! vertex buffers
			DIRTY_VIEWPORT = 1 << 5, //! viewport rectangles
			DIRTY_SCISSOR  = 1 << 6, //! scissor rectangles
			DIRTY_BLEND    = 1 << 7, //! blend state
			DIRTY_RASTER   = 1 << 8, //! encoder state like cull mode
			DIRTY_ALL      = (1 << 9) - 1,
		};
		// clang-format on
		BitField<DirtyFlag> dirty = DIRTY_NONE;

		LocalVector<MDUniformSet *> uniform_sets;
		uint32_t dynamic_offsets = 0;
		// Bit mask of the uniform sets that are dirty, to prevent redundant binding.
		uint64_t uniform_set_mask = 0;

		_FORCE_INLINE_ void reset();
		void end_encoding();

		_ALWAYS_INLINE_ const MDSubpass &get_subpass() const {
			DEV_ASSERT(pass != nullptr);
			return pass->subpasses[current_subpass];
		}

		_FORCE_INLINE_ void mark_viewport_dirty() {
			if (viewports.is_empty()) {
				return;
			}
			dirty.set_flag(DirtyFlag::DIRTY_VIEWPORT);
		}

		_FORCE_INLINE_ void mark_scissors_dirty() {
			if (scissors.is_empty()) {
				return;
			}
			dirty.set_flag(DirtyFlag::DIRTY_SCISSOR);
		}

		_FORCE_INLINE_ void mark_vertex_dirty() {
			if (vertex_buffers.is_empty()) {
				return;
			}
			dirty.set_flag(DirtyFlag::DIRTY_VERTEX);
		}

		_FORCE_INLINE_ void mark_uniforms_dirty(std::initializer_list<uint32_t> l) {
			if (uniform_sets.is_empty()) {
				return;
			}
			for (uint32_t i : l) {
				if (i < uniform_sets.size() && uniform_sets[i] != nullptr) {
					uniform_set_mask |= 1 << i;
				}
			}
			dirty.set_flag(DirtyFlag::DIRTY_UNIFORMS);
		}

		_FORCE_INLINE_ void mark_uniforms_dirty(void) {
			if (uniform_sets.is_empty()) {
				return;
			}
			for (uint32_t i = 0; i < uniform_sets.size(); i++) {
				if (uniform_sets[i] != nullptr) {
					uniform_set_mask |= 1 << i;
				}
			}
			dirty.set_flag(DirtyFlag::DIRTY_UNIFORMS);
		}

		_FORCE_INLINE_ void mark_blend_dirty() {
			if (!blend_constants.has_value()) {
				return;
			}
			dirty.set_flag(DirtyFlag::DIRTY_BLEND);
		}

		MTLScissorRect clip_to_render_area(MTLScissorRect p_rect) const {
			uint32_t raLeft = render_area.position.x;
			uint32_t raRight = raLeft + render_area.size.width;
			uint32_t raBottom = render_area.position.y;
			uint32_t raTop = raBottom + render_area.size.height;

			p_rect.x = CLAMP(p_rect.x, raLeft, MAX(raRight - 1, raLeft));
			p_rect.y = CLAMP(p_rect.y, raBottom, MAX(raTop - 1, raBottom));
			p_rect.width = MIN(p_rect.width, raRight - p_rect.x);
			p_rect.height = MIN(p_rect.height, raTop - p_rect.y);

			return p_rect;
		}

		Rect2i clip_to_render_area(Rect2i p_rect) const {
			int32_t raLeft = render_area.position.x;
			int32_t raRight = raLeft + render_area.size.width;
			int32_t raBottom = render_area.position.y;
			int32_t raTop = raBottom + render_area.size.height;

			p_rect.position.x = CLAMP(p_rect.position.x, raLeft, MAX(raRight - 1, raLeft));
			p_rect.position.y = CLAMP(p_rect.position.y, raBottom, MAX(raTop - 1, raBottom));
			p_rect.size.width = MIN(p_rect.size.width, raRight - p_rect.position.x);
			p_rect.size.height = MIN(p_rect.size.height, raTop - p_rect.position.y);

			return p_rect;
		}

	} render;

	// State specific for a compute pass.
	struct ComputeState {
		MDComputePipeline *pipeline = nullptr;
		id<MTLComputeCommandEncoder> encoder = nil;
		ResourceTracker resource_tracker;
		// clang-format off
		enum DirtyFlag: uint16_t {
			DIRTY_NONE     = 0,
			DIRTY_PIPELINE = 1 << 0, //! pipeline state
			DIRTY_UNIFORMS = 1 << 1, //! uniform sets
			DIRTY_PUSH     = 1 << 2, //! push constants
			DIRTY_ALL      = (1 << 3) - 1,
		};
		// clang-format on
		BitField<DirtyFlag> dirty = DIRTY_NONE;

		LocalVector<MDUniformSet *> uniform_sets;
		uint32_t dynamic_offsets = 0;
		// Bit mask of the uniform sets that are dirty, to prevent redundant binding.
		uint64_t uniform_set_mask = 0;

		_FORCE_INLINE_ void reset();
		void end_encoding();

		_FORCE_INLINE_ void mark_uniforms_dirty(void) {
			if (uniform_sets.is_empty()) {
				return;
			}
			for (uint32_t i = 0; i < uniform_sets.size(); i++) {
				if (uniform_sets[i] != nullptr) {
					uniform_set_mask |= 1 << i;
				}
			}
			dirty.set_flag(DirtyFlag::DIRTY_UNIFORMS);
		}
	} compute;

	// State specific to a blit pass.
	struct {
		id<MTLBlitCommandEncoder> encoder = nil;
		_FORCE_INLINE_ void reset() {
			encoder = nil;
		}
	} blit;

	_FORCE_INLINE_ id<MTLCommandBuffer> get_command_buffer() const {
		return commandBuffer;
	}

	void begin() override;
	void commit() override;
	void end() override;

	void bind_pipeline(RDD::PipelineID p_pipeline) override;
	void encode_push_constant_data(RDD::ShaderID p_shader, VectorView<uint32_t> p_data) override;

#pragma mark - Render Commands

	void render_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) override;
	void render_clear_attachments(VectorView<RDD::AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) override;
	void render_set_viewport(VectorView<Rect2i> p_viewports) override;
	void render_set_scissor(VectorView<Rect2i> p_scissors) override;
	void render_set_blend_constants(const Color &p_constants) override;
	void render_begin_pass(RDD::RenderPassID p_render_pass,
			RDD::FramebufferID p_frameBuffer,
			RDD::CommandBufferType p_cmd_buffer_type,
			const Rect2i &p_rect,
			VectorView<RDD::RenderPassClearValue> p_clear_values) override;
	void render_next_subpass() override;
	void render_draw(uint32_t p_vertex_count,
			uint32_t p_instance_count,
			uint32_t p_base_vertex,
			uint32_t p_first_instance) override;
	void render_bind_vertex_buffers(uint32_t p_binding_count, const RDD::BufferID *p_buffers, const uint64_t *p_offsets, uint64_t p_dynamic_offsets) override;
	void render_bind_index_buffer(RDD::BufferID p_buffer, RDD::IndexBufferFormat p_format, uint64_t p_offset) override;

	void render_draw_indexed(uint32_t p_index_count,
			uint32_t p_instance_count,
			uint32_t p_first_index,
			int32_t p_vertex_offset,
			uint32_t p_first_instance) override;

	void render_draw_indexed_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) override;
	void render_draw_indexed_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) override;
	void render_draw_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) override;
	void render_draw_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) override;

	void render_end_pass() override;

#pragma mark - Compute Commands

	void compute_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) override;
	void compute_dispatch(uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) override;
	void compute_dispatch_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset) override;

#pragma mark - Transfer

private:
	id<MTLRenderCommandEncoder> get_new_render_encoder_with_descriptor(MTLRenderPassDescriptor *p_desc);

public:
	void resolve_texture(RDD::TextureID p_src_texture, RDD::TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, RDD::TextureID p_dst_texture, RDD::TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) override;
	void clear_color_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, const Color &p_color, const RDD::TextureSubresourceRange &p_subresources) override;
	void clear_buffer(RDD::BufferID p_buffer, uint64_t p_offset, uint64_t p_size) override;
	void copy_buffer(RDD::BufferID p_src_buffer, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferCopyRegion> p_regions) override;
	void copy_texture(RDD::TextureID p_src_texture, RDD::TextureID p_dst_texture, VectorView<RDD::TextureCopyRegion> p_regions) override;
	void copy_buffer_to_texture(RDD::BufferID p_src_buffer, RDD::TextureID p_dst_texture, VectorView<RDD::BufferTextureCopyRegion> p_regions) override;
	void copy_texture_to_buffer(RDD::TextureID p_src_texture, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferTextureCopyRegion> p_regions) override;

#pragma mark - Synchronization

	void pipeline_barrier(BitField<RDD::PipelineStageBits> p_src_stages,
			BitField<RDD::PipelineStageBits> p_dst_stages,
			VectorView<RDD::MemoryAccessBarrier> p_memory_barriers,
			VectorView<RDD::BufferBarrier> p_buffer_barriers,
			VectorView<RDD::TextureBarrier> p_texture_barriers) override;

#pragma mark - Debugging

	void begin_label(const char *p_label_name, const Color &p_color) override;
	void end_label() override;

	MDCommandBuffer(id<MTLCommandQueue> p_queue, ::RenderingDeviceDriverMetal *p_device_driver);
	MDCommandBuffer() = default;
};

} // namespace MTL3

namespace rid {
#define MAKE_ID(FROM, TO)                \
	_FORCE_INLINE_ TO make(FROM p_obj) { \
		return TO(owned(p_obj));         \
	}

MAKE_ID(id<MTLCommandQueue>, RDD::CommandPoolID)

#undef MAKE_ID
} //namespace rid
