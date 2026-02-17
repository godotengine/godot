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

#include "metal_objects_shared.h"

#include "servers/rendering/rendering_device_driver.h"

#include <Metal/Metal.hpp>

#include <initializer_list>
#include <optional>

namespace MTL3 {

// These types are defined in the global namespace (metal_objects_shared.h / rendering_device_driver_metal.h)
using ::MDAttachment;
using ::MDAttachmentType;
using ::MDCommandBufferBase;
using ::MDCommandBufferStateType;
using ::MDFrameBuffer;
using ::MDRenderPass;
using ::MDRingBuffer;
using ::MDSubpass;
using ::RenderStateBase;

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
typedef HashMap<MTL::Resource *, ResourceUsageEntry> ResourceToStageUsage;

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

	void merge_from(const ::ResourceUsageMap &p_from);
	void encode(MTL::RenderCommandEncoder *p_enc);
	void encode(MTL::ComputeCommandEncoder *p_enc);
	void reset();
};

struct BindingCache {
	struct BufferBinding {
		MTL::Buffer *buffer = nullptr;
		NS::UInteger offset = 0;

		bool operator!=(const BufferBinding &p_other) const {
			return buffer != p_other.buffer || offset != p_other.offset;
		}
	};

	LocalVector<MTL::Texture *> textures;
	LocalVector<MTL::SamplerState *> samplers;
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
	_FORCE_INLINE_ bool update(NS::Range p_range, MTL::Texture *const *p_values) {
		if (p_range.length == 0) {
			return false;
		}
		uint32_t required = (uint32_t)(p_range.location + p_range.length);
		ensure_size(textures, required);
		bool changed = false;
		for (NS::UInteger i = 0; i < p_range.length; ++i) {
			uint32_t slot = (uint32_t)(p_range.location + i);
			MTL::Texture *value = p_values[i];
			if (textures[slot] != value) {
				textures[slot] = value;
				changed = true;
			}
		}
		return changed;
	}

	_FORCE_INLINE_ bool update(NS::Range p_range, MTL::SamplerState *const *p_values) {
		if (p_range.length == 0) {
			return false;
		}
		uint32_t required = (uint32_t)(p_range.location + p_range.length);
		ensure_size(samplers, required);
		bool changed = false;
		for (NS::UInteger i = 0; i < p_range.length; ++i) {
			uint32_t slot = (uint32_t)(p_range.location + i);
			MTL::SamplerState *value = p_values[i];
			if (samplers[slot] != value) {
				samplers[slot] = value;
				changed = true;
			}
		}
		return changed;
	}

	_FORCE_INLINE_ bool update(NS::Range p_range, MTL::Buffer *const *p_values, const NS::UInteger *p_offsets) {
		if (p_range.length == 0) {
			return false;
		}
		uint32_t required = (uint32_t)(p_range.location + p_range.length);
		ensure_size(buffers, required);
		BufferBinding *buffers_ptr = buffers.ptr() + p_range.location;
		bool changed = false;
		for (NS::UInteger i = 0; i < p_range.length; ++i) {
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

	_FORCE_INLINE_ bool update(MTL::Buffer *p_buffer, NS::UInteger p_offset, uint32_t p_index) {
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
	MTL::CommandEncoder *encoder;
	BindingCache &cache;
	enum Mode {
		RENDER,
		COMPUTE
	};
	Mode mode;

	void set(MTL::Buffer **p_buffers, const NS::UInteger *p_offsets, NS::Range p_range);
	void set(MTL::Buffer *p_buffer, NS::UInteger p_offset, uint32_t p_index);
	void set(MTL::Texture **p_textures, NS::Range p_range);
	void set(MTL::SamplerState **p_samplers, NS::Range p_range);

	DirectEncoder(MTL::CommandEncoder *p_encoder, BindingCache &p_cache, Mode p_mode) :
			encoder(p_encoder), cache(p_cache), mode(p_mode) {}
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDCommandBuffer : public MDCommandBufferBase {
	friend class MDUniformSet;

private:
#pragma mark - Common State

	BindingCache binding_cache;

#pragma mark - Argument Buffer Ring Allocator

	using Alloc = MDRingBuffer::Allocation;

	// Used for argument buffers that contain dynamic uniforms.
	MDRingBuffer _scratch;

	/// Allocates from the ring buffer for dynamic argument buffers.
	Alloc allocate_arg_buffer(uint32_t p_size);

	struct {
		NS::SharedPtr<MTL::ResidencySet> rs;
	} _frame_state;

#pragma mark - Synchronization

	enum {
		STAGE_RENDER,
		STAGE_COMPUTE,
		STAGE_BLIT,
		STAGE_MAX,
	};
	bool use_barriers = false;
	MTL::Stages pending_after_stages[STAGE_MAX] = { 0, 0, 0 };
	MTL::Stages pending_before_queue_stages[STAGE_MAX] = { 0, 0, 0 };
	void _encode_barrier(MTL::CommandEncoder *p_enc);

	void reset();

	MTL::CommandQueue *queue = nullptr;
	NS::SharedPtr<MTL::CommandBuffer> commandBuffer;
	bool state_begin = false;

	MTL::CommandBuffer *command_buffer();

	void _end_compute_dispatch();
	void _end_blit();
	MTL::BlitCommandEncoder *_ensure_blit_encoder();

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

#pragma mark - Compute

	void _compute_set_dirty_state();
	void _compute_bind_uniform_sets();
	void _bind_uniforms_argument_buffers_compute(MDUniformSet *p_set, MDShader *p_shader, uint32_t p_set_index, uint32_t p_dynamic_offsets);

protected:
	void mark_push_constants_dirty() override;
	RenderStateBase &get_render_state_base() override { return render; }
	uint32_t get_current_view_count() const override { return render.get_subpass().view_count; }
	MDRenderPass *get_render_pass() const override { return render.pass; }
	MDFrameBuffer *get_frame_buffer() const override { return render.frameBuffer; }
	const MDSubpass &get_current_subpass() const override { return render.get_subpass(); }
	LocalVector<RDD::RenderPassClearValue> &get_clear_values() override { return render.clear_values; }
	const Rect2i &get_render_area() const override { return render.render_area; }
	void end_render_encoding() override { render.end_encoding(); }

public:
	struct RenderState : public RenderStateBase {
		MDRenderPass *pass = nullptr;
		MDFrameBuffer *frameBuffer = nullptr;
		MDRenderPipeline *pipeline = nullptr;
		LocalVector<RDD::RenderPassClearValue> clear_values;
		uint32_t current_subpass = UINT32_MAX;
		Rect2i render_area = {};
		bool is_rendering_entire_area = false;
		NS::SharedPtr<MTL::RenderPassDescriptor> desc;
		NS::SharedPtr<MTL::RenderCommandEncoder> encoder;
		MTL::Buffer *index_buffer = nullptr; // Buffer is owned by RDD.
		MTL::IndexType index_type = MTL::IndexTypeUInt16;
		uint32_t index_offset = 0;
		LocalVector<MTL::Buffer *> vertex_buffers;
		LocalVector<NS::UInteger> vertex_offsets;
		ResourceTracker resource_tracker;

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

		MTL::ScissorRect clip_to_render_area(MTL::ScissorRect p_rect) const {
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
		NS::SharedPtr<MTL::ComputeCommandEncoder> encoder;
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
		NS::SharedPtr<MTL::BlitCommandEncoder> encoder;
		_FORCE_INLINE_ void reset() {
			encoder.reset();
		}
	} blit;

	_FORCE_INLINE_ MTL::CommandBuffer *get_command_buffer() const {
		return commandBuffer.get();
	}

	void begin() override;
	void commit() override;
	void end() override;

	void bind_pipeline(RDD::PipelineID p_pipeline) override;

#pragma mark - Render Commands

	void render_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) override;
	void render_clear_attachments(VectorView<RDD::AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) override;
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
	MTL::RenderCommandEncoder *get_new_render_encoder_with_descriptor(MTL::RenderPassDescriptor *p_desc);

public:
	void resolve_texture(RDD::TextureID p_src_texture, RDD::TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, RDD::TextureID p_dst_texture, RDD::TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) override;
	void clear_color_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, const Color &p_color, const RDD::TextureSubresourceRange &p_subresources) override;
	void clear_depth_stencil_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, float p_depth, uint8_t p_stencil, const RDD::TextureSubresourceRange &p_subresources) override;
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
			VectorView<RDD::TextureBarrier> p_texture_barriers,
			VectorView<RDD::AccelerationStructureBarrier> p_acceleration_structure_barriers) override;

#pragma mark - Debugging

	void begin_label(const char *p_label_name, const Color &p_color) override;
	void end_label() override;

	MDCommandBuffer(MTL::CommandQueue *p_queue, ::RenderingDeviceDriverMetal *p_device_driver);
	MDCommandBuffer() = default;
};

} // namespace MTL3

// C++ helper to get mipmap level size from texture
_FORCE_INLINE_ static MTL::Size mipmapLevelSizeFromTexture(MTL::Texture *p_tex, NS::UInteger p_level) {
	MTL::Size lvlSize;
	lvlSize.width = MAX(p_tex->width() >> p_level, 1UL);
	lvlSize.height = MAX(p_tex->height() >> p_level, 1UL);
	lvlSize.depth = MAX(p_tex->depth() >> p_level, 1UL);
	return lvlSize;
}
