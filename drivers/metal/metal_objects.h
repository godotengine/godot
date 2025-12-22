/**************************************************************************/
/*  metal_objects.h                                                       */
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

#import "metal_device_properties.h"
#import "metal_objects_shared.h"
#import "metal_utils.h"
#import "pixel_formats.h"
#import "sha256_digest.h"

#include "servers/rendering/rendering_device_driver.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <simd/simd.h>
#import <zlib.h>
#import <initializer_list>
#import <memory>
#import <optional>

enum StageResourceUsage : uint32_t {
	ResourceUnused = 0,
	VertexRead = (MTLResourceUsageRead << RDD::SHADER_STAGE_VERTEX * 2),
	VertexWrite = (MTLResourceUsageWrite << RDD::SHADER_STAGE_VERTEX * 2),
	FragmentRead = (MTLResourceUsageRead << RDD::SHADER_STAGE_FRAGMENT * 2),
	FragmentWrite = (MTLResourceUsageWrite << RDD::SHADER_STAGE_FRAGMENT * 2),
	TesselationControlRead = (MTLResourceUsageRead << RDD::SHADER_STAGE_TESSELATION_CONTROL * 2),
	TesselationControlWrite = (MTLResourceUsageWrite << RDD::SHADER_STAGE_TESSELATION_CONTROL * 2),
	TesselationEvaluationRead = (MTLResourceUsageRead << RDD::SHADER_STAGE_TESSELATION_EVALUATION * 2),
	TesselationEvaluationWrite = (MTLResourceUsageWrite << RDD::SHADER_STAGE_TESSELATION_EVALUATION * 2),
	ComputeRead = (MTLResourceUsageRead << RDD::SHADER_STAGE_COMPUTE * 2),
	ComputeWrite = (MTLResourceUsageWrite << RDD::SHADER_STAGE_COMPUTE * 2),
};

typedef id<MTLResource> __unsafe_unretained MTLResourceUnsafe;

template <>
struct HashMapHasherDefaultImpl<MTLResourceUnsafe> {
	static _FORCE_INLINE_ uint32_t hash(const MTLResourceUnsafe p_pointer) { return hash_one_uint64((uint64_t)p_pointer); }
};

typedef LocalVector<MTLResourceUnsafe> ResourceVector;
typedef HashMap<StageResourceUsage, ResourceVector> ResourceUsageMap;

struct ResourceUsageEntry {
	StageResourceUsage usage = ResourceUnused;
	uint32_t unused = 0;

	ResourceUsageEntry() {}
	ResourceUsageEntry(StageResourceUsage p_usage) :
			usage(p_usage) {}
};

template <>
struct is_zero_constructible<ResourceUsageEntry> : std::true_type {};

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

enum class MDCommandBufferStateType {
	None,
	Render,
	Compute,
	Blit,
};

enum class MDPipelineType {
	None,
	Render,
	Compute,
};

class MDRenderPass;
class MDPipeline;
class MDRenderPipeline;
class MDComputePipeline;
class RenderingDeviceDriverMetal;
class MDUniformSet;
class MDShader;

struct MetalBufferDynamicInfo;

using RDM = RenderingDeviceDriverMetal;

#pragma mark - Resource Factory

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDResourceFactory {
private:
	RenderingDeviceDriverMetal *device_driver;

	id<MTLFunction> new_func(NSString *p_source, NSString *p_name, NSError **p_error);
	id<MTLFunction> new_clear_vert_func(ClearAttKey &p_key);
	id<MTLFunction> new_clear_frag_func(ClearAttKey &p_key);
	NSString *get_format_type_string(MTLPixelFormat p_fmt);

public:
	id<MTLRenderPipelineState> new_clear_pipeline_state(ClearAttKey &p_key, NSError **p_error);
	id<MTLDepthStencilState> new_depth_stencil_state(bool p_use_depth, bool p_use_stencil);

	MDResourceFactory(RenderingDeviceDriverMetal *p_device_driver) :
			device_driver(p_device_driver) {}
	~MDResourceFactory() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDResourceCache {
private:
	typedef HashMap<ClearAttKey, id<MTLRenderPipelineState>> HashMap;
	std::unique_ptr<MDResourceFactory> resource_factory;
	HashMap clear_states;

	struct {
		id<MTLDepthStencilState> all;
		id<MTLDepthStencilState> depth_only;
		id<MTLDepthStencilState> stencil_only;
		id<MTLDepthStencilState> none;
	} clear_depth_stencil_state;

public:
	id<MTLRenderPipelineState> get_clear_render_pipeline_state(ClearAttKey &p_key, NSError **p_error);
	id<MTLDepthStencilState> get_depth_stencil_state(bool p_use_depth, bool p_use_stencil);

	explicit MDResourceCache(RenderingDeviceDriverMetal *p_device_driver) :
			resource_factory(new MDResourceFactory(p_device_driver)) {}
	~MDResourceCache() = default;
};

enum class MDAttachmentType : uint8_t {
	None = 0,
	Color = 1 << 0,
	Depth = 1 << 1,
	Stencil = 1 << 2,
};

_FORCE_INLINE_ MDAttachmentType &operator|=(MDAttachmentType &p_a, MDAttachmentType p_b) {
	flags::set(p_a, p_b);
	return p_a;
}

_FORCE_INLINE_ bool operator&(MDAttachmentType p_a, MDAttachmentType p_b) {
	return uint8_t(p_a) & uint8_t(p_b);
}

struct MDSubpass {
	uint32_t subpass_index = 0;
	uint32_t view_count = 0;
	LocalVector<RDD::AttachmentReference> input_references;
	LocalVector<RDD::AttachmentReference> color_references;
	RDD::AttachmentReference depth_stencil_reference;
	LocalVector<RDD::AttachmentReference> resolve_references;

	MTLFmtCaps getRequiredFmtCapsForAttachmentAt(uint32_t p_index) const;
};

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDAttachment {
private:
	uint32_t index = 0;
	uint32_t firstUseSubpassIndex = 0;
	uint32_t lastUseSubpassIndex = 0;

public:
	MTLPixelFormat format = MTLPixelFormatInvalid;
	MDAttachmentType type = MDAttachmentType::None;
	MTLLoadAction loadAction = MTLLoadActionDontCare;
	MTLStoreAction storeAction = MTLStoreActionDontCare;
	MTLLoadAction stencilLoadAction = MTLLoadActionDontCare;
	MTLStoreAction stencilStoreAction = MTLStoreActionDontCare;
	uint32_t samples = 1;

	/*!
	 * @brief Returns true if this attachment is first used in the given subpass.
	 * @param p_subpass
	 * @return
	 */
	_FORCE_INLINE_ bool isFirstUseOf(MDSubpass const &p_subpass) const {
		return p_subpass.subpass_index == firstUseSubpassIndex;
	}

	/*!
	 * @brief Returns true if this attachment is last used in the given subpass.
	 * @param p_subpass
	 * @return
	 */
	_FORCE_INLINE_ bool isLastUseOf(MDSubpass const &p_subpass) const {
		return p_subpass.subpass_index == lastUseSubpassIndex;
	}

	void linkToSubpass(MDRenderPass const &p_pass);

	MTLStoreAction getMTLStoreAction(MDSubpass const &p_subpass,
			bool p_is_rendering_entire_area,
			bool p_has_resolve,
			bool p_can_resolve,
			bool p_is_stencil) const;
	bool configureDescriptor(MTLRenderPassAttachmentDescriptor *p_desc,
			PixelFormats &p_pf,
			MDSubpass const &p_subpass,
			id<MTLTexture> p_attachment,
			bool p_is_rendering_entire_area,
			bool p_has_resolve,
			bool p_can_resolve,
			bool p_is_stencil) const;
	/** Returns whether this attachment should be cleared in the subpass. */
	bool shouldClear(MDSubpass const &p_subpass, bool p_is_stencil) const;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDRenderPass {
public:
	Vector<MDAttachment> attachments;
	Vector<MDSubpass> subpasses;

	uint32_t get_sample_count() const {
		return attachments.is_empty() ? 1 : attachments[0].samples;
	}

	MDRenderPass(Vector<MDAttachment> &p_attachments, Vector<MDSubpass> &p_subpasses);
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

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDCommandBuffer {
	friend class MDUniformSet;

private:
#pragma mark - Common State

	// From RenderingDevice
	static constexpr uint32_t MAX_PUSH_CONSTANT_SIZE = 128;

	uint8_t push_constant_data[MAX_PUSH_CONSTANT_SIZE];
	uint32_t push_constant_data_len = 0;
	uint32_t push_constant_binding = UINT32_MAX;

	BindingCache binding_cache;

	void reset();

	RenderingDeviceDriverMetal *device_driver = nullptr;
	id<MTLCommandQueue> queue = nil;
	id<MTLCommandBuffer> commandBuffer = nil;
	bool state_begin = false;

	_FORCE_INLINE_ id<MTLCommandBuffer> command_buffer() {
		DEV_ASSERT(state_begin);
		if (commandBuffer == nil) {
			commandBuffer = queue.commandBuffer;
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

	void _populate_vertices(simd::float4 *p_vertices, Size2i p_fb_size, VectorView<Rect2i> p_rects);
	uint32_t _populate_vertices(simd::float4 *p_vertices, uint32_t p_index, Rect2i const &p_rect, Size2i p_fb_size);
	void _end_render_pass();
	void _render_clear_render_area();

#pragma mark - Compute

	void _compute_set_dirty_state();
	void _compute_bind_uniform_sets();

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

	void begin();
	void commit();
	void end();

	void bind_pipeline(RDD::PipelineID p_pipeline);
	void encode_push_constant_data(RDD::ShaderID p_shader, VectorView<uint32_t> p_data);

#pragma mark - Render Commands

	void render_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets);
	void render_clear_attachments(VectorView<RDD::AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects);
	void render_set_viewport(VectorView<Rect2i> p_viewports);
	void render_set_scissor(VectorView<Rect2i> p_scissors);
	void render_set_blend_constants(const Color &p_constants);
	void render_begin_pass(RDD::RenderPassID p_render_pass,
			RDD::FramebufferID p_frameBuffer,
			RDD::CommandBufferType p_cmd_buffer_type,
			const Rect2i &p_rect,
			VectorView<RDD::RenderPassClearValue> p_clear_values);
	void render_next_subpass();
	void render_draw(uint32_t p_vertex_count,
			uint32_t p_instance_count,
			uint32_t p_base_vertex,
			uint32_t p_first_instance);
	void render_bind_vertex_buffers(uint32_t p_binding_count, const RDD::BufferID *p_buffers, const uint64_t *p_offsets, uint64_t p_dynamic_offsets);
	void render_bind_index_buffer(RDD::BufferID p_buffer, RDD::IndexBufferFormat p_format, uint64_t p_offset);

	void render_draw_indexed(uint32_t p_index_count,
			uint32_t p_instance_count,
			uint32_t p_first_index,
			int32_t p_vertex_offset,
			uint32_t p_first_instance);

	void render_draw_indexed_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride);
	void render_draw_indexed_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride);
	void render_draw_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride);
	void render_draw_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride);

	void render_end_pass();

#pragma mark - Compute Commands

	void compute_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets);
	void compute_dispatch(uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups);
	void compute_dispatch_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset);

#pragma mark - Transfer

private:
	void encodeRenderCommandEncoderWithDescriptor(MTLRenderPassDescriptor *p_desc, NSString *p_label);

public:
	void resolve_texture(RDD::TextureID p_src_texture, RDD::TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, RDD::TextureID p_dst_texture, RDD::TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap);
	void clear_color_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, const Color &p_color, const RDD::TextureSubresourceRange &p_subresources);
	void clear_buffer(RDD::BufferID p_buffer, uint64_t p_offset, uint64_t p_size);
	void copy_buffer(RDD::BufferID p_src_buffer, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferCopyRegion> p_regions);
	void copy_texture(RDD::TextureID p_src_texture, RDD::TextureID p_dst_texture, VectorView<RDD::TextureCopyRegion> p_regions);
	void copy_buffer_to_texture(RDD::BufferID p_src_buffer, RDD::TextureID p_dst_texture, VectorView<RDD::BufferTextureCopyRegion> p_regions);
	void copy_texture_to_buffer(RDD::TextureID p_src_texture, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferTextureCopyRegion> p_regions);

#pragma mark - Debugging

	void begin_label(const char *p_label_name, const Color &p_color);
	void end_label();

	MDCommandBuffer(id<MTLCommandQueue> p_queue, RenderingDeviceDriverMetal *p_device_driver) :
			device_driver(p_device_driver), queue(p_queue) {
		type = MDCommandBufferStateType::None;
	}

	MDCommandBuffer() = default;
};

#if (TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED < 140000) || (TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED < 170000)
#define MTLBindingAccess MTLArgumentAccess
#define MTLBindingAccessReadOnly MTLArgumentAccessReadOnly
#define MTLBindingAccessReadWrite MTLArgumentAccessReadWrite
#define MTLBindingAccessWriteOnly MTLArgumentAccessWriteOnly
#endif

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) UniformInfo {
	uint32_t binding;
	BitField<RDD::ShaderStage> active_stages;
	MTLDataType dataType = MTLDataTypeNone;
	MTLBindingAccess access = MTLBindingAccessReadOnly;
	MTLResourceUsage usage = 0;
	MTLTextureType textureType = MTLTextureType2D;
	uint32_t imageFormat = 0;
	uint32_t arrayLength = 0;
	bool isMultisampled = 0;

	struct Indexes {
		uint32_t buffer = UINT32_MAX;
		uint32_t texture = UINT32_MAX;
		uint32_t sampler = UINT32_MAX;
	};
	Indexes slot;
	Indexes arg_buffer;

	enum class IndexType {
		SLOT,
		ARG,
	};

	_FORCE_INLINE_ Indexes &get_indexes(IndexType p_type) {
		switch (p_type) {
			case IndexType::SLOT:
				return slot;
			case IndexType::ARG:
				return arg_buffer;
		}
	}
};

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) UniformSet {
	LocalVector<UniformInfo> uniforms;
	LocalVector<uint32_t> dynamic_uniforms;
	uint32_t buffer_size = 0;
};

struct ShaderCacheEntry;

enum class ShaderLoadStrategy {
	IMMEDIATE,
	LAZY,

	/// The default strategy is to load the shader immediately.
	DEFAULT = IMMEDIATE,
};

/// A Metal shader library.
@interface MDLibrary : NSObject {
	ShaderCacheEntry *_entry;
	NSString *_original_source;
};
- (id<MTLLibrary>)library;
- (NSError *)error;
- (void)setLabel:(NSString *)label;
#ifdef DEV_ENABLED
- (NSString *)originalSource;
#endif

+ (instancetype)newLibraryWithCacheEntry:(ShaderCacheEntry *)entry
								  device:(id<MTLDevice>)device
								  source:(NSString *)source
								 options:(MTLCompileOptions *)options
								strategy:(ShaderLoadStrategy)strategy;

+ (instancetype)newLibraryWithCacheEntry:(ShaderCacheEntry *)entry
								  device:(id<MTLDevice>)device
#ifdef DEV_ENABLED
								  source:(NSString *)source
#endif
									data:(dispatch_data_t)data;
@end

/// A cache entry for a Metal shader library.
struct ShaderCacheEntry {
	RenderingDeviceDriverMetal &owner;
	/// A hash of the Metal shader source code.
	SHA256Digest key;
	CharString name;
	RD::ShaderStage stage = RD::SHADER_STAGE_VERTEX;
	/// This reference must be weak, to ensure that when the last strong reference to the library
	/// is released, the cache entry is freed.
	MDLibrary *__weak library = nil;

	/// Notify the cache that this entry is no longer needed.
	void notify_free() const;

	ShaderCacheEntry(RenderingDeviceDriverMetal &p_owner, SHA256Digest p_key) :
			owner(p_owner), key(p_key) {
	}
	~ShaderCacheEntry() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) DynamicOffsetLayout {
	struct Data {
		uint8_t offset : 4;
		uint8_t count : 4;
	};

	union {
		Data data[MAX_DYNAMIC_BUFFERS];
		uint64_t _val = 0;
	};

public:
	_FORCE_INLINE_ bool is_empty() const { return _val == 0; }

	_FORCE_INLINE_ uint32_t get_count(uint32_t p_set_index) const {
		return data[p_set_index].count;
	}

	_FORCE_INLINE_ uint32_t get_offset(uint32_t p_set_index) const {
		return data[p_set_index].offset;
	}

	_FORCE_INLINE_ void set_offset_count(uint32_t p_set_index, uint8_t p_offset, uint8_t p_count) {
		data[p_set_index].offset = p_offset;
		data[p_set_index].count = p_count;
	}

	_FORCE_INLINE_ uint32_t get_offset_index_shift(uint32_t p_set_index, uint32_t p_dynamic_index = 0) const {
		return (data[p_set_index].offset + p_dynamic_index) * 4u;
	}
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDShader {
public:
	CharString name;
	Vector<UniformSet> sets;
	struct {
		BitField<RDD::ShaderStage> stages = {};
		uint32_t binding = UINT32_MAX;
		uint32_t size = 0;
	} push_constants;
	DynamicOffsetLayout dynamic_offset_layout;
	bool uses_argument_buffers = true;

	MDShader(CharString p_name, Vector<UniformSet> p_sets, bool p_uses_argument_buffers) :
			name(p_name), sets(p_sets), uses_argument_buffers(p_uses_argument_buffers) {}
	virtual ~MDShader() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDComputeShader final : public MDShader {
public:
	MTLSize local = {};

	MDLibrary *kernel;

	MDComputeShader(CharString p_name, Vector<UniformSet> p_sets, bool p_uses_argument_buffers, MDLibrary *p_kernel);
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDRenderShader final : public MDShader {
public:
	bool needs_view_mask_buffer = false;

	MDLibrary *vert;
	MDLibrary *frag;

	MDRenderShader(CharString p_name,
			Vector<UniformSet> p_sets,
			bool p_needs_view_mask_buffer,
			bool p_uses_argument_buffers,
			MDLibrary *p_vert, MDLibrary *p_frag);
};

_FORCE_INLINE_ StageResourceUsage &operator|=(StageResourceUsage &p_a, uint32_t p_b) {
	p_a = StageResourceUsage(uint32_t(p_a) | p_b);
	return p_a;
}

_FORCE_INLINE_ StageResourceUsage stage_resource_usage(RDC::ShaderStage p_stage, MTLResourceUsage p_usage) {
	return StageResourceUsage(p_usage << (p_stage * 2));
}

_FORCE_INLINE_ MTLResourceUsage resource_usage_for_stage(StageResourceUsage p_usage, RDC::ShaderStage p_stage) {
	return MTLResourceUsage((p_usage >> (p_stage * 2)) & 0b11);
}

template <>
struct HashMapComparatorDefault<RDD::ShaderID> {
	static bool compare(const RDD::ShaderID &p_lhs, const RDD::ShaderID &p_rhs) {
		return p_lhs.id == p_rhs.id;
	}
};

template <>
struct HashMapComparatorDefault<RDD::BufferID> {
	static bool compare(const RDD::BufferID &p_lhs, const RDD::BufferID &p_rhs) {
		return p_lhs.id == p_rhs.id;
	}
};

template <>
struct HashMapComparatorDefault<RDD::TextureID> {
	static bool compare(const RDD::TextureID &p_lhs, const RDD::TextureID &p_rhs) {
		return p_lhs.id == p_rhs.id;
	}
};

template <>
struct HashMapHasherDefaultImpl<RDD::BufferID> {
	static _FORCE_INLINE_ uint32_t hash(const RDD::BufferID &p_value) {
		return HashMapHasherDefaultImpl<uint64_t>::hash(p_value.id);
	}
};

template <>
struct HashMapHasherDefaultImpl<RDD::TextureID> {
	static _FORCE_INLINE_ uint32_t hash(const RDD::TextureID &p_value) {
		return HashMapHasherDefaultImpl<uint64_t>::hash(p_value.id);
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

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDUniformSet {
public:
	uint32_t index = 0;
	id<MTLBuffer> arg_buffer = nil;
	ResourceUsageMap usage_to_resources;
	LocalVector<RDD::BoundUniform> uniforms;

	void bind_uniforms_argument_buffers(MDShader *p_shader, MDCommandBuffer::RenderState &p_state, uint32_t p_set_index, uint32_t p_dynamic_offsets, uint32_t p_frame_idx, uint32_t p_frame_count);
	void bind_uniforms_argument_buffers(MDShader *p_shader, MDCommandBuffer::ComputeState &p_state, uint32_t p_set_index, uint32_t p_dynamic_offsets, uint32_t p_frame_idx, uint32_t p_frame_count);
	void bind_uniforms_direct(MDShader *p_shader, DirectEncoder p_enc, uint32_t p_set_index, uint32_t p_dynamic_offsets);
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDPipeline {
public:
	MDPipelineType type;

	explicit MDPipeline(MDPipelineType p_type) :
			type(p_type) {}
	virtual ~MDPipeline() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDRenderPipeline final : public MDPipeline {
public:
	id<MTLRenderPipelineState> state = nil;
	id<MTLDepthStencilState> depth_stencil = nil;
	uint32_t push_constant_size = 0;
	uint32_t push_constant_stages_mask = 0;
	SampleCount sample_count = SampleCount1;

	struct {
		MTLCullMode cull_mode = MTLCullModeNone;
		MTLTriangleFillMode fill_mode = MTLTriangleFillModeFill;
		MTLDepthClipMode clip_mode = MTLDepthClipModeClip;
		MTLWinding winding = MTLWindingClockwise;
		MTLPrimitiveType render_primitive = MTLPrimitiveTypePoint;

		struct {
			bool enabled = false;
		} depth_test;

		struct {
			bool enabled = false;
			float depth_bias = 0.0;
			float slope_scale = 0.0;
			float clamp = 0.0;
			_FORCE_INLINE_ void apply(id<MTLRenderCommandEncoder> __unsafe_unretained p_enc) const {
				if (!enabled) {
					return;
				}
				[p_enc setDepthBias:depth_bias slopeScale:slope_scale clamp:clamp];
			}
		} depth_bias;

		struct {
			bool enabled = false;
			uint32_t front_reference = 0;
			uint32_t back_reference = 0;
			_FORCE_INLINE_ void apply(id<MTLRenderCommandEncoder> __unsafe_unretained p_enc) const {
				if (!enabled) {
					return;
				}
				[p_enc setStencilFrontReferenceValue:front_reference backReferenceValue:back_reference];
			}
		} stencil;

		struct {
			bool enabled = false;
			float r = 0.0;
			float g = 0.0;
			float b = 0.0;
			float a = 0.0;

			_FORCE_INLINE_ void apply(id<MTLRenderCommandEncoder> __unsafe_unretained p_enc) const {
				//if (!enabled)
				//	return;
				[p_enc setBlendColorRed:r green:g blue:b alpha:a];
			}
		} blend;

		_FORCE_INLINE_ void apply(id<MTLRenderCommandEncoder> __unsafe_unretained p_enc) const {
			[p_enc setCullMode:cull_mode];
			[p_enc setTriangleFillMode:fill_mode];
			[p_enc setDepthClipMode:clip_mode];
			[p_enc setFrontFacingWinding:winding];
			depth_bias.apply(p_enc);
			stencil.apply(p_enc);
			blend.apply(p_enc);
		}

	} raster_state;

	MDRenderShader *shader = nil;

	MDRenderPipeline() :
			MDPipeline(MDPipelineType::Render) {}
	~MDRenderPipeline() final = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MDComputePipeline final : public MDPipeline {
public:
	id<MTLComputePipelineState> state = nil;
	struct {
		MTLSize local = {};
	} compute_state;

	MDComputeShader *shader = nil;

	explicit MDComputePipeline(id<MTLComputePipelineState> p_state) :
			MDPipeline(MDPipelineType::Compute), state(p_state) {}
	~MDComputePipeline() final = default;
};

namespace rid {
#define MAKE_ID(FROM, TO)                \
	_FORCE_INLINE_ TO make(FROM p_obj) { \
		return TO(owned(p_obj));         \
	}

MAKE_ID(id<MTLCommandQueue>, RDD::CommandPoolID)

#undef MAKE_ID
} //namespace rid

namespace MTL {

_FORCE_INLINE_ static Transform3D simd_to_transform3D(const simd_float4x4 &matrix) {
	Transform3D transform(Vector3(matrix.columns[0].x, matrix.columns[0].y, matrix.columns[0].z),
			Vector3(matrix.columns[1].x, matrix.columns[1].y, matrix.columns[1].z),
			Vector3(matrix.columns[2].x, matrix.columns[2].y, matrix.columns[2].z),
			Vector3(matrix.columns[3].x, matrix.columns[3].y, matrix.columns[3].z));
	return transform;
}

_FORCE_INLINE_ static Projection simd_to_projection(const simd_float4x4 &matrix) {
	Projection projection(Vector4(matrix.columns[0].x, matrix.columns[0].y, matrix.columns[0].z, matrix.columns[0].w),
			Vector4(matrix.columns[1].x, matrix.columns[1].y, matrix.columns[1].z, matrix.columns[1].w),
			Vector4(matrix.columns[2].x, matrix.columns[2].y, matrix.columns[2].z, matrix.columns[2].w),
			Vector4(matrix.columns[3].x, matrix.columns[3].y, matrix.columns[3].z, matrix.columns[3].w));
	return projection;
}

_FORCE_INLINE_ static Rect2i rect_from_mtl_viewport(MTLViewport viewport) {
	return Rect2i(viewport.originX, viewport.originY, viewport.width, viewport.height);
}

_FORCE_INLINE_ static RD::TextureType texture_type_from_metal(MTLTextureType p_type) {
	switch (p_type) {
		case MTLTextureType1D:
			return RD::TEXTURE_TYPE_1D;
		case MTLTextureType2D:
			return RD::TEXTURE_TYPE_2D;
		case MTLTextureType3D:
			return RD::TEXTURE_TYPE_3D;
		case MTLTextureTypeCube:
			return RD::TEXTURE_TYPE_CUBE;
		case MTLTextureType1DArray:
			return RD::TEXTURE_TYPE_1D_ARRAY;
		case MTLTextureType2DArray:
			return RD::TEXTURE_TYPE_2D_ARRAY;
		case MTLTextureTypeCubeArray:
			return RD::TEXTURE_TYPE_CUBE_ARRAY;
		default:
			return RD::TEXTURE_TYPE_MAX; // Fallback for unknown types
	}
}

_FORCE_INLINE_ static RD::TextureSamples texture_samples_from_metal(int p_sample_count) {
	switch (p_sample_count) {
		case 1:
			return RD::TEXTURE_SAMPLES_1;
		case 2:
			return RD::TEXTURE_SAMPLES_2;
		case 4:
			return RD::TEXTURE_SAMPLES_4;
		case 8:
			return RD::TEXTURE_SAMPLES_8;
		case 16:
			return RD::TEXTURE_SAMPLES_16;
		case 32:
			return RD::TEXTURE_SAMPLES_32;
		case 64:
			return RD::TEXTURE_SAMPLES_64;
		default:
			return RD::TEXTURE_SAMPLES_MAX;
	}
}

} //namespace MTL
