/**************************************************************************/
/*  metal_objects_shared.h                                                */
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

#import "metal_device_properties.h"
#import "metal_utils.h"
#import "pixel_formats.h"
#import "sha256_digest.h"

#import <memory>

class RenderingDeviceDriverMetal;

using RDC = RenderingDeviceCommons;

// These types can be used in Vector and other containers that use
// pointer operations not supported by ARC.
namespace MTL {
#define MTL_CLASS(name)                               \
	class name {                                      \
	public:                                           \
		name(id<MTL##name> obj = nil) : m_obj(obj) {} \
		operator id<MTL##name>() const {              \
			return m_obj;                             \
		}                                             \
		id<MTL##name> m_obj;                          \
	};

MTL_CLASS(Texture)

} //namespace MTL

typedef id<MTLResource> __unsafe_unretained MTLResourceUnsafe;

template <>
struct HashMapHasherDefaultImpl<MTLResourceUnsafe> {
	static _FORCE_INLINE_ uint32_t hash(const MTLResourceUnsafe p_pointer) { return hash_one_uint64((uint64_t)p_pointer); }
};

enum ShaderStageUsage : uint32_t {
	None = 0,
	Vertex = RDD::SHADER_STAGE_VERTEX_BIT,
	Fragment = RDD::SHADER_STAGE_FRAGMENT_BIT,
	TesselationControl = RDD::SHADER_STAGE_TESSELATION_CONTROL_BIT,
	TesselationEvaluation = RDD::SHADER_STAGE_TESSELATION_EVALUATION_BIT,
	Compute = RDD::SHADER_STAGE_COMPUTE_BIT,
};

_FORCE_INLINE_ ShaderStageUsage &operator|=(ShaderStageUsage &p_a, int p_b) {
	p_a = ShaderStageUsage(uint32_t(p_a) | uint32_t(p_b));
	return p_a;
}

struct ClearAttKey {
	const static uint32_t COLOR_COUNT = MAX_COLOR_ATTACHMENT_COUNT;
	const static uint32_t DEPTH_INDEX = COLOR_COUNT;
	const static uint32_t STENCIL_INDEX = DEPTH_INDEX + 1;
	const static uint32_t ATTACHMENT_COUNT = STENCIL_INDEX + 1;

	enum Flags : uint16_t {
		CLEAR_FLAGS_NONE = 0,
		CLEAR_FLAGS_LAYERED = 1 << 0,
	};

	Flags flags = CLEAR_FLAGS_NONE;
	uint16_t sample_count = 0;
	uint16_t pixel_formats[ATTACHMENT_COUNT] = { 0 };

	_FORCE_INLINE_ void set_color_format(uint32_t p_idx, MTLPixelFormat p_fmt) { pixel_formats[p_idx] = p_fmt; }
	_FORCE_INLINE_ void set_depth_format(MTLPixelFormat p_fmt) { pixel_formats[DEPTH_INDEX] = p_fmt; }
	_FORCE_INLINE_ void set_stencil_format(MTLPixelFormat p_fmt) { pixel_formats[STENCIL_INDEX] = p_fmt; }
	_FORCE_INLINE_ MTLPixelFormat depth_format() const { return (MTLPixelFormat)pixel_formats[DEPTH_INDEX]; }
	_FORCE_INLINE_ MTLPixelFormat stencil_format() const { return (MTLPixelFormat)pixel_formats[STENCIL_INDEX]; }
	_FORCE_INLINE_ void enable_layered_rendering() { flags::set(flags, CLEAR_FLAGS_LAYERED); }

	_FORCE_INLINE_ bool is_enabled(uint32_t p_idx) const { return pixel_formats[p_idx] != 0; }
	_FORCE_INLINE_ bool is_depth_enabled() const { return pixel_formats[DEPTH_INDEX] != 0; }
	_FORCE_INLINE_ bool is_stencil_enabled() const { return pixel_formats[STENCIL_INDEX] != 0; }
	_FORCE_INLINE_ bool is_layered_rendering_enabled() const { return flags::any(flags, CLEAR_FLAGS_LAYERED); }

	_FORCE_INLINE_ bool operator==(const ClearAttKey &p_rhs) const {
		return memcmp(this, &p_rhs, sizeof(ClearAttKey)) == 0;
	}

	uint32_t hash() const {
		uint32_t h = hash_murmur3_one_32(flags);
		h = hash_murmur3_one_32(sample_count, h);
		h = hash_murmur3_buffer(pixel_formats, ATTACHMENT_COUNT * sizeof(pixel_formats[0]), h);
		return hash_fmix32(h);
	}
};

#pragma mark - Ring Buffer

/// A ring buffer backed by MTLBuffer instances for transient GPU allocations.
/// Allocations are 16-byte aligned with a minimum size of 16 bytes.
/// When the current buffer is exhausted, a new buffer is allocated.
class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDRingBuffer {
public:
	static constexpr uint32_t DEFAULT_BUFFER_SIZE = 512 * 1024;
	static constexpr uint32_t MIN_BLOCK_SIZE = 16;
	static constexpr uint32_t ALIGNMENT = 16;

	struct Allocation {
		void *ptr = nullptr;
		id<MTLBuffer> buffer = nil;
		uint64_t gpu_address = 0;
		uint32_t offset = 0;

		_FORCE_INLINE_ bool is_valid() const { return ptr != nullptr; }
	};

private:
	id<MTLDevice> device = nil;
	LocalVector<id<MTLBuffer>> buffers;
	LocalVector<uint32_t> heads;
	uint32_t current_segment = 0;
	uint32_t buffer_size = DEFAULT_BUFFER_SIZE;
	bool changed = false;

	_FORCE_INLINE_ uint32_t alloc_segment() {
		id<MTLBuffer> buffer = [device newBufferWithLength:buffer_size
												   options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
		buffers.push_back(buffer);
		heads.push_back(0);
		changed = true;

		return buffers.size() - 1;
	}

public:
	MDRingBuffer() = default;

	MDRingBuffer(id<MTLDevice> p_device, uint32_t p_buffer_size = DEFAULT_BUFFER_SIZE) :
			device(p_device), buffer_size(p_buffer_size) {}

	~MDRingBuffer() {
		for (uint32_t i = 0; i < buffers.size(); i++) {
			buffers[i] = nil;
		}
	}

	/// Allocates a block of memory from the ring buffer.
	/// Returns an Allocation with the pointer, buffer, and offset.
	_FORCE_INLINE_ Allocation allocate(uint32_t p_size) {
		p_size = MAX(p_size, MIN_BLOCK_SIZE);
		p_size = (p_size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

		if (buffers.is_empty()) {
			alloc_segment();
		}

		uint32_t aligned_head = (heads[current_segment] + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

		if (aligned_head + p_size > buffer_size) {
			// Current segment exhausted, try to find one with space or allocate new.
			bool found = false;
			for (uint32_t i = 0; i < buffers.size(); i++) {
				uint32_t ah = (heads[i] + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
				if (ah + p_size <= buffer_size) {
					current_segment = i;
					aligned_head = ah;
					found = true;
					break;
				}
			}

			if (!found) {
				current_segment = alloc_segment();
				aligned_head = 0;
			}
		}

		id<MTLBuffer> buffer = buffers[current_segment];
		Allocation alloc;
		alloc.buffer = buffer;
		alloc.offset = aligned_head;
		alloc.ptr = static_cast<uint8_t *>([buffer contents]) + aligned_head;
		if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
			alloc.gpu_address = buffer.gpuAddress + aligned_head;
		}
		heads[current_segment] = aligned_head + p_size;

		return alloc;
	}

	/// Resets all segments for reuse. Call at frame boundaries when GPU work is complete.
	_FORCE_INLINE_ void reset() {
		for (uint32_t &head : heads) {
			head = 0;
		}
		current_segment = 0;
	}

	/// Returns true if buffers were added or removed since last clear_changed().
	_FORCE_INLINE_ bool is_changed() const { return changed; }

	/// Clears the changed flag.
	_FORCE_INLINE_ void clear_changed() { changed = false; }

	/// Returns a Span of all backing buffers.
	_FORCE_INLINE_ Span<const id<MTLBuffer> __unsafe_unretained> get_buffers() const {
		return Span<const id<MTLBuffer> __unsafe_unretained>(buffers.ptr(), buffers.size());
	}

	/// Returns the number of buffer segments currently allocated.
	_FORCE_INLINE_ uint32_t get_segment_count() const {
		return buffers.size();
	}
};

#pragma mark - Resource Factory

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDResourceFactory {
private:
	id<MTLDevice> device;
	PixelFormats &pixel_formats;
	uint32_t max_buffer_count;

	id<MTLFunction> new_func(NSString *p_source, NSString *p_name, NSError **p_error);
	id<MTLFunction> new_clear_vert_func(ClearAttKey &p_key);
	id<MTLFunction> new_clear_frag_func(ClearAttKey &p_key);
	NSString *get_format_type_string(MTLPixelFormat p_fmt);

	_FORCE_INLINE_ uint32_t get_vertex_buffer_index(uint32_t p_binding) {
		return (max_buffer_count - 1) - p_binding;
	}

public:
	id<MTLRenderPipelineState> new_clear_pipeline_state(ClearAttKey &p_key, NSError **p_error);
	id<MTLRenderPipelineState> new_empty_draw_pipeline_state(ClearAttKey &p_key, NSError **p_error);
	id<MTLDepthStencilState> new_depth_stencil_state(bool p_use_depth, bool p_use_stencil);

	MDResourceFactory(id<MTLDevice> p_device, PixelFormats &p_pixel_formats, uint32_t p_max_buffer_count) :
			device(p_device), pixel_formats(p_pixel_formats), max_buffer_count(p_max_buffer_count) {}
	~MDResourceFactory() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDResourceCache {
private:
	typedef HashMap<ClearAttKey, id<MTLRenderPipelineState>> HashMap;
	std::unique_ptr<MDResourceFactory> resource_factory;
	HashMap clear_states;
	HashMap empty_draw_states;

	struct {
		id<MTLDepthStencilState> all;
		id<MTLDepthStencilState> depth_only;
		id<MTLDepthStencilState> stencil_only;
		id<MTLDepthStencilState> none;
	} clear_depth_stencil_state;

public:
	id<MTLRenderPipelineState> get_clear_render_pipeline_state(ClearAttKey &p_key, NSError **p_error);
	id<MTLRenderPipelineState> get_empty_draw_pipeline_state(ClearAttKey &p_key, NSError **p_error);
	id<MTLDepthStencilState> get_depth_stencil_state(bool p_use_depth, bool p_use_stencil);

	explicit MDResourceCache(id<MTLDevice> p_device, PixelFormats &p_pixel_formats, uint32_t p_max_buffer_count) :
			resource_factory(new MDResourceFactory(p_device, p_pixel_formats, p_max_buffer_count)) {}
	~MDResourceCache() = default;
};

/**
 * Returns an index that can be used to map a shader stage to an index in a fixed-size array that is used for
 * a single pipeline type.
 */
_FORCE_INLINE_ static uint32_t to_index(RDD::ShaderStage p_s) {
	switch (p_s) {
		case RenderingDeviceCommons::SHADER_STAGE_VERTEX:
		case RenderingDeviceCommons::SHADER_STAGE_TESSELATION_CONTROL:
		case RenderingDeviceCommons::SHADER_STAGE_TESSELATION_EVALUATION:
		case RenderingDeviceCommons::SHADER_STAGE_COMPUTE:
		default:
			return 0;
		case RenderingDeviceCommons::SHADER_STAGE_FRAGMENT:
			return 1;
	}
}

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDFrameBuffer {
	Vector<MTL::Texture> textures;

public:
	Size2i size;
	MDFrameBuffer(Vector<MTL::Texture> p_textures, Size2i p_size) :
			textures(p_textures), size(p_size) {}
	MDFrameBuffer() {}

	/// Returns the texture at the given index.
	_ALWAYS_INLINE_ MTL::Texture get_texture(uint32_t p_idx) const {
		return textures[p_idx];
	}

	/// Returns true if the texture at the given index is not nil.
	_ALWAYS_INLINE_ bool has_texture(uint32_t p_idx) const {
		return textures[p_idx] != nil;
	}

	/// Set the texture at the given index.
	_ALWAYS_INLINE_ void set_texture(uint32_t p_idx, MTL::Texture p_texture) {
		textures.write[p_idx] = p_texture;
	}

	/// Unset or nil the texture at the given index.
	_ALWAYS_INLINE_ void unset_texture(uint32_t p_idx) {
		textures.write[p_idx] = nil;
	}

	/// Resizes buffers to the specified size.
	_ALWAYS_INLINE_ void set_texture_count(uint32_t p_size) {
		textures.resize(p_size);
	}

	virtual ~MDFrameBuffer() = default;
};

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

// These functions are used to convert between Objective-C objects and
// the RIDs used by Godot, respecting automatic reference counting.
namespace rid {

// Converts an Objective-C object to a pointer, and incrementing the
// reference count.
_FORCE_INLINE_ void *owned(id p_id) {
	return (__bridge_retained void *)p_id;
}

#define MAKE_ID(FROM, TO)                \
	_FORCE_INLINE_ TO make(FROM p_obj) { \
		return TO(owned(p_obj));         \
	}

// These are shared for Metal and Metal 4 drivers

MAKE_ID(id<MTLTexture>, RDD::TextureID)
MAKE_ID(id<MTLBuffer>, RDD::BufferID)
MAKE_ID(id<MTLSamplerState>, RDD::SamplerID)
MAKE_ID(MTLVertexDescriptor *, RDD::VertexFormatID)

#undef MAKE_ID

// Converts a pointer to an Objective-C object without changing the reference count.
_FORCE_INLINE_ auto get(RDD::ID p_id) {
	return (p_id.id) ? (__bridge ::id)(void *)p_id.id : nil;
}

// Converts a pointer to an Objective-C object, and decrements the reference count.
_FORCE_INLINE_ auto release(RDD::ID p_id) {
	return (__bridge_transfer ::id)(void *)p_id.id;
}

} // namespace rid

#pragma mark - Render Pass Types

class MDRenderPass;

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

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDAttachment {
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

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDRenderPass {
public:
	Vector<MDAttachment> attachments;
	Vector<MDSubpass> subpasses;

	uint32_t get_sample_count() const {
		return attachments.is_empty() ? 1 : attachments[0].samples;
	}

	MDRenderPass(Vector<MDAttachment> &p_attachments, Vector<MDSubpass> &p_subpasses);
};

#pragma mark - Command Buffer Base

/// Abstract base class for Metal command buffers, shared between MTL3 and MTL4 implementations.
class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDCommandBufferBase {
	LocalVector<id> _retained_resources;

protected:
	void release_resources();

public:
	virtual ~MDCommandBufferBase() = default;

	virtual void begin() = 0;
	virtual void commit() = 0;
	virtual void end() = 0;

	virtual void bind_pipeline(RDD::PipelineID p_pipeline) = 0;
	virtual void encode_push_constant_data(RDD::ShaderID p_shader, VectorView<uint32_t> p_data) = 0;

	void retain_resource(id p_resource);

#pragma mark - Render Commands

	virtual void render_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) = 0;
	virtual void render_clear_attachments(VectorView<RDD::AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) = 0;
	virtual void render_set_viewport(VectorView<Rect2i> p_viewports) = 0;
	virtual void render_set_scissor(VectorView<Rect2i> p_scissors) = 0;
	virtual void render_set_blend_constants(const Color &p_constants) = 0;
	virtual void render_begin_pass(RDD::RenderPassID p_render_pass,
			RDD::FramebufferID p_frameBuffer,
			RDD::CommandBufferType p_cmd_buffer_type,
			const Rect2i &p_rect,
			VectorView<RDD::RenderPassClearValue> p_clear_values) = 0;
	virtual void render_next_subpass() = 0;
	virtual void render_draw(uint32_t p_vertex_count,
			uint32_t p_instance_count,
			uint32_t p_base_vertex,
			uint32_t p_first_instance) = 0;
	virtual void render_bind_vertex_buffers(uint32_t p_binding_count, const RDD::BufferID *p_buffers, const uint64_t *p_offsets, uint64_t p_dynamic_offsets) = 0;
	virtual void render_bind_index_buffer(RDD::BufferID p_buffer, RDD::IndexBufferFormat p_format, uint64_t p_offset) = 0;

	virtual void render_draw_indexed(uint32_t p_index_count,
			uint32_t p_instance_count,
			uint32_t p_first_index,
			int32_t p_vertex_offset,
			uint32_t p_first_instance) = 0;

	virtual void render_draw_indexed_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) = 0;
	virtual void render_draw_indexed_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) = 0;
	virtual void render_draw_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) = 0;
	virtual void render_draw_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) = 0;

	virtual void render_end_pass() = 0;

#pragma mark - Compute Commands

	virtual void compute_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) = 0;
	virtual void compute_dispatch(uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) = 0;
	virtual void compute_dispatch_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset) = 0;

#pragma mark - Transfer

	virtual void resolve_texture(RDD::TextureID p_src_texture, RDD::TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, RDD::TextureID p_dst_texture, RDD::TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) = 0;
	virtual void clear_color_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, const Color &p_color, const RDD::TextureSubresourceRange &p_subresources) = 0;
	virtual void clear_buffer(RDD::BufferID p_buffer, uint64_t p_offset, uint64_t p_size) = 0;
	virtual void copy_buffer(RDD::BufferID p_src_buffer, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferCopyRegion> p_regions) = 0;
	virtual void copy_texture(RDD::TextureID p_src_texture, RDD::TextureID p_dst_texture, VectorView<RDD::TextureCopyRegion> p_regions) = 0;
	virtual void copy_buffer_to_texture(RDD::BufferID p_src_buffer, RDD::TextureID p_dst_texture, VectorView<RDD::BufferTextureCopyRegion> p_regions) = 0;
	virtual void copy_texture_to_buffer(RDD::TextureID p_src_texture, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferTextureCopyRegion> p_regions) = 0;

#pragma mark - Synchronization

	virtual void pipeline_barrier(BitField<RDD::PipelineStageBits> p_src_stages,
			BitField<RDD::PipelineStageBits> p_dst_stages,
			VectorView<RDD::MemoryAccessBarrier> p_memory_barriers,
			VectorView<RDD::BufferBarrier> p_buffer_barriers,
			VectorView<RDD::TextureBarrier> p_texture_barriers) = 0;

#pragma mark - Debugging

	virtual void begin_label(const char *p_label_name, const Color &p_color) = 0;
	virtual void end_label() = 0;
};

#pragma mark - Uniform Types

#if (TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED < 140000) || (TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED < 170000)
#define MTLBindingAccess MTLArgumentAccess
#define MTLBindingAccessReadOnly MTLArgumentAccessReadOnly
#define MTLBindingAccessReadWrite MTLArgumentAccessReadWrite
#define MTLBindingAccessWriteOnly MTLArgumentAccessWriteOnly
#endif

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) UniformInfo {
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

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) UniformSet {
	LocalVector<UniformInfo> uniforms;
	LocalVector<uint32_t> dynamic_uniforms;
	uint32_t buffer_size = 0;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) DynamicOffsetLayout {
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

#pragma mark - Shader Types

struct ShaderCacheEntry;

enum class ShaderLoadStrategy {
	IMMEDIATE,
	LAZY,

	/// The default strategy is to load the shader immediately.
	DEFAULT = IMMEDIATE,
};

/// A Metal shader library.
class MDLibrary : public std::enable_shared_from_this<MDLibrary> {
protected:
	ShaderCacheEntry *_entry = nullptr;
#ifdef DEV_ENABLED
	NSString *_original_source = nil;
#endif

	MDLibrary(ShaderCacheEntry *p_entry
#ifdef DEV_ENABLED
			,
			NSString *p_source
#endif
	);

public:
	virtual ~MDLibrary();

	virtual id<MTLLibrary> get_library() = 0;
	virtual NSError *get_error() = 0;
	virtual void set_label(NSString *p_label);
#ifdef DEV_ENABLED
	NSString *get_original_source() const { return _original_source; }
#endif

	static std::shared_ptr<MDLibrary> create(ShaderCacheEntry *p_entry,
			id<MTLDevice> p_device,
			NSString *p_source,
			MTLCompileOptions *p_options,
			ShaderLoadStrategy p_strategy);

	static std::shared_ptr<MDLibrary> create(ShaderCacheEntry *p_entry,
			id<MTLDevice> p_device,
#ifdef DEV_ENABLED
			NSString *p_source,
#endif
			dispatch_data_t p_data);
};

/// A cache entry for a Metal shader library.
struct ShaderCacheEntry {
	RenderingDeviceDriverMetal &owner;
	/// A hash of the Metal shader source code.
	SHA256Digest key;
	CharString name;
	RD::ShaderStage stage = RD::SHADER_STAGE_VERTEX;
	/// Weak reference to the library; allows cache lookup without preventing cleanup.
	std::weak_ptr<MDLibrary> library;

	/// Notify the cache that this entry is no longer needed.
	void notify_free() const;

	ShaderCacheEntry(RenderingDeviceDriverMetal &p_owner, SHA256Digest p_key) :
			owner(p_owner), key(p_key) {
	}
	~ShaderCacheEntry() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDShader {
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

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDComputeShader final : public MDShader {
public:
	MTLSize local = {};

	std::shared_ptr<MDLibrary> kernel;

	MDComputeShader(CharString p_name, Vector<UniformSet> p_sets, bool p_uses_argument_buffers, std::shared_ptr<MDLibrary> p_kernel);
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDRenderShader final : public MDShader {
public:
	bool needs_view_mask_buffer = false;

	std::shared_ptr<MDLibrary> vert;
	std::shared_ptr<MDLibrary> frag;

	MDRenderShader(CharString p_name,
			Vector<UniformSet> p_sets,
			bool p_needs_view_mask_buffer,
			bool p_uses_argument_buffers,
			std::shared_ptr<MDLibrary> p_vert, std::shared_ptr<MDLibrary> p_frag);
};

#pragma mark - Uniform Set

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

typedef LocalVector<MTLResourceUnsafe> ResourceVector;
typedef HashMap<StageResourceUsage, ResourceVector> ResourceUsageMap;

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

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDUniformSet {
public:
	id<MTLBuffer> arg_buffer = nil;
	Vector<uint8_t> arg_buffer_data; // Stored for dynamic uniform sets.
	ResourceUsageMap usage_to_resources; // Used by Metal 3 for resource tracking.
	Vector<RDD::BoundUniform> uniforms;
};

#pragma mark - Pipeline Types

enum class MDPipelineType {
	None,
	Render,
	Compute,
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDPipeline {
public:
	MDPipelineType type;

	explicit MDPipeline(MDPipelineType p_type) :
			type(p_type) {}
	virtual ~MDPipeline() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDRenderPipeline final : public MDPipeline {
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
			API_AVAILABLE(macos(26.0), ios(26.0), tvos(26.0), visionos(26.0))
			_FORCE_INLINE_ void apply(id<MTL4RenderCommandEncoder> __unsafe_unretained p_enc) const {
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
			API_AVAILABLE(macos(26.0), ios(26.0), tvos(26.0), visionos(26.0))
			_FORCE_INLINE_ void apply(id<MTL4RenderCommandEncoder> __unsafe_unretained p_enc) const {
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
			API_AVAILABLE(macos(26.0), ios(26.0), tvos(26.0), visionos(26.0))
			_FORCE_INLINE_ void apply(id<MTL4RenderCommandEncoder> __unsafe_unretained p_enc) const {
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

		API_AVAILABLE(macos(26.0), ios(26.0), tvos(26.0), visionos(26.0))
		_FORCE_INLINE_ void apply(id<MTL4RenderCommandEncoder> __unsafe_unretained p_enc) const {
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

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDComputePipeline final : public MDPipeline {
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
