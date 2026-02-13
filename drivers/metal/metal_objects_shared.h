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

#include "metal_device_properties.h"
#include "metal_utils.h"
#include "pixel_formats.h"
#include "sha256_digest.h"

#include <CoreFoundation/CoreFoundation.h>
#include <memory>
#include <optional>

class RenderingDeviceDriverMetal;

using RDC = RenderingDeviceCommons;

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

	_FORCE_INLINE_ void set_color_format(uint32_t p_idx, MTL::PixelFormat p_fmt) { pixel_formats[p_idx] = p_fmt; }
	_FORCE_INLINE_ void set_depth_format(MTL::PixelFormat p_fmt) { pixel_formats[DEPTH_INDEX] = p_fmt; }
	_FORCE_INLINE_ void set_stencil_format(MTL::PixelFormat p_fmt) { pixel_formats[STENCIL_INDEX] = p_fmt; }
	_FORCE_INLINE_ MTL::PixelFormat depth_format() const { return (MTL::PixelFormat)pixel_formats[DEPTH_INDEX]; }
	_FORCE_INLINE_ MTL::PixelFormat stencil_format() const { return (MTL::PixelFormat)pixel_formats[STENCIL_INDEX]; }
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
		MTL::Buffer *buffer = nullptr;
		uint64_t gpu_address = 0;
		uint32_t offset = 0;

		_FORCE_INLINE_ bool is_valid() const { return ptr != nullptr; }
	};

private:
	MTL::Device *device = nullptr;
	LocalVector<MTL::Buffer *> buffers;
	LocalVector<uint32_t> heads;
	uint32_t current_segment = 0;
	uint32_t buffer_size = DEFAULT_BUFFER_SIZE;
	bool changed = false;

	_FORCE_INLINE_ uint32_t alloc_segment() {
		MTL::Buffer *buffer = device->newBuffer(buffer_size, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
		buffers.push_back(buffer);
		heads.push_back(0);
		changed = true;

		return buffers.size() - 1;
	}

public:
	MDRingBuffer() = default;

	MDRingBuffer(MTL::Device *p_device, uint32_t p_buffer_size = DEFAULT_BUFFER_SIZE) :
			device(p_device), buffer_size(p_buffer_size) {}

	~MDRingBuffer() {
		for (MTL::Buffer *buffer : buffers) {
			buffer->release();
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

		MTL::Buffer *buffer = buffers[current_segment];
		Allocation alloc;
		alloc.buffer = buffer;
		alloc.offset = aligned_head;
		alloc.ptr = static_cast<uint8_t *>(buffer->contents()) + aligned_head;
		if (__builtin_available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
			alloc.gpu_address = buffer->gpuAddress() + aligned_head;
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
	_FORCE_INLINE_ Span<MTL::Buffer *const> get_buffers() const {
		return Span<MTL::Buffer *const>(buffers.ptr(), buffers.size());
	}

	/// Returns the number of buffer segments currently allocated.
	_FORCE_INLINE_ uint32_t get_segment_count() const {
		return buffers.size();
	}
};

#pragma mark - Resource Factory

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDResourceFactory {
private:
	MTL::Device *device;
	PixelFormats &pixel_formats;
	uint32_t max_buffer_count;

	NS::SharedPtr<MTL::Function> new_func(NS::String *p_source, NS::String *p_name, NS::Error **p_error);
	NS::SharedPtr<MTL::Function> new_clear_vert_func(ClearAttKey &p_key);
	NS::SharedPtr<MTL::Function> new_clear_frag_func(ClearAttKey &p_key);
	const char *get_format_type_string(MTL::PixelFormat p_fmt) const;

	_FORCE_INLINE_ uint32_t get_vertex_buffer_index(uint32_t p_binding) {
		return (max_buffer_count - 1) - p_binding;
	}

public:
	NS::SharedPtr<MTL::RenderPipelineState> new_clear_pipeline_state(ClearAttKey &p_key, NS::Error **p_error);
	NS::SharedPtr<MTL::RenderPipelineState> new_empty_draw_pipeline_state(ClearAttKey &p_key, NS::Error **p_error);
	NS::SharedPtr<MTL::DepthStencilState> new_depth_stencil_state(bool p_use_depth, bool p_use_stencil);

	MDResourceFactory(MTL::Device *p_device, PixelFormats &p_pixel_formats, uint32_t p_max_buffer_count) :
			device(p_device), pixel_formats(p_pixel_formats), max_buffer_count(p_max_buffer_count) {}
	~MDResourceFactory() = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDResourceCache {
private:
	typedef HashMap<ClearAttKey, NS::SharedPtr<MTL::RenderPipelineState>> HashMap;
	std::unique_ptr<MDResourceFactory> resource_factory;
	HashMap clear_states;
	HashMap empty_draw_states;

	struct {
		NS::SharedPtr<MTL::DepthStencilState> all;
		NS::SharedPtr<MTL::DepthStencilState> depth_only;
		NS::SharedPtr<MTL::DepthStencilState> stencil_only;
		NS::SharedPtr<MTL::DepthStencilState> none;
	} clear_depth_stencil_state;

public:
	MTL::RenderPipelineState *get_clear_render_pipeline_state(ClearAttKey &p_key, NS::Error **p_error);
	MTL::RenderPipelineState *get_empty_draw_pipeline_state(ClearAttKey &p_key, NS::Error **p_error);
	MTL::DepthStencilState *get_depth_stencil_state(bool p_use_depth, bool p_use_stencil);

	explicit MDResourceCache(MTL::Device *p_device, PixelFormats &p_pixel_formats, uint32_t p_max_buffer_count) :
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
	Vector<MTL::Texture *> textures;

public:
	Size2i size;
	MDFrameBuffer(Vector<MTL::Texture *> p_textures, Size2i p_size) :
			textures(p_textures), size(p_size) {}
	MDFrameBuffer() {}

	/// Returns the texture at the given index.
	_ALWAYS_INLINE_ MTL::Texture *get_texture(uint32_t p_idx) const {
		return textures[p_idx];
	}

	/// Returns true if the texture at the given index is not nil.
	_ALWAYS_INLINE_ bool has_texture(uint32_t p_idx) const {
		return textures[p_idx] != nullptr;
	}

	/// Set the texture at the given index.
	_ALWAYS_INLINE_ void set_texture(uint32_t p_idx, MTL::Texture *p_texture) {
		textures.write[p_idx] = p_texture;
	}

	/// Unset or nil the texture at the given index.
	_ALWAYS_INLINE_ void unset_texture(uint32_t p_idx) {
		textures.write[p_idx] = nullptr;
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

namespace rid {

template <typename T>
_FORCE_INLINE_ T *get(RDD::ID p_id) {
	return reinterpret_cast<T *>(p_id.id);
}

template <typename T>
_FORCE_INLINE_ T *get(uint64_t p_id) {
	return reinterpret_cast<T *>(p_id);
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
	MTL::PixelFormat format = MTL::PixelFormatInvalid;
	MDAttachmentType type = MDAttachmentType::None;
	MTL::LoadAction loadAction = MTL::LoadActionDontCare;
	MTL::StoreAction storeAction = MTL::StoreActionDontCare;
	MTL::LoadAction stencilLoadAction = MTL::LoadActionDontCare;
	MTL::StoreAction stencilStoreAction = MTL::StoreActionDontCare;
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

	MTL::StoreAction getMTLStoreAction(MDSubpass const &p_subpass,
			bool p_is_rendering_entire_area,
			bool p_has_resolve,
			bool p_can_resolve,
			bool p_is_stencil) const;
	bool configureDescriptor(MTL::RenderPassAttachmentDescriptor *p_desc,
			PixelFormats &p_pf,
			MDSubpass const &p_subpass,
			MTL::Texture *p_attachment,
			bool p_is_rendering_entire_area,
			bool p_has_resolve,
			bool p_can_resolve,
			bool p_is_stencil) const {
		p_desc->setTexture(p_attachment);

		MTL::LoadAction load;
		if (!p_is_rendering_entire_area || !isFirstUseOf(p_subpass)) {
			load = MTL::LoadActionLoad;
		} else {
			load = p_is_stencil ? (MTL::LoadAction)stencilLoadAction : (MTL::LoadAction)loadAction;
		}

		p_desc->setLoadAction(load);

		MTL::PixelFormat mtlFmt = p_attachment->pixelFormat();
		bool isDepthFormat = p_pf.isDepthFormat(mtlFmt);
		bool isStencilFormat = p_pf.isStencilFormat(mtlFmt);
		if (isStencilFormat && !p_is_stencil && !isDepthFormat) {
			p_desc->setStoreAction(MTL::StoreActionDontCare);
		} else {
			p_desc->setStoreAction(getMTLStoreAction(p_subpass, p_is_rendering_entire_area, p_has_resolve, p_can_resolve, p_is_stencil));
		}

		return load == MTL::LoadActionClear;
	}

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

#pragma mark - Command Buffer Helpers

_FORCE_INLINE_ static MTL::Size MTLSizeFromVector3i(Vector3i p_size) {
	return MTL::Size{ (NS::UInteger)p_size.x, (NS::UInteger)p_size.y, (NS::UInteger)p_size.z };
}

_FORCE_INLINE_ static MTL::Origin MTLOriginFromVector3i(Vector3i p_origin) {
	return MTL::Origin{ (NS::UInteger)p_origin.x, (NS::UInteger)p_origin.y, (NS::UInteger)p_origin.z };
}

// Clamps the size so that the sum of the origin and size do not exceed the maximum size.
_FORCE_INLINE_ static MTL::Size clampMTLSize(MTL::Size p_size, MTL::Origin p_origin, MTL::Size p_max_size) {
	MTL::Size clamped;
	clamped.width = MIN(p_size.width, p_max_size.width - p_origin.x);
	clamped.height = MIN(p_size.height, p_max_size.height - p_origin.y);
	clamped.depth = MIN(p_size.depth, p_max_size.depth - p_origin.z);
	return clamped;
}

API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0))
_FORCE_INLINE_ static bool isArrayTexture(MTL::TextureType p_type) {
	return (p_type == MTL::TextureType3D ||
			p_type == MTL::TextureType2DArray ||
			p_type == MTL::TextureType2DMultisampleArray ||
			p_type == MTL::TextureType1DArray);
}

_FORCE_INLINE_ static bool operator==(MTL::Size p_a, MTL::Size p_b) {
	return p_a.width == p_b.width && p_a.height == p_b.height && p_a.depth == p_b.depth;
}

#pragma mark - Pipeline Stage Conversion

GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

_FORCE_INLINE_ static MTL::Stages convert_src_pipeline_stages_to_metal(BitField<RDD::PipelineStageBits> p_stages) {
	p_stages.clear_flag(RDD::PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	// BOTTOM_OF_PIPE or ALL_COMMANDS means "all prior work must complete".
	if (p_stages & (RDD::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT | RDD::PIPELINE_STAGE_ALL_COMMANDS_BIT)) {
		return MTL::StageAll;
	}

	MTL::Stages mtlStages = 0;

	// Vertex stage mappings.
	if (p_stages & (RDD::PIPELINE_STAGE_DRAW_INDIRECT_BIT | RDD::PIPELINE_STAGE_VERTEX_INPUT_BIT | RDD::PIPELINE_STAGE_VERTEX_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | RDD::PIPELINE_STAGE_GEOMETRY_SHADER_BIT)) {
		mtlStages |= MTL::StageVertex;
	}

	// Fragment stage mappings.
	// Includes resolve and clear_storage, which on Metal use the render pipeline.
	if (p_stages & (RDD::PIPELINE_STAGE_FRAGMENT_SHADER_BIT | RDD::PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT | RDD::PIPELINE_STAGE_RESOLVE_BIT | RDD::PIPELINE_STAGE_CLEAR_STORAGE_BIT)) {
		mtlStages |= MTL::StageFragment;
	}

	// Compute stage.
	if (p_stages & RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
		mtlStages |= MTL::StageDispatch;
	}

	// Blit stage (transfer operations).
	if (p_stages & RDD::PIPELINE_STAGE_COPY_BIT) {
		mtlStages |= MTL::StageBlit;
	}

	// ALL_GRAPHICS_BIT special case.
	if (p_stages & RDD::PIPELINE_STAGE_ALL_GRAPHICS_BIT) {
		mtlStages |= (MTL::StageVertex | MTL::StageFragment);
	}

	return mtlStages;
}

_FORCE_INLINE_ static MTL::Stages convert_dst_pipeline_stages_to_metal(BitField<RDD::PipelineStageBits> p_stages) {
	p_stages.clear_flag(RDD::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	// TOP_OF_PIPE or ALL_COMMANDS means "wait before any work starts".
	if (p_stages & (RDD::PIPELINE_STAGE_ALL_COMMANDS_BIT | RDD::PIPELINE_STAGE_TOP_OF_PIPE_BIT)) {
		return MTL::StageAll;
	}

	MTL::Stages mtlStages = 0;

	// Vertex stage mappings.
	if (p_stages & (RDD::PIPELINE_STAGE_DRAW_INDIRECT_BIT | RDD::PIPELINE_STAGE_VERTEX_INPUT_BIT | RDD::PIPELINE_STAGE_VERTEX_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | RDD::PIPELINE_STAGE_GEOMETRY_SHADER_BIT)) {
		mtlStages |= MTL::StageVertex;
	}

	// Fragment stage mappings.
	// Includes resolve and clear_storage, which on Metal use the render pipeline.
	if (p_stages & (RDD::PIPELINE_STAGE_FRAGMENT_SHADER_BIT | RDD::PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT | RDD::PIPELINE_STAGE_RESOLVE_BIT | RDD::PIPELINE_STAGE_CLEAR_STORAGE_BIT)) {
		mtlStages |= MTL::StageFragment;
	}

	// Compute stage.
	if (p_stages & RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
		mtlStages |= MTL::StageDispatch;
	}

	// Blit stage (transfer operations).
	if (p_stages & RDD::PIPELINE_STAGE_COPY_BIT) {
		mtlStages |= MTL::StageBlit;
	}

	// ALL_GRAPHICS_BIT special case.
	if (p_stages & RDD::PIPELINE_STAGE_ALL_GRAPHICS_BIT) {
		mtlStages |= (MTL::StageVertex | MTL::StageFragment);
	}

	return mtlStages;
}

GODOT_CLANG_WARNING_POP

#pragma mark - Command Buffer Base

enum class MDCommandBufferStateType {
	None,
	Render,
	Compute,
	Blit, // Only used by Metal 3
};

/// Base struct for render state shared between MTL3 and MTL4 implementations.
struct RenderStateBase {
	LocalVector<MTL::Viewport> viewports;
	LocalVector<MTL::ScissorRect> scissors;
	std::optional<Color> blend_constants;

	// clang-format off
	enum DirtyFlag : uint16_t {
		DIRTY_NONE     = 0,
		DIRTY_PIPELINE = 1 << 0,
		DIRTY_UNIFORMS = 1 << 1,
		DIRTY_PUSH     = 1 << 2,
		DIRTY_DEPTH    = 1 << 3,
		DIRTY_VERTEX   = 1 << 4,
		DIRTY_VIEWPORT = 1 << 5,
		DIRTY_SCISSOR  = 1 << 6,
		DIRTY_BLEND    = 1 << 7,
		DIRTY_RASTER   = 1 << 8,
		DIRTY_ALL      = (1 << 9) - 1,
	};
	// clang-format on
	BitField<DirtyFlag> dirty = DIRTY_NONE;
};

/// Abstract base class for Metal command buffers, shared between MTL3 and MTL4 implementations.
class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDCommandBufferBase {
	LocalVector<CFTypeRef> _retained_resources;

protected:
	// From RenderingDevice
	static constexpr uint32_t MAX_PUSH_CONSTANT_SIZE = 128;

	MDCommandBufferStateType type = MDCommandBufferStateType::None;

	uint8_t push_constant_data[MAX_PUSH_CONSTANT_SIZE];
	uint32_t push_constant_data_len = 0;
	uint32_t push_constant_binding = UINT32_MAX;

	::RenderingDeviceDriverMetal *device_driver = nullptr;

	void release_resources();

	/// Called when push constants are modified to mark the appropriate dirty flags.
	virtual void mark_push_constants_dirty() = 0;

	/// Returns a reference to the render state base for viewport/scissor/blend operations.
	virtual RenderStateBase &get_render_state_base() = 0;

	/// Returns the view count for the current subpass.
	virtual uint32_t get_current_view_count() const = 0;

	/// Accessors for render pass state.
	virtual MDRenderPass *get_render_pass() const = 0;
	virtual MDFrameBuffer *get_frame_buffer() const = 0;
	virtual const MDSubpass &get_current_subpass() const = 0;
	virtual LocalVector<RDD::RenderPassClearValue> &get_clear_values() = 0;
	virtual const Rect2i &get_render_area() const = 0;
	virtual void end_render_encoding() = 0;

	void _populate_vertices(simd::float4 *p_vertices, Size2i p_fb_size, VectorView<Rect2i> p_rects);
	uint32_t _populate_vertices(simd::float4 *p_vertices, uint32_t p_index, Rect2i const &p_rect, Size2i p_fb_size);
	void _end_render_pass();
	void _render_clear_render_area();

public:
	virtual ~MDCommandBufferBase() { release_resources(); }

	virtual void begin() = 0;
	virtual void commit() = 0;
	virtual void end() = 0;

	virtual void bind_pipeline(RDD::PipelineID p_pipeline) = 0;
	void encode_push_constant_data(RDD::ShaderID p_shader, VectorView<uint32_t> p_data);

	void retain_resource(CFTypeRef p_resource);

#pragma mark - Render Commands

	virtual void render_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) = 0;
	virtual void render_clear_attachments(VectorView<RDD::AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) = 0;
	void render_set_viewport(VectorView<Rect2i> p_viewports);
	void render_set_scissor(VectorView<Rect2i> p_scissors);
	void render_set_blend_constants(const Color &p_constants);
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
	virtual void clear_depth_stencil_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, float p_depth, uint8_t p_stencil, const RDD::TextureSubresourceRange &p_subresources) = 0;
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
			VectorView<RDD::TextureBarrier> p_texture_barriers,
			VectorView<RDD::AccelerationStructureBarrier> p_acceleration_structure_barriers) = 0;

#pragma mark - Debugging

	virtual void begin_label(const char *p_label_name, const Color &p_color) = 0;
	virtual void end_label() = 0;
};

#pragma mark - Uniform Types

struct API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) UniformInfo {
	uint32_t binding;
	BitField<RDD::ShaderStage> active_stages;
	MTL::DataType dataType = MTL::DataTypeNone;
	MTL::BindingAccess access = MTL::BindingAccessReadOnly;
	MTL::ResourceUsage usage = 0;
	MTL::TextureType textureType = MTL::TextureType2D;
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

class MDLibrary; // Forward declaration for C++ code
struct ShaderCacheEntry; // Forward declaration for C++ code

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
	NS::SharedPtr<NS::String> _original_source = nullptr;
#endif

	MDLibrary(ShaderCacheEntry *p_entry
#ifdef DEV_ENABLED
			,
			NS::String *p_source
#endif
	);

public:
	virtual ~MDLibrary();

	virtual MTL::Library *get_library() = 0;
	virtual NS::Error *get_error() = 0;
	virtual void set_label(NS::String *p_label);
#ifdef DEV_ENABLED
	NS::String *get_original_source() const { return _original_source.get(); }
#endif

	static std::shared_ptr<MDLibrary> create(ShaderCacheEntry *p_entry,
			MTL::Device *p_device,
			NS::String *p_source,
			MTL::CompileOptions *p_options,
			ShaderLoadStrategy p_strategy);

	static std::shared_ptr<MDLibrary> create(ShaderCacheEntry *p_entry,
			MTL::Device *p_device,
#ifdef DEV_ENABLED
			NS::String *p_source,
#endif
			dispatch_data_t p_data);
};

/// A cache entry for a Metal shader library.
struct ShaderCacheEntry {
	RenderingDeviceDriverMetal &owner;
	/// A hash of the Metal shader source code.
	SHA256Digest key;
	CharString name;
	RDC::ShaderStage stage = RDC::SHADER_STAGE_VERTEX;
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
	MTL::Size local = {};

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
	VertexRead = (MTL::ResourceUsageRead << RDD::SHADER_STAGE_VERTEX * 2),
	VertexWrite = (MTL::ResourceUsageWrite << RDD::SHADER_STAGE_VERTEX * 2),
	FragmentRead = (MTL::ResourceUsageRead << RDD::SHADER_STAGE_FRAGMENT * 2),
	FragmentWrite = (MTL::ResourceUsageWrite << RDD::SHADER_STAGE_FRAGMENT * 2),
	TesselationControlRead = (MTL::ResourceUsageRead << RDD::SHADER_STAGE_TESSELATION_CONTROL * 2),
	TesselationControlWrite = (MTL::ResourceUsageWrite << RDD::SHADER_STAGE_TESSELATION_CONTROL * 2),
	TesselationEvaluationRead = (MTL::ResourceUsageRead << RDD::SHADER_STAGE_TESSELATION_EVALUATION * 2),
	TesselationEvaluationWrite = (MTL::ResourceUsageWrite << RDD::SHADER_STAGE_TESSELATION_EVALUATION * 2),
	ComputeRead = (MTL::ResourceUsageRead << RDD::SHADER_STAGE_COMPUTE * 2),
	ComputeWrite = (MTL::ResourceUsageWrite << RDD::SHADER_STAGE_COMPUTE * 2),
};

typedef LocalVector<MTL::Resource *> ResourceVector;
typedef HashMap<StageResourceUsage, ResourceVector> ResourceUsageMap;

_FORCE_INLINE_ StageResourceUsage &operator|=(StageResourceUsage &p_a, uint32_t p_b) {
	p_a = StageResourceUsage(uint32_t(p_a) | p_b);
	return p_a;
}

_FORCE_INLINE_ StageResourceUsage stage_resource_usage(RDC::ShaderStage p_stage, MTL::ResourceUsage p_usage) {
	return StageResourceUsage(p_usage << (p_stage * 2));
}

_FORCE_INLINE_ MTL::ResourceUsage resource_usage_for_stage(StageResourceUsage p_usage, RDC::ShaderStage p_stage) {
	return MTL::ResourceUsage((p_usage >> (p_stage * 2)) & 0b11);
}

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDUniformSet {
public:
	NS::SharedPtr<MTL::Buffer> arg_buffer;
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
	NS::SharedPtr<MTL::RenderPipelineState> state;
	NS::SharedPtr<MTL::DepthStencilState> depth_stencil;
	uint32_t push_constant_size = 0;
	uint32_t push_constant_stages_mask = 0;
	SampleCount sample_count = SampleCount1;

	struct {
		MTL::CullMode cull_mode = MTL::CullModeNone;
		MTL::TriangleFillMode fill_mode = MTL::TriangleFillModeFill;
		MTL::DepthClipMode clip_mode = MTL::DepthClipModeClip;
		MTL::Winding winding = MTL::WindingClockwise;
		MTL::PrimitiveType render_primitive = MTL::PrimitiveTypePoint;

		struct {
			bool enabled = false;
		} depth_test;

		struct {
			bool enabled = false;
			float depth_bias = 0.0;
			float slope_scale = 0.0;
			float clamp = 0.0;

			template <typename T>
			_FORCE_INLINE_ void apply(T *p_enc) const {
				if (!enabled) {
					return;
				}
				p_enc->setDepthBias(depth_bias, slope_scale, clamp);
			}
		} depth_bias;

		struct {
			bool enabled = false;
			uint32_t front_reference = 0;
			uint32_t back_reference = 0;

			template <typename T>
			_FORCE_INLINE_ void apply(T *p_enc) const {
				if (!enabled) {
					return;
				}
				p_enc->setStencilReferenceValues(front_reference, back_reference);
			}
		} stencil;

		struct {
			bool enabled = false;
			float r = 0.0;
			float g = 0.0;
			float b = 0.0;
			float a = 0.0;

			template <typename T>
			_FORCE_INLINE_ void apply(T *p_enc) const {
				p_enc->setBlendColor(r, g, b, a);
			}
		} blend;

		template <typename T>
		_FORCE_INLINE_ void apply(T *p_enc) const {
			p_enc->setCullMode(cull_mode);
			p_enc->setTriangleFillMode(fill_mode);
			p_enc->setDepthClipMode(clip_mode);
			p_enc->setFrontFacingWinding(winding);
			depth_bias.apply(p_enc);
			stencil.apply(p_enc);
			blend.apply(p_enc);
		}

	} raster_state;

	MDRenderShader *shader = nullptr;

	MDRenderPipeline() :
			MDPipeline(MDPipelineType::Render) {}
	~MDRenderPipeline() final = default;
};

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0), visionos(2.0)) MDComputePipeline final : public MDPipeline {
public:
	NS::SharedPtr<MTL::ComputePipelineState> state;
	struct {
		MTL::Size local = {};
	} compute_state;

	MDComputeShader *shader = nullptr;

	explicit MDComputePipeline(NS::SharedPtr<MTL::ComputePipelineState> p_state) :
			MDPipeline(MDPipelineType::Compute), state(std::move(p_state)) {}
	~MDComputePipeline() final = default;
};
