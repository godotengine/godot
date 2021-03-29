/*************************************************************************/
/*  rendering_device_vulkan.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef RENDERING_DEVICE_VULKAN_H
#define RENDERING_DEVICE_VULKAN_H

#include "core/os/thread_safe.h"
#include "core/templates/local_vector.h"
#include "core/templates/oa_hash_map.h"
#include "core/templates/rid_owner.h"
#include "servers/rendering/rendering_device.h"

#ifdef DEBUG_ENABLED
#ifndef _DEBUG
#define _DEBUG
#endif
#endif
#include "vk_mem_alloc.h"

#include <vulkan/vulkan.h>

class VulkanContext;

class RenderingDeviceVulkan : public RenderingDevice {
	_THREAD_SAFE_CLASS_

	// Miscellaneous tables that map
	// our enums to enums used
	// by vulkan.

	VkPhysicalDeviceLimits limits;
	static const VkFormat vulkan_formats[DATA_FORMAT_MAX];
	static const char *named_formats[DATA_FORMAT_MAX];
	static const VkCompareOp compare_operators[COMPARE_OP_MAX];
	static const VkStencilOp stencil_operations[STENCIL_OP_MAX];
	static const VkSampleCountFlagBits rasterization_sample_count[TEXTURE_SAMPLES_MAX];
	static const VkLogicOp logic_operations[RenderingDevice::LOGIC_OP_MAX];
	static const VkBlendFactor blend_factors[RenderingDevice::BLEND_FACTOR_MAX];
	static const VkBlendOp blend_operations[RenderingDevice::BLEND_OP_MAX];
	static const VkSamplerAddressMode address_modes[SAMPLER_REPEAT_MODE_MAX];
	static const VkBorderColor sampler_border_colors[SAMPLER_BORDER_COLOR_MAX];
	static const VkImageType vulkan_image_type[TEXTURE_TYPE_MAX];

	// Functions used for format
	// validation, and ensures the
	// user passes valid data.

	static int get_format_vertex_size(DataFormat p_format);
	static uint32_t get_image_format_pixel_size(DataFormat p_format);
	static void get_compressed_image_format_block_dimensions(DataFormat p_format, uint32_t &r_w, uint32_t &r_h);
	uint32_t get_compressed_image_format_block_byte_size(DataFormat p_format);
	static uint32_t get_compressed_image_format_pixel_rshift(DataFormat p_format);
	static uint32_t get_image_format_required_size(DataFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_depth, uint32_t p_mipmaps, uint32_t *r_blockw = nullptr, uint32_t *r_blockh = nullptr, uint32_t *r_depth = nullptr);
	static uint32_t get_image_required_mipmaps(uint32_t p_width, uint32_t p_height, uint32_t p_depth);
	static bool format_has_stencil(DataFormat p_format);

	/***************************/
	/**** ID INFRASTRUCTURE ****/
	/***************************/

	enum IDType {
		ID_TYPE_FRAMEBUFFER_FORMAT,
		ID_TYPE_VERTEX_FORMAT,
		ID_TYPE_DRAW_LIST,
		ID_TYPE_SPLIT_DRAW_LIST,
		ID_TYPE_COMPUTE_LIST,
		ID_TYPE_MAX,
		ID_BASE_SHIFT = 58 //5 bits for ID types
	};

	VkDevice device = VK_NULL_HANDLE;

	Map<RID, Set<RID>> dependency_map; //IDs to IDs that depend on it
	Map<RID, Set<RID>> reverse_dependency_map; //same as above, but in reverse

	void _add_dependency(RID p_id, RID p_depends_on);
	void _free_dependencies(RID p_id);

	/*****************/
	/**** TEXTURE ****/
	/*****************/

	// In Vulkan, the concept of textures does not exist,
	// instead there is the image (the memory pretty much,
	// the view (how the memory is interpreted) and the
	// sampler (how it's sampled from the shader).
	//
	// Texture here includes the first two stages, but
	// It's possible to create textures sharing the image
	// but with different views. The main use case for this
	// is textures that can be read as both SRGB/Linear,
	// or slices of a texture (a mipmap, a layer, a 3D slice)
	// for a framebuffer to render into it.

	struct Texture {
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = nullptr;
		VmaAllocationInfo allocation_info;
		VkImageView view = VK_NULL_HANDLE;

		TextureType type;
		DataFormat format;
		TextureSamples samples;
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t depth = 0;
		uint32_t layers = 0;
		uint32_t mipmaps = 0;
		uint32_t usage_flags = 0;
		uint32_t base_mipmap = 0;
		uint32_t base_layer = 0;

		Vector<DataFormat> allowed_shared_formats;

		VkImageLayout layout;

		uint64_t used_in_frame = 0;
		bool used_in_transfer = false;
		bool used_in_raster = false;
		bool used_in_compute = false;

		uint32_t read_aspect_mask = 0;
		uint32_t barrier_aspect_mask = 0;
		bool bound = false; //bound to framebffer
		RID owner;
	};

	RID_Owner<Texture, true> texture_owner;
	uint32_t texture_upload_region_size_px = 0;

	Vector<uint8_t> _texture_get_data_from_image(Texture *tex, VkImage p_image, VmaAllocation p_allocation, uint32_t p_layer, bool p_2d = false);

	/*****************/
	/**** SAMPLER ****/
	/*****************/

	RID_Owner<VkSampler> sampler_owner;

	/***************************/
	/**** BUFFER MANAGEMENT ****/
	/***************************/

	// These are temporary buffers on CPU memory that hold
	// the information until the CPU fetches it and places it
	// either on GPU buffers, or images (textures). It ensures
	// updates are properly synchronized with whathever the
	// GPU is doing.
	//
	// The logic here is as follows, only 3 of these
	// blocks are created at the beginning (one per frame)
	// they can each belong to a frame (assigned to current when
	// used) and they can only be reused after the same frame is
	// recycled.
	//
	// When CPU requires to allocate more than what is available,
	// more of these buffers are created. If a limit is reached,
	// then a fence will ensure will wait for blocks allocated
	// in previous frames are processed. If that fails, then
	// another fence will ensure everything pending for the current
	// frame is processed (effectively stalling).
	//
	// See the comments in the code to understand better how it works.

	struct StagingBufferBlock {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = nullptr;
		uint64_t frame_used = 0;
		uint32_t fill_amount = 0;
	};

	Vector<StagingBufferBlock> staging_buffer_blocks;
	int staging_buffer_current = 0;
	uint32_t staging_buffer_block_size = 0;
	uint64_t staging_buffer_max_size = 0;
	bool staging_buffer_used = false;

	Error _staging_buffer_allocate(uint32_t p_amount, uint32_t p_required_align, uint32_t &r_alloc_offset, uint32_t &r_alloc_size, bool p_can_segment = true, bool p_on_draw_command_buffer = false);
	Error _insert_staging_block();

	struct Buffer {
		uint32_t size = 0;
		uint32_t usage = 0;
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = nullptr;
		VkDescriptorBufferInfo buffer_info; //used for binding
		Buffer() {
		}
	};

	Error _buffer_allocate(Buffer *p_buffer, uint32_t p_size, uint32_t p_usage, VmaMemoryUsage p_mapping);
	Error _buffer_free(Buffer *p_buffer);
	Error _buffer_update(Buffer *p_buffer, size_t p_offset, const uint8_t *p_data, size_t p_data_size, bool p_use_draw_command_buffer = false, uint32_t p_required_align = 32);

	void _full_barrier(bool p_sync_with_draw);
	void _memory_barrier(VkPipelineStageFlags p_src_stage_mask, VkPipelineStageFlags p_dst_stage_mask, VkAccessFlags p_src_access, VkAccessFlags p_dst_sccess, bool p_sync_with_draw);
	void _buffer_memory_barrier(VkBuffer buffer, uint64_t p_from, uint64_t p_size, VkPipelineStageFlags p_src_stage_mask, VkPipelineStageFlags p_dst_stage_mask, VkAccessFlags p_src_access, VkAccessFlags p_dst_sccess, bool p_sync_with_draw);

	/*********************/
	/**** FRAMEBUFFER ****/
	/*********************/

	// In Vulkan, framebuffers work similar to how they
	// do in OpenGL, with the exception that
	// the "format" (vkRenderPass) is not dynamic
	// and must be more or less the same as the one
	// used for the render pipelines.

	struct FramebufferFormatKey {
		Vector<AttachmentFormat> attachments;
		bool operator<(const FramebufferFormatKey &p_key) const {
			int as = attachments.size();
			int bs = p_key.attachments.size();
			if (as != bs) {
				return as < bs;
			}

			const AttachmentFormat *af_a = attachments.ptr();
			const AttachmentFormat *af_b = p_key.attachments.ptr();
			for (int i = 0; i < as; i++) {
				const AttachmentFormat &a = af_a[i];
				const AttachmentFormat &b = af_b[i];
				if (a.format != b.format) {
					return a.format < b.format;
				}
				if (a.samples != b.samples) {
					return a.samples < b.samples;
				}
				if (a.usage_flags != b.usage_flags) {
					return a.usage_flags < b.usage_flags;
				}
			}

			return false; //equal
		}
	};

	VkRenderPass _render_pass_create(const Vector<AttachmentFormat> &p_format, InitialAction p_initial_action, FinalAction p_final_action, InitialAction p_initial_depth_action, FinalAction p_final_depthcolor_action, int *r_color_attachment_count = nullptr);

	// This is a cache and it's never freed, it ensures
	// IDs for a given format are always unique.
	Map<FramebufferFormatKey, FramebufferFormatID> framebuffer_format_cache;
	struct FramebufferFormat {
		const Map<FramebufferFormatKey, FramebufferFormatID>::Element *E;
		VkRenderPass render_pass = VK_NULL_HANDLE; //here for constructing shaders, never used, see section (7.2. Render Pass Compatibility from Vulkan spec)
		int color_attachments = 0; //used for pipeline validation
		TextureSamples samples;
	};

	Map<FramebufferFormatID, FramebufferFormat> framebuffer_formats;

	struct Framebuffer {
		FramebufferFormatID format_id = 0;
		struct VersionKey {
			InitialAction initial_color_action;
			FinalAction final_color_action;
			InitialAction initial_depth_action;
			FinalAction final_depth_action;
			bool operator<(const VersionKey &p_key) const {
				if (initial_color_action == p_key.initial_color_action) {
					if (final_color_action == p_key.final_color_action) {
						if (initial_depth_action == p_key.initial_depth_action) {
							return final_depth_action < p_key.final_depth_action;
						} else {
							return initial_depth_action < p_key.initial_depth_action;
						}
					} else {
						return final_color_action < p_key.final_color_action;
					}
				} else {
					return initial_color_action < p_key.initial_color_action;
				}
			}
		};

		uint32_t storage_mask = 0;
		Vector<RID> texture_ids;

		struct Version {
			VkFramebuffer framebuffer = VK_NULL_HANDLE;
			VkRenderPass render_pass = VK_NULL_HANDLE; //this one is owned
		};

		Map<VersionKey, Version> framebuffers;
		Size2 size;
	};

	RID_Owner<Framebuffer, true> framebuffer_owner;

	/***********************/
	/**** VERTEX BUFFER ****/
	/***********************/

	// Vertex buffers in Vulkan are similar to how
	// they work in OpenGL, except that instead of
	// an attribute index, there is a buffer binding
	// index (for binding the buffers in real-time)
	// and a location index (what is used in the shader).
	//
	// This mapping is done here internally, and it's not
	// exposed.

	RID_Owner<Buffer, true> vertex_buffer_owner;

	struct VertexDescriptionKey {
		Vector<VertexAttribute> vertex_formats;
		bool operator==(const VertexDescriptionKey &p_key) const {
			int vdc = vertex_formats.size();
			int vdck = p_key.vertex_formats.size();

			if (vdc != vdck) {
				return false;
			} else {
				const VertexAttribute *a_ptr = vertex_formats.ptr();
				const VertexAttribute *b_ptr = p_key.vertex_formats.ptr();
				for (int i = 0; i < vdc; i++) {
					const VertexAttribute &a = a_ptr[i];
					const VertexAttribute &b = b_ptr[i];

					if (a.location != b.location) {
						return false;
					}
					if (a.offset != b.offset) {
						return false;
					}
					if (a.format != b.format) {
						return false;
					}
					if (a.stride != b.stride) {
						return false;
					}
					if (a.frequency != b.frequency) {
						return false;
					}
				}
				return true; //they are equal
			}
		}

		uint32_t hash() const {
			int vdc = vertex_formats.size();
			uint32_t h = hash_djb2_one_32(vdc);
			const VertexAttribute *ptr = vertex_formats.ptr();
			for (int i = 0; i < vdc; i++) {
				const VertexAttribute &vd = ptr[i];
				h = hash_djb2_one_32(vd.location, h);
				h = hash_djb2_one_32(vd.offset, h);
				h = hash_djb2_one_32(vd.format, h);
				h = hash_djb2_one_32(vd.stride, h);
				h = hash_djb2_one_32(vd.frequency, h);
			}
			return h;
		}
	};

	struct VertexDescriptionHash {
		static _FORCE_INLINE_ uint32_t hash(const VertexDescriptionKey &p_key) {
			return p_key.hash();
		}
	};

	// This is a cache and it's never freed, it ensures that
	// ID used for a specific format always remain the same.
	HashMap<VertexDescriptionKey, VertexFormatID, VertexDescriptionHash> vertex_format_cache;

	struct VertexDescriptionCache {
		Vector<VertexAttribute> vertex_formats;
		VkVertexInputBindingDescription *bindings = nullptr;
		VkVertexInputAttributeDescription *attributes = nullptr;
		VkPipelineVertexInputStateCreateInfo create_info;
	};

	Map<VertexFormatID, VertexDescriptionCache> vertex_formats;

	struct VertexArray {
		RID buffer;
		VertexFormatID description = 0;
		int vertex_count = 0;
		uint32_t max_instances_allowed = 0;

		Vector<VkBuffer> buffers; //not owned, just referenced
		Vector<VkDeviceSize> offsets;
	};

	RID_Owner<VertexArray, true> vertex_array_owner;

	struct IndexBuffer : public Buffer {
		uint32_t max_index = 0; //used for validation
		uint32_t index_count = 0;
		VkIndexType index_type = VK_INDEX_TYPE_NONE_NV;
		bool supports_restart_indices = false;
	};

	RID_Owner<IndexBuffer, true> index_buffer_owner;

	struct IndexArray {
		uint32_t max_index = 0; //remember the maximum index here too, for validation
		VkBuffer buffer; //not owned, inherited from index buffer
		uint32_t offset = 0;
		uint32_t indices = 0;
		VkIndexType index_type = VK_INDEX_TYPE_NONE_NV;
		bool supports_restart_indices = false;
	};

	RID_Owner<IndexArray, true> index_array_owner;

	/****************/
	/**** SHADER ****/
	/****************/

	// Vulkan specifies a really complex behavior for the application
	// in order to tell when descriptor sets need to be re-bound (or not).
	// "When binding a descriptor set (see Descriptor Set Binding) to set
	//  number N, if the previously bound descriptor sets for sets zero
	//  through N-1 were all bound using compatible pipeline layouts,
	//  then performing this binding does not disturb any of the lower numbered sets.
	//  If, additionally, the previous bound descriptor set for set N was
	//  bound using a pipeline layout compatible for set N, then the bindings
	//  in sets numbered greater than N are also not disturbed."
	// As a result, we need to figure out quickly when something is no longer "compatible".
	// in order to avoid costly rebinds.

	enum {
		MAX_UNIFORM_SETS = 16
	};

	struct UniformInfo {
		UniformType type = UniformType::UNIFORM_TYPE_MAX;
		int binding = 0;
		uint32_t stages = 0;
		int length = 0; //size of arrays (in total elements), or ubos (in bytes * total elements)

		bool operator!=(const UniformInfo &p_info) const {
			return (binding != p_info.binding || type != p_info.type || stages != p_info.stages || length != p_info.length);
		}

		bool operator<(const UniformInfo &p_info) const {
			if (binding != p_info.binding) {
				return binding < p_info.binding;
			}
			if (type != p_info.type) {
				return type < p_info.type;
			}
			if (stages != p_info.stages) {
				return stages < p_info.stages;
			}
			return length < p_info.length;
		}
	};

	struct UniformSetFormat {
		Vector<UniformInfo> uniform_info;
		bool operator<(const UniformSetFormat &p_format) const {
			uint32_t size = uniform_info.size();
			uint32_t psize = p_format.uniform_info.size();

			if (size != psize) {
				return size < psize;
			}

			const UniformInfo *infoptr = uniform_info.ptr();
			const UniformInfo *pinfoptr = p_format.uniform_info.ptr();

			for (uint32_t i = 0; i < size; i++) {
				if (infoptr[i] != pinfoptr[i]) {
					return infoptr[i] < pinfoptr[i];
				}
			}

			return false;
		}
	};

	// Always grows, never shrinks, ensuring unique IDs, but we assume
	// the amount of formats will never be a problem, as the amount of shaders
	// in a game is limited.
	Map<UniformSetFormat, uint32_t> uniform_set_format_cache;

	// Shaders in Vulkan are just pretty much
	// precompiled blocks of SPIR-V bytecode. They
	// are most likely not really compiled to host
	// assembly until a pipeline is created.
	//
	// When supplying the shaders, this implementation
	// will use the reflection abilities of glslang to
	// understand and cache everything required to
	// create and use the descriptor sets (Vulkan's
	// biggest pain).
	//
	// Additionally, hashes are created for every set
	// to do quick validation and ensuring the user
	// does not submit something invalid.

	struct Shader {
		struct Set {
			Vector<UniformInfo> uniform_info;
			VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
		};

		uint32_t vertex_input_mask = 0; //inputs used, this is mostly for validation
		int fragment_outputs = 0;

		struct PushConstant {
			uint32_t push_constant_size = 0;
			uint32_t push_constants_vk_stage = 0;
		};

		PushConstant push_constant;

		uint32_t compute_local_size[3] = { 0, 0, 0 };

		bool is_compute = false;
		int max_output = 0;
		Vector<Set> sets;
		Vector<uint32_t> set_formats;
		Vector<VkPipelineShaderStageCreateInfo> pipeline_stages;
		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	};

	String _shader_uniform_debug(RID p_shader, int p_set = -1);

	RID_Owner<Shader, true> shader_owner;

	/******************/
	/**** UNIFORMS ****/
	/******************/

	// Descriptor sets require allocation from a pool.
	// The documentation on how to use pools properly
	// is scarce, and the documentation is strange.
	//
	// Basically, you can mix and match pools as you
	// like, but you'll run into fragmentation issues.
	// Because of this, the recommended approach is to
	// create a a pool for every descriptor set type,
	// as this prevents fragmentation.
	//
	// This is implemented here as a having a list of
	// pools (each can contain up to 64 sets) for each
	// set layout. The amount of sets for each type
	// is used as the key.

	enum {
		MAX_DESCRIPTOR_POOL_ELEMENT = 65535
	};

	struct DescriptorPoolKey {
		union {
			struct {
				uint16_t uniform_type[UNIFORM_TYPE_MAX]; // Using 16 bits because, for sending arrays, each element is a pool set.
			};
			struct {
				uint64_t key1;
				uint64_t key2;
				uint64_t key3;
			};
		};
		bool operator<(const DescriptorPoolKey &p_key) const {
			if (key1 != p_key.key1) {
				return key1 < p_key.key1;
			}
			if (key2 != p_key.key2) {
				return key2 < p_key.key2;
			}

			return key3 < p_key.key3;
		}
		DescriptorPoolKey() {
			key1 = 0;
			key2 = 0;
			key3 = 0;
		}
	};

	struct DescriptorPool {
		VkDescriptorPool pool;
		uint32_t usage;
	};

	Map<DescriptorPoolKey, Set<DescriptorPool *>> descriptor_pools;
	uint32_t max_descriptors_per_pool = 0;

	DescriptorPool *_descriptor_pool_allocate(const DescriptorPoolKey &p_key);
	void _descriptor_pool_free(const DescriptorPoolKey &p_key, DescriptorPool *p_pool);

	RID_Owner<Buffer, true> uniform_buffer_owner;
	RID_Owner<Buffer, true> storage_buffer_owner;

	//texture buffer needs a view
	struct TextureBuffer {
		Buffer buffer;
		VkBufferView view = VK_NULL_HANDLE;
	};

	RID_Owner<TextureBuffer, true> texture_buffer_owner;

	// This structure contains the descriptor set. They _need_ to be allocated
	// for a shader (and will be erased when this shader is erased), but should
	// work for other shaders as long as the hash matches. This covers using
	// them in shader variants.
	//
	// Keep also in mind that you can share buffers between descriptor sets, so
	// the above restriction is not too serious.

	struct UniformSet {
		uint32_t format = 0;
		RID shader_id;
		uint32_t shader_set = 0;
		DescriptorPool *pool = nullptr;
		DescriptorPoolKey pool_key;
		VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
		//VkPipelineLayout pipeline_layout; //not owned, inherited from shader
		struct AttachableTexture {
			uint32_t bind;
			RID texture;
		};

		LocalVector<AttachableTexture> attachable_textures; //used for validation
		Vector<Texture *> mutable_sampled_textures; //used for layout change
		Vector<Texture *> mutable_storage_textures; //used for layout change
	};

	RID_Owner<UniformSet, true> uniform_set_owner;

	/*******************/
	/**** PIPELINES ****/
	/*******************/

	// Render pipeline contains ALL the
	// information required for drawing.
	// This includes all the rasterizer state
	// as well as shader used, framebuffer format,
	// etc.
	// While the pipeline is just a single object
	// (VkPipeline) a lot of values are also saved
	// here to do validation (vulkan does none by
	// default) and warn the user if something
	// was not supplied as intended.

	struct RenderPipeline {
		//Cached values for validation
#ifdef DEBUG_ENABLED
		struct Validation {
			FramebufferFormatID framebuffer_format = 0;
			uint32_t dynamic_state = 0;
			VertexFormatID vertex_format = 0;
			bool uses_restart_indices = false;
			uint32_t primitive_minimum = 0;
			uint32_t primitive_divisor = 0;
		} validation;
#endif
		//Actual pipeline
		RID shader;
		Vector<uint32_t> set_formats;
		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE; // not owned, needed for push constants
		VkPipeline pipeline = VK_NULL_HANDLE;
		uint32_t push_constant_size = 0;
		uint32_t push_constant_stages = 0;
	};

	RID_Owner<RenderPipeline, true> render_pipeline_owner;

	struct ComputePipeline {
		RID shader;
		Vector<uint32_t> set_formats;
		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE; // not owned, needed for push constants
		VkPipeline pipeline = VK_NULL_HANDLE;
		uint32_t push_constant_size = 0;
		uint32_t push_constant_stages = 0;
		uint32_t local_group_size[3] = { 0, 0, 0 };
	};

	RID_Owner<ComputePipeline, true> compute_pipeline_owner;

	/*******************/
	/**** DRAW LIST ****/
	/*******************/

	// Draw list contains both the command buffer
	// used for drawing as well as a LOT of
	// information used for validation. This
	// validation is cheap so most of it can
	// also run in release builds.

	// When using split command lists, this is
	// implemented internally using secondary command
	// buffers. As they can be created in threads,
	// each needs it's own command pool.

	struct SplitDrawListAllocator {
		VkCommandPool command_pool = VK_NULL_HANDLE;
		Vector<VkCommandBuffer> command_buffers; //one for each frame
	};

	Vector<SplitDrawListAllocator> split_draw_list_allocators;

	struct DrawList {
		VkCommandBuffer command_buffer = VK_NULL_HANDLE; // If persistent, this is owned, otherwise it's shared with the ringbuffer.
		Rect2i viewport;

		struct SetState {
			uint32_t pipeline_expected_format = 0;
			uint32_t uniform_set_format = 0;
			VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
			RID uniform_set;
			bool bound = false;
		};

		struct State {
			SetState sets[MAX_UNIFORM_SETS];
			uint32_t set_count = 0;
			RID pipeline;
			RID pipeline_shader;
			VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
			RID vertex_array;
			RID index_array;
			uint32_t pipeline_push_constant_stages = 0;
		} state;

#ifdef DEBUG_ENABLED
		struct Validation {
			bool active = true; // Means command buffer was not closed, so you can keep adding things.
			FramebufferFormatID framebuffer_format = INVALID_ID;
			// Actual render pass values.
			uint32_t dynamic_state = 0;
			VertexFormatID vertex_format = INVALID_ID;
			uint32_t vertex_array_size = 0;
			uint32_t vertex_max_instances_allowed = 0xFFFFFFFF;
			bool index_buffer_uses_restart_indices = false;
			uint32_t index_array_size = 0;
			uint32_t index_array_max_index = 0;
			uint32_t index_array_offset = 0;
			Vector<uint32_t> set_formats;
			Vector<bool> set_bound;
			Vector<RID> set_rids;
			// Last pipeline set values.
			bool pipeline_active = false;
			uint32_t pipeline_dynamic_state = 0;
			VertexFormatID pipeline_vertex_format = INVALID_ID;
			RID pipeline_shader;
			uint32_t invalid_set_from = 0;
			bool pipeline_uses_restart_indices = false;
			uint32_t pipeline_primitive_divisor = 0;
			uint32_t pipeline_primitive_minimum = 0;
			Vector<uint32_t> pipeline_set_formats;
			uint32_t pipeline_push_constant_size = 0;
			bool pipeline_push_constant_supplied = false;
		} validation;
#else
		struct Validation {
			uint32_t vertex_array_size = 0;
			uint32_t index_array_size = 0;
			uint32_t index_array_offset;
		} validation;
#endif
	};

	DrawList *draw_list = nullptr; // One for regular draw lists, multiple for split.
	uint32_t draw_list_count = 0;
	bool draw_list_split = false;
	Vector<RID> draw_list_bound_textures;
	Vector<RID> draw_list_storage_textures;
	bool draw_list_unbind_color_textures = false;
	bool draw_list_unbind_depth_textures = false;

	void _draw_list_insert_clear_region(DrawList *draw_list, Framebuffer *framebuffer, Point2i viewport_offset, Point2i viewport_size, bool p_clear_color, const Vector<Color> &p_clear_colors, bool p_clear_depth, float p_depth, uint32_t p_stencil);
	Error _draw_list_setup_framebuffer(Framebuffer *p_framebuffer, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, VkFramebuffer *r_framebuffer, VkRenderPass *r_render_pass);
	Error _draw_list_render_pass_begin(Framebuffer *framebuffer, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_colors, float p_clear_depth, uint32_t p_clear_stencil, Point2i viewport_offset, Point2i viewport_size, VkFramebuffer vkframebuffer, VkRenderPass render_pass, VkCommandBuffer command_buffer, VkSubpassContents subpass_contents, const Vector<RID> &p_storage_textures);
	_FORCE_INLINE_ DrawList *_get_draw_list_ptr(DrawListID p_id);
	Buffer *_get_buffer_from_owner(RID p_buffer, VkPipelineStageFlags &dst_stage_mask, VkAccessFlags &dst_access, uint32_t p_post_barrier);

	/**********************/
	/**** COMPUTE LIST ****/
	/**********************/

	struct ComputeList {
		VkCommandBuffer command_buffer = VK_NULL_HANDLE; // If persistent, this is owned, otherwise it's shared with the ringbuffer.

		struct SetState {
			uint32_t pipeline_expected_format = 0;
			uint32_t uniform_set_format = 0;
			VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
			RID uniform_set;
			bool bound = false;
		};

		struct State {
			Set<Texture *> textures_to_sampled_layout;
			SetState sets[MAX_UNIFORM_SETS];
			uint32_t set_count = 0;
			RID pipeline;
			RID pipeline_shader;
			uint32_t local_group_size[3] = { 0, 0, 0 };
			VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
			uint32_t pipeline_push_constant_stages = 0;
			bool allow_draw_overlap;
		} state;

#ifdef DEBUG_ENABLED
		struct Validation {
			bool active = true; // Means command buffer was not closed, so you can keep adding things.
			Vector<uint32_t> set_formats;
			Vector<bool> set_bound;
			Vector<RID> set_rids;
			// Last pipeline set values.
			bool pipeline_active = false;
			RID pipeline_shader;
			uint32_t invalid_set_from = 0;
			Vector<uint32_t> pipeline_set_formats;
			uint32_t pipeline_push_constant_size = 0;
			bool pipeline_push_constant_supplied = false;
		} validation;
#endif
	};

	ComputeList *compute_list = nullptr;

	/**************************/
	/**** FRAME MANAGEMENT ****/
	/**************************/

	// This is the frame structure. There are normally
	// 3 of these (used for triple buffering), or 2
	// (double buffering). They are cycled constantly.
	//
	// It contains two command buffers, one that is
	// used internally for setting up (creating stuff)
	// and another used mostly for drawing.
	//
	// They also contains a list of things that need
	// to be disposed of when deleted, which can't
	// happen immediately due to the asynchronous
	// nature of the GPU. They will get deleted
	// when the frame is cycled.

	struct Frame {
		//list in usage order, from last to free to first to free
		List<Buffer> buffers_to_dispose_of;
		List<Texture> textures_to_dispose_of;
		List<Framebuffer> framebuffers_to_dispose_of;
		List<VkSampler> samplers_to_dispose_of;
		List<Shader> shaders_to_dispose_of;
		List<VkBufferView> buffer_views_to_dispose_of;
		List<UniformSet> uniform_sets_to_dispose_of;
		List<RenderPipeline> render_pipelines_to_dispose_of;
		List<ComputePipeline> compute_pipelines_to_dispose_of;

		VkCommandPool command_pool = VK_NULL_HANDLE;
		VkCommandBuffer setup_command_buffer = VK_NULL_HANDLE; //used at the beginning of every frame for set-up
		VkCommandBuffer draw_command_buffer = VK_NULL_HANDLE; //used at the beginning of every frame for set-up

		struct Timestamp {
			String description;
			uint64_t value = 0;
		};

		VkQueryPool timestamp_pool;

		String *timestamp_names = nullptr;
		uint64_t *timestamp_cpu_values = nullptr;
		uint32_t timestamp_count = 0;
		String *timestamp_result_names = nullptr;
		uint64_t *timestamp_cpu_result_values = nullptr;
		uint64_t *timestamp_result_values = nullptr;
		uint32_t timestamp_result_count = 0;
		uint64_t index = 0;
	};

	uint32_t max_timestamp_query_elements = 0;

	Frame *frames = nullptr; //frames available, for main device they are cycled (usually 3), for local devices only 1
	int frame = 0; //current frame
	int frame_count = 0; //total amount of frames
	uint64_t frames_drawn = 0;
	RID local_device;
	bool local_device_processing = false;

	void _free_pending_resources(int p_frame);

	VmaAllocator allocator = nullptr;

	VulkanContext *context = nullptr;

	void _free_internal(RID p_id);
	void _flush(bool p_current_frame);

	bool screen_prepared = false;

	template <class T>
	void _free_rids(T &p_owner, const char *p_type);

	void _finalize_command_bufers();
	void _begin_frame();

public:
	virtual RID texture_create(const TextureFormat &p_format, const TextureView &p_view, const Vector<Vector<uint8_t>> &p_data = Vector<Vector<uint8_t>>());
	virtual RID texture_create_shared(const TextureView &p_view, RID p_with_texture);

	virtual RID texture_create_shared_from_slice(const TextureView &p_view, RID p_with_texture, uint32_t p_layer, uint32_t p_mipmap, TextureSliceType p_slice_type = TEXTURE_SLICE_2D);
	virtual Error texture_update(RID p_texture, uint32_t p_layer, const Vector<uint8_t> &p_data, uint32_t p_post_barrier = BARRIER_MASK_ALL);
	virtual Vector<uint8_t> texture_get_data(RID p_texture, uint32_t p_layer);

	virtual bool texture_is_format_supported_for_usage(DataFormat p_format, uint32_t p_usage) const;
	virtual bool texture_is_shared(RID p_texture);
	virtual bool texture_is_valid(RID p_texture);

	virtual Error texture_copy(RID p_from_texture, RID p_to_texture, const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_size, uint32_t p_src_mipmap, uint32_t p_dst_mipmap, uint32_t p_src_layer, uint32_t p_dst_layer, uint32_t p_post_barrier = BARRIER_MASK_ALL);
	virtual Error texture_clear(RID p_texture, const Color &p_color, uint32_t p_base_mipmap, uint32_t p_mipmaps, uint32_t p_base_layer, uint32_t p_layers, uint32_t p_post_barrier = BARRIER_MASK_ALL);
	virtual Error texture_resolve_multisample(RID p_from_texture, RID p_to_texture, uint32_t p_post_barrier = BARRIER_MASK_ALL);

	/*********************/
	/**** FRAMEBUFFER ****/
	/*********************/

	virtual FramebufferFormatID framebuffer_format_create(const Vector<AttachmentFormat> &p_format);
	virtual FramebufferFormatID framebuffer_format_create_empty(TextureSamples p_samples = TEXTURE_SAMPLES_1);
	virtual TextureSamples framebuffer_format_get_texture_samples(FramebufferFormatID p_format);

	virtual RID framebuffer_create(const Vector<RID> &p_texture_attachments, FramebufferFormatID p_format_check = INVALID_ID);
	virtual RID framebuffer_create_empty(const Size2i &p_size, TextureSamples p_samples = TEXTURE_SAMPLES_1, FramebufferFormatID p_format_check = INVALID_ID);

	virtual FramebufferFormatID framebuffer_get_format(RID p_framebuffer);

	/*****************/
	/**** SAMPLER ****/
	/*****************/

	virtual RID sampler_create(const SamplerState &p_state);

	/**********************/
	/**** VERTEX ARRAY ****/
	/**********************/

	virtual RID vertex_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>(), bool p_use_as_storage = false);

	// Internally reference counted, this ID is warranted to be unique for the same description, but needs to be freed as many times as it was allocated
	virtual VertexFormatID vertex_format_create(const Vector<VertexAttribute> &p_vertex_formats);
	virtual RID vertex_array_create(uint32_t p_vertex_count, VertexFormatID p_vertex_format, const Vector<RID> &p_src_buffers);

	virtual RID index_buffer_create(uint32_t p_size_indices, IndexBufferFormat p_format, const Vector<uint8_t> &p_data = Vector<uint8_t>(), bool p_use_restart_indices = false);

	virtual RID index_array_create(RID p_index_buffer, uint32_t p_index_offset, uint32_t p_index_count);

	/****************/
	/**** SHADER ****/
	/****************/

	virtual RID shader_create(const Vector<ShaderStageData> &p_stages);
	virtual uint32_t shader_get_vertex_input_attribute_mask(RID p_shader);

	/*****************/
	/**** UNIFORM ****/
	/*****************/

	virtual RID uniform_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>());
	virtual RID storage_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>(), uint32_t p_usage = 0);
	virtual RID texture_buffer_create(uint32_t p_size_elements, DataFormat p_format, const Vector<uint8_t> &p_data = Vector<uint8_t>());

	virtual RID uniform_set_create(const Vector<Uniform> &p_uniforms, RID p_shader, uint32_t p_shader_set);
	virtual bool uniform_set_is_valid(RID p_uniform_set);

	virtual Error buffer_update(RID p_buffer, uint32_t p_offset, uint32_t p_size, const void *p_data, uint32_t p_post_barrier = BARRIER_MASK_ALL); //works for any buffer
	virtual Error buffer_clear(RID p_buffer, uint32_t p_offset, uint32_t p_size, uint32_t p_post_barrier = BARRIER_MASK_ALL);
	virtual Vector<uint8_t> buffer_get_data(RID p_buffer);

	/*************************/
	/**** RENDER PIPELINE ****/
	/*************************/

	virtual RID render_pipeline_create(RID p_shader, FramebufferFormatID p_framebuffer_format, VertexFormatID p_vertex_format, RenderPrimitive p_render_primitive, const PipelineRasterizationState &p_rasterization_state, const PipelineMultisampleState &p_multisample_state, const PipelineDepthStencilState &p_depth_stencil_state, const PipelineColorBlendState &p_blend_state, int p_dynamic_state_flags = 0);
	virtual bool render_pipeline_is_valid(RID p_pipeline);

	/**************************/
	/**** COMPUTE PIPELINE ****/
	/**************************/

	virtual RID compute_pipeline_create(RID p_shader);
	virtual bool compute_pipeline_is_valid(RID p_pipeline);

	/****************/
	/**** SCREEN ****/
	/****************/

	virtual int screen_get_width(DisplayServer::WindowID p_screen = 0) const;
	virtual int screen_get_height(DisplayServer::WindowID p_screen = 0) const;
	virtual FramebufferFormatID screen_get_framebuffer_format() const;

	/********************/
	/**** DRAW LISTS ****/
	/********************/

	virtual DrawListID draw_list_begin_for_screen(DisplayServer::WindowID p_screen = 0, const Color &p_clear_color = Color());

	virtual DrawListID draw_list_begin(RID p_framebuffer, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const Vector<RID> &p_storage_textures = Vector<RID>());
	virtual Error draw_list_begin_split(RID p_framebuffer, uint32_t p_splits, DrawListID *r_split_ids, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const Vector<RID> &p_storage_textures = Vector<RID>());

	virtual void draw_list_bind_render_pipeline(DrawListID p_list, RID p_render_pipeline);
	virtual void draw_list_bind_uniform_set(DrawListID p_list, RID p_uniform_set, uint32_t p_index);
	virtual void draw_list_bind_vertex_array(DrawListID p_list, RID p_vertex_array);
	virtual void draw_list_bind_index_array(DrawListID p_list, RID p_index_array);
	virtual void draw_list_set_line_width(DrawListID p_list, float p_width);
	virtual void draw_list_set_push_constant(DrawListID p_list, const void *p_data, uint32_t p_data_size);

	virtual void draw_list_draw(DrawListID p_list, bool p_use_indices, uint32_t p_instances = 1, uint32_t p_procedural_vertices = 0);

	virtual void draw_list_enable_scissor(DrawListID p_list, const Rect2 &p_rect);
	virtual void draw_list_disable_scissor(DrawListID p_list);

	virtual void draw_list_end(uint32_t p_post_barrier = BARRIER_MASK_ALL);

	/***********************/
	/**** COMPUTE LISTS ****/
	/***********************/

	virtual ComputeListID compute_list_begin(bool p_allow_draw_overlap = false);
	virtual void compute_list_bind_compute_pipeline(ComputeListID p_list, RID p_compute_pipeline);
	virtual void compute_list_bind_uniform_set(ComputeListID p_list, RID p_uniform_set, uint32_t p_index);
	virtual void compute_list_set_push_constant(ComputeListID p_list, const void *p_data, uint32_t p_data_size);
	virtual void compute_list_add_barrier(ComputeListID p_list);

	virtual void compute_list_dispatch(ComputeListID p_list, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups);
	virtual void compute_list_dispatch_threads(ComputeListID p_list, uint32_t p_x_threads, uint32_t p_y_threads, uint32_t p_z_threads);
	virtual void compute_list_dispatch_indirect(ComputeListID p_list, RID p_buffer, uint32_t p_offset);
	virtual void compute_list_end(uint32_t p_post_barrier = BARRIER_MASK_ALL);

	virtual void barrier(uint32_t p_from = BARRIER_MASK_ALL, uint32_t p_to = BARRIER_MASK_ALL);
	virtual void full_barrier();

	/**************/
	/**** FREE ****/
	/**************/

	virtual void free(RID p_id);

	/****************/
	/**** Timing ****/
	/****************/

	virtual void capture_timestamp(const String &p_name);
	virtual uint32_t get_captured_timestamps_count() const;
	virtual uint64_t get_captured_timestamps_frame() const;
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const;
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const;
	virtual String get_captured_timestamp_name(uint32_t p_index) const;

	/****************/
	/**** Limits ****/
	/****************/

	virtual int limit_get(Limit p_limit);

	virtual void prepare_screen_for_drawing();
	void initialize(VulkanContext *p_context, bool p_local_device = false);
	void finalize();

	virtual void swap_buffers(); //for main device

	virtual void submit(); //for local device
	virtual void sync(); //for local device

	virtual uint32_t get_frame_delay() const;

	virtual RenderingDevice *create_local_device();

	virtual uint64_t get_memory_usage() const;

	virtual void set_resource_name(RID p_id, const String p_name);

	virtual void draw_command_begin_label(String p_label_name, const Color p_color = Color(1, 1, 1, 1));
	virtual void draw_command_insert_label(String p_label_name, const Color p_color = Color(1, 1, 1, 1));
	virtual void draw_command_end_label();

	virtual String get_device_vendor_name() const;
	virtual String get_device_name() const;
	virtual String get_device_pipeline_cache_uuid() const;

	RenderingDeviceVulkan();
	~RenderingDeviceVulkan();
};

#endif // RENDERING_DEVICE_VULKAN_H
