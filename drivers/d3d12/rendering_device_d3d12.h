/**************************************************************************/
/*  rendering_device_d3d12.h                                              */
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

#ifndef RENDERING_DEVICE_D3D12_H
#define RENDERING_DEVICE_D3D12_H

#include "core/os/thread_safe.h"
#include "core/templates/local_vector.h"
#include "core/templates/oa_hash_map.h"
#include "core/templates/rid_owner.h"
#include "drivers/d3d12/d3d12_context.h"
#include "servers/rendering/rendering_device.h"

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

#define D3D12_BITCODE_OFFSETS_NUM_STAGES 3

struct dxil_validator;

class RenderingDeviceD3D12 : public RenderingDevice {
	_THREAD_SAFE_CLASS_
	// Miscellaneous tables that map
	// our enums to enums used
	// by DXGI/D3D12.

	D3D12Context::DeviceLimits limits = {};
	struct D3D12Format {
		DXGI_FORMAT family = DXGI_FORMAT_UNKNOWN;
		DXGI_FORMAT general_format = DXGI_FORMAT_UNKNOWN;
		UINT swizzle = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
	};
	static const D3D12Format d3d12_formats[DATA_FORMAT_MAX];
	static const char *named_formats[DATA_FORMAT_MAX];
	static const D3D12_COMPARISON_FUNC compare_operators[COMPARE_OP_MAX];
	static const D3D12_STENCIL_OP stencil_operations[STENCIL_OP_MAX];
	static const UINT rasterization_sample_count[TEXTURE_SAMPLES_MAX];
	static const D3D12_LOGIC_OP logic_operations[RenderingDevice::LOGIC_OP_MAX];
	static const D3D12_BLEND blend_factors[RenderingDevice::BLEND_FACTOR_MAX];
	static const D3D12_BLEND_OP blend_operations[RenderingDevice::BLEND_OP_MAX];
	static const D3D12_TEXTURE_ADDRESS_MODE address_modes[SAMPLER_REPEAT_MODE_MAX];
	static const FLOAT sampler_border_colors[SAMPLER_BORDER_COLOR_MAX][4];
	static const D3D12_RESOURCE_DIMENSION d3d12_texture_dimension[TEXTURE_TYPE_MAX];

	// Functions used for format
	// validation, and ensures the
	// user passes valid data.

	static int get_format_vertex_size(DataFormat p_format);
	static uint32_t get_image_format_pixel_size(DataFormat p_format);
	static void get_compressed_image_format_block_dimensions(DataFormat p_format, uint32_t &r_w, uint32_t &r_h);
	uint32_t get_compressed_image_format_block_byte_size(DataFormat p_format);
	static uint32_t get_compressed_image_format_pixel_rshift(DataFormat p_format);
	static uint32_t get_image_format_plane_count(DataFormat p_format);
	static uint32_t get_image_format_required_size(DataFormat p_format, uint32_t p_width, uint32_t p_height, uint32_t p_depth, uint32_t p_mipmaps, uint32_t *r_blockw = nullptr, uint32_t *r_blockh = nullptr, uint32_t *r_depth = nullptr);
	static uint32_t get_image_required_mipmaps(uint32_t p_width, uint32_t p_height, uint32_t p_depth);
	static bool format_has_stencil(DataFormat p_format);

	Mutex dxil_mutex;
	HashMap<int, dxil_validator *> dxil_validators; // One per WorkerThreadPool thread used for shader compilation, plus one (-1) for all the other.

	dxil_validator *get_dxil_validator_for_current_thread();

	class DescriptorsHeap {
		D3D12_DESCRIPTOR_HEAP_DESC desc = {};
		ComPtr<ID3D12DescriptorHeap> heap;
		uint32_t handle_size = 0;

	public:
		class Walker { // Texas Ranger.
			friend class DescriptorsHeap;

			uint32_t handle_size = 0;
			uint32_t handle_count = 0;
			D3D12_CPU_DESCRIPTOR_HANDLE first_cpu_handle = {};
			D3D12_GPU_DESCRIPTOR_HANDLE first_gpu_handle = {};
			uint32_t handle_index = 0;

		public:
			D3D12_CPU_DESCRIPTOR_HANDLE get_curr_cpu_handle();
			D3D12_GPU_DESCRIPTOR_HANDLE get_curr_gpu_handle();
			_FORCE_INLINE_ void rewind() { handle_index = 0; }
			void advance(uint32_t p_count = 1);
			uint32_t get_current_handle_index() const { return handle_index; }
			uint32_t get_free_handles() { return handle_count - handle_index; }
			bool is_at_eof() { return handle_index == handle_count; }
		};

		Error allocate(ID3D12Device *m_device, D3D12_DESCRIPTOR_HEAP_TYPE m_type, uint32_t m_descriptor_count, bool p_for_gpu);
		uint32_t get_descriptor_count() const { return desc.NumDescriptors; }
		ID3D12DescriptorHeap *get_heap() const { return heap.Get(); }

		Walker make_walker() const;
	};

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
		ID_BASE_SHIFT = 58 // 5 bits for ID types.
	};

	ComPtr<ID3D12Device> device;

	HashMap<RID, HashSet<RID>> dependency_map; // IDs to IDs that depend on it.
	HashMap<RID, HashSet<RID>> reverse_dependency_map; // Same as above, but in reverse.

	void _add_dependency(RID p_id, RID p_depends_on);
	void _free_dependencies(RID p_id);

	/******************/
	/**** RESOURCE ****/
	/******************/

	class ResourceState {
		D3D12_RESOURCE_STATES states = D3D12_RESOURCE_STATE_COMMON;

	public:
		void extend(D3D12_RESOURCE_STATES p_states_to_add);
		D3D12_RESOURCE_STATES get_state_mask() const { return states; }

		ResourceState() {}
		ResourceState(D3D12_RESOURCE_STATES p_states) :
				states(p_states) {}
	};

	struct Resource {
		struct States {
			// As many subresources as mipmaps * layers; planes (for depth-stencil) are tracked together.
			LocalVector<D3D12_RESOURCE_STATES> subresource_states; // Used only if not a view.
			uint32_t last_batch_transitioned_to_uav = 0;
			uint32_t last_batch_with_uav_barrier = 0;
		};

		ID3D12Resource *resource = nullptr;
		D3D12MA::Allocation *allocation = nullptr;

		States own_states; // Used only if not a view.
		States *states = nullptr; // Non-null only if a view.

		States *get_states_ptr() { return states ? states : &own_states; }
	};

	struct BarrierRequest {
		static const uint32_t MAX_GROUPS = 4;
		// Maybe this is too much data to have it locally. Benchmarking may reveal that
		// cache would be used better by having a maximum of local subresource masks and beyond
		// that have an allocated vector with the rest.
		static const uint32_t MAX_SUBRESOURCES = 4096; // Must be multiple of 64.
		ID3D12Resource *dx_resource;
		uint8_t subres_mask_qwords;
		uint8_t planes;
		struct Group {
			ResourceState state;
			uint64_t subres_mask[MAX_SUBRESOURCES / 64];
		} groups[MAX_GROUPS];
		uint8_t groups_count;
		static const D3D12_RESOURCE_STATES DELETED_GROUP = D3D12_RESOURCE_STATE_COMMON;
	};
	HashMap<Resource::States *, BarrierRequest> res_barriers_requests;

	LocalVector<D3D12_RESOURCE_BARRIER> res_barriers;
	uint32_t res_barriers_count = 0;
	uint32_t res_barriers_batch = 0;
#ifdef DEV_ENABLED
	int frame_barriers_count = 0;
	int frame_barriers_batches_count = 0;
	uint64_t frame_barriers_cpu_time = 0;
#endif

	void _resource_transition_batch(Resource *p_resource, uint32_t p_subresource, uint32_t p_num_planes, D3D12_RESOURCE_STATES p_new_state, ID3D12Resource *p_resource_override = nullptr);
	void _resource_transitions_flush(ID3D12GraphicsCommandList *p_command_list);

	/*****************/
	/**** TEXTURE ****/
	/*****************/

	struct Texture : Resource {
		D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
		D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
		D3D12_UNORDERED_ACCESS_VIEW_DESC owner_uav_desc = {}; // [[CROSS_FAMILY_ALIASING]].

		TextureType type;
		DataFormat format;
		uint32_t planes = 1;
		TextureSamples samples;
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t depth = 0;
		uint32_t layers = 0;
		uint32_t mipmaps = 0;
		uint32_t owner_layers = 0;
		uint32_t owner_mipmaps = 0;
		uint32_t usage_flags = 0;
		uint32_t base_mipmap = 0;
		uint32_t base_layer = 0;

		Vector<DataFormat> allowed_shared_formats;
		TightLocalVector<ID3D12Resource *> aliases; // [[CROSS_FAMILY_ALIASING]].
		ID3D12Resource *owner_resource = nullptr; // Always the one of the main format passed to creation. [[CROSS_FAMILY_ALIASING]].

		bool is_resolve_buffer = false;

		bool bound = false; // Bound to framebffer.
		RID owner;
	};

	RID_Owner<Texture, true> texture_owner;
	uint32_t texture_upload_region_size_px = 0;

	Vector<uint8_t> _texture_get_data_from_image(Texture *tex, uint32_t p_layer, bool p_2d = false);
	Error _texture_update(Texture *p_texture, uint32_t p_layer, const Vector<uint8_t> &p_data, BitField<BarrierMask> p_post_barrier, ID3D12GraphicsCommandList *p_command_list);

	/*****************/
	/**** SAMPLER ****/
	/*****************/

	RID_Owner<D3D12_SAMPLER_DESC> sampler_owner;

	/***************************/
	/**** BUFFER MANAGEMENT ****/
	/***************************/

	// These are temporary buffers on CPU memory that hold
	// the information until the CPU fetches it and places it
	// either on GPU buffers, or images (textures). It ensures
	// updates are properly synchronized with whatever the
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
		ID3D12Resource *resource = nullptr; // Owned, but ComPtr would have too much overhead in a Vector.
		D3D12MA::Allocation *allocation = nullptr;
		uint64_t frame_used = 0;
		uint32_t fill_amount = 0;
	};

	Vector<StagingBufferBlock> staging_buffer_blocks;
	int staging_buffer_current = 0;
	uint32_t staging_buffer_block_size = 0;
	uint64_t staging_buffer_max_size = 0;
	bool staging_buffer_used = false;

	Error _staging_buffer_allocate(uint32_t p_amount, uint32_t p_required_align, uint32_t &r_alloc_offset, uint32_t &r_alloc_size, bool p_can_segment = true);
	Error _insert_staging_block();

	struct Buffer : Resource {
		uint32_t size = 0;
		D3D12_RESOURCE_STATES usage = {};
		uint32_t last_execution = 0;
	};

	Error _buffer_allocate(Buffer *p_buffer, uint32_t p_size, D3D12_RESOURCE_STATES p_usage, D3D12_HEAP_TYPE p_heap_type);
	Error _buffer_free(Buffer *p_buffer);
	Error _buffer_update(Buffer *p_buffer, size_t p_offset, const uint8_t *p_data, size_t p_data_size, bool p_use_draw_command_list = false, uint32_t p_required_align = 32);

	/*********************/
	/**** FRAMEBUFFER ****/
	/*********************/

	static D3D12_RENDER_TARGET_VIEW_DESC _make_rtv_for_texture(const RenderingDeviceD3D12::Texture *p_texture, uint32_t p_mipmap_offset = 0, uint32_t p_layer_offset = 0, uint32_t p_layers = UINT32_MAX);
	static D3D12_DEPTH_STENCIL_VIEW_DESC _make_dsv_for_texture(const RenderingDeviceD3D12::Texture *p_texture);

	// In Vulkan we'd create some structures the driver uses for render pass based rendering.
	// (Dynamic rendering is supported on Vulkan 1.3+, though, but Godot is not using it.)
	// In contrast, in D3D12 we'll go the dynamic rendering way, since it's more convenient
	// and render pass based render setup is not available on every version.
	// Therefore, we just need to keep the data at hand and use it where appropriate.

	struct FramebufferFormat {
		Vector<AttachmentFormat> attachments;
		Vector<FramebufferPass> passes;
		Vector<TextureSamples> pass_samples;
		uint32_t view_count = 1;
		uint32_t max_supported_sample_count = 1;
	};

	bool _framebuffer_format_preprocess(FramebufferFormat *p_fb_format, uint32_t p_view_count);

	HashMap<FramebufferFormatID, FramebufferFormat> framebuffer_formats;

	struct Framebuffer {
		DisplayServer::WindowID window_id = DisplayServer::INVALID_WINDOW_ID;
		FramebufferFormatID format_id = 0;
		Vector<RID> texture_ids; // Empty if for screen.
		InvalidationCallback invalidated_callback = nullptr;
		void *invalidated_callback_userdata = nullptr;
		Vector<uint32_t> attachments_handle_inds; // RTV heap index for color; DSV heap index for DSV.
		Size2 size;
		uint32_t view_count = 1;
		DescriptorsHeap rtv_heap; // Used only if not for screen and some color attachments.
		D3D12_CPU_DESCRIPTOR_HANDLE screen_rtv_handle = {}; // Used only if for screen.
		DescriptorsHeap dsv_heap; // Used only if not for screen and some depth-stencil attachments.
	};

	RID_Owner<Framebuffer, true> framebuffer_owner;

	/***********************/
	/**** VERTEX BUFFER ****/
	/***********************/

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
				return true; // They are equal.
			}
		}

		uint32_t hash() const {
			int vdc = vertex_formats.size();
			uint32_t h = hash_murmur3_one_32(vdc);
			const VertexAttribute *ptr = vertex_formats.ptr();
			for (int i = 0; i < vdc; i++) {
				const VertexAttribute &vd = ptr[i];
				h = hash_murmur3_one_32(vd.location, h);
				h = hash_murmur3_one_32(vd.offset, h);
				h = hash_murmur3_one_32(vd.format, h);
				h = hash_murmur3_one_32(vd.stride, h);
				h = hash_murmur3_one_32(vd.frequency, h);
			}
			return hash_fmix32(h);
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
		Vector<D3D12_INPUT_ELEMENT_DESC> elements_desc;
	};

	HashMap<VertexFormatID, VertexDescriptionCache> vertex_formats;

	struct VertexArray {
		Vector<Buffer *> unique_buffers;
		VertexFormatID description = 0;
		int vertex_count = 0;
		uint32_t max_instances_allowed = 0;
		Vector<D3D12_VERTEX_BUFFER_VIEW> views;
	};

	RID_Owner<VertexArray, true> vertex_array_owner;

	struct IndexBuffer : public Buffer {
		uint32_t max_index = 0; // Used for validation.
		uint32_t index_count = 0;
		DXGI_FORMAT index_format = {};
		bool supports_restart_indices = false;
	};

	RID_Owner<IndexBuffer, true> index_buffer_owner;

	struct IndexArray {
		IndexBuffer *buffer = nullptr;
		uint32_t max_index = 0; // Remember the maximum index here too, for validation.
		uint32_t offset = 0;
		uint32_t indices = 0;
		bool supports_restart_indices = false;
		D3D12_INDEX_BUFFER_VIEW view = {};
	};

	RID_Owner<IndexArray, true> index_array_owner;

	/****************/
	/**** SHADER ****/
	/****************/

	static const uint32_t ROOT_SIGNATURE_SIZE = 256;
	static const uint32_t PUSH_CONSTANT_SIZE = 128; // Mimicking Vulkan.

	enum {
		// We can only aim to set a maximum here, since depending on the shader
		// there may be more or less root signature free for descriptor tables.
		// Therefore, we'll have to rely on the final check at runtime, when building
		// the root signature structure for a given shader.
		// To be precise, these may be present or not, and their size vary statically:
		// - Push constant (we'll assume this is always present to avoid reserving much
		//   more space for descriptor sets than needed for almost any imaginable case,
		//   given that most shader templates feature push constants).
		// - NIR-DXIL runtime data.
		MAX_UNIFORM_SETS = (ROOT_SIGNATURE_SIZE - PUSH_CONSTANT_SIZE) / sizeof(uint32_t),
	};

	enum ResourceClass {
		RES_CLASS_INVALID,
		RES_CLASS_CBV,
		RES_CLASS_SRV,
		RES_CLASS_UAV,
	};

	struct UniformBindingInfo {
		uint32_t stages = 0; // Actual shader stages using the uniform (0 if totally optimized out).
		ResourceClass res_class = RES_CLASS_INVALID;
		struct RootSignatureLocation {
			uint32_t root_param_idx = UINT32_MAX;
			uint32_t range_idx = UINT32_MAX;
		};
		struct {
			RootSignatureLocation resource;
			RootSignatureLocation sampler;
		} root_sig_locations;
	};

	struct UniformInfo {
		UniformType type = UniformType::UNIFORM_TYPE_MAX;
		bool writable = false;
		int binding = 0;
		int length = 0; // Size of arrays (in total elements), or ubos (in bytes * total elements).

		bool operator!=(const UniformInfo &p_info) const {
			return (binding != p_info.binding || type != p_info.type || writable != p_info.writable || length != p_info.length);
		}

		bool operator<(const UniformInfo &p_info) const {
			if (binding != p_info.binding) {
				return binding < p_info.binding;
			}
			if (type != p_info.type) {
				return type < p_info.type;
			}
			if (writable != p_info.writable) {
				return writable < p_info.writable;
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
	RBMap<UniformSetFormat, uint32_t> uniform_set_format_cache;
	Vector<RBMap<UniformSetFormat, uint32_t>::Element *> uniform_set_format_cache_reverse;

	struct Shader {
		struct ShaderUniformInfo {
			UniformInfo info;
			UniformBindingInfo binding;

			bool operator<(const ShaderUniformInfo &p_info) const {
				return *((UniformInfo *)this) < (const UniformInfo &)p_info;
			}
		};
		struct Set {
			Vector<ShaderUniformInfo> uniforms;
			struct {
				uint32_t resources = 0;
				uint32_t samplers = 0;
			} num_root_params;
		};

		uint64_t vertex_input_mask = 0; // Inputs used, this is mostly for validation.
		uint32_t fragment_output_mask = 0;

		uint32_t spirv_push_constant_size = 0;
		uint32_t dxil_push_constant_size = 0;
		uint32_t nir_runtime_data_root_param_idx = UINT32_MAX;

		uint32_t compute_local_size[3] = { 0, 0, 0 };

		struct SpecializationConstant {
			PipelineSpecializationConstant constant;
			uint64_t stages_bit_offsets[D3D12_BITCODE_OFFSETS_NUM_STAGES];
		};

		bool is_compute = false;
		Vector<Set> sets;
		Vector<uint32_t> set_formats;
		Vector<SpecializationConstant> specialization_constants;
		uint32_t spirv_specialization_constants_ids_mask = 0;
		HashMap<ShaderStage, Vector<uint8_t>> stages_bytecode;
		String name; // Used for debug.

		ComPtr<ID3D12RootSignature> root_signature;
		ComPtr<ID3D12RootSignatureDeserializer> root_signature_deserializer;
		const D3D12_ROOT_SIGNATURE_DESC *root_signature_desc = nullptr; // Owned by the deserializer.
		uint32_t root_signature_crc = 0;
	};

	String _shader_uniform_debug(RID p_shader, int p_set = -1);

	RID_Owner<Shader, true> shader_owner;

	uint32_t _shader_patch_dxil_specialization_constant(
			PipelineSpecializationConstantType p_type,
			const void *p_value,
			const uint64_t (&p_stages_bit_offsets)[D3D12_BITCODE_OFFSETS_NUM_STAGES],
			HashMap<ShaderStage, Vector<uint8_t>> &r_stages_bytecodes,
			bool p_is_first_patch);
	bool _shader_sign_dxil_bytecode(ShaderStage p_stage, Vector<uint8_t> &r_dxil_blob);

	/******************/
	/**** UNIFORMS ****/
	/******************/

	RID_Owner<Buffer, true> uniform_buffer_owner;
	RID_Owner<Buffer, true> storage_buffer_owner;

	// Texture buffer needs a view.
	struct TextureBuffer {
		Buffer buffer;
	};

	RID_Owner<TextureBuffer, true> texture_buffer_owner;

	struct RootDescriptorTable {
		uint32_t root_param_idx = UINT32_MAX;
		D3D12_GPU_DESCRIPTOR_HANDLE start_gpu_handle = {};
	};

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
		struct {
			DescriptorsHeap resources;
			DescriptorsHeap samplers;
		} desc_heaps;
		struct StateRequirement {
			Resource *resource;
			bool is_buffer;
			D3D12_RESOURCE_STATES states;
			uint64_t shader_uniform_idx_mask;
		};
		struct AttachableTexture {
			uint32_t bind;
			RID texture;
		};

		struct RecentBind {
			uint64_t execution_index = 0;
			uint32_t root_signature_crc = 0;
			struct {
				LocalVector<RootDescriptorTable> resources;
				LocalVector<RootDescriptorTable> samplers;
			} root_tables;
			int uses = 0;
		} recent_binds[4]; // A better amount may be empirically found.

		LocalVector<AttachableTexture> attachable_textures; // Used for validation.
		Vector<StateRequirement> resource_states;
		InvalidationCallback invalidated_callback = nullptr;
		void *invalidated_callback_userdata = nullptr;

#ifdef DEV_ENABLED
		// Filthy, but useful for dev.
		struct ResourceDescInfo {
			D3D12_DESCRIPTOR_RANGE_TYPE type;
			D3D12_SRV_DIMENSION srv_dimension;
		};
		LocalVector<ResourceDescInfo> _resources_desc_info;
		const Shader *_shader = nullptr;
#endif
	};

	RID_Owner<UniformSet, true> uniform_set_owner;

	void _bind_uniform_set(UniformSet *p_uniform_set, const Shader::Set &p_shader_set, const Vector<UniformBindingInfo> &p_bindings, ID3D12GraphicsCommandList *p_command_list, bool p_for_compute);
	void _apply_uniform_set_resource_states(const UniformSet *p_uniform_set, const Shader::Set &p_shader_set);

	/*******************/
	/**** PIPELINES ****/
	/*******************/

	Error _apply_specialization_constants(
			const Shader *p_shader,
			const Vector<PipelineSpecializationConstant> &p_specialization_constants,
			HashMap<ShaderStage, Vector<uint8_t>> &r_final_stages_bytecode);
#ifdef DEV_ENABLED
	String _build_pipeline_blob_filename(
			const Vector<uint8_t> &p_blob,
			const Shader *p_shader,
			const Vector<PipelineSpecializationConstant> &p_specialization_constants,
			const String &p_extra_name_suffix = "",
			const String &p_forced_id = "");
	void _save_pso_blob(
			ID3D12PipelineState *p_pso,
			const Shader *p_shader,
			const Vector<PipelineSpecializationConstant> &p_specialization_constants);
	void _save_stages_bytecode(
			const HashMap<ShaderStage, Vector<uint8_t>> &p_stages_bytecode,
			const Shader *p_shader,
			const RID p_shader_rid,
			const Vector<PipelineSpecializationConstant> &p_specialization_constants);
#endif

	// Render pipeline contains ALL the
	// information required for drawing.
	// This includes all the rasterizer state
	// as well as shader used, framebuffer format,
	// etc.
	// Some parameters aren't fixed in D3D12,
	// so they are stored in an ancillary
	// dynamic parameters structure to be set
	// on pipeline activation via several calls.

	struct RenderPipeline {
		// Cached values for validation.
#ifdef DEBUG_ENABLED
		struct Validation {
			FramebufferFormatID framebuffer_format = 0;
			uint32_t render_pass = 0;
			uint32_t dynamic_state = 0;
			VertexFormatID vertex_format = 0;
			bool uses_restart_indices = false;
			uint32_t primitive_minimum = 0;
			uint32_t primitive_divisor = 0;
		} validation;
#endif
		RID shader;
		Vector<uint32_t> set_formats;
		uint32_t bindings_id = 0;
		ComPtr<ID3D12PipelineState> pso;
		uint32_t root_signature_crc = 0;
		uint32_t spirv_push_constant_size = 0;
		uint32_t dxil_push_constant_size = 0;
		uint32_t nir_runtime_data_root_param_idx = UINT32_MAX;
		struct DynamicParams {
			D3D12_PRIMITIVE_TOPOLOGY primitive_topology = {};
			Color blend_constant;
			float depth_bounds_min = 0.0f;
			float depth_bounds_max = 0.0f;
			uint32_t stencil_reference = 0;
		} dyn_params;
	};

	HashMap<uint32_t, Vector<Vector<UniformBindingInfo>>> pipeline_bindings;
	uint32_t next_pipeline_binding_id = 1;

	RID_Owner<RenderPipeline, true> render_pipeline_owner;

	struct ComputePipeline {
		RID shader;
		Vector<uint32_t> set_formats;
		uint32_t bindings_id = 0;
		ComPtr<ID3D12PipelineState> pso;
		uint32_t root_signature_crc = 0;
		uint32_t spirv_push_constant_size = 0;
		uint32_t dxil_push_constant_size = 0;
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
	// implemented internally using bundles.
	// As they can be created in threads,
	// each needs its own command allocator.

	struct SplitDrawListAllocator {
		// All pointers are owned, but not using ComPtr to avoid overhead in the vector.
		ID3D12CommandAllocator *command_allocator = nullptr;
		Vector<ID3D12GraphicsCommandList *> command_lists; // One for each frame.
	};

	Vector<SplitDrawListAllocator> split_draw_list_allocators;

	struct DrawList {
		ID3D12GraphicsCommandList *command_list = nullptr; // If persistent, this is owned, otherwise it's shared with the ringbuffer.
		Rect2i viewport;
		bool viewport_set = false;

		struct SetState {
			uint32_t pipeline_expected_format = 0;
			uint32_t uniform_set_format = 0;
			RID uniform_set;
			bool bound = false;
#ifdef DEV_ENABLED
			// Filthy, but useful for dev.
			const Vector<UniformInfo> *_pipeline_expected_format = nullptr;
			const UniformSet *_uniform_set = nullptr;
#endif
		};

		struct State {
			SetState sets[MAX_UNIFORM_SETS];
			uint32_t set_count = 0;
			RID pipeline;
			ID3D12PipelineState *pso = nullptr;
			ID3D12PipelineState *bound_pso = nullptr;
			RID pipeline_shader;
			uint32_t pipeline_dxil_push_constant_size = 0;
			uint32_t pipeline_bindings_id = 0;
			uint32_t root_signature_crc = 0;
			RID vertex_array;
			RID index_array;
#ifdef DEV_ENABLED
			// Filthy, but useful for dev.
			Shader *_shader = nullptr;
#endif
		} state;

#ifdef DEBUG_ENABLED
		struct Validation {
			bool active = true; // Means command buffer was not closed, so you can keep adding things.
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
			bool pipeline_uses_restart_indices = false;
			uint32_t pipeline_primitive_divisor = 0;
			uint32_t pipeline_primitive_minimum = 0;
			uint32_t pipeline_spirv_push_constant_size = 0;
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
	uint32_t draw_list_subpass_count = 0;
	uint32_t draw_list_count = 0;
	Framebuffer curr_screen_framebuffer; // Only valid while a screen draw list is open.
	Framebuffer *draw_list_framebuffer = nullptr;
	FinalAction draw_list_final_color_action = FINAL_ACTION_DISCARD;
	FinalAction draw_list_final_depth_action = FINAL_ACTION_DISCARD;
	Vector2 draw_list_viewport_size = {};
	uint32_t draw_list_current_subpass = 0;

	bool draw_list_split = false;
	Vector<RID> draw_list_bound_textures;
	bool draw_list_unbind_color_textures = false;
	bool draw_list_unbind_depth_textures = false;

	struct {
		RID texture_bound;
		bool configured = false;
	} vrs_state;
	uint32_t vrs_state_execution_index = 0;

	Error _draw_list_render_pass_begin(Framebuffer *framebuffer, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_colors, float p_clear_depth, uint32_t p_clear_stencil, const Rect2 &p_region, Point2i viewport_offset, Point2i viewport_size, ID3D12GraphicsCommandList *command_list, const Vector<RID> &p_storage_textures);
	_FORCE_INLINE_ DrawList *_get_draw_list_ptr(DrawListID p_id);
	Buffer *_get_buffer_from_owner(RID p_buffer);
	Error _draw_list_allocate(const Rect2i &p_viewport, uint32_t p_splits, uint32_t p_subpass);
	void _draw_list_free(Rect2i *r_last_viewport = nullptr);
	void _draw_list_subpass_begin();
	void _draw_list_subpass_end();

	/**********************/
	/**** COMPUTE LIST ****/
	/**********************/

	struct ComputeList {
		ID3D12GraphicsCommandList *command_list = nullptr; // If persistent, this is owned, otherwise it's shared with the ringbuffer.

		struct SetState {
			uint32_t pipeline_expected_format = 0;
			uint32_t uniform_set_format = 0;
			RID uniform_set;
			bool bound = false;
#ifdef DEV_ENABLED
			// Filthy, but useful for dev.
			const Vector<UniformInfo> *_pipeline_expected_format = nullptr;
			const UniformSet *_uniform_set = nullptr;
#endif
		};

		struct State {
			HashSet<Texture *> textures_to_sampled_layout;
			SetState sets[MAX_UNIFORM_SETS];
			uint32_t set_count = 0;
			RID pipeline;
			ID3D12PipelineState *pso = nullptr;
			ID3D12PipelineState *bound_pso = nullptr;
			RID pipeline_shader;
			uint32_t pipeline_dxil_push_constant_size = 0;
			uint32_t pipeline_bindings_id = 0;
			uint32_t root_signature_crc = 0;
			uint32_t local_group_size[3] = { 0, 0, 0 };
			bool allow_draw_overlap;
#ifdef DEV_ENABLED
			// Filthy, but useful for dev.
			Shader *_shader = nullptr;
#endif
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
			uint32_t pipeline_spirv_push_constant_size = 0;
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
		// List in usage order, from last to free to first to free.
		List<Buffer> buffers_to_dispose_of;
		List<Texture> textures_to_dispose_of;
		List<Framebuffer> framebuffers_to_dispose_of;
		List<Shader> shaders_to_dispose_of;
		List<RenderPipeline> render_pipelines_to_dispose_of;
		List<ComputePipeline> compute_pipelines_to_dispose_of;
		struct {
			DescriptorsHeap resources;
			DescriptorsHeap samplers;
			DescriptorsHeap aux;
			DescriptorsHeap rtv;
		} desc_heaps;
		struct {
			DescriptorsHeap::Walker resources;
			DescriptorsHeap::Walker samplers;
			DescriptorsHeap::Walker aux;
			DescriptorsHeap::Walker rtv;
		} desc_heap_walkers;
		struct {
			bool resources;
			bool samplers;
			bool aux;
			bool rtv;
		} desc_heaps_exhausted_reported;
		CD3DX12_CPU_DESCRIPTOR_HANDLE null_rtv_handle = {}; // For [[MANUAL_SUBPASSES]].

		ComPtr<ID3D12CommandAllocator> setup_command_allocator;
		ComPtr<ID3D12CommandAllocator> draw_command_allocator;
		ComPtr<ID3D12GraphicsCommandList> setup_command_list; // Used at the beginning of every frame for set-up.
		ComPtr<ID3D12GraphicsCommandList> draw_command_list;

		struct Timestamp {
			String description;
			uint64_t value = 0;
		};

		ComPtr<ID3D12QueryHeap> timestamp_heap;

		TightLocalVector<String> timestamp_names;
		TightLocalVector<uint64_t> timestamp_cpu_values;
		uint32_t timestamp_count = 0;
		TightLocalVector<String> timestamp_result_names;
		TightLocalVector<uint64_t> timestamp_cpu_result_values;
		Buffer timestamp_result_values_buffer;
		TightLocalVector<uint64_t> timestamp_result_values;
		uint32_t timestamp_result_count = 0;
		uint64_t index = 0;
		uint64_t execution_index = 0;
#ifdef DEV_ENABLED
		uint32_t uniform_set_reused = 0;
#endif
	};

	uint32_t max_timestamp_query_elements = 0;

	TightLocalVector<Frame> frames; // Frames available, for main device they are cycled (usually 3), for local devices only 1.
	int frame = 0; // Current frame.
	int frame_count = 0; // Total amount of frames.
	uint64_t frames_drawn = 0;
	uint32_t execution_index = 0; // Gets incremented on every call to ExecuteCommandLists (each frame and each flush).
	RID local_device;
	bool local_device_processing = false;

	void _free_pending_resources(int p_frame);

//#define USE_SMALL_ALLOCS_POOL // Disabled by now; seems not to be beneficial as it is in Vulkan.
#ifdef USE_SMALL_ALLOCS_POOL
	union AllocPoolKey {
		struct {
			D3D12_HEAP_TYPE heap_type;
			D3D12_HEAP_FLAGS heap_flags;
		};
		uint64_t key;
	};
	HashMap<uint64_t, ComPtr<D3D12MA::Pool>> small_allocs_pools;
	D3D12MA::Pool *_find_or_create_small_allocs_pool(D3D12_HEAP_TYPE p_heap_type, D3D12_HEAP_FLAGS p_heap_flags);
#endif

	ComPtr<ID3D12CommandSignature> indirect_dispatch_cmd_sig;
	RID aux_resource; // Used for causing full barriers.

	D3D12Context *context = nullptr;

	uint64_t image_memory = 0;
	uint64_t buffer_memory = 0;

	void _free_internal(RID p_id);
	void _flush(bool p_flush_current_frame);

	bool screen_prepared = false;

	template <class T>
	void _free_rids(T &p_owner, const char *p_type);

	void _finalize_command_bufers();
	void _begin_frame();

#ifdef DEV_ENABLED
	HashMap<RID, String> resource_names;
#endif

	HashMap<DXGI_FORMAT, uint32_t> format_sample_counts_mask_cache;
	uint32_t _find_max_common_supported_sample_count(const DXGI_FORMAT *p_formats, uint32_t p_num_formats);

public:
	virtual RID texture_create(const TextureFormat &p_format, const TextureView &p_view, const Vector<Vector<uint8_t>> &p_data = Vector<Vector<uint8_t>>());
	virtual RID texture_create_shared(const TextureView &p_view, RID p_with_texture);
	virtual RID texture_create_from_extension(TextureType p_type, DataFormat p_format, TextureSamples p_samples, BitField<RenderingDevice::TextureUsageBits> p_flags, uint64_t p_image, uint64_t p_width, uint64_t p_height, uint64_t p_depth, uint64_t p_layers);

	virtual RID texture_create_shared_from_slice(const TextureView &p_view, RID p_with_texture, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_mipmaps = 1, TextureSliceType p_slice_type = TEXTURE_SLICE_2D, uint32_t p_layers = 0);
	virtual Error texture_update(RID p_texture, uint32_t p_layer, const Vector<uint8_t> &p_data, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);
	virtual Vector<uint8_t> texture_get_data(RID p_texture, uint32_t p_layer);

	virtual bool texture_is_format_supported_for_usage(DataFormat p_format, BitField<RenderingDevice::TextureUsageBits> p_usage) const;
	virtual bool texture_is_shared(RID p_texture);
	virtual bool texture_is_valid(RID p_texture);
	virtual TextureFormat texture_get_format(RID p_texture);
	virtual Size2i texture_size(RID p_texture);
	virtual uint64_t texture_get_native_handle(RID p_texture);

	virtual Error texture_copy(RID p_from_texture, RID p_to_texture, const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_size, uint32_t p_src_mipmap, uint32_t p_dst_mipmap, uint32_t p_src_layer, uint32_t p_dst_layer, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);
	virtual Error texture_clear(RID p_texture, const Color &p_color, uint32_t p_base_mipmap, uint32_t p_mipmaps, uint32_t p_base_layer, uint32_t p_layers, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);
	virtual Error texture_resolve_multisample(RID p_from_texture, RID p_to_texture, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);

	/*********************/
	/**** FRAMEBUFFER ****/
	/*********************/

	virtual FramebufferFormatID framebuffer_format_create(const Vector<AttachmentFormat> &p_format, uint32_t p_view_count = 1);
	virtual FramebufferFormatID framebuffer_format_create_multipass(const Vector<AttachmentFormat> &p_attachments, const Vector<FramebufferPass> &p_passes, uint32_t p_view_count = 1);
	virtual FramebufferFormatID framebuffer_format_create_empty(TextureSamples p_samples = TEXTURE_SAMPLES_1);
	virtual TextureSamples framebuffer_format_get_texture_samples(FramebufferFormatID p_format, uint32_t p_pass = 0);

	virtual RID framebuffer_create(const Vector<RID> &p_texture_attachments, FramebufferFormatID p_format_check = INVALID_ID, uint32_t p_view_count = 1);
	virtual RID framebuffer_create_multipass(const Vector<RID> &p_texture_attachments, const Vector<FramebufferPass> &p_passes, FramebufferFormatID p_format_check = INVALID_ID, uint32_t p_view_count = 1);
	virtual RID framebuffer_create_empty(const Size2i &p_size, TextureSamples p_samples = TEXTURE_SAMPLES_1, FramebufferFormatID p_format_check = INVALID_ID);
	virtual bool framebuffer_is_valid(RID p_framebuffer) const;
	virtual void framebuffer_set_invalidation_callback(RID p_framebuffer, InvalidationCallback p_callback, void *p_userdata);

	virtual FramebufferFormatID framebuffer_get_format(RID p_framebuffer);

	/*****************/
	/**** SAMPLER ****/
	/*****************/

	virtual RID sampler_create(const SamplerState &p_state);
	virtual bool sampler_is_format_supported_for_filter(DataFormat p_format, SamplerFilter p_sampler_filter) const;

	/**********************/
	/**** VERTEX ARRAY ****/
	/**********************/

	virtual RID vertex_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>(), bool p_use_as_storage = false);

	// Internally reference counted, this ID is warranted to be unique for the same description, but needs to be freed as many times as it was allocated.
	virtual VertexFormatID vertex_format_create(const Vector<VertexAttribute> &p_vertex_formats);
	virtual RID vertex_array_create(uint32_t p_vertex_count, VertexFormatID p_vertex_format, const Vector<RID> &p_src_buffers, const Vector<uint64_t> &p_offsets = Vector<uint64_t>());

	virtual RID index_buffer_create(uint32_t p_size_indices, IndexBufferFormat p_format, const Vector<uint8_t> &p_data = Vector<uint8_t>(), bool p_use_restart_indices = false);

	virtual RID index_array_create(RID p_index_buffer, uint32_t p_index_offset, uint32_t p_index_count);

	/****************/
	/**** SHADER ****/
	/****************/

	virtual String shader_get_binary_cache_key() const;
	virtual Vector<uint8_t> shader_compile_binary_from_spirv(const Vector<ShaderStageSPIRVData> &p_spirv, const String &p_shader_name = "");

	virtual RID shader_create_from_bytecode(const Vector<uint8_t> &p_shader_binary, RID p_placeholder = RID());
	virtual RID shader_create_placeholder();

	virtual uint64_t shader_get_vertex_input_attribute_mask(RID p_shader);

	/*****************/
	/**** UNIFORM ****/
	/*****************/

	virtual RID uniform_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>());
	virtual RID storage_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>(), BitField<StorageBufferUsage> p_usage = 0);
	virtual RID texture_buffer_create(uint32_t p_size_elements, DataFormat p_format, const Vector<uint8_t> &p_data = Vector<uint8_t>());

	virtual RID uniform_set_create(const Vector<Uniform> &p_uniforms, RID p_shader, uint32_t p_shader_set);
	virtual bool uniform_set_is_valid(RID p_uniform_set);
	virtual void uniform_set_set_invalidation_callback(RID p_uniform_set, InvalidationCallback p_callback, void *p_userdata);

	virtual Error buffer_copy(RID p_src_buffer, RID p_dst_buffer, uint32_t p_src_offset, uint32_t p_dst_offset, uint32_t p_size, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);
	virtual Error buffer_update(RID p_buffer, uint32_t p_offset, uint32_t p_size, const void *p_data, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS); // Works for any buffer.
	virtual Error buffer_clear(RID p_buffer, uint32_t p_offset, uint32_t p_size, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);
	virtual Vector<uint8_t> buffer_get_data(RID p_buffer, uint32_t p_offset = 0, uint32_t p_size = 0);

	/*************************/
	/**** RENDER PIPELINE ****/
	/*************************/

	virtual RID render_pipeline_create(RID p_shader, FramebufferFormatID p_framebuffer_format, VertexFormatID p_vertex_format, RenderPrimitive p_render_primitive, const PipelineRasterizationState &p_rasterization_state, const PipelineMultisampleState &p_multisample_state, const PipelineDepthStencilState &p_depth_stencil_state, const PipelineColorBlendState &p_blend_state, BitField<PipelineDynamicStateFlags> p_dynamic_state_flags = 0, uint32_t p_for_render_pass = 0, const Vector<PipelineSpecializationConstant> &p_specialization_constants = Vector<PipelineSpecializationConstant>());
	virtual bool render_pipeline_is_valid(RID p_pipeline);

	/**************************/
	/**** COMPUTE PIPELINE ****/
	/**************************/

	virtual RID compute_pipeline_create(RID p_shader, const Vector<PipelineSpecializationConstant> &p_specialization_constants = Vector<PipelineSpecializationConstant>());
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

	virtual void draw_list_set_blend_constants(DrawListID p_list, const Color &p_color);
	virtual void draw_list_bind_render_pipeline(DrawListID p_list, RID p_render_pipeline);
	virtual void draw_list_bind_uniform_set(DrawListID p_list, RID p_uniform_set, uint32_t p_index);
	virtual void draw_list_bind_vertex_array(DrawListID p_list, RID p_vertex_array);
	virtual void draw_list_bind_index_array(DrawListID p_list, RID p_index_array);
	virtual void draw_list_set_line_width(DrawListID p_list, float p_width);
	virtual void draw_list_set_push_constant(DrawListID p_list, const void *p_data, uint32_t p_data_size);

	virtual void draw_list_draw(DrawListID p_list, bool p_use_indices, uint32_t p_instances = 1, uint32_t p_procedural_vertices = 0);

	virtual void draw_list_enable_scissor(DrawListID p_list, const Rect2 &p_rect);
	virtual void draw_list_disable_scissor(DrawListID p_list);

	virtual uint32_t draw_list_get_current_pass();
	virtual DrawListID draw_list_switch_to_next_pass();
	virtual Error draw_list_switch_to_next_pass_split(uint32_t p_splits, DrawListID *r_split_ids);

	virtual void draw_list_end(BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);

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
	virtual void compute_list_end(BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);

	virtual void barrier(BitField<BarrierMask> p_from = BARRIER_MASK_ALL_BARRIERS, BitField<BarrierMask> p_to = BARRIER_MASK_ALL_BARRIERS);
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

	virtual uint64_t limit_get(Limit p_limit) const;

	virtual void prepare_screen_for_drawing();

	void initialize(D3D12Context *p_context, bool p_local_device = false);
	void finalize();

	virtual void swap_buffers(); // For main device.

	virtual void submit(); // For local device.
	virtual void sync(); // For local device.

	virtual uint32_t get_frame_delay() const;

	virtual RenderingDevice *create_local_device();

	virtual uint64_t get_memory_usage(MemoryType p_type) const;

	virtual void set_resource_name(RID p_id, const String p_name);

	virtual void draw_command_begin_label(String p_label_name, const Color p_color = Color(1, 1, 1, 1));
	virtual void draw_command_insert_label(String p_label_name, const Color p_color = Color(1, 1, 1, 1));
	virtual void draw_command_end_label();

	virtual String get_device_vendor_name() const;
	virtual String get_device_name() const;
	virtual RenderingDevice::DeviceType get_device_type() const;
	virtual String get_device_api_version() const;
	virtual String get_device_pipeline_cache_uuid() const;

	virtual uint64_t get_driver_resource(DriverResource p_resource, RID p_rid = RID(), uint64_t p_index = 0);

	virtual bool has_feature(const Features p_feature) const;

	RenderingDeviceD3D12();
	~RenderingDeviceD3D12();
};

#endif // RENDERING_DEVICE_D3D12_H
