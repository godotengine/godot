/**************************************************************************/
/*  webgpu_objects.h                                                      */
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

#include "servers/rendering/rendering_device_driver.h"

#include <webgpu/webgpu.h>

using RDD = RenderingDeviceDriver;

// =============================================================================
// Buffer
// =============================================================================

struct WGBuffer {
	WGPUBuffer handle = nullptr;
	uint64_t size = 0;
	WGPUBufferUsage usage = 0;
	uint8_t *shadow_map = nullptr; // For CPU-side mapping emulation.
	bool map_dirty = false;
	bool is_readback = false; // True for staging buffers that need GPU→CPU readback.
	bool map_complete = false; // Set by async map callback.
	bool map_pending = false; // True while wgpuBufferMapAsync callback is in flight.
	bool freed = false; // Source freed while map pending; callback will clean up.
	WGPUFuture map_future = {};

	// Dirty range tracking for shadow buffer flushes. On WebGPU, buffer_unmap()
	// must copy shadow_map data to the GPU buffer via wgpuQueueWriteBuffer. Without
	// range tracking, the entire buffer (often 32MB for staging) is copied even when
	// only a few bytes were written — costing ~10ms per unmap on web. These fields
	// limit the flush to the actually-modified region.
	uint64_t dirty_offset = 0;
	uint64_t dirty_end = 0; // Exclusive end. 0 means "no explicit range set."

	// Dynamic persistent rotation (Task 7.5).
	// `frame_idx` is UINT32_MAX for non-dynamic buffers. For BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT
	// buffers, buffer_create sets frame_idx=0 and per_frame_size to the aligned single-frame
	// slice. The physical buffer holds frame_count slices; each frame the CPU writes to one
	// slice and the GPU reads from it via a dynamic offset of frame_idx * per_frame_size.
	uint32_t frame_idx = UINT32_MAX;
	uint64_t per_frame_size = 0;
	bool is_dynamic() const { return frame_idx != UINT32_MAX; }
};

// =============================================================================
// Texture
// =============================================================================

struct WGTexture {
	WGPUTexture handle = nullptr; // null for shared/sliced views (they don't own the GPU texture)
	WGPUTexture view_source = nullptr; // always the owning WGPUTexture; inherited by shared/sliced textures
	WGPUTextureView default_view = nullptr;

	// Returns the underlying WGPUTexture for any GPU op that needs the actual
	// resource (copies, debug labels). Shared / sliced views have handle==nullptr
	// and must read view_source instead — passing nullptr to wgpu* functions
	// causes "Required member is undefined" / dispatch crashes in emscripten.
	// Use this for *any* WGPUTexture read into a Copy/CommandEncoder/etc.
	// Lifecycle paths (creation, release) still touch handle directly so they
	// only operate on owning textures.
	inline WGPUTexture gpu_handle() const {
		DEV_ASSERT(view_source || handle);
		return view_source ? view_source : handle;
	}
	WGPUTextureFormat format = WGPUTextureFormat_Undefined;
	// Godot-side DataFormat remembered at creation so readback / allocation-size
	// paths can look up true bytes-per-pixel instead of hardcoding bpp=4.
	RDD::DataFormat rd_format = RDD::DATA_FORMAT_MAX;
	WGPUTextureDimension dimension = WGPUTextureDimension_2D;
	WGPUTextureViewDimension view_dimension = WGPUTextureViewDimension_2D;
	uint32_t width = 0, height = 0, depth = 1;
	uint32_t mipmaps = 1;
	uint32_t layers = 1;
	uint32_t sample_count = 1;
	// For slice views (texture_create_shared_from_slice), the base offset of
	// this view into the parent texture. 0/0 for non-slice textures.
	uint32_t base_mipmap = 0;
	uint32_t base_layer = 0;
	WGPUTextureUsage usage = 0;
	// sRGB/linear sibling listed in viewFormats at creation; Undefined if views can't change format.
	WGPUTextureFormat view_format_alt = WGPUTextureFormat_Undefined;
};

// =============================================================================
// Sampler (stored directly as WGPUSampler handle in SamplerID)
// =============================================================================

// =============================================================================
// Vertex Format
// =============================================================================

struct WGVertexFormat {
	struct Attribute {
		uint32_t location = 0;
		WGPUVertexFormat format = (WGPUVertexFormat)0; // 0 = no valid format (Undefined was removed in Dawn)
		uint64_t offset = 0;
		uint32_t binding = 0;
	};
	struct Binding {
		uint32_t stride = 0;
		WGPUVertexStepMode step_mode = WGPUVertexStepMode_Vertex;
	};
	LocalVector<Attribute> attributes;
	LocalVector<Binding> bindings;
};

// =============================================================================
// Shader
// =============================================================================

struct WGShader {
	// Per-stage shader modules. Indexed by RDD::ShaderStage enum value.
	WGPUShaderModule stage_modules[6] = {}; // SHADER_STAGE_MAX = 6 (vertex/frag/tess×2/compute/max)
	// WGSL whose module is created by the first pipeline using it; most variants never get one.
	CharString stage_wgsl[6];
	// Never reused, unlike the WGShader address or its layout handles; keys WGUniformSet::rebind_cache.
	uint64_t uid = 0;

	HashMap<uint64_t, uint32_t> binding_remaps; // (set << 32 | Godot binding) -> first scalar binding.

	// Per-stage raw SPIR-V bytes. Stored for deferred specialization constant patching.
	PackedByteArray stage_spirv[6];

	WGPUPipelineLayout pipeline_layout = nullptr;
	LocalVector<WGPUBindGroupLayout> bind_group_layouts; // One per descriptor set.

	// Push constant emulation.
	uint32_t push_constant_size = 0;
	uint32_t push_constant_bind_group = UINT32_MAX; // UINT32_MAX = no push constants.
	uint32_t push_constant_binding = 0;
	// Merged bind group layout for the PC group when it also has material uniforms.
	// nullptr = PC group has no other uniforms; pipeline uses the universal PC-only layout.
	WGPUBindGroupLayout merged_pc_group_layout = nullptr;

	// Reflection data for uniform set creation:
	//   bind_group_infos[set_index].entries[binding_index] → layout entry + Godot type.
	struct BindGroupEntry {
		WGPUBindGroupLayoutEntry layout_entry = {};
		RDD::UniformType godot_type = RDD::UNIFORM_TYPE_MAX;
		uint32_t array_length = 0; // >1 for binding_array (e.g., texture arrays)
	};
	struct BindGroupInfo {
		LocalVector<BindGroupEntry> entries;
	};
	LocalVector<BindGroupInfo> bind_group_infos;

	// Depth alias bindings: maps (set << 16 | alias_binding) → depth_binding.
	// Used by uniform_set_create to add extra bind group entries for split depth textures.
	HashMap<uint32_t, uint32_t> depth_alias_bindings;

	// Read-only storage texture → sampled texture conversions.
	// Set of (set << 16 | binding) keys. When readonly-and-readwrite-storage-textures
	// is unavailable, read-only storage textures are converted to sampled textures
	// in the WGSL and BGL so the shader compiles on Firefox/wgpu.
	HashSet<uint32_t> read_storage_to_sampled;

	// (set << 16 | binding) of storage buffers the vertex stage declares. Fragment read_write
	// declarations of these are demoted to read: a BGL entry can't be writable and vertex-visible.
	HashSet<uint32_t> vertex_storage_buffers;

	// Pipeline layout gap indices — slots between set_count and pc_group that
	// need empty bind groups bound before draw (Firefox/wgpu requirement).
	LocalVector<uint32_t> gap_bind_group_indices;

	String name;

	// True if the base shader modules were compiled with WGSL override
	// declarations (spirv_to_wgsl_with_overrides). When set, specialization
	// constants are passed as WGPUConstantEntry at pipeline creation time
	// instead of patching SPIR-V + runtime Tint conversion.
	bool has_override_declarations = false;

	// Per-stage set of override constant IDs found in the WGSL output.
	// Used to filter WGPUConstantEntry per stage — WebGPU requires that
	// every constant key passed to a stage actually exists in that module.
	HashSet<uint32_t> stage_override_ids[6];
};

// =============================================================================
// Render Pass (metadata — no WebGPU object)
// =============================================================================

struct WGRenderPass {
	struct SubpassInfo {
		LocalVector<RDD::AttachmentReference> input_references;
		LocalVector<RDD::AttachmentReference> color_references;
		RDD::AttachmentReference depth_stencil_reference;
		LocalVector<RDD::AttachmentReference> resolve_references;
	};
	LocalVector<RDD::Attachment> attachments;
	LocalVector<SubpassInfo> subpasses;
	uint32_t view_count = 0;
	bool is_swap_chain_pass = false; // True for swap chain render passes — used to strip alpha writes.
};

// =============================================================================
// Framebuffer (metadata — no WebGPU object)
// =============================================================================

struct WGFramebuffer {
	LocalVector<WGTexture *> attachments;
	LocalVector<WGPUTextureView> attachment_views;
	uint32_t width = 0, height = 0;
	WGRenderPass *render_pass = nullptr;
};

// =============================================================================
// Pipeline
// =============================================================================

struct WGPipelineWrapper {
	enum Type { RENDER,
		COMPUTE };
	Type type = RENDER;
	union {
		WGPURenderPipeline render_handle;
		WGPUComputePipeline compute_handle;
	};
	WGShader *shader = nullptr;
	// Specialized shader modules created with pipeline-specific specialization constants.
	// If non-null, these are owned by this pipeline and must be released.
	WGPUShaderModule specialized_modules[6] = {};
	// WebGPU has no pipeline-level stencil reference — it must be set dynamically
	// via wgpuRenderPassEncoderSetStencilReference(). Stored here at creation time,
	// applied when the pipeline is bound.
	uint32_t stencil_reference = 0;
	// WebGPU bakes stripIndexFormat into the pipeline state, but Godot only knows
	// the index format at bind time. For strip topologies we create both Uint16 and
	// Uint32 variants and select the correct one at draw time.
	WGPURenderPipeline render_handle_u16 = nullptr;
	bool is_strip = false;
	// Creation arguments of a pipeline that is created on first bind (web only).
	struct WGDeferredPipeline *deferred = nullptr;

	WGPipelineWrapper() : render_handle(nullptr) {}
};

struct WGDeferredPipeline {
	RDD::VertexFormatID vertex_format;
	RDD::RenderPrimitive render_primitive = RDD::RENDER_PRIMITIVE_TRIANGLES;
	RDD::PipelineRasterizationState rasterization_state;
	RDD::PipelineMultisampleState multisample_state;
	RDD::PipelineDepthStencilState depth_stencil_state;
	RDD::PipelineColorBlendState blend_state;
	LocalVector<int32_t> color_attachments;
	BitField<RDD::PipelineDynamicStateFlags> dynamic_state;
	RDD::RenderPassID render_pass;
	uint32_t render_subpass = 0;
	LocalVector<RDD::PipelineSpecializationConstant> specialization_constants;
};

// =============================================================================
// Uniform Set (Bind Group)
// =============================================================================

struct WGUniformSet {
	WGPUBindGroup handle = nullptr;
	uint32_t set_index = 0;
	LocalVector<WGPUTextureView> temp_views; // Re-dimensioned views for Cube↔2D fixups.
	// Rebind cache: when a bind group is created with shader A's BGL but
	// needs to be used with shader B's pipeline (different BGL), we
	// re-create the bind group with the correct BGL and cache it.
	LocalVector<WGPUBindGroupEntry> cached_entries;
	WGShader *source_shader = nullptr;
	HashMap<uint64_t, WGPUBindGroup> rebind_cache; // Key: target WGShader::uid << 8 | set index.

	// Maps binding index → WGTexture* for texture entries, used during rebind
	// to create compatible views when the target BGL has a different dimension.
	HashMap<uint32_t, WGTexture *> bound_textures;

	// Dynamic buffers in binding order (Task 7.5). Populated by uniform_set_create
	// whenever a UNIFORM_TYPE_*_BUFFER_DYNAMIC binding resolves to a WGBuffer with
	// is_dynamic()==true. command_bind_*_uniform_sets walks this list in order to
	// unpack per-set dynamic offsets from the mask returned by
	// uniform_sets_get_dynamic_offsets and pass them to wgpuRenderPassEncoderSetBindGroup.
	LocalVector<WGBuffer *> dynamic_buffers;
};

// =============================================================================
// Swap Chain
// =============================================================================

struct WGSwapChain {
	WGPUSurface surface = nullptr;
	WGPUTextureFormat format = WGPUTextureFormat_BGRA8Unorm;
	uint32_t width = 0, height = 0;
	WGRenderPass *render_pass = nullptr;
	uint64_t surface_id = 0; // Context driver SurfaceID — used to look up size on resize.
	WGPUTexture current_texture = nullptr; // Released after each frame.
	WGPUTextureView current_view = nullptr;
	WGFramebuffer *current_framebuffer = nullptr;
	bool configured = false;
};

// =============================================================================
// Command Queue / Pool / Buffer
// =============================================================================

struct WGCommandQueue {
	WGPUQueue queue = nullptr;
};

struct WGCommandBuffer;

struct WGCommandPool {
	RDD::CommandBufferType buffer_type = RDD::COMMAND_BUFFER_TYPE_PRIMARY;
	LocalVector<WGCommandBuffer *> buffers; // Owned; freed with the pool like Vulkan command buffers.
};

struct WGCommandBuffer {
	WGPUCommandEncoder encoder = nullptr;
	WGPUCommandBuffer finished_buffer = nullptr;

	// Current active encoder states.
	WGPURenderPassEncoder render_encoder = nullptr;
	WGPUComputePassEncoder compute_encoder = nullptr;

	enum ActiveEncoder { NONE,
		RENDER,
		COMPUTE };
	ActiveEncoder active_encoder = NONE;

	// Push constant emulation state.
	static constexpr uint32_t MAX_PUSH_CONSTANT_SIZE = 128;
	uint8_t push_constant_data[MAX_PUSH_CONSTANT_SIZE] = {};
	uint32_t push_constant_data_len = 0;
	bool push_constants_dirty = false;
	// Combined bind group for the PC group (includes both material resources and PC ring buffer).
	// Updated by command_bind_render_uniform_sets when the PC group is bound.
	WGPUBindGroup current_pc_bind_group = nullptr;
	// Last PC ring offset successfully flushed by _flush_push_constants.
	// Used by command_bind_render_uniform_sets to set the correct dynamic offset
	// when rebinding the merged PC group (avoids stale offset=0 placeholder).
	uint32_t last_flushed_pc_offset = 0;
	// Task 7.5: When the PC group also contains material dynamic-offset UBOs,
	// _flush_push_constants must preserve those offsets while patching in the new
	// PC ring offset. Stored in binding order; the PC ring's own slot is appended
	// at pc_group_material_dyn_count by _flush_push_constants each draw.
	static constexpr uint32_t MAX_PC_GROUP_MATERIAL_DYN = 7; // MAX_DYNAMIC_BUFFERS - 1 (PC ring takes 1 slot).
	uint32_t pc_group_material_dyn_offsets[MAX_PC_GROUP_MATERIAL_DYN] = {};
	uint32_t pc_group_material_dyn_count = 0;

	// Render state tracking.
	struct RenderState {
		WGRenderPass *render_pass = nullptr;
		WGFramebuffer *framebuffer = nullptr;
		uint32_t current_subpass = 0;
		WGPipelineWrapper *current_pipeline = nullptr;
		WGPUIndexFormat current_index_format = WGPUIndexFormat_Uint32;
		// Right and bottom edges of the render area, which scissors are clamped to.
		uint32_t render_area_width = 0;
		uint32_t render_area_height = 0;

		// Dynamic state of the current pass, re-applied when a push constant page restarts it.
		bool viewport_set = false;
		float viewport[4] = {};
		bool scissor_set = false;
		uint32_t scissor[4] = {};
		bool blend_constant_set = false;
		WGPUColor blend_constant = {};

		// Redundant state elimination for per-draw calls.
		WGPUBuffer current_index_buffer = nullptr;
		uint64_t current_index_offset = 0;
		static constexpr uint32_t MAX_VERTEX_BINDINGS = 8;
		WGPUBuffer current_vertex_buffers[MAX_VERTEX_BINDINGS] = {};
		uint64_t current_vertex_offsets[MAX_VERTEX_BINDINGS] = {};
		// Textures used as render attachments in the CURRENT pass only.
		// Used to detect intra-pass sync scope violations when a texture
		// has both TextureBinding and RenderAttachment usage in the same pass.
		// Note: cross-pass synchronization is guaranteed by the WebGPU spec
		// (passes within a single command encoder execute sequentially with
		// implicit barriers at usage scope boundaries).
		static constexpr uint32_t MAX_ATTACHMENT_TEXTURES = 64;
		WGPUTexture current_pass_attachments[MAX_ATTACHMENT_TEXTURES] = {};
		uint32_t current_pass_attachment_count = 0;
		void reset_current_pass_attachments() { current_pass_attachment_count = 0; }
		void add_current_pass_attachment(WGPUTexture t) {
			if (!t) {
				return;
			}
			for (uint32_t i = 0; i < current_pass_attachment_count; i++) {
				if (current_pass_attachments[i] == t) {
					return;
				}
			}
			if (current_pass_attachment_count < MAX_ATTACHMENT_TEXTURES) {
				current_pass_attachments[current_pass_attachment_count++] = t;
			}
		}
	} render_state;

	// Bind group state tracking for redundancy elimination.
	// Tracks what's currently bound at each bind group slot to skip redundant SetBindGroup calls.
	static constexpr uint32_t MAX_BIND_GROUPS = 4;
	WGPUBindGroup bound_bind_groups[MAX_BIND_GROUPS] = {};
	WGShader *bound_shader = nullptr; // Shader whose BGL the bound groups were created for.

	// Full bind group state for mid-pass restart (new push constant page).
	// Stores the resolved bind group + dynamic offsets for each slot so we can
	// rebind everything after ending/restarting the render pass.
	static constexpr uint32_t MAX_BIND_GROUP_DYN_OFFSETS = 9; // 8 material + 1 PC ring
	struct BoundGroupState {
		WGPUBindGroup group = nullptr;
		uint32_t dynamic_offsets[MAX_BIND_GROUP_DYN_OFFSETS] = {};
		uint32_t dynamic_offset_count = 0;
	};
	BoundGroupState last_bound_state[MAX_BIND_GROUPS] = {};

	void invalidate_bind_groups() {
		for (uint32_t i = 0; i < MAX_BIND_GROUPS; i++) {
			bound_bind_groups[i] = nullptr;
			last_bound_state[i] = {}; // A pass restart must not rebind groups from an older pass.
		}
		bound_shader = nullptr;
		current_pc_bind_group = nullptr;
		last_flushed_pc_offset = 0;
		pc_group_material_dyn_count = 0;
	}

	// Helpers.
	void end_active_encoder() {
		if (render_encoder) {
			wgpuRenderPassEncoderEnd(render_encoder);
			wgpuRenderPassEncoderRelease(render_encoder);
			render_encoder = nullptr;
		}
		if (compute_encoder) {
			wgpuComputePassEncoderEnd(compute_encoder);
			wgpuComputePassEncoderRelease(compute_encoder);
			compute_encoder = nullptr;
		}
		active_encoder = NONE;
	}
};

// =============================================================================
// Fence
// =============================================================================

struct WGFence {
	uint32_t pending_callbacks = 0; // wgpuQueueOnSubmittedWorkDone callbacks in flight (a fence can be resubmitted).
	bool freed = false; // Freed while callbacks pending; the last callback deletes it.
	WGPUFuture future = {}; // Most recent submission.
};

// =============================================================================
// Semaphore (mostly no-op on WebGPU — single queue)
// =============================================================================

struct WGSemaphore {};

#endif // WEBGPU_ENABLED
