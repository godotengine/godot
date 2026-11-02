/**************************************************************************/
/*  rendering_device_driver_webgpu.cpp                                    */
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

#ifdef WEBGPU_ENABLED

#include "rendering_device_driver_webgpu.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/templates/hash_map.h"
#include "drivers/webgpu/pixel_formats_webgpu.h"
#include "drivers/webgpu/rendering_context_driver_webgpu.h"
#include "drivers/webgpu/rendering_shader_container_webgpu.h"
#include "drivers/webgpu/shader_metadata_webgpu.h"
#include "drivers/webgpu/spirv_preprocess.h"
#include "drivers/webgpu/tint_wrapper.h"

#include <webgpu/webgpu.h>
#ifdef WEB_ENABLED
#include <emscripten/emscripten.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Define WEBGPU_VERBOSE to enable diagnostic console.log prints in the browser.
// These are disabled by default for production builds.
// #define WEBGPU_VERBOSE

#if defined(WEBGPU_VERBOSE) && defined(WEB_ENABLED)
#define WEBGPU_DIAG(...) EM_ASM(__VA_ARGS__)
#define WEBGPU_DIAG_INT(...) EM_ASM_INT(__VA_ARGS__)
#else
#define WEBGPU_DIAG(...) ((void)0)
#define WEBGPU_DIAG_INT(...) 0
#endif

#ifdef WEB_ENABLED
static constexpr WGPUCallbackMode WEBGPU_CALLBACK_MODE = WGPUCallbackMode_AllowSpontaneous;
#else
// Deliver native completion callbacks while polling on the engine owner thread.
static constexpr WGPUCallbackMode WEBGPU_CALLBACK_MODE = WGPUCallbackMode_AllowProcessEvents;
#endif

// Fence work-done callback: fires when wgpuQueueSubmit work completes on GPU.
// Newer emdawnwebgpu packages add a WGPUStringView message parameter to
// WGPUQueueWorkDoneCallback (4 params); earlier ones have 3. Macro presence
// does NOT discriminate the two (both define WGPU_STRLEN), so provide both
// signatures as overloads and let the assignment to the port's own
// WGPUQueueWorkDoneCallback typedef select the matching one.
static void _fence_work_done_body(void *p_userdata1) {
	WGFence *fence = (WGFence *)p_userdata1;
	if (!fence) {
		return;
	}
	fence->pending_callbacks--;
	if (fence->freed && fence->pending_callbacks == 0) {
		memdelete(fence);
	}
}

[[maybe_unused]] static void _fence_work_done_callback(WGPUQueueWorkDoneStatus p_status, void *p_userdata1, void *p_userdata2) {
	_fence_work_done_body(p_userdata1);
}

[[maybe_unused]] static void _fence_work_done_callback(WGPUQueueWorkDoneStatus p_status, WGPUStringView p_message, void *p_userdata1, void *p_userdata2) {
	_fence_work_done_body(p_userdata1);
}

// Parse "@group(G[u]) @binding(B[u])" from a WGSL string.
// Handles the optional 'u'/'i' suffix that Tint emits for integer literals.
// Returns true and sets r_grp/r_bnd on success.
static bool parse_group_binding(const char *p, unsigned int &r_grp, unsigned int &r_bnd) {
	if (strncmp(p, "@group(", 7) != 0) {
		return false;
	}
	const char *g = p + 7;
	char *end = nullptr;
	unsigned long grp = strtoul(g, &end, 10);
	if (end == g) {
		return false;
	}
	if (*end == 'u' || *end == 'i') {
		end++;
	}
	if (*end != ')') {
		return false;
	}
	end++; // skip ')'
	while (*end == ' ') {
		end++;
	}
	if (strncmp(end, "@binding(", 9) != 0) {
		return false;
	}
	const char *b = end + 9;
	unsigned long bnd = strtoul(b, &end, 10);
	if (end == b) {
		return false;
	}
	if (*end == 'u' || *end == 'i') {
		end++;
	}
	if (*end != ')') {
		return false;
	}
	r_grp = (unsigned int)grp;
	r_bnd = (unsigned int)bnd;
	return true;
}

// =============================================================================
// Storage Texture Format Validation
// =============================================================================

// Returns true for 32-bit float formats that require the float32-filterable
// feature for linear sampling (R32Float, RG32Float, RGBA32Float).
static bool _is_float32_format(WGPUTextureFormat p_format) {
	switch (p_format) {
		case WGPUTextureFormat_R32Float:
		case WGPUTextureFormat_RG32Float:
		case WGPUTextureFormat_RGBA32Float:
			return true;
		default:
			return false;
	}
}

// Maps a WGPUTextureFormat to the WGPUTextureSampleType needed for a sampled
// texture BGL entry. Integer formats → Uint/Sint, everything else → UnfilterableFloat.
static WGPUTextureSampleType _texture_sample_type_for_format(WGPUTextureFormat p_format) {
	if (p_format == WGPUTextureFormat_RGB10A2Uint) {
		return WGPUTextureSampleType_Uint;
	}
	uint32_t channels = 0;
	WGTexelType type = WG_TEXEL_FLOAT32;
	if (wgpu_format_texel_info(p_format, channels, type)) {
		if (type == WG_TEXEL_UINT8 || type == WG_TEXEL_UINT16 || type == WG_TEXEL_UINT32) {
			return WGPUTextureSampleType_Uint;
		}
		if (type == WG_TEXEL_SINT8 || type == WG_TEXEL_SINT16 || type == WG_TEXEL_SINT32) {
			return WGPUTextureSampleType_Sint;
		}
	}
	return WGPUTextureSampleType_UnfilterableFloat;
}

// Sample type of a sampled-texture BGL entry, from the WGSL declaration at that binding.
static WGPUTextureSampleType _bgl_texture_sample_type(bool p_depth, bool p_multisampled, WGPUTextureSampleType p_wgsl_type) {
	if (p_depth) {
		return WGPUTextureSampleType_Depth;
	}
	if (p_wgsl_type == WGPUTextureSampleType_Uint || p_wgsl_type == WGPUTextureSampleType_Sint || p_wgsl_type == WGPUTextureSampleType_UnfilterableFloat) {
		return p_wgsl_type;
	}
	// Filtering is illegal for multisampled textures.
	return p_multisampled ? WGPUTextureSampleType_UnfilterableFloat : WGPUTextureSampleType_Float;
}

// Whether a texture bound with this sample type reads its depth aspect, which needs a depth-only view.
static bool _reads_depth(WGPUTextureSampleType p_sample_type, WGPUTextureFormat p_format) {
	return p_sample_type == WGPUTextureSampleType_Depth || (p_sample_type == WGPUTextureSampleType_UnfilterableFloat && is_depth_format_wgpu(p_format));
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

RenderingDeviceDriverWebGPU::RenderingDeviceDriverWebGPU(RenderingContextDriverWebGPU *p_context_driver) {
	context_driver = p_context_driver;
}

RenderingDeviceDriverWebGPU::~RenderingDeviceDriverWebGPU() {
	for (const KeyValue<ClearPipelineKey, WGPURenderPipeline> &kv : clear_pipelines) {
		wgpuRenderPipelineRelease(kv.value);
	}
	clear_pipelines.clear();
	if (clear_pipeline_layout) {
		wgpuPipelineLayoutRelease(clear_pipeline_layout);
		clear_pipeline_layout = nullptr;
	}

	// Release push constant resources.
	if (push_constant_bind_group) {
		wgpuBindGroupRelease(push_constant_bind_group);
		push_constant_bind_group = nullptr;
	}
	if (push_constant_bind_group_layout) {
		wgpuBindGroupLayoutRelease(push_constant_bind_group_layout);
		push_constant_bind_group_layout = nullptr;
	}
	if (push_constant_ring_buffer) {
		wgpuBufferRelease(push_constant_ring_buffer);
		push_constant_ring_buffer = nullptr;
	}
	for (WGPUBuffer page : push_constant_pages) {
		wgpuBufferRelease(page);
	}
	push_constant_pages.clear();
	if (empty_bind_group) {
		wgpuBindGroupRelease(empty_bind_group);
		empty_bind_group = nullptr;
	}
	if (empty_bind_group_layout) {
		wgpuBindGroupLayoutRelease(empty_bind_group_layout);
		empty_bind_group_layout = nullptr;
	}

	// Release fallback textures and views.
	if (fallback_float_texture_view) {
		wgpuTextureViewRelease(fallback_float_texture_view);
		fallback_float_texture_view = nullptr;
	}
	if (fallback_float_texture) {
		wgpuTextureRelease(fallback_float_texture);
		fallback_float_texture = nullptr;
	}
	if (fallback_cube_texture_view) {
		wgpuTextureViewRelease(fallback_cube_texture_view);
		fallback_cube_texture_view = nullptr;
	}
	if (fallback_cube_texture) {
		wgpuTextureRelease(fallback_cube_texture);
		fallback_cube_texture = nullptr;
	}
	if (fallback_ms_texture_view) {
		wgpuTextureViewRelease(fallback_ms_texture_view);
		fallback_ms_texture_view = nullptr;
	}
	if (fallback_ms_texture) {
		wgpuTextureRelease(fallback_ms_texture);
		fallback_ms_texture = nullptr;
	}

	// Release dummy samplers.
	if (dummy_filtering_sampler) {
		wgpuSamplerRelease(dummy_filtering_sampler);
		dummy_filtering_sampler = nullptr;
	}
	if (dummy_comparison_sampler) {
		wgpuSamplerRelease(dummy_comparison_sampler);
		dummy_comparison_sampler = nullptr;
	}
	for (const KeyValue<uint64_t, Pair<WGPUTexture, WGPUTextureView>> &kv : unused_binding_views) {
		wgpuTextureViewRelease(kv.value.second);
		wgpuTextureRelease(kv.value.first);
	}
	unused_binding_views.clear();

	// Clean up readback cache — release persistent staging buffers and shadow memory.
	for (KeyValue<uint64_t, ReadbackEntry *> &kv : _readback_cache) {
		ReadbackEntry *entry = kv.value;
		if (entry->map_pending) {
			entry->cancelled = true;
			continue; // The callback owns cleanup until the map finishes.
		}
		if (entry->staging) {
			wgpuBufferRelease(entry->staging);
		}
		if (entry->shadow) {
			memfree(entry->shadow);
		}
		memdelete(entry);
	}
	_readback_cache.clear();

	if (shader_container_format) {
		memdelete(shader_container_format);
		shader_container_format = nullptr;
	}
}

// =============================================================================
// GENERIC
// =============================================================================

Error RenderingDeviceDriverWebGPU::initialize(uint32_t p_device_index, uint32_t p_frame_count) {
	// Initialize Tint shader compiler (SPIR-V → WGSL).
	tint_wrapper_initialize();

	device = context_driver->get_device();
	queue = context_driver->get_queue();
	ERR_FAIL_COND_V(device == nullptr, ERR_CANT_CREATE);
	ERR_FAIL_COND_V(queue == nullptr, ERR_CANT_CREATE);

	frame_count = p_frame_count;

	// Query device limits.
	_check_capabilities();

	// Create push constant ring buffer.
	{
		WGPUBufferDescriptor desc = {};
		desc.size = PUSH_CONSTANT_RING_SIZE;
		desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
		desc.mappedAtCreation = false;
		push_constant_ring_buffer = wgpuDeviceCreateBuffer(device, &desc);
		ERR_FAIL_COND_V(push_constant_ring_buffer == nullptr, ERR_CANT_CREATE);
		push_constant_shadow.resize(PUSH_CONSTANT_RING_SIZE);
	}

	// Create a universal push constant bind group layout:
	//   PUSH_CONSTANT_RING_BINDING, all stages, read-only storage buffer with dynamic offset.
	// All shaders with push constants use this same layout for their push
	// constant slot in the pipeline layout (hasDynamicOffset=true allows
	// reusing one bind group with different ring buffer offsets per draw).
	{
		WGPUBindGroupLayoutEntry pc_entry = {};
		pc_entry.binding = PUSH_CONSTANT_RING_BINDING;
		pc_entry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
		pc_entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
		pc_entry.buffer.hasDynamicOffset = true;
		pc_entry.buffer.minBindingSize = 0; // Flexible — works for any push constant size.

		WGPUBindGroupLayoutDescriptor layout_desc = {};
		layout_desc.entryCount = 1;
		layout_desc.entries = &pc_entry;
		push_constant_bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &layout_desc);
		ERR_FAIL_COND_V(push_constant_bind_group_layout == nullptr, ERR_CANT_CREATE);

		// Create the bind group (backed by the ring buffer).
		// Dynamic offset is applied at bind time via SetBindGroup(..., 1, &offset).
		WGPUBindGroupEntry bg_entry = {};
		bg_entry.binding = PUSH_CONSTANT_RING_BINDING;
		bg_entry.buffer = push_constant_ring_buffer;
		bg_entry.offset = 0;
		bg_entry.size = PUSH_CONSTANT_SLOT_ALIGNMENT;

		WGPUBindGroupDescriptor bg_desc = {};
		bg_desc.layout = push_constant_bind_group_layout;
		bg_desc.entryCount = 1;
		bg_desc.entries = &bg_entry;
		push_constant_bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);
		ERR_FAIL_COND_V(push_constant_bind_group == nullptr, ERR_CANT_CREATE);

		// Region clears read their colors from the push constant ring at group 0.
		WGPUPipelineLayoutDescriptor pl_desc = {};
		pl_desc.bindGroupLayoutCount = 1;
		pl_desc.bindGroupLayouts = &push_constant_bind_group_layout;
		clear_pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pl_desc);
		ERR_FAIL_COND_V(clear_pipeline_layout == nullptr, ERR_CANT_CREATE);
		clear_shader.push_constant_bind_group = 0;
		clear_shader.push_constant_size = WGCommandBuffer::MAX_PUSH_CONSTANT_SIZE;
	}

	// Create a persistent empty bind group for pipeline layout gaps.
	// Firefox/wgpu requires all bind group slots to be set before draw calls,
	// even if the BGL at that slot has zero entries.
	{
		WGPUBindGroupLayoutDescriptor empty_desc = {};
		empty_desc.entryCount = 0;
		empty_bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &empty_desc);
		ERR_FAIL_COND_V(empty_bind_group_layout == nullptr, ERR_CANT_CREATE);

		WGPUBindGroupDescriptor bg_desc = {};
		bg_desc.layout = empty_bind_group_layout;
		bg_desc.entryCount = 0;
		empty_bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);
		ERR_FAIL_COND_V(empty_bind_group == nullptr, ERR_CANT_CREATE);
	}

	// Create a small fallback float texture (4x4, RGBA8Unorm, all zeros).
	// Used when Godot provides a depth-format fallback texture to a BGL entry
	// that expects Float sample type. WebGPU requires format/sample-type match.
	{
		WGPUTextureDescriptor td = {};
		td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
		td.dimension = WGPUTextureDimension_2D;
		td.size = { 4, 4, 1 };
		td.format = WGPUTextureFormat_RGBA8Unorm;
		td.mipLevelCount = 1;
		td.sampleCount = 1;
		fallback_float_texture = wgpuDeviceCreateTexture(device, &td);
		if (fallback_float_texture) {
			WGPUTextureViewDescriptor vd = {};
			vd.format = WGPUTextureFormat_RGBA8Unorm;
			vd.dimension = WGPUTextureViewDimension_2D;
			vd.mipLevelCount = 1;
			vd.arrayLayerCount = 1;
			vd.aspect = WGPUTextureAspect_All;
			fallback_float_texture_view = wgpuTextureCreateView(fallback_float_texture, &vd);
		}
	}

	// Create fallback cube texture (4x4, 6 layers) for BGL entries expecting Cube views.
	{
		WGPUTextureDescriptor td = {};
		td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
		td.dimension = WGPUTextureDimension_2D;
		td.size = { 4, 4, 6 };
		td.format = WGPUTextureFormat_RGBA8Unorm;
		td.mipLevelCount = 1;
		td.sampleCount = 1;
		fallback_cube_texture = wgpuDeviceCreateTexture(device, &td);
		if (fallback_cube_texture) {
			WGPUTextureViewDescriptor vd = {};
			vd.format = WGPUTextureFormat_RGBA8Unorm;
			vd.dimension = WGPUTextureViewDimension_Cube;
			vd.mipLevelCount = 1;
			vd.arrayLayerCount = 6;
			vd.aspect = WGPUTextureAspect_All;
			fallback_cube_texture_view = wgpuTextureCreateView(fallback_cube_texture, &vd);
		}
	}

	// Create fallback multisampled texture (4x4, 4x MSAA, RenderAttachment usage
	// required for MSAA textures). Used by ResolveRasterShaderRD and similar MSAA
	// depth resolve shaders when the source is a depth format (can't be sampled
	// as UnfilterableFloat in WebGPU). The shader reads zeros but no GPU errors.
	{
		WGPUTextureDescriptor td = {};
		// MSAA textures require RenderAttachment usage; we never render to this
		// texture but the usage flag is still required by the WebGPU spec.
		td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment;
		td.dimension = WGPUTextureDimension_2D;
		td.size = { 4, 4, 1 };
		td.format = WGPUTextureFormat_R32Float; // Non-filterable float.
		td.mipLevelCount = 1;
		td.sampleCount = 4;
		fallback_ms_texture = wgpuDeviceCreateTexture(device, &td);
		if (fallback_ms_texture) {
			WGPUTextureViewDescriptor vd = {};
			vd.format = WGPUTextureFormat_R32Float;
			vd.dimension = WGPUTextureViewDimension_2D;
			vd.mipLevelCount = 1;
			vd.arrayLayerCount = 1;
			vd.aspect = WGPUTextureAspect_All;
			fallback_ms_texture_view = wgpuTextureCreateView(fallback_ms_texture, &vd);
		}
	}

	// Create dummy samplers for BGL rebinding.
	// When a bind group must be re-created with a different BGL (Comparison↔Filtering),
	// these dummy samplers are substituted for the mismatched entries.
	{
		WGPUSamplerDescriptor sd = {};
		sd.addressModeU = WGPUAddressMode_ClampToEdge;
		sd.addressModeV = WGPUAddressMode_ClampToEdge;
		sd.addressModeW = WGPUAddressMode_ClampToEdge;
		sd.magFilter = WGPUFilterMode_Linear;
		sd.minFilter = WGPUFilterMode_Linear;
		sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
		sd.maxAnisotropy = 1;
		dummy_filtering_sampler = wgpuDeviceCreateSampler(device, &sd);

		WGPUSamplerDescriptor csd = {};
		csd.addressModeU = WGPUAddressMode_ClampToEdge;
		csd.addressModeV = WGPUAddressMode_ClampToEdge;
		csd.addressModeW = WGPUAddressMode_ClampToEdge;
		csd.magFilter = WGPUFilterMode_Linear;
		csd.minFilter = WGPUFilterMode_Linear;
		csd.mipmapFilter = WGPUMipmapFilterMode_Linear;
		csd.maxAnisotropy = 1;
		csd.compare = WGPUCompareFunction_Less;
		dummy_comparison_sampler = wgpuDeviceCreateSampler(device, &csd);
	}

	// Create shader container format.
	shader_container_format = memnew(RenderingShaderContainerFormatWebGPU);

	// Always-on: uncaptured error listener so WebGPU validation errors appear
	// in the browser console before any abort(). Lightweight — no extra API calls.
#ifdef WEB_ENABLED
	MAIN_THREAD_EM_ASM({
		var d = Module['preinitializedWebGPUDevice'];
		if (d && !d._uncapturedPatched) {
			d.addEventListener(
					'uncapturederror', function(e) { console.error('[Godot-WebGPU] uncaptured error: ' + e.error.constructor.name + ' | ' + e.error.message); });
			d._uncapturedPatched = true;
		}
		if (d && d.lost && !d._lostPatched) {
			d.lost.then(function(info) {
				console.error('[Godot-WebGPU] DEVICE LOST: reason=' + info.reason + ' | ' + info.message);
			});
			d._lostPatched = true;
		}
	});
#endif

	// Install main-thread JS diagnostic patches early — as soon as the device is
	// ready, before any pipelines are created. This ensures we intercept EVERY
	// createRenderPipeline/createShaderModule/createBindGroupLayout call. We also
	// kick off a parallel createRenderPipelineAsync() to catch validation errors
	// that the error-scope path misses (Dawn may defer validation past popErrorScope).
	//
	// Gated behind WEBGPU_VERBOSE because the parallel async pipeline creation
	// doubles every render-pipeline compile, and the per-module getCompilationInfo()
	// forces a sync compile path. On shiny_gen with ~383 shaders, leaving these on
	// adds ~12s to startup. Re-enable by uncommenting #define WEBGPU_VERBOSE above.
#if defined(WEBGPU_VERBOSE) && defined(WEB_ENABLED)
	MAIN_THREAD_EM_ASM({
		var d = Module['preinitializedWebGPUDevice'];
		if (!d) {
			console.error('[DIAG-PATCH] preinitializedWebGPUDevice missing');
			return;
		}

		// Always-on uncaptured error logger.
		if (!d._uncapturedPatched) {
			d.addEventListener(
					'uncapturederror', function(e) {
				console.error('[Godot] WebGPU uncaptured error: ' + e.error.constructor.name);
				console.error(e.error.message); });
			d._uncapturedPatched = true;
		}

		if (!d._pipelinePatched) {
			var origCreate = d.createRenderPipeline.bind(d);
			var origCreateAsync = d.createRenderPipelineAsync.bind(d);
			var pipeCount = 0;
			d.createRenderPipeline = function(desc) {
				var myId = pipeCount++;
				var label = (desc && desc.label) || '(unlabeled)';
				// Error-scope path (may miss deferred-validation errors).
				d.pushErrorScope('validation');
				var pipeline = origCreate(desc);
				d.popErrorScope().then(function(err) {
					if (err) {
						console.error('[JS-PCREATE-FAIL#' + myId + '] label="' + label + '" | ' + err.message.substring(0, 2000));
					}
				});
				// Parallel async path (authoritative — catches all validation errors).
				origCreateAsync(desc).then(
						function(_p){
								// Success — no-op.
						},
						function(err) {
					var msg = (err && err.message) ? err.message : String(err);
					console.error('[JS-PCREATE-ASYNC-FAIL#' + myId + '] label="' + label + '" | ' + msg.substring(0, 2000)); });
				// Log every ENTER so we can correlate JS IDs with Godot pipe#N.
				console.log('[JS-PCREATE-ENTER#' + myId + '] label="' + label + '"');
				return pipeline;
			}
			d._pipelinePatched = true;
			console.log('[DIAG-PATCH] createRenderPipeline patched on main thread (initialize)');
		}

		if (!d._shaderModPatched) {
			var origCreateMod = d.createShaderModule.bind(d);
			var modCount = 0;
			d.createShaderModule = function(desc) {
				var myId = modCount++;
				var label = (desc && desc.label) || '(unlabeled)';
				d.pushErrorScope('validation');
				var mod = origCreateMod(desc);
				d.popErrorScope().then(function(err) {
					if (err) {
						console.error('[JS-SMCREATE-FAIL#' + myId + '] label="' + label + '" | ' + err.message.substring(0, 2000));
					}
				});
				if (mod && mod.getCompilationInfo) {
					mod.getCompilationInfo().then(function(info) {
						if (!info || !info.messages || info.messages.length == = 0) {
							return;
						}
						for (var i = 0; i < info.messages.length; i++) {
							var m = info.messages[i];
							if (m.type == = 'error') {
								console.error('[JS-SMCOMPILE-ERR#' + myId + '] label="' + label + '" line=' + m.lineNum + ':' + m.linePos + ' | ' + m.message.substring(0, 1200));
							}
						}
					});
				}
				return mod;
			}
			d._shaderModPatched = true;
			console.log('[DIAG-PATCH] createShaderModule patched on main thread (initialize)');
		}

		if (!d._bglPatched) {
			var origBGL = d.createBindGroupLayout.bind(d);
			d.createBindGroupLayout = function(desc) {
				var label = (desc && desc.label) || '(unlabeled)';
				d.pushErrorScope('validation');
				var layout = origBGL(desc);
				d.popErrorScope().then(function(err) {
					if (err) {
						console.error('[JS-BGL-FAIL] label="' + label + '" | ' + err.message.substring(0, 2000));
					}
				});
				return layout;
			}
			d._bglPatched = true;
		}

		if (!d._plytPatched) {
			var origPLYT = d.createPipelineLayout.bind(d);
			d.createPipelineLayout = function(desc) {
				var label = (desc && desc.label) || '(unlabeled)';
				d.pushErrorScope('validation');
				var layout = origPLYT(desc);
				d.popErrorScope().then(function(err) {
					if (err) {
						console.error('[JS-PLYT-FAIL] label="' + label + '" | ' + err.message.substring(0, 2000));
					}
				});
				return layout;
			}
			d._plytPatched = true;
		}
	});
#endif // WEBGPU_VERBOSE

	return OK;
}

void RenderingDeviceDriverWebGPU::_check_capabilities() {
	has_core_features = wgpuDeviceHasFeature(device, WGPUFeatureName_CoreFeaturesAndLimits);
	capabilities.device_family = DEVICE_UNKNOWN;
	capabilities.version_major = 1;
	capabilities.version_minor = 0;

	// Query actual device limits.
	device_limits = WGPU_LIMITS_INIT;
	WGPUStatus status = wgpuDeviceGetLimits(device, &device_limits);
	if (status != WGPUStatus_Success) {
		WARN_PRINT("WebGPU: Failed to query device limits, using spec minimums.");
	}

	// float32-filterable: required for linear sampling of R32Float / RG32Float / RGBA32Float.
	// Forward Mobile's HDR post-processing path samples 32F render targets with linear
	// samplers, so without this feature those samplers must fall back to NEAREST.
	// Query the JS feature name: Dawn's C enum values vary between port versions.
#ifdef WEB_ENABLED
	float32_filterable_supported = EM_ASM_INT({
		return Module['preinitializedWebGPUDevice'].features.has('float32-filterable') ? 1 : 0;
	});
#else
	float32_filterable_supported = wgpuDeviceHasFeature(device, WGPUFeatureName_Float32Filterable);
#endif
	if (float32_filterable_supported) {
		print_verbose("WebGPU: float32-filterable feature is available.");
	} else {
		WARN_PRINT("WebGPU: float32-filterable feature NOT available — 32F linear sampling will fall back to nearest.");
	}

	// float32-blendable: required for blending on R32Float / RG32Float / RGBA32Float
	// render targets. Without this, blend operations on float32 targets silently fail
	// (particles, post-processing compositing).
	// Query the JS feature name for compatibility with both supported port generations.
#ifdef WEB_ENABLED
	float32_blendable_supported = EM_ASM_INT({
		return Module['preinitializedWebGPUDevice'].features.has('float32-blendable') ? 1 : 0;
	});
#else
	float32_blendable_supported = wgpuDeviceHasFeature(device, WGPUFeatureName_Float32Blendable);
#endif
	if (float32_blendable_supported) {
		print_verbose("WebGPU: float32-blendable feature is available.");
	} else {
		WARN_PRINT("WebGPU: float32-blendable feature NOT available — blend on float32 targets will be disabled.");
	}

	// texture-formats-tier1: adds storage binding support for r8unorm, rg8unorm, etc.
	// The emdawnwebgpu 4.0.10 header lacks the WGPUFeatureName enum value for this
	// feature, so query the JS device object directly.
#ifdef WEB_ENABLED
	has_texture_formats_tier1 = (bool)EM_ASM_INT({
		var d = Module['preinitializedWebGPUDevice'];
		return (d && d.features && d.features.has('texture-formats-tier1')) ? 1 : 0;
	});
	has_texture_formats_tier2 = (bool)EM_ASM_INT({
		return Module['preinitializedWebGPUDevice'].features.has('texture-formats-tier2') ? 1 : 0;
	});
#else
	has_texture_formats_tier1 = wgpuDeviceHasFeature(device, WGPUFeatureName_TextureFormatsTier1);
	has_texture_formats_tier2 = wgpuDeviceHasFeature(device, WGPUFeatureName_TextureFormatsTier2);
#endif
	if (has_texture_formats_tier1) {
		print_verbose("WebGPU: texture-formats-tier1 feature is available — r8/rg8 storage formats supported natively.");
	}

	// readonly-and-readwrite-storage-textures: allows read and read_write access
	// modes on storage textures. Without this, only write-only is valid.
#ifdef WEB_ENABLED
	has_rw_storage_textures = (bool)EM_ASM_INT({
		return navigator.gpu.wgslLanguageFeatures.has('readonly_and_readwrite_storage_textures') ? 1 : 0;
	});
#else
	has_rw_storage_textures = wgpuInstanceHasWGSLLanguageFeature(context_driver->get_instance(), WGPUWGSLLanguageFeatureName_ReadonlyAndReadwriteStorageTextures);
#endif
	if (has_rw_storage_textures) {
		print_verbose("WebGPU: readonly-and-readwrite-storage-textures feature is available.");
	} else {
		print_verbose("WebGPU: readonly-and-readwrite-storage-textures NOT available — read/write storage textures are unavailable.");
	}

	// Optional texture-compression families. The JS shell opts in to these when
	// the adapter supports them; we must match that here so Godot only advertises
	// the corresponding DataFormats when they'll actually work.
	has_texture_compression_bc = wgpuDeviceHasFeature(device, WGPUFeatureName_TextureCompressionBC);
	has_texture_compression_etc2 = wgpuDeviceHasFeature(device, WGPUFeatureName_TextureCompressionETC2);
	has_texture_compression_astc = wgpuDeviceHasFeature(device, WGPUFeatureName_TextureCompressionASTC);
	if (has_texture_compression_bc) {
		print_verbose("WebGPU: texture-compression-bc feature is available.");
	}
	if (has_texture_compression_etc2) {
		print_verbose("WebGPU: texture-compression-etc2 feature is available.");
	}
	if (has_texture_compression_astc) {
		print_verbose("WebGPU: texture-compression-astc feature is available.");
	}

	// Multiview not supported in WebGPU.
	multiview_capabilities.is_supported = false;

	// Fragment shading rate not supported.
	fsr_capabilities = {};

	// Fragment density map not supported.
	fdm_capabilities = {};
}

// =============================================================================
// BUFFERS
// =============================================================================

RDD::BufferID RenderingDeviceDriverWebGPU::buffer_create(uint64_t p_size, BitField<BufferUsageBits> p_usage, MemoryAllocationType p_allocation_type, uint64_t p_frames_drawn) {
	WGBuffer *buf = memnew(WGBuffer);

	// WebGPU buffer sizes must be a multiple of 4.
	uint64_t aligned_size = (p_size + 3) & ~3ULL;

	// Task 7.5: BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT means Godot wants to rotate
	// through `frame_count` slices of this buffer. Each slice must be aligned to
	// the device's minUniformBufferOffsetAlignment (typically 256) because the
	// slice offset is passed as a dynamic offset to wgpuRenderPassEncoderSetBindGroup,
	// and dynamic offsets must be multiples of that alignment. Without this
	// alignment pass, the original Phase 7 commit (8d48436801) would fail validation
	// whenever p_size was not already a 256-multiple.
	const bool is_dynamic_persistent = p_usage.has_flag(BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT);
	if (is_dynamic_persistent && frame_count > 1) {
		uint32_t alignment = device_limits.minUniformBufferOffsetAlignment;
		if (alignment == 0 || alignment == WGPU_LIMIT_U32_UNDEFINED) {
			alignment = 256; // WebGPU spec default minimum.
		}
		aligned_size = (aligned_size + alignment - 1) & ~(uint64_t)(alignment - 1);
		buf->per_frame_size = aligned_size;
		aligned_size *= frame_count;
		buf->frame_idx = 0; // Mark as dynamic — first slice is index 0.
	}

	buf->usage = _buffer_usage_to_wgpu(p_usage);
	buf->usage |= WGPUBufferUsage_CopyDst; // Always allow writes.
	buf->size = aligned_size;

	// Don't add MapRead to generic CPU buffers — it conflicts with CopySrc.
	// Compute readback uses buffer_get_data_direct() which creates its own
	// staging buffer with the correct CopyDst|MapRead usage.
	//
	// Task 7.8: Narrow exception — a "CPU + TRANSFER_TO + no TRANSFER_FROM"
	// buffer is a pure GPU→CPU staging buffer (dest-only, never the source of
	// another copy). For these, WebGPU spec permits MapRead (since CopySrc is
	// not present, there's no CopySrc↔MapRead conflict), and the existing
	// buffer_map() async-readback codepath needs MapRead to be valid.
	const bool wants_transfer_to = p_usage.has_flag(BUFFER_USAGE_TRANSFER_TO_BIT);
	const bool wants_transfer_from = p_usage.has_flag(BUFFER_USAGE_TRANSFER_FROM_BIT);
	if (p_allocation_type == MEMORY_ALLOCATION_TYPE_CPU && wants_transfer_to && !wants_transfer_from) {
		buf->is_readback = true;
		buf->usage |= WGPUBufferUsage_MapRead;
		// CopySrc must NOT be set for MapRead to be legal; _buffer_usage_to_wgpu
		// derives CopySrc from TRANSFER_FROM_BIT, which we've just confirmed is off.
	}

	WGPUBufferDescriptor desc = {};
	desc.size = aligned_size;
	desc.usage = buf->usage;
	desc.mappedAtCreation = false;

	buf->handle = wgpuDeviceCreateBuffer(device, &desc);
	if (buf->handle == nullptr) {
		memdelete(buf);
		ERR_FAIL_V(BufferID());
	}

	return BufferID(buf);
}

RDD::BufferID RenderingDeviceDriverWebGPU::buffer_create_with_data(uint64_t p_size, BitField<BufferUsageBits> p_usage, MemoryAllocationType p_allocation_type, const uint8_t *p_data, uint64_t p_data_size) {
	WGBuffer *buf = memnew(WGBuffer);

	uint64_t aligned_size = (p_size + 3) & ~3ULL;

	buf->usage = _buffer_usage_to_wgpu(p_usage);
	buf->usage |= WGPUBufferUsage_CopyDst; // Always allow writes.
	buf->size = aligned_size;

	WGPUBufferDescriptor desc = {};
	desc.size = aligned_size;
	desc.usage = buf->usage;
	desc.mappedAtCreation = true;

	buf->handle = wgpuDeviceCreateBuffer(device, &desc);
	if (buf->handle == nullptr) {
		memdelete(buf);
		ERR_FAIL_V(BufferID());
	}

	// Write initial data directly into the mapped range — no staging buffer,
	// no wgpuQueueWriteBuffer, no command encoder copy needed.
	void *mapped = wgpuBufferGetMappedRange(buf->handle, 0, aligned_size);
	if (mapped) {
		memcpy(mapped, p_data, p_data_size);
		// Zero-fill any padding between data end and aligned buffer size.
		if (aligned_size > p_data_size) {
			memset((uint8_t *)mapped + p_data_size, 0, aligned_size - p_data_size);
		}
	}
	wgpuBufferUnmap(buf->handle);

	return BufferID(buf);
}

bool RenderingDeviceDriverWebGPU::buffer_set_texel_format(BufferID p_buffer, DataFormat p_format) {
	// WebGPU has no texel buffer views. Stub: store format, emulate later if needed.
	return true;
}

void RenderingDeviceDriverWebGPU::buffer_free(BufferID p_buffer) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL(buf);
#ifndef WEB_ENABLED
	if (buf->map_pending) {
		buffer_map(p_buffer); // Complete the native future before releasing its userdata.
	}
#endif

	// Drop any readback-cache entry for this buffer.
	uint64_t key = (uint64_t)(uintptr_t)buf;
	if (_readback_cache.has(key)) {
		ReadbackEntry *entry = _readback_cache[key];
		if (!entry->map_complete) {
			entry->cancelled = true;
		} else {
			if (entry->staging) {
				wgpuBufferRelease(entry->staging);
			}
			if (entry->shadow) {
				memfree(entry->shadow);
			}
			memdelete(entry);
		}
		_readback_cache.erase(key);
	}

	// If an async map callback is in flight for this buffer, mark it freed
	// and let the callback handle cleanup (use-after-free prevention).
	if (buf->map_pending) {
		buf->freed = true;
		return;
	}

	if (buf->handle) {
		wgpuBufferRelease(buf->handle);
	}
	if (buf->shadow_map) {
		memfree(buf->shadow_map);
	}
	memdelete(buf);
}

uint64_t RenderingDeviceDriverWebGPU::buffer_get_allocation_size(BufferID p_buffer) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL_V(buf, 0);
	return buf->size;
}

// Callback for deferred buffer map readback.
// Copies GPU buffer data into the WGBuffer shadow_map when the async map resolves.
static void _buffer_deferred_map_cb(WGPUMapAsyncStatus p_status, WGPUStringView p_message, void *p_userdata1, void *p_userdata2) {
	WGBuffer *buf = (WGBuffer *)p_userdata1;
	if (!buf) {
		return;
	}

	buf->map_pending = false;

	// Buffer was freed while this async map was in flight.
	if (buf->freed) {
		if (buf->handle) {
			if (p_status == WGPUMapAsyncStatus_Success) {
				wgpuBufferUnmap(buf->handle);
			}
			wgpuBufferRelease(buf->handle);
		}
		if (buf->shadow_map) {
			memfree(buf->shadow_map);
		}
		memdelete(buf);
		return;
	}

	if (p_status == WGPUMapAsyncStatus_Success) {
		const void *mapped = wgpuBufferGetConstMappedRange(buf->handle, 0, buf->size);
		if (mapped && buf->shadow_map) {
			memcpy(buf->shadow_map, mapped, buf->size);
		}
		wgpuBufferUnmap(buf->handle);
	} else {
		ERR_PRINT(vformat("WebGPU _buffer_deferred_map_cb: FAILED status=%d", (int)p_status));
	}
	buf->map_complete = true;
}

uint8_t *RenderingDeviceDriverWebGPU::buffer_map(BufferID p_buffer) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL_V(buf, nullptr);

	// For readback buffers (staging buffers used by buffer_get_data), perform
	// async map to get actual GPU data.  This is critical for compute shader
	// readback, screenshot capture, and any CPU-side read of GPU results.
	//
	// The pattern: wgpuBufferMapAsync with AllowSpontaneous callback mode,
	// then wgpuQueueSubmit (empty) to flush, followed by polling via
	// wgpuBufferGetMapState until the map completes.
	if (buf->is_readback && buf->handle) {
#ifndef WEB_ENABLED
		if (!buf->map_pending && !buf->map_complete) {
			buffer_initiate_async_map(p_buffer);
		}
		if (buf->map_pending) {
			WGPUFutureWaitInfo wait = { buf->map_future, false };
			ERR_FAIL_COND_V_MSG(wgpuInstanceWaitAny(context_driver->get_instance(), 1, &wait, 30000000000ULL) != WGPUWaitStatus_Success, nullptr, "WebGPU: buffer readback timed out.");
		}
#endif
		// WebGPU async readback: the buffer may have been mapped by
		// buffer_initiate_async_map() (called after GPU submit). We try
		// ProcessEvents to deliver any pending callbacks, and also check the
		// map state directly — if the buffer is Mapped, we can read it now
		// even if the C callback hasn't fired yet.
		if (!buf->shadow_map) {
			buf->shadow_map = (uint8_t *)memalloc(buf->size);
			memset(buf->shadow_map, 0, buf->size);
		}

		// Try to deliver pending callbacks.
		WGPUInstance inst = context_driver ? context_driver->get_instance() : nullptr;
		if (inst) {
			wgpuInstanceProcessEvents(inst);
		}

		// If the callback already fired, shadow has data.
		if (buf->map_complete) {
			buf->map_complete = false;
			return buf->shadow_map;
		}

		// Callback didn't fire, but check if the buffer is mapped anyway.
		// emdawnwebgpu may have completed the map at the JS level even
		// though the C callback hasn't been delivered yet.
		WGPUBufferMapState state = wgpuBufferGetMapState(buf->handle);
		if (state == WGPUBufferMapState_Mapped) {
			const void *mapped = wgpuBufferGetConstMappedRange(buf->handle, 0, buf->size);
			if (mapped) {
				memcpy(buf->shadow_map, mapped, buf->size);
				// Buffer was mapped at JS level before C callback fired.
			}
			wgpuBufferUnmap(buf->handle);
			return buf->shadow_map;
		}

		return buf->shadow_map;
	}

	// For non-readback buffers, use the shadow CPU buffer (for upload staging).
	// Mark dirty so buffer_unmap() will flush to GPU.
	if (!buf->shadow_map) {
		buf->shadow_map = (uint8_t *)memalloc(buf->size);
		memset(buf->shadow_map, 0, buf->size);
	}
	buf->map_dirty = true;
	return buf->shadow_map;
}

void RenderingDeviceDriverWebGPU::buffer_unmap(BufferID p_buffer) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL(buf);
	if (buf->shadow_map && buf->map_dirty) {
		// Flush only the dirty range if one was set, otherwise fall back to the
		// full buffer. command_copy_buffer_to_texture and command_copy_buffer
		// clear map_dirty after handling the transfer themselves, so in practice
		// this path is only hit for persistent dynamic buffers or unknown callers.
		uint64_t flush_offset = 0;
		uint64_t flush_size = buf->size;
		if (buf->dirty_end > buf->dirty_offset) {
			flush_offset = buf->dirty_offset;
			flush_size = buf->dirty_end - buf->dirty_offset;
		}
		wgpuQueueWriteBuffer(queue, buf->handle, flush_offset, buf->shadow_map + flush_offset, flush_size);
		buf->map_dirty = false;
		buf->dirty_offset = 0;
		buf->dirty_end = 0;
	}
}

uint8_t *RenderingDeviceDriverWebGPU::buffer_persistent_map_advance(BufferID p_buffer, uint64_t p_frames_drawn) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL_V(buf, nullptr);
	if (!buf->shadow_map) {
		buf->shadow_map = (uint8_t *)memalloc(buf->size);
		memset(buf->shadow_map, 0, buf->size);
	}
	// Task 7.5: For dynamic persistent buffers, rotate to the next slice and
	// return a pointer into that slice. Godot writes one frame's worth of data
	// here, and the GPU reads from `frame_idx * per_frame_size` via dynamic offset.
	if (buf->is_dynamic() && buf->per_frame_size > 0 && frame_count > 1) {
		buf->frame_idx = (buf->frame_idx + 1) % frame_count;
		uint64_t slice_offset = (uint64_t)buf->frame_idx * buf->per_frame_size;
		// Set dirty range to just this frame's slice so buffer_flush() only
		// writes per_frame_size bytes instead of the entire multi-frame buffer.
		buf->dirty_offset = slice_offset;
		buf->dirty_end = slice_offset + buf->per_frame_size;
		return buf->shadow_map + slice_offset;
	}
	return buf->shadow_map;
}

uint64_t RenderingDeviceDriverWebGPU::buffer_get_dynamic_offsets(Span<BufferID> p_buffers) {
	// Task 7.5: Standalone path — packs frame_idx for each dynamic buffer into
	// 2-bit shifted slots (matching the Vulkan pattern). Consumed by the render
	// graph for buffers passed to command_render_bind_vertex_buffers (which
	// currently doesn't honor dynamic offsets on WebGPU; harmless to return here).
	uint64_t mask = 0;
	uint64_t shift = 0;
	for (const BufferID &bid : p_buffers) {
		const WGBuffer *buf = (const WGBuffer *)bid.id;
		if (!buf || !buf->is_dynamic()) {
			continue;
		}
		mask |= (uint64_t)buf->frame_idx << shift;
		shift += 2; // Matches Vulkan: frame_count never exceeds 4.
	}
	return mask;
}

void RenderingDeviceDriverWebGPU::buffer_flush(BufferID p_buffer) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	if (buf && buf->shadow_map) {
		// Flush only the dirty range if one was set (e.g., by
		// buffer_persistent_map_advance), otherwise fall back to full buffer.
		uint64_t flush_offset = 0;
		uint64_t flush_size = buf->size;
		if (buf->dirty_end > buf->dirty_offset) {
			flush_offset = buf->dirty_offset;
			flush_size = buf->dirty_end - buf->dirty_offset;
		}
		wgpuQueueWriteBuffer(queue, buf->handle, flush_offset, buf->shadow_map + flush_offset, flush_size);
		buf->dirty_offset = 0;
		buf->dirty_end = 0;
	}
}

void RenderingDeviceDriverWebGPU::buffer_write_direct(BufferID p_buffer, uint64_t p_offset, uint64_t p_size, const void *p_data) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL(buf);
	if ((p_size & 3) == 0) {
		wgpuQueueWriteBuffer(queue, buf->handle, p_offset, p_data, p_size);
		return;
	}
	// WebGPU writes are 4-byte multiples; pad a copy instead of reading past the caller's data.
	LocalVector<uint8_t> padded;
	padded.resize((p_size + 3) & ~3ULL);
	memset(padded.ptr() + p_size, 0, padded.size() - p_size);
	memcpy(padded.ptr(), p_data, p_size);
	wgpuQueueWriteBuffer(queue, buf->handle, p_offset, padded.ptr(), padded.size());
}

uint64_t RenderingDeviceDriverWebGPU::buffer_get_device_address(BufferID p_buffer) {
	return 0; // No device addresses in WebGPU.
}

void RenderingDeviceDriverWebGPU::buffer_initiate_async_map(BufferID p_buffer) {
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	if (!buf || !buf->is_readback || !buf->handle) {
		WARN_PRINT("WebGPU buffer_initiate_async_map: skipped (null/not-readback)");
		return;
	}

	// Consume any completed previous map so shadow_map has its data,
	// then start a fresh map for the data the GPU just wrote.
	if (buf->map_complete) {
		buf->map_complete = false;
	}

	// Only initiate if the buffer isn't already pending a map.
	WGPUBufferMapState state = wgpuBufferGetMapState(buf->handle);
	if (state == WGPUBufferMapState_Unmapped) {
		if (!buf->shadow_map) {
			buf->shadow_map = (uint8_t *)memalloc(buf->size);
			memset(buf->shadow_map, 0, buf->size);
		}
		buf->map_pending = true;
		WGPUBufferMapCallbackInfo cb = {};
		cb.mode = WEBGPU_CALLBACK_MODE;
		cb.callback = _buffer_deferred_map_cb;
		cb.userdata1 = buf;
#ifndef WEB_ENABLED
		cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
		buf->map_future = wgpuBufferMapAsync(buf->handle, WGPUMapMode_Read, 0, buf->size, cb);
	} else {
		// Buffer already pending or mapped, skip duplicate mapAsync.
	}
}

uint32_t RenderingDeviceDriverWebGPU::texture_get_gpu_pixel_size(TextureID p_texture) {
	WGTexture *tex = (WGTexture *)(p_texture.id);
	if (!tex || tex->rd_format == DATA_FORMAT_MAX) {
		return 0;
	}
	// Non-zero whenever the GPU format was promoted or downgraded, even at equal size
	// (e.g. RGBA16Unorm stored as RGBA16Float), so RD routes the data through the converters.
	return (tex->format != data_format_to_wgpu(tex->rd_format)) ? wgpu_format_pixel_size(tex->format) : 0;
}

// Converts rows of texels between two plain color formats with the same channel count
// (the promotions/downgrades done by texture_create). Unknown pairs are copied as-is.
static void _convert_texels(WGPUTextureFormat p_src_format, WGPUTextureFormat p_dst_format,
		const uint8_t *p_src, uint32_t p_src_pitch, uint8_t *p_dst, uint32_t p_dst_pitch,
		uint32_t p_width, uint32_t p_height) {
	uint32_t src_channels = 0, dst_channels = 0;
	WGTexelType src_type = WG_TEXEL_FLOAT32, dst_type = WG_TEXEL_FLOAT32;
	if (!wgpu_format_texel_info(p_src_format, src_channels, src_type) ||
			!wgpu_format_texel_info(p_dst_format, dst_channels, dst_type) || src_channels != dst_channels) {
		const uint32_t row_bytes = p_width * MIN(wgpu_format_pixel_size(p_src_format), wgpu_format_pixel_size(p_dst_format));
		for (uint32_t y = 0; y < p_height; y++) {
			memcpy(p_dst + y * p_dst_pitch, p_src + y * p_src_pitch, row_bytes);
		}
		return;
	}
	const uint32_t src_size = wg_texel_component_size(src_type);
	const uint32_t dst_size = wg_texel_component_size(dst_type);
	const uint32_t components = p_width * src_channels;
	for (uint32_t y = 0; y < p_height; y++) {
		const uint8_t *src_row = p_src + y * p_src_pitch;
		uint8_t *dst_row = p_dst + y * p_dst_pitch;
		for (uint32_t c = 0; c < components; c++) {
			wg_texel_encode(dst_type, wg_texel_decode(src_type, src_row + c * src_size), dst_row + c * dst_size);
		}
	}
}

void RenderingDeviceDriverWebGPU::texture_readback_convert(TextureID p_texture,
		const uint8_t *p_src, uint32_t p_src_pitch,
		uint8_t *p_dst, uint32_t p_dst_pitch,
		uint32_t p_width, uint32_t p_height) {
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL(tex);
	_convert_texels(tex->format, data_format_to_wgpu(tex->rd_format), p_src, p_src_pitch, p_dst, p_dst_pitch, p_width, p_height);
}

void RenderingDeviceDriverWebGPU::texture_upload_convert(TextureID p_texture,
		const uint8_t *p_src, uint32_t p_src_pitch,
		uint8_t *p_dst, uint32_t p_dst_pitch,
		uint32_t p_width, uint32_t p_height) {
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL(tex);
	_convert_texels(data_format_to_wgpu(tex->rd_format), tex->format, p_src, p_src_pitch, p_dst, p_dst_pitch, p_width, p_height);
}

// =============================================================================
// ASYNC BUFFER READBACK (WebGPU-specific)
// =============================================================================

void RenderingDeviceDriverWebGPU::_readback_map_cb(WGPUMapAsyncStatus p_status, WGPUStringView p_message, void *p_userdata1, void *p_userdata2) {
	ReadbackEntry *entry = (ReadbackEntry *)p_userdata1;
	if (!entry) {
		return;
	}

	// Source was freed while the async map was in flight. Clean up the
	// orphaned entry that was kept alive specifically for this callback.
	if (entry->cancelled) {
		if (entry->staging) {
			if (p_status == WGPUMapAsyncStatus_Success) {
				wgpuBufferUnmap(entry->staging);
			}
			wgpuBufferRelease(entry->staging);
		}
		if (entry->shadow) {
			memfree(entry->shadow);
		}
		memdelete(entry);
		return;
	}

	if (!entry->staging) {
		return;
	}

	if (p_status == WGPUMapAsyncStatus_Success) {
		const void *mapped = wgpuBufferGetConstMappedRange(entry->staging, 0, entry->size);
		if (mapped && entry->shadow) {
			memcpy(entry->shadow, mapped, entry->size);
			entry->has_data = true;
		}
		wgpuBufferUnmap(entry->staging);
	}
	entry->map_pending = false;
	entry->map_complete = true;
}

bool RenderingDeviceDriverWebGPU::buffer_get_data_direct(BufferID p_buffer, uint64_t p_offset, uint64_t p_size, Vector<uint8_t> &r_data) {
#ifndef WEB_ENABLED
	return false; // Native callers can use RD's synchronous staging-buffer path.
#else
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL_V(buf, false);

	uint64_t key = (uint64_t)(uintptr_t)buf;
	ReadbackEntry *entry = nullptr;

	if (_readback_cache.has(key)) {
		entry = _readback_cache[key];
	}

	// First call for this buffer: create its persistent staging buffer.
	if (!entry) {
		entry = memnew(ReadbackEntry);
		entry->size = buf->size;
		entry->shadow = (uint8_t *)memalloc(buf->size);
		memset(entry->shadow, 0, buf->size);

		// Create persistent staging buffer with CopyDst + MapRead.
		WGPUBufferDescriptor desc = {};
		desc.size = (buf->size + 3) & ~3ULL;
		desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
		entry->staging = wgpuDeviceCreateBuffer(device, &desc);
		if (!entry->staging) {
			memfree(entry->shadow);
			memdelete(entry);
			ERR_FAIL_V_MSG(false, "WebGPU: buffer_get_data_direct: failed to create staging buffer.");
		}
		entry->map_complete = false;
		entry->has_data = false;

		_readback_cache[key] = entry;
	}

	// Return the latest completed readback, or empty data while none has landed yet.
	// Always report the request as handled: RD's fallback would stall and read zeros.
	if (entry->has_data) {
		r_data.resize(p_size);
		memcpy(r_data.ptrw(), entry->shadow + p_offset, p_size);
	} else {
		r_data.clear();
	}

	// Re-arm after every finished map, including failed ones, so later calls see fresh data.
	if (!entry->map_pending) {
		entry->map_pending = true;
		entry->map_complete = false;
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
		wgpuCommandEncoderCopyBufferToBuffer(encoder, buf->handle, 0, entry->staging, 0, entry->size);
		WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
		wgpuQueueSubmit(queue, 1, &cmd);
		wgpuCommandBufferRelease(cmd);
		wgpuCommandEncoderRelease(encoder);

		WGPUBufferMapCallbackInfo cb = {};
		cb.mode = WEBGPU_CALLBACK_MODE;
		cb.callback = _readback_map_cb;
		cb.userdata1 = entry;
		wgpuBufferMapAsync(entry->staging, WGPUMapMode_Read, 0, entry->size, cb);
	}
	return true;
#endif
}

WGPUBufferUsage RenderingDeviceDriverWebGPU::_buffer_usage_to_wgpu(BitField<BufferUsageBits> p_usage) const {
	WGPUBufferUsage flags = 0;
	if (p_usage.has_flag(BUFFER_USAGE_TRANSFER_FROM_BIT)) {
		flags |= WGPUBufferUsage_CopySrc;
	}
	if (p_usage.has_flag(BUFFER_USAGE_TRANSFER_TO_BIT)) {
		flags |= WGPUBufferUsage_CopyDst;
	}
	if (p_usage.has_flag(BUFFER_USAGE_UNIFORM_BIT)) {
		flags |= WGPUBufferUsage_Uniform;
	}
	if (p_usage.has_flag(BUFFER_USAGE_STORAGE_BIT) || p_usage.has_flag(BUFFER_USAGE_TEXEL_BIT)) {
		flags |= WGPUBufferUsage_Storage;
	}
	if (p_usage.has_flag(BUFFER_USAGE_INDEX_BIT)) {
		flags |= WGPUBufferUsage_Index;
	}
	if (p_usage.has_flag(BUFFER_USAGE_VERTEX_BIT)) {
		flags |= WGPUBufferUsage_Vertex;
	}
	if (p_usage.has_flag(BUFFER_USAGE_INDIRECT_BIT)) {
		flags |= WGPUBufferUsage_Indirect;
	}
	return flags;
}

// =============================================================================
// TEXTURES
// =============================================================================

static bool _is_norm16_format(WGPUTextureFormat p_format) {
	switch (p_format) {
		case WGPUTextureFormat_R16Unorm:
		case WGPUTextureFormat_R16Snorm:
		case WGPUTextureFormat_RG16Unorm:
		case WGPUTextureFormat_RG16Snorm:
		case WGPUTextureFormat_RGBA16Unorm:
		case WGPUTextureFormat_RGBA16Snorm:
			return true;
		default:
			return false;
	}
}

// Returns the sRGB counterpart format for view compatibility, or Undefined if none.
static WGPUTextureFormat _get_srgb_view_format(WGPUTextureFormat p_format) {
	switch (p_format) {
		case WGPUTextureFormat_RGBA8Unorm:
			return WGPUTextureFormat_RGBA8UnormSrgb;
		case WGPUTextureFormat_RGBA8UnormSrgb:
			return WGPUTextureFormat_RGBA8Unorm;
		case WGPUTextureFormat_BGRA8Unorm:
			return WGPUTextureFormat_BGRA8UnormSrgb;
		case WGPUTextureFormat_BGRA8UnormSrgb:
			return WGPUTextureFormat_BGRA8Unorm;
		case WGPUTextureFormat_BC1RGBAUnorm:
			return WGPUTextureFormat_BC1RGBAUnormSrgb;
		case WGPUTextureFormat_BC1RGBAUnormSrgb:
			return WGPUTextureFormat_BC1RGBAUnorm;
		case WGPUTextureFormat_BC2RGBAUnorm:
			return WGPUTextureFormat_BC2RGBAUnormSrgb;
		case WGPUTextureFormat_BC2RGBAUnormSrgb:
			return WGPUTextureFormat_BC2RGBAUnorm;
		case WGPUTextureFormat_BC3RGBAUnorm:
			return WGPUTextureFormat_BC3RGBAUnormSrgb;
		case WGPUTextureFormat_BC3RGBAUnormSrgb:
			return WGPUTextureFormat_BC3RGBAUnorm;
		case WGPUTextureFormat_ETC2RGB8Unorm:
			return WGPUTextureFormat_ETC2RGB8UnormSrgb;
		case WGPUTextureFormat_ETC2RGB8UnormSrgb:
			return WGPUTextureFormat_ETC2RGB8Unorm;
		case WGPUTextureFormat_ETC2RGB8A1Unorm:
			return WGPUTextureFormat_ETC2RGB8A1UnormSrgb;
		case WGPUTextureFormat_ETC2RGB8A1UnormSrgb:
			return WGPUTextureFormat_ETC2RGB8A1Unorm;
		case WGPUTextureFormat_ETC2RGBA8Unorm:
			return WGPUTextureFormat_ETC2RGBA8UnormSrgb;
		case WGPUTextureFormat_ETC2RGBA8UnormSrgb:
			return WGPUTextureFormat_ETC2RGBA8Unorm;
		default:
			return WGPUTextureFormat_Undefined;
	}
}

RDD::TextureID RenderingDeviceDriverWebGPU::texture_create(const TextureFormat &p_format, const TextureView &p_view) {
	ERR_FAIL_COND_V_MSG(!texture_is_format_supported_for_samples(p_format.format, p_format.samples), TextureID(), "WebGPU: texture format does not support the requested sample count.");
	if (p_format.samples != TEXTURE_SAMPLES_1) {
		ERR_FAIL_COND_V_MSG(p_format.texture_type != TEXTURE_TYPE_2D || p_format.array_layers != 1 || p_format.mipmaps != 1,
				TextureID(), "WebGPU: multisampled textures must be 2D with one layer and one mip level.");
		const WGPUTextureUsage usage = _texture_usage_to_wgpu(p_format.usage_bits);
		ERR_FAIL_COND_V_MSG(!(usage & WGPUTextureUsage_RenderAttachment) || (usage & WGPUTextureUsage_StorageBinding),
				TextureID(), "WebGPU: multisampled textures require attachment usage and cannot use storage binding.");
	}
	WGTexture *tex = memnew(WGTexture);

	tex->format = _data_format_to_wgpu(p_format.format);
	tex->rd_format = p_format.format;
	tex->dimension = _texture_type_to_dimension(p_format.texture_type);
	tex->view_dimension = _texture_type_to_view_dimension(p_format.texture_type);
	tex->width = p_format.width;
	tex->height = p_format.height;
	tex->depth = p_format.depth;
	tex->mipmaps = p_format.mipmaps;
	tex->layers = p_format.array_layers;
	tex->sample_count = 1 << p_format.samples;
	tex->usage = _texture_usage_to_wgpu(p_format.usage_bits);
	// Format aliases can use RD's copy fallback even when callers do not request readback.
	if (tex->sample_count == 1 && p_format.shareable_formats.size() > 1) {
		tex->usage |= WGPUTextureUsage_CopySrc;
	}

	// WebGPU does not support R8/RG8/R16/RG16 as storage texel formats.
	// Upgrade to 32-bit equivalents when storage binding is needed.
	if (tex->usage & WGPUTextureUsage_StorageBinding) {
		tex->format = _promote_storage_format(tex->format);
		// Read-only storage is bound as a sampled texture; see _postprocess_wgsl().
		tex->usage |= WGPUTextureUsage_TextureBinding;
	}

	// When float32-filterable is not supported (e.g. Adreno GPU), float32
	// textures cannot use linear filtering and get substituted with blank
	// fallbacks at bind time, losing all data.  Downgrade to float16 which
	// is universally filterable and preserves the data with sufficient
	// precision.  Only target textures that are not storage/render-attachment/
	// multisampled (those need exact format matching with compute shaders).
	if (!float32_filterable_supported &&
			!(tex->usage & WGPUTextureUsage_StorageBinding) &&
			!(tex->usage & WGPUTextureUsage_RenderAttachment) &&
			tex->sample_count == 1) {
		if (tex->format == WGPUTextureFormat_RGBA32Float) {
			tex->format = WGPUTextureFormat_RGBA16Float;
			WEBGPU_DIAG({ console.log('[F32-DOWNGRADE] RGBA32Float→RGBA16Float size=' + $0 + 'x' + $1 + ' usage=0x' + ($2).toString(16)); }, (int)tex->width, (int)tex->height, (int)tex->usage);
		} else if (tex->format == WGPUTextureFormat_RG32Float) {
			tex->format = WGPUTextureFormat_RG16Float;
			WEBGPU_DIAG({ console.log('[F32-DOWNGRADE] RG32Float→RG16Float size=' + $0 + 'x' + $1 + ' usage=0x' + ($2).toString(16)); }, (int)tex->width, (int)tex->height, (int)tex->usage);
		} else if (tex->format == WGPUTextureFormat_R32Float) {
			tex->format = WGPUTextureFormat_R16Float;
			WEBGPU_DIAG({ console.log('[F32-DOWNGRADE] R32Float→R16Float size=' + $0 + 'x' + $1 + ' usage=0x' + ($2).toString(16)); }, (int)tex->width, (int)tex->height, (int)tex->usage);
		}
	}

	// WebGPU has no texture component swizzle (unlike Vulkan's VkComponentSwizzle).
	// Textures with non-identity swizzles (e.g., R8 with R→all broadcast) must be
	// handled at a higher level. The driver stores the swizzle for potential future
	// use but does NOT promote the format (Godot's upload data wouldn't match).
	// See: servers/rendering/renderer_rd/storage_rd/texture_storage.cpp which
	// handles swizzle emulation for backends that don't support it natively.

	WGPUTextureDescriptor desc = {};
	desc.dimension = tex->dimension;
	desc.format = tex->format;
	desc.size.width = tex->width;
	desc.size.height = tex->height;
	desc.size.depthOrArrayLayers = (tex->dimension == WGPUTextureDimension_3D) ? tex->depth : tex->layers;
	desc.mipLevelCount = tex->mipmaps;
	desc.sampleCount = tex->sample_count;
	desc.usage = tex->usage;

	// WebGPU views may only change between sRGB and linear variants, and only of formats
	// listed in viewFormats; sRGB viewFormats are incompatible with StorageBinding.
	// texture_can_make_shared_with_format sends every other shareable format to RD's fallback.
	WGPUTextureFormat srgb_compat = _get_srgb_view_format(tex->format);
	if (srgb_compat != WGPUTextureFormat_Undefined && !(tex->usage & WGPUTextureUsage_StorageBinding)) {
		for (DataFormat shareable : p_format.shareable_formats) {
			if (_data_format_to_wgpu(shareable) == srgb_compat) {
				tex->view_format_alt = srgb_compat;
				desc.viewFormatCount = 1;
				desc.viewFormats = &tex->view_format_alt;
				break;
			}
		}
	}

	// Label at creation: Dawn's error messages use the descriptor label, not later SetLabel calls.
	char _tex_fallback[128];
	snprintf(_tex_fallback, sizeof(_tex_fallback), "unnamed %ux%ux%u mip%u fmt%d usage0x%x",
			tex->width, tex->height, tex->layers, tex->mipmaps, (int)tex->format, (unsigned)tex->usage);
	desc.label = { _tex_fallback, WGPU_STRLEN };

	tex->handle = wgpuDeviceCreateTexture(device, &desc);
	if (tex->handle == nullptr) {
		memdelete(tex);
		ERR_FAIL_V(TextureID());
	}
	tex->view_source = tex->handle; // Always the owning WGPUTexture; inherited by shared/sliced textures.

	// Create default view.
	WGPUTextureViewDescriptor view_desc = {};
	view_desc.format = tex->format;
	view_desc.dimension = tex->view_dimension;
	view_desc.baseMipLevel = 0;
	view_desc.mipLevelCount = tex->mipmaps;
	view_desc.baseArrayLayer = 0;
	view_desc.arrayLayerCount = tex->layers;
	view_desc.aspect = WGPUTextureAspect_All;

	tex->default_view = wgpuTextureCreateView(tex->handle, &view_desc);
	if (tex->default_view == nullptr) {
		wgpuTextureRelease(tex->handle);
		memdelete(tex);
		ERR_FAIL_V_MSG(TextureID(), "WebGPU: wgpuTextureCreateView failed for default view.");
	}

	return TextureID(tex);
}

RDD::TextureID RenderingDeviceDriverWebGPU::texture_create_from_extension(uint64_t p_native_texture, TextureType p_type, DataFormat p_format, uint32_t p_array_layers, bool p_depth_stencil, uint32_t p_mipmaps) {
	// Not supported on web platform.
	ERR_FAIL_V_MSG(TextureID(), "WebGPU: texture_create_from_extension not supported.");
}

RDD::TextureID RenderingDeviceDriverWebGPU::texture_create_shared(TextureID p_original_texture, const TextureView &p_view) {
	WGTexture *orig = (WGTexture *)(p_original_texture.id);
	ERR_FAIL_NULL_V(orig, TextureID());

	WGTexture *tex = memnew(WGTexture);
	*tex = *orig; // Copy base properties.

	// Create a new view with potentially different format. The texture's own format
	// keeps its GPU format, which may be promoted (e.g. R8 storage stored as R32Float).
	WGPUTextureViewDescriptor view_desc = {};
	if (p_view.format != DATA_FORMAT_MAX && p_view.format != orig->rd_format) {
		view_desc.format = _data_format_to_wgpu(p_view.format);
		tex->format = view_desc.format;
		tex->rd_format = p_view.format;
	} else {
		view_desc.format = orig->format;
	}
	view_desc.dimension = tex->view_dimension;
	view_desc.baseMipLevel = 0;
	view_desc.mipLevelCount = tex->mipmaps;
	view_desc.baseArrayLayer = 0;
	view_desc.arrayLayerCount = tex->layers;
	view_desc.aspect = WGPUTextureAspect_All;

	// view_source was already inherited from orig via *tex = *orig.
	if (tex->view_source == nullptr) {
		memdelete(tex);
		ERR_FAIL_V_MSG(TextureID(), "WebGPU: texture_create_shared: original texture has no GPU handle (view_source is null).");
	}
	tex->default_view = wgpuTextureCreateView(tex->view_source, &view_desc);
	if (tex->default_view == nullptr) {
		memdelete(tex);
		ERR_FAIL_V_MSG(TextureID(), "WebGPU: wgpuTextureCreateView failed for shared texture view.");
	}
	tex->handle = nullptr; // Shared texture does not own the WGPUTexture.

	return TextureID(tex);
}

RDD::TextureID RenderingDeviceDriverWebGPU::texture_create_shared_from_slice(TextureID p_original_texture, const TextureView &p_view, TextureSliceType p_slice_type, uint32_t p_layer, uint32_t p_layers, uint32_t p_mipmap, uint32_t p_mipmaps) {
	WGTexture *orig = (WGTexture *)(p_original_texture.id);
	ERR_FAIL_NULL_V(orig, TextureID());

	WGTexture *tex = memnew(WGTexture);
	*tex = *orig;

	WGPUTextureViewDescriptor view_desc = {};
	view_desc.format = orig->format;
	if (p_view.format != DATA_FORMAT_MAX && p_view.format != orig->rd_format) {
		view_desc.format = _data_format_to_wgpu(p_view.format);
		tex->format = view_desc.format;
		tex->rd_format = p_view.format;
	}
	view_desc.baseMipLevel = p_mipmap;
	view_desc.mipLevelCount = p_mipmaps;
	view_desc.baseArrayLayer = p_layer;
	view_desc.arrayLayerCount = p_layers;
	view_desc.aspect = WGPUTextureAspect_All;

	switch (p_slice_type) {
		case TEXTURE_SLICE_2D:
			view_desc.dimension = WGPUTextureViewDimension_2D;
			break;
		case TEXTURE_SLICE_CUBEMAP:
			view_desc.dimension = WGPUTextureViewDimension_Cube;
			break;
		case TEXTURE_SLICE_3D:
			view_desc.dimension = WGPUTextureViewDimension_3D;
			break;
		case TEXTURE_SLICE_2D_ARRAY:
			view_desc.dimension = WGPUTextureViewDimension_2DArray;
			break;
		default:
			view_desc.dimension = orig->view_dimension;
			break;
	}

	// view_source was already inherited from orig via *tex = *orig.
	if (tex->view_source == nullptr) {
		memdelete(tex);
		ERR_FAIL_V_MSG(TextureID(), "WebGPU: texture_create_shared_from_slice: original texture has no GPU handle (view_source is null).");
	}

	tex->default_view = wgpuTextureCreateView(tex->view_source, &view_desc);
	if (tex->default_view == nullptr) {
		memdelete(tex);
		ERR_FAIL_V_MSG(TextureID(), "WebGPU: wgpuTextureCreateView failed for sliced texture view.");
	}
	tex->handle = nullptr;
	tex->layers = p_layers;
	tex->mipmaps = p_mipmaps;

	// Update slice-specific metadata so downstream code (uniform set creation,
	// dimension fixups, allocation queries, subresource tracking) sees the
	// correct subresource instead of the parent's full extent.
	//
	// Bug this fixes: without this, tex->view_dimension still says 2DArray
	// (inherited from parent), so uniform_set_create's SAMPLER_WITH_TEXTURE
	// fixup thinks the slice doesn't match shader-expected 2D and re-creates
	// a view from view_source with baseMipLevel=0/baseArrayLayer=0 — i.e.
	// ignoring the slice and sampling the parent's mip 0 layer 0.
	tex->view_dimension = view_desc.dimension;

	// Track the slice's base offset into the parent, so future dimension
	// fixups can create correct subresource views rather than resetting to 0.
	tex->base_mipmap = p_mipmap;
	tex->base_layer = p_layer;

	// Update width/height to the mip level's extent so allocation-size and
	// similar queries return values for the actual subresource.
	for (uint32_t m = 0; m < p_mipmap; m++) {
		tex->width = MAX(1u, tex->width >> 1);
		tex->height = MAX(1u, tex->height >> 1);
		if (tex->depth > 1) {
			tex->depth = MAX(1u, tex->depth >> 1);
		}
	}

	return TextureID(tex);
}

void RenderingDeviceDriverWebGPU::texture_free(TextureID p_texture) {
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL(tex);

	// Drop any readback-cache entries keyed on this texture before its
	// pointer can be recycled by a future allocation. The cache key is
	// (uintptr_t)tex ^ (layer << 48) — see texture_get_data — so we mask
	// the high 16 bits when comparing to match across all layers.
	// If an async map is still in flight (!map_complete), we can't free the
	// entry here — the callback holds a pointer to it. Instead, mark it
	// cancelled; the callback will release the staging buffer/shadow and
	// delete the entry when it fires.
	{
		const uint64_t LAYER_MASK = 0xFFFF000000000000ULL;
		const uint64_t base = (uint64_t)(uintptr_t)tex;
		LocalVector<uint64_t> to_remove;
		for (KeyValue<uint64_t, ReadbackEntry *> &kv : _readback_cache) {
			if ((kv.key & ~LAYER_MASK) == (base & ~LAYER_MASK)) {
				ReadbackEntry *entry = kv.value;
				if (!entry->map_complete) {
					// Async map still in flight — let the callback clean up.
					entry->cancelled = true;
				} else {
					if (entry->staging) {
						wgpuBufferRelease(entry->staging);
					}
					if (entry->shadow) {
						memfree(entry->shadow);
					}
					memdelete(entry);
				}
				to_remove.push_back(kv.key);
			}
		}
		for (uint64_t k : to_remove) {
			_readback_cache.erase(k);
		}
	}

	if (tex->default_view) {
		wgpuTextureViewRelease(tex->default_view);
	}
	if (tex->handle) {
		wgpuTextureRelease(tex->handle);
	}
	memdelete(tex);
}

uint64_t RenderingDeviceDriverWebGPU::texture_get_allocation_size(TextureID p_texture) {
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL_V(tex, 0);
	// Use Godot's format-aware helper (handles compressed block sizes, mipmaps,
	// and per-format bpp). Fall back to a conservative bpp=4 estimate when the
	// rd_format wasn't stored (shared/sliced views, etc.).
	if (tex->rd_format != DATA_FORMAT_MAX) {
		uint64_t layer_size = get_image_format_required_size(tex->rd_format, tex->width, tex->height, tex->depth, tex->mipmaps);
		return layer_size * tex->layers;
	}
	return (uint64_t)tex->width * tex->height * tex->depth * tex->layers * 4;
}

void RenderingDeviceDriverWebGPU::texture_get_copyable_layout(TextureID p_texture, const TextureSubresource &p_subresource, TextureCopyableLayout *r_layout) {
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL(tex);
	ERR_FAIL_NULL(r_layout);

	uint32_t mip_width = MAX(1u, tex->width >> p_subresource.mipmap);
	uint32_t mip_height = MAX(1u, tex->height >> p_subresource.mipmap);
	uint32_t mip_depth = (tex->dimension == WGPUTextureDimension_3D) ? MAX(1u, tex->depth >> p_subresource.mipmap) : 1;
	uint32_t row_bytes = 0;
	uint32_t rows_for_size = mip_height;

	if (tex->rd_format != DATA_FORMAT_MAX) {
		// Compressed formats (BCn / ETC2 / ASTC) pack their data in fixed-size
		// blocks. Row pitch is (blocks_wide * block_byte_size) and "height" is
		// measured in blocks for the copy layout.
		uint32_t block_w = 1, block_h = 1;
		get_compressed_image_format_block_dimensions(tex->rd_format, block_w, block_h);
		if (block_w > 1 || block_h > 1) {
			uint32_t block_byte_size = get_compressed_image_format_block_byte_size(tex->rd_format);
			uint32_t blocks_wide = (mip_width + block_w - 1) / block_w;
			uint32_t blocks_tall = (mip_height + block_h - 1) / block_h;
			row_bytes = blocks_wide * block_byte_size;
			rows_for_size = blocks_tall;
		} else {
			// Use the actual GPU format's pixel size, not the original rd_format.
			// R8/RG8/R16/RG16 textures with STORAGE usage are promoted to 32-bit
			// equivalents (e.g. R8Unorm → R32Float), so the GPU-side bpp differs.
			row_bytes = mip_width * wgpu_format_pixel_size(tex->format);
		}
	} else {
		row_bytes = mip_width * 4; // Conservative fallback.
	}

	// WebGPU requires 256-byte row alignment for buffer <-> texture copies.
	r_layout->row_pitch = ((row_bytes + 255) / 256) * 256;
	r_layout->size = (uint64_t)r_layout->row_pitch * rows_for_size * mip_depth;
}

Vector<uint8_t> RenderingDeviceDriverWebGPU::texture_get_data(TextureID p_texture, uint32_t p_layer) {
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL_V(tex, Vector<uint8_t>());

	// Same frame-deferred readback pattern as buffer_get_data_direct.
	// First call: copy texture → staging buffer, initiate async map, return zeros.
	// Subsequent calls: return cached data from completed async map.

	uint64_t key = (uint64_t)(uintptr_t)tex ^ ((uint64_t)p_layer << 48);
	ReadbackEntry *entry = nullptr;

	if (_readback_cache.has(key)) {
		entry = _readback_cache[key];
	}

	// Two pixel sizes: the original Godot format (rd_bpp) for the output
	// Vector, and the actual GPU format (gpu_bpp) for staging buffer sizing
	// and the CopyTextureToBuffer stride. These diverge when the driver
	// promotes storage formats (R8→R32Float) or downgrades float32→float16.
	uint32_t rd_bpp = (tex->rd_format != DATA_FORMAT_MAX) ? get_image_format_pixel_size(tex->rd_format) : 4;
	if (rd_bpp == 0) {
		rd_bpp = 4;
	}
	uint32_t gpu_bpp = wgpu_format_pixel_size(tex->format);
	if (gpu_bpp == 0) {
		gpu_bpp = rd_bpp;
	}
	uint32_t mip_w = tex->width;
	uint32_t mip_h = tex->height;
	uint32_t row_pitch = ((mip_w * gpu_bpp + 255) / 256) * 256; // 256-byte aligned
	uint64_t buffer_size = (uint64_t)row_pitch * mip_h;

	// If a readback is in flight, deliver pending callbacks and then probe the
	// buffer's map state directly. emdawnwebgpu may have completed the map at
	// the JS level even though the C callback hasn't been delivered yet (same
	// trick as buffer_map). If mapped, copy data into shadow and mark complete.
	if (entry && !entry->map_complete && entry->map_pending) {
		WGPUInstance inst = context_driver ? context_driver->get_instance() : nullptr;
		if (inst) {
			wgpuInstanceProcessEvents(inst);
		}
		if (!entry->map_complete) {
			WGPUBufferMapState state = wgpuBufferGetMapState(entry->staging);
			if (state == WGPUBufferMapState_Mapped) {
				const void *mapped = wgpuBufferGetConstMappedRange(entry->staging, 0, entry->size);
				if (mapped && entry->shadow) {
					memcpy(entry->shadow, mapped, entry->size);
					entry->has_data = true;
				}
				wgpuBufferUnmap(entry->staging);
				entry->map_pending = false;
				entry->map_complete = true;
			}
		}
	}

	// Return cached data if previous readback completed.
	if (entry && entry->has_data && entry->map_complete) {
		Vector<uint8_t> result;
		result.resize(mip_w * mip_h * rd_bpp);
		uint8_t *dst = result.ptrw();

		if (texture_get_gpu_pixel_size(p_texture) != 0) {
			// Format was promoted or downgraded — convert GPU data back to
			// the original Godot format (e.g. R32Float→R8, Float16→Float32).
			uint32_t dst_pitch = mip_w * rd_bpp;
			texture_readback_convert(p_texture, entry->shadow, row_pitch,
					dst, dst_pitch, mip_w, mip_h);
		} else {
			// No format divergence — un-pad from staging to tightly packed output.
			uint32_t tight_row = mip_w * rd_bpp;
			for (uint32_t y = 0; y < mip_h; y++) {
				memcpy(dst + y * tight_row, entry->shadow + y * row_pitch, tight_row);
			}
		}

		// Mark consumed; do NOT auto-requeue. Each top-level call (e.g.
		// viewport.get_texture().get_image()) should reflect GPU state at call
		// time, not the previous call's snapshot. The next call to
		// texture_get_data falls through to the queue-fresh-copy path below.
		// For continuous readback the caller can simply call again next frame
		// and will get a fresh capture (one-frame extra latency, but always
		// current). Earlier streaming-style auto-requeue here caused
		// scroll-screenshot to show t_(n-1) on subsequent scrolls because the
		// post-Path-A copy snapshotted the *post-scroll* frame, not the next
		// pre-scroll frame.
		entry->has_data = false;

		return result;
	}

	// First call — create staging buffer and initiate readback.
	if (!entry) {
		entry = memnew(ReadbackEntry);
		entry->size = buffer_size;
		entry->shadow = (uint8_t *)memalloc(buffer_size);
		memset(entry->shadow, 0, buffer_size);

		WGPUBufferDescriptor desc = {};
		desc.size = (buffer_size + 3) & ~3ULL;
		desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
		entry->staging = wgpuDeviceCreateBuffer(device, &desc);
		if (!entry->staging) {
			memfree(entry->shadow);
			memdelete(entry);
			ERR_FAIL_V_MSG(Vector<uint8_t>(), "WebGPU: texture_get_data: failed to create staging buffer.");
		}
		entry->map_complete = false;
		entry->has_data = false;

		_readback_cache[key] = entry;
	} else if (!entry->map_complete) {
		// Readback in flight from a previous call. Don't queue another copy /
		// mapAsync — wgpuBufferMapAsync on an already-pending buffer is a
		// validation error. Return empty to signal "not ready" — callers must
		// handle this (null Image) rather than receiving misleading zeros.
		return Vector<uint8_t>();
	}

	// Reaching here means: either a brand-new entry, or an existing entry
	// whose data was just consumed by Path A (has_data=false, map_complete=true).
	// In the latter case we're about to issue a fresh mapAsync, so reset
	// map_complete first — otherwise the next call's in-flight check would
	// see stale state.
	entry->map_complete = false;
	entry->map_pending = true;
	entry->has_data = false;

	// Copy texture to staging buffer.
	WGPUCommandEncoderDescriptor enc_desc = {};
	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &enc_desc);
	WGPUTexelCopyTextureInfo src_info = {};
	src_info.texture = tex->gpu_handle();
	src_info.mipLevel = 0;
	src_info.origin = { 0, 0, p_layer };
	WGPUTexelCopyBufferInfo dst_info = {};
	dst_info.buffer = entry->staging;
	dst_info.layout.offset = 0;
	dst_info.layout.bytesPerRow = row_pitch;
	dst_info.layout.rowsPerImage = mip_h;
	WGPUExtent3D extent = { mip_w, mip_h, 1 };
	wgpuCommandEncoderCopyTextureToBuffer(encoder, &src_info, &dst_info, &extent);
	WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
	wgpuQueueSubmit(queue, 1, &cmd);
	wgpuCommandBufferRelease(cmd);
	wgpuCommandEncoderRelease(encoder);

	WGPUBufferMapCallbackInfo cb = {};
	cb.mode = WEBGPU_CALLBACK_MODE;
	cb.callback = _readback_map_cb;
	cb.userdata1 = entry;
	wgpuBufferMapAsync(entry->staging, WGPUMapMode_Read, 0, entry->size, cb);

	// Return empty on first call — signals "not ready" to the caller.
	// Data will be available on the next call after frame_post_draw.
	return Vector<uint8_t>();
}

bool RenderingDeviceDriverWebGPU::texture_is_format_supported_for_samples(DataFormat p_format, TextureSamples p_samples) {
	if (p_samples != TEXTURE_SAMPLES_1 && p_samples != TEXTURE_SAMPLES_4) {
		return false;
	}
	if (texture_get_usages_supported_by_format(p_format, false) == 0) {
		return false;
	}
	if (p_samples == TEXTURE_SAMPLES_1) {
		return true;
	}
	const WGPUTextureFormat format = _data_format_to_wgpu(p_format);
	return wgpu_format_supports_multisampling(format, has_core_features, has_texture_formats_tier1, format == WGPUTextureFormat_RG11B10Ufloat && wgpuDeviceHasFeature(device, WGPUFeatureName_RG11B10UfloatRenderable));
}

BitField<RDD::TextureUsageBits> RenderingDeviceDriverWebGPU::texture_get_usages_supported_by_format(DataFormat p_format, bool p_cpu_readable) {
	// If there's no WebGPU equivalent, the format is unsupported.
	if (_data_format_to_wgpu(p_format) == WGPUTextureFormat_Undefined) {
		return 0;
	}

	// These bits apply to every supported format.
	BitField<TextureUsageBits> flags = TEXTURE_USAGE_SAMPLING_BIT | TEXTURE_USAGE_CAN_UPDATE_BIT | TEXTURE_USAGE_CAN_COPY_FROM_BIT | TEXTURE_USAGE_CAN_COPY_TO_BIT;

	// Classify the format into depth/stencil, compressed, or plain color.
	bool is_depth_stencil = false;
	bool is_compressed = false;
	// Which optional compression family the format belongs to (so we can gate on
	// the corresponding adapter feature). Only meaningful when is_compressed.
	bool needs_bc = false;
	bool needs_etc2 = false;
	bool needs_astc = false;

	switch (p_format) {
		// Depth and depth/stencil formats.
		case DATA_FORMAT_D16_UNORM:
		case DATA_FORMAT_X8_D24_UNORM_PACK32:
		case DATA_FORMAT_D32_SFLOAT:
		case DATA_FORMAT_S8_UINT:
		case DATA_FORMAT_D16_UNORM_S8_UINT:
		case DATA_FORMAT_D24_UNORM_S8_UINT:
		case DATA_FORMAT_D32_SFLOAT_S8_UINT:
			is_depth_stencil = true;
			break;

		// BC compressed formats.
		case DATA_FORMAT_BC1_RGB_UNORM_BLOCK:
		case DATA_FORMAT_BC1_RGB_SRGB_BLOCK:
		case DATA_FORMAT_BC1_RGBA_UNORM_BLOCK:
		case DATA_FORMAT_BC1_RGBA_SRGB_BLOCK:
		case DATA_FORMAT_BC2_UNORM_BLOCK:
		case DATA_FORMAT_BC2_SRGB_BLOCK:
		case DATA_FORMAT_BC3_UNORM_BLOCK:
		case DATA_FORMAT_BC3_SRGB_BLOCK:
		case DATA_FORMAT_BC4_UNORM_BLOCK:
		case DATA_FORMAT_BC4_SNORM_BLOCK:
		case DATA_FORMAT_BC5_UNORM_BLOCK:
		case DATA_FORMAT_BC5_SNORM_BLOCK:
		case DATA_FORMAT_BC6H_UFLOAT_BLOCK:
		case DATA_FORMAT_BC6H_SFLOAT_BLOCK:
		case DATA_FORMAT_BC7_UNORM_BLOCK:
		case DATA_FORMAT_BC7_SRGB_BLOCK:
			is_compressed = true;
			needs_bc = true;
			break;
		// ETC2 compressed formats.
		case DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
		case DATA_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
		case DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
		case DATA_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
		case DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
		case DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
		case DATA_FORMAT_EAC_R11_UNORM_BLOCK:
		case DATA_FORMAT_EAC_R11_SNORM_BLOCK:
		case DATA_FORMAT_EAC_R11G11_UNORM_BLOCK:
		case DATA_FORMAT_EAC_R11G11_SNORM_BLOCK:
			is_compressed = true;
			needs_etc2 = true;
			break;
		// ASTC compressed formats.
		case DATA_FORMAT_ASTC_4x4_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_4x4_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_5x4_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_5x4_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_5x5_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_5x5_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_6x5_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_6x5_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_6x6_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_6x6_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_8x5_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_8x5_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_8x6_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_8x6_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_8x8_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_8x8_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_10x5_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_10x5_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_10x6_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_10x6_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_10x8_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_10x8_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_10x10_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_10x10_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_12x10_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_12x10_SRGB_BLOCK:
		case DATA_FORMAT_ASTC_12x12_UNORM_BLOCK:
		case DATA_FORMAT_ASTC_12x12_SRGB_BLOCK:
			is_compressed = true;
			needs_astc = true;
			break;

		default:
			break;
	}

	if (p_format == DATA_FORMAT_D32_SFLOAT_S8_UINT && !wgpuDeviceHasFeature(device, WGPUFeatureName_Depth32FloatStencil8)) {
		return 0;
	}
	if (is_depth_stencil) {
		flags.set_flag(TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
		// Depth textures can't be updated via CAN_UPDATE (CopyDst is not allowed).
		flags.clear_flag(TEXTURE_USAGE_CAN_UPDATE_BIT);
		return flags;
	}

	if (is_compressed) {
		// Gate on the corresponding optional feature — if the device didn't
		// request it, the WGPU format technically exists in the enum but any
		// attempt to create a texture with it will fail validation. Reporting
		// it as unsupported here lets Godot fall back to CPU decompression.
		if ((needs_bc && !has_texture_compression_bc) ||
				(needs_etc2 && !has_texture_compression_etc2) ||
				(needs_astc && !has_texture_compression_astc)) {
			return 0;
		}
		// Compressed textures: sampling + copy only, no render attachment, no
		// storage. CAN_UPDATE stays set — it maps to WGPUTextureUsage_CopyDst,
		// which is valid for compressed textures (uploaded via queue.writeTexture).
		return flags;
	}

	// Plain color format: render attachment is generally supported.
	flags.set_flag(TEXTURE_USAGE_COLOR_ATTACHMENT_BIT);

	// Without texture-formats-tier1, 16-bit normalized formats are stored as half floats.
	// Hide CAN_UPDATE so RD converts CPU images to exact 32-bit floats instead.
	if (!has_texture_formats_tier1 && _is_norm16_format(data_format_to_wgpu(p_format))) {
		flags.clear_flag(TEXTURE_USAGE_CAN_UPDATE_BIT);
	}

	// Storage binding: only formats explicitly listed in the WebGPU spec support it.
	// See https://gpuweb.github.io/gpuweb/#storage-texel-format-capability-matrix
	// Formats not natively supported (R8, RG8, R16, RG16) are included here because
	// Godot's renderer uses them with storage; _promote_storage_format and the WGSL
	// rewrites upgrade them to 32-bit equivalents, and the converters translate the data.
	switch (p_format) {
		case DATA_FORMAT_R8_UNORM:
		case DATA_FORMAT_R8_SNORM:
		case DATA_FORMAT_R8_UINT:
		case DATA_FORMAT_R8_SINT:
		case DATA_FORMAT_R8G8_UNORM:
		case DATA_FORMAT_R8G8_SNORM:
		case DATA_FORMAT_R8G8_UINT:
		case DATA_FORMAT_R8G8_SINT:
		case DATA_FORMAT_R8G8B8A8_UNORM:
		case DATA_FORMAT_R8G8B8A8_SNORM:
		case DATA_FORMAT_R8G8B8A8_UINT:
		case DATA_FORMAT_R8G8B8A8_SINT:
		case DATA_FORMAT_R16_SFLOAT:
		case DATA_FORMAT_R16_UINT:
		case DATA_FORMAT_R16_SINT:
		case DATA_FORMAT_R16_SNORM:
		case DATA_FORMAT_R16_UNORM:
		case DATA_FORMAT_R16G16_SFLOAT:
		case DATA_FORMAT_R16G16_SNORM:
		case DATA_FORMAT_R16G16_UNORM:
		case DATA_FORMAT_R16G16_SINT:
		case DATA_FORMAT_R16G16_UINT:
		case DATA_FORMAT_R16G16B16A16_SFLOAT:
		case DATA_FORMAT_R16G16B16A16_UINT:
		case DATA_FORMAT_R16G16B16A16_SINT:
		case DATA_FORMAT_R32_SFLOAT:
		case DATA_FORMAT_R32_UINT:
		case DATA_FORMAT_R32_SINT:
		case DATA_FORMAT_R32G32_SFLOAT:
		case DATA_FORMAT_R32G32_UINT:
		case DATA_FORMAT_R32G32_SINT:
		case DATA_FORMAT_R32G32B32A32_SFLOAT:
		case DATA_FORMAT_R32G32B32A32_UINT:
		case DATA_FORMAT_R32G32B32A32_SINT:
			flags.set_flag(TEXTURE_USAGE_STORAGE_BIT);
			break;
		default:
			break;
	}

	return flags;
}

bool RenderingDeviceDriverWebGPU::texture_can_make_shared_with_format(TextureID p_texture, DataFormat p_format, bool &r_raw_reinterpretation) {
	r_raw_reinterpretation = false;
	const WGTexture *tex = (const WGTexture *)(p_texture.id);
	ERR_FAIL_NULL_V(tex, false);
	// Only the sRGB/linear sibling registered in viewFormats can be viewed directly.
	const WGPUTextureFormat format = _data_format_to_wgpu(p_format);
	return format == tex->format || (format != WGPUTextureFormat_Undefined && format == tex->view_format_alt);
}

WGPUTextureUsage RenderingDeviceDriverWebGPU::_texture_usage_to_wgpu(BitField<TextureUsageBits> p_usage) const {
	WGPUTextureUsage flags = 0;
	if (p_usage.has_flag(TEXTURE_USAGE_SAMPLING_BIT)) {
		flags |= WGPUTextureUsage_TextureBinding;
	}
	if (p_usage.has_flag(TEXTURE_USAGE_COLOR_ATTACHMENT_BIT) || p_usage.has_flag(TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
		flags |= WGPUTextureUsage_RenderAttachment;
	}
	if (p_usage.has_flag(TEXTURE_USAGE_STORAGE_BIT) || p_usage.has_flag(TEXTURE_USAGE_STORAGE_ATOMIC_BIT)) {
		flags |= WGPUTextureUsage_StorageBinding;
	}
	if (p_usage.has_flag(TEXTURE_USAGE_CPU_READ_BIT) || p_usage.has_flag(TEXTURE_USAGE_CAN_COPY_FROM_BIT)) {
		flags |= WGPUTextureUsage_CopySrc;
	}
	if (p_usage.has_flag(TEXTURE_USAGE_CAN_UPDATE_BIT) || p_usage.has_flag(TEXTURE_USAGE_CAN_COPY_TO_BIT)) {
		flags |= WGPUTextureUsage_CopyDst;
	}
	return flags;
}

WGPUTextureDimension RenderingDeviceDriverWebGPU::_texture_type_to_dimension(TextureType p_type) const {
	switch (p_type) {
		case TEXTURE_TYPE_1D:
			return WGPUTextureDimension_1D;
		case TEXTURE_TYPE_3D:
			return WGPUTextureDimension_3D;
		default:
			return WGPUTextureDimension_2D;
	}
}

WGPUTextureViewDimension RenderingDeviceDriverWebGPU::_texture_type_to_view_dimension(TextureType p_type) const {
	switch (p_type) {
		case TEXTURE_TYPE_1D:
			return WGPUTextureViewDimension_1D;
		case TEXTURE_TYPE_2D:
			return WGPUTextureViewDimension_2D;
		case TEXTURE_TYPE_3D:
			return WGPUTextureViewDimension_3D;
		case TEXTURE_TYPE_CUBE:
			return WGPUTextureViewDimension_Cube;
		case TEXTURE_TYPE_2D_ARRAY:
			return WGPUTextureViewDimension_2DArray;
		case TEXTURE_TYPE_CUBE_ARRAY:
			return WGPUTextureViewDimension_CubeArray;
		default:
			return WGPUTextureViewDimension_2D;
	}
}

WGPUTextureFormat RenderingDeviceDriverWebGPU::_data_format_to_wgpu(DataFormat p_format) const {
	WGPUTextureFormat fmt = data_format_to_wgpu(p_format);
	if (fmt != WGPUTextureFormat_Undefined) {
		// 16-bit normalized formats need texture-formats-tier1; otherwise store them as half floats
		// (the upload/readback converters translate the data, see texture_get_gpu_pixel_size).
		if (!has_texture_formats_tier1 && _is_norm16_format(fmt)) {
			switch (fmt) {
				case WGPUTextureFormat_R16Unorm:
				case WGPUTextureFormat_R16Snorm:
					return WGPUTextureFormat_R16Float;
				case WGPUTextureFormat_RG16Unorm:
				case WGPUTextureFormat_RG16Snorm:
					return WGPUTextureFormat_RG16Float;
				default:
					return WGPUTextureFormat_RGBA16Float;
			}
		}
		return fmt;
	}

	// Fallback approximations for formats not in the base WebGPU spec.
	switch (p_format) {
		// No depth16+stencil8 in WebGPU; use depth24plus-stencil8 as nearest approximation.
		case DATA_FORMAT_D16_UNORM_S8_UINT:
			return WGPUTextureFormat_Depth24PlusStencil8;
		// 3-component (RGB) formats don't exist as WebGPU *texture* formats, only as
		// vertex attribute formats (handled by _data_format_to_wgpu_vertex). Return
		// Undefined silently here — texture_get_usages_supported_by_format probes every
		// DataFormat for texture-usage compat, and we don't want to spam warnings for
		// legitimate vertex-only formats.
		case DATA_FORMAT_R32G32B32_UINT:
		case DATA_FORMAT_R32G32B32_SINT:
		case DATA_FORMAT_R32G32B32_SFLOAT:
		case DATA_FORMAT_R16G16B16_UNORM:
		case DATA_FORMAT_R16G16B16_SNORM:
		case DATA_FORMAT_R16G16B16_UINT:
		case DATA_FORMAT_R16G16B16_SINT:
		case DATA_FORMAT_R16G16B16_SFLOAT:
		case DATA_FORMAT_R8G8B8_UNORM:
		case DATA_FORMAT_R8G8B8_SNORM:
		case DATA_FORMAT_R8G8B8_UINT:
		case DATA_FORMAT_R8G8B8_SINT:
			return WGPUTextureFormat_Undefined;
		default: {
			static thread_local LocalVector<int> warned_formats;
			int ifmt = (int)p_format;
			bool already_warned = false;
			for (uint32_t i = 0; i < warned_formats.size(); i++) {
				if (warned_formats[i] == ifmt) {
					already_warned = true;
					break;
				}
			}
			if (!already_warned) {
				warned_formats.push_back(ifmt);
				WARN_PRINT(vformat("WebGPU: Unsupported DataFormat %d (further occurrences suppressed)", ifmt));
			}
			return WGPUTextureFormat_Undefined;
		}
	}
}

// WebGPU does not support R8/RG8/R16/RG16 as storage texel formats. Any texture or
// render target that must carry STORAGE usage has to be created as a 32-bit variant
// instead. This helper centralizes that promotion so texture_create() and
// render_pipeline_create() stay in sync — otherwise a texture upgraded to R32Float
// gets bound to a pipeline that was built for R8Unorm and every submit fails with a
// GPUValidationError. Canvas SDF (R8_UNORM + STORAGE_BIT + COLOR_ATTACHMENT_BIT) is
// the motivating case.
WGPUTextureFormat RenderingDeviceDriverWebGPU::_promote_storage_format(WGPUTextureFormat p_format) const {
	switch (p_format) {
		// 8-bit formats: with texture-formats-tier1, these are valid storage texel
		// formats natively. Keeping the original format preserves blendable/filterable
		// properties (R32Float is not blendable on Adreno without float32-blendable).
		// The WGSL r8/rg8 replacement is also tier1-conditional, so shader and texture
		// formats stay in sync.
		case WGPUTextureFormat_R8Unorm:
		case WGPUTextureFormat_R8Snorm:
			if (has_texture_formats_tier1) {
				return p_format;
			}
			return WGPUTextureFormat_R32Float;
		case WGPUTextureFormat_R8Uint:
			if (has_texture_formats_tier1) {
				return p_format;
			}
			return WGPUTextureFormat_R32Uint;
		case WGPUTextureFormat_R8Sint:
			if (has_texture_formats_tier1) {
				return p_format;
			}
			return WGPUTextureFormat_R32Sint;
		case WGPUTextureFormat_RG8Unorm:
		case WGPUTextureFormat_RG8Snorm:
			if (has_texture_formats_tier1) {
				return p_format;
			}
			return WGPUTextureFormat_RG32Float;
		case WGPUTextureFormat_RG8Uint:
			if (has_texture_formats_tier1) {
				return p_format;
			}
			return WGPUTextureFormat_RG32Uint;
		case WGPUTextureFormat_RG8Sint:
			if (has_texture_formats_tier1) {
				return p_format;
			}
			return WGPUTextureFormat_RG32Sint;
		// 16-bit float/int formats: always promote (the WGSL text gets the same rewrite) since
		// read_write access needs a 32-bit format. Normalized 16-bit formats only exist with
		// texture-formats-tier1, which makes them read/write-only storage formats natively.
		case WGPUTextureFormat_R16Float:
			return WGPUTextureFormat_R32Float;
		case WGPUTextureFormat_R16Uint:
			return WGPUTextureFormat_R32Uint;
		case WGPUTextureFormat_R16Sint:
			return WGPUTextureFormat_R32Sint;
		case WGPUTextureFormat_RG16Float:
			return WGPUTextureFormat_RG32Float;
		case WGPUTextureFormat_RG16Uint:
			return WGPUTextureFormat_RG32Uint;
		case WGPUTextureFormat_RG16Sint:
			return WGPUTextureFormat_RG32Sint;
		default:
			return p_format;
	}
}

RDD::DataFormat RenderingDeviceDriverWebGPU::_wgpu_to_data_format(WGPUTextureFormat p_format) const {
	// TODO: Full reverse mapping. For now, handle common cases.
	switch (p_format) {
		case WGPUTextureFormat_BGRA8Unorm:
			return DATA_FORMAT_B8G8R8A8_UNORM;
		case WGPUTextureFormat_RGBA8Unorm:
			return DATA_FORMAT_R8G8B8A8_UNORM;
		default:
			return DATA_FORMAT_MAX;
	}
}

// =============================================================================
// SAMPLERS
// =============================================================================

RDD::SamplerID RenderingDeviceDriverWebGPU::sampler_create(const SamplerState &p_state) {
	WGPUSamplerDescriptor desc = {};

	auto map_filter = [](SamplerFilter f) -> WGPUFilterMode {
		return (f == SAMPLER_FILTER_LINEAR) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
	};
	auto map_address = [](SamplerRepeatMode m) -> WGPUAddressMode {
		switch (m) {
			case SAMPLER_REPEAT_MODE_REPEAT:
				return WGPUAddressMode_Repeat;
			case SAMPLER_REPEAT_MODE_MIRRORED_REPEAT:
				return WGPUAddressMode_MirrorRepeat;
			default:
				return WGPUAddressMode_ClampToEdge;
		}
	};
	auto map_compare = [](CompareOperator op) -> WGPUCompareFunction {
		switch (op) {
			case COMPARE_OP_NEVER:
				return WGPUCompareFunction_Never;
			case COMPARE_OP_LESS:
				return WGPUCompareFunction_Less;
			case COMPARE_OP_EQUAL:
				return WGPUCompareFunction_Equal;
			case COMPARE_OP_LESS_OR_EQUAL:
				return WGPUCompareFunction_LessEqual;
			case COMPARE_OP_GREATER:
				return WGPUCompareFunction_Greater;
			case COMPARE_OP_NOT_EQUAL:
				return WGPUCompareFunction_NotEqual;
			case COMPARE_OP_GREATER_OR_EQUAL:
				return WGPUCompareFunction_GreaterEqual;
			case COMPARE_OP_ALWAYS:
				return WGPUCompareFunction_Always;
			default:
				return WGPUCompareFunction_Undefined;
		}
	};

	desc.magFilter = map_filter(p_state.mag_filter);
	desc.minFilter = map_filter(p_state.min_filter);
	desc.mipmapFilter = (p_state.mip_filter == SAMPLER_FILTER_LINEAR) ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
	desc.addressModeU = map_address(p_state.repeat_u);
	desc.addressModeV = map_address(p_state.repeat_v);
	desc.addressModeW = map_address(p_state.repeat_w);
	// WebGPU rejects negative LOD clamps, which Godot's samplers use to mean "unclamped".
	desc.lodMinClamp = MAX(0.0f, p_state.min_lod);
	desc.lodMaxClamp = MAX(desc.lodMinClamp, p_state.max_lod);

	if (p_state.enable_compare) {
		desc.compare = map_compare(p_state.compare_op);
	}

	if (p_state.use_anisotropy) {
		desc.maxAnisotropy = (uint16_t)p_state.anisotropy_max;
		// WebGPU requires minFilter, magFilter, and mipmapFilter to all be
		// Linear when maxAnisotropy > 1.
		if (desc.maxAnisotropy > 1) {
			desc.minFilter = WGPUFilterMode_Linear;
			desc.magFilter = WGPUFilterMode_Linear;
			desc.mipmapFilter = WGPUMipmapFilterMode_Linear;
		}
	} else {
		desc.maxAnisotropy = 1;
	}

	WGPUSampler sampler = wgpuDeviceCreateSampler(device, &desc);
	ERR_FAIL_COND_V(sampler == nullptr, SamplerID());

	// Store the WGPUSampler handle directly as the ID (no wrapper struct needed).
	return SamplerID((uint64_t)sampler);
}

void RenderingDeviceDriverWebGPU::sampler_free(SamplerID p_sampler) {
	WGPUSampler sampler = (WGPUSampler)(p_sampler.id);
	if (sampler) {
		wgpuSamplerRelease(sampler);
	}
}

bool RenderingDeviceDriverWebGPU::sampler_is_format_supported_for_filter(DataFormat p_format, SamplerFilter p_filter) {
	if (p_filter == SAMPLER_FILTER_NEAREST) {
		return true; // All formats support nearest filtering.
	}
	// Linear filtering rules on WebGPU:
	//   - 32-bit float formats (R32F / RG32F / RGBA32F) require the "float32-filterable"
	//     optional feature. Without it the sampler must use NEAREST.
	//   - Integer formats (UINT / SINT) are never filterable in any API.
	//   - All other formats (8/16-bit normalized, 16F, SRGB, BCn/ETC2/ASTC, depth) are
	//     filterable on any spec-compliant WebGPU implementation.
	switch (p_format) {
		// 32-bit float formats — gated on the float32-filterable optional feature.
		case DATA_FORMAT_R32_SFLOAT:
		case DATA_FORMAT_R32G32_SFLOAT:
		case DATA_FORMAT_R32G32B32_SFLOAT:
		case DATA_FORMAT_R32G32B32A32_SFLOAT:
			return float32_filterable_supported;

		// Integer formats — never filterable.
		case DATA_FORMAT_R8_UINT:
		case DATA_FORMAT_R8_SINT:
		case DATA_FORMAT_R8G8_UINT:
		case DATA_FORMAT_R8G8_SINT:
		case DATA_FORMAT_R8G8B8A8_UINT:
		case DATA_FORMAT_R8G8B8A8_SINT:
		case DATA_FORMAT_R16_UINT:
		case DATA_FORMAT_R16_SINT:
		case DATA_FORMAT_R16G16_UINT:
		case DATA_FORMAT_R16G16_SINT:
		case DATA_FORMAT_R16G16B16A16_UINT:
		case DATA_FORMAT_R16G16B16A16_SINT:
		case DATA_FORMAT_R32_UINT:
		case DATA_FORMAT_R32_SINT:
		case DATA_FORMAT_R32G32_UINT:
		case DATA_FORMAT_R32G32_SINT:
		case DATA_FORMAT_R32G32B32A32_UINT:
		case DATA_FORMAT_R32G32B32A32_SINT:
			return false;

		default:
			return true;
	}
}

// =============================================================================
// VERTEX FORMAT
// =============================================================================

RDD::VertexFormatID RenderingDeviceDriverWebGPU::vertex_format_create(Span<VertexAttribute> p_vertex_attribs, const VertexAttributeBindingsMap &p_vertex_bindings) {
	WGVertexFormat *vf = memnew(WGVertexFormat);

	// Build attribute list.
	for (uint32_t i = 0; i < p_vertex_attribs.size(); i++) {
		const VertexAttribute &va = p_vertex_attribs[i];
		WGVertexFormat::Attribute attr;
		attr.location = va.location;
		attr.format = data_format_to_wgpu_vertex(va.format);
		if (attr.format == (WGPUVertexFormat)0) {
			WARN_PRINT(vformat("WebGPU: Unsupported vertex DataFormat %d", (int)va.format));
		}
		attr.offset = va.offset;
		attr.binding = (va.binding == UINT32_MAX) ? i : va.binding;
		vf->attributes.push_back(attr);
	}

	// Build binding list from p_vertex_bindings map.
	if (!p_vertex_bindings.is_empty()) {
		uint32_t max_binding = 0;
		for (const KeyValue<uint32_t, VertexAttributeBinding> &kv : p_vertex_bindings) {
			max_binding = MAX(max_binding, kv.key);
		}
		vf->bindings.resize(max_binding + 1);
		for (const KeyValue<uint32_t, VertexAttributeBinding> &kv : p_vertex_bindings) {
			vf->bindings[kv.key].stride = kv.value.stride;
			vf->bindings[kv.key].step_mode = (kv.value.frequency == VERTEX_FREQUENCY_INSTANCE)
					? WGPUVertexStepMode_Instance
					: WGPUVertexStepMode_Vertex;
		}
	} else {
		// Build bindings from vertex attribute data (stride/frequency stored per-attribute in older API).
		HashMap<uint32_t, bool> seen;
		for (uint32_t i = 0; i < p_vertex_attribs.size(); i++) {
			const VertexAttribute &va = p_vertex_attribs[i];
			uint32_t binding = (va.binding == UINT32_MAX) ? i : va.binding;
			if (!seen.has(binding)) {
				if (vf->bindings.size() <= binding) {
					vf->bindings.resize(binding + 1);
				}
				vf->bindings[binding].stride = va.stride;
				vf->bindings[binding].step_mode = (va.frequency == VERTEX_FREQUENCY_INSTANCE)
						? WGPUVertexStepMode_Instance
						: WGPUVertexStepMode_Vertex;
				seen[binding] = true;
			}
		}
	}

	return VertexFormatID(vf);
}

void RenderingDeviceDriverWebGPU::vertex_format_free(VertexFormatID p_vertex_format) {
	WGVertexFormat *vf = (WGVertexFormat *)(p_vertex_format.id);
	ERR_FAIL_NULL(vf);
	memdelete(vf);
}

// =============================================================================
// BARRIERS (ALL NO-OPS)
// =============================================================================

void RenderingDeviceDriverWebGPU::command_pipeline_barrier(
		CommandBufferID p_cmd_buffer,
		BitField<PipelineStageBits> p_src_stages,
		BitField<PipelineStageBits> p_dst_stages,
		VectorView<MemoryAccessBarrier> p_memory_barriers,
		VectorView<BufferBarrier> p_buffer_barriers,
		VectorView<TextureBarrier> p_texture_barriers,
		VectorView<AccelerationStructureBarrier> p_acceleration_structure_barriers) {
	// No-op: WebGPU handles synchronization automatically.
}

// =============================================================================
// FENCES
// =============================================================================

RDD::FenceID RenderingDeviceDriverWebGPU::fence_create() {
	WGFence *fence = memnew(WGFence);
	return FenceID(fence);
}

Error RenderingDeviceDriverWebGPU::fence_wait(FenceID p_fence) {
	WGFence *fence = (WGFence *)(p_fence.id);
	ERR_FAIL_NULL_V(fence, ERR_INVALID_PARAMETER);
#ifndef WEB_ENABLED
	if (fence->pending_callbacks > 0) {
		WGPUFutureWaitInfo wait = { fence->future, false };
		ERR_FAIL_COND_V_MSG(wgpuInstanceWaitAny(context_driver->get_instance(), 1, &wait, 30000000000ULL) != WGPUWaitStatus_Success, ERR_TIMEOUT, "WebGPU: GPU fence timed out.");
	}
	return OK;
#else
	// The browser can't block on GPU work (no Asyncify), so this only delivers pending
	// callbacks and returns. RD waits on the previous frame's fence, which has had a
	// full frame to complete; WebGPU keeps resources alive for in-flight work anyway.
	WGPUInstance inst = context_driver ? context_driver->get_instance() : nullptr;
	if (inst) {
		wgpuInstanceProcessEvents(inst);
	}
	return OK;
#endif
}

void RenderingDeviceDriverWebGPU::fence_free(FenceID p_fence) {
	WGFence *fence = (WGFence *)(p_fence.id);
	if (!fence) {
		return;
	}
#ifndef WEB_ENABLED
	if (fence->pending_callbacks > 0) {
		fence_wait(p_fence);
	}
#endif

	// The callbacks still in flight hold the pointer; the last one deletes the fence.
	if (fence->pending_callbacks > 0) {
		fence->freed = true;
		return;
	}

	memdelete(fence);
}

// =============================================================================
// SEMAPHORES (NO-OPS - single queue)
// =============================================================================

RDD::SemaphoreID RenderingDeviceDriverWebGPU::semaphore_create() {
	WGSemaphore *sem = memnew(WGSemaphore);
	return SemaphoreID(sem);
}

void RenderingDeviceDriverWebGPU::semaphore_free(SemaphoreID p_semaphore) {
	WGSemaphore *sem = (WGSemaphore *)(p_semaphore.id);
	ERR_FAIL_NULL(sem);
	memdelete(sem);
}

// =============================================================================
// COMMAND BUFFERS
// =============================================================================

RDD::CommandQueueFamilyID RenderingDeviceDriverWebGPU::command_queue_family_get(BitField<CommandQueueFamilyBits> p_cmd_queue_family_bits, RenderingContextDriver::SurfaceID p_surface) {
	return CommandQueueFamilyID(1); // Single family that supports everything.
}

RDD::CommandQueueID RenderingDeviceDriverWebGPU::command_queue_create(CommandQueueFamilyID p_cmd_queue_family, bool p_identify_as_main_queue) {
	WGCommandQueue *cq = memnew(WGCommandQueue);
	cq->queue = queue; // Share the single device queue.
	return CommandQueueID(cq);
}

Error RenderingDeviceDriverWebGPU::command_queue_execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_semaphores, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_semaphores, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains) {
	// Submit all command buffers; submission consumes them (released below).
	LocalVector<WGPUCommandBuffer> wgpu_cmd_buffers;
	for (uint32_t i = 0; i < p_cmd_buffers.size(); i++) {
		WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffers[i].id);
		if (cmd && cmd->finished_buffer) {
			wgpu_cmd_buffers.push_back(cmd->finished_buffer);
			cmd->finished_buffer = nullptr;
		}
	}

	if (wgpu_cmd_buffers.size() > 0) {
		// Flush batched push constant data to GPU before submitting command buffers.
		_flush_push_constant_shadow();

		wgpuQueueSubmit(queue, wgpu_cmd_buffers.size(), wgpu_cmd_buffers.ptr());
		for (WGPUCommandBuffer finished : wgpu_cmd_buffers) {
			wgpuCommandBufferRelease(finished);
		}
		// Diagnostic: log submit count for the first few frames.
#if defined(WEBGPU_VERBOSE) && defined(WEB_ENABLED)
		static int _submit_log = 0;
		if (_submit_log < 10) {
			EM_ASM({ console.log('[DIAG-SUBMIT] frame=' + $0 + ' cmds=' + $1); }, _submit_log, (int)wgpu_cmd_buffers.size());
			_submit_log++;
		}
#endif // WEBGPU_VERBOSE
	}

	// Signal fence when GPU work completes via async callback.
	if (p_cmd_fence) {
		WGFence *fence = (WGFence *)(p_cmd_fence.id);
		if (fence) {
#ifndef WEB_ENABLED
			// WaitAnyOnly callbacks only fire when waited on, so finish the previous one first.
			if (fence->pending_callbacks > 0) {
				fence_wait(p_cmd_fence);
			}
#endif
			fence->pending_callbacks++;
			WGPUQueueWorkDoneCallbackInfo cb = {};
			cb.mode = WEBGPU_CALLBACK_MODE;
			// Overload set (see definitions near the top of this file): the
			// port's WGPUQueueWorkDoneCallback typedef selects the matching
			// signature. A "no matching conversion" error here means a new
			// emdawnwebgpu changed the callback shape again.
			cb.callback = _fence_work_done_callback;
			cb.userdata1 = fence;
			cb.userdata2 = nullptr;
#ifndef WEB_ENABLED
			cb.mode = WGPUCallbackMode_WaitAnyOnly;
#endif
			fence->future = wgpuQueueOnSubmittedWorkDone(queue, cb);
		}
	}

	// Present swap chains.
	for (uint32_t i = 0; i < p_swap_chains.size(); i++) {
		WGSwapChain *sc = (WGSwapChain *)(p_swap_chains[i].id);
		if (sc && sc->surface) {
#ifndef __EMSCRIPTEN__
			wgpuSurfacePresent(sc->surface);
#endif
			// Post-submit diagnostic clear DISABLED — only pre-submit CYAN is active.
			// This lets us see if Godot's submit overwrites the cyan.

			// Release the surface texture and view so the browser can composite the
			// frame within this requestAnimationFrame callback. If we hold onto the
			// texture reference until next frame's acquire, the browser won't present
			// the rendered content.
			if (sc->current_view) {
				wgpuTextureViewRelease(sc->current_view);
				sc->current_view = nullptr;
			}
			if (sc->current_texture) {
				wgpuTextureRelease(sc->current_texture);
				sc->current_texture = nullptr;
			}
		}
	}

	return OK;
}

void RenderingDeviceDriverWebGPU::command_queue_free(CommandQueueID p_cmd_queue) {
	WGCommandQueue *cq = (WGCommandQueue *)(p_cmd_queue.id);
	ERR_FAIL_NULL(cq);
	memdelete(cq);
}

RDD::CommandPoolID RenderingDeviceDriverWebGPU::command_pool_create(CommandQueueFamilyID p_cmd_queue_family, CommandBufferType p_cmd_buffer_type) {
	WGCommandPool *pool = memnew(WGCommandPool);
	pool->buffer_type = p_cmd_buffer_type;
	return CommandPoolID(pool);
}

bool RenderingDeviceDriverWebGPU::command_pool_reset(CommandPoolID p_cmd_pool) {
	// Nothing to reset — each command buffer creates its own encoder.
	return true;
}

void RenderingDeviceDriverWebGPU::command_pool_free(CommandPoolID p_cmd_pool) {
	WGCommandPool *pool = (WGCommandPool *)(p_cmd_pool.id);
	ERR_FAIL_NULL(pool);
	for (WGCommandBuffer *cmd : pool->buffers) {
		cmd->end_active_encoder();
		if (cmd->encoder) {
			wgpuCommandEncoderRelease(cmd->encoder);
		}
		if (cmd->finished_buffer) {
			wgpuCommandBufferRelease(cmd->finished_buffer);
		}
		memdelete(cmd);
	}
	memdelete(pool);
}

RDD::CommandBufferID RenderingDeviceDriverWebGPU::command_buffer_create(CommandPoolID p_cmd_pool) {
	WGCommandPool *pool = (WGCommandPool *)(p_cmd_pool.id);
	ERR_FAIL_NULL_V(pool, CommandBufferID());
	WGCommandBuffer *cmd = memnew(WGCommandBuffer);
	pool->buffers.push_back(cmd);
	return CommandBufferID(cmd);
}

bool RenderingDeviceDriverWebGPU::command_buffer_begin(CommandBufferID p_cmd_buffer) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL_V(cmd, false);

	// A buffer recorded but never submitted still owns its finished WGPUCommandBuffer.
	if (cmd->finished_buffer) {
		wgpuCommandBufferRelease(cmd->finished_buffer);
		cmd->finished_buffer = nullptr;
	}

	WGPUCommandEncoderDescriptor desc = {};
	cmd->encoder = wgpuDeviceCreateCommandEncoder(device, &desc);
	ERR_FAIL_COND_V(cmd->encoder == nullptr, false);

	cmd->active_encoder = WGCommandBuffer::NONE;
	cmd->push_constants_dirty = false;
	cmd->push_constant_data_len = 0;

	return true;
}

bool RenderingDeviceDriverWebGPU::command_buffer_begin_secondary(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, uint32_t p_subpass, FramebufferID p_framebuffer) {
	// WebGPU has no secondary command buffers. Treat as primary.
	return command_buffer_begin(p_cmd_buffer);
}

void RenderingDeviceDriverWebGPU::command_buffer_end(CommandBufferID p_cmd_buffer) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);

	// End any active render/compute pass.
	cmd->end_active_encoder();

	// Finish the command encoder.
	if (cmd->encoder) {
		cmd->finished_buffer = wgpuCommandEncoderFinish(cmd->encoder, nullptr);
		wgpuCommandEncoderRelease(cmd->encoder);
		cmd->encoder = nullptr;
	}
}

void RenderingDeviceDriverWebGPU::command_buffer_execute_secondary(CommandBufferID p_cmd_buffer, VectorView<CommandBufferID> p_secondary_cmd_buffers) {
	// No-op: WebGPU has no secondary command buffers.
}

// =============================================================================
// SWAP CHAIN
// =============================================================================

RDD::SwapChainID RenderingDeviceDriverWebGPU::swap_chain_create(RenderingContextDriver::SurfaceID p_surface) {
	WGSwapChain *sc = memnew(WGSwapChain);
	sc->surface = context_driver->surface_get_handle(p_surface);
	sc->surface_id = p_surface;
	sc->format = context_driver->surface_get_format(p_surface);

	// Create a render pass descriptor for this swap chain.
	// Used by swap_chain_get_render_pass() so the RD layer can create compatible pipelines.
	WGRenderPass *rp = memnew(WGRenderPass);
	RDD::Attachment att;
	att.format = swap_chain_get_format(SwapChainID(sc));
	att.samples = TEXTURE_SAMPLES_1;
	att.load_op = ATTACHMENT_LOAD_OP_CLEAR;
	att.store_op = ATTACHMENT_STORE_OP_STORE;
	att.stencil_load_op = ATTACHMENT_LOAD_OP_DONT_CARE;
	att.stencil_store_op = ATTACHMENT_STORE_OP_DONT_CARE;
	att.initial_layout = TEXTURE_LAYOUT_UNDEFINED;
	att.final_layout = TEXTURE_LAYOUT_UNDEFINED;
	rp->attachments.push_back(att);

	WGRenderPass::SubpassInfo subpass;
	RDD::AttachmentReference color_ref;
	color_ref.attachment = 0;
	color_ref.layout = TEXTURE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	subpass.color_references.push_back(color_ref);
	rp->subpasses.push_back(subpass);
	rp->is_swap_chain_pass = true;
	sc->render_pass = rp;

	return SwapChainID(sc);
}

Error RenderingDeviceDriverWebGPU::swap_chain_resize(CommandQueueID p_cmd_queue, SwapChainID p_swap_chain, uint32_t p_desired_framebuffer_count) {
	WGSwapChain *sc = (WGSwapChain *)(p_swap_chain.id);
	ERR_FAIL_NULL_V(sc, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(sc->surface == nullptr, ERR_INVALID_PARAMETER);

	uint32_t width = context_driver->surface_get_width(sc->surface_id);
	uint32_t height = context_driver->surface_get_height(sc->surface_id);
	if (width == 0 || height == 0) {
		return ERR_SKIP; // Canvas not yet sized.
	}

	// If previously configured, unconfigure first to reconfigure with new dimensions.
	if (sc->configured) {
		wgpuSurfaceUnconfigure(sc->surface);
		sc->configured = false;
	}

	WGPUSurfaceConfiguration config = {};
	config.device = device;
	config.format = sc->format;
	config.usage = WGPUTextureUsage_RenderAttachment;
	config.alphaMode = context_driver->surface_get_alpha_mode(sc->surface_id);
	config.presentMode = context_driver->surface_get_present_mode(sc->surface_id);
	config.width = width;
	config.height = height;
	wgpuSurfaceConfigure(sc->surface, &config);

	// Uncaptured error listener + monkey patches are installed in initialize().

	sc->width = width;
	sc->height = height;
	sc->configured = true;
	// Clear the needs_resize flag so swap_chain_acquire_framebuffer doesn't
	context_driver->surface_set_needs_resize(sc->surface_id, false);
	return OK;
}

RDD::FramebufferID RenderingDeviceDriverWebGPU::swap_chain_acquire_framebuffer(CommandQueueID p_cmd_queue, SwapChainID p_swap_chain, bool &r_resize_required) {
	WGSwapChain *sc = (WGSwapChain *)(p_swap_chain.id);
	ERR_FAIL_NULL_V(sc, FramebufferID());
	if (!sc->configured || context_driver->surface_get_needs_resize(sc->surface_id)) {
		// Not yet sized or surface dimensions changed — request a resize.
		if (!sc->configured) {
			print_verbose("WebGPU: swap_chain_acquire_framebuffer: not configured, requesting resize");
		}
		r_resize_required = true;
		return FramebufferID();
	}

	// Release resources from the previous frame.
	if (sc->current_framebuffer) {
		memdelete(sc->current_framebuffer);
		sc->current_framebuffer = nullptr;
	}
	if (sc->current_view) {
		wgpuTextureViewRelease(sc->current_view);
		sc->current_view = nullptr;
	}
	if (sc->current_texture) {
		wgpuTextureRelease(sc->current_texture);
		sc->current_texture = nullptr;
	}

	// Acquire the next swap chain texture.
	WGPUSurfaceTexture surface_texture = {};
	wgpuSurfaceGetCurrentTexture(sc->surface, &surface_texture);

	// Diagnostic: log surface texture status for the first few frames.
	static int _st_log = 0;
	if (_st_log < 10) {
		WEBGPU_DIAG({ console.log('[SURFACE] status=' + $0 + ' texture=' + ($1 ? 'valid' : 'NULL')); }, (int)surface_texture.status, (int)(surface_texture.texture != nullptr));
		_st_log++;
	}

	if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Lost) {
		ERR_PRINT_ONCE("WebGPU: surface lost — device or surface is no longer usable.");
		return FramebufferID();
	}

	if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal ||
			surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated) {
		WARN_PRINT_ONCE(vformat("WebGPU: wgpuSurfaceGetCurrentTexture: resize-needed status=%d", (int)surface_texture.status));
		r_resize_required = true;
		return FramebufferID();
	}
	if (surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal) {
		WARN_PRINT_ONCE(vformat("WebGPU: wgpuSurfaceGetCurrentTexture: unexpected status=%d", (int)surface_texture.status));
		return FramebufferID();
	}

	sc->current_texture = surface_texture.texture;

	// Get actual surface texture dimensions. On web, wgpuSurfaceConfigure may be called
	// with OS physical-pixel dimensions (e.g. 1152×648 on HiDPI), but the browser canvas
	// may be CSS-sized (e.g. 756×417). wgpuTextureGetWidth gives the real GPU texture size.
	uint32_t actual_w = wgpuTextureGetWidth(sc->current_texture);
	uint32_t actual_h = wgpuTextureGetHeight(sc->current_texture);
	if (actual_w != sc->width || actual_h != sc->height) {
		WARN_PRINT_ONCE(vformat("WebGPU: surface texture (%d x %d) differs from configured (%d x %d) — using actual size",
				actual_w, actual_h, sc->width, sc->height));
		sc->width = actual_w;
		sc->height = actual_h;
		// Sync the context driver's stored surface dimensions so that screen_get_width/height
		// (used by the blit dst_rect normalization) returns the actual canvas size, not the
		// configured HiDPI physical-pixel size. Without this, the blit only covers a fraction
		// of the swap chain surface.
		context_driver->surface_set_size(sc->surface_id, actual_w, actual_h);
	}

	// Create a view for this frame's swap chain texture.
	WGPUTextureViewDescriptor view_desc = {};
	view_desc.format = sc->format;
	view_desc.dimension = WGPUTextureViewDimension_2D;
	view_desc.baseMipLevel = 0;
	view_desc.mipLevelCount = 1;
	view_desc.baseArrayLayer = 0;
	view_desc.arrayLayerCount = 1;
	view_desc.aspect = WGPUTextureAspect_All;
	sc->current_view = wgpuTextureCreateView(sc->current_texture, &view_desc);
	ERR_FAIL_COND_V(sc->current_view == nullptr, FramebufferID());

	// Wrap in a WGFramebuffer. The WGTexture pointer is null since we manage
	// the texture lifetime through the swap chain, not the framebuffer.
	WGFramebuffer *fb = memnew(WGFramebuffer);
	fb->render_pass = sc->render_pass;
	fb->width = sc->width;
	fb->height = sc->height;
	fb->attachments.push_back(nullptr);
	fb->attachment_views.push_back(sc->current_view);
	sc->current_framebuffer = fb;

	r_resize_required = false;
	return FramebufferID(fb);
}

RDD::RenderPassID RenderingDeviceDriverWebGPU::swap_chain_get_render_pass(SwapChainID p_swap_chain) {
	WGSwapChain *sc = (WGSwapChain *)(p_swap_chain.id);
	ERR_FAIL_NULL_V(sc, RenderPassID());
	return RenderPassID(sc->render_pass);
}

RDD::DataFormat RenderingDeviceDriverWebGPU::swap_chain_get_format(SwapChainID p_swap_chain) {
	WGSwapChain *sc = (WGSwapChain *)(p_swap_chain.id);
	ERR_FAIL_NULL_V(sc, DATA_FORMAT_MAX);
	return _wgpu_to_data_format(sc->format);
}

void RenderingDeviceDriverWebGPU::swap_chain_free(SwapChainID p_swap_chain) {
	WGSwapChain *sc = (WGSwapChain *)(p_swap_chain.id);
	ERR_FAIL_NULL(sc);
	if (sc->current_framebuffer) {
		memdelete(sc->current_framebuffer);
	}
	if (sc->current_view) {
		wgpuTextureViewRelease(sc->current_view);
	}
	if (sc->current_texture) {
		wgpuTextureRelease(sc->current_texture);
	}
	if (sc->render_pass) {
		memdelete(sc->render_pass);
	}
	if (sc->surface && sc->configured) {
		wgpuSurfaceUnconfigure(sc->surface);
	}
	memdelete(sc);
}

// =============================================================================
// FRAMEBUFFER
// =============================================================================

RDD::FramebufferID RenderingDeviceDriverWebGPU::framebuffer_create(RenderPassID p_render_pass, VectorView<TextureID> p_attachments, uint32_t p_width, uint32_t p_height) {
	WGFramebuffer *fb = memnew(WGFramebuffer);
	fb->render_pass = (WGRenderPass *)(p_render_pass.id);
	fb->width = p_width;
	fb->height = p_height;

	for (uint32_t i = 0; i < p_attachments.size(); i++) {
		WGTexture *tex = (WGTexture *)(p_attachments[i].id);
		fb->attachments.push_back(tex);
		fb->attachment_views.push_back(tex ? tex->default_view : nullptr);
	}

	return FramebufferID(fb);
}

void RenderingDeviceDriverWebGPU::framebuffer_free(FramebufferID p_framebuffer) {
	WGFramebuffer *fb = (WGFramebuffer *)(p_framebuffer.id);
	ERR_FAIL_NULL(fb);
	memdelete(fb);
}

// =============================================================================
// SHADER
// =============================================================================

// Helper: map Godot stage mask bits to WGPUShaderStage visibility flags.
// After Tint processing (comparison splitting, type changes), the WGSL shader
// may reference bindings in stages not predicted by the original SPIR-V reflection.
// To avoid visibility mismatches, OR in both Vertex and Fragment for any render
// shader that uses either stage. Compute stays separate.
static WGPUShaderStage _stages_to_wgpu_visibility(uint32_t p_stage_mask) {
	WGPUShaderStage vis = WGPUShaderStage_None;
	bool has_render = (p_stage_mask & ((1u << RDD::SHADER_STAGE_VERTEX) | (1u << RDD::SHADER_STAGE_FRAGMENT) | (1u << RDD::SHADER_STAGE_TESSELATION_CONTROL) | (1u << RDD::SHADER_STAGE_TESSELATION_EVALUATION))) != 0;
	if (has_render) {
		vis = (WGPUShaderStage)(WGPUShaderStage_Vertex | WGPUShaderStage_Fragment);
	}
	if (p_stage_mask & (1u << RDD::SHADER_STAGE_COMPUTE)) {
		vis = (WGPUShaderStage)(vis | WGPUShaderStage_Compute);
	}
	return vis;
}

static bool _collect_storage_buffer_bindings(const char *p_wgsl, HashSet<uint32_t> &r_keys) {
	webgpu_metadata::Shader reflection_info;
	if (!webgpu_metadata::parse(p_wgsl, reflection_info)) {
		return false;
	}
	for (const auto &resource : reflection_info.resources) {
		if (resource.kind == webgpu_metadata::STORAGE_BUFFER || resource.kind == webgpu_metadata::READ_ONLY_STORAGE_BUFFER) {
			r_keys.insert((resource.group << 16) | resource.binding);
		}
	}
	return true;
}

// Shared by shader_create_from_container and _create_module_with_spec_constants, so specialized
// modules always match the base shader's bind group layouts.
CharString RenderingDeviceDriverWebGPU::_postprocess_wgsl(const char *p_wgsl, ShaderStage p_stage, const HashSet<uint32_t> &p_vertex_storage_buffers, WGSLRewrites &r_rewrites) const {
	CharString cs(p_wgsl);

	// Remap unsupported 8-bit storage texture format names in WGSL.
	// r8* and rg8* are not valid base WebGPU storage texel formats — remap to
	// 32-bit equivalents. With texture-formats-tier1 these formats are valid
	// natively, so skip the remap to preserve blendable/filterable properties.
	if (!has_texture_formats_tier1 &&
			(strstr(cs.get_data(), "r8unorm") || strstr(cs.get_data(), "r8snorm") ||
					strstr(cs.get_data(), "r8uint") || strstr(cs.get_data(), "r8sint") ||
					strstr(cs.get_data(), "rg8unorm") || strstr(cs.get_data(), "rg8snorm") ||
					strstr(cs.get_data(), "rg8uint") || strstr(cs.get_data(), "rg8sint"))) {
		String ws(cs.get_data());
		ws = ws.replace("rg8unorm", "rg32float");
		ws = ws.replace("rg8snorm", "rg32float");
		ws = ws.replace("rg8uint", "rg32uint");
		ws = ws.replace("rg8sint", "rg32sint");
		ws = ws.replace("r8unorm", "r32float");
		ws = ws.replace("r8snorm", "r32float");
		ws = ws.replace("r8uint", "r32uint");
		ws = ws.replace("r8sint", "r32sint");
		cs = ws.utf8();
	}

	// If texture-formats-tier1 is not available, remap 16-bit SNORM/UNORM storage
	// texture format names to their float equivalents in the WGSL text. The format
	// string lengths are preserved (pad with spaces) so scan offsets remain valid.
	// r16snorm  → r16float  (same 8 chars)
	// r16unorm  → r16float  (same 8 chars)
	// rg16snorm → rg16float (same 9 chars — "rg16float" is 9 chars, perfect)
	// rg16unorm → rg16float (same 9 chars)
	// rgba16snorm → rgba16float (11 vs 11 — perfect)
	// rgba16unorm → rgba16float (11 vs 11 — perfect)
	if (!has_texture_formats_tier1) {
		char *q = cs.ptrw();
		while (*q) {
			if (strncmp(q, "rgba16snorm", 11) == 0) {
				memcpy(q, "rgba16float", 11);
				q += 11;
			} else if (strncmp(q, "rgba16unorm", 11) == 0) {
				memcpy(q, "rgba16float", 11);
				q += 11;
			} else if (strncmp(q, "rg16snorm", 9) == 0) {
				memcpy(q, "rg16float", 9);
				q += 9;
			} else if (strncmp(q, "rg16unorm", 9) == 0) {
				memcpy(q, "rg16float", 9);
				q += 9;
			} else if (strncmp(q, "r16snorm", 8) == 0) {
				memcpy(q, "r16float", 8);
				q += 8;
			} else if (strncmp(q, "r16unorm", 8) == 0) {
				memcpy(q, "r16float", 8);
				q += 8;
			} else {
				q++;
			}
		}
	}

	// 16-bit float/int storage formats must match _promote_storage_format, which widens them to
	// 32 bits so read_write access stays valid. Normalized 16-bit names only remain with
	// texture-formats-tier1, where they are native storage formats. Replacements keep string length.
	{
		char *q = cs.ptrw();
		while (*q) {
			// RG16 float/int → rg32 equivalents
			if (strncmp(q, "rg16float", 9) == 0) {
				memcpy(q, "rg32float", 9);
				q += 9;
			} else if (strncmp(q, "rg16uint", 8) == 0) {
				memcpy(q, "rg32uint", 8);
				q += 8;
			} else if (strncmp(q, "rg16sint", 8) == 0) {
				memcpy(q, "rg32sint", 8);
				q += 8;
			}
			// R16 float/int → r32 equivalents
			else if (strncmp(q, "r16float", 8) == 0) {
				memcpy(q, "r32float", 8);
				q += 8;
			} else if (strncmp(q, "r16uint", 7) == 0) {
				memcpy(q, "r32uint", 7);
				q += 7;
			} else if (strncmp(q, "r16sint", 7) == 0) {
				memcpy(q, "r32sint", 7);
				q += 7;
			} else {
				q++;
			}
		}
	}

	// WebGPU forbids writable storage buffers in the vertex stage, so demote them to read. A fragment
	// read_write buffer the vertex stage also declares is demoted too: one BGL entry can't be both
	// writable and vertex-visible. Other fragment buffers keep read_write (writes must work).
	if (p_stage == SHADER_STAGE_VERTEX || p_stage == SHADER_STAGE_FRAGMENT) {
		char *p = cs.ptrw();
		while ((p = strstr(p, "@group(")) != nullptr) {
			unsigned int grp = 0, bnd = 0;
			if (parse_group_binding(p, grp, bnd) &&
					(p_stage == SHADER_STAGE_VERTEX || p_vertex_storage_buffers.has((grp << 16) | bnd))) {
				for (char *fwd = p; *fwd && *fwd != ';'; fwd++) {
					if (strncmp(fwd, "var<storage, read_write>", 24) == 0) {
						memcpy(fwd, "var<storage, read>      ", 24); // Same length.
						break;
					}
				}
			}
			p++;
		}
	}

	// A sampled snapshot cannot preserve reads after writes within one dispatch.
	const char *storage = cs.get_data();
	while ((storage = strstr(storage, "texture_storage_")) != nullptr) {
		const char *format = strchr(storage, '<');
		const char *comma = format ? strchr(format, ',') : nullptr;
		if (!comma) {
			break;
		}
		const char *access = comma + 1;
		while (*access == ' ' || *access == '\t') {
			access++;
		}
		if (strncmp(access, "read_write>", 11) == 0 &&
				!wgpu_storage_format_supports_read_write(wgpu_storage_format_from_wgsl(format + 1), has_rw_storage_textures, has_texture_formats_tier2)) {
			r_rewrites.error = "WebGPU: read/write storage texture format is unsupported on this adapter. Use supported R32 formats or separate read and write textures.";
			return CharString();
		}
		storage = comma + 1;
	}

	// Convert read-only storage textures to sampled textures. They then count against the sampled-texture budget, not
	// the 8 storage textures per stage (SDFGI reads 12), and Firefox/wgpu needs no readonly-and-readwrite feature.
	if (strstr(cs.get_data(), ", read>") || strstr(cs.get_data(), ",read>")) {
		struct ReadOnlyInfo {
			uint32_t grp, bnd;
			String var_name, dim_type, fmt;
		};
		LocalVector<ReadOnlyInfo> ro_infos;

		// Scan for texture_storage_*<fmt, read> declarations.
		{
			const char *p = cs.get_data();
			while ((p = strstr(p, "@group(")) != nullptr) {
				unsigned int grp = 0, bnd = 0;
				if (!parse_group_binding(p, grp, bnd)) {
					p++;
					continue;
				}
				const char *semi = strchr(p, ';');
				if (!semi) {
					p++;
					continue;
				}
				// Look for ", read>" or ",read>" but NOT "read_write>"
				const char *rd = strstr(p, ",read>");
				if (!rd || rd >= semi) {
					rd = strstr(p, ", read>");
				}
				if (!rd || rd >= semi) {
					p = semi;
					continue;
				}
				// Make sure it's not read_write (already handled above).
				if (rd > cs.get_data() && *(rd - 1) == '_') {
					p = semi;
					continue;
				} // part of "read_write"
				// Check for "read_write>" pattern — skip if found.
				const char *rw_check = strstr(p, "read_write>");
				if (rw_check && rw_check < semi && rw_check < rd) {
					p = semi;
					continue;
				}

				const char *ts = strstr(p, "texture_storage_");
				if (!ts || ts >= semi) {
					p = semi;
					continue;
				}
				// Variable name: "var NAME:"
				const char *vp = strstr(p, "var ");
				if (!vp || vp > ts) {
					p = semi;
					continue;
				}
				vp += 4;
				const char *colon = strchr(vp, ':');
				if (!colon || colon > ts) {
					p = semi;
					continue;
				}
				const char *ne = colon;
				while (ne > vp && (*(ne - 1) == ' ' || *(ne - 1) == '\t')) {
					ne--;
				}
				if (ne <= vp) {
					p = semi;
					continue;
				}
				// Dim type: texture_storage_Xd
				const char *lt = strchr(ts, '<');
				if (!lt || lt > rd) {
					p = semi;
					continue;
				}
				// Format: between < and , before read
				// Note: rd points to ",read>" (includes comma) so comma == rd is expected.
				// Use > not >= so the format-access separator comma is correctly accepted.
				const char *comma = strchr(lt + 1, ',');
				if (!comma || comma > rd) {
					p = semi;
					continue;
				}
				int fmt_len = (int)(comma - lt - 1);
				if (fmt_len <= 0 || fmt_len >= 64) {
					p = semi;
					continue;
				}

				ReadOnlyInfo info;
				info.grp = grp;
				info.bnd = bnd;
				{
					char buf[256];
					int len = (int)(ne - vp);
					if (len > 255) {
						len = 255;
					}
					memcpy(buf, vp, len);
					buf[len] = '\0';
					info.var_name = buf;
				}
				{
					char buf[256];
					int len = (int)(lt - ts);
					if (len > 255) {
						len = 255;
					}
					memcpy(buf, ts, len);
					buf[len] = '\0';
					info.dim_type = buf;
				}
				{
					char buf[64];
					if (fmt_len > 63) {
						fmt_len = 63;
					}
					memcpy(buf, lt + 1, fmt_len);
					buf[fmt_len] = '\0';
					info.fmt = String(buf).strip_edges();
				}
				ro_infos.push_back(info);
				p = semi;
			}
		}

		if (ro_infos.size() > 0) {
			String ws(cs.get_data());
			for (const ReadOnlyInfo &info : ro_infos) {
				// Map texture_storage_Xd → texture_Xd and determine component type.
				String sampled_dim = info.dim_type.replace("texture_storage_", "texture_");
				String comp_type = "f32";
				if (info.fmt.find("uint") != -1) {
					comp_type = "u32";
				} else if (info.fmt.find("sint") != -1) {
					comp_type = "i32";
				}

				// Replace the full declaration:
				//   texture_storage_Xd<fmt, read> → texture_Xd<comp_type>
				// Try both spacing variants.
				String old_decl = info.dim_type + "<" + info.fmt + ", read>";
				String new_decl = sampled_dim + "<" + comp_type + ">";
				ws = ws.replace(old_decl, new_decl);
				old_decl = info.dim_type + "<" + info.fmt + ",read>";
				ws = ws.replace(old_decl, new_decl);

				// Add mip level parameter to textureLoad calls for this variable.
				// textureLoad(var, coords) → textureLoad(var, coords, 0)
				{
					String search = "textureLoad(" + info.var_name;
					int64_t pos = 0;
					while ((pos = ws.find(search, pos)) != -1) {
						// Skip other variables sharing the prefix, such as v_1 for v.
						const char32_t next = pos + search.length() < ws.length() ? ws[pos + search.length()] : U' ';
						if (is_ascii_identifier_char(next)) {
							pos += search.length();
							continue;
						}
						int64_t paren_start = pos + String("textureLoad").length();
						int depth = 0;
						int64_t insert_mip = -1;
						for (int64_t k = paren_start; k < ws.length(); k++) {
							if (ws[k] == '(') {
								depth++;
							} else if (ws[k] == ')') {
								depth--;
								if (depth == 0) {
									insert_mip = k;
									break;
								}
							}
						}
						if (insert_mip != -1) {
							ws = ws.substr(0, insert_mip) + ", 0" + ws.substr(insert_mip);
							pos = insert_mip + 3;
						} else {
							pos += search.length();
						}
					}
				}

				// Record the conversion for BGL creation and populate the
				// format map so the BGL entry gets the correct sample type.
				// The WGSL declaration is about to be removed, so the later
				// format-scanning pass won't find it — we must record it now.
				uint32_t key = ((uint32_t)info.grp << 16) | info.bnd;
				r_rewrites.read_storage_to_sampled.insert(key);
				{
					WGPUTextureFormat tf = WGPUTextureFormat_RGBA8Unorm; // fallback
					if (info.fmt == "rgba8unorm") {
						tf = WGPUTextureFormat_RGBA8Unorm;
					} else if (info.fmt == "rgba8snorm") {
						tf = WGPUTextureFormat_RGBA8Snorm;
					} else if (info.fmt == "rgba8uint") {
						tf = WGPUTextureFormat_RGBA8Uint;
					} else if (info.fmt == "rgba8sint") {
						tf = WGPUTextureFormat_RGBA8Sint;
					} else if (info.fmt == "rgba16float") {
						tf = WGPUTextureFormat_RGBA16Float;
					} else if (info.fmt == "rgba16uint") {
						tf = WGPUTextureFormat_RGBA16Uint;
					} else if (info.fmt == "rgba16sint") {
						tf = WGPUTextureFormat_RGBA16Sint;
					} else if (info.fmt == "rgba32float") {
						tf = WGPUTextureFormat_RGBA32Float;
					} else if (info.fmt == "rgba32uint") {
						tf = WGPUTextureFormat_RGBA32Uint;
					} else if (info.fmt == "rgba32sint") {
						tf = WGPUTextureFormat_RGBA32Sint;
					} else if (info.fmt == "rg32float") {
						tf = WGPUTextureFormat_RG32Float;
					} else if (info.fmt == "rg32uint") {
						tf = WGPUTextureFormat_RG32Uint;
					} else if (info.fmt == "rg32sint") {
						tf = WGPUTextureFormat_RG32Sint;
					} else if (info.fmt == "r32float") {
						tf = WGPUTextureFormat_R32Float;
					} else if (info.fmt == "r32uint") {
						tf = WGPUTextureFormat_R32Uint;
					} else if (info.fmt == "r32sint") {
						tf = WGPUTextureFormat_R32Sint;
					} else if (info.fmt == "r8unorm") {
						tf = WGPUTextureFormat_R8Unorm;
					} else if (info.fmt == "r8snorm") {
						tf = WGPUTextureFormat_R8Snorm;
					} else if (info.fmt == "r8uint") {
						tf = WGPUTextureFormat_R8Uint;
					} else if (info.fmt == "r8sint") {
						tf = WGPUTextureFormat_R8Sint;
					} else if (info.fmt == "rg8unorm") {
						tf = WGPUTextureFormat_RG8Unorm;
					} else if (info.fmt == "rg8snorm") {
						tf = WGPUTextureFormat_RG8Snorm;
					} else if (info.fmt == "rg8uint") {
						tf = WGPUTextureFormat_RG8Uint;
					} else if (info.fmt == "rg8sint") {
						tf = WGPUTextureFormat_RG8Sint;
					} else if (info.fmt == "bgra8unorm") {
						tf = WGPUTextureFormat_BGRA8Unorm;
					}
					r_rewrites.read_storage_formats[key] = tf;
				}
			}
			WEBGPU_DIAG({ console.log('[READ-CONVERT] Converted ' + $0 + ' read-only storage texture(s) to sampled'); }, (int)ro_infos.size());
			print_verbose(vformat("WebGPU: bound %d read-only storage texture(s) as sampled.", (int)ro_infos.size()));

			cs = ws.utf8();
		}
	}

	return cs;
}

static bool _is_handle_uniform(RDD::UniformType p_type) {
	return p_type == RDD::UNIFORM_TYPE_SAMPLER || p_type == RDD::UNIFORM_TYPE_TEXTURE ||
			p_type == RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE || p_type == RDD::UNIFORM_TYPE_IMAGE ||
			p_type == RDD::UNIFORM_TYPE_INPUT_ATTACHMENT;
}

RDD::ShaderID RenderingDeviceDriverWebGPU::shader_create_from_container(const Ref<RenderingShaderContainer> &p_shader_container, const Vector<ImmutableSampler> &p_immutable_samplers) {
	ERR_FAIL_COND_V(p_shader_container.is_null(), ShaderID());

	Ref<RenderingShaderContainerWebGPU> wg_container = p_shader_container;
	ERR_FAIL_COND_V(wg_container.is_null(), ShaderID());

	RenderingDeviceCommons::ShaderReflection shader_refl = p_shader_container->get_shader_reflection();

	WGShader *shader = memnew(WGShader);
	shader->uid = shader_uid_counter.increment();
	shader->name = String(p_shader_container->shader_name.ptr());
	shader->push_constant_bind_group = wg_container->get_push_constant_bind_group();
	shader->push_constant_binding = wg_container->get_push_constant_binding();
	shader->push_constant_size = shader_refl.push_constant_size;

	if (!RenderingShaderContainerWebGPU::expand_bindings(shader_refl, shader->binding_remaps)) {
		memdelete(shader);
		return ShaderID();
	}

	print_verbose(vformat("WebGPU: shader_create_from_container '%s' (%d stages, push_const_size=%d)", shader->name, (int)p_shader_container->shaders.size(), (int)shader_refl.push_constant_size));

	String error_text;

	// Maps (set_index << 16 | binding) to the WGSL texture view dimension detected from Tint output.
	// Used so SAMPLER_WITH_TEXTURE and TEXTURE BGL entries get the correct viewDimension.
	HashMap<uint32_t, WGPUTextureViewDimension> wgsl_tex_dims;

	// Maps (set_index << 16 | binding) → true if the WGSL storage buffer is read-only (var<storage, read>),
	// false if read-write (var<storage, read_write>). Used to correctly set BGL buffer type for SSBOs.
	HashMap<uint32_t, bool> wgsl_ssbo_readonly;

	// Maps (set_index << 16 | binding) → storage texture access mode detected from Tint WGSL output.
	// Tint emits: texture_storage_2d<format, write/read/read_write>. Used for IMAGE BGL entries.
	HashMap<uint32_t, WGPUStorageTextureAccess> wgsl_storage_tex_access;

	// Maps (set_index << 16 | binding) → storage texture format from Tint metadata. Used for IMAGE BGL entries.
	HashMap<uint32_t, WGPUTextureFormat> wgsl_storage_tex_format;

	// Maps (set_index << 16 | binding) → true if Tint output has var<uniform> (e.g. for texture buffers).
	HashMap<uint32_t, bool> wgsl_is_uniform;

	// Maps (set_index << 16 | binding) → true if the WGSL has texture_depth_* at this binding.
	HashMap<uint32_t, bool> wgsl_is_depth_texture;

	// Maps (set_index << 16 | binding) → Uint/Sint for texture_*<u32> / texture_*<i32> (usampler/isampler).
	HashMap<uint32_t, WGPUTextureSampleType> wgsl_tex_sample_type;
	HashSet<uint32_t> wgsl_filtered_textures; // Float textures a stage samples; they can't bind as unfilterable.

	// Maps (set_index << 16 | binding) → true if the WGSL has sampler_comparison at this binding.
	HashMap<uint32_t, bool> wgsl_is_comparison_sampler;

	// Maps (set_index << 16 | binding) → true if the WGSL has texture_multisampled_*
	// or texture_depth_multisampled_* at this binding. Used so BGL entries set
	// texture.multisampled=true to match sampler2DMS (GLSL) bindings.
	HashMap<uint32_t, bool> wgsl_is_multisampled_texture;

	// Depth alias bindings: Tint splits mixed-usage depth textures into two globals
	// (one Depth at binding B, one Float alias at binding B+1). Track (set,B+1) pairs
	// so we can add extra BGL and bind group entries.
	// Maps (set_index << 16 | alias_binding) → depth_binding (the adjacent depth texture).
	HashMap<uint32_t, uint32_t> wgsl_depth_alias_bindings;

	// Maps (set_index << 16 | binding) → WGPUShaderStage bitmask of stages that actually
	// declare this storage buffer binding in their WGSL.  Used to set per-stage visibility
	// so fragment-only storage buffers don't consume vertex buffer slots on Metal.
	// Firefox/wgpu enforces Metal's limit of 8 storage buffers per shader stage.
	HashMap<uint32_t, uint32_t> wgsl_buffer_stages;
	HashMap<uint32_t, uint32_t> wgsl_resource_stages;
	bool has_resource_usage = false;

	// Read-write storage texture splits: maps (set << 16 | write_binding) → shadow_read_binding.
	// Populated when readonly-and-readwrite-storage-textures is unavailable.

	// Read-only storage texture → sampled texture conversions.
	// Set of (set << 16 | binding) keys for read-only storage textures converted to texture_2d.
	HashSet<uint32_t> wgsl_read_storage_to_sampled;

	// --- Create one WGPUShaderModule per stage ---
	bool detected_override_declarations = false;
	Vector<RenderingShaderContainer::Shader> &stage_shaders = p_shader_container->shaders;
	// Translate every stage first: fragment storage rewrites depend on what the vertex stage declares.
	LocalVector<CharString> stage_wgsl;
	stage_wgsl.resize(stage_shaders.size());
	for (int i = 0; i < stage_shaders.size(); i++) {
		const RenderingShaderContainer::Shader &s = stage_shaders[i];

		Vector<uint8_t> spv_bytes;
		if (!wg_container->get_stage_code(i, spv_bytes, stage_wgsl[i])) {
			error_text = "WebGPU: failed to load baked shader stage.";
			break;
		}
		shader->stage_spirv[(int)s.shader_stage] = spv_bytes;
		if (s.shader_stage == SHADER_STAGE_VERTEX) {
			if (!_collect_storage_buffer_bindings(stage_wgsl[i].get_data(), shader->vertex_storage_buffers)) {
				error_text = "WebGPU: invalid vertex resource metadata.";
				break;
			}
		}
	}

	for (int i = 0; error_text.is_empty() && i < stage_shaders.size(); i++) {
		const RenderingShaderContainer::Shader &s = stage_shaders[i];

		WGSLRewrites rewrites;
		const CharString wgsl_cs = _postprocess_wgsl(stage_wgsl[i].get_data(), s.shader_stage, shader->vertex_storage_buffers, rewrites);
		const char *wgsl_str = wgsl_cs.get_data();
		if (!rewrites.error.is_empty()) {
			error_text = rewrites.error;
			break;
		}
		for (const uint32_t key : rewrites.read_storage_to_sampled) {
			wgsl_read_storage_to_sampled.insert(key);
		}
		for (const KeyValue<uint32_t, WGPUTextureFormat> &kv : rewrites.read_storage_formats) {
			wgsl_storage_tex_format[kv.key] = kv.value;
		}

		webgpu_metadata::Shader reflection_info;
		if (!webgpu_metadata::parse(wgsl_str, reflection_info)) {
			error_text = "WebGPU: invalid shader resource metadata.";
			break;
		}
		detected_override_declarations |= reflection_info.has_overrides;
		for (uint32_t id : reflection_info.overrides) {
			shader->stage_override_ids[s.shader_stage].insert(id);
		}
		has_resource_usage = true;
		const uint32_t stage = s.shader_stage == SHADER_STAGE_VERTEX ? WGPUShaderStage_Vertex : (s.shader_stage == SHADER_STAGE_FRAGMENT ? WGPUShaderStage_Fragment : WGPUShaderStage_Compute);
		static const WGPUTextureViewDimension dimensions[] = {
			WGPUTextureViewDimension_Undefined, WGPUTextureViewDimension_1D, WGPUTextureViewDimension_2D,
			WGPUTextureViewDimension_2DArray, WGPUTextureViewDimension_3D, WGPUTextureViewDimension_Cube, WGPUTextureViewDimension_CubeArray
		};
		for (const auto &resource : reflection_info.resources) {
			const uint32_t key = (resource.group << 16) | resource.binding;
			if (resource.used) {
				wgsl_resource_stages[key] |= stage;
			}
			if (resource.dimension != webgpu_metadata::NONE) {
				wgsl_tex_dims[key] = dimensions[resource.dimension];
				if (resource.sample_type == webgpu_metadata::UNFILTERABLE_FLOAT) {
					if (!wgsl_filtered_textures.has(key)) {
						wgsl_tex_sample_type[key] = WGPUTextureSampleType_UnfilterableFloat;
					}
				} else if (resource.sample_type == webgpu_metadata::FLOAT) {
					if (resource.used) {
						wgsl_filtered_textures.insert(key);
						const WGPUTextureSampleType *type = wgsl_tex_sample_type.getptr(key);
						if (type && *type == WGPUTextureSampleType_UnfilterableFloat) {
							wgsl_tex_sample_type.erase(key);
						}
					}
				} else {
					wgsl_tex_sample_type[key] = resource.sample_type == webgpu_metadata::UINT ? WGPUTextureSampleType_Uint : WGPUTextureSampleType_Sint;
				}
			}
			if (resource.depth_alias != UINT32_MAX) {
				wgsl_depth_alias_bindings[key] = resource.depth_alias;
			}
			switch (resource.kind) {
				case webgpu_metadata::UNIFORM_BUFFER:
					wgsl_is_uniform[key] = true;
					break;
				case webgpu_metadata::STORAGE_BUFFER:
				case webgpu_metadata::READ_ONLY_STORAGE_BUFFER: {
					const bool readonly = resource.kind == webgpu_metadata::READ_ONLY_STORAGE_BUFFER || s.shader_stage == SHADER_STAGE_VERTEX ||
							(s.shader_stage == SHADER_STAGE_FRAGMENT && shader->vertex_storage_buffers.has(key));
					bool *previous = wgsl_ssbo_readonly.getptr(key);
					wgsl_ssbo_readonly[key] = previous ? (*previous && readonly) : readonly;
					if (resource.used) {
						wgsl_buffer_stages[key] |= stage;
					}
				} break;
				case webgpu_metadata::COMPARISON_SAMPLER:
					wgsl_is_comparison_sampler[key] = true;
					break;
				case webgpu_metadata::DEPTH_MULTISAMPLED_TEXTURE:
					wgsl_is_multisampled_texture[key] = true;
					[[fallthrough]];
				case webgpu_metadata::DEPTH_TEXTURE:
					wgsl_is_depth_texture[key] = true;
					break;
				case webgpu_metadata::MULTISAMPLED_TEXTURE:
					wgsl_is_multisampled_texture[key] = true;
					break;
				case webgpu_metadata::WRITE_ONLY_STORAGE_TEXTURE:
				case webgpu_metadata::READ_ONLY_STORAGE_TEXTURE:
				case webgpu_metadata::READ_WRITE_STORAGE_TEXTURE: {
					const std::string format = resource.format + ",";
					wgsl_storage_tex_format[key] = wgpu_storage_format_from_wgsl(format.c_str());
					wgsl_storage_tex_access[key] = resource.kind == webgpu_metadata::WRITE_ONLY_STORAGE_TEXTURE ? WGPUStorageTextureAccess_WriteOnly : (resource.kind == webgpu_metadata::READ_ONLY_STORAGE_TEXTURE ? WGPUStorageTextureAccess_ReadOnly : WGPUStorageTextureAccess_ReadWrite);
				} break;
				default:
					break;
			}
		}

		if (s.shader_stage < 6) {
			shader->stage_wgsl[s.shader_stage] = wgsl_cs;
		}
	}

	shader->has_override_declarations = detected_override_declarations;
	if (detected_override_declarations) {
		print_verbose(vformat("WebGPU: shader '%s' has override declarations — will use pipeline constants for specialization.", shader->name));
	}

	if (!error_text.is_empty()) {
		goto cleanup;
	}

	{ // Block scope: variables here (LocalVector, String) have non-trivial
	  // destructors and must not be jumped over by goto.

		// --- Build WGPUBindGroupLayout for each descriptor set ---
		const uint32_t set_count = (uint32_t)shader_refl.uniform_sets.size();
		shader->bind_group_infos.resize(set_count);
		shader->bind_group_layouts.resize(set_count);

		for (uint32_t set = 0; set < set_count; set++) {
			const Vector<RenderingDeviceCommons::ShaderUniform> &set_uniforms = shader_refl.uniform_sets[set];

			// Count entries — combined sampler+texture expands to 2 entries each.
			uint32_t entry_count = 0;
			for (int i = 0; i < set_uniforms.size(); i++) {
				if (set_uniforms[i].type == RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE) {
					entry_count += 2; // Sampler + texture.
				} else {
					entry_count += 1;
				}
			}

			LocalVector<WGPUBindGroupLayoutEntry> entries;
			entries.resize(entry_count);

			WGShader::BindGroupInfo &bgi = shader->bind_group_infos[set];
			bgi.entries.resize(set_uniforms.size());

			uint32_t e_idx = 0;
			for (int u_idx = 0; u_idx < set_uniforms.size(); u_idx++) {
				const RenderingDeviceCommons::ShaderUniform &u = set_uniforms[u_idx];
				WGShader::BindGroupEntry &bge = bgi.entries[u_idx];
				WGPUShaderStage vis = _stages_to_wgpu_visibility((uint32_t)u.stages);

				bge.godot_type = u.type;

				switch (u.type) {
					case RDD::UNIFORM_TYPE_SAMPLER: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						// Descriptor arrays were expanded into scalar uniforms before translation.
						{
							uint32_t k = ((uint32_t)set << 16) | (u.binding * 2);
							entry.sampler.type = (wgsl_is_comparison_sampler.has(k) && wgsl_is_comparison_sampler[k])
									? WGPUSamplerBindingType_Comparison
									: WGPUSamplerBindingType_Filtering;
						}
						bge.layout_entry = entry;
						bge.array_length = 1;
					} break;

					case RDD::UNIFORM_TYPE_TEXTURE:
					case RDD::UNIFORM_TYPE_INPUT_ATTACHMENT: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						// Descriptor arrays were expanded into scalar uniforms before translation.
						{
							uint32_t k = ((uint32_t)set << 16) | (u.binding * 2);
							bool is_ms = wgsl_is_multisampled_texture.has(k) && wgsl_is_multisampled_texture[k];
							bool is_depth = wgsl_is_depth_texture.has(k) && wgsl_is_depth_texture[k];
							const WGPUTextureSampleType *wgsl_type = wgsl_tex_sample_type.getptr(k);
							entry.texture.sampleType = _bgl_texture_sample_type(is_depth, is_ms, wgsl_type ? *wgsl_type : WGPUTextureSampleType_Float);
							entry.texture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;
							entry.texture.multisampled = is_ms;
						}
						bge.layout_entry = entry;
						bge.array_length = 1;
					} break;

					case RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE: {
						// Combined sampler+texture split by our SPIR-V preprocessor:
						// Sampler at binding*2+0, texture at binding*2+1 in the modified SPIR-V → matches Tint WGSL output.
						const uint32_t ms_key = ((uint32_t)set << 16) | (u.binding * 2);
						if (wgsl_tex_dims.has(ms_key)) {
							// Multisampled textures can't be sampled, so the preprocessor keeps only the texture, at binding*2.
							WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
							entry = {};
							entry.binding = u.binding * 2;
							entry.visibility = vis;
							const bool is_depth = wgsl_is_depth_texture.has(ms_key) && wgsl_is_depth_texture[ms_key];
							const WGPUTextureSampleType *wgsl_type = wgsl_tex_sample_type.getptr(ms_key);
							entry.texture.sampleType = _bgl_texture_sample_type(is_depth, true, wgsl_type ? *wgsl_type : WGPUTextureSampleType_Float);
							entry.texture.viewDimension = wgsl_tex_dims[ms_key];
							entry.texture.multisampled = true;
							bge.layout_entry = entry;
							break;
						}
						WGPUBindGroupLayoutEntry &samp_entry = entries[e_idx++];
						samp_entry = {};
						samp_entry.binding = u.binding * 2 + 0;
						samp_entry.visibility = vis;
						{
							uint32_t k = ((uint32_t)set << 16) | (u.binding * 2 + 0);
							samp_entry.sampler.type = (wgsl_is_comparison_sampler.has(k) && wgsl_is_comparison_sampler[k])
									? WGPUSamplerBindingType_Comparison
									: WGPUSamplerBindingType_Filtering;
						}

						WGPUBindGroupLayoutEntry &tex_entry = entries[e_idx++];
						tex_entry = {};
						tex_entry.binding = u.binding * 2 + 1;
						tex_entry.visibility = vis;
						{
							uint32_t k = ((uint32_t)set << 16) | (u.binding * 2 + 1);
							bool is_ms = wgsl_is_multisampled_texture.has(k) && wgsl_is_multisampled_texture[k];
							bool is_depth = wgsl_is_depth_texture.has(k) && wgsl_is_depth_texture[k];
							const WGPUTextureSampleType *wgsl_type = wgsl_tex_sample_type.getptr(k);
							tex_entry.texture.sampleType = _bgl_texture_sample_type(is_depth, is_ms, wgsl_type ? *wgsl_type : WGPUTextureSampleType_Float);
							tex_entry.texture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;
							tex_entry.texture.multisampled = is_ms;
							// MSAA texture bindings with UnfilterableFloat require a NonFiltering sampler —
							// override the sampler for this combined binding.
							if (is_ms && !is_depth) {
								samp_entry.sampler.type = WGPUSamplerBindingType_NonFiltering;
							}
						}

						bge.layout_entry = tex_entry; // Store texture entry as the primary.
					} break;

					case RDD::UNIFORM_TYPE_IMAGE: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						uint32_t k = ((uint32_t)set << 16) | (u.binding * 2);
						if (wgsl_read_storage_to_sampled.has(k)) {
							// Read-only storage texture was converted to sampled texture_2d.
							// Create a sampled texture BGL entry instead of storage.
							WGPUTextureFormat fmt = wgsl_storage_tex_format.has(k) ? wgsl_storage_tex_format[k] : WGPUTextureFormat_RGBA8Unorm;
							entry.texture.sampleType = _texture_sample_type_for_format(fmt);
							entry.texture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;
							entry.texture.multisampled = false;
						} else {
							WGPUTextureFormat fmt = wgsl_storage_tex_format.has(k) ? wgsl_storage_tex_format[k] : WGPUTextureFormat_RGBA8Unorm;
							WGPUStorageTextureAccess access = wgsl_storage_tex_access.has(k)
									? wgsl_storage_tex_access[k]
									: (u.writable ? WGPUStorageTextureAccess_WriteOnly : WGPUStorageTextureAccess_ReadOnly);
							entry.storageTexture.access = access;
							entry.storageTexture.format = fmt;
							entry.storageTexture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;
						}
						bge.layout_entry = entry;
					} break;

					case RDD::UNIFORM_TYPE_UNIFORM_BUFFER: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						entry.buffer.type = WGPUBufferBindingType_Uniform;
						entry.buffer.hasDynamicOffset = false;
						entry.buffer.minBindingSize = 0;
						bge.layout_entry = entry;
					} break;

					case RDD::UNIFORM_TYPE_STORAGE_BUFFER: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						uint32_t k = ((uint32_t)set << 16) | (u.binding * 2);
						// Check if WGSL scan shows this is actually a storage texture.
						if (wgsl_storage_tex_format.has(k)) {
							WGPUTextureFormat fmt = wgsl_storage_tex_format[k];
							WGPUStorageTextureAccess access = wgsl_storage_tex_access.has(k)
									? wgsl_storage_tex_access[k]
									: (u.writable ? WGPUStorageTextureAccess_WriteOnly : WGPUStorageTextureAccess_ReadOnly);
							entry.storageTexture.access = access;
							entry.storageTexture.format = fmt;
							entry.storageTexture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;

						} else if (wgsl_is_uniform.has(k) && wgsl_is_uniform[k]) {
							// Tint emitted var<uniform> for this binding.
							entry.buffer.type = WGPUBufferBindingType_Uniform;
						} else {
							bool is_readonly = wgsl_ssbo_readonly.has(k) ? wgsl_ssbo_readonly[k] : !u.writable;
							entry.buffer.type = is_readonly ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Storage;
							// Use per-stage visibility from Tint metadata for storage buffers.
							// Firefox/wgpu enforces Metal's limit of 8 storage buffers per shader stage.
							if (wgsl_buffer_stages.has(k)) {
								entry.visibility = (WGPUShaderStage)wgsl_buffer_stages[k];
							} else if (!wgsl_buffer_stages.is_empty()) {
								// Buffer is declared in SPIR-V but not used by any entry point.
								// Set visibility to None — doesn't count against any stage's limit.
								entry.visibility = (WGPUShaderStage)0;
							}
							// Writable storage can't be vertex-visible; _postprocess_wgsl made every
							// buffer the vertex stage declares read-only.
							if (!is_readonly) {
								entry.visibility = (WGPUShaderStage)(entry.visibility & ~WGPUShaderStage_Vertex);
							}
						}
						bge.layout_entry = entry;
					} break;

					// Task 7.5: Dynamic variants — hasDynamicOffset=true allows the
					// bind group to be set with per-frame offsets via wgpuRenderPassEncoderSetBindGroup.
					case RDD::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						entry.buffer.type = WGPUBufferBindingType_Uniform;
						entry.buffer.hasDynamicOffset = true;
						entry.buffer.minBindingSize = 0;
						bge.layout_entry = entry;
					} break;

					case RDD::UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						bool is_storage_tex = false;
						{
							uint32_t k = ((uint32_t)set << 16) | (u.binding * 2);
							if (wgsl_storage_tex_format.has(k)) {
								is_storage_tex = true;
								WGPUTextureFormat fmt = wgsl_storage_tex_format[k];
								WGPUStorageTextureAccess access = wgsl_storage_tex_access.has(k)
										? wgsl_storage_tex_access[k]
										: (u.writable ? WGPUStorageTextureAccess_WriteOnly : WGPUStorageTextureAccess_ReadOnly);
								entry.storageTexture.access = access;
								entry.storageTexture.format = fmt;
								entry.storageTexture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;
							} else if (wgsl_is_uniform.has(k) && wgsl_is_uniform[k]) {
								entry.buffer.type = WGPUBufferBindingType_Uniform;
							} else {
								bool is_readonly = wgsl_ssbo_readonly.has(k) ? wgsl_ssbo_readonly[k] : !u.writable;
								entry.buffer.type = is_readonly ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Storage;
								// Use per-stage visibility from Tint metadata (see UNIFORM_TYPE_STORAGE_BUFFER above).
								if (wgsl_buffer_stages.has(k)) {
									entry.visibility = (WGPUShaderStage)wgsl_buffer_stages[k];
								} else if (!wgsl_buffer_stages.is_empty()) {
									entry.visibility = (WGPUShaderStage)0;
								}
								if (!is_readonly) {
									entry.visibility = (WGPUShaderStage)(entry.visibility & ~WGPUShaderStage_Vertex);
								}
							}
						}
						// Only buffer bindings support dynamic offsets; storage textures must not set it.
						entry.buffer.hasDynamicOffset = !is_storage_tex;
						entry.buffer.minBindingSize = 0;
						bge.layout_entry = entry;
					} break;

					// WebGPU has no texel buffers (TBOs); emulate as uniform or storage buffers based on WGSL.
					case RDD::UNIFORM_TYPE_TEXTURE_BUFFER:
					case RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						// Tint may convert TBOs to var<uniform> or var<storage,read> depending on usage.
						{
							uint32_t k = ((uint32_t)set << 16) | (u.binding * 2);
							if (wgsl_is_uniform.has(k) && wgsl_is_uniform[k]) {
								entry.buffer.type = WGPUBufferBindingType_Uniform;
							} else if (wgsl_ssbo_readonly.has(k)) {
								entry.buffer.type = wgsl_ssbo_readonly[k] ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Storage;
							} else {
								entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage; // default fallback
							}
						}
						entry.buffer.hasDynamicOffset = false;
						entry.buffer.minBindingSize = 0;
						bge.layout_entry = entry;
					} break;

					case RDD::UNIFORM_TYPE_IMAGE_BUFFER: {
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						uint32_t k = ((uint32_t)set << 16) | (u.binding * 2);
						// Tint may convert image buffers to texture_storage_*.
						if (wgsl_storage_tex_format.has(k)) {
							WGPUTextureFormat fmt = wgsl_storage_tex_format[k];
							WGPUStorageTextureAccess access = wgsl_storage_tex_access.has(k)
									? wgsl_storage_tex_access[k]
									: (u.writable ? WGPUStorageTextureAccess_WriteOnly : WGPUStorageTextureAccess_ReadOnly);
							entry.storageTexture.access = access;
							entry.storageTexture.format = fmt;
							entry.storageTexture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;
						} else {
							entry.buffer.type = WGPUBufferBindingType_Storage;
							entry.buffer.hasDynamicOffset = false;
							entry.buffer.minBindingSize = 0;
						}
						bge.layout_entry = entry;
					} break;

					default:
						WARN_PRINT_ONCE(vformat("WebGPU: unhandled uniform type %d in bind group layout.", (int)u.type));
						WGPUBindGroupLayoutEntry &entry = entries[e_idx++];
						entry = {};
						entry.binding = u.binding * 2; // Preprocessing doubles all non-combined bindings.
						entry.visibility = vis;
						entry.buffer.type = WGPUBufferBindingType_Uniform;
						bge.layout_entry = entry;
						break;
				}
			}
			entries.resize(e_idx); // Multisampled combined samplers use one entry, not two.

			// Add BGL entries for depth alias variables.
			// Tint splits mixed-usage depth textures: original→Depth at binding B,
			// clone→Float at binding B+1 (named "*_depth_alias").
			for (const KeyValue<uint32_t, uint32_t> &kv : wgsl_depth_alias_bindings) {
				uint32_t alias_key = kv.key;
				uint32_t alias_grp = alias_key >> 16;
				uint32_t alias_bnd = alias_key & 0xFFFF;
				if (alias_grp != set) {
					continue;
				}
				WGPUBindGroupLayoutEntry alias_entry = {};
				alias_entry.binding = alias_bnd;
				alias_entry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
				alias_entry.texture.sampleType = WGPUTextureSampleType_Float;
				alias_entry.texture.viewDimension = wgsl_tex_dims.has(alias_key) ? wgsl_tex_dims[alias_key] : WGPUTextureViewDimension_2D;
				alias_entry.texture.multisampled = false;
				entries.push_back(alias_entry);
			}

			// Unused samplers must not count against WebGPU's per-stage limits.
			// Tint reflection includes resources reached through helper functions.
			if (has_resource_usage) {
				for (auto &entry : entries) {
					uint32_t key = (set << 16) | entry.binding;
					if (wgsl_depth_alias_bindings.has(key)) {
						key = (set << 16) | wgsl_depth_alias_bindings[key];
					}
					entry.visibility = (WGPUShaderStage)(wgsl_resource_stages.has(key) ? wgsl_resource_stages[key] : 0);
				}
				for (auto &binding : bgi.entries) {
					for (const auto &entry : entries) {
						if (entry.binding == binding.layout_entry.binding) {
							binding.layout_entry.visibility = entry.visibility;
							break;
						}
					}
				}
			}

			// Log any duplicate binding indices for debugging.
			for (uint32_t a = 0; a < entries.size(); a++) {
				for (uint32_t b = a + 1; b < entries.size(); b++) {
					if (entries[a].binding == entries[b].binding) {
						WEBGPU_DIAG({ console.error('[BGL-DUP] set=' + $0 + ' binding=' + $1 + ' idx_a=' + $2 + ' idx_b=' + $3); }, (int)set, (int)entries[a].binding, (int)a, (int)b);
					}
				}
			}

			WGPUBindGroupLayoutDescriptor layout_desc = {};
			layout_desc.entryCount = entries.size();
			layout_desc.entries = entries.size() > 0 ? entries.ptr() : nullptr;

			// Label the BGL so JS-side patch logs identify it.
			String _bgl_label = "bgl:" + shader->name + ":set" + itos((int)set);
			CharString _bgl_label_cs = _bgl_label.utf8();
			layout_desc.label = { _bgl_label_cs.get_data(), WGPU_STRLEN };

			shader->bind_group_layouts[set] = wgpuDeviceCreateBindGroupLayout(device, &layout_desc);
			if (shader->bind_group_layouts[set] == nullptr) {
				error_text = "WebGPU: wgpuDeviceCreateBindGroupLayout failed.";
				goto cleanup;
			}
		}

		// Store depth alias bindings on the shader for use during uniform_set_create.
		shader->depth_alias_bindings = wgsl_depth_alias_bindings;

		shader->read_storage_to_sampled = wgsl_read_storage_to_sampled;

		// --- Build WGPUPipelineLayout ---
		// The number of bind groups in the pipeline layout must cover:
		//   - All descriptor sets (0 .. set_count - 1)
		//   - The push constant bind group slot (if any)
		const bool has_pc = wg_container->has_push_constants();
		const uint32_t pc_group = has_pc ? wg_container->get_push_constant_bind_group() : UINT32_MAX;
		const uint32_t total_groups = has_pc ? MAX(set_count, pc_group + 1) : set_count;

		LocalVector<WGPUBindGroupLayout> all_layouts;
		all_layouts.resize(total_groups);

		// Use the persistent empty layout for gap slots between sets and push constant slot.

		for (uint32_t i = 0; i < total_groups; i++) {
			if (has_pc && i == pc_group) {
				// Build a merged layout: the shader's own group entries (material uniforms,
				// textures, etc.) PLUS the PC ring-buffer entry at PUSH_CONSTANT_RING_BINDING with hasDynamicOffset.
				// This is needed because calling setBindGroup() twice for the same group index
				// overrides the first binding — we must combine both into one layout/bind group.
				bool has_existing = (i < set_count && i < (uint32_t)shader->bind_group_infos.size() &&
						!shader->bind_group_infos[i].entries.is_empty());
				if (has_existing) {
					// Rebuild all layout entries for this group (both sampler AND texture for
					// combined types — bge.layout_entry only stores the texture entry).
					LocalVector<WGPUBindGroupLayoutEntry> merged_entries;
					const Vector<RenderingDeviceCommons::ShaderUniform> &pc_uniforms = shader_refl.uniform_sets[i];
					for (int u_idx2 = 0; u_idx2 < pc_uniforms.size(); u_idx2++) {
						const RenderingDeviceCommons::ShaderUniform &pu = pc_uniforms[u_idx2];
						WGPUShaderStage pvis = _stages_to_wgpu_visibility((uint32_t)pu.stages);
						// A multisampled combined sampler has only its texture entry, copied below.
						if (pu.type == RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE && !wgsl_tex_dims.has(((uint32_t)i << 16) | (pu.binding * 2))) {
							WGPUBindGroupLayoutEntry se = {}, te = {};
							se.binding = pu.binding * 2 + 0;
							se.visibility = pvis;
							{
								uint32_t k = ((uint32_t)i << 16) | (pu.binding * 2 + 0);
								se.sampler.type = (wgsl_is_comparison_sampler.has(k) && wgsl_is_comparison_sampler[k])
										? WGPUSamplerBindingType_Comparison
										: WGPUSamplerBindingType_Filtering;
							}
							te.binding = pu.binding * 2 + 1;
							te.visibility = pvis;
							if (has_resource_usage) {
								uint32_t sampler_key = (i << 16) | se.binding;
								uint32_t texture_key = (i << 16) | te.binding;
								se.visibility = (WGPUShaderStage)(wgsl_resource_stages.has(sampler_key) ? wgsl_resource_stages[sampler_key] : 0);
								te.visibility = (WGPUShaderStage)(wgsl_resource_stages.has(texture_key) ? wgsl_resource_stages[texture_key] : 0);
							}
							{
								uint32_t k = ((uint32_t)i << 16) | (pu.binding * 2 + 1);
								bool is_ms = wgsl_is_multisampled_texture.has(k) && wgsl_is_multisampled_texture[k];
								bool is_depth = wgsl_is_depth_texture.has(k) && wgsl_is_depth_texture[k];
								const WGPUTextureSampleType *wgsl_type = wgsl_tex_sample_type.getptr(k);
								te.texture.sampleType = _bgl_texture_sample_type(is_depth, is_ms, wgsl_type ? *wgsl_type : WGPUTextureSampleType_Float);
								te.texture.viewDimension = wgsl_tex_dims.has(k) ? wgsl_tex_dims[k] : WGPUTextureViewDimension_2D;
								te.texture.multisampled = is_ms;
								if (is_ms && !is_depth) {
									se.sampler.type = WGPUSamplerBindingType_NonFiltering;
								}
							}
							merged_entries.push_back(se);
							merged_entries.push_back(te);
						} else {
							// For all other types, just copy the existing bge.layout_entry.
							merged_entries.push_back(shader->bind_group_infos[i].entries[u_idx2].layout_entry);
						}
					}
					// Add the PC ring-buffer entry: PUSH_CONSTANT_RING_BINDING, read-only storage, hasDynamicOffset=true.
					// Safety: skip if any existing entry is already at PUSH_CONSTANT_RING_BINDING.
					bool has_pc_binding = false;
					for (const auto &me : merged_entries) {
						if (me.binding == PUSH_CONSTANT_RING_BINDING) {
							has_pc_binding = true;
							break;
						}
					}
					WGPUBindGroupLayoutEntry pc_entry = {};
					pc_entry.binding = PUSH_CONSTANT_RING_BINDING;
					pc_entry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
					pc_entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
					pc_entry.buffer.hasDynamicOffset = true;
					pc_entry.buffer.minBindingSize = 0;
					if (!has_pc_binding) {
						merged_entries.push_back(pc_entry);
					}

					// Add depth alias entries to the merged layout.
					for (const KeyValue<uint32_t, uint32_t> &kv : wgsl_depth_alias_bindings) {
						uint32_t alias_grp = kv.key >> 16;
						uint32_t alias_bnd = kv.key & 0xFFFF;
						if (alias_grp != i) {
							continue;
						}
						WGPUBindGroupLayoutEntry ae = {};
						ae.binding = alias_bnd;
						ae.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
						ae.texture.sampleType = WGPUTextureSampleType_Float;
						ae.texture.viewDimension = wgsl_tex_dims.has(kv.key) ? wgsl_tex_dims[kv.key] : WGPUTextureViewDimension_2D;
						ae.texture.multisampled = false;
						merged_entries.push_back(ae);
					}
					WGPUBindGroupLayoutDescriptor merged_desc = {};
					merged_desc.entryCount = merged_entries.size();
					merged_desc.entries = merged_entries.ptr();
					shader->merged_pc_group_layout = wgpuDeviceCreateBindGroupLayout(device, &merged_desc);
					if (!shader->merged_pc_group_layout) {
						error_text = "WebGPU: failed to create merged PC+material bind group layout.";
						goto cleanup;
					}
					all_layouts[i] = shader->merged_pc_group_layout;
				} else {
					// No material uniforms at this group — use the universal PC-only layout.
					all_layouts[i] = push_constant_bind_group_layout;
				}
			} else if (i < set_count) {
				// Check if this set is empty (no uniforms defined in GLSL).
				// E.g., CanvasShaderRD has sets 0, 2, 3 but no set 1.
				bool is_empty_set = (i >= (uint32_t)shader->bind_group_infos.size() ||
						shader->bind_group_infos[i].entries.is_empty());
				if (is_empty_set) {
					all_layouts[i] = empty_bind_group_layout;
					shader->gap_bind_group_indices.push_back(i);
				} else {
					all_layouts[i] = shader->bind_group_layouts[i];
				}
			} else {
				// Gap slot — use empty layout and record for pre-binding at draw time.
				all_layouts[i] = empty_bind_group_layout;
				shader->gap_bind_group_indices.push_back(i);
			}
		}

		WGPUPipelineLayoutDescriptor pl_desc = {};
		pl_desc.bindGroupLayoutCount = total_groups;
		pl_desc.bindGroupLayouts = all_layouts.size() > 0 ? all_layouts.ptr() : nullptr;

		// Label the pipeline layout for JS-side patch logs.
		String _pl_label = "plyt:" + shader->name;
		CharString _pl_label_cs = _pl_label.utf8();
		pl_desc.label = { _pl_label_cs.get_data(), WGPU_STRLEN };

		shader->pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pl_desc);
		if (shader->pipeline_layout == nullptr) {
			error_text = "WebGPU: wgpuDeviceCreatePipelineLayout failed.";
			goto cleanup;
		}

		return ShaderID(shader);

	} // End block scope.

cleanup:
	// Clean up partially-constructed shader (mirrors shader_free).
	for (int i = 0; i < 6; i++) {
		if (shader->stage_modules[i]) {
			wgpuShaderModuleRelease(shader->stage_modules[i]);
		}
	}
	if (shader->pipeline_layout) {
		wgpuPipelineLayoutRelease(shader->pipeline_layout);
	}
	for (WGPUBindGroupLayout &layout : shader->bind_group_layouts) {
		if (layout) {
			wgpuBindGroupLayoutRelease(layout);
		}
	}
	if (shader->merged_pc_group_layout) {
		wgpuBindGroupLayoutRelease(shader->merged_pc_group_layout);
	}
	memdelete(shader);
	ERR_FAIL_V_MSG(ShaderID(), error_text);
}

uint32_t RenderingDeviceDriverWebGPU::shader_get_layout_hash(ShaderID p_shader) {
	WGShader *shader = (WGShader *)(p_shader.id);
	if (!shader) {
		return 0;
	}
	// Use the pipeline layout pointer as a cheap hash identifier.
	return (uint32_t)(uint64_t)(void *)(shader->pipeline_layout);
}

void RenderingDeviceDriverWebGPU::shader_free(ShaderID p_shader) {
	WGShader *shader = (WGShader *)(p_shader.id);
	ERR_FAIL_NULL(shader);
	for (int i = 0; i < 6; i++) {
		if (shader->stage_modules[i]) {
			wgpuShaderModuleRelease(shader->stage_modules[i]);
			shader->stage_modules[i] = nullptr;
		}
	}
	if (shader->pipeline_layout) {
		wgpuPipelineLayoutRelease(shader->pipeline_layout);
	}
	for (WGPUBindGroupLayout &layout : shader->bind_group_layouts) {
		if (layout) {
			wgpuBindGroupLayoutRelease(layout);
		}
	}
	if (shader->merged_pc_group_layout) {
		wgpuBindGroupLayoutRelease(shader->merged_pc_group_layout);
	}
	memdelete(shader);
}

void RenderingDeviceDriverWebGPU::shader_destroy_modules(ShaderID p_shader) {
	WGShader *shader = (WGShader *)(p_shader.id);
	ERR_FAIL_NULL(shader);
	for (int i = 0; i < 6; i++) {
		if (shader->stage_modules[i]) {
			wgpuShaderModuleRelease(shader->stage_modules[i]);
			shader->stage_modules[i] = nullptr;
		}
	}
}

// =============================================================================
// UNIFORM SET
// =============================================================================

RDD::UniformSetID RenderingDeviceDriverWebGPU::uniform_set_create(VectorView<BoundUniform> p_uniforms, ShaderID p_shader, uint32_t p_set_index, int p_linear_pool_index) {
	WGShader *shader = (WGShader *)(p_shader.id);
	ERR_FAIL_NULL_V(shader, UniformSetID());
	ERR_FAIL_COND_V(p_set_index >= (uint32_t)shader->bind_group_layouts.size(), UniformSetID());

	WGPUBindGroupLayout layout = shader->bind_group_layouts[p_set_index];
	ERR_FAIL_NULL_V(layout, UniformSetID());

	// If this set is also the push constant group AND the shader has a merged layout
	// (PC ring buffer + material uniforms combined), switch to the merged layout.
	// We will also inject a PC ring buffer entry into the entries array below.
	const bool is_pc_group = (shader->push_constant_size > 0 &&
			p_set_index == shader->push_constant_bind_group &&
			shader->merged_pc_group_layout != nullptr);
	if (is_pc_group) {
		layout = shader->merged_pc_group_layout;
	}

	LocalVector<BoundUniform> expanded_uniforms;
	for (uint32_t source_index = 0; source_index < p_uniforms.size(); source_index++) {
		const BoundUniform &source = p_uniforms[source_index];
		const uint32_t *binding = shader->binding_remaps.getptr((uint64_t(p_set_index) << 32) | source.binding);
		ERR_FAIL_NULL_V(binding, UniformSetID());
		const uint32_t stride = source.type == UNIFORM_TYPE_SAMPLER_WITH_TEXTURE ? 2 : 1;
		const uint32_t count = _is_handle_uniform(source.type) ? source.ids.size() / stride : 1;
		for (uint32_t element = 0; element < count; element++) {
			BoundUniform uniform;
			uniform.type = source.type;
			uniform.binding = *binding + element;
			uniform.immutable_sampler = source.immutable_sampler;
			if (_is_handle_uniform(source.type)) {
				for (uint32_t id = 0; id < stride; id++) {
					uniform.ids.push_back(source.ids[element * stride + id]);
				}
			} else {
				uniform.ids = source.ids;
			}
			expanded_uniforms.push_back(uniform);
		}
	}

	// Reject overlapping writable aliases instead of silently redirecting their writes.
	HashMap<WGPUBuffer, bool> storage_buffers;
	for (const BoundUniform &uniform : expanded_uniforms) {
		if ((uniform.type != UNIFORM_TYPE_STORAGE_BUFFER && uniform.type != UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC) || uniform.ids.is_empty()) {
			continue;
		}
		WGPUBufferBindingType access = WGPUBufferBindingType_Undefined;
		for (const auto &info : shader->bind_group_infos[p_set_index].entries) {
			if (info.layout_entry.binding == uniform.binding * 2 && info.layout_entry.visibility != 0) {
				access = info.layout_entry.buffer.type;
				break;
			}
		}
		if (access == WGPUBufferBindingType_Undefined) {
			continue;
		}
		WGBuffer *buffer = (WGBuffer *)uniform.ids[0].id;
		ERR_FAIL_NULL_V(buffer, UniformSetID());
		bool writable = access == WGPUBufferBindingType_Storage;
		const bool *previous = storage_buffers.getptr(buffer->handle);
		ERR_FAIL_COND_V_MSG(previous && (*previous || writable), UniformSetID(), "WebGPU: overlapping writable storage-buffer bindings are unsupported. Use distinct buffers.");
		storage_buffers[buffer->handle] = writable;
	}

	// Each BoundUniform may expand to one or two WGPUBindGroupEntry items.
	LocalVector<WGPUBindGroupEntry> entries;
	entries.reserve(expanded_uniforms.size() * 2);

	// Allocate the uniform set early so texture handlers can store temp views.
	WGUniformSet *us = memnew(WGUniformSet);
	us->set_index = p_set_index;

	for (uint32_t i = 0; i < expanded_uniforms.size(); i++) {
		const BoundUniform &uniform = expanded_uniforms[i];
		// WebGPU has no immutable sampler concept — always provide the sampler
		// in the bind group (unlike Vulkan where it's baked into the pipeline layout).

		switch (uniform.type) {
			case UNIFORM_TYPE_SAMPLER: {
				if (uniform.ids.size() > 0) {
					WGPUBindGroupEntry entry = {};
					entry.binding = uniform.binding * 2;
					entry.sampler = (WGPUSampler)(uniform.ids[0].id);
					entries.push_back(entry);
				}
			} break;

			case UNIFORM_TYPE_TEXTURE:
			case UNIFORM_TYPE_INPUT_ATTACHMENT: {
				WGPUTextureViewDimension expected_dim = WGPUTextureViewDimension_Undefined;
				WGPUTextureSampleType expected_sample = WGPUTextureSampleType_Undefined;
				if (p_set_index < (uint32_t)shader->bind_group_infos.size()) {
					for (const auto &bge : shader->bind_group_infos[p_set_index].entries) {
						if (bge.layout_entry.binding == uniform.binding * 2) {
							expected_dim = bge.layout_entry.texture.viewDimension;
							expected_sample = bge.layout_entry.texture.sampleType;
							break;
						}
					}
				}
				if (uniform.ids.size() > 0) {
					WGTexture *tex = (WGTexture *)(uniform.ids[0].id);
					if (tex != nullptr) {
						WGPUBindGroupEntry entry = {};
						entry.binding = uniform.binding * 2;

						// Fix depth/float mismatch: if layout expects Float but
						// texture has a depth format (common with Godot's depth
						// fallback textures), substitute a float fallback texture.
						if (expected_sample == WGPUTextureSampleType_Float &&
								is_depth_format_wgpu(tex->format) &&
								fallback_float_texture_view != nullptr) {
							// Also check if we need cube dimension for the depth fallback.
							if (expected_dim == WGPUTextureViewDimension_Cube && fallback_cube_texture_view != nullptr) {
								entry.textureView = fallback_cube_texture_view;
							} else {
								entry.textureView = fallback_float_texture_view;
							}
						} else if (expected_sample == WGPUTextureSampleType_Float &&
								!float32_filterable_supported &&
								_is_float32_format(tex->format) &&
								fallback_float_texture_view != nullptr) {
							// R32Float/RG32Float/RGBA32Float textures are unfilterable
							// without the float32-filterable feature (e.g., Android Chrome
							// on Adreno). Substitute a filterable RGBA8 fallback to avoid
							// validation errors. The texture data is lost but rendering
							// continues without GPU errors.
							if (expected_dim == WGPUTextureViewDimension_Cube && fallback_cube_texture_view != nullptr) {
								entry.textureView = fallback_cube_texture_view;
							} else {
								entry.textureView = fallback_float_texture_view;
							}
						} else if (tex->dimension == WGPUTextureDimension_3D && expected_dim != WGPUTextureViewDimension_3D &&
								expected_dim != WGPUTextureViewDimension_Undefined && fallback_float_texture_view != nullptr) {
							// 3D textures have only 3D views. A layout expecting another dimension came from an
							// unused binding (such as an absent VoxelGI instance), so its contents are never read.
							entry.textureView = fallback_float_texture_view;
						} else if (expected_dim != WGPUTextureViewDimension_Undefined &&
								(expected_dim != tex->view_dimension || (_reads_depth(expected_sample, tex->format) && has_stencil_wgpu(tex->format)))) {
							// Fix dimension mismatch: if the layout expects a different dimension
							// than the texture's default view (e.g., 2D vs Cube), create a
							// compatible view or use a fallback texture.
							if (expected_dim == WGPUTextureViewDimension_Cube && tex->layers < 6 &&
									fallback_cube_texture_view != nullptr) {
								// Can't create a cube view from a texture with < 6 layers.
								entry.textureView = fallback_cube_texture_view;
							} else if (tex->view_source != nullptr) {
								// Use slice base offsets so slice views don't
								// silently remap to mip 0 / layer 0 of the parent.
								const bool depth = _reads_depth(expected_sample, tex->format);
								WGPUTextureViewDescriptor vd = {};
								vd.format = depth ? WGPUTextureFormat_Undefined : tex->format;
								vd.dimension = expected_dim;
								vd.baseMipLevel = tex->base_mipmap;
								vd.mipLevelCount = tex->mipmaps;
								vd.baseArrayLayer = tex->base_layer;
								vd.arrayLayerCount = (expected_dim == WGPUTextureViewDimension_2D) ? 1 : tex->layers;
								vd.aspect = depth ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
								WGPUTextureView fixed_view = wgpuTextureCreateView(tex->view_source, &vd);
								if (fixed_view) {
									entry.textureView = fixed_view;
									us->temp_views.push_back(fixed_view);
								} else {
									entry.textureView = tex->default_view;
								}
							} else {
								entry.textureView = tex->default_view;
							}
						} else {
							entry.textureView = tex->default_view;
						}
						us->bound_textures[entry.binding] = tex;
						entries.push_back(entry);
					}
				}
			} break;

			case UNIFORM_TYPE_SAMPLER_WITH_TEXTURE: {
				// Godot pairs sampler+texture as ids[j*2] / ids[j*2+1].
				// WebGPU needs separate sampler and texture entries.
				// Sampler at binding*2+j*2+0, texture at binding*2+j*2+1 (matches layout and SPIR-V preprocessor).
				// Look up expected texture dimension and sample type from the shader layout.
				WGPUTextureViewDimension swt_expected_dim = WGPUTextureViewDimension_Undefined;
				WGPUTextureSampleType swt_expected_sample = WGPUTextureSampleType_Float;
				bool swt_expected_ms = false;
				bool texture_only = false; // A multisampled texture without its sampler, at binding*2.
				if (p_set_index < (uint32_t)shader->bind_group_infos.size()) {
					for (const auto &bge : shader->bind_group_infos[p_set_index].entries) {
						if (bge.layout_entry.binding == uniform.binding * 2 && bge.layout_entry.texture.sampleType != WGPUTextureSampleType_BindingNotUsed) {
							texture_only = true;
							break;
						}
					}
				}
				if (texture_only) {
					WGTexture *tex = uniform.ids.size() > 1 ? (WGTexture *)(uniform.ids[1].id) : nullptr;
					if (tex && tex->default_view) {
						WGPUBindGroupEntry te = {};
						te.binding = uniform.binding * 2;
						te.textureView = tex->sample_count > 1 || fallback_ms_texture_view == nullptr ? tex->default_view : fallback_ms_texture_view;
						if (tex->sample_count > 1 && has_stencil_wgpu(tex->format) && tex->view_source) {
							// Only the depth aspect of a depth-stencil texture can be bound.
							WGPUTextureViewDescriptor vd = {};
							vd.dimension = WGPUTextureViewDimension_2D;
							vd.baseMipLevel = tex->base_mipmap;
							vd.mipLevelCount = 1;
							vd.baseArrayLayer = tex->base_layer;
							vd.arrayLayerCount = 1;
							vd.aspect = WGPUTextureAspect_DepthOnly;
							WGPUTextureView depth_view = wgpuTextureCreateView(tex->view_source, &vd);
							if (depth_view) {
								te.textureView = depth_view;
								us->temp_views.push_back(depth_view);
							}
						}
						entries.push_back(te);
					}
					break;
				}
				if (p_set_index < (uint32_t)shader->bind_group_infos.size()) {
					uint32_t tex_binding = uniform.binding * 2 + 1;
					for (const auto &bge : shader->bind_group_infos[p_set_index].entries) {
						if (bge.layout_entry.binding == tex_binding) {
							swt_expected_dim = bge.layout_entry.texture.viewDimension;
							swt_expected_sample = bge.layout_entry.texture.sampleType;
							swt_expected_ms = (bool)bge.layout_entry.texture.multisampled;
							break;
						}
					}
				}
				for (uint32_t j = 0; j < uniform.ids.size() / 2; j++) {
					WGPUSampler sampler = (WGPUSampler)(uniform.ids[j * 2 + 0].id);
					WGTexture *tex = (WGTexture *)(uniform.ids[j * 2 + 1].id);
					if (sampler) {
						WGPUBindGroupEntry se = {};
						se.binding = uniform.binding * 2 + j * 2 + 0;
						se.sampler = sampler;
						entries.push_back(se);
					}
					if (tex && tex->default_view) {
						WGPUBindGroupEntry te = {};
						te.binding = uniform.binding * 2 + j * 2 + 1;
						// Check sample-count mismatch first: if the BGL expects a multisampled
						// texture but the bound texture isn't multisampled (or vice versa),
						// we can't use either the real texture or a non-MS fallback. Substitute
						// the MSAA fallback for this case. This handles ResolveRasterShaderRD
						// which binds an MSAA depth texture into a float MSAA slot (WebGPU
						// forbids sampling depth as float, so the MSAA fallback is used).
						bool tex_is_ms = (tex->sample_count > 1);
						if (swt_expected_ms && (!tex_is_ms || (is_depth_format_wgpu(tex->format) && swt_expected_sample != WGPUTextureSampleType_Depth)) &&
								fallback_ms_texture_view != nullptr) {
							te.textureView = fallback_ms_texture_view;
						} else if (swt_expected_sample == WGPUTextureSampleType_Float &&
								is_depth_format_wgpu(tex->format) && fallback_float_texture_view != nullptr) {
							// Only substitute when the shader actually expects a color texture.
							if (swt_expected_dim == WGPUTextureViewDimension_Cube && fallback_cube_texture_view != nullptr) {
								te.textureView = fallback_cube_texture_view;
							} else {
								te.textureView = fallback_float_texture_view;
							}
						} else if (!float32_filterable_supported &&
								_is_float32_format(tex->format) &&
								fallback_float_texture_view != nullptr) {
							// R32Float/RG32Float/RGBA32Float unfilterable without feature.
							if (swt_expected_dim == WGPUTextureViewDimension_Cube && fallback_cube_texture_view != nullptr) {
								te.textureView = fallback_cube_texture_view;
							} else {
								te.textureView = fallback_float_texture_view;
							}
						} else if (swt_expected_dim != WGPUTextureViewDimension_Undefined &&
								(swt_expected_dim != tex->view_dimension || (_reads_depth(swt_expected_sample, tex->format) && has_stencil_wgpu(tex->format)))) {
							// Fix dimension mismatch (e.g., Cube↔2D with fallback textures).
							if (swt_expected_dim == WGPUTextureViewDimension_Cube && tex->layers < 6 &&
									fallback_cube_texture_view != nullptr) {
								te.textureView = fallback_cube_texture_view;
							} else if (tex->view_source != nullptr) {
								// Use slice base offsets so slice views don't
								// silently remap to mip 0 / layer 0 of the parent.
								WGPUTextureViewDescriptor vd = {};
								vd.format = _reads_depth(swt_expected_sample, tex->format) ? WGPUTextureFormat_Undefined : tex->format;
								vd.dimension = swt_expected_dim;
								vd.baseMipLevel = tex->base_mipmap;
								vd.mipLevelCount = tex->mipmaps;
								vd.baseArrayLayer = tex->base_layer;
								vd.arrayLayerCount = (swt_expected_dim == WGPUTextureViewDimension_2D) ? 1 : tex->layers;
								vd.aspect = _reads_depth(swt_expected_sample, tex->format) ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
								WGPUTextureView fixed_view = wgpuTextureCreateView(tex->view_source, &vd);
								if (fixed_view) {
									te.textureView = fixed_view;
									us->temp_views.push_back(fixed_view);
								} else {
									te.textureView = tex->default_view;
								}
							} else {
								te.textureView = tex->default_view;
							}
						} else {
							te.textureView = tex->default_view;
						}
						us->bound_textures[te.binding] = tex;
						entries.push_back(te);
					}
				}
			} break;

			case UNIFORM_TYPE_IMAGE: {
				if (uniform.ids.size() > 0) {
					WGTexture *tex = (WGTexture *)(uniform.ids[0].id);
					if (tex != nullptr) {
						WGPUBindGroupEntry entry = {};
						entry.binding = uniform.binding * 2;
						entry.textureView = tex->default_view;
						us->bound_textures[entry.binding] = tex;
						entries.push_back(entry);
					}
				}
			} break;

			case UNIFORM_TYPE_UNIFORM_BUFFER: {
				WGBuffer *buf = (WGBuffer *)(uniform.ids[0].id);
				ERR_CONTINUE_MSG(buf == nullptr, "WebGPU: null buffer in uniform set.");
				WGPUBindGroupEntry entry = {};
				entry.binding = uniform.binding * 2; // Preprocessing doubles all non-combined bindings.
				entry.buffer = buf->handle;
				entry.offset = 0;
				entry.size = buf->size;
				entries.push_back(entry);
			} break;

			case UNIFORM_TYPE_STORAGE_BUFFER: {
				WGBuffer *buf = (WGBuffer *)(uniform.ids[0].id);
				ERR_CONTINUE_MSG(buf == nullptr, "WebGPU: null buffer in storage uniform.");
				WGPUBindGroupEntry entry = {};
				entry.binding = uniform.binding * 2; // Preprocessing doubles all non-combined bindings.
				entry.offset = 0;
				entry.buffer = buf->handle;
				entry.size = buf->size;
				entries.push_back(entry);
			} break;

			case UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC:
			case UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC: {
				WGBuffer *buf = (WGBuffer *)(uniform.ids[0].id);
				ERR_CONTINUE_MSG(buf == nullptr, "WebGPU: null buffer in dynamic uniform.");
				WGPUBindGroupEntry entry = {};
				entry.binding = uniform.binding * 2;
				entry.buffer = buf->handle;
				entry.offset = 0;
				// Task 7.5: For dynamic persistent buffers, bind ONE slice (per_frame_size),
				// not the full physical allocation. The rest of the buffer holds the other
				// frames' slices, reached via the dynamic offset at bind time.
				if (buf->is_dynamic() && buf->per_frame_size > 0) {
					entry.size = buf->per_frame_size;
					us->dynamic_buffers.push_back(buf);
				} else {
					entry.size = buf->size;
				}
				entries.push_back(entry);
			} break;

			case UNIFORM_TYPE_TEXTURE_BUFFER:
			case UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER: {
				// Tint converts texture buffers to uniform/storage buffers. Bind as buffer.
				WGBuffer *buf = (WGBuffer *)(uniform.ids[0].id);
				ERR_CONTINUE_MSG(buf == nullptr, "WebGPU: null buffer in texture buffer uniform.");
				WGPUBindGroupEntry entry = {};
				entry.binding = uniform.binding * 2;
				entry.buffer = buf->handle;
				entry.offset = 0;
				entry.size = buf->size;
				entries.push_back(entry);
			} break;

			case UNIFORM_TYPE_IMAGE_BUFFER: {
				WGBuffer *buf = (WGBuffer *)(uniform.ids[0].id);
				ERR_CONTINUE_MSG(buf == nullptr, "WebGPU: null buffer in image buffer uniform.");
				WGPUBindGroupEntry entry = {};
				entry.binding = uniform.binding * 2;
				entry.buffer = buf->handle;
				entry.offset = 0;
				entry.size = buf->size;
				entries.push_back(entry);
			} break;

			default: {
				WARN_PRINT_ONCE(vformat("WebGPU: unhandled uniform type %d in uniform_set_create.", (int)uniform.type));
			} break;
		}
	}

	// Add depth alias bind group entries: for each depth alias binding, provide
	// the same texture view as the paired depth texture at (alias_binding - 1).
	for (const KeyValue<uint32_t, uint32_t> &kv : shader->depth_alias_bindings) {
		uint32_t alias_grp = kv.key >> 16;
		uint32_t alias_bnd = kv.key & 0xFFFF;
		uint32_t depth_bnd = kv.value;
		if (alias_grp != p_set_index) {
			continue;
		}
		// Find the texture view from the depth texture entry.
		WGPUTextureView alias_view = nullptr;
		for (const auto &e : entries) {
			if (e.binding == depth_bnd && e.textureView != nullptr) {
				alias_view = e.textureView;
				break;
			}
		}
		if (alias_view) {
			WGPUBindGroupEntry alias_entry = {};
			alias_entry.binding = alias_bnd;
			// The alias is supposed to sample depth as Float, but alias_view may
			// point at a Depth-format texture. Substitute the float fallback.
			if (alias_view == fallback_float_texture_view) {
				alias_entry.textureView = alias_view; // Already substituted.
			} else {
				// Check if the source texture view came from a depth-format texture.
				// We can't query the view's format, so check if the alias BGL expects Float.
				// For safety, always use fallback for depth alias entries since
				// they always represent Float sampling of depth data.
				if (fallback_float_texture_view != nullptr) {
					alias_entry.textureView = fallback_float_texture_view;
				} else {
					alias_entry.textureView = alias_view;
				}
			}
			entries.push_back(alias_entry);
		}
	}

	// If this is the merged PC group, inject the push constant ring buffer at PUSH_CONSTANT_RING_BINDING.
	// Dynamic offset is 0 here; _flush_push_constants will rebind with the correct offset.
	if (is_pc_group) {
		WGPUBindGroupEntry pc_entry = {};
		pc_entry.binding = PUSH_CONSTANT_RING_BINDING;
		pc_entry.buffer = push_constant_ring_buffer;
		pc_entry.offset = 0;
		pc_entry.size = PUSH_CONSTANT_SLOT_ALIGNMENT;
		entries.push_back(pc_entry);
	}

	// Log any duplicate binding indices in bind group entries.
	for (uint32_t a = 0; a < entries.size(); a++) {
		for (uint32_t b = a + 1; b < entries.size(); b++) {
			if (entries[a].binding == entries[b].binding) {
				WEBGPU_DIAG({ console.error('[BG-DUP] set=' + $0 + ' binding=' + $1 + ' idx_a=' + $2 + ' idx_b=' + $3); }, (int)p_set_index, (int)entries[a].binding, (int)a, (int)b);
			}
		}
	}

	// Cache the real resources and source shader for BGL rebinding; another shader may use the unused bindings.
	us->cached_entries.resize(entries.size());
	for (uint32_t i = 0; i < entries.size(); i++) {
		us->cached_entries[i] = entries[i];
	}
	us->source_shader = shader;
	_fit_unused_bindings(entries, shader, p_set_index);

	WGPUBindGroupDescriptor bg_desc = {};
	bg_desc.layout = layout;
	bg_desc.entryCount = entries.size();
	bg_desc.entries = entries.size() > 0 ? entries.ptr() : nullptr;

	WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);
	if (bg == nullptr) {
		uniform_set_free(UniformSetID(us)); // Releases temporary views.
		ERR_FAIL_V_MSG(UniformSetID(), "WebGPU: wgpuDeviceCreateBindGroup failed.");
	}

	us->handle = bg;
	return UniformSetID(us);
}

WGPUTextureView RenderingDeviceDriverWebGPU::_unused_binding_view(const WGPUBindGroupLayoutEntry &p_layout) {
	const bool storage = p_layout.storageTexture.access != WGPUStorageTextureAccess_BindingNotUsed;
	WGPUTextureViewDimension dimension = storage ? p_layout.storageTexture.viewDimension : p_layout.texture.viewDimension;
	if (dimension == WGPUTextureViewDimension_Undefined) {
		dimension = WGPUTextureViewDimension_2D;
	}
	WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
	if (storage) {
		format = p_layout.storageTexture.format;
	} else if (p_layout.texture.sampleType == WGPUTextureSampleType_UnfilterableFloat) {
		format = WGPUTextureFormat_R32Float;
	} else if (p_layout.texture.sampleType == WGPUTextureSampleType_Uint) {
		format = WGPUTextureFormat_R32Uint;
	} else if (p_layout.texture.sampleType == WGPUTextureSampleType_Sint) {
		format = WGPUTextureFormat_R32Sint;
	} else if (p_layout.texture.sampleType == WGPUTextureSampleType_Depth) {
		format = WGPUTextureFormat_Depth32Float;
	}
	const bool multisampled = !storage && p_layout.texture.multisampled;
	// Per binding as well: writable storage dummies would alias within one bind group.
	const uint64_t key = (uint64_t(storage) << 63) | (uint64_t(multisampled) << 62) | (uint64_t(dimension) << 56) | (uint64_t(p_layout.binding) << 32) | uint64_t(format);
	if (const Pair<WGPUTexture, WGPUTextureView> *existing = unused_binding_views.getptr(key)) {
		return existing->second;
	}

	const bool cube = dimension == WGPUTextureViewDimension_Cube || dimension == WGPUTextureViewDimension_CubeArray;
	WGPUTextureDescriptor desc = {};
	desc.dimension = dimension == WGPUTextureViewDimension_1D ? WGPUTextureDimension_1D : (dimension == WGPUTextureViewDimension_3D ? WGPUTextureDimension_3D : WGPUTextureDimension_2D);
	desc.size = { 1, 1, cube ? 6u : 1u };
	desc.format = format;
	desc.mipLevelCount = 1;
	desc.sampleCount = multisampled ? 4 : 1;
	desc.usage = storage ? WGPUTextureUsage_StorageBinding : (WGPUTextureUsage_TextureBinding | (multisampled ? WGPUTextureUsage_RenderAttachment : WGPUTextureUsage_None));
	WGPUTexture texture = wgpuDeviceCreateTexture(device, &desc);
	ERR_FAIL_NULL_V(texture, nullptr);
	WGPUTextureViewDescriptor view_desc = {};
	view_desc.dimension = dimension;
	view_desc.mipLevelCount = 1;
	view_desc.arrayLayerCount = cube ? 6 : 1;
	WGPUTextureView view = wgpuTextureCreateView(texture, &view_desc);
	unused_binding_views.insert(key, Pair<WGPUTexture, WGPUTextureView>(texture, view));
	return view;
}

void RenderingDeviceDriverWebGPU::_fit_unused_bindings(LocalVector<WGPUBindGroupEntry> &r_entries, const WGShader *p_shader, uint32_t p_set_idx) {
	if (p_set_idx >= (uint32_t)p_shader->bind_group_infos.size()) {
		return;
	}
	for (WGPUBindGroupEntry &entry : r_entries) {
		for (const WGShader::BindGroupEntry &bge : p_shader->bind_group_infos[p_set_idx].entries) {
			const WGPUBindGroupLayoutEntry &layout = bge.layout_entry;
			if (layout.binding != entry.binding || layout.visibility != WGPUShaderStage_None) {
				continue;
			}
			if (entry.sampler != nullptr && layout.sampler.type != WGPUSamplerBindingType_BindingNotUsed) {
				entry.sampler = layout.sampler.type == WGPUSamplerBindingType_Comparison ? dummy_comparison_sampler : dummy_filtering_sampler;
			} else if (entry.textureView != nullptr) {
				WGPUTextureView view = _unused_binding_view(layout);
				if (view) {
					entry.textureView = view;
				}
			}
			break;
		}
	}
}

WGPUBindGroup RenderingDeviceDriverWebGPU::_get_compatible_bind_group(WGUniformSet *p_us, WGShader *p_target_shader, uint32_t p_set_idx) {
	// Fast path: same shader or no shader info.
	if (!p_target_shader || p_us->source_shader == p_target_shader) {
		return p_us->handle;
	}
	if (p_set_idx >= (uint32_t)p_target_shader->bind_group_layouts.size()) {
		return p_us->handle;
	}

	// Check if the pipeline shader uses the same BGL or a merged PC layout.
	WGPUBindGroupLayout target_layout = p_target_shader->bind_group_layouts[p_set_idx];
	if (p_target_shader->merged_pc_group_layout && p_set_idx == p_target_shader->push_constant_bind_group) {
		target_layout = p_target_shader->merged_pc_group_layout;
	}

	// If source shader uses the same BGL, no rebind needed.
	WGPUBindGroupLayout source_layout = nullptr;
	if (p_us->source_shader && p_set_idx < (uint32_t)p_us->source_shader->bind_group_layouts.size()) {
		source_layout = p_us->source_shader->bind_group_layouts[p_set_idx];
		if (p_us->source_shader->merged_pc_group_layout && p_set_idx == p_us->source_shader->push_constant_bind_group) {
			source_layout = p_us->source_shader->merged_pc_group_layout;
		}
	}
	if (source_layout == target_layout) {
		return p_us->handle;
	}

	// Check rebind cache. Keyed on the shader's uid, not the layout handle, which a freed
	// shader can hand on to a new layout.
	const uint64_t cache_key = (p_target_shader->uid << 8) | p_set_idx;
	if (WGPUBindGroup *cached = p_us->rebind_cache.getptr(cache_key)) {
		return *cached ? *cached : p_us->handle;
	}

	// Build adapted entries: copy cached entries and fix sampler type mismatches.
	LocalVector<WGPUBindGroupEntry> adapted;
	adapted.resize(p_us->cached_entries.size());
	for (uint32_t i = 0; i < p_us->cached_entries.size(); i++) {
		adapted[i] = p_us->cached_entries[i];
	}

	// Check sampler and texture entries for type mismatches between source and target BGLs.
	if (p_set_idx < (uint32_t)p_target_shader->bind_group_infos.size()) {
		for (auto &entry : adapted) {
			// --- Sampler adaptation ---
			if (entry.sampler != nullptr) {
				WGPUSamplerBindingType target_samp_type = WGPUSamplerBindingType_BindingNotUsed;
				for (const auto &bge : p_target_shader->bind_group_infos[p_set_idx].entries) {
					if (bge.layout_entry.binding == entry.binding &&
							bge.layout_entry.sampler.type != WGPUSamplerBindingType_BindingNotUsed) {
						target_samp_type = bge.layout_entry.sampler.type;
						break;
					}
				}
				// Fallback: check source shader for SWT sampler info.
				if (target_samp_type == WGPUSamplerBindingType_BindingNotUsed &&
						p_us->source_shader && p_set_idx < (uint32_t)p_us->source_shader->bind_group_infos.size()) {
					for (const auto &bge : p_us->source_shader->bind_group_infos[p_set_idx].entries) {
						if (bge.layout_entry.binding == entry.binding &&
								bge.layout_entry.sampler.type != WGPUSamplerBindingType_BindingNotUsed) {
							target_samp_type = bge.layout_entry.sampler.type;
							break;
						}
					}
				}

				WGPUSamplerBindingType source_samp_type = WGPUSamplerBindingType_BindingNotUsed;
				if (p_us->source_shader && p_set_idx < (uint32_t)p_us->source_shader->bind_group_infos.size()) {
					for (const auto &bge : p_us->source_shader->bind_group_infos[p_set_idx].entries) {
						if (bge.layout_entry.binding == entry.binding &&
								bge.layout_entry.sampler.type != WGPUSamplerBindingType_BindingNotUsed) {
							source_samp_type = bge.layout_entry.sampler.type;
							break;
						}
					}
				}

				if ((target_samp_type == WGPUSamplerBindingType_Filtering ||
							target_samp_type == WGPUSamplerBindingType_NonFiltering) &&
						source_samp_type == WGPUSamplerBindingType_Comparison && dummy_filtering_sampler) {
					entry.sampler = dummy_filtering_sampler;
				} else if (target_samp_type == WGPUSamplerBindingType_Comparison &&
						source_samp_type != WGPUSamplerBindingType_Comparison && dummy_comparison_sampler) {
					entry.sampler = dummy_comparison_sampler;
				}
			}

			// --- Texture view adaptation ---
			// If the target BGL expects Float but the entry has a depth-format texture view,
			// substitute the fallback float texture view.
			if (entry.textureView != nullptr) {
				WGPUTextureSampleType target_tex_sample = WGPUTextureSampleType_BindingNotUsed;
				for (const auto &bge : p_target_shader->bind_group_infos[p_set_idx].entries) {
					if (bge.layout_entry.binding == entry.binding &&
							bge.layout_entry.texture.sampleType != WGPUTextureSampleType_BindingNotUsed) {
						target_tex_sample = bge.layout_entry.texture.sampleType;
						break;
					}
				}
				// Check source BGL to detect if the texture was originally depth.
				WGPUTextureSampleType source_tex_sample = WGPUTextureSampleType_BindingNotUsed;
				if (p_us->source_shader && p_set_idx < (uint32_t)p_us->source_shader->bind_group_infos.size()) {
					for (const auto &bge : p_us->source_shader->bind_group_infos[p_set_idx].entries) {
						if (bge.layout_entry.binding == entry.binding &&
								bge.layout_entry.texture.sampleType != WGPUTextureSampleType_BindingNotUsed) {
							source_tex_sample = bge.layout_entry.texture.sampleType;
							break;
						}
					}
				}
				// Source was Depth, target expects Float → substitute fallback.
				if (source_tex_sample == WGPUTextureSampleType_Depth &&
						target_tex_sample == WGPUTextureSampleType_Float &&
						fallback_float_texture_view != nullptr) {
					entry.textureView = fallback_float_texture_view;
				}

				// --- Texture view dimension adaptation ---
				// If the target BGL expects a different dimension (e.g. 2D vs Cube),
				// create a compatible view from the original texture.
				WGPUTextureViewDimension target_dim = WGPUTextureViewDimension_Undefined;
				for (const auto &bge : p_target_shader->bind_group_infos[p_set_idx].entries) {
					if (bge.layout_entry.binding == entry.binding &&
							bge.layout_entry.texture.viewDimension != WGPUTextureViewDimension_Undefined) {
						target_dim = bge.layout_entry.texture.viewDimension;
						break;
					}
				}
				if (target_dim != WGPUTextureViewDimension_Undefined &&
						p_us->bound_textures.has(entry.binding)) {
					WGTexture *tex = p_us->bound_textures[entry.binding];
					if (tex && tex->dimension == WGPUTextureDimension_3D) {
						// 3D textures have only 3D views; another dimension means an unused binding, see uniform_set_create().
						entry.textureView = target_dim == WGPUTextureViewDimension_3D || fallback_float_texture_view == nullptr ? tex->default_view : fallback_float_texture_view;
					} else if (tex && tex->view_dimension == target_dim && // A previous variant may have replaced the default view.
							((target_tex_sample == WGPUTextureSampleType_Depth && is_depth_format_wgpu(tex->format)) ||
									(entry.textureView != fallback_float_texture_view && entry.textureView != fallback_cube_texture_view && entry.textureView != fallback_ms_texture_view))) {
						entry.textureView = tex->default_view;
					}
					if (tex && tex->dimension != WGPUTextureDimension_3D && tex->view_source && (tex->view_dimension != target_dim || (_reads_depth(target_tex_sample, tex->format) && has_stencil_wgpu(tex->format)))) {
						// Use slice base offsets so slice views don't
						// silently remap to mip 0 / layer 0 of the parent.
						WGPUTextureViewDescriptor vd = {};
						vd.format = _reads_depth(target_tex_sample, tex->format) ? WGPUTextureFormat_Undefined : tex->format;
						vd.dimension = target_dim;
						vd.baseMipLevel = tex->base_mipmap;
						vd.mipLevelCount = tex->mipmaps;
						vd.baseArrayLayer = tex->base_layer;
						vd.arrayLayerCount = (target_dim == WGPUTextureViewDimension_2D) ? 1 : tex->layers;
						vd.aspect = _reads_depth(target_tex_sample, tex->format) ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
						WGPUTextureView fixed = wgpuTextureCreateView(tex->view_source, &vd);
						if (fixed) {
							entry.textureView = fixed;
							p_us->temp_views.push_back(fixed);
						}
					}
				}
			}
		}
	}

	// Collect valid binding indices from the target BGL.
	// The target BGL includes entries from bind_group_infos (reflecting Godot uniforms)
	// plus depth alias entries and potentially a push constant entry.
	HashSet<uint32_t> target_bindings;
	if (p_set_idx < (uint32_t)p_target_shader->bind_group_infos.size()) {
		for (const auto &bge : p_target_shader->bind_group_infos[p_set_idx].entries) {
			target_bindings.insert(bge.layout_entry.binding);
			// For SWT, the layout_entry stores the texture binding; the sampler is at binding-1.
			if (bge.godot_type == RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE) {
				target_bindings.insert(bge.layout_entry.binding - 1); // Sampler binding.
			}
		}
	}
	// Add depth alias bindings.
	for (const KeyValue<uint32_t, uint32_t> &kv : p_target_shader->depth_alias_bindings) {
		uint32_t alias_grp = kv.key >> 16;
		uint32_t alias_bnd = kv.key & 0xFFFF;
		if (alias_grp == p_set_idx) {
			target_bindings.insert(alias_bnd);
		}
	}
	// Add push constant ring buffer binding if this is the PC group.
	if (p_target_shader->merged_pc_group_layout && p_set_idx == p_target_shader->push_constant_bind_group) {
		target_bindings.insert(PUSH_CONSTANT_RING_BINDING);
	}

	// Filter adapted entries: only keep entries whose binding exists in the target BGL.
	LocalVector<WGPUBindGroupEntry> filtered;
	for (const auto &e : adapted) {
		if (target_bindings.has(e.binding)) {
			filtered.push_back(e);
		}
	}
	_fit_unused_bindings(filtered, p_target_shader, p_set_idx);

	// Create re-bound bind group with target layout.
	perf.bind_group_cache_misses++;
	WGPUBindGroupDescriptor desc = {};
	desc.layout = target_layout;
	desc.entryCount = filtered.size();
	desc.entries = filtered.size() > 0 ? filtered.ptr() : nullptr;
	WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &desc);
	if (!bg) {
		WARN_PRINT(vformat("WebGPU: bind group rebind failed for set=%d src=%s tgt=%s entries=%d",
				(int)p_set_idx,
				p_us->source_shader ? p_us->source_shader->name : String("null"),
				p_target_shader->name,
				(int)filtered.size()));
	}
	p_us->rebind_cache[cache_key] = bg; // Cache even if null.
	return bg ? bg : p_us->handle;
}

void RenderingDeviceDriverWebGPU::uniform_set_free(UniformSetID p_uniform_set) {
	WGUniformSet *us = (WGUniformSet *)(p_uniform_set.id);
	ERR_FAIL_NULL(us);
	for (WGPUTextureView v : us->temp_views) {
		if (v) {
			wgpuTextureViewRelease(v);
		}
	}
	// Release cached rebind bind groups.
	for (const KeyValue<uint64_t, WGPUBindGroup> &kv : us->rebind_cache) {
		if (kv.value) {
			wgpuBindGroupRelease(kv.value);
		}
	}
	if (us->handle) {
		wgpuBindGroupRelease(us->handle);
	}
	memdelete(us);
}

uint32_t RenderingDeviceDriverWebGPU::uniform_sets_get_dynamic_offsets(VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count) const {
	// Task 7.5: Pack frame_idx for every dynamic buffer across the given sets into
	// 4-bit slots of a single uint32_t mask. Matches the Vulkan driver's encoding —
	// command_bind_*_uniform_sets reads the same mask and unpacks (frame_idx & 0xF).
	uint32_t mask = 0;
	uint32_t shift = 0;
	for (uint32_t i = 0; i < p_set_count; i++) {
		const WGUniformSet *us = (const WGUniformSet *)p_uniform_sets[i].id;
		if (!us) {
			continue;
		}
		for (const WGBuffer *buf : us->dynamic_buffers) {
			if (!buf) {
				continue;
			}
			mask |= (buf->frame_idx & 0xFu) << shift;
			shift += 4u;
		}
	}
	return mask;
}

void RenderingDeviceDriverWebGPU::command_uniform_set_prepare_for_use(CommandBufferID p_cmd_buffer, UniformSetID p_uniform_set, ShaderID p_shader, uint32_t p_set_index) {
	// No-op: WebGPU doesn't need explicit preparation.
}

// =============================================================================
// TRANSFER
// =============================================================================

void RenderingDeviceDriverWebGPU::command_clear_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, uint64_t p_offset, uint64_t p_size) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(buf);

	cmd->end_active_encoder();

	uint64_t size = (p_size == BUFFER_WHOLE_SIZE) ? (buf->size - p_offset) : p_size;
	size = (size + 3) & ~3ULL; // Must be multiple of 4.
	wgpuCommandEncoderClearBuffer(cmd->encoder, buf->handle, p_offset, size);
}

void RenderingDeviceDriverWebGPU::command_copy_buffer(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, BufferID p_dst_buffer, VectorView<BufferCopyRegion> p_regions) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGBuffer *src = (WGBuffer *)(p_src_buffer.id);
	WGBuffer *dst = (WGBuffer *)(p_dst_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(src);
	ERR_FAIL_NULL(dst);

	// If the source buffer has a shadow map (CPU-side staging buffer), flush
	// the shadow map to the staging buffer's GPU handle first via the queue,
	// then use the command encoder to copy staging→dst.  We must NOT write
	// directly to dst via wgpuQueueWriteBuffer because writeBuffer calls are
	// NOT ordered relative to draw commands within the same submission — they
	// all take effect before any draw, so every canvas in the frame would see
	// the last-written value rather than the per-canvas value.  Using the
	// encoder copy preserves the staging→dst copy order relative to draws.
	if (src->shadow_map) {
		for (uint32_t i = 0; i < p_regions.size(); i++) {
			const BufferCopyRegion &region = p_regions[i];
			uint64_t size = (region.size + 3) & ~3ULL;
			wgpuQueueWriteBuffer(queue, src->handle, region.src_offset, src->shadow_map + region.src_offset, size);
		}
		// The specific regions have been flushed above. Clear map_dirty so the
		// subsequent buffer_unmap() doesn't redundantly flush the entire buffer.
		src->map_dirty = false;
		// Fall through to encoder copy below.
	}

	cmd->end_active_encoder();

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		const BufferCopyRegion &region = p_regions[i];
		uint64_t size = (region.size + 3) & ~3ULL;
		wgpuCommandEncoderCopyBufferToBuffer(cmd->encoder, src->handle, region.src_offset, dst->handle, region.dst_offset, size);
	}
}

void RenderingDeviceDriverWebGPU::command_copy_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<TextureCopyRegion> p_regions) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGTexture *src = (WGTexture *)(p_src_texture.id);
	WGTexture *dst = (WGTexture *)(p_dst_texture.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(src);
	ERR_FAIL_NULL(dst);

	cmd->end_active_encoder();

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		const TextureCopyRegion &region = p_regions[i];

		WGPUTexelCopyTextureInfo src_copy = {};
		src_copy.texture = src->gpu_handle();
		src_copy.mipLevel = region.src_subresources.mipmap;
		src_copy.origin = { (uint32_t)region.src_offset.x, (uint32_t)region.src_offset.y, region.src_subresources.base_layer };
		src_copy.aspect = WGPUTextureAspect_All;

		WGPUTexelCopyTextureInfo dst_copy = {};
		dst_copy.texture = dst->gpu_handle();
		dst_copy.mipLevel = region.dst_subresources.mipmap;
		dst_copy.origin = { (uint32_t)region.dst_offset.x, (uint32_t)region.dst_offset.y, region.dst_subresources.base_layer };
		dst_copy.aspect = WGPUTextureAspect_All;

		WGPUExtent3D extent = { (uint32_t)region.size.x, (uint32_t)region.size.y, (uint32_t)region.size.z };

		wgpuCommandEncoderCopyTextureToTexture(cmd->encoder, &src_copy, &dst_copy, &extent);
	}
}

void RenderingDeviceDriverWebGPU::command_resolve_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)p_cmd_buffer.id;
	WGTexture *src = (WGTexture *)p_src_texture.id;
	WGTexture *dst = (WGTexture *)p_dst_texture.id;
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(src);
	ERR_FAIL_NULL(dst);
	cmd->end_active_encoder();

	WGPUTextureViewDescriptor view = {};
	view.format = src->format;
	view.dimension = WGPUTextureViewDimension_2D;
	view.aspect = WGPUTextureAspect_All;
	view.mipLevelCount = 1;
	view.arrayLayerCount = 1;
	view.baseMipLevel = src->base_mipmap + p_src_mipmap;
	view.baseArrayLayer = src->base_layer + p_src_layer;
	WGPUTextureView source_view = wgpuTextureCreateView(src->gpu_handle(), &view);
	view.format = dst->format;
	view.baseMipLevel = dst->base_mipmap + p_dst_mipmap;
	view.baseArrayLayer = dst->base_layer + p_dst_layer;
	WGPUTexture intermediate = nullptr;
	if (!(dst->usage & WGPUTextureUsage_RenderAttachment)) {
		// Godot permits a copy-only resolve destination. WebGPU requires a
		// render attachment, so resolve into a temporary and copy it afterward.
		WGPUTextureDescriptor descriptor = {};
		descriptor.dimension = WGPUTextureDimension_2D;
		descriptor.size = { MAX(1u, dst->width >> p_dst_mipmap), MAX(1u, dst->height >> p_dst_mipmap), 1 };
		descriptor.format = dst->format;
		descriptor.mipLevelCount = 1;
		descriptor.sampleCount = 1;
		descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
		intermediate = wgpuDeviceCreateTexture(device, &descriptor);
		view.baseMipLevel = 0;
		view.baseArrayLayer = 0;
	}
	WGPUTextureView target_view = wgpuTextureCreateView(intermediate ? intermediate : dst->gpu_handle(), &view);

	WGPURenderPassColorAttachment color = {};
	color.view = source_view;
	color.resolveTarget = target_view;
	color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
	color.loadOp = WGPULoadOp_Load;
	color.storeOp = WGPUStoreOp_Store;
	WGPURenderPassDescriptor pass = {};
	pass.colorAttachmentCount = 1;
	pass.colorAttachments = &color;
	WGPURenderPassEncoder encoder = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &pass);
	wgpuRenderPassEncoderEnd(encoder);
	wgpuRenderPassEncoderRelease(encoder);
	if (intermediate) {
		WGPUTexelCopyTextureInfo from = {};
		from.texture = intermediate;
		from.aspect = WGPUTextureAspect_All;
		WGPUTexelCopyTextureInfo to = {};
		to.texture = dst->gpu_handle();
		to.mipLevel = dst->base_mipmap + p_dst_mipmap;
		to.origin.z = dst->base_layer + p_dst_layer;
		to.aspect = WGPUTextureAspect_All;
		WGPUExtent3D size = { MAX(1u, dst->width >> p_dst_mipmap), MAX(1u, dst->height >> p_dst_mipmap), 1 };
		wgpuCommandEncoderCopyTextureToTexture(cmd->encoder, &from, &to, &size);
		wgpuTextureRelease(intermediate);
	}
	wgpuTextureViewRelease(target_view);
	wgpuTextureViewRelease(source_view);
}

// Encode a clear Color into the raw texel bytes for a given WGPUTextureFormat.
// r_texel must point to at least 16 bytes and be pre-zeroed; packed formats stay zero.
static void _encode_clear_texel(WGPUTextureFormat p_format, const Color &p_color, uint8_t *r_texel) {
	uint32_t channels = 0;
	WGTexelType type = WG_TEXEL_FLOAT32;
	if (!wgpu_format_texel_info(p_format, channels, type)) {
		return;
	}
	const bool bgra = (p_format == WGPUTextureFormat_BGRA8Unorm || p_format == WGPUTextureFormat_BGRA8UnormSrgb);
	const float components[4] = { bgra ? p_color.b : p_color.r, p_color.g, bgra ? p_color.r : p_color.b, p_color.a };
	for (uint32_t c = 0; c < channels; c++) {
		wg_texel_encode(type, components[c], r_texel + c * wg_texel_component_size(type));
	}
}

void RenderingDeviceDriverWebGPU::command_clear_color_texture(CommandBufferID p_cmd_buffer, TextureID p_texture, TextureLayout p_texture_layout, const Color &p_color, const TextureSubresourceRange &p_subresources) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(tex);

	cmd->end_active_encoder();

	// 3D textures have depth slices, not array layers.
	const bool is_3d = (tex->dimension == WGPUTextureDimension_3D);

	if (tex->usage & WGPUTextureUsage_RenderAttachment) {
		// Fast path: clear via a zero-draw render pass (requires RenderAttachment usage).
		for (uint32_t mip = p_subresources.base_mipmap; mip < p_subresources.base_mipmap + p_subresources.mipmap_count; mip++) {
			const uint32_t slices = is_3d ? MAX(1u, tex->depth >> mip) : 1;
			for (uint32_t layer = p_subresources.base_layer; layer < p_subresources.base_layer + p_subresources.layer_count; layer++) {
				WGPUTextureViewDescriptor view_desc = {};
				view_desc.format = tex->format;
				view_desc.dimension = is_3d ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D;
				view_desc.baseMipLevel = mip;
				view_desc.mipLevelCount = 1;
				view_desc.baseArrayLayer = is_3d ? 0 : layer;
				view_desc.arrayLayerCount = 1;
				view_desc.aspect = WGPUTextureAspect_All;

				WGPUTextureView view = wgpuTextureCreateView(tex->view_source, &view_desc);

				for (uint32_t slice = 0; slice < slices; slice++) {
					WGPURenderPassColorAttachment color_att = {};
					color_att.view = view;
					color_att.depthSlice = is_3d ? slice : WGPU_DEPTH_SLICE_UNDEFINED;
					color_att.loadOp = WGPULoadOp_Clear;
					color_att.storeOp = WGPUStoreOp_Store;
					color_att.clearValue = { p_color.r, p_color.g, p_color.b, p_color.a };

					WGPURenderPassDescriptor rp_desc = {};
					rp_desc.colorAttachmentCount = 1;
					rp_desc.colorAttachments = &color_att;

					WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &rp_desc);
					wgpuRenderPassEncoderEnd(pass);
					wgpuRenderPassEncoderRelease(pass);
				}
				wgpuTextureViewRelease(view);
			}
		}
		return;
	}

	// Textures without RenderAttachment usage (e.g. storage-only compute textures): copy a
	// buffer filled with the clear color, recorded in the encoder so it stays in command order.
	ERR_FAIL_COND_MSG(!(tex->usage & WGPUTextureUsage_CopyDst),
			"Cannot clear texture: texture has neither RenderAttachment nor CopyDst usage.");

	const uint32_t bpp = wgpu_format_pixel_size(tex->format);
	ERR_FAIL_COND_MSG(bpp == 0, "Cannot clear texture via copy: unsupported format.");

	uint8_t texel[16] = {};
	_encode_clear_texel(tex->format, p_color, texel);
	bool is_zero = true;
	for (uint32_t i = 0; i < bpp; i++) {
		is_zero = is_zero && texel[i] == 0;
	}

	for (uint32_t mip = p_subresources.base_mipmap; mip < p_subresources.base_mipmap + p_subresources.mipmap_count; mip++) {
		const uint32_t w = MAX(1u, tex->width >> mip);
		const uint32_t h = MAX(1u, tex->height >> mip);
		const uint32_t images = is_3d ? MAX(1u, tex->depth >> mip) : p_subresources.layer_count;
		const uint32_t bytes_per_row = STEPIFY(w * bpp, 256u);

		// New buffers are zero-initialized, so only non-zero colors need filling.
		WGPUBufferDescriptor buf_desc = {};
		buf_desc.size = (uint64_t)bytes_per_row * h * images;
		buf_desc.usage = WGPUBufferUsage_CopySrc;
		buf_desc.mappedAtCreation = !is_zero;
		WGPUBuffer fill = wgpuDeviceCreateBuffer(device, &buf_desc);
		ERR_FAIL_NULL_MSG(fill, "WebGPU: failed to create clear staging buffer.");
		if (!is_zero) {
			uint8_t *ptr = (uint8_t *)wgpuBufferGetMappedRange(fill, 0, buf_desc.size);
			if (ptr) {
				for (uint32_t row = 0; row < h * images; row++) {
					for (uint32_t x = 0; x < w; x++) {
						memcpy(ptr + (uint64_t)row * bytes_per_row + x * bpp, texel, bpp);
					}
				}
			}
			wgpuBufferUnmap(fill);
		}

		WGPUTexelCopyBufferInfo src = {};
		src.buffer = fill;
		src.layout.bytesPerRow = bytes_per_row;
		src.layout.rowsPerImage = h;

		WGPUTexelCopyTextureInfo dst = {};
		dst.texture = tex->gpu_handle();
		dst.mipLevel = mip;
		dst.origin = { 0, 0, is_3d ? 0 : p_subresources.base_layer };
		dst.aspect = WGPUTextureAspect_All;

		WGPUExtent3D extent = { w, h, images };
		wgpuCommandEncoderCopyBufferToTexture(cmd->encoder, &src, &dst, &extent);
		wgpuBufferRelease(fill); // The recorded copy keeps it alive.
	}
}

void RenderingDeviceDriverWebGPU::command_clear_depth_stencil_texture(CommandBufferID p_cmd_buffer, TextureID p_texture, TextureLayout p_texture_layout, float p_depth, uint8_t p_stencil, const TextureSubresourceRange &p_subresources) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGTexture *tex = (WGTexture *)(p_texture.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(tex);

	cmd->end_active_encoder();

	bool has_depth = is_depth_format_wgpu(tex->format);
	bool has_stencil = has_stencil_wgpu(tex->format);

	for (uint32_t mip = p_subresources.base_mipmap; mip < p_subresources.base_mipmap + p_subresources.mipmap_count; mip++) {
		for (uint32_t layer = p_subresources.base_layer; layer < p_subresources.base_layer + p_subresources.layer_count; layer++) {
			WGPUTextureViewDescriptor view_desc = {};
			view_desc.format = tex->format;
			view_desc.dimension = WGPUTextureViewDimension_2D;
			view_desc.baseMipLevel = mip;
			view_desc.mipLevelCount = 1;
			view_desc.baseArrayLayer = layer;
			view_desc.arrayLayerCount = 1;
			view_desc.aspect = WGPUTextureAspect_All;

			WGPUTextureView view = wgpuTextureCreateView(tex->view_source, &view_desc);

			WGPURenderPassDepthStencilAttachment ds_att = {};
			ds_att.view = view;
			if (has_depth) {
				ds_att.depthLoadOp = WGPULoadOp_Clear;
				ds_att.depthStoreOp = WGPUStoreOp_Store;
				ds_att.depthClearValue = p_depth;
			} else {
				ds_att.depthLoadOp = WGPULoadOp_Undefined;
				ds_att.depthStoreOp = WGPUStoreOp_Undefined;
				ds_att.depthReadOnly = true;
			}
			if (has_stencil) {
				ds_att.stencilLoadOp = WGPULoadOp_Clear;
				ds_att.stencilStoreOp = WGPUStoreOp_Store;
				ds_att.stencilClearValue = p_stencil;
			} else {
				ds_att.stencilLoadOp = WGPULoadOp_Undefined;
				ds_att.stencilStoreOp = WGPUStoreOp_Undefined;
				ds_att.stencilReadOnly = true;
			}

			WGPURenderPassDescriptor rp_desc = {};
			rp_desc.depthStencilAttachment = &ds_att;

			WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &rp_desc);
			wgpuRenderPassEncoderEnd(pass);
			wgpuRenderPassEncoderRelease(pass);
			wgpuTextureViewRelease(view);
		}
	}
}

// Records one buffer→texture copy covering p_depth_or_layers images. CPU-staged sources are flushed with
// wgpuQueueWriteBuffer: RD never reuses a staging region within a frame, and the encoder copy keeps
// the texture write in command order (queue.writeTexture would run before already-recorded work).
void RenderingDeviceDriverWebGPU::_copy_buffer_to_texture(WGCommandBuffer *p_cmd_buf, WGBuffer *p_src, WGTexture *p_dst, const BufferTextureCopyRegion &p_region, uint32_t p_depth_or_layers) {
	// Compressed formats must pass a block-aligned copy extent. For mip levels
	// smaller than the block size (e.g. a 2x2 mip of a BC3 texture), the physical
	// size is the next block multiple, and WebGPU rejects non-multiple extents.
	uint32_t block_w = 1, block_h = 1;
	if (p_dst->rd_format != DATA_FORMAT_MAX) {
		get_compressed_image_format_block_dimensions(p_dst->rd_format, block_w, block_h);
	}
	const uint32_t copy_w = STEPIFY((uint32_t)p_region.texture_region_size.x, block_w);
	const uint32_t copy_h = STEPIFY((uint32_t)p_region.texture_region_size.y, block_h);

	WGPUTexelCopyBufferInfo src_info = {};
	src_info.buffer = p_src->handle;
	src_info.layout.offset = p_region.buffer_offset;
	src_info.layout.bytesPerRow = STEPIFY(p_region.row_pitch, 256u);
	src_info.layout.rowsPerImage = copy_h / block_h;

	if (p_src->shadow_map) {
		const uint64_t size = MIN((uint64_t)src_info.layout.bytesPerRow * src_info.layout.rowsPerImage * p_depth_or_layers, p_src->size - p_region.buffer_offset);
		wgpuQueueWriteBuffer(queue, p_src->handle, p_region.buffer_offset, p_src->shadow_map + p_region.buffer_offset, STEPIFY(size, 4u));
	}

	WGPUTexelCopyTextureInfo dst_copy = {};
	dst_copy.texture = p_dst->gpu_handle();
	dst_copy.mipLevel = p_region.texture_subresource.mipmap;
	// For 3D textures origin.z is a depth slice, otherwise the first array layer.
	dst_copy.origin = { (uint32_t)p_region.texture_offset.x, (uint32_t)p_region.texture_offset.y,
		p_dst->dimension == WGPUTextureDimension_3D ? (uint32_t)p_region.texture_offset.z : p_region.texture_subresource.layer };
	dst_copy.aspect = WGPUTextureAspect_All;

	const WGPUExtent3D extent = { copy_w, copy_h, p_depth_or_layers };
	wgpuCommandEncoderCopyBufferToTexture(p_cmd_buf->encoder, &src_info, &dst_copy, &extent);
}

void RenderingDeviceDriverWebGPU::command_copy_buffer_to_texture(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<BufferTextureCopyRegion> p_regions) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGBuffer *src = (WGBuffer *)(p_src_buffer.id);
	WGTexture *dst = (WGTexture *)(p_dst_texture.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(src);
	ERR_FAIL_NULL(dst);

	cmd->end_active_encoder();

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		_copy_buffer_to_texture(cmd, src, dst, p_regions[i], MAX(1, p_regions[i].texture_region_size.z));
	}

	// The copied regions were flushed above; skip buffer_unmap()'s full-buffer flush.
	src->map_dirty = false;
}

// Direct CPU->GPU multi-layer texture write, outside command recording (RD calls it at texture
// creation, before any recorded use), so queue.writeTexture ordering is safe here. Takes a CPU
// pointer directly instead of going through a transfer worker's GPU staging buffer + shadow_map.
//
// Save vs transfer-worker path on tier 1024 (75 layers, 300 MB total):
//   - 1x wgpuDeviceCreateBuffer(300 MB) eliminated (peak VRAM -300 MB)
//   - 1x command encoder begin/end eliminated
//   - 1x pipeline-barrier no-op eliminated
//   - The single multi-layer wgpuQueueWriteTexture is the same as before.
void RenderingDeviceDriverWebGPU::texture_initialize_direct_layered(TextureID p_dst_texture, TextureLayout p_dst_layout, const uint8_t *p_cpu_data, uint64_t p_total_size, uint32_t p_aligned_bpr, uint32_t p_rows_per_image, uint32_t p_width, uint32_t p_height, uint32_t p_layer_count, uint32_t p_base_layer, uint32_t p_mip_level) {
	WGTexture *dst = (WGTexture *)(p_dst_texture.id);
	ERR_FAIL_NULL(dst);
	ERR_FAIL_NULL(p_cpu_data);
	if (p_layer_count == 0 || p_total_size == 0) {
		return;
	}

	// Block-aligned copy extent (matches command_copy_buffer_to_texture).
	uint32_t block_w = 1, block_h = 1;
	if (dst->rd_format != DATA_FORMAT_MAX) {
		get_compressed_image_format_block_dimensions(dst->rd_format, block_w, block_h);
	}
	uint32_t copy_w = p_width;
	uint32_t copy_h = p_height;
	if (block_w > 1 || block_h > 1) {
		copy_w = ((copy_w + block_w - 1) / block_w) * block_w;
		copy_h = ((copy_h + block_h - 1) / block_h) * block_h;
	}

	WGPUTexelCopyTextureInfo dst_copy = {};
	dst_copy.texture = dst->gpu_handle();
	dst_copy.mipLevel = p_mip_level;
	dst_copy.origin = { 0, 0, p_base_layer };
	dst_copy.aspect = WGPUTextureAspect_All;

	WGPUExtent3D extent = { copy_w, copy_h, p_layer_count };

	WGPUTexelCopyBufferLayout layout = {};
	layout.offset = 0;
	layout.bytesPerRow = p_aligned_bpr;
	layout.rowsPerImage = p_rows_per_image;

	wgpuQueueWriteTexture(queue, &dst_copy, p_cpu_data, p_total_size, &layout, &extent);
}

void RenderingDeviceDriverWebGPU::command_copy_texture_to_buffer(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, BufferID p_dst_buffer, VectorView<BufferTextureCopyRegion> p_regions) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGTexture *src = (WGTexture *)(p_src_texture.id);
	WGBuffer *dst = (WGBuffer *)(p_dst_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(src);
	ERR_FAIL_NULL(dst);

	cmd->end_active_encoder();

	// Block-alignment: see _copy_buffer_to_texture for the rationale.
	uint32_t block_w = 1, block_h = 1;
	if (src->rd_format != DATA_FORMAT_MAX) {
		get_compressed_image_format_block_dimensions(src->rd_format, block_w, block_h);
	}

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		const BufferTextureCopyRegion &region = p_regions[i];

		uint32_t copy_w = (uint32_t)region.texture_region_size.x;
		uint32_t copy_h = (uint32_t)region.texture_region_size.y;
		if (block_w > 1 || block_h > 1) {
			copy_w = ((copy_w + block_w - 1) / block_w) * block_w;
			copy_h = ((copy_h + block_h - 1) / block_h) * block_h;
		}

		WGPUTexelCopyTextureInfo src_copy = {};
		src_copy.texture = src->gpu_handle();
		src_copy.mipLevel = region.texture_subresource.mipmap;
		src_copy.origin = { (uint32_t)region.texture_offset.x, (uint32_t)region.texture_offset.y,
			src->dimension == WGPUTextureDimension_3D ? (uint32_t)region.texture_offset.z : region.texture_subresource.layer };
		src_copy.aspect = WGPUTextureAspect_All;

		// RD passes the row pitch from texture_get_copyable_layout (GPU format, block rows).
		uint32_t row_bytes = region.row_pitch;
		if (row_bytes == 0) {
			row_bytes = (block_w > 1 || block_h > 1)
					? (copy_w / block_w) * get_compressed_image_format_block_byte_size(src->rd_format)
					: copy_w * wgpu_format_pixel_size(src->format);
		}
		WGPUTexelCopyBufferInfo dst_info = {};
		dst_info.buffer = dst->handle;
		dst_info.layout.offset = region.buffer_offset;
		dst_info.layout.bytesPerRow = STEPIFY(row_bytes, 256u);
		dst_info.layout.rowsPerImage = copy_h / block_h;

		WGPUExtent3D extent = { copy_w, copy_h, (uint32_t)region.texture_region_size.z };

		wgpuCommandEncoderCopyTextureToBuffer(cmd->encoder, &src_copy, &dst_info, &extent);
	}
}

// =============================================================================
// PIPELINE
// =============================================================================

void RenderingDeviceDriverWebGPU::pipeline_free(PipelineID p_pipeline) {
	WGPipelineWrapper *pw = (WGPipelineWrapper *)(p_pipeline.id);
	ERR_FAIL_NULL(pw);
	if (pw->deferred) {
		memdelete(pw->deferred);
	}
	if (pw->type == WGPipelineWrapper::RENDER && pw->render_handle) {
		wgpuRenderPipelineRelease(pw->render_handle);
		if (pw->render_handle_u16) {
			wgpuRenderPipelineRelease(pw->render_handle_u16);
		}
	} else if (pw->type == WGPipelineWrapper::COMPUTE && pw->compute_handle) {
		wgpuComputePipelineRelease(pw->compute_handle);
	}
	// Release any specialized shader modules owned by this pipeline.
	for (int i = 0; i < 6; i++) {
		if (pw->specialized_modules[i]) {
			wgpuShaderModuleRelease(pw->specialized_modules[i]);
		}
	}
	memdelete(pw);
}

void RenderingDeviceDriverWebGPU::command_bind_push_constants(CommandBufferID p_cmd_buffer, ShaderID p_shader, uint32_t p_first_index, VectorView<uint32_t> p_data) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);

	uint32_t offset = p_first_index * sizeof(uint32_t);
	uint32_t size = p_data.size() * sizeof(uint32_t);
	ERR_FAIL_COND(offset + size > WGCommandBuffer::MAX_PUSH_CONSTANT_SIZE);

	memcpy(cmd->push_constant_data + offset, &p_data[0], size);
	cmd->push_constant_data_len = MAX(cmd->push_constant_data_len, offset + size);
	cmd->push_constants_dirty = true;
}

void RenderingDeviceDriverWebGPU::_flush_push_constant_shadow() {
	// Page 0 lives in the ring itself; later pages go to their staging buffers.
	uint64_t start = push_constant_shadow_dirty_start;
	while (start < push_constant_shadow_dirty_end) {
		const uint32_t page = start / PUSH_CONSTANT_RING_SIZE;
		const uint64_t page_end = MIN((uint64_t)(page + 1) * PUSH_CONSTANT_RING_SIZE, push_constant_shadow_dirty_end);
		WGPUBuffer target = page == 0 ? push_constant_ring_buffer : push_constant_pages[page - 1];
		wgpuQueueWriteBuffer(queue, target, start - (uint64_t)page * PUSH_CONSTANT_RING_SIZE, push_constant_shadow.ptr() + start, page_end - start);
		start = page_end;
	}
	push_constant_shadow_dirty_start = UINT64_MAX;
	push_constant_shadow_dirty_end = 0;
}

void RenderingDeviceDriverWebGPU::_flush_push_constants(WGCommandBuffer *p_cmd_buf, WGShader *p_shader) {
	if (!p_cmd_buf->push_constants_dirty || p_cmd_buf->push_constant_data_len == 0 || !p_shader) {
		perf.push_constant_skipped++;
		return;
	}
	if (p_shader->push_constant_bind_group == UINT32_MAX || !push_constant_bind_group) {
		p_cmd_buf->push_constants_dirty = false;
		perf.push_constant_skipped++;
		return; // Shader has no push constants.
	}

	perf.push_constant_writes++;

	// Write push constant data to CPU shadow buffer (batched GPU write at submit time).
	uint32_t aligned_size = (p_cmd_buf->push_constant_data_len + PUSH_CONSTANT_SLOT_ALIGNMENT - 1) & ~(PUSH_CONSTANT_SLOT_ALIGNMENT - 1);

	if (push_constant_ring_offset + aligned_size > PUSH_CONSTANT_RING_SIZE) {
		// The ring is full: start a new page instead of submitting mid-recording (which would run
		// this buffer before earlier, still unsubmitted ones). The page is staged in its own buffer
		// and copied into the ring here, after every recorded draw that reads the previous page.
		perf.ring_overflows++;
		if (push_constant_pages.size() <= push_constant_page) {
			WGPUBufferDescriptor desc = {};
			desc.size = PUSH_CONSTANT_RING_SIZE;
			desc.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
			WGPUBuffer page = wgpuDeviceCreateBuffer(device, &desc);
			ERR_FAIL_NULL_MSG(page, "WebGPU: failed to allocate a push constant page.");
			push_constant_pages.push_back(page);
		}
		const bool was_compute = (p_cmd_buf->compute_encoder != nullptr);
		const bool was_render = (p_cmd_buf->render_encoder != nullptr);
		p_cmd_buf->end_active_encoder();

		push_constant_page++;
		if (push_constant_shadow.size() < (push_constant_page + 1) * PUSH_CONSTANT_RING_SIZE) {
			push_constant_shadow.resize((push_constant_page + 1) * PUSH_CONSTANT_RING_SIZE);
		}
		wgpuCommandEncoderCopyBufferToBuffer(p_cmd_buf->encoder, push_constant_pages[push_constant_page - 1], 0, push_constant_ring_buffer, 0, PUSH_CONSTANT_RING_SIZE);
		push_constant_ring_offset = 0;

		if (was_compute || was_render) {
			_restart_pass_after_push_constant_page(p_cmd_buf, was_compute);
		}
	}

	const uint64_t shadow_offset = (uint64_t)push_constant_page * PUSH_CONSTANT_RING_SIZE + push_constant_ring_offset;
	memcpy(push_constant_shadow.ptr() + shadow_offset, p_cmd_buf->push_constant_data, p_cmd_buf->push_constant_data_len);

	// Track dirty range for batched flush.
	push_constant_shadow_dirty_start = MIN(push_constant_shadow_dirty_start, shadow_offset);
	push_constant_shadow_dirty_end = MAX(push_constant_shadow_dirty_end, shadow_offset + aligned_size);

	uint32_t dynamic_offset = push_constant_ring_offset;

	// Choose which bind group to rebind at the PC group:
	// - If the shader has a merged layout (material + PC) AND the material uniform set
	//   was previously bound, use that combined bind group.
	// - Otherwise, fall back to the universal PC-only bind group.
	WGPUBindGroup pc_bind_group_to_use;
	bool use_merged = (p_shader->merged_pc_group_layout && p_cmd_buf->current_pc_bind_group != nullptr);
	if (use_merged) {
		pc_bind_group_to_use = p_cmd_buf->current_pc_bind_group;
	} else {
		pc_bind_group_to_use = push_constant_bind_group;
	}

	// Task 7.5: If the merged PC group also contains material dynamic-offset UBOs,
	// we need to preserve those offsets while patching in the new PC ring offset.
	// command_bind_*_uniform_sets saved the material offsets in
	// pc_group_material_dyn_offsets[] in binding order; the PC ring's own dynamic
	// offset is appended last (the PC ring binding is always the highest binding
	// in the merged group — see rendering_shader_container_webgpu.cpp).
	uint32_t dyn_offsets[WGCommandBuffer::MAX_PC_GROUP_MATERIAL_DYN + 1];
	uint32_t num_dyn_offsets;
	if (use_merged && p_cmd_buf->pc_group_material_dyn_count > 0) {
		uint32_t mat_count = p_cmd_buf->pc_group_material_dyn_count;
		if (mat_count > WGCommandBuffer::MAX_PC_GROUP_MATERIAL_DYN) {
			mat_count = WGCommandBuffer::MAX_PC_GROUP_MATERIAL_DYN;
		}
		for (uint32_t j = 0; j < mat_count; j++) {
			dyn_offsets[j] = p_cmd_buf->pc_group_material_dyn_offsets[j];
		}
		dyn_offsets[mat_count] = dynamic_offset;
		num_dyn_offsets = mat_count + 1;
	} else {
		dyn_offsets[0] = dynamic_offset;
		num_dyn_offsets = 1;
	}

	// Bind the push constant ring buffer bind group with the assembled dynamic offsets.
	if (p_cmd_buf->render_encoder) {
		wgpuRenderPassEncoderSetBindGroup(p_cmd_buf->render_encoder, p_shader->push_constant_bind_group, pc_bind_group_to_use, num_dyn_offsets, dyn_offsets);
	} else if (p_cmd_buf->compute_encoder) {
		wgpuComputePassEncoderSetBindGroup(p_cmd_buf->compute_encoder, p_shader->push_constant_bind_group, pc_bind_group_to_use, num_dyn_offsets, dyn_offsets);
	}

	p_cmd_buf->last_flushed_pc_offset = push_constant_ring_offset;
	push_constant_ring_offset += aligned_size;
	p_cmd_buf->push_constants_dirty = false;
}

// Re-opens the pass that a new push constant page ended, with all bound state.
// shortcut: attachments the original pass stored with Discard (e.g. MSAA colour) lose the draws
// before the page switch; only hit beyond PUSH_CONSTANT_RING_SIZE of push constants per frame.
void RenderingDeviceDriverWebGPU::_restart_pass_after_push_constant_page(WGCommandBuffer *p_cmd_buf, bool p_compute) {
	if (p_compute) {
		// Restart compute pass and rebind state.
		WGPUComputePassDescriptor cp_desc = {};
		p_cmd_buf->compute_encoder = wgpuCommandEncoderBeginComputePass(p_cmd_buf->encoder, &cp_desc);
		p_cmd_buf->active_encoder = WGCommandBuffer::COMPUTE;

		// Rebind compute pipeline.
		WGPipelineWrapper *pw = p_cmd_buf->render_state.current_pipeline;
		if (pw) {
			wgpuComputePassEncoderSetPipeline(p_cmd_buf->compute_encoder, pw->compute_handle);
			if (pw->shader) {
				for (uint32_t gap_idx : pw->shader->gap_bind_group_indices) {
					wgpuComputePassEncoderSetBindGroup(p_cmd_buf->compute_encoder, gap_idx, empty_bind_group, 0, nullptr);
				}
			}
		}

		// Rebind all bind groups from saved state.
		for (uint32_t i = 0; i < WGCommandBuffer::MAX_BIND_GROUPS; i++) {
			const auto &bs = p_cmd_buf->last_bound_state[i];
			if (bs.group) {
				wgpuComputePassEncoderSetBindGroup(p_cmd_buf->compute_encoder, i, bs.group, bs.dynamic_offset_count, bs.dynamic_offsets);
			}
		}

		// Invalidate redundancy caches.
		for (uint32_t i = 0; i < WGCommandBuffer::MAX_BIND_GROUPS; i++) {
			p_cmd_buf->bound_bind_groups[i] = nullptr;
		}
	} else {
		// Restart render pass with LoadOp::Load to preserve framebuffer content.
		WGRenderPass *rp = p_cmd_buf->render_state.render_pass;
		WGFramebuffer *fb = p_cmd_buf->render_state.framebuffer;
		if (rp && fb) {
			const WGRenderPass::SubpassInfo &subpass = rp->subpasses[p_cmd_buf->render_state.current_subpass];

			LocalVector<WGPURenderPassColorAttachment> color_attachments;
			for (uint32_t i = 0; i < subpass.color_references.size(); i++) {
				const RDD::AttachmentReference &ref = subpass.color_references[i];
				WGPURenderPassColorAttachment att = {};
				att.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

				if (ref.attachment == RDD::AttachmentReference::UNUSED) {
					att.view = nullptr;
					att.loadOp = WGPULoadOp_Load;
					att.storeOp = WGPUStoreOp_Discard;
					color_attachments.push_back(att);
					continue;
				}

				if (ref.attachment < fb->attachment_views.size()) {
					att.view = fb->attachment_views[ref.attachment];
				}
				if (i < subpass.resolve_references.size()) {
					const RDD::AttachmentReference &res_ref = subpass.resolve_references[i];
					if (res_ref.attachment != RDD::AttachmentReference::UNUSED &&
							res_ref.attachment < fb->attachment_views.size()) {
						att.resolveTarget = fb->attachment_views[res_ref.attachment];
					}
				}
				att.loadOp = WGPULoadOp_Load;
				att.storeOp = WGPUStoreOp_Store;
				color_attachments.push_back(att);
			}

			WGPURenderPassDepthStencilAttachment ds_att = {};
			WGPURenderPassDepthStencilAttachment *ds_att_ptr = nullptr;
			const RDD::AttachmentReference &ds_ref = subpass.depth_stencil_reference;
			if (ds_ref.attachment != RDD::AttachmentReference::UNUSED &&
					ds_ref.attachment < fb->attachment_views.size()) {
				ds_att.view = fb->attachment_views[ds_ref.attachment];
				if (ds_ref.attachment < rp->attachments.size()) {
					WGPUTextureFormat wgpu_fmt = _data_format_to_wgpu(rp->attachments[ds_ref.attachment].format);
					bool has_depth = is_depth_format_wgpu(wgpu_fmt);
					bool has_stencil = has_stencil_wgpu(wgpu_fmt);
					if (has_depth) {
						ds_att.depthLoadOp = WGPULoadOp_Load;
						ds_att.depthStoreOp = WGPUStoreOp_Store;
					} else {
						ds_att.depthLoadOp = WGPULoadOp_Undefined;
						ds_att.depthStoreOp = WGPUStoreOp_Undefined;
						ds_att.depthReadOnly = true;
					}
					if (has_stencil) {
						ds_att.stencilLoadOp = WGPULoadOp_Load;
						ds_att.stencilStoreOp = WGPUStoreOp_Store;
					} else {
						ds_att.stencilLoadOp = WGPULoadOp_Undefined;
						ds_att.stencilStoreOp = WGPUStoreOp_Undefined;
						ds_att.stencilReadOnly = true;
					}
				} else {
					ds_att.depthLoadOp = WGPULoadOp_Load;
					ds_att.depthStoreOp = WGPUStoreOp_Store;
					ds_att.depthClearValue = 1.0f;
					ds_att.stencilLoadOp = WGPULoadOp_Load;
					ds_att.stencilStoreOp = WGPUStoreOp_Store;
				}
				ds_att_ptr = &ds_att;
			}

			WGPURenderPassDescriptor pass_desc = {};
			pass_desc.colorAttachmentCount = color_attachments.size();
			pass_desc.colorAttachments = color_attachments.ptr();
			pass_desc.depthStencilAttachment = ds_att_ptr;

			p_cmd_buf->render_encoder = wgpuCommandEncoderBeginRenderPass(p_cmd_buf->encoder, &pass_desc);
			p_cmd_buf->active_encoder = WGCommandBuffer::RENDER;

			_rebind_render_state(p_cmd_buf);
		}
	}
}

// Re-applies the tracked pipeline, bind groups, buffers and dynamic state to the current render encoder.
void RenderingDeviceDriverWebGPU::_rebind_render_state(WGCommandBuffer *p_cmd_buf) {
	// Rebind render pipeline.
	WGPipelineWrapper *pw = p_cmd_buf->render_state.current_pipeline;
	if (pw) {
		wgpuRenderPassEncoderSetPipeline(p_cmd_buf->render_encoder, pw->render_handle);
		wgpuRenderPassEncoderSetStencilReference(p_cmd_buf->render_encoder, pw->stencil_reference);
		if (pw->shader) {
			for (uint32_t gap_idx : pw->shader->gap_bind_group_indices) {
				wgpuRenderPassEncoderSetBindGroup(p_cmd_buf->render_encoder, gap_idx, empty_bind_group, 0, nullptr);
			}
		}
	}

	// Rebind all bind groups from saved state.
	for (uint32_t i = 0; i < WGCommandBuffer::MAX_BIND_GROUPS; i++) {
		const auto &bs = p_cmd_buf->last_bound_state[i];
		if (bs.group) {
			wgpuRenderPassEncoderSetBindGroup(p_cmd_buf->render_encoder, i, bs.group, bs.dynamic_offset_count, bs.dynamic_offsets);
		}
	}

	// Rebind vertex buffers.
	for (uint32_t i = 0; i < WGCommandBuffer::RenderState::MAX_VERTEX_BINDINGS; i++) {
		if (p_cmd_buf->render_state.current_vertex_buffers[i]) {
			wgpuRenderPassEncoderSetVertexBuffer(p_cmd_buf->render_encoder, i,
					p_cmd_buf->render_state.current_vertex_buffers[i],
					p_cmd_buf->render_state.current_vertex_offsets[i], WGPU_WHOLE_SIZE);
		}
	}

	// Rebind index buffer.
	if (p_cmd_buf->render_state.current_index_buffer) {
		wgpuRenderPassEncoderSetIndexBuffer(p_cmd_buf->render_encoder,
				p_cmd_buf->render_state.current_index_buffer,
				p_cmd_buf->render_state.current_index_format,
				p_cmd_buf->render_state.current_index_offset, WGPU_WHOLE_SIZE);
	}

	// A new pass starts with default viewport, scissor and blend constant.
	const WGCommandBuffer::RenderState &rs = p_cmd_buf->render_state;
	if (rs.viewport_set) {
		wgpuRenderPassEncoderSetViewport(p_cmd_buf->render_encoder, rs.viewport[0], rs.viewport[1], rs.viewport[2], rs.viewport[3], 0.0f, 1.0f);
	}
	if (rs.scissor_set) {
		wgpuRenderPassEncoderSetScissorRect(p_cmd_buf->render_encoder, rs.scissor[0], rs.scissor[1], rs.scissor[2], rs.scissor[3]);
	}
	if (rs.blend_constant_set) {
		wgpuRenderPassEncoderSetBlendConstant(p_cmd_buf->render_encoder, &rs.blend_constant);
	}

	// Invalidate redundancy caches (state was just rebound fresh).
	for (uint32_t i = 0; i < WGCommandBuffer::MAX_BIND_GROUPS; i++) {
		p_cmd_buf->bound_bind_groups[i] = nullptr;
	}
}

bool RenderingDeviceDriverWebGPU::pipeline_cache_create(const Vector<uint8_t> &p_data) {
	return true; // No-op.
}

void RenderingDeviceDriverWebGPU::pipeline_cache_free() {
	// No-op.
}

size_t RenderingDeviceDriverWebGPU::pipeline_cache_query_size() {
	// We don't implement a pipeline cache, but must return non-zero so
	// RenderingDevice::update_pipeline_cache() doesn't emit an error every frame.
	return 1;
}

Vector<uint8_t> RenderingDeviceDriverWebGPU::pipeline_cache_serialize() {
	return Vector<uint8_t>();
}

// =============================================================================
// RENDERING
// =============================================================================

RDD::RenderPassID RenderingDeviceDriverWebGPU::render_pass_create(VectorView<Attachment> p_attachments, VectorView<Subpass> p_subpasses, VectorView<SubpassDependency> p_subpass_dependencies, uint32_t p_view_count, AttachmentReference p_fragment_density_map_attachment) {
	WGRenderPass *rp = memnew(WGRenderPass);
	rp->view_count = p_view_count;

	for (uint32_t i = 0; i < p_attachments.size(); i++) {
		rp->attachments.push_back(p_attachments[i]);
	}

	for (uint32_t i = 0; i < p_subpasses.size(); i++) {
		WGRenderPass::SubpassInfo sp;
		sp.color_references = p_subpasses[i].color_references;
		sp.input_references = p_subpasses[i].input_references;
		sp.depth_stencil_reference = p_subpasses[i].depth_stencil_reference;
		sp.resolve_references = p_subpasses[i].resolve_references;
		rp->subpasses.push_back(sp);
	}

	return RenderPassID(rp);
}

void RenderingDeviceDriverWebGPU::render_pass_free(RenderPassID p_render_pass) {
	WGRenderPass *rp = (WGRenderPass *)(p_render_pass.id);
	ERR_FAIL_NULL(rp);
	memdelete(rp);
}

// The size WebGPU validates viewports and scissors against. fb->width/height may be the Godot-level
// render area (window size) rather than the attachment size.
static Size2i _framebuffer_extent(const WGFramebuffer *p_fb) {
	if (!p_fb->attachments.is_empty() && p_fb->attachments[0] != nullptr) {
		return Size2i(p_fb->attachments[0]->width, p_fb->attachments[0]->height);
	}
	return Size2i(p_fb->width, p_fb->height);
}

void RenderingDeviceDriverWebGPU::command_begin_render_pass(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, FramebufferID p_framebuffer, CommandBufferType p_cmd_buffer_type, const Rect2i &p_rect, VectorView<RenderPassClearValue> p_clear_values) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGRenderPass *rp = (WGRenderPass *)(p_render_pass.id);
	WGFramebuffer *fb = (WGFramebuffer *)(p_framebuffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(rp);
	ERR_FAIL_NULL(fb);

	perf.render_passes++;

	// End any active encoder.
	cmd->end_active_encoder();

	// Invalidate bind group state tracking (new encoder = clean state).
	cmd->invalidate_bind_groups();

	// Store render state.
	cmd->render_state.render_pass = rp;
	cmd->render_state.framebuffer = fb;
	cmd->render_state.current_subpass = 0;
	cmd->render_state.render_area_width = p_rect.size.x > 0 ? (uint32_t)p_rect.get_end().x : fb->width;
	cmd->render_state.render_area_height = p_rect.size.y > 0 ? (uint32_t)p_rect.get_end().y : fb->height;

	// WebGPU has no render area: a partial one loads and stores every attachment and its clears are
	// drawn into p_rect after the pass begins, like the Metal driver does.
	const bool full_area = p_rect.size == Size2i() || p_rect.encloses(Rect2i(Point2i(), _framebuffer_extent(fb)));
	AttachmentClear region_clears[CLEAR_MAX_COLORS + 1];
	uint32_t region_clear_count = 0;

	ERR_FAIL_COND(rp->subpasses.size() == 0);
	const WGRenderPass::SubpassInfo &subpass = rp->subpasses[0];

	// Track ONLY subpass-referenced attachments for intra-pass conflict detection.
	// Only textures in the actual render pass descriptor create sync scope constraints.
	cmd->render_state.reset_current_pass_attachments();
	for (uint32_t i = 0; i < subpass.color_references.size(); i++) {
		uint32_t att_idx = subpass.color_references[i].attachment;
		if (att_idx != RDD::AttachmentReference::UNUSED && att_idx < fb->attachments.size()) {
			WGTexture *tex = fb->attachments[att_idx];
			if (tex) {
				cmd->render_state.add_current_pass_attachment(tex->gpu_handle());
			}
		}
	}
	if (subpass.depth_stencil_reference.attachment != RDD::AttachmentReference::UNUSED &&
			subpass.depth_stencil_reference.attachment < fb->attachments.size()) {
		WGTexture *tex = fb->attachments[subpass.depth_stencil_reference.attachment];
		if (tex) {
			cmd->render_state.add_current_pass_attachment(tex->gpu_handle());
		}
	}

	// Keep commands in Godot's submission order. Submitting here would execute
	// this buffer ahead of earlier, completed buffers in the render graph chain.

	// --- Helper lambdas for op mapping ---
	auto map_load_op = [](AttachmentLoadOp op) -> WGPULoadOp {
		switch (op) {
			case ATTACHMENT_LOAD_OP_LOAD:
				return WGPULoadOp_Load;
			case ATTACHMENT_LOAD_OP_CLEAR:
				return WGPULoadOp_Clear;
			default:
				return WGPULoadOp_Clear; // DONT_CARE → Clear (WebGPU has no DONT_CARE; Clear is safest)
		}
	};
	auto map_store_op = [](AttachmentStoreOp op) -> WGPUStoreOp {
		switch (op) {
			case ATTACHMENT_STORE_OP_STORE:
				return WGPUStoreOp_Store;
			default:
				return WGPUStoreOp_Discard; // DONT_CARE
		}
	};

	// --- Build color attachments ---
	// clear_values is indexed by attachment index, matching Vulkan/Godot convention.
	LocalVector<WGPURenderPassColorAttachment> color_attachments;
	for (uint32_t i = 0; i < subpass.color_references.size(); i++) {
		const RDD::AttachmentReference &ref = subpass.color_references[i];

		WGPURenderPassColorAttachment att = {};
		att.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

		if (ref.attachment == RDD::AttachmentReference::UNUSED) {
			// Unused attachment slot: provide a dummy null entry.
			att.view = nullptr;
			att.loadOp = WGPULoadOp_Load;
			att.storeOp = WGPUStoreOp_Discard;
			color_attachments.push_back(att);
			continue;
		}

		// Attachment view from framebuffer.
		if (ref.attachment < fb->attachment_views.size()) {
			att.view = fb->attachment_views[ref.attachment];
		}

		// Resolve target (MSAA).
		if (i < subpass.resolve_references.size()) {
			const RDD::AttachmentReference &res_ref = subpass.resolve_references[i];
			if (res_ref.attachment != RDD::AttachmentReference::UNUSED &&
					res_ref.attachment < fb->attachment_views.size()) {
				att.resolveTarget = fb->attachment_views[res_ref.attachment];
			}
		}

		// Load/store ops and clear value from attachment description.
		if (ref.attachment < rp->attachments.size()) {
			const RDD::Attachment &attach_desc = rp->attachments[ref.attachment];
			att.loadOp = map_load_op(attach_desc.load_op);
			// WebGPU emulates Vulkan subpasses as separate render passes.
			// For the first subpass in a multi-subpass pass, force Store to
			// preserve MSAA data for subsequent subpasses (e.g. transparent pass).
			if (rp->subpasses.size() > 1) {
				att.storeOp = WGPUStoreOp_Store;
			} else {
				att.storeOp = map_store_op(attach_desc.store_op);
			}

			if (att.loadOp == WGPULoadOp_Clear && ref.attachment < p_clear_values.size()) {
				const Color &c = p_clear_values[ref.attachment].color;
				att.clearValue = { c.r, c.g, c.b, c.a };
			}

			if (!full_area) {
				if (attach_desc.load_op == ATTACHMENT_LOAD_OP_CLEAR && ref.attachment < p_clear_values.size() && i < CLEAR_MAX_COLORS) {
					AttachmentClear &clear = region_clears[region_clear_count++];
					clear.aspect = TEXTURE_ASPECT_COLOR_BIT;
					clear.color_attachment = i;
					clear.value = p_clear_values[ref.attachment];
				}
				att.loadOp = WGPULoadOp_Load;
				att.storeOp = WGPUStoreOp_Store;
			}

			// WebGPU swap chain textures have undefined content each frame.
			// Force LOAD→CLEAR so the alpha channel starts at 1.0 (opaque).
			if (rp->is_swap_chain_pass && att.loadOp == WGPULoadOp_Load) {
				att.loadOp = WGPULoadOp_Clear;
				att.clearValue = { 0.0, 0.0, 0.0, 1.0 };
			}
		} else {
			att.loadOp = WGPULoadOp_Load;
			att.storeOp = WGPUStoreOp_Store;
		}

		color_attachments.push_back(att);
	}

	// --- Build depth/stencil attachment ---
	WGPURenderPassDepthStencilAttachment ds_att = {};
	WGPURenderPassDepthStencilAttachment *ds_att_ptr = nullptr;

	const RDD::AttachmentReference &ds_ref = subpass.depth_stencil_reference;
	if (ds_ref.attachment != RDD::AttachmentReference::UNUSED &&
			ds_ref.attachment < fb->attachment_views.size()) {
		ds_att.view = fb->attachment_views[ds_ref.attachment];

		if (ds_ref.attachment < rp->attachments.size()) {
			const RDD::Attachment &attach_desc = rp->attachments[ds_ref.attachment];
			WGPUTextureFormat wgpu_fmt = _data_format_to_wgpu(attach_desc.format);
			bool has_depth = is_depth_format_wgpu(wgpu_fmt);
			bool has_stencil = has_stencil_wgpu(wgpu_fmt);

			if (has_depth) {
				ds_att.depthLoadOp = map_load_op(attach_desc.load_op);
				// Force Store for multi-subpass passes (see color attachment comment).
				if (rp->subpasses.size() > 1) {
					ds_att.depthStoreOp = WGPUStoreOp_Store;
				} else {
					ds_att.depthStoreOp = map_store_op(attach_desc.store_op);
				}
				if (ds_att.depthLoadOp == WGPULoadOp_Clear && ds_ref.attachment < p_clear_values.size()) {
					ds_att.depthClearValue = p_clear_values[ds_ref.attachment].depth;
				} else {
					ds_att.depthClearValue = 1.0f;
				}
			} else {
				ds_att.depthLoadOp = WGPULoadOp_Undefined;
				ds_att.depthStoreOp = WGPUStoreOp_Undefined;
				ds_att.depthReadOnly = true;
			}

			if (has_stencil) {
				ds_att.stencilLoadOp = map_load_op(attach_desc.stencil_load_op);
				// Force Store for multi-subpass passes (see color attachment comment).
				if (rp->subpasses.size() > 1) {
					ds_att.stencilStoreOp = WGPUStoreOp_Store;
				} else {
					ds_att.stencilStoreOp = map_store_op(attach_desc.stencil_store_op);
				}
				if (ds_att.stencilLoadOp == WGPULoadOp_Clear && ds_ref.attachment < p_clear_values.size()) {
					ds_att.stencilClearValue = p_clear_values[ds_ref.attachment].stencil;
				}
			} else {
				ds_att.stencilLoadOp = WGPULoadOp_Undefined;
				ds_att.stencilStoreOp = WGPUStoreOp_Undefined;
				ds_att.stencilReadOnly = true;
			}

			if (!full_area) {
				AttachmentClear clear;
				if (has_depth) {
					if (attach_desc.load_op == ATTACHMENT_LOAD_OP_CLEAR) {
						clear.aspect.set_flag(TEXTURE_ASPECT_DEPTH_BIT);
					}
					ds_att.depthLoadOp = WGPULoadOp_Load;
					ds_att.depthStoreOp = WGPUStoreOp_Store;
				}
				if (has_stencil) {
					if (attach_desc.stencil_load_op == ATTACHMENT_LOAD_OP_CLEAR) {
						clear.aspect.set_flag(TEXTURE_ASPECT_STENCIL_BIT);
					}
					ds_att.stencilLoadOp = WGPULoadOp_Load;
					ds_att.stencilStoreOp = WGPUStoreOp_Store;
				}
				if (!clear.aspect.is_empty() && ds_ref.attachment < p_clear_values.size()) {
					clear.value = p_clear_values[ds_ref.attachment];
					region_clears[region_clear_count++] = clear;
				}
			}
		} else {
			// Fallback: load and store everything.
			ds_att.depthLoadOp = WGPULoadOp_Load;
			ds_att.depthStoreOp = WGPUStoreOp_Store;
			ds_att.depthClearValue = 1.0f;
			ds_att.stencilLoadOp = WGPULoadOp_Load;
			ds_att.stencilStoreOp = WGPUStoreOp_Store;
		}
		ds_att_ptr = &ds_att;
	}

	// --- Begin render pass ---
	// --- Diagnostic: verify swap chain view maps to correct JS object ---
#if defined(WEBGPU_VERBOSE) && defined(WEB_ENABLED)
	if (rp->is_swap_chain_pass && color_attachments.size() > 0) {
		static int _sc_view_log = 0;
		if (_sc_view_log < 5) {
			WGPUTextureView view = color_attachments[0].view;
			int load_op_int = (int)color_attachments[0].loadOp;
			int store_op_int = (int)color_attachments[0].storeOp;
			EM_ASM({
				var viewPtr = $0;
				var jsView = WebGPU.getJsObject(viewPtr);
				var viewType = jsView ? jsView.constructor.name : 'null';
				var viewLabel = jsView ? jsView.label : 'N/A';
				console.log('[SC-VIEW#' + $3 + '] viewPtr=' + viewPtr +
					' jsType=' + viewType +
					' label=' + viewLabel +
					' loadOp=' + $1 + ' storeOp=' + $2);
				// Check if the view's texture matches the canvas surface texture.
				var canvas = document.querySelector('#canvas');
				if (canvas) {
					var ctx = canvas.getContext('webgpu');
					if (ctx) {
						try {
							var surfTex = ctx.getCurrentTexture();
							var surfView = surfTex.createView();
							console.log('[SC-VIEW#' + $3 + '] surfaceTex=' + surfTex.width + 'x' + surfTex.height +
								' fmt=' + surfTex.format + ' usage=' + surfTex.usage);
						} catch (e) {
							console.log('[SC-VIEW#' + $3 + '] getCurrentTexture error: ' + e.message);
						}
					}
				} }, (int)(uintptr_t)view, load_op_int, store_op_int, _sc_view_log);
			_sc_view_log++;
		}
	}
#endif // WEBGPU_VERBOSE

	WGPURenderPassDescriptor pass_desc = {};
	pass_desc.colorAttachmentCount = color_attachments.size();
	pass_desc.colorAttachments = color_attachments.ptr();
	pass_desc.depthStencilAttachment = ds_att_ptr;

	cmd->render_encoder = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &pass_desc);
	cmd->active_encoder = WGCommandBuffer::RENDER;

	// Reset cached state — new render pass requires binding everything fresh.
	cmd->render_state.viewport_set = false;
	cmd->render_state.scissor_set = false;
	cmd->render_state.blend_constant_set = false;
	cmd->render_state.current_pipeline = nullptr;
	cmd->render_state.current_index_buffer = nullptr;
	cmd->render_state.current_index_offset = 0;
	memset(cmd->render_state.current_vertex_buffers, 0, sizeof(cmd->render_state.current_vertex_buffers));
	memset(cmd->render_state.current_vertex_offsets, 0, sizeof(cmd->render_state.current_vertex_offsets));

	if (region_clear_count > 0) {
		_render_clear(cmd, VectorView(region_clears, region_clear_count), p_rect);
	}
}

void RenderingDeviceDriverWebGPU::command_end_render_pass(CommandBufferID p_cmd_buffer) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);

	// Temporary: log render pass ends to track pass boundaries.
	static int _rp_end_log = 0;
	if (_rp_end_log < 30) {
		WEBGPU_DIAG({ console.log('[RP-END#' + $0 + '] ' + $1 + 'x' + $2); }, _rp_end_log, cmd->render_state.render_area_width, cmd->render_state.render_area_height);
		_rp_end_log++;
	}

	if (cmd->render_encoder) {
		wgpuRenderPassEncoderEnd(cmd->render_encoder);
		wgpuRenderPassEncoderRelease(cmd->render_encoder);
		cmd->render_encoder = nullptr;
	}
	cmd->active_encoder = WGCommandBuffer::NONE;
	cmd->render_state = {};
}

void RenderingDeviceDriverWebGPU::command_next_render_subpass(CommandBufferID p_cmd_buffer, CommandBufferType p_cmd_buffer_type) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);

	// End current render pass encoder.
	if (cmd->render_encoder) {
		wgpuRenderPassEncoderEnd(cmd->render_encoder);
		wgpuRenderPassEncoderRelease(cmd->render_encoder);
		cmd->render_encoder = nullptr;
	}

	// Advance to next subpass.
	cmd->render_state.current_subpass++;

	WGRenderPass *rp = cmd->render_state.render_pass;
	WGFramebuffer *fb = cmd->render_state.framebuffer;
	ERR_FAIL_NULL(rp);
	ERR_FAIL_NULL(fb);
	ERR_FAIL_COND(cmd->render_state.current_subpass >= rp->subpasses.size());

	const WGRenderPass::SubpassInfo &subpass = rp->subpasses[cmd->render_state.current_subpass];

	// --- Build color attachments for the new subpass ---
	// For subsequent subpasses, we LOAD (not clear) color/depth since we want to preserve
	// what was drawn in previous subpasses on shared attachments.
	LocalVector<WGPURenderPassColorAttachment> color_attachments;
	for (uint32_t i = 0; i < subpass.color_references.size(); i++) {
		const RDD::AttachmentReference &ref = subpass.color_references[i];

		WGPURenderPassColorAttachment att = {};
		att.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

		if (ref.attachment == RDD::AttachmentReference::UNUSED) {
			att.view = nullptr;
			att.loadOp = WGPULoadOp_Load;
			att.storeOp = WGPUStoreOp_Discard;
			color_attachments.push_back(att);
			continue;
		}

		if (ref.attachment < fb->attachment_views.size()) {
			att.view = fb->attachment_views[ref.attachment];
		}

		// Resolve target (MSAA).
		if (i < subpass.resolve_references.size()) {
			const RDD::AttachmentReference &res_ref = subpass.resolve_references[i];
			if (res_ref.attachment != RDD::AttachmentReference::UNUSED &&
					res_ref.attachment < fb->attachment_views.size()) {
				att.resolveTarget = fb->attachment_views[res_ref.attachment];
			}
		}

		// Load from previous subpass output, store for next use.
		att.loadOp = WGPULoadOp_Load;
		att.storeOp = WGPUStoreOp_Store;

		color_attachments.push_back(att);
	}

	// --- Build depth/stencil attachment ---
	WGPURenderPassDepthStencilAttachment ds_att = {};
	WGPURenderPassDepthStencilAttachment *ds_att_ptr = nullptr;

	const RDD::AttachmentReference &ds_ref = subpass.depth_stencil_reference;
	if (ds_ref.attachment != RDD::AttachmentReference::UNUSED &&
			ds_ref.attachment < fb->attachment_views.size()) {
		ds_att.view = fb->attachment_views[ds_ref.attachment];

		if (ds_ref.attachment < rp->attachments.size()) {
			const RDD::Attachment &attach_desc = rp->attachments[ds_ref.attachment];
			WGPUTextureFormat wgpu_fmt = _data_format_to_wgpu(attach_desc.format);
			bool has_depth = is_depth_format_wgpu(wgpu_fmt);
			bool has_stencil = has_stencil_wgpu(wgpu_fmt);

			if (has_depth) {
				ds_att.depthLoadOp = WGPULoadOp_Load;
				ds_att.depthStoreOp = WGPUStoreOp_Store;
				ds_att.depthClearValue = 0.0f;
			} else {
				ds_att.depthLoadOp = WGPULoadOp_Undefined;
				ds_att.depthStoreOp = WGPUStoreOp_Undefined;
				ds_att.depthReadOnly = true;
			}

			if (has_stencil) {
				ds_att.stencilLoadOp = WGPULoadOp_Load;
				ds_att.stencilStoreOp = WGPUStoreOp_Store;
			} else {
				ds_att.stencilLoadOp = WGPULoadOp_Undefined;
				ds_att.stencilStoreOp = WGPUStoreOp_Undefined;
				ds_att.stencilReadOnly = true;
			}
		} else {
			ds_att.depthLoadOp = WGPULoadOp_Load;
			ds_att.depthStoreOp = WGPUStoreOp_Store;
			ds_att.depthClearValue = 1.0f;
			ds_att.stencilLoadOp = WGPULoadOp_Load;
			ds_att.stencilStoreOp = WGPUStoreOp_Store;
		}
		ds_att_ptr = &ds_att;
	}

	// --- Begin new render pass for this subpass ---
	WGPURenderPassDescriptor pass_desc = {};
	pass_desc.colorAttachmentCount = color_attachments.size();
	pass_desc.colorAttachments = color_attachments.ptr();
	pass_desc.depthStencilAttachment = ds_att_ptr;

	cmd->render_encoder = wgpuCommandEncoderBeginRenderPass(cmd->encoder, &pass_desc);
	cmd->active_encoder = WGCommandBuffer::RENDER;

	// Reset pipeline state — new render pass requires re-binding everything.
	cmd->render_state.viewport_set = false;
	cmd->render_state.scissor_set = false;
	cmd->render_state.blend_constant_set = false;
	cmd->render_state.current_pipeline = nullptr;
	cmd->render_state.current_index_buffer = nullptr;
	cmd->render_state.current_index_offset = 0;
	memset(cmd->render_state.current_vertex_buffers, 0, sizeof(cmd->render_state.current_vertex_buffers));
	memset(cmd->render_state.current_vertex_offsets, 0, sizeof(cmd->render_state.current_vertex_offsets));
}

void RenderingDeviceDriverWebGPU::command_render_set_viewport(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_viewports) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->render_encoder);

	if (p_viewports.size() > 0) {
		const Rect2i &vp = p_viewports[0]; // WebGPU supports only one viewport.
		WGCommandBuffer::RenderState &rs = cmd->render_state;
		rs.viewport[0] = (float)vp.position.x;
		rs.viewport[1] = (float)vp.position.y;
		rs.viewport[2] = (float)vp.size.x;
		rs.viewport[3] = (float)vp.size.y;
		rs.viewport_set = true;
		wgpuRenderPassEncoderSetViewport(cmd->render_encoder, rs.viewport[0], rs.viewport[1], rs.viewport[2], rs.viewport[3], 0.0f, 1.0f);
	}
}

void RenderingDeviceDriverWebGPU::command_render_set_scissor(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_scissors) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->render_encoder);

	if (p_scissors.size() > 0) {
		const Rect2i &sr = p_scissors[0];
		uint32_t x = MAX(sr.position.x, 0);
		uint32_t y = MAX(sr.position.y, 0);
		uint32_t w = MAX(sr.size.x, 0);
		uint32_t h = MAX(sr.size.y, 0);
		// Clamp scissor to the SMALLER of (render area) and (actual framebuffer size).
		// WebGPU requires scissor to fit within attachment dimensions.
		uint32_t clamp_w = 0;
		uint32_t clamp_h = 0;
		if (cmd->render_state.render_area_width > 0) {
			clamp_w = cmd->render_state.render_area_width;
			clamp_h = cmd->render_state.render_area_height;
		}
		if (cmd->render_state.framebuffer) {
			const Size2i fb_size = _framebuffer_extent(cmd->render_state.framebuffer);
			clamp_w = clamp_w > 0 ? MIN(clamp_w, (uint32_t)fb_size.x) : (uint32_t)fb_size.x;
			clamp_h = clamp_h > 0 ? MIN(clamp_h, (uint32_t)fb_size.y) : (uint32_t)fb_size.y;
		}
		if (clamp_w > 0 && clamp_h > 0) {
			if (x >= clamp_w || y >= clamp_h) {
				w = 0;
				h = 0;
				x = 0;
				y = 0;
			} else {
				w = MIN(w, clamp_w - x);
				h = MIN(h, clamp_h - y);
			}
		}
		WGCommandBuffer::RenderState &rs = cmd->render_state;
		rs.scissor[0] = x;
		rs.scissor[1] = y;
		rs.scissor[2] = w;
		rs.scissor[3] = h;
		rs.scissor_set = true;
		wgpuRenderPassEncoderSetScissorRect(cmd->render_encoder, x, y, w, h);
	}
}

void RenderingDeviceDriverWebGPU::command_render_clear_attachments(CommandBufferID p_cmd_buffer, VectorView<AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	_render_clear(cmd, p_attachment_clears, p_rects);
}

// WGSL type a fragment output needs to write a colour target of this format.
static const char *_wgsl_color_output_type(WGPUTextureFormat p_format) {
	if (p_format == WGPUTextureFormat_RGB10A2Uint) {
		return "vec4u";
	}
	uint32_t channels = 0;
	WGTexelType type = WG_TEXEL_FLOAT32;
	if (wgpu_format_texel_info(p_format, channels, type)) {
		switch (type) {
			case WG_TEXEL_UINT8:
			case WG_TEXEL_UINT16:
			case WG_TEXEL_UINT32:
				return "vec4u";
			case WG_TEXEL_SINT8:
			case WG_TEXEL_SINT16:
			case WG_TEXEL_SINT32:
				return "vec4i";
			default:
				break;
		}
	}
	return "vec4f";
}

WGPURenderPipeline RenderingDeviceDriverWebGPU::_get_clear_pipeline(const ClearPipelineKey &p_key) {
	if (const WGPURenderPipeline *cached = clear_pipelines.getptr(p_key)) {
		return *cached;
	}

	// A triangle covering the viewport; depth comes from the viewport's depth range.
	String wgsl = vformat("@group(0) @binding(%d) var<storage, read> clear_colors: array<vec4f, %d>;\n", PUSH_CONSTANT_RING_BINDING, CLEAR_MAX_COLORS);
	wgsl += "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
			"\treturn vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.5, 1.0);\n"
			"}\n";
	String outputs;
	String writes;
	WGPUColorTargetState targets[CLEAR_MAX_COLORS] = {};
	for (uint32_t i = 0; i < p_key.color_count; i++) {
		targets[i].format = p_key.color_formats[i];
		targets[i].writeMask = WGPUColorWriteMask_None;
		if (p_key.color_write_mask & (1u << i)) {
			targets[i].writeMask = WGPUColorWriteMask_All;
			const char *type = _wgsl_color_output_type(p_key.color_formats[i]);
			outputs += vformat("\t@location(%d) c%d: %s,\n", i, i, type);
			writes += vformat("\to.c%d = %s(clear_colors[%d]);\n", i, type, i);
		}
	}
	if (outputs.is_empty()) {
		wgsl += "@fragment fn fs() {}\n";
	} else {
		wgsl += "struct ClearOutput {\n" + outputs + "}\n@fragment fn fs() -> ClearOutput {\n\tvar o: ClearOutput;\n" + writes + "\treturn o;\n}\n";
	}

	const CharString wgsl_cs = wgsl.utf8();
	WGPUShaderSourceWGSL wgsl_source = {};
	wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgsl_source.code = { wgsl_cs.get_data(), WGPU_STRLEN };
	WGPUShaderModuleDescriptor module_desc = {};
	module_desc.nextInChain = &wgsl_source.chain;
	module_desc.label = { "clear_attachments", WGPU_STRLEN };
	WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &module_desc);
	ERR_FAIL_NULL_V_MSG(module, nullptr, "WebGPU: failed to create the clear shader module.");

	WGPUFragmentState fragment = {};
	fragment.module = module;
	fragment.entryPoint = { "fs", WGPU_STRLEN };
	fragment.targetCount = p_key.color_count;
	fragment.targets = targets;

	WGPUDepthStencilState depth_stencil = {};
	WGPURenderPipelineDescriptor desc = {};
	if (p_key.depth_stencil_format != WGPUTextureFormat_Undefined) {
		const WGPUStencilOperation stencil_op = (p_key.depth_stencil_write & 2) ? WGPUStencilOperation_Replace : WGPUStencilOperation_Keep;
		depth_stencil.format = p_key.depth_stencil_format;
		depth_stencil.depthWriteEnabled = (p_key.depth_stencil_write & 1) ? WGPUOptionalBool_True : WGPUOptionalBool_False;
		depth_stencil.depthCompare = is_depth_format_wgpu(p_key.depth_stencil_format) ? WGPUCompareFunction_Always : WGPUCompareFunction_Undefined;
		depth_stencil.stencilFront = { WGPUCompareFunction_Always, stencil_op, stencil_op, stencil_op };
		depth_stencil.stencilBack = depth_stencil.stencilFront;
		depth_stencil.stencilReadMask = 0xFF;
		depth_stencil.stencilWriteMask = (p_key.depth_stencil_write & 2) ? 0xFF : 0;
		desc.depthStencil = &depth_stencil;
	}

	desc.label = { "clear_attachments", WGPU_STRLEN };
	desc.layout = clear_pipeline_layout;
	desc.vertex.module = module;
	desc.vertex.entryPoint = { "vs", WGPU_STRLEN };
	desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	desc.primitive.cullMode = WGPUCullMode_None;
	desc.multisample.count = p_key.sample_count;
	desc.multisample.mask = 0xFFFFFFFFu;
	desc.fragment = &fragment;

	WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
	wgpuShaderModuleRelease(module);
	ERR_FAIL_NULL_V_MSG(pipeline, nullptr, "WebGPU: failed to create the clear pipeline.");
	clear_pipelines.insert(p_key, pipeline);
	return pipeline;
}

void RenderingDeviceDriverWebGPU::_render_clear(WGCommandBuffer *p_cmd_buf, VectorView<AttachmentClear> p_clears, VectorView<Rect2i> p_rects) {
	WGCommandBuffer::RenderState &rs = p_cmd_buf->render_state;
	ERR_FAIL_COND(!p_cmd_buf->render_encoder || !rs.render_pass || !rs.framebuffer);
	const WGRenderPass::SubpassInfo &subpass = rs.render_pass->subpasses[rs.current_subpass];
	const WGFramebuffer *fb = rs.framebuffer;
	ERR_FAIL_COND_MSG(subpass.color_references.size() > CLEAR_MAX_COLORS, "WebGPU: too many colour attachments to clear.");

	// Attachment views are the textures' default views, so the texture formats match the pass.
	auto attachment = [fb](uint32_t p_index) -> const WGTexture * {
		return p_index < fb->attachments.size() ? fb->attachments[p_index] : nullptr;
	};
	ClearPipelineKey key;
	key.color_count = subpass.color_references.size();
	for (uint32_t i = 0; i < key.color_count; i++) {
		if (const WGTexture *tex = attachment(subpass.color_references[i].attachment)) {
			key.color_formats[i] = tex->format;
			key.sample_count = tex->sample_count;
		}
	}
	if (const WGTexture *tex = attachment(subpass.depth_stencil_reference.attachment)) {
		key.depth_stencil_format = tex->format;
		key.sample_count = tex->sample_count;
	}

	float colors[CLEAR_MAX_COLORS][4] = {};
	float depth = 0.0f;
	uint32_t stencil = 0;
	for (uint32_t i = 0; i < p_clears.size(); i++) {
		const AttachmentClear &clear = p_clears[i];
		if (clear.aspect.has_flag(TEXTURE_ASPECT_COLOR_BIT)) {
			ERR_CONTINUE(clear.color_attachment >= key.color_count || key.color_formats[clear.color_attachment] == WGPUTextureFormat_Undefined);
			key.color_write_mask |= 1u << clear.color_attachment;
			const Color &c = clear.value.color;
			colors[clear.color_attachment][0] = c.r;
			colors[clear.color_attachment][1] = c.g;
			colors[clear.color_attachment][2] = c.b;
			colors[clear.color_attachment][3] = c.a;
		}
		if (clear.aspect.has_flag(TEXTURE_ASPECT_DEPTH_BIT) && is_depth_format_wgpu(key.depth_stencil_format)) {
			key.depth_stencil_write |= 1;
			depth = CLAMP(clear.value.depth, 0.0f, 1.0f);
		}
		if (clear.aspect.has_flag(TEXTURE_ASPECT_STENCIL_BIT) && has_stencil_wgpu(key.depth_stencil_format)) {
			key.depth_stencil_write |= 2;
			stencil = clear.value.stencil;
		}
	}
	if (key.color_write_mask == 0 && key.depth_stencil_write == 0) {
		return;
	}
	WGPURenderPipeline pipeline = _get_clear_pipeline(key);
	ERR_FAIL_NULL(pipeline);

	// Send the colors as push constants (this may restart the pass on a new ring page), then restore
	// the caller's, which the next draw must re-flush because group 0 now holds the clear colors.
	static_assert(sizeof(colors) == WGCommandBuffer::MAX_PUSH_CONSTANT_SIZE);
	uint8_t saved_push_constants[WGCommandBuffer::MAX_PUSH_CONSTANT_SIZE];
	memcpy(saved_push_constants, p_cmd_buf->push_constant_data, sizeof(saved_push_constants));
	const uint32_t saved_push_constant_len = p_cmd_buf->push_constant_data_len;
	memcpy(p_cmd_buf->push_constant_data, colors, sizeof(colors));
	p_cmd_buf->push_constant_data_len = sizeof(colors);
	p_cmd_buf->push_constants_dirty = true;
	_flush_push_constants(p_cmd_buf, &clear_shader);
	memcpy(p_cmd_buf->push_constant_data, saved_push_constants, sizeof(saved_push_constants));
	p_cmd_buf->push_constant_data_len = saved_push_constant_len;
	p_cmd_buf->push_constants_dirty = true;

	WGPURenderPassEncoder encoder = p_cmd_buf->render_encoder;
	wgpuRenderPassEncoderSetPipeline(encoder, pipeline);
	wgpuRenderPassEncoderSetStencilReference(encoder, stencil);
	const Rect2i extent(Point2i(), _framebuffer_extent(fb));
	for (uint32_t i = 0; i < p_rects.size(); i++) {
		const Rect2i rect = p_rects[i].intersection(extent);
		if (!rect.has_area()) {
			continue;
		}
		wgpuRenderPassEncoderSetViewport(encoder, rect.position.x, rect.position.y, rect.size.x, rect.size.y, depth, depth);
		wgpuRenderPassEncoderSetScissorRect(encoder, rect.position.x, rect.position.y, rect.size.x, rect.size.y);
		wgpuRenderPassEncoderDraw(encoder, 3, 1, 0, 0);
	}

	// Restore the pipeline, bind groups and dynamic state; untracked viewport/scissor go back to the defaults.
	_rebind_render_state(p_cmd_buf);
	if (!rs.viewport_set) {
		wgpuRenderPassEncoderSetViewport(encoder, 0.0f, 0.0f, extent.size.x, extent.size.y, 0.0f, 1.0f);
	}
	if (!rs.scissor_set) {
		wgpuRenderPassEncoderSetScissorRect(encoder, 0, 0, extent.size.x, extent.size.y);
	}
}

void RenderingDeviceDriverWebGPU::command_bind_render_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGPipelineWrapper *pw = (WGPipelineWrapper *)(p_pipeline.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(pw);
	ERR_FAIL_COND(!cmd->render_encoder);

	if (cmd->render_state.current_pipeline == pw) {
		return; // Pipeline already bound, skip redundant call.
	}
	if (pw->deferred) {
		_create_deferred_pipeline(pw);
	}
	ERR_FAIL_NULL(pw->render_handle);

	wgpuRenderPassEncoderSetPipeline(cmd->render_encoder, pw->render_handle);
	wgpuRenderPassEncoderSetStencilReference(cmd->render_encoder, pw->stencil_reference);
	cmd->render_state.current_pipeline = pw;
	perf.set_pipeline_calls++;

	// Pre-bind empty bind groups at pipeline layout gap slots.
	// Firefox/wgpu requires all bind group indices to be set before draw calls.
	if (pw->shader) {
		for (uint32_t gap_idx : pw->shader->gap_bind_group_indices) {
			wgpuRenderPassEncoderSetBindGroup(cmd->render_encoder, gap_idx, empty_bind_group, 0, nullptr);
			perf.gap_bind_group_calls++;
		}
	}
}

void RenderingDeviceDriverWebGPU::command_bind_render_uniform_sets(CommandBufferID p_cmd_buffer, VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->render_encoder);

	WGShader *pipeline_shader = cmd->render_state.current_pipeline ? cmd->render_state.current_pipeline->shader : nullptr;

	// Invalidate bind group tracking if the pipeline shader changed.
	if (pipeline_shader != cmd->bound_shader) {
		cmd->invalidate_bind_groups();
		cmd->bound_shader = pipeline_shader;
	}

	// Task 7.5: Unpack 4-bit frame indices from p_dynamic_offsets as we walk the sets.
	// Every set with `us->dynamic_buffers.size()` entries consumes that many 4-bit
	// slots in the mask (in binding order, matching uniform_sets_get_dynamic_offsets).
	static constexpr uint32_t MAX_DYNAMIC_BUFFERS = 8;
	uint32_t dyn_shift = 0;

	for (uint32_t i = 0; i < p_set_count; i++) {
		WGUniformSet *us = (WGUniformSet *)(p_uniform_sets[i].id);
		if (us && us->handle) {
			uint32_t set_idx = p_first_set_index + i;

			// Get a bind group compatible with the current pipeline's BGL.
			WGPUBindGroup bg_to_bind = _get_compatible_bind_group(us, pipeline_shader, set_idx);

			// Unpack dynamic offsets for this set's material dynamic buffers, in binding order.
			uint32_t set_dyn_offsets[MAX_DYNAMIC_BUFFERS] = {};
			uint32_t num_dyn = us->dynamic_buffers.size();
			if (num_dyn > MAX_DYNAMIC_BUFFERS) {
				num_dyn = MAX_DYNAMIC_BUFFERS;
			}
			for (uint32_t j = 0; j < num_dyn; j++) {
				uint32_t frame_idx = (p_dynamic_offsets >> dyn_shift) & 0xFu;
				dyn_shift += 4u;
				const WGBuffer *dbuf = us->dynamic_buffers[j];
				set_dyn_offsets[j] = (uint32_t)(frame_idx * (dbuf ? dbuf->per_frame_size : 0));
			}

			const bool is_pc_merged = (pipeline_shader && pipeline_shader->push_constant_size > 0 &&
					set_idx == pipeline_shader->push_constant_bind_group &&
					pipeline_shader->merged_pc_group_layout != nullptr);

			if (is_pc_merged) {
				// PC ring buffer is at PUSH_CONSTANT_RING_BINDING, the highest
				// binding in the merged group, so its dynamic offset is appended last.
				// Save the material offsets so _flush_push_constants can preserve them
				// while patching in the real PC ring offset before each draw.
				cmd->pc_group_material_dyn_count = num_dyn;
				for (uint32_t j = 0; j < num_dyn && j < WGCommandBuffer::MAX_PC_GROUP_MATERIAL_DYN; j++) {
					cmd->pc_group_material_dyn_offsets[j] = set_dyn_offsets[j];
				}
				// Append last-flushed PC ring offset so the draw is correct even if
				// _flush_push_constants doesn't fire (e.g. firstInstance optimization).
				if (num_dyn < MAX_DYNAMIC_BUFFERS) {
					set_dyn_offsets[num_dyn] = cmd->last_flushed_pc_offset;
					num_dyn++;
				}
				cmd->current_pc_bind_group = bg_to_bind;
				wgpuRenderPassEncoderSetBindGroup(cmd->render_encoder, set_idx, bg_to_bind, num_dyn, set_dyn_offsets);
				perf.set_bind_group_calls++;
				// Save full state for mid-pass restart.
				if (set_idx < WGCommandBuffer::MAX_BIND_GROUPS) {
					auto &bs = cmd->last_bound_state[set_idx];
					bs.group = bg_to_bind;
					bs.dynamic_offset_count = num_dyn;
					for (uint32_t j = 0; j < num_dyn && j < WGCommandBuffer::MAX_BIND_GROUP_DYN_OFFSETS; j++) {
						bs.dynamic_offsets[j] = set_dyn_offsets[j];
					}
				}
				// PC group always rebinds per draw (via _flush_push_constants); don't cache.
				if (set_idx < WGCommandBuffer::MAX_BIND_GROUPS) {
					cmd->bound_bind_groups[set_idx] = nullptr;
				}
			} else if (pipeline_shader && pipeline_shader->push_constant_size > 0 &&
					set_idx == pipeline_shader->push_constant_bind_group) {
				// PC group, but shader has no merged layout → just clear the PC pointer.
				cmd->current_pc_bind_group = nullptr;
				cmd->pc_group_material_dyn_count = 0;
				wgpuRenderPassEncoderSetBindGroup(cmd->render_encoder, set_idx, bg_to_bind, num_dyn, num_dyn ? set_dyn_offsets : nullptr);
				perf.set_bind_group_calls++;
				// Save full state for mid-pass restart.
				if (set_idx < WGCommandBuffer::MAX_BIND_GROUPS) {
					auto &bs = cmd->last_bound_state[set_idx];
					bs.group = bg_to_bind;
					bs.dynamic_offset_count = num_dyn;
					for (uint32_t j = 0; j < num_dyn && j < WGCommandBuffer::MAX_BIND_GROUP_DYN_OFFSETS; j++) {
						bs.dynamic_offsets[j] = set_dyn_offsets[j];
					}
					cmd->bound_bind_groups[set_idx] = num_dyn > 0 ? nullptr : bg_to_bind;
				}
			} else if (num_dyn > 0) {
				// Non-PC set with material dynamic buffers: must always rebind because
				// the frame_idx rotates — bypass the redundant-bind cache.
				wgpuRenderPassEncoderSetBindGroup(cmd->render_encoder, set_idx, bg_to_bind, num_dyn, set_dyn_offsets);
				perf.set_bind_group_calls++;
				// Save full state for mid-pass restart.
				if (set_idx < WGCommandBuffer::MAX_BIND_GROUPS) {
					auto &bs = cmd->last_bound_state[set_idx];
					bs.group = bg_to_bind;
					bs.dynamic_offset_count = num_dyn;
					for (uint32_t j = 0; j < num_dyn && j < WGCommandBuffer::MAX_BIND_GROUP_DYN_OFFSETS; j++) {
						bs.dynamic_offsets[j] = set_dyn_offsets[j];
					}
					cmd->bound_bind_groups[set_idx] = nullptr;
				}
			} else {
				// Static non-PC set — skip if already bound.
				if (set_idx < WGCommandBuffer::MAX_BIND_GROUPS && cmd->bound_bind_groups[set_idx] == bg_to_bind) {
					continue;
				}
				wgpuRenderPassEncoderSetBindGroup(cmd->render_encoder, set_idx, bg_to_bind, 0, nullptr);
				perf.set_bind_group_calls++;
				// Save full state for mid-pass restart.
				if (set_idx < WGCommandBuffer::MAX_BIND_GROUPS) {
					auto &bs = cmd->last_bound_state[set_idx];
					bs.group = bg_to_bind;
					bs.dynamic_offset_count = 0;
					cmd->bound_bind_groups[set_idx] = bg_to_bind;
				}
			}
		}
	}
}

void RenderingDeviceDriverWebGPU::command_render_draw(CommandBufferID p_cmd_buffer, uint32_t p_vertex_count, uint32_t p_instance_count, uint32_t p_base_vertex, uint32_t p_first_instance) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->render_encoder);

	if (cmd->render_state.current_pipeline) {
		_flush_push_constants(cmd, cmd->render_state.current_pipeline->shader);
	}

	perf.draw_calls++;
	if (p_first_instance > 0) {
		perf.first_instance_draws++;
	}
	wgpuRenderPassEncoderDraw(cmd->render_encoder, p_vertex_count, p_instance_count, p_base_vertex, p_first_instance);
}

void RenderingDeviceDriverWebGPU::command_render_draw_indexed(CommandBufferID p_cmd_buffer, uint32_t p_index_count, uint32_t p_instance_count, uint32_t p_first_index, int32_t p_vertex_offset, uint32_t p_first_instance) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->render_encoder);

	WGPipelineWrapper *pw = cmd->render_state.current_pipeline;
	if (pw) {
		_flush_push_constants(cmd, pw->shader);
		// For strip topologies, ensure the pipeline variant matches the bound index format.
		// Always select the correct variant — the format may have changed since pipeline bind.
		if (pw->is_strip && pw->render_handle_u16) {
			WGPURenderPipeline needed = (cmd->render_state.current_index_format == WGPUIndexFormat_Uint16)
					? pw->render_handle_u16
					: pw->render_handle;
			wgpuRenderPassEncoderSetPipeline(cmd->render_encoder, needed);
		}
	}

	perf.draw_calls++;
	if (p_first_instance > 0) {
		perf.first_instance_draws++;
	}
	wgpuRenderPassEncoderDrawIndexed(cmd->render_encoder, p_index_count, p_instance_count, p_first_index, p_vertex_offset, p_first_instance);
}

void RenderingDeviceDriverWebGPU::command_render_draw_indexed_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGBuffer *indirect = (WGBuffer *)(p_indirect_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(indirect);
	ERR_FAIL_COND(!cmd->render_encoder);

	WGPipelineWrapper *pw = cmd->render_state.current_pipeline;
	if (pw) {
		_flush_push_constants(cmd, pw->shader);
		if (pw->is_strip && pw->render_handle_u16) {
			WGPURenderPipeline needed = (cmd->render_state.current_index_format == WGPUIndexFormat_Uint16)
					? pw->render_handle_u16
					: pw->render_handle;
			wgpuRenderPassEncoderSetPipeline(cmd->render_encoder, needed);
		}
	}

	// WebGPU has no multi-draw-indirect — must loop.
	for (uint32_t i = 0; i < p_draw_count; i++) {
		wgpuRenderPassEncoderDrawIndexedIndirect(cmd->render_encoder, indirect->handle, p_offset + i * p_stride);
	}
}

void RenderingDeviceDriverWebGPU::command_render_draw_indexed_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	// TODO: Read count from buffer (requires async readback). For now, use max count.
	command_render_draw_indexed_indirect(p_cmd_buffer, p_indirect_buffer, p_offset, p_max_draw_count, p_stride);
}

void RenderingDeviceDriverWebGPU::command_render_draw_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGBuffer *indirect = (WGBuffer *)(p_indirect_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(indirect);
	ERR_FAIL_COND(!cmd->render_encoder);

	if (cmd->render_state.current_pipeline) {
		_flush_push_constants(cmd, cmd->render_state.current_pipeline->shader);
	}

	for (uint32_t i = 0; i < p_draw_count; i++) {
		wgpuRenderPassEncoderDrawIndirect(cmd->render_encoder, indirect->handle, p_offset + i * p_stride);
	}
}

void RenderingDeviceDriverWebGPU::command_render_draw_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	command_render_draw_indirect(p_cmd_buffer, p_indirect_buffer, p_offset, p_max_draw_count, p_stride);
}

void RenderingDeviceDriverWebGPU::command_render_bind_vertex_buffers(CommandBufferID p_cmd_buffer, uint32_t p_binding_count, const BufferID *p_buffers, const uint64_t *p_offsets, uint64_t p_dynamic_offsets) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->render_encoder);

	for (uint32_t i = 0; i < p_binding_count; i++) {
		WGBuffer *buf = (WGBuffer *)(p_buffers[i].id);
		if (buf) {
			uint64_t offset = p_offsets[i];
			if (buf->is_dynamic()) {
				uint64_t frame_idx = p_dynamic_offsets & 0x3;
				p_dynamic_offsets >>= 2;
				offset += frame_idx * buf->per_frame_size;
			}
			// Skip redundant SetVertexBuffer calls.
			if (i < WGCommandBuffer::RenderState::MAX_VERTEX_BINDINGS &&
					cmd->render_state.current_vertex_buffers[i] == buf->handle &&
					cmd->render_state.current_vertex_offsets[i] == offset) {
				continue;
			}
			wgpuRenderPassEncoderSetVertexBuffer(cmd->render_encoder, i, buf->handle, offset, WGPU_WHOLE_SIZE);
			perf.set_vertex_buffer_calls++;
			if (i < WGCommandBuffer::RenderState::MAX_VERTEX_BINDINGS) {
				cmd->render_state.current_vertex_buffers[i] = buf->handle;
				cmd->render_state.current_vertex_offsets[i] = offset;
			}
		}
	}
}

void RenderingDeviceDriverWebGPU::command_render_bind_index_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, IndexBufferFormat p_format, uint64_t p_offset) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGBuffer *buf = (WGBuffer *)(p_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(buf);
	ERR_FAIL_COND(!cmd->render_encoder);

	WGPUIndexFormat format = (p_format == INDEX_BUFFER_FORMAT_UINT16) ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;

	if (cmd->render_state.current_index_buffer == buf->handle &&
			cmd->render_state.current_index_offset == p_offset &&
			cmd->render_state.current_index_format == format) {
		return; // Index buffer already bound, skip redundant call.
	}

	wgpuRenderPassEncoderSetIndexBuffer(cmd->render_encoder, buf->handle, format, p_offset, WGPU_WHOLE_SIZE);
	cmd->render_state.current_index_buffer = buf->handle;
	cmd->render_state.current_index_offset = p_offset;
	cmd->render_state.current_index_format = format;
}

void RenderingDeviceDriverWebGPU::command_render_set_blend_constants(CommandBufferID p_cmd_buffer, const Color &p_constants) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->render_encoder);

	cmd->render_state.blend_constant = { p_constants.r, p_constants.g, p_constants.b, p_constants.a };
	cmd->render_state.blend_constant_set = true;
	wgpuRenderPassEncoderSetBlendConstant(cmd->render_encoder, &cmd->render_state.blend_constant);
}

void RenderingDeviceDriverWebGPU::command_render_set_line_width(CommandBufferID p_cmd_buffer, float p_width) {
	// No-op: WebGPU doesn't support line width > 1.0.
}

// =============================================================================
// Specialization Constant Support
// =============================================================================

// SPIR-V opcodes for specialization constant handling.
static constexpr uint16_t SPV_OP_DECORATE = 71;
static constexpr uint16_t SPV_OP_SPEC_CONSTANT_TRUE = 48;
static constexpr uint16_t SPV_OP_SPEC_CONSTANT_FALSE = 49;
static constexpr uint16_t SPV_OP_SPEC_CONSTANT = 50;
static constexpr uint16_t SPV_DECORATION_SPEC_ID = 1;

static inline uint32_t _spv_read_word(const uint8_t *p_data, uint32_t p_word_index) {
	const uint8_t *p = p_data + p_word_index * 4;
	return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
}

static inline void _spv_write_word(uint8_t *p_data, uint32_t p_word_index, uint32_t p_value) {
	uint8_t *p = p_data + p_word_index * 4;
	p[0] = p_value & 0xFF;
	p[1] = (p_value >> 8) & 0xFF;
	p[2] = (p_value >> 16) & 0xFF;
	p[3] = (p_value >> 24) & 0xFF;
}

// Patches SPIR-V bytecode to apply specialization constant values.
// Returns a modified copy with OpSpecConstant* values updated.
static PackedByteArray _patch_spirv_spec_constants(const PackedByteArray &p_spirv, VectorView<RDD::PipelineSpecializationConstant> p_constants) {
	if (p_constants.size() == 0 || p_spirv.size() < 20) {
		return p_spirv;
	}

	// Build a map: SpecId → value (as uint32_t).
	HashMap<uint32_t, uint32_t> spec_values;
	for (uint32_t i = 0; i < p_constants.size(); i++) {
		const RDD::PipelineSpecializationConstant &c = p_constants[i];
		uint32_t val = 0;
		switch (c.type) {
			case RDD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL:
				val = c.bool_value ? 1 : 0;
				break;
			case RDD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT:
				val = (uint32_t)c.int_value;
				break;
			case RDD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT:
				memcpy(&val, &c.float_value, sizeof(float));
				break;
		}
		spec_values[c.constant_id] = val;
	}

	// First pass: scan OpDecorate for SpecId → result_id mapping.
	HashMap<uint32_t, uint32_t> spec_id_to_result_id; // SpecId → result_id
	uint32_t total_words = p_spirv.size() / 4;
	uint32_t pos = 5; // Skip SPIR-V header (5 words).
	while (pos < total_words) {
		uint32_t w0 = _spv_read_word(p_spirv.ptr(), pos);
		uint32_t wc = w0 >> 16;
		uint32_t op = w0 & 0xFFFF;
		if (wc == 0 || pos + wc > total_words) {
			break;
		}
		// OpDecorate target decoration [literal...]
		// For SpecId: OpDecorate result_id SpecId(1) literal_spec_id
		if (op == SPV_OP_DECORATE && wc >= 4) {
			uint32_t target = _spv_read_word(p_spirv.ptr(), pos + 1);
			uint32_t decoration = _spv_read_word(p_spirv.ptr(), pos + 2);
			if (decoration == SPV_DECORATION_SPEC_ID) {
				uint32_t spec_id = _spv_read_word(p_spirv.ptr(), pos + 3);
				spec_id_to_result_id[spec_id] = target;
			}
		}
		pos += wc;
	}

	// Build result_id → value map.
	HashMap<uint32_t, uint32_t> result_to_value;
	for (const KeyValue<uint32_t, uint32_t> &kv : spec_id_to_result_id) {
		if (spec_values.has(kv.key)) {
			result_to_value[kv.value] = spec_values[kv.key];
		}
	}

	if (result_to_value.is_empty()) {
		return p_spirv; // No matching spec constants to patch.
	}

	// Second pass: patch the SPIR-V.
	PackedByteArray out = p_spirv;
	pos = 5;
	while (pos < total_words) {
		uint32_t w0 = _spv_read_word(out.ptr(), pos);
		uint32_t wc = w0 >> 16;
		uint32_t op = w0 & 0xFFFF;
		if (wc == 0 || pos + wc > total_words) {
			break;
		}

		if (op == SPV_OP_SPEC_CONSTANT_TRUE || op == SPV_OP_SPEC_CONSTANT_FALSE) {
			// OpSpecConstantTrue/False: wc=3, [type_id, result_id]
			if (wc >= 3) {
				uint32_t result_id = _spv_read_word(out.ptr(), pos + 2);
				if (result_to_value.has(result_id)) {
					uint32_t val = result_to_value[result_id];
					uint16_t new_op = val ? SPV_OP_SPEC_CONSTANT_TRUE : SPV_OP_SPEC_CONSTANT_FALSE;
					_spv_write_word(out.ptrw(), pos, (wc << 16) | new_op);
				}
			}
		} else if (op == SPV_OP_SPEC_CONSTANT) {
			// OpSpecConstant: wc>=4, [type_id, result_id, value...]
			if (wc >= 4) {
				uint32_t result_id = _spv_read_word(out.ptr(), pos + 2);
				if (result_to_value.has(result_id)) {
					_spv_write_word(out.ptrw(), pos + 3, result_to_value[result_id]);
				}
			}
		}

		pos += wc;
	}

	return out;
}

WGPUShaderModule RenderingDeviceDriverWebGPU::_stage_module(WGShader *p_shader, ShaderStage p_stage) {
	MutexLock lock(stage_module_mutex);
	WGPUShaderModule &module = p_shader->stage_modules[p_stage];
	if (module == nullptr && p_shader->stage_wgsl[p_stage].length() > 0) {
		WGPUShaderSourceWGSL wgsl_source = {};
		wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
		wgsl_source.code = WGPUStringView{ p_shader->stage_wgsl[p_stage].get_data(), WGPU_STRLEN };
		WGPUShaderModuleDescriptor desc = {};
		desc.nextInChain = (WGPUChainedStruct *)&wgsl_source;
		// Label the module so Dawn error messages (and JS-side patch logs) identify it.
		const CharString label = ("mod:" + p_shader->name + ":stg" + itos((int)p_stage)).utf8();
		desc.label = { label.get_data(), WGPU_STRLEN };
		module = wgpuDeviceCreateShaderModule(device, &desc);
		ERR_FAIL_NULL_V_MSG(module, nullptr, vformat("WebGPU: wgpuDeviceCreateShaderModule failed for stage %d.", (int)p_stage));
		p_shader->stage_wgsl[p_stage] = CharString();
	}
	return module;
}

// Creates a WGPUShaderModule from SPIR-V with specialization constants applied.
// Returns nullptr on failure.
WGPUShaderModule RenderingDeviceDriverWebGPU::_create_module_with_spec_constants(
		const PackedByteArray &p_spirv,
		VectorView<PipelineSpecializationConstant> p_constants,
		ShaderStage p_stage,
		const WGShader *p_shader) {
	PackedByteArray patched = _patch_spirv_spec_constants(p_spirv, p_constants);

	// Reuse the container compiler and its shader cache for specialized modules.
	char *wgsl_str = RenderingShaderContainerWebGPU::compile_spirv(patched.ptr(), (int)patched.size());

	if (!wgsl_str) {
		ERR_PRINT("WebGPU: SPIR-V→WGSL conversion failed for specialized shader module.");
		return nullptr;
	}

	WGSLRewrites rewrites; // Already mirrored by p_shader's layouts.
	const CharString wgsl_cs = _postprocess_wgsl(wgsl_str, p_stage, p_shader->vertex_storage_buffers, rewrites);
	free(wgsl_str);
	ERR_FAIL_COND_V_MSG(!rewrites.error.is_empty(), nullptr, rewrites.error);

	WGPUShaderSourceWGSL wgsl_source = {};
	wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgsl_source.code = { wgsl_cs.get_data(), WGPU_STRLEN };

	WGPUShaderModuleDescriptor desc = {};
	desc.nextInChain = &wgsl_source.chain;

	// Label the specialized module so the JS-side createShaderModule patch identifies it.
	static int _spec_mod_id = 0;
	int _specid = _spec_mod_id++;
	String _spec_label = "specmod#" + itos(_specid) + ":stg" + itos((int)p_stage);
	CharString _spec_label_cs = _spec_label.utf8();
	desc.label = { _spec_label_cs.get_data(), WGPU_STRLEN };

	return wgpuDeviceCreateShaderModule(device, &desc);
}

RDD::PipelineID RenderingDeviceDriverWebGPU::render_pipeline_create(
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
		VectorView<PipelineSpecializationConstant> p_specialization_constants) {
	ERR_FAIL_COND_V_MSG(p_multisample_state.sample_count != TEXTURE_SAMPLES_1 && p_multisample_state.sample_count != TEXTURE_SAMPLES_4,
			PipelineID(), "WebGPU: render pipelines only support sample counts 1 and 4.");
	WGRenderPass *render_pass = (WGRenderPass *)p_render_pass.id;
	ERR_FAIL_NULL_V(render_pass, PipelineID());
	ERR_FAIL_UNSIGNED_INDEX_V(p_render_subpass, render_pass->subpasses.size(), PipelineID());
	const WGRenderPass::SubpassInfo &subpass = render_pass->subpasses[p_render_subpass];
	for (const AttachmentReference &ref : subpass.color_references) {
		if (ref.attachment != AttachmentReference::UNUSED) {
			ERR_FAIL_COND_V_MSG(!texture_is_format_supported_for_samples(render_pass->attachments[ref.attachment].format, p_multisample_state.sample_count),
					PipelineID(), "WebGPU: color attachment format does not support the pipeline sample count.");
		}
	}
	if (subpass.depth_stencil_reference.attachment != AttachmentReference::UNUSED) {
		ERR_FAIL_COND_V_MSG(!texture_is_format_supported_for_samples(render_pass->attachments[subpass.depth_stencil_reference.attachment].format, p_multisample_state.sample_count),
				PipelineID(), "WebGPU: depth attachment format does not support the pipeline sample count.");
	}
#ifdef WEB_ENABLED
	// Chrome compiles pipelines one by one on its GPU thread and presents no frames meanwhile, and Godot creates many
	// pipelines it never binds, which stalled Forward+ startup for half a minute. Create each on first bind instead.
	WGShader *shader = (WGShader *)(p_shader.id);
	ERR_FAIL_COND_V(!shader, PipelineID());
	ERR_FAIL_COND_V(!p_render_pass.id, PipelineID());
	WGPipelineWrapper *pw = memnew(WGPipelineWrapper);
	pw->type = WGPipelineWrapper::RENDER;
	pw->shader = shader;
	WGDeferredPipeline *d = memnew(WGDeferredPipeline);
	d->vertex_format = p_vertex_format;
	d->render_primitive = p_render_primitive;
	d->rasterization_state = p_rasterization_state;
	d->multisample_state = p_multisample_state;
	d->depth_stencil_state = p_depth_stencil_state;
	d->blend_state = p_blend_state;
	for (uint32_t i = 0; i < p_color_attachments.size(); i++) {
		d->color_attachments.push_back(p_color_attachments[i]);
	}
	d->dynamic_state = p_dynamic_state;
	d->render_pass = p_render_pass;
	d->render_subpass = p_render_subpass;
	for (uint32_t i = 0; i < p_specialization_constants.size(); i++) {
		d->specialization_constants.push_back(p_specialization_constants[i]);
	}
	pw->deferred = d;
	return PipelineID(pw);
#else
	return _render_pipeline_create_now(p_shader, p_vertex_format, p_render_primitive, p_rasterization_state, p_multisample_state,
			p_depth_stencil_state, p_blend_state, p_color_attachments, p_dynamic_state, p_render_pass, p_render_subpass, p_specialization_constants);
#endif
}

void RenderingDeviceDriverWebGPU::_create_deferred_pipeline(WGPipelineWrapper *p_pw) {
	WGDeferredPipeline *d = p_pw->deferred;
	p_pw->deferred = nullptr;
	ShaderID shader(p_pw->shader);
	PipelineID built = p_pw->type == WGPipelineWrapper::COMPUTE
			? _compute_pipeline_create_now(shader, d->specialization_constants)
			: _render_pipeline_create_now(shader, d->vertex_format, d->render_primitive, d->rasterization_state, d->multisample_state,
					  d->depth_stencil_state, d->blend_state, d->color_attachments, d->dynamic_state, d->render_pass, d->render_subpass,
					  d->specialization_constants);
	memdelete(d);
	if (built.id) {
		WGPipelineWrapper *b = (WGPipelineWrapper *)(built.id);
		*p_pw = *b;
		memdelete(b);
	}
}

RDD::PipelineID RenderingDeviceDriverWebGPU::_render_pipeline_create_now(
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
		VectorView<PipelineSpecializationConstant> p_specialization_constants) {
	WGShader *shader = (WGShader *)(p_shader.id);
	ERR_FAIL_COND_V(!shader, PipelineID());
	WGVertexFormat *vf = p_vertex_format.id ? (WGVertexFormat *)(p_vertex_format.id) : nullptr;
	WGRenderPass *rp = (WGRenderPass *)(p_render_pass.id);
	ERR_FAIL_COND_V(!rp, PipelineID());
	ERR_FAIL_COND_V(p_render_subpass >= (uint32_t)rp->subpasses.size(), PipelineID());
	const WGRenderPass::SubpassInfo &subpass = rp->subpasses[p_render_subpass];

	// --- Specialization constants ---
	// If the base shader modules were compiled with WGSL override declarations
	// (spirv_to_wgsl_with_overrides), we can pass specialization values as
	// pipeline constants instead of creating specialized shader modules.
	// This eliminates all runtime SPIR-V patching and Tint conversion.
	WGPUShaderModule vertex_module = _stage_module(shader, SHADER_STAGE_VERTEX);
	WGPUShaderModule fragment_module = _stage_module(shader, SHADER_STAGE_FRAGMENT);
	WGPUShaderModule specialized_vertex = nullptr;
	WGPUShaderModule specialized_fragment = nullptr;

	// Build per-stage WGPUConstantEntry arrays from specialization constants.
	// WebGPU requires that every constant key passed to a stage actually exists
	// as an @id(N) override in that stage's module — unlike Vulkan which silently
	// ignores unknown spec constant IDs. So we filter by stage_override_ids.
	LocalVector<WGPUConstantEntry> vertex_constants;
	LocalVector<WGPUConstantEntry> frag_constants;
	LocalVector<CharString> constant_key_storage; // keep key strings alive
	bool use_override_path = shader->has_override_declarations;
	if (p_specialization_constants.size() > 0) {
		if (use_override_path) {
			// Override path: pass constants at pipeline creation, reuse base modules.
			const HashSet<uint32_t> &vtx_ids = shader->stage_override_ids[SHADER_STAGE_VERTEX];
			const HashSet<uint32_t> &frg_ids = shader->stage_override_ids[SHADER_STAGE_FRAGMENT];
			for (uint32_t i = 0; i < p_specialization_constants.size(); i++) {
				const PipelineSpecializationConstant &sc = p_specialization_constants[i];
				bool in_vtx = vtx_ids.has(sc.constant_id);
				bool in_frg = frg_ids.has(sc.constant_id);
				if (!in_vtx && !in_frg) {
					continue; // This constant ID doesn't exist in any stage's WGSL.
				}
				WGPUConstantEntry entry = {};
				String key_str = itos(sc.constant_id);
				CharString key_cs = key_str.utf8();
				constant_key_storage.push_back(key_cs);
				entry.key = { constant_key_storage[constant_key_storage.size() - 1].get_data(), WGPU_STRLEN };
				switch (sc.type) {
					case PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL:
						entry.value = sc.bool_value ? 1.0 : 0.0;
						break;
					case PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT:
						entry.value = (double)sc.int_value;
						break;
					case PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT:
						entry.value = (double)sc.float_value;
						break;
				}
				if (in_vtx) {
					vertex_constants.push_back(entry);
				}
				if (in_frg) {
					frag_constants.push_back(entry);
				}
			}
		} else {
			// Legacy path: patch SPIR-V and create specialized modules via Tint.
			if (!shader->stage_spirv[SHADER_STAGE_VERTEX].is_empty()) {
				specialized_vertex = _create_module_with_spec_constants(
						shader->stage_spirv[SHADER_STAGE_VERTEX], p_specialization_constants, SHADER_STAGE_VERTEX, shader);
				if (specialized_vertex) {
					vertex_module = specialized_vertex;
				}
			}
			if (!shader->stage_spirv[SHADER_STAGE_FRAGMENT].is_empty()) {
				specialized_fragment = _create_module_with_spec_constants(
						shader->stage_spirv[SHADER_STAGE_FRAGMENT], p_specialization_constants, SHADER_STAGE_FRAGMENT, shader);
				if (specialized_fragment) {
					fragment_module = specialized_fragment;
				}
			}
		}
	}

	// --- Vertex state ---
	LocalVector<WGPUVertexBufferLayout> vb_layouts;
	LocalVector<WGPUVertexAttribute> vb_attrs;
	if (vf && !vf->attributes.is_empty()) {
		// Group attributes by binding, counting totals first.
		HashMap<uint32_t, uint32_t> binding_attr_count;
		for (const WGVertexFormat::Attribute &a : vf->attributes) {
			binding_attr_count[a.binding]++;
		}
		uint32_t total_attrs = vf->attributes.size();
		vb_attrs.resize(total_attrs); // Pre-allocate — no reallocation after this.

		// Determine start index per binding.
		HashMap<uint32_t, uint32_t> binding_start;
		uint32_t pos = 0;
		for (uint32_t b = 0; b < vf->bindings.size(); b++) {
			if (binding_attr_count.has(b)) {
				binding_start[b] = pos;
				pos += binding_attr_count[b];
			}
		}

		// Fill attrs per binding.
		HashMap<uint32_t, uint32_t> fill_pos;
		for (const WGVertexFormat::Attribute &a : vf->attributes) {
			uint32_t idx = binding_start[a.binding] + (fill_pos.has(a.binding) ? fill_pos[a.binding]++ : (fill_pos[a.binding] = 1, 0));
			vb_attrs[idx].format = a.format;
			vb_attrs[idx].offset = a.offset;
			vb_attrs[idx].shaderLocation = a.location;
		}

		// (debug removed)

		// Build layouts (one per binding that has attributes).
		for (uint32_t b = 0; b < vf->bindings.size(); b++) {
			if (!binding_attr_count.has(b)) {
				continue;
			}
			WGPUVertexBufferLayout layout = {};
			layout.arrayStride = vf->bindings[b].stride;
			layout.stepMode = vf->bindings[b].step_mode;
			layout.attributeCount = binding_attr_count[b];
			layout.attributes = vb_attrs.ptr() + binding_start[b];
			vb_layouts.push_back(layout);
		}
	}

	WGPUVertexState vertex_state = {};
	vertex_state.module = vertex_module;
	vertex_state.entryPoint = { "main", WGPU_STRLEN };
	vertex_state.bufferCount = vb_layouts.size();
	vertex_state.buffers = vb_layouts.ptr();
	if (use_override_path && !vertex_constants.is_empty()) {
		vertex_state.constantCount = vertex_constants.size();
		vertex_state.constants = vertex_constants.ptr();
	}

	// --- Primitive state ---
	WGPUPrimitiveState primitive = {};
	bool is_strip = false;
	switch (p_render_primitive) {
		case RENDER_PRIMITIVE_POINTS:
			primitive.topology = WGPUPrimitiveTopology_PointList;
			break;
		case RENDER_PRIMITIVE_LINES:
		case RENDER_PRIMITIVE_LINES_WITH_ADJACENCY:
			primitive.topology = WGPUPrimitiveTopology_LineList;
			break;
		case RENDER_PRIMITIVE_LINESTRIPS:
		case RENDER_PRIMITIVE_LINESTRIPS_WITH_ADJACENCY:
			primitive.topology = WGPUPrimitiveTopology_LineStrip;
			is_strip = true;
			break;
		case RENDER_PRIMITIVE_TRIANGLES:
		case RENDER_PRIMITIVE_TRIANGLES_WITH_ADJACENCY:
			primitive.topology = WGPUPrimitiveTopology_TriangleList;
			break;
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS:
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_AJACENCY:
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_RESTART_INDEX:
			primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
			is_strip = true;
			break;
		default:
			primitive.topology = WGPUPrimitiveTopology_TriangleList;
			break;
	}
	if (is_strip) {
		primitive.stripIndexFormat = WGPUIndexFormat_Uint32;
	}
	switch (p_rasterization_state.cull_mode) {
		case POLYGON_CULL_FRONT:
			primitive.cullMode = WGPUCullMode_Front;
			break;
		case POLYGON_CULL_BACK:
			primitive.cullMode = WGPUCullMode_Back;
			break;
		default:
			primitive.cullMode = WGPUCullMode_None;
			break;
	}
	primitive.frontFace = (p_rasterization_state.front_face == POLYGON_FRONT_FACE_CLOCKWISE)
			? WGPUFrontFace_CW
			: WGPUFrontFace_CCW;

	// --- Multisample state ---
	WGPUMultisampleState multisample = {};
	// TextureSamples enum: 0=1x, 1=2x, 2=4x, 3=8x ... → 1 << sample_count
	multisample.count = (uint32_t)(1u << (uint32_t)p_multisample_state.sample_count);
	multisample.mask = 0xFFFFFFFFu;
	multisample.alphaToCoverageEnabled = p_multisample_state.enable_alpha_to_coverage;

	// --- Depth/stencil helpers ---
	auto map_compare = [](CompareOperator op) -> WGPUCompareFunction {
		switch (op) {
			case COMPARE_OP_NEVER:
				return WGPUCompareFunction_Never;
			case COMPARE_OP_LESS:
				return WGPUCompareFunction_Less;
			case COMPARE_OP_EQUAL:
				return WGPUCompareFunction_Equal;
			case COMPARE_OP_LESS_OR_EQUAL:
				return WGPUCompareFunction_LessEqual;
			case COMPARE_OP_GREATER:
				return WGPUCompareFunction_Greater;
			case COMPARE_OP_NOT_EQUAL:
				return WGPUCompareFunction_NotEqual;
			case COMPARE_OP_GREATER_OR_EQUAL:
				return WGPUCompareFunction_GreaterEqual;
			default:
				return WGPUCompareFunction_Always;
		}
	};
	auto map_stencil_op = [](StencilOperation op) -> WGPUStencilOperation {
		switch (op) {
			case STENCIL_OP_KEEP:
				return WGPUStencilOperation_Keep;
			case STENCIL_OP_ZERO:
				return WGPUStencilOperation_Zero;
			case STENCIL_OP_REPLACE:
				return WGPUStencilOperation_Replace;
			case STENCIL_OP_INCREMENT_AND_CLAMP:
				return WGPUStencilOperation_IncrementClamp;
			case STENCIL_OP_DECREMENT_AND_CLAMP:
				return WGPUStencilOperation_DecrementClamp;
			case STENCIL_OP_INVERT:
				return WGPUStencilOperation_Invert;
			case STENCIL_OP_INCREMENT_AND_WRAP:
				return WGPUStencilOperation_IncrementWrap;
			case STENCIL_OP_DECREMENT_AND_WRAP:
				return WGPUStencilOperation_DecrementWrap;
			default:
				return WGPUStencilOperation_Keep;
		}
	};

	// --- Depth/stencil state ---
	WGPUDepthStencilState ds = {};
	WGPUDepthStencilState *ds_ptr = nullptr;
	if ((int32_t)subpass.depth_stencil_reference.attachment != ATTACHMENT_UNUSED) {
		uint32_t ds_att = (uint32_t)subpass.depth_stencil_reference.attachment;
		ds.format = (ds_att < (uint32_t)rp->attachments.size())
				? _data_format_to_wgpu(rp->attachments[ds_att].format)
				: WGPUTextureFormat_Depth24Plus;

		// Like Vulkan/D3D12/Metal, a disabled depth test also disables depth writes.
		ds.depthWriteEnabled = (p_depth_stencil_state.enable_depth_test && p_depth_stencil_state.enable_depth_write)
				? WGPUOptionalBool_True
				: WGPUOptionalBool_False;
		ds.depthCompare = p_depth_stencil_state.enable_depth_test
				? map_compare(p_depth_stencil_state.depth_compare_operator)
				: WGPUCompareFunction_Always;

		if (p_depth_stencil_state.enable_stencil) {
			ds.stencilFront.compare = map_compare(p_depth_stencil_state.front_op.compare);
			ds.stencilFront.failOp = map_stencil_op(p_depth_stencil_state.front_op.fail);
			ds.stencilFront.depthFailOp = map_stencil_op(p_depth_stencil_state.front_op.depth_fail);
			ds.stencilFront.passOp = map_stencil_op(p_depth_stencil_state.front_op.pass);
			ds.stencilBack.compare = map_compare(p_depth_stencil_state.back_op.compare);
			ds.stencilBack.failOp = map_stencil_op(p_depth_stencil_state.back_op.fail);
			ds.stencilBack.depthFailOp = map_stencil_op(p_depth_stencil_state.back_op.depth_fail);
			ds.stencilBack.passOp = map_stencil_op(p_depth_stencil_state.back_op.pass);
			ds.stencilReadMask = p_depth_stencil_state.front_op.compare_mask;
			ds.stencilWriteMask = p_depth_stencil_state.front_op.write_mask;
			if (p_depth_stencil_state.front_op.compare_mask != p_depth_stencil_state.back_op.compare_mask ||
					p_depth_stencil_state.front_op.write_mask != p_depth_stencil_state.back_op.write_mask) {
				WARN_PRINT_ONCE("WebGPU: Front and back stencil masks differ, but WebGPU only supports a single mask pair. Back-face masks will be ignored (front-face masks used for both).");
			}
		} else {
			ds.stencilFront = { WGPUCompareFunction_Always, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep };
			ds.stencilBack = { WGPUCompareFunction_Always, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep };
			ds.stencilReadMask = 0xFF;
			ds.stencilWriteMask = 0xFF;
		}
		if (p_rasterization_state.depth_bias_enabled) {
			ds.depthBias = (int32_t)p_rasterization_state.depth_bias_constant_factor;
			ds.depthBiasSlopeScale = p_rasterization_state.depth_bias_slope_factor;
			ds.depthBiasClamp = p_rasterization_state.depth_bias_clamp;
		}
		ds_ptr = &ds;
	}

	// --- Blend factor/op helpers ---
	auto blend_factor = [](BlendFactor f) -> WGPUBlendFactor {
		switch (f) {
			case BLEND_FACTOR_ZERO:
				return WGPUBlendFactor_Zero;
			case BLEND_FACTOR_ONE:
				return WGPUBlendFactor_One;
			case BLEND_FACTOR_SRC_COLOR:
				return WGPUBlendFactor_Src;
			case BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
				return WGPUBlendFactor_OneMinusSrc;
			case BLEND_FACTOR_DST_COLOR:
				return WGPUBlendFactor_Dst;
			case BLEND_FACTOR_ONE_MINUS_DST_COLOR:
				return WGPUBlendFactor_OneMinusDst;
			case BLEND_FACTOR_SRC_ALPHA:
				return WGPUBlendFactor_SrcAlpha;
			case BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
				return WGPUBlendFactor_OneMinusSrcAlpha;
			case BLEND_FACTOR_DST_ALPHA:
				return WGPUBlendFactor_DstAlpha;
			case BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
				return WGPUBlendFactor_OneMinusDstAlpha;
			case BLEND_FACTOR_CONSTANT_COLOR:
				return WGPUBlendFactor_Constant;
			case BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
				return WGPUBlendFactor_OneMinusConstant;
			case BLEND_FACTOR_CONSTANT_ALPHA:
				return WGPUBlendFactor_Constant;
			case BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA:
				return WGPUBlendFactor_OneMinusConstant;
			case BLEND_FACTOR_SRC_ALPHA_SATURATE:
				return WGPUBlendFactor_SrcAlphaSaturated;
			default:
				return WGPUBlendFactor_Zero;
		}
	};
	auto blend_op = [](BlendOperation op) -> WGPUBlendOperation {
		switch (op) {
			case BLEND_OP_ADD:
				return WGPUBlendOperation_Add;
			case BLEND_OP_SUBTRACT:
				return WGPUBlendOperation_Subtract;
			case BLEND_OP_REVERSE_SUBTRACT:
				return WGPUBlendOperation_ReverseSubtract;
			case BLEND_OP_MINIMUM:
				return WGPUBlendOperation_Min;
			case BLEND_OP_MAXIMUM:
				return WGPUBlendOperation_Max;
			default:
				return WGPUBlendOperation_Add;
		}
	};

	// --- Color targets (one per fragment shader @location()) ---
	uint32_t color_target_count = p_color_attachments.size();
	LocalVector<WGPUColorTargetState> color_targets;
	LocalVector<WGPUBlendState> blend_states;
	color_targets.resize(color_target_count);
	blend_states.resize(color_target_count);
	// Depth-only passes have no color targets. Calling memset with a null
	// pointer is undefined even for zero bytes; GCC can then remove the
	// LocalVector destructor's null checks and report invalid frees.
	if (color_target_count > 0) {
		memset(color_targets.ptr(), 0, sizeof(WGPUColorTargetState) * color_target_count);
		memset(blend_states.ptr(), 0, sizeof(WGPUBlendState) * color_target_count);
	}

	for (uint32_t i = 0; i < color_target_count; i++) {
		// Get texture format from subpass color reference i.
		WGPUTextureFormat fmt = WGPUTextureFormat_RGBA8Unorm; // Placeholder for unused slots.
		if (i < (uint32_t)subpass.color_references.size()) {
			int32_t att_idx = subpass.color_references[i].attachment;
			if (att_idx != ATTACHMENT_UNUSED && (uint32_t)att_idx < (uint32_t)rp->attachments.size()) {
				const RDD::Attachment &att_desc = rp->attachments[att_idx];
				fmt = _data_format_to_wgpu(att_desc.format);
				// Mirror the R8/RG8/R16/RG16 → 32-bit promotion in texture_create() for
				// attachments that will also be used as storage textures. Canvas SDF
				// (renderer_canvas_render_rd.cpp) declares R8_UNORM + STORAGE_BIT +
				// COLOR_ATTACHMENT_BIT; the texture gets upgraded to R32Float but the
				// pipeline would otherwise still target R8Unorm → validation mismatch.
				if (att_desc.usage_flags & TEXTURE_USAGE_STORAGE_BIT) {
					fmt = _promote_storage_format(fmt);
				}
				if (fmt == WGPUTextureFormat_Undefined) {
					fmt = WGPUTextureFormat_RGBA8Unorm;
				}
			}
		}
		color_targets[i].format = fmt;

		if (p_color_attachments[i] == ATTACHMENT_UNUSED) {
			// A null target: a format would require an attachment there even with all writes disabled.
			color_targets[i].format = WGPUTextureFormat_Undefined;
			color_targets[i].writeMask = WGPUColorWriteMask_None;
			continue;
		}

		// Blend state from p_blend_state.attachments[i].
		if (i < (uint32_t)p_blend_state.attachments.size()) {
			const PipelineColorBlendState::Attachment &ba = p_blend_state.attachments[i];
			WGPUColorWriteMask mask = WGPUColorWriteMask_None;
			if (ba.write_r) {
				mask |= WGPUColorWriteMask_Red;
			}
			if (ba.write_g) {
				mask |= WGPUColorWriteMask_Green;
			}
			if (ba.write_b) {
				mask |= WGPUColorWriteMask_Blue;
			}
			if (ba.write_a) {
				mask |= WGPUColorWriteMask_Alpha;
			}
			// Strip alpha writes for ALL pipelines targeting the swap chain format.
			// Chrome ignores CompositeAlphaMode_Opaque and composites alpha=0
			// against a gray/white background. The swap chain (BGRA8Unorm) is the
			// only BGRA render target — internal targets use RGBA formats.
			// Stripping alpha for blended pipelines too ensures the clear value's
			// alpha=1 is never overwritten by shader output.
			if (fmt == WGPUTextureFormat_BGRA8Unorm) {
				mask &= ~WGPUColorWriteMask_Alpha;
				static int _alpha_strip_log = 0;
				if (_alpha_strip_log < 10) {
					[[maybe_unused]] const char *sname = (p_shader.id) ? ((WGShader *)(p_shader.id))->name.utf8().get_data() : "?";
					WEBGPU_DIAG({ console.log('[ALPHA-STRIP] Pipeline #' + $0 + ' fmt=BGRA8Unorm mask=' + $1 + ' blend=' + $2 + ' shader=' + UTF8ToString($3)); }, _alpha_strip_log, (int)mask, ba.enable_blend ? 1 : 0, sname);
					_alpha_strip_log++;
				}
			}
			color_targets[i].writeMask = mask;
			if (ba.enable_blend) {
				// float32-blendable: if the device lacks this feature, blending on
				// float32 render targets causes GPU validation errors. Skip the
				// blend state for these targets — writes still happen (writeMask),
				// but the GPU composites without blending.
				if (!float32_blendable_supported && _is_float32_format(fmt)) {
					static int _f32_blend_skip_log = 0;
					if (_f32_blend_skip_log < 10) {
						[[maybe_unused]] const char *sname = (p_shader.id) ? ((WGShader *)(p_shader.id))->name.utf8().get_data() : "?";
						WEBGPU_DIAG({ console.log('[FLOAT32-BLEND-SKIP] Pipeline fmt=' + $0 + ' shader=' + UTF8ToString($1) + ' — device lacks float32-blendable, disabling blend'); }, (int)fmt, sname);
						_f32_blend_skip_log++;
					}
					// Don't set blend — leave color_targets[i].blend = nullptr.
				} else {
					blend_states[i].color = { blend_op(ba.color_blend_op), blend_factor(ba.src_color_blend_factor), blend_factor(ba.dst_color_blend_factor) };
					blend_states[i].alpha = { blend_op(ba.alpha_blend_op), blend_factor(ba.src_alpha_blend_factor), blend_factor(ba.dst_alpha_blend_factor) };
					color_targets[i].blend = &blend_states[i];
				}
			}
		} else {
			color_targets[i].writeMask = WGPUColorWriteMask_All;
		}
	}

	// --- Fragment state ---
	WGPUFragmentState frag = {};
	frag.module = fragment_module;
	frag.entryPoint = { "main", WGPU_STRLEN };
	frag.targetCount = color_target_count;
	frag.targets = color_targets.ptr();
	if (use_override_path && !frag_constants.is_empty()) {
		frag.constantCount = frag_constants.size();
		frag.constants = frag_constants.ptr();
	}

	// --- Build and create render pipeline ---
	WGPURenderPipelineDescriptor desc = {};
	desc.layout = shader->pipeline_layout;
	desc.vertex = vertex_state;
	desc.primitive = primitive;
	desc.depthStencil = ds_ptr;
	desc.multisample = multisample;
	if (frag.module) {
		desc.fragment = &frag;
	}

	// Label the pipeline with the shader name + a monotonic id so Dawn's
	// validation error messages (and the JS-side create patch) identify it.
	static int _pcreate_id = 0;
	int _pid = _pcreate_id++;
	String _pipeline_label_str = "pipe#" + itos(_pid) + ":" + shader->name;
	CharString _pipeline_label_cs = _pipeline_label_str.utf8();
	desc.label = { _pipeline_label_cs.get_data(), WGPU_STRLEN };

	WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
	if (!pipeline) {
		if (specialized_vertex) {
			wgpuShaderModuleRelease(specialized_vertex);
		}
		if (specialized_fragment) {
			wgpuShaderModuleRelease(specialized_fragment);
		}
		ERR_FAIL_V_MSG(PipelineID(), "WebGPU: Failed to create render pipeline.");
	}

	WGPipelineWrapper *pw = memnew(WGPipelineWrapper);
	pw->type = WGPipelineWrapper::RENDER;
	pw->render_handle = pipeline;
	pw->shader = shader;
	pw->specialized_modules[SHADER_STAGE_VERTEX] = specialized_vertex;
	pw->specialized_modules[SHADER_STAGE_FRAGMENT] = specialized_fragment;
	pw->stencil_reference = p_depth_stencil_state.front_op.reference;
	pw->is_strip = is_strip;
	if (is_strip) {
		// Create a Uint16 variant — WebGPU bakes stripIndexFormat into the pipeline,
		// but Godot only knows the index format at bind time.
		desc.primitive.stripIndexFormat = WGPUIndexFormat_Uint16;
		pw->render_handle_u16 = wgpuDeviceCreateRenderPipeline(device, &desc);
		// Non-fatal if this fails — Uint32 variant still works for the common case.
		if (!pw->render_handle_u16) {
			WARN_PRINT_ONCE("WebGPU: failed to create Uint16 strip pipeline variant.");
		}
	}
	return PipelineID(pw);
}

// =============================================================================
// COMPUTE
// =============================================================================

void RenderingDeviceDriverWebGPU::command_bind_compute_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGPipelineWrapper *pw = (WGPipelineWrapper *)(p_pipeline.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(pw);
	if (pw->deferred) {
		_create_deferred_pipeline(pw);
	}
	ERR_FAIL_NULL(pw->compute_handle);

	// Ensure we're in a compute pass (end any active render pass).
	if (cmd->active_encoder == WGCommandBuffer::RENDER) {
		cmd->end_active_encoder();
	}
	if (cmd->active_encoder != WGCommandBuffer::COMPUTE) {
		WGPUComputePassDescriptor pass_desc = {};
		cmd->compute_encoder = wgpuCommandEncoderBeginComputePass(cmd->encoder, &pass_desc);
		cmd->active_encoder = WGCommandBuffer::COMPUTE;
	}

	wgpuComputePassEncoderSetPipeline(cmd->compute_encoder, pw->compute_handle);
	cmd->render_state.current_pipeline = pw;

	// Pre-bind empty bind groups at pipeline layout gap slots.
	if (pw->shader) {
		for (uint32_t gap_idx : pw->shader->gap_bind_group_indices) {
			wgpuComputePassEncoderSetBindGroup(cmd->compute_encoder, gap_idx, empty_bind_group, 0, nullptr);
		}
	}
}

void RenderingDeviceDriverWebGPU::command_bind_compute_uniform_sets(CommandBufferID p_cmd_buffer, VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->compute_encoder);

	WGShader *pipeline_shader = cmd->render_state.current_pipeline ? cmd->render_state.current_pipeline->shader : nullptr;

	// Task 7.5: mirror the render path's dynamic offset unpacking.
	static constexpr uint32_t MAX_DYNAMIC_BUFFERS = 8;
	uint32_t dyn_shift = 0;

	for (uint32_t i = 0; i < p_set_count; i++) {
		WGUniformSet *us = (WGUniformSet *)(p_uniform_sets[i].id);
		if (us && us->handle) {
			uint32_t set_idx = p_first_set_index + i;
			WGPUBindGroup bg_to_bind = _get_compatible_bind_group(us, pipeline_shader, set_idx);

			uint32_t set_dyn_offsets[MAX_DYNAMIC_BUFFERS] = {};
			uint32_t num_dyn = us->dynamic_buffers.size();
			if (num_dyn > MAX_DYNAMIC_BUFFERS) {
				num_dyn = MAX_DYNAMIC_BUFFERS;
			}
			for (uint32_t j = 0; j < num_dyn; j++) {
				uint32_t frame_idx = (p_dynamic_offsets >> dyn_shift) & 0xFu;
				dyn_shift += 4u;
				const WGBuffer *dbuf = us->dynamic_buffers[j];
				set_dyn_offsets[j] = (uint32_t)(frame_idx * (dbuf ? dbuf->per_frame_size : 0));
			}

			const bool is_pc_merged = (pipeline_shader && pipeline_shader->push_constant_size > 0 &&
					set_idx == pipeline_shader->push_constant_bind_group &&
					pipeline_shader->merged_pc_group_layout != nullptr);

			if (is_pc_merged) {
				// Save material offsets; _flush_push_constants preserves them each dispatch.
				cmd->pc_group_material_dyn_count = num_dyn;
				for (uint32_t j = 0; j < num_dyn && j < WGCommandBuffer::MAX_PC_GROUP_MATERIAL_DYN; j++) {
					cmd->pc_group_material_dyn_offsets[j] = set_dyn_offsets[j];
				}
				if (num_dyn < MAX_DYNAMIC_BUFFERS) {
					set_dyn_offsets[num_dyn] = cmd->last_flushed_pc_offset;
					num_dyn++;
				}
				cmd->current_pc_bind_group = bg_to_bind;
				wgpuComputePassEncoderSetBindGroup(cmd->compute_encoder, set_idx, bg_to_bind, num_dyn, set_dyn_offsets);
			} else if (pipeline_shader && pipeline_shader->push_constant_size > 0 &&
					set_idx == pipeline_shader->push_constant_bind_group) {
				cmd->current_pc_bind_group = nullptr;
				cmd->pc_group_material_dyn_count = 0;
				wgpuComputePassEncoderSetBindGroup(cmd->compute_encoder, set_idx, bg_to_bind, num_dyn, num_dyn ? set_dyn_offsets : nullptr);
			} else {
				wgpuComputePassEncoderSetBindGroup(cmd->compute_encoder, set_idx, bg_to_bind, num_dyn, num_dyn ? set_dyn_offsets : nullptr);
			}
			// Save full state for mid-pass restart on ring overflow.
			if (set_idx < WGCommandBuffer::MAX_BIND_GROUPS) {
				auto &bs = cmd->last_bound_state[set_idx];
				bs.group = bg_to_bind;
				bs.dynamic_offset_count = num_dyn;
				for (uint32_t j = 0; j < num_dyn && j < WGCommandBuffer::MAX_BIND_GROUP_DYN_OFFSETS; j++) {
					bs.dynamic_offsets[j] = set_dyn_offsets[j];
				}
			}
		}
	}
}

void RenderingDeviceDriverWebGPU::command_compute_dispatch(CommandBufferID p_cmd_buffer, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_COND(!cmd->compute_encoder);

	if (cmd->render_state.current_pipeline) {
		_flush_push_constants(cmd, cmd->render_state.current_pipeline->shader);
	}

	wgpuComputePassEncoderDispatchWorkgroups(cmd->compute_encoder, p_x_groups, p_y_groups, p_z_groups);
}

void RenderingDeviceDriverWebGPU::command_compute_dispatch_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	WGBuffer *indirect = (WGBuffer *)(p_indirect_buffer.id);
	ERR_FAIL_NULL(cmd);
	ERR_FAIL_NULL(indirect);
	ERR_FAIL_COND(!cmd->compute_encoder);

	if (cmd->render_state.current_pipeline) {
		_flush_push_constants(cmd, cmd->render_state.current_pipeline->shader);
	}

	wgpuComputePassEncoderDispatchWorkgroupsIndirect(cmd->compute_encoder, indirect->handle, p_offset);
}

RDD::PipelineID RenderingDeviceDriverWebGPU::compute_pipeline_create(ShaderID p_shader, VectorView<PipelineSpecializationConstant> p_specialization_constants) {
#ifdef WEB_ENABLED
	// Created on first bind, see render_pipeline_create.
	WGShader *shader = (WGShader *)(p_shader.id);
	ERR_FAIL_COND_V(!shader, PipelineID());
	WGPipelineWrapper *pw = memnew(WGPipelineWrapper);
	pw->type = WGPipelineWrapper::COMPUTE;
	pw->shader = shader;
	pw->deferred = memnew(WGDeferredPipeline);
	for (uint32_t i = 0; i < p_specialization_constants.size(); i++) {
		pw->deferred->specialization_constants.push_back(p_specialization_constants[i]);
	}
	return PipelineID(pw);
#else
	return _compute_pipeline_create_now(p_shader, p_specialization_constants);
#endif
}

RDD::PipelineID RenderingDeviceDriverWebGPU::_compute_pipeline_create_now(ShaderID p_shader, VectorView<PipelineSpecializationConstant> p_specialization_constants) {
	WGShader *shader = (WGShader *)(p_shader.id);
	ERR_FAIL_COND_V(!shader, PipelineID());
	WGPUShaderModule compute_module = _stage_module(shader, SHADER_STAGE_COMPUTE);
	ERR_FAIL_COND_V_MSG(!compute_module, PipelineID(),
			"WebGPU: compute_pipeline_create called with a shader that has no compute stage.");

	// Specialization constants: override path or legacy module-patching path.
	WGPUShaderModule specialized_compute = nullptr;
	LocalVector<WGPUConstantEntry> compute_constants;
	LocalVector<CharString> compute_key_storage;
	bool use_override_path = shader->has_override_declarations;
	if (p_specialization_constants.size() > 0) {
		if (use_override_path) {
			const HashSet<uint32_t> &cmp_ids = shader->stage_override_ids[SHADER_STAGE_COMPUTE];
			for (uint32_t i = 0; i < p_specialization_constants.size(); i++) {
				const PipelineSpecializationConstant &sc = p_specialization_constants[i];
				if (!cmp_ids.has(sc.constant_id)) {
					continue; // This constant ID doesn't exist in the compute stage's WGSL.
				}
				WGPUConstantEntry entry = {};
				String key_str = itos(sc.constant_id);
				CharString key_cs = key_str.utf8();
				compute_key_storage.push_back(key_cs);
				entry.key = { compute_key_storage[compute_key_storage.size() - 1].get_data(), WGPU_STRLEN };
				switch (sc.type) {
					case PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL:
						entry.value = sc.bool_value ? 1.0 : 0.0;
						break;
					case PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT:
						entry.value = (double)sc.int_value;
						break;
					case PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT:
						entry.value = (double)sc.float_value;
						break;
				}
				compute_constants.push_back(entry);
			}
		} else if (!shader->stage_spirv[SHADER_STAGE_COMPUTE].is_empty()) {
			specialized_compute = _create_module_with_spec_constants(
					shader->stage_spirv[SHADER_STAGE_COMPUTE], p_specialization_constants, SHADER_STAGE_COMPUTE, shader);
			if (specialized_compute) {
				compute_module = specialized_compute;
			}
		}
	}

	WGPUComputePipelineDescriptor desc = {};
	desc.layout = shader->pipeline_layout;
	desc.compute.module = compute_module;
	desc.compute.entryPoint = { "main", WGPU_STRLEN };
	CharString label = shader->name.utf8();
	desc.label = { label.get_data(), WGPU_STRLEN };
	if (use_override_path && !compute_constants.is_empty()) {
		desc.compute.constantCount = compute_constants.size();
		desc.compute.constants = compute_constants.ptr();
	}

	WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &desc);
	if (!pipeline) {
		if (specialized_compute) {
			wgpuShaderModuleRelease(specialized_compute);
		}
		ERR_FAIL_V_MSG(PipelineID(), "WebGPU: Failed to create compute pipeline.");
	}

	WGPipelineWrapper *pw = memnew(WGPipelineWrapper);
	pw->type = WGPipelineWrapper::COMPUTE;
	pw->compute_handle = pipeline;
	pw->shader = shader;
	pw->specialized_modules[SHADER_STAGE_COMPUTE] = specialized_compute;
	return PipelineID(pw);
}

// =============================================================================
// QUERIES
// =============================================================================

// Timestamp queries aren't exposed: their async readback can't be completed within a frame on
// web. Like Metal, hand out a dummy pool whose results are zeros.
RDD::QueryPoolID RenderingDeviceDriverWebGPU::timestamp_query_pool_create(uint32_t p_query_count) {
	return QueryPoolID(1);
}

void RenderingDeviceDriverWebGPU::timestamp_query_pool_free(QueryPoolID p_pool_id) {
}

void RenderingDeviceDriverWebGPU::timestamp_query_pool_get_results(QueryPoolID p_pool_id, uint32_t p_query_count, uint64_t *r_results) {
	memset(r_results, 0, sizeof(uint64_t) * p_query_count);
}

uint64_t RenderingDeviceDriverWebGPU::timestamp_query_result_to_time(uint64_t p_result) {
	return p_result;
}

void RenderingDeviceDriverWebGPU::command_timestamp_query_pool_reset(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_query_count) {
}

void RenderingDeviceDriverWebGPU::command_timestamp_write(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_index) {
}

// =============================================================================
// LABELS & DEBUG
// =============================================================================

void RenderingDeviceDriverWebGPU::command_begin_label(CommandBufferID p_cmd_buffer, const char *p_label_name, const Color &p_color) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);

	if (cmd->render_encoder) {
		wgpuRenderPassEncoderPushDebugGroup(cmd->render_encoder, WGPUStringView{ p_label_name, WGPU_STRLEN });
	} else if (cmd->compute_encoder) {
		wgpuComputePassEncoderPushDebugGroup(cmd->compute_encoder, WGPUStringView{ p_label_name, WGPU_STRLEN });
	} else if (cmd->encoder) {
		wgpuCommandEncoderPushDebugGroup(cmd->encoder, WGPUStringView{ p_label_name, WGPU_STRLEN });
	}
}

void RenderingDeviceDriverWebGPU::command_end_label(CommandBufferID p_cmd_buffer) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);

	if (cmd->render_encoder) {
		wgpuRenderPassEncoderPopDebugGroup(cmd->render_encoder);
	} else if (cmd->compute_encoder) {
		wgpuComputePassEncoderPopDebugGroup(cmd->compute_encoder);
	} else if (cmd->encoder) {
		wgpuCommandEncoderPopDebugGroup(cmd->encoder);
	}
}

void RenderingDeviceDriverWebGPU::command_insert_breadcrumb(CommandBufferID p_cmd_buffer, uint32_t p_data) {
	WGCommandBuffer *cmd = (WGCommandBuffer *)(p_cmd_buffer.id);
	ERR_FAIL_NULL(cmd);

	char marker[32];
	snprintf(marker, sizeof(marker), "breadcrumb_%u", p_data);

	if (cmd->render_encoder) {
		wgpuRenderPassEncoderInsertDebugMarker(cmd->render_encoder, WGPUStringView{ marker, WGPU_STRLEN });
	} else if (cmd->compute_encoder) {
		wgpuComputePassEncoderInsertDebugMarker(cmd->compute_encoder, WGPUStringView{ marker, WGPU_STRLEN });
	} else if (cmd->encoder) {
		wgpuCommandEncoderInsertDebugMarker(cmd->encoder, WGPUStringView{ marker, WGPU_STRLEN });
	}
}

// =============================================================================
// SUBMISSION
// =============================================================================

void RenderingDeviceDriverWebGPU::begin_segment(uint32_t p_frame_index, uint32_t p_frames_drawn) {
	frame_index = p_frame_index;
	frames_drawn = p_frames_drawn;

#if defined(WEBGPU_VERBOSE) && defined(WEB_ENABLED)
	// Performance counters, logged once per second.
	perf.frames_since_log++;
	double now = EM_ASM_DOUBLE({ return performance.now(); });
	if (perf.last_log_time == 0) {
		perf.last_log_time = now;
	} else if (now - perf.last_log_time >= 1000.0) {
		double elapsed = (now - perf.last_log_time) / 1000.0;
		uint32_t fps = (uint32_t)(perf.frames_since_log / elapsed);
		uint32_t f = perf.frames_since_log > 0 ? perf.frames_since_log : 1;
		EM_ASM({ console.log('[PERF] fps=' + $0 +
						 ' draws/f=' + $1 +
						 ' SetBG/f=' + $2 +
						 ' PC/f=' + $3 +
						 ' RP/f=' + $4 +
						 ' SetVB/f=' + $5 +
						 ' FI/f=' + $6 +
						 ' RingOF/f=' + $7); }, fps, perf.draw_calls / f, perf.set_bind_group_calls / f, perf.push_constant_writes / f, perf.render_passes / f, perf.set_vertex_buffer_calls / f, perf.first_instance_draws / f, perf.ring_overflows / f);
		perf.reset();
		perf.frames_since_log = 0;
		perf.last_log_time = now;
	}
#endif

	// Reset push constant ring buffer offset and shadow buffer tracking at the start of each segment.
	push_constant_ring_offset = 0;
	push_constant_page = 0;
	push_constant_shadow_dirty_start = UINT64_MAX;
	push_constant_shadow_dirty_end = 0;
}

void RenderingDeviceDriverWebGPU::end_segment() {
	// Nothing to do.
}

// =============================================================================
// MISC
// =============================================================================

void RenderingDeviceDriverWebGPU::set_object_name(ObjectType p_type, ID p_driver_id, const String &p_name) {
	// Propagate names to Dawn so validation errors reference the Godot resource
	// (otherwise Dawn reports "[Texture (unlabeled ...)]"). CharString must live
	// until wgpuXxxSetLabel returns — Dawn copies the string internally.
	if (p_driver_id.id == 0) {
		return;
	}
	const CharString name_utf8 = p_name.utf8();
	const WGPUStringView label = { name_utf8.get_data(), WGPU_STRLEN };
	switch (p_type) {
		case OBJECT_TYPE_ACCELERATION_STRUCTURE:
		case OBJECT_TYPE_RAYTRACING_PIPELINE:
			break;
		case OBJECT_TYPE_TEXTURE: {
			WGTexture *tex = (WGTexture *)(p_driver_id.id);
			if (tex && tex->handle) {
				wgpuTextureSetLabel(tex->handle, label);
			}
		} break;
		case OBJECT_TYPE_SAMPLER: {
			// SamplerID stores the raw WGPUSampler handle directly.
			WGPUSampler sampler = (WGPUSampler)(p_driver_id.id);
			wgpuSamplerSetLabel(sampler, label);
		} break;
		case OBJECT_TYPE_BUFFER: {
			WGBuffer *buf = (WGBuffer *)(p_driver_id.id);
			if (buf && buf->handle) {
				wgpuBufferSetLabel(buf->handle, label);
			}
		} break;
		case OBJECT_TYPE_SHADER: {
			WGShader *shader = (WGShader *)(p_driver_id.id);
			if (shader) {
				shader->name = p_name;
				for (uint32_t i = 0; i < 6; i++) {
					if (shader->stage_modules[i]) {
						wgpuShaderModuleSetLabel(shader->stage_modules[i], label);
					}
				}
			}
		} break;
		case OBJECT_TYPE_UNIFORM_SET: {
			WGUniformSet *us = (WGUniformSet *)(p_driver_id.id);
			if (us && us->handle) {
				wgpuBindGroupSetLabel(us->handle, label);
			}
		} break;
		case OBJECT_TYPE_PIPELINE: {
			WGPipelineWrapper *pw = (WGPipelineWrapper *)(p_driver_id.id);
			if (!pw) {
				break;
			}
			if (pw->type == WGPipelineWrapper::RENDER && pw->render_handle) {
				wgpuRenderPipelineSetLabel(pw->render_handle, label);
			} else if (pw->type == WGPipelineWrapper::COMPUTE && pw->compute_handle) {
				wgpuComputePipelineSetLabel(pw->compute_handle, label);
			}
		} break;
	}
}

uint64_t RenderingDeviceDriverWebGPU::get_resource_native_handle(DriverResource p_type, ID p_driver_id) {
	switch (p_type) {
		case DRIVER_RESOURCE_LOGICAL_DEVICE:
			return (uint64_t)device;
		case DRIVER_RESOURCE_TOPMOST_OBJECT:
			return (uint64_t)context_driver->get_instance();
		case DRIVER_RESOURCE_COMMAND_QUEUE:
			return (uint64_t)queue;
		case DRIVER_RESOURCE_TEXTURE: {
			const WGTexture *tex = (const WGTexture *)p_driver_id.id;
			return tex ? (uint64_t)tex->gpu_handle() : 0;
		}
		case DRIVER_RESOURCE_TEXTURE_VIEW: {
			const WGTexture *tex = (const WGTexture *)p_driver_id.id;
			return tex ? (uint64_t)tex->default_view : 0;
		}
		case DRIVER_RESOURCE_TEXTURE_DATA_FORMAT: {
			const WGTexture *tex = (const WGTexture *)p_driver_id.id;
			return tex ? (uint64_t)tex->format : 0;
		}
		case DRIVER_RESOURCE_SAMPLER:
			return p_driver_id.id; // SamplerID is the WGPUSampler.
		case DRIVER_RESOURCE_UNIFORM_SET: {
			const WGUniformSet *us = (const WGUniformSet *)p_driver_id.id;
			return us ? (uint64_t)us->handle : 0;
		}
		case DRIVER_RESOURCE_BUFFER: {
			const WGBuffer *buf = (const WGBuffer *)p_driver_id.id;
			return buf ? (uint64_t)buf->handle : 0;
		}
		case DRIVER_RESOURCE_COMPUTE_PIPELINE: {
			WGPipelineWrapper *pw = (WGPipelineWrapper *)p_driver_id.id;
			if (pw && pw->deferred) {
				_create_deferred_pipeline(pw);
			}
			return (pw && pw->type == WGPipelineWrapper::COMPUTE) ? (uint64_t)pw->compute_handle : 0;
		}
		case DRIVER_RESOURCE_RENDER_PIPELINE: {
			WGPipelineWrapper *pw = (WGPipelineWrapper *)p_driver_id.id;
			if (pw && pw->deferred) {
				_create_deferred_pipeline(pw);
			}
			return (pw && pw->type == WGPipelineWrapper::RENDER) ? (uint64_t)pw->render_handle : 0;
		}
		default:
			return 0;
	}
}

uint64_t RenderingDeviceDriverWebGPU::get_total_memory_used() {
	return 0; // TODO: Track internally.
}

uint64_t RenderingDeviceDriverWebGPU::get_lazily_memory_used() {
	return 0;
}

uint64_t RenderingDeviceDriverWebGPU::limit_get(Limit p_limit) {
	switch (p_limit) {
		case LIMIT_MAX_BOUND_UNIFORM_SETS:
			return device_limits.maxBindGroups;
		case LIMIT_MAX_FRAMEBUFFER_COLOR_ATTACHMENTS:
			return device_limits.maxColorAttachments;
		case LIMIT_MAX_TEXTURES_PER_UNIFORM_SET:
			return device_limits.maxSampledTexturesPerShaderStage;
		case LIMIT_MAX_SAMPLERS_PER_UNIFORM_SET:
			return device_limits.maxSamplersPerShaderStage;
		case LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET:
			return device_limits.maxStorageBuffersPerShaderStage;
		case LIMIT_MAX_STORAGE_IMAGES_PER_UNIFORM_SET:
			return device_limits.maxStorageTexturesPerShaderStage;
		case LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET:
			return device_limits.maxUniformBuffersPerShaderStage;
		case LIMIT_MAX_PUSH_CONSTANT_SIZE:
			return 128; // Emulated via ring buffer.
		case LIMIT_MAX_UNIFORM_BUFFER_SIZE:
			return device_limits.maxUniformBufferBindingSize;
		case LIMIT_MAX_TEXTURES_PER_SHADER_STAGE:
			return device_limits.maxSampledTexturesPerShaderStage;
		case LIMIT_MAX_SAMPLERS_PER_SHADER_STAGE:
			return device_limits.maxSamplersPerShaderStage;
		case LIMIT_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE:
			return device_limits.maxStorageBuffersPerShaderStage;
		case LIMIT_MAX_STORAGE_IMAGES_PER_SHADER_STAGE:
			return device_limits.maxStorageTexturesPerShaderStage;
		case LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE:
			return device_limits.maxUniformBuffersPerShaderStage;
		case LIMIT_MAX_TEXTURE_ARRAY_LAYERS:
			return device_limits.maxTextureArrayLayers;
		case LIMIT_MAX_TEXTURE_SIZE_1D:
			return device_limits.maxTextureDimension1D;
		case LIMIT_MAX_TEXTURE_SIZE_2D:
			return device_limits.maxTextureDimension2D;
		case LIMIT_MAX_TEXTURE_SIZE_3D:
			return device_limits.maxTextureDimension3D;
		case LIMIT_MAX_TEXTURE_SIZE_CUBE:
			return device_limits.maxTextureDimension2D; // Cube faces are 2D.
		case LIMIT_MAX_VERTEX_INPUT_ATTRIBUTE_OFFSET:
			return device_limits.maxVertexBufferArrayStride;
		case LIMIT_MAX_VERTEX_INPUT_ATTRIBUTES:
			return device_limits.maxVertexAttributes;
		case LIMIT_MAX_VERTEX_INPUT_BINDINGS:
			return device_limits.maxVertexBuffers;
		case LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE:
			return device_limits.maxComputeWorkgroupStorageSize;
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X:
			return device_limits.maxComputeWorkgroupsPerDimension;
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y:
			return device_limits.maxComputeWorkgroupsPerDimension;
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z:
			return device_limits.maxComputeWorkgroupsPerDimension;
		case LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS:
			return device_limits.maxComputeInvocationsPerWorkgroup;
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X:
			return device_limits.maxComputeWorkgroupSizeX;
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y:
			return device_limits.maxComputeWorkgroupSizeY;
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Z:
			return device_limits.maxComputeWorkgroupSizeZ;
		case LIMIT_MAX_SHADER_VARYINGS:
			return device_limits.maxInterStageShaderVariables;
		case LIMIT_SUBGROUP_SIZE:
			return 0; // Subgroups not available in WebGPU.
		case LIMIT_SUBGROUP_MIN_SIZE:
			return 0;
		case LIMIT_SUBGROUP_MAX_SIZE:
			return 0;
		case LIMIT_SUBGROUP_IN_SHADERS:
			return 0;
		case LIMIT_SUBGROUP_OPERATIONS:
			return 0;
		default:
			return 0;
	}
}

uint64_t RenderingDeviceDriverWebGPU::api_trait_get(ApiTrait p_trait) {
	switch (p_trait) {
		case API_TRAIT_BUFFER_MAP_REQUIRES_FLUSH:
			return 1;
		case API_TRAIT_HONORS_PIPELINE_BARRIERS:
			return 0; // No barriers in WebGPU.
		case API_TRAIT_SHADER_CHANGE_INVALIDATION:
			return SHADER_CHANGE_INVALIDATION_ALL_BOUND_UNIFORM_SETS;
		case API_TRAIT_TEXTURE_TRANSFER_ALIGNMENT:
			return 256;
		case API_TRAIT_TEXTURE_DATA_ROW_PITCH_STEP:
			return 256;
		case API_TRAIT_SECONDARY_VIEWPORT_SCISSOR:
			return 0;
		case API_TRAIT_CLEARS_WITH_COPY_ENGINE:
			return 0;
		case API_TRAIT_USE_GENERAL_IN_COPY_QUEUES:
			return 0;
		case API_TRAIT_BUFFERS_REQUIRE_TRANSITIONS:
			return 0;
		case API_TRAIT_TEXTURE_OUTPUTS_REQUIRE_CLEARS:
			return 0;
		// Skip transfer-worker path for layered Texture2DArray uploads; we
		// implement texture_initialize_direct_layered using wgpuQueueWriteTexture
		// directly (no GPU staging buffer, no command encoder, no barriers).
		// Eliminates a same-size wasted VRAM allocation per Texture2DArray
		// upload and the queue serialization that came with it.
		case API_TRAIT_TEXTURE_INITIALIZE_DIRECT_WRITE:
			return 1;
		// Use mappedAtCreation for buffer creation with initial data —
		// bypasses staging buffer + wgpuQueueWriteBuffer overhead entirely.
		case API_TRAIT_BUFFER_CREATE_MAPPED_AT_CREATION:
			return 1;
		// Cap the staging buffer pool to 16 MB. WebGPU staging blocks use
		// CPU shadow memory (memalloc), so idle blocks waste heap after
		// the loading spike. 16 MB is generous for per-frame dynamic
		// updates; overflow stalls briefly and reuses blocks.
		case API_TRAIT_STAGING_BUFFER_MAX_SIZE_MB:
			return 16;
		default:
			return RenderingDeviceDriver::api_trait_get(p_trait);
	}
}

RDD::DriverWorkarounds RenderingDeviceDriverWebGPU::get_driver_workarounds() const {
	DriverWorkarounds workarounds;
#ifdef WEB_ENABLED
	// mapAsync can't complete inside a synchronous call without Asyncify; see texture_get_data().
	workarounds.async_texture_readback = true;
#endif
	workarounds.no_input_attachments = true;
	workarounds.no_texture_swizzle = true;
	workarounds.no_storage_render_targets = true; // Dawn rejects sRGB view formats on storage textures.
	workarounds.limited_read_write_storage = !has_rw_storage_textures || !has_texture_formats_tier2;
	// Without texture-formats-tier1, R8 storage textures are promoted to R32, which supports read-write access.
	workarounds.limited_read_write_storage_8bit = !has_rw_storage_textures || (has_texture_formats_tier1 && !has_texture_formats_tier2);
	workarounds.force_omni_dual_paraboloid = true;
	workarounds.merge_shadow_passes = true;
	workarounds.batch_instance_draws = true;
	workarounds.skeleton_atlas = true;
	// Safari exposes 44 sampled textures per stage and Chrome on Metal 10 storage buffers; the full clustered scene
	// shader needs 48 and 11. Bakes must match, so the policy follows the driver rather than the adapter.
	workarounds.compact_scene_bindings = true;
	return workarounds;
}

bool RenderingDeviceDriverWebGPU::has_feature(Features p_feature) {
	switch (p_feature) {
		case SUPPORTS_HALF_FLOAT:
			return false; // WebGPU shader-f16 extension not reliably available; avoid f16 in generated SPIR-V.
		case SUPPORTS_FRAGMENT_SHADER_WITH_ONLY_SIDE_EFFECTS:
			return false; // WebGPU requires at least one color or depth/stencil attachment.
		case SUPPORTS_IMAGE_ATOMIC_32_BIT:
			return false; // WebGPU has no image atomics support.
		case SUPPORTS_MULTIVIEW:
			return false; // Not available in WebGPU.
		case SUPPORTS_ATTACHMENT_VRS:
			return false; // Not available in WebGPU.
		case SUPPORTS_METALFX_SPATIAL:
		case SUPPORTS_METALFX_TEMPORAL:
			return false; // Metal-only features.
		case SUPPORTS_BUFFER_DEVICE_ADDRESS:
			return false; // Not available in WebGPU.
		case SUPPORTS_VULKAN_MEMORY_MODEL:
			return false; // Not available in WebGPU.
		case SUPPORTS_FRAMEBUFFER_DEPTH_RESOLVE:
			return false; // Not available in WebGPU.
		case SUPPORTS_POINT_SIZE:
			return false; // Point size not controllable in WebGPU.
		default:
			return false;
	}
}

const RDD::MultiviewCapabilities &RenderingDeviceDriverWebGPU::get_multiview_capabilities() {
	return multiview_capabilities;
}

const RDD::FragmentShadingRateCapabilities &RenderingDeviceDriverWebGPU::get_fragment_shading_rate_capabilities() {
	return fsr_capabilities;
}

const RDD::FragmentDensityMapCapabilities &RenderingDeviceDriverWebGPU::get_fragment_density_map_capabilities() {
	return fdm_capabilities;
}

String RenderingDeviceDriverWebGPU::get_api_name() const {
	return "WebGPU";
}

String RenderingDeviceDriverWebGPU::get_api_version() const {
	return "1.0";
}

String RenderingDeviceDriverWebGPU::get_pipeline_cache_uuid() const {
	return "";
}

const RDD::Capabilities &RenderingDeviceDriverWebGPU::get_capabilities() const {
	return capabilities;
}

const RenderingShaderContainerFormat &RenderingDeviceDriverWebGPU::get_shader_container_format() const {
	return *shader_container_format;
}

#endif // WEBGPU_ENABLED
