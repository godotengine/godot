/**************************************************************************/
/*  metal3_objects.mm                                                     */
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

#import "metal3_objects.h"

#import "metal_utils.h"
#import "pixel_formats.h"
#import "rendering_device_driver_metal3.h"
#import "rendering_shader_container_metal.h"

#import <algorithm>

// We have to undefine these macros because they are defined in NSObjCRuntime.h.
#undef MIN
#undef MAX

using namespace MTL3;

MDCommandBuffer::MDCommandBuffer(id<MTLCommandQueue> p_queue, ::RenderingDeviceDriverMetal *p_device_driver) :
		_scratch(p_queue.device), device_driver(p_device_driver), queue(p_queue) {
	type = MDCommandBufferStateType::None;
	use_barriers = device_driver->use_barriers;
	if (use_barriers) {
		// Already validated availability if use_barriers is true.
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
		MTLResidencySetDescriptor *rs_desc = [MTLResidencySetDescriptor new];
		[rs_desc setInitialCapacity:10];
		rs_desc.label = @"Command Residency Set";
		NSError *error;
		_frame_state.rs = [p_queue.device newResidencySetWithDescriptor:rs_desc error:&error];
		CRASH_COND_MSG(error != nil, vformat("Failed to create residency set: %s", String(error.localizedDescription.UTF8String)));
		GODOT_CLANG_WARNING_POP
	}
}

void MDCommandBuffer::begin_label(const char *p_label_name, const Color &p_color) {
	NSString *s = [[NSString alloc] initWithBytesNoCopy:(void *)p_label_name length:strlen(p_label_name) encoding:NSUTF8StringEncoding freeWhenDone:NO];
	[command_buffer() pushDebugGroup:s];
}

void MDCommandBuffer::end_label() {
	[command_buffer() popDebugGroup];
}

void MDCommandBuffer::begin() {
	DEV_ASSERT(commandBuffer == nil && !state_begin);
	state_begin = true;
	bzero(pending_after_stages, sizeof(pending_after_stages));
	bzero(pending_before_queue_stages, sizeof(pending_before_queue_stages));
	binding_cache.clear();
	_scratch.reset();
	release_resources();
}

MDCommandBuffer::Alloc MDCommandBuffer::allocate_arg_buffer(uint32_t p_size) {
	return _scratch.allocate(p_size);
}

void MDCommandBuffer::end() {
	switch (type) {
		case MDCommandBufferStateType::None:
			return;
		case MDCommandBufferStateType::Render:
			return render_end_pass();
		case MDCommandBufferStateType::Compute:
			return _end_compute_dispatch();
		case MDCommandBufferStateType::Blit:
			return _end_blit();
	}
}

void MDCommandBuffer::commit() {
	end();
	if (use_barriers) {
		if (_scratch.is_changed()) {
			for (id<MTLBuffer> buf : _scratch.get_buffers()) {
				[_frame_state.rs addAllocation:buf];
			}
			_scratch.clear_changed();
			[_frame_state.rs commit];
		}
	}
	[commandBuffer commit];
	commandBuffer = nil;
	state_begin = false;
}

GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

static MTLStages convert_src_pipeline_stages_to_metal(BitField<RDD::PipelineStageBits> p_stages) {
	p_stages.clear_flag(RDD::PIPELINE_STAGE_TOP_OF_PIPE_BIT);

	if (p_stages.has_flag(RDD::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT | RDD::PIPELINE_STAGE_ALL_COMMANDS_BIT)) {
		return MTLStageAll;
	}

	MTLStages mtlStages = 0;

	// Vertex stage mappings
	if (p_stages & (RDD::PIPELINE_STAGE_DRAW_INDIRECT_BIT | RDD::PIPELINE_STAGE_VERTEX_INPUT_BIT | RDD::PIPELINE_STAGE_VERTEX_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | RDD::PIPELINE_STAGE_GEOMETRY_SHADER_BIT)) {
		mtlStages |= MTLStageVertex;
	}

	// Fragment stage mappings (includes resolve, which on Metal is handled in the render pipeline)
	if (p_stages & (RDD::PIPELINE_STAGE_FRAGMENT_SHADER_BIT | RDD::PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT | RDD::PIPELINE_STAGE_RESOLVE_BIT | RDD::PIPELINE_STAGE_CLEAR_STORAGE_BIT)) {
		mtlStages |= MTLStageFragment;
	}

	// Compute stage
	if (p_stages & RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
		mtlStages |= MTLStageDispatch;
	}

	// Blit stage (transfer operations)
	if (p_stages & (RDD::PIPELINE_STAGE_COPY_BIT)) {
		mtlStages |= MTLStageBlit;
	}

	// ALL_GRAPHICS_BIT special case
	if (p_stages & RDD::PIPELINE_STAGE_ALL_GRAPHICS_BIT) {
		mtlStages |= (MTLStageVertex | MTLStageFragment);
	}

	return mtlStages;
}

static MTLStages convert_dst_pipeline_stages_to_metal(BitField<RDD::PipelineStageBits> p_stages) {
	p_stages.clear_flag(RDD::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

	if (p_stages.has_flag(RDD::PIPELINE_STAGE_ALL_COMMANDS_BIT | RDD::PIPELINE_STAGE_TOP_OF_PIPE_BIT)) {
		return MTLStageAll;
	}

	MTLStages mtlStages = 0;

	if (p_stages & (RDD::PIPELINE_STAGE_DRAW_INDIRECT_BIT | RDD::PIPELINE_STAGE_VERTEX_INPUT_BIT | RDD::PIPELINE_STAGE_VERTEX_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | RDD::PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | RDD::PIPELINE_STAGE_GEOMETRY_SHADER_BIT)) {
		mtlStages |= MTLStageVertex;
	}

	if (p_stages & (RDD::PIPELINE_STAGE_FRAGMENT_SHADER_BIT | RDD::PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | RDD::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT | RDD::PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT | RDD::PIPELINE_STAGE_RESOLVE_BIT)) {
		mtlStages |= MTLStageFragment;
	}

	if (p_stages & RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
		mtlStages |= MTLStageDispatch;
	}

	if (p_stages & (RDD::PIPELINE_STAGE_COPY_BIT | RDD::PIPELINE_STAGE_CLEAR_STORAGE_BIT)) {
		mtlStages |= MTLStageBlit;
	}

	if (p_stages & RDD::PIPELINE_STAGE_ALL_GRAPHICS_BIT) {
		mtlStages |= (MTLStageVertex | MTLStageFragment);
	}

	return mtlStages;
}

void MDCommandBuffer::_encode_barrier(id<MTLCommandEncoder> p_enc) {
	DEV_ASSERT(p_enc);

	static const MTLStages empty_stages[STAGE_MAX] = { 0, 0, 0 };
	if (memcmp(&pending_before_queue_stages, empty_stages, sizeof(pending_before_queue_stages)) == 0) {
		return;
	}

	int stage = STAGE_MAX;
	if ([p_enc conformsToProtocol:@protocol(MTLRenderCommandEncoder)] && pending_after_stages[STAGE_RENDER] != 0) {
		stage = STAGE_RENDER;
	} else if ([p_enc conformsToProtocol:@protocol(MTLComputeCommandEncoder)] && pending_after_stages[STAGE_COMPUTE] != 0) {
		stage = STAGE_COMPUTE;
	} else if ([p_enc conformsToProtocol:@protocol(MTLBlitCommandEncoder)] && pending_after_stages[STAGE_BLIT] != 0) {
		stage = STAGE_BLIT;
	}

	if (stage == STAGE_MAX) {
		return;
	}

	[p_enc barrierAfterQueueStages:pending_after_stages[stage] beforeStages:pending_before_queue_stages[stage]];
	pending_before_queue_stages[stage] = 0;
	pending_after_stages[stage] = 0;
}

void MDCommandBuffer::pipeline_barrier(BitField<RDD::PipelineStageBits> p_src_stages,
		BitField<RDD::PipelineStageBits> p_dst_stages,
		VectorView<RDD::MemoryAccessBarrier> p_memory_barriers,
		VectorView<RDD::BufferBarrier> p_buffer_barriers,
		VectorView<RDD::TextureBarrier> p_texture_barriers) {
	MTLStages after_stages = convert_src_pipeline_stages_to_metal(p_src_stages);
	if (after_stages == 0) {
		return;
	}

	MTLStages before_stages = convert_dst_pipeline_stages_to_metal(p_dst_stages);
	if (before_stages == 0) {
		return;
	}

	// Encode intra-encoder memory barrier if an encoder is active for matching stages.
	if (render.encoder != nil) {
		MTLRenderStages render_after = static_cast<MTLRenderStages>(after_stages & (MTLStageVertex | MTLStageFragment));
		MTLRenderStages render_before = static_cast<MTLRenderStages>(before_stages & (MTLStageVertex | MTLStageFragment));
		if (render_after != 0 && render_before != 0) {
			[render.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures afterStages:render_after beforeStages:render_before];
		}
	} else if (compute.encoder != nil) {
		if (after_stages & MTLStageDispatch) {
			[compute.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
		}
	}
	// Blit encoder has no memory barrier API.

	// Also cache for inter-pass barriers.
	if (after_stages & (MTLStageVertex | MTLStageFragment)) {
		pending_after_stages[STAGE_RENDER] |= after_stages;
		pending_before_queue_stages[STAGE_RENDER] |= before_stages;
	}

	if (after_stages & MTLStageDispatch) {
		pending_after_stages[STAGE_COMPUTE] |= after_stages;
		pending_before_queue_stages[STAGE_COMPUTE] |= before_stages;
	}

	if (after_stages & MTLStageBlit) {
		pending_after_stages[STAGE_BLIT] |= after_stages;
		pending_before_queue_stages[STAGE_BLIT] |= before_stages;
	}
}

GODOT_CLANG_WARNING_POP

void MDCommandBuffer::bind_pipeline(RDD::PipelineID p_pipeline) {
	MDPipeline *p = (MDPipeline *)(p_pipeline.id);

	// End current encoder if it is a compute encoder or blit encoder,
	// as they do not have a defined end boundary in the RDD like render.
	if (type == MDCommandBufferStateType::Compute) {
		_end_compute_dispatch();
	} else if (type == MDCommandBufferStateType::Blit) {
		_end_blit();
	}

	if (p->type == MDPipelineType::Render) {
		DEV_ASSERT(type == MDCommandBufferStateType::Render);
		MDRenderPipeline *rp = (MDRenderPipeline *)p;

		if (render.encoder == nil) {
			// This error would happen if the render pass failed.
			ERR_FAIL_NULL_MSG(render.desc, "Render pass descriptor is null.");

			// This condition occurs when there are no attachments when calling render_next_subpass()
			// and is due to the SUPPORTS_FRAGMENT_SHADER_WITH_ONLY_SIDE_EFFECTS flag.
			render.desc.defaultRasterSampleCount = static_cast<NSUInteger>(rp->sample_count);

// NOTE(sgc): This is to test rdar://FB13605547 and will be deleted once fix is confirmed.
#if 0
			if (render.pipeline->sample_count == 4) {
				static id<MTLTexture> tex = nil;
				static id<MTLTexture> res_tex = nil;
				static dispatch_once_t onceToken;
				dispatch_once(&onceToken, ^{
					Size2i sz = render.frameBuffer->size;
					MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:sz.width height:sz.height mipmapped:NO];
					td.textureType = MTLTextureType2DMultisample;
					td.storageMode = MTLStorageModeMemoryless;
					td.usage = MTLTextureUsageRenderTarget;
					td.sampleCount = render.pipeline->sample_count;
					tex = [device_driver->get_device() newTextureWithDescriptor:td];

					td.textureType = MTLTextureType2D;
					td.storageMode = MTLStorageModePrivate;
					td.usage = MTLTextureUsageShaderWrite;
					td.sampleCount = 1;
					res_tex = [device_driver->get_device() newTextureWithDescriptor:td];
				});
				render.desc.colorAttachments[0].texture = tex;
				render.desc.colorAttachments[0].loadAction = MTLLoadActionClear;
				render.desc.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;

				render.desc.colorAttachments[0].resolveTexture = res_tex;
			}
#endif
			render.encoder = [command_buffer() renderCommandEncoderWithDescriptor:render.desc];
			_encode_barrier(render.encoder);
		}

		if (render.pipeline != rp) {
			render.dirty.set_flag((RenderState::DirtyFlag)(RenderState::DIRTY_PIPELINE | RenderState::DIRTY_RASTER));
			// Mark all uniforms as dirty, as variants of a shader pipeline may have a different entry point ABI,
			// due to setting force_active_argument_buffer_resources = true for spirv_cross::CompilerMSL::Options.
			// As a result, uniform sets with the same layout will generate redundant binding warnings when
			// capturing a Metal frame in Xcode.
			//
			// If we don't mark as dirty, then some bindings will generate a validation error.
			// binding_cache.clear();
			render.mark_uniforms_dirty();

			if (render.pipeline != nullptr && render.pipeline->depth_stencil != rp->depth_stencil) {
				render.dirty.set_flag(RenderState::DIRTY_DEPTH);
			}
			if (rp->raster_state.blend.enabled) {
				render.dirty.set_flag(RenderState::DIRTY_BLEND);
			}
			render.pipeline = rp;
		}
	} else if (p->type == MDPipelineType::Compute) {
		DEV_ASSERT(type == MDCommandBufferStateType::None);
		type = MDCommandBufferStateType::Compute;

		if (compute.pipeline != p) {
			compute.dirty.set_flag(ComputeState::DIRTY_PIPELINE);
			binding_cache.clear();
			compute.mark_uniforms_dirty();
			compute.pipeline = (MDComputePipeline *)p;
		}
	}
}

void MDCommandBuffer::encode_push_constant_data(RDD::ShaderID p_shader, VectorView<uint32_t> p_data) {
	switch (type) {
		case MDCommandBufferStateType::Render:
		case MDCommandBufferStateType::Compute: {
			MDShader *shader = (MDShader *)(p_shader.id);
			if (shader->push_constants.binding == UINT32_MAX) {
				return;
			}
			push_constant_binding = shader->push_constants.binding;
			void const *ptr = p_data.ptr();
			push_constant_data_len = p_data.size() * sizeof(uint32_t);
			DEV_ASSERT(push_constant_data_len <= sizeof(push_constant_data));
			memcpy(push_constant_data, ptr, push_constant_data_len);
			if (push_constant_data_len > 0) {
				switch (type) {
					case MDCommandBufferStateType::Render:
						render.dirty.set_flag(RenderState::DirtyFlag::DIRTY_PUSH);
						break;
					case MDCommandBufferStateType::Compute:
						compute.dirty.set_flag(ComputeState::DirtyFlag::DIRTY_PUSH);
						break;
					default:
						break;
				}
			}
		} break;
		case MDCommandBufferStateType::Blit:
		case MDCommandBufferStateType::None:
			return;
	}
}

id<MTLBlitCommandEncoder> MDCommandBuffer::_ensure_blit_encoder() {
	switch (type) {
		case MDCommandBufferStateType::None:
			break;
		case MDCommandBufferStateType::Render:
			render_end_pass();
			break;
		case MDCommandBufferStateType::Compute:
			_end_compute_dispatch();
			break;
		case MDCommandBufferStateType::Blit:
			return blit.encoder;
	}

	type = MDCommandBufferStateType::Blit;
	blit.encoder = command_buffer().blitCommandEncoder;
	_encode_barrier(blit.encoder);

	return blit.encoder;
}

_FORCE_INLINE_ static MTLSize mipmapLevelSizeFromTexture(id<MTLTexture> p_tex, NSUInteger p_level) {
	MTLSize lvlSize;
	lvlSize.width = MAX(p_tex.width >> p_level, 1UL);
	lvlSize.height = MAX(p_tex.height >> p_level, 1UL);
	lvlSize.depth = MAX(p_tex.depth >> p_level, 1UL);
	return lvlSize;
}

void MDCommandBuffer::resolve_texture(RDD::TextureID p_src_texture, RDD::TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, RDD::TextureID p_dst_texture, RDD::TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) {
	id<MTLTexture> src_tex = rid::get(p_src_texture);
	id<MTLTexture> dst_tex = rid::get(p_dst_texture);

	MTLRenderPassDescriptor *mtlRPD = [MTLRenderPassDescriptor renderPassDescriptor];
	MTLRenderPassColorAttachmentDescriptor *mtlColorAttDesc = mtlRPD.colorAttachments[0];
	mtlColorAttDesc.loadAction = MTLLoadActionLoad;
	mtlColorAttDesc.storeAction = MTLStoreActionMultisampleResolve;

	mtlColorAttDesc.texture = src_tex;
	mtlColorAttDesc.resolveTexture = dst_tex;
	mtlColorAttDesc.level = p_src_mipmap;
	mtlColorAttDesc.slice = p_src_layer;
	mtlColorAttDesc.resolveLevel = p_dst_mipmap;
	mtlColorAttDesc.resolveSlice = p_dst_layer;
	id<MTLRenderCommandEncoder> enc = get_new_render_encoder_with_descriptor(mtlRPD);
	enc.label = @"Resolve Image";
	[enc endEncoding];
}

void MDCommandBuffer::clear_color_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, const Color &p_color, const RDD::TextureSubresourceRange &p_subresources) {
	id<MTLTexture> src_tex = rid::get(p_texture);

	if (src_tex.parentTexture) {
		// Clear via the parent texture rather than the view.
		src_tex = src_tex.parentTexture;
	}

	PixelFormats &pf = device_driver->get_pixel_formats();

	if (pf.isDepthFormat(src_tex.pixelFormat) || pf.isStencilFormat(src_tex.pixelFormat)) {
		ERR_FAIL_MSG("invalid: depth or stencil texture format");
	}

	MTLRenderPassDescriptor *desc = MTLRenderPassDescriptor.renderPassDescriptor;

	if (p_subresources.aspect.has_flag(RDD::TEXTURE_ASPECT_COLOR_BIT)) {
		MTLRenderPassColorAttachmentDescriptor *caDesc = desc.colorAttachments[0];
		caDesc.texture = src_tex;
		caDesc.loadAction = MTLLoadActionClear;
		caDesc.storeAction = MTLStoreActionStore;
		caDesc.clearColor = MTLClearColorMake(p_color.r, p_color.g, p_color.b, p_color.a);

		// Extract the mipmap levels that are to be updated.
		uint32_t mipLvlStart = p_subresources.base_mipmap;
		uint32_t mipLvlCnt = p_subresources.mipmap_count;
		uint32_t mipLvlEnd = mipLvlStart + mipLvlCnt;

		uint32_t levelCount = src_tex.mipmapLevelCount;

		// Extract the cube or array layers (slices) that are to be updated.
		bool is3D = src_tex.textureType == MTLTextureType3D;
		uint32_t layerStart = is3D ? 0 : p_subresources.base_layer;
		uint32_t layerCnt = p_subresources.layer_count;
		uint32_t layerEnd = layerStart + layerCnt;

		MetalFeatures const &features = device_driver->get_device_properties().features;

		// Iterate across mipmap levels and layers, and perform and empty render to clear each.
		for (uint32_t mipLvl = mipLvlStart; mipLvl < mipLvlEnd; mipLvl++) {
			ERR_FAIL_INDEX_MSG(mipLvl, levelCount, "mip level out of range");

			caDesc.level = mipLvl;

			// If a 3D image, we need to get the depth for each level.
			if (is3D) {
				layerCnt = mipmapLevelSizeFromTexture(src_tex, mipLvl).depth;
				layerEnd = layerStart + layerCnt;
			}

			if ((features.layeredRendering && src_tex.sampleCount == 1) || features.multisampleLayeredRendering) {
				// We can clear all layers at once.
				if (is3D) {
					caDesc.depthPlane = layerStart;
				} else {
					caDesc.slice = layerStart;
				}
				desc.renderTargetArrayLength = layerCnt;
				id<MTLRenderCommandEncoder> enc = get_new_render_encoder_with_descriptor(desc);
				enc.label = @"Clear Image";
				[enc endEncoding];
			} else {
				for (uint32_t layer = layerStart; layer < layerEnd; layer++) {
					if (is3D) {
						caDesc.depthPlane = layer;
					} else {
						caDesc.slice = layer;
					}
					id<MTLRenderCommandEncoder> enc = get_new_render_encoder_with_descriptor(desc);
					enc.label = @"Clear Image";
					[enc endEncoding];
				}
			}
		}
	}
}

void MDCommandBuffer::clear_buffer(RDD::BufferID p_buffer, uint64_t p_offset, uint64_t p_size) {
	id<MTLBlitCommandEncoder> blit_enc = _ensure_blit_encoder();
	const RDM::BufferInfo *buffer = (const RDM::BufferInfo *)p_buffer.id;

	[blit_enc fillBuffer:buffer->metal_buffer
				   range:NSMakeRange(p_offset, p_size)
				   value:0];
}

void MDCommandBuffer::copy_buffer(RDD::BufferID p_src_buffer, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferCopyRegion> p_regions) {
	const RDM::BufferInfo *src = (const RDM::BufferInfo *)p_src_buffer.id;
	const RDM::BufferInfo *dst = (const RDM::BufferInfo *)p_dst_buffer.id;

	id<MTLBlitCommandEncoder> enc = _ensure_blit_encoder();

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		RDD::BufferCopyRegion region = p_regions[i];
		[enc copyFromBuffer:src->metal_buffer
					 sourceOffset:region.src_offset
						 toBuffer:dst->metal_buffer
				destinationOffset:region.dst_offset
							 size:region.size];
	}
}

static MTLSize MTLSizeFromVector3i(Vector3i p_size) {
	return MTLSizeMake(p_size.x, p_size.y, p_size.z);
}

static MTLOrigin MTLOriginFromVector3i(Vector3i p_origin) {
	return MTLOriginMake(p_origin.x, p_origin.y, p_origin.z);
}

// Clamps the size so that the sum of the origin and size do not exceed the maximum size.
static inline MTLSize clampMTLSize(MTLSize p_size, MTLOrigin p_origin, MTLSize p_max_size) {
	MTLSize clamped;
	clamped.width = MIN(p_size.width, p_max_size.width - p_origin.x);
	clamped.height = MIN(p_size.height, p_max_size.height - p_origin.y);
	clamped.depth = MIN(p_size.depth, p_max_size.depth - p_origin.z);
	return clamped;
}

API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0))
static bool isArrayTexture(MTLTextureType p_type) {
	return (p_type == MTLTextureType3D ||
			p_type == MTLTextureType2DArray ||
			p_type == MTLTextureType2DMultisampleArray ||
			p_type == MTLTextureType1DArray);
}

_FORCE_INLINE_ static bool operator==(MTLSize p_a, MTLSize p_b) {
	return p_a.width == p_b.width && p_a.height == p_b.height && p_a.depth == p_b.depth;
}

void MDCommandBuffer::copy_texture(RDD::TextureID p_src_texture, RDD::TextureID p_dst_texture, VectorView<RDD::TextureCopyRegion> p_regions) {
	id<MTLTexture> src = rid::get(p_src_texture);
	id<MTLTexture> dst = rid::get(p_dst_texture);

	id<MTLBlitCommandEncoder> enc = _ensure_blit_encoder();
	PixelFormats &pf = device_driver->get_pixel_formats();

	MTLPixelFormat src_fmt = src.pixelFormat;
	bool src_is_compressed = pf.getFormatType(src_fmt) == MTLFormatType::Compressed;
	MTLPixelFormat dst_fmt = dst.pixelFormat;
	bool dst_is_compressed = pf.getFormatType(dst_fmt) == MTLFormatType::Compressed;

	// Validate copy.
	if (src.sampleCount != dst.sampleCount || pf.getBytesPerBlock(src_fmt) != pf.getBytesPerBlock(dst_fmt)) {
		ERR_FAIL_MSG("Cannot copy between incompatible pixel formats, such as formats of different pixel sizes, or between images with different sample counts.");
	}

	// If source and destination have different formats and at least one is compressed, a temporary buffer is required.
	bool need_tmp_buffer = (src_fmt != dst_fmt) && (src_is_compressed || dst_is_compressed);
	if (need_tmp_buffer) {
		ERR_FAIL_MSG("not implemented: copy with intermediate buffer");
	}

	if (src_fmt != dst_fmt) {
		// Map the source pixel format to the dst through a texture view on the source texture.
		src = [src newTextureViewWithPixelFormat:dst_fmt];
	}

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		RDD::TextureCopyRegion region = p_regions[i];

		MTLSize extent = MTLSizeFromVector3i(region.size);

		// If copies can be performed using direct texture-texture copying, do so.
		uint32_t src_level = region.src_subresources.mipmap;
		uint32_t src_base_layer = region.src_subresources.base_layer;
		MTLSize src_extent = mipmapLevelSizeFromTexture(src, src_level);
		uint32_t dst_level = region.dst_subresources.mipmap;
		uint32_t dst_base_layer = region.dst_subresources.base_layer;
		MTLSize dst_extent = mipmapLevelSizeFromTexture(dst, dst_level);

		// All layers may be copied at once, if the extent completely covers both images.
		if (src_extent == extent && dst_extent == extent) {
			[enc copyFromTexture:src
						 sourceSlice:src_base_layer
						 sourceLevel:src_level
						   toTexture:dst
					destinationSlice:dst_base_layer
					destinationLevel:dst_level
						  sliceCount:region.src_subresources.layer_count
						  levelCount:1];
		} else {
			MTLOrigin src_origin = MTLOriginFromVector3i(region.src_offset);
			MTLSize src_size = clampMTLSize(extent, src_origin, src_extent);
			uint32_t layer_count = 0;
			if ((src.textureType == MTLTextureType3D) != (dst.textureType == MTLTextureType3D)) {
				// In the case, the number of layers to copy is in extent.depth. Use that value,
				// then clamp the depth, so we don't try to copy more than Metal will allow.
				layer_count = extent.depth;
				src_size.depth = 1;
			} else {
				layer_count = region.src_subresources.layer_count;
			}
			MTLOrigin dst_origin = MTLOriginFromVector3i(region.dst_offset);

			for (uint32_t layer = 0; layer < layer_count; layer++) {
				// We can copy between a 3D and a 2D image easily. Just copy between
				// one slice of the 2D image and one plane of the 3D image at a time.
				if ((src.textureType == MTLTextureType3D) == (dst.textureType == MTLTextureType3D)) {
					[enc copyFromTexture:src
								  sourceSlice:src_base_layer + layer
								  sourceLevel:src_level
								 sourceOrigin:src_origin
								   sourceSize:src_size
									toTexture:dst
							 destinationSlice:dst_base_layer + layer
							 destinationLevel:dst_level
							destinationOrigin:dst_origin];
				} else if (src.textureType == MTLTextureType3D) {
					[enc copyFromTexture:src
								  sourceSlice:src_base_layer
								  sourceLevel:src_level
								 sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z + layer)
								   sourceSize:src_size
									toTexture:dst
							 destinationSlice:dst_base_layer + layer
							 destinationLevel:dst_level
							destinationOrigin:dst_origin];
				} else {
					DEV_ASSERT(dst.textureType == MTLTextureType3D);
					[enc copyFromTexture:src
								  sourceSlice:src_base_layer + layer
								  sourceLevel:src_level
								 sourceOrigin:src_origin
								   sourceSize:src_size
									toTexture:dst
							 destinationSlice:dst_base_layer
							 destinationLevel:dst_level
							destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z + layer)];
				}
			}
		}
	}
}

void MDCommandBuffer::copy_buffer_to_texture(RDD::BufferID p_src_buffer, RDD::TextureID p_dst_texture, VectorView<RDD::BufferTextureCopyRegion> p_regions) {
	_copy_texture_buffer(CopySource::Buffer, p_dst_texture, p_src_buffer, p_regions);
}

void MDCommandBuffer::copy_texture_to_buffer(RDD::TextureID p_src_texture, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferTextureCopyRegion> p_regions) {
	_copy_texture_buffer(CopySource::Texture, p_src_texture, p_dst_buffer, p_regions);
}

void MDCommandBuffer::_copy_texture_buffer(CopySource p_source,
		RDD::TextureID p_texture,
		RDD::BufferID p_buffer,
		VectorView<RDD::BufferTextureCopyRegion> p_regions) {
	const RDM::BufferInfo *buffer = (const RDM::BufferInfo *)p_buffer.id;
	id<MTLTexture> texture = rid::get(p_texture);

	id<MTLBlitCommandEncoder> enc = _ensure_blit_encoder();

	PixelFormats &pf = device_driver->get_pixel_formats();
	MTLPixelFormat mtlPixFmt = texture.pixelFormat;

	MTLBlitOption options = MTLBlitOptionNone;
	if (pf.isPVRTCFormat(mtlPixFmt)) {
		options |= MTLBlitOptionRowLinearPVRTC;
	}

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		RDD::BufferTextureCopyRegion region = p_regions[i];

		uint32_t mip_level = region.texture_subresource.mipmap;
		MTLOrigin txt_origin = MTLOriginMake(region.texture_offset.x, region.texture_offset.y, region.texture_offset.z);
		MTLSize src_extent = mipmapLevelSizeFromTexture(texture, mip_level);
		MTLSize txt_size = clampMTLSize(MTLSizeMake(region.texture_region_size.x, region.texture_region_size.y, region.texture_region_size.z),
				txt_origin,
				src_extent);

		uint32_t buffImgWd = region.texture_region_size.x;
		uint32_t buffImgHt = region.texture_region_size.y;

		NSUInteger bytesPerRow = pf.getBytesPerRow(mtlPixFmt, buffImgWd);
		NSUInteger bytesPerImg = pf.getBytesPerLayer(mtlPixFmt, bytesPerRow, buffImgHt);

		MTLBlitOption blit_options = options;

		if (pf.isDepthFormat(mtlPixFmt) && pf.isStencilFormat(mtlPixFmt)) {
			// Don't reduce depths of 32-bit depth/stencil formats.
			if (region.texture_subresource.aspect == RDD::TEXTURE_ASPECT_DEPTH) {
				if (pf.getBytesPerTexel(mtlPixFmt) != 4) {
					bytesPerRow -= buffImgWd;
					bytesPerImg -= buffImgWd * buffImgHt;
				}
				blit_options |= MTLBlitOptionDepthFromDepthStencil;
			} else if (region.texture_subresource.aspect == RDD::TEXTURE_ASPECT_STENCIL) {
				// The stencil component is always 1 byte per pixel.
				bytesPerRow = buffImgWd;
				bytesPerImg = buffImgWd * buffImgHt;
				blit_options |= MTLBlitOptionStencilFromDepthStencil;
			}
		}

		if (!isArrayTexture(texture.textureType)) {
			bytesPerImg = 0;
		}

		if (p_source == CopySource::Buffer) {
			[enc copyFromBuffer:buffer->metal_buffer
						   sourceOffset:region.buffer_offset
					  sourceBytesPerRow:bytesPerRow
					sourceBytesPerImage:bytesPerImg
							 sourceSize:txt_size
							  toTexture:texture
					   destinationSlice:region.texture_subresource.layer
					   destinationLevel:mip_level
					  destinationOrigin:txt_origin
								options:blit_options];
		} else {
			[enc copyFromTexture:texture
								 sourceSlice:region.texture_subresource.layer
								 sourceLevel:mip_level
								sourceOrigin:txt_origin
								  sourceSize:txt_size
									toBuffer:buffer->metal_buffer
						   destinationOffset:region.buffer_offset
					  destinationBytesPerRow:bytesPerRow
					destinationBytesPerImage:bytesPerImg
									 options:blit_options];
		}
	}
}

id<MTLRenderCommandEncoder> MDCommandBuffer::get_new_render_encoder_with_descriptor(MTLRenderPassDescriptor *p_desc) {
	switch (type) {
		case MDCommandBufferStateType::None:
			break;
		case MDCommandBufferStateType::Render:
			render_end_pass();
			break;
		case MDCommandBufferStateType::Compute:
			_end_compute_dispatch();
			break;
		case MDCommandBufferStateType::Blit:
			_end_blit();
			break;
	}

	id<MTLRenderCommandEncoder> enc = [command_buffer() renderCommandEncoderWithDescriptor:p_desc];
	_encode_barrier(enc);
	return enc;
}

#pragma mark - Render Commands

void MDCommandBuffer::render_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);

	render.dynamic_offsets |= p_dynamic_offsets;

	if (uint32_t new_size = p_first_set_index + p_set_count; render.uniform_sets.size() < new_size) {
		uint32_t s = render.uniform_sets.size();
		render.uniform_sets.resize(new_size);
		// Set intermediate values to null.
		std::fill(&render.uniform_sets[s], render.uniform_sets.end().operator->(), nullptr);
	}

	const MDShader *shader = (const MDShader *)p_shader.id;
	DynamicOffsetLayout layout = shader->dynamic_offset_layout;

	for (size_t i = 0; i < p_set_count; ++i) {
		MDUniformSet *set = (MDUniformSet *)(p_uniform_sets[i].id);

		uint32_t index = p_first_set_index + i;
		if (render.uniform_sets[index] != set || layout.get_count(index) > 0) {
			render.dirty.set_flag(RenderState::DIRTY_UNIFORMS);
			render.uniform_set_mask |= 1ULL << index;
			render.uniform_sets[index] = set;
		}
	}
}

void MDCommandBuffer::render_clear_attachments(VectorView<RDD::AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);

	const MDSubpass &subpass = render.get_subpass();

	uint32_t vertex_count = p_rects.size() * 6 * subpass.view_count;
	simd::float4 *vertices = ALLOCA_ARRAY(simd::float4, vertex_count);
	simd::float4 clear_colors[ClearAttKey::ATTACHMENT_COUNT];

	Size2i size = render.frameBuffer->size;
	Rect2i render_area = render.clip_to_render_area({ { 0, 0 }, size });
	size = Size2i(render_area.position.x + render_area.size.width, render_area.position.y + render_area.size.height);
	_populate_vertices(vertices, size, p_rects);

	ClearAttKey key;
	key.sample_count = render.pass->get_sample_count();
	if (subpass.view_count > 1) {
		key.enable_layered_rendering();
	}

	float depth_value = 0;
	uint32_t stencil_value = 0;

	for (uint32_t i = 0; i < p_attachment_clears.size(); i++) {
		RDD::AttachmentClear const &attClear = p_attachment_clears[i];
		uint32_t attachment_index;
		if (attClear.aspect.has_flag(RDD::TEXTURE_ASPECT_COLOR_BIT)) {
			attachment_index = attClear.color_attachment;
		} else {
			attachment_index = subpass.depth_stencil_reference.attachment;
		}

		MDAttachment const &mda = render.pass->attachments[attachment_index];
		if (attClear.aspect.has_flag(RDD::TEXTURE_ASPECT_COLOR_BIT)) {
			key.set_color_format(attachment_index, mda.format);
			clear_colors[attachment_index] = {
				attClear.value.color.r,
				attClear.value.color.g,
				attClear.value.color.b,
				attClear.value.color.a
			};
		}

		if (attClear.aspect.has_flag(RDD::TEXTURE_ASPECT_DEPTH_BIT)) {
			key.set_depth_format(mda.format);
			depth_value = attClear.value.depth;
		}

		if (attClear.aspect.has_flag(RDD::TEXTURE_ASPECT_STENCIL_BIT)) {
			key.set_stencil_format(mda.format);
			stencil_value = attClear.value.stencil;
		}
	}
	clear_colors[ClearAttKey::DEPTH_INDEX] = {
		depth_value,
		depth_value,
		depth_value,
		depth_value
	};

	id<MTLRenderCommandEncoder> enc = render.encoder;

	MDResourceCache &cache = device_driver->get_resource_cache();

	[enc pushDebugGroup:@"ClearAttachments"];
	[enc setRenderPipelineState:cache.get_clear_render_pipeline_state(key, nil)];
	[enc setDepthStencilState:cache.get_depth_stencil_state(
									  key.is_depth_enabled(),
									  key.is_stencil_enabled())];
	[enc setStencilReferenceValue:stencil_value];
	[enc setCullMode:MTLCullModeNone];
	[enc setTriangleFillMode:MTLTriangleFillModeFill];
	[enc setDepthBias:0 slopeScale:0 clamp:0];
	[enc setViewport:{ 0, 0, (double)size.width, (double)size.height, 0.0, 1.0 }];
	[enc setScissorRect:{ 0, 0, (NSUInteger)size.width, (NSUInteger)size.height }];

	[enc setVertexBytes:clear_colors length:sizeof(clear_colors) atIndex:0];
	[enc setFragmentBytes:clear_colors length:sizeof(clear_colors) atIndex:0];
	[enc setVertexBytes:vertices length:vertex_count * sizeof(vertices[0]) atIndex:device_driver->get_metal_buffer_index_for_vertex_attribute_binding(VERT_CONTENT_BUFFER_INDEX)];

	[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:vertex_count];
	[enc popDebugGroup];

	render.dirty.set_flag((RenderState::DirtyFlag)(RenderState::DIRTY_PIPELINE | RenderState::DIRTY_DEPTH | RenderState::DIRTY_RASTER));
	binding_cache.clear();
	render.mark_uniforms_dirty({ 0 }); // Mark index 0 dirty, if there is already a binding for index 0.
	render.mark_viewport_dirty();
	render.mark_scissors_dirty();
	render.mark_vertex_dirty();
	render.mark_blend_dirty();
}

void MDCommandBuffer::_render_set_dirty_state() {
	_render_bind_uniform_sets();

	if (render.dirty.has_flag(RenderState::DIRTY_PUSH)) {
		if (push_constant_binding != UINT32_MAX) {
			[render.encoder setVertexBytes:push_constant_data
									length:push_constant_data_len
								   atIndex:push_constant_binding];
			[render.encoder setFragmentBytes:push_constant_data
									  length:push_constant_data_len
									 atIndex:push_constant_binding];
		}
	}

	MDSubpass const &subpass = render.get_subpass();
	if (subpass.view_count > 1) {
		uint32_t view_range[2] = { 0, subpass.view_count };
		[render.encoder setVertexBytes:view_range length:sizeof(view_range) atIndex:VIEW_MASK_BUFFER_INDEX];
		[render.encoder setFragmentBytes:view_range length:sizeof(view_range) atIndex:VIEW_MASK_BUFFER_INDEX];
	}

	if (render.dirty.has_flag(RenderState::DIRTY_PIPELINE)) {
		[render.encoder setRenderPipelineState:render.pipeline->state];
	}

	if (render.dirty.has_flag(RenderState::DIRTY_VIEWPORT)) {
		[render.encoder setViewports:render.viewports.ptr() count:render.viewports.size()];
	}

	if (render.dirty.has_flag(RenderState::DIRTY_DEPTH)) {
		[render.encoder setDepthStencilState:render.pipeline->depth_stencil];
	}

	if (render.dirty.has_flag(RenderState::DIRTY_RASTER)) {
		render.pipeline->raster_state.apply(render.encoder);
	}

	if (render.dirty.has_flag(RenderState::DIRTY_SCISSOR) && !render.scissors.is_empty()) {
		size_t len = render.scissors.size();
		MTLScissorRect *rects = ALLOCA_ARRAY(MTLScissorRect, len);
		for (size_t i = 0; i < len; i++) {
			rects[i] = render.clip_to_render_area(render.scissors[i]);
		}
		[render.encoder setScissorRects:rects count:len];
	}

	if (render.dirty.has_flag(RenderState::DIRTY_BLEND) && render.blend_constants.has_value()) {
		[render.encoder setBlendColorRed:render.blend_constants->r green:render.blend_constants->g blue:render.blend_constants->b alpha:render.blend_constants->a];
	}

	if (render.dirty.has_flag(RenderState::DIRTY_VERTEX)) {
		uint32_t p_binding_count = render.vertex_buffers.size();
		if (p_binding_count > 0) {
			uint32_t first = device_driver->get_metal_buffer_index_for_vertex_attribute_binding(p_binding_count - 1);
			[render.encoder setVertexBuffers:render.vertex_buffers.ptr()
									 offsets:render.vertex_offsets.ptr()
								   withRange:NSMakeRange(first, p_binding_count)];
		}
	}

	if (!use_barriers) {
		render.resource_tracker.encode(render.encoder);
	}

	render.dirty.clear();
}

void MDCommandBuffer::render_set_viewport(VectorView<Rect2i> p_viewports) {
	render.viewports.resize(p_viewports.size());
	for (uint32_t i = 0; i < p_viewports.size(); i += 1) {
		Rect2i const &vp = p_viewports[i];
		render.viewports[i] = {
			.originX = static_cast<double>(vp.position.x),
			.originY = static_cast<double>(vp.position.y),
			.width = static_cast<double>(vp.size.width),
			.height = static_cast<double>(vp.size.height),
			.znear = 0.0,
			.zfar = 1.0,
		};
	}

	render.dirty.set_flag(RenderState::DIRTY_VIEWPORT);
}

void MDCommandBuffer::render_set_scissor(VectorView<Rect2i> p_scissors) {
	render.scissors.resize(p_scissors.size());
	for (uint32_t i = 0; i < p_scissors.size(); i += 1) {
		Rect2i const &vp = p_scissors[i];
		render.scissors[i] = {
			.x = static_cast<NSUInteger>(vp.position.x),
			.y = static_cast<NSUInteger>(vp.position.y),
			.width = static_cast<NSUInteger>(vp.size.width),
			.height = static_cast<NSUInteger>(vp.size.height),
		};
	}

	render.dirty.set_flag(RenderState::DIRTY_SCISSOR);
}

void MDCommandBuffer::render_set_blend_constants(const Color &p_constants) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	if (render.blend_constants != p_constants) {
		render.blend_constants = p_constants;
		render.dirty.set_flag(RenderState::DIRTY_BLEND);
	}
}

void ResourceTracker::merge_from(const ResourceUsageMap &p_from) {
	for (KeyValue<StageResourceUsage, ResourceVector> const &keyval : p_from) {
		ResourceVector *resources = _current.getptr(keyval.key);
		if (resources == nullptr) {
			resources = &_current.insert(keyval.key, ResourceVector())->value;
		}
		// Reserve space for the new resources, assuming they are all added.
		resources->reserve(resources->size() + keyval.value.size());

		uint32_t i = 0, j = 0;
		MTLResourceUnsafe *resources_ptr = resources->ptr();
		const MTLResourceUnsafe *keyval_ptr = keyval.value.ptr();
		// 2-way merge.
		while (i < resources->size() && j < keyval.value.size()) {
			if (resources_ptr[i] < keyval_ptr[j]) {
				i++;
			} else if (resources_ptr[i] > keyval_ptr[j]) {
				ResourceUsageEntry *existing = nullptr;
				if ((existing = _previous.getptr(keyval_ptr[j])) == nullptr) {
					existing = &_previous.insert(keyval_ptr[j], keyval.key)->value;
					resources->insert(i, keyval_ptr[j]);
				} else {
					if (existing->usage != keyval.key) {
						existing->usage |= keyval.key;
						resources->insert(i, keyval_ptr[j]);
					}
				}
				i++;
				j++;
			} else {
				i++;
				j++;
			}
		}
		// Append the remaining resources.
		for (; j < keyval.value.size(); j++) {
			ResourceUsageEntry *existing = nullptr;
			if ((existing = _previous.getptr(keyval_ptr[j])) == nullptr) {
				existing = &_previous.insert(keyval_ptr[j], keyval.key)->value;
				resources->push_back(keyval_ptr[j]);
			} else {
				if (existing->usage != keyval.key) {
					existing->usage |= keyval.key;
					resources->push_back(keyval_ptr[j]);
				}
			}
		}
	}
}

void ResourceTracker::encode(id<MTLRenderCommandEncoder> __unsafe_unretained p_enc) {
	for (KeyValue<StageResourceUsage, ResourceVector> const &keyval : _current) {
		if (keyval.value.is_empty()) {
			continue;
		}

		MTLResourceUsage vert_usage = resource_usage_for_stage(keyval.key, RDD::ShaderStage::SHADER_STAGE_VERTEX);
		MTLResourceUsage frag_usage = resource_usage_for_stage(keyval.key, RDD::ShaderStage::SHADER_STAGE_FRAGMENT);
		if (vert_usage == frag_usage) {
			[p_enc useResources:keyval.value.ptr() count:keyval.value.size() usage:vert_usage stages:MTLRenderStageVertex | MTLRenderStageFragment];
		} else {
			if (vert_usage != 0) {
				[p_enc useResources:keyval.value.ptr() count:keyval.value.size() usage:vert_usage stages:MTLRenderStageVertex];
			}
			if (frag_usage != 0) {
				[p_enc useResources:keyval.value.ptr() count:keyval.value.size() usage:frag_usage stages:MTLRenderStageFragment];
			}
		}
	}

	// Keep the keys for now and clear the vectors to reduce churn.
	for (KeyValue<StageResourceUsage, ResourceVector> &v : _current) {
		v.value.clear();
	}
}

void ResourceTracker::encode(id<MTLComputeCommandEncoder> __unsafe_unretained p_enc) {
	for (KeyValue<StageResourceUsage, ResourceVector> const &keyval : _current) {
		if (keyval.value.is_empty()) {
			continue;
		}
		MTLResourceUsage usage = resource_usage_for_stage(keyval.key, RDD::ShaderStage::SHADER_STAGE_COMPUTE);
		if (usage != 0) {
			[p_enc useResources:keyval.value.ptr() count:keyval.value.size() usage:usage];
		}
	}

	// Keep the keys for now and clear the vectors to reduce churn.
	for (KeyValue<StageResourceUsage, ResourceVector> &v : _current) {
		v.value.clear();
	}
}

void ResourceTracker::reset() {
	// Keep the keys for now, as they are likely to be used repeatedly.
	for (KeyValue<MTLResourceUnsafe, ResourceUsageEntry> &v : _previous) {
		if (v.value.usage == ResourceUnused) {
			v.value.unused++;
			if (v.value.unused >= RESOURCE_UNUSED_CLEANUP_COUNT) {
				_scratch.push_back(v.key);
			}
		} else {
			v.value = ResourceUnused;
			v.value.unused = 0;
		}
	}

	// Clear up resources that weren't used for the last pass.
	for (const MTLResourceUnsafe &res : _scratch) {
		_previous.erase(res);
	}
	_scratch.clear();
}

void MDCommandBuffer::_render_bind_uniform_sets() {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	if (!render.dirty.has_flag(RenderState::DIRTY_UNIFORMS)) {
		return;
	}

	render.dirty.clear_flag(RenderState::DIRTY_UNIFORMS);
	uint64_t set_uniforms = render.uniform_set_mask;
	render.uniform_set_mask = 0;

	MDRenderShader *shader = render.pipeline->shader;
	const uint32_t dynamic_offsets = render.dynamic_offsets;

	while (set_uniforms != 0) {
		// Find the index of the next set bit.
		uint32_t index = (uint32_t)__builtin_ctzll(set_uniforms);
		// Clear the set bit.
		set_uniforms &= (set_uniforms - 1);
		MDUniformSet *set = render.uniform_sets[index];
		if (set == nullptr || index >= (uint32_t)shader->sets.size()) {
			continue;
		}
		if (shader->uses_argument_buffers) {
			_bind_uniforms_argument_buffers(set, shader, index, dynamic_offsets);
		} else {
			DirectEncoder de(render.encoder, binding_cache);
			_bind_uniforms_direct(set, shader, de, index, dynamic_offsets);
		}
	}
}

void MDCommandBuffer::_populate_vertices(simd::float4 *p_vertices, Size2i p_fb_size, VectorView<Rect2i> p_rects) {
	uint32_t idx = 0;
	for (uint32_t i = 0; i < p_rects.size(); i++) {
		Rect2i const &rect = p_rects[i];
		idx = _populate_vertices(p_vertices, idx, rect, p_fb_size);
	}
}

uint32_t MDCommandBuffer::_populate_vertices(simd::float4 *p_vertices, uint32_t p_index, Rect2i const &p_rect, Size2i p_fb_size) {
	// Determine the positions of the four edges of the
	// clear rectangle as a fraction of the attachment size.
	float leftPos = (float)(p_rect.position.x) / (float)p_fb_size.width;
	float rightPos = (float)(p_rect.size.width) / (float)p_fb_size.width + leftPos;
	float bottomPos = (float)(p_rect.position.y) / (float)p_fb_size.height;
	float topPos = (float)(p_rect.size.height) / (float)p_fb_size.height + bottomPos;

	// Transform to clip-space coordinates, which are bounded by (-1.0 < p < 1.0) in clip-space.
	leftPos = (leftPos * 2.0f) - 1.0f;
	rightPos = (rightPos * 2.0f) - 1.0f;
	bottomPos = (bottomPos * 2.0f) - 1.0f;
	topPos = (topPos * 2.0f) - 1.0f;

	simd::float4 vtx;

	uint32_t idx = p_index;
	uint32_t endLayer = render.get_subpass().view_count;

	for (uint32_t layer = 0; layer < endLayer; layer++) {
		vtx.z = 0.0;
		vtx.w = (float)layer;

		// Top left vertex - First triangle.
		vtx.y = topPos;
		vtx.x = leftPos;
		p_vertices[idx++] = vtx;

		// Bottom left vertex.
		vtx.y = bottomPos;
		vtx.x = leftPos;
		p_vertices[idx++] = vtx;

		// Bottom right vertex.
		vtx.y = bottomPos;
		vtx.x = rightPos;
		p_vertices[idx++] = vtx;

		// Bottom right vertex - Second triangle.
		p_vertices[idx++] = vtx;

		// Top right vertex.
		vtx.y = topPos;
		vtx.x = rightPos;
		p_vertices[idx++] = vtx;

		// Top left vertex.
		vtx.y = topPos;
		vtx.x = leftPos;
		p_vertices[idx++] = vtx;
	}

	return idx;
}

void MDCommandBuffer::render_begin_pass(RDD::RenderPassID p_render_pass, RDD::FramebufferID p_frameBuffer, RDD::CommandBufferType p_cmd_buffer_type, const Rect2i &p_rect, VectorView<RDD::RenderPassClearValue> p_clear_values) {
	DEV_ASSERT(command_buffer() != nil);
	end();

	MDRenderPass *pass = (MDRenderPass *)(p_render_pass.id);
	MDFrameBuffer *fb = (MDFrameBuffer *)(p_frameBuffer.id);

	type = MDCommandBufferStateType::Render;
	render.pass = pass;
	render.current_subpass = UINT32_MAX;
	render.render_area = p_rect;
	render.clear_values.resize(p_clear_values.size());
	for (uint32_t i = 0; i < p_clear_values.size(); i++) {
		render.clear_values[i] = p_clear_values[i];
	}
	render.is_rendering_entire_area = (p_rect.position == Point2i(0, 0)) && p_rect.size == fb->size;
	render.frameBuffer = fb;
	render_next_subpass();
}

void MDCommandBuffer::_end_render_pass() {
	MDFrameBuffer const &fb_info = *render.frameBuffer;
	MDSubpass const &subpass = render.get_subpass();

	PixelFormats &pf = device_driver->get_pixel_formats();

	for (uint32_t i = 0; i < subpass.resolve_references.size(); i++) {
		uint32_t color_index = subpass.color_references[i].attachment;
		uint32_t resolve_index = subpass.resolve_references[i].attachment;
		DEV_ASSERT((color_index == RDD::AttachmentReference::UNUSED) == (resolve_index == RDD::AttachmentReference::UNUSED));
		if (color_index == RDD::AttachmentReference::UNUSED || !fb_info.has_texture(color_index)) {
			continue;
		}

		id<MTLTexture> resolve_tex = fb_info.get_texture(resolve_index);

		CRASH_COND_MSG(!flags::all(pf.getCapabilities(resolve_tex.pixelFormat), kMTLFmtCapsResolve), "not implemented: unresolvable texture types");
		// see: https://github.com/KhronosGroup/MoltenVK/blob/d20d13fe2735adb845636a81522df1b9d89c0fba/MoltenVK/MoltenVK/GPUObjects/MVKRenderPass.mm#L407
	}

	render.end_encoding();
}

void MDCommandBuffer::_render_clear_render_area() {
	MDRenderPass const &pass = *render.pass;
	MDSubpass const &subpass = render.get_subpass();

	uint32_t ds_index = subpass.depth_stencil_reference.attachment;
	bool clear_depth = (ds_index != RDD::AttachmentReference::UNUSED && pass.attachments[ds_index].shouldClear(subpass, false));
	bool clear_stencil = (ds_index != RDD::AttachmentReference::UNUSED && pass.attachments[ds_index].shouldClear(subpass, true));

	uint32_t color_count = subpass.color_references.size();
	uint32_t clears_size = color_count + (clear_depth || clear_stencil ? 1 : 0);
	if (clears_size == 0) {
		return;
	}

	RDD::AttachmentClear *clears = ALLOCA_ARRAY(RDD::AttachmentClear, clears_size);
	uint32_t clears_count = 0;

	for (uint32_t i = 0; i < color_count; i++) {
		uint32_t idx = subpass.color_references[i].attachment;
		if (idx != RDD::AttachmentReference::UNUSED && pass.attachments[idx].shouldClear(subpass, false)) {
			clears[clears_count++] = { .aspect = RDD::TEXTURE_ASPECT_COLOR_BIT, .color_attachment = idx, .value = render.clear_values[idx] };
		}
	}

	if (clear_depth || clear_stencil) {
		MDAttachment const &attachment = pass.attachments[ds_index];
		BitField<RDD::TextureAspectBits> bits = {};
		if (clear_depth && attachment.type & MDAttachmentType::Depth) {
			bits.set_flag(RDD::TEXTURE_ASPECT_DEPTH_BIT);
		}
		if (clear_stencil && attachment.type & MDAttachmentType::Stencil) {
			bits.set_flag(RDD::TEXTURE_ASPECT_STENCIL_BIT);
		}

		clears[clears_count++] = { .aspect = bits, .color_attachment = ds_index, .value = render.clear_values[ds_index] };
	}

	if (clears_count == 0) {
		return;
	}

	render_clear_attachments(VectorView(clears, clears_count), { render.render_area });
}

void MDCommandBuffer::render_next_subpass() {
	DEV_ASSERT(command_buffer() != nil);

	if (render.current_subpass == UINT32_MAX) {
		render.current_subpass = 0;
	} else {
		_end_render_pass();
		render.current_subpass++;
	}

	MDFrameBuffer const &fb = *render.frameBuffer;
	MDRenderPass const &pass = *render.pass;
	MDSubpass const &subpass = render.get_subpass();

	MTLRenderPassDescriptor *desc = MTLRenderPassDescriptor.renderPassDescriptor;

	if (subpass.view_count > 1) {
		desc.renderTargetArrayLength = subpass.view_count;
	}

	PixelFormats &pf = device_driver->get_pixel_formats();

	uint32_t attachmentCount = 0;
	for (uint32_t i = 0; i < subpass.color_references.size(); i++) {
		uint32_t idx = subpass.color_references[i].attachment;
		if (idx == RDD::AttachmentReference::UNUSED) {
			continue;
		}

		attachmentCount += 1;
		MTLRenderPassColorAttachmentDescriptor *ca = desc.colorAttachments[i];

		uint32_t resolveIdx = subpass.resolve_references.is_empty() ? RDD::AttachmentReference::UNUSED : subpass.resolve_references[i].attachment;
		bool has_resolve = resolveIdx != RDD::AttachmentReference::UNUSED;
		bool can_resolve = true;
		if (resolveIdx != RDD::AttachmentReference::UNUSED) {
			id<MTLTexture> resolve_tex = fb.get_texture(resolveIdx);
			can_resolve = flags::all(pf.getCapabilities(resolve_tex.pixelFormat), kMTLFmtCapsResolve);
			if (can_resolve) {
				ca.resolveTexture = resolve_tex;
			} else {
				CRASH_NOW_MSG("unimplemented: using a texture format that is not supported for resolve");
			}
		}

		MDAttachment const &attachment = pass.attachments[idx];

		id<MTLTexture> tex = fb.get_texture(idx);
		ERR_FAIL_NULL_MSG(tex, "Frame buffer color texture is null.");

		if ((attachment.type & MDAttachmentType::Color)) {
			if (attachment.configureDescriptor(ca, pf, subpass, tex, render.is_rendering_entire_area, has_resolve, can_resolve, false)) {
				Color clearColor = render.clear_values[idx].color;
				ca.clearColor = MTLClearColorMake(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
			}
		}
	}

	if (subpass.depth_stencil_reference.attachment != RDD::AttachmentReference::UNUSED) {
		attachmentCount += 1;
		uint32_t idx = subpass.depth_stencil_reference.attachment;
		MDAttachment const &attachment = pass.attachments[idx];
		id<MTLTexture> tex = fb.get_texture(idx);
		ERR_FAIL_NULL_MSG(tex, "Frame buffer depth / stencil texture is null.");
		if (attachment.type & MDAttachmentType::Depth) {
			MTLRenderPassDepthAttachmentDescriptor *da = desc.depthAttachment;
			if (attachment.configureDescriptor(da, pf, subpass, tex, render.is_rendering_entire_area, false, false, false)) {
				da.clearDepth = render.clear_values[idx].depth;
			}
		}

		if (attachment.type & MDAttachmentType::Stencil) {
			MTLRenderPassStencilAttachmentDescriptor *sa = desc.stencilAttachment;
			if (attachment.configureDescriptor(sa, pf, subpass, tex, render.is_rendering_entire_area, false, false, true)) {
				sa.clearStencil = render.clear_values[idx].stencil;
			}
		}
	}

	desc.renderTargetWidth = MAX((NSUInteger)MIN(render.render_area.position.x + render.render_area.size.width, fb.size.width), 1u);
	desc.renderTargetHeight = MAX((NSUInteger)MIN(render.render_area.position.y + render.render_area.size.height, fb.size.height), 1u);

	if (attachmentCount == 0) {
		// If there are no attachments, delay the creation of the encoder,
		// so we can use a matching sample count for the pipeline, by setting
		// the defaultRasterSampleCount from the pipeline's sample count.
		render.desc = desc;
	} else {
		render.encoder = [command_buffer() renderCommandEncoderWithDescriptor:desc];
		_encode_barrier(render.encoder);

		if (!render.is_rendering_entire_area) {
			_render_clear_render_area();
		}
		// With a new encoder, all state is dirty.
		render.dirty.set_flag(RenderState::DIRTY_ALL);
	}
}

void MDCommandBuffer::render_draw(uint32_t p_vertex_count,
		uint32_t p_instance_count,
		uint32_t p_base_vertex,
		uint32_t p_first_instance) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	ERR_FAIL_NULL_MSG(render.pipeline, "No pipeline set for render command buffer.");

	_render_set_dirty_state();

	MDSubpass const &subpass = render.get_subpass();
	if (subpass.view_count > 1) {
		p_instance_count *= subpass.view_count;
	}

	DEV_ASSERT(render.dirty == 0);

	id<MTLRenderCommandEncoder> enc = render.encoder;

	[enc drawPrimitives:render.pipeline->raster_state.render_primitive
			  vertexStart:p_base_vertex
			  vertexCount:p_vertex_count
			instanceCount:p_instance_count
			 baseInstance:p_first_instance];
}

void MDCommandBuffer::render_bind_vertex_buffers(uint32_t p_binding_count, const RDD::BufferID *p_buffers, const uint64_t *p_offsets, uint64_t p_dynamic_offsets) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);

	render.vertex_buffers.resize(p_binding_count);
	render.vertex_offsets.resize(p_binding_count);

	// Are the existing buffer bindings the same?
	bool same = true;

	// Reverse the buffers, as their bindings are assigned in descending order.
	for (uint32_t i = 0; i < p_binding_count; i += 1) {
		const RenderingDeviceDriverMetal::BufferInfo *buf_info = (const RenderingDeviceDriverMetal::BufferInfo *)p_buffers[p_binding_count - i - 1].id;

		NSUInteger dynamic_offset = 0;
		if (buf_info->is_dynamic()) {
			const MetalBufferDynamicInfo *dyn_buf = (const MetalBufferDynamicInfo *)buf_info;
			uint64_t frame_idx = p_dynamic_offsets & 0x3;
			p_dynamic_offsets >>= 2;
			dynamic_offset = frame_idx * dyn_buf->size_bytes;
		}
		if (render.vertex_buffers[i] != buf_info->metal_buffer) {
			render.vertex_buffers[i] = buf_info->metal_buffer;
			same = false;
		}

		render.vertex_offsets[i] = dynamic_offset + p_offsets[p_binding_count - i - 1];
	}

	if (render.encoder) {
		uint32_t first = device_driver->get_metal_buffer_index_for_vertex_attribute_binding(p_binding_count - 1);
		if (same) {
			NSUInteger *offset_ptr = render.vertex_offsets.ptr();
			for (uint32_t i = first; i < first + p_binding_count; i++) {
				[render.encoder setVertexBufferOffset:*offset_ptr atIndex:i];
				offset_ptr++;
			}
		} else {
			[render.encoder setVertexBuffers:render.vertex_buffers.ptr()
									 offsets:render.vertex_offsets.ptr()
								   withRange:NSMakeRange(first, p_binding_count)];
		}
		render.dirty.clear_flag(RenderState::DIRTY_VERTEX);
	} else {
		render.dirty.set_flag(RenderState::DIRTY_VERTEX);
	}
}

void MDCommandBuffer::render_bind_index_buffer(RDD::BufferID p_buffer, RDD::IndexBufferFormat p_format, uint64_t p_offset) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);

	const RenderingDeviceDriverMetal::BufferInfo *buffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_buffer.id;

	render.index_buffer = buffer->metal_buffer;
	render.index_type = p_format == RDD::IndexBufferFormat::INDEX_BUFFER_FORMAT_UINT16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
	render.index_offset = p_offset;
}

void MDCommandBuffer::render_draw_indexed(uint32_t p_index_count,
		uint32_t p_instance_count,
		uint32_t p_first_index,
		int32_t p_vertex_offset,
		uint32_t p_first_instance) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	ERR_FAIL_NULL_MSG(render.pipeline, "No pipeline set for render command buffer.");

	_render_set_dirty_state();

	MDSubpass const &subpass = render.get_subpass();
	if (subpass.view_count > 1) {
		p_instance_count *= subpass.view_count;
	}

	id<MTLRenderCommandEncoder> enc = render.encoder;

	uint32_t index_offset = render.index_offset;
	index_offset += p_first_index * (render.index_type == MTLIndexTypeUInt16 ? sizeof(uint16_t) : sizeof(uint32_t));

	[enc drawIndexedPrimitives:render.pipeline->raster_state.render_primitive
					indexCount:p_index_count
					 indexType:render.index_type
				   indexBuffer:render.index_buffer
			 indexBufferOffset:index_offset
				 instanceCount:p_instance_count
					baseVertex:p_vertex_offset
				  baseInstance:p_first_instance];
}

void MDCommandBuffer::render_draw_indexed_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	ERR_FAIL_NULL_MSG(render.pipeline, "No pipeline set for render command buffer.");

	_render_set_dirty_state();

	id<MTLRenderCommandEncoder> enc = render.encoder;

	const RenderingDeviceDriverMetal::BufferInfo *indirect_buffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_indirect_buffer.id;
	NSUInteger indirect_offset = p_offset;

	for (uint32_t i = 0; i < p_draw_count; i++) {
		[enc drawIndexedPrimitives:render.pipeline->raster_state.render_primitive
						   indexType:render.index_type
						 indexBuffer:render.index_buffer
				   indexBufferOffset:0
					  indirectBuffer:indirect_buffer->metal_buffer
				indirectBufferOffset:indirect_offset];
		indirect_offset += p_stride;
	}
}

void MDCommandBuffer::render_draw_indexed_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	ERR_FAIL_MSG("not implemented");
}

void MDCommandBuffer::render_draw_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	ERR_FAIL_NULL_MSG(render.pipeline, "No pipeline set for render command buffer.");

	_render_set_dirty_state();

	id<MTLRenderCommandEncoder> enc = render.encoder;

	const RenderingDeviceDriverMetal::BufferInfo *indirect_buffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_indirect_buffer.id;
	NSUInteger indirect_offset = p_offset;

	for (uint32_t i = 0; i < p_draw_count; i++) {
		[enc drawPrimitives:render.pipeline->raster_state.render_primitive
					  indirectBuffer:indirect_buffer->metal_buffer
				indirectBufferOffset:indirect_offset];
		indirect_offset += p_stride;
	}
}

void MDCommandBuffer::render_draw_indirect_count(RDD::BufferID p_indirect_buffer, uint64_t p_offset, RDD::BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	ERR_FAIL_MSG("not implemented");
}

void MDCommandBuffer::render_end_pass() {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);

	render.end_encoding();
	render.reset();
	reset();
}

#pragma mark - RenderState

void MDCommandBuffer::RenderState::reset() {
	pass = nil;
	frameBuffer = nil;
	pipeline = nil;
	current_subpass = UINT32_MAX;
	render_area = {};
	is_rendering_entire_area = false;
	desc = nil;
	encoder = nil;
	index_buffer = nil;
	index_type = MTLIndexTypeUInt16;
	dirty = DIRTY_NONE;
	uniform_sets.clear();
	dynamic_offsets = 0;
	uniform_set_mask = 0;
	clear_values.clear();
	viewports.clear();
	scissors.clear();
	blend_constants.reset();
	bzero(vertex_buffers.ptr(), sizeof(id<MTLBuffer> __unsafe_unretained) * vertex_buffers.size());
	vertex_buffers.clear();
	bzero(vertex_offsets.ptr(), sizeof(NSUInteger) * vertex_offsets.size());
	vertex_offsets.clear();
	resource_tracker.reset();
}

void MDCommandBuffer::RenderState::end_encoding() {
	if (encoder == nil) {
		return;
	}

	[encoder endEncoding];
	encoder = nil;
}

#pragma mark - ComputeState

void MDCommandBuffer::ComputeState::end_encoding() {
	if (encoder == nil) {
		return;
	}

	[encoder endEncoding];
	encoder = nil;
}

#pragma mark - Compute

void MDCommandBuffer::_compute_set_dirty_state() {
	if (compute.dirty.has_flag(ComputeState::DIRTY_PIPELINE)) {
		compute.encoder = [command_buffer() computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
		_encode_barrier(compute.encoder);
		[compute.encoder setComputePipelineState:compute.pipeline->state];
	}

	_compute_bind_uniform_sets();

	if (compute.dirty.has_flag(ComputeState::DIRTY_PUSH)) {
		if (push_constant_binding != UINT32_MAX) {
			[compute.encoder setBytes:push_constant_data
							   length:push_constant_data_len
							  atIndex:push_constant_binding];
		}
	}

	if (!use_barriers) {
		compute.resource_tracker.encode(compute.encoder);
	}

	compute.dirty.clear();
}

void MDCommandBuffer::_compute_bind_uniform_sets() {
	DEV_ASSERT(type == MDCommandBufferStateType::Compute);
	if (!compute.dirty.has_flag(ComputeState::DIRTY_UNIFORMS)) {
		return;
	}

	compute.dirty.clear_flag(ComputeState::DIRTY_UNIFORMS);
	uint64_t set_uniforms = compute.uniform_set_mask;
	compute.uniform_set_mask = 0;

	MDComputeShader *shader = compute.pipeline->shader;
	const uint32_t dynamic_offsets = compute.dynamic_offsets;

	while (set_uniforms != 0) {
		// Find the index of the next set bit.
		uint32_t index = (uint32_t)__builtin_ctzll(set_uniforms);
		// Clear the set bit.
		set_uniforms &= (set_uniforms - 1);
		MDUniformSet *set = compute.uniform_sets[index];
		if (set == nullptr || index >= (uint32_t)shader->sets.size()) {
			continue;
		}
		if (shader->uses_argument_buffers) {
			_bind_uniforms_argument_buffers_compute(set, shader, index, dynamic_offsets);
		} else {
			DirectEncoder de(compute.encoder, binding_cache);
			_bind_uniforms_direct(set, shader, de, index, dynamic_offsets);
		}
	}
}

void MDCommandBuffer::ComputeState::reset() {
	pipeline = nil;
	encoder = nil;
	dirty = DIRTY_NONE;
	uniform_sets.clear();
	dynamic_offsets = 0;
	uniform_set_mask = 0;
	resource_tracker.reset();
}

void MDCommandBuffer::compute_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(type == MDCommandBufferStateType::Compute);

	compute.dynamic_offsets |= p_dynamic_offsets;

	if (uint32_t new_size = p_first_set_index + p_set_count; compute.uniform_sets.size() < new_size) {
		uint32_t s = compute.uniform_sets.size();
		compute.uniform_sets.resize(new_size);
		// Set intermediate values to null.
		std::fill(&compute.uniform_sets[s], compute.uniform_sets.end().operator->(), nullptr);
	}

	const MDShader *shader = (const MDShader *)p_shader.id;
	DynamicOffsetLayout layout = shader->dynamic_offset_layout;

	for (size_t i = 0; i < p_set_count; ++i) {
		MDUniformSet *set = (MDUniformSet *)(p_uniform_sets[i].id);

		uint32_t index = p_first_set_index + i;
		if (compute.uniform_sets[index] != set || layout.get_count(index) > 0) {
			compute.dirty.set_flag(ComputeState::DIRTY_UNIFORMS);
			compute.uniform_set_mask |= 1ULL << index;
			compute.uniform_sets[index] = set;
		}
	}
}

void MDCommandBuffer::compute_dispatch(uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) {
	DEV_ASSERT(type == MDCommandBufferStateType::Compute);

	_compute_set_dirty_state();

	MTLRegion region = MTLRegionMake3D(0, 0, 0, p_x_groups, p_y_groups, p_z_groups);

	id<MTLComputeCommandEncoder> enc = compute.encoder;
	[enc dispatchThreadgroups:region.size threadsPerThreadgroup:compute.pipeline->compute_state.local];
}

void MDCommandBuffer::compute_dispatch_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset) {
	DEV_ASSERT(type == MDCommandBufferStateType::Compute);

	_compute_set_dirty_state();

	const RenderingDeviceDriverMetal::BufferInfo *indirectBuffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_indirect_buffer.id;

	id<MTLComputeCommandEncoder> enc = compute.encoder;
	[enc dispatchThreadgroupsWithIndirectBuffer:indirectBuffer->metal_buffer indirectBufferOffset:p_offset threadsPerThreadgroup:compute.pipeline->compute_state.local];
}

void MDCommandBuffer::reset() {
	push_constant_binding = UINT32_MAX;
	push_constant_data_len = 0;
	type = MDCommandBufferStateType::None;
	binding_cache.clear();
}

void MDCommandBuffer::_end_compute_dispatch() {
	DEV_ASSERT(type == MDCommandBufferStateType::Compute);

	compute.end_encoding();
	compute.reset();
	reset();
}

void MDCommandBuffer::_end_blit() {
	DEV_ASSERT(type == MDCommandBufferStateType::Blit);

	[blit.encoder endEncoding];
	blit.reset();
	reset();
}

MDComputeShader::MDComputeShader(CharString p_name,
		Vector<UniformSet> p_sets,
		bool p_uses_argument_buffers,
		std::shared_ptr<MDLibrary> p_kernel) :
		MDShader(p_name, p_sets, p_uses_argument_buffers), kernel(std::move(p_kernel)) {
}

MDRenderShader::MDRenderShader(CharString p_name,
		Vector<UniformSet> p_sets,
		bool p_needs_view_mask_buffer,
		bool p_uses_argument_buffers,
		std::shared_ptr<MDLibrary> p_vert, std::shared_ptr<MDLibrary> p_frag) :
		MDShader(p_name, p_sets, p_uses_argument_buffers),
		needs_view_mask_buffer(p_needs_view_mask_buffer),
		vert(std::move(p_vert)),
		frag(std::move(p_frag)) {
}

void DirectEncoder::set(__unsafe_unretained id<MTLTexture> *p_textures, NSRange p_range) {
	if (cache.update(p_range, p_textures)) {
		switch (mode) {
			case RENDER: {
				id<MTLRenderCommandEncoder> __unsafe_unretained enc = (id<MTLRenderCommandEncoder>)encoder;
				[enc setVertexTextures:p_textures withRange:p_range];
				[enc setFragmentTextures:p_textures withRange:p_range];
			} break;
			case COMPUTE: {
				id<MTLComputeCommandEncoder> __unsafe_unretained enc = (id<MTLComputeCommandEncoder>)encoder;
				[enc setTextures:p_textures withRange:p_range];
			} break;
		}
	}
}

void DirectEncoder::set(__unsafe_unretained id<MTLBuffer> *p_buffers, const NSUInteger *p_offsets, NSRange p_range) {
	if (cache.update(p_range, p_buffers, p_offsets)) {
		switch (mode) {
			case RENDER: {
				id<MTLRenderCommandEncoder> __unsafe_unretained enc = (id<MTLRenderCommandEncoder>)encoder;
				[enc setVertexBuffers:p_buffers offsets:p_offsets withRange:p_range];
				[enc setFragmentBuffers:p_buffers offsets:p_offsets withRange:p_range];
			} break;
			case COMPUTE: {
				id<MTLComputeCommandEncoder> __unsafe_unretained enc = (id<MTLComputeCommandEncoder>)encoder;
				[enc setBuffers:p_buffers offsets:p_offsets withRange:p_range];
			} break;
		}
	}
}

void DirectEncoder::set(id<MTLBuffer> __unsafe_unretained p_buffer, const NSUInteger p_offset, uint32_t p_index) {
	if (cache.update(p_buffer, p_offset, p_index)) {
		switch (mode) {
			case RENDER: {
				id<MTLRenderCommandEncoder> __unsafe_unretained enc = (id<MTLRenderCommandEncoder>)encoder;
				[enc setVertexBuffer:p_buffer offset:p_offset atIndex:p_index];
				[enc setFragmentBuffer:p_buffer offset:p_offset atIndex:p_index];
			} break;
			case COMPUTE: {
				id<MTLComputeCommandEncoder> __unsafe_unretained enc = (id<MTLComputeCommandEncoder>)encoder;
				[enc setBuffer:p_buffer offset:p_offset atIndex:p_index];
			} break;
		}
	}
}

void DirectEncoder::set(__unsafe_unretained id<MTLSamplerState> *p_samplers, NSRange p_range) {
	if (cache.update(p_range, p_samplers)) {
		switch (mode) {
			case RENDER: {
				id<MTLRenderCommandEncoder> __unsafe_unretained enc = (id<MTLRenderCommandEncoder>)encoder;
				[enc setVertexSamplerStates:p_samplers withRange:p_range];
				[enc setFragmentSamplerStates:p_samplers withRange:p_range];
			} break;
			case COMPUTE: {
				id<MTLComputeCommandEncoder> __unsafe_unretained enc = (id<MTLComputeCommandEncoder>)encoder;
				[enc setSamplerStates:p_samplers withRange:p_range];
			} break;
		}
	}
}

GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability-new")

void MDCommandBuffer::_bind_uniforms_argument_buffers(MDUniformSet *p_set, MDShader *p_shader, uint32_t p_set_index, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(p_shader->uses_argument_buffers);
	DEV_ASSERT(render.encoder != nil);

	id<MTLRenderCommandEncoder> __unsafe_unretained enc = render.encoder;
	render.resource_tracker.merge_from(p_set->usage_to_resources);

	const UniformSet &shader_set = p_shader->sets[p_set_index];

	// Check if this set has dynamic uniforms.
	if (!shader_set.dynamic_uniforms.is_empty()) {
		// Allocate from the ring buffer.
		uint32_t buffer_size = p_set->arg_buffer_data.size();
		MDRingBuffer::Allocation alloc = allocate_arg_buffer(buffer_size);

		// Copy the base argument buffer data.
		memcpy(alloc.ptr, p_set->arg_buffer_data.ptr(), buffer_size);

		// Update dynamic buffer GPU addresses.
		uint64_t *ptr = (uint64_t *)alloc.ptr;
		DynamicOffsetLayout layout = p_shader->dynamic_offset_layout;
		uint32_t dynamic_index = 0;

		for (uint32_t i : shader_set.dynamic_uniforms) {
			RDD::BoundUniform const &uniform = p_set->uniforms[i];
			const UniformInfo &ui = shader_set.uniforms[i];
			const UniformInfo::Indexes &idx = ui.arg_buffer;

			uint32_t shift = layout.get_offset_index_shift(p_set_index, dynamic_index);
			dynamic_index++;
			uint32_t frame_idx = (p_dynamic_offsets >> shift) & 0xf;

			const MetalBufferDynamicInfo *buf_info = (const MetalBufferDynamicInfo *)uniform.ids[0].id;
			uint64_t gpu_address = buf_info->metal_buffer.gpuAddress + frame_idx * buf_info->size_bytes;
			*(MTLGPUAddress *)(ptr + idx.buffer) = gpu_address;
		}

		[enc setVertexBuffer:alloc.buffer offset:alloc.offset atIndex:p_set_index];
		[enc setFragmentBuffer:alloc.buffer offset:alloc.offset atIndex:p_set_index];
	} else {
		[enc setVertexBuffer:p_set->arg_buffer offset:0 atIndex:p_set_index];
		[enc setFragmentBuffer:p_set->arg_buffer offset:0 atIndex:p_set_index];
	}
}

void MDCommandBuffer::_bind_uniforms_direct(MDUniformSet *p_set, MDShader *p_shader, DirectEncoder p_enc, uint32_t p_set_index, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(!p_shader->uses_argument_buffers);

	UniformSet const &set = p_shader->sets[p_set_index];
	DynamicOffsetLayout layout = p_shader->dynamic_offset_layout;
	uint32_t dynamic_index = 0;

	for (uint32_t i = 0; i < MIN(p_set->uniforms.size(), set.uniforms.size()); i++) {
		RDD::BoundUniform const &uniform = p_set->uniforms[i];
		const UniformInfo &ui = set.uniforms[i];
		const UniformInfo::Indexes &indexes = ui.slot;

		uint32_t frame_idx;
		if (uniform.is_dynamic()) {
			uint32_t shift = layout.get_offset_index_shift(p_set_index, dynamic_index);
			dynamic_index++;
			frame_idx = (p_dynamic_offsets >> shift) & 0xf;
		} else {
			frame_idx = 0;
		}

		switch (uniform.type) {
			case RDD::UNIFORM_TYPE_SAMPLER: {
				size_t count = uniform.ids.size();
				id<MTLSamplerState> __unsafe_unretained *objects = ALLOCA_ARRAY(id<MTLSamplerState> __unsafe_unretained, count);
				for (size_t j = 0; j < count; j += 1) {
					objects[j] = rid::get(uniform.ids[j].id);
				}
				NSRange sampler_range = NSMakeRange(indexes.sampler, count);
				p_enc.set(objects, sampler_range);
			} break;
			case RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE: {
				size_t count = uniform.ids.size() / 2;
				id<MTLTexture> __unsafe_unretained *textures = ALLOCA_ARRAY(id<MTLTexture> __unsafe_unretained, count);
				id<MTLSamplerState> __unsafe_unretained *samplers = ALLOCA_ARRAY(id<MTLSamplerState> __unsafe_unretained, count);
				for (uint32_t j = 0; j < count; j += 1) {
					id<MTLSamplerState> sampler = rid::get(uniform.ids[j * 2 + 0]);
					id<MTLTexture> texture = rid::get(uniform.ids[j * 2 + 1]);
					samplers[j] = sampler;
					textures[j] = texture;
				}
				NSRange sampler_range = NSMakeRange(indexes.sampler, count);
				NSRange texture_range = NSMakeRange(indexes.texture, count);
				p_enc.set(samplers, sampler_range);
				p_enc.set(textures, texture_range);
			} break;
			case RDD::UNIFORM_TYPE_TEXTURE: {
				size_t count = uniform.ids.size();
				id<MTLTexture> __unsafe_unretained *objects = ALLOCA_ARRAY(id<MTLTexture> __unsafe_unretained, count);
				for (size_t j = 0; j < count; j += 1) {
					id<MTLTexture> obj = rid::get(uniform.ids[j]);
					objects[j] = obj;
				}
				NSRange texture_range = NSMakeRange(indexes.texture, count);
				p_enc.set(objects, texture_range);
			} break;
			case RDD::UNIFORM_TYPE_IMAGE: {
				size_t count = uniform.ids.size();
				id<MTLTexture> __unsafe_unretained *objects = ALLOCA_ARRAY(id<MTLTexture> __unsafe_unretained, count);
				for (size_t j = 0; j < count; j += 1) {
					id<MTLTexture> obj = rid::get(uniform.ids[j]);
					objects[j] = obj;
				}
				NSRange texture_range = NSMakeRange(indexes.texture, count);
				p_enc.set(objects, texture_range);

				if (indexes.buffer != UINT32_MAX) {
					// Emulated atomic image access.
					id<MTLBuffer> __unsafe_unretained *bufs = ALLOCA_ARRAY(id<MTLBuffer> __unsafe_unretained, count);
					for (size_t j = 0; j < count; j += 1) {
						id<MTLTexture> obj = rid::get(uniform.ids[j]);
						id<MTLTexture> tex = obj.parentTexture ? obj.parentTexture : obj;
						id<MTLBuffer> buf = tex.buffer;
						bufs[j] = buf;
					}
					NSUInteger *offs = ALLOCA_ARRAY(NSUInteger, count);
					bzero(offs, sizeof(NSUInteger) * count);
					NSRange buffer_range = NSMakeRange(indexes.buffer, count);
					p_enc.set(bufs, offs, buffer_range);
				}
			} break;
			case RDD::UNIFORM_TYPE_TEXTURE_BUFFER: {
				ERR_PRINT("not implemented: UNIFORM_TYPE_TEXTURE_BUFFER");
			} break;
			case RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER: {
				ERR_PRINT("not implemented: UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER");
			} break;
			case RDD::UNIFORM_TYPE_IMAGE_BUFFER: {
				CRASH_NOW_MSG("not implemented: UNIFORM_TYPE_IMAGE_BUFFER");
			} break;
			case RDD::UNIFORM_TYPE_UNIFORM_BUFFER:
			case RDD::UNIFORM_TYPE_STORAGE_BUFFER: {
				const RDM::BufferInfo *buf_info = (const RDM::BufferInfo *)uniform.ids[0].id;
				p_enc.set(buf_info->metal_buffer, 0, indexes.buffer);
			} break;
			case RDD::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC:
			case RDD::UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC: {
				const MetalBufferDynamicInfo *buf_info = (const MetalBufferDynamicInfo *)uniform.ids[0].id;
				p_enc.set(buf_info->metal_buffer, frame_idx * buf_info->size_bytes, indexes.buffer);
			} break;
			case RDD::UNIFORM_TYPE_INPUT_ATTACHMENT: {
				size_t count = uniform.ids.size();
				id<MTLTexture> __unsafe_unretained *objects = ALLOCA_ARRAY(id<MTLTexture> __unsafe_unretained, count);
				for (size_t j = 0; j < count; j += 1) {
					id<MTLTexture> obj = rid::get(uniform.ids[j]);
					objects[j] = obj;
				}
				NSRange texture_range = NSMakeRange(indexes.texture, count);
				p_enc.set(objects, texture_range);
			} break;
			default: {
				DEV_ASSERT(false);
			}
		}
	}
}

void MDCommandBuffer::_bind_uniforms_argument_buffers_compute(MDUniformSet *p_set, MDShader *p_shader, uint32_t p_set_index, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(p_shader->uses_argument_buffers);
	DEV_ASSERT(compute.encoder != nil);

	id<MTLComputeCommandEncoder> enc = compute.encoder;
	compute.resource_tracker.merge_from(p_set->usage_to_resources);

	const UniformSet &shader_set = p_shader->sets[p_set_index];

	// Check if this set has dynamic uniforms.
	if (!shader_set.dynamic_uniforms.is_empty()) {
		// Allocate from the ring buffer.
		uint32_t buffer_size = p_set->arg_buffer_data.size();
		MDRingBuffer::Allocation alloc = allocate_arg_buffer(buffer_size);

		// Copy the base argument buffer data.
		memcpy(alloc.ptr, p_set->arg_buffer_data.ptr(), buffer_size);

		// Update dynamic buffer GPU addresses.
		uint64_t *ptr = (uint64_t *)alloc.ptr;
		DynamicOffsetLayout layout = p_shader->dynamic_offset_layout;
		uint32_t dynamic_index = 0;

		for (uint32_t i : shader_set.dynamic_uniforms) {
			RDD::BoundUniform const &uniform = p_set->uniforms[i];
			const UniformInfo &ui = shader_set.uniforms[i];
			const UniformInfo::Indexes &idx = ui.arg_buffer;

			uint32_t shift = layout.get_offset_index_shift(p_set_index, dynamic_index);
			dynamic_index++;
			uint32_t frame_idx = (p_dynamic_offsets >> shift) & 0xf;

			const MetalBufferDynamicInfo *buf_info = (const MetalBufferDynamicInfo *)uniform.ids[0].id;
			uint64_t gpu_address = buf_info->metal_buffer.gpuAddress + frame_idx * buf_info->size_bytes;
			*(MTLGPUAddress *)(ptr + idx.buffer) = gpu_address;
		}

		[enc setBuffer:alloc.buffer offset:alloc.offset atIndex:p_set_index];
	} else {
		[enc setBuffer:p_set->arg_buffer offset:0 atIndex:p_set_index];
	}
}

GODOT_CLANG_WARNING_POP
