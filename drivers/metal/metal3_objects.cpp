/**************************************************************************/
/*  metal3_objects.cpp                                                    */
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

#include "metal3_objects.h"

#include "metal_utils.h"
#include "pixel_formats.h"
#include "rendering_device_driver_metal3.h"
#include "rendering_shader_container_metal.h"

#include <algorithm>

using namespace MTL3;

MDCommandBuffer::MDCommandBuffer(MTL::CommandQueue *p_queue, ::RenderingDeviceDriverMetal *p_device_driver) :
		_scratch(p_queue->device()), queue(p_queue) {
	device_driver = p_device_driver;
	type = MDCommandBufferStateType::None;
	use_barriers = device_driver->use_barriers;
	if (use_barriers) {
		// Already validated availability if use_barriers is true.
		MTL::Device *device = p_queue->device();
		NS::SharedPtr<MTL::ResidencySetDescriptor> rs_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
		rs_desc->setInitialCapacity(10);
		rs_desc->setLabel(MTLSTR("Command Residency Set"));
		NS::Error *error = nullptr;
		_frame_state.rs = NS::TransferPtr(device->newResidencySet(rs_desc.get(), &error));
		CRASH_COND_MSG(error != nullptr, vformat("Failed to create residency set: %s", String(error->localizedDescription()->utf8String())));
	}
}

void MDCommandBuffer::begin_label(const char *p_label_name, const Color &p_color) {
	NS::SharedPtr<NS::String> s = NS::TransferPtr(NS::String::alloc()->init(p_label_name, NS::UTF8StringEncoding));
	command_buffer()->pushDebugGroup(s.get());
}

void MDCommandBuffer::end_label() {
	command_buffer()->popDebugGroup();
}

void MDCommandBuffer::begin() {
	DEV_ASSERT(commandBuffer.get() == nullptr && !state_begin);
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
			Span<MTL::Buffer *const> bufs = _scratch.get_buffers();
			_frame_state.rs->addAllocations(reinterpret_cast<const MTL::Allocation *const *>(bufs.ptr()), bufs.size());
			_scratch.clear_changed();
			_frame_state.rs->commit();
		}
	}
	commandBuffer->commit();
	commandBuffer.reset();
	state_begin = false;
}

MTL::CommandBuffer *MDCommandBuffer::command_buffer() {
	DEV_ASSERT(state_begin);
	if (commandBuffer.get() == nullptr) {
		commandBuffer = NS::RetainPtr(queue->commandBuffer());
		if (use_barriers) {
			commandBuffer->useResidencySet(_frame_state.rs.get());
		}
	}
	return commandBuffer.get();
}

void MDCommandBuffer::_encode_barrier(MTL::CommandEncoder *p_enc) {
	DEV_ASSERT(p_enc);

	static const MTL::Stages empty_stages[STAGE_MAX] = { 0, 0, 0 };
	if (memcmp(&pending_before_queue_stages, empty_stages, sizeof(pending_before_queue_stages)) == 0) {
		return;
	}

	int stage = STAGE_MAX;
	// Determine encoder type by checking if it's the current active encoder.
	if (render.encoder.get() == p_enc && pending_after_stages[STAGE_RENDER] != 0) {
		stage = STAGE_RENDER;
	} else if (compute.encoder.get() == p_enc && pending_after_stages[STAGE_COMPUTE] != 0) {
		stage = STAGE_COMPUTE;
	} else if (blit.encoder.get() == p_enc && pending_after_stages[STAGE_BLIT] != 0) {
		stage = STAGE_BLIT;
	}

	if (stage == STAGE_MAX) {
		return;
	}

	p_enc->barrierAfterQueueStages(pending_after_stages[stage], pending_before_queue_stages[stage]);
	pending_before_queue_stages[stage] = 0;
	pending_after_stages[stage] = 0;
}

void MDCommandBuffer::pipeline_barrier(BitField<RDD::PipelineStageBits> p_src_stages,
		BitField<RDD::PipelineStageBits> p_dst_stages,
		VectorView<RDD::MemoryAccessBarrier> p_memory_barriers,
		VectorView<RDD::BufferBarrier> p_buffer_barriers,
		VectorView<RDD::TextureBarrier> p_texture_barriers,
		VectorView<RDD::AccelerationStructureBarrier> p_acceleration_structure_barriers) {
	MTL::Stages after_stages = convert_src_pipeline_stages_to_metal(p_src_stages);
	if (after_stages == 0) {
		return;
	}

	MTL::Stages before_stages = convert_dst_pipeline_stages_to_metal(p_dst_stages);
	if (before_stages == 0) {
		return;
	}

	// Encode intra-encoder memory barrier if an encoder is active for matching stages.
	if (render.encoder.get() != nullptr) {
		MTL::RenderStages render_after = static_cast<MTL::RenderStages>(after_stages & (MTL::StageVertex | MTL::StageFragment));
		MTL::RenderStages render_before = static_cast<MTL::RenderStages>(before_stages & (MTL::StageVertex | MTL::StageFragment));
		if (render_after != 0 && render_before != 0) {
			render.encoder->memoryBarrier(MTL::BarrierScopeBuffers | MTL::BarrierScopeTextures, render_after, render_before);
		}
	} else if (compute.encoder.get() != nullptr) {
		if (after_stages & MTL::StageDispatch) {
			compute.encoder->memoryBarrier(MTL::BarrierScopeBuffers | MTL::BarrierScopeTextures);
		}
	}
	// Blit encoder has no memory barrier API.

	// Also cache for inter-pass barriers based on DESTINATION stages,
	// since barrierAfterQueueStages is called on the encoder that must wait.
	if (before_stages & (MTL::StageVertex | MTL::StageFragment)) {
		pending_after_stages[STAGE_RENDER] |= after_stages;
		pending_before_queue_stages[STAGE_RENDER] |= before_stages;
	}

	if (before_stages & MTL::StageDispatch) {
		pending_after_stages[STAGE_COMPUTE] |= after_stages;
		pending_before_queue_stages[STAGE_COMPUTE] |= before_stages;
	}

	if (before_stages & MTL::StageBlit) {
		pending_after_stages[STAGE_BLIT] |= after_stages;
		pending_before_queue_stages[STAGE_BLIT] |= before_stages;
	}
}

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

		if (render.encoder.get() == nullptr) {
			// This error would happen if the render pass failed.
			ERR_FAIL_NULL_MSG(render.desc.get(), "Render pass descriptor is null.");

			// This condition occurs when there are no attachments when calling render_next_subpass()
			// and is due to the SUPPORTS_FRAGMENT_SHADER_WITH_ONLY_SIDE_EFFECTS flag.
			render.desc->setDefaultRasterSampleCount(static_cast<NS::UInteger>(rp->sample_count));

			render.encoder = NS::RetainPtr(command_buffer()->renderCommandEncoder(render.desc.get()));
			_encode_barrier(render.encoder.get());
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

void MDCommandBuffer::mark_push_constants_dirty() {
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

MTL::BlitCommandEncoder *MDCommandBuffer::_ensure_blit_encoder() {
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
			return blit.encoder.get();
	}

	type = MDCommandBufferStateType::Blit;
	blit.encoder = NS::RetainPtr(command_buffer()->blitCommandEncoder());
	_encode_barrier(blit.encoder.get());

	return blit.encoder.get();
}

void MDCommandBuffer::resolve_texture(RDD::TextureID p_src_texture, RDD::TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, RDD::TextureID p_dst_texture, RDD::TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) {
	MTL::Texture *src_tex = rid::get<MTL::Texture>(p_src_texture);
	MTL::Texture *dst_tex = rid::get<MTL::Texture>(p_dst_texture);

	NS::SharedPtr<MTL::RenderPassDescriptor> mtlRPD = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());
	MTL::RenderPassColorAttachmentDescriptor *mtlColorAttDesc = mtlRPD->colorAttachments()->object(0);
	mtlColorAttDesc->setLoadAction(MTL::LoadActionLoad);
	mtlColorAttDesc->setStoreAction(MTL::StoreActionMultisampleResolve);

	mtlColorAttDesc->setTexture(src_tex);
	mtlColorAttDesc->setResolveTexture(dst_tex);
	mtlColorAttDesc->setLevel(p_src_mipmap);
	mtlColorAttDesc->setSlice(p_src_layer);
	mtlColorAttDesc->setResolveLevel(p_dst_mipmap);
	mtlColorAttDesc->setResolveSlice(p_dst_layer);
	MTL::RenderCommandEncoder *enc = get_new_render_encoder_with_descriptor(mtlRPD.get());
	enc->setLabel(MTLSTR("Resolve Image"));
	enc->endEncoding();
}

void MDCommandBuffer::clear_color_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, const Color &p_color, const RDD::TextureSubresourceRange &p_subresources) {
	MTL::Texture *src_tex = rid::get<MTL::Texture>(p_texture);

	if (src_tex->parentTexture()) {
		// Clear via the parent texture rather than the view.
		src_tex = src_tex->parentTexture();
	}

	PixelFormats &pf = device_driver->get_pixel_formats();

	if (pf.isDepthFormat(src_tex->pixelFormat()) || pf.isStencilFormat(src_tex->pixelFormat())) {
		ERR_FAIL_MSG("invalid: depth or stencil texture format");
	}

	NS::SharedPtr<MTL::RenderPassDescriptor> desc = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());

	if (p_subresources.aspect.has_flag(RDD::TEXTURE_ASPECT_COLOR_BIT)) {
		MTL::RenderPassColorAttachmentDescriptor *caDesc = desc->colorAttachments()->object(0);
		caDesc->setTexture(src_tex);
		caDesc->setLoadAction(MTL::LoadActionClear);
		caDesc->setStoreAction(MTL::StoreActionStore);
		caDesc->setClearColor(MTL::ClearColor(p_color.r, p_color.g, p_color.b, p_color.a));

		// Extract the mipmap levels that are to be updated.
		uint32_t mipLvlStart = p_subresources.base_mipmap;
		uint32_t mipLvlCnt = p_subresources.mipmap_count;
		uint32_t mipLvlEnd = mipLvlStart + mipLvlCnt;

		uint32_t levelCount = src_tex->mipmapLevelCount();

		// Extract the cube or array layers (slices) that are to be updated.
		bool is3D = src_tex->textureType() == MTL::TextureType3D;
		uint32_t layerStart = is3D ? 0 : p_subresources.base_layer;
		uint32_t layerCnt = p_subresources.layer_count;
		uint32_t layerEnd = layerStart + layerCnt;

		MetalFeatures const &features = device_driver->get_device_properties().features;

		// Iterate across mipmap levels and layers, and perform and empty render to clear each.
		for (uint32_t mipLvl = mipLvlStart; mipLvl < mipLvlEnd; mipLvl++) {
			ERR_FAIL_INDEX_MSG(mipLvl, levelCount, "mip level out of range");

			caDesc->setLevel(mipLvl);

			// If a 3D image, we need to get the depth for each level.
			if (is3D) {
				layerCnt = mipmapLevelSizeFromTexture(src_tex, mipLvl).depth;
				layerEnd = layerStart + layerCnt;
			}

			if ((features.layeredRendering && src_tex->sampleCount() == 1) || features.multisampleLayeredRendering) {
				// We can clear all layers at once.
				if (is3D) {
					caDesc->setDepthPlane(layerStart);
				} else {
					caDesc->setSlice(layerStart);
				}
				desc->setRenderTargetArrayLength(layerCnt);
				MTL::RenderCommandEncoder *enc = get_new_render_encoder_with_descriptor(desc.get());
				enc->setLabel(MTLSTR("Clear Image"));
				enc->endEncoding();
			} else {
				for (uint32_t layer = layerStart; layer < layerEnd; layer++) {
					if (is3D) {
						caDesc->setDepthPlane(layer);
					} else {
						caDesc->setSlice(layer);
					}
					MTL::RenderCommandEncoder *enc = get_new_render_encoder_with_descriptor(desc.get());
					enc->setLabel(MTLSTR("Clear Image"));
					enc->endEncoding();
				}
			}
		}
	}
}

void MDCommandBuffer::clear_buffer(RDD::BufferID p_buffer, uint64_t p_offset, uint64_t p_size) {
	MTL::BlitCommandEncoder *blit_enc = _ensure_blit_encoder();
	const RDM::BufferInfo *buffer = (const RDM::BufferInfo *)p_buffer.id;

	blit_enc->fillBuffer(buffer->metal_buffer.get(), NS::Range(p_offset, p_size), 0);
}

void MDCommandBuffer::clear_depth_stencil_texture(RDD::TextureID p_texture, RDD::TextureLayout p_texture_layout, float p_depth, uint8_t p_stencil, const RDD::TextureSubresourceRange &p_subresources) {
	MTL::Texture *src_tex = rid::get<MTL::Texture>(p_texture);

	if (src_tex->parentTexture()) {
		// Clear via the parent texture rather than the view.
		src_tex = src_tex->parentTexture();
	}

	PixelFormats &pf = device_driver->get_pixel_formats();

	bool is_depth_format = pf.isDepthFormat(src_tex->pixelFormat());
	bool is_stencil_format = pf.isStencilFormat(src_tex->pixelFormat());

	if (!is_depth_format && !is_stencil_format) {
		ERR_FAIL_MSG("invalid: color texture format");
	}

	bool clear_depth = is_depth_format && p_subresources.aspect.has_flag(RDD::TEXTURE_ASPECT_DEPTH_BIT);
	bool clear_stencil = is_stencil_format && p_subresources.aspect.has_flag(RDD::TEXTURE_ASPECT_STENCIL_BIT);

	if (clear_depth || clear_stencil) {
		NS::SharedPtr<MTL::RenderPassDescriptor> desc = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());

		MTL::RenderPassDepthAttachmentDescriptor *daDesc = desc->depthAttachment();
		if (clear_depth) {
			daDesc->setTexture(src_tex);
			daDesc->setLoadAction(MTL::LoadActionClear);
			daDesc->setStoreAction(MTL::StoreActionStore);
			daDesc->setClearDepth(p_depth);
		}

		MTL::RenderPassStencilAttachmentDescriptor *saDesc = desc->stencilAttachment();
		if (clear_stencil) {
			saDesc->setTexture(src_tex);
			saDesc->setLoadAction(MTL::LoadActionClear);
			saDesc->setStoreAction(MTL::StoreActionStore);
			saDesc->setClearStencil(p_stencil);
		}

		// Extract the mipmap levels that are to be updated.
		uint32_t mipLvlStart = p_subresources.base_mipmap;
		uint32_t mipLvlCnt = p_subresources.mipmap_count;
		uint32_t mipLvlEnd = mipLvlStart + mipLvlCnt;

		uint32_t levelCount = src_tex->mipmapLevelCount();

		// Extract the cube or array layers (slices) that are to be updated.
		bool is3D = src_tex->textureType() == MTL::TextureType3D;
		uint32_t layerStart = is3D ? 0 : p_subresources.base_layer;
		uint32_t layerCnt = p_subresources.layer_count;
		uint32_t layerEnd = layerStart + layerCnt;

		MetalFeatures const &features = device_driver->get_device_properties().features;

		// Iterate across mipmap levels and layers, and perform and empty render to clear each.
		for (uint32_t mipLvl = mipLvlStart; mipLvl < mipLvlEnd; mipLvl++) {
			ERR_FAIL_INDEX_MSG(mipLvl, levelCount, "mip level out of range");

			if (clear_depth) {
				daDesc->setLevel(mipLvl);
			}
			if (clear_stencil) {
				saDesc->setLevel(mipLvl);
			}

			// If a 3D image, we need to get the depth for each level.
			if (is3D) {
				layerCnt = mipmapLevelSizeFromTexture(src_tex, mipLvl).depth;
				layerEnd = layerStart + layerCnt;
			}

			if ((features.layeredRendering && src_tex->sampleCount() == 1) || features.multisampleLayeredRendering) {
				// We can clear all layers at once.
				if (is3D) {
					if (clear_depth) {
						daDesc->setDepthPlane(layerStart);
					}
					if (clear_stencil) {
						saDesc->setDepthPlane(layerStart);
					}
				} else {
					if (clear_depth) {
						daDesc->setSlice(layerStart);
					}
					if (clear_stencil) {
						saDesc->setSlice(layerStart);
					}
				}
				desc->setRenderTargetArrayLength(layerCnt);
				MTL::RenderCommandEncoder *enc = get_new_render_encoder_with_descriptor(desc.get());
				enc->setLabel(MTLSTR("Clear Image"));
				enc->endEncoding();
			} else {
				for (uint32_t layer = layerStart; layer < layerEnd; layer++) {
					if (is3D) {
						if (clear_depth) {
							daDesc->setDepthPlane(layer);
						}
						if (clear_stencil) {
							saDesc->setDepthPlane(layer);
						}
					} else {
						if (clear_depth) {
							daDesc->setSlice(layer);
						}
						if (clear_stencil) {
							saDesc->setSlice(layer);
						}
					}
					MTL::RenderCommandEncoder *enc = get_new_render_encoder_with_descriptor(desc.get());
					enc->setLabel(MTLSTR("Clear Image"));
					enc->endEncoding();
				}
			}
		}
	}
}

void MDCommandBuffer::copy_buffer(RDD::BufferID p_src_buffer, RDD::BufferID p_dst_buffer, VectorView<RDD::BufferCopyRegion> p_regions) {
	const RDM::BufferInfo *src = (const RDM::BufferInfo *)p_src_buffer.id;
	const RDM::BufferInfo *dst = (const RDM::BufferInfo *)p_dst_buffer.id;

	MTL::BlitCommandEncoder *enc = _ensure_blit_encoder();

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		RDD::BufferCopyRegion region = p_regions[i];
		enc->copyFromBuffer(src->metal_buffer.get(), region.src_offset,
				dst->metal_buffer.get(), region.dst_offset, region.size);
	}
}

void MDCommandBuffer::copy_texture(RDD::TextureID p_src_texture, RDD::TextureID p_dst_texture, VectorView<RDD::TextureCopyRegion> p_regions) {
	MTL::Texture *src = rid::get<MTL::Texture>(p_src_texture);
	MTL::Texture *dst = rid::get<MTL::Texture>(p_dst_texture);

	MTL::BlitCommandEncoder *enc = _ensure_blit_encoder();
	PixelFormats &pf = device_driver->get_pixel_formats();

	MTL::PixelFormat src_fmt = src->pixelFormat();
	bool src_is_compressed = pf.getFormatType(src_fmt) == MTLFormatType::Compressed;
	MTL::PixelFormat dst_fmt = dst->pixelFormat();
	bool dst_is_compressed = pf.getFormatType(dst_fmt) == MTLFormatType::Compressed;

	// Validate copy.
	if (src->sampleCount() != dst->sampleCount() || pf.getBytesPerBlock(src_fmt) != pf.getBytesPerBlock(dst_fmt)) {
		ERR_FAIL_MSG("Cannot copy between incompatible pixel formats, such as formats of different pixel sizes, or between images with different sample counts.");
	}

	// If source and destination have different formats and at least one is compressed, a temporary buffer is required.
	bool need_tmp_buffer = (src_fmt != dst_fmt) && (src_is_compressed || dst_is_compressed);
	if (need_tmp_buffer) {
		ERR_FAIL_MSG("not implemented: copy with intermediate buffer");
	}

	if (src_fmt != dst_fmt) {
		// Map the source pixel format to the dst through a texture view on the source texture.
		src = src->newTextureView(dst_fmt);
	}

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		RDD::TextureCopyRegion region = p_regions[i];

		MTL::Size extent = MTLSizeFromVector3i(region.size);

		// If copies can be performed using direct texture-texture copying, do so.
		uint32_t src_level = region.src_subresources.mipmap;
		uint32_t src_base_layer = region.src_subresources.base_layer;
		MTL::Size src_extent = mipmapLevelSizeFromTexture(src, src_level);
		uint32_t dst_level = region.dst_subresources.mipmap;
		uint32_t dst_base_layer = region.dst_subresources.base_layer;
		MTL::Size dst_extent = mipmapLevelSizeFromTexture(dst, dst_level);

		// All layers may be copied at once, if the extent completely covers both images.
		if (src_extent == extent && dst_extent == extent) {
			enc->copyFromTexture(src, src_base_layer, src_level,
					dst, dst_base_layer, dst_level,
					region.src_subresources.layer_count, 1);
		} else {
			MTL::Origin src_origin = MTLOriginFromVector3i(region.src_offset);
			MTL::Size src_size = clampMTLSize(extent, src_origin, src_extent);
			uint32_t layer_count = 0;
			if ((src->textureType() == MTL::TextureType3D) != (dst->textureType() == MTL::TextureType3D)) {
				// In the case, the number of layers to copy is in extent.depth. Use that value,
				// then clamp the depth, so we don't try to copy more than Metal will allow.
				layer_count = extent.depth;
				src_size.depth = 1;
			} else {
				layer_count = region.src_subresources.layer_count;
			}
			MTL::Origin dst_origin = MTLOriginFromVector3i(region.dst_offset);

			for (uint32_t layer = 0; layer < layer_count; layer++) {
				// We can copy between a 3D and a 2D image easily. Just copy between
				// one slice of the 2D image and one plane of the 3D image at a time.
				if ((src->textureType() == MTL::TextureType3D) == (dst->textureType() == MTL::TextureType3D)) {
					enc->copyFromTexture(src, src_base_layer + layer, src_level, src_origin, src_size,
							dst, dst_base_layer + layer, dst_level, dst_origin);
				} else if (src->textureType() == MTL::TextureType3D) {
					enc->copyFromTexture(src, src_base_layer, src_level,
							MTL::Origin(src_origin.x, src_origin.y, src_origin.z + layer), src_size,
							dst, dst_base_layer + layer, dst_level, dst_origin);
				} else {
					DEV_ASSERT(dst->textureType() == MTL::TextureType3D);
					enc->copyFromTexture(src, src_base_layer + layer, src_level, src_origin, src_size,
							dst, dst_base_layer, dst_level,
							MTL::Origin(dst_origin.x, dst_origin.y, dst_origin.z + layer));
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
	MTL::Texture *texture = rid::get<MTL::Texture>(p_texture);

	MTL::BlitCommandEncoder *enc = _ensure_blit_encoder();

	PixelFormats &pf = device_driver->get_pixel_formats();
	MTL::PixelFormat mtlPixFmt = texture->pixelFormat();

	MTL::BlitOption options = MTL::BlitOptionNone;
	if (pf.isPVRTCFormat(mtlPixFmt)) {
		options |= MTL::BlitOptionRowLinearPVRTC;
	}

	for (uint32_t i = 0; i < p_regions.size(); i++) {
		RDD::BufferTextureCopyRegion region = p_regions[i];

		uint32_t mip_level = region.texture_subresource.mipmap;
		MTL::Origin txt_origin = MTL::Origin(region.texture_offset.x, region.texture_offset.y, region.texture_offset.z);
		MTL::Size src_extent = mipmapLevelSizeFromTexture(texture, mip_level);
		MTL::Size txt_size = clampMTLSize(MTL::Size(region.texture_region_size.x, region.texture_region_size.y, region.texture_region_size.z),
				txt_origin,
				src_extent);

		uint32_t buffImgWd = region.texture_region_size.x;
		uint32_t buffImgHt = region.texture_region_size.y;

		NS::UInteger bytesPerRow = pf.getBytesPerRow(mtlPixFmt, buffImgWd);
		NS::UInteger bytesPerImg = pf.getBytesPerLayer(mtlPixFmt, bytesPerRow, buffImgHt);

		MTL::BlitOption blit_options = options;

		if (pf.isDepthFormat(mtlPixFmt) && pf.isStencilFormat(mtlPixFmt)) {
			// Don't reduce depths of 32-bit depth/stencil formats.
			if (region.texture_subresource.aspect == RDD::TEXTURE_ASPECT_DEPTH) {
				if (pf.getBytesPerTexel(mtlPixFmt) != 4) {
					bytesPerRow -= buffImgWd;
					bytesPerImg -= buffImgWd * buffImgHt;
				}
				blit_options |= MTL::BlitOptionDepthFromDepthStencil;
			} else if (region.texture_subresource.aspect == RDD::TEXTURE_ASPECT_STENCIL) {
				// The stencil component is always 1 byte per pixel.
				bytesPerRow = buffImgWd;
				bytesPerImg = buffImgWd * buffImgHt;
				blit_options |= MTL::BlitOptionStencilFromDepthStencil;
			}
		}

		if (!isArrayTexture(texture->textureType())) {
			bytesPerImg = 0;
		}

		if (p_source == CopySource::Buffer) {
			enc->copyFromBuffer(buffer->metal_buffer.get(), region.buffer_offset, bytesPerRow, bytesPerImg, txt_size,
					texture, region.texture_subresource.layer, mip_level, txt_origin, blit_options);
		} else {
			enc->copyFromTexture(texture, region.texture_subresource.layer, mip_level, txt_origin, txt_size,
					buffer->metal_buffer.get(), region.buffer_offset, bytesPerRow, bytesPerImg, blit_options);
		}
	}
}

MTL::RenderCommandEncoder *MDCommandBuffer::get_new_render_encoder_with_descriptor(MTL::RenderPassDescriptor *p_desc) {
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

	MTL::RenderCommandEncoder *enc = command_buffer()->renderCommandEncoder(p_desc);
	_encode_barrier(enc);
	return enc;
}

#pragma mark - Render Commands

void MDCommandBuffer::render_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);

	if (uint32_t new_size = p_first_set_index + p_set_count; render.uniform_sets.size() < new_size) {
		uint32_t s = render.uniform_sets.size();
		render.uniform_sets.resize(new_size);
		// Set intermediate values to null.
		std::fill(&render.uniform_sets[s], render.uniform_sets.end().operator->(), nullptr);
	}

	const MDShader *shader = (const MDShader *)p_shader.id;
	DynamicOffsetLayout layout = shader->dynamic_offset_layout;

	// Clear bits for sets being bound, then OR new values.
	for (uint32_t i = 0; i < p_set_count && render.dynamic_offsets != 0; i++) {
		uint32_t set_index = p_first_set_index + i;
		uint32_t count = layout.get_count(set_index);
		if (count > 0) {
			uint32_t shift = layout.get_offset_index_shift(set_index);
			uint32_t mask = ((1u << (count * 4u)) - 1u) << shift;
			render.dynamic_offsets &= ~mask; // Clear this set's bits
		}
	}
	render.dynamic_offsets |= p_dynamic_offsets;

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

	MTL::RenderCommandEncoder *enc = render.encoder.get();

	MDResourceCache &cache = device_driver->get_resource_cache();

	enc->pushDebugGroup(MTLSTR("ClearAttachments"));
	enc->setRenderPipelineState(cache.get_clear_render_pipeline_state(key, nullptr));
	enc->setDepthStencilState(cache.get_depth_stencil_state(
			key.is_depth_enabled(),
			key.is_stencil_enabled()));
	enc->setStencilReferenceValue(stencil_value);
	enc->setCullMode(MTL::CullModeNone);
	enc->setTriangleFillMode(MTL::TriangleFillModeFill);
	enc->setDepthBias(0, 0, 0);
	enc->setViewport(MTL::Viewport{ 0, 0, (double)size.width, (double)size.height, 0.0, 1.0 });
	enc->setScissorRect(MTL::ScissorRect{ 0, 0, (NS::UInteger)size.width, (NS::UInteger)size.height });

	enc->setVertexBytes(clear_colors, sizeof(clear_colors), 0);
	enc->setFragmentBytes(clear_colors, sizeof(clear_colors), 0);
	enc->setVertexBytes(vertices, vertex_count * sizeof(vertices[0]), device_driver->get_metal_buffer_index_for_vertex_attribute_binding(VERT_CONTENT_BUFFER_INDEX));

	enc->drawPrimitives(MTL::PrimitiveTypeTriangle, (NS::UInteger)0, vertex_count);
	enc->popDebugGroup();

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
			render.encoder->setVertexBytes(push_constant_data, push_constant_data_len, push_constant_binding);
			render.encoder->setFragmentBytes(push_constant_data, push_constant_data_len, push_constant_binding);
		}
	}

	MDSubpass const &subpass = render.get_subpass();
	if (subpass.view_count > 1) {
		uint32_t view_range[2] = { 0, subpass.view_count };
		render.encoder->setVertexBytes(view_range, sizeof(view_range), VIEW_MASK_BUFFER_INDEX);
		render.encoder->setFragmentBytes(view_range, sizeof(view_range), VIEW_MASK_BUFFER_INDEX);
	}

	if (render.dirty.has_flag(RenderState::DIRTY_PIPELINE)) {
		render.encoder->setRenderPipelineState(render.pipeline->state.get());
	}

	if (render.dirty.has_flag(RenderState::DIRTY_VIEWPORT)) {
		render.encoder->setViewports(reinterpret_cast<const MTL::Viewport *>(render.viewports.ptr()), render.viewports.size());
	}

	if (render.dirty.has_flag(RenderState::DIRTY_DEPTH)) {
		render.encoder->setDepthStencilState(render.pipeline->depth_stencil.get());
	}

	if (render.dirty.has_flag(RenderState::DIRTY_RASTER)) {
		render.pipeline->raster_state.apply(render.encoder.get());
	}

	if (render.dirty.has_flag(RenderState::DIRTY_SCISSOR) && !render.scissors.is_empty()) {
		size_t len = render.scissors.size();
		MTL::ScissorRect *rects = ALLOCA_ARRAY(MTL::ScissorRect, len);
		for (size_t i = 0; i < len; i++) {
			rects[i] = render.clip_to_render_area(render.scissors[i]);
		}
		render.encoder->setScissorRects(rects, len);
	}

	if (render.dirty.has_flag(RenderState::DIRTY_BLEND) && render.blend_constants.has_value()) {
		render.encoder->setBlendColor(render.blend_constants->r, render.blend_constants->g, render.blend_constants->b, render.blend_constants->a);
	}

	if (render.dirty.has_flag(RenderState::DIRTY_VERTEX)) {
		uint32_t p_binding_count = render.vertex_buffers.size();
		if (p_binding_count > 0) {
			uint32_t first = device_driver->get_metal_buffer_index_for_vertex_attribute_binding(p_binding_count - 1);
			render.encoder->setVertexBuffers(render.vertex_buffers.ptr(), render.vertex_offsets.ptr(), NS::Range(first, p_binding_count));
		}
	}

	if (!use_barriers) {
		render.resource_tracker.encode(render.encoder.get());
	}

	render.dirty.clear();
}

void ResourceTracker::merge_from(const ::ResourceUsageMap &p_from) {
	for (KeyValue<StageResourceUsage, ::ResourceVector> const &keyval : p_from) {
		ResourceVector *resources = _current.getptr(keyval.key);
		if (resources == nullptr) {
			resources = &_current.insert(keyval.key, ResourceVector())->value;
		}
		resources->reserve(resources->size() + keyval.value.size());

		MTL::Resource *const *keyval_ptr = (MTL::Resource *const *)(void *)keyval.value.ptr();

		// Helper to check if a resource needs to be added based on previous usage.
		auto should_add_resource = [this, usage = keyval.key](MTL::Resource *res) -> bool {
			ResourceUsageEntry *existing = _previous.getptr(res);
			if (existing == nullptr) {
				_previous.insert(res, usage);
				return true;
			}
			if (existing->usage != usage) {
				existing->usage |= usage;
				return true;
			}
			return false;
		};

		// 2-way merge of sorted resource lists.
		uint32_t i = 0, j = 0;
		while (i < resources->size() && j < keyval.value.size()) {
			MTL::Resource *current_res = resources->ptr()[i];
			MTL::Resource *new_res = keyval_ptr[j];

			if (current_res < new_res) {
				i++;
			} else if (current_res > new_res) {
				if (should_add_resource(new_res)) {
					resources->insert(i, new_res);
				}
				i++;
				j++;
			} else {
				i++;
				j++;
			}
		}

		// Append any remaining resources from the input.
		for (; j < keyval.value.size(); j++) {
			if (should_add_resource(keyval_ptr[j])) {
				resources->push_back(keyval_ptr[j]);
			}
		}
	}
}

void ResourceTracker::encode(MTL::RenderCommandEncoder *p_enc) {
	for (KeyValue<StageResourceUsage, ResourceVector> const &keyval : _current) {
		if (keyval.value.is_empty()) {
			continue;
		}

		MTL::ResourceUsage vert_usage = (MTL::ResourceUsage)resource_usage_for_stage(keyval.key, RDD::ShaderStage::SHADER_STAGE_VERTEX);
		MTL::ResourceUsage frag_usage = (MTL::ResourceUsage)resource_usage_for_stage(keyval.key, RDD::ShaderStage::SHADER_STAGE_FRAGMENT);
		const MTL::Resource **resources = (const MTL::Resource **)(void *)keyval.value.ptr();
		NS::UInteger count = keyval.value.size();
		if (vert_usage == frag_usage) {
			p_enc->useResources(resources, count, vert_usage, MTL::RenderStageVertex | MTL::RenderStageFragment);
		} else {
			if (vert_usage != 0) {
				p_enc->useResources(resources, count, vert_usage, MTL::RenderStageVertex);
			}
			if (frag_usage != 0) {
				p_enc->useResources(resources, count, frag_usage, MTL::RenderStageFragment);
			}
		}
	}

	// Keep the keys for now and clear the vectors to reduce churn.
	for (KeyValue<StageResourceUsage, ResourceVector> &v : _current) {
		v.value.clear();
	}
}

void ResourceTracker::encode(MTL::ComputeCommandEncoder *p_enc) {
	for (KeyValue<StageResourceUsage, ResourceVector> const &keyval : _current) {
		if (keyval.value.is_empty()) {
			continue;
		}
		MTL::ResourceUsage usage = (MTL::ResourceUsage)resource_usage_for_stage(keyval.key, RDD::ShaderStage::SHADER_STAGE_COMPUTE);
		if (usage != 0) {
			const MTL::Resource **resources = (const MTL::Resource **)(void *)keyval.value.ptr();
			p_enc->useResources(resources, keyval.value.size(), usage);
		}
	}

	// Keep the keys for now and clear the vectors to reduce churn.
	for (KeyValue<StageResourceUsage, ResourceVector> &v : _current) {
		v.value.clear();
	}
}

void ResourceTracker::reset() {
	// Keep the keys for now, as they are likely to be used repeatedly.
	for (KeyValue<MTL::Resource *, ResourceUsageEntry> &v : _previous) {
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
	for (MTL::Resource *res : _scratch) {
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
			DirectEncoder de(render.encoder.get(), binding_cache, DirectEncoder::RENDER);
			_bind_uniforms_direct(set, shader, de, index, dynamic_offsets);
		}
	}
}

void MDCommandBuffer::render_begin_pass(RDD::RenderPassID p_render_pass, RDD::FramebufferID p_frameBuffer, RDD::CommandBufferType p_cmd_buffer_type, const Rect2i &p_rect, VectorView<RDD::RenderPassClearValue> p_clear_values) {
	DEV_ASSERT(command_buffer() != nullptr);
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

void MDCommandBuffer::render_next_subpass() {
	DEV_ASSERT(command_buffer() != nullptr);

	if (render.current_subpass == UINT32_MAX) {
		render.current_subpass = 0;
	} else {
		_end_render_pass();
		render.current_subpass++;
	}

	MDFrameBuffer const &fb = *render.frameBuffer;
	MDRenderPass const &pass = *render.pass;
	MDSubpass const &subpass = render.get_subpass();

	NS::SharedPtr<MTL::RenderPassDescriptor> desc = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());

	if (subpass.view_count > 1) {
		desc->setRenderTargetArrayLength(subpass.view_count);
	}

	PixelFormats &pf = device_driver->get_pixel_formats();

	uint32_t attachmentCount = 0;
	for (uint32_t i = 0; i < subpass.color_references.size(); i++) {
		uint32_t idx = subpass.color_references[i].attachment;
		if (idx == RDD::AttachmentReference::UNUSED) {
			continue;
		}

		attachmentCount += 1;
		MTL::RenderPassColorAttachmentDescriptor *ca = desc->colorAttachments()->object(i);

		uint32_t resolveIdx = subpass.resolve_references.is_empty() ? RDD::AttachmentReference::UNUSED : subpass.resolve_references[i].attachment;
		bool has_resolve = resolveIdx != RDD::AttachmentReference::UNUSED;
		bool can_resolve = true;
		if (resolveIdx != RDD::AttachmentReference::UNUSED) {
			MTL::Texture *resolve_tex = fb.get_texture(resolveIdx);
			can_resolve = flags::all(pf.getCapabilities(resolve_tex->pixelFormat()), kMTLFmtCapsResolve);
			if (can_resolve) {
				ca->setResolveTexture(resolve_tex);
			} else {
				CRASH_NOW_MSG("unimplemented: using a texture format that is not supported for resolve");
			}
		}

		MDAttachment const &attachment = pass.attachments[idx];

		MTL::Texture *tex = fb.get_texture(idx);
		ERR_FAIL_NULL_MSG(tex, "Frame buffer color texture is null.");

		if ((attachment.type & MDAttachmentType::Color)) {
			if (attachment.configureDescriptor(ca, pf, subpass, tex, render.is_rendering_entire_area, has_resolve, can_resolve, false)) {
				Color clearColor = render.clear_values[idx].color;
				ca->setClearColor(MTL::ClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a));
			}
		}
	}

	if (subpass.depth_stencil_reference.attachment != RDD::AttachmentReference::UNUSED) {
		attachmentCount += 1;
		uint32_t idx = subpass.depth_stencil_reference.attachment;
		MDAttachment const &attachment = pass.attachments[idx];
		MTL::Texture *tex = fb.get_texture(idx);
		ERR_FAIL_NULL_MSG(tex, "Frame buffer depth / stencil texture is null.");
		if (attachment.type & MDAttachmentType::Depth) {
			MTL::RenderPassDepthAttachmentDescriptor *da = desc->depthAttachment();
			if (attachment.configureDescriptor(da, pf, subpass, tex, render.is_rendering_entire_area, false, false, false)) {
				da->setClearDepth(render.clear_values[idx].depth);
			}
		}

		if (attachment.type & MDAttachmentType::Stencil) {
			MTL::RenderPassStencilAttachmentDescriptor *sa = desc->stencilAttachment();
			if (attachment.configureDescriptor(sa, pf, subpass, tex, render.is_rendering_entire_area, false, false, true)) {
				sa->setClearStencil(render.clear_values[idx].stencil);
			}
		}
	}

	desc->setRenderTargetWidth(MAX((NS::UInteger)MIN(render.render_area.position.x + render.render_area.size.width, fb.size.width), 1u));
	desc->setRenderTargetHeight(MAX((NS::UInteger)MIN(render.render_area.position.y + render.render_area.size.height, fb.size.height), 1u));

	if (attachmentCount == 0) {
		// If there are no attachments, delay the creation of the encoder,
		// so we can use a matching sample count for the pipeline, by setting
		// the defaultRasterSampleCount from the pipeline's sample count.
		render.desc = desc;
	} else {
		render.encoder = NS::RetainPtr(command_buffer()->renderCommandEncoder(desc.get()));
		_encode_barrier(render.encoder.get());

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

	MTL::RenderCommandEncoder *enc = render.encoder.get();
	enc->drawPrimitives(render.pipeline->raster_state.render_primitive, p_base_vertex, p_vertex_count, p_instance_count, p_first_instance);
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

		NS::UInteger dynamic_offset = 0;
		if (buf_info->is_dynamic()) {
			const MetalBufferDynamicInfo *dyn_buf = (const MetalBufferDynamicInfo *)buf_info;
			uint64_t frame_idx = p_dynamic_offsets & 0x3;
			p_dynamic_offsets >>= 2;
			dynamic_offset = frame_idx * dyn_buf->size_bytes;
		}
		if (render.vertex_buffers[i] != buf_info->metal_buffer.get()) {
			render.vertex_buffers[i] = buf_info->metal_buffer.get();
			same = false;
		}

		render.vertex_offsets[i] = dynamic_offset + p_offsets[p_binding_count - i - 1];
	}

	if (render.encoder.get() != nullptr) {
		uint32_t first = device_driver->get_metal_buffer_index_for_vertex_attribute_binding(p_binding_count - 1);
		if (same) {
			NS::UInteger *offset_ptr = render.vertex_offsets.ptr();
			for (uint32_t i = first; i < first + p_binding_count; i++) {
				render.encoder->setVertexBufferOffset(*offset_ptr, i);
				offset_ptr++;
			}
		} else {
			render.encoder->setVertexBuffers(render.vertex_buffers.ptr(), render.vertex_offsets.ptr(), NS::Range(first, p_binding_count));
		}
		render.dirty.clear_flag(RenderState::DIRTY_VERTEX);
	} else {
		render.dirty.set_flag(RenderState::DIRTY_VERTEX);
	}
}

void MDCommandBuffer::render_bind_index_buffer(RDD::BufferID p_buffer, RDD::IndexBufferFormat p_format, uint64_t p_offset) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);

	const RenderingDeviceDriverMetal::BufferInfo *buffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_buffer.id;

	render.index_buffer = buffer->metal_buffer.get();
	render.index_type = p_format == RDD::IndexBufferFormat::INDEX_BUFFER_FORMAT_UINT16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;
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

	MTL::RenderCommandEncoder *enc = render.encoder.get();

	uint32_t index_offset = render.index_offset;
	index_offset += p_first_index * (render.index_type == MTL::IndexTypeUInt16 ? sizeof(uint16_t) : sizeof(uint32_t));

	enc->drawIndexedPrimitives(render.pipeline->raster_state.render_primitive, p_index_count, render.index_type, render.index_buffer, index_offset, p_instance_count, p_vertex_offset, p_first_instance);
}

void MDCommandBuffer::render_draw_indexed_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	ERR_FAIL_NULL_MSG(render.pipeline, "No pipeline set for render command buffer.");

	_render_set_dirty_state();

	MTL::RenderCommandEncoder *enc = render.encoder.get();

	const RenderingDeviceDriverMetal::BufferInfo *indirect_buffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_indirect_buffer.id;
	NS::UInteger indirect_offset = p_offset;

	for (uint32_t i = 0; i < p_draw_count; i++) {
		enc->drawIndexedPrimitives(render.pipeline->raster_state.render_primitive, render.index_type, render.index_buffer, 0, indirect_buffer->metal_buffer.get(), indirect_offset);
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

	MTL::RenderCommandEncoder *enc = render.encoder.get();

	const RenderingDeviceDriverMetal::BufferInfo *indirect_buffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_indirect_buffer.id;
	NS::UInteger indirect_offset = p_offset;

	for (uint32_t i = 0; i < p_draw_count; i++) {
		enc->drawPrimitives(render.pipeline->raster_state.render_primitive, indirect_buffer->metal_buffer.get(), indirect_offset);
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
	pass = nullptr;
	frameBuffer = nullptr;
	pipeline = nullptr;
	current_subpass = UINT32_MAX;
	render_area = {};
	is_rendering_entire_area = false;
	desc.reset();
	encoder.reset();
	index_buffer = nullptr;
	index_type = MTL::IndexTypeUInt16;
	dirty = DIRTY_NONE;
	uniform_sets.clear();
	dynamic_offsets = 0;
	uniform_set_mask = 0;
	clear_values.clear();
	viewports.clear();
	scissors.clear();
	blend_constants.reset();
	bzero(vertex_buffers.ptr(), sizeof(MTL::Buffer *) * vertex_buffers.size());
	vertex_buffers.clear();
	bzero(vertex_offsets.ptr(), sizeof(NS::UInteger) * vertex_offsets.size());
	vertex_offsets.clear();
	resource_tracker.reset();
}

void MDCommandBuffer::RenderState::end_encoding() {
	if (encoder.get() == nullptr) {
		return;
	}

	encoder->endEncoding();
	encoder.reset();
}

#pragma mark - ComputeState

void MDCommandBuffer::ComputeState::end_encoding() {
	if (encoder.get() == nullptr) {
		return;
	}

	encoder->endEncoding();
	encoder.reset();
}

#pragma mark - Compute

void MDCommandBuffer::_compute_set_dirty_state() {
	if (compute.dirty.has_flag(ComputeState::DIRTY_PIPELINE)) {
		compute.encoder = NS::RetainPtr(command_buffer()->computeCommandEncoder(MTL::DispatchTypeConcurrent));
		_encode_barrier(compute.encoder.get());
		compute.encoder->setComputePipelineState(compute.pipeline->state.get());
	}

	_compute_bind_uniform_sets();

	if (compute.dirty.has_flag(ComputeState::DIRTY_PUSH)) {
		if (push_constant_binding != UINT32_MAX) {
			compute.encoder->setBytes(push_constant_data, push_constant_data_len, push_constant_binding);
		}
	}

	if (!use_barriers) {
		compute.resource_tracker.encode(compute.encoder.get());
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
			DirectEncoder de(compute.encoder.get(), binding_cache, DirectEncoder::COMPUTE);
			_bind_uniforms_direct(set, shader, de, index, dynamic_offsets);
		}
	}
}

void MDCommandBuffer::ComputeState::reset() {
	pipeline = nullptr;
	encoder.reset();
	dirty = DIRTY_NONE;
	uniform_sets.clear();
	dynamic_offsets = 0;
	uniform_set_mask = 0;
	resource_tracker.reset();
}

void MDCommandBuffer::compute_bind_uniform_sets(VectorView<RDD::UniformSetID> p_uniform_sets, RDD::ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(type == MDCommandBufferStateType::Compute);

	if (uint32_t new_size = p_first_set_index + p_set_count; compute.uniform_sets.size() < new_size) {
		uint32_t s = compute.uniform_sets.size();
		compute.uniform_sets.resize(new_size);
		// Set intermediate values to null.
		std::fill(&compute.uniform_sets[s], compute.uniform_sets.end().operator->(), nullptr);
	}

	const MDShader *shader = (const MDShader *)p_shader.id;
	DynamicOffsetLayout layout = shader->dynamic_offset_layout;

	// Clear bits for sets being bound, then OR new values.
	for (uint32_t i = 0; i < p_set_count && compute.dynamic_offsets != 0; i++) {
		uint32_t set_index = p_first_set_index + i;
		uint32_t count = layout.get_count(set_index);
		if (count > 0) {
			uint32_t shift = layout.get_offset_index_shift(set_index);
			uint32_t mask = ((1u << (count * 4u)) - 1u) << shift;
			compute.dynamic_offsets &= ~mask; // Clear this set's bits
		}
	}
	compute.dynamic_offsets |= p_dynamic_offsets;

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

	MTL::Size size = MTL::Size(p_x_groups, p_y_groups, p_z_groups);

	MTL::ComputeCommandEncoder *enc = compute.encoder.get();
	enc->dispatchThreadgroups(size, compute.pipeline->compute_state.local);
}

void MDCommandBuffer::compute_dispatch_indirect(RDD::BufferID p_indirect_buffer, uint64_t p_offset) {
	DEV_ASSERT(type == MDCommandBufferStateType::Compute);

	_compute_set_dirty_state();

	const RenderingDeviceDriverMetal::BufferInfo *indirectBuffer = (const RenderingDeviceDriverMetal::BufferInfo *)p_indirect_buffer.id;

	MTL::ComputeCommandEncoder *enc = compute.encoder.get();
	enc->dispatchThreadgroups(indirectBuffer->metal_buffer.get(), p_offset, compute.pipeline->compute_state.local);
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

	blit.encoder->endEncoding();
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

void DirectEncoder::set(MTL::Texture **p_textures, NS::Range p_range) {
	if (cache.update(p_range, p_textures)) {
		switch (mode) {
			case RENDER: {
				MTL::RenderCommandEncoder *enc = static_cast<MTL::RenderCommandEncoder *>(encoder);
				enc->setVertexTextures(p_textures, p_range);
				enc->setFragmentTextures(p_textures, p_range);
			} break;
			case COMPUTE: {
				MTL::ComputeCommandEncoder *enc = static_cast<MTL::ComputeCommandEncoder *>(encoder);
				enc->setTextures(p_textures, p_range);
			} break;
		}
	}
}

void DirectEncoder::set(MTL::Buffer **p_buffers, const NS::UInteger *p_offsets, NS::Range p_range) {
	if (cache.update(p_range, p_buffers, p_offsets)) {
		switch (mode) {
			case RENDER: {
				MTL::RenderCommandEncoder *enc = static_cast<MTL::RenderCommandEncoder *>(encoder);
				enc->setVertexBuffers(p_buffers, p_offsets, p_range);
				enc->setFragmentBuffers(p_buffers, p_offsets, p_range);
			} break;
			case COMPUTE: {
				MTL::ComputeCommandEncoder *enc = static_cast<MTL::ComputeCommandEncoder *>(encoder);
				enc->setBuffers(p_buffers, p_offsets, p_range);
			} break;
		}
	}
}

void DirectEncoder::set(MTL::Buffer *p_buffer, NS::UInteger p_offset, uint32_t p_index) {
	if (cache.update(p_buffer, p_offset, p_index)) {
		switch (mode) {
			case RENDER: {
				MTL::RenderCommandEncoder *enc = static_cast<MTL::RenderCommandEncoder *>(encoder);
				enc->setVertexBuffer(p_buffer, p_offset, p_index);
				enc->setFragmentBuffer(p_buffer, p_offset, p_index);
			} break;
			case COMPUTE: {
				MTL::ComputeCommandEncoder *enc = static_cast<MTL::ComputeCommandEncoder *>(encoder);
				enc->setBuffer(p_buffer, p_offset, p_index);
			} break;
		}
	}
}

void DirectEncoder::set(MTL::SamplerState **p_samplers, NS::Range p_range) {
	if (cache.update(p_range, p_samplers)) {
		switch (mode) {
			case RENDER: {
				MTL::RenderCommandEncoder *enc = static_cast<MTL::RenderCommandEncoder *>(encoder);
				enc->setVertexSamplerStates(p_samplers, p_range);
				enc->setFragmentSamplerStates(p_samplers, p_range);
			} break;
			case COMPUTE: {
				MTL::ComputeCommandEncoder *enc = static_cast<MTL::ComputeCommandEncoder *>(encoder);
				enc->setSamplerStates(p_samplers, p_range);
			} break;
		}
	}
}

GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability-new")

void MDCommandBuffer::_bind_uniforms_argument_buffers(MDUniformSet *p_set, MDShader *p_shader, uint32_t p_set_index, uint32_t p_dynamic_offsets) {
	DEV_ASSERT(p_shader->uses_argument_buffers);
	DEV_ASSERT(render.encoder.get() != nullptr);

	MTL::RenderCommandEncoder *enc = render.encoder.get();
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
			uint64_t gpu_address = buf_info->metal_buffer.get()->gpuAddress() + frame_idx * buf_info->size_bytes;
			*(uint64_t *)(ptr + idx.buffer) = gpu_address;
		}

		enc->setVertexBuffer(alloc.buffer, alloc.offset, p_set_index);
		enc->setFragmentBuffer(alloc.buffer, alloc.offset, p_set_index);
	} else {
		enc->setVertexBuffer(p_set->arg_buffer.get(), 0, p_set_index);
		enc->setFragmentBuffer(p_set->arg_buffer.get(), 0, p_set_index);
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
				MTL::SamplerState **objects = ALLOCA_ARRAY(MTL::SamplerState *, count);
				for (size_t j = 0; j < count; j += 1) {
					objects[j] = rid::get<MTL::SamplerState>(uniform.ids[j]);
				}
				NS::Range sampler_range = { indexes.sampler, count };
				p_enc.set(objects, sampler_range);
			} break;
			case RDD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE: {
				size_t count = uniform.ids.size() / 2;
				MTL::Texture **textures = ALLOCA_ARRAY(MTL::Texture *, count);
				MTL::SamplerState **samplers = ALLOCA_ARRAY(MTL::SamplerState *, count);
				for (uint32_t j = 0; j < count; j += 1) {
					samplers[j] = rid::get<MTL::SamplerState>(uniform.ids[j * 2 + 0]);
					textures[j] = rid::get<MTL::Texture>(uniform.ids[j * 2 + 1]);
				}
				NS::Range sampler_range = { indexes.sampler, count };
				NS::Range texture_range = { indexes.texture, count };
				p_enc.set(samplers, sampler_range);
				p_enc.set(textures, texture_range);
			} break;
			case RDD::UNIFORM_TYPE_TEXTURE: {
				size_t count = uniform.ids.size();
				MTL::Texture **objects = ALLOCA_ARRAY(MTL::Texture *, count);
				for (size_t j = 0; j < count; j += 1) {
					objects[j] = rid::get<MTL::Texture>(uniform.ids[j]);
				}
				NS::Range texture_range = { indexes.texture, count };
				p_enc.set(objects, texture_range);
			} break;
			case RDD::UNIFORM_TYPE_IMAGE: {
				size_t count = uniform.ids.size();
				MTL::Texture **objects = ALLOCA_ARRAY(MTL::Texture *, count);
				for (size_t j = 0; j < count; j += 1) {
					objects[j] = rid::get<MTL::Texture>(uniform.ids[j]);
				}
				NS::Range texture_range = { indexes.texture, count };
				p_enc.set(objects, texture_range);

				if (indexes.buffer != UINT32_MAX) {
					// Emulated atomic image access.
					MTL::Buffer **bufs = ALLOCA_ARRAY(MTL::Buffer *, count);
					for (size_t j = 0; j < count; j += 1) {
						MTL::Texture *obj = objects[j];
						MTL::Texture *tex = obj->parentTexture() ? obj->parentTexture() : obj;
						bufs[j] = tex->buffer();
					}
					NS::UInteger *offs = ALLOCA_ARRAY(NS::UInteger, count);
					bzero(offs, sizeof(NS::UInteger) * count);
					NS::Range buffer_range = { indexes.buffer, count };
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
				p_enc.set(buf_info->metal_buffer.get(), 0, indexes.buffer);
			} break;
			case RDD::UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC:
			case RDD::UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC: {
				const MetalBufferDynamicInfo *buf_info = (const MetalBufferDynamicInfo *)uniform.ids[0].id;
				p_enc.set(buf_info->metal_buffer.get(), frame_idx * buf_info->size_bytes, indexes.buffer);
			} break;
			case RDD::UNIFORM_TYPE_INPUT_ATTACHMENT: {
				size_t count = uniform.ids.size();
				MTL::Texture **objects = ALLOCA_ARRAY(MTL::Texture *, count);
				for (size_t j = 0; j < count; j += 1) {
					objects[j] = rid::get<MTL::Texture>(uniform.ids[j]);
				}
				NS::Range texture_range = { indexes.texture, count };
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
	DEV_ASSERT(compute.encoder.get() != nullptr);

	MTL::ComputeCommandEncoder *enc = compute.encoder.get();
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
			uint64_t gpu_address = buf_info->metal_buffer.get()->gpuAddress() + frame_idx * buf_info->size_bytes;
			*(uint64_t *)(ptr + idx.buffer) = gpu_address;
		}

		enc->setBuffer(alloc.buffer, alloc.offset, p_set_index);
	} else {
		enc->setBuffer(p_set->arg_buffer.get(), 0, p_set_index);
	}
}

GODOT_CLANG_WARNING_POP
