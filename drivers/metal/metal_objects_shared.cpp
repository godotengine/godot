/**************************************************************************/
/*  metal_objects_shared.cpp                                              */
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

#include "metal_objects_shared.h"

#include "rendering_device_driver_metal.h"

#include <os/signpost.h>
#include <simd/simd.h>
#include <string>

#pragma mark - Resource Factory

NS::SharedPtr<MTL::Function> MDResourceFactory::new_func(NS::String *p_source, NS::String *p_name, NS::Error **p_error) {
	NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
	NS::SharedPtr<MTL::CompileOptions> options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
	NS::Error *err = nullptr;
	NS::SharedPtr<MTL::Library> mtlLib = NS::TransferPtr(device->newLibrary(p_source, options.get(), &err));
	if (err) {
		if (p_error != nullptr) {
			*p_error = err;
		}
	}
	return NS::TransferPtr(mtlLib->newFunction(p_name));
}

NS::SharedPtr<MTL::Function> MDResourceFactory::new_clear_vert_func(ClearAttKey &p_key) {
	NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
	char msl[1024];
	snprintf(msl, sizeof(msl), R"(
#include <metal_stdlib>
using namespace metal;

typedef struct {
    float4 a_position [[attribute(0)]];
} AttributesPos;

typedef struct {
    float4 colors[9];
} ClearColorsIn;

typedef struct {
    float4 v_position [[position]];
    uint layer%s;
} VaryingsPos;

vertex VaryingsPos vertClear(AttributesPos attributes [[stage_in]], constant ClearColorsIn& ccIn [[buffer(0)]]) {
    VaryingsPos varyings;
    varyings.v_position = float4(attributes.a_position.x, -attributes.a_position.y, ccIn.colors[%d].r, 1.0);
    varyings.layer = uint(attributes.a_position.w);
    return varyings;
}
)",
			p_key.is_layered_rendering_enabled() ? " [[render_target_array_index]]" : "", ClearAttKey::DEPTH_INDEX);

	return new_func(NS::String::string(msl, NS::UTF8StringEncoding), MTLSTR("vertClear"), nullptr);
}

NS::SharedPtr<MTL::Function> MDResourceFactory::new_clear_frag_func(ClearAttKey &p_key) {
	NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
	std::string msl;
	msl.reserve(2048);

	msl += R"(
#include <metal_stdlib>
using namespace metal;

typedef struct {
    float4 v_position [[position]];
} VaryingsPos;

typedef struct {
    float4 colors[9];
} ClearColorsIn;

typedef struct {
)";

	char line[128];
	for (uint32_t caIdx = 0; caIdx < ClearAttKey::COLOR_COUNT; caIdx++) {
		if (p_key.is_enabled(caIdx)) {
			const char *typeStr = get_format_type_string((MTL::PixelFormat)p_key.pixel_formats[caIdx]);
			snprintf(line, sizeof(line), "    %s4 color%u [[color(%u)]];\n", typeStr, caIdx, caIdx);
			msl += line;
		}
	}
	msl += R"(} ClearColorsOut;

fragment ClearColorsOut fragClear(VaryingsPos varyings [[stage_in]], constant ClearColorsIn& ccIn [[buffer(0)]]) {

    ClearColorsOut ccOut;
)";
	for (uint32_t caIdx = 0; caIdx < ClearAttKey::COLOR_COUNT; caIdx++) {
		if (p_key.is_enabled(caIdx)) {
			const char *typeStr = get_format_type_string((MTL::PixelFormat)p_key.pixel_formats[caIdx]);
			snprintf(line, sizeof(line), "    ccOut.color%u = %s4(ccIn.colors[%u]);\n", caIdx, typeStr, caIdx);
			msl += line;
		}
	}
	msl += R"(    return ccOut;
})";

	return new_func(NS::String::string(msl.c_str(), NS::UTF8StringEncoding), MTLSTR("fragClear"), nullptr);
}

const char *MDResourceFactory::get_format_type_string(MTL::PixelFormat p_fmt) const {
	switch (pixel_formats.getFormatType(p_fmt)) {
		case MTLFormatType::ColorInt8:
		case MTLFormatType::ColorInt16:
			return "short";
		case MTLFormatType::ColorUInt8:
		case MTLFormatType::ColorUInt16:
			return "ushort";
		case MTLFormatType::ColorInt32:
			return "int";
		case MTLFormatType::ColorUInt32:
			return "uint";
		case MTLFormatType::ColorHalf:
			return "half";
		case MTLFormatType::ColorFloat:
		case MTLFormatType::DepthStencil:
		case MTLFormatType::Compressed:
			return "float";
		case MTLFormatType::None:
		default:
			return "unexpected_MTLPixelFormatInvalid";
	}
}

NS::SharedPtr<MTL::DepthStencilState> MDResourceFactory::new_depth_stencil_state(bool p_use_depth, bool p_use_stencil) {
	NS::SharedPtr<MTL::DepthStencilDescriptor> dsDesc = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
	dsDesc->setDepthCompareFunction(MTL::CompareFunctionAlways);
	dsDesc->setDepthWriteEnabled(p_use_depth);

	if (p_use_stencil) {
		NS::SharedPtr<MTL::StencilDescriptor> sDesc = NS::TransferPtr(MTL::StencilDescriptor::alloc()->init());
		sDesc->setStencilCompareFunction(MTL::CompareFunctionAlways);
		sDesc->setStencilFailureOperation(MTL::StencilOperationReplace);
		sDesc->setDepthFailureOperation(MTL::StencilOperationReplace);
		sDesc->setDepthStencilPassOperation(MTL::StencilOperationReplace);

		dsDesc->setFrontFaceStencil(sDesc.get());
		dsDesc->setBackFaceStencil(sDesc.get());
	} else {
		dsDesc->setFrontFaceStencil(nullptr);
		dsDesc->setBackFaceStencil(nullptr);
	}

	return NS::TransferPtr(device->newDepthStencilState(dsDesc.get()));
}

NS::SharedPtr<MTL::RenderPipelineState> MDResourceFactory::new_clear_pipeline_state(ClearAttKey &p_key, NS::Error **p_error) {
	NS::SharedPtr<MTL::Function> vtxFunc = new_clear_vert_func(p_key);
	NS::SharedPtr<MTL::Function> fragFunc = new_clear_frag_func(p_key);
	NS::SharedPtr<MTL::RenderPipelineDescriptor> plDesc = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
	plDesc->setLabel(MTLSTR("ClearRenderAttachments"));
	plDesc->setVertexFunction(vtxFunc.get());
	plDesc->setFragmentFunction(fragFunc.get());
	plDesc->setRasterSampleCount(p_key.sample_count);
	plDesc->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);

	for (uint32_t caIdx = 0; caIdx < ClearAttKey::COLOR_COUNT; caIdx++) {
		MTL::RenderPipelineColorAttachmentDescriptor *colorDesc = plDesc->colorAttachments()->object(caIdx);
		colorDesc->setPixelFormat((MTL::PixelFormat)p_key.pixel_formats[caIdx]);
		colorDesc->setWriteMask(p_key.is_enabled(caIdx) ? MTL::ColorWriteMaskAll : MTL::ColorWriteMaskNone);
	}

	MTL::PixelFormat mtlDepthFormat = (MTL::PixelFormat)p_key.depth_format();
	if (pixel_formats.isDepthFormat(mtlDepthFormat)) {
		plDesc->setDepthAttachmentPixelFormat(mtlDepthFormat);
	}

	MTL::PixelFormat mtlStencilFormat = (MTL::PixelFormat)p_key.stencil_format();
	if (pixel_formats.isStencilFormat(mtlStencilFormat)) {
		plDesc->setStencilAttachmentPixelFormat(mtlStencilFormat);
	}

	MTL::VertexDescriptor *vtxDesc = plDesc->vertexDescriptor();

	// Vertex attribute descriptors.
	NS::UInteger vtxBuffIdx = get_vertex_buffer_index(VERT_CONTENT_BUFFER_INDEX);
	NS::UInteger vtxStride = 0;

	// Vertex location.
	MTL::VertexAttributeDescriptor *vaDesc = vtxDesc->attributes()->object(0);
	vaDesc->setFormat(MTL::VertexFormatFloat4);
	vaDesc->setBufferIndex(vtxBuffIdx);
	vaDesc->setOffset(vtxStride);
	vtxStride += sizeof(simd::float4);

	// Vertex attribute buffer.
	MTL::VertexBufferLayoutDescriptor *vbDesc = vtxDesc->layouts()->object(vtxBuffIdx);
	vbDesc->setStepFunction(MTL::VertexStepFunctionPerVertex);
	vbDesc->setStepRate(1);
	vbDesc->setStride(vtxStride);

	NS::Error *err = nullptr;
	NS::SharedPtr<MTL::RenderPipelineState> state = NS::TransferPtr(device->newRenderPipelineState(plDesc.get(), &err));
	if (p_error != nullptr) {
		*p_error = err;
	}
	return state;
}

NS::SharedPtr<MTL::RenderPipelineState> MDResourceFactory::new_empty_draw_pipeline_state(ClearAttKey &p_key, NS::Error **p_error) {
	DEV_ASSERT(!p_key.is_layered_rendering_enabled());
	DEV_ASSERT(p_key.is_enabled(0));
	DEV_ASSERT(!p_key.is_depth_enabled());
	DEV_ASSERT(!p_key.is_stencil_enabled());

	NS::SharedPtr<NS::AutoreleasePool> pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
	static const char *msl = R"(#include <metal_stdlib>
using namespace metal;

struct FullscreenNoopOut {
    float4 position [[position]];
};

vertex FullscreenNoopOut fullscreenNoopVert(uint vid [[vertex_id]]) {
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 pos = positions[vid];

    FullscreenNoopOut out;
    out.position = float4(pos, 0.0, 1.0);
    return out;
}

fragment void fullscreenNoopFrag(float4 gl_FragCoord [[position]]) {
}
)";

	NS::Error *err = nullptr;
	NS::SharedPtr<MTL::CompileOptions> options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
	NS::SharedPtr<MTL::Library> mtlLib = NS::TransferPtr(device->newLibrary(NS::String::string(msl, NS::UTF8StringEncoding), options.get(), &err));
	if (err && p_error != nullptr) {
		*p_error = err;
	}

	if (mtlLib.get() == nullptr) {
		return {};
	}

	NS::SharedPtr<MTL::Function> vtxFunc = NS::TransferPtr(mtlLib->newFunction(MTLSTR("fullscreenNoopVert")));
	NS::SharedPtr<MTL::Function> fragFunc = NS::TransferPtr(mtlLib->newFunction(MTLSTR("fullscreenNoopFrag")));

	NS::SharedPtr<MTL::RenderPipelineDescriptor> plDesc = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
	plDesc->setLabel(MTLSTR("EmptyDrawFullscreenTriangle"));
	plDesc->setVertexFunction(vtxFunc.get());
	plDesc->setFragmentFunction(fragFunc.get());
	plDesc->setRasterSampleCount(p_key.sample_count ? p_key.sample_count : 1);
	plDesc->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);

	MTL::RenderPipelineColorAttachmentDescriptor *colorDesc = plDesc->colorAttachments()->object(0);
	colorDesc->setPixelFormat((MTL::PixelFormat)p_key.pixel_formats[0]);
	colorDesc->setWriteMask(MTL::ColorWriteMaskNone);

	err = nullptr;
	NS::SharedPtr<MTL::RenderPipelineState> state = NS::TransferPtr(device->newRenderPipelineState(plDesc.get(), &err));
	if (p_error != nullptr && err != nullptr) {
		*p_error = err;
	}
	return state;
}

#pragma mark - Resource Cache

MTL::RenderPipelineState *MDResourceCache::get_clear_render_pipeline_state(ClearAttKey &p_key, NS::Error **p_error) {
	HashMap::ConstIterator it = clear_states.find(p_key);
	if (it != clear_states.end()) {
		return it->value.get();
	}

	NS::SharedPtr<MTL::RenderPipelineState> state = resource_factory->new_clear_pipeline_state(p_key, p_error);
	MTL::RenderPipelineState *result = state.get();
	clear_states[p_key] = std::move(state);
	return result;
}

MTL::RenderPipelineState *MDResourceCache::get_empty_draw_pipeline_state(ClearAttKey &p_key, NS::Error **p_error) {
	HashMap::ConstIterator it = empty_draw_states.find(p_key);
	if (it != empty_draw_states.end()) {
		return it->value.get();
	}

	NS::SharedPtr<MTL::RenderPipelineState> state = resource_factory->new_empty_draw_pipeline_state(p_key, p_error);
	MTL::RenderPipelineState *result = state.get();
	empty_draw_states[p_key] = std::move(state);
	return result;
}

MTL::DepthStencilState *MDResourceCache::get_depth_stencil_state(bool p_use_depth, bool p_use_stencil) {
	if (p_use_depth && p_use_stencil) {
		if (!clear_depth_stencil_state.all) {
			clear_depth_stencil_state.all = resource_factory->new_depth_stencil_state(true, true);
		}
		return clear_depth_stencil_state.all.get();
	} else if (p_use_depth) {
		if (!clear_depth_stencil_state.depth_only) {
			clear_depth_stencil_state.depth_only = resource_factory->new_depth_stencil_state(true, false);
		}
		return clear_depth_stencil_state.depth_only.get();
	} else if (p_use_stencil) {
		if (!clear_depth_stencil_state.stencil_only) {
			clear_depth_stencil_state.stencil_only = resource_factory->new_depth_stencil_state(false, true);
		}
		return clear_depth_stencil_state.stencil_only.get();
	} else {
		if (!clear_depth_stencil_state.none) {
			clear_depth_stencil_state.none = resource_factory->new_depth_stencil_state(false, false);
		}
		return clear_depth_stencil_state.none.get();
	}
}

#pragma mark - Render Pass Types

MTLFmtCaps MDSubpass::getRequiredFmtCapsForAttachmentAt(uint32_t p_index) const {
	MTLFmtCaps caps = kMTLFmtCapsNone;

	for (RDD::AttachmentReference const &ar : input_references) {
		if (ar.attachment == p_index) {
			flags::set(caps, kMTLFmtCapsRead);
			break;
		}
	}

	for (RDD::AttachmentReference const &ar : color_references) {
		if (ar.attachment == p_index) {
			flags::set(caps, kMTLFmtCapsColorAtt);
			break;
		}
	}

	for (RDD::AttachmentReference const &ar : resolve_references) {
		if (ar.attachment == p_index) {
			flags::set(caps, kMTLFmtCapsResolve);
			break;
		}
	}

	if (depth_stencil_reference.attachment == p_index) {
		flags::set(caps, kMTLFmtCapsDSAtt);
	}

	return caps;
}

void MDAttachment::linkToSubpass(const MDRenderPass &p_pass) {
	firstUseSubpassIndex = UINT32_MAX;
	lastUseSubpassIndex = 0;

	for (MDSubpass const &subpass : p_pass.subpasses) {
		MTLFmtCaps reqCaps = subpass.getRequiredFmtCapsForAttachmentAt(index);
		if (reqCaps) {
			firstUseSubpassIndex = MIN(subpass.subpass_index, firstUseSubpassIndex);
			lastUseSubpassIndex = MAX(subpass.subpass_index, lastUseSubpassIndex);
		}
	}
}

MTL::StoreAction MDAttachment::getMTLStoreAction(MDSubpass const &p_subpass,
		bool p_is_rendering_entire_area,
		bool p_has_resolve,
		bool p_can_resolve,
		bool p_is_stencil) const {
	if (!p_is_rendering_entire_area || !isLastUseOf(p_subpass)) {
		return p_has_resolve && p_can_resolve ? MTL::StoreActionStoreAndMultisampleResolve : MTL::StoreActionStore;
	}

	switch (p_is_stencil ? stencilStoreAction : storeAction) {
		case MTL::StoreActionStore:
			return p_has_resolve && p_can_resolve ? MTL::StoreActionStoreAndMultisampleResolve : MTL::StoreActionStore;
		case MTL::StoreActionDontCare:
			return p_has_resolve ? (p_can_resolve ? MTL::StoreActionMultisampleResolve : MTL::StoreActionStore) : MTL::StoreActionDontCare;

		default:
			return MTL::StoreActionStore;
	}
}

bool MDAttachment::shouldClear(const MDSubpass &p_subpass, bool p_is_stencil) const {
	// If the subpass is not the first subpass to use this attachment, don't clear this attachment.
	if (p_subpass.subpass_index != firstUseSubpassIndex) {
		return false;
	}
	return (p_is_stencil ? stencilLoadAction : loadAction) == MTL::LoadActionClear;
}

MDRenderPass::MDRenderPass(Vector<MDAttachment> &p_attachments, Vector<MDSubpass> &p_subpasses) :
		attachments(p_attachments), subpasses(p_subpasses) {
	for (MDAttachment &att : attachments) {
		att.linkToSubpass(*this);
	}
}

#pragma mark - Command Buffer Base

void MDCommandBufferBase::retain_resource(CFTypeRef p_resource) {
	CFRetain(p_resource);
	_retained_resources.push_back(p_resource);
}

void MDCommandBufferBase::release_resources() {
	for (CFTypeRef r : _retained_resources) {
		CFRelease(r);
	}
	_retained_resources.clear();
}

void MDCommandBufferBase::render_set_viewport(VectorView<Rect2i> p_viewports) {
	RenderStateBase &state = get_render_state_base();
	state.viewports.resize(p_viewports.size());
	for (uint32_t i = 0; i < p_viewports.size(); i += 1) {
		Rect2i const &vp = p_viewports[i];
		state.viewports[i] = {
			.originX = static_cast<double>(vp.position.x),
			.originY = static_cast<double>(vp.position.y),
			.width = static_cast<double>(vp.size.width),
			.height = static_cast<double>(vp.size.height),
			.znear = 0.0,
			.zfar = 1.0,
		};
	}
	state.dirty.set_flag(RenderStateBase::DIRTY_VIEWPORT);
}

void MDCommandBufferBase::render_set_scissor(VectorView<Rect2i> p_scissors) {
	RenderStateBase &state = get_render_state_base();
	state.scissors.resize(p_scissors.size());
	for (uint32_t i = 0; i < p_scissors.size(); i += 1) {
		Rect2i const &vp = p_scissors[i];
		state.scissors[i] = {
			.x = static_cast<NS::UInteger>(vp.position.x),
			.y = static_cast<NS::UInteger>(vp.position.y),
			.width = static_cast<NS::UInteger>(vp.size.width),
			.height = static_cast<NS::UInteger>(vp.size.height),
		};
	}
	state.dirty.set_flag(RenderStateBase::DIRTY_SCISSOR);
}

void MDCommandBufferBase::render_set_blend_constants(const Color &p_constants) {
	DEV_ASSERT(type == MDCommandBufferStateType::Render);
	RenderStateBase &state = get_render_state_base();
	if (state.blend_constants != p_constants) {
		state.blend_constants = p_constants;
		state.dirty.set_flag(RenderStateBase::DIRTY_BLEND);
	}
}

void MDCommandBufferBase::_populate_vertices(simd::float4 *p_vertices, Size2i p_fb_size, VectorView<Rect2i> p_rects) {
	uint32_t idx = 0;
	for (uint32_t i = 0; i < p_rects.size(); i++) {
		Rect2i const &rect = p_rects[i];
		idx = _populate_vertices(p_vertices, idx, rect, p_fb_size);
	}
}

uint32_t MDCommandBufferBase::_populate_vertices(simd::float4 *p_vertices, uint32_t p_index, Rect2i const &p_rect, Size2i p_fb_size) {
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
	uint32_t endLayer = get_current_view_count();

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

void MDCommandBufferBase::_end_render_pass() {
	MDFrameBuffer const &fb_info = *get_frame_buffer();
	MDSubpass const &subpass = get_current_subpass();

	PixelFormats &pf = device_driver->get_pixel_formats();

	for (uint32_t i = 0; i < subpass.resolve_references.size(); i++) {
		uint32_t color_index = subpass.color_references[i].attachment;
		uint32_t resolve_index = subpass.resolve_references[i].attachment;
		DEV_ASSERT((color_index == RDD::AttachmentReference::UNUSED) == (resolve_index == RDD::AttachmentReference::UNUSED));
		if (color_index == RDD::AttachmentReference::UNUSED || !fb_info.has_texture(color_index)) {
			continue;
		}

		MTL::Texture *resolve_tex = fb_info.get_texture(resolve_index);

		CRASH_COND_MSG(!flags::all(pf.getCapabilities(resolve_tex->pixelFormat()), kMTLFmtCapsResolve), "not implemented: unresolvable texture types");
		// see: https://github.com/KhronosGroup/MoltenVK/blob/d20d13fe2735adb845636a81522df1b9d89c0fba/MoltenVK/MoltenVK/GPUObjects/MVKRenderPass.mm#L407
	}

	end_render_encoding();
}

void MDCommandBufferBase::_render_clear_render_area() {
	MDRenderPass const &pass = *get_render_pass();
	MDSubpass const &subpass = get_current_subpass();
	LocalVector<RDD::RenderPassClearValue> &clear_values = get_clear_values();

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
			clears[clears_count++] = { .aspect = RDD::TEXTURE_ASPECT_COLOR_BIT, .color_attachment = idx, .value = clear_values[idx] };
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

		clears[clears_count++] = { .aspect = bits, .color_attachment = ds_index, .value = clear_values[ds_index] };
	}

	if (clears_count == 0) {
		return;
	}

	render_clear_attachments(VectorView(clears, clears_count), { get_render_area() });
}

void MDCommandBufferBase::encode_push_constant_data(RDD::ShaderID p_shader, VectorView<uint32_t> p_data) {
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
				mark_push_constants_dirty();
			}
		} break;
		case MDCommandBufferStateType::Blit:
		case MDCommandBufferStateType::None:
			return;
	}
}

#pragma mark - Metal Library

static const char *SHADER_STAGE_NAMES[] = {
	[RDC::SHADER_STAGE_VERTEX] = "vert",
	[RDC::SHADER_STAGE_FRAGMENT] = "frag",
	[RDC::SHADER_STAGE_TESSELATION_CONTROL] = "tess_ctrl",
	[RDC::SHADER_STAGE_TESSELATION_EVALUATION] = "tess_eval",
	[RDC::SHADER_STAGE_COMPUTE] = "comp",
};

void ShaderCacheEntry::notify_free() const {
	owner.shader_cache_free_entry(key);
}

#pragma mark - MDLibrary

MDLibrary::MDLibrary(ShaderCacheEntry *p_entry
#ifdef DEV_ENABLED
		,
		NS::String *p_source
#endif
		) :
		_entry(p_entry) {
#ifdef DEV_ENABLED
	_original_source = NS::RetainPtr(p_source);
#endif
}

MDLibrary::~MDLibrary() {
	_entry->notify_free();
}

void MDLibrary::set_label(NS::String *p_label) {
}

#pragma mark - MDLazyLibrary

/// Loads the MTLLibrary when the library is first accessed.
class MDLazyLibrary final : public MDLibrary {
	NS::SharedPtr<MTL::Library> _library;
	NS::Error *_error = nullptr;
	std::shared_mutex _mu;
	bool _loaded = false;
	MTL::Device *_device = nullptr;
	NS::SharedPtr<NS::String> _source;
	NS::SharedPtr<MTL::CompileOptions> _options;

	void _load();

public:
	MDLazyLibrary(ShaderCacheEntry *p_entry,
			MTL::Device *p_device,
			NS::String *p_source,
			MTL::CompileOptions *p_options);

	MTL::Library *get_library() override;
	NS::Error *get_error() override;
};

MDLazyLibrary::MDLazyLibrary(ShaderCacheEntry *p_entry,
		MTL::Device *p_device,
		NS::String *p_source,
		MTL::CompileOptions *p_options) :
		MDLibrary(p_entry
#ifdef DEV_ENABLED
				,
				p_source
#endif
				),
		_device(p_device),
		_source(NS::RetainPtr(p_source)),
		_options(NS::RetainPtr(p_options)) {
}

void MDLazyLibrary::_load() {
	{
		std::shared_lock<std::shared_mutex> lock(_mu);
		if (_loaded) {
			return;
		}
	}

	std::unique_lock<std::shared_mutex> lock(_mu);
	if (_loaded) {
		return;
	}

	os_signpost_id_t compile_id = (os_signpost_id_t)(uintptr_t)this;
	os_signpost_interval_begin(LOG_INTERVALS, compile_id, "shader_compile",
			"shader_name=%{public}s stage=%{public}s hash=%X",
			_entry->name.get_data(), SHADER_STAGE_NAMES[_entry->stage], _entry->key.short_sha());
	NS::Error *error = nullptr;
	_library = NS::TransferPtr(_device->newLibrary(_source.get(), _options.get(), &error));
	os_signpost_interval_end(LOG_INTERVALS, compile_id, "shader_compile");
	_error = error;
	_device = nullptr;
	_source.reset();
	_options.reset();
	_loaded = true;
}

MTL::Library *MDLazyLibrary::get_library() {
	_load();
	return _library.get();
}

NS::Error *MDLazyLibrary::get_error() {
	_load();
	return _error;
}

#pragma mark - MDImmediateLibrary

/// Loads the MTLLibrary immediately on initialization, using Metal's async compilation API.
class MDImmediateLibrary final : public MDLibrary {
	NS::SharedPtr<MTL::Library> _library;
	NS::Error *_error = nullptr;
	std::mutex _cv_mutex;
	std::condition_variable _cv;
	std::atomic<bool> _complete{ false };
	bool _ready = false;

public:
	MDImmediateLibrary(ShaderCacheEntry *p_entry,
			MTL::Device *p_device,
			NS::String *p_source,
			MTL::CompileOptions *p_options);

	MTL::Library *get_library() override;
	NS::Error *get_error() override;
};

MDImmediateLibrary::MDImmediateLibrary(ShaderCacheEntry *p_entry,
		MTL::Device *p_device,
		NS::String *p_source,
		MTL::CompileOptions *p_options) :
		MDLibrary(p_entry
#ifdef DEV_ENABLED
				,
				p_source
#endif
		) {
	os_signpost_id_t compile_id = (os_signpost_id_t)(uintptr_t)this;
	os_signpost_interval_begin(LOG_INTERVALS, compile_id, "shader_compile",
			"shader_name=%{public}s stage=%{public}s hash=%X",
			p_entry->name.get_data(), SHADER_STAGE_NAMES[p_entry->stage], p_entry->key.short_sha());

	// Use Metal's async compilation API with std::function callback.
	p_device->newLibrary(p_source, p_options, [this, compile_id, p_entry](MTL::Library *library, NS::Error *error) {
		os_signpost_interval_end(LOG_INTERVALS, compile_id, "shader_compile");
		if (library) {
			_library = NS::RetainPtr(library);
		}
		_error = error;
		if (error) {
			ERR_PRINT(vformat(U"Error compiling shader %s: %s", p_entry->name.get_data(), error->localizedDescription()->utf8String()));
		}

		{
			std::lock_guard<std::mutex> lock(_cv_mutex);
			_ready = true;
		}
		_cv.notify_all();
		_complete = true;
	});
}

MTL::Library *MDImmediateLibrary::get_library() {
	if (!_complete) {
		std::unique_lock<std::mutex> lock(_cv_mutex);
		_cv.wait(lock, [this] { return _ready; });
	}
	return _library.get();
}

NS::Error *MDImmediateLibrary::get_error() {
	if (!_complete) {
		std::unique_lock<std::mutex> lock(_cv_mutex);
		_cv.wait(lock, [this] { return _ready; });
	}
	return _error;
}

#pragma mark - MDBinaryLibrary

/// Loads the MTLLibrary from pre-compiled binary data.
class MDBinaryLibrary final : public MDLibrary {
	NS::SharedPtr<MTL::Library> _library;
	NS::Error *_error = nullptr;

public:
	MDBinaryLibrary(ShaderCacheEntry *p_entry,
			MTL::Device *p_device,
#ifdef DEV_ENABLED
			NS::String *p_source,
#endif
			dispatch_data_t p_data);

	MTL::Library *get_library() override;
	NS::Error *get_error() override;
};

MDBinaryLibrary::MDBinaryLibrary(ShaderCacheEntry *p_entry,
		MTL::Device *p_device,
#ifdef DEV_ENABLED
		NS::String *p_source,
#endif
		dispatch_data_t p_data) :
		MDLibrary(p_entry
#ifdef DEV_ENABLED
				,
				p_source
#endif
		) {
	NS::Error *error = nullptr;
	_library = NS::TransferPtr(p_device->newLibrary(p_data, &error));
	if (error != nullptr) {
		_error = error;
		ERR_PRINT(vformat("Unable to load shader library: %s", error->localizedDescription()->utf8String()));
	}
}

MTL::Library *MDBinaryLibrary::get_library() {
	return _library.get();
}

NS::Error *MDBinaryLibrary::get_error() {
	return _error;
}

#pragma mark - MDLibrary Factory Methods

std::shared_ptr<MDLibrary> MDLibrary::create(ShaderCacheEntry *p_entry,
		MTL::Device *p_device,
		NS::String *p_source,
		MTL::CompileOptions *p_options,
		ShaderLoadStrategy p_strategy) {
	std::shared_ptr<MDLibrary> lib;
	switch (p_strategy) {
		case ShaderLoadStrategy::IMMEDIATE:
			[[fallthrough]];
		default:
			lib = std::make_shared<MDImmediateLibrary>(p_entry, p_device, p_source, p_options);
			break;
		case ShaderLoadStrategy::LAZY:
			lib = std::make_shared<MDLazyLibrary>(p_entry, p_device, p_source, p_options);
			break;
	}
	p_entry->library = lib;
	return lib;
}

std::shared_ptr<MDLibrary> MDLibrary::create(ShaderCacheEntry *p_entry,
		MTL::Device *p_device,
#ifdef DEV_ENABLED
		NS::String *p_source,
#endif
		dispatch_data_t p_data) {
	std::shared_ptr<MDLibrary> lib = std::make_shared<MDBinaryLibrary>(p_entry, p_device,
#ifdef DEV_ENABLED
			p_source,
#endif
			p_data);
	p_entry->library = lib;
	return lib;
}
