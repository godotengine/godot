/**************************************************************************/
/*  metal_objects_shared.mm                                               */
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

#import "metal_objects_shared.h"

#import "rendering_device_driver_metal.h"

#import <os/signpost.h>
#import <simd/simd.h>

#pragma mark - Resource Factory

id<MTLFunction> MDResourceFactory::new_func(NSString *p_source, NSString *p_name, NSError **p_error) {
	@autoreleasepool {
		NSError *err = nil;
		MTLCompileOptions *options = [MTLCompileOptions new];
		id<MTLLibrary> mtlLib = [device newLibraryWithSource:p_source
													 options:options
													   error:&err];
		if (err) {
			if (p_error != nil) {
				*p_error = err;
			}
		}
		return [mtlLib newFunctionWithName:p_name];
	}
}

id<MTLFunction> MDResourceFactory::new_clear_vert_func(ClearAttKey &p_key) {
	@autoreleasepool {
		NSString *msl = [NSString stringWithFormat:@R"(
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
)", p_key.is_layered_rendering_enabled() ? " [[render_target_array_index]]" : "", ClearAttKey::DEPTH_INDEX];

		return new_func(msl, @"vertClear", nil);
	}
}

id<MTLFunction> MDResourceFactory::new_clear_frag_func(ClearAttKey &p_key) {
	@autoreleasepool {
		NSMutableString *msl = [NSMutableString stringWithCapacity:2048];

		[msl appendFormat:@R"(
#include <metal_stdlib>
using namespace metal;

typedef struct {
    float4 v_position [[position]];
} VaryingsPos;

typedef struct {
    float4 colors[9];
} ClearColorsIn;

typedef struct {
)"];

		for (uint32_t caIdx = 0; caIdx < ClearAttKey::COLOR_COUNT; caIdx++) {
			if (p_key.is_enabled(caIdx)) {
				NSString *typeStr = get_format_type_string((MTLPixelFormat)p_key.pixel_formats[caIdx]);
				[msl appendFormat:@"    %@4 color%u [[color(%u)]];\n", typeStr, caIdx, caIdx];
			}
		}
		[msl appendFormat:@R"(} ClearColorsOut;

fragment ClearColorsOut fragClear(VaryingsPos varyings [[stage_in]], constant ClearColorsIn& ccIn [[buffer(0)]]) {

    ClearColorsOut ccOut;
)"];
		for (uint32_t caIdx = 0; caIdx < ClearAttKey::COLOR_COUNT; caIdx++) {
			if (p_key.is_enabled(caIdx)) {
				NSString *typeStr = get_format_type_string((MTLPixelFormat)p_key.pixel_formats[caIdx]);
				[msl appendFormat:@"    ccOut.color%u = %@4(ccIn.colors[%u]);\n", caIdx, typeStr, caIdx];
			}
		}
		[msl appendString:@R"(    return ccOut;
})"];

		return new_func(msl, @"fragClear", nil);
	}
}

NSString *MDResourceFactory::get_format_type_string(MTLPixelFormat p_fmt) {
	switch (pixel_formats.getFormatType(p_fmt)) {
		case MTLFormatType::ColorInt8:
		case MTLFormatType::ColorInt16:
			return @"short";
		case MTLFormatType::ColorUInt8:
		case MTLFormatType::ColorUInt16:
			return @"ushort";
		case MTLFormatType::ColorInt32:
			return @"int";
		case MTLFormatType::ColorUInt32:
			return @"uint";
		case MTLFormatType::ColorHalf:
			return @"half";
		case MTLFormatType::ColorFloat:
		case MTLFormatType::DepthStencil:
		case MTLFormatType::Compressed:
			return @"float";
		case MTLFormatType::None:
			return @"unexpected_MTLPixelFormatInvalid";
	}
}

id<MTLDepthStencilState> MDResourceFactory::new_depth_stencil_state(bool p_use_depth, bool p_use_stencil) {
	MTLDepthStencilDescriptor *dsDesc = [MTLDepthStencilDescriptor new];
	dsDesc.depthCompareFunction = MTLCompareFunctionAlways;
	dsDesc.depthWriteEnabled = p_use_depth;

	if (p_use_stencil) {
		MTLStencilDescriptor *sDesc = [MTLStencilDescriptor new];
		sDesc.stencilCompareFunction = MTLCompareFunctionAlways;
		sDesc.stencilFailureOperation = MTLStencilOperationReplace;
		sDesc.depthFailureOperation = MTLStencilOperationReplace;
		sDesc.depthStencilPassOperation = MTLStencilOperationReplace;

		dsDesc.frontFaceStencil = sDesc;
		dsDesc.backFaceStencil = sDesc;
	} else {
		dsDesc.frontFaceStencil = nil;
		dsDesc.backFaceStencil = nil;
	}

	return [device newDepthStencilStateWithDescriptor:dsDesc];
}

id<MTLRenderPipelineState> MDResourceFactory::new_clear_pipeline_state(ClearAttKey &p_key, NSError **p_error) {
	id<MTLFunction> vtxFunc = new_clear_vert_func(p_key);
	id<MTLFunction> fragFunc = new_clear_frag_func(p_key);
	MTLRenderPipelineDescriptor *plDesc = [MTLRenderPipelineDescriptor new];
	plDesc.label = @"ClearRenderAttachments";
	plDesc.vertexFunction = vtxFunc;
	plDesc.fragmentFunction = fragFunc;
	plDesc.rasterSampleCount = p_key.sample_count;
	plDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

	for (uint32_t caIdx = 0; caIdx < ClearAttKey::COLOR_COUNT; caIdx++) {
		MTLRenderPipelineColorAttachmentDescriptor *colorDesc = plDesc.colorAttachments[caIdx];
		colorDesc.pixelFormat = (MTLPixelFormat)p_key.pixel_formats[caIdx];
		colorDesc.writeMask = p_key.is_enabled(caIdx) ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
	}

	MTLPixelFormat mtlDepthFormat = p_key.depth_format();
	if (pixel_formats.isDepthFormat(mtlDepthFormat)) {
		plDesc.depthAttachmentPixelFormat = mtlDepthFormat;
	}

	MTLPixelFormat mtlStencilFormat = p_key.stencil_format();
	if (pixel_formats.isStencilFormat(mtlStencilFormat)) {
		plDesc.stencilAttachmentPixelFormat = mtlStencilFormat;
	}

	MTLVertexDescriptor *vtxDesc = plDesc.vertexDescriptor;

	// Vertex attribute descriptors.
	MTLVertexAttributeDescriptorArray *vaDescArray = vtxDesc.attributes;
	MTLVertexAttributeDescriptor *vaDesc;
	NSUInteger vtxBuffIdx = get_vertex_buffer_index(VERT_CONTENT_BUFFER_INDEX);
	NSUInteger vtxStride = 0;

	// Vertex location.
	vaDesc = vaDescArray[0];
	vaDesc.format = MTLVertexFormatFloat4;
	vaDesc.bufferIndex = vtxBuffIdx;
	vaDesc.offset = vtxStride;
	vtxStride += sizeof(simd::float4);

	// Vertex attribute buffer.
	MTLVertexBufferLayoutDescriptorArray *vbDescArray = vtxDesc.layouts;
	MTLVertexBufferLayoutDescriptor *vbDesc = vbDescArray[vtxBuffIdx];
	vbDesc.stepFunction = MTLVertexStepFunctionPerVertex;
	vbDesc.stepRate = 1;
	vbDesc.stride = vtxStride;

	return [device newRenderPipelineStateWithDescriptor:plDesc error:p_error];
}

id<MTLRenderPipelineState> MDResourceFactory::new_empty_draw_pipeline_state(ClearAttKey &p_key, NSError **p_error) {
	DEV_ASSERT(!p_key.is_layered_rendering_enabled());
	DEV_ASSERT(p_key.is_enabled(0));
	DEV_ASSERT(!p_key.is_depth_enabled());
	DEV_ASSERT(!p_key.is_stencil_enabled());

	@autoreleasepool {
		NSMutableString *msl = [NSMutableString stringWithCapacity:512];
		[msl appendString:@"#include <metal_stdlib>\nusing namespace metal;\n\n"];
		[msl appendString:@"struct FullscreenNoopOut {\n    float4 position [[position]];\n};\n\n"];
		[msl appendString:@"vertex FullscreenNoopOut fullscreenNoopVert(uint vid [[vertex_id]]) {\n"];
		[msl appendString:@"    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };\n"];
		[msl appendString:@"    float2 pos = positions[vid];\n\n"];
		[msl appendString:@"    FullscreenNoopOut out;\n"];
		[msl appendString:@"    out.position = float4(pos, 0.0, 1.0);\n"];
		[msl appendString:@"    return out;\n"];
		[msl appendString:@"}\n\nfragment void fullscreenNoopFrag(float4 gl_FragCoord [[position]]) {\n}\n"];

		NSError *err = nil;
		MTLCompileOptions *options = [MTLCompileOptions new];
		id<MTLLibrary> mtlLib = [device newLibraryWithSource:msl options:options error:&err];
		if (err && p_error != nil) {
			*p_error = err;
		}

		if (mtlLib == nil) {
			return nil;
		}

		id<MTLFunction> vtxFunc = [mtlLib newFunctionWithName:@"fullscreenNoopVert"];
		id<MTLFunction> fragFunc = [mtlLib newFunctionWithName:@"fullscreenNoopFrag"];

		MTLRenderPipelineDescriptor *plDesc = [MTLRenderPipelineDescriptor new];
		plDesc.label = @"EmptyDrawFullscreenTriangle";
		plDesc.vertexFunction = vtxFunc;
		plDesc.fragmentFunction = fragFunc;
		plDesc.rasterSampleCount = p_key.sample_count ? p_key.sample_count : 1;
		plDesc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

		MTLRenderPipelineColorAttachmentDescriptor *colorDesc = plDesc.colorAttachments[0];
		colorDesc.pixelFormat = (MTLPixelFormat)p_key.pixel_formats[0];
		colorDesc.writeMask = MTLColorWriteMaskNone;

		return [device newRenderPipelineStateWithDescriptor:plDesc error:p_error];
	}
}

#pragma mark - Resource Cache

id<MTLRenderPipelineState> MDResourceCache::get_clear_render_pipeline_state(ClearAttKey &p_key, NSError **p_error) {
	HashMap::ConstIterator it = clear_states.find(p_key);
	if (it != clear_states.end()) {
		return it->value;
	}

	id<MTLRenderPipelineState> state = resource_factory->new_clear_pipeline_state(p_key, p_error);
	clear_states[p_key] = state;
	return state;
}

id<MTLRenderPipelineState> MDResourceCache::get_empty_draw_pipeline_state(ClearAttKey &p_key, NSError **p_error) {
	HashMap::ConstIterator it = empty_draw_states.find(p_key);
	if (it != empty_draw_states.end()) {
		return it->value;
	}

	id<MTLRenderPipelineState> state = resource_factory->new_empty_draw_pipeline_state(p_key, p_error);
	empty_draw_states[p_key] = state;
	return state;
}

id<MTLDepthStencilState> MDResourceCache::get_depth_stencil_state(bool p_use_depth, bool p_use_stencil) {
	if (p_use_depth && p_use_stencil) {
		if (clear_depth_stencil_state.all == nil) {
			clear_depth_stencil_state.all = resource_factory->new_depth_stencil_state(true, true);
		}
		return clear_depth_stencil_state.all;
	} else if (p_use_depth) {
		if (clear_depth_stencil_state.depth_only == nil) {
			clear_depth_stencil_state.depth_only = resource_factory->new_depth_stencil_state(true, false);
		}
		return clear_depth_stencil_state.depth_only;
	} else if (p_use_stencil) {
		if (clear_depth_stencil_state.stencil_only == nil) {
			clear_depth_stencil_state.stencil_only = resource_factory->new_depth_stencil_state(false, true);
		}
		return clear_depth_stencil_state.stencil_only;
	} else {
		if (clear_depth_stencil_state.none == nil) {
			clear_depth_stencil_state.none = resource_factory->new_depth_stencil_state(false, false);
		}
		return clear_depth_stencil_state.none;
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

MTLStoreAction MDAttachment::getMTLStoreAction(MDSubpass const &p_subpass,
		bool p_is_rendering_entire_area,
		bool p_has_resolve,
		bool p_can_resolve,
		bool p_is_stencil) const {
	if (!p_is_rendering_entire_area || !isLastUseOf(p_subpass)) {
		return p_has_resolve && p_can_resolve ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore;
	}

	switch (p_is_stencil ? stencilStoreAction : storeAction) {
		case MTLStoreActionStore:
			return p_has_resolve && p_can_resolve ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore;
		case MTLStoreActionDontCare:
			return p_has_resolve ? (p_can_resolve ? MTLStoreActionMultisampleResolve : MTLStoreActionStore) : MTLStoreActionDontCare;

		default:
			return MTLStoreActionStore;
	}
}

bool MDAttachment::configureDescriptor(MTLRenderPassAttachmentDescriptor *p_desc,
		PixelFormats &p_pf,
		MDSubpass const &p_subpass,
		id<MTLTexture> p_attachment,
		bool p_is_rendering_entire_area,
		bool p_has_resolve,
		bool p_can_resolve,
		bool p_is_stencil) const {
	p_desc.texture = p_attachment;

	MTLLoadAction load;
	if (!p_is_rendering_entire_area || !isFirstUseOf(p_subpass)) {
		load = MTLLoadActionLoad;
	} else {
		load = p_is_stencil ? stencilLoadAction : loadAction;
	}

	p_desc.loadAction = load;

	MTLPixelFormat mtlFmt = p_attachment.pixelFormat;
	bool isDepthFormat = p_pf.isDepthFormat(mtlFmt);
	bool isStencilFormat = p_pf.isStencilFormat(mtlFmt);
	if (isStencilFormat && !p_is_stencil && !isDepthFormat) {
		p_desc.storeAction = MTLStoreActionDontCare;
	} else {
		p_desc.storeAction = getMTLStoreAction(p_subpass, p_is_rendering_entire_area, p_has_resolve, p_can_resolve, p_is_stencil);
	}

	return load == MTLLoadActionClear;
}

bool MDAttachment::shouldClear(const MDSubpass &p_subpass, bool p_is_stencil) const {
	// If the subpass is not the first subpass to use this attachment, don't clear this attachment.
	if (p_subpass.subpass_index != firstUseSubpassIndex) {
		return false;
	}
	return (p_is_stencil ? stencilLoadAction : loadAction) == MTLLoadActionClear;
}

MDRenderPass::MDRenderPass(Vector<MDAttachment> &p_attachments, Vector<MDSubpass> &p_subpasses) :
		attachments(p_attachments), subpasses(p_subpasses) {
	for (MDAttachment &att : attachments) {
		att.linkToSubpass(*this);
	}
}

#pragma mark - Command Buffer Base

void MDCommandBufferBase::retain_resource(id p_resource) {
	_retained_resources.push_back(p_resource);
}

void MDCommandBufferBase::release_resources() {
	_retained_resources.clear();
}

#pragma mark - Metal Library

static const char *SHADER_STAGE_NAMES[] = {
	[RD::SHADER_STAGE_VERTEX] = "vert",
	[RD::SHADER_STAGE_FRAGMENT] = "frag",
	[RD::SHADER_STAGE_TESSELATION_CONTROL] = "tess_ctrl",
	[RD::SHADER_STAGE_TESSELATION_EVALUATION] = "tess_eval",
	[RD::SHADER_STAGE_COMPUTE] = "comp",
};

void ShaderCacheEntry::notify_free() const {
	owner.shader_cache_free_entry(key);
}

#pragma mark - MDLibrary

MDLibrary::MDLibrary(ShaderCacheEntry *p_entry
#ifdef DEV_ENABLED
		,
		NSString *p_source
#endif
		) :
		_entry(p_entry) {
#ifdef DEV_ENABLED
	_original_source = p_source;
#endif
}

MDLibrary::~MDLibrary() {
	_entry->notify_free();
}

void MDLibrary::set_label(NSString *p_label) {
}

#pragma mark - MDLazyLibrary

/// Loads the MTLLibrary when the library is first accessed.
class MDLazyLibrary final : public MDLibrary {
	id<MTLLibrary> _library = nil;
	NSError *_error = nil;
	std::shared_mutex _mu;
	bool _loaded = false;
	id<MTLDevice> _device = nil;
	NSString *_source = nil;
	MTLCompileOptions *_options = nil;

	void _load();

public:
	MDLazyLibrary(ShaderCacheEntry *p_entry,
			id<MTLDevice> p_device,
			NSString *p_source,
			MTLCompileOptions *p_options);

	id<MTLLibrary> get_library() override;
	NSError *get_error() override;
};

MDLazyLibrary::MDLazyLibrary(ShaderCacheEntry *p_entry,
		id<MTLDevice> p_device,
		NSString *p_source,
		MTLCompileOptions *p_options) :
		MDLibrary(p_entry
#ifdef DEV_ENABLED
				,
				p_source
#endif
				),
		_device(p_device),
		_source(p_source),
		_options(p_options) {
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
	NSError *error = nil;
	_library = [_device newLibraryWithSource:_source options:_options error:&error];
	os_signpost_interval_end(LOG_INTERVALS, compile_id, "shader_compile");
	_error = error;
	_device = nil;
	_source = nil;
	_options = nil;
	_loaded = true;
}

id<MTLLibrary> MDLazyLibrary::get_library() {
	_load();
	return _library;
}

NSError *MDLazyLibrary::get_error() {
	_load();
	return _error;
}

#pragma mark - MDImmediateLibrary

/// Loads the MTLLibrary immediately on initialization, using an asynchronous API.
class MDImmediateLibrary final : public MDLibrary {
	id<MTLLibrary> _library = nil;
	NSError *_error = nil;
	std::mutex _cv_mutex;
	std::condition_variable _cv;
	std::atomic<bool> _complete{ false };
	bool _ready = false;

public:
	MDImmediateLibrary(ShaderCacheEntry *p_entry,
			id<MTLDevice> p_device,
			NSString *p_source,
			MTLCompileOptions *p_options);

	id<MTLLibrary> get_library() override;
	NSError *get_error() override;
};

MDImmediateLibrary::MDImmediateLibrary(ShaderCacheEntry *p_entry,
		id<MTLDevice> p_device,
		NSString *p_source,
		MTLCompileOptions *p_options) :
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

	[p_device newLibraryWithSource:p_source
						   options:p_options
				 completionHandler:^(id<MTLLibrary> library, NSError *error) {
					 os_signpost_interval_end(LOG_INTERVALS, compile_id, "shader_compile");
					 _library = library;
					 _error = error;
					 if (error) {
						 ERR_PRINT(vformat(U"Error compiling shader %s: %s", p_entry->name.get_data(), error.localizedDescription.UTF8String));
					 }

					 {
						 std::lock_guard<std::mutex> lock(_cv_mutex);
						 _ready = true;
					 }
					 _cv.notify_all();
					 _complete = true;
				 }];
}

id<MTLLibrary> MDImmediateLibrary::get_library() {
	if (!_complete) {
		std::unique_lock<std::mutex> lock(_cv_mutex);
		_cv.wait(lock, [this] { return _ready; });
	}
	return _library;
}

NSError *MDImmediateLibrary::get_error() {
	if (!_complete) {
		std::unique_lock<std::mutex> lock(_cv_mutex);
		_cv.wait(lock, [this] { return _ready; });
	}
	return _error;
}

#pragma mark - MDBinaryLibrary

/// Loads the MTLLibrary from pre-compiled binary data.
class MDBinaryLibrary final : public MDLibrary {
	id<MTLLibrary> _library = nil;
	NSError *_error = nil;

public:
	MDBinaryLibrary(ShaderCacheEntry *p_entry,
			id<MTLDevice> p_device,
#ifdef DEV_ENABLED
			NSString *p_source,
#endif
			dispatch_data_t p_data);

	id<MTLLibrary> get_library() override;
	NSError *get_error() override;
};

MDBinaryLibrary::MDBinaryLibrary(ShaderCacheEntry *p_entry,
		id<MTLDevice> p_device,
#ifdef DEV_ENABLED
		NSString *p_source,
#endif
		dispatch_data_t p_data) :
		MDLibrary(p_entry
#ifdef DEV_ENABLED
				,
				p_source
#endif
		) {
	NSError *error = nil;
	_library = [p_device newLibraryWithData:p_data error:&error];
	if (error != nil) {
		_error = error;
		NSString *desc = [error description];
		ERR_PRINT(vformat("Unable to load shader library: %s", desc.UTF8String));
	}
}

id<MTLLibrary> MDBinaryLibrary::get_library() {
	return _library;
}

NSError *MDBinaryLibrary::get_error() {
	return _error;
}

#pragma mark - MDLibrary Factory Methods

std::shared_ptr<MDLibrary> MDLibrary::create(ShaderCacheEntry *p_entry,
		id<MTLDevice> p_device,
		NSString *p_source,
		MTLCompileOptions *p_options,
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
		id<MTLDevice> p_device,
#ifdef DEV_ENABLED
		NSString *p_source,
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
