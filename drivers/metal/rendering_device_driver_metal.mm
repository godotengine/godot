/**************************************************************************/
/*  rendering_device_driver_metal.mm                                      */
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

#import "rendering_device_driver_metal.h"

#import "pixel_formats.h"
#import "rendering_context_driver_metal.h"
#import "rendering_shader_container_metal.h"

#include "core/io/marshalls.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "drivers/apple/foundation_helpers.h"

#import <Metal/MTLTexture.h>
#import <Metal/Metal.h>
#import <os/log.h>
#import <os/signpost.h>
#include <algorithm>

#ifndef MTLGPUAddress
typedef uint64_t MTLGPUAddress;
#endif

#pragma mark - Logging

extern os_log_t LOG_DRIVER;
// Used for dynamic tracing.
extern os_log_t LOG_INTERVALS;

/*****************/
/**** GENERIC ****/
/*****************/

// RDD::CompareOperator == VkCompareOp.
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_NEVER, MTLCompareFunctionNever));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_LESS, MTLCompareFunctionLess));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_EQUAL, MTLCompareFunctionEqual));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_LESS_OR_EQUAL, MTLCompareFunctionLessEqual));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_GREATER, MTLCompareFunctionGreater));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_NOT_EQUAL, MTLCompareFunctionNotEqual));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_GREATER_OR_EQUAL, MTLCompareFunctionGreaterEqual));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_ALWAYS, MTLCompareFunctionAlways));

_FORCE_INLINE_ MTLSize mipmapLevelSizeFromSize(MTLSize p_size, NSUInteger p_level) {
	if (p_level == 0) {
		return p_size;
	}

	MTLSize lvlSize;
	lvlSize.width = MAX(p_size.width >> p_level, 1UL);
	lvlSize.height = MAX(p_size.height >> p_level, 1UL);
	lvlSize.depth = MAX(p_size.depth >> p_level, 1UL);
	return lvlSize;
}

/*****************/
/**** BUFFERS ****/
/*****************/

RDD::BufferID RenderingDeviceDriverMetal::buffer_create(uint64_t p_size, BitField<BufferUsageBits> p_usage, MemoryAllocationType p_allocation_type, uint64_t p_frames_drawn) {
	const uint64_t original_size = p_size;
	if (p_usage.has_flag(BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT)) {
		p_size = round_up_to_alignment(p_size, 16u) * _frame_count;
	}

	MTLResourceOptions options = 0;
	switch (p_allocation_type) {
		case MEMORY_ALLOCATION_TYPE_CPU:
			options = MTLResourceHazardTrackingModeTracked | MTLResourceStorageModeShared;
			break;
		case MEMORY_ALLOCATION_TYPE_GPU:
			if (p_usage.has_flag(BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT)) {
				options = MTLResourceHazardTrackingModeUntracked | MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
			} else {
				options = MTLResourceHazardTrackingModeTracked | MTLResourceStorageModePrivate;
			}
			break;
	}

	id<MTLBuffer> obj = [device newBufferWithLength:p_size options:options];
	ERR_FAIL_NULL_V_MSG(obj, BufferID(), "Can't create buffer of size: " + itos(p_size));

	BufferInfo *buf_info;
	if (p_usage.has_flag(BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT)) {
		MetalBufferDynamicInfo *dyn_buffer = memnew(MetalBufferDynamicInfo);
		buf_info = dyn_buffer;
#ifdef DEBUG_ENABLED
		dyn_buffer->last_frame_mapped = p_frames_drawn - 1ul;
#endif
		dyn_buffer->set_frame_index(0u);
		dyn_buffer->size_bytes = round_up_to_alignment(original_size, 16u);
	} else {
		buf_info = memnew(BufferInfo);
	}
	buf_info->metal_buffer = obj;

	return BufferID(buf_info);
}

bool RenderingDeviceDriverMetal::buffer_set_texel_format(BufferID p_buffer, DataFormat p_format) {
	// Nothing to do.
	return true;
}

void RenderingDeviceDriverMetal::buffer_free(BufferID p_buffer) {
	BufferInfo *buf_info = (BufferInfo *)p_buffer.id;
	buf_info->metal_buffer = nil; // Tell ARC to release.

	if (buf_info->is_dynamic()) {
		memdelete((MetalBufferDynamicInfo *)buf_info);
	} else {
		memdelete(buf_info);
	}
}

uint64_t RenderingDeviceDriverMetal::buffer_get_allocation_size(BufferID p_buffer) {
	const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
	return buf_info->metal_buffer.allocatedSize;
}

uint8_t *RenderingDeviceDriverMetal::buffer_map(BufferID p_buffer) {
	const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
	ERR_FAIL_COND_V_MSG(buf_info->metal_buffer.storageMode != MTLStorageModeShared, nullptr, "Unable to map private buffers");
	return (uint8_t *)buf_info->metal_buffer.contents;
}

void RenderingDeviceDriverMetal::buffer_unmap(BufferID p_buffer) {
	// Nothing to do.
}

uint8_t *RenderingDeviceDriverMetal::buffer_persistent_map_advance(BufferID p_buffer, uint64_t p_frames_drawn) {
	MetalBufferDynamicInfo *buf_info = (MetalBufferDynamicInfo *)p_buffer.id;
	ERR_FAIL_COND_V_MSG(!buf_info->is_dynamic(), nullptr, "Buffer must have BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT. Use buffer_map() instead.");
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V_MSG(buf_info->last_frame_mapped == p_frames_drawn, nullptr, "Buffers with BUFFER_USAGE_DYNAMIC_PERSISTENT_BIT must only be mapped once per frame. Otherwise there could be race conditions with the GPU. Amalgamate all data uploading into one map(), use an extra buffer or remove the bit.");
	buf_info->last_frame_mapped = p_frames_drawn;
#endif
	return (uint8_t *)buf_info->metal_buffer.contents + buf_info->next_frame_index(_frame_count) * buf_info->size_bytes;
}

uint64_t RenderingDeviceDriverMetal::buffer_get_dynamic_offsets(Span<BufferID> p_buffers) {
	uint64_t mask = 0u;
	uint64_t shift = 0u;

	for (const BufferID &buf : p_buffers) {
		const BufferInfo *buf_info = (const BufferInfo *)buf.id;
		if (!buf_info->is_dynamic()) {
			continue;
		}
		mask |= buf_info->frame_index() << shift;
		// We can encode the frame index in 2 bits since frame_count won't be > 4.
		shift += 2UL;
	}

	return mask;
}

void RenderingDeviceDriverMetal::buffer_flush(BufferID p_buffer) {
	// Nothing to do.
}

uint64_t RenderingDeviceDriverMetal::buffer_get_device_address(BufferID p_buffer) {
	if (@available(iOS 16.0, macOS 13.0, *)) {
		const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
		return buf_info->metal_buffer.gpuAddress;
	} else {
#if DEV_ENABLED
		WARN_PRINT_ONCE("buffer_get_device_address is not supported on this OS version.");
#endif
		return 0;
	}
}

#pragma mark - Texture

#pragma mark - Format Conversions

static const MTLTextureType TEXTURE_TYPE[RD::TEXTURE_TYPE_MAX] = {
	MTLTextureType1D,
	MTLTextureType2D,
	MTLTextureType3D,
	MTLTextureTypeCube,
	MTLTextureType1DArray,
	MTLTextureType2DArray,
	MTLTextureTypeCubeArray,
};

bool RenderingDeviceDriverMetal::is_valid_linear(TextureFormat const &p_format) const {
	MTLFormatType ft = pixel_formats->getFormatType(p_format.format);

	return p_format.texture_type == TEXTURE_TYPE_2D // Linear textures must be 2D textures.
			&& ft != MTLFormatType::DepthStencil && ft != MTLFormatType::Compressed // Linear textures must not be depth/stencil or compressed formats.)
			&& p_format.mipmaps == 1 // Linear textures must have 1 mipmap level.
			&& p_format.array_layers == 1 // Linear textures must have 1 array layer.
			&& p_format.samples == TEXTURE_SAMPLES_1; // Linear textures must have 1 sample.
}

RDD::TextureID RenderingDeviceDriverMetal::texture_create(const TextureFormat &p_format, const TextureView &p_view) {
	MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
	desc.textureType = TEXTURE_TYPE[p_format.texture_type];

	PixelFormats &formats = *pixel_formats;
	desc.pixelFormat = formats.getMTLPixelFormat(p_format.format);
	MTLFmtCaps format_caps = formats.getCapabilities(desc.pixelFormat);

	desc.width = p_format.width;
	desc.height = p_format.height;
	desc.depth = p_format.depth;
	desc.mipmapLevelCount = p_format.mipmaps;

	if (p_format.texture_type == TEXTURE_TYPE_1D_ARRAY ||
			p_format.texture_type == TEXTURE_TYPE_2D_ARRAY) {
		desc.arrayLength = p_format.array_layers;
	} else if (p_format.texture_type == TEXTURE_TYPE_CUBE_ARRAY) {
		desc.arrayLength = p_format.array_layers / 6;
	}

	// TODO(sgc): Evaluate lossy texture support (perhaps as a project option?)
	//  https://developer.apple.com/videos/play/tech-talks/10876?time=459
	// desc.compressionType = MTLTextureCompressionTypeLossy;

	if (p_format.samples > TEXTURE_SAMPLES_1) {
		SampleCount supported = (*device_properties).find_nearest_supported_sample_count(p_format.samples);

		if (supported > SampleCount1) {
			bool ok = p_format.texture_type == TEXTURE_TYPE_2D || p_format.texture_type == TEXTURE_TYPE_2D_ARRAY;
			if (ok) {
				switch (p_format.texture_type) {
					case TEXTURE_TYPE_2D:
						desc.textureType = MTLTextureType2DMultisample;
						break;
					case TEXTURE_TYPE_2D_ARRAY:
						desc.textureType = MTLTextureType2DMultisampleArray;
						break;
					default:
						break;
				}
				desc.sampleCount = (NSUInteger)supported;
				if (p_format.mipmaps > 1) {
					// For a buffer-backed or multi-sample texture, the value must be 1.
					WARN_PRINT("mipmaps == 1 for multi-sample textures");
					desc.mipmapLevelCount = 1;
				}
			} else {
				WARN_PRINT("Unsupported multi-sample texture type; disabling multi-sample");
			}
		}
	}

	static const MTLTextureSwizzle COMPONENT_SWIZZLE[TEXTURE_SWIZZLE_MAX] = {
		static_cast<MTLTextureSwizzle>(255), // IDENTITY
		MTLTextureSwizzleZero,
		MTLTextureSwizzleOne,
		MTLTextureSwizzleRed,
		MTLTextureSwizzleGreen,
		MTLTextureSwizzleBlue,
		MTLTextureSwizzleAlpha,
	};

	MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(
			p_view.swizzle_r != TEXTURE_SWIZZLE_IDENTITY ? COMPONENT_SWIZZLE[p_view.swizzle_r] : MTLTextureSwizzleRed,
			p_view.swizzle_g != TEXTURE_SWIZZLE_IDENTITY ? COMPONENT_SWIZZLE[p_view.swizzle_g] : MTLTextureSwizzleGreen,
			p_view.swizzle_b != TEXTURE_SWIZZLE_IDENTITY ? COMPONENT_SWIZZLE[p_view.swizzle_b] : MTLTextureSwizzleBlue,
			p_view.swizzle_a != TEXTURE_SWIZZLE_IDENTITY ? COMPONENT_SWIZZLE[p_view.swizzle_a] : MTLTextureSwizzleAlpha);

	// Represents a swizzle operation that is a no-op.
	static MTLTextureSwizzleChannels IDENTITY_SWIZZLE = {
		.red = MTLTextureSwizzleRed,
		.green = MTLTextureSwizzleGreen,
		.blue = MTLTextureSwizzleBlue,
		.alpha = MTLTextureSwizzleAlpha,
	};

	bool no_swizzle = memcmp(&IDENTITY_SWIZZLE, &swizzle, sizeof(MTLTextureSwizzleChannels)) == 0;
	if (!no_swizzle) {
		desc.swizzle = swizzle;
	}

	// Usage.

	MTLResourceOptions options = 0;
	bool is_linear = false;
#if defined(VISIONOS_ENABLED)
	const bool supports_memoryless = true;
#else
	GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wdeprecated-declarations")
	const bool supports_memoryless = (*device_properties).features.highestFamily >= MTLGPUFamilyApple2 && (*device_properties).features.highestFamily < MTLGPUFamilyMac1;
	GODOT_CLANG_WARNING_POP
#endif
	if (supports_memoryless && p_format.usage_bits & TEXTURE_USAGE_TRANSIENT_BIT) {
		options = MTLResourceStorageModeMemoryless | MTLResourceHazardTrackingModeTracked;
		desc.storageMode = MTLStorageModeMemoryless;
	} else {
		options = MTLResourceCPUCacheModeDefaultCache | MTLResourceHazardTrackingModeTracked;
		if (p_format.usage_bits & TEXTURE_USAGE_CPU_READ_BIT) {
			options |= MTLResourceStorageModeShared;
			// The user has indicated they want to read from the texture on the CPU,
			// so we'll see if we can use a linear format.
			// A linear format is a texture that is backed by a buffer,
			// which allows for CPU access to the texture data via a pointer.
			is_linear = is_valid_linear(p_format);
		} else {
			options |= MTLResourceStorageModePrivate;
		}
	}
	desc.resourceOptions = options;

	if (p_format.usage_bits & TEXTURE_USAGE_SAMPLING_BIT) {
		desc.usage |= MTLTextureUsageShaderRead;
	}

	if (p_format.usage_bits & TEXTURE_USAGE_STORAGE_BIT) {
		desc.usage |= MTLTextureUsageShaderWrite;
	}

	bool can_be_attachment = flags::any(format_caps, (kMTLFmtCapsColorAtt | kMTLFmtCapsDSAtt));

	if (flags::any(p_format.usage_bits, TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
			can_be_attachment) {
		desc.usage |= MTLTextureUsageRenderTarget;
	}

	if (p_format.usage_bits & TEXTURE_USAGE_INPUT_ATTACHMENT_BIT) {
		desc.usage |= MTLTextureUsageShaderRead;
	}

	if (p_format.usage_bits & TEXTURE_USAGE_STORAGE_ATOMIC_BIT) {
		ERR_FAIL_COND_V_MSG((format_caps & kMTLFmtCapsAtomic) == 0, RDD::TextureID(), "Atomic operations on this texture format are not supported.");
		ERR_FAIL_COND_V_MSG(!device_properties->features.supports_native_image_atomics, RDD::TextureID(), "Atomic operations on textures are not supported on this OS version. Check SUPPORTS_IMAGE_ATOMIC_32_BIT.");
		// If supports_native_image_atomics is true, this condition should always succeed, as it is set the same.
		if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
			desc.usage |= MTLTextureUsageShaderAtomic;
		}
	}

	if (p_format.usage_bits & TEXTURE_USAGE_VRS_ATTACHMENT_BIT) {
		ERR_FAIL_V_MSG(RDD::TextureID(), "unsupported: TEXTURE_USAGE_VRS_ATTACHMENT_BIT");
	}

	if (flags::any(p_format.usage_bits, TEXTURE_USAGE_CAN_UPDATE_BIT | TEXTURE_USAGE_CAN_COPY_TO_BIT) &&
			can_be_attachment && no_swizzle) {
		// Per MoltenVK, can be cleared as a render attachment.
		desc.usage |= MTLTextureUsageRenderTarget;
	}
	if (p_format.usage_bits & TEXTURE_USAGE_CAN_COPY_FROM_BIT) {
		// Covered by blits.
	}

	// Create texture views with a different component layout.
	if (!p_format.shareable_formats.is_empty()) {
		desc.usage |= MTLTextureUsagePixelFormatView;
	}

	// Allocate memory.

	id<MTLTexture> obj = nil;
	if (is_linear) {
		// Linear textures are restricted to 2D textures, a single mipmap level and a single array layer.
		MTLPixelFormat pixel_format = desc.pixelFormat;
		size_t row_alignment = get_texel_buffer_alignment_for_format(p_format.format);
		size_t bytes_per_row = formats.getBytesPerRow(pixel_format, p_format.width);
		bytes_per_row = round_up_to_alignment(bytes_per_row, row_alignment);
		size_t bytes_per_layer = formats.getBytesPerLayer(pixel_format, bytes_per_row, p_format.height);
		size_t byte_count = bytes_per_layer * p_format.depth * p_format.array_layers;

		id<MTLBuffer> buf = [device newBufferWithLength:byte_count options:options];
		obj = [buf newTextureWithDescriptor:desc offset:0 bytesPerRow:bytes_per_row];
	} else {
		obj = [device newTextureWithDescriptor:desc];
	}
	ERR_FAIL_NULL_V_MSG(obj, TextureID(), "Unable to create texture.");

	return rid::make(obj);
}

RDD::TextureID RenderingDeviceDriverMetal::texture_create_from_extension(uint64_t p_native_texture, TextureType p_type, DataFormat p_format, uint32_t p_array_layers, bool p_depth_stencil, uint32_t p_mipmaps) {
	id<MTLTexture> res = (__bridge id<MTLTexture>)(void *)(uintptr_t)p_native_texture;

	// If the requested format is different, we need to create a view.
	MTLPixelFormat format = pixel_formats->getMTLPixelFormat(p_format);
	if (res.pixelFormat != format) {
		MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(
				MTLTextureSwizzleRed,
				MTLTextureSwizzleGreen,
				MTLTextureSwizzleBlue,
				MTLTextureSwizzleAlpha);
		res = [res newTextureViewWithPixelFormat:format
									 textureType:res.textureType
										  levels:NSMakeRange(0, res.mipmapLevelCount)
										  slices:NSMakeRange(0, p_array_layers)
										 swizzle:swizzle];
		ERR_FAIL_NULL_V_MSG(res, TextureID(), "Unable to create texture view.");
	}

	return rid::make(res);
}

RDD::TextureID RenderingDeviceDriverMetal::texture_create_shared(TextureID p_original_texture, const TextureView &p_view) {
	id<MTLTexture> src_texture = rid::get(p_original_texture);

	NSUInteger slices = src_texture.arrayLength;
	if (src_texture.textureType == MTLTextureTypeCube) {
		// Metal expects Cube textures to have a slice count of 6.
		slices = 6;
	} else if (src_texture.textureType == MTLTextureTypeCubeArray) {
		// Metal expects Cube Array textures to have 6 slices per layer.
		slices *= 6;
	}

#if DEV_ENABLED
	if (src_texture.sampleCount > 1) {
		// TODO(sgc): is it ok to create a shared texture from a multi-sample texture?
		WARN_PRINT("Is it safe to create a shared texture from multi-sample texture?");
	}
#endif

	MTLPixelFormat format = pixel_formats->getMTLPixelFormat(p_view.format);

	static const MTLTextureSwizzle component_swizzle[TEXTURE_SWIZZLE_MAX] = {
		static_cast<MTLTextureSwizzle>(255), // IDENTITY
		MTLTextureSwizzleZero,
		MTLTextureSwizzleOne,
		MTLTextureSwizzleRed,
		MTLTextureSwizzleGreen,
		MTLTextureSwizzleBlue,
		MTLTextureSwizzleAlpha,
	};

#define SWIZZLE(C, CHAN) (p_view.swizzle_##C != TEXTURE_SWIZZLE_IDENTITY ? component_swizzle[p_view.swizzle_##C] : MTLTextureSwizzle##CHAN)
	MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(
			SWIZZLE(r, Red),
			SWIZZLE(g, Green),
			SWIZZLE(b, Blue),
			SWIZZLE(a, Alpha));
#undef SWIZZLE
	id<MTLTexture> obj = [src_texture newTextureViewWithPixelFormat:format
														textureType:src_texture.textureType
															 levels:NSMakeRange(0, src_texture.mipmapLevelCount)
															 slices:NSMakeRange(0, slices)
															swizzle:swizzle];
	ERR_FAIL_NULL_V_MSG(obj, TextureID(), "Unable to create shared texture");
	return rid::make(obj);
}

RDD::TextureID RenderingDeviceDriverMetal::texture_create_shared_from_slice(TextureID p_original_texture, const TextureView &p_view, TextureSliceType p_slice_type, uint32_t p_layer, uint32_t p_layers, uint32_t p_mipmap, uint32_t p_mipmaps) {
	id<MTLTexture> src_texture = rid::get(p_original_texture);

	static const MTLTextureType VIEW_TYPES[] = {
		MTLTextureType1D, // MTLTextureType1D
		MTLTextureType1D, // MTLTextureType1DArray
		MTLTextureType2D, // MTLTextureType2D
		MTLTextureType2D, // MTLTextureType2DArray
		MTLTextureType2D, // MTLTextureType2DMultisample
		MTLTextureType2D, // MTLTextureTypeCube
		MTLTextureType2D, // MTLTextureTypeCubeArray
		MTLTextureType2D, // MTLTextureType3D
		MTLTextureType2D, // MTLTextureType2DMultisampleArray
	};

	MTLTextureType textureType = VIEW_TYPES[src_texture.textureType];
	switch (p_slice_type) {
		case TEXTURE_SLICE_2D: {
			textureType = MTLTextureType2D;
		} break;
		case TEXTURE_SLICE_3D: {
			textureType = MTLTextureType3D;
		} break;
		case TEXTURE_SLICE_CUBEMAP: {
			textureType = MTLTextureTypeCube;
		} break;
		case TEXTURE_SLICE_2D_ARRAY: {
			textureType = MTLTextureType2DArray;
		} break;
		case TEXTURE_SLICE_MAX: {
			ERR_FAIL_V_MSG(TextureID(), "Invalid texture slice type");
		} break;
	}

	MTLPixelFormat format = pixel_formats->getMTLPixelFormat(p_view.format);

	static const MTLTextureSwizzle component_swizzle[TEXTURE_SWIZZLE_MAX] = {
		static_cast<MTLTextureSwizzle>(255), // IDENTITY
		MTLTextureSwizzleZero,
		MTLTextureSwizzleOne,
		MTLTextureSwizzleRed,
		MTLTextureSwizzleGreen,
		MTLTextureSwizzleBlue,
		MTLTextureSwizzleAlpha,
	};

#define SWIZZLE(C, CHAN) (p_view.swizzle_##C != TEXTURE_SWIZZLE_IDENTITY ? component_swizzle[p_view.swizzle_##C] : MTLTextureSwizzle##CHAN)
	MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(
			SWIZZLE(r, Red),
			SWIZZLE(g, Green),
			SWIZZLE(b, Blue),
			SWIZZLE(a, Alpha));
#undef SWIZZLE
	id<MTLTexture> obj = [src_texture newTextureViewWithPixelFormat:format
														textureType:textureType
															 levels:NSMakeRange(p_mipmap, p_mipmaps)
															 slices:NSMakeRange(p_layer, p_layers)
															swizzle:swizzle];
	ERR_FAIL_NULL_V_MSG(obj, TextureID(), "Unable to create shared texture");
	return rid::make(obj);
}

void RenderingDeviceDriverMetal::texture_free(TextureID p_texture) {
	rid::release(p_texture);
}

uint64_t RenderingDeviceDriverMetal::texture_get_allocation_size(TextureID p_texture) {
	id<MTLTexture> __unsafe_unretained obj = rid::get(p_texture);
	return obj.allocatedSize;
}

void RenderingDeviceDriverMetal::texture_get_copyable_layout(TextureID p_texture, const TextureSubresource &p_subresource, TextureCopyableLayout *r_layout) {
	id<MTLTexture> __unsafe_unretained obj = rid::get(p_texture);

	PixelFormats &pf = *pixel_formats;
	DataFormat format = pf.getDataFormat(obj.pixelFormat);

	uint32_t w = MAX(1u, obj.width >> p_subresource.mipmap);
	uint32_t h = MAX(1u, obj.height >> p_subresource.mipmap);
	uint32_t d = MAX(1u, obj.depth >> p_subresource.mipmap);

	uint32_t bw = 0, bh = 0;
	get_compressed_image_format_block_dimensions(format, bw, bh);

	uint32_t sbw = 0, sbh = 0;
	*r_layout = {};
	r_layout->size = get_image_format_required_size(format, w, h, d, 1, &sbw, &sbh);
	r_layout->row_pitch = r_layout->size / ((sbh / bh) * d);
}

Vector<uint8_t> RenderingDeviceDriverMetal::texture_get_data(TextureID p_texture, uint32_t p_layer) {
	id<MTLTexture> obj = rid::get(p_texture);
	ERR_FAIL_COND_V_MSG(obj.storageMode != MTLStorageModeShared, Vector<uint8_t>(), "Texture must be created with TEXTURE_USAGE_CPU_READ_BIT set.");

	if (obj.buffer) {
		ERR_FAIL_COND_V_MSG(p_layer > 0, Vector<uint8_t>(), "A linear texture has a single layer.");
		ERR_FAIL_COND_V_MSG(obj.mipmapLevelCount > 1, Vector<uint8_t>(), "A linear texture has a single mipmap level.");
		Vector<uint8_t> image_data;
		image_data.resize_uninitialized(obj.buffer.length);
		memcpy(image_data.ptrw(), obj.buffer.contents, obj.buffer.length);
		return image_data;
	}

	DataFormat tex_format = pixel_formats->getDataFormat(obj.pixelFormat);
	uint32_t tex_w = obj.width;
	uint32_t tex_h = obj.height;
	uint32_t tex_d = obj.depth;
	uint32_t tex_mipmaps = obj.mipmapLevelCount;

	// Must iteratively copy the texture data to a buffer.

	uint32_t tight_mip_size = get_image_format_required_size(tex_format, tex_w, tex_h, tex_d, tex_mipmaps);

	Vector<uint8_t> image_data;
	image_data.resize(tight_mip_size);

	uint32_t pixel_size = get_image_format_pixel_size(tex_format);
	uint32_t pixel_rshift = get_compressed_image_format_pixel_rshift(tex_format);
	uint32_t blockw = 0, blockh = 0;
	get_compressed_image_format_block_dimensions(tex_format, blockw, blockh);

	uint8_t *dest_ptr = image_data.ptrw();

	for (uint32_t mm_i = 0; mm_i < tex_mipmaps; mm_i++) {
		uint32_t bw = STEPIFY(tex_w, blockw);
		uint32_t bh = STEPIFY(tex_h, blockh);

		uint32_t bytes_per_row = (bw * pixel_size) >> pixel_rshift;
		uint32_t bytes_per_img = bytes_per_row * bh;
		uint32_t mip_size = bytes_per_img * tex_d;

		[obj getBytes:(void *)dest_ptr
				  bytesPerRow:bytes_per_row
				bytesPerImage:bytes_per_img
				   fromRegion:MTLRegionMake3D(0, 0, 0, bw, bh, tex_d)
				  mipmapLevel:mm_i
						slice:p_layer];

		dest_ptr += mip_size;

		// Next mipmap level.
		tex_w = MAX(blockw, tex_w >> 1);
		tex_h = MAX(blockh, tex_h >> 1);
		tex_d = MAX(1u, tex_d >> 1);
	}

	// Ensure that the destination pointer is at the end of the image data.
	DEV_ASSERT(dest_ptr - image_data.ptr() == image_data.size());

	return image_data;
}

BitField<RDD::TextureUsageBits> RenderingDeviceDriverMetal::texture_get_usages_supported_by_format(DataFormat p_format, bool p_cpu_readable) {
	PixelFormats &pf = *pixel_formats;
	if (pf.getMTLPixelFormat(p_format) == MTLPixelFormatInvalid) {
		return 0;
	}

	MTLFmtCaps caps = pf.getCapabilities(p_format);

	// Everything supported by default makes an all-or-nothing check easier for the caller.
	BitField<RDD::TextureUsageBits> supported = INT64_MAX;
	supported.clear_flag(TEXTURE_USAGE_VRS_ATTACHMENT_BIT); // No VRS support for Metal.

	if (!flags::any(caps, kMTLFmtCapsColorAtt)) {
		supported.clear_flag(TEXTURE_USAGE_COLOR_ATTACHMENT_BIT);
	}
	if (!flags::any(caps, kMTLFmtCapsDSAtt)) {
		supported.clear_flag(TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}
	if (!flags::any(caps, kMTLFmtCapsRead)) {
		supported.clear_flag(TEXTURE_USAGE_SAMPLING_BIT);
	}
	if (!flags::any(caps, kMTLFmtCapsAtomic)) {
		supported.clear_flag(TEXTURE_USAGE_STORAGE_ATOMIC_BIT);
	}

	return supported;
}

bool RenderingDeviceDriverMetal::texture_can_make_shared_with_format(TextureID p_texture, DataFormat p_format, bool &r_raw_reinterpretation) {
	r_raw_reinterpretation = false;
	return true;
}

#pragma mark - Sampler

static const MTLCompareFunction COMPARE_OPERATORS[RD::COMPARE_OP_MAX] = {
	MTLCompareFunctionNever,
	MTLCompareFunctionLess,
	MTLCompareFunctionEqual,
	MTLCompareFunctionLessEqual,
	MTLCompareFunctionGreater,
	MTLCompareFunctionNotEqual,
	MTLCompareFunctionGreaterEqual,
	MTLCompareFunctionAlways,
};

static const MTLStencilOperation STENCIL_OPERATIONS[RD::STENCIL_OP_MAX] = {
	MTLStencilOperationKeep,
	MTLStencilOperationZero,
	MTLStencilOperationReplace,
	MTLStencilOperationIncrementClamp,
	MTLStencilOperationDecrementClamp,
	MTLStencilOperationInvert,
	MTLStencilOperationIncrementWrap,
	MTLStencilOperationDecrementWrap,
};

static const MTLBlendFactor BLEND_FACTORS[RD::BLEND_FACTOR_MAX] = {
	MTLBlendFactorZero,
	MTLBlendFactorOne,
	MTLBlendFactorSourceColor,
	MTLBlendFactorOneMinusSourceColor,
	MTLBlendFactorDestinationColor,
	MTLBlendFactorOneMinusDestinationColor,
	MTLBlendFactorSourceAlpha,
	MTLBlendFactorOneMinusSourceAlpha,
	MTLBlendFactorDestinationAlpha,
	MTLBlendFactorOneMinusDestinationAlpha,
	MTLBlendFactorBlendColor,
	MTLBlendFactorOneMinusBlendColor,
	MTLBlendFactorBlendAlpha,
	MTLBlendFactorOneMinusBlendAlpha,
	MTLBlendFactorSourceAlphaSaturated,
	MTLBlendFactorSource1Color,
	MTLBlendFactorOneMinusSource1Color,
	MTLBlendFactorSource1Alpha,
	MTLBlendFactorOneMinusSource1Alpha,
};
static const MTLBlendOperation BLEND_OPERATIONS[RD::BLEND_OP_MAX] = {
	MTLBlendOperationAdd,
	MTLBlendOperationSubtract,
	MTLBlendOperationReverseSubtract,
	MTLBlendOperationMin,
	MTLBlendOperationMax,
};

static const API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MTLSamplerAddressMode ADDRESS_MODES[RD::SAMPLER_REPEAT_MODE_MAX] = {
	MTLSamplerAddressModeRepeat,
	MTLSamplerAddressModeMirrorRepeat,
	MTLSamplerAddressModeClampToEdge,
	MTLSamplerAddressModeClampToBorderColor,
	MTLSamplerAddressModeMirrorClampToEdge,
};

static const API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MTLSamplerBorderColor SAMPLER_BORDER_COLORS[RD::SAMPLER_BORDER_COLOR_MAX] = {
	MTLSamplerBorderColorTransparentBlack,
	MTLSamplerBorderColorTransparentBlack,
	MTLSamplerBorderColorOpaqueBlack,
	MTLSamplerBorderColorOpaqueBlack,
	MTLSamplerBorderColorOpaqueWhite,
	MTLSamplerBorderColorOpaqueWhite,
};

RDD::SamplerID RenderingDeviceDriverMetal::sampler_create(const SamplerState &p_state) {
	MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
	desc.supportArgumentBuffers = YES;

	desc.magFilter = p_state.mag_filter == SAMPLER_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
	desc.minFilter = p_state.min_filter == SAMPLER_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
	desc.mipFilter = p_state.mip_filter == SAMPLER_FILTER_LINEAR ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;

	desc.sAddressMode = ADDRESS_MODES[p_state.repeat_u];
	desc.tAddressMode = ADDRESS_MODES[p_state.repeat_v];
	desc.rAddressMode = ADDRESS_MODES[p_state.repeat_w];

	if (p_state.use_anisotropy) {
		desc.maxAnisotropy = p_state.anisotropy_max;
	}

	desc.compareFunction = COMPARE_OPERATORS[p_state.compare_op];

	desc.lodMinClamp = p_state.min_lod;
	desc.lodMaxClamp = p_state.max_lod;

	desc.borderColor = SAMPLER_BORDER_COLORS[p_state.border_color];

	desc.normalizedCoordinates = !p_state.unnormalized_uvw;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000 || __IPHONE_OS_VERSION_MAX_ALLOWED >= 260000 || __TV_OS_VERSION_MAX_ALLOWED >= 260000 || __VISION_OS_VERSION_MAX_ALLOWED >= 260000
	if (p_state.lod_bias != 0.0) {
		if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)) {
			desc.lodBias = p_state.lod_bias;
		}
	}
#endif

	id<MTLSamplerState> obj = [device newSamplerStateWithDescriptor:desc];
	ERR_FAIL_NULL_V_MSG(obj, SamplerID(), "newSamplerStateWithDescriptor failed");
	return rid::make(obj);
}

void RenderingDeviceDriverMetal::sampler_free(SamplerID p_sampler) {
	rid::release(p_sampler);
}

bool RenderingDeviceDriverMetal::sampler_is_format_supported_for_filter(DataFormat p_format, SamplerFilter p_filter) {
	switch (p_filter) {
		case SAMPLER_FILTER_NEAREST:
			return true;
		case SAMPLER_FILTER_LINEAR: {
			MTLFmtCaps caps = pixel_formats->getCapabilities(p_format);
			return flags::any(caps, kMTLFmtCapsFilter);
		}
	}
}

#pragma mark - Vertex Array

RDD::VertexFormatID RenderingDeviceDriverMetal::vertex_format_create(Span<VertexAttribute> p_vertex_attribs, const VertexAttributeBindingsMap &p_vertex_bindings) {
	MTLVertexDescriptor *desc = MTLVertexDescriptor.vertexDescriptor;

	for (const VertexAttributeBindingsMap::KV &kv : p_vertex_bindings) {
		uint32_t idx = get_metal_buffer_index_for_vertex_attribute_binding(kv.key);
		MTLVertexBufferLayoutDescriptor *ld = desc.layouts[idx];
		if (kv.value.stride != 0) {
			ld.stepFunction = kv.value.frequency == VERTEX_FREQUENCY_VERTEX ? MTLVertexStepFunctionPerVertex : MTLVertexStepFunctionPerInstance;
			ld.stepRate = 1;
			ld.stride = kv.value.stride;
		} else {
			ld.stepFunction = MTLVertexStepFunctionConstant;
			ld.stepRate = 0;
			ld.stride = 0;
		}
		DEV_ASSERT(ld.stride == desc.layouts[idx].stride);
	}

	for (const VertexAttribute &vf : p_vertex_attribs) {
		desc.attributes[vf.location].format = pixel_formats->getMTLVertexFormat(vf.format);
		desc.attributes[vf.location].offset = vf.offset;
		uint32_t idx = get_metal_buffer_index_for_vertex_attribute_binding(vf.binding);
		desc.attributes[vf.location].bufferIndex = idx;
		if (vf.stride == 0) {
			// Constant attribute, so we must determine the stride to satisfy Metal API.
			uint32_t stride = desc.layouts[idx].stride;
			desc.layouts[idx].stride = std::max(stride, vf.offset + pixel_formats->getBytesPerBlock(vf.format));
		}
	}

	return rid::make(desc);
}

void RenderingDeviceDriverMetal::vertex_format_free(VertexFormatID p_vertex_format) {
	rid::release(p_vertex_format);
}

#pragma mark - Barriers

void RenderingDeviceDriverMetal::command_pipeline_barrier(
		CommandBufferID p_cmd_buffer,
		BitField<PipelineStageBits> p_src_stages,
		BitField<PipelineStageBits> p_dst_stages,
		VectorView<MemoryAccessBarrier> p_memory_barriers,
		VectorView<BufferBarrier> p_buffer_barriers,
		VectorView<TextureBarrier> p_texture_barriers) {
	WARN_PRINT_ONCE("not implemented");
}

#pragma mark - Fences

RDD::FenceID RenderingDeviceDriverMetal::fence_create() {
	Fence *fence = nullptr;
	if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, visionOS 1.0, *)) {
		fence = memnew(FenceEvent([device newSharedEvent]));
	} else {
		fence = memnew(FenceSemaphore());
	}
	return FenceID(fence);
}

Error RenderingDeviceDriverMetal::fence_wait(FenceID p_fence) {
	Fence *fence = (Fence *)(p_fence.id);
	return fence->wait(1000);
}

void RenderingDeviceDriverMetal::fence_free(FenceID p_fence) {
	Fence *fence = (Fence *)(p_fence.id);
	memdelete(fence);
}

#pragma mark - Semaphores

RDD::SemaphoreID RenderingDeviceDriverMetal::semaphore_create() {
	// Metal doesn't use semaphores, as their purpose within Godot is to ensure ordering of command buffer execution.
	return SemaphoreID(1);
}

void RenderingDeviceDriverMetal::semaphore_free(SemaphoreID p_semaphore) {
}

#pragma mark - Queues

RDD::CommandQueueFamilyID RenderingDeviceDriverMetal::command_queue_family_get(BitField<CommandQueueFamilyBits> p_cmd_queue_family_bits, RenderingContextDriver::SurfaceID p_surface) {
	if (p_cmd_queue_family_bits.has_flag(COMMAND_QUEUE_FAMILY_GRAPHICS_BIT) || (p_surface != 0)) {
		return CommandQueueFamilyID(COMMAND_QUEUE_FAMILY_GRAPHICS_BIT);
	} else if (p_cmd_queue_family_bits.has_flag(COMMAND_QUEUE_FAMILY_COMPUTE_BIT)) {
		return CommandQueueFamilyID(COMMAND_QUEUE_FAMILY_COMPUTE_BIT);
	} else if (p_cmd_queue_family_bits.has_flag(COMMAND_QUEUE_FAMILY_TRANSFER_BIT)) {
		return CommandQueueFamilyID(COMMAND_QUEUE_FAMILY_TRANSFER_BIT);
	} else {
		return CommandQueueFamilyID();
	}
}

RDD::CommandQueueID RenderingDeviceDriverMetal::command_queue_create(CommandQueueFamilyID p_cmd_queue_family, bool p_identify_as_main_queue) {
	return CommandQueueID(1);
}

Error RenderingDeviceDriverMetal::command_queue_execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID>, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID>, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains) {
	uint32_t size = p_cmd_buffers.size();
	if (size == 0) {
		return OK;
	}

	for (uint32_t i = 0; i < size - 1; i++) {
		MDCommandBuffer *cmd_buffer = (MDCommandBuffer *)(p_cmd_buffers[i].id);
		cmd_buffer->commit();
	}

	// The last command buffer will signal the fence and semaphores.
	MDCommandBuffer *cmd_buffer = (MDCommandBuffer *)(p_cmd_buffers[size - 1].id);
	Fence *fence = (Fence *)(p_cmd_fence.id);
	if (fence != nullptr) {
		cmd_buffer->end();
		id<MTLCommandBuffer> cb = cmd_buffer->get_command_buffer();
		fence->signal(cb);
	}

	for (uint32_t i = 0; i < p_swap_chains.size(); i++) {
		SwapChain *swap_chain = (SwapChain *)(p_swap_chains[i].id);
		RenderingContextDriverMetal::Surface *metal_surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
		metal_surface->present(cmd_buffer);
	}

	cmd_buffer->commit();

	if (p_swap_chains.size() > 0) {
		// Used as a signal that we're presenting, so this is the end of a frame.
		[device_scope endScope];
		[device_scope beginScope];
	}

	return OK;
}

void RenderingDeviceDriverMetal::command_queue_free(CommandQueueID p_cmd_queue) {
}

#pragma mark - Command Buffers

// ----- POOL -----

RDD::CommandPoolID RenderingDeviceDriverMetal::command_pool_create(CommandQueueFamilyID p_cmd_queue_family, CommandBufferType p_cmd_buffer_type) {
	DEV_ASSERT(p_cmd_buffer_type == COMMAND_BUFFER_TYPE_PRIMARY);
	return rid::make(device_queue);
}

bool RenderingDeviceDriverMetal::command_pool_reset(CommandPoolID p_cmd_pool) {
	return true;
}

void RenderingDeviceDriverMetal::command_pool_free(CommandPoolID p_cmd_pool) {
	rid::release(p_cmd_pool);
}

// ----- BUFFER -----

RDD::CommandBufferID RenderingDeviceDriverMetal::command_buffer_create(CommandPoolID p_cmd_pool) {
	id<MTLCommandQueue> queue = rid::get(p_cmd_pool);
	MDCommandBuffer *obj = new MDCommandBuffer(queue, this);
	command_buffers.push_back(obj);
	return CommandBufferID(obj);
}

bool RenderingDeviceDriverMetal::command_buffer_begin(CommandBufferID p_cmd_buffer) {
	MDCommandBuffer *obj = (MDCommandBuffer *)(p_cmd_buffer.id);
	obj->begin();
	return true;
}

bool RenderingDeviceDriverMetal::command_buffer_begin_secondary(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, uint32_t p_subpass, FramebufferID p_framebuffer) {
	ERR_FAIL_V_MSG(false, "not implemented");
}

void RenderingDeviceDriverMetal::command_buffer_end(CommandBufferID p_cmd_buffer) {
	MDCommandBuffer *obj = (MDCommandBuffer *)(p_cmd_buffer.id);
	obj->end();
}

void RenderingDeviceDriverMetal::command_buffer_execute_secondary(CommandBufferID p_cmd_buffer, VectorView<CommandBufferID> p_secondary_cmd_buffers) {
	ERR_FAIL_MSG("not implemented");
}

#pragma mark - Swap Chain

void RenderingDeviceDriverMetal::_swap_chain_release(SwapChain *p_swap_chain) {
	_swap_chain_release_buffers(p_swap_chain);
}

void RenderingDeviceDriverMetal::_swap_chain_release_buffers(SwapChain *p_swap_chain) {
}

RDD::SwapChainID RenderingDeviceDriverMetal::swap_chain_create(RenderingContextDriver::SurfaceID p_surface) {
	RenderingContextDriverMetal::Surface const *surface = (RenderingContextDriverMetal::Surface *)(p_surface);

	// Create the render pass that will be used to draw to the swap chain's framebuffers.
	RDD::Attachment attachment;
	attachment.format = pixel_formats->getDataFormat(surface->get_pixel_format());
	attachment.samples = RDD::TEXTURE_SAMPLES_1;
	attachment.load_op = RDD::ATTACHMENT_LOAD_OP_CLEAR;
	attachment.store_op = RDD::ATTACHMENT_STORE_OP_STORE;

	RDD::Subpass subpass;
	RDD::AttachmentReference color_ref;
	color_ref.attachment = 0;
	color_ref.aspect.set_flag(RDD::TEXTURE_ASPECT_COLOR_BIT);
	subpass.color_references.push_back(color_ref);

	RenderPassID render_pass = render_pass_create(attachment, subpass, {}, 1, RDD::AttachmentReference());
	ERR_FAIL_COND_V(!render_pass, SwapChainID());

	// Create the empty swap chain until it is resized.
	SwapChain *swap_chain = memnew(SwapChain);
	swap_chain->surface = p_surface;
	swap_chain->data_format = attachment.format;
	swap_chain->render_pass = render_pass;
	return SwapChainID(swap_chain);
}

Error RenderingDeviceDriverMetal::swap_chain_resize(CommandQueueID p_cmd_queue, SwapChainID p_swap_chain, uint32_t p_desired_framebuffer_count) {
	DEV_ASSERT(p_cmd_queue.id != 0);
	DEV_ASSERT(p_swap_chain.id != 0);

	SwapChain *swap_chain = (SwapChain *)(p_swap_chain.id);
	RenderingContextDriverMetal::Surface *surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
	surface->resize(p_desired_framebuffer_count);

	// Once everything's been created correctly, indicate the surface no longer needs to be resized.
	context_driver->surface_set_needs_resize(swap_chain->surface, false);

	return OK;
}

RDD::FramebufferID RenderingDeviceDriverMetal::swap_chain_acquire_framebuffer(CommandQueueID p_cmd_queue, SwapChainID p_swap_chain, bool &r_resize_required) {
	DEV_ASSERT(p_cmd_queue.id != 0);
	DEV_ASSERT(p_swap_chain.id != 0);

	SwapChain *swap_chain = (SwapChain *)(p_swap_chain.id);
	if (context_driver->surface_get_needs_resize(swap_chain->surface)) {
		r_resize_required = true;
		return FramebufferID();
	}

	RenderingContextDriverMetal::Surface *metal_surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
	return metal_surface->acquire_next_frame_buffer();
}

RDD::RenderPassID RenderingDeviceDriverMetal::swap_chain_get_render_pass(SwapChainID p_swap_chain) {
	const SwapChain *swap_chain = (const SwapChain *)(p_swap_chain.id);
	return swap_chain->render_pass;
}

RDD::DataFormat RenderingDeviceDriverMetal::swap_chain_get_format(SwapChainID p_swap_chain) {
	const SwapChain *swap_chain = (const SwapChain *)(p_swap_chain.id);
	return swap_chain->data_format;
}

void RenderingDeviceDriverMetal::swap_chain_set_max_fps(SwapChainID p_swap_chain, int p_max_fps) {
	SwapChain *swap_chain = (SwapChain *)(p_swap_chain.id);
	RenderingContextDriverMetal::Surface *metal_surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
	metal_surface->set_max_fps(p_max_fps);
}

void RenderingDeviceDriverMetal::swap_chain_free(SwapChainID p_swap_chain) {
	SwapChain *swap_chain = (SwapChain *)(p_swap_chain.id);
	_swap_chain_release(swap_chain);
	render_pass_free(swap_chain->render_pass);
	memdelete(swap_chain);
}

#pragma mark - Frame buffer

RDD::FramebufferID RenderingDeviceDriverMetal::framebuffer_create(RenderPassID p_render_pass, VectorView<TextureID> p_attachments, uint32_t p_width, uint32_t p_height) {
	MDRenderPass *pass = (MDRenderPass *)(p_render_pass.id);

	Vector<MTL::Texture> textures;
	textures.resize(p_attachments.size());

	for (uint32_t i = 0; i < p_attachments.size(); i += 1) {
		MDAttachment const &a = pass->attachments[i];
		id<MTLTexture> tex = rid::get(p_attachments[i]);
		if (tex == nil) {
#if DEV_ENABLED
			WARN_PRINT("Invalid texture for attachment " + itos(i));
#endif
		}
		if (a.samples > 1) {
			if (tex.sampleCount != a.samples) {
#if DEV_ENABLED
				WARN_PRINT("Mismatched sample count for attachment " + itos(i) + "; expected " + itos(a.samples) + ", got " + itos(tex.sampleCount));
#endif
			}
		}
		textures.write[i] = tex;
	}

	MDFrameBuffer *fb = new MDFrameBuffer(textures, Size2i(p_width, p_height));
	return FramebufferID(fb);
}

void RenderingDeviceDriverMetal::framebuffer_free(FramebufferID p_framebuffer) {
	MDFrameBuffer *obj = (MDFrameBuffer *)(p_framebuffer.id);
	delete obj;
}

#pragma mark - Shader

void RenderingDeviceDriverMetal::shader_cache_free_entry(const SHA256Digest &key) {
	if (ShaderCacheEntry **pentry = _shader_cache.getptr(key); pentry != nullptr) {
		ShaderCacheEntry *entry = *pentry;
		_shader_cache.erase(key);
		entry->library = nil;
		memdelete(entry);
	}
}

template <typename T, typename U>
struct is_layout_compatible
		: std::bool_constant<
				  sizeof(T) == sizeof(U) &&
				  alignof(T) == alignof(U) &&
				  std::is_trivially_copyable_v<T> &&
				  std::is_trivially_copyable_v<U>> {};
static_assert(is_layout_compatible<UniformInfo::Indexes, RenderingShaderContainerMetal::UniformData::Indexes>::value, "UniformInfo::Indexes layout does not match RenderingShaderContainerMetal::UniformData::Indexes layout");

API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0))
static void update_uniform_info(const RenderingShaderContainerMetal::UniformData &p_data, UniformInfo &r_ui) {
	r_ui.active_stages = p_data.active_stages;
	r_ui.dataType = static_cast<MTLDataType>(p_data.data_type);
	memcpy(&r_ui.slot, &p_data.slot, sizeof(UniformInfo::Indexes));
	memcpy(&r_ui.arg_buffer, &p_data.arg_buffer, sizeof(UniformInfo::Indexes));
	r_ui.access = static_cast<MTLBindingAccess>(p_data.access);
	r_ui.usage = static_cast<MTLResourceUsage>(p_data.usage);
	r_ui.textureType = static_cast<MTLTextureType>(p_data.texture_type);
	r_ui.imageFormat = p_data.image_format;
	r_ui.arrayLength = p_data.array_length;
	r_ui.isMultisampled = p_data.is_multisampled;
}

RDD::ShaderID RenderingDeviceDriverMetal::shader_create_from_container(const Ref<RenderingShaderContainer> &p_shader_container, const Vector<ImmutableSampler> &p_immutable_samplers) {
	Ref<RenderingShaderContainerMetal> shader_container = p_shader_container;
	using RSCM = RenderingShaderContainerMetal;

	CharString shader_name = shader_container->shader_name;
	RSCM::HeaderData &mtl_reflection_data = shader_container->mtl_reflection_data;
	Vector<RenderingShaderContainer::Shader> &shaders = shader_container->shaders;
	Vector<RSCM::StageData> &mtl_shaders = shader_container->mtl_shaders;

	// We need to regenerate the shader if the cache is moved to an incompatible device or argument buffer support differs.
	ERR_FAIL_COND_V_MSG(!device_properties->features.argument_buffers_supported() && mtl_reflection_data.uses_argument_buffers(),
			RDD::ShaderID(),
			"Shader was compiled with argument buffers enabled, but this device does not support them");

	ERR_FAIL_COND_V_MSG(device_properties->features.msl_max_version < mtl_reflection_data.msl_version,
			RDD::ShaderID(),
			"Shader was compiled for a newer version of Metal");

	MTLGPUFamily compiled_gpu_family = static_cast<MTLGPUFamily>(mtl_reflection_data.profile.gpu);
	ERR_FAIL_COND_V_MSG(device_properties->features.highestFamily < compiled_gpu_family,
			RDD::ShaderID(),
			"Shader was generated for a newer Apple GPU");

	MTLCompileOptions *options = [MTLCompileOptions new];
	uint32_t major = mtl_reflection_data.msl_version / 10000;
	uint32_t minor = (mtl_reflection_data.msl_version / 100) % 100;
	options.languageVersion = MTLLanguageVersion((major << 0x10) + minor);
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
		options.enableLogging = mtl_reflection_data.needs_debug_logging();
	}

	HashMap<RD::ShaderStage, MDLibrary *> libraries;

	bool is_compute = false;
	Vector<uint8_t> decompressed_code;
	for (uint32_t shader_index = 0; shader_index < shaders.size(); shader_index++) {
		const RenderingShaderContainer::Shader &shader = shaders[shader_index];
		const RSCM::StageData &shader_data = mtl_shaders[shader_index];

		if (shader.shader_stage == RD::ShaderStage::SHADER_STAGE_COMPUTE) {
			is_compute = true;
		}

		if (ShaderCacheEntry **p = _shader_cache.getptr(shader_data.hash); p != nullptr) {
			libraries[shader.shader_stage] = (*p)->library;
			continue;
		}

		if (shader.code_decompressed_size > 0) {
			decompressed_code.resize(shader.code_decompressed_size);
			bool decompressed = shader_container->decompress_code(shader.code_compressed_bytes.ptr(), shader.code_compressed_bytes.size(), shader.code_compression_flags, decompressed_code.ptrw(), decompressed_code.size());
			ERR_FAIL_COND_V_MSG(!decompressed, RDD::ShaderID(), vformat("Failed to decompress code on shader stage %s.", String(RDD::SHADER_STAGE_NAMES[shader.shader_stage])));
		} else {
			decompressed_code = shader.code_compressed_bytes;
		}

		ShaderCacheEntry *cd = memnew(ShaderCacheEntry(*this, shader_data.hash));
		cd->name = shader_name;
		cd->stage = shader.shader_stage;

		NSString *source = [[NSString alloc] initWithBytes:(void *)decompressed_code.ptr()
													length:shader_data.source_size
												  encoding:NSUTF8StringEncoding];

		MDLibrary *library = nil;
		if (shader_data.library_size > 0) {
			ERR_FAIL_COND_V_MSG(mtl_reflection_data.os_min_version > device_properties->os_version,
					RDD::ShaderID(),
					"Metal shader binary was generated for a newer target OS");
			dispatch_data_t binary = dispatch_data_create(decompressed_code.ptr() + shader_data.source_size, shader_data.library_size, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
			library = [MDLibrary newLibraryWithCacheEntry:cd
												   device:device
#if DEV_ENABLED
												   source:source
#endif
													 data:binary];
		} else {
			options.preserveInvariance = shader_data.is_position_invariant;
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 150000 || __IPHONE_OS_VERSION_MIN_REQUIRED >= 180000 || __TV_OS_VERSION_MIN_REQUIRED >= 180000 || defined(VISIONOS_ENABLED)
			options.mathMode = MTLMathModeFast;
#else
			options.fastMathEnabled = YES;
#endif
			library = [MDLibrary newLibraryWithCacheEntry:cd
												   device:device
												   source:source
												  options:options
												 strategy:_shader_load_strategy];
		}

		_shader_cache[shader_data.hash] = cd;
		libraries[shader.shader_stage] = library;
	}

	ShaderReflection refl = shader_container->get_shader_reflection();
	RSCM::MetalShaderReflection mtl_refl = shader_container->get_metal_shader_reflection();

	Vector<UniformSet> uniform_sets;
	uint32_t uniform_sets_count = mtl_refl.uniform_sets.size();
	uniform_sets.resize(uniform_sets_count);

	DynamicOffsetLayout dynamic_offset_layout;
	uint8_t dynamic_offset = 0;

	// Create sets.
	for (uint32_t i = 0; i < uniform_sets_count; i++) {
		UniformSet &set = uniform_sets.write[i];
		const Vector<ShaderUniform> &refl_set = refl.uniform_sets.ptr()[i];
		const Vector<RSCM::UniformData> &mtl_set = mtl_refl.uniform_sets.ptr()[i];
		uint32_t set_size = mtl_set.size();
		set.uniforms.resize(set_size);

		uint8_t dynamic_count = 0;

		LocalVector<UniformInfo>::Iterator iter = set.uniforms.begin();
		for (uint32_t j = 0; j < set_size; j++) {
			const ShaderUniform &uniform = refl_set.ptr()[j];
			const RSCM::UniformData &bind = mtl_set.ptr()[j];

			switch (uniform.type) {
				case UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC:
				case UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC: {
					set.dynamic_uniforms.push_back(j);
					dynamic_count++;
				} break;
				default: {
				} break;
			}

			UniformInfo &ui = *iter;
			++iter;
			update_uniform_info(bind, ui);
			ui.binding = uniform.binding;

			if (ui.arg_buffer.texture == UINT32_MAX && ui.arg_buffer.buffer == UINT32_MAX && ui.arg_buffer.sampler == UINT32_MAX) {
				// No bindings.
				continue;
			}
#define VAL(x) (x == UINT32_MAX ? 0 : x)
			uint32_t max = std::max({ VAL(ui.arg_buffer.texture), VAL(ui.arg_buffer.buffer), VAL(ui.arg_buffer.sampler) });
			max += ui.arrayLength > 0 ? ui.arrayLength - 1 : 0;
			set.buffer_size = std::max(set.buffer_size, (max + 1) * (uint32_t)sizeof(uint64_t));
#undef VAL
		}

		if (dynamic_count > 0) {
			dynamic_offset_layout.set_offset_count(i, dynamic_offset, dynamic_count);
			dynamic_offset += dynamic_count;
		}
	}

	MDShader *shader = nullptr;
	if (is_compute) {
		MDComputeShader *cs = new MDComputeShader(
				shader_name,
				uniform_sets,
				mtl_reflection_data.uses_argument_buffers(),
				libraries[RD::ShaderStage::SHADER_STAGE_COMPUTE]);

		cs->local = MTLSizeMake(refl.compute_local_size[0], refl.compute_local_size[1], refl.compute_local_size[2]);
		shader = cs;
	} else {
		MDRenderShader *rs = new MDRenderShader(
				shader_name,
				uniform_sets,
				mtl_reflection_data.needs_view_mask_buffer(),
				mtl_reflection_data.uses_argument_buffers(),
				libraries[RD::ShaderStage::SHADER_STAGE_VERTEX],
				libraries[RD::ShaderStage::SHADER_STAGE_FRAGMENT]);
		shader = rs;
	}

	shader->push_constants.stages = refl.push_constant_stages;
	shader->push_constants.size = refl.push_constant_size;
	shader->push_constants.binding = mtl_reflection_data.push_constant_binding;
	shader->dynamic_offset_layout = dynamic_offset_layout;

	return RDD::ShaderID(shader);
}

void RenderingDeviceDriverMetal::shader_free(ShaderID p_shader) {
	MDShader *obj = (MDShader *)p_shader.id;
	delete obj;
}

void RenderingDeviceDriverMetal::shader_destroy_modules(ShaderID p_shader) {
	// TODO.
}

/*********************/
/**** UNIFORM SET ****/
/*********************/

RDD::UniformSetID RenderingDeviceDriverMetal::uniform_set_create(VectorView<BoundUniform> p_uniforms, ShaderID p_shader, uint32_t p_set_index, int p_linear_pool_index) {
	//p_linear_pool_index = -1; // TODO:? Linear pools not implemented or not supported by API backend.

	MDShader *shader = (MDShader *)(p_shader.id);
	ERR_FAIL_INDEX_V_MSG(p_set_index, shader->sets.size(), UniformSetID(), "Set index out of range");
	const UniformSet &shader_set = shader->sets.get(p_set_index);
	MDUniformSet *set = memnew(MDUniformSet);

	if (device_properties->features.argument_buffers_supported()) {
		// If argument buffers are enabled, we have already verified availability, so we can skip the runtime check.
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability-new")

		set->arg_buffer = [device newBufferWithLength:shader_set.buffer_size options:MTLResourceStorageModeShared];
		uint64_t *ptr = (uint64_t *)set->arg_buffer.contents;

		HashMap<MTLResourceUnsafe, StageResourceUsage, HashMapHasherDefault> bound_resources;
		auto add_usage = [&bound_resources](MTLResourceUnsafe res, BitField<RDD::ShaderStage> stage, MTLResourceUsage usage) {
			StageResourceUsage *sru = bound_resources.getptr(res);
			if (sru == nullptr) {
				sru = &bound_resources.insert(res, ResourceUnused)->value;
			}
			if (stage.has_flag(RDD::SHADER_STAGE_VERTEX_BIT)) {
				*sru |= stage_resource_usage(RDD::SHADER_STAGE_VERTEX, usage);
			}
			if (stage.has_flag(RDD::SHADER_STAGE_FRAGMENT_BIT)) {
				*sru |= stage_resource_usage(RDD::SHADER_STAGE_FRAGMENT, usage);
			}
			if (stage.has_flag(RDD::SHADER_STAGE_COMPUTE_BIT)) {
				*sru |= stage_resource_usage(RDD::SHADER_STAGE_COMPUTE, usage);
			}
		};

		// Ensure the argument buffer exists for this set as some shader pipelines may
		// have been generated with argument buffers enabled.
		for (uint32_t i = 0; i < p_uniforms.size(); i += 1) {
			const BoundUniform &uniform = p_uniforms[i];
			const UniformInfo &ui = shader_set.uniforms[i];
			const UniformInfo::Indexes &idx = ui.arg_buffer;

			switch (uniform.type) {
				case UNIFORM_TYPE_SAMPLER: {
					size_t count = uniform.ids.size();
					for (size_t j = 0; j < count; j += 1) {
						id<MTLSamplerState> sampler = rid::get(uniform.ids[j]);
						*(MTLResourceID *)(ptr + idx.sampler + j) = sampler.gpuResourceID;
					}
				} break;
				case UNIFORM_TYPE_SAMPLER_WITH_TEXTURE: {
					uint32_t count = uniform.ids.size() / 2;
					for (uint32_t j = 0; j < count; j += 1) {
						id<MTLSamplerState> sampler = rid::get(uniform.ids[j * 2 + 0]);
						id<MTLTexture> texture = rid::get(uniform.ids[j * 2 + 1]);
						*(MTLResourceID *)(ptr + idx.texture + j) = texture.gpuResourceID;
						*(MTLResourceID *)(ptr + idx.sampler + j) = sampler.gpuResourceID;

						add_usage(texture, ui.active_stages, ui.usage);
					}
				} break;
				case UNIFORM_TYPE_TEXTURE: {
					size_t count = uniform.ids.size();
					for (size_t j = 0; j < count; j += 1) {
						id<MTLTexture> texture = rid::get(uniform.ids[j]);
						*(MTLResourceID *)(ptr + idx.texture + j) = texture.gpuResourceID;

						add_usage(texture, ui.active_stages, ui.usage);
					}
				} break;
				case UNIFORM_TYPE_IMAGE: {
					size_t count = uniform.ids.size();
					for (size_t j = 0; j < count; j += 1) {
						id<MTLTexture> texture = rid::get(uniform.ids[j]);
						*(MTLResourceID *)(ptr + idx.texture + j) = texture.gpuResourceID;
						add_usage(texture, ui.active_stages, ui.usage);

						if (idx.buffer != UINT32_MAX) {
							// Emulated atomic image access.
							id<MTLBuffer> buffer = (texture.parentTexture ? texture.parentTexture : texture).buffer;
							*(MTLGPUAddress *)(ptr + idx.buffer + j) = buffer.gpuAddress;

							add_usage(buffer, ui.active_stages, ui.usage);
						}
					}
				} break;
				case UNIFORM_TYPE_TEXTURE_BUFFER: {
					ERR_PRINT("not implemented: UNIFORM_TYPE_TEXTURE_BUFFER");
				} break;
				case UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER: {
					ERR_PRINT("not implemented: UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER");
				} break;
				case UNIFORM_TYPE_IMAGE_BUFFER: {
					CRASH_NOW_MSG("not implemented: UNIFORM_TYPE_IMAGE_BUFFER");
				} break;
				case UNIFORM_TYPE_STORAGE_BUFFER:
				case UNIFORM_TYPE_UNIFORM_BUFFER: {
					const BufferInfo *buffer = (const BufferInfo *)uniform.ids[0].id;
					*(MTLGPUAddress *)(ptr + idx.buffer) = buffer->metal_buffer.gpuAddress;

					add_usage(buffer->metal_buffer, ui.active_stages, ui.usage);
				} break;
				case UNIFORM_TYPE_INPUT_ATTACHMENT: {
					size_t count = uniform.ids.size();
					for (size_t j = 0; j < count; j += 1) {
						id<MTLTexture> texture = rid::get(uniform.ids[j]);
						*(MTLResourceID *)(ptr + idx.texture + j) = texture.gpuResourceID;

						add_usage(texture, ui.active_stages, ui.usage);
					}
				} break;
				case UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC:
				case UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC: {
					// Dynamic buffers are not supported by argument buffers currently.
					// so we do not encode them, as there shouldn't be any runtime shaders that used them.
				} break;
				default: {
					DEV_ASSERT(false);
				}
			}
		}

		for (KeyValue<MTLResourceUnsafe, StageResourceUsage> const &keyval : bound_resources) {
			ResourceVector *resources = set->usage_to_resources.getptr(keyval.value);
			if (resources == nullptr) {
				resources = &set->usage_to_resources.insert(keyval.value, ResourceVector())->value;
			}
			int64_t pos = resources->span().bisect(keyval.key, true);
			if (pos == resources->size() || (*resources)[pos] != keyval.key) {
				resources->insert(pos, keyval.key);
			}
		}

		GODOT_CLANG_WARNING_POP
	}
	Vector<BoundUniform> bound_uniforms;
	bound_uniforms.resize(p_uniforms.size());
	for (uint32_t i = 0; i < p_uniforms.size(); i += 1) {
		bound_uniforms.write[i] = p_uniforms[i];
	}
	set->uniforms = bound_uniforms;
	set->index = p_set_index;

	return UniformSetID(set);
}

void RenderingDeviceDriverMetal::uniform_set_free(UniformSetID p_uniform_set) {
	MDUniformSet *obj = (MDUniformSet *)p_uniform_set.id;
	memdelete(obj);
}

uint32_t RenderingDeviceDriverMetal::uniform_sets_get_dynamic_offsets(VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count) const {
	const MDShader *shader = (const MDShader *)p_shader.id;
	const DynamicOffsetLayout layout = shader->dynamic_offset_layout;

	if (layout.is_empty()) {
		return 0u;
	}

	uint32_t mask = 0u;

	for (uint32_t i = 0; i < p_set_count; i++) {
		const uint32_t index = p_first_set_index + i;
		uint32_t shift = layout.get_offset_index_shift(index);
		const uint32_t count = layout.get_count(index);
		DEV_ASSERT(shader->sets[index].dynamic_uniforms.size() == count);
		if (count == 0) {
			continue;
		}

		const MDUniformSet *usi = (const MDUniformSet *)p_uniform_sets[i].id;
		for (uint32_t uniform_index : shader->sets[index].dynamic_uniforms) {
			const RDD::BoundUniform &uniform = usi->uniforms[uniform_index];
			DEV_ASSERT(uniform.is_dynamic());
			const MetalBufferDynamicInfo *buf_info = (const MetalBufferDynamicInfo *)uniform.ids[0].id;
			mask |= buf_info->frame_index() << shift;
			shift += 4u;
		}
	}

	return mask;
}

void RenderingDeviceDriverMetal::command_uniform_set_prepare_for_use(CommandBufferID p_cmd_buffer, UniformSetID p_uniform_set, ShaderID p_shader, uint32_t p_set_index) {
}

#pragma mark - Transfer

void RenderingDeviceDriverMetal::command_clear_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, uint64_t p_offset, uint64_t p_size) {
	MDCommandBuffer *cmd = (MDCommandBuffer *)(p_cmd_buffer.id);
	cmd->clear_buffer(p_buffer, p_offset, p_size);
}

void RenderingDeviceDriverMetal::command_copy_buffer(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, BufferID p_dst_buffer, VectorView<BufferCopyRegion> p_regions) {
	MDCommandBuffer *cmd = (MDCommandBuffer *)(p_cmd_buffer.id);
	cmd->copy_buffer(p_src_buffer, p_dst_buffer, p_regions);
}

void RenderingDeviceDriverMetal::command_copy_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<TextureCopyRegion> p_regions) {
	MDCommandBuffer *cmd = (MDCommandBuffer *)(p_cmd_buffer.id);
	cmd->copy_texture(p_src_texture, p_dst_texture, p_regions);
}

void RenderingDeviceDriverMetal::command_resolve_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->resolve_texture(p_src_texture, p_src_texture_layout, p_src_layer, p_src_mipmap, p_dst_texture, p_dst_texture_layout, p_dst_layer, p_dst_mipmap);
}

void RenderingDeviceDriverMetal::command_clear_color_texture(CommandBufferID p_cmd_buffer, TextureID p_texture, TextureLayout p_texture_layout, const Color &p_color, const TextureSubresourceRange &p_subresources) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->clear_color_texture(p_texture, p_texture_layout, p_color, p_subresources);
}

void RenderingDeviceDriverMetal::command_copy_buffer_to_texture(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<BufferTextureCopyRegion> p_regions) {
	MDCommandBuffer *cmd = (MDCommandBuffer *)(p_cmd_buffer.id);
	cmd->copy_buffer_to_texture(p_src_buffer, p_dst_texture, p_regions);
}

void RenderingDeviceDriverMetal::command_copy_texture_to_buffer(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, BufferID p_dst_buffer, VectorView<BufferTextureCopyRegion> p_regions) {
	MDCommandBuffer *cmd = (MDCommandBuffer *)(p_cmd_buffer.id);
	cmd->copy_texture_to_buffer(p_src_texture, p_dst_buffer, p_regions);
}

#pragma mark - Pipeline

void RenderingDeviceDriverMetal::pipeline_free(PipelineID p_pipeline_id) {
	MDPipeline *obj = (MDPipeline *)(p_pipeline_id.id);
	delete obj;
}

// ----- BINDING -----

void RenderingDeviceDriverMetal::command_bind_push_constants(CommandBufferID p_cmd_buffer, ShaderID p_shader, uint32_t p_dst_first_index, VectorView<uint32_t> p_data) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->encode_push_constant_data(p_shader, p_data);
}

// ----- CACHE -----

String RenderingDeviceDriverMetal::_pipeline_get_cache_path() const {
	String path = OS::get_singleton()->get_user_data_dir() + "/metal/pipelines";
	path += "." + context_device.name.validate_filename().replace_char(' ', '_').to_lower();
	if (Engine::get_singleton()->is_editor_hint()) {
		path += ".editor";
	}
	path += ".cache";

	return path;
}

bool RenderingDeviceDriverMetal::pipeline_cache_create(const Vector<uint8_t> &p_data) {
	return false;
	CharString path = _pipeline_get_cache_path().utf8();
	NSString *nPath = [[NSString alloc] initWithBytesNoCopy:path.ptrw()
													 length:path.length()
												   encoding:NSUTF8StringEncoding
											   freeWhenDone:NO];
	MTLBinaryArchiveDescriptor *desc = [MTLBinaryArchiveDescriptor new];
	if ([[NSFileManager defaultManager] fileExistsAtPath:nPath]) {
		desc.url = [NSURL fileURLWithPath:nPath];
	}
	NSError *error = nil;
	archive = [device newBinaryArchiveWithDescriptor:desc error:&error];
	return true;
}

void RenderingDeviceDriverMetal::pipeline_cache_free() {
	archive = nil;
}

size_t RenderingDeviceDriverMetal::pipeline_cache_query_size() {
	return archive_count * 1024;
}

Vector<uint8_t> RenderingDeviceDriverMetal::pipeline_cache_serialize() {
	if (!archive) {
		return Vector<uint8_t>();
	}

	CharString path = _pipeline_get_cache_path().utf8();

	NSString *nPath = [[NSString alloc] initWithBytesNoCopy:path.ptrw()
													 length:path.length()
												   encoding:NSUTF8StringEncoding
											   freeWhenDone:NO];
	NSURL *target = [NSURL fileURLWithPath:nPath];
	NSError *error = nil;
	if ([archive serializeToURL:target error:&error]) {
		return Vector<uint8_t>();
	} else {
		print_line(error.localizedDescription.UTF8String);
		return Vector<uint8_t>();
	}
}

#pragma mark - Rendering

// ----- SUBPASS -----

RDD::RenderPassID RenderingDeviceDriverMetal::render_pass_create(VectorView<Attachment> p_attachments, VectorView<Subpass> p_subpasses, VectorView<SubpassDependency> p_subpass_dependencies, uint32_t p_view_count, AttachmentReference p_fragment_density_map_attachment) {
	PixelFormats &pf = *pixel_formats;

	size_t subpass_count = p_subpasses.size();

	Vector<MDSubpass> subpasses;
	subpasses.resize(subpass_count);
	for (uint32_t i = 0; i < subpass_count; i++) {
		MDSubpass &subpass = subpasses.write[i];
		subpass.subpass_index = i;
		subpass.view_count = p_view_count;
		subpass.input_references = p_subpasses[i].input_references;
		subpass.color_references = p_subpasses[i].color_references;
		subpass.depth_stencil_reference = p_subpasses[i].depth_stencil_reference;
		subpass.resolve_references = p_subpasses[i].resolve_references;
	}

	static const MTLLoadAction LOAD_ACTIONS[] = {
		[ATTACHMENT_LOAD_OP_LOAD] = MTLLoadActionLoad,
		[ATTACHMENT_LOAD_OP_CLEAR] = MTLLoadActionClear,
		[ATTACHMENT_LOAD_OP_DONT_CARE] = MTLLoadActionDontCare,
	};

	static const MTLStoreAction STORE_ACTIONS[] = {
		[ATTACHMENT_STORE_OP_STORE] = MTLStoreActionStore,
		[ATTACHMENT_STORE_OP_DONT_CARE] = MTLStoreActionDontCare,
	};

	Vector<MDAttachment> attachments;
	attachments.resize(p_attachments.size());

	for (uint32_t i = 0; i < p_attachments.size(); i++) {
		Attachment const &a = p_attachments[i];
		MDAttachment &mda = attachments.write[i];
		MTLPixelFormat format = pf.getMTLPixelFormat(a.format);
		mda.format = format;
		if (a.samples > TEXTURE_SAMPLES_1) {
			mda.samples = (*device_properties).find_nearest_supported_sample_count(a.samples);
		}
		mda.loadAction = LOAD_ACTIONS[a.load_op];
		mda.storeAction = STORE_ACTIONS[a.store_op];
		bool is_depth = pf.isDepthFormat(format);
		if (is_depth) {
			mda.type |= MDAttachmentType::Depth;
		}
		bool is_stencil = pf.isStencilFormat(format);
		if (is_stencil) {
			mda.type |= MDAttachmentType::Stencil;
			mda.stencilLoadAction = LOAD_ACTIONS[a.stencil_load_op];
			mda.stencilStoreAction = STORE_ACTIONS[a.stencil_store_op];
		}
		if (!is_depth && !is_stencil) {
			mda.type |= MDAttachmentType::Color;
		}
	}
	MDRenderPass *obj = new MDRenderPass(attachments, subpasses);
	return RenderPassID(obj);
}

void RenderingDeviceDriverMetal::render_pass_free(RenderPassID p_render_pass) {
	MDRenderPass *obj = (MDRenderPass *)(p_render_pass.id);
	delete obj;
}

// ----- COMMANDS -----

void RenderingDeviceDriverMetal::command_begin_render_pass(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, FramebufferID p_framebuffer, CommandBufferType p_cmd_buffer_type, const Rect2i &p_rect, VectorView<RenderPassClearValue> p_clear_values) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_begin_pass(p_render_pass, p_framebuffer, p_cmd_buffer_type, p_rect, p_clear_values);
}

void RenderingDeviceDriverMetal::command_end_render_pass(CommandBufferID p_cmd_buffer) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_end_pass();
}

void RenderingDeviceDriverMetal::command_next_render_subpass(CommandBufferID p_cmd_buffer, CommandBufferType p_cmd_buffer_type) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_next_subpass();
}

void RenderingDeviceDriverMetal::command_render_set_viewport(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_viewports) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_set_viewport(p_viewports);
}

void RenderingDeviceDriverMetal::command_render_set_scissor(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_scissors) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_set_scissor(p_scissors);
}

void RenderingDeviceDriverMetal::command_render_clear_attachments(CommandBufferID p_cmd_buffer, VectorView<AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_clear_attachments(p_attachment_clears, p_rects);
}

void RenderingDeviceDriverMetal::command_bind_render_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->bind_pipeline(p_pipeline);
}

void RenderingDeviceDriverMetal::command_bind_render_uniform_sets(CommandBufferID p_cmd_buffer, VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_bind_uniform_sets(p_uniform_sets, p_shader, p_first_set_index, p_set_count, p_dynamic_offsets);
}

void RenderingDeviceDriverMetal::command_render_draw(CommandBufferID p_cmd_buffer, uint32_t p_vertex_count, uint32_t p_instance_count, uint32_t p_base_vertex, uint32_t p_first_instance) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_draw(p_vertex_count, p_instance_count, p_base_vertex, p_first_instance);
}

void RenderingDeviceDriverMetal::command_render_draw_indexed(CommandBufferID p_cmd_buffer, uint32_t p_index_count, uint32_t p_instance_count, uint32_t p_first_index, int32_t p_vertex_offset, uint32_t p_first_instance) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_draw_indexed(p_index_count, p_instance_count, p_first_index, p_vertex_offset, p_first_instance);
}

void RenderingDeviceDriverMetal::command_render_draw_indexed_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_draw_indexed_indirect(p_indirect_buffer, p_offset, p_draw_count, p_stride);
}

void RenderingDeviceDriverMetal::command_render_draw_indexed_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_draw_indexed_indirect_count(p_indirect_buffer, p_offset, p_count_buffer, p_count_buffer_offset, p_max_draw_count, p_stride);
}

void RenderingDeviceDriverMetal::command_render_draw_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_draw_indirect(p_indirect_buffer, p_offset, p_draw_count, p_stride);
}

void RenderingDeviceDriverMetal::command_render_draw_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_draw_indirect_count(p_indirect_buffer, p_offset, p_count_buffer, p_count_buffer_offset, p_max_draw_count, p_stride);
}

void RenderingDeviceDriverMetal::command_render_bind_vertex_buffers(CommandBufferID p_cmd_buffer, uint32_t p_binding_count, const BufferID *p_buffers, const uint64_t *p_offsets, uint64_t p_dynamic_offsets) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_bind_vertex_buffers(p_binding_count, p_buffers, p_offsets, p_dynamic_offsets);
}

void RenderingDeviceDriverMetal::command_render_bind_index_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, IndexBufferFormat p_format, uint64_t p_offset) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_bind_index_buffer(p_buffer, p_format, p_offset);
}

void RenderingDeviceDriverMetal::command_render_set_blend_constants(CommandBufferID p_cmd_buffer, const Color &p_constants) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->render_set_blend_constants(p_constants);
}

void RenderingDeviceDriverMetal::command_render_set_line_width(CommandBufferID p_cmd_buffer, float p_width) {
	if (!Math::is_equal_approx(p_width, 1.0f)) {
		ERR_FAIL_MSG("Setting line widths other than 1.0 is not supported by the Metal rendering driver.");
	}
}

// ----- PIPELINE -----

RenderingDeviceDriverMetal::Result<id<MTLFunction>> RenderingDeviceDriverMetal::_create_function(MDLibrary *p_library, NSString *p_name, VectorView<PipelineSpecializationConstant> &p_specialization_constants) {
	id<MTLLibrary> library = p_library.library;
	if (!library) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Failed to compile Metal library");
	}

	id<MTLFunction> function = [library newFunctionWithName:p_name];
	ERR_FAIL_NULL_V_MSG(function, ERR_CANT_CREATE, "No function named main0");

	if (function.functionConstantsDictionary.count == 0) {
		return function;
	}

	NSArray<MTLFunctionConstant *> *constants = function.functionConstantsDictionary.allValues;
	bool is_sorted = true;
	for (uint32_t i = 1; i < constants.count; i++) {
		if (constants[i - 1].index > constants[i].index) {
			is_sorted = false;
			break;
		}
	}

	if (!is_sorted) {
		constants = [constants sortedArrayUsingComparator:^NSComparisonResult(MTLFunctionConstant *a, MTLFunctionConstant *b) {
			if (a.index < b.index) {
				return NSOrderedAscending;
			} else if (a.index > b.index) {
				return NSOrderedDescending;
			} else {
				return NSOrderedSame;
			}
		}];
	}

	// Initialize an array of integers representing the indexes of p_specialization_constants
	uint32_t *indexes = (uint32_t *)alloca(p_specialization_constants.size() * sizeof(uint32_t));
	for (uint32_t i = 0; i < p_specialization_constants.size(); i++) {
		indexes[i] = i;
	}
	// Sort the array of integers based on the values in p_specialization_constants
	std::sort(indexes, &indexes[p_specialization_constants.size()], [&](int a, int b) {
		return p_specialization_constants[a].constant_id < p_specialization_constants[b].constant_id;
	});

	MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];
	uint32_t i = 0;
	uint32_t j = 0;
	while (i < constants.count && j < p_specialization_constants.size()) {
		MTLFunctionConstant *curr = constants[i];
		PipelineSpecializationConstant const &sc = p_specialization_constants[indexes[j]];
		if (curr.index == sc.constant_id) {
			switch (curr.type) {
				case MTLDataTypeBool:
				case MTLDataTypeFloat:
				case MTLDataTypeInt:
				case MTLDataTypeUInt: {
					[constantValues setConstantValue:&sc.int_value
												type:curr.type
											 atIndex:sc.constant_id];
				} break;
				default:
					ERR_FAIL_V_MSG(function, "Invalid specialization constant type");
			}
			i++;
			j++;
		} else if (curr.index < sc.constant_id) {
			i++;
		} else {
			j++;
		}
	}

	if (i != constants.count) {
		MTLFunctionConstant *curr = constants[i];
		if (curr.index == R32UI_ALIGNMENT_CONSTANT_ID) {
			uint32_t alignment = 16; // TODO(sgc): is this always correct?
			[constantValues setConstantValue:&alignment
										type:curr.type
									 atIndex:curr.index];
			i++;
		}
	}

	NSError *err = nil;
	function = [library newFunctionWithName:@"main0"
							 constantValues:constantValues
									  error:&err];
	ERR_FAIL_NULL_V_MSG(function, ERR_CANT_CREATE, String("specialized function failed: ") + err.localizedDescription.UTF8String);

	return function;
}

// RDD::PolygonCullMode == MTLCullMode.
static_assert(ENUM_MEMBERS_EQUAL(RDD::POLYGON_CULL_DISABLED, MTLCullModeNone));
static_assert(ENUM_MEMBERS_EQUAL(RDD::POLYGON_CULL_FRONT, MTLCullModeFront));
static_assert(ENUM_MEMBERS_EQUAL(RDD::POLYGON_CULL_BACK, MTLCullModeBack));

// RDD::StencilOperation == MTLStencilOperation.
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_KEEP, MTLStencilOperationKeep));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_ZERO, MTLStencilOperationZero));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_REPLACE, MTLStencilOperationReplace));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_INCREMENT_AND_CLAMP, MTLStencilOperationIncrementClamp));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_DECREMENT_AND_CLAMP, MTLStencilOperationDecrementClamp));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_INVERT, MTLStencilOperationInvert));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_INCREMENT_AND_WRAP, MTLStencilOperationIncrementWrap));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_DECREMENT_AND_WRAP, MTLStencilOperationDecrementWrap));

// RDD::BlendOperation == MTLBlendOperation.
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_ADD, MTLBlendOperationAdd));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_SUBTRACT, MTLBlendOperationSubtract));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_REVERSE_SUBTRACT, MTLBlendOperationReverseSubtract));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_MINIMUM, MTLBlendOperationMin));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_MAXIMUM, MTLBlendOperationMax));

RDD::PipelineID RenderingDeviceDriverMetal::render_pipeline_create(
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
	MDRenderShader *shader = (MDRenderShader *)(p_shader.id);
	MTLVertexDescriptor *vert_desc = rid::get(p_vertex_format);
	MDRenderPass *pass = (MDRenderPass *)(p_render_pass.id);

	os_signpost_id_t reflect_id = os_signpost_id_make_with_pointer(LOG_INTERVALS, shader);
	os_signpost_interval_begin(LOG_INTERVALS, reflect_id, "render_pipeline_create", "shader_name=%{public}s", shader->name.get_data());
	DEFER([=]() {
		os_signpost_interval_end(LOG_INTERVALS, reflect_id, "render_pipeline_create");
	});

	os_signpost_event_emit(LOG_DRIVER, OS_SIGNPOST_ID_EXCLUSIVE, "create_pipeline");

	MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];

	{
		MDSubpass const &subpass = pass->subpasses[p_render_subpass];
		for (uint32_t i = 0; i < subpass.color_references.size(); i++) {
			uint32_t attachment = subpass.color_references[i].attachment;
			if (attachment != AttachmentReference::UNUSED) {
				MDAttachment const &a = pass->attachments[attachment];
				desc.colorAttachments[i].pixelFormat = a.format;
			}
		}

		if (subpass.depth_stencil_reference.attachment != AttachmentReference::UNUSED) {
			uint32_t attachment = subpass.depth_stencil_reference.attachment;
			MDAttachment const &a = pass->attachments[attachment];

			if (a.type & MDAttachmentType::Depth) {
				desc.depthAttachmentPixelFormat = a.format;
			}

			if (a.type & MDAttachmentType::Stencil) {
				desc.stencilAttachmentPixelFormat = a.format;
			}
		}
	}

	desc.vertexDescriptor = vert_desc;
	desc.label = [NSString stringWithUTF8String:shader->name.get_data()];

	// Input assembly & tessellation.

	MDRenderPipeline *pipeline = new MDRenderPipeline();

	switch (p_render_primitive) {
		case RENDER_PRIMITIVE_POINTS:
			desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
			break;
		case RENDER_PRIMITIVE_LINES:
		case RENDER_PRIMITIVE_LINES_WITH_ADJACENCY:
		case RENDER_PRIMITIVE_LINESTRIPS_WITH_ADJACENCY:
		case RENDER_PRIMITIVE_LINESTRIPS:
			desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
			break;
		case RENDER_PRIMITIVE_TRIANGLES:
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS:
		case RENDER_PRIMITIVE_TRIANGLES_WITH_ADJACENCY:
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_AJACENCY:
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_RESTART_INDEX:
			desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
			break;
		case RENDER_PRIMITIVE_TESSELATION_PATCH:
			desc.maxTessellationFactor = p_rasterization_state.patch_control_points;
			desc.tessellationPartitionMode = MTLTessellationPartitionModeInteger;
			ERR_FAIL_V_MSG(PipelineID(), "tessellation not implemented");
			break;
		case RENDER_PRIMITIVE_MAX:
		default:
			desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassUnspecified;
			break;
	}

	switch (p_render_primitive) {
		case RENDER_PRIMITIVE_POINTS:
			pipeline->raster_state.render_primitive = MTLPrimitiveTypePoint;
			break;
		case RENDER_PRIMITIVE_LINES:
		case RENDER_PRIMITIVE_LINES_WITH_ADJACENCY:
			pipeline->raster_state.render_primitive = MTLPrimitiveTypeLine;
			break;
		case RENDER_PRIMITIVE_LINESTRIPS:
		case RENDER_PRIMITIVE_LINESTRIPS_WITH_ADJACENCY:
			pipeline->raster_state.render_primitive = MTLPrimitiveTypeLineStrip;
			break;
		case RENDER_PRIMITIVE_TRIANGLES:
		case RENDER_PRIMITIVE_TRIANGLES_WITH_ADJACENCY:
			pipeline->raster_state.render_primitive = MTLPrimitiveTypeTriangle;
			break;
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS:
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_AJACENCY:
		case RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_RESTART_INDEX:
			pipeline->raster_state.render_primitive = MTLPrimitiveTypeTriangleStrip;
			break;
		default:
			break;
	}

	// Rasterization.
	desc.rasterizationEnabled = !p_rasterization_state.discard_primitives;
	pipeline->raster_state.clip_mode = p_rasterization_state.enable_depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
	pipeline->raster_state.fill_mode = p_rasterization_state.wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;

	static const MTLCullMode CULL_MODE[3] = {
		MTLCullModeNone,
		MTLCullModeFront,
		MTLCullModeBack,
	};
	pipeline->raster_state.cull_mode = CULL_MODE[p_rasterization_state.cull_mode];
	pipeline->raster_state.winding = (p_rasterization_state.front_face == POLYGON_FRONT_FACE_CLOCKWISE) ? MTLWindingClockwise : MTLWindingCounterClockwise;
	pipeline->raster_state.depth_bias.enabled = p_rasterization_state.depth_bias_enabled;
	pipeline->raster_state.depth_bias.depth_bias = p_rasterization_state.depth_bias_constant_factor;
	pipeline->raster_state.depth_bias.slope_scale = p_rasterization_state.depth_bias_slope_factor;
	pipeline->raster_state.depth_bias.clamp = p_rasterization_state.depth_bias_clamp;
	// In Metal there is no line width.
	if (!Math::is_equal_approx(p_rasterization_state.line_width, 1.0f)) {
		WARN_PRINT("unsupported: line width");
	}

	// Multisample.
	if (p_multisample_state.enable_sample_shading) {
		WARN_PRINT("unsupported: multi-sample shading");
	}

	if (p_multisample_state.sample_count > TEXTURE_SAMPLES_1) {
		pipeline->sample_count = (*device_properties).find_nearest_supported_sample_count(p_multisample_state.sample_count);
	}
	desc.rasterSampleCount = static_cast<NSUInteger>(pipeline->sample_count);
	desc.alphaToCoverageEnabled = p_multisample_state.enable_alpha_to_coverage;
	desc.alphaToOneEnabled = p_multisample_state.enable_alpha_to_one;

	// Depth buffer.
	bool depth_enabled = p_depth_stencil_state.enable_depth_test && desc.depthAttachmentPixelFormat != MTLPixelFormatInvalid;
	bool stencil_enabled = p_depth_stencil_state.enable_stencil && desc.stencilAttachmentPixelFormat != MTLPixelFormatInvalid;

	if (depth_enabled || stencil_enabled) {
		MTLDepthStencilDescriptor *ds_desc = [MTLDepthStencilDescriptor new];

		pipeline->raster_state.depth_test.enabled = depth_enabled;
		ds_desc.depthWriteEnabled = p_depth_stencil_state.enable_depth_write;
		ds_desc.depthCompareFunction = COMPARE_OPERATORS[p_depth_stencil_state.depth_compare_operator];
		if (p_depth_stencil_state.enable_depth_range) {
			WARN_PRINT("unsupported: depth range");
		}

		if (stencil_enabled) {
			pipeline->raster_state.stencil.enabled = true;
			pipeline->raster_state.stencil.front_reference = p_depth_stencil_state.front_op.reference;
			pipeline->raster_state.stencil.back_reference = p_depth_stencil_state.back_op.reference;

			{
				// Front.
				MTLStencilDescriptor *sd = [MTLStencilDescriptor new];
				sd.stencilFailureOperation = STENCIL_OPERATIONS[p_depth_stencil_state.front_op.fail];
				sd.depthStencilPassOperation = STENCIL_OPERATIONS[p_depth_stencil_state.front_op.pass];
				sd.depthFailureOperation = STENCIL_OPERATIONS[p_depth_stencil_state.front_op.depth_fail];
				sd.stencilCompareFunction = COMPARE_OPERATORS[p_depth_stencil_state.front_op.compare];
				sd.readMask = p_depth_stencil_state.front_op.compare_mask;
				sd.writeMask = p_depth_stencil_state.front_op.write_mask;
				ds_desc.frontFaceStencil = sd;
			}
			{
				// Back.
				MTLStencilDescriptor *sd = [MTLStencilDescriptor new];
				sd.stencilFailureOperation = STENCIL_OPERATIONS[p_depth_stencil_state.back_op.fail];
				sd.depthStencilPassOperation = STENCIL_OPERATIONS[p_depth_stencil_state.back_op.pass];
				sd.depthFailureOperation = STENCIL_OPERATIONS[p_depth_stencil_state.back_op.depth_fail];
				sd.stencilCompareFunction = COMPARE_OPERATORS[p_depth_stencil_state.back_op.compare];
				sd.readMask = p_depth_stencil_state.back_op.compare_mask;
				sd.writeMask = p_depth_stencil_state.back_op.write_mask;
				ds_desc.backFaceStencil = sd;
			}
		}

		pipeline->depth_stencil = [device newDepthStencilStateWithDescriptor:ds_desc];
		ERR_FAIL_NULL_V_MSG(pipeline->depth_stencil, PipelineID(), "Failed to create depth stencil state");
	} else {
		// TODO(sgc): FB13671991 raised as Apple docs state calling setDepthStencilState:nil is valid, but currently generates an exception
		pipeline->depth_stencil = get_resource_cache().get_depth_stencil_state(false, false);
	}

	// Blend state.
	{
		for (uint32_t i = 0; i < p_color_attachments.size(); i++) {
			if (p_color_attachments[i] == ATTACHMENT_UNUSED) {
				continue;
			}

			const PipelineColorBlendState::Attachment &bs = p_blend_state.attachments[i];

			MTLRenderPipelineColorAttachmentDescriptor *ca_desc = desc.colorAttachments[p_color_attachments[i]];
			ca_desc.blendingEnabled = bs.enable_blend;

			ca_desc.sourceRGBBlendFactor = BLEND_FACTORS[bs.src_color_blend_factor];
			ca_desc.destinationRGBBlendFactor = BLEND_FACTORS[bs.dst_color_blend_factor];
			ca_desc.rgbBlendOperation = BLEND_OPERATIONS[bs.color_blend_op];

			ca_desc.sourceAlphaBlendFactor = BLEND_FACTORS[bs.src_alpha_blend_factor];
			ca_desc.destinationAlphaBlendFactor = BLEND_FACTORS[bs.dst_alpha_blend_factor];
			ca_desc.alphaBlendOperation = BLEND_OPERATIONS[bs.alpha_blend_op];

			ca_desc.writeMask = MTLColorWriteMaskNone;
			if (bs.write_r) {
				ca_desc.writeMask |= MTLColorWriteMaskRed;
			}
			if (bs.write_g) {
				ca_desc.writeMask |= MTLColorWriteMaskGreen;
			}
			if (bs.write_b) {
				ca_desc.writeMask |= MTLColorWriteMaskBlue;
			}
			if (bs.write_a) {
				ca_desc.writeMask |= MTLColorWriteMaskAlpha;
			}
		}

		pipeline->raster_state.blend.r = p_blend_state.blend_constant.r;
		pipeline->raster_state.blend.g = p_blend_state.blend_constant.g;
		pipeline->raster_state.blend.b = p_blend_state.blend_constant.b;
		pipeline->raster_state.blend.a = p_blend_state.blend_constant.a;
	}

	// Dynamic state.

	if (p_dynamic_state.has_flag(DYNAMIC_STATE_DEPTH_BIAS)) {
		pipeline->raster_state.depth_bias.enabled = true;
	}

	if (p_dynamic_state.has_flag(DYNAMIC_STATE_BLEND_CONSTANTS)) {
		pipeline->raster_state.blend.enabled = true;
	}

	if (p_dynamic_state.has_flag(DYNAMIC_STATE_DEPTH_BOUNDS)) {
		// TODO(sgc): ??
	}

	if (p_dynamic_state.has_flag(DYNAMIC_STATE_STENCIL_COMPARE_MASK)) {
		// TODO(sgc): ??
	}

	if (p_dynamic_state.has_flag(DYNAMIC_STATE_STENCIL_WRITE_MASK)) {
		// TODO(sgc): ??
	}

	if (p_dynamic_state.has_flag(DYNAMIC_STATE_STENCIL_REFERENCE)) {
		pipeline->raster_state.stencil.enabled = true;
	}

	if (shader->vert != nil) {
		Result<id<MTLFunction>> function_or_err = _create_function(shader->vert, @"main0", p_specialization_constants);
		ERR_FAIL_COND_V(std::holds_alternative<Error>(function_or_err), PipelineID());
		desc.vertexFunction = std::get<id<MTLFunction>>(function_or_err);
	}

	if (shader->frag != nil) {
		Result<id<MTLFunction>> function_or_err = _create_function(shader->frag, @"main0", p_specialization_constants);
		ERR_FAIL_COND_V(std::holds_alternative<Error>(function_or_err), PipelineID());
		desc.fragmentFunction = std::get<id<MTLFunction>>(function_or_err);
	}

	if (archive) {
		desc.binaryArchives = @[ archive ];
	}

	NSError *error = nil;
	pipeline->state = [device newRenderPipelineStateWithDescriptor:desc
															 error:&error];
	pipeline->shader = shader;

	ERR_FAIL_COND_V_MSG(error != nil, PipelineID(), ([NSString stringWithFormat:@"error creating pipeline: %@", error.localizedDescription].UTF8String));

	if (archive) {
		if ([archive addRenderPipelineFunctionsWithDescriptor:desc error:&error]) {
			archive_count += 1;
		} else {
			print_error(error.localizedDescription.UTF8String);
		}
	}

	return PipelineID(pipeline);
}

#pragma mark - Compute

// ----- COMMANDS -----

void RenderingDeviceDriverMetal::command_bind_compute_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->bind_pipeline(p_pipeline);
}

void RenderingDeviceDriverMetal::command_bind_compute_uniform_sets(CommandBufferID p_cmd_buffer, VectorView<UniformSetID> p_uniform_sets, ShaderID p_shader, uint32_t p_first_set_index, uint32_t p_set_count, uint32_t p_dynamic_offsets) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->compute_bind_uniform_sets(p_uniform_sets, p_shader, p_first_set_index, p_set_count, p_dynamic_offsets);
}

void RenderingDeviceDriverMetal::command_compute_dispatch(CommandBufferID p_cmd_buffer, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->compute_dispatch(p_x_groups, p_y_groups, p_z_groups);
}

void RenderingDeviceDriverMetal::command_compute_dispatch_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->compute_dispatch_indirect(p_indirect_buffer, p_offset);
}

// ----- PIPELINE -----

RDD::PipelineID RenderingDeviceDriverMetal::compute_pipeline_create(ShaderID p_shader, VectorView<PipelineSpecializationConstant> p_specialization_constants) {
	MDComputeShader *shader = (MDComputeShader *)(p_shader.id);

	os_signpost_id_t reflect_id = os_signpost_id_make_with_pointer(LOG_INTERVALS, shader);
	os_signpost_interval_begin(LOG_INTERVALS, reflect_id, "compute_pipeline_create", "shader_name=%{public}s", shader->name.get_data());
	DEFER([=]() {
		os_signpost_interval_end(LOG_INTERVALS, reflect_id, "compute_pipeline_create");
	});

	os_signpost_event_emit(LOG_DRIVER, OS_SIGNPOST_ID_EXCLUSIVE, "create_pipeline");

	Result<id<MTLFunction>> function_or_err = _create_function(shader->kernel, @"main0", p_specialization_constants);
	ERR_FAIL_COND_V(std::holds_alternative<Error>(function_or_err), PipelineID());
	id<MTLFunction> function = std::get<id<MTLFunction>>(function_or_err);

	MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
	desc.computeFunction = function;
	desc.label = conv::to_nsstring(shader->name);
	if (archive) {
		desc.binaryArchives = @[ archive ];
	}

	NSError *error;
	id<MTLComputePipelineState> state = [device newComputePipelineStateWithDescriptor:desc
																			  options:MTLPipelineOptionNone
																		   reflection:nil
																				error:&error];
	ERR_FAIL_COND_V_MSG(error != nil, PipelineID(), ([NSString stringWithFormat:@"error creating pipeline: %@", error.localizedDescription].UTF8String));

	MDComputePipeline *pipeline = new MDComputePipeline(state);
	pipeline->compute_state.local = shader->local;
	pipeline->shader = shader;

	if (archive) {
		if ([archive addComputePipelineFunctionsWithDescriptor:desc error:&error]) {
			archive_count += 1;
		} else {
			print_error(error.localizedDescription.UTF8String);
		}
	}

	return PipelineID(pipeline);
}

#pragma mark - Queries

// ----- TIMESTAMP -----

RDD::QueryPoolID RenderingDeviceDriverMetal::timestamp_query_pool_create(uint32_t p_query_count) {
	return QueryPoolID(1);
}

void RenderingDeviceDriverMetal::timestamp_query_pool_free(QueryPoolID p_pool_id) {
}

void RenderingDeviceDriverMetal::timestamp_query_pool_get_results(QueryPoolID p_pool_id, uint32_t p_query_count, uint64_t *r_results) {
	// Metal doesn't support timestamp queries, so we just clear the buffer.
	bzero(r_results, p_query_count * sizeof(uint64_t));
}

uint64_t RenderingDeviceDriverMetal::timestamp_query_result_to_time(uint64_t p_result) {
	return p_result;
}

void RenderingDeviceDriverMetal::command_timestamp_query_pool_reset(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_query_count) {
}

void RenderingDeviceDriverMetal::command_timestamp_write(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_index) {
}

#pragma mark - Labels

void RenderingDeviceDriverMetal::command_begin_label(CommandBufferID p_cmd_buffer, const char *p_label_name, const Color &p_color) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->begin_label(p_label_name, p_color);
}

void RenderingDeviceDriverMetal::command_end_label(CommandBufferID p_cmd_buffer) {
	MDCommandBuffer *cb = (MDCommandBuffer *)(p_cmd_buffer.id);
	cb->end_label();
}

#pragma mark - Debug

void RenderingDeviceDriverMetal::command_insert_breadcrumb(CommandBufferID p_cmd_buffer, uint32_t p_data) {
	// TODO: Implement.
}

#pragma mark - Submission

void RenderingDeviceDriverMetal::begin_segment(uint32_t p_frame_index, uint32_t p_frames_drawn) {
	_frame_index = p_frame_index;
	_frames_drawn = p_frames_drawn;
}

void RenderingDeviceDriverMetal::end_segment() {
}

#pragma mark - Misc

void RenderingDeviceDriverMetal::set_object_name(ObjectType p_type, ID p_driver_id, const String &p_name) {
	switch (p_type) {
		case OBJECT_TYPE_TEXTURE: {
			id<MTLTexture> tex = rid::get(p_driver_id);
			tex.label = [NSString stringWithUTF8String:p_name.utf8().get_data()];
		} break;
		case OBJECT_TYPE_SAMPLER: {
			// Can't set label after creation.
		} break;
		case OBJECT_TYPE_BUFFER: {
			const BufferInfo *buf_info = (const BufferInfo *)p_driver_id.id;
			buf_info->metal_buffer.label = [NSString stringWithUTF8String:p_name.utf8().get_data()];
		} break;
		case OBJECT_TYPE_SHADER: {
			NSString *label = [NSString stringWithUTF8String:p_name.utf8().get_data()];
			MDShader *shader = (MDShader *)(p_driver_id.id);
			if (MDRenderShader *rs = dynamic_cast<MDRenderShader *>(shader); rs != nullptr) {
				[rs->vert setLabel:label];
				[rs->frag setLabel:label];
			} else if (MDComputeShader *cs = dynamic_cast<MDComputeShader *>(shader); cs != nullptr) {
				[cs->kernel setLabel:label];
			} else {
				DEV_ASSERT(false);
			}
		} break;
		case OBJECT_TYPE_UNIFORM_SET: {
			MDUniformSet *set = (MDUniformSet *)(p_driver_id.id);
			set->arg_buffer.label = [NSString stringWithUTF8String:p_name.utf8().get_data()];
		} break;
		case OBJECT_TYPE_PIPELINE: {
			// Can't set label after creation.
		} break;
		default: {
			DEV_ASSERT(false);
		}
	}
}

uint64_t RenderingDeviceDriverMetal::get_resource_native_handle(DriverResource p_type, ID p_driver_id) {
	switch (p_type) {
		case DRIVER_RESOURCE_LOGICAL_DEVICE: {
			return (uint64_t)(uintptr_t)(__bridge void *)device;
		}
		case DRIVER_RESOURCE_PHYSICAL_DEVICE: {
			return 0;
		}
		case DRIVER_RESOURCE_TOPMOST_OBJECT: {
			return 0;
		}
		case DRIVER_RESOURCE_COMMAND_QUEUE: {
			return (uint64_t)(uintptr_t)(__bridge void *)device_queue;
		}
		case DRIVER_RESOURCE_QUEUE_FAMILY: {
			return 0;
		}
		case DRIVER_RESOURCE_TEXTURE: {
			return p_driver_id.id;
		}
		case DRIVER_RESOURCE_TEXTURE_VIEW: {
			return p_driver_id.id;
		}
		case DRIVER_RESOURCE_TEXTURE_DATA_FORMAT: {
			return 0;
		}
		case DRIVER_RESOURCE_SAMPLER: {
			return p_driver_id.id;
		}
		case DRIVER_RESOURCE_UNIFORM_SET: {
			return 0;
		}
		case DRIVER_RESOURCE_BUFFER: {
			return p_driver_id.id;
		}
		case DRIVER_RESOURCE_COMPUTE_PIPELINE: {
			MDComputePipeline *pipeline = (MDComputePipeline *)(p_driver_id.id);
			return (uint64_t)(uintptr_t)(__bridge void *)pipeline->state;
		}
		case DRIVER_RESOURCE_RENDER_PIPELINE: {
			MDRenderPipeline *pipeline = (MDRenderPipeline *)(p_driver_id.id);
			return (uint64_t)(uintptr_t)(__bridge void *)pipeline->state;
		}
		default: {
			return 0;
		}
	}
}

uint64_t RenderingDeviceDriverMetal::get_total_memory_used() {
	return device.currentAllocatedSize;
}

uint64_t RenderingDeviceDriverMetal::get_lazily_memory_used() {
	return 0; // TODO: Track this (grep for memoryless in Godot's Metal backend).
}

uint64_t RenderingDeviceDriverMetal::limit_get(Limit p_limit) {
	MetalDeviceProperties const &props = (*device_properties);
	MetalLimits const &limits = props.limits;
	uint64_t safe_unbounded = ((uint64_t)1 << 30);
#if defined(DEV_ENABLED)
#define UNKNOWN(NAME)                                                            \
	case NAME:                                                                   \
		WARN_PRINT_ONCE("Returning maximum value for unknown limit " #NAME "."); \
		return safe_unbounded;
#else
#define UNKNOWN(NAME) \
	case NAME:        \
		return safe_unbounded
#endif

	// clang-format off
	switch (p_limit) {
		case LIMIT_MAX_BOUND_UNIFORM_SETS:
			return limits.maxBoundDescriptorSets;
		case LIMIT_MAX_FRAMEBUFFER_COLOR_ATTACHMENTS:
			return limits.maxColorAttachments;
		case LIMIT_MAX_TEXTURES_PER_UNIFORM_SET:
			return limits.maxTexturesPerArgumentBuffer;
		case LIMIT_MAX_SAMPLERS_PER_UNIFORM_SET:
			return limits.maxSamplersPerArgumentBuffer;
		case LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET:
			return limits.maxBuffersPerArgumentBuffer;
		case LIMIT_MAX_STORAGE_IMAGES_PER_UNIFORM_SET:
			return limits.maxTexturesPerArgumentBuffer;
		case LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET:
			return limits.maxBuffersPerArgumentBuffer;
		case LIMIT_MAX_DRAW_INDEXED_INDEX:
			return limits.maxDrawIndexedIndexValue;
		case LIMIT_MAX_FRAMEBUFFER_HEIGHT:
			return limits.maxFramebufferHeight;
		case LIMIT_MAX_FRAMEBUFFER_WIDTH:
			return limits.maxFramebufferWidth;
		case LIMIT_MAX_TEXTURE_ARRAY_LAYERS:
			return limits.maxImageArrayLayers;
		case LIMIT_MAX_TEXTURE_SIZE_1D:
			return limits.maxImageDimension1D;
		case LIMIT_MAX_TEXTURE_SIZE_2D:
			return limits.maxImageDimension2D;
		case LIMIT_MAX_TEXTURE_SIZE_3D:
			return limits.maxImageDimension3D;
		case LIMIT_MAX_TEXTURE_SIZE_CUBE:
			return limits.maxImageDimensionCube;
		case LIMIT_MAX_TEXTURES_PER_SHADER_STAGE:
			return limits.maxTexturesPerArgumentBuffer;
		case LIMIT_MAX_SAMPLERS_PER_SHADER_STAGE:
			return limits.maxSamplersPerArgumentBuffer;
		case LIMIT_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE:
			return limits.maxBuffersPerArgumentBuffer;
		case LIMIT_MAX_STORAGE_IMAGES_PER_SHADER_STAGE:
			return limits.maxTexturesPerArgumentBuffer;
		case LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE:
			return limits.maxBuffersPerArgumentBuffer;
		case LIMIT_MAX_PUSH_CONSTANT_SIZE:
			return limits.maxBufferLength;
		case LIMIT_MAX_UNIFORM_BUFFER_SIZE:
			return limits.maxBufferLength;
		case LIMIT_MAX_VERTEX_INPUT_ATTRIBUTE_OFFSET:
			return limits.maxVertexDescriptorLayoutStride;
		case LIMIT_MAX_VERTEX_INPUT_ATTRIBUTES:
			return limits.maxVertexInputAttributes;
		case LIMIT_MAX_VERTEX_INPUT_BINDINGS:
			return limits.maxVertexInputBindings;
		case LIMIT_MAX_VERTEX_INPUT_BINDING_STRIDE:
			return limits.maxVertexInputBindingStride;
		case LIMIT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT:
			return limits.minUniformBufferOffsetAlignment;
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X:
			return limits.maxComputeWorkGroupCount.width;
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y:
			return limits.maxComputeWorkGroupCount.height;
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z:
			return limits.maxComputeWorkGroupCount.depth;
		case LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS:
			return std::max({ limits.maxThreadsPerThreadGroup.width, limits.maxThreadsPerThreadGroup.height, limits.maxThreadsPerThreadGroup.depth });
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X:
			return limits.maxThreadsPerThreadGroup.width;
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y:
			return limits.maxThreadsPerThreadGroup.height;
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Z:
			return limits.maxThreadsPerThreadGroup.depth;
		case LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE:
			return limits.maxThreadGroupMemoryAllocation;
		case LIMIT_MAX_VIEWPORT_DIMENSIONS_X:
			return limits.maxViewportDimensionX;
		case LIMIT_MAX_VIEWPORT_DIMENSIONS_Y:
			return limits.maxViewportDimensionY;
		case LIMIT_SUBGROUP_SIZE:
			// MoltenVK sets the subgroupSize to the same as the maxSubgroupSize.
			return limits.maxSubgroupSize;
		case LIMIT_SUBGROUP_MIN_SIZE:
			return limits.minSubgroupSize;
		case LIMIT_SUBGROUP_MAX_SIZE:
			return limits.maxSubgroupSize;
		case LIMIT_SUBGROUP_IN_SHADERS:
			return (uint64_t)limits.subgroupSupportedShaderStages;
		case LIMIT_SUBGROUP_OPERATIONS:
			return (uint64_t)limits.subgroupSupportedOperations;
		case LIMIT_METALFX_TEMPORAL_SCALER_MIN_SCALE:
			return (uint64_t)((1.0 / limits.temporalScalerInputContentMaxScale) * 1000'000);
		case LIMIT_METALFX_TEMPORAL_SCALER_MAX_SCALE:
			return (uint64_t)((1.0 / limits.temporalScalerInputContentMinScale) * 1000'000);
		case LIMIT_MAX_SHADER_VARYINGS:
			return limits.maxShaderVaryings;
		default: {
#ifdef DEV_ENABLED
			WARN_PRINT("Returning maximum value for unknown limit " + itos(p_limit) + ".");
#endif
			return safe_unbounded;
		}
	}
	// clang-format on
	return 0;
}

uint64_t RenderingDeviceDriverMetal::api_trait_get(ApiTrait p_trait) {
	switch (p_trait) {
		case API_TRAIT_HONORS_PIPELINE_BARRIERS:
			return false;
		case API_TRAIT_CLEARS_WITH_COPY_ENGINE:
			return false;
		default:
			return RenderingDeviceDriver::api_trait_get(p_trait);
	}
}

bool RenderingDeviceDriverMetal::has_feature(Features p_feature) {
	switch (p_feature) {
		case SUPPORTS_HALF_FLOAT:
			return true;
		case SUPPORTS_FRAGMENT_SHADER_WITH_ONLY_SIDE_EFFECTS:
			return true;
		case SUPPORTS_BUFFER_DEVICE_ADDRESS:
			return device_properties->features.supports_gpu_address;
		case SUPPORTS_METALFX_SPATIAL:
			return device_properties->features.metal_fx_spatial;
		case SUPPORTS_METALFX_TEMPORAL:
			return device_properties->features.metal_fx_temporal;
		case SUPPORTS_IMAGE_ATOMIC_32_BIT:
			return device_properties->features.supports_native_image_atomics;
		case SUPPORTS_VULKAN_MEMORY_MODEL:
			return true;
		case SUPPORTS_POINT_SIZE:
			return true;
		default:
			return false;
	}
}

const RDD::MultiviewCapabilities &RenderingDeviceDriverMetal::get_multiview_capabilities() {
	return multiview_capabilities;
}

const RDD::FragmentShadingRateCapabilities &RenderingDeviceDriverMetal::get_fragment_shading_rate_capabilities() {
	return fsr_capabilities;
}

const RDD::FragmentDensityMapCapabilities &RenderingDeviceDriverMetal::get_fragment_density_map_capabilities() {
	return fdm_capabilities;
}

String RenderingDeviceDriverMetal::get_api_version() const {
	return vformat("%d.%d", capabilities.version_major, capabilities.version_minor);
}

String RenderingDeviceDriverMetal::get_pipeline_cache_uuid() const {
	return pipeline_cache_id;
}

const RDD::Capabilities &RenderingDeviceDriverMetal::get_capabilities() const {
	return capabilities;
}

bool RenderingDeviceDriverMetal::is_composite_alpha_supported(CommandQueueID p_queue) const {
	// The CAMetalLayer.opaque property is configured according to this global setting.
	return OS::get_singleton()->is_layered_allowed();
}

size_t RenderingDeviceDriverMetal::get_texel_buffer_alignment_for_format(RDD::DataFormat p_format) const {
	return [device minimumLinearTextureAlignmentForPixelFormat:pixel_formats->getMTLPixelFormat(p_format)];
}

size_t RenderingDeviceDriverMetal::get_texel_buffer_alignment_for_format(MTLPixelFormat p_format) const {
	return [device minimumLinearTextureAlignmentForPixelFormat:p_format];
}

/******************/

RenderingDeviceDriverMetal::RenderingDeviceDriverMetal(RenderingContextDriverMetal *p_context_driver) :
		context_driver(p_context_driver) {
	DEV_ASSERT(p_context_driver != nullptr);

#if TARGET_OS_OSX
	if (String res = OS::get_singleton()->get_environment("GODOT_MTL_SHADER_LOAD_STRATEGY"); res == U"lazy") {
		_shader_load_strategy = ShaderLoadStrategy::LAZY;
	}
#else
	// Always use the lazy strategy on other OSs like iOS, tvOS, or visionOS.
	_shader_load_strategy = ShaderLoadStrategy::LAZY;
#endif
}

RenderingDeviceDriverMetal::~RenderingDeviceDriverMetal() {
	for (MDCommandBuffer *cb : command_buffers) {
		delete cb;
	}

	for (KeyValue<SHA256Digest, ShaderCacheEntry *> &kv : _shader_cache) {
		memdelete(kv.value);
	}

	if (shader_container_format != nullptr) {
		memdelete(shader_container_format);
	}

	if (pixel_formats != nullptr) {
		memdelete(pixel_formats);
	}

	if (device_properties != nullptr) {
		memdelete(device_properties);
	}
}

#pragma mark - Initialization

Error RenderingDeviceDriverMetal::_create_device() {
	device = context_driver->get_metal_device();

	device_queue = [device newCommandQueue];
	ERR_FAIL_NULL_V(device_queue, ERR_CANT_CREATE);

	device_scope = [MTLCaptureManager.sharedCaptureManager newCaptureScopeWithCommandQueue:device_queue];
	device_scope.label = @"Godot Frame";
	[device_scope beginScope]; // Allow Xcode to capture the first frame, if desired.

	resource_cache = std::make_unique<MDResourceCache>(this);

	return OK;
}

void RenderingDeviceDriverMetal::_check_capabilities() {
	capabilities.device_family = DEVICE_METAL;
	parse_msl_version(device_properties->features.msl_target_version, capabilities.version_major, capabilities.version_minor);
}

API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0))
static MetalDeviceProfile device_profile_from_properties(MetalDeviceProperties *p_device_properties) {
	using DP = MetalDeviceProfile;
	NSOperatingSystemVersion os_version = NSProcessInfo.processInfo.operatingSystemVersion;
	MetalDeviceProfile res;
	res.min_os_version = MinOsVersion(os_version.majorVersion, os_version.minorVersion, os_version.patchVersion);
#if TARGET_OS_OSX
	res.platform = DP::Platform::macOS;
#else
	res.platform = DP::Platform::iOS;
#endif
	res.features = {
		.msl_version = p_device_properties->features.msl_target_version,
		.use_argument_buffers = p_device_properties->features.argument_buffers_enabled(),
		.simdPermute = p_device_properties->features.simdPermute,
	};

	// highestFamily will only be set to an Apple GPU family
	switch (p_device_properties->features.highestFamily) {
		case MTLGPUFamilyApple1:
			res.gpu = DP::GPU::Apple1;
			break;
		case MTLGPUFamilyApple2:
			res.gpu = DP::GPU::Apple2;
			break;
		case MTLGPUFamilyApple3:
			res.gpu = DP::GPU::Apple3;
			break;
		case MTLGPUFamilyApple4:
			res.gpu = DP::GPU::Apple4;
			break;
		case MTLGPUFamilyApple5:
			res.gpu = DP::GPU::Apple5;
			break;
		case MTLGPUFamilyApple6:
			res.gpu = DP::GPU::Apple6;
			break;
		case MTLGPUFamilyApple7:
			res.gpu = DP::GPU::Apple7;
			break;
		case MTLGPUFamilyApple8:
			res.gpu = DP::GPU::Apple8;
			break;
		case MTLGPUFamilyApple9:
			res.gpu = DP::GPU::Apple9;
			break;
		default: {
			// Programming error if the default case is hit.
			CRASH_NOW_MSG("Unsupported GPU family");
		} break;
	}

	return res;
}

Error RenderingDeviceDriverMetal::initialize(uint32_t p_device_index, uint32_t p_frame_count) {
	context_device = context_driver->device_get(p_device_index);
	Error err = _create_device();
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);

	device_properties = memnew(MetalDeviceProperties(device));
	device_profile = device_profile_from_properties(device_properties);
	shader_container_format = memnew(RenderingShaderContainerFormatMetal(&device_profile));

	_check_capabilities();

	_frame_count = p_frame_count;

	// Set the pipeline cache ID based on the Metal version.
	pipeline_cache_id = "metal-driver-" + get_api_version();

	pixel_formats = memnew(PixelFormats(device, device_properties->features));
	if (device_properties->features.layeredRendering) {
		multiview_capabilities.is_supported = true;
		multiview_capabilities.max_view_count = device_properties->limits.maxViewports;
		// NOTE: I'm not sure what the limit is as I don't see it referenced anywhere
		multiview_capabilities.max_instance_count = UINT32_MAX;

		print_verbose("- Metal multiview supported:");
		print_verbose("  max view count: " + itos(multiview_capabilities.max_view_count));
		print_verbose("  max instances: " + itos(multiview_capabilities.max_instance_count));
	} else {
		print_verbose("- Metal multiview not supported");
	}

	// The Metal renderer requires Apple4 family. This is 2017 era A11 chips and newer.
	if (device_properties->features.highestFamily < MTLGPUFamilyApple4) {
		String error_string = vformat("Your Apple GPU does not support the following features, which are required to use Metal-based renderers in Godot:\n\n");
		if (!device_properties->features.imageCubeArray) {
			error_string += "- No support for image cube arrays.\n";
		}

#if defined(APPLE_EMBEDDED_ENABLED)
		// Apple Embedded platforms exports currently don't exit themselves when this method returns `ERR_CANT_CREATE`.
		OS::get_singleton()->alert(error_string + "\nClick OK to exit (black screen will be visible).");
#else
		OS::get_singleton()->alert(error_string + "\nClick OK to exit.");
#endif

		return ERR_CANT_CREATE;
	}

	return OK;
}

const RenderingShaderContainerFormat &RenderingDeviceDriverMetal::get_shader_container_format() const {
	return *shader_container_format;
}
