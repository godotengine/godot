/**************************************************************************/
/*  pixel_formats_webgpu.h                                                */
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

#include "servers/rendering/rendering_device_commons.h"

#include <webgpu/webgpu.h>

// WebGPU's format table defines 4x support; 2x and 8x are never valid.
static inline bool wgpu_format_supports_multisampling(WGPUTextureFormat p_format, bool p_core, bool p_tier1, bool p_rg11b10_renderable) {
	switch (p_format) {
		case WGPUTextureFormat_R8Unorm:
		case WGPUTextureFormat_RG8Unorm:
		case WGPUTextureFormat_RGBA8Unorm:
		case WGPUTextureFormat_RGBA8UnormSrgb:
		case WGPUTextureFormat_BGRA8Unorm:
		case WGPUTextureFormat_R16Float:
		case WGPUTextureFormat_RG16Float:
		case WGPUTextureFormat_RGB10A2Unorm:
		case WGPUTextureFormat_Stencil8:
		case WGPUTextureFormat_Depth16Unorm:
		case WGPUTextureFormat_Depth24Plus:
		case WGPUTextureFormat_Depth24PlusStencil8:
		case WGPUTextureFormat_Depth32Float:
		case WGPUTextureFormat_Depth32FloatStencil8:
			return true;
		case WGPUTextureFormat_BGRA8UnormSrgb:
		case WGPUTextureFormat_RGBA16Float:
		case WGPUTextureFormat_R32Float:
		case WGPUTextureFormat_R8Uint:
		case WGPUTextureFormat_R8Sint:
		case WGPUTextureFormat_RG8Uint:
		case WGPUTextureFormat_RG8Sint:
		case WGPUTextureFormat_RGBA8Uint:
		case WGPUTextureFormat_RGBA8Sint:
		case WGPUTextureFormat_R16Uint:
		case WGPUTextureFormat_R16Sint:
		case WGPUTextureFormat_RG16Uint:
		case WGPUTextureFormat_RG16Sint:
		case WGPUTextureFormat_RGBA16Uint:
		case WGPUTextureFormat_RGBA16Sint:
		case WGPUTextureFormat_RGB10A2Uint:
			return p_core;
		case WGPUTextureFormat_R8Snorm:
		case WGPUTextureFormat_RG8Snorm:
		case WGPUTextureFormat_RGBA8Snorm:
		case WGPUTextureFormat_R16Unorm:
		case WGPUTextureFormat_R16Snorm:
		case WGPUTextureFormat_RG16Unorm:
		case WGPUTextureFormat_RG16Snorm:
		case WGPUTextureFormat_RGBA16Unorm:
		case WGPUTextureFormat_RGBA16Snorm:
			return p_tier1;
		case WGPUTextureFormat_RG11B10Ufloat:
			return p_rg11b10_renderable;
		default:
			return false;
	}
}

// Storage declarations use a format name followed by a comma.
static inline WGPUTextureFormat wgpu_storage_format_from_wgsl(const char *p_format) {
	if (strncmp(p_format, "rgba8unorm,", 11) == 0) {
		return WGPUTextureFormat_RGBA8Unorm;
	}
	if (strncmp(p_format, "rgba8snorm,", 11) == 0) {
		return WGPUTextureFormat_RGBA8Snorm;
	}
	if (strncmp(p_format, "rgba8uint,", 10) == 0) {
		return WGPUTextureFormat_RGBA8Uint;
	}
	if (strncmp(p_format, "rgba8sint,", 10) == 0) {
		return WGPUTextureFormat_RGBA8Sint;
	}
	if (strncmp(p_format, "rgba16float,", 12) == 0) {
		return WGPUTextureFormat_RGBA16Float;
	}
	if (strncmp(p_format, "rgba16uint,", 11) == 0) {
		return WGPUTextureFormat_RGBA16Uint;
	}
	if (strncmp(p_format, "rgba16sint,", 11) == 0) {
		return WGPUTextureFormat_RGBA16Sint;
	}
	if (strncmp(p_format, "rgba32float,", 12) == 0) {
		return WGPUTextureFormat_RGBA32Float;
	}
	if (strncmp(p_format, "rgba32uint,", 11) == 0) {
		return WGPUTextureFormat_RGBA32Uint;
	}
	if (strncmp(p_format, "rgba32sint,", 11) == 0) {
		return WGPUTextureFormat_RGBA32Sint;
	}
	if (strncmp(p_format, "rg32float,", 10) == 0) {
		return WGPUTextureFormat_RG32Float;
	}
	if (strncmp(p_format, "rg32uint,", 9) == 0) {
		return WGPUTextureFormat_RG32Uint;
	}
	if (strncmp(p_format, "rg32sint,", 9) == 0) {
		return WGPUTextureFormat_RG32Sint;
	}
	if (strncmp(p_format, "r32float,", 9) == 0) {
		return WGPUTextureFormat_R32Float;
	}
	if (strncmp(p_format, "r32uint,", 8) == 0) {
		return WGPUTextureFormat_R32Uint;
	}
	if (strncmp(p_format, "r32sint,", 8) == 0) {
		return WGPUTextureFormat_R32Sint;
	}
	if (strncmp(p_format, "r16float,", 9) == 0) {
		return WGPUTextureFormat_R16Float;
	}
	if (strncmp(p_format, "r16uint,", 8) == 0) {
		return WGPUTextureFormat_R16Uint;
	}
	if (strncmp(p_format, "r16sint,", 8) == 0) {
		return WGPUTextureFormat_R16Sint;
	}
	if (strncmp(p_format, "r16snorm,", 9) == 0) {
		return WGPUTextureFormat_R16Snorm;
	}
	if (strncmp(p_format, "r16unorm,", 9) == 0) {
		return WGPUTextureFormat_R16Unorm;
	}
	if (strncmp(p_format, "r8unorm,", 8) == 0) {
		return WGPUTextureFormat_R8Unorm;
	}
	if (strncmp(p_format, "r8snorm,", 8) == 0) {
		return WGPUTextureFormat_R8Snorm;
	}
	if (strncmp(p_format, "r8uint,", 7) == 0) {
		return WGPUTextureFormat_R8Uint;
	}
	if (strncmp(p_format, "r8sint,", 7) == 0) {
		return WGPUTextureFormat_R8Sint;
	}
	if (strncmp(p_format, "rg8unorm,", 9) == 0) {
		return WGPUTextureFormat_RG8Unorm;
	}
	if (strncmp(p_format, "rg8snorm,", 9) == 0) {
		return WGPUTextureFormat_RG8Snorm;
	}
	if (strncmp(p_format, "rg8uint,", 8) == 0) {
		return WGPUTextureFormat_RG8Uint;
	}
	if (strncmp(p_format, "rg8sint,", 8) == 0) {
		return WGPUTextureFormat_RG8Sint;
	}
	if (strncmp(p_format, "rg16float,", 10) == 0) {
		return WGPUTextureFormat_RG16Float;
	}
	if (strncmp(p_format, "rg16uint,", 9) == 0) {
		return WGPUTextureFormat_RG16Uint;
	}
	if (strncmp(p_format, "rg16sint,", 9) == 0) {
		return WGPUTextureFormat_RG16Sint;
	}
	if (strncmp(p_format, "rg16snorm,", 10) == 0) {
		return WGPUTextureFormat_RG16Snorm;
	}
	if (strncmp(p_format, "rg16unorm,", 10) == 0) {
		return WGPUTextureFormat_RG16Unorm;
	}
	if (strncmp(p_format, "rgba16snorm,", 12) == 0) {
		return WGPUTextureFormat_RGBA16Snorm;
	}
	if (strncmp(p_format, "rgba16unorm,", 12) == 0) {
		return WGPUTextureFormat_RGBA16Unorm;
	}
	if (strncmp(p_format, "bgra8unorm,", 11) == 0) {
		return WGPUTextureFormat_BGRA8Unorm;
	}
	return WGPUTextureFormat_Undefined;
}

// The WGSL language feature alone only enables read/write access for R32 formats.
static inline bool wgpu_storage_format_supports_read_write(WGPUTextureFormat p_format, bool p_language_feature, bool p_tier2) {
	if (!p_language_feature) {
		return false;
	}
	switch (p_format) {
		case WGPUTextureFormat_R32Float:
		case WGPUTextureFormat_R32Uint:
		case WGPUTextureFormat_R32Sint:
			return true;
		case WGPUTextureFormat_R8Unorm:
		case WGPUTextureFormat_R8Uint:
		case WGPUTextureFormat_R8Sint:
		case WGPUTextureFormat_RGBA8Unorm:
		case WGPUTextureFormat_RGBA8Uint:
		case WGPUTextureFormat_RGBA8Sint:
		case WGPUTextureFormat_R16Uint:
		case WGPUTextureFormat_R16Sint:
		case WGPUTextureFormat_R16Float:
		case WGPUTextureFormat_RGBA16Uint:
		case WGPUTextureFormat_RGBA16Sint:
		case WGPUTextureFormat_RGBA16Float:
		case WGPUTextureFormat_RGBA32Uint:
		case WGPUTextureFormat_RGBA32Sint:
		case WGPUTextureFormat_RGBA32Float:
			return p_tier2;
		default:
			return false;
	}
}

// Formats omitted here are unsupported by WebGPU.
static inline WGPUTextureFormat data_format_to_wgpu(RenderingDeviceCommons::DataFormat p_format) {
	using DF = RenderingDeviceCommons::DataFormat;
	switch (p_format) {
		case DF::DATA_FORMAT_R8_UNORM:
			return WGPUTextureFormat_R8Unorm;
		case DF::DATA_FORMAT_R8_SNORM:
			return WGPUTextureFormat_R8Snorm;
		case DF::DATA_FORMAT_R8_UINT:
			return WGPUTextureFormat_R8Uint;
		case DF::DATA_FORMAT_R8_SINT:
			return WGPUTextureFormat_R8Sint;
		case DF::DATA_FORMAT_R8G8_UNORM:
			return WGPUTextureFormat_RG8Unorm;
		case DF::DATA_FORMAT_R8G8_SNORM:
			return WGPUTextureFormat_RG8Snorm;
		case DF::DATA_FORMAT_R8G8_UINT:
			return WGPUTextureFormat_RG8Uint;
		case DF::DATA_FORMAT_R8G8_SINT:
			return WGPUTextureFormat_RG8Sint;
		case DF::DATA_FORMAT_R8G8B8A8_UNORM:
			return WGPUTextureFormat_RGBA8Unorm;
		case DF::DATA_FORMAT_R8G8B8A8_SNORM:
			return WGPUTextureFormat_RGBA8Snorm;
		case DF::DATA_FORMAT_R8G8B8A8_UINT:
			return WGPUTextureFormat_RGBA8Uint;
		case DF::DATA_FORMAT_R8G8B8A8_SINT:
			return WGPUTextureFormat_RGBA8Sint;
		case DF::DATA_FORMAT_R8G8B8A8_SRGB:
			return WGPUTextureFormat_RGBA8UnormSrgb;
		case DF::DATA_FORMAT_B8G8R8A8_UNORM:
			return WGPUTextureFormat_BGRA8Unorm;
		case DF::DATA_FORMAT_B8G8R8A8_SRGB:
			return WGPUTextureFormat_BGRA8UnormSrgb;
		case DF::DATA_FORMAT_A8B8G8R8_UNORM_PACK32:
			return WGPUTextureFormat_RGBA8Unorm;
		case DF::DATA_FORMAT_A8B8G8R8_SNORM_PACK32:
			return WGPUTextureFormat_RGBA8Snorm;
		case DF::DATA_FORMAT_A8B8G8R8_UINT_PACK32:
			return WGPUTextureFormat_RGBA8Uint;
		case DF::DATA_FORMAT_A8B8G8R8_SINT_PACK32:
			return WGPUTextureFormat_RGBA8Sint;
		case DF::DATA_FORMAT_A8B8G8R8_SRGB_PACK32:
			return WGPUTextureFormat_RGBA8UnormSrgb;
		case DF::DATA_FORMAT_A2B10G10R10_UNORM_PACK32:
			return WGPUTextureFormat_RGB10A2Unorm;
		case DF::DATA_FORMAT_A2B10G10R10_UINT_PACK32:
			return WGPUTextureFormat_RGB10A2Uint;
		case DF::DATA_FORMAT_R16_UNORM:
			return WGPUTextureFormat_R16Unorm;
		case DF::DATA_FORMAT_R16_SNORM:
			return WGPUTextureFormat_R16Snorm;
		case DF::DATA_FORMAT_R16_UINT:
			return WGPUTextureFormat_R16Uint;
		case DF::DATA_FORMAT_R16_SINT:
			return WGPUTextureFormat_R16Sint;
		case DF::DATA_FORMAT_R16_SFLOAT:
			return WGPUTextureFormat_R16Float;
		case DF::DATA_FORMAT_R16G16_UNORM:
			return WGPUTextureFormat_RG16Unorm;
		case DF::DATA_FORMAT_R16G16_SNORM:
			return WGPUTextureFormat_RG16Snorm;
		case DF::DATA_FORMAT_R16G16_UINT:
			return WGPUTextureFormat_RG16Uint;
		case DF::DATA_FORMAT_R16G16_SINT:
			return WGPUTextureFormat_RG16Sint;
		case DF::DATA_FORMAT_R16G16_SFLOAT:
			return WGPUTextureFormat_RG16Float;
		case DF::DATA_FORMAT_R16G16B16A16_UNORM:
			return WGPUTextureFormat_RGBA16Unorm;
		case DF::DATA_FORMAT_R16G16B16A16_SNORM:
			return WGPUTextureFormat_RGBA16Snorm;
		case DF::DATA_FORMAT_R16G16B16A16_UINT:
			return WGPUTextureFormat_RGBA16Uint;
		case DF::DATA_FORMAT_R16G16B16A16_SINT:
			return WGPUTextureFormat_RGBA16Sint;
		case DF::DATA_FORMAT_R16G16B16A16_SFLOAT:
			return WGPUTextureFormat_RGBA16Float;
		case DF::DATA_FORMAT_R32_UINT:
			return WGPUTextureFormat_R32Uint;
		case DF::DATA_FORMAT_R32_SINT:
			return WGPUTextureFormat_R32Sint;
		case DF::DATA_FORMAT_R32_SFLOAT:
			return WGPUTextureFormat_R32Float;
		case DF::DATA_FORMAT_R32G32_UINT:
			return WGPUTextureFormat_RG32Uint;
		case DF::DATA_FORMAT_R32G32_SINT:
			return WGPUTextureFormat_RG32Sint;
		case DF::DATA_FORMAT_R32G32_SFLOAT:
			return WGPUTextureFormat_RG32Float;
		case DF::DATA_FORMAT_R32G32B32A32_UINT:
			return WGPUTextureFormat_RGBA32Uint;
		case DF::DATA_FORMAT_R32G32B32A32_SINT:
			return WGPUTextureFormat_RGBA32Sint;
		case DF::DATA_FORMAT_R32G32B32A32_SFLOAT:
			return WGPUTextureFormat_RGBA32Float;
		case DF::DATA_FORMAT_B10G11R11_UFLOAT_PACK32:
			return WGPUTextureFormat_RG11B10Ufloat;
		case DF::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32:
			return WGPUTextureFormat_RGB9E5Ufloat;
		case DF::DATA_FORMAT_D16_UNORM:
			return WGPUTextureFormat_Depth16Unorm;
		case DF::DATA_FORMAT_X8_D24_UNORM_PACK32:
			return WGPUTextureFormat_Depth24Plus;
		case DF::DATA_FORMAT_D32_SFLOAT:
			return WGPUTextureFormat_Depth32Float;
		case DF::DATA_FORMAT_S8_UINT:
			return WGPUTextureFormat_Stencil8;
		case DF::DATA_FORMAT_D24_UNORM_S8_UINT:
			return WGPUTextureFormat_Depth24PlusStencil8;
		case DF::DATA_FORMAT_D32_SFLOAT_S8_UINT:
			return WGPUTextureFormat_Depth32FloatStencil8;
		case DF::DATA_FORMAT_BC1_RGB_UNORM_BLOCK:
			return WGPUTextureFormat_BC1RGBAUnorm;
		case DF::DATA_FORMAT_BC1_RGB_SRGB_BLOCK:
			return WGPUTextureFormat_BC1RGBAUnormSrgb;
		case DF::DATA_FORMAT_BC1_RGBA_UNORM_BLOCK:
			return WGPUTextureFormat_BC1RGBAUnorm;
		case DF::DATA_FORMAT_BC1_RGBA_SRGB_BLOCK:
			return WGPUTextureFormat_BC1RGBAUnormSrgb;
		case DF::DATA_FORMAT_BC2_UNORM_BLOCK:
			return WGPUTextureFormat_BC2RGBAUnorm;
		case DF::DATA_FORMAT_BC2_SRGB_BLOCK:
			return WGPUTextureFormat_BC2RGBAUnormSrgb;
		case DF::DATA_FORMAT_BC3_UNORM_BLOCK:
			return WGPUTextureFormat_BC3RGBAUnorm;
		case DF::DATA_FORMAT_BC3_SRGB_BLOCK:
			return WGPUTextureFormat_BC3RGBAUnormSrgb;
		case DF::DATA_FORMAT_BC4_UNORM_BLOCK:
			return WGPUTextureFormat_BC4RUnorm;
		case DF::DATA_FORMAT_BC4_SNORM_BLOCK:
			return WGPUTextureFormat_BC4RSnorm;
		case DF::DATA_FORMAT_BC5_UNORM_BLOCK:
			return WGPUTextureFormat_BC5RGUnorm;
		case DF::DATA_FORMAT_BC5_SNORM_BLOCK:
			return WGPUTextureFormat_BC5RGSnorm;
		case DF::DATA_FORMAT_BC6H_UFLOAT_BLOCK:
			return WGPUTextureFormat_BC6HRGBUfloat;
		case DF::DATA_FORMAT_BC6H_SFLOAT_BLOCK:
			return WGPUTextureFormat_BC6HRGBFloat;
		case DF::DATA_FORMAT_BC7_UNORM_BLOCK:
			return WGPUTextureFormat_BC7RGBAUnorm;
		case DF::DATA_FORMAT_BC7_SRGB_BLOCK:
			return WGPUTextureFormat_BC7RGBAUnormSrgb;
		case DF::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
			return WGPUTextureFormat_ETC2RGB8Unorm;
		case DF::DATA_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
			return WGPUTextureFormat_ETC2RGB8UnormSrgb;
		case DF::DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
			return WGPUTextureFormat_ETC2RGB8A1Unorm;
		case DF::DATA_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
			return WGPUTextureFormat_ETC2RGB8A1UnormSrgb;
		case DF::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
			return WGPUTextureFormat_ETC2RGBA8Unorm;
		case DF::DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
			return WGPUTextureFormat_ETC2RGBA8UnormSrgb;
		case DF::DATA_FORMAT_EAC_R11_UNORM_BLOCK:
			return WGPUTextureFormat_EACR11Unorm;
		case DF::DATA_FORMAT_EAC_R11_SNORM_BLOCK:
			return WGPUTextureFormat_EACR11Snorm;
		case DF::DATA_FORMAT_EAC_R11G11_UNORM_BLOCK:
			return WGPUTextureFormat_EACRG11Unorm;
		case DF::DATA_FORMAT_EAC_R11G11_SNORM_BLOCK:
			return WGPUTextureFormat_EACRG11Snorm;
		case DF::DATA_FORMAT_ASTC_4x4_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC4x4Unorm;
		case DF::DATA_FORMAT_ASTC_4x4_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC4x4UnormSrgb;
		case DF::DATA_FORMAT_ASTC_5x4_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC5x4Unorm;
		case DF::DATA_FORMAT_ASTC_5x4_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC5x4UnormSrgb;
		case DF::DATA_FORMAT_ASTC_5x5_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC5x5Unorm;
		case DF::DATA_FORMAT_ASTC_5x5_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC5x5UnormSrgb;
		case DF::DATA_FORMAT_ASTC_6x5_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC6x5Unorm;
		case DF::DATA_FORMAT_ASTC_6x5_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC6x5UnormSrgb;
		case DF::DATA_FORMAT_ASTC_6x6_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC6x6Unorm;
		case DF::DATA_FORMAT_ASTC_6x6_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC6x6UnormSrgb;
		case DF::DATA_FORMAT_ASTC_8x5_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC8x5Unorm;
		case DF::DATA_FORMAT_ASTC_8x5_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC8x5UnormSrgb;
		case DF::DATA_FORMAT_ASTC_8x6_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC8x6Unorm;
		case DF::DATA_FORMAT_ASTC_8x6_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC8x6UnormSrgb;
		case DF::DATA_FORMAT_ASTC_8x8_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC8x8Unorm;
		case DF::DATA_FORMAT_ASTC_8x8_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC8x8UnormSrgb;
		case DF::DATA_FORMAT_ASTC_10x5_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC10x5Unorm;
		case DF::DATA_FORMAT_ASTC_10x5_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC10x5UnormSrgb;
		case DF::DATA_FORMAT_ASTC_10x6_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC10x6Unorm;
		case DF::DATA_FORMAT_ASTC_10x6_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC10x6UnormSrgb;
		case DF::DATA_FORMAT_ASTC_10x8_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC10x8Unorm;
		case DF::DATA_FORMAT_ASTC_10x8_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC10x8UnormSrgb;
		case DF::DATA_FORMAT_ASTC_10x10_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC10x10Unorm;
		case DF::DATA_FORMAT_ASTC_10x10_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC10x10UnormSrgb;
		case DF::DATA_FORMAT_ASTC_12x10_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC12x10Unorm;
		case DF::DATA_FORMAT_ASTC_12x10_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC12x10UnormSrgb;
		case DF::DATA_FORMAT_ASTC_12x12_UNORM_BLOCK:
			return WGPUTextureFormat_ASTC12x12Unorm;
		case DF::DATA_FORMAT_ASTC_12x12_SRGB_BLOCK:
			return WGPUTextureFormat_ASTC12x12UnormSrgb;
		default:
			return WGPUTextureFormat_Undefined;
	}
}

// ============================================================================
// DataFormat → WGPUVertexFormat mapping (for vertex buffers)
// ============================================================================
//
// WebGPU vertex formats are more limited than texture formats but support
// some formats that textures don't (e.g., Float32x3 for RGB32F vertices).
//
// This mapping is used in vertex_format_create() to build WGPUVertexAttribute
// descriptors from Godot VertexAttribute descriptions.
// ============================================================================

static inline WGPUVertexFormat data_format_to_wgpu_vertex(RenderingDeviceCommons::DataFormat p_format) {
	using DF = RenderingDeviceCommons::DataFormat;
	switch (p_format) {
		// 8-bit
		case DF::DATA_FORMAT_R8_UINT:
			return WGPUVertexFormat_Uint8;
		case DF::DATA_FORMAT_R8_SINT:
			return WGPUVertexFormat_Sint8;
		case DF::DATA_FORMAT_R8G8_UINT:
			return WGPUVertexFormat_Uint8x2;
		case DF::DATA_FORMAT_R8G8_SINT:
			return WGPUVertexFormat_Sint8x2;
		case DF::DATA_FORMAT_R8G8B8A8_UINT:
			return WGPUVertexFormat_Uint8x4;
		case DF::DATA_FORMAT_R8G8B8A8_SINT:
			return WGPUVertexFormat_Sint8x4;
		case DF::DATA_FORMAT_R8_UNORM:
			return WGPUVertexFormat_Unorm8;
		case DF::DATA_FORMAT_R8_SNORM:
			return WGPUVertexFormat_Snorm8;
		case DF::DATA_FORMAT_R8G8_UNORM:
			return WGPUVertexFormat_Unorm8x2;
		case DF::DATA_FORMAT_R8G8_SNORM:
			return WGPUVertexFormat_Snorm8x2;
		case DF::DATA_FORMAT_R8G8B8A8_UNORM:
			return WGPUVertexFormat_Unorm8x4;
		case DF::DATA_FORMAT_R8G8B8A8_SNORM:
			return WGPUVertexFormat_Snorm8x4;

		// 16-bit
		case DF::DATA_FORMAT_R16_UINT:
			return WGPUVertexFormat_Uint16;
		case DF::DATA_FORMAT_R16_SINT:
			return WGPUVertexFormat_Sint16;
		case DF::DATA_FORMAT_R16G16_UINT:
			return WGPUVertexFormat_Uint16x2;
		case DF::DATA_FORMAT_R16G16_SINT:
			return WGPUVertexFormat_Sint16x2;
		case DF::DATA_FORMAT_R16G16B16A16_UINT:
			return WGPUVertexFormat_Uint16x4;
		case DF::DATA_FORMAT_R16G16B16A16_SINT:
			return WGPUVertexFormat_Sint16x4;
		case DF::DATA_FORMAT_R16_UNORM:
			return WGPUVertexFormat_Unorm16;
		case DF::DATA_FORMAT_R16_SNORM:
			return WGPUVertexFormat_Snorm16;
		case DF::DATA_FORMAT_R16G16_UNORM:
			return WGPUVertexFormat_Unorm16x2;
		case DF::DATA_FORMAT_R16G16_SNORM:
			return WGPUVertexFormat_Snorm16x2;
		case DF::DATA_FORMAT_R16G16B16A16_UNORM:
			return WGPUVertexFormat_Unorm16x4;
		case DF::DATA_FORMAT_R16G16B16A16_SNORM:
			return WGPUVertexFormat_Snorm16x4;
		case DF::DATA_FORMAT_R16_SFLOAT:
			return WGPUVertexFormat_Float16;
		case DF::DATA_FORMAT_R16G16_SFLOAT:
			return WGPUVertexFormat_Float16x2;
		case DF::DATA_FORMAT_R16G16B16A16_SFLOAT:
			return WGPUVertexFormat_Float16x4;

		// 32-bit float
		case DF::DATA_FORMAT_R32_SFLOAT:
			return WGPUVertexFormat_Float32;
		case DF::DATA_FORMAT_R32G32_SFLOAT:
			return WGPUVertexFormat_Float32x2;
		case DF::DATA_FORMAT_R32G32B32_SFLOAT:
			return WGPUVertexFormat_Float32x3; // 3-comp OK for vertex!
		case DF::DATA_FORMAT_R32G32B32A32_SFLOAT:
			return WGPUVertexFormat_Float32x4;

		// 32-bit integer
		case DF::DATA_FORMAT_R32_UINT:
			return WGPUVertexFormat_Uint32;
		case DF::DATA_FORMAT_R32G32_UINT:
			return WGPUVertexFormat_Uint32x2;
		case DF::DATA_FORMAT_R32G32B32_UINT:
			return WGPUVertexFormat_Uint32x3; // 3-comp OK for vertex!
		case DF::DATA_FORMAT_R32G32B32A32_UINT:
			return WGPUVertexFormat_Uint32x4;
		case DF::DATA_FORMAT_R32_SINT:
			return WGPUVertexFormat_Sint32;
		case DF::DATA_FORMAT_R32G32_SINT:
			return WGPUVertexFormat_Sint32x2;
		case DF::DATA_FORMAT_R32G32B32_SINT:
			return WGPUVertexFormat_Sint32x3; // 3-comp OK for vertex!
		case DF::DATA_FORMAT_R32G32B32A32_SINT:
			return WGPUVertexFormat_Sint32x4;

		// 10-10-10-2 packed (A2B10G10R10)
		case DF::DATA_FORMAT_A2B10G10R10_UNORM_PACK32:
			return WGPUVertexFormat_Unorm10_10_10_2;

		default:
			return (WGPUVertexFormat)0; // WGPUVertexFormat_Undefined was removed in Dawn
	}
}

// ============================================================================
// Depth format helper: is this a depth or depth-stencil format?
// ============================================================================

static inline bool is_depth_format_wgpu(WGPUTextureFormat p_format) {
	switch (p_format) {
		case WGPUTextureFormat_Depth16Unorm:
		case WGPUTextureFormat_Depth24Plus:
		case WGPUTextureFormat_Depth24PlusStencil8:
		case WGPUTextureFormat_Depth32Float:
		case WGPUTextureFormat_Depth32FloatStencil8:
			return true;
		default:
			return false;
	}
}

static inline bool has_stencil_wgpu(WGPUTextureFormat p_format) {
	switch (p_format) {
		case WGPUTextureFormat_Stencil8:
		case WGPUTextureFormat_Depth24PlusStencil8:
		case WGPUTextureFormat_Depth32FloatStencil8:
			return true;
		default:
			return false;
	}
}

// ============================================================================
// Bytes per pixel for supported formats (used for buffer<->texture copies)
// ============================================================================

static inline uint32_t wgpu_format_pixel_size(WGPUTextureFormat p_format) {
	switch (p_format) {
		// 1 byte
		case WGPUTextureFormat_R8Unorm:
		case WGPUTextureFormat_R8Snorm:
		case WGPUTextureFormat_R8Uint:
		case WGPUTextureFormat_R8Sint:
		case WGPUTextureFormat_Stencil8:
			return 1;

		// 2 bytes
		case WGPUTextureFormat_R16Uint:
		case WGPUTextureFormat_R16Sint:
		case WGPUTextureFormat_R16Float:
		case WGPUTextureFormat_R16Unorm:
		case WGPUTextureFormat_R16Snorm:
		case WGPUTextureFormat_RG8Unorm:
		case WGPUTextureFormat_RG8Snorm:
		case WGPUTextureFormat_RG8Uint:
		case WGPUTextureFormat_RG8Sint:
		case WGPUTextureFormat_Depth16Unorm:
			return 2;

		// 4 bytes
		case WGPUTextureFormat_R32Uint:
		case WGPUTextureFormat_R32Sint:
		case WGPUTextureFormat_R32Float:
		case WGPUTextureFormat_RG16Uint:
		case WGPUTextureFormat_RG16Sint:
		case WGPUTextureFormat_RG16Float:
		case WGPUTextureFormat_RG16Unorm:
		case WGPUTextureFormat_RG16Snorm:
		case WGPUTextureFormat_RGBA8Unorm:
		case WGPUTextureFormat_RGBA8UnormSrgb:
		case WGPUTextureFormat_RGBA8Snorm:
		case WGPUTextureFormat_RGBA8Uint:
		case WGPUTextureFormat_RGBA8Sint:
		case WGPUTextureFormat_BGRA8Unorm:
		case WGPUTextureFormat_BGRA8UnormSrgb:
		case WGPUTextureFormat_RGB10A2Unorm:
		case WGPUTextureFormat_RGB10A2Uint:
		case WGPUTextureFormat_RG11B10Ufloat:
		case WGPUTextureFormat_RGB9E5Ufloat:
		case WGPUTextureFormat_Depth32Float:
		case WGPUTextureFormat_Depth24Plus:
			return 4;

		// 8 bytes
		case WGPUTextureFormat_RG32Uint:
		case WGPUTextureFormat_RG32Sint:
		case WGPUTextureFormat_RG32Float:
		case WGPUTextureFormat_RGBA16Uint:
		case WGPUTextureFormat_RGBA16Sint:
		case WGPUTextureFormat_RGBA16Float:
		case WGPUTextureFormat_RGBA16Unorm:
		case WGPUTextureFormat_RGBA16Snorm:
		case WGPUTextureFormat_Depth24PlusStencil8:
		case WGPUTextureFormat_Depth32FloatStencil8:
			return 8;

		// 16 bytes
		case WGPUTextureFormat_RGBA32Uint:
		case WGPUTextureFormat_RGBA32Sint:
		case WGPUTextureFormat_RGBA32Float:
			return 16;

		default:
			return 0; // Compressed or unknown — use block size calculations instead
	}
}

// ============================================================================
// Per-component encoding of plain color formats (CPU-side conversions)
// ============================================================================

enum WGTexelType : uint8_t {
	WG_TEXEL_UNORM8,
	WG_TEXEL_SNORM8,
	WG_TEXEL_UINT8,
	WG_TEXEL_SINT8,
	WG_TEXEL_UNORM16,
	WG_TEXEL_SNORM16,
	WG_TEXEL_UINT16,
	WG_TEXEL_SINT16,
	WG_TEXEL_FLOAT16,
	WG_TEXEL_UINT32,
	WG_TEXEL_SINT32,
	WG_TEXEL_FLOAT32,
};

static inline uint32_t wg_texel_component_size(WGTexelType p_type) {
	return p_type <= WG_TEXEL_SINT8 ? 1 : (p_type <= WG_TEXEL_FLOAT16 ? 2 : 4);
}

// Packed, depth and compressed formats return false.
static inline bool wgpu_format_texel_info(WGPUTextureFormat p_format, uint32_t &r_channels, WGTexelType &r_type) {
#define WG_TEXEL_CASE(m_format, m_channels, m_type) \
	case WGPUTextureFormat_##m_format: \
		r_channels = m_channels; \
		r_type = m_type; \
		return true;
	switch (p_format) {
		WG_TEXEL_CASE(R8Unorm, 1, WG_TEXEL_UNORM8)
		WG_TEXEL_CASE(R8Snorm, 1, WG_TEXEL_SNORM8)
		WG_TEXEL_CASE(R8Uint, 1, WG_TEXEL_UINT8)
		WG_TEXEL_CASE(R8Sint, 1, WG_TEXEL_SINT8)
		WG_TEXEL_CASE(RG8Unorm, 2, WG_TEXEL_UNORM8)
		WG_TEXEL_CASE(RG8Snorm, 2, WG_TEXEL_SNORM8)
		WG_TEXEL_CASE(RG8Uint, 2, WG_TEXEL_UINT8)
		WG_TEXEL_CASE(RG8Sint, 2, WG_TEXEL_SINT8)
		WG_TEXEL_CASE(RGBA8Unorm, 4, WG_TEXEL_UNORM8)
		WG_TEXEL_CASE(RGBA8UnormSrgb, 4, WG_TEXEL_UNORM8)
		WG_TEXEL_CASE(BGRA8Unorm, 4, WG_TEXEL_UNORM8)
		WG_TEXEL_CASE(BGRA8UnormSrgb, 4, WG_TEXEL_UNORM8)
		WG_TEXEL_CASE(RGBA8Snorm, 4, WG_TEXEL_SNORM8)
		WG_TEXEL_CASE(RGBA8Uint, 4, WG_TEXEL_UINT8)
		WG_TEXEL_CASE(RGBA8Sint, 4, WG_TEXEL_SINT8)
		WG_TEXEL_CASE(R16Unorm, 1, WG_TEXEL_UNORM16)
		WG_TEXEL_CASE(R16Snorm, 1, WG_TEXEL_SNORM16)
		WG_TEXEL_CASE(R16Uint, 1, WG_TEXEL_UINT16)
		WG_TEXEL_CASE(R16Sint, 1, WG_TEXEL_SINT16)
		WG_TEXEL_CASE(R16Float, 1, WG_TEXEL_FLOAT16)
		WG_TEXEL_CASE(RG16Unorm, 2, WG_TEXEL_UNORM16)
		WG_TEXEL_CASE(RG16Snorm, 2, WG_TEXEL_SNORM16)
		WG_TEXEL_CASE(RG16Uint, 2, WG_TEXEL_UINT16)
		WG_TEXEL_CASE(RG16Sint, 2, WG_TEXEL_SINT16)
		WG_TEXEL_CASE(RG16Float, 2, WG_TEXEL_FLOAT16)
		WG_TEXEL_CASE(RGBA16Unorm, 4, WG_TEXEL_UNORM16)
		WG_TEXEL_CASE(RGBA16Snorm, 4, WG_TEXEL_SNORM16)
		WG_TEXEL_CASE(RGBA16Uint, 4, WG_TEXEL_UINT16)
		WG_TEXEL_CASE(RGBA16Sint, 4, WG_TEXEL_SINT16)
		WG_TEXEL_CASE(RGBA16Float, 4, WG_TEXEL_FLOAT16)
		WG_TEXEL_CASE(R32Uint, 1, WG_TEXEL_UINT32)
		WG_TEXEL_CASE(R32Sint, 1, WG_TEXEL_SINT32)
		WG_TEXEL_CASE(R32Float, 1, WG_TEXEL_FLOAT32)
		WG_TEXEL_CASE(RG32Uint, 2, WG_TEXEL_UINT32)
		WG_TEXEL_CASE(RG32Sint, 2, WG_TEXEL_SINT32)
		WG_TEXEL_CASE(RG32Float, 2, WG_TEXEL_FLOAT32)
		WG_TEXEL_CASE(RGBA32Uint, 4, WG_TEXEL_UINT32)
		WG_TEXEL_CASE(RGBA32Sint, 4, WG_TEXEL_SINT32)
		WG_TEXEL_CASE(RGBA32Float, 4, WG_TEXEL_FLOAT32)
		default:
			return false;
	}
#undef WG_TEXEL_CASE
}

// Normalized and float components decode to their shader-visible value, integers to their integer value.
static inline double wg_texel_decode(WGTexelType p_type, const uint8_t *p_src) {
	uint16_t u16;
	int16_t i16;
	uint32_t u32;
	int32_t i32;
	float f32;
	switch (p_type) {
		case WG_TEXEL_UNORM8:
			return p_src[0] / 255.0;
		case WG_TEXEL_SNORM8:
			return MAX((int8_t)p_src[0] / 127.0, -1.0);
		case WG_TEXEL_UINT8:
			return p_src[0];
		case WG_TEXEL_SINT8:
			return (int8_t)p_src[0];
		case WG_TEXEL_UNORM16:
			memcpy(&u16, p_src, 2);
			return u16 / 65535.0;
		case WG_TEXEL_SNORM16:
			memcpy(&i16, p_src, 2);
			return MAX(i16 / 32767.0, -1.0);
		case WG_TEXEL_UINT16:
			memcpy(&u16, p_src, 2);
			return u16;
		case WG_TEXEL_SINT16:
			memcpy(&i16, p_src, 2);
			return i16;
		case WG_TEXEL_FLOAT16:
			memcpy(&u16, p_src, 2);
			return Math::half_to_float(u16);
		case WG_TEXEL_UINT32:
			memcpy(&u32, p_src, 4);
			return u32;
		case WG_TEXEL_SINT32:
			memcpy(&i32, p_src, 4);
			return i32;
		case WG_TEXEL_FLOAT32:
			memcpy(&f32, p_src, 4);
			return f32;
	}
	return 0.0;
}

static inline void wg_texel_encode(WGTexelType p_type, double p_value, uint8_t *r_dst) {
	uint16_t u16;
	int16_t i16;
	uint32_t u32;
	int32_t i32;
	float f32;
	switch (p_type) {
		case WG_TEXEL_UNORM8:
			r_dst[0] = (uint8_t)Math::round(CLAMP(p_value, 0.0, 1.0) * 255.0);
			break;
		case WG_TEXEL_SNORM8:
			r_dst[0] = (uint8_t)(int8_t)Math::round(CLAMP(p_value, -1.0, 1.0) * 127.0);
			break;
		case WG_TEXEL_UINT8:
			r_dst[0] = (uint8_t)CLAMP(Math::round(p_value), 0.0, 255.0);
			break;
		case WG_TEXEL_SINT8:
			r_dst[0] = (uint8_t)(int8_t)CLAMP(Math::round(p_value), -128.0, 127.0);
			break;
		case WG_TEXEL_UNORM16:
			u16 = (uint16_t)Math::round(CLAMP(p_value, 0.0, 1.0) * 65535.0);
			memcpy(r_dst, &u16, 2);
			break;
		case WG_TEXEL_SNORM16:
			i16 = (int16_t)Math::round(CLAMP(p_value, -1.0, 1.0) * 32767.0);
			memcpy(r_dst, &i16, 2);
			break;
		case WG_TEXEL_UINT16:
			u16 = (uint16_t)CLAMP(Math::round(p_value), 0.0, 65535.0);
			memcpy(r_dst, &u16, 2);
			break;
		case WG_TEXEL_SINT16:
			i16 = (int16_t)CLAMP(Math::round(p_value), -32768.0, 32767.0);
			memcpy(r_dst, &i16, 2);
			break;
		case WG_TEXEL_FLOAT16:
			u16 = Math::make_half_float((float)p_value);
			memcpy(r_dst, &u16, 2);
			break;
		case WG_TEXEL_UINT32:
			u32 = (uint32_t)CLAMP(Math::round(p_value), 0.0, 4294967295.0);
			memcpy(r_dst, &u32, 4);
			break;
		case WG_TEXEL_SINT32:
			i32 = (int32_t)CLAMP(Math::round(p_value), -2147483648.0, 2147483647.0);
			memcpy(r_dst, &i32, 4);
			break;
		case WG_TEXEL_FLOAT32:
			f32 = (float)p_value;
			memcpy(r_dst, &f32, 4);
			break;
	}
}
