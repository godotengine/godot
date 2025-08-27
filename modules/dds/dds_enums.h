/**************************************************************************/
/*  dds_enums.h                                                           */
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

#include "core/io/image.h"

// Reference: https://docs.microsoft.com/en-us/windows/win32/direct3ddds/dds-header

enum {
	DDS_MAGIC = make_fourcc("DDS "),
	DDS_HEADER_SIZE = 124,
	DDS_PIXELFORMAT_SIZE = 32,

	DDSD_PITCH = 0x00000008,
	DDSD_LINEARSIZE = 0x00080000,
	DDSD_MIPMAPCOUNT = 0x00020000,
	DDSD_CAPS = 0x1,
	DDSD_HEIGHT = 0x2,
	DDSD_WIDTH = 0x4,
	DDSD_PIXELFORMAT = 0x1000,
	DDPF_ALPHAPIXELS = 0x00000001,
	DDPF_ALPHAONLY = 0x00000002,
	DDPF_FOURCC = 0x00000004,
	DDPF_RGB = 0x00000040,
	DDPF_RG_SNORM = 0x00080000,

	DDSC2_CUBEMAP = 0x200,
	DDSC2_VOLUME = 0x200000,

	DX10D_1D = 2,
	DX10D_2D = 3,
	DX10D_3D = 4,
};

enum DDSFourCC {
	DDFCC_DXT1 = make_fourcc("DXT1"),
	DDFCC_DXT2 = make_fourcc("DXT2"),
	DDFCC_DXT3 = make_fourcc("DXT3"),
	DDFCC_DXT4 = make_fourcc("DXT4"),
	DDFCC_DXT5 = make_fourcc("DXT5"),
	DDFCC_ATI1 = make_fourcc("ATI1"),
	DDFCC_BC4U = make_fourcc("BC4U"),
	DDFCC_ATI2 = make_fourcc("ATI2"),
	DDFCC_BC5U = make_fourcc("BC5U"),
	DDFCC_A2XY = make_fourcc("A2XY"),
	DDFCC_DX10 = make_fourcc("DX10"),
	DDFCC_RGBA16 = 36,
	DDFCC_R16F = 111,
	DDFCC_RG16F = 112,
	DDFCC_RGBA16F = 113,
	DDFCC_R32F = 114,
	DDFCC_RG32F = 115,
	DDFCC_RGBA32F = 116,
};

// Reference: https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
enum DXGIFormat {
	DXGI_R32G32B32A32_FLOAT = 2,
	DXGI_R32G32B32_FLOAT = 6,
	DXGI_R16G16B16A16_FLOAT = 10,
	DXGI_R16G16B16A16_UNORM = 11,
	DXGI_R16G16B16A16_UINT = 12,
	DXGI_R32G32_FLOAT = 16,
	DXGI_R10G10B10A2_UNORM = 24,
	DXGI_R8G8B8A8_UNORM = 28,
	DXGI_R8G8B8A8_UNORM_SRGB = 29,
	DXGI_R16G16_FLOAT = 34,
	DXGI_R16G16_UNORM = 35,
	DXGI_R16G16_UINT = 36,
	DXGI_R32_FLOAT = 41,
	DXGI_R8G8_UNORM = 49,
	DXGI_R16_FLOAT = 54,
	DXGI_R16_UNORM = 56,
	DXGI_R16_UINT = 57,
	DXGI_R8_UNORM = 61,
	DXGI_A8_UNORM = 65,
	DXGI_R9G9B9E5 = 67,
	DXGI_BC1_UNORM = 71,
	DXGI_BC1_UNORM_SRGB = 72,
	DXGI_BC2_UNORM = 74,
	DXGI_BC2_UNORM_SRGB = 75,
	DXGI_BC3_UNORM = 77,
	DXGI_BC3_UNORM_SRGB = 78,
	DXGI_BC4_UNORM = 80,
	DXGI_BC5_UNORM = 83,
	DXGI_B5G6R5_UNORM = 85,
	DXGI_B5G5R5A1_UNORM = 86,
	DXGI_B8G8R8A8_UNORM = 87,
	DXGI_BC6H_UF16 = 95,
	DXGI_BC6H_SF16 = 96,
	DXGI_BC7_UNORM = 98,
	DXGI_BC7_UNORM_SRGB = 99,
	DXGI_B4G4R4A4_UNORM = 115,
};

// The legacy bitmasked format names here represent the actual data layout in the files,
// while their official names are flipped (e.g. RGBA8 layout is officially called ABGR8).
enum DDSFormat {
	DDS_DXT1,
	DDS_DXT3,
	DDS_DXT5,
	DDS_ATI1,
	DDS_ATI2,
	DDS_BC6U,
	DDS_BC6S,
	DDS_BC7,
	DDS_R16,
	DDS_RG16,
	DDS_RGBA16,
	DDS_R16I,
	DDS_RG16I,
	DDS_RGBA16I,
	DDS_R16F,
	DDS_RG16F,
	DDS_RGBA16F,
	DDS_R32F,
	DDS_RG32F,
	DDS_RGB32F,
	DDS_RGBA32F,
	DDS_RGB9E5,
	DDS_RGB8,
	DDS_RGBA8,
	DDS_RGBX8,
	DDS_BGR8,
	DDS_BGRA8,
	DDS_BGRX8,
	DDS_BGR5A1,
	DDS_BGR565,
	DDS_B2GR3,
	DDS_B2GR3A8,
	DDS_BGR10A2,
	DDS_RGB10A2,
	DDS_BGRA4,
	DDS_LUMINANCE,
	DDS_LUMINANCE_ALPHA,
	DDS_LUMINANCE_ALPHA_4,
	DDS_MAX
};

enum DDSType {
	DDST_2D = 1,
	DDST_CUBEMAP,
	DDST_3D,

	DDST_TYPE_MASK = 0x7F,
	DDST_ARRAY = 0x80,
};

struct DDSFormatInfo {
	const char *name = nullptr;
	bool compressed = false;
	uint32_t divisor = 0;
	uint32_t block_size = 0;
	Image::Format format = Image::Format::FORMAT_BPTC_RGBA;
};

static const DDSFormatInfo dds_format_info[DDS_MAX] = {
	{ "DXT1/BC1", true, 4, 8, Image::FORMAT_DXT1 },
	{ "DXT2/DXT3/BC2", true, 4, 16, Image::FORMAT_DXT3 },
	{ "DXT4/DXT5/BC3", true, 4, 16, Image::FORMAT_DXT5 },
	{ "ATI1/BC4", true, 4, 8, Image::FORMAT_RGTC_R },
	{ "ATI2/A2XY/BC5", true, 4, 16, Image::FORMAT_RGTC_RG },
	{ "BC6UF", true, 4, 16, Image::FORMAT_BPTC_RGBFU },
	{ "BC6SF", true, 4, 16, Image::FORMAT_BPTC_RGBF },
	{ "BC7", true, 4, 16, Image::FORMAT_BPTC_RGBA },
	{ "R16", false, 1, 2, Image::FORMAT_R16 },
	{ "RG16", false, 1, 4, Image::FORMAT_RG16 },
	{ "RGBA16", false, 1, 8, Image::FORMAT_RGBA16 },
	{ "R16I", false, 1, 2, Image::FORMAT_R16I },
	{ "RG16I", false, 1, 4, Image::FORMAT_RG16I },
	{ "RGBA16I", false, 1, 8, Image::FORMAT_RGBA16I },
	{ "R16F", false, 1, 2, Image::FORMAT_RH },
	{ "RG16F", false, 1, 4, Image::FORMAT_RGH },
	{ "RGBA16F", false, 1, 8, Image::FORMAT_RGBAH },
	{ "R32F", false, 1, 4, Image::FORMAT_RF },
	{ "RG32F", false, 1, 8, Image::FORMAT_RGF },
	{ "RGB32F", false, 1, 12, Image::FORMAT_RGBF },
	{ "RGBA32F", false, 1, 16, Image::FORMAT_RGBAF },
	{ "RGB9E5", false, 1, 4, Image::FORMAT_RGBE9995 },
	{ "RGB8", false, 1, 3, Image::FORMAT_RGB8 },
	{ "RGBA8", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "RGBX8", false, 1, 4, Image::FORMAT_RGB8 },
	{ "BGR8", false, 1, 3, Image::FORMAT_RGB8 },
	{ "BGRA8", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGRX8", false, 1, 4, Image::FORMAT_RGB8 },
	{ "BGR5A1", false, 1, 2, Image::FORMAT_RGBA8 },
	{ "BGR565", false, 1, 2, Image::FORMAT_RGB565 },
	{ "B2GR3", false, 1, 1, Image::FORMAT_RGB8 },
	{ "B2GR3A8", false, 1, 2, Image::FORMAT_RGBA8 },
	{ "BGR10A2", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "RGB10A2", false, 1, 4, Image::FORMAT_RGBA8 },
	{ "BGRA4", false, 1, 2, Image::FORMAT_RGBA4444 },
	{ "GRAYSCALE", false, 1, 1, Image::FORMAT_L8 },
	{ "GRAYSCALE_ALPHA", false, 1, 2, Image::FORMAT_LA8 },
	{ "GRAYSCALE_ALPHA_4", false, 1, 1, Image::FORMAT_LA8 },
};
