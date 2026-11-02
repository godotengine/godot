/**************************************************************************/
/*  texel_convert_test.cpp                                                */
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

#include "pixel_formats_webgpu.h"

#include <cassert>
#include <cstdio>

static double round_trip(WGPUTextureFormat p_format, double p_value) {
	uint32_t channels = 0;
	WGTexelType type = WG_TEXEL_FLOAT32;
	assert(wgpu_format_texel_info(p_format, channels, type));
	uint8_t bytes[4] = {};
	wg_texel_encode(type, p_value, bytes);
	return wg_texel_decode(type, bytes);
}

int main() {
	// A format must support multisampling in every enabled feature level, not just be renderable.
	for (bool core : { false, true }) {
		for (bool tier1 : { false, true }) {
			for (bool rg11b10 : { false, true }) {
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_RGBA8Unorm, core, tier1, rg11b10));
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_Depth24PlusStencil8, core, tier1, rg11b10));
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_RGBA16Float, core, tier1, rg11b10) == core);
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_RG8Uint, core, tier1, rg11b10) == core);
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_R32Float, core, tier1, rg11b10) == core);
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_RGBA8Snorm, core, tier1, rg11b10) == tier1);
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_RGBA16Unorm, core, tier1, rg11b10) == tier1);
				assert(wgpu_format_supports_multisampling(WGPUTextureFormat_RG11B10Ufloat, core, tier1, rg11b10) == rg11b10);
				assert(!wgpu_format_supports_multisampling(WGPUTextureFormat_RG32Float, core, tier1, rg11b10));
				assert(!wgpu_format_supports_multisampling(WGPUTextureFormat_RGBA32Float, core, tier1, rg11b10));
				assert(!wgpu_format_supports_multisampling(WGPUTextureFormat_R32Uint, core, tier1, rg11b10));
				assert(!wgpu_format_supports_multisampling(WGPUTextureFormat_BC1RGBAUnorm, core, tier1, rg11b10));
				assert(!wgpu_format_supports_multisampling(WGPUTextureFormat_Undefined, core, tier1, rg11b10));
			}
		}
	}
	// WGSL language support and texture-format capabilities are independent.
	assert(wgpu_storage_format_from_wgsl("rgba16sint, read_write>") == WGPUTextureFormat_RGBA16Sint);
	assert(wgpu_storage_format_from_wgsl("r32uint,read_write>") == WGPUTextureFormat_R32Uint);
	assert(wgpu_storage_format_from_wgsl("rgba16sint_extra, read_write>") == WGPUTextureFormat_Undefined);
	assert(wgpu_storage_format_supports_read_write(WGPUTextureFormat_R32Uint, true, false));
	assert(!wgpu_storage_format_supports_read_write(WGPUTextureFormat_R32Uint, false, true));
	assert(!wgpu_storage_format_supports_read_write(WGPUTextureFormat_RGBA16Sint, true, false));
	assert(wgpu_storage_format_supports_read_write(WGPUTextureFormat_RGBA16Sint, true, true));
	assert(!wgpu_storage_format_supports_read_write(WGPUTextureFormat_RG32Float, true, true));
	assert(!wgpu_storage_format_supports_read_write(WGPUTextureFormat_RGBA8Snorm, true, true));

	// Signed 8-bit formats keep their sign (R8 storage promotions go through these).
	assert(round_trip(WGPUTextureFormat_R8Sint, -5) == -5);
	assert(round_trip(WGPUTextureFormat_R8Snorm, -1.0) == -1.0);
	assert(round_trip(WGPUTextureFormat_R8Uint, 300) == 255);
	assert(round_trip(WGPUTextureFormat_R8Unorm, 1.0) == 1.0);

	// 16-bit normalized formats stored as half floats without texture-formats-tier1.
	uint8_t unorm16[2] = { 0xFF, 0xFF };
	assert(wg_texel_decode(WG_TEXEL_UNORM16, unorm16) == 1.0);
	uint8_t half[2] = {};
	wg_texel_encode(WG_TEXEL_FLOAT16, wg_texel_decode(WG_TEXEL_UNORM16, unorm16), half);
	assert(wg_texel_decode(WG_TEXEL_FLOAT16, half) == 1.0);
	assert(round_trip(WGPUTextureFormat_R16Snorm, -0.5) > -0.5001 && round_trip(WGPUTextureFormat_R16Snorm, -0.5) < -0.4999);

	// Channel counts follow the format, not the byte size (RG16Float is 4 bytes, 2 channels).
	uint32_t channels = 0;
	WGTexelType type = WG_TEXEL_FLOAT32;
	assert(wgpu_format_texel_info(WGPUTextureFormat_RG16Float, channels, type) && channels == 2 && type == WG_TEXEL_FLOAT16);
	assert(!wgpu_format_texel_info(WGPUTextureFormat_RGB10A2Unorm, channels, type));
	assert(wgpu_format_pixel_size(WGPUTextureFormat_RGBA16Unorm) == 8);

	std::puts("PASS: multisample/storage format capabilities and texel sign, range and channel layout");
}
