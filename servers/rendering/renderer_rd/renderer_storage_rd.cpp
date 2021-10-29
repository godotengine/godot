/*************************************************************************/
/*  renderer_storage_rd.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "renderer_storage_rd.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/math/math_defs.h"
#include "renderer_compositor_rd.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/shader_language.h"

bool RendererStorageRD::can_create_resources_async() const {
	return true;
}

Ref<Image> RendererStorageRD::_validate_texture_format(const Ref<Image> &p_image, TextureToRDFormat &r_format) {
	Ref<Image> image = p_image->duplicate();

	switch (p_image->get_format()) {
		case Image::FORMAT_L8: {
			r_format.format = RD::DATA_FORMAT_R8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //luminance
		case Image::FORMAT_LA8: {
			r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_G;
		} break; //luminance-alpha
		case Image::FORMAT_R8: {
			r_format.format = RD::DATA_FORMAT_R8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RG8: {
			r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGB8: {
			//this format is not mandatory for specification, check if supported first
			if (false && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R8G8B8_UNORM, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT) && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R8G8B8_SRGB, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_R8G8B8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8_SRGB;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_RGBA8: {
			r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_RGBA4444: {
			r_format.format = RD::DATA_FORMAT_B4G4R4A4_UNORM_PACK16;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_B; //needs swizzle
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_RGB565: {
			r_format.format = RD::DATA_FORMAT_B5G6R5_UNORM_PACK16;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_RF: {
			r_format.format = RD::DATA_FORMAT_R32_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //float
		case Image::FORMAT_RGF: {
			r_format.format = RD::DATA_FORMAT_R32G32_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGBF: {
			//this format is not mandatory for specification, check if supported first
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R32G32B32_SFLOAT, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
				image->convert(Image::FORMAT_RGBAF);
			}

			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGBAF: {
			r_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break;
		case Image::FORMAT_RH: {
			r_format.format = RD::DATA_FORMAT_R16_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break; //half float
		case Image::FORMAT_RGH: {
			r_format.format = RD::DATA_FORMAT_R16G16_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_RGBH: {
			//this format is not mandatory for specification, check if supported first
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_R16G16B16_SFLOAT, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_R16G16B16_SFLOAT;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				image->convert(Image::FORMAT_RGBAH);
			}

			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_RGBAH: {
			r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break;
		case Image::FORMAT_RGBE9995: {
			r_format.format = RD::DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32;
#ifndef _MSC_VER
#warning TODO need to make a function in Image to swap bits for this
#endif
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_IDENTITY;
		} break;
		case Image::FORMAT_DXT1: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC1_RGB_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break; //s3tc bc1
		case Image::FORMAT_DXT3: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC2_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC2_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC2_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break; //bc2
		case Image::FORMAT_DXT5: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC3_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC3_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC3_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break; //bc3
		case Image::FORMAT_RGTC_R: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC4_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC4_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_R8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_RGTC_RG: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC5_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC5_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_RG8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_BPTC_RGBA: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC7_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC7_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC7_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;

		} break; //btpc bc7
		case Image::FORMAT_BPTC_RGBF: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC6H_SFLOAT_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC6H_SFLOAT_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				image->decompress();
				image->convert(Image::FORMAT_RGBAH);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //float bc6h
		case Image::FORMAT_BPTC_RGBFU: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC6H_UFLOAT_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC6H_UFLOAT_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
				image->decompress();
				image->convert(Image::FORMAT_RGBAH);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //unsigned float bc6hu
		case Image::FORMAT_PVRTC1_2: {
			//this is not properly supported by MoltekVK it seems, so best to use ETC2
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
				r_format.format_srgb = RD::DATA_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break; //pvrtc
		case Image::FORMAT_PVRTC1_2A: {
			//this is not properly supported by MoltekVK it seems, so best to use ETC2
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;
				r_format.format_srgb = RD::DATA_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_PVRTC1_4: {
			//this is not properly supported by MoltekVK it seems, so best to use ETC2
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;
				r_format.format_srgb = RD::DATA_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_PVRTC1_4A: {
			//this is not properly supported by MoltekVK it seems, so best to use ETC2
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;
				r_format.format_srgb = RD::DATA_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_ETC2_R11: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_R8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break; //etc2
		case Image::FORMAT_ETC2_R11S: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11_SNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11_SNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8_SNORM;
				image->decompress();
				image->convert(Image::FORMAT_R8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break; //signed: {} break; NOT srgb.
		case Image::FORMAT_ETC2_RG11: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11G11_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11G11_UNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8_UNORM;
				image->decompress();
				image->convert(Image::FORMAT_RG8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_ETC2_RG11S: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_EAC_R11G11_SNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_EAC_R11G11_SNORM_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8_SNORM;
				image->decompress();
				image->convert(Image::FORMAT_RG8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_ETC:
		case Image::FORMAT_ETC2_RGB8: {
			//ETC2 is backwards compatible with ETC1, and all modern platforms support it
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;

		} break;
		case Image::FORMAT_ETC2_RGBA8: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_ETC2_RGB8A1: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_A;
		} break;
		case Image::FORMAT_ETC2_RA_AS_RG: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_A;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;
		case Image::FORMAT_DXT5_RA_AS_RG: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC3_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC3_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC3_SRGB_BLOCK;
			} else {
				//not supported, reconvert
				r_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
				r_format.format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
				image->decompress();
				image->convert(Image::FORMAT_RGBA8);
			}
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_A;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_ZERO;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		} break;

		default: {
		}
	}

	return image;
}

RID RendererStorageRD::texture_allocate() {
	return texture_owner.allocate_rid();
}

void RendererStorageRD::texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) {
	ERR_FAIL_COND(p_image.is_null());
	ERR_FAIL_COND(p_image->is_empty());

	TextureToRDFormat ret_format;
	Ref<Image> image = _validate_texture_format(p_image, ret_format);

	Texture texture;

	texture.type = Texture::TYPE_2D;

	texture.width = p_image->get_width();
	texture.height = p_image->get_height();
	texture.layers = 1;
	texture.mipmaps = p_image->get_mipmap_count() + 1;
	texture.depth = 1;
	texture.format = p_image->get_format();
	texture.validated_format = image->get_format();

	texture.rd_type = RD::TEXTURE_TYPE_2D;
	texture.rd_format = ret_format.format;
	texture.rd_format_srgb = ret_format.format_srgb;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = texture.rd_format;
		rd_format.width = texture.width;
		rd_format.height = texture.height;
		rd_format.depth = 1;
		rd_format.array_layers = 1;
		rd_format.mipmaps = texture.mipmaps;
		rd_format.texture_type = texture.rd_type;
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
			rd_format.shareable_formats.push_back(texture.rd_format);
			rd_format.shareable_formats.push_back(texture.rd_format_srgb);
		}
	}
	{
		rd_view.swizzle_r = ret_format.swizzle_r;
		rd_view.swizzle_g = ret_format.swizzle_g;
		rd_view.swizzle_b = ret_format.swizzle_b;
		rd_view.swizzle_a = ret_format.swizzle_a;
	}
	Vector<uint8_t> data = image->get_data(); //use image data
	Vector<Vector<uint8_t>> data_slices;
	data_slices.push_back(data);
	texture.rd_texture = RD::get_singleton()->texture_create(rd_format, rd_view, data_slices);
	ERR_FAIL_COND(texture.rd_texture.is_null());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND(texture.rd_texture_srgb.is_null());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	texture_owner.initialize_rid(p_texture, texture);
}

void RendererStorageRD::texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) {
	ERR_FAIL_COND(p_layers.size() == 0);

	ERR_FAIL_COND(p_layered_type == RS::TEXTURE_LAYERED_CUBEMAP && p_layers.size() != 6);
	ERR_FAIL_COND(p_layered_type == RS::TEXTURE_LAYERED_CUBEMAP_ARRAY && (p_layers.size() < 6 || (p_layers.size() % 6) != 0));

	TextureToRDFormat ret_format;
	Vector<Ref<Image>> images;
	{
		int valid_width = 0;
		int valid_height = 0;
		bool valid_mipmaps = false;
		Image::Format valid_format = Image::FORMAT_MAX;

		for (int i = 0; i < p_layers.size(); i++) {
			ERR_FAIL_COND(p_layers[i]->is_empty());

			if (i == 0) {
				valid_width = p_layers[i]->get_width();
				valid_height = p_layers[i]->get_height();
				valid_format = p_layers[i]->get_format();
				valid_mipmaps = p_layers[i]->has_mipmaps();
			} else {
				ERR_FAIL_COND(p_layers[i]->get_width() != valid_width);
				ERR_FAIL_COND(p_layers[i]->get_height() != valid_height);
				ERR_FAIL_COND(p_layers[i]->get_format() != valid_format);
				ERR_FAIL_COND(p_layers[i]->has_mipmaps() != valid_mipmaps);
			}

			images.push_back(_validate_texture_format(p_layers[i], ret_format));
		}
	}

	Texture texture;

	texture.type = Texture::TYPE_LAYERED;
	texture.layered_type = p_layered_type;

	texture.width = p_layers[0]->get_width();
	texture.height = p_layers[0]->get_height();
	texture.layers = p_layers.size();
	texture.mipmaps = p_layers[0]->get_mipmap_count() + 1;
	texture.depth = 1;
	texture.format = p_layers[0]->get_format();
	texture.validated_format = images[0]->get_format();

	switch (p_layered_type) {
		case RS::TEXTURE_LAYERED_2D_ARRAY: {
			texture.rd_type = RD::TEXTURE_TYPE_2D_ARRAY;
		} break;
		case RS::TEXTURE_LAYERED_CUBEMAP: {
			texture.rd_type = RD::TEXTURE_TYPE_CUBE;
		} break;
		case RS::TEXTURE_LAYERED_CUBEMAP_ARRAY: {
			texture.rd_type = RD::TEXTURE_TYPE_CUBE_ARRAY;
		} break;
	}

	texture.rd_format = ret_format.format;
	texture.rd_format_srgb = ret_format.format_srgb;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = texture.rd_format;
		rd_format.width = texture.width;
		rd_format.height = texture.height;
		rd_format.depth = 1;
		rd_format.array_layers = texture.layers;
		rd_format.mipmaps = texture.mipmaps;
		rd_format.texture_type = texture.rd_type;
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
			rd_format.shareable_formats.push_back(texture.rd_format);
			rd_format.shareable_formats.push_back(texture.rd_format_srgb);
		}
	}
	{
		rd_view.swizzle_r = ret_format.swizzle_r;
		rd_view.swizzle_g = ret_format.swizzle_g;
		rd_view.swizzle_b = ret_format.swizzle_b;
		rd_view.swizzle_a = ret_format.swizzle_a;
	}
	Vector<Vector<uint8_t>> data_slices;
	for (int i = 0; i < images.size(); i++) {
		Vector<uint8_t> data = images[i]->get_data(); //use image data
		data_slices.push_back(data);
	}
	texture.rd_texture = RD::get_singleton()->texture_create(rd_format, rd_view, data_slices);
	ERR_FAIL_COND(texture.rd_texture.is_null());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND(texture.rd_texture_srgb.is_null());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	texture_owner.initialize_rid(p_texture, texture);
}

void RendererStorageRD::texture_3d_initialize(RID p_texture, Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) {
	ERR_FAIL_COND(p_data.size() == 0);
	Image::Image3DValidateError verr = Image::validate_3d_image(p_format, p_width, p_height, p_depth, p_mipmaps, p_data);
	if (verr != Image::VALIDATE_3D_OK) {
		ERR_FAIL_MSG(Image::get_3d_image_validation_error_text(verr));
	}

	TextureToRDFormat ret_format;
	Image::Format validated_format = Image::FORMAT_MAX;
	Vector<uint8_t> all_data;
	uint32_t mipmap_count = 0;
	Vector<Texture::BufferSlice3D> slices;
	{
		Vector<Ref<Image>> images;
		uint32_t all_data_size = 0;
		images.resize(p_data.size());
		for (int i = 0; i < p_data.size(); i++) {
			TextureToRDFormat f;
			images.write[i] = _validate_texture_format(p_data[i], f);
			if (i == 0) {
				ret_format = f;
				validated_format = images[0]->get_format();
			}

			all_data_size += images[i]->get_data().size();
		}

		all_data.resize(all_data_size); //consolidate all data here
		uint32_t offset = 0;
		Size2i prev_size;
		for (int i = 0; i < p_data.size(); i++) {
			uint32_t s = images[i]->get_data().size();

			memcpy(&all_data.write[offset], images[i]->get_data().ptr(), s);
			{
				Texture::BufferSlice3D slice;
				slice.size.width = images[i]->get_width();
				slice.size.height = images[i]->get_height();
				slice.offset = offset;
				slice.buffer_size = s;
				slices.push_back(slice);
			}
			offset += s;

			Size2i img_size(images[i]->get_width(), images[i]->get_height());
			if (img_size != prev_size) {
				mipmap_count++;
			}
			prev_size = img_size;
		}
	}

	Texture texture;

	texture.type = Texture::TYPE_3D;
	texture.width = p_width;
	texture.height = p_height;
	texture.depth = p_depth;
	texture.mipmaps = mipmap_count;
	texture.format = p_data[0]->get_format();
	texture.validated_format = validated_format;

	texture.buffer_size_3d = all_data.size();
	texture.buffer_slices_3d = slices;

	texture.rd_type = RD::TEXTURE_TYPE_3D;
	texture.rd_format = ret_format.format;
	texture.rd_format_srgb = ret_format.format_srgb;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = texture.rd_format;
		rd_format.width = texture.width;
		rd_format.height = texture.height;
		rd_format.depth = texture.depth;
		rd_format.array_layers = 1;
		rd_format.mipmaps = texture.mipmaps;
		rd_format.texture_type = texture.rd_type;
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
			rd_format.shareable_formats.push_back(texture.rd_format);
			rd_format.shareable_formats.push_back(texture.rd_format_srgb);
		}
	}
	{
		rd_view.swizzle_r = ret_format.swizzle_r;
		rd_view.swizzle_g = ret_format.swizzle_g;
		rd_view.swizzle_b = ret_format.swizzle_b;
		rd_view.swizzle_a = ret_format.swizzle_a;
	}
	Vector<Vector<uint8_t>> data_slices;
	data_slices.push_back(all_data); //one slice

	texture.rd_texture = RD::get_singleton()->texture_create(rd_format, rd_view, data_slices);
	ERR_FAIL_COND(texture.rd_texture.is_null());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND(texture.rd_texture_srgb.is_null());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	texture_owner.initialize_rid(p_texture, texture);
}

void RendererStorageRD::texture_proxy_initialize(RID p_texture, RID p_base) {
	Texture *tex = texture_owner.get_or_null(p_base);
	ERR_FAIL_COND(!tex);
	Texture proxy_tex = *tex;

	proxy_tex.rd_view.format_override = tex->rd_format;
	proxy_tex.rd_texture = RD::get_singleton()->texture_create_shared(proxy_tex.rd_view, tex->rd_texture);
	if (proxy_tex.rd_texture_srgb.is_valid()) {
		proxy_tex.rd_view.format_override = tex->rd_format_srgb;
		proxy_tex.rd_texture_srgb = RD::get_singleton()->texture_create_shared(proxy_tex.rd_view, tex->rd_texture);
	}
	proxy_tex.proxy_to = p_base;
	proxy_tex.is_render_target = false;
	proxy_tex.is_proxy = true;
	proxy_tex.proxies.clear();

	texture_owner.initialize_rid(p_texture, proxy_tex);

	tex->proxies.push_back(p_texture);
}

void RendererStorageRD::_texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer, bool p_immediate) {
	ERR_FAIL_COND(p_image.is_null() || p_image->is_empty());

	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->is_render_target);
	ERR_FAIL_COND(p_image->get_width() != tex->width || p_image->get_height() != tex->height);
	ERR_FAIL_COND(p_image->get_format() != tex->format);

	if (tex->type == Texture::TYPE_LAYERED) {
		ERR_FAIL_INDEX(p_layer, tex->layers);
	}

#ifdef TOOLS_ENABLED
	tex->image_cache_2d.unref();
#endif
	TextureToRDFormat f;
	Ref<Image> validated = _validate_texture_format(p_image, f);

	RD::get_singleton()->texture_update(tex->rd_texture, p_layer, validated->get_data());
}

void RendererStorageRD::texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	_texture_2d_update(p_texture, p_image, p_layer, false);
}

void RendererStorageRD::texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->type != Texture::TYPE_3D);
	Image::Image3DValidateError verr = Image::validate_3d_image(tex->format, tex->width, tex->height, tex->depth, tex->mipmaps > 1, p_data);
	if (verr != Image::VALIDATE_3D_OK) {
		ERR_FAIL_MSG(Image::get_3d_image_validation_error_text(verr));
	}

	Vector<uint8_t> all_data;
	{
		Vector<Ref<Image>> images;
		uint32_t all_data_size = 0;
		images.resize(p_data.size());
		for (int i = 0; i < p_data.size(); i++) {
			Ref<Image> image = p_data[i];
			if (image->get_format() != tex->validated_format) {
				image = image->duplicate();
				image->convert(tex->validated_format);
			}
			all_data_size += images[i]->get_data().size();
			images.push_back(image);
		}

		all_data.resize(all_data_size); //consolidate all data here
		uint32_t offset = 0;

		for (int i = 0; i < p_data.size(); i++) {
			uint32_t s = images[i]->get_data().size();
			memcpy(&all_data.write[offset], images[i]->get_data().ptr(), s);
			offset += s;
		}
	}

	RD::get_singleton()->texture_update(tex->rd_texture, 0, all_data);
}

void RendererStorageRD::texture_proxy_update(RID p_texture, RID p_proxy_to) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(!tex->is_proxy);
	Texture *proxy_to = texture_owner.get_or_null(p_proxy_to);
	ERR_FAIL_COND(!proxy_to);
	ERR_FAIL_COND(proxy_to->is_proxy);

	if (tex->proxy_to.is_valid()) {
		//unlink proxy
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture)) {
			RD::get_singleton()->free(tex->rd_texture);
			tex->rd_texture = RID();
		}
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture_srgb)) {
			RD::get_singleton()->free(tex->rd_texture_srgb);
			tex->rd_texture_srgb = RID();
		}
		Texture *prev_tex = texture_owner.get_or_null(tex->proxy_to);
		ERR_FAIL_COND(!prev_tex);
		prev_tex->proxies.erase(p_texture);
	}

	*tex = *proxy_to;

	tex->proxy_to = p_proxy_to;
	tex->is_render_target = false;
	tex->is_proxy = true;
	tex->proxies.clear();
	proxy_to->proxies.push_back(p_texture);

	tex->rd_view.format_override = tex->rd_format;
	tex->rd_texture = RD::get_singleton()->texture_create_shared(tex->rd_view, proxy_to->rd_texture);
	if (tex->rd_texture_srgb.is_valid()) {
		tex->rd_view.format_override = tex->rd_format_srgb;
		tex->rd_texture_srgb = RD::get_singleton()->texture_create_shared(tex->rd_view, proxy_to->rd_texture);
	}
}

//these two APIs can be used together or in combination with the others.
void RendererStorageRD::texture_2d_placeholder_initialize(RID p_texture) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			image->set_pixel(i, j, Color(1, 0, 1, 1));
		}
	}

	texture_2d_initialize(p_texture, image);
}

void RendererStorageRD::texture_2d_layered_placeholder_initialize(RID p_texture, RS::TextureLayeredType p_layered_type) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			image->set_pixel(i, j, Color(1, 0, 1, 1));
		}
	}

	Vector<Ref<Image>> images;
	if (p_layered_type == RS::TEXTURE_LAYERED_2D_ARRAY) {
		images.push_back(image);
	} else {
		//cube
		for (int i = 0; i < 6; i++) {
			images.push_back(image);
		}
	}

	texture_2d_layered_initialize(p_texture, images, p_layered_type);
}

void RendererStorageRD::texture_3d_placeholder_initialize(RID p_texture) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instantiate();
	image->create(4, 4, false, Image::FORMAT_RGBA8);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			image->set_pixel(i, j, Color(1, 0, 1, 1));
		}
	}

	Vector<Ref<Image>> images;
	//cube
	for (int i = 0; i < 4; i++) {
		images.push_back(image);
	}

	texture_3d_initialize(p_texture, Image::FORMAT_RGBA8, 4, 4, 4, false, images);
}

Ref<Image> RendererStorageRD::texture_2d_get(RID p_texture) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, Ref<Image>());

#ifdef TOOLS_ENABLED
	if (tex->image_cache_2d.is_valid()) {
		return tex->image_cache_2d;
	}
#endif
	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(tex->rd_texture, 0);
	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instantiate();
	image->create(tex->width, tex->height, tex->mipmaps > 1, tex->validated_format, data);
	ERR_FAIL_COND_V(image->is_empty(), Ref<Image>());
	if (tex->format != tex->validated_format) {
		image->convert(tex->format);
	}

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		tex->image_cache_2d = image;
	}
#endif

	return image;
}

Ref<Image> RendererStorageRD::texture_2d_layer_get(RID p_texture, int p_layer) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, Ref<Image>());

	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(tex->rd_texture, p_layer);
	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instantiate();
	image->create(tex->width, tex->height, tex->mipmaps > 1, tex->validated_format, data);
	ERR_FAIL_COND_V(image->is_empty(), Ref<Image>());
	if (tex->format != tex->validated_format) {
		image->convert(tex->format);
	}

	return image;
}

Vector<Ref<Image>> RendererStorageRD::texture_3d_get(RID p_texture) const {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND_V(!tex, Vector<Ref<Image>>());
	ERR_FAIL_COND_V(tex->type != Texture::TYPE_3D, Vector<Ref<Image>>());

	Vector<uint8_t> all_data = RD::get_singleton()->texture_get_data(tex->rd_texture, 0);

	ERR_FAIL_COND_V(all_data.size() != (int)tex->buffer_size_3d, Vector<Ref<Image>>());

	Vector<Ref<Image>> ret;

	for (int i = 0; i < tex->buffer_slices_3d.size(); i++) {
		const Texture::BufferSlice3D &bs = tex->buffer_slices_3d[i];
		ERR_FAIL_COND_V(bs.offset >= (uint32_t)all_data.size(), Vector<Ref<Image>>());
		ERR_FAIL_COND_V(bs.offset + bs.buffer_size > (uint32_t)all_data.size(), Vector<Ref<Image>>());
		Vector<uint8_t> sub_region = all_data.subarray(bs.offset, bs.offset + bs.buffer_size - 1);

		Ref<Image> img;
		img.instantiate();
		img->create(bs.size.width, bs.size.height, false, tex->validated_format, sub_region);
		ERR_FAIL_COND_V(img->is_empty(), Vector<Ref<Image>>());
		if (tex->format != tex->validated_format) {
			img->convert(tex->format);
		}

		ret.push_back(img);
	}

	return ret;
}

void RendererStorageRD::texture_replace(RID p_texture, RID p_by_texture) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->proxy_to.is_valid()); //can't replace proxy
	Texture *by_tex = texture_owner.get_or_null(p_by_texture);
	ERR_FAIL_COND(!by_tex);
	ERR_FAIL_COND(by_tex->proxy_to.is_valid()); //can't replace proxy

	if (tex == by_tex) {
		return;
	}

	if (tex->rd_texture_srgb.is_valid()) {
		RD::get_singleton()->free(tex->rd_texture_srgb);
	}
	RD::get_singleton()->free(tex->rd_texture);

	if (tex->canvas_texture) {
		memdelete(tex->canvas_texture);
		tex->canvas_texture = nullptr;
	}

	Vector<RID> proxies_to_update = tex->proxies;
	Vector<RID> proxies_to_redirect = by_tex->proxies;

	*tex = *by_tex;

	tex->proxies = proxies_to_update; //restore proxies, so they can be updated

	if (tex->canvas_texture) {
		tex->canvas_texture->diffuse = p_texture; //update
	}

	for (int i = 0; i < proxies_to_update.size(); i++) {
		texture_proxy_update(proxies_to_update[i], p_texture);
	}
	for (int i = 0; i < proxies_to_redirect.size(); i++) {
		texture_proxy_update(proxies_to_redirect[i], p_texture);
	}
	//delete last, so proxies can be updated
	texture_owner.free(p_by_texture);

	if (decal_atlas.textures.has(p_texture)) {
		//belongs to decal atlas..

		decal_atlas.dirty = true; //mark it dirty since it was most likely modified
	}
}

void RendererStorageRD::texture_set_size_override(RID p_texture, int p_width, int p_height) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->type != Texture::TYPE_2D);
	tex->width_2d = p_width;
	tex->height_2d = p_height;
}

void RendererStorageRD::texture_set_path(RID p_texture, const String &p_path) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	tex->path = p_path;
}

String RendererStorageRD::texture_get_path(RID p_texture) const {
	return String();
}

void RendererStorageRD::texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	tex->detect_3d_callback_ud = p_userdata;
	tex->detect_3d_callback = p_callback;
}

void RendererStorageRD::texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	tex->detect_normal_callback_ud = p_userdata;
	tex->detect_normal_callback = p_callback;
}

void RendererStorageRD::texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.get_or_null(p_texture);
	ERR_FAIL_COND(!tex);
	tex->detect_roughness_callback_ud = p_userdata;
	tex->detect_roughness_callback = p_callback;
}

void RendererStorageRD::texture_debug_usage(List<RS::TextureInfo> *r_info) {
}

void RendererStorageRD::texture_set_proxy(RID p_proxy, RID p_base) {
}

void RendererStorageRD::texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {
}

Size2 RendererStorageRD::texture_size_with_proxy(RID p_proxy) {
	return texture_2d_get_size(p_proxy);
}

/* CANVAS TEXTURE */

void RendererStorageRD::CanvasTexture::clear_sets() {
	if (cleared_cache) {
		return;
	}
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			if (RD::get_singleton()->uniform_set_is_valid(uniform_sets[i][j])) {
				RD::get_singleton()->free(uniform_sets[i][j]);
				uniform_sets[i][j] = RID();
			}
		}
	}
	cleared_cache = true;
}

RendererStorageRD::CanvasTexture::~CanvasTexture() {
	clear_sets();
}

RID RendererStorageRD::canvas_texture_allocate() {
	return canvas_texture_owner.allocate_rid();
}
void RendererStorageRD::canvas_texture_initialize(RID p_rid) {
	canvas_texture_owner.initialize_rid(p_rid);
}

void RendererStorageRD::canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	switch (p_channel) {
		case RS::CANVAS_TEXTURE_CHANNEL_DIFFUSE: {
			ct->diffuse = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_NORMAL: {
			ct->normal_map = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_SPECULAR: {
			ct->specular = p_texture;
		} break;
	}

	ct->clear_sets();
}

void RendererStorageRD::canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_specular_color, float p_shininess) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ct->specular_color.r = p_specular_color.r;
	ct->specular_color.g = p_specular_color.g;
	ct->specular_color.b = p_specular_color.b;
	ct->specular_color.a = p_shininess;
	ct->clear_sets();
}

void RendererStorageRD::canvas_texture_set_texture_filter(RID p_canvas_texture, RS::CanvasItemTextureFilter p_filter) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ct->texture_filter = p_filter;
	ct->clear_sets();
}

void RendererStorageRD::canvas_texture_set_texture_repeat(RID p_canvas_texture, RS::CanvasItemTextureRepeat p_repeat) {
	CanvasTexture *ct = canvas_texture_owner.get_or_null(p_canvas_texture);
	ct->texture_repeat = p_repeat;
	ct->clear_sets();
}

bool RendererStorageRD::canvas_texture_get_uniform_set(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, RID p_base_shader, int p_base_set, RID &r_uniform_set, Size2i &r_size, Color &r_specular_shininess, bool &r_use_normal, bool &r_use_specular) {
	CanvasTexture *ct = nullptr;

	Texture *t = texture_owner.get_or_null(p_texture);

	if (t) {
		//regular texture
		if (!t->canvas_texture) {
			t->canvas_texture = memnew(CanvasTexture);
			t->canvas_texture->diffuse = p_texture;
		}

		ct = t->canvas_texture;
	} else {
		ct = canvas_texture_owner.get_or_null(p_texture);
	}

	if (!ct) {
		return false; //invalid texture RID
	}

	RS::CanvasItemTextureFilter filter = ct->texture_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT ? ct->texture_filter : p_base_filter;
	ERR_FAIL_COND_V(filter == RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, false);

	RS::CanvasItemTextureRepeat repeat = ct->texture_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT ? ct->texture_repeat : p_base_repeat;
	ERR_FAIL_COND_V(repeat == RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, false);

	RID uniform_set = ct->uniform_sets[filter][repeat];
	if (!RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//create and update
		Vector<RD::Uniform> uniforms;
		{ //diffuse
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 0;

			t = texture_owner.get_or_null(ct->diffuse);
			if (!t) {
				u.ids.push_back(texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE));
				ct->size_cache = Size2i(1, 1);
			} else {
				u.ids.push_back(t->rd_texture);
				ct->size_cache = Size2i(t->width_2d, t->height_2d);
			}
			uniforms.push_back(u);
		}
		{ //normal
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;

			t = texture_owner.get_or_null(ct->normal_map);
			if (!t) {
				u.ids.push_back(texture_rd_get_default(DEFAULT_RD_TEXTURE_NORMAL));
				ct->use_normal_cache = false;
			} else {
				u.ids.push_back(t->rd_texture);
				ct->use_normal_cache = true;
			}
			uniforms.push_back(u);
		}
		{ //specular
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2;

			t = texture_owner.get_or_null(ct->specular);
			if (!t) {
				u.ids.push_back(texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE));
				ct->use_specular_cache = false;
			} else {
				u.ids.push_back(t->rd_texture);
				ct->use_specular_cache = true;
			}
			uniforms.push_back(u);
		}
		{ //sampler
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 3;
			u.ids.push_back(sampler_rd_get_default(filter, repeat));
			uniforms.push_back(u);
		}

		uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_base_shader, p_base_set);
		ct->uniform_sets[filter][repeat] = uniform_set;
		ct->cleared_cache = false;
	}

	r_uniform_set = uniform_set;
	r_size = ct->size_cache;
	r_specular_shininess = ct->specular_color;
	r_use_normal = ct->use_normal_cache;
	r_use_specular = ct->use_specular_cache;

	return true;
}

/* SHADER API */

RID RendererStorageRD::shader_allocate() {
	return shader_owner.allocate_rid();
}
void RendererStorageRD::shader_initialize(RID p_rid) {
	Shader shader;
	shader.data = nullptr;
	shader.type = SHADER_TYPE_MAX;

	shader_owner.initialize_rid(p_rid, shader);
}

void RendererStorageRD::shader_set_code(RID p_shader, const String &p_code) {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	shader->code = p_code;
	String mode_string = ShaderLanguage::get_shader_type(p_code);

	ShaderType new_type;
	if (mode_string == "canvas_item") {
		new_type = SHADER_TYPE_2D;
	} else if (mode_string == "particles") {
		new_type = SHADER_TYPE_PARTICLES;
	} else if (mode_string == "spatial") {
		new_type = SHADER_TYPE_3D;
	} else if (mode_string == "sky") {
		new_type = SHADER_TYPE_SKY;
	} else if (mode_string == "fog") {
		new_type = SHADER_TYPE_FOG;
	} else {
		new_type = SHADER_TYPE_MAX;
	}

	if (new_type != shader->type) {
		if (shader->data) {
			memdelete(shader->data);
			shader->data = nullptr;
		}

		for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
			Material *material = E->get();
			material->shader_type = new_type;
			if (material->data) {
				memdelete(material->data);
				material->data = nullptr;
			}
		}

		shader->type = new_type;

		if (new_type < SHADER_TYPE_MAX && shader_data_request_func[new_type]) {
			shader->data = shader_data_request_func[new_type]();
		} else {
			shader->type = SHADER_TYPE_MAX; //invalid
		}

		for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
			Material *material = E->get();
			if (shader->data) {
				material->data = material_data_request_func[new_type](shader->data);
				material->data->self = material->self;
				material->data->set_next_pass(material->next_pass);
				material->data->set_render_priority(material->priority);
			}
			material->shader_type = new_type;
		}

		if (shader->data) {
			for (const KeyValue<StringName, RID> &E : shader->default_texture_parameter) {
				shader->data->set_default_texture_param(E.key, E.value);
			}
		}
	}

	if (shader->data) {
		shader->data->set_code(p_code);
	}

	for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
		Material *material = E->get();
		material->dependency.changed_notify(DEPENDENCY_CHANGED_MATERIAL);
		_material_queue_update(material, true, true);
	}
}

String RendererStorageRD::shader_get_code(RID p_shader) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, String());
	return shader->code;
}

void RendererStorageRD::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);
	if (shader->data) {
		return shader->data->get_param_list(p_param_list);
	}
}

void RendererStorageRD::shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);

	if (p_texture.is_valid() && texture_owner.owns(p_texture)) {
		shader->default_texture_parameter[p_name] = p_texture;
	} else {
		shader->default_texture_parameter.erase(p_name);
	}
	if (shader->data) {
		shader->data->set_default_texture_param(p_name, p_texture);
	}
	for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
		Material *material = E->get();
		_material_queue_update(material, false, true);
	}
}

RID RendererStorageRD::shader_get_default_texture_param(RID p_shader, const StringName &p_name) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, RID());
	if (shader->default_texture_parameter.has(p_name)) {
		return shader->default_texture_parameter[p_name];
	}

	return RID();
}

Variant RendererStorageRD::shader_get_param_default(RID p_shader, const StringName &p_param) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, Variant());
	if (shader->data) {
		return shader->data->get_default_parameter(p_param);
	}
	return Variant();
}

void RendererStorageRD::shader_set_data_request_function(ShaderType p_shader_type, ShaderDataRequestFunction p_function) {
	ERR_FAIL_INDEX(p_shader_type, SHADER_TYPE_MAX);
	shader_data_request_func[p_shader_type] = p_function;
}

RS::ShaderNativeSourceCode RendererStorageRD::shader_get_native_source_code(RID p_shader) const {
	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND_V(!shader, RS::ShaderNativeSourceCode());
	if (shader->data) {
		return shader->data->get_native_source_code();
	}
	return RS::ShaderNativeSourceCode();
}

/* COMMON MATERIAL API */

RID RendererStorageRD::material_allocate() {
	return material_owner.allocate_rid();
}
void RendererStorageRD::material_initialize(RID p_rid) {
	material_owner.initialize_rid(p_rid);
	Material *material = material_owner.get_or_null(p_rid);
	material->self = p_rid;
}

void RendererStorageRD::_material_queue_update(Material *material, bool p_uniform, bool p_texture) {
	if (material->update_element.in_list()) {
		return;
	}

	material_update_list.add(&material->update_element);

	material->uniform_dirty = material->uniform_dirty || p_uniform;
	material->texture_dirty = material->texture_dirty || p_texture;
}

void RendererStorageRD::material_set_shader(RID p_material, RID p_shader) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (material->data) {
		memdelete(material->data);
		material->data = nullptr;
	}

	if (material->shader) {
		material->shader->owners.erase(material);
		material->shader = nullptr;
		material->shader_type = SHADER_TYPE_MAX;
	}

	if (p_shader.is_null()) {
		material->dependency.changed_notify(DEPENDENCY_CHANGED_MATERIAL);
		material->shader_id = 0;
		return;
	}

	Shader *shader = shader_owner.get_or_null(p_shader);
	ERR_FAIL_COND(!shader);
	material->shader = shader;
	material->shader_type = shader->type;
	material->shader_id = p_shader.get_local_index();
	shader->owners.insert(material);

	if (shader->type == SHADER_TYPE_MAX) {
		return;
	}

	ERR_FAIL_COND(shader->data == nullptr);

	material->data = material_data_request_func[shader->type](shader->data);
	material->data->self = p_material;
	material->data->set_next_pass(material->next_pass);
	material->data->set_render_priority(material->priority);
	//updating happens later
	material->dependency.changed_notify(DEPENDENCY_CHANGED_MATERIAL);
	_material_queue_update(material, true, true);
}

void RendererStorageRD::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (p_value.get_type() == Variant::NIL) {
		material->params.erase(p_param);
	} else {
		ERR_FAIL_COND(p_value.get_type() == Variant::OBJECT); //object not allowed
		material->params[p_param] = p_value;
	}

	if (material->shader && material->shader->data) { //shader is valid
		bool is_texture = material->shader->data->is_param_texture(p_param);
		_material_queue_update(material, !is_texture, is_texture);
	} else {
		_material_queue_update(material, true, true);
	}
}

Variant RendererStorageRD::material_get_param(RID p_material, const StringName &p_param) const {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, Variant());
	if (material->params.has(p_param)) {
		return material->params[p_param];
	} else {
		return Variant();
	}
}

void RendererStorageRD::material_set_next_pass(RID p_material, RID p_next_material) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);

	if (material->next_pass == p_next_material) {
		return;
	}

	material->next_pass = p_next_material;
	if (material->data) {
		material->data->set_next_pass(p_next_material);
	}

	material->dependency.changed_notify(DEPENDENCY_CHANGED_MATERIAL);
}

void RendererStorageRD::material_set_render_priority(RID p_material, int priority) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	material->priority = priority;
	if (material->data) {
		material->data->set_render_priority(priority);
	}
}

bool RendererStorageRD::material_is_animated(RID p_material) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, false);
	if (material->shader && material->shader->data) {
		if (material->shader->data->is_animated()) {
			return true;
		} else if (material->next_pass.is_valid()) {
			return material_is_animated(material->next_pass);
		}
	}
	return false; //by default nothing is animated
}

bool RendererStorageRD::material_casts_shadows(RID p_material) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND_V(!material, true);
	if (material->shader && material->shader->data) {
		if (material->shader->data->casts_shadows()) {
			return true;
		} else if (material->next_pass.is_valid()) {
			return material_casts_shadows(material->next_pass);
		}
	}
	return true; //by default everything casts shadows
}

void RendererStorageRD::material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	if (material->shader && material->shader->data) {
		material->shader->data->get_instance_param_list(r_parameters);

		if (material->next_pass.is_valid()) {
			material_get_instance_shader_parameters(material->next_pass, r_parameters);
		}
	}
}

void RendererStorageRD::material_update_dependency(RID p_material, DependencyTracker *p_instance) {
	Material *material = material_owner.get_or_null(p_material);
	ERR_FAIL_COND(!material);
	p_instance->update_dependency(&material->dependency);
	if (material->next_pass.is_valid()) {
		material_update_dependency(material->next_pass, p_instance);
	}
}

void RendererStorageRD::material_set_data_request_function(ShaderType p_shader_type, MaterialDataRequestFunction p_function) {
	ERR_FAIL_INDEX(p_shader_type, SHADER_TYPE_MAX);
	material_data_request_func[p_shader_type] = p_function;
}

_FORCE_INLINE_ static void _fill_std140_variant_ubo_value(ShaderLanguage::DataType type, int p_array_size, const Variant &value, uint8_t *data, bool p_linear_color) {
	switch (type) {
		case ShaderLanguage::TYPE_BOOL: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = (r[i] != 0) ? 1 : 0;
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				bool v = value;
				gui[0] = v ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_BVEC2: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();
				int count = 2 * p_array_size;

				for (int i = 0, j = 0; i < count; i += 2, j += 4) {
					if (i < s) {
						gui[j] = r[i] ? 1 : 0;
						gui[j + 1] = r[i + 1] ? 1 : 0;
					} else {
						gui[j] = 0;
						gui[j + 1] = 0;
					}
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = v & 1 ? 1 : 0;
				gui[1] = v & 2 ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_BVEC3: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();
				int count = 3 * p_array_size;

				for (int i = 0, j = 0; i < count; i += 3, j += 4) {
					if (i < s) {
						gui[j] = r[i] ? 1 : 0;
						gui[j + 1] = r[i + 1] ? 1 : 0;
						gui[j + 2] = r[i + 2] ? 1 : 0;
					} else {
						gui[j] = 0;
						gui[j + 1] = 0;
						gui[j + 2] = 0;
					}
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = (v & 1) ? 1 : 0;
				gui[1] = (v & 2) ? 1 : 0;
				gui[2] = (v & 4) ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_BVEC4: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				const PackedInt32Array &ba = value;
				int s = ba.size();
				const int *r = ba.ptr();
				int count = 4 * p_array_size;

				for (int i = 0; i < count; i += 4) {
					if (i < s) {
						gui[i] = r[i] ? 1 : 0;
						gui[i + 1] = r[i + 1] ? 1 : 0;
						gui[i + 2] = r[i + 2] ? 1 : 0;
						gui[i + 3] = r[i + 3] ? 1 : 0;
					} else {
						gui[i] = 0;
						gui[i + 1] = 0;
						gui[i + 2] = 0;
						gui[i + 3] = 0;
					}
				}
			} else {
				int v = value;
				gui[0] = (v & 1) ? 1 : 0;
				gui[1] = (v & 2) ? 1 : 0;
				gui[2] = (v & 4) ? 1 : 0;
				gui[3] = (v & 8) ? 1 : 0;
			}
		} break;
		case ShaderLanguage::TYPE_INT: {
			int32_t *gui = (int32_t *)data;

			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
				const int *r = iv.ptr();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = r[i];
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = v;
			}
		} break;
		case ShaderLanguage::TYPE_IVEC2: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 2 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 2, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
				}
				gui[j + 2] = 0; // ignored
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_IVEC3: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 3 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 3, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
					gui[j + 2] = r[i + 2];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
					gui[j + 2] = 0;
				}
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_IVEC4: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 4 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0; i < count; i += 4) {
				if (i < s) {
					gui[i] = r[i];
					gui[i + 1] = r[i + 1];
					gui[i + 2] = r[i + 2];
					gui[i + 3] = r[i + 3];
				} else {
					gui[i] = 0;
					gui[i + 1] = 0;
					gui[i + 2] = 0;
					gui[i + 3] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_UINT: {
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size > 0) {
				Vector<int> iv = value;
				int s = iv.size();
				const int *r = iv.ptr();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = r[i];
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				int v = value;
				gui[0] = v;
			}
		} break;
		case ShaderLanguage::TYPE_UVEC2: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 2 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 2, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
				}
				gui[j + 2] = 0; // ignored
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_UVEC3: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 3 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0, j = 0; i < count; i += 3, j += 4) {
				if (i < s) {
					gui[j] = r[i];
					gui[j + 1] = r[i + 1];
					gui[j + 2] = r[i + 2];
				} else {
					gui[j] = 0;
					gui[j + 1] = 0;
					gui[j + 2] = 0;
				}
				gui[j + 3] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_UVEC4: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			if (p_array_size <= 0) {
				p_array_size = 1;
			}
			int count = 4 * p_array_size;

			const int *r = iv.ptr();
			for (int i = 0; i < count; i++) {
				if (i < s) {
					gui[i] = r[i];
					gui[i + 1] = r[i + 1];
					gui[i + 2] = r[i + 2];
					gui[i + 3] = r[i + 3];
				} else {
					gui[i] = 0;
					gui[i + 1] = 0;
					gui[i + 2] = 0;
					gui[i + 3] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_FLOAT: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = a[i];
					} else {
						gui[j] = 0;
					}
					gui[j + 1] = 0; // ignored
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				float v = value;
				gui[0] = v;
			}
		} break;
		case ShaderLanguage::TYPE_VEC2: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedVector2Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = a[i].x;
						gui[j + 1] = a[i].y;
					} else {
						gui[j] = 0;
						gui[j + 1] = 0;
					}
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
				}
			} else {
				Vector2 v = value;
				gui[0] = v.x;
				gui[1] = v.y;
			}
		} break;
		case ShaderLanguage::TYPE_VEC3: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedVector3Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
					if (i < s) {
						gui[j] = a[i].x;
						gui[j + 1] = a[i].y;
						gui[j + 2] = a[i].z;
					} else {
						gui[j] = 0;
						gui[j + 1] = 0;
						gui[j + 2] = 0;
					}
					gui[j + 3] = 0; // ignored
				}
			} else {
				Vector3 v = value;
				gui[0] = v.x;
				gui[1] = v.y;
				gui[2] = v.z;
			}
		} break;
		case ShaderLanguage::TYPE_VEC4: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				if (value.get_type() == Variant::PACKED_COLOR_ARRAY) {
					const PackedColorArray &a = value;
					int s = a.size();

					for (int i = 0, j = 0; i < p_array_size; i++, j += 4) {
						if (i < s) {
							Color color = a[i];
							if (p_linear_color) {
								color = color.to_linear();
							}
							gui[j] = color.r;
							gui[j + 1] = color.g;
							gui[j + 2] = color.b;
							gui[j + 3] = color.a;
						} else {
							gui[j] = 0;
							gui[j + 1] = 0;
							gui[j + 2] = 0;
							gui[j + 3] = 0;
						}
					}
				} else {
					const PackedFloat32Array &a = value;
					int s = a.size();
					int count = 4 * p_array_size;

					for (int i = 0; i < count; i += 4) {
						if (i + 3 < s) {
							gui[i] = a[i];
							gui[i + 1] = a[i + 1];
							gui[i + 2] = a[i + 2];
							gui[i + 3] = a[i + 3];
						} else {
							gui[i] = 0;
							gui[i + 1] = 0;
							gui[i + 2] = 0;
							gui[i + 3] = 0;
						}
					}
				}
			} else {
				if (value.get_type() == Variant::COLOR) {
					Color v = value;

					if (p_linear_color) {
						v = v.to_linear();
					}

					gui[0] = v.r;
					gui[1] = v.g;
					gui[2] = v.b;
					gui[3] = v.a;
				} else if (value.get_type() == Variant::RECT2) {
					Rect2 v = value;

					gui[0] = v.position.x;
					gui[1] = v.position.y;
					gui[2] = v.size.x;
					gui[3] = v.size.y;
				} else if (value.get_type() == Variant::QUATERNION) {
					Quaternion v = value;

					gui[0] = v.x;
					gui[1] = v.y;
					gui[2] = v.z;
					gui[3] = v.w;
				} else {
					Plane v = value;

					gui[0] = v.normal.x;
					gui[1] = v.normal.y;
					gui[2] = v.normal.z;
					gui[3] = v.d;
				}
			}
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size * 4; i += 4, j += 8) {
					if (i + 3 < s) {
						gui[j] = a[i];
						gui[j + 1] = a[i + 1];

						gui[j + 4] = a[i + 2];
						gui[j + 5] = a[i + 3];
					} else {
						gui[j] = 1;
						gui[j + 1] = 0;

						gui[j + 4] = 0;
						gui[j + 5] = 1;
					}
					gui[j + 2] = 0; // ignored
					gui[j + 3] = 0; // ignored
					gui[j + 6] = 0; // ignored
					gui[j + 7] = 0; // ignored
				}
			} else {
				Transform2D v = value;

				//in std140 members of mat2 are treated as vec4s
				gui[0] = v.elements[0][0];
				gui[1] = v.elements[0][1];
				gui[2] = 0; // ignored
				gui[3] = 0; // ignored

				gui[4] = v.elements[1][0];
				gui[5] = v.elements[1][1];
				gui[6] = 0; // ignored
				gui[7] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0, j = 0; i < p_array_size * 9; i += 9, j += 12) {
					if (i + 8 < s) {
						gui[j] = a[i];
						gui[j + 1] = a[i + 1];
						gui[j + 2] = a[i + 2];

						gui[j + 4] = a[i + 3];
						gui[j + 5] = a[i + 4];
						gui[j + 6] = a[i + 5];

						gui[j + 8] = a[i + 6];
						gui[j + 9] = a[i + 7];
						gui[j + 10] = a[i + 8];
					} else {
						gui[j] = 1;
						gui[j + 1] = 0;
						gui[j + 2] = 0;

						gui[j + 4] = 0;
						gui[j + 5] = 1;
						gui[j + 6] = 0;

						gui[j + 8] = 0;
						gui[j + 9] = 0;
						gui[j + 10] = 1;
					}
					gui[j + 3] = 0; // ignored
					gui[j + 7] = 0; // ignored
					gui[j + 11] = 0; // ignored
				}
			} else {
				Basis v = value;
				gui[0] = v.elements[0][0];
				gui[1] = v.elements[1][0];
				gui[2] = v.elements[2][0];
				gui[3] = 0; // ignored

				gui[4] = v.elements[0][1];
				gui[5] = v.elements[1][1];
				gui[6] = v.elements[2][1];
				gui[7] = 0; // ignored

				gui[8] = v.elements[0][2];
				gui[9] = v.elements[1][2];
				gui[10] = v.elements[2][2];
				gui[11] = 0; // ignored
			}
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			float *gui = (float *)data;

			if (p_array_size > 0) {
				const PackedFloat32Array &a = value;
				int s = a.size();

				for (int i = 0; i < p_array_size * 16; i += 16) {
					if (i + 15 < s) {
						gui[i] = a[i];
						gui[i + 1] = a[i + 1];
						gui[i + 2] = a[i + 2];
						gui[i + 3] = a[i + 3];

						gui[i + 4] = a[i + 4];
						gui[i + 5] = a[i + 5];
						gui[i + 6] = a[i + 6];
						gui[i + 7] = a[i + 7];

						gui[i + 8] = a[i + 8];
						gui[i + 9] = a[i + 9];
						gui[i + 10] = a[i + 10];
						gui[i + 11] = a[i + 11];

						gui[i + 12] = a[i + 12];
						gui[i + 13] = a[i + 13];
						gui[i + 14] = a[i + 14];
						gui[i + 15] = a[i + 15];
					} else {
						gui[i] = 1;
						gui[i + 1] = 0;
						gui[i + 2] = 0;
						gui[i + 3] = 0;

						gui[i + 4] = 0;
						gui[i + 5] = 1;
						gui[i + 6] = 0;
						gui[i + 7] = 0;

						gui[i + 8] = 0;
						gui[i + 9] = 0;
						gui[i + 10] = 1;
						gui[i + 11] = 0;

						gui[i + 12] = 0;
						gui[i + 13] = 0;
						gui[i + 14] = 0;
						gui[i + 15] = 1;
					}
				}
			} else {
				Transform3D v = value;
				gui[0] = v.basis.elements[0][0];
				gui[1] = v.basis.elements[1][0];
				gui[2] = v.basis.elements[2][0];
				gui[3] = 0;

				gui[4] = v.basis.elements[0][1];
				gui[5] = v.basis.elements[1][1];
				gui[6] = v.basis.elements[2][1];
				gui[7] = 0;

				gui[8] = v.basis.elements[0][2];
				gui[9] = v.basis.elements[1][2];
				gui[10] = v.basis.elements[2][2];
				gui[11] = 0;

				gui[12] = v.origin.x;
				gui[13] = v.origin.y;
				gui[14] = v.origin.z;
				gui[15] = 1;
			}
		} break;
		default: {
		}
	}
}

_FORCE_INLINE_ static void _fill_std140_ubo_value(ShaderLanguage::DataType type, const Vector<ShaderLanguage::ConstantNode::Value> &value, uint8_t *data) {
	switch (type) {
		case ShaderLanguage::TYPE_BOOL: {
			uint32_t *gui = (uint32_t *)data;
			*gui = value[0].boolean ? 1 : 0;
		} break;
		case ShaderLanguage::TYPE_BVEC2: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].boolean ? 1 : 0;
			gui[1] = value[1].boolean ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_BVEC3: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].boolean ? 1 : 0;
			gui[1] = value[1].boolean ? 1 : 0;
			gui[2] = value[2].boolean ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_BVEC4: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].boolean ? 1 : 0;
			gui[1] = value[1].boolean ? 1 : 0;
			gui[2] = value[2].boolean ? 1 : 0;
			gui[3] = value[3].boolean ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_INT: {
			int32_t *gui = (int32_t *)data;
			gui[0] = value[0].sint;

		} break;
		case ShaderLanguage::TYPE_IVEC2: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 2; i++) {
				gui[i] = value[i].sint;
			}

		} break;
		case ShaderLanguage::TYPE_IVEC3: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 3; i++) {
				gui[i] = value[i].sint;
			}

		} break;
		case ShaderLanguage::TYPE_IVEC4: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 4; i++) {
				gui[i] = value[i].sint;
			}

		} break;
		case ShaderLanguage::TYPE_UINT: {
			uint32_t *gui = (uint32_t *)data;
			gui[0] = value[0].uint;

		} break;
		case ShaderLanguage::TYPE_UVEC2: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 2; i++) {
				gui[i] = value[i].uint;
			}
		} break;
		case ShaderLanguage::TYPE_UVEC3: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 3; i++) {
				gui[i] = value[i].uint;
			}

		} break;
		case ShaderLanguage::TYPE_UVEC4: {
			int32_t *gui = (int32_t *)data;

			for (int i = 0; i < 4; i++) {
				gui[i] = value[i].uint;
			}
		} break;
		case ShaderLanguage::TYPE_FLOAT: {
			float *gui = (float *)data;
			gui[0] = value[0].real;

		} break;
		case ShaderLanguage::TYPE_VEC2: {
			float *gui = (float *)data;

			for (int i = 0; i < 2; i++) {
				gui[i] = value[i].real;
			}

		} break;
		case ShaderLanguage::TYPE_VEC3: {
			float *gui = (float *)data;

			for (int i = 0; i < 3; i++) {
				gui[i] = value[i].real;
			}

		} break;
		case ShaderLanguage::TYPE_VEC4: {
			float *gui = (float *)data;

			for (int i = 0; i < 4; i++) {
				gui[i] = value[i].real;
			}
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			float *gui = (float *)data;

			//in std140 members of mat2 are treated as vec4s
			gui[0] = value[0].real;
			gui[1] = value[1].real;
			gui[2] = 0;
			gui[3] = 0;
			gui[4] = value[2].real;
			gui[5] = value[3].real;
			gui[6] = 0;
			gui[7] = 0;
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			float *gui = (float *)data;

			gui[0] = value[0].real;
			gui[1] = value[1].real;
			gui[2] = value[2].real;
			gui[3] = 0;
			gui[4] = value[3].real;
			gui[5] = value[4].real;
			gui[6] = value[5].real;
			gui[7] = 0;
			gui[8] = value[6].real;
			gui[9] = value[7].real;
			gui[10] = value[8].real;
			gui[11] = 0;
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			float *gui = (float *)data;

			for (int i = 0; i < 16; i++) {
				gui[i] = value[i].real;
			}
		} break;
		default: {
		}
	}
}

_FORCE_INLINE_ static void _fill_std140_ubo_empty(ShaderLanguage::DataType type, int p_array_size, uint8_t *data) {
	if (p_array_size <= 0) {
		p_array_size = 1;
	}

	switch (type) {
		case ShaderLanguage::TYPE_BOOL:
		case ShaderLanguage::TYPE_INT:
		case ShaderLanguage::TYPE_UINT:
		case ShaderLanguage::TYPE_FLOAT: {
			memset(data, 0, 4 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_BVEC2:
		case ShaderLanguage::TYPE_IVEC2:
		case ShaderLanguage::TYPE_UVEC2:
		case ShaderLanguage::TYPE_VEC2: {
			memset(data, 0, 8 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_BVEC3:
		case ShaderLanguage::TYPE_IVEC3:
		case ShaderLanguage::TYPE_UVEC3:
		case ShaderLanguage::TYPE_VEC3:
		case ShaderLanguage::TYPE_BVEC4:
		case ShaderLanguage::TYPE_IVEC4:
		case ShaderLanguage::TYPE_UVEC4:
		case ShaderLanguage::TYPE_VEC4: {
			memset(data, 0, 16 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			memset(data, 0, 32 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			memset(data, 0, 48 * p_array_size);
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			memset(data, 0, 64 * p_array_size);
		} break;

		default: {
		}
	}
}

void RendererStorageRD::MaterialData::update_uniform_buffer(const Map<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Map<StringName, Variant> &p_parameters, uint8_t *p_buffer, uint32_t p_buffer_size, bool p_use_linear_color) {
	bool uses_global_buffer = false;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : p_uniforms) {
		if (E.value.order < 0) {
			continue; // texture, does not go here
		}

		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue; //instance uniforms don't appear in the bufferr
		}

		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL) {
			//this is a global variable, get the index to it
			RendererStorageRD *rs = base_singleton;

			GlobalVariables::Variable *gv = rs->global_variables.variables.getptr(E.key);
			uint32_t index = 0;
			if (gv) {
				index = gv->buffer_index;
			} else {
				WARN_PRINT("Shader uses global uniform '" + E.key + "', but it was removed at some point. Material will not display correctly.");
			}

			uint32_t offset = p_uniform_offsets[E.value.order];
			uint32_t *intptr = (uint32_t *)&p_buffer[offset];
			*intptr = index;
			uses_global_buffer = true;
			continue;
		}

		//regular uniform
		uint32_t offset = p_uniform_offsets[E.value.order];
#ifdef DEBUG_ENABLED
		uint32_t size = ShaderLanguage::get_type_size(E.value.type);
		ERR_CONTINUE(offset + size > p_buffer_size);
#endif
		uint8_t *data = &p_buffer[offset];
		const Map<StringName, Variant>::Element *V = p_parameters.find(E.key);

		if (V) {
			//user provided
			_fill_std140_variant_ubo_value(E.value.type, E.value.array_size, V->get(), data, p_use_linear_color);

		} else if (E.value.default_value.size()) {
			//default value
			_fill_std140_ubo_value(E.value.type, E.value.default_value, data);
			//value=E.value.default_value;
		} else {
			//zero because it was not provided
			if (E.value.type == ShaderLanguage::TYPE_VEC4 && E.value.hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
				//colors must be set as black, with alpha as 1.0
				_fill_std140_variant_ubo_value(E.value.type, E.value.array_size, Color(0, 0, 0, 1), data, p_use_linear_color);
			} else {
				//else just zero it out
				_fill_std140_ubo_empty(E.value.type, E.value.array_size, data);
			}
		}
	}

	if (uses_global_buffer != (global_buffer_E != nullptr)) {
		RendererStorageRD *rs = base_singleton;
		if (uses_global_buffer) {
			global_buffer_E = rs->global_variables.materials_using_buffer.push_back(self);
		} else {
			rs->global_variables.materials_using_buffer.erase(global_buffer_E);
			global_buffer_E = nullptr;
		}
	}
}

RendererStorageRD::MaterialData::~MaterialData() {
	if (global_buffer_E) {
		//unregister global buffers
		RendererStorageRD *rs = base_singleton;
		rs->global_variables.materials_using_buffer.erase(global_buffer_E);
	}

	if (global_texture_E) {
		//unregister global textures
		RendererStorageRD *rs = base_singleton;

		for (const KeyValue<StringName, uint64_t> &E : used_global_textures) {
			GlobalVariables::Variable *v = rs->global_variables.variables.getptr(E.key);
			if (v) {
				v->texture_materials.erase(self);
			}
		}
		//unregister material from those using global textures
		rs->global_variables.materials_using_texture.erase(global_texture_E);
	}

	if (uniform_buffer.is_valid()) {
		RD::get_singleton()->free(uniform_buffer);
	}
}

void RendererStorageRD::MaterialData::update_textures(const Map<StringName, Variant> &p_parameters, const Map<StringName, RID> &p_default_textures, const Vector<ShaderCompilerRD::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures, bool p_use_linear_color) {
	RendererStorageRD *singleton = (RendererStorageRD *)RendererStorage::base_singleton;
#ifdef TOOLS_ENABLED
	Texture *roughness_detect_texture = nullptr;
	RS::TextureDetectRoughnessChannel roughness_channel = RS::TEXTURE_DETECT_ROUGHNESS_R;
	Texture *normal_detect_texture = nullptr;
#endif

	bool uses_global_textures = false;
	global_textures_pass++;

	for (int i = 0, k = 0; i < p_texture_uniforms.size(); i++) {
		const StringName &uniform_name = p_texture_uniforms[i].name;
		int uniform_array_size = p_texture_uniforms[i].array_size;

		Vector<RID> textures;

		if (p_texture_uniforms[i].global) {
			RendererStorageRD *rs = base_singleton;

			uses_global_textures = true;

			GlobalVariables::Variable *v = rs->global_variables.variables.getptr(uniform_name);
			if (v) {
				if (v->buffer_index >= 0) {
					WARN_PRINT("Shader uses global uniform texture '" + String(uniform_name) + "', but it changed type and is no longer a texture!.");

				} else {
					Map<StringName, uint64_t>::Element *E = used_global_textures.find(uniform_name);
					if (!E) {
						E = used_global_textures.insert(uniform_name, global_textures_pass);
						v->texture_materials.insert(self);
					} else {
						E->get() = global_textures_pass;
					}

					textures.push_back(v->override.get_type() != Variant::NIL ? v->override : v->value);
				}

			} else {
				WARN_PRINT("Shader uses global uniform texture '" + String(uniform_name) + "', but it was removed at some point. Material will not display correctly.");
			}
		} else {
			const Map<StringName, Variant>::Element *V = p_parameters.find(uniform_name);
			if (V) {
				if (V->get().is_array()) {
					Array array = (Array)V->get();
					if (uniform_array_size > 0) {
						for (int j = 0; j < array.size(); j++) {
							textures.push_back(array[j]);
						}
					} else {
						if (array.size() > 0) {
							textures.push_back(array[0]);
						}
					}
				} else {
					textures.push_back(V->get());
				}
			}

			if (uniform_array_size > 0) {
				if (textures.size() < uniform_array_size) {
					const Map<StringName, RID>::Element *W = p_default_textures.find(uniform_name);
					for (int j = textures.size(); j < uniform_array_size; j++) {
						if (W) {
							textures.push_back(W->get());
						} else {
							textures.push_back(RID());
						}
					}
				}
			} else if (textures.is_empty()) {
				const Map<StringName, RID>::Element *W = p_default_textures.find(uniform_name);
				if (W) {
					textures.push_back(W->get());
				}
			}
		}

		RID rd_texture;

		if (textures.is_empty()) {
			//check default usage
			switch (p_texture_uniforms[i].type) {
				case ShaderLanguage::TYPE_ISAMPLER2D:
				case ShaderLanguage::TYPE_USAMPLER2D:
				case ShaderLanguage::TYPE_SAMPLER2D: {
					switch (p_texture_uniforms[i].hint) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK:
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO: {
							rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_BLACK);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_NONE: {
							rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_NORMAL);
						} break;
						case ShaderLanguage::ShaderNode::Uniform::HINT_ANISO: {
							rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_ANISO);
						} break;
						default: {
							rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE);
						} break;
					}
				} break;

				case ShaderLanguage::TYPE_SAMPLERCUBE: {
					switch (p_texture_uniforms[i].hint) {
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK:
						case ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO: {
							rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_CUBEMAP_BLACK);
						} break;
						default: {
							rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_CUBEMAP_WHITE);
						} break;
					}
				} break;
				case ShaderLanguage::TYPE_SAMPLERCUBEARRAY: {
					rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK);
				} break;

				case ShaderLanguage::TYPE_ISAMPLER3D:
				case ShaderLanguage::TYPE_USAMPLER3D:
				case ShaderLanguage::TYPE_SAMPLER3D: {
					rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_3D_WHITE);
				} break;

				case ShaderLanguage::TYPE_ISAMPLER2DARRAY:
				case ShaderLanguage::TYPE_USAMPLER2DARRAY:
				case ShaderLanguage::TYPE_SAMPLER2DARRAY: {
					rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE);
				} break;

				default: {
				}
			}
#ifdef TOOLS_ENABLED
			if (roughness_detect_texture && normal_detect_texture && normal_detect_texture->path != String()) {
				roughness_detect_texture->detect_roughness_callback(roughness_detect_texture->detect_roughness_callback_ud, normal_detect_texture->path, roughness_channel);
			}
#endif
			if (uniform_array_size > 0) {
				for (int j = 0; j < uniform_array_size; j++) {
					p_textures[k++] = rd_texture;
				}
			} else {
				p_textures[k++] = rd_texture;
			}
		} else {
			bool srgb = p_use_linear_color && (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_ALBEDO || p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO);

			for (int j = 0; j < textures.size(); j++) {
				Texture *tex = singleton->texture_owner.get_or_null(textures[j]);

				if (tex) {
					rd_texture = (srgb && tex->rd_texture_srgb.is_valid()) ? tex->rd_texture_srgb : tex->rd_texture;
#ifdef TOOLS_ENABLED
					if (tex->detect_3d_callback && p_use_linear_color) {
						tex->detect_3d_callback(tex->detect_3d_callback_ud);
					}
					if (tex->detect_normal_callback && (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_NORMAL || p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_NORMAL)) {
						if (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_NORMAL) {
							normal_detect_texture = tex;
						}
						tex->detect_normal_callback(tex->detect_normal_callback_ud);
					}
					if (tex->detect_roughness_callback && (p_texture_uniforms[i].hint >= ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_R || p_texture_uniforms[i].hint <= ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_GRAY)) {
						//find the normal texture
						roughness_detect_texture = tex;
						roughness_channel = RS::TextureDetectRoughnessChannel(p_texture_uniforms[i].hint - ShaderLanguage::ShaderNode::Uniform::HINT_ROUGHNESS_R);
					}
#endif
				}
				if (rd_texture.is_null()) {
					rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE);
				}
#ifdef TOOLS_ENABLED
				if (roughness_detect_texture && normal_detect_texture && normal_detect_texture->path != String()) {
					roughness_detect_texture->detect_roughness_callback(roughness_detect_texture->detect_roughness_callback_ud, normal_detect_texture->path, roughness_channel);
				}
#endif
				p_textures[k++] = rd_texture;
			}
		}
	}
	{
		//for textures no longer used, unregister them
		List<Map<StringName, uint64_t>::Element *> to_delete;
		RendererStorageRD *rs = base_singleton;

		for (Map<StringName, uint64_t>::Element *E = used_global_textures.front(); E; E = E->next()) {
			if (E->get() != global_textures_pass) {
				to_delete.push_back(E);

				GlobalVariables::Variable *v = rs->global_variables.variables.getptr(E->key());
				if (v) {
					v->texture_materials.erase(self);
				}
			}
		}

		while (to_delete.front()) {
			used_global_textures.erase(to_delete.front()->get());
			to_delete.pop_front();
		}
		//handle registering/unregistering global textures
		if (uses_global_textures != (global_texture_E != nullptr)) {
			if (uses_global_textures) {
				global_texture_E = rs->global_variables.materials_using_texture.push_back(self);
			} else {
				rs->global_variables.materials_using_texture.erase(global_texture_E);
				global_texture_E = nullptr;
			}
		}
	}
}

void RendererStorageRD::MaterialData::free_parameters_uniform_set(RID p_uniform_set) {
	if (p_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(p_uniform_set)) {
		RD::get_singleton()->uniform_set_set_invalidation_callback(p_uniform_set, nullptr, nullptr);
		RD::get_singleton()->free(p_uniform_set);
	}
}

bool RendererStorageRD::MaterialData::update_parameters_uniform_set(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty, const Map<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Vector<ShaderCompilerRD::GeneratedCode::Texture> &p_texture_uniforms, const Map<StringName, RID> &p_default_texture_params, uint32_t p_ubo_size, RID &uniform_set, RID p_shader, uint32_t p_shader_uniform_set, uint32_t p_barrier) {
	if ((uint32_t)ubo_data.size() != p_ubo_size) {
		p_uniform_dirty = true;
		if (uniform_buffer.is_valid()) {
			RD::get_singleton()->free(uniform_buffer);
			uniform_buffer = RID();
		}

		ubo_data.resize(p_ubo_size);
		if (ubo_data.size()) {
			uniform_buffer = RD::get_singleton()->uniform_buffer_create(ubo_data.size());
			memset(ubo_data.ptrw(), 0, ubo_data.size()); //clear
		}

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->uniform_set_set_invalidation_callback(uniform_set, nullptr, nullptr);
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	//check whether buffer changed
	if (p_uniform_dirty && ubo_data.size()) {
		update_uniform_buffer(p_uniforms, p_uniform_offsets, p_parameters, ubo_data.ptrw(), ubo_data.size(), false);
		RD::get_singleton()->buffer_update(uniform_buffer, 0, ubo_data.size(), ubo_data.ptrw(), p_barrier);
	}

	uint32_t tex_uniform_count = 0U;
	for (int i = 0; i < p_texture_uniforms.size(); i++) {
		tex_uniform_count += uint32_t(p_texture_uniforms[i].array_size > 0 ? p_texture_uniforms[i].array_size : 1);
	}

	if ((uint32_t)texture_cache.size() != tex_uniform_count || p_textures_dirty) {
		texture_cache.resize(tex_uniform_count);
		p_textures_dirty = true;

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->uniform_set_set_invalidation_callback(uniform_set, nullptr, nullptr);
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	if (p_textures_dirty && tex_uniform_count) {
		update_textures(p_parameters, p_default_texture_params, p_texture_uniforms, texture_cache.ptrw(), true);
	}

	if (p_ubo_size == 0 && p_texture_uniforms.size() == 0) {
		// This material does not require an uniform set, so don't create it.
		return false;
	}

	if (!p_textures_dirty && uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//no reason to update uniform set, only UBO (or nothing) was needed to update
		return false;
	}

	Vector<RD::Uniform> uniforms;

	{
		if (p_ubo_size) {
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 0;
			u.ids.push_back(uniform_buffer);
			uniforms.push_back(u);
		}

		const RID *textures = texture_cache.ptrw();
		for (int i = 0, k = 0; i < p_texture_uniforms.size(); i++) {
			const int array_size = p_texture_uniforms[i].array_size;

			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1 + k;
			if (array_size > 0) {
				for (int j = 0; j < array_size; j++) {
					u.ids.push_back(textures[k++]);
				}
			} else {
				u.ids.push_back(textures[k++]);
			}
			uniforms.push_back(u);
		}
	}

	uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_shader, p_shader_uniform_set);

	RD::get_singleton()->uniform_set_set_invalidation_callback(uniform_set, _material_uniform_set_erased, &self);

	return true;
}

void RendererStorageRD::_material_uniform_set_erased(const RID &p_set, void *p_material) {
	RID rid = *(RID *)p_material;
	Material *material = base_singleton->material_owner.get_or_null(rid);
	if (material) {
		material->dependency.changed_notify(DEPENDENCY_CHANGED_MATERIAL);
	}
}

void RendererStorageRD::material_force_update_textures(RID p_material, ShaderType p_shader_type) {
	Material *material = material_owner.get_or_null(p_material);
	if (material->shader_type != p_shader_type) {
		return;
	}
	if (material->data) {
		material->data->update_parameters(material->params, false, true);
	}
}

void RendererStorageRD::_update_queued_materials() {
	while (material_update_list.first()) {
		Material *material = material_update_list.first()->self();
		bool uniforms_changed = false;

		if (material->data) {
			uniforms_changed = material->data->update_parameters(material->params, material->uniform_dirty, material->texture_dirty);
		}
		material->texture_dirty = false;
		material->uniform_dirty = false;

		material_update_list.remove(&material->update_element);

		if (uniforms_changed) {
			//some implementations such as 3D renderer cache the matreial uniform set, so update is required
			material->dependency.changed_notify(DEPENDENCY_CHANGED_MATERIAL);
		}
	}
}

/* MESH API */

RID RendererStorageRD::mesh_allocate() {
	return mesh_owner.allocate_rid();
}
void RendererStorageRD::mesh_initialize(RID p_rid) {
	mesh_owner.initialize_rid(p_rid, Mesh());
}

void RendererStorageRD::mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) {
	ERR_FAIL_COND(p_blend_shape_count < 0);

	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);

	ERR_FAIL_COND(mesh->surface_count > 0); //surfaces already exist

	mesh->blend_shape_count = p_blend_shape_count;
}

/// Returns stride
void RendererStorageRD::mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);

	ERR_FAIL_COND(mesh->surface_count == RS::MAX_MESH_SURFACES);

#ifdef DEBUG_ENABLED
	//do a validation, to catch errors first
	{
		uint32_t stride = 0;
		uint32_t attrib_stride = 0;
		uint32_t skin_stride = 0;

		for (int i = 0; i < RS::ARRAY_WEIGHTS; i++) {
			if ((p_surface.format & (1 << i))) {
				switch (i) {
					case RS::ARRAY_VERTEX: {
						if (p_surface.format & RS::ARRAY_FLAG_USE_2D_VERTICES) {
							stride += sizeof(float) * 2;
						} else {
							stride += sizeof(float) * 3;
						}

					} break;
					case RS::ARRAY_NORMAL: {
						stride += sizeof(int32_t);

					} break;
					case RS::ARRAY_TANGENT: {
						stride += sizeof(int32_t);

					} break;
					case RS::ARRAY_COLOR: {
						attrib_stride += sizeof(uint32_t);
					} break;
					case RS::ARRAY_TEX_UV: {
						attrib_stride += sizeof(float) * 2;

					} break;
					case RS::ARRAY_TEX_UV2: {
						attrib_stride += sizeof(float) * 2;

					} break;
					case RS::ARRAY_CUSTOM0:
					case RS::ARRAY_CUSTOM1:
					case RS::ARRAY_CUSTOM2:
					case RS::ARRAY_CUSTOM3: {
						int idx = i - RS::ARRAY_CUSTOM0;
						uint32_t fmt_shift[RS::ARRAY_CUSTOM_COUNT] = { RS::ARRAY_FORMAT_CUSTOM0_SHIFT, RS::ARRAY_FORMAT_CUSTOM1_SHIFT, RS::ARRAY_FORMAT_CUSTOM2_SHIFT, RS::ARRAY_FORMAT_CUSTOM3_SHIFT };
						uint32_t fmt = (p_surface.format >> fmt_shift[idx]) & RS::ARRAY_FORMAT_CUSTOM_MASK;
						uint32_t fmtsize[RS::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };
						attrib_stride += fmtsize[fmt];

					} break;
					case RS::ARRAY_WEIGHTS:
					case RS::ARRAY_BONES: {
						//uses a separate array
						bool use_8 = p_surface.format & RS::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
						skin_stride += sizeof(int16_t) * (use_8 ? 16 : 8);
					} break;
				}
			}
		}

		int expected_size = stride * p_surface.vertex_count;
		ERR_FAIL_COND_MSG(expected_size != p_surface.vertex_data.size(), "Size of vertex data provided (" + itos(p_surface.vertex_data.size()) + ") does not match expected (" + itos(expected_size) + ")");

		int bs_expected_size = expected_size * mesh->blend_shape_count;

		ERR_FAIL_COND_MSG(bs_expected_size != p_surface.blend_shape_data.size(), "Size of blend shape data provided (" + itos(p_surface.blend_shape_data.size()) + ") does not match expected (" + itos(bs_expected_size) + ")");

		int expected_attrib_size = attrib_stride * p_surface.vertex_count;
		ERR_FAIL_COND_MSG(expected_attrib_size != p_surface.attribute_data.size(), "Size of attribute data provided (" + itos(p_surface.attribute_data.size()) + ") does not match expected (" + itos(expected_attrib_size) + ")");

		if ((p_surface.format & RS::ARRAY_FORMAT_WEIGHTS) && (p_surface.format & RS::ARRAY_FORMAT_BONES)) {
			expected_size = skin_stride * p_surface.vertex_count;
			ERR_FAIL_COND_MSG(expected_size != p_surface.skin_data.size(), "Size of skin data provided (" + itos(p_surface.skin_data.size()) + ") does not match expected (" + itos(expected_size) + ")");
		}
	}

#endif

	Mesh::Surface *s = memnew(Mesh::Surface);

	s->format = p_surface.format;
	s->primitive = p_surface.primitive;

	bool use_as_storage = (p_surface.skin_data.size() || mesh->blend_shape_count > 0);

	s->vertex_buffer = RD::get_singleton()->vertex_buffer_create(p_surface.vertex_data.size(), p_surface.vertex_data, use_as_storage);
	s->vertex_buffer_size = p_surface.vertex_data.size();

	if (p_surface.attribute_data.size()) {
		s->attribute_buffer = RD::get_singleton()->vertex_buffer_create(p_surface.attribute_data.size(), p_surface.attribute_data);
	}
	if (p_surface.skin_data.size()) {
		s->skin_buffer = RD::get_singleton()->vertex_buffer_create(p_surface.skin_data.size(), p_surface.skin_data, use_as_storage);
		s->skin_buffer_size = p_surface.skin_data.size();
	}

	s->vertex_count = p_surface.vertex_count;

	if (p_surface.format & RS::ARRAY_FORMAT_BONES) {
		mesh->has_bone_weights = true;
	}

	if (p_surface.index_count) {
		bool is_index_16 = p_surface.vertex_count <= 65536;

		s->index_buffer = RD::get_singleton()->index_buffer_create(p_surface.index_count, is_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, p_surface.index_data, false);
		s->index_count = p_surface.index_count;
		s->index_array = RD::get_singleton()->index_array_create(s->index_buffer, 0, s->index_count);
		if (p_surface.lods.size()) {
			s->lods = memnew_arr(Mesh::Surface::LOD, p_surface.lods.size());
			s->lod_count = p_surface.lods.size();

			for (int i = 0; i < p_surface.lods.size(); i++) {
				uint32_t indices = p_surface.lods[i].index_data.size() / (is_index_16 ? 2 : 4);
				s->lods[i].index_buffer = RD::get_singleton()->index_buffer_create(indices, is_index_16 ? RD::INDEX_BUFFER_FORMAT_UINT16 : RD::INDEX_BUFFER_FORMAT_UINT32, p_surface.lods[i].index_data);
				s->lods[i].index_array = RD::get_singleton()->index_array_create(s->lods[i].index_buffer, 0, indices);
				s->lods[i].edge_length = p_surface.lods[i].edge_length;
				s->lods[i].index_count = indices;
			}
		}
	}

	s->aabb = p_surface.aabb;
	s->bone_aabbs = p_surface.bone_aabbs; //only really useful for returning them.

	if (mesh->blend_shape_count > 0) {
		s->blend_shape_buffer = RD::get_singleton()->storage_buffer_create(p_surface.blend_shape_data.size(), p_surface.blend_shape_data);
	}

	if (use_as_storage) {
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 0;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(s->vertex_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 1;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			if (s->skin_buffer.is_valid()) {
				u.ids.push_back(s->skin_buffer);
			} else {
				u.ids.push_back(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 2;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			if (s->blend_shape_buffer.is_valid()) {
				u.ids.push_back(s->blend_shape_buffer);
			} else {
				u.ids.push_back(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}

		s->uniform_set = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_SURFACE);
	}

	if (mesh->surface_count == 0) {
		mesh->bone_aabbs = p_surface.bone_aabbs;
		mesh->aabb = p_surface.aabb;
	} else {
		if (mesh->bone_aabbs.size() < p_surface.bone_aabbs.size()) {
			// ArrayMesh::_surface_set_data only allocates bone_aabbs up to max_bone
			// Each surface may affect different numbers of bones.
			mesh->bone_aabbs.resize(p_surface.bone_aabbs.size());
		}
		for (int i = 0; i < p_surface.bone_aabbs.size(); i++) {
			mesh->bone_aabbs.write[i].merge_with(p_surface.bone_aabbs[i]);
		}
		mesh->aabb.merge_with(p_surface.aabb);
	}

	s->material = p_surface.material;

	mesh->surfaces = (Mesh::Surface **)memrealloc(mesh->surfaces, sizeof(Mesh::Surface *) * (mesh->surface_count + 1));
	mesh->surfaces[mesh->surface_count] = s;
	mesh->surface_count++;

	for (MeshInstance *mi : mesh->instances) {
		_mesh_instance_add_surface(mi, mesh, mesh->surface_count - 1);
	}

	mesh->dependency.changed_notify(DEPENDENCY_CHANGED_MESH);

	for (Set<Mesh *>::Element *E = mesh->shadow_owners.front(); E; E = E->next()) {
		Mesh *shadow_owner = E->get();
		shadow_owner->shadow_mesh = RID();
		shadow_owner->dependency.changed_notify(DEPENDENCY_CHANGED_MESH);
	}

	mesh->material_cache.clear();
}

int RendererStorageRD::mesh_get_blend_shape_count(RID p_mesh) const {
	const Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);
	return mesh->blend_shape_count;
}

void RendererStorageRD::mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX((int)p_mode, 2);

	mesh->blend_shape_mode = p_mode;
}

RS::BlendShapeMode RendererStorageRD::mesh_get_blend_shape_mode(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, RS::BLEND_SHAPE_MODE_NORMALIZED);
	return mesh->blend_shape_mode;
}

void RendererStorageRD::mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(p_data.size() == 0);
	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->vertex_buffer, p_offset, data_size, r);
}

void RendererStorageRD::mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(p_data.size() == 0);
	ERR_FAIL_COND(mesh->surfaces[p_surface]->attribute_buffer.is_null());
	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->attribute_buffer, p_offset, data_size, r);
}

void RendererStorageRD::mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(p_data.size() == 0);
	ERR_FAIL_COND(mesh->surfaces[p_surface]->skin_buffer.is_null());
	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->skin_buffer, p_offset, data_size, r);
}

void RendererStorageRD::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	mesh->surfaces[p_surface]->material = p_material;

	mesh->dependency.changed_notify(DEPENDENCY_CHANGED_MATERIAL);
	mesh->material_cache.clear();
}

RID RendererStorageRD::mesh_surface_get_material(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());

	return mesh->surfaces[p_surface]->material;
}

RS::SurfaceData RendererStorageRD::mesh_get_surface(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, RS::SurfaceData());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RS::SurfaceData());

	Mesh::Surface &s = *mesh->surfaces[p_surface];

	RS::SurfaceData sd;
	sd.format = s.format;
	sd.vertex_data = RD::get_singleton()->buffer_get_data(s.vertex_buffer);
	if (s.attribute_buffer.is_valid()) {
		sd.attribute_data = RD::get_singleton()->buffer_get_data(s.attribute_buffer);
	}
	if (s.skin_buffer.is_valid()) {
		sd.skin_data = RD::get_singleton()->buffer_get_data(s.skin_buffer);
	}
	sd.vertex_count = s.vertex_count;
	sd.index_count = s.index_count;
	sd.primitive = s.primitive;

	if (sd.index_count) {
		sd.index_data = RD::get_singleton()->buffer_get_data(s.index_buffer);
	}
	sd.aabb = s.aabb;
	for (uint32_t i = 0; i < s.lod_count; i++) {
		RS::SurfaceData::LOD lod;
		lod.edge_length = s.lods[i].edge_length;
		lod.index_data = RD::get_singleton()->buffer_get_data(s.lods[i].index_buffer);
		sd.lods.push_back(lod);
	}

	sd.bone_aabbs = s.bone_aabbs;

	if (s.blend_shape_buffer.is_valid()) {
		sd.blend_shape_data = RD::get_singleton()->buffer_get_data(s.blend_shape_buffer);
	}

	return sd;
}

int RendererStorageRD::mesh_get_surface_count(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, 0);
	return mesh->surface_count;
}

void RendererStorageRD::mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	mesh->custom_aabb = p_aabb;
}

AABB RendererStorageRD::mesh_get_custom_aabb(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());
	return mesh->custom_aabb;
}

AABB RendererStorageRD::mesh_get_aabb(RID p_mesh, RID p_skeleton) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());

	if (mesh->custom_aabb != AABB()) {
		return mesh->custom_aabb;
	}

	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	if (!skeleton || skeleton->size == 0) {
		return mesh->aabb;
	}

	AABB aabb;

	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		AABB laabb;
		if ((mesh->surfaces[i]->format & RS::ARRAY_FORMAT_BONES) && mesh->surfaces[i]->bone_aabbs.size()) {
			int bs = mesh->surfaces[i]->bone_aabbs.size();
			const AABB *skbones = mesh->surfaces[i]->bone_aabbs.ptr();

			int sbs = skeleton->size;
			ERR_CONTINUE(bs > sbs);
			const float *baseptr = skeleton->data.ptr();

			bool first = true;

			if (skeleton->use_2d) {
				for (int j = 0; j < bs; j++) {
					if (skbones[0].size == Vector3()) {
						continue; //bone is unused
					}

					const float *dataptr = baseptr + j * 8;

					Transform3D mtx;

					mtx.basis.elements[0].x = dataptr[0];
					mtx.basis.elements[1].x = dataptr[1];
					mtx.origin.x = dataptr[3];

					mtx.basis.elements[0].y = dataptr[4];
					mtx.basis.elements[1].y = dataptr[5];
					mtx.origin.y = dataptr[7];

					AABB baabb = mtx.xform(skbones[j]);

					if (first) {
						laabb = baabb;
						first = false;
					} else {
						laabb.merge_with(baabb);
					}
				}
			} else {
				for (int j = 0; j < bs; j++) {
					if (skbones[0].size == Vector3()) {
						continue; //bone is unused
					}

					const float *dataptr = baseptr + j * 12;

					Transform3D mtx;

					mtx.basis.elements[0][0] = dataptr[0];
					mtx.basis.elements[0][1] = dataptr[1];
					mtx.basis.elements[0][2] = dataptr[2];
					mtx.origin.x = dataptr[3];
					mtx.basis.elements[1][0] = dataptr[4];
					mtx.basis.elements[1][1] = dataptr[5];
					mtx.basis.elements[1][2] = dataptr[6];
					mtx.origin.y = dataptr[7];
					mtx.basis.elements[2][0] = dataptr[8];
					mtx.basis.elements[2][1] = dataptr[9];
					mtx.basis.elements[2][2] = dataptr[10];
					mtx.origin.z = dataptr[11];

					AABB baabb = mtx.xform(skbones[j]);
					if (first) {
						laabb = baabb;
						first = false;
					} else {
						laabb.merge_with(baabb);
					}
				}
			}

			if (laabb.size == Vector3()) {
				laabb = mesh->surfaces[i]->aabb;
			}
		} else {
			laabb = mesh->surfaces[i]->aabb;
		}

		if (i == 0) {
			aabb = laabb;
		} else {
			aabb.merge_with(laabb);
		}
	}

	return aabb;
}

void RendererStorageRD::mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);

	Mesh *shadow_mesh = mesh_owner.get_or_null(mesh->shadow_mesh);
	if (shadow_mesh) {
		shadow_mesh->shadow_owners.erase(mesh);
	}
	mesh->shadow_mesh = p_shadow_mesh;

	shadow_mesh = mesh_owner.get_or_null(mesh->shadow_mesh);

	if (shadow_mesh) {
		shadow_mesh->shadow_owners.insert(mesh);
	}

	mesh->dependency.changed_notify(DEPENDENCY_CHANGED_MESH);
}

void RendererStorageRD::mesh_clear(RID p_mesh) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		Mesh::Surface &s = *mesh->surfaces[i];
		RD::get_singleton()->free(s.vertex_buffer); //clears arrays as dependency automatically, including all versions
		if (s.attribute_buffer.is_valid()) {
			RD::get_singleton()->free(s.attribute_buffer);
		}
		if (s.skin_buffer.is_valid()) {
			RD::get_singleton()->free(s.skin_buffer);
		}
		if (s.versions) {
			memfree(s.versions); //reallocs, so free with memfree.
		}

		if (s.index_buffer.is_valid()) {
			RD::get_singleton()->free(s.index_buffer);
		}

		if (s.lod_count) {
			for (uint32_t j = 0; j < s.lod_count; j++) {
				RD::get_singleton()->free(s.lods[j].index_buffer);
			}
			memdelete_arr(s.lods);
		}

		if (s.blend_shape_buffer.is_valid()) {
			RD::get_singleton()->free(s.blend_shape_buffer);
		}

		memdelete(mesh->surfaces[i]);
	}
	if (mesh->surfaces) {
		memfree(mesh->surfaces);
	}

	mesh->surfaces = nullptr;
	mesh->surface_count = 0;
	mesh->material_cache.clear();
	//clear instance data
	for (MeshInstance *mi : mesh->instances) {
		_mesh_instance_clear(mi);
	}
	mesh->has_bone_weights = false;
	mesh->dependency.changed_notify(DEPENDENCY_CHANGED_MESH);

	for (Set<Mesh *>::Element *E = mesh->shadow_owners.front(); E; E = E->next()) {
		Mesh *shadow_owner = E->get();
		shadow_owner->shadow_mesh = RID();
		shadow_owner->dependency.changed_notify(DEPENDENCY_CHANGED_MESH);
	}
}

bool RendererStorageRD::mesh_needs_instance(RID p_mesh, bool p_has_skeleton) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, false);

	return mesh->blend_shape_count > 0 || (mesh->has_bone_weights && p_has_skeleton);
}

/* MESH INSTANCE */

RID RendererStorageRD::mesh_instance_create(RID p_base) {
	Mesh *mesh = mesh_owner.get_or_null(p_base);
	ERR_FAIL_COND_V(!mesh, RID());

	RID rid = mesh_instance_owner.make_rid();
	MeshInstance *mi = mesh_instance_owner.get_or_null(rid);

	mi->mesh = mesh;

	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		_mesh_instance_add_surface(mi, mesh, i);
	}

	mi->I = mesh->instances.push_back(mi);

	mi->dirty = true;

	return rid;
}
void RendererStorageRD::mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);
	if (mi->skeleton == p_skeleton) {
		return;
	}
	mi->skeleton = p_skeleton;
	mi->skeleton_version = 0;
	mi->dirty = true;
}

void RendererStorageRD::mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);
	ERR_FAIL_COND(!mi);
	ERR_FAIL_INDEX(p_shape, (int)mi->blend_weights.size());
	mi->blend_weights[p_shape] = p_weight;
	mi->weights_dirty = true;
	//will be eventually updated
}

void RendererStorageRD::_mesh_instance_clear(MeshInstance *mi) {
	for (uint32_t i = 0; i < mi->surfaces.size(); i++) {
		if (mi->surfaces[i].vertex_buffer.is_valid()) {
			RD::get_singleton()->free(mi->surfaces[i].vertex_buffer);
		}
		if (mi->surfaces[i].versions) {
			for (uint32_t j = 0; j < mi->surfaces[i].version_count; j++) {
				RD::get_singleton()->free(mi->surfaces[i].versions[j].vertex_array);
			}
			memfree(mi->surfaces[i].versions);
		}
	}
	mi->surfaces.clear();

	if (mi->blend_weights_buffer.is_valid()) {
		RD::get_singleton()->free(mi->blend_weights_buffer);
	}
	mi->blend_weights.clear();
	mi->weights_dirty = false;
	mi->skeleton_version = 0;
}

void RendererStorageRD::_mesh_instance_add_surface(MeshInstance *mi, Mesh *mesh, uint32_t p_surface) {
	if (mesh->blend_shape_count > 0 && mi->blend_weights_buffer.is_null()) {
		mi->blend_weights.resize(mesh->blend_shape_count);
		for (uint32_t i = 0; i < mi->blend_weights.size(); i++) {
			mi->blend_weights[i] = 0;
		}
		mi->blend_weights_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * mi->blend_weights.size(), mi->blend_weights.to_byte_array());
		mi->weights_dirty = true;
	}

	MeshInstance::Surface s;
	if (mesh->blend_shape_count > 0 || (mesh->surfaces[p_surface]->format & RS::ARRAY_FORMAT_BONES)) {
		//surface warrants transform
		s.vertex_buffer = RD::get_singleton()->vertex_buffer_create(mesh->surfaces[p_surface]->vertex_buffer_size, Vector<uint8_t>(), true);

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.binding = 1;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.ids.push_back(s.vertex_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.binding = 2;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			if (mi->blend_weights_buffer.is_valid()) {
				u.ids.push_back(mi->blend_weights_buffer);
			} else {
				u.ids.push_back(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}
		s.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_INSTANCE);
	}

	mi->surfaces.push_back(s);
	mi->dirty = true;
}

void RendererStorageRD::mesh_instance_check_for_update(RID p_mesh_instance) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);

	bool needs_update = mi->dirty;

	if (mi->weights_dirty && !mi->weight_update_list.in_list()) {
		dirty_mesh_instance_weights.add(&mi->weight_update_list);
		needs_update = true;
	}

	if (mi->array_update_list.in_list()) {
		return;
	}

	if (!needs_update && mi->skeleton.is_valid()) {
		Skeleton *sk = skeleton_owner.get_or_null(mi->skeleton);
		if (sk && sk->version != mi->skeleton_version) {
			needs_update = true;
		}
	}

	if (needs_update) {
		dirty_mesh_instance_arrays.add(&mi->array_update_list);
	}
}

void RendererStorageRD::update_mesh_instances() {
	while (dirty_mesh_instance_weights.first()) {
		MeshInstance *mi = dirty_mesh_instance_weights.first()->self();

		if (mi->blend_weights_buffer.is_valid()) {
			RD::get_singleton()->buffer_update(mi->blend_weights_buffer, 0, mi->blend_weights.size() * sizeof(float), mi->blend_weights.ptr());
		}
		dirty_mesh_instance_weights.remove(&mi->weight_update_list);
		mi->weights_dirty = false;
	}
	if (dirty_mesh_instance_arrays.first() == nullptr) {
		return; //nothing to do
	}

	//process skeletons and blend shapes
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	while (dirty_mesh_instance_arrays.first()) {
		MeshInstance *mi = dirty_mesh_instance_arrays.first()->self();

		Skeleton *sk = skeleton_owner.get_or_null(mi->skeleton);

		for (uint32_t i = 0; i < mi->surfaces.size(); i++) {
			if (mi->surfaces[i].uniform_set == RID() || mi->mesh->surfaces[i]->uniform_set == RID()) {
				continue;
			}

			bool array_is_2d = mi->mesh->surfaces[i]->format & RS::ARRAY_FLAG_USE_2D_VERTICES;

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, skeleton_shader.pipeline[array_is_2d ? SkeletonShader::SHADER_MODE_2D : SkeletonShader::SHADER_MODE_3D]);

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mi->surfaces[i].uniform_set, SkeletonShader::UNIFORM_SET_INSTANCE);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mi->mesh->surfaces[i]->uniform_set, SkeletonShader::UNIFORM_SET_SURFACE);
			if (sk && sk->uniform_set_mi.is_valid()) {
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, sk->uniform_set_mi, SkeletonShader::UNIFORM_SET_SKELETON);
			} else {
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, skeleton_shader.default_skeleton_uniform_set, SkeletonShader::UNIFORM_SET_SKELETON);
			}

			SkeletonShader::PushConstant push_constant;

			push_constant.has_normal = mi->mesh->surfaces[i]->format & RS::ARRAY_FORMAT_NORMAL;
			push_constant.has_tangent = mi->mesh->surfaces[i]->format & RS::ARRAY_FORMAT_TANGENT;
			push_constant.has_skeleton = sk != nullptr && sk->use_2d == array_is_2d && (mi->mesh->surfaces[i]->format & RS::ARRAY_FORMAT_BONES);
			push_constant.has_blend_shape = mi->mesh->blend_shape_count > 0;

			push_constant.vertex_count = mi->mesh->surfaces[i]->vertex_count;
			push_constant.vertex_stride = (mi->mesh->surfaces[i]->vertex_buffer_size / mi->mesh->surfaces[i]->vertex_count) / 4;
			push_constant.skin_stride = (mi->mesh->surfaces[i]->skin_buffer_size / mi->mesh->surfaces[i]->vertex_count) / 4;
			push_constant.skin_weight_offset = (mi->mesh->surfaces[i]->format & RS::ARRAY_FLAG_USE_8_BONE_WEIGHTS) ? 4 : 2;

			push_constant.blend_shape_count = mi->mesh->blend_shape_count;
			push_constant.normalized_blend_shapes = mi->mesh->blend_shape_mode == RS::BLEND_SHAPE_MODE_NORMALIZED;
			push_constant.pad0 = 0;
			push_constant.pad1 = 0;

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(SkeletonShader::PushConstant));

			//dispatch without barrier, so all is done at the same time
			RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.vertex_count, 1, 1);
		}

		mi->dirty = false;
		if (sk) {
			mi->skeleton_version = sk->version;
		}
		dirty_mesh_instance_arrays.remove(&mi->array_update_list);
	}

	RD::get_singleton()->compute_list_end();
}

void RendererStorageRD::_mesh_surface_generate_version_for_input_mask(Mesh::Surface::Version &v, Mesh::Surface *s, uint32_t p_input_mask, MeshInstance::Surface *mis) {
	Vector<RD::VertexAttribute> attributes;
	Vector<RID> buffers;

	uint32_t stride = 0;
	uint32_t attribute_stride = 0;
	uint32_t skin_stride = 0;

	for (int i = 0; i < RS::ARRAY_INDEX; i++) {
		RD::VertexAttribute vd;
		RID buffer;
		vd.location = i;

		if (!(s->format & (1 << i))) {
			// Not supplied by surface, use default value
			buffer = mesh_default_rd_buffers[i];
			vd.stride = 0;
			switch (i) {
				case RS::ARRAY_VERTEX: {
					vd.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;

				} break;
				case RS::ARRAY_NORMAL: {
					vd.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
				} break;
				case RS::ARRAY_TANGENT: {
					vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
				} break;
				case RS::ARRAY_COLOR: {
					vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;

				} break;
				case RS::ARRAY_TEX_UV: {
					vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;

				} break;
				case RS::ARRAY_TEX_UV2: {
					vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
				} break;
				case RS::ARRAY_CUSTOM0:
				case RS::ARRAY_CUSTOM1:
				case RS::ARRAY_CUSTOM2:
				case RS::ARRAY_CUSTOM3: {
					//assumed weights too
					vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
				} break;
				case RS::ARRAY_BONES: {
					//assumed weights too
					vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
				} break;
				case RS::ARRAY_WEIGHTS: {
					//assumed weights too
					vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
				} break;
			}
		} else {
			//Supplied, use it

			vd.stride = 1; //mark that it needs a stride set (default uses 0)

			switch (i) {
				case RS::ARRAY_VERTEX: {
					vd.offset = stride;

					if (s->format & RS::ARRAY_FLAG_USE_2D_VERTICES) {
						vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
						stride += sizeof(float) * 2;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
						stride += sizeof(float) * 3;
					}

					if (mis) {
						buffer = mis->vertex_buffer;
					} else {
						buffer = s->vertex_buffer;
					}

				} break;
				case RS::ARRAY_NORMAL: {
					vd.offset = stride;

					vd.format = RD::DATA_FORMAT_A2B10G10R10_UNORM_PACK32;

					stride += sizeof(uint32_t);
					if (mis) {
						buffer = mis->vertex_buffer;
					} else {
						buffer = s->vertex_buffer;
					}
				} break;
				case RS::ARRAY_TANGENT: {
					vd.offset = stride;

					vd.format = RD::DATA_FORMAT_A2B10G10R10_UNORM_PACK32;
					stride += sizeof(uint32_t);
					if (mis) {
						buffer = mis->vertex_buffer;
					} else {
						buffer = s->vertex_buffer;
					}
				} break;
				case RS::ARRAY_COLOR: {
					vd.offset = attribute_stride;

					vd.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
					attribute_stride += sizeof(int8_t) * 4;
					buffer = s->attribute_buffer;
				} break;
				case RS::ARRAY_TEX_UV: {
					vd.offset = attribute_stride;

					vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
					attribute_stride += sizeof(float) * 2;
					buffer = s->attribute_buffer;

				} break;
				case RS::ARRAY_TEX_UV2: {
					vd.offset = attribute_stride;

					vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
					attribute_stride += sizeof(float) * 2;
					buffer = s->attribute_buffer;
				} break;
				case RS::ARRAY_CUSTOM0:
				case RS::ARRAY_CUSTOM1:
				case RS::ARRAY_CUSTOM2:
				case RS::ARRAY_CUSTOM3: {
					vd.offset = attribute_stride;

					int idx = i - RS::ARRAY_CUSTOM0;
					uint32_t fmt_shift[RS::ARRAY_CUSTOM_COUNT] = { RS::ARRAY_FORMAT_CUSTOM0_SHIFT, RS::ARRAY_FORMAT_CUSTOM1_SHIFT, RS::ARRAY_FORMAT_CUSTOM2_SHIFT, RS::ARRAY_FORMAT_CUSTOM3_SHIFT };
					uint32_t fmt = (s->format >> fmt_shift[idx]) & RS::ARRAY_FORMAT_CUSTOM_MASK;
					uint32_t fmtsize[RS::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };
					RD::DataFormat fmtrd[RS::ARRAY_CUSTOM_MAX] = { RD::DATA_FORMAT_R8G8B8A8_UNORM, RD::DATA_FORMAT_R8G8B8A8_SNORM, RD::DATA_FORMAT_R16G16_SFLOAT, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, RD::DATA_FORMAT_R32_SFLOAT, RD::DATA_FORMAT_R32G32_SFLOAT, RD::DATA_FORMAT_R32G32B32_SFLOAT, RD::DATA_FORMAT_R32G32B32A32_SFLOAT };
					vd.format = fmtrd[fmt];
					attribute_stride += fmtsize[fmt];
					buffer = s->attribute_buffer;
				} break;
				case RS::ARRAY_BONES: {
					vd.offset = skin_stride;

					vd.format = RD::DATA_FORMAT_R16G16B16A16_UINT;
					skin_stride += sizeof(int16_t) * 4;
					buffer = s->skin_buffer;
				} break;
				case RS::ARRAY_WEIGHTS: {
					vd.offset = skin_stride;

					vd.format = RD::DATA_FORMAT_R16G16B16A16_UNORM;
					skin_stride += sizeof(int16_t) * 4;
					buffer = s->skin_buffer;
				} break;
			}
		}

		if (!(p_input_mask & (1 << i))) {
			continue; // Shader does not need this, skip it (but computing stride was important anyway)
		}

		attributes.push_back(vd);
		buffers.push_back(buffer);
	}

	//update final stride
	for (int i = 0; i < attributes.size(); i++) {
		if (attributes[i].stride == 0) {
			continue; //default location
		}
		int loc = attributes[i].location;

		if (loc < RS::ARRAY_COLOR) {
			attributes.write[i].stride = stride;
		} else if (loc < RS::ARRAY_BONES) {
			attributes.write[i].stride = attribute_stride;
		} else {
			attributes.write[i].stride = skin_stride;
		}
	}

	v.input_mask = p_input_mask;
	v.vertex_format = RD::get_singleton()->vertex_format_create(attributes);
	v.vertex_array = RD::get_singleton()->vertex_array_create(s->vertex_count, v.vertex_format, buffers);
}

////////////////// MULTIMESH

RID RendererStorageRD::multimesh_allocate() {
	return multimesh_owner.allocate_rid();
}
void RendererStorageRD::multimesh_initialize(RID p_rid) {
	multimesh_owner.initialize_rid(p_rid, MultiMesh());
}

void RendererStorageRD::multimesh_allocate_data(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors, bool p_use_custom_data) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	if (multimesh->instances == p_instances && multimesh->xform_format == p_transform_format && multimesh->uses_colors == p_use_colors && multimesh->uses_custom_data == p_use_custom_data) {
		return;
	}

	if (multimesh->buffer.is_valid()) {
		RD::get_singleton()->free(multimesh->buffer);
		multimesh->buffer = RID();
		multimesh->uniform_set_2d = RID(); //cleared by dependency
		multimesh->uniform_set_3d = RID(); //cleared by dependency
	}

	if (multimesh->data_cache_dirty_regions) {
		memdelete_arr(multimesh->data_cache_dirty_regions);
		multimesh->data_cache_dirty_regions = nullptr;
		multimesh->data_cache_used_dirty_regions = 0;
	}

	multimesh->instances = p_instances;
	multimesh->xform_format = p_transform_format;
	multimesh->uses_colors = p_use_colors;
	multimesh->color_offset_cache = p_transform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12;
	multimesh->uses_custom_data = p_use_custom_data;
	multimesh->custom_data_offset_cache = multimesh->color_offset_cache + (p_use_colors ? 4 : 0);
	multimesh->stride_cache = multimesh->custom_data_offset_cache + (p_use_custom_data ? 4 : 0);
	multimesh->buffer_set = false;

	//print_line("allocate, elements: " + itos(p_instances) + " 2D: " + itos(p_transform_format == RS::MULTIMESH_TRANSFORM_2D) + " colors " + itos(multimesh->uses_colors) + " data " + itos(multimesh->uses_custom_data) + " stride " + itos(multimesh->stride_cache) + " total size " + itos(multimesh->stride_cache * multimesh->instances));
	multimesh->data_cache = Vector<float>();
	multimesh->aabb = AABB();
	multimesh->aabb_dirty = false;
	multimesh->visible_instances = MIN(multimesh->visible_instances, multimesh->instances);

	if (multimesh->instances) {
		multimesh->buffer = RD::get_singleton()->storage_buffer_create(multimesh->instances * multimesh->stride_cache * 4);
	}

	multimesh->dependency.changed_notify(DEPENDENCY_CHANGED_MULTIMESH);
}

int RendererStorageRD::multimesh_get_instance_count(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, 0);
	return multimesh->instances;
}

void RendererStorageRD::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	if (multimesh->mesh == p_mesh) {
		return;
	}
	multimesh->mesh = p_mesh;

	if (multimesh->instances == 0) {
		return;
	}

	if (multimesh->data_cache.size()) {
		//we have a data cache, just mark it dirt
		_multimesh_mark_all_dirty(multimesh, false, true);
	} else if (multimesh->instances) {
		//need to re-create AABB unfortunately, calling this has a penalty
		if (multimesh->buffer_set) {
			Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(multimesh->buffer);
			const uint8_t *r = buffer.ptr();
			const float *data = (const float *)r;
			_multimesh_re_create_aabb(multimesh, data, multimesh->instances);
		}
	}

	multimesh->dependency.changed_notify(DEPENDENCY_CHANGED_MESH);
}

#define MULTIMESH_DIRTY_REGION_SIZE 512

void RendererStorageRD::_multimesh_make_local(MultiMesh *multimesh) const {
	if (multimesh->data_cache.size() > 0) {
		return; //already local
	}
	ERR_FAIL_COND(multimesh->data_cache.size() > 0);
	// this means that the user wants to load/save individual elements,
	// for this, the data must reside on CPU, so just copy it there.
	multimesh->data_cache.resize(multimesh->instances * multimesh->stride_cache);
	{
		float *w = multimesh->data_cache.ptrw();

		if (multimesh->buffer_set) {
			Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(multimesh->buffer);
			{
				const uint8_t *r = buffer.ptr();
				memcpy(w, r, buffer.size());
			}
		} else {
			memset(w, 0, (size_t)multimesh->instances * multimesh->stride_cache * sizeof(float));
		}
	}
	uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
	multimesh->data_cache_dirty_regions = memnew_arr(bool, data_cache_dirty_region_count);
	for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
		multimesh->data_cache_dirty_regions[i] = false;
	}
	multimesh->data_cache_used_dirty_regions = 0;
}

void RendererStorageRD::_multimesh_mark_dirty(MultiMesh *multimesh, int p_index, bool p_aabb) {
	uint32_t region_index = p_index / MULTIMESH_DIRTY_REGION_SIZE;
#ifdef DEBUG_ENABLED
	uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
	ERR_FAIL_UNSIGNED_INDEX(region_index, data_cache_dirty_region_count); //bug
#endif
	if (!multimesh->data_cache_dirty_regions[region_index]) {
		multimesh->data_cache_dirty_regions[region_index] = true;
		multimesh->data_cache_used_dirty_regions++;
	}

	if (p_aabb) {
		multimesh->aabb_dirty = true;
	}

	if (!multimesh->dirty) {
		multimesh->dirty_list = multimesh_dirty_list;
		multimesh_dirty_list = multimesh;
		multimesh->dirty = true;
	}
}

void RendererStorageRD::_multimesh_mark_all_dirty(MultiMesh *multimesh, bool p_data, bool p_aabb) {
	if (p_data) {
		uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;

		for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
			if (!multimesh->data_cache_dirty_regions[i]) {
				multimesh->data_cache_dirty_regions[i] = true;
				multimesh->data_cache_used_dirty_regions++;
			}
		}
	}

	if (p_aabb) {
		multimesh->aabb_dirty = true;
	}

	if (!multimesh->dirty) {
		multimesh->dirty_list = multimesh_dirty_list;
		multimesh_dirty_list = multimesh;
		multimesh->dirty = true;
	}
}

void RendererStorageRD::_multimesh_re_create_aabb(MultiMesh *multimesh, const float *p_data, int p_instances) {
	ERR_FAIL_COND(multimesh->mesh.is_null());
	AABB aabb;
	AABB mesh_aabb = mesh_get_aabb(multimesh->mesh);
	for (int i = 0; i < p_instances; i++) {
		const float *data = p_data + multimesh->stride_cache * i;
		Transform3D t;

		if (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_3D) {
			t.basis.elements[0][0] = data[0];
			t.basis.elements[0][1] = data[1];
			t.basis.elements[0][2] = data[2];
			t.origin.x = data[3];
			t.basis.elements[1][0] = data[4];
			t.basis.elements[1][1] = data[5];
			t.basis.elements[1][2] = data[6];
			t.origin.y = data[7];
			t.basis.elements[2][0] = data[8];
			t.basis.elements[2][1] = data[9];
			t.basis.elements[2][2] = data[10];
			t.origin.z = data[11];

		} else {
			t.basis.elements[0].x = data[0];
			t.basis.elements[1].x = data[1];
			t.origin.x = data[3];

			t.basis.elements[0].y = data[4];
			t.basis.elements[1].y = data[5];
			t.origin.y = data[7];
		}

		if (i == 0) {
			aabb = t.xform(mesh_aabb);
		} else {
			aabb.merge_with(t.xform(mesh_aabb));
		}
	}

	multimesh->aabb = aabb;
}

void RendererStorageRD::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_3D);

	_multimesh_make_local(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache;

		dataptr[0] = p_transform.basis.elements[0][0];
		dataptr[1] = p_transform.basis.elements[0][1];
		dataptr[2] = p_transform.basis.elements[0][2];
		dataptr[3] = p_transform.origin.x;
		dataptr[4] = p_transform.basis.elements[1][0];
		dataptr[5] = p_transform.basis.elements[1][1];
		dataptr[6] = p_transform.basis.elements[1][2];
		dataptr[7] = p_transform.origin.y;
		dataptr[8] = p_transform.basis.elements[2][0];
		dataptr[9] = p_transform.basis.elements[2][1];
		dataptr[10] = p_transform.basis.elements[2][2];
		dataptr[11] = p_transform.origin.z;
	}

	_multimesh_mark_dirty(multimesh, p_index, true);
}

void RendererStorageRD::multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_2D);

	_multimesh_make_local(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache;

		dataptr[0] = p_transform.elements[0][0];
		dataptr[1] = p_transform.elements[1][0];
		dataptr[2] = 0;
		dataptr[3] = p_transform.elements[2][0];
		dataptr[4] = p_transform.elements[0][1];
		dataptr[5] = p_transform.elements[1][1];
		dataptr[6] = 0;
		dataptr[7] = p_transform.elements[2][1];
	}

	_multimesh_mark_dirty(multimesh, p_index, true);
}

void RendererStorageRD::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(!multimesh->uses_colors);

	_multimesh_make_local(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache + multimesh->color_offset_cache;

		dataptr[0] = p_color.r;
		dataptr[1] = p_color.g;
		dataptr[2] = p_color.b;
		dataptr[3] = p_color.a;
	}

	_multimesh_mark_dirty(multimesh, p_index, false);
}

void RendererStorageRD::multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(!multimesh->uses_custom_data);

	_multimesh_make_local(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache + multimesh->custom_data_offset_cache;

		dataptr[0] = p_color.r;
		dataptr[1] = p_color.g;
		dataptr[2] = p_color.b;
		dataptr[3] = p_color.a;
	}

	_multimesh_mark_dirty(multimesh, p_index, false);
}

RID RendererStorageRD::multimesh_get_mesh(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, RID());

	return multimesh->mesh;
}

Transform3D RendererStorageRD::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Transform3D());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Transform3D());
	ERR_FAIL_COND_V(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_3D, Transform3D());

	_multimesh_make_local(multimesh);

	Transform3D t;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache;

		t.basis.elements[0][0] = dataptr[0];
		t.basis.elements[0][1] = dataptr[1];
		t.basis.elements[0][2] = dataptr[2];
		t.origin.x = dataptr[3];
		t.basis.elements[1][0] = dataptr[4];
		t.basis.elements[1][1] = dataptr[5];
		t.basis.elements[1][2] = dataptr[6];
		t.origin.y = dataptr[7];
		t.basis.elements[2][0] = dataptr[8];
		t.basis.elements[2][1] = dataptr[9];
		t.basis.elements[2][2] = dataptr[10];
		t.origin.z = dataptr[11];
	}

	return t;
}

Transform2D RendererStorageRD::multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Transform2D());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Transform2D());
	ERR_FAIL_COND_V(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_2D, Transform2D());

	_multimesh_make_local(multimesh);

	Transform2D t;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache;

		t.elements[0][0] = dataptr[0];
		t.elements[1][0] = dataptr[1];
		t.elements[2][0] = dataptr[3];
		t.elements[0][1] = dataptr[4];
		t.elements[1][1] = dataptr[5];
		t.elements[2][1] = dataptr[7];
	}

	return t;
}

Color RendererStorageRD::multimesh_instance_get_color(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Color());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Color());
	ERR_FAIL_COND_V(!multimesh->uses_colors, Color());

	_multimesh_make_local(multimesh);

	Color c;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache + multimesh->color_offset_cache;

		c.r = dataptr[0];
		c.g = dataptr[1];
		c.b = dataptr[2];
		c.a = dataptr[3];
	}

	return c;
}

Color RendererStorageRD::multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Color());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Color());
	ERR_FAIL_COND_V(!multimesh->uses_custom_data, Color());

	_multimesh_make_local(multimesh);

	Color c;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache + multimesh->custom_data_offset_cache;

		c.r = dataptr[0];
		c.g = dataptr[1];
		c.b = dataptr[2];
		c.a = dataptr[3];
	}

	return c;
}

void RendererStorageRD::multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_COND(p_buffer.size() != (multimesh->instances * (int)multimesh->stride_cache));

	{
		const float *r = p_buffer.ptr();
		RD::get_singleton()->buffer_update(multimesh->buffer, 0, p_buffer.size() * sizeof(float), r);
		multimesh->buffer_set = true;
	}

	if (multimesh->data_cache.size()) {
		//if we have a data cache, just update it
		multimesh->data_cache = p_buffer;
		{
			//clear dirty since nothing will be dirty anymore
			uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
			for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
				multimesh->data_cache_dirty_regions[i] = false;
			}
			multimesh->data_cache_used_dirty_regions = 0;
		}

		_multimesh_mark_all_dirty(multimesh, false, true); //update AABB
	} else if (multimesh->mesh.is_valid()) {
		//if we have a mesh set, we need to re-generate the AABB from the new data
		const float *data = p_buffer.ptr();

		_multimesh_re_create_aabb(multimesh, data, multimesh->instances);
		multimesh->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
	}
}

Vector<float> RendererStorageRD::multimesh_get_buffer(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Vector<float>());
	if (multimesh->buffer.is_null()) {
		return Vector<float>();
	} else if (multimesh->data_cache.size()) {
		return multimesh->data_cache;
	} else {
		//get from memory

		Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(multimesh->buffer);
		Vector<float> ret;
		ret.resize(multimesh->instances * multimesh->stride_cache);
		{
			float *w = ret.ptrw();
			const uint8_t *r = buffer.ptr();
			memcpy(w, r, buffer.size());
		}

		return ret;
	}
}

void RendererStorageRD::multimesh_set_visible_instances(RID p_multimesh, int p_visible) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_COND(p_visible < -1 || p_visible > multimesh->instances);
	if (multimesh->visible_instances == p_visible) {
		return;
	}

	if (multimesh->data_cache.size()) {
		//there is a data cache..
		_multimesh_mark_all_dirty(multimesh, false, true);
	}

	multimesh->visible_instances = p_visible;

	multimesh->dependency.changed_notify(DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES);
}

int RendererStorageRD::multimesh_get_visible_instances(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, 0);
	return multimesh->visible_instances;
}

AABB RendererStorageRD::multimesh_get_aabb(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, AABB());
	if (multimesh->aabb_dirty) {
		const_cast<RendererStorageRD *>(this)->_update_dirty_multimeshes();
	}
	return multimesh->aabb;
}

void RendererStorageRD::_update_dirty_multimeshes() {
	while (multimesh_dirty_list) {
		MultiMesh *multimesh = multimesh_dirty_list;

		if (multimesh->data_cache.size()) { //may have been cleared, so only process if it exists
			const float *data = multimesh->data_cache.ptr();

			uint32_t visible_instances = multimesh->visible_instances >= 0 ? multimesh->visible_instances : multimesh->instances;

			if (multimesh->data_cache_used_dirty_regions) {
				uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
				uint32_t visible_region_count = visible_instances == 0 ? 0 : (visible_instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;

				uint32_t region_size = multimesh->stride_cache * MULTIMESH_DIRTY_REGION_SIZE * sizeof(float);

				if (multimesh->data_cache_used_dirty_regions > 32 || multimesh->data_cache_used_dirty_regions > visible_region_count / 2) {
					//if there too many dirty regions, or represent the majority of regions, just copy all, else transfer cost piles up too much
					RD::get_singleton()->buffer_update(multimesh->buffer, 0, MIN(visible_region_count * region_size, multimesh->instances * (uint32_t)multimesh->stride_cache * (uint32_t)sizeof(float)), data);
				} else {
					//not that many regions? update them all
					for (uint32_t i = 0; i < visible_region_count; i++) {
						if (multimesh->data_cache_dirty_regions[i]) {
							uint32_t offset = i * region_size;
							uint32_t size = multimesh->stride_cache * (uint32_t)multimesh->instances * (uint32_t)sizeof(float);
							RD::get_singleton()->buffer_update(multimesh->buffer, offset, MIN(region_size, size - offset), &data[i * region_size]);
						}
					}
				}

				for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
					multimesh->data_cache_dirty_regions[i] = false;
				}

				multimesh->data_cache_used_dirty_regions = 0;
			}

			if (multimesh->aabb_dirty) {
				//aabb is dirty..
				_multimesh_re_create_aabb(multimesh, data, visible_instances);
				multimesh->aabb_dirty = false;
				multimesh->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
			}
		}

		multimesh_dirty_list = multimesh->dirty_list;

		multimesh->dirty_list = nullptr;
		multimesh->dirty = false;
	}

	multimesh_dirty_list = nullptr;
}

/* PARTICLES */

RID RendererStorageRD::particles_allocate() {
	return particles_owner.allocate_rid();
}
void RendererStorageRD::particles_initialize(RID p_rid) {
	particles_owner.initialize_rid(p_rid, Particles());
}

void RendererStorageRD::particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	if (particles->mode == p_mode) {
		return;
	}

	_particles_free_data(particles);

	particles->mode = p_mode;
}

void RendererStorageRD::particles_set_emitting(RID p_particles, bool p_emitting) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->emitting = p_emitting;
}

bool RendererStorageRD::particles_get_emitting(RID p_particles) {
	ERR_FAIL_COND_V_MSG(RSG::threaded, false, "This function should never be used with threaded rendering, as it stalls the renderer.");
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, false);

	return particles->emitting;
}

void RendererStorageRD::_particles_free_data(Particles *particles) {
	if (particles->particle_buffer.is_valid()) {
		RD::get_singleton()->free(particles->particle_buffer);
		particles->particle_buffer = RID();
		RD::get_singleton()->free(particles->particle_instance_buffer);
		particles->particle_instance_buffer = RID();
	}

	if (particles->frame_params_buffer.is_valid()) {
		RD::get_singleton()->free(particles->frame_params_buffer);
		particles->frame_params_buffer = RID();
	}
	particles->particles_transforms_buffer_uniform_set = RID();

	if (RD::get_singleton()->uniform_set_is_valid(particles->trail_bind_pose_uniform_set)) {
		RD::get_singleton()->free(particles->trail_bind_pose_uniform_set);
	}
	particles->trail_bind_pose_uniform_set = RID();

	if (particles->trail_bind_pose_buffer.is_valid()) {
		RD::get_singleton()->free(particles->trail_bind_pose_buffer);
		particles->trail_bind_pose_buffer = RID();
	}
	if (RD::get_singleton()->uniform_set_is_valid(particles->collision_textures_uniform_set)) {
		RD::get_singleton()->free(particles->collision_textures_uniform_set);
	}
	particles->collision_textures_uniform_set = RID();

	if (particles->particles_sort_buffer.is_valid()) {
		RD::get_singleton()->free(particles->particles_sort_buffer);
		particles->particles_sort_buffer = RID();
		particles->particles_sort_uniform_set = RID();
	}

	if (particles->emission_buffer != nullptr) {
		particles->emission_buffer = nullptr;
		particles->emission_buffer_data.clear();
		RD::get_singleton()->free(particles->emission_storage_buffer);
		particles->emission_storage_buffer = RID();
	}

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		//will need to be re-created
		RD::get_singleton()->free(particles->particles_material_uniform_set);
	}
	particles->particles_material_uniform_set = RID();
}

void RendererStorageRD::particles_set_amount(RID p_particles, int p_amount) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	if (particles->amount == p_amount) {
		return;
	}

	_particles_free_data(particles);

	particles->amount = p_amount;

	particles->prev_ticks = 0;
	particles->phase = 0;
	particles->prev_phase = 0;
	particles->clear = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_lifetime(RID p_particles, double p_lifetime) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->lifetime = p_lifetime;
}

void RendererStorageRD::particles_set_one_shot(RID p_particles, bool p_one_shot) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->one_shot = p_one_shot;
}

void RendererStorageRD::particles_set_pre_process_time(RID p_particles, double p_time) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->pre_process_time = p_time;
}
void RendererStorageRD::particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->explosiveness = p_ratio;
}
void RendererStorageRD::particles_set_randomness_ratio(RID p_particles, real_t p_ratio) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->randomness = p_ratio;
}

void RendererStorageRD::particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->custom_aabb = p_aabb;
	particles->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_set_speed_scale(RID p_particles, double p_scale) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->speed_scale = p_scale;
}
void RendererStorageRD::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->use_local_coords = p_enable;
	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_fixed_fps(RID p_particles, int p_fps) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->fixed_fps = p_fps;

	_particles_free_data(particles);

	particles->prev_ticks = 0;
	particles->phase = 0;
	particles->prev_phase = 0;
	particles->clear = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_interpolate(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->interpolate = p_enable;
}

void RendererStorageRD::particles_set_fractional_delta(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->fractional_delta = p_enable;
}

void RendererStorageRD::particles_set_trails(RID p_particles, bool p_enable, double p_length) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(p_length < 0.1);
	p_length = MIN(10.0, p_length);

	particles->trails_enabled = p_enable;
	particles->trail_length = p_length;

	_particles_free_data(particles);

	particles->prev_ticks = 0;
	particles->phase = 0;
	particles->prev_phase = 0;
	particles->clear = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	if (particles->trail_bind_pose_buffer.is_valid() && particles->trail_bind_poses.size() != p_bind_poses.size()) {
		_particles_free_data(particles);

		particles->prev_ticks = 0;
		particles->phase = 0;
		particles->prev_phase = 0;
		particles->clear = true;
	}
	particles->trail_bind_poses = p_bind_poses;
	particles->trail_bind_poses_dirty = true;

	particles->dependency.changed_notify(DEPENDENCY_CHANGED_PARTICLES);
}

void RendererStorageRD::particles_set_collision_base_size(RID p_particles, real_t p_size) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->collision_base_size = p_size;
}

void RendererStorageRD::particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->transform_align = p_transform_align;
}

void RendererStorageRD::particles_set_process_material(RID p_particles, RID p_material) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->process_material = p_material;
}

void RendererStorageRD::particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->draw_order = p_order;
}

void RendererStorageRD::particles_set_draw_passes(RID p_particles, int p_passes) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->draw_passes.resize(p_passes);
}

void RendererStorageRD::particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_INDEX(p_pass, particles->draw_passes.size());
	particles->draw_passes.write[p_pass] = p_mesh;
}

void RendererStorageRD::particles_restart(RID p_particles) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->restart_request = true;
}

void RendererStorageRD::_particles_allocate_emission_buffer(Particles *particles) {
	ERR_FAIL_COND(particles->emission_buffer != nullptr);

	particles->emission_buffer_data.resize(sizeof(ParticleEmissionBuffer::Data) * particles->amount + sizeof(uint32_t) * 4);
	memset(particles->emission_buffer_data.ptrw(), 0, particles->emission_buffer_data.size());
	particles->emission_buffer = (ParticleEmissionBuffer *)particles->emission_buffer_data.ptrw();
	particles->emission_buffer->particle_max = particles->amount;

	particles->emission_storage_buffer = RD::get_singleton()->storage_buffer_create(particles->emission_buffer_data.size(), particles->emission_buffer_data);

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		//will need to be re-created
		RD::get_singleton()->free(particles->particles_material_uniform_set);
		particles->particles_material_uniform_set = RID();
	}
}

void RendererStorageRD::particles_set_subemitter(RID p_particles, RID p_subemitter_particles) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(p_particles == p_subemitter_particles);

	particles->sub_emitter = p_subemitter_particles;

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		RD::get_singleton()->free(particles->particles_material_uniform_set);
		particles->particles_material_uniform_set = RID(); //clear and force to re create sub emitting
	}
}

void RendererStorageRD::particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(particles->amount == 0);

	if (particles->emitting) {
		particles->clear = true;
		particles->emitting = false;
	}

	if (particles->emission_buffer == nullptr) {
		_particles_allocate_emission_buffer(particles);
	}

	if (particles->inactive) {
		//in case it was inactive, make active again
		particles->inactive = false;
		particles->inactive_time = 0;
	}

	int32_t idx = particles->emission_buffer->particle_count;
	if (idx < particles->emission_buffer->particle_max) {
		store_transform(p_transform, particles->emission_buffer->data[idx].xform);

		particles->emission_buffer->data[idx].velocity[0] = p_velocity.x;
		particles->emission_buffer->data[idx].velocity[1] = p_velocity.y;
		particles->emission_buffer->data[idx].velocity[2] = p_velocity.z;

		particles->emission_buffer->data[idx].custom[0] = p_custom.r;
		particles->emission_buffer->data[idx].custom[1] = p_custom.g;
		particles->emission_buffer->data[idx].custom[2] = p_custom.b;
		particles->emission_buffer->data[idx].custom[3] = p_custom.a;

		particles->emission_buffer->data[idx].color[0] = p_color.r;
		particles->emission_buffer->data[idx].color[1] = p_color.g;
		particles->emission_buffer->data[idx].color[2] = p_color.b;
		particles->emission_buffer->data[idx].color[3] = p_color.a;

		particles->emission_buffer->data[idx].flags = p_emit_flags;
		particles->emission_buffer->particle_count++;
	}
}

void RendererStorageRD::particles_request_process(RID p_particles) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	if (!particles->dirty) {
		particles->dirty = true;
		particles->update_list = particle_update_list;
		particle_update_list = particles;
	}
}

AABB RendererStorageRD::particles_get_current_aabb(RID p_particles) {
	if (RSG::threaded) {
		WARN_PRINT_ONCE("Calling this function with threaded rendering enabled stalls the renderer, use with care.");
	}

	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());

	int total_amount = particles->amount;
	if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
		total_amount *= particles->trail_bind_poses.size();
	}

	Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(particles->particle_buffer);
	ERR_FAIL_COND_V(buffer.size() != (int)(total_amount * sizeof(ParticleData)), AABB());

	Transform3D inv = particles->emission_transform.affine_inverse();

	AABB aabb;
	if (buffer.size()) {
		bool first = true;

		const ParticleData *particle_data = reinterpret_cast<const ParticleData *>(buffer.ptr());
		for (int i = 0; i < total_amount; i++) {
			if (particle_data[i].active) {
				Vector3 pos = Vector3(particle_data[i].xform[12], particle_data[i].xform[13], particle_data[i].xform[14]);
				if (!particles->use_local_coords) {
					pos = inv.xform(pos);
				}
				if (first) {
					aabb.position = pos;
					first = false;
				} else {
					aabb.expand_to(pos);
				}
			}
		}
	}

	float longest_axis_size = 0;
	for (int i = 0; i < particles->draw_passes.size(); i++) {
		if (particles->draw_passes[i].is_valid()) {
			AABB maabb = mesh_get_aabb(particles->draw_passes[i], RID());
			longest_axis_size = MAX(maabb.get_longest_axis_size(), longest_axis_size);
		}
	}

	aabb.grow_by(longest_axis_size);

	return aabb;
}

AABB RendererStorageRD::particles_get_aabb(RID p_particles) const {
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());

	return particles->custom_aabb;
}

void RendererStorageRD::particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	particles->emission_transform = p_transform;
}

int RendererStorageRD::particles_get_draw_passes(RID p_particles) const {
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, 0);

	return particles->draw_passes.size();
}

RID RendererStorageRD::particles_get_draw_pass_mesh(RID p_particles, int p_pass) const {
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, RID());
	ERR_FAIL_INDEX_V(p_pass, particles->draw_passes.size(), RID());

	return particles->draw_passes[p_pass];
}

void RendererStorageRD::particles_add_collision(RID p_particles, RID p_particles_collision_instance) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->collisions.insert(p_particles_collision_instance);
}

void RendererStorageRD::particles_remove_collision(RID p_particles, RID p_particles_collision_instance) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->collisions.erase(p_particles_collision_instance);
}

void RendererStorageRD::particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);
	particles->has_sdf_collision = p_enable;
	particles->sdf_collision_transform = p_xform;
	particles->sdf_collision_to_screen = p_to_screen;
	particles->sdf_collision_texture = p_texture;
}

void RendererStorageRD::_particles_process(Particles *p_particles, double p_delta) {
	if (p_particles->particles_material_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(p_particles->particles_material_uniform_set)) {
		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.ids.push_back(p_particles->frame_params_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 1;
			u.ids.push_back(p_particles->particle_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 2;
			if (p_particles->emission_storage_buffer.is_valid()) {
				u.ids.push_back(p_particles->emission_storage_buffer);
			} else {
				u.ids.push_back(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 3;
			Particles *sub_emitter = particles_owner.get_or_null(p_particles->sub_emitter);
			if (sub_emitter) {
				if (sub_emitter->emission_buffer == nullptr) { //no emission buffer, allocate emission buffer
					_particles_allocate_emission_buffer(sub_emitter);
				}
				u.ids.push_back(sub_emitter->emission_storage_buffer);
			} else {
				u.ids.push_back(default_rd_storage_buffer);
			}
			uniforms.push_back(u);
		}

		p_particles->particles_material_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.default_shader_rd, 1);
	}

	double new_phase = Math::fmod((double)p_particles->phase + (p_delta / p_particles->lifetime) * p_particles->speed_scale, 1.0);

	//move back history (if there is any)
	for (uint32_t i = p_particles->frame_history.size() - 1; i > 0; i--) {
		p_particles->frame_history[i] = p_particles->frame_history[i - 1];
	}
	//update current frame
	ParticlesFrameParams &frame_params = p_particles->frame_history[0];

	if (p_particles->clear) {
		p_particles->cycle_number = 0;
		p_particles->random_seed = Math::rand();
	} else if (new_phase < p_particles->phase) {
		if (p_particles->one_shot) {
			p_particles->emitting = false;
		}
		p_particles->cycle_number++;
	}

	frame_params.emitting = p_particles->emitting;
	frame_params.system_phase = new_phase;
	frame_params.prev_system_phase = p_particles->phase;

	p_particles->phase = new_phase;

	frame_params.time = RendererCompositorRD::singleton->get_total_time();
	frame_params.delta = p_delta * p_particles->speed_scale;
	frame_params.random_seed = p_particles->random_seed;
	frame_params.explosiveness = p_particles->explosiveness;
	frame_params.randomness = p_particles->randomness;

	if (p_particles->use_local_coords) {
		store_transform(Transform3D(), frame_params.emission_transform);
	} else {
		store_transform(p_particles->emission_transform, frame_params.emission_transform);
	}

	frame_params.cycle = p_particles->cycle_number;
	frame_params.frame = p_particles->frame_counter++;
	frame_params.pad0 = 0;
	frame_params.pad1 = 0;
	frame_params.pad2 = 0;

	{ //collision and attractors

		frame_params.collider_count = 0;
		frame_params.attractor_count = 0;
		frame_params.particle_size = p_particles->collision_base_size;

		RID collision_3d_textures[ParticlesFrameParams::MAX_3D_TEXTURES];
		RID collision_heightmap_texture;

		Transform3D to_particles;
		if (p_particles->use_local_coords) {
			to_particles = p_particles->emission_transform.affine_inverse();
		}

		if (p_particles->has_sdf_collision && RD::get_singleton()->texture_is_valid(p_particles->sdf_collision_texture)) {
			//2D collision

			Transform2D xform = p_particles->sdf_collision_transform; //will use dotproduct manually so invert beforehand
			Transform2D revert = xform.affine_inverse();
			frame_params.collider_count = 1;
			frame_params.colliders[0].transform[0] = xform.elements[0][0];
			frame_params.colliders[0].transform[1] = xform.elements[0][1];
			frame_params.colliders[0].transform[2] = 0;
			frame_params.colliders[0].transform[3] = xform.elements[2][0];

			frame_params.colliders[0].transform[4] = xform.elements[1][0];
			frame_params.colliders[0].transform[5] = xform.elements[1][1];
			frame_params.colliders[0].transform[6] = 0;
			frame_params.colliders[0].transform[7] = xform.elements[2][1];

			frame_params.colliders[0].transform[8] = revert.elements[0][0];
			frame_params.colliders[0].transform[9] = revert.elements[0][1];
			frame_params.colliders[0].transform[10] = 0;
			frame_params.colliders[0].transform[11] = revert.elements[2][0];

			frame_params.colliders[0].transform[12] = revert.elements[1][0];
			frame_params.colliders[0].transform[13] = revert.elements[1][1];
			frame_params.colliders[0].transform[14] = 0;
			frame_params.colliders[0].transform[15] = revert.elements[2][1];

			frame_params.colliders[0].extents[0] = p_particles->sdf_collision_to_screen.size.x;
			frame_params.colliders[0].extents[1] = p_particles->sdf_collision_to_screen.size.y;
			frame_params.colliders[0].extents[2] = p_particles->sdf_collision_to_screen.position.x;
			frame_params.colliders[0].scale = p_particles->sdf_collision_to_screen.position.y;
			frame_params.colliders[0].texture_index = 0;
			frame_params.colliders[0].type = ParticlesFrameParams::COLLISION_TYPE_2D_SDF;

			collision_heightmap_texture = p_particles->sdf_collision_texture;

			//replace in all other history frames where used because parameters are no longer valid if screen moves
			for (uint32_t i = 1; i < p_particles->frame_history.size(); i++) {
				if (p_particles->frame_history[i].collider_count > 0 && p_particles->frame_history[i].colliders[0].type == ParticlesFrameParams::COLLISION_TYPE_2D_SDF) {
					p_particles->frame_history[i].colliders[0] = frame_params.colliders[0];
				}
			}
		}

		uint32_t collision_3d_textures_used = 0;
		for (const Set<RID>::Element *E = p_particles->collisions.front(); E; E = E->next()) {
			ParticlesCollisionInstance *pci = particles_collision_instance_owner.get_or_null(E->get());
			if (!pci || !pci->active) {
				continue;
			}
			ParticlesCollision *pc = particles_collision_owner.get_or_null(pci->collision);
			ERR_CONTINUE(!pc);

			Transform3D to_collider = pci->transform;
			if (p_particles->use_local_coords) {
				to_collider = to_particles * to_collider;
			}
			Vector3 scale = to_collider.basis.get_scale();
			to_collider.basis.orthonormalize();

			if (pc->type <= RS::PARTICLES_COLLISION_TYPE_VECTOR_FIELD_ATTRACT) {
				//attractor
				if (frame_params.attractor_count >= ParticlesFrameParams::MAX_ATTRACTORS) {
					continue;
				}

				ParticlesFrameParams::Attractor &attr = frame_params.attractors[frame_params.attractor_count];

				store_transform(to_collider, attr.transform);
				attr.strength = pc->attractor_strength;
				attr.attenuation = pc->attractor_attenuation;
				attr.directionality = pc->attractor_directionality;

				switch (pc->type) {
					case RS::PARTICLES_COLLISION_TYPE_SPHERE_ATTRACT: {
						attr.type = ParticlesFrameParams::ATTRACTOR_TYPE_SPHERE;
						float radius = pc->radius;
						radius *= (scale.x + scale.y + scale.z) / 3.0;
						attr.extents[0] = radius;
						attr.extents[1] = radius;
						attr.extents[2] = radius;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_BOX_ATTRACT: {
						attr.type = ParticlesFrameParams::ATTRACTOR_TYPE_BOX;
						Vector3 extents = pc->extents * scale;
						attr.extents[0] = extents.x;
						attr.extents[1] = extents.y;
						attr.extents[2] = extents.z;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_VECTOR_FIELD_ATTRACT: {
						if (collision_3d_textures_used >= ParticlesFrameParams::MAX_3D_TEXTURES) {
							continue;
						}
						attr.type = ParticlesFrameParams::ATTRACTOR_TYPE_VECTOR_FIELD;
						Vector3 extents = pc->extents * scale;
						attr.extents[0] = extents.x;
						attr.extents[1] = extents.y;
						attr.extents[2] = extents.z;
						attr.texture_index = collision_3d_textures_used;

						collision_3d_textures[collision_3d_textures_used] = pc->field_texture;
						collision_3d_textures_used++;
					} break;
					default: {
					}
				}

				frame_params.attractor_count++;
			} else {
				//collider
				if (frame_params.collider_count >= ParticlesFrameParams::MAX_COLLIDERS) {
					continue;
				}

				ParticlesFrameParams::Collider &col = frame_params.colliders[frame_params.collider_count];

				store_transform(to_collider, col.transform);
				switch (pc->type) {
					case RS::PARTICLES_COLLISION_TYPE_SPHERE_COLLIDE: {
						col.type = ParticlesFrameParams::COLLISION_TYPE_SPHERE;
						float radius = pc->radius;
						radius *= (scale.x + scale.y + scale.z) / 3.0;
						col.extents[0] = radius;
						col.extents[1] = radius;
						col.extents[2] = radius;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_BOX_COLLIDE: {
						col.type = ParticlesFrameParams::COLLISION_TYPE_BOX;
						Vector3 extents = pc->extents * scale;
						col.extents[0] = extents.x;
						col.extents[1] = extents.y;
						col.extents[2] = extents.z;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_SDF_COLLIDE: {
						if (collision_3d_textures_used >= ParticlesFrameParams::MAX_3D_TEXTURES) {
							continue;
						}
						col.type = ParticlesFrameParams::COLLISION_TYPE_SDF;
						Vector3 extents = pc->extents * scale;
						col.extents[0] = extents.x;
						col.extents[1] = extents.y;
						col.extents[2] = extents.z;
						col.texture_index = collision_3d_textures_used;
						col.scale = (scale.x + scale.y + scale.z) * 0.333333333333; //non uniform scale non supported

						collision_3d_textures[collision_3d_textures_used] = pc->field_texture;
						collision_3d_textures_used++;
					} break;
					case RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE: {
						if (collision_heightmap_texture != RID()) { //already taken
							continue;
						}

						col.type = ParticlesFrameParams::COLLISION_TYPE_HEIGHT_FIELD;
						Vector3 extents = pc->extents * scale;
						col.extents[0] = extents.x;
						col.extents[1] = extents.y;
						col.extents[2] = extents.z;
						collision_heightmap_texture = pc->heightfield_texture;
					} break;
					default: {
					}
				}

				frame_params.collider_count++;
			}
		}

		bool different = false;
		if (collision_3d_textures_used == p_particles->collision_3d_textures_used) {
			for (int i = 0; i < ParticlesFrameParams::MAX_3D_TEXTURES; i++) {
				if (p_particles->collision_3d_textures[i] != collision_3d_textures[i]) {
					different = true;
					break;
				}
			}
		}

		if (collision_heightmap_texture != p_particles->collision_heightmap_texture) {
			different = true;
		}

		bool uniform_set_valid = RD::get_singleton()->uniform_set_is_valid(p_particles->collision_textures_uniform_set);

		if (different || !uniform_set_valid) {
			if (uniform_set_valid) {
				RD::get_singleton()->free(p_particles->collision_textures_uniform_set);
			}

			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				for (uint32_t i = 0; i < ParticlesFrameParams::MAX_3D_TEXTURES; i++) {
					RID rd_tex;
					if (i < collision_3d_textures_used) {
						Texture *t = texture_owner.get_or_null(collision_3d_textures[i]);
						if (t && t->type == Texture::TYPE_3D) {
							rd_tex = t->rd_texture;
						}
					}

					if (rd_tex == RID()) {
						rd_tex = default_rd_textures[DEFAULT_RD_TEXTURE_3D_WHITE];
					}
					u.ids.push_back(rd_tex);
				}
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 1;
				if (collision_heightmap_texture.is_valid()) {
					u.ids.push_back(collision_heightmap_texture);
				} else {
					u.ids.push_back(default_rd_textures[DEFAULT_RD_TEXTURE_BLACK]);
				}
				uniforms.push_back(u);
			}
			p_particles->collision_textures_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.default_shader_rd, 2);
		}
	}

	ParticlesShader::PushConstant push_constant;

	int process_amount = p_particles->amount;

	if (p_particles->trails_enabled && p_particles->trail_bind_poses.size() > 1) {
		process_amount *= p_particles->trail_bind_poses.size();
	}
	push_constant.clear = p_particles->clear;
	push_constant.total_particles = p_particles->amount;
	push_constant.lifetime = p_particles->lifetime;
	push_constant.trail_size = p_particles->trail_params.size();
	push_constant.use_fractional_delta = p_particles->fractional_delta;
	push_constant.sub_emitter_mode = !p_particles->emitting && p_particles->emission_buffer && (p_particles->emission_buffer->particle_count > 0 || p_particles->force_sub_emit);
	push_constant.trail_pass = false;

	p_particles->force_sub_emit = false; //reset

	Particles *sub_emitter = particles_owner.get_or_null(p_particles->sub_emitter);

	if (sub_emitter && sub_emitter->emission_storage_buffer.is_valid()) {
		//	print_line("updating subemitter buffer");
		int32_t zero[4] = { 0, sub_emitter->amount, 0, 0 };
		RD::get_singleton()->buffer_update(sub_emitter->emission_storage_buffer, 0, sizeof(uint32_t) * 4, zero);
		push_constant.can_emit = true;

		if (sub_emitter->emitting) {
			sub_emitter->emitting = false;
			sub_emitter->clear = true; //will need to clear if it was emitting, sorry
		}
		//make sure the sub emitter processes particles too
		sub_emitter->inactive = false;
		sub_emitter->inactive_time = 0;

		sub_emitter->force_sub_emit = true;

	} else {
		push_constant.can_emit = false;
	}

	if (p_particles->emission_buffer && p_particles->emission_buffer->particle_count) {
		RD::get_singleton()->buffer_update(p_particles->emission_storage_buffer, 0, sizeof(uint32_t) * 4 + sizeof(ParticleEmissionBuffer::Data) * p_particles->emission_buffer->particle_count, p_particles->emission_buffer);
		p_particles->emission_buffer->particle_count = 0;
	}

	p_particles->clear = false;

	if (p_particles->trail_params.size() > 1) {
		//fill the trail params
		for (uint32_t i = 0; i < p_particles->trail_params.size(); i++) {
			uint32_t src_idx = i * p_particles->frame_history.size() / p_particles->trail_params.size();
			p_particles->trail_params[i] = p_particles->frame_history[src_idx];
		}
	} else {
		p_particles->trail_params[0] = p_particles->frame_history[0];
	}

	RD::get_singleton()->buffer_update(p_particles->frame_params_buffer, 0, sizeof(ParticlesFrameParams) * p_particles->trail_params.size(), p_particles->trail_params.ptr());

	ParticlesMaterialData *m = (ParticlesMaterialData *)material_get_data(p_particles->process_material, SHADER_TYPE_PARTICLES);
	if (!m) {
		m = (ParticlesMaterialData *)material_get_data(particles_shader.default_material, SHADER_TYPE_PARTICLES);
	}

	ERR_FAIL_COND(!m);

	p_particles->has_collision_cache = m->shader_data->uses_collision;

	//todo should maybe compute all particle systems together?
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, m->shader_data->pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles_shader.base_uniform_set, 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_particles->particles_material_uniform_set, 1);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_particles->collision_textures_uniform_set, 2);

	if (m->uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(m->uniform_set)) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, m->uniform_set, 3);
	}

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ParticlesShader::PushConstant));

	if (p_particles->trails_enabled && p_particles->trail_bind_poses.size() > 1) {
		//trails requires two passes in order to catch particle starts
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, process_amount / p_particles->trail_bind_poses.size(), 1, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);

		push_constant.trail_pass = true;
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ParticlesShader::PushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, process_amount - p_particles->amount, 1, 1);
	} else {
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, process_amount, 1, 1);
	}

	RD::get_singleton()->compute_list_end();
}

void RendererStorageRD::particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) {
	Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND(!particles);

	if (particles->draw_order != RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY) {
		return;
	}

	if (particles->particle_buffer.is_null()) {
		return; //particles have not processed yet
	}

	bool do_sort = particles->draw_order == RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH;

	//copy to sort buffer
	if (do_sort && particles->particles_sort_buffer == RID()) {
		uint32_t size = particles->amount;
		if (size & 1) {
			size++; //make multiple of 16
		}
		size *= sizeof(float) * 2;
		particles->particles_sort_buffer = RD::get_singleton()->storage_buffer_create(size);

		{
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.ids.push_back(particles->particles_sort_buffer);
				uniforms.push_back(u);
			}

			particles->particles_sort_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, ParticlesShader::COPY_MODE_FILL_SORT_BUFFER), 1);
		}
	}

	ParticlesShader::CopyPushConstant copy_push_constant;

	if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
		int fixed_fps = 60.0;
		if (particles->fixed_fps > 0) {
			fixed_fps = particles->fixed_fps;
		}

		copy_push_constant.trail_size = particles->trail_bind_poses.size();
		copy_push_constant.trail_total = particles->frame_history.size();
		copy_push_constant.frame_delta = 1.0 / fixed_fps;
	} else {
		copy_push_constant.trail_size = 1;
		copy_push_constant.trail_total = 1;
		copy_push_constant.frame_delta = 0.0;
	}

	copy_push_constant.order_by_lifetime = (particles->draw_order == RS::PARTICLES_DRAW_ORDER_LIFETIME || particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME);
	copy_push_constant.lifetime_split = MIN(particles->amount * particles->phase, particles->amount - 1);
	copy_push_constant.lifetime_reverse = particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME;

	copy_push_constant.frame_remainder = particles->interpolate ? particles->frame_remainder : 0.0;
	copy_push_constant.total_particles = particles->amount;

	Vector3 axis = -p_axis; // cameras look to z negative

	if (particles->use_local_coords) {
		axis = particles->emission_transform.basis.xform_inv(axis).normalized();
	}

	copy_push_constant.sort_direction[0] = axis.x;
	copy_push_constant.sort_direction[1] = axis.y;
	copy_push_constant.sort_direction[2] = axis.z;

	copy_push_constant.align_up[0] = p_up_axis.x;
	copy_push_constant.align_up[1] = p_up_axis.y;
	copy_push_constant.align_up[2] = p_up_axis.z;

	copy_push_constant.align_mode = particles->transform_align;

	if (do_sort) {
		RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[ParticlesShader::COPY_MODE_FILL_SORT_BUFFER]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_sort_uniform_set, 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->trail_bind_pose_uniform_set, 2);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

		RD::get_singleton()->compute_list_dispatch_threads(compute_list, particles->amount, 1, 1);

		RD::get_singleton()->compute_list_end();
		effects->sort_buffer(particles->particles_sort_uniform_set, particles->amount);
	}

	copy_push_constant.total_particles *= copy_push_constant.total_particles;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[do_sort ? ParticlesShader::COPY_MODE_FILL_INSTANCES_WITH_SORT_BUFFER : (particles->mode == RS::PARTICLES_MODE_2D ? ParticlesShader::COPY_MODE_FILL_INSTANCES_2D : ParticlesShader::COPY_MODE_FILL_INSTANCES)]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
	if (do_sort) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_sort_uniform_set, 1);
	}
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->trail_bind_pose_uniform_set, 2);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, copy_push_constant.total_particles, 1, 1);

	RD::get_singleton()->compute_list_end();
}

void RendererStorageRD::_particles_update_buffers(Particles *particles) {
	if (particles->amount > 0 && particles->particle_buffer.is_null()) {
		int total_amount = particles->amount;
		if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
			total_amount *= particles->trail_bind_poses.size();
		}

		uint32_t xform_size = particles->mode == RS::PARTICLES_MODE_2D ? 2 : 3;

		particles->particle_buffer = RD::get_singleton()->storage_buffer_create(sizeof(ParticleData) * total_amount);

		particles->particle_instance_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * 4 * (xform_size + 1 + 1) * total_amount);
		//needs to clear it

		{
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 1;
				u.ids.push_back(particles->particle_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.ids.push_back(particles->particle_instance_buffer);
				uniforms.push_back(u);
			}

			particles->particles_copy_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, 0), 0);
		}
	}
}
void RendererStorageRD::update_particles() {
	while (particle_update_list) {
		//use transform feedback to process particles

		Particles *particles = particle_update_list;

		//take and remove
		particle_update_list = particles->update_list;
		particles->update_list = nullptr;
		particles->dirty = false;

		_particles_update_buffers(particles);

		if (particles->restart_request) {
			particles->prev_ticks = 0;
			particles->phase = 0;
			particles->prev_phase = 0;
			particles->clear = true;
			particles->restart_request = false;
		}

		if (particles->inactive && !particles->emitting) {
			//go next
			continue;
		}

		if (particles->emitting) {
			if (particles->inactive) {
				//restart system from scratch
				particles->prev_ticks = 0;
				particles->phase = 0;
				particles->prev_phase = 0;
				particles->clear = true;
			}
			particles->inactive = false;
			particles->inactive_time = 0;
		} else {
			particles->inactive_time += particles->speed_scale * RendererCompositorRD::singleton->get_frame_delta_time();
			if (particles->inactive_time > particles->lifetime * 1.2) {
				particles->inactive = true;
				continue;
			}
		}

#ifndef _MSC_VER
#warning Should use display refresh rate for all this
#endif

		float screen_hz = 60;

		int fixed_fps = 0;
		if (particles->fixed_fps > 0) {
			fixed_fps = particles->fixed_fps;
		} else if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
			fixed_fps = screen_hz;
		}
		{
			//update trails
			int history_size = 1;
			int trail_steps = 1;
			if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
				history_size = MAX(1, int(particles->trail_length * fixed_fps));
				trail_steps = particles->trail_bind_poses.size();
			}

			if (uint32_t(history_size) != particles->frame_history.size()) {
				particles->frame_history.resize(history_size);
				memset(particles->frame_history.ptr(), 0, sizeof(ParticlesFrameParams) * history_size);
			}

			if (uint32_t(trail_steps) != particles->trail_params.size() || particles->frame_params_buffer.is_null()) {
				particles->trail_params.resize(trail_steps);
				if (particles->frame_params_buffer.is_valid()) {
					RD::get_singleton()->free(particles->frame_params_buffer);
				}
				particles->frame_params_buffer = RD::get_singleton()->storage_buffer_create(sizeof(ParticlesFrameParams) * trail_steps);
			}

			if (particles->trail_bind_poses.size() > 1 && particles->trail_bind_pose_buffer.is_null()) {
				particles->trail_bind_pose_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * 16 * particles->trail_bind_poses.size());
				particles->trail_bind_poses_dirty = true;
			}

			if (particles->trail_bind_pose_uniform_set.is_null()) {
				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 0;
					if (particles->trail_bind_pose_buffer.is_valid()) {
						u.ids.push_back(particles->trail_bind_pose_buffer);
					} else {
						u.ids.push_back(default_rd_storage_buffer);
					}
					uniforms.push_back(u);
				}

				particles->trail_bind_pose_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, 0), 2);
			}

			if (particles->trail_bind_pose_buffer.is_valid() && particles->trail_bind_poses_dirty) {
				if (particles_shader.pose_update_buffer.size() < uint32_t(particles->trail_bind_poses.size()) * 16) {
					particles_shader.pose_update_buffer.resize(particles->trail_bind_poses.size() * 16);
				}

				for (int i = 0; i < particles->trail_bind_poses.size(); i++) {
					store_transform(particles->trail_bind_poses[i], &particles_shader.pose_update_buffer[i * 16]);
				}

				RD::get_singleton()->buffer_update(particles->trail_bind_pose_buffer, 0, particles->trail_bind_poses.size() * 16 * sizeof(float), particles_shader.pose_update_buffer.ptr());
			}
		}

		bool zero_time_scale = Engine::get_singleton()->get_time_scale() <= 0.0;

		if (particles->clear && particles->pre_process_time > 0.0) {
			double frame_time;
			if (fixed_fps > 0) {
				frame_time = 1.0 / fixed_fps;
			} else {
				frame_time = 1.0 / 30.0;
			}

			double todo = particles->pre_process_time;

			while (todo >= 0) {
				_particles_process(particles, frame_time);
				todo -= frame_time;
			}
		}

		if (fixed_fps > 0) {
			double frame_time;
			double decr;
			if (zero_time_scale) {
				frame_time = 0.0;
				decr = 1.0 / fixed_fps;
			} else {
				frame_time = 1.0 / fixed_fps;
				decr = frame_time;
			}
			double delta = RendererCompositorRD::singleton->get_frame_delta_time();
			if (delta > 0.1) { //avoid recursive stalls if fps goes below 10
				delta = 0.1;
			} else if (delta <= 0.0) { //unlikely but..
				delta = 0.001;
			}
			double todo = particles->frame_remainder + delta;

			while (todo >= frame_time) {
				_particles_process(particles, frame_time);
				todo -= decr;
			}

			particles->frame_remainder = todo;

		} else {
			if (zero_time_scale) {
				_particles_process(particles, 0.0);
			} else {
				_particles_process(particles, RendererCompositorRD::singleton->get_frame_delta_time());
			}
		}

		//copy particles to instance buffer

		if (particles->draw_order != RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD && particles->transform_align != RS::PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY) {
			//does not need view dependent operation, do copy here
			ParticlesShader::CopyPushConstant copy_push_constant;

			int total_amount = particles->amount;
			if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
				total_amount *= particles->trail_bind_poses.size();
			}

			copy_push_constant.total_particles = total_amount;
			copy_push_constant.frame_remainder = particles->interpolate ? particles->frame_remainder : 0.0;
			copy_push_constant.align_mode = particles->transform_align;
			copy_push_constant.align_up[0] = 0;
			copy_push_constant.align_up[1] = 0;
			copy_push_constant.align_up[2] = 0;

			if (particles->trails_enabled && particles->trail_bind_poses.size() > 1) {
				copy_push_constant.trail_size = particles->trail_bind_poses.size();
				copy_push_constant.trail_total = particles->frame_history.size();
				copy_push_constant.frame_delta = 1.0 / fixed_fps;
			} else {
				copy_push_constant.trail_size = 1;
				copy_push_constant.trail_total = 1;
				copy_push_constant.frame_delta = 0.0;
			}

			copy_push_constant.order_by_lifetime = (particles->draw_order == RS::PARTICLES_DRAW_ORDER_LIFETIME || particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME);
			copy_push_constant.lifetime_split = MIN(particles->amount * particles->phase, particles->amount - 1);
			copy_push_constant.lifetime_reverse = particles->draw_order == RS::PARTICLES_DRAW_ORDER_REVERSE_LIFETIME;

			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[particles->mode == RS::PARTICLES_MODE_2D ? ParticlesShader::COPY_MODE_FILL_INSTANCES_2D : ParticlesShader::COPY_MODE_FILL_INSTANCES]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->trail_bind_pose_uniform_set, 2);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

			RD::get_singleton()->compute_list_dispatch_threads(compute_list, total_amount, 1, 1);

			RD::get_singleton()->compute_list_end();
		}

		particles->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
	}
}

bool RendererStorageRD::particles_is_inactive(RID p_particles) const {
	ERR_FAIL_COND_V_MSG(RSG::threaded, false, "This function should never be used with threaded rendering, as it stalls the renderer.");
	const Particles *particles = particles_owner.get_or_null(p_particles);
	ERR_FAIL_COND_V(!particles, false);
	return !particles->emitting && particles->inactive;
}

/* SKY SHADER */

void RendererStorageRD::ParticlesShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_collision = false;

	if (code == String()) {
		return; //just invalid, but no error
	}

	ShaderCompilerRD::GeneratedCode gen_code;
	ShaderCompilerRD::IdentifierActions actions;
	actions.entry_point_stages["start"] = ShaderCompilerRD::STAGE_COMPUTE;
	actions.entry_point_stages["process"] = ShaderCompilerRD::STAGE_COMPUTE;

	/*
	uses_time = false;

	actions.render_mode_flags["use_half_res_pass"] = &uses_half_res;
	actions.render_mode_flags["use_quarter_res_pass"] = &uses_quarter_res;

	actions.usage_flag_pointers["TIME"] = &uses_time;
*/

	actions.usage_flag_pointers["COLLIDED"] = &uses_collision;

	actions.uniforms = &uniforms;

	Error err = base_singleton->particles_shader.compiler.compile(RS::SHADER_PARTICLES, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = base_singleton->particles_shader.shader.version_create();
	}

	base_singleton->particles_shader.shader.version_set_compute_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompilerRD::STAGE_COMPUTE], gen_code.defines);
	ERR_FAIL_COND(!base_singleton->particles_shader.shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//update pipelines

	pipeline = RD::get_singleton()->compute_pipeline_create(base_singleton->particles_shader.shader.version_get_shader(version, 0));

	valid = true;
}

void RendererStorageRD::ParticlesShaderData::set_default_texture_param(const StringName &p_name, RID p_texture) {
	if (!p_texture.is_valid()) {
		default_texture_params.erase(p_name);
	} else {
		default_texture_params[p_name] = p_texture;
	}
}

void RendererStorageRD::ParticlesShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL || E.value.scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		if (E.value.texture_order >= 0) {
			order[E.value.texture_order + 100000] = E.key;
		} else {
			order[E.value.order] = E.key;
		}
	}

	for (const KeyValue<int, StringName> &E : order) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E.value]);
		pi.name = E.value;
		p_param_list->push_back(pi);
	}
}

void RendererStorageRD::ParticlesShaderData::get_instance_param_list(List<RendererStorage::InstanceShaderParam> *p_param_list) const {
	for (const KeyValue<StringName, ShaderLanguage::ShaderNode::Uniform> &E : uniforms) {
		if (E.value.scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RendererStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E.value);
		p.info.name = E.key; //supply name
		p.index = E.value.instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E.value.default_value, E.value.type, E.value.array_size, E.value.hint);
		p_param_list->push_back(p);
	}
}

bool RendererStorageRD::ParticlesShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RendererStorageRD::ParticlesShaderData::is_animated() const {
	return false;
}

bool RendererStorageRD::ParticlesShaderData::casts_shadows() const {
	return false;
}

Variant RendererStorageRD::ParticlesShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.array_size, uniform.hint);
	}
	return Variant();
}

RS::ShaderNativeSourceCode RendererStorageRD::ParticlesShaderData::get_native_source_code() const {
	return base_singleton->particles_shader.shader.version_get_native_source_code(version);
}

RendererStorageRD::ParticlesShaderData::ParticlesShaderData() {
	valid = false;
}

RendererStorageRD::ParticlesShaderData::~ParticlesShaderData() {
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		base_singleton->particles_shader.shader.version_free(version);
	}
}

RendererStorageRD::ShaderData *RendererStorageRD::_create_particles_shader_func() {
	ParticlesShaderData *shader_data = memnew(ParticlesShaderData);
	return shader_data;
}

bool RendererStorageRD::ParticlesMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	uniform_set_updated = true;

	return update_parameters_uniform_set(p_parameters, p_uniform_dirty, p_textures_dirty, shader_data->uniforms, shader_data->ubo_offsets.ptr(), shader_data->texture_uniforms, shader_data->default_texture_params, shader_data->ubo_size, uniform_set, base_singleton->particles_shader.shader.version_get_shader(shader_data->version, 0), 3);
}

RendererStorageRD::ParticlesMaterialData::~ParticlesMaterialData() {
	free_parameters_uniform_set(uniform_set);
}

RendererStorageRD::MaterialData *RendererStorageRD::_create_particles_material_func(ParticlesShaderData *p_shader) {
	ParticlesMaterialData *material_data = memnew(ParticlesMaterialData);
	material_data->shader_data = p_shader;
	material_data->last_frame = false;
	//update will happen later anyway so do nothing.
	return material_data;
}
////////

/* PARTICLES COLLISION API */

RID RendererStorageRD::particles_collision_allocate() {
	return particles_collision_owner.allocate_rid();
}
void RendererStorageRD::particles_collision_initialize(RID p_rid) {
	particles_collision_owner.initialize_rid(p_rid, ParticlesCollision());
}

RID RendererStorageRD::particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, RID());
	ERR_FAIL_COND_V(particles_collision->type != RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE, RID());

	if (particles_collision->heightfield_texture == RID()) {
		//create
		int resolutions[RS::PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_MAX] = { 256, 512, 1024, 2048, 4096, 8192 };
		Size2i size;
		if (particles_collision->extents.x > particles_collision->extents.z) {
			size.x = resolutions[particles_collision->heightfield_resolution];
			size.y = int32_t(particles_collision->extents.z / particles_collision->extents.x * size.x);
		} else {
			size.y = resolutions[particles_collision->heightfield_resolution];
			size.x = int32_t(particles_collision->extents.x / particles_collision->extents.z * size.y);
		}

		RD::TextureFormat tf;
		tf.format = RD::DATA_FORMAT_D32_SFLOAT;
		tf.width = size.x;
		tf.height = size.y;
		tf.texture_type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		particles_collision->heightfield_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		Vector<RID> fb_tex;
		fb_tex.push_back(particles_collision->heightfield_texture);
		particles_collision->heightfield_fb = RD::get_singleton()->framebuffer_create(fb_tex);
		particles_collision->heightfield_fb_size = size;
	}

	return particles_collision->heightfield_fb;
}

void RendererStorageRD::particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	if (p_type == particles_collision->type) {
		return;
	}

	if (particles_collision->heightfield_texture.is_valid()) {
		RD::get_singleton()->free(particles_collision->heightfield_texture);
		particles_collision->heightfield_texture = RID();
	}
	particles_collision->type = p_type;
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	particles_collision->cull_mask = p_cull_mask;
}

void RendererStorageRD::particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->radius = p_radius;
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->extents = p_extents;
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_strength = p_strength;
}

void RendererStorageRD::particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_directionality = p_directionality;
}

void RendererStorageRD::particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_attenuation = p_curve;
}

void RendererStorageRD::particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->field_texture = p_texture;
}

void RendererStorageRD::particles_collision_height_field_update(RID p_particles_collision) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	particles_collision->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	ERR_FAIL_INDEX(p_resolution, RS::PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_MAX);

	if (particles_collision->heightfield_resolution == p_resolution) {
		return;
	}

	particles_collision->heightfield_resolution = p_resolution;

	if (particles_collision->heightfield_texture.is_valid()) {
		RD::get_singleton()->free(particles_collision->heightfield_texture);
		particles_collision->heightfield_texture = RID();
	}
}

AABB RendererStorageRD::particles_collision_get_aabb(RID p_particles_collision) const {
	ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, AABB());

	switch (particles_collision->type) {
		case RS::PARTICLES_COLLISION_TYPE_SPHERE_ATTRACT:
		case RS::PARTICLES_COLLISION_TYPE_SPHERE_COLLIDE: {
			AABB aabb;
			aabb.position = -Vector3(1, 1, 1) * particles_collision->radius;
			aabb.size = Vector3(2, 2, 2) * particles_collision->radius;
			return aabb;
		}
		default: {
			AABB aabb;
			aabb.position = -particles_collision->extents;
			aabb.size = particles_collision->extents * 2;
			return aabb;
		}
	}

	return AABB();
}

Vector3 RendererStorageRD::particles_collision_get_extents(RID p_particles_collision) const {
	const ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, Vector3());
	return particles_collision->extents;
}

bool RendererStorageRD::particles_collision_is_heightfield(RID p_particles_collision) const {
	const ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, false);
	return particles_collision->type == RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE;
}

RID RendererStorageRD::particles_collision_instance_create(RID p_collision) {
	ParticlesCollisionInstance pci;
	pci.collision = p_collision;
	return particles_collision_instance_owner.make_rid(pci);
}
void RendererStorageRD::particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) {
	ParticlesCollisionInstance *pci = particles_collision_instance_owner.get_or_null(p_collision_instance);
	ERR_FAIL_COND(!pci);
	pci->transform = p_transform;
}
void RendererStorageRD::particles_collision_instance_set_active(RID p_collision_instance, bool p_active) {
	ParticlesCollisionInstance *pci = particles_collision_instance_owner.get_or_null(p_collision_instance);
	ERR_FAIL_COND(!pci);
	pci->active = p_active;
}

/* FOG VOLUMES */

RID RendererStorageRD::fog_volume_allocate() {
	return fog_volume_owner.allocate_rid();
}
void RendererStorageRD::fog_volume_initialize(RID p_rid) {
	fog_volume_owner.initialize_rid(p_rid, FogVolume());
}

void RendererStorageRD::fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);

	if (p_shape == fog_volume->shape) {
		return;
	}

	fog_volume->shape = p_shape;
	fog_volume->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);

	fog_volume->extents = p_extents;
	fog_volume->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::fog_volume_set_material(RID p_fog_volume, RID p_material) {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND(!fog_volume);
	fog_volume->material = p_material;
}

RID RendererStorageRD::fog_volume_get_material(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, RID());

	return fog_volume->material;
}

RS::FogVolumeShape RendererStorageRD::fog_volume_get_shape(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, RS::FOG_VOLUME_SHAPE_BOX);

	return fog_volume->shape;
}

AABB RendererStorageRD::fog_volume_get_aabb(RID p_fog_volume) const {
	FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, AABB());

	switch (fog_volume->shape) {
		case RS::FOG_VOLUME_SHAPE_ELLIPSOID:
		case RS::FOG_VOLUME_SHAPE_BOX: {
			AABB aabb;
			aabb.position = -fog_volume->extents;
			aabb.size = fog_volume->extents * 2;
			return aabb;
		}
		default: {
			// Need some size otherwise will get culled
			return AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
		}
	}

	return AABB();
}

Vector3 RendererStorageRD::fog_volume_get_extents(RID p_fog_volume) const {
	const FogVolume *fog_volume = fog_volume_owner.get_or_null(p_fog_volume);
	ERR_FAIL_COND_V(!fog_volume, Vector3());
	return fog_volume->extents;
}

/* VISIBILITY NOTIFIER */

RID RendererStorageRD::visibility_notifier_allocate() {
	return visibility_notifier_owner.allocate_rid();
}
void RendererStorageRD::visibility_notifier_initialize(RID p_notifier) {
	visibility_notifier_owner.initialize_rid(p_notifier, VisibilityNotifier());
}
void RendererStorageRD::visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);
	vn->aabb = p_aabb;
	vn->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}
void RendererStorageRD::visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);
	vn->enter_callback = p_enter_callbable;
	vn->exit_callback = p_exit_callable;
}

AABB RendererStorageRD::visibility_notifier_get_aabb(RID p_notifier) const {
	const VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND_V(!vn, AABB());
	return vn->aabb;
}
void RendererStorageRD::visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) {
	VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_notifier);
	ERR_FAIL_COND(!vn);

	if (p_enter) {
		if (!vn->enter_callback.is_null()) {
			if (p_deferred) {
				vn->enter_callback.call_deferred(nullptr, 0);
			} else {
				Variant r;
				Callable::CallError ce;
				vn->enter_callback.call(nullptr, 0, r, ce);
			}
		}
	} else {
		if (!vn->exit_callback.is_null()) {
			if (p_deferred) {
				vn->exit_callback.call_deferred(nullptr, 0);
			} else {
				Variant r;
				Callable::CallError ce;
				vn->exit_callback.call(nullptr, 0, r, ce);
			}
		}
	}
}

/* SKELETON API */

RID RendererStorageRD::skeleton_allocate() {
	return skeleton_owner.allocate_rid();
}
void RendererStorageRD::skeleton_initialize(RID p_rid) {
	skeleton_owner.initialize_rid(p_rid, Skeleton());
}

void RendererStorageRD::_skeleton_make_dirty(Skeleton *skeleton) {
	if (!skeleton->dirty) {
		skeleton->dirty = true;
		skeleton->dirty_list = skeleton_dirty_list;
		skeleton_dirty_list = skeleton;
	}
}

void RendererStorageRD::skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);
	ERR_FAIL_COND(!skeleton);
	ERR_FAIL_COND(p_bones < 0);

	if (skeleton->size == p_bones && skeleton->use_2d == p_2d_skeleton) {
		return;
	}

	skeleton->size = p_bones;
	skeleton->use_2d = p_2d_skeleton;
	skeleton->uniform_set_3d = RID();

	if (skeleton->buffer.is_valid()) {
		RD::get_singleton()->free(skeleton->buffer);
		skeleton->buffer = RID();
		skeleton->data.resize(0);
		skeleton->uniform_set_mi = RID();
	}

	if (skeleton->size) {
		skeleton->data.resize(skeleton->size * (skeleton->use_2d ? 8 : 12));
		skeleton->buffer = RD::get_singleton()->storage_buffer_create(skeleton->data.size() * sizeof(float));
		memset(skeleton->data.ptrw(), 0, skeleton->data.size() * sizeof(float));

		_skeleton_make_dirty(skeleton);

		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.binding = 0;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.ids.push_back(skeleton->buffer);
				uniforms.push_back(u);
			}
			skeleton->uniform_set_mi = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_SKELETON);
		}
	}

	skeleton->dependency.changed_notify(DEPENDENCY_CHANGED_SKELETON_DATA);
}

int RendererStorageRD::skeleton_get_bone_count(RID p_skeleton) const {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);
	ERR_FAIL_COND_V(!skeleton, 0);

	return skeleton->size;
}

void RendererStorageRD::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_COND(!skeleton);
	ERR_FAIL_INDEX(p_bone, skeleton->size);
	ERR_FAIL_COND(skeleton->use_2d);

	float *dataptr = skeleton->data.ptrw() + p_bone * 12;

	dataptr[0] = p_transform.basis.elements[0][0];
	dataptr[1] = p_transform.basis.elements[0][1];
	dataptr[2] = p_transform.basis.elements[0][2];
	dataptr[3] = p_transform.origin.x;
	dataptr[4] = p_transform.basis.elements[1][0];
	dataptr[5] = p_transform.basis.elements[1][1];
	dataptr[6] = p_transform.basis.elements[1][2];
	dataptr[7] = p_transform.origin.y;
	dataptr[8] = p_transform.basis.elements[2][0];
	dataptr[9] = p_transform.basis.elements[2][1];
	dataptr[10] = p_transform.basis.elements[2][2];
	dataptr[11] = p_transform.origin.z;

	_skeleton_make_dirty(skeleton);
}

Transform3D RendererStorageRD::skeleton_bone_get_transform(RID p_skeleton, int p_bone) const {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_COND_V(!skeleton, Transform3D());
	ERR_FAIL_INDEX_V(p_bone, skeleton->size, Transform3D());
	ERR_FAIL_COND_V(skeleton->use_2d, Transform3D());

	const float *dataptr = skeleton->data.ptr() + p_bone * 12;

	Transform3D t;

	t.basis.elements[0][0] = dataptr[0];
	t.basis.elements[0][1] = dataptr[1];
	t.basis.elements[0][2] = dataptr[2];
	t.origin.x = dataptr[3];
	t.basis.elements[1][0] = dataptr[4];
	t.basis.elements[1][1] = dataptr[5];
	t.basis.elements[1][2] = dataptr[6];
	t.origin.y = dataptr[7];
	t.basis.elements[2][0] = dataptr[8];
	t.basis.elements[2][1] = dataptr[9];
	t.basis.elements[2][2] = dataptr[10];
	t.origin.z = dataptr[11];

	return t;
}

void RendererStorageRD::skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_COND(!skeleton);
	ERR_FAIL_INDEX(p_bone, skeleton->size);
	ERR_FAIL_COND(!skeleton->use_2d);

	float *dataptr = skeleton->data.ptrw() + p_bone * 8;

	dataptr[0] = p_transform.elements[0][0];
	dataptr[1] = p_transform.elements[1][0];
	dataptr[2] = 0;
	dataptr[3] = p_transform.elements[2][0];
	dataptr[4] = p_transform.elements[0][1];
	dataptr[5] = p_transform.elements[1][1];
	dataptr[6] = 0;
	dataptr[7] = p_transform.elements[2][1];

	_skeleton_make_dirty(skeleton);
}

Transform2D RendererStorageRD::skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_COND_V(!skeleton, Transform2D());
	ERR_FAIL_INDEX_V(p_bone, skeleton->size, Transform2D());
	ERR_FAIL_COND_V(!skeleton->use_2d, Transform2D());

	const float *dataptr = skeleton->data.ptr() + p_bone * 8;

	Transform2D t;
	t.elements[0][0] = dataptr[0];
	t.elements[1][0] = dataptr[1];
	t.elements[2][0] = dataptr[3];
	t.elements[0][1] = dataptr[4];
	t.elements[1][1] = dataptr[5];
	t.elements[2][1] = dataptr[7];

	return t;
}

void RendererStorageRD::skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	ERR_FAIL_COND(!skeleton->use_2d);

	skeleton->base_transform_2d = p_base_transform;
}

void RendererStorageRD::_update_dirty_skeletons() {
	while (skeleton_dirty_list) {
		Skeleton *skeleton = skeleton_dirty_list;

		if (skeleton->size) {
			RD::get_singleton()->buffer_update(skeleton->buffer, 0, skeleton->data.size() * sizeof(float), skeleton->data.ptr());
		}

		skeleton_dirty_list = skeleton->dirty_list;

		skeleton->dependency.changed_notify(DEPENDENCY_CHANGED_SKELETON_BONES);

		skeleton->version++;

		skeleton->dirty = false;
		skeleton->dirty_list = nullptr;
	}

	skeleton_dirty_list = nullptr;
}

/* LIGHT */

void RendererStorageRD::_light_initialize(RID p_light, RS::LightType p_type) {
	Light light;
	light.type = p_type;

	light.param[RS::LIGHT_PARAM_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_INDIRECT_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_SPECULAR] = 0.5;
	light.param[RS::LIGHT_PARAM_RANGE] = 1.0;
	light.param[RS::LIGHT_PARAM_SIZE] = 0.0;
	light.param[RS::LIGHT_PARAM_ATTENUATION] = 1.0;
	light.param[RS::LIGHT_PARAM_SPOT_ANGLE] = 45;
	light.param[RS::LIGHT_PARAM_SPOT_ATTENUATION] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_MAX_DISTANCE] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET] = 0.1;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET] = 0.3;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET] = 0.6;
	light.param[RS::LIGHT_PARAM_SHADOW_FADE_START] = 0.8;
	light.param[RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_BIAS] = 0.02;
	light.param[RS::LIGHT_PARAM_SHADOW_BLUR] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE] = 20.0;
	light.param[RS::LIGHT_PARAM_SHADOW_VOLUMETRIC_FOG_FADE] = 0.1;
	light.param[RS::LIGHT_PARAM_TRANSMITTANCE_BIAS] = 0.05;

	light_owner.initialize_rid(p_light, light);
}

RID RendererStorageRD::directional_light_allocate() {
	return light_owner.allocate_rid();
}
void RendererStorageRD::directional_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_DIRECTIONAL);
}

RID RendererStorageRD::omni_light_allocate() {
	return light_owner.allocate_rid();
}
void RendererStorageRD::omni_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_OMNI);
}

RID RendererStorageRD::spot_light_allocate() {
	return light_owner.allocate_rid();
}
void RendererStorageRD::spot_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_SPOT);
}

void RendererStorageRD::light_set_color(RID p_light, const Color &p_color) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->color = p_color;
}

void RendererStorageRD::light_set_param(RID p_light, RS::LightParam p_param, float p_value) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_param, RS::LIGHT_PARAM_MAX);

	if (light->param[p_param] == p_value) {
		return;
	}

	switch (p_param) {
		case RS::LIGHT_PARAM_RANGE:
		case RS::LIGHT_PARAM_SPOT_ANGLE:
		case RS::LIGHT_PARAM_SHADOW_MAX_DISTANCE:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS:
		case RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE:
		case RS::LIGHT_PARAM_SHADOW_BIAS: {
			light->version++;
			light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
		} break;
		case RS::LIGHT_PARAM_SIZE: {
			if ((light->param[p_param] > CMP_EPSILON) != (p_value > CMP_EPSILON)) {
				//changing from no size to size and the opposite
				light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR);
			}
		} break;
		default: {
		}
	}

	light->param[p_param] = p_value;
}

void RendererStorageRD::light_set_shadow(RID p_light, bool p_enabled) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);
	light->shadow = p_enabled;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_shadow_color(RID p_light, const Color &p_color) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);
	light->shadow_color = p_color;
}

void RendererStorageRD::light_set_projector(RID p_light, RID p_texture) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	if (light->projector == p_texture) {
		return;
	}

	if (light->type != RS::LIGHT_DIRECTIONAL && light->projector.is_valid()) {
		texture_remove_from_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
	}

	light->projector = p_texture;

	if (light->type != RS::LIGHT_DIRECTIONAL) {
		if (light->projector.is_valid()) {
			texture_add_to_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
		}
		light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR);
	}
}

void RendererStorageRD::light_set_negative(RID p_light, bool p_enable) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->negative = p_enable;
}

void RendererStorageRD::light_set_cull_mask(RID p_light, uint32_t p_mask) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->cull_mask = p_mask;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->reverse_cull = p_enabled;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->bake_mode = p_bake_mode;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->max_sdfgi_cascade = p_cascade;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->omni_shadow_mode = p_mode;

	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

RS::LightOmniShadowMode RendererStorageRD::light_omni_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_OMNI_SHADOW_CUBE);

	return light->omni_shadow_mode;
}

void RendererStorageRD::light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_shadow_mode = p_mode;
	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

void RendererStorageRD::light_directional_set_blend_splits(RID p_light, bool p_enable) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_blend_splits = p_enable;
	light->version++;
	light->dependency.changed_notify(DEPENDENCY_CHANGED_LIGHT);
}

bool RendererStorageRD::light_directional_get_blend_splits(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, false);

	return light->directional_blend_splits;
}

void RendererStorageRD::light_directional_set_sky_only(RID p_light, bool p_sky_only) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_sky_only = p_sky_only;
}

bool RendererStorageRD::light_directional_is_sky_only(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, false);

	return light->directional_sky_only;
}

RS::LightDirectionalShadowMode RendererStorageRD::light_directional_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL);

	return light->directional_shadow_mode;
}

uint32_t RendererStorageRD::light_get_max_sdfgi_cascade(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->max_sdfgi_cascade;
}

RS::LightBakeMode RendererStorageRD::light_get_bake_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_BAKE_DISABLED);

	return light->bake_mode;
}

uint64_t RendererStorageRD::light_get_version(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->version;
}

AABB RendererStorageRD::light_get_aabb(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, AABB());

	switch (light->type) {
		case RS::LIGHT_SPOT: {
			float len = light->param[RS::LIGHT_PARAM_RANGE];
			float size = Math::tan(Math::deg2rad(light->param[RS::LIGHT_PARAM_SPOT_ANGLE])) * len;
			return AABB(Vector3(-size, -size, -len), Vector3(size * 2, size * 2, len));
		};
		case RS::LIGHT_OMNI: {
			float r = light->param[RS::LIGHT_PARAM_RANGE];
			return AABB(-Vector3(r, r, r), Vector3(r, r, r) * 2);
		};
		case RS::LIGHT_DIRECTIONAL: {
			return AABB();
		};
	}

	ERR_FAIL_V(AABB());
}

/* REFLECTION PROBE */

RID RendererStorageRD::reflection_probe_allocate() {
	return reflection_probe_owner.allocate_rid();
}
void RendererStorageRD::reflection_probe_initialize(RID p_reflection_probe) {
	reflection_probe_owner.initialize_rid(p_reflection_probe, ReflectionProbe());
}

void RendererStorageRD::reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->update_mode = p_mode;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->intensity = p_intensity;
}

void RendererStorageRD::reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_mode = p_mode;
}

void RendererStorageRD::reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color = p_color;
}

void RendererStorageRD::reflection_probe_set_ambient_energy(RID p_probe, float p_energy) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color_energy = p_energy;
}

void RendererStorageRD::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->max_distance = p_distance;

	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	if (reflection_probe->extents == p_extents) {
		return;
	}
	reflection_probe->extents = p_extents;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->origin_offset = p_offset;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->interior = p_enable;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->box_projection = p_enable;
}

void RendererStorageRD::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->enable_shadows = p_enable;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->cull_mask = p_layers;
	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void RendererStorageRD::reflection_probe_set_resolution(RID p_probe, int p_resolution) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);
	ERR_FAIL_COND(p_resolution < 32);

	reflection_probe->resolution = p_resolution;
}

void RendererStorageRD::reflection_probe_set_lod_threshold(RID p_probe, float p_ratio) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->lod_threshold = p_ratio;

	reflection_probe->dependency.changed_notify(DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

AABB RendererStorageRD::reflection_probe_get_aabb(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, AABB());

	AABB aabb;
	aabb.position = -reflection_probe->extents;
	aabb.size = reflection_probe->extents * 2.0;

	return aabb;
}

RS::ReflectionProbeUpdateMode RendererStorageRD::reflection_probe_get_update_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_UPDATE_ALWAYS);

	return reflection_probe->update_mode;
}

uint32_t RendererStorageRD::reflection_probe_get_cull_mask(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->cull_mask;
}

Vector3 RendererStorageRD::reflection_probe_get_extents(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->extents;
}

Vector3 RendererStorageRD::reflection_probe_get_origin_offset(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->origin_offset;
}

bool RendererStorageRD::reflection_probe_renders_shadows(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->enable_shadows;
}

float RendererStorageRD::reflection_probe_get_origin_max_distance(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->max_distance;
}

float RendererStorageRD::reflection_probe_get_lod_threshold(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->lod_threshold;
}

int RendererStorageRD::reflection_probe_get_resolution(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->resolution;
}

float RendererStorageRD::reflection_probe_get_intensity(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->intensity;
}

bool RendererStorageRD::reflection_probe_is_interior(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->interior;
}

bool RendererStorageRD::reflection_probe_is_box_projection(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->box_projection;
}

RS::ReflectionProbeAmbientMode RendererStorageRD::reflection_probe_get_ambient_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_AMBIENT_DISABLED);
	return reflection_probe->ambient_mode;
}

Color RendererStorageRD::reflection_probe_get_ambient_color(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Color());

	return reflection_probe->ambient_color;
}
float RendererStorageRD::reflection_probe_get_ambient_color_energy(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->ambient_color_energy;
}

RID RendererStorageRD::decal_allocate() {
	return decal_owner.allocate_rid();
}
void RendererStorageRD::decal_initialize(RID p_decal) {
	decal_owner.initialize_rid(p_decal, Decal());
}

void RendererStorageRD::decal_set_extents(RID p_decal, const Vector3 &p_extents) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->extents = p_extents;
	decal->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	ERR_FAIL_INDEX(p_type, RS::DECAL_TEXTURE_MAX);

	if (decal->textures[p_type] == p_texture) {
		return;
	}

	ERR_FAIL_COND(p_texture.is_valid() && !texture_owner.owns(p_texture));

	if (decal->textures[p_type].is_valid() && texture_owner.owns(decal->textures[p_type])) {
		texture_remove_from_decal_atlas(decal->textures[p_type]);
	}

	decal->textures[p_type] = p_texture;

	if (decal->textures[p_type].is_valid()) {
		texture_add_to_decal_atlas(decal->textures[p_type]);
	}

	decal->dependency.changed_notify(DEPENDENCY_CHANGED_DECAL);
}

void RendererStorageRD::decal_set_emission_energy(RID p_decal, float p_energy) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->emission_energy = p_energy;
}

void RendererStorageRD::decal_set_albedo_mix(RID p_decal, float p_mix) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->albedo_mix = p_mix;
}

void RendererStorageRD::decal_set_modulate(RID p_decal, const Color &p_modulate) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->modulate = p_modulate;
}

void RendererStorageRD::decal_set_cull_mask(RID p_decal, uint32_t p_layers) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->cull_mask = p_layers;
	decal->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

void RendererStorageRD::decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->distance_fade = p_enabled;
	decal->distance_fade_begin = p_begin;
	decal->distance_fade_length = p_length;
}

void RendererStorageRD::decal_set_fade(RID p_decal, float p_above, float p_below) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->upper_fade = p_above;
	decal->lower_fade = p_below;
}

void RendererStorageRD::decal_set_normal_fade(RID p_decal, float p_fade) {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND(!decal);
	decal->normal_fade = p_fade;
}

AABB RendererStorageRD::decal_get_aabb(RID p_decal) const {
	Decal *decal = decal_owner.get_or_null(p_decal);
	ERR_FAIL_COND_V(!decal, AABB());

	return AABB(-decal->extents, decal->extents * 2.0);
}

RID RendererStorageRD::voxel_gi_allocate() {
	return voxel_gi_owner.allocate_rid();
}
void RendererStorageRD::voxel_gi_initialize(RID p_voxel_gi) {
	voxel_gi_owner.initialize_rid(p_voxel_gi, VoxelGI());
}

void RendererStorageRD::voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	if (voxel_gi->octree_buffer.is_valid()) {
		RD::get_singleton()->free(voxel_gi->octree_buffer);
		RD::get_singleton()->free(voxel_gi->data_buffer);
		if (voxel_gi->sdf_texture.is_valid()) {
			RD::get_singleton()->free(voxel_gi->sdf_texture);
		}

		voxel_gi->sdf_texture = RID();
		voxel_gi->octree_buffer = RID();
		voxel_gi->data_buffer = RID();
		voxel_gi->octree_buffer_size = 0;
		voxel_gi->data_buffer_size = 0;
		voxel_gi->cell_count = 0;
	}

	voxel_gi->to_cell_xform = p_to_cell_xform;
	voxel_gi->bounds = p_aabb;
	voxel_gi->octree_size = p_octree_size;
	voxel_gi->level_counts = p_level_counts;

	if (p_octree_cells.size()) {
		ERR_FAIL_COND(p_octree_cells.size() % 32 != 0); //cells size must be a multiple of 32

		uint32_t cell_count = p_octree_cells.size() / 32;

		ERR_FAIL_COND(p_data_cells.size() != (int)cell_count * 16); //see that data size matches

		voxel_gi->cell_count = cell_count;
		voxel_gi->octree_buffer = RD::get_singleton()->storage_buffer_create(p_octree_cells.size(), p_octree_cells);
		voxel_gi->octree_buffer_size = p_octree_cells.size();
		voxel_gi->data_buffer = RD::get_singleton()->storage_buffer_create(p_data_cells.size(), p_data_cells);
		voxel_gi->data_buffer_size = p_data_cells.size();

		if (p_distance_field.size()) {
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = voxel_gi->octree_size.x;
			tf.height = voxel_gi->octree_size.y;
			tf.depth = voxel_gi->octree_size.z;
			tf.texture_type = RD::TEXTURE_TYPE_3D;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
			Vector<Vector<uint8_t>> s;
			s.push_back(p_distance_field);
			voxel_gi->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView(), s);
		}
#if 0
			{
				RD::TextureFormat tf;
				tf.format = RD::DATA_FORMAT_R8_UNORM;
				tf.width = voxel_gi->octree_size.x;
				tf.height = voxel_gi->octree_size.y;
				tf.depth = voxel_gi->octree_size.z;
				tf.type = RD::TEXTURE_TYPE_3D;
				tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UNORM);
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UINT);
				voxel_gi->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			}
			RID shared_tex;
			{
				RD::TextureView tv;
				tv.format_override = RD::DATA_FORMAT_R8_UINT;
				shared_tex = RD::get_singleton()->texture_create_shared(tv, voxel_gi->sdf_texture);
			}
			//update SDF texture
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 1;
				u.ids.push_back(voxel_gi->octree_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.ids.push_back(voxel_gi->data_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 3;
				u.ids.push_back(shared_tex);
				uniforms.push_back(u);
			}

			RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, voxel_gi_sdf_shader_version_shader, 0);

			{
				uint32_t push_constant[4] = { 0, 0, 0, 0 };

				for (int i = 0; i < voxel_gi->level_counts.size() - 1; i++) {
					push_constant[0] += voxel_gi->level_counts[i];
				}
				push_constant[1] = push_constant[0] + voxel_gi->level_counts[voxel_gi->level_counts.size() - 1];

				print_line("offset: " + itos(push_constant[0]));
				print_line("size: " + itos(push_constant[1]));
				//create SDF
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, voxel_gi_sdf_shader_pipeline);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, push_constant, sizeof(uint32_t) * 4);
				RD::get_singleton()->compute_list_dispatch(compute_list, voxel_gi->octree_size.x / 4, voxel_gi->octree_size.y / 4, voxel_gi->octree_size.z / 4);
				RD::get_singleton()->compute_list_end();
			}

			RD::get_singleton()->free(uniform_set);
			RD::get_singleton()->free(shared_tex);
		}
#endif
	}

	voxel_gi->version++;
	voxel_gi->data_version++;

	voxel_gi->dependency.changed_notify(DEPENDENCY_CHANGED_AABB);
}

AABB RendererStorageRD::voxel_gi_get_bounds(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, AABB());

	return voxel_gi->bounds;
}

Vector3i RendererStorageRD::voxel_gi_get_octree_size(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector3i());
	return voxel_gi->octree_size;
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_octree_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->octree_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->octree_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_data_cells(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(voxel_gi->data_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RendererStorageRD::voxel_gi_get_distance_field(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<uint8_t>());

	if (voxel_gi->data_buffer.is_valid()) {
		return RD::get_singleton()->texture_get_data(voxel_gi->sdf_texture, 0);
	}
	return Vector<uint8_t>();
}

Vector<int> RendererStorageRD::voxel_gi_get_level_counts(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Vector<int>());

	return voxel_gi->level_counts;
}

Transform3D RendererStorageRD::voxel_gi_get_to_cell_xform(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, Transform3D());

	return voxel_gi->to_cell_xform;
}

void RendererStorageRD::voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->dynamic_range = p_range;
	voxel_gi->version++;
}

float RendererStorageRD::voxel_gi_get_dynamic_range(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);

	return voxel_gi->dynamic_range;
}

void RendererStorageRD::voxel_gi_set_propagation(RID p_voxel_gi, float p_range) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->propagation = p_range;
	voxel_gi->version++;
}

float RendererStorageRD::voxel_gi_get_propagation(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->propagation;
}

void RendererStorageRD::voxel_gi_set_energy(RID p_voxel_gi, float p_energy) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->energy = p_energy;
}

float RendererStorageRD::voxel_gi_get_energy(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->energy;
}

void RendererStorageRD::voxel_gi_set_bias(RID p_voxel_gi, float p_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->bias = p_bias;
}

float RendererStorageRD::voxel_gi_get_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->bias;
}

void RendererStorageRD::voxel_gi_set_normal_bias(RID p_voxel_gi, float p_normal_bias) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->normal_bias = p_normal_bias;
}

float RendererStorageRD::voxel_gi_get_normal_bias(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->normal_bias;
}

void RendererStorageRD::voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->anisotropy_strength = p_strength;
}

float RendererStorageRD::voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->anisotropy_strength;
}

void RendererStorageRD::voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->interior = p_enable;
}

void RendererStorageRD::voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND(!voxel_gi);

	voxel_gi->use_two_bounces = p_enable;
	voxel_gi->version++;
}

bool RendererStorageRD::voxel_gi_is_using_two_bounces(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, false);
	return voxel_gi->use_two_bounces;
}

bool RendererStorageRD::voxel_gi_is_interior(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->interior;
}

uint32_t RendererStorageRD::voxel_gi_get_version(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->version;
}

uint32_t RendererStorageRD::voxel_gi_get_data_version(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, 0);
	return voxel_gi->data_version;
}

RID RendererStorageRD::voxel_gi_get_octree_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());
	return voxel_gi->octree_buffer;
}

RID RendererStorageRD::voxel_gi_get_data_buffer(RID p_voxel_gi) const {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());
	return voxel_gi->data_buffer;
}

RID RendererStorageRD::voxel_gi_get_sdf_texture(RID p_voxel_gi) {
	VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_voxel_gi);
	ERR_FAIL_COND_V(!voxel_gi, RID());

	return voxel_gi->sdf_texture;
}

/* LIGHTMAP API */

RID RendererStorageRD::lightmap_allocate() {
	return lightmap_owner.allocate_rid();
}

void RendererStorageRD::lightmap_initialize(RID p_lightmap) {
	lightmap_owner.initialize_rid(p_lightmap, Lightmap());
}

void RendererStorageRD::lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	lightmap_array_version++;

	//erase lightmap users
	if (lm->light_texture.is_valid()) {
		Texture *t = texture_owner.get_or_null(lm->light_texture);
		if (t) {
			t->lightmap_users.erase(p_lightmap);
		}
	}

	Texture *t = texture_owner.get_or_null(p_light);
	lm->light_texture = p_light;
	lm->uses_spherical_harmonics = p_uses_spherical_haromics;

	RID default_2d_array = default_rd_textures[DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE];
	if (!t) {
		if (using_lightmap_array) {
			if (lm->array_index >= 0) {
				lightmap_textures.write[lm->array_index] = default_2d_array;
				lm->array_index = -1;
			}
		}

		return;
	}

	t->lightmap_users.insert(p_lightmap);

	if (using_lightmap_array) {
		if (lm->array_index < 0) {
			//not in array, try to put in array
			for (int i = 0; i < lightmap_textures.size(); i++) {
				if (lightmap_textures[i] == default_2d_array) {
					lm->array_index = i;
					break;
				}
			}
		}
		ERR_FAIL_COND_MSG(lm->array_index < 0, "Maximum amount of lightmaps in use (" + itos(lightmap_textures.size()) + ") has been exceeded, lightmap will nod display properly.");

		lightmap_textures.write[lm->array_index] = t->rd_texture;
	}
}

void RendererStorageRD::lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->bounds = p_bounds;
}

void RendererStorageRD::lightmap_set_probe_interior(RID p_lightmap, bool p_interior) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->interior = p_interior;
}

void RendererStorageRD::lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	if (p_points.size()) {
		ERR_FAIL_COND(p_points.size() * 9 != p_point_sh.size());
		ERR_FAIL_COND((p_tetrahedra.size() % 4) != 0);
		ERR_FAIL_COND((p_bsp_tree.size() % 6) != 0);
	}

	lm->points = p_points;
	lm->bsp_tree = p_bsp_tree;
	lm->point_sh = p_point_sh;
	lm->tetrahedra = p_tetrahedra;
}

PackedVector3Array RendererStorageRD::lightmap_get_probe_capture_points(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedVector3Array());

	return lm->points;
}

PackedColorArray RendererStorageRD::lightmap_get_probe_capture_sh(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedColorArray());
	return lm->point_sh;
}

PackedInt32Array RendererStorageRD::lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->tetrahedra;
}

PackedInt32Array RendererStorageRD::lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->bsp_tree;
}

void RendererStorageRD::lightmap_set_probe_capture_update_speed(float p_speed) {
	lightmap_probe_capture_update_speed = p_speed;
}

void RendererStorageRD::lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	for (int i = 0; i < 9; i++) {
		r_sh[i] = Color(0, 0, 0, 0);
	}

	if (!lm->points.size() || !lm->bsp_tree.size() || !lm->tetrahedra.size()) {
		return;
	}

	static_assert(sizeof(Lightmap::BSP) == 24);

	const Lightmap::BSP *bsp = (const Lightmap::BSP *)lm->bsp_tree.ptr();
	int32_t node = 0;
	while (node >= 0) {
		if (Plane(bsp[node].plane[0], bsp[node].plane[1], bsp[node].plane[2], bsp[node].plane[3]).is_point_over(p_point)) {
#ifdef DEBUG_ENABLED
			ERR_FAIL_COND(bsp[node].over >= 0 && bsp[node].over < node);
#endif

			node = bsp[node].over;
		} else {
#ifdef DEBUG_ENABLED
			ERR_FAIL_COND(bsp[node].under >= 0 && bsp[node].under < node);
#endif
			node = bsp[node].under;
		}
	}

	if (node == Lightmap::BSP::EMPTY_LEAF) {
		return; //nothing could be done
	}

	node = ABS(node) - 1;

	uint32_t *tetrahedron = (uint32_t *)&lm->tetrahedra[node * 4];
	Vector3 points[4] = { lm->points[tetrahedron[0]], lm->points[tetrahedron[1]], lm->points[tetrahedron[2]], lm->points[tetrahedron[3]] };
	const Color *sh_colors[4]{ &lm->point_sh[tetrahedron[0] * 9], &lm->point_sh[tetrahedron[1] * 9], &lm->point_sh[tetrahedron[2] * 9], &lm->point_sh[tetrahedron[3] * 9] };
	Color barycentric = Geometry3D::tetrahedron_get_barycentric_coords(points[0], points[1], points[2], points[3], p_point);

	for (int i = 0; i < 4; i++) {
		float c = CLAMP(barycentric[i], 0.0, 1.0);
		for (int j = 0; j < 9; j++) {
			r_sh[j] += sh_colors[i][j] * c;
		}
	}
}

bool RendererStorageRD::lightmap_is_interior(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, false);
	return lm->interior;
}

AABB RendererStorageRD::lightmap_get_aabb(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, AABB());
	return lm->bounds;
}

/* RENDER TARGET API */

void RendererStorageRD::_clear_render_target(RenderTarget *rt) {
	//free in reverse dependency order
	if (rt->framebuffer.is_valid()) {
		RD::get_singleton()->free(rt->framebuffer);
		rt->framebuffer_uniform_set = RID(); //chain deleted
	}

	if (rt->color.is_valid()) {
		RD::get_singleton()->free(rt->color);
	}

	if (rt->backbuffer.is_valid()) {
		RD::get_singleton()->free(rt->backbuffer);
		rt->backbuffer = RID();
		for (int i = 0; i < rt->backbuffer_mipmaps.size(); i++) {
			//just erase copies, since the rest are erased by dependency
			RD::get_singleton()->free(rt->backbuffer_mipmaps[i].mipmap_copy);
		}
		rt->backbuffer_mipmaps.clear();
		rt->backbuffer_uniform_set = RID(); //chain deleted
	}

	_render_target_clear_sdf(rt);

	rt->framebuffer = RID();
	rt->color = RID();
}

void RendererStorageRD::_update_render_target(RenderTarget *rt) {
	if (rt->texture.is_null()) {
		//create a placeholder until updated
		rt->texture = texture_allocate();
		texture_2d_placeholder_initialize(rt->texture);
		Texture *tex = texture_owner.get_or_null(rt->texture);
		tex->is_render_target = true;
	}

	_clear_render_target(rt);

	if (rt->size.width == 0 || rt->size.height == 0) {
		return;
	}
	//until we implement support for HDR monitors (and render target is attached to screen), this is enough.
	rt->color_format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	rt->color_format_srgb = RD::DATA_FORMAT_R8G8B8A8_SRGB;
	rt->image_format = rt->flags[RENDER_TARGET_TRANSPARENT] ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8;

	RD::TextureFormat rd_format;
	RD::TextureView rd_view;
	{ //attempt register
		rd_format.format = rt->color_format;
		rd_format.width = rt->size.width;
		rd_format.height = rt->size.height;
		rd_format.depth = 1;
		rd_format.array_layers = rt->view_count; // for stereo we create two (or more) layers, need to see if we can make fallback work like this too if we don't have multiview
		rd_format.mipmaps = 1;
		if (rd_format.array_layers > 1) { // why are we not using rt->texture_type ??
			rd_format.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;
		} else {
			rd_format.texture_type = RD::TEXTURE_TYPE_2D;
		}
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		rd_format.shareable_formats.push_back(rt->color_format);
		rd_format.shareable_formats.push_back(rt->color_format_srgb);
	}

	rt->color = RD::get_singleton()->texture_create(rd_format, rd_view);
	ERR_FAIL_COND(rt->color.is_null());

	Vector<RID> fb_textures;
	fb_textures.push_back(rt->color);
	rt->framebuffer = RD::get_singleton()->framebuffer_create(fb_textures, RenderingDevice::INVALID_ID, rt->view_count);
	if (rt->framebuffer.is_null()) {
		_clear_render_target(rt);
		ERR_FAIL_COND(rt->framebuffer.is_null());
	}

	{ //update texture

		Texture *tex = texture_owner.get_or_null(rt->texture);

		//free existing textures
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture)) {
			RD::get_singleton()->free(tex->rd_texture);
		}
		if (RD::get_singleton()->texture_is_valid(tex->rd_texture_srgb)) {
			RD::get_singleton()->free(tex->rd_texture_srgb);
		}

		tex->rd_texture = RID();
		tex->rd_texture_srgb = RID();

		//create shared textures to the color buffer,
		//so transparent can be supported
		RD::TextureView view;
		view.format_override = rt->color_format;
		if (!rt->flags[RENDER_TARGET_TRANSPARENT]) {
			view.swizzle_a = RD::TEXTURE_SWIZZLE_ONE;
		}
		tex->rd_texture = RD::get_singleton()->texture_create_shared(view, rt->color);
		if (rt->color_format_srgb != RD::DATA_FORMAT_MAX) {
			view.format_override = rt->color_format_srgb;
			tex->rd_texture_srgb = RD::get_singleton()->texture_create_shared(view, rt->color);
		}
		tex->rd_view = view;
		tex->width = rt->size.width;
		tex->height = rt->size.height;
		tex->width_2d = rt->size.width;
		tex->height_2d = rt->size.height;
		tex->rd_format = rt->color_format;
		tex->rd_format_srgb = rt->color_format_srgb;
		tex->format = rt->image_format;

		Vector<RID> proxies = tex->proxies; //make a copy, since update may change it
		for (int i = 0; i < proxies.size(); i++) {
			texture_proxy_update(proxies[i], rt->texture);
		}
	}
}

void RendererStorageRD::_create_render_target_backbuffer(RenderTarget *rt) {
	ERR_FAIL_COND(rt->backbuffer.is_valid());

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(rt->size.width, rt->size.height, Image::FORMAT_RGBA8);
	RD::TextureFormat tf;
	tf.format = rt->color_format;
	tf.width = rt->size.width;
	tf.height = rt->size.height;
	tf.texture_type = RD::TEXTURE_TYPE_2D;
	tf.usage_bits = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tf.mipmaps = mipmaps_required;

	rt->backbuffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
	rt->backbuffer_mipmap0 = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->backbuffer, 0, 0);

	{
		Vector<RID> fb_tex;
		fb_tex.push_back(rt->backbuffer_mipmap0);
		rt->backbuffer_fb = RD::get_singleton()->framebuffer_create(fb_tex);
	}

	if (rt->framebuffer_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rt->framebuffer_uniform_set)) {
		//the new one will require the backbuffer.
		RD::get_singleton()->free(rt->framebuffer_uniform_set);
		rt->framebuffer_uniform_set = RID();
	}
	//create mipmaps
	for (uint32_t i = 1; i < mipmaps_required; i++) {
		RenderTarget::BackbufferMipmap mm;
		{
			mm.mipmap = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->backbuffer, 0, i);
		}

		{
			Size2 mm_size = Image::get_image_mipmap_size(tf.width, tf.height, Image::FORMAT_RGBA8, i);

			RD::TextureFormat mmtf = tf;
			mmtf.width = mm_size.width;
			mmtf.height = mm_size.height;
			mmtf.mipmaps = 1;

			mm.mipmap_copy = RD::get_singleton()->texture_create(mmtf, RD::TextureView());
		}

		rt->backbuffer_mipmaps.push_back(mm);
	}
}

RID RendererStorageRD::render_target_create() {
	RenderTarget render_target;

	render_target.was_used = false;
	render_target.clear_requested = false;

	for (int i = 0; i < RENDER_TARGET_FLAG_MAX; i++) {
		render_target.flags[i] = false;
	}
	_update_render_target(&render_target);
	return render_target_owner.make_rid(render_target);
}

void RendererStorageRD::render_target_set_position(RID p_render_target, int p_x, int p_y) {
	//unused for this render target
}

void RendererStorageRD::render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (rt->size.x != p_width || rt->size.y != p_height || rt->view_count != p_view_count) {
		rt->size.x = p_width;
		rt->size.y = p_height;
		rt->view_count = p_view_count;
		_update_render_target(rt);
	}
}

RID RendererStorageRD::render_target_get_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->texture;
}

void RendererStorageRD::render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {
}

void RendererStorageRD::render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->flags[p_flag] = p_value;
	_update_render_target(rt);
}

bool RendererStorageRD::render_target_was_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->was_used;
}

void RendererStorageRD::render_target_set_as_unused(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->was_used = false;
}

Size2 RendererStorageRD::render_target_get_size(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Size2());

	return rt->size;
}

RID RendererStorageRD::render_target_get_rd_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->framebuffer;
}

RID RendererStorageRD::render_target_get_rd_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->color;
}

RID RendererStorageRD::render_target_get_rd_backbuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer;
}

RID RendererStorageRD::render_target_get_rd_backbuffer_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	return rt->backbuffer_fb;
}

void RendererStorageRD::render_target_request_clear(RID p_render_target, const Color &p_clear_color) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = true;
	rt->clear_color = p_clear_color;
}

bool RendererStorageRD::render_target_is_clear_requested(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->clear_requested;
}

Color RendererStorageRD::render_target_get_clear_request_color(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Color());
	return rt->clear_color;
}

void RendererStorageRD::render_target_disable_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = false;
}

void RendererStorageRD::render_target_do_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->clear_requested) {
		return;
	}
	Vector<Color> clear_colors;
	clear_colors.push_back(rt->clear_color);
	RD::get_singleton()->draw_list_begin(rt->framebuffer, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, clear_colors);
	RD::get_singleton()->draw_list_end();
	rt->clear_requested = false;
}

void RendererStorageRD::render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (rt->sdf_oversize == p_size && rt->sdf_scale == p_scale) {
		return;
	}

	rt->sdf_oversize = p_size;
	rt->sdf_scale = p_scale;

	_render_target_clear_sdf(rt);
}

Rect2i RendererStorageRD::_render_target_get_sdf_rect(const RenderTarget *rt) const {
	Size2i margin;
	int scale;
	switch (rt->sdf_oversize) {
		case RS::VIEWPORT_SDF_OVERSIZE_100_PERCENT: {
			scale = 100;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_120_PERCENT: {
			scale = 120;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_150_PERCENT: {
			scale = 150;
		} break;
		case RS::VIEWPORT_SDF_OVERSIZE_200_PERCENT: {
			scale = 200;
		} break;
		default: {
		}
	}

	margin = (rt->size * scale / 100) - rt->size;

	Rect2i r(Vector2i(), rt->size);
	r.position -= margin;
	r.size += margin * 2;

	return r;
}

Rect2i RendererStorageRD::render_target_get_sdf_rect(RID p_render_target) const {
	const RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Rect2i());

	return _render_target_get_sdf_rect(rt);
}

void RendererStorageRD::render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->sdf_enabled = p_enabled;
}

bool RendererStorageRD::render_target_is_sdf_enabled(RID p_render_target) const {
	const RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);

	return rt->sdf_enabled;
}

RID RendererStorageRD::render_target_get_sdf_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	if (rt->sdf_buffer_read.is_null()) {
		// no texture, create a dummy one for the 2D uniform set
		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		memset(pv.ptrw(), 0, 16 * 4);
		Vector<Vector<uint8_t>> vpv;

		rt->sdf_buffer_read = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
	}

	return rt->sdf_buffer_read;
}

void RendererStorageRD::_render_target_allocate_sdf(RenderTarget *rt) {
	ERR_FAIL_COND(rt->sdf_buffer_write_fb.is_valid());
	if (rt->sdf_buffer_read.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_read);
		rt->sdf_buffer_read = RID();
	}

	Size2i size = _render_target_get_sdf_rect(rt).size;

	RD::TextureFormat tformat;
	tformat.format = RD::DATA_FORMAT_R8_UNORM;
	tformat.width = size.width;
	tformat.height = size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
	tformat.texture_type = RD::TEXTURE_TYPE_2D;

	rt->sdf_buffer_write = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	{
		Vector<RID> write_fb;
		write_fb.push_back(rt->sdf_buffer_write);
		rt->sdf_buffer_write_fb = RD::get_singleton()->framebuffer_create(write_fb);
	}

	int scale;
	switch (rt->sdf_scale) {
		case RS::VIEWPORT_SDF_SCALE_100_PERCENT: {
			scale = 100;
		} break;
		case RS::VIEWPORT_SDF_SCALE_50_PERCENT: {
			scale = 50;
		} break;
		case RS::VIEWPORT_SDF_SCALE_25_PERCENT: {
			scale = 25;
		} break;
		default: {
			scale = 100;
		} break;
	}

	rt->process_size = size * scale / 100;
	rt->process_size.x = MAX(rt->process_size.x, 1);
	rt->process_size.y = MAX(rt->process_size.y, 1);

	tformat.format = RD::DATA_FORMAT_R16G16_SINT;
	tformat.width = rt->process_size.width;
	tformat.height = rt->process_size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

	rt->sdf_buffer_process[0] = RD::get_singleton()->texture_create(tformat, RD::TextureView());
	rt->sdf_buffer_process[1] = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	tformat.format = RD::DATA_FORMAT_R16_SNORM;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;

	rt->sdf_buffer_read = RD::get_singleton()->texture_create(tformat, RD::TextureView());

	{
		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 1;
			u.ids.push_back(rt->sdf_buffer_write);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 2;
			u.ids.push_back(rt->sdf_buffer_read);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 3;
			u.ids.push_back(rt->sdf_buffer_process[0]);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
			u.binding = 4;
			u.ids.push_back(rt->sdf_buffer_process[1]);
			uniforms.push_back(u);
		}

		rt->sdf_buffer_process_uniform_sets[0] = RD::get_singleton()->uniform_set_create(uniforms, rt_sdf.shader.version_get_shader(rt_sdf.shader_version, 0), 0);
		SWAP(uniforms.write[2].ids.write[0], uniforms.write[3].ids.write[0]);
		rt->sdf_buffer_process_uniform_sets[1] = RD::get_singleton()->uniform_set_create(uniforms, rt_sdf.shader.version_get_shader(rt_sdf.shader_version, 0), 0);
	}
}

void RendererStorageRD::_render_target_clear_sdf(RenderTarget *rt) {
	if (rt->sdf_buffer_read.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_read);
		rt->sdf_buffer_read = RID();
	}
	if (rt->sdf_buffer_write_fb.is_valid()) {
		RD::get_singleton()->free(rt->sdf_buffer_write);
		RD::get_singleton()->free(rt->sdf_buffer_process[0]);
		RD::get_singleton()->free(rt->sdf_buffer_process[1]);
		rt->sdf_buffer_write = RID();
		rt->sdf_buffer_write_fb = RID();
		rt->sdf_buffer_process[0] = RID();
		rt->sdf_buffer_process[1] = RID();
		rt->sdf_buffer_process_uniform_sets[0] = RID();
		rt->sdf_buffer_process_uniform_sets[1] = RID();
	}
}

RID RendererStorageRD::render_target_get_sdf_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (rt->sdf_buffer_write_fb.is_null()) {
		_render_target_allocate_sdf(rt);
	}

	return rt->sdf_buffer_write_fb;
}
void RendererStorageRD::render_target_sdf_process(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	ERR_FAIL_COND(rt->sdf_buffer_write_fb.is_null());

	RenderTargetSDF::PushConstant push_constant;

	Rect2i r = _render_target_get_sdf_rect(rt);

	push_constant.size[0] = r.size.width;
	push_constant.size[1] = r.size.height;
	push_constant.stride = 0;
	push_constant.shift = 0;
	push_constant.base_size[0] = r.size.width;
	push_constant.base_size[1] = r.size.height;

	bool shrink = false;

	switch (rt->sdf_scale) {
		case RS::VIEWPORT_SDF_SCALE_50_PERCENT: {
			push_constant.size[0] >>= 1;
			push_constant.size[1] >>= 1;
			push_constant.shift = 1;
			shrink = true;
		} break;
		case RS::VIEWPORT_SDF_SCALE_25_PERCENT: {
			push_constant.size[0] >>= 2;
			push_constant.size[1] >>= 2;
			push_constant.shift = 2;
			shrink = true;
		} break;
		default: {
		};
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	/* Load */

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[shrink ? RenderTargetSDF::SHADER_LOAD_SHRINK : RenderTargetSDF::SHADER_LOAD]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[1], 0); //fill [0]
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);

	/* Process */

	int stride = nearest_power_of_2_templated(MAX(push_constant.size[0], push_constant.size[1]) / 2);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[RenderTargetSDF::SHADER_PROCESS]);

	RD::get_singleton()->compute_list_add_barrier(compute_list);
	bool swap = false;

	//jumpflood
	while (stride > 0) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[swap ? 1 : 0], 0);
		push_constant.stride = stride;
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);
		stride /= 2;
		swap = !swap;
		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}

	/* Store */

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, rt_sdf.pipelines[shrink ? RenderTargetSDF::SHADER_STORE_SHRINK : RenderTargetSDF::SHADER_STORE]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, rt->sdf_buffer_process_uniform_sets[swap ? 1 : 0], 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(RenderTargetSDF::PushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, push_constant.size[0], push_constant.size[1], 1);

	RD::get_singleton()->compute_list_end();
}

void RendererStorageRD::render_target_copy_to_back_buffer(RID p_render_target, const Rect2i &p_region, bool p_gen_mipmaps) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//single texture copy for backbuffer
	//RD::get_singleton()->texture_copy(rt->color, rt->backbuffer_mipmap0, Vector3(region.position.x, region.position.y, 0), Vector3(region.position.x, region.position.y, 0), Vector3(region.size.x, region.size.y, 1), 0, 0, 0, 0, true);
	effects->copy_to_rect(rt->color, rt->backbuffer_mipmap0, region, false, false, false, true, true);

	if (!p_gen_mipmaps) {
		return;
	}

	//then mipmap blur
	RID prev_texture = rt->color; //use color, not backbuffer, as bb has mipmaps.

	for (int i = 0; i < rt->backbuffer_mipmaps.size(); i++) {
		region.position.x >>= 1;
		region.position.y >>= 1;
		region.size.x = MAX(1, region.size.x >> 1);
		region.size.y = MAX(1, region.size.y >> 1);

		const RenderTarget::BackbufferMipmap &mm = rt->backbuffer_mipmaps[i];
		effects->gaussian_blur(prev_texture, mm.mipmap, mm.mipmap_copy, region, true);
		prev_texture = mm.mipmap;
	}
}

void RendererStorageRD::render_target_clear_back_buffer(RID p_render_target, const Rect2i &p_region, const Color &p_color) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//single texture copy for backbuffer
	effects->set_color(rt->backbuffer_mipmap0, p_color, region, true);
}

void RendererStorageRD::render_target_gen_back_buffer_mipmaps(RID p_render_target, const Rect2i &p_region) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).intersection(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//then mipmap blur
	RID prev_texture = rt->backbuffer_mipmap0;

	for (int i = 0; i < rt->backbuffer_mipmaps.size(); i++) {
		region.position.x >>= 1;
		region.position.y >>= 1;
		region.size.x = MAX(1, region.size.x >> 1);
		region.size.y = MAX(1, region.size.y >> 1);

		const RenderTarget::BackbufferMipmap &mm = rt->backbuffer_mipmaps[i];
		effects->gaussian_blur(prev_texture, mm.mipmap, mm.mipmap_copy, region, true);
		prev_texture = mm.mipmap;
	}
}

RID RendererStorageRD::render_target_get_framebuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->framebuffer_uniform_set;
}
RID RendererStorageRD::render_target_get_backbuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer_uniform_set;
}

void RendererStorageRD::render_target_set_framebuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->framebuffer_uniform_set = p_uniform_set;
}
void RendererStorageRD::render_target_set_backbuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->backbuffer_uniform_set = p_uniform_set;
}

void RendererStorageRD::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
	if (mesh_owner.owns(p_base)) {
		Mesh *mesh = mesh_owner.get_or_null(p_base);
		p_instance->update_dependency(&mesh->dependency);
	} else if (multimesh_owner.owns(p_base)) {
		MultiMesh *multimesh = multimesh_owner.get_or_null(p_base);
		p_instance->update_dependency(&multimesh->dependency);
		if (multimesh->mesh.is_valid()) {
			base_update_dependency(multimesh->mesh, p_instance);
		}
	} else if (reflection_probe_owner.owns(p_base)) {
		ReflectionProbe *rp = reflection_probe_owner.get_or_null(p_base);
		p_instance->update_dependency(&rp->dependency);
	} else if (decal_owner.owns(p_base)) {
		Decal *decal = decal_owner.get_or_null(p_base);
		p_instance->update_dependency(&decal->dependency);
	} else if (voxel_gi_owner.owns(p_base)) {
		VoxelGI *gip = voxel_gi_owner.get_or_null(p_base);
		p_instance->update_dependency(&gip->dependency);
	} else if (lightmap_owner.owns(p_base)) {
		Lightmap *lm = lightmap_owner.get_or_null(p_base);
		p_instance->update_dependency(&lm->dependency);
	} else if (light_owner.owns(p_base)) {
		Light *l = light_owner.get_or_null(p_base);
		p_instance->update_dependency(&l->dependency);
	} else if (particles_owner.owns(p_base)) {
		Particles *p = particles_owner.get_or_null(p_base);
		p_instance->update_dependency(&p->dependency);
	} else if (particles_collision_owner.owns(p_base)) {
		ParticlesCollision *pc = particles_collision_owner.get_or_null(p_base);
		p_instance->update_dependency(&pc->dependency);
	} else if (fog_volume_owner.owns(p_base)) {
		FogVolume *fv = fog_volume_owner.get_or_null(p_base);
		p_instance->update_dependency(&fv->dependency);
	} else if (visibility_notifier_owner.owns(p_base)) {
		VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_base);
		p_instance->update_dependency(&vn->dependency);
	}
}

void RendererStorageRD::skeleton_update_dependency(RID p_skeleton, DependencyTracker *p_instance) {
	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);
	ERR_FAIL_COND(!skeleton);

	p_instance->update_dependency(&skeleton->dependency);
}

RS::InstanceType RendererStorageRD::get_base_type(RID p_rid) const {
	if (mesh_owner.owns(p_rid)) {
		return RS::INSTANCE_MESH;
	}
	if (multimesh_owner.owns(p_rid)) {
		return RS::INSTANCE_MULTIMESH;
	}
	if (reflection_probe_owner.owns(p_rid)) {
		return RS::INSTANCE_REFLECTION_PROBE;
	}
	if (decal_owner.owns(p_rid)) {
		return RS::INSTANCE_DECAL;
	}
	if (voxel_gi_owner.owns(p_rid)) {
		return RS::INSTANCE_VOXEL_GI;
	}
	if (light_owner.owns(p_rid)) {
		return RS::INSTANCE_LIGHT;
	}
	if (lightmap_owner.owns(p_rid)) {
		return RS::INSTANCE_LIGHTMAP;
	}
	if (particles_owner.owns(p_rid)) {
		return RS::INSTANCE_PARTICLES;
	}
	if (particles_collision_owner.owns(p_rid)) {
		return RS::INSTANCE_PARTICLES_COLLISION;
	}
	if (fog_volume_owner.owns(p_rid)) {
		return RS::INSTANCE_FOG_VOLUME;
	}
	if (visibility_notifier_owner.owns(p_rid)) {
		return RS::INSTANCE_VISIBLITY_NOTIFIER;
	}

	return RS::INSTANCE_NONE;
}

void RendererStorageRD::texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
	if (!decal_atlas.textures.has(p_texture)) {
		DecalAtlas::Texture t;
		t.users = 1;
		t.panorama_to_dp_users = p_panorama_to_dp ? 1 : 0;
		decal_atlas.textures[p_texture] = t;
		decal_atlas.dirty = true;
	} else {
		DecalAtlas::Texture *t = decal_atlas.textures.getptr(p_texture);
		t->users++;
		if (p_panorama_to_dp) {
			t->panorama_to_dp_users++;
		}
	}
}

void RendererStorageRD::texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
	DecalAtlas::Texture *t = decal_atlas.textures.getptr(p_texture);
	ERR_FAIL_COND(!t);
	t->users--;
	if (p_panorama_to_dp) {
		ERR_FAIL_COND(t->panorama_to_dp_users == 0);
		t->panorama_to_dp_users--;
	}
	if (t->users == 0) {
		decal_atlas.textures.erase(p_texture);
		//do not mark it dirty, there is no need to since it remains working
	}
}

RID RendererStorageRD::decal_atlas_get_texture() const {
	return decal_atlas.texture;
}

RID RendererStorageRD::decal_atlas_get_texture_srgb() const {
	return decal_atlas.texture_srgb;
}

void RendererStorageRD::_update_decal_atlas() {
	if (!decal_atlas.dirty) {
		return; //nothing to do
	}

	decal_atlas.dirty = false;

	if (decal_atlas.texture.is_valid()) {
		RD::get_singleton()->free(decal_atlas.texture);
		decal_atlas.texture = RID();
		decal_atlas.texture_srgb = RID();
		decal_atlas.texture_mipmaps.clear();
	}

	int border = 1 << decal_atlas.mipmaps;

	if (decal_atlas.textures.size()) {
		//generate atlas
		Vector<DecalAtlas::SortItem> itemsv;
		itemsv.resize(decal_atlas.textures.size());
		int base_size = 8;
		const RID *K = nullptr;

		int idx = 0;
		while ((K = decal_atlas.textures.next(K))) {
			DecalAtlas::SortItem &si = itemsv.write[idx];

			Texture *src_tex = texture_owner.get_or_null(*K);

			si.size.width = (src_tex->width / border) + 1;
			si.size.height = (src_tex->height / border) + 1;
			si.pixel_size = Size2i(src_tex->width, src_tex->height);

			if (base_size < si.size.width) {
				base_size = nearest_power_of_2_templated(si.size.width);
			}

			si.texture = *K;
			idx++;
		}

		//sort items by size
		itemsv.sort();

		//attempt to create atlas
		int item_count = itemsv.size();
		DecalAtlas::SortItem *items = itemsv.ptrw();

		int atlas_height = 0;

		while (true) {
			Vector<int> v_offsetsv;
			v_offsetsv.resize(base_size);

			int *v_offsets = v_offsetsv.ptrw();
			memset(v_offsets, 0, sizeof(int) * base_size);

			int max_height = 0;

			for (int i = 0; i < item_count; i++) {
				//best fit
				DecalAtlas::SortItem &si = items[i];
				int best_idx = -1;
				int best_height = 0x7FFFFFFF;
				for (int j = 0; j <= base_size - si.size.width; j++) {
					int height = 0;
					for (int k = 0; k < si.size.width; k++) {
						int h = v_offsets[k + j];
						if (h > height) {
							height = h;
							if (height > best_height) {
								break; //already bad
							}
						}
					}

					if (height < best_height) {
						best_height = height;
						best_idx = j;
					}
				}

				//update
				for (int k = 0; k < si.size.width; k++) {
					v_offsets[k + best_idx] = best_height + si.size.height;
				}

				si.pos.x = best_idx;
				si.pos.y = best_height;

				if (si.pos.y + si.size.height > max_height) {
					max_height = si.pos.y + si.size.height;
				}
			}

			if (max_height <= base_size * 2) {
				atlas_height = max_height;
				break; //good ratio, break;
			}

			base_size *= 2;
		}

		decal_atlas.size.width = base_size * border;
		decal_atlas.size.height = nearest_power_of_2_templated(atlas_height * border);

		for (int i = 0; i < item_count; i++) {
			DecalAtlas::Texture *t = decal_atlas.textures.getptr(items[i].texture);
			t->uv_rect.position = items[i].pos * border + Vector2i(border / 2, border / 2);
			t->uv_rect.size = items[i].pixel_size;

			t->uv_rect.position /= Size2(decal_atlas.size);
			t->uv_rect.size /= Size2(decal_atlas.size);
		}
	} else {
		//use border as size, so it at least has enough mipmaps
		decal_atlas.size.width = border;
		decal_atlas.size.height = border;
	}

	//blit textures

	RD::TextureFormat tformat;
	tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
	tformat.width = decal_atlas.size.width;
	tformat.height = decal_atlas.size.height;
	tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tformat.texture_type = RD::TEXTURE_TYPE_2D;
	tformat.mipmaps = decal_atlas.mipmaps;
	tformat.shareable_formats.push_back(RD::DATA_FORMAT_R8G8B8A8_UNORM);
	tformat.shareable_formats.push_back(RD::DATA_FORMAT_R8G8B8A8_SRGB);

	decal_atlas.texture = RD::get_singleton()->texture_create(tformat, RD::TextureView());
	RD::get_singleton()->texture_clear(decal_atlas.texture, Color(0, 0, 0, 0), 0, decal_atlas.mipmaps, 0, 1);

	{
		//create the framebuffer

		Size2i s = decal_atlas.size;

		for (int i = 0; i < decal_atlas.mipmaps; i++) {
			DecalAtlas::MipMap mm;
			mm.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), decal_atlas.texture, 0, i);
			Vector<RID> fb;
			fb.push_back(mm.texture);
			mm.fb = RD::get_singleton()->framebuffer_create(fb);
			mm.size = s;
			decal_atlas.texture_mipmaps.push_back(mm);

			s.width = MAX(1, s.width >> 1);
			s.height = MAX(1, s.height >> 1);
		}
		{
			//create the SRGB variant
			RD::TextureView rd_view;
			rd_view.format_override = RD::DATA_FORMAT_R8G8B8A8_SRGB;
			decal_atlas.texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, decal_atlas.texture);
		}
	}

	RID prev_texture;
	for (int i = 0; i < decal_atlas.texture_mipmaps.size(); i++) {
		const DecalAtlas::MipMap &mm = decal_atlas.texture_mipmaps[i];

		Color clear_color(0, 0, 0, 0);

		if (decal_atlas.textures.size()) {
			if (i == 0) {
				Vector<Color> cc;
				cc.push_back(clear_color);

				RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(mm.fb, RD::INITIAL_ACTION_CLEAR, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_DISCARD, cc);

				const RID *K = nullptr;
				while ((K = decal_atlas.textures.next(K))) {
					DecalAtlas::Texture *t = decal_atlas.textures.getptr(*K);
					Texture *src_tex = texture_owner.get_or_null(*K);
					effects->copy_to_atlas_fb(src_tex->rd_texture, mm.fb, t->uv_rect, draw_list, false, t->panorama_to_dp_users > 0);
				}

				RD::get_singleton()->draw_list_end();

				prev_texture = mm.texture;
			} else {
				effects->copy_to_fb_rect(prev_texture, mm.fb, Rect2i(Point2i(), mm.size));
				prev_texture = mm.texture;
			}
		} else {
			RD::get_singleton()->texture_clear(mm.texture, clear_color, 0, 1, 0, 1);
		}
	}
}

int32_t RendererStorageRD::_global_variable_allocate(uint32_t p_elements) {
	int32_t idx = 0;
	while (idx + p_elements <= global_variables.buffer_size) {
		if (global_variables.buffer_usage[idx].elements == 0) {
			bool valid = true;
			for (uint32_t i = 1; i < p_elements; i++) {
				if (global_variables.buffer_usage[idx + i].elements > 0) {
					valid = false;
					idx += i + global_variables.buffer_usage[idx + i].elements;
					break;
				}
			}

			if (!valid) {
				continue; //if not valid, idx is in new position
			}

			return idx;
		} else {
			idx += global_variables.buffer_usage[idx].elements;
		}
	}

	return -1;
}

void RendererStorageRD::_global_variable_store_in_buffer(int32_t p_index, RS::GlobalVariableType p_type, const Variant &p_value) {
	switch (p_type) {
		case RS::GLOBAL_VAR_TYPE_BOOL: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			bool b = p_value;
			bv.x = b ? 1.0 : 0.0;
			bv.y = 0.0;
			bv.z = 0.0;
			bv.w = 0.0;

		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC2: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = 0.0;
			bv.w = 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC3: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = (bvec & 4) ? 1.0 : 0.0;
			bv.w = 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_BVEC4: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			uint32_t bvec = p_value;
			bv.x = (bvec & 1) ? 1.0 : 0.0;
			bv.y = (bvec & 2) ? 1.0 : 0.0;
			bv.z = (bvec & 4) ? 1.0 : 0.0;
			bv.w = (bvec & 8) ? 1.0 : 0.0;
		} break;
		case RS::GLOBAL_VAR_TYPE_INT: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			int32_t v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC2: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Vector2i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC3: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Vector3i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_IVEC4: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Vector<int32_t> v = p_value;
			bv.x = v.size() >= 1 ? v[0] : 0;
			bv.y = v.size() >= 2 ? v[1] : 0;
			bv.z = v.size() >= 3 ? v[2] : 0;
			bv.w = v.size() >= 4 ? v[3] : 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_RECT2I: {
			GlobalVariables::ValueInt &bv = *(GlobalVariables::ValueInt *)&global_variables.buffer_values[p_index];
			Rect2i v = p_value;
			bv.x = v.position.x;
			bv.y = v.position.y;
			bv.z = v.size.x;
			bv.w = v.size.y;
		} break;
		case RS::GLOBAL_VAR_TYPE_UINT: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			uint32_t v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC2: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			Vector2i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC3: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			Vector3i v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_UVEC4: {
			GlobalVariables::ValueUInt &bv = *(GlobalVariables::ValueUInt *)&global_variables.buffer_values[p_index];
			Vector<int32_t> v = p_value;
			bv.x = v.size() >= 1 ? v[0] : 0;
			bv.y = v.size() >= 2 ? v[1] : 0;
			bv.z = v.size() >= 3 ? v[2] : 0;
			bv.w = v.size() >= 4 ? v[3] : 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_FLOAT: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			float v = p_value;
			bv.x = v;
			bv.y = 0;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC2: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Vector2 v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = 0;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC3: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Vector3 v = p_value;
			bv.x = v.x;
			bv.y = v.y;
			bv.z = v.z;
			bv.w = 0;
		} break;
		case RS::GLOBAL_VAR_TYPE_VEC4: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Plane v = p_value;
			bv.x = v.normal.x;
			bv.y = v.normal.y;
			bv.z = v.normal.z;
			bv.w = v.d;
		} break;
		case RS::GLOBAL_VAR_TYPE_COLOR: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Color v = p_value;
			bv.x = v.r;
			bv.y = v.g;
			bv.z = v.b;
			bv.w = v.a;

			GlobalVariables::Value &bv_linear = global_variables.buffer_values[p_index + 1];
			v = v.to_linear();
			bv_linear.x = v.r;
			bv_linear.y = v.g;
			bv_linear.z = v.b;
			bv_linear.w = v.a;

		} break;
		case RS::GLOBAL_VAR_TYPE_RECT2: {
			GlobalVariables::Value &bv = global_variables.buffer_values[p_index];
			Rect2 v = p_value;
			bv.x = v.position.x;
			bv.y = v.position.y;
			bv.z = v.size.x;
			bv.w = v.size.y;
		} break;
		case RS::GLOBAL_VAR_TYPE_MAT2: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Vector<float> m2 = p_value;
			if (m2.size() < 4) {
				m2.resize(4);
			}
			bv[0].x = m2[0];
			bv[0].y = m2[1];
			bv[0].z = 0;
			bv[0].w = 0;

			bv[1].x = m2[2];
			bv[1].y = m2[3];
			bv[1].z = 0;
			bv[1].w = 0;

		} break;
		case RS::GLOBAL_VAR_TYPE_MAT3: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Basis v = p_value;
			bv[0].x = v.elements[0][0];
			bv[0].y = v.elements[1][0];
			bv[0].z = v.elements[2][0];
			bv[0].w = 0;

			bv[1].x = v.elements[0][1];
			bv[1].y = v.elements[1][1];
			bv[1].z = v.elements[2][1];
			bv[1].w = 0;

			bv[2].x = v.elements[0][2];
			bv[2].y = v.elements[1][2];
			bv[2].z = v.elements[2][2];
			bv[2].w = 0;

		} break;
		case RS::GLOBAL_VAR_TYPE_MAT4: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];

			Vector<float> m2 = p_value;
			if (m2.size() < 16) {
				m2.resize(16);
			}

			bv[0].x = m2[0];
			bv[0].y = m2[1];
			bv[0].z = m2[2];
			bv[0].w = m2[3];

			bv[1].x = m2[4];
			bv[1].y = m2[5];
			bv[1].z = m2[6];
			bv[1].w = m2[7];

			bv[2].x = m2[8];
			bv[2].y = m2[9];
			bv[2].z = m2[10];
			bv[2].w = m2[11];

			bv[3].x = m2[12];
			bv[3].y = m2[13];
			bv[3].z = m2[14];
			bv[3].w = m2[15];

		} break;
		case RS::GLOBAL_VAR_TYPE_TRANSFORM_2D: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Transform2D v = p_value;
			bv[0].x = v.elements[0][0];
			bv[0].y = v.elements[0][1];
			bv[0].z = 0;
			bv[0].w = 0;

			bv[1].x = v.elements[1][0];
			bv[1].y = v.elements[1][1];
			bv[1].z = 0;
			bv[1].w = 0;

			bv[2].x = v.elements[2][0];
			bv[2].y = v.elements[2][1];
			bv[2].z = 1;
			bv[2].w = 0;

		} break;
		case RS::GLOBAL_VAR_TYPE_TRANSFORM: {
			GlobalVariables::Value *bv = &global_variables.buffer_values[p_index];
			Transform3D v = p_value;
			bv[0].x = v.basis.elements[0][0];
			bv[0].y = v.basis.elements[1][0];
			bv[0].z = v.basis.elements[2][0];
			bv[0].w = 0;

			bv[1].x = v.basis.elements[0][1];
			bv[1].y = v.basis.elements[1][1];
			bv[1].z = v.basis.elements[2][1];
			bv[1].w = 0;

			bv[2].x = v.basis.elements[0][2];
			bv[2].y = v.basis.elements[1][2];
			bv[2].z = v.basis.elements[2][2];
			bv[2].w = 0;

			bv[3].x = v.origin.x;
			bv[3].y = v.origin.y;
			bv[3].z = v.origin.z;
			bv[3].w = 1;

		} break;
		default: {
			ERR_FAIL();
		}
	}
}

void RendererStorageRD::_global_variable_mark_buffer_dirty(int32_t p_index, int32_t p_elements) {
	int32_t prev_chunk = -1;

	for (int32_t i = 0; i < p_elements; i++) {
		int32_t chunk = (p_index + i) / GlobalVariables::BUFFER_DIRTY_REGION_SIZE;
		if (chunk != prev_chunk) {
			if (!global_variables.buffer_dirty_regions[chunk]) {
				global_variables.buffer_dirty_regions[chunk] = true;
				global_variables.buffer_dirty_region_count++;
			}
		}

		prev_chunk = chunk;
	}
}

void RendererStorageRD::global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) {
	ERR_FAIL_COND(global_variables.variables.has(p_name));
	GlobalVariables::Variable gv;
	gv.type = p_type;
	gv.value = p_value;
	gv.buffer_index = -1;

	if (p_type >= RS::GLOBAL_VAR_TYPE_SAMPLER2D) {
		//is texture
		global_variables.must_update_texture_materials = true; //normally there are none
	} else {
		gv.buffer_elements = 1;
		if (p_type == RS::GLOBAL_VAR_TYPE_COLOR || p_type == RS::GLOBAL_VAR_TYPE_MAT2) {
			//color needs to elements to store srgb and linear
			gv.buffer_elements = 2;
		}
		if (p_type == RS::GLOBAL_VAR_TYPE_MAT3 || p_type == RS::GLOBAL_VAR_TYPE_TRANSFORM_2D) {
			//color needs to elements to store srgb and linear
			gv.buffer_elements = 3;
		}
		if (p_type == RS::GLOBAL_VAR_TYPE_MAT4 || p_type == RS::GLOBAL_VAR_TYPE_TRANSFORM) {
			//color needs to elements to store srgb and linear
			gv.buffer_elements = 4;
		}

		//is vector, allocate in buffer and update index
		gv.buffer_index = _global_variable_allocate(gv.buffer_elements);
		ERR_FAIL_COND_MSG(gv.buffer_index < 0, vformat("Failed allocating global variable '%s' out of buffer memory. Consider increasing it in the Project Settings.", String(p_name)));
		global_variables.buffer_usage[gv.buffer_index].elements = gv.buffer_elements;
		_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.value);
		_global_variable_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);

		global_variables.must_update_buffer_materials = true; //normally there are none
	}

	global_variables.variables[p_name] = gv;
}

void RendererStorageRD::global_variable_remove(const StringName &p_name) {
	if (!global_variables.variables.has(p_name)) {
		return;
	}
	GlobalVariables::Variable &gv = global_variables.variables[p_name];

	if (gv.buffer_index >= 0) {
		global_variables.buffer_usage[gv.buffer_index].elements = 0;
		global_variables.must_update_buffer_materials = true;
	} else {
		global_variables.must_update_texture_materials = true;
	}

	global_variables.variables.erase(p_name);
}

Vector<StringName> RendererStorageRD::global_variable_get_list() const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(Vector<StringName>(), "This function should never be used outside the editor, it can severely damage performance.");
	}

	const StringName *K = nullptr;
	Vector<StringName> names;
	while ((K = global_variables.variables.next(K))) {
		names.push_back(*K);
	}
	names.sort_custom<StringName::AlphCompare>();
	return names;
}

void RendererStorageRD::global_variable_set(const StringName &p_name, const Variant &p_value) {
	ERR_FAIL_COND(!global_variables.variables.has(p_name));
	GlobalVariables::Variable &gv = global_variables.variables[p_name];
	gv.value = p_value;
	if (gv.override.get_type() == Variant::NIL) {
		if (gv.buffer_index >= 0) {
			//buffer
			_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.value);
			_global_variable_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);
		} else {
			//texture
			for (Set<RID>::Element *E = gv.texture_materials.front(); E; E = E->next()) {
				Material *material = material_owner.get_or_null(E->get());
				ERR_CONTINUE(!material);
				_material_queue_update(material, false, true);
			}
		}
	}
}

void RendererStorageRD::global_variable_set_override(const StringName &p_name, const Variant &p_value) {
	if (!global_variables.variables.has(p_name)) {
		return; //variable may not exist
	}

	ERR_FAIL_COND(p_value.get_type() == Variant::OBJECT);

	GlobalVariables::Variable &gv = global_variables.variables[p_name];

	gv.override = p_value;

	if (gv.buffer_index >= 0) {
		//buffer
		if (gv.override.get_type() == Variant::NIL) {
			_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.value);
		} else {
			_global_variable_store_in_buffer(gv.buffer_index, gv.type, gv.override);
		}

		_global_variable_mark_buffer_dirty(gv.buffer_index, gv.buffer_elements);
	} else {
		//texture
		for (Set<RID>::Element *E = gv.texture_materials.front(); E; E = E->next()) {
			Material *material = material_owner.get_or_null(E->get());
			ERR_CONTINUE(!material);
			_material_queue_update(material, false, true);
		}
	}
}

Variant RendererStorageRD::global_variable_get(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(Variant(), "This function should never be used outside the editor, it can severely damage performance.");
	}

	if (!global_variables.variables.has(p_name)) {
		return Variant();
	}

	return global_variables.variables[p_name].value;
}

RS::GlobalVariableType RendererStorageRD::global_variable_get_type_internal(const StringName &p_name) const {
	if (!global_variables.variables.has(p_name)) {
		return RS::GLOBAL_VAR_TYPE_MAX;
	}

	return global_variables.variables[p_name].type;
}

RS::GlobalVariableType RendererStorageRD::global_variable_get_type(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(RS::GLOBAL_VAR_TYPE_MAX, "This function should never be used outside the editor, it can severely damage performance.");
	}

	return global_variable_get_type_internal(p_name);
}

void RendererStorageRD::global_variables_load_settings(bool p_load_textures) {
	List<PropertyInfo> settings;
	ProjectSettings::get_singleton()->get_property_list(&settings);

	for (const PropertyInfo &E : settings) {
		if (E.name.begins_with("shader_globals/")) {
			StringName name = E.name.get_slice("/", 1);
			Dictionary d = ProjectSettings::get_singleton()->get(E.name);

			ERR_CONTINUE(!d.has("type"));
			ERR_CONTINUE(!d.has("value"));

			String type = d["type"];

			static const char *global_var_type_names[RS::GLOBAL_VAR_TYPE_MAX] = {
				"bool",
				"bvec2",
				"bvec3",
				"bvec4",
				"int",
				"ivec2",
				"ivec3",
				"ivec4",
				"rect2i",
				"uint",
				"uvec2",
				"uvec3",
				"uvec4",
				"float",
				"vec2",
				"vec3",
				"vec4",
				"color",
				"rect2",
				"mat2",
				"mat3",
				"mat4",
				"transform_2d",
				"transform",
				"sampler2D",
				"sampler2DArray",
				"sampler3D",
				"samplerCube",
			};

			RS::GlobalVariableType gvtype = RS::GLOBAL_VAR_TYPE_MAX;

			for (int i = 0; i < RS::GLOBAL_VAR_TYPE_MAX; i++) {
				if (global_var_type_names[i] == type) {
					gvtype = RS::GlobalVariableType(i);
					break;
				}
			}

			ERR_CONTINUE(gvtype == RS::GLOBAL_VAR_TYPE_MAX); //type invalid

			Variant value = d["value"];

			if (gvtype >= RS::GLOBAL_VAR_TYPE_SAMPLER2D) {
				//textire
				if (!p_load_textures) {
					value = RID();
					continue;
				}

				String path = value;
				RES resource = ResourceLoader::load(path);
				ERR_CONTINUE(resource.is_null());
				value = resource;
			}

			if (global_variables.variables.has(name)) {
				//has it, update it
				global_variable_set(name, value);
			} else {
				global_variable_add(name, gvtype, value);
			}
		}
	}
}

void RendererStorageRD::global_variables_clear() {
	global_variables.variables.clear(); //not right but for now enough
}

RID RendererStorageRD::global_variables_get_storage_buffer() const {
	return global_variables.buffer;
}

int32_t RendererStorageRD::global_variables_instance_allocate(RID p_instance) {
	ERR_FAIL_COND_V(global_variables.instance_buffer_pos.has(p_instance), -1);
	int32_t pos = _global_variable_allocate(ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES);
	global_variables.instance_buffer_pos[p_instance] = pos; //save anyway
	ERR_FAIL_COND_V_MSG(pos < 0, -1, "Too many instances using shader instance variables. Increase buffer size in Project Settings.");
	global_variables.buffer_usage[pos].elements = ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES;
	return pos;
}

void RendererStorageRD::global_variables_instance_free(RID p_instance) {
	ERR_FAIL_COND(!global_variables.instance_buffer_pos.has(p_instance));
	int32_t pos = global_variables.instance_buffer_pos[p_instance];
	if (pos >= 0) {
		global_variables.buffer_usage[pos].elements = 0;
	}
	global_variables.instance_buffer_pos.erase(p_instance);
}

void RendererStorageRD::global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) {
	if (!global_variables.instance_buffer_pos.has(p_instance)) {
		return; //just not allocated, ignore
	}
	int32_t pos = global_variables.instance_buffer_pos[p_instance];

	if (pos < 0) {
		return; //again, not allocated, ignore
	}
	ERR_FAIL_INDEX(p_index, ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES);
	ERR_FAIL_COND_MSG(p_value.get_type() > Variant::COLOR, "Unsupported variant type for instance parameter: " + Variant::get_type_name(p_value.get_type())); //anything greater not supported

	ShaderLanguage::DataType datatype_from_value[Variant::COLOR + 1] = {
		ShaderLanguage::TYPE_MAX, //nil
		ShaderLanguage::TYPE_BOOL, //bool
		ShaderLanguage::TYPE_INT, //int
		ShaderLanguage::TYPE_FLOAT, //float
		ShaderLanguage::TYPE_MAX, //string
		ShaderLanguage::TYPE_VEC2, //vec2
		ShaderLanguage::TYPE_IVEC2, //vec2i
		ShaderLanguage::TYPE_VEC4, //rect2
		ShaderLanguage::TYPE_IVEC4, //rect2i
		ShaderLanguage::TYPE_VEC3, // vec3
		ShaderLanguage::TYPE_IVEC3, //vec3i
		ShaderLanguage::TYPE_MAX, //xform2d not supported here
		ShaderLanguage::TYPE_VEC4, //plane
		ShaderLanguage::TYPE_VEC4, //quat
		ShaderLanguage::TYPE_MAX, //aabb not supported here
		ShaderLanguage::TYPE_MAX, //basis not supported here
		ShaderLanguage::TYPE_MAX, //xform not supported here
		ShaderLanguage::TYPE_VEC4 //color
	};

	ShaderLanguage::DataType datatype = datatype_from_value[p_value.get_type()];

	ERR_FAIL_COND_MSG(datatype == ShaderLanguage::TYPE_MAX, "Unsupported variant type for instance parameter: " + Variant::get_type_name(p_value.get_type())); //anything greater not supported

	pos += p_index;

	_fill_std140_variant_ubo_value(datatype, 0, p_value, (uint8_t *)&global_variables.buffer_values[pos], true); //instances always use linear color in this renderer
	_global_variable_mark_buffer_dirty(pos, 1);
}

void RendererStorageRD::_update_global_variables() {
	if (global_variables.buffer_dirty_region_count > 0) {
		uint32_t total_regions = global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE;
		if (total_regions / global_variables.buffer_dirty_region_count <= 4) {
			// 25% of regions dirty, just update all buffer
			RD::get_singleton()->buffer_update(global_variables.buffer, 0, sizeof(GlobalVariables::Value) * global_variables.buffer_size, global_variables.buffer_values);
			memset(global_variables.buffer_dirty_regions, 0, sizeof(bool) * total_regions);
		} else {
			uint32_t region_byte_size = sizeof(GlobalVariables::Value) * GlobalVariables::BUFFER_DIRTY_REGION_SIZE;

			for (uint32_t i = 0; i < total_regions; i++) {
				if (global_variables.buffer_dirty_regions[i]) {
					RD::get_singleton()->buffer_update(global_variables.buffer, i * region_byte_size, region_byte_size, global_variables.buffer_values);

					global_variables.buffer_dirty_regions[i] = false;
				}
			}
		}

		global_variables.buffer_dirty_region_count = 0;
	}

	if (global_variables.must_update_buffer_materials) {
		// only happens in the case of a buffer variable added or removed,
		// so not often.
		for (const RID &E : global_variables.materials_using_buffer) {
			Material *material = material_owner.get_or_null(E);
			ERR_CONTINUE(!material); //wtf

			_material_queue_update(material, true, false);
		}

		global_variables.must_update_buffer_materials = false;
	}

	if (global_variables.must_update_texture_materials) {
		// only happens in the case of a buffer variable added or removed,
		// so not often.
		for (const RID &E : global_variables.materials_using_texture) {
			Material *material = material_owner.get_or_null(E);
			ERR_CONTINUE(!material); //wtf

			_material_queue_update(material, false, true);
			print_line("update material texture?");
		}

		global_variables.must_update_texture_materials = false;
	}
}

void RendererStorageRD::update_dirty_resources() {
	_update_global_variables(); //must do before materials, so it can queue them for update
	_update_queued_materials();
	_update_dirty_multimeshes();
	_update_dirty_skeletons();
	_update_decal_atlas();
}

bool RendererStorageRD::has_os_feature(const String &p_feature) const {
	if (p_feature == "rgtc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC5_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "s3tc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "bptc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC7_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if ((p_feature == "etc" || p_feature == "etc2") && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	if (p_feature == "pvrtc" && RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG, RD::TEXTURE_USAGE_SAMPLING_BIT)) {
		return true;
	}

	return false;
}

bool RendererStorageRD::free(RID p_rid) {
	if (texture_owner.owns(p_rid)) {
		Texture *t = texture_owner.get_or_null(p_rid);

		ERR_FAIL_COND_V(!t, false);
		ERR_FAIL_COND_V(t->is_render_target, false);

		if (RD::get_singleton()->texture_is_valid(t->rd_texture_srgb)) {
			//erase this first, as it's a dependency of the one below
			RD::get_singleton()->free(t->rd_texture_srgb);
		}
		if (RD::get_singleton()->texture_is_valid(t->rd_texture)) {
			RD::get_singleton()->free(t->rd_texture);
		}

		if (t->is_proxy && t->proxy_to.is_valid()) {
			Texture *proxy_to = texture_owner.get_or_null(t->proxy_to);
			if (proxy_to) {
				proxy_to->proxies.erase(p_rid);
			}
		}

		if (decal_atlas.textures.has(p_rid)) {
			decal_atlas.textures.erase(p_rid);
			//there is not much a point of making it dirty, just let it be.
		}

		for (int i = 0; i < t->proxies.size(); i++) {
			Texture *p = texture_owner.get_or_null(t->proxies[i]);
			ERR_CONTINUE(!p);
			p->proxy_to = RID();
			p->rd_texture = RID();
			p->rd_texture_srgb = RID();
		}

		if (t->canvas_texture) {
			memdelete(t->canvas_texture);
		}
		texture_owner.free(p_rid);

	} else if (canvas_texture_owner.owns(p_rid)) {
		canvas_texture_owner.free(p_rid);
	} else if (shader_owner.owns(p_rid)) {
		Shader *shader = shader_owner.get_or_null(p_rid);
		//make material unreference this
		while (shader->owners.size()) {
			material_set_shader(shader->owners.front()->get()->self, RID());
		}
		//clear data if exists
		if (shader->data) {
			memdelete(shader->data);
		}
		shader_owner.free(p_rid);

	} else if (material_owner.owns(p_rid)) {
		Material *material = material_owner.get_or_null(p_rid);
		material_set_shader(p_rid, RID()); //clean up shader
		material->dependency.deleted_notify(p_rid);

		material_owner.free(p_rid);
	} else if (mesh_owner.owns(p_rid)) {
		mesh_clear(p_rid);
		mesh_set_shadow_mesh(p_rid, RID());
		Mesh *mesh = mesh_owner.get_or_null(p_rid);
		mesh->dependency.deleted_notify(p_rid);
		if (mesh->instances.size()) {
			ERR_PRINT("deleting mesh with active instances");
		}
		if (mesh->shadow_owners.size()) {
			for (Set<Mesh *>::Element *E = mesh->shadow_owners.front(); E; E = E->next()) {
				Mesh *shadow_owner = E->get();
				shadow_owner->shadow_mesh = RID();
				shadow_owner->dependency.changed_notify(DEPENDENCY_CHANGED_MESH);
			}
		}
		mesh_owner.free(p_rid);
	} else if (mesh_instance_owner.owns(p_rid)) {
		MeshInstance *mi = mesh_instance_owner.get_or_null(p_rid);
		_mesh_instance_clear(mi);
		mi->mesh->instances.erase(mi->I);
		mi->I = nullptr;

		mesh_instance_owner.free(p_rid);

	} else if (multimesh_owner.owns(p_rid)) {
		_update_dirty_multimeshes();
		multimesh_allocate_data(p_rid, 0, RS::MULTIMESH_TRANSFORM_2D);
		MultiMesh *multimesh = multimesh_owner.get_or_null(p_rid);
		multimesh->dependency.deleted_notify(p_rid);
		multimesh_owner.free(p_rid);
	} else if (skeleton_owner.owns(p_rid)) {
		_update_dirty_skeletons();
		skeleton_allocate_data(p_rid, 0);
		Skeleton *skeleton = skeleton_owner.get_or_null(p_rid);
		skeleton->dependency.deleted_notify(p_rid);
		skeleton_owner.free(p_rid);
	} else if (reflection_probe_owner.owns(p_rid)) {
		ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_rid);
		reflection_probe->dependency.deleted_notify(p_rid);
		reflection_probe_owner.free(p_rid);
	} else if (decal_owner.owns(p_rid)) {
		Decal *decal = decal_owner.get_or_null(p_rid);
		for (int i = 0; i < RS::DECAL_TEXTURE_MAX; i++) {
			if (decal->textures[i].is_valid() && texture_owner.owns(decal->textures[i])) {
				texture_remove_from_decal_atlas(decal->textures[i]);
			}
		}
		decal->dependency.deleted_notify(p_rid);
		decal_owner.free(p_rid);
	} else if (voxel_gi_owner.owns(p_rid)) {
		voxel_gi_allocate_data(p_rid, Transform3D(), AABB(), Vector3i(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<int>()); //deallocate
		VoxelGI *voxel_gi = voxel_gi_owner.get_or_null(p_rid);
		voxel_gi->dependency.deleted_notify(p_rid);
		voxel_gi_owner.free(p_rid);
	} else if (lightmap_owner.owns(p_rid)) {
		lightmap_set_textures(p_rid, RID(), false);
		Lightmap *lightmap = lightmap_owner.get_or_null(p_rid);
		lightmap->dependency.deleted_notify(p_rid);
		lightmap_owner.free(p_rid);

	} else if (light_owner.owns(p_rid)) {
		light_set_projector(p_rid, RID()); //clear projector
		// delete the texture
		Light *light = light_owner.get_or_null(p_rid);
		light->dependency.deleted_notify(p_rid);
		light_owner.free(p_rid);

	} else if (particles_owner.owns(p_rid)) {
		update_particles();
		Particles *particles = particles_owner.get_or_null(p_rid);
		particles->dependency.deleted_notify(p_rid);
		_particles_free_data(particles);
		particles_owner.free(p_rid);
	} else if (particles_collision_owner.owns(p_rid)) {
		ParticlesCollision *particles_collision = particles_collision_owner.get_or_null(p_rid);

		if (particles_collision->heightfield_texture.is_valid()) {
			RD::get_singleton()->free(particles_collision->heightfield_texture);
		}
		particles_collision->dependency.deleted_notify(p_rid);
		particles_collision_owner.free(p_rid);
	} else if (visibility_notifier_owner.owns(p_rid)) {
		VisibilityNotifier *vn = visibility_notifier_owner.get_or_null(p_rid);
		vn->dependency.deleted_notify(p_rid);
		visibility_notifier_owner.free(p_rid);
	} else if (particles_collision_instance_owner.owns(p_rid)) {
		particles_collision_instance_owner.free(p_rid);
	} else if (fog_volume_owner.owns(p_rid)) {
		FogVolume *fog_volume = fog_volume_owner.get_or_null(p_rid);
		fog_volume->dependency.deleted_notify(p_rid);
		fog_volume_owner.free(p_rid);
	} else if (render_target_owner.owns(p_rid)) {
		RenderTarget *rt = render_target_owner.get_or_null(p_rid);

		_clear_render_target(rt);

		if (rt->texture.is_valid()) {
			Texture *tex = texture_owner.get_or_null(rt->texture);
			tex->is_render_target = false;
			free(rt->texture);
		}

		render_target_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

void RendererStorageRD::init_effects(bool p_prefer_raster_effects) {
	effects = memnew(EffectsRD(p_prefer_raster_effects));
}

EffectsRD *RendererStorageRD::get_effects() {
	ERR_FAIL_NULL_V_MSG(effects, nullptr, "Effects haven't been initialised yet.");
	return effects;
}

void RendererStorageRD::capture_timestamps_begin() {
	RD::get_singleton()->capture_timestamp("Frame Begin");
}

void RendererStorageRD::capture_timestamp(const String &p_name) {
	RD::get_singleton()->capture_timestamp(p_name);
}

uint32_t RendererStorageRD::get_captured_timestamps_count() const {
	return RD::get_singleton()->get_captured_timestamps_count();
}

uint64_t RendererStorageRD::get_captured_timestamps_frame() const {
	return RD::get_singleton()->get_captured_timestamps_frame();
}

uint64_t RendererStorageRD::get_captured_timestamp_gpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_gpu_time(p_index);
}

uint64_t RendererStorageRD::get_captured_timestamp_cpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_cpu_time(p_index);
}

String RendererStorageRD::get_captured_timestamp_name(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_name(p_index);
}

void RendererStorageRD::update_memory_info() {
	texture_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TEXTURES);
	buffer_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_BUFFERS);
	total_mem_cache = RenderingDevice::get_singleton()->get_memory_usage(RenderingDevice::MEMORY_TOTAL);
}
uint64_t RendererStorageRD::get_rendering_info(RS::RenderingInfo p_info) {
	if (p_info == RS::RENDERING_INFO_TEXTURE_MEM_USED) {
		return texture_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_BUFFER_MEM_USED) {
		return buffer_mem_cache;
	} else if (p_info == RS::RENDERING_INFO_VIDEO_MEM_USED) {
		return total_mem_cache;
	}
	return 0;
}

String RendererStorageRD::get_video_adapter_name() const {
	return RenderingDevice::get_singleton()->get_device_name();
}
String RendererStorageRD::get_video_adapter_vendor() const {
	return RenderingDevice::get_singleton()->get_device_vendor_name();
}

RendererStorageRD *RendererStorageRD::base_singleton = nullptr;

RendererStorageRD::RendererStorageRD() {
	base_singleton = this;

	for (int i = 0; i < SHADER_TYPE_MAX; i++) {
		shader_data_request_func[i] = nullptr;
	}

	static_assert(sizeof(GlobalVariables::Value) == 16);

	global_variables.buffer_size = GLOBAL_GET("rendering/limits/global_shader_variables/buffer_size");
	global_variables.buffer_size = MAX(4096, global_variables.buffer_size);
	global_variables.buffer_values = memnew_arr(GlobalVariables::Value, global_variables.buffer_size);
	memset(global_variables.buffer_values, 0, sizeof(GlobalVariables::Value) * global_variables.buffer_size);
	global_variables.buffer_usage = memnew_arr(GlobalVariables::ValueUsage, global_variables.buffer_size);
	global_variables.buffer_dirty_regions = memnew_arr(bool, global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE);
	memset(global_variables.buffer_dirty_regions, 0, sizeof(bool) * global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE);
	global_variables.buffer = RD::get_singleton()->storage_buffer_create(sizeof(GlobalVariables::Value) * global_variables.buffer_size);

	{ //create default textures

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);

			//take the chance and initialize decal atlas to something
			decal_atlas.texture = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
			decal_atlas.texture_srgb = decal_atlas.texture;
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 128);
			pv.set(i * 4 + 1, 128);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_NORMAL] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 128);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_ANISO] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		default_rd_textures[DEFAULT_RD_TEXTURE_MULTIMESH_BUFFER] = RD::get_singleton()->texture_buffer_create(16, RD::DATA_FORMAT_R8G8B8A8_UNORM, pv);

		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			tformat.format = RD::DATA_FORMAT_R8G8B8A8_UINT;
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_2D_UINT] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default cubemap

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 6;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_CUBE_ARRAY;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			for (int i = 0; i < 6; i++) {
				vpv.push_back(pv);
			}
			default_rd_textures[DEFAULT_RD_TEXTURE_CUBEMAP_ARRAY_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default cubemap array

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 6;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_CUBE;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			for (int i = 0; i < 6; i++) {
				vpv.push_back(pv);
			}
			default_rd_textures[DEFAULT_RD_TEXTURE_CUBEMAP_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default cubemap white array

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 6;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_CUBE;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			for (int i = 0; i < 6; i++) {
				vpv.push_back(pv);
			}
			default_rd_textures[DEFAULT_RD_TEXTURE_CUBEMAP_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default 3D

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.depth = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_3D;

		Vector<uint8_t> pv;
		pv.resize(64 * 4);
		for (int i = 0; i < 64; i++) {
			pv.set(i * 4 + 0, 0);
			pv.set(i * 4 + 1, 0);
			pv.set(i * 4 + 2, 0);
			pv.set(i * 4 + 3, 0);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_3D_BLACK] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
		for (int i = 0; i < 64; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_3D_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	{ //create default array

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.array_layers = 1;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.texture_type = RD::TEXTURE_TYPE_2D_ARRAY;

		Vector<uint8_t> pv;
		pv.resize(16 * 4);
		for (int i = 0; i < 16; i++) {
			pv.set(i * 4 + 0, 255);
			pv.set(i * 4 + 1, 255);
			pv.set(i * 4 + 2, 255);
			pv.set(i * 4 + 3, 255);
		}

		{
			Vector<Vector<uint8_t>> vpv;
			vpv.push_back(pv);
			default_rd_textures[DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE] = RD::get_singleton()->texture_create(tformat, RD::TextureView(), vpv);
		}
	}

	//default samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::SamplerState sampler_state;
			switch (i) {
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.max_lod = 0;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}

				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_NEAREST;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					if (GLOBAL_GET("rendering/textures/default_filters/use_nearest_mipmap_filter")) {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_NEAREST;
					} else {
						sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					}
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/textures/default_filters/anisotropic_filtering_level"));

				} break;
				default: {
				}
			}
			switch (j) {
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_REPEAT;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_w = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
				} break;
				default: {
				}
			}

			default_rd_samplers[i][j] = RD::get_singleton()->sampler_create(sampler_state);
		}
	}

	//default rd buffers
	{
		Vector<uint8_t> buffer;
		{
			buffer.resize(sizeof(float) * 3);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 0.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_VERTEX] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //normal
			buffer.resize(sizeof(float) * 3);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 1.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_NORMAL] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //tangent
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 1.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
				fptr[3] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_TANGENT] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //color
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 1.0;
				fptr[1] = 1.0;
				fptr[2] = 1.0;
				fptr[3] = 1.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_COLOR] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //tex uv 1
			buffer.resize(sizeof(float) * 2);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 0.0;
				fptr[1] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_TEX_UV] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}
		{ //tex uv 2
			buffer.resize(sizeof(float) * 2);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 0.0;
				fptr[1] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_TEX_UV2] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		for (int i = 0; i < RS::ARRAY_CUSTOM_COUNT; i++) {
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 0.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
				fptr[3] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_CUSTOM0 + i] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //bones
			buffer.resize(sizeof(uint32_t) * 4);
			{
				uint8_t *w = buffer.ptrw();
				uint32_t *fptr = (uint32_t *)w;
				fptr[0] = 0;
				fptr[1] = 0;
				fptr[2] = 0;
				fptr[3] = 0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_BONES] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}

		{ //weights
			buffer.resize(sizeof(float) * 4);
			{
				uint8_t *w = buffer.ptrw();
				float *fptr = (float *)w;
				fptr[0] = 0.0;
				fptr[1] = 0.0;
				fptr[2] = 0.0;
				fptr[3] = 0.0;
			}
			mesh_default_rd_buffers[DEFAULT_RD_BUFFER_WEIGHTS] = RD::get_singleton()->vertex_buffer_create(buffer.size(), buffer);
		}
	}

	using_lightmap_array = true; // high end
	if (using_lightmap_array) {
		uint32_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);

		if (textures_per_stage <= 256) {
			lightmap_textures.resize(32);
		} else {
			lightmap_textures.resize(1024);
		}

		for (int i = 0; i < lightmap_textures.size(); i++) {
			lightmap_textures.write[i] = default_rd_textures[DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE];
		}
	}

	lightmap_probe_capture_update_speed = GLOBAL_GET("rendering/lightmapping/probe_capture/update_speed");

	/* Particles */

	{
		// Initialize particles
		Vector<String> particles_modes;
		particles_modes.push_back("");
		particles_shader.shader.initialize(particles_modes, String());
	}
	shader_set_data_request_function(RendererStorageRD::SHADER_TYPE_PARTICLES, _create_particles_shader_funcs);
	material_set_data_request_function(RendererStorageRD::SHADER_TYPE_PARTICLES, _create_particles_material_funcs);

	{
		ShaderCompilerRD::DefaultIdentifierActions actions;

		actions.renames["COLOR"] = "PARTICLE.color";
		actions.renames["VELOCITY"] = "PARTICLE.velocity";
		//actions.renames["MASS"] = "mass"; ?
		actions.renames["ACTIVE"] = "particle_active";
		actions.renames["RESTART"] = "restart";
		actions.renames["CUSTOM"] = "PARTICLE.custom";
		actions.renames["TRANSFORM"] = "PARTICLE.xform";
		actions.renames["TIME"] = "FRAME.time";
		actions.renames["PI"] = _MKSTR(Math_PI);
		actions.renames["TAU"] = _MKSTR(Math_TAU);
		actions.renames["E"] = _MKSTR(Math_E);
		actions.renames["LIFETIME"] = "params.lifetime";
		actions.renames["DELTA"] = "local_delta";
		actions.renames["NUMBER"] = "particle_number";
		actions.renames["INDEX"] = "index";
		//actions.renames["GRAVITY"] = "current_gravity";
		actions.renames["EMISSION_TRANSFORM"] = "FRAME.emission_transform";
		actions.renames["RANDOM_SEED"] = "FRAME.random_seed";
		actions.renames["FLAG_EMIT_POSITION"] = "EMISSION_FLAG_HAS_POSITION";
		actions.renames["FLAG_EMIT_ROT_SCALE"] = "EMISSION_FLAG_HAS_ROTATION_SCALE";
		actions.renames["FLAG_EMIT_VELOCITY"] = "EMISSION_FLAG_HAS_VELOCITY";
		actions.renames["FLAG_EMIT_COLOR"] = "EMISSION_FLAG_HAS_COLOR";
		actions.renames["FLAG_EMIT_CUSTOM"] = "EMISSION_FLAG_HAS_CUSTOM";
		actions.renames["RESTART_POSITION"] = "restart_position";
		actions.renames["RESTART_ROT_SCALE"] = "restart_rotation_scale";
		actions.renames["RESTART_VELOCITY"] = "restart_velocity";
		actions.renames["RESTART_COLOR"] = "restart_color";
		actions.renames["RESTART_CUSTOM"] = "restart_custom";
		actions.renames["emit_subparticle"] = "emit_subparticle";
		actions.renames["COLLIDED"] = "collided";
		actions.renames["COLLISION_NORMAL"] = "collision_normal";
		actions.renames["COLLISION_DEPTH"] = "collision_depth";
		actions.renames["ATTRACTOR_FORCE"] = "attractor_force";

		actions.render_mode_defines["disable_force"] = "#define DISABLE_FORCE\n";
		actions.render_mode_defines["disable_velocity"] = "#define DISABLE_VELOCITY\n";
		actions.render_mode_defines["keep_data"] = "#define ENABLE_KEEP_DATA\n";
		actions.render_mode_defines["collision_use_scale"] = "#define USE_COLLISON_SCALE\n";

		actions.sampler_array_name = "material_samplers";
		actions.base_texture_binding_index = 1;
		actions.texture_layout_set = 3;
		actions.base_uniform_string = "material.";
		actions.base_varying_index = 10;

		actions.default_filter = ShaderLanguage::FILTER_LINEAR_MIPMAP;
		actions.default_repeat = ShaderLanguage::REPEAT_ENABLE;
		actions.global_buffer_array_variable = "global_variables.data";

		particles_shader.compiler.initialize(actions);
	}

	{
		// default material and shader for particles shader
		particles_shader.default_shader = shader_allocate();
		shader_initialize(particles_shader.default_shader);
		shader_set_code(particles_shader.default_shader, R"(
// Default particles shader.

shader_type particles;

void process() {
	COLOR = vec4(1.0);
}
)");
		particles_shader.default_material = material_allocate();
		material_initialize(particles_shader.default_material);
		material_set_shader(particles_shader.default_material, particles_shader.default_shader);

		ParticlesMaterialData *md = (ParticlesMaterialData *)material_get_data(particles_shader.default_material, RendererStorageRD::SHADER_TYPE_PARTICLES);
		particles_shader.default_shader_rd = particles_shader.shader.version_get_shader(md->shader_data->version, 0);

		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
			u.binding = 1;
			u.ids.resize(12);
			RID *ids_ptr = u.ids.ptrw();
			ids_ptr[0] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[1] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[2] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[3] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[4] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[5] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
			ids_ptr[6] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[7] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[8] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[9] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[10] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			ids_ptr[11] = sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC, RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 2;
			u.ids.push_back(global_variables_get_storage_buffer());
			uniforms.push_back(u);
		}

		particles_shader.base_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.default_shader_rd, 0);
	}

	default_rd_storage_buffer = RD::get_singleton()->storage_buffer_create(sizeof(uint32_t) * 4);

	{
		Vector<String> copy_modes;
		copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n");
		copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n#define MODE_2D\n");
		copy_modes.push_back("\n#define MODE_FILL_SORT_BUFFER\n#define USE_SORT_BUFFER\n");
		copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n#define USE_SORT_BUFFER\n");

		particles_shader.copy_shader.initialize(copy_modes);

		particles_shader.copy_shader_version = particles_shader.copy_shader.version_create();

		for (int i = 0; i < ParticlesShader::COPY_MODE_MAX; i++) {
			particles_shader.copy_pipelines[i] = RD::get_singleton()->compute_pipeline_create(particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, i));
		}
	}

	{
		Vector<String> sdf_modes;
		sdf_modes.push_back("\n#define MODE_LOAD\n");
		sdf_modes.push_back("\n#define MODE_LOAD_SHRINK\n");
		sdf_modes.push_back("\n#define MODE_PROCESS\n");
		sdf_modes.push_back("\n#define MODE_PROCESS_OPTIMIZED\n");
		sdf_modes.push_back("\n#define MODE_STORE\n");
		sdf_modes.push_back("\n#define MODE_STORE_SHRINK\n");

		rt_sdf.shader.initialize(sdf_modes);

		rt_sdf.shader_version = rt_sdf.shader.version_create();

		for (int i = 0; i < RenderTargetSDF::SHADER_MAX; i++) {
			rt_sdf.pipelines[i] = RD::get_singleton()->compute_pipeline_create(rt_sdf.shader.version_get_shader(rt_sdf.shader_version, i));
		}
	}
	{
		Vector<String> skeleton_modes;
		skeleton_modes.push_back("\n#define MODE_2D\n");
		skeleton_modes.push_back("");

		skeleton_shader.shader.initialize(skeleton_modes);
		skeleton_shader.version = skeleton_shader.shader.version_create();
		for (int i = 0; i < SkeletonShader::SHADER_MODE_MAX; i++) {
			skeleton_shader.version_shader[i] = skeleton_shader.shader.version_get_shader(skeleton_shader.version, i);
			skeleton_shader.pipeline[i] = RD::get_singleton()->compute_pipeline_create(skeleton_shader.version_shader[i]);
		}

		{
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.binding = 0;
				u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.ids.push_back(default_rd_storage_buffer);
				uniforms.push_back(u);
			}
			skeleton_shader.default_skeleton_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, skeleton_shader.version_shader[0], SkeletonShader::UNIFORM_SET_SKELETON);
		}
	}
}

RendererStorageRD::~RendererStorageRD() {
	memdelete_arr(global_variables.buffer_values);
	memdelete_arr(global_variables.buffer_usage);
	memdelete_arr(global_variables.buffer_dirty_regions);
	RD::get_singleton()->free(global_variables.buffer);

	//def textures
	for (int i = 0; i < DEFAULT_RD_TEXTURE_MAX; i++) {
		RD::get_singleton()->free(default_rd_textures[i]);
	}

	//def samplers
	for (int i = 1; i < RS::CANVAS_ITEM_TEXTURE_FILTER_MAX; i++) {
		for (int j = 1; j < RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX; j++) {
			RD::get_singleton()->free(default_rd_samplers[i][j]);
		}
	}

	//def buffers
	for (int i = 0; i < DEFAULT_RD_BUFFER_MAX; i++) {
		RD::get_singleton()->free(mesh_default_rd_buffers[i]);
	}

	particles_shader.copy_shader.version_free(particles_shader.copy_shader_version);
	rt_sdf.shader.version_free(rt_sdf.shader_version);

	skeleton_shader.shader.version_free(skeleton_shader.version);

	RenderingServer::get_singleton()->free(particles_shader.default_material);
	RenderingServer::get_singleton()->free(particles_shader.default_shader);

	RD::get_singleton()->free(default_rd_storage_buffer);

	if (decal_atlas.textures.size()) {
		ERR_PRINT("Decal Atlas: " + itos(decal_atlas.textures.size()) + " textures were not removed from the atlas.");
	}

	if (decal_atlas.texture.is_valid()) {
		RD::get_singleton()->free(decal_atlas.texture);
	}

	if (effects) {
		memdelete(effects);
		effects = nullptr;
	}
}
