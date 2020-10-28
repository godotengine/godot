/*************************************************************************/
/*  rasterizer_storage_rd.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rasterizer_storage_rd.h"

#include "core/engine.h"
#include "core/io/resource_loader.h"
#include "core/project_settings.h"
#include "rasterizer_rd.h"
#include "servers/rendering/shader_language.h"

Ref<Image> RasterizerStorageRD::_validate_texture_format(const Ref<Image> &p_image, TextureToRDFormat &r_format) {
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
		case Image::FORMAT_PVRTC2: {
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
		case Image::FORMAT_PVRTC2A: {
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
		case Image::FORMAT_PVRTC4: {
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
		case Image::FORMAT_PVRTC4A: {
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

RID RasterizerStorageRD::texture_2d_create(const Ref<Image> &p_image) {
	ERR_FAIL_COND_V(p_image.is_null(), RID());
	ERR_FAIL_COND_V(p_image->empty(), RID());

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
		rd_format.type = texture.rd_type;
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
	ERR_FAIL_COND_V(texture.rd_texture.is_null(), RID());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND_V(texture.rd_texture_srgb.is_null(), RID());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	return texture_owner.make_rid(texture);
}

RID RasterizerStorageRD::texture_2d_layered_create(const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) {
	ERR_FAIL_COND_V(p_layers.size() == 0, RID());

	ERR_FAIL_COND_V(p_layered_type == RS::TEXTURE_LAYERED_CUBEMAP && p_layers.size() != 6, RID());
	ERR_FAIL_COND_V(p_layered_type == RS::TEXTURE_LAYERED_CUBEMAP_ARRAY && (p_layers.size() < 6 || (p_layers.size() % 6) != 0), RID());

	TextureToRDFormat ret_format;
	Vector<Ref<Image>> images;
	{
		int valid_width = 0;
		int valid_height = 0;
		bool valid_mipmaps = false;
		Image::Format valid_format = Image::FORMAT_MAX;

		for (int i = 0; i < p_layers.size(); i++) {
			ERR_FAIL_COND_V(p_layers[i]->empty(), RID());

			if (i == 0) {
				valid_width = p_layers[i]->get_width();
				valid_height = p_layers[i]->get_height();
				valid_format = p_layers[i]->get_format();
				valid_mipmaps = p_layers[i]->has_mipmaps();
			} else {
				ERR_FAIL_COND_V(p_layers[i]->get_width() != valid_width, RID());
				ERR_FAIL_COND_V(p_layers[i]->get_height() != valid_height, RID());
				ERR_FAIL_COND_V(p_layers[i]->get_format() != valid_format, RID());
				ERR_FAIL_COND_V(p_layers[i]->has_mipmaps() != valid_mipmaps, RID());
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
		rd_format.type = texture.rd_type;
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
	ERR_FAIL_COND_V(texture.rd_texture.is_null(), RID());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND_V(texture.rd_texture_srgb.is_null(), RID());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	return texture_owner.make_rid(texture);
}

RID RasterizerStorageRD::texture_3d_create(Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) {
	ERR_FAIL_COND_V(p_data.size() == 0, RID());
	Image::Image3DValidateError verr = Image::validate_3d_image(p_format, p_width, p_height, p_depth, p_mipmaps, p_data);
	if (verr != Image::VALIDATE_3D_OK) {
		ERR_FAIL_V_MSG(RID(), Image::get_3d_image_validation_error_text(verr));
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

			copymem(&all_data.write[offset], images[i]->get_data().ptr(), s);
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
		rd_format.type = texture.rd_type;
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
	ERR_FAIL_COND_V(texture.rd_texture.is_null(), RID());
	if (texture.rd_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = texture.rd_format_srgb;
		texture.rd_texture_srgb = RD::get_singleton()->texture_create_shared(rd_view, texture.rd_texture);
		if (texture.rd_texture_srgb.is_null()) {
			RD::get_singleton()->free(texture.rd_texture);
			ERR_FAIL_COND_V(texture.rd_texture_srgb.is_null(), RID());
		}
	}

	//used for 2D, overridable
	texture.width_2d = texture.width;
	texture.height_2d = texture.height;
	texture.is_render_target = false;
	texture.rd_view = rd_view;
	texture.is_proxy = false;

	return texture_owner.make_rid(texture);
}

RID RasterizerStorageRD::texture_proxy_create(RID p_base) {
	Texture *tex = texture_owner.getornull(p_base);
	ERR_FAIL_COND_V(!tex, RID());
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

	RID rid = texture_owner.make_rid(proxy_tex);

	tex->proxies.push_back(rid);

	return rid;
}

void RasterizerStorageRD::_texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer, bool p_immediate) {
	ERR_FAIL_COND(p_image.is_null() || p_image->empty());

	Texture *tex = texture_owner.getornull(p_texture);
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

	RD::get_singleton()->texture_update(tex->rd_texture, p_layer, validated->get_data(), !p_immediate);
}

void RasterizerStorageRD::texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	_texture_2d_update(p_texture, p_image, p_layer, true);
}

void RasterizerStorageRD::texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer) {
	_texture_2d_update(p_texture, p_image, p_layer, false);
}

void RasterizerStorageRD::texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) {
	Texture *tex = texture_owner.getornull(p_texture);
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
			copymem(&all_data.write[offset], images[i]->get_data().ptr(), s);
			offset += s;
		}
	}

	RD::get_singleton()->texture_update(tex->rd_texture, 0, all_data, true);
}

void RasterizerStorageRD::texture_proxy_update(RID p_texture, RID p_proxy_to) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(!tex->is_proxy);
	Texture *proxy_to = texture_owner.getornull(p_proxy_to);
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
		Texture *prev_tex = texture_owner.getornull(tex->proxy_to);
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
RID RasterizerStorageRD::texture_2d_placeholder_create() {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instance();
	image->create(4, 4, false, Image::FORMAT_RGBA8);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			image->set_pixel(i, j, Color(1, 0, 1, 1));
		}
	}

	return texture_2d_create(image);
}

RID RasterizerStorageRD::texture_2d_layered_placeholder_create(RS::TextureLayeredType p_layered_type) {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instance();
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

	return texture_2d_layered_create(images, p_layered_type);
}

RID RasterizerStorageRD::texture_3d_placeholder_create() {
	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instance();
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

	return texture_3d_create(Image::FORMAT_RGBA8, 4, 4, 4, false, images);
}

Ref<Image> RasterizerStorageRD::texture_2d_get(RID p_texture) const {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND_V(!tex, Ref<Image>());

#ifdef TOOLS_ENABLED
	if (tex->image_cache_2d.is_valid()) {
		return tex->image_cache_2d;
	}
#endif
	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(tex->rd_texture, 0);
	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instance();
	image->create(tex->width, tex->height, tex->mipmaps > 1, tex->validated_format, data);
	ERR_FAIL_COND_V(image->empty(), Ref<Image>());
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

Ref<Image> RasterizerStorageRD::texture_2d_layer_get(RID p_texture, int p_layer) const {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND_V(!tex, Ref<Image>());

	Vector<uint8_t> data = RD::get_singleton()->texture_get_data(tex->rd_texture, p_layer);
	ERR_FAIL_COND_V(data.size() == 0, Ref<Image>());
	Ref<Image> image;
	image.instance();
	image->create(tex->width, tex->height, tex->mipmaps > 1, tex->validated_format, data);
	ERR_FAIL_COND_V(image->empty(), Ref<Image>());
	if (tex->format != tex->validated_format) {
		image->convert(tex->format);
	}

	return image;
}

Vector<Ref<Image>> RasterizerStorageRD::texture_3d_get(RID p_texture) const {
	Texture *tex = texture_owner.getornull(p_texture);
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
		img.instance();
		img->create(bs.size.width, bs.size.height, false, tex->validated_format, sub_region);
		ERR_FAIL_COND_V(img->empty(), Vector<Ref<Image>>());
		if (tex->format != tex->validated_format) {
			img->convert(tex->format);
		}

		ret.push_back(img);
	}

	return ret;
}

void RasterizerStorageRD::texture_replace(RID p_texture, RID p_by_texture) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->proxy_to.is_valid()); //can't replace proxy
	Texture *by_tex = texture_owner.getornull(p_by_texture);
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

void RasterizerStorageRD::texture_set_size_override(RID p_texture, int p_width, int p_height) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->type != Texture::TYPE_2D);
	tex->width_2d = p_width;
	tex->height_2d = p_height;
}

void RasterizerStorageRD::texture_set_path(RID p_texture, const String &p_path) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	tex->path = p_path;
}

String RasterizerStorageRD::texture_get_path(RID p_texture) const {
	return String();
}

void RasterizerStorageRD::texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	tex->detect_3d_callback_ud = p_userdata;
	tex->detect_3d_callback = p_callback;
}

void RasterizerStorageRD::texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	tex->detect_normal_callback_ud = p_userdata;
	tex->detect_normal_callback = p_callback;
}

void RasterizerStorageRD::texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	tex->detect_roughness_callback_ud = p_userdata;
	tex->detect_roughness_callback = p_callback;
}

void RasterizerStorageRD::texture_debug_usage(List<RS::TextureInfo> *r_info) {
}

void RasterizerStorageRD::texture_set_proxy(RID p_proxy, RID p_base) {
}

void RasterizerStorageRD::texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {
}

Size2 RasterizerStorageRD::texture_size_with_proxy(RID p_proxy) {
	return texture_2d_get_size(p_proxy);
}

/* CANVAS TEXTURE */

void RasterizerStorageRD::CanvasTexture::clear_sets() {
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

RasterizerStorageRD::CanvasTexture::~CanvasTexture() {
	clear_sets();
}

RID RasterizerStorageRD::canvas_texture_create() {
	return canvas_texture_owner.make_rid(memnew(CanvasTexture));
}

void RasterizerStorageRD::canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) {
	CanvasTexture *ct = canvas_texture_owner.getornull(p_canvas_texture);
	switch (p_channel) {
		case RS::CANVAS_TEXTURE_CHANNEL_DIFFUSE: {
			ct->diffuse = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_NORMAL: {
			ct->normalmap = p_texture;
		} break;
		case RS::CANVAS_TEXTURE_CHANNEL_SPECULAR: {
			ct->specular = p_texture;
		} break;
	}

	ct->clear_sets();
}

void RasterizerStorageRD::canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_specular_color, float p_shininess) {
	CanvasTexture *ct = canvas_texture_owner.getornull(p_canvas_texture);
	ct->specular_color.r = p_specular_color.r;
	ct->specular_color.g = p_specular_color.g;
	ct->specular_color.b = p_specular_color.b;
	ct->specular_color.a = p_shininess;
	ct->clear_sets();
}

void RasterizerStorageRD::canvas_texture_set_texture_filter(RID p_canvas_texture, RS::CanvasItemTextureFilter p_filter) {
	CanvasTexture *ct = canvas_texture_owner.getornull(p_canvas_texture);
	ct->texture_filter = p_filter;
	ct->clear_sets();
}

void RasterizerStorageRD::canvas_texture_set_texture_repeat(RID p_canvas_texture, RS::CanvasItemTextureRepeat p_repeat) {
	CanvasTexture *ct = canvas_texture_owner.getornull(p_canvas_texture);
	ct->texture_repeat = p_repeat;
	ct->clear_sets();
}

bool RasterizerStorageRD::canvas_texture_get_unifom_set(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, RID p_base_shader, int p_base_set, RID &r_uniform_set, Size2i &r_size, Color &r_specular_shininess, bool &r_use_normal, bool &r_use_specular) {
	CanvasTexture *ct = nullptr;

	Texture *t = texture_owner.getornull(p_texture);

	if (t) {
		//regular texture
		if (!t->canvas_texture) {
			t->canvas_texture = memnew(CanvasTexture);
			t->canvas_texture->diffuse = p_texture;
		}

		ct = t->canvas_texture;
	} else {
		ct = canvas_texture_owner.getornull(p_texture);
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
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 0;

			t = texture_owner.getornull(ct->diffuse);
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
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1;

			t = texture_owner.getornull(ct->normalmap);
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
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 2;

			t = texture_owner.getornull(ct->specular);
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
			u.type = RD::UNIFORM_TYPE_SAMPLER;
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

RID RasterizerStorageRD::shader_create() {
	Shader shader;
	shader.data = nullptr;
	shader.type = SHADER_TYPE_MAX;

	return shader_owner.make_rid(shader);
}

void RasterizerStorageRD::shader_set_code(RID p_shader, const String &p_code) {
	Shader *shader = shader_owner.getornull(p_shader);
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

		for (Map<StringName, RID>::Element *E = shader->default_texture_parameter.front(); E; E = E->next()) {
			shader->data->set_default_texture_param(E->key(), E->get());
		}
	}

	if (shader->data) {
		shader->data->set_code(p_code);
	}

	for (Set<Material *>::Element *E = shader->owners.front(); E; E = E->next()) {
		Material *material = E->get();
		material->instance_dependency.instance_notify_changed(false, true);
		_material_queue_update(material, true, true);
	}
}

String RasterizerStorageRD::shader_get_code(RID p_shader) const {
	Shader *shader = shader_owner.getornull(p_shader);
	ERR_FAIL_COND_V(!shader, String());
	return shader->code;
}

void RasterizerStorageRD::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const {
	Shader *shader = shader_owner.getornull(p_shader);
	ERR_FAIL_COND(!shader);
	if (shader->data) {
		return shader->data->get_param_list(p_param_list);
	}
}

void RasterizerStorageRD::shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) {
	Shader *shader = shader_owner.getornull(p_shader);
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

RID RasterizerStorageRD::shader_get_default_texture_param(RID p_shader, const StringName &p_name) const {
	Shader *shader = shader_owner.getornull(p_shader);
	ERR_FAIL_COND_V(!shader, RID());
	if (shader->default_texture_parameter.has(p_name)) {
		return shader->default_texture_parameter[p_name];
	}

	return RID();
}

Variant RasterizerStorageRD::shader_get_param_default(RID p_shader, const StringName &p_param) const {
	Shader *shader = shader_owner.getornull(p_shader);
	ERR_FAIL_COND_V(!shader, Variant());
	if (shader->data) {
		return shader->data->get_default_parameter(p_param);
	}
	return Variant();
}

void RasterizerStorageRD::shader_set_data_request_function(ShaderType p_shader_type, ShaderDataRequestFunction p_function) {
	ERR_FAIL_INDEX(p_shader_type, SHADER_TYPE_MAX);
	shader_data_request_func[p_shader_type] = p_function;
}

/* COMMON MATERIAL API */

RID RasterizerStorageRD::material_create() {
	Material material;
	material.data = nullptr;
	material.shader = nullptr;
	material.shader_type = SHADER_TYPE_MAX;
	material.update_next = nullptr;
	material.update_requested = false;
	material.uniform_dirty = false;
	material.texture_dirty = false;
	material.priority = 0;
	RID id = material_owner.make_rid(material);
	{
		Material *material_ptr = material_owner.getornull(id);
		material_ptr->self = id;
	}
	return id;
}

void RasterizerStorageRD::_material_queue_update(Material *material, bool p_uniform, bool p_texture) {
	if (material->update_requested) {
		return;
	}

	material->update_next = material_update_list;
	material_update_list = material;
	material->update_requested = true;
	material->uniform_dirty = material->uniform_dirty || p_uniform;
	material->texture_dirty = material->texture_dirty || p_texture;
}

void RasterizerStorageRD::material_set_shader(RID p_material, RID p_shader) {
	Material *material = material_owner.getornull(p_material);
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
		material->instance_dependency.instance_notify_changed(false, true);
		return;
	}

	Shader *shader = shader_owner.getornull(p_shader);
	ERR_FAIL_COND(!shader);
	material->shader = shader;
	material->shader_type = shader->type;
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
	material->instance_dependency.instance_notify_changed(false, true);
	_material_queue_update(material, true, true);
}

void RasterizerStorageRD::material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) {
	Material *material = material_owner.getornull(p_material);
	ERR_FAIL_COND(!material);

	if (p_value.get_type() == Variant::NIL) {
		material->params.erase(p_param);
	} else {
		material->params[p_param] = p_value;
	}

	if (material->shader && material->shader->data) { //shader is valid
		bool is_texture = material->shader->data->is_param_texture(p_param);
		_material_queue_update(material, !is_texture, is_texture);
	} else {
		_material_queue_update(material, true, true);
	}
}

Variant RasterizerStorageRD::material_get_param(RID p_material, const StringName &p_param) const {
	Material *material = material_owner.getornull(p_material);
	ERR_FAIL_COND_V(!material, Variant());
	if (material->params.has(p_param)) {
		return material->params[p_param];
	} else {
		return Variant();
	}
}

void RasterizerStorageRD::material_set_next_pass(RID p_material, RID p_next_material) {
	Material *material = material_owner.getornull(p_material);
	ERR_FAIL_COND(!material);

	if (material->next_pass == p_next_material) {
		return;
	}

	material->next_pass = p_next_material;
	if (material->data) {
		material->data->set_next_pass(p_next_material);
	}

	material->instance_dependency.instance_notify_changed(false, true);
}

void RasterizerStorageRD::material_set_render_priority(RID p_material, int priority) {
	Material *material = material_owner.getornull(p_material);
	ERR_FAIL_COND(!material);
	material->priority = priority;
	if (material->data) {
		material->data->set_render_priority(priority);
	}
}

bool RasterizerStorageRD::material_is_animated(RID p_material) {
	Material *material = material_owner.getornull(p_material);
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

bool RasterizerStorageRD::material_casts_shadows(RID p_material) {
	Material *material = material_owner.getornull(p_material);
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

void RasterizerStorageRD::material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) {
	Material *material = material_owner.getornull(p_material);
	ERR_FAIL_COND(!material);
	if (material->shader && material->shader->data) {
		material->shader->data->get_instance_param_list(r_parameters);

		if (material->next_pass.is_valid()) {
			material_get_instance_shader_parameters(material->next_pass, r_parameters);
		}
	}
}

void RasterizerStorageRD::material_update_dependency(RID p_material, RasterizerScene::InstanceBase *p_instance) {
	Material *material = material_owner.getornull(p_material);
	ERR_FAIL_COND(!material);
	p_instance->update_dependency(&material->instance_dependency);
	if (material->next_pass.is_valid()) {
		material_update_dependency(material->next_pass, p_instance);
	}
}

void RasterizerStorageRD::material_set_data_request_function(ShaderType p_shader_type, MaterialDataRequestFunction p_function) {
	ERR_FAIL_INDEX(p_shader_type, SHADER_TYPE_MAX);
	material_data_request_func[p_shader_type] = p_function;
}

_FORCE_INLINE_ static void _fill_std140_variant_ubo_value(ShaderLanguage::DataType type, const Variant &value, uint8_t *data, bool p_linear_color) {
	switch (type) {
		case ShaderLanguage::TYPE_BOOL: {
			bool v = value;

			uint32_t *gui = (uint32_t *)data;
			*gui = v ? 1 : 0;
		} break;
		case ShaderLanguage::TYPE_BVEC2: {
			int v = value;
			uint32_t *gui = (uint32_t *)data;
			gui[0] = v & 1 ? 1 : 0;
			gui[1] = v & 2 ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_BVEC3: {
			int v = value;
			uint32_t *gui = (uint32_t *)data;
			gui[0] = (v & 1) ? 1 : 0;
			gui[1] = (v & 2) ? 1 : 0;
			gui[2] = (v & 4) ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_BVEC4: {
			int v = value;
			uint32_t *gui = (uint32_t *)data;
			gui[0] = (v & 1) ? 1 : 0;
			gui[1] = (v & 2) ? 1 : 0;
			gui[2] = (v & 4) ? 1 : 0;
			gui[3] = (v & 8) ? 1 : 0;

		} break;
		case ShaderLanguage::TYPE_INT: {
			int v = value;
			int32_t *gui = (int32_t *)data;
			gui[0] = v;

		} break;
		case ShaderLanguage::TYPE_IVEC2: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 2; i++) {
				if (i < s) {
					gui[i] = r[i];
				} else {
					gui[i] = 0;
				}
			}

		} break;
		case ShaderLanguage::TYPE_IVEC3: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 3; i++) {
				if (i < s) {
					gui[i] = r[i];
				} else {
					gui[i] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_IVEC4: {
			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 4; i++) {
				if (i < s) {
					gui[i] = r[i];
				} else {
					gui[i] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_UINT: {
			int v = value;
			uint32_t *gui = (uint32_t *)data;
			gui[0] = v;

		} break;
		case ShaderLanguage::TYPE_UVEC2: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 2; i++) {
				if (i < s) {
					gui[i] = r[i];
				} else {
					gui[i] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_UVEC3: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 3; i++) {
				if (i < s) {
					gui[i] = r[i];
				} else {
					gui[i] = 0;
				}
			}

		} break;
		case ShaderLanguage::TYPE_UVEC4: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 4; i++) {
				if (i < s) {
					gui[i] = r[i];
				} else {
					gui[i] = 0;
				}
			}
		} break;
		case ShaderLanguage::TYPE_FLOAT: {
			float v = value;
			float *gui = (float *)data;
			gui[0] = v;

		} break;
		case ShaderLanguage::TYPE_VEC2: {
			Vector2 v = value;
			float *gui = (float *)data;
			gui[0] = v.x;
			gui[1] = v.y;

		} break;
		case ShaderLanguage::TYPE_VEC3: {
			Vector3 v = value;
			float *gui = (float *)data;
			gui[0] = v.x;
			gui[1] = v.y;
			gui[2] = v.z;

		} break;
		case ShaderLanguage::TYPE_VEC4: {
			float *gui = (float *)data;

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
			} else if (value.get_type() == Variant::QUAT) {
				Quat v = value;

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
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			Transform2D v = value;
			float *gui = (float *)data;

			//in std140 members of mat2 are treated as vec4s
			gui[0] = v.elements[0][0];
			gui[1] = v.elements[0][1];
			gui[2] = 0;
			gui[3] = 0;
			gui[4] = v.elements[1][0];
			gui[5] = v.elements[1][1];
			gui[6] = 0;
			gui[7] = 0;
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			Basis v = value;
			float *gui = (float *)data;

			gui[0] = v.elements[0][0];
			gui[1] = v.elements[1][0];
			gui[2] = v.elements[2][0];
			gui[3] = 0;
			gui[4] = v.elements[0][1];
			gui[5] = v.elements[1][1];
			gui[6] = v.elements[2][1];
			gui[7] = 0;
			gui[8] = v.elements[0][2];
			gui[9] = v.elements[1][2];
			gui[10] = v.elements[2][2];
			gui[11] = 0;
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			Transform v = value;
			float *gui = (float *)data;

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

_FORCE_INLINE_ static void _fill_std140_ubo_empty(ShaderLanguage::DataType type, uint8_t *data) {
	switch (type) {
		case ShaderLanguage::TYPE_BOOL:
		case ShaderLanguage::TYPE_INT:
		case ShaderLanguage::TYPE_UINT:
		case ShaderLanguage::TYPE_FLOAT: {
			zeromem(data, 4);
		} break;
		case ShaderLanguage::TYPE_BVEC2:
		case ShaderLanguage::TYPE_IVEC2:
		case ShaderLanguage::TYPE_UVEC2:
		case ShaderLanguage::TYPE_VEC2: {
			zeromem(data, 8);
		} break;
		case ShaderLanguage::TYPE_BVEC3:
		case ShaderLanguage::TYPE_IVEC3:
		case ShaderLanguage::TYPE_UVEC3:
		case ShaderLanguage::TYPE_VEC3:
		case ShaderLanguage::TYPE_BVEC4:
		case ShaderLanguage::TYPE_IVEC4:
		case ShaderLanguage::TYPE_UVEC4:
		case ShaderLanguage::TYPE_VEC4: {
			zeromem(data, 16);
		} break;
		case ShaderLanguage::TYPE_MAT2: {
			zeromem(data, 32);
		} break;
		case ShaderLanguage::TYPE_MAT3: {
			zeromem(data, 48);
		} break;
		case ShaderLanguage::TYPE_MAT4: {
			zeromem(data, 64);
		} break;

		default: {
		}
	}
}

void RasterizerStorageRD::MaterialData::update_uniform_buffer(const Map<StringName, ShaderLanguage::ShaderNode::Uniform> &p_uniforms, const uint32_t *p_uniform_offsets, const Map<StringName, Variant> &p_parameters, uint8_t *p_buffer, uint32_t p_buffer_size, bool p_use_linear_color) {
	bool uses_global_buffer = false;

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = p_uniforms.front(); E; E = E->next()) {
		if (E->get().order < 0) {
			continue; // texture, does not go here
		}

		if (E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue; //instance uniforms don't appear in the bufferr
		}

		if (E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL) {
			//this is a global variable, get the index to it
			RasterizerStorageRD *rs = base_singleton;

			GlobalVariables::Variable *gv = rs->global_variables.variables.getptr(E->key());
			uint32_t index = 0;
			if (gv) {
				index = gv->buffer_index;
			} else {
				WARN_PRINT("Shader uses global uniform '" + E->key() + "', but it was removed at some point. Material will not display correctly.");
			}

			uint32_t offset = p_uniform_offsets[E->get().order];
			uint32_t *intptr = (uint32_t *)&p_buffer[offset];
			*intptr = index;
			uses_global_buffer = true;
			continue;
		}

		//regular uniform
		uint32_t offset = p_uniform_offsets[E->get().order];
#ifdef DEBUG_ENABLED
		uint32_t size = ShaderLanguage::get_type_size(E->get().type);
		ERR_CONTINUE(offset + size > p_buffer_size);
#endif
		uint8_t *data = &p_buffer[offset];
		const Map<StringName, Variant>::Element *V = p_parameters.find(E->key());

		if (V) {
			//user provided
			_fill_std140_variant_ubo_value(E->get().type, V->get(), data, p_use_linear_color);

		} else if (E->get().default_value.size()) {
			//default value
			_fill_std140_ubo_value(E->get().type, E->get().default_value, data);
			//value=E->get().default_value;
		} else {
			//zero because it was not provided
			if (E->get().type == ShaderLanguage::TYPE_VEC4 && E->get().hint == ShaderLanguage::ShaderNode::Uniform::HINT_COLOR) {
				//colors must be set as black, with alpha as 1.0
				_fill_std140_variant_ubo_value(E->get().type, Color(0, 0, 0, 1), data, p_use_linear_color);
			} else {
				//else just zero it out
				_fill_std140_ubo_empty(E->get().type, data);
			}
		}
	}

	if (uses_global_buffer != (global_buffer_E != nullptr)) {
		RasterizerStorageRD *rs = base_singleton;
		if (uses_global_buffer) {
			global_buffer_E = rs->global_variables.materials_using_buffer.push_back(self);
		} else {
			rs->global_variables.materials_using_buffer.erase(global_buffer_E);
			global_buffer_E = nullptr;
		}
	}
}

RasterizerStorageRD::MaterialData::~MaterialData() {
	if (global_buffer_E) {
		//unregister global buffers
		RasterizerStorageRD *rs = base_singleton;
		rs->global_variables.materials_using_buffer.erase(global_buffer_E);
	}

	if (global_texture_E) {
		//unregister global textures
		RasterizerStorageRD *rs = base_singleton;

		for (Map<StringName, uint64_t>::Element *E = used_global_textures.front(); E; E = E->next()) {
			GlobalVariables::Variable *v = rs->global_variables.variables.getptr(E->key());
			if (v) {
				v->texture_materials.erase(self);
			}
		}
		//unregister material from those using global textures
		rs->global_variables.materials_using_texture.erase(global_texture_E);
	}
}

void RasterizerStorageRD::MaterialData::update_textures(const Map<StringName, Variant> &p_parameters, const Map<StringName, RID> &p_default_textures, const Vector<ShaderCompilerRD::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures, bool p_use_linear_color) {
	RasterizerStorageRD *singleton = (RasterizerStorageRD *)RasterizerStorage::base_singleton;
#ifdef TOOLS_ENABLED
	Texture *roughness_detect_texture = nullptr;
	RS::TextureDetectRoughnessChannel roughness_channel = RS::TEXTURE_DETECT_ROUGNHESS_R;
	Texture *normal_detect_texture = nullptr;
#endif

	bool uses_global_textures = false;
	global_textures_pass++;

	for (int i = 0; i < p_texture_uniforms.size(); i++) {
		const StringName &uniform_name = p_texture_uniforms[i].name;

		RID texture;

		if (p_texture_uniforms[i].global) {
			RasterizerStorageRD *rs = base_singleton;

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

					texture = v->override.get_type() != Variant::NIL ? v->override : v->value;
				}

			} else {
				WARN_PRINT("Shader uses global uniform texture '" + String(uniform_name) + "', but it was removed at some point. Material will not display correctly.");
			}
		} else {
			if (!texture.is_valid()) {
				const Map<StringName, Variant>::Element *V = p_parameters.find(uniform_name);
				if (V) {
					texture = V->get();
				}
			}

			if (!texture.is_valid()) {
				const Map<StringName, RID>::Element *W = p_default_textures.find(uniform_name);
				if (W) {
					texture = W->get();
				}
			}
		}

		RID rd_texture;

		if (texture.is_null()) {
			//check default usage
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
		} else {
			bool srgb = p_use_linear_color && (p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_ALBEDO || p_texture_uniforms[i].hint == ShaderLanguage::ShaderNode::Uniform::HINT_BLACK_ALBEDO);

			Texture *tex = singleton->texture_owner.getornull(texture);

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
				//wtf
				rd_texture = singleton->texture_rd_get_default(DEFAULT_RD_TEXTURE_WHITE);
			}
		}

		p_textures[i] = rd_texture;
	}
#ifdef TOOLS_ENABLED
	if (roughness_detect_texture && normal_detect_texture && normal_detect_texture->path != String()) {
		roughness_detect_texture->detect_roughness_callback(roughness_detect_texture->detect_roughness_callback_ud, normal_detect_texture->path, roughness_channel);
	}
#endif
	{
		//for textures no longer used, unregister them
		List<Map<StringName, uint64_t>::Element *> to_delete;
		RasterizerStorageRD *rs = base_singleton;

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

void RasterizerStorageRD::material_force_update_textures(RID p_material, ShaderType p_shader_type) {
	Material *material = material_owner.getornull(p_material);
	if (material->shader_type != p_shader_type) {
		return;
	}
	if (material->data) {
		material->data->update_parameters(material->params, false, true);
	}
}

void RasterizerStorageRD::_update_queued_materials() {
	Material *material = material_update_list;
	while (material) {
		Material *next = material->update_next;

		if (material->data) {
			material->data->update_parameters(material->params, material->uniform_dirty, material->texture_dirty);
		}
		material->update_requested = false;
		material->texture_dirty = false;
		material->uniform_dirty = false;
		material->update_next = nullptr;
		material = next;
	}
	material_update_list = nullptr;
}

/* MESH API */

RID RasterizerStorageRD::mesh_create() {
	return mesh_owner.make_rid(Mesh());
}

/// Returns stride
void RasterizerStorageRD::mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND(!mesh);

	//ensure blend shape consistency
	ERR_FAIL_COND(mesh->blend_shape_count && p_surface.blend_shapes.size() != (int)mesh->blend_shape_count);
	ERR_FAIL_COND(mesh->blend_shape_count && p_surface.bone_aabbs.size() != mesh->bone_aabbs.size());

#ifdef DEBUG_ENABLED
	//do a validation, to catch errors first
	{
		uint32_t stride = 0;

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
						if (p_surface.format & RS::ARRAY_COMPRESS_NORMAL) {
							stride += sizeof(int8_t) * 4;
						} else {
							stride += sizeof(float) * 4;
						}

					} break;
					case RS::ARRAY_TANGENT: {
						if (p_surface.format & RS::ARRAY_COMPRESS_TANGENT) {
							stride += sizeof(int8_t) * 4;
						} else {
							stride += sizeof(float) * 4;
						}

					} break;
					case RS::ARRAY_COLOR: {
						if (p_surface.format & RS::ARRAY_COMPRESS_COLOR) {
							stride += sizeof(int8_t) * 4;
						} else {
							stride += sizeof(float) * 4;
						}

					} break;
					case RS::ARRAY_TEX_UV: {
						if (p_surface.format & RS::ARRAY_COMPRESS_TEX_UV) {
							stride += sizeof(int16_t) * 2;
						} else {
							stride += sizeof(float) * 2;
						}

					} break;
					case RS::ARRAY_TEX_UV2: {
						if (p_surface.format & RS::ARRAY_COMPRESS_TEX_UV2) {
							stride += sizeof(int16_t) * 2;
						} else {
							stride += sizeof(float) * 2;
						}

					} break;
					case RS::ARRAY_BONES: {
						//assumed weights too

						//unique format, internally 16 bits, exposed as single array for 32

						stride += sizeof(int32_t) * 4;

					} break;
				}
			}
		}

		int expected_size = stride * p_surface.vertex_count;
		ERR_FAIL_COND_MSG(expected_size != p_surface.vertex_data.size(), "Size of data provided (" + itos(p_surface.vertex_data.size()) + ") does not match expected (" + itos(expected_size) + ")");
	}

#endif

	Mesh::Surface *s = memnew(Mesh::Surface);

	s->format = p_surface.format;
	s->primitive = p_surface.primitive;

	s->vertex_buffer = RD::get_singleton()->vertex_buffer_create(p_surface.vertex_data.size(), p_surface.vertex_data);
	s->vertex_count = p_surface.vertex_count;

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
			}
		}
	}

	s->aabb = p_surface.aabb;
	s->bone_aabbs = p_surface.bone_aabbs; //only really useful for returning them.

	for (int i = 0; i < p_surface.blend_shapes.size(); i++) {
		if (p_surface.blend_shapes[i].size() != p_surface.vertex_data.size()) {
			memdelete(s);
			ERR_FAIL_COND(p_surface.blend_shapes[i].size() != p_surface.vertex_data.size());
		}
		RID vertex_buffer = RD::get_singleton()->vertex_buffer_create(p_surface.blend_shapes[i].size(), p_surface.blend_shapes[i]);
		s->blend_shapes.push_back(vertex_buffer);
	}

	mesh->blend_shape_count = p_surface.blend_shapes.size();

	if (mesh->surface_count == 0) {
		mesh->bone_aabbs = p_surface.bone_aabbs;
		mesh->aabb = p_surface.aabb;
	} else {
		for (int i = 0; i < p_surface.bone_aabbs.size(); i++) {
			mesh->bone_aabbs.write[i].merge_with(p_surface.bone_aabbs[i]);
		}
		mesh->aabb.merge_with(p_surface.aabb);
	}

	s->material = p_surface.material;

	mesh->surfaces = (Mesh::Surface **)memrealloc(mesh->surfaces, sizeof(Mesh::Surface *) * (mesh->surface_count + 1));
	mesh->surfaces[mesh->surface_count] = s;
	mesh->surface_count++;

	mesh->instance_dependency.instance_notify_changed(true, true);

	mesh->material_cache.clear();
}

int RasterizerStorageRD::mesh_get_blend_shape_count(RID p_mesh) const {
	const Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);
	return mesh->blend_shape_count;
}

void RasterizerStorageRD::mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX((int)p_mode, 2);

	mesh->blend_shape_mode = p_mode;
}

RS::BlendShapeMode RasterizerStorageRD::mesh_get_blend_shape_mode(RID p_mesh) const {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND_V(!mesh, RS::BLEND_SHAPE_MODE_NORMALIZED);
	return mesh->blend_shape_mode;
}

void RasterizerStorageRD::mesh_surface_update_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	ERR_FAIL_COND(p_data.size() == 0);
	uint64_t data_size = p_data.size();
	const uint8_t *r = p_data.ptr();

	RD::get_singleton()->buffer_update(mesh->surfaces[p_surface]->vertex_buffer, p_offset, data_size, r);
}

void RasterizerStorageRD::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	mesh->surfaces[p_surface]->material = p_material;

	mesh->instance_dependency.instance_notify_changed(false, true);
	mesh->material_cache.clear();
}

RID RasterizerStorageRD::mesh_surface_get_material(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND_V(!mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());

	return mesh->surfaces[p_surface]->material;
}

RS::SurfaceData RasterizerStorageRD::mesh_get_surface(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND_V(!mesh, RS::SurfaceData());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RS::SurfaceData());

	Mesh::Surface &s = *mesh->surfaces[p_surface];

	RS::SurfaceData sd;
	sd.format = s.format;
	sd.vertex_data = RD::get_singleton()->buffer_get_data(s.vertex_buffer);
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

	for (int i = 0; i < s.blend_shapes.size(); i++) {
		Vector<uint8_t> bs = RD::get_singleton()->buffer_get_data(s.blend_shapes[i]);
		sd.blend_shapes.push_back(bs);
	}

	return sd;
}

int RasterizerStorageRD::mesh_get_surface_count(RID p_mesh) const {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND_V(!mesh, 0);
	return mesh->surface_count;
}

void RasterizerStorageRD::mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND(!mesh);
	mesh->custom_aabb = p_aabb;
}

AABB RasterizerStorageRD::mesh_get_custom_aabb(RID p_mesh) const {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());
	return mesh->custom_aabb;
}

AABB RasterizerStorageRD::mesh_get_aabb(RID p_mesh, RID p_skeleton) {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());

	if (mesh->custom_aabb != AABB()) {
		return mesh->custom_aabb;
	}

	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);

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

					Transform mtx;

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

					Transform mtx;

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

void RasterizerStorageRD::mesh_clear(RID p_mesh) {
	Mesh *mesh = mesh_owner.getornull(p_mesh);
	ERR_FAIL_COND(!mesh);
	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		Mesh::Surface &s = *mesh->surfaces[i];
		RD::get_singleton()->free(s.vertex_buffer); //clears arrays as dependency automatically, including all versions
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

		for (int32_t j = 0; j < s.blend_shapes.size(); j++) {
			RD::get_singleton()->free(s.blend_shapes[j]);
		}

		if (s.blend_shape_base_buffer.is_valid()) {
			RD::get_singleton()->free(s.blend_shape_base_buffer);
		}

		memdelete(mesh->surfaces[i]);
	}
	if (mesh->surfaces) {
		memfree(mesh->surfaces);
	}

	mesh->surfaces = nullptr;
	mesh->surface_count = 0;
	mesh->material_cache.clear();
	mesh->instance_dependency.instance_notify_changed(true, true);
}

void RasterizerStorageRD::_mesh_surface_generate_version_for_input_mask(Mesh::Surface *s, uint32_t p_input_mask) {
	uint32_t version = s->version_count;
	s->version_count++;
	s->versions = (Mesh::Surface::Version *)memrealloc(s->versions, sizeof(Mesh::Surface::Version) * s->version_count);

	Mesh::Surface::Version &v = s->versions[version];

	Vector<RD::VertexAttribute> attributes;
	Vector<RID> buffers;

	uint32_t stride = 0;

	for (int i = 0; i < RS::ARRAY_WEIGHTS; i++) {
		RD::VertexAttribute vd;
		RID buffer;
		vd.location = i;

		if (!(s->format & (1 << i))) {
			// Not supplied by surface, use default value
			buffer = mesh_default_rd_buffers[i];
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
				case RS::ARRAY_BONES: {
					//assumed weights too
					vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
				} break;
			}
		} else {
			//Supplied, use it

			vd.offset = stride;
			vd.stride = 1; //mark that it needs a stride set
			buffer = s->vertex_buffer;

			switch (i) {
				case RS::ARRAY_VERTEX: {
					if (s->format & RS::ARRAY_FLAG_USE_2D_VERTICES) {
						vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
						stride += sizeof(float) * 2;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32B32_SFLOAT;
						stride += sizeof(float) * 3;
					}

				} break;
				case RS::ARRAY_NORMAL: {
					if (s->format & RS::ARRAY_COMPRESS_NORMAL) {
						vd.format = RD::DATA_FORMAT_R8G8B8A8_SNORM;
						stride += sizeof(int8_t) * 4;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
						stride += sizeof(float) * 4;
					}

				} break;
				case RS::ARRAY_TANGENT: {
					if (s->format & RS::ARRAY_COMPRESS_TANGENT) {
						vd.format = RD::DATA_FORMAT_R8G8B8A8_SNORM;
						stride += sizeof(int8_t) * 4;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
						stride += sizeof(float) * 4;
					}

				} break;
				case RS::ARRAY_COLOR: {
					if (s->format & RS::ARRAY_COMPRESS_COLOR) {
						vd.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
						stride += sizeof(int8_t) * 4;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
						stride += sizeof(float) * 4;
					}

				} break;
				case RS::ARRAY_TEX_UV: {
					if (s->format & RS::ARRAY_COMPRESS_TEX_UV) {
						vd.format = RD::DATA_FORMAT_R16G16_SFLOAT;
						stride += sizeof(int16_t) * 2;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
						stride += sizeof(float) * 2;
					}

				} break;
				case RS::ARRAY_TEX_UV2: {
					if (s->format & RS::ARRAY_COMPRESS_TEX_UV2) {
						vd.format = RD::DATA_FORMAT_R16G16_SFLOAT;
						stride += sizeof(int16_t) * 2;
					} else {
						vd.format = RD::DATA_FORMAT_R32G32_SFLOAT;
						stride += sizeof(float) * 2;
					}

				} break;
				case RS::ARRAY_BONES: {
					//assumed weights too

					//unique format, internally 16 bits, exposed as single array for 32

					vd.format = RD::DATA_FORMAT_R32G32B32A32_UINT;
					stride += sizeof(int32_t) * 4;

				} break;
			}
		}

		if (!(p_input_mask & (1 << i))) {
			continue; // Shader does not need this, skip it
		}

		attributes.push_back(vd);
		buffers.push_back(buffer);
	}

	//update final stride
	for (int i = 0; i < attributes.size(); i++) {
		if (attributes[i].stride == 1) {
			attributes.write[i].stride = stride;
		}
	}

	v.input_mask = p_input_mask;
	v.vertex_format = RD::get_singleton()->vertex_format_create(attributes);
	v.vertex_array = RD::get_singleton()->vertex_array_create(s->vertex_count, v.vertex_format, buffers);
}

////////////////// MULTIMESH

RID RasterizerStorageRD::multimesh_create() {
	return multimesh_owner.make_rid(MultiMesh());
}

void RasterizerStorageRD::multimesh_allocate(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors, bool p_use_custom_data) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	if (multimesh->instances == p_instances && multimesh->xform_format == p_transform_format && multimesh->uses_colors == p_use_colors && multimesh->uses_custom_data == p_use_custom_data) {
		return;
	}

	if (multimesh->buffer.is_valid()) {
		RD::get_singleton()->free(multimesh->buffer);
		multimesh->buffer = RID();
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
}

int RasterizerStorageRD::multimesh_get_instance_count(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, 0);
	return multimesh->instances;
}

void RasterizerStorageRD::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

	multimesh->instance_dependency.instance_notify_changed(true, true);
}

#define MULTIMESH_DIRTY_REGION_SIZE 512

void RasterizerStorageRD::_multimesh_make_local(MultiMesh *multimesh) const {
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
				copymem(w, r, buffer.size());
			}
		} else {
			zeromem(w, multimesh->instances * multimesh->stride_cache * sizeof(float));
		}
	}
	uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
	multimesh->data_cache_dirty_regions = memnew_arr(bool, data_cache_dirty_region_count);
	for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
		multimesh->data_cache_dirty_regions[i] = false;
	}
	multimesh->data_cache_used_dirty_regions = 0;
}

void RasterizerStorageRD::_multimesh_mark_dirty(MultiMesh *multimesh, int p_index, bool p_aabb) {
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

void RasterizerStorageRD::_multimesh_mark_all_dirty(MultiMesh *multimesh, bool p_data, bool p_aabb) {
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

void RasterizerStorageRD::_multimesh_re_create_aabb(MultiMesh *multimesh, const float *p_data, int p_instances) {
	ERR_FAIL_COND(multimesh->mesh.is_null());
	AABB aabb;
	AABB mesh_aabb = mesh_get_aabb(multimesh->mesh);
	for (int i = 0; i < p_instances; i++) {
		const float *data = p_data + multimesh->stride_cache * i;
		Transform t;

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

void RasterizerStorageRD::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

void RasterizerStorageRD::multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

void RasterizerStorageRD::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

void RasterizerStorageRD::multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

RID RasterizerStorageRD::multimesh_get_mesh(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, RID());

	return multimesh->mesh;
}

Transform RasterizerStorageRD::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Transform());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Transform());
	ERR_FAIL_COND_V(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_3D, Transform());

	_multimesh_make_local(multimesh);

	Transform t;
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

Transform2D RasterizerStorageRD::multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

Color RasterizerStorageRD::multimesh_instance_get_color(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

Color RasterizerStorageRD::multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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

void RasterizerStorageRD::multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_COND(p_buffer.size() != (multimesh->instances * (int)multimesh->stride_cache));

	{
		const float *r = p_buffer.ptr();
		RD::get_singleton()->buffer_update(multimesh->buffer, 0, p_buffer.size() * sizeof(float), r, false);
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
		multimesh->instance_dependency.instance_notify_changed(true, false);
	}
}

Vector<float> RasterizerStorageRD::multimesh_get_buffer(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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
			copymem(w, r, buffer.size());
		}

		return ret;
	}
}

void RasterizerStorageRD::multimesh_set_visible_instances(RID p_multimesh, int p_visible) {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
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
}

int RasterizerStorageRD::multimesh_get_visible_instances(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, 0);
	return multimesh->visible_instances;
}

AABB RasterizerStorageRD::multimesh_get_aabb(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.getornull(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, AABB());
	if (multimesh->aabb_dirty) {
		const_cast<RasterizerStorageRD *>(this)->_update_dirty_multimeshes();
	}
	return multimesh->aabb;
}

void RasterizerStorageRD::_update_dirty_multimeshes() {
	while (multimesh_dirty_list) {
		MultiMesh *multimesh = multimesh_dirty_list;

		if (multimesh->data_cache.size()) { //may have been cleared, so only process if it exists
			const float *data = multimesh->data_cache.ptr();

			uint32_t visible_instances = multimesh->visible_instances >= 0 ? multimesh->visible_instances : multimesh->instances;

			if (multimesh->data_cache_used_dirty_regions) {
				uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
				uint32_t visible_region_count = (visible_instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;

				uint32_t region_size = multimesh->stride_cache * MULTIMESH_DIRTY_REGION_SIZE * sizeof(float);

				if (multimesh->data_cache_used_dirty_regions > 32 || multimesh->data_cache_used_dirty_regions > visible_region_count / 2) {
					//if there too many dirty regions, or represent the majority of regions, just copy all, else transfer cost piles up too much
					RD::get_singleton()->buffer_update(multimesh->buffer, 0, MIN(visible_region_count * region_size, multimesh->instances * multimesh->stride_cache * sizeof(float)), data, false);
				} else {
					//not that many regions? update them all
					for (uint32_t i = 0; i < visible_region_count; i++) {
						if (multimesh->data_cache_dirty_regions[i]) {
							uint64_t offset = i * region_size;
							uint64_t size = multimesh->stride_cache * multimesh->instances * sizeof(float);
							RD::get_singleton()->buffer_update(multimesh->buffer, offset, MIN(region_size, size - offset), &data[i * region_size], false);
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
				multimesh->instance_dependency.instance_notify_changed(true, false);
			}
		}

		multimesh_dirty_list = multimesh->dirty_list;

		multimesh->dirty_list = nullptr;
		multimesh->dirty = false;
	}

	multimesh_dirty_list = nullptr;
}

/* PARTICLES */

RID RasterizerStorageRD::particles_create() {
	return particles_owner.make_rid(Particles());
}

void RasterizerStorageRD::particles_set_emitting(RID p_particles, bool p_emitting) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->emitting = p_emitting;
}

bool RasterizerStorageRD::particles_get_emitting(RID p_particles) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND_V(!particles, false);

	return particles->emitting;
}

void RasterizerStorageRD::_particles_free_data(Particles *particles) {
	if (!particles->particle_buffer.is_valid()) {
		return;
	}
	RD::get_singleton()->free(particles->particle_buffer);
	RD::get_singleton()->free(particles->frame_params_buffer);
	RD::get_singleton()->free(particles->particle_instance_buffer);
	particles->particles_transforms_buffer_uniform_set = RID();
	particles->particle_buffer = RID();

	if (RD::get_singleton()->uniform_set_is_valid(particles->collision_textures_uniform_set)) {
		RD::get_singleton()->free(particles->collision_textures_uniform_set);
	}

	if (particles->particles_sort_buffer.is_valid()) {
		RD::get_singleton()->free(particles->particles_sort_buffer);
		particles->particles_sort_buffer = RID();
	}

	if (particles->emission_buffer != nullptr) {
		particles->emission_buffer = nullptr;
		particles->emission_buffer_data.clear();
		RD::get_singleton()->free(particles->emission_storage_buffer);
		particles->emission_storage_buffer = RID();
	}
}

void RasterizerStorageRD::particles_set_amount(RID p_particles, int p_amount) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	if (particles->amount == p_amount) {
		return;
	}

	_particles_free_data(particles);

	particles->amount = p_amount;

	if (particles->amount > 0) {
		particles->particle_buffer = RD::get_singleton()->storage_buffer_create(sizeof(ParticleData) * p_amount);
		particles->frame_params_buffer = RD::get_singleton()->storage_buffer_create(sizeof(ParticlesFrameParams) * 1);
		particles->particle_instance_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * 4 * (3 + 1 + 1) * p_amount);
		//needs to clear it

		{
			Vector<RD::Uniform> uniforms;

			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 1;
				u.ids.push_back(particles->particle_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.ids.push_back(particles->particle_instance_buffer);
				uniforms.push_back(u);
			}

			particles->particles_copy_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, 0), 0);
		}
	}

	particles->prev_ticks = 0;
	particles->phase = 0;
	particles->prev_phase = 0;
	particles->clear = true;
}

void RasterizerStorageRD::particles_set_lifetime(RID p_particles, float p_lifetime) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	particles->lifetime = p_lifetime;
}

void RasterizerStorageRD::particles_set_one_shot(RID p_particles, bool p_one_shot) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	particles->one_shot = p_one_shot;
}

void RasterizerStorageRD::particles_set_pre_process_time(RID p_particles, float p_time) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	particles->pre_process_time = p_time;
}
void RasterizerStorageRD::particles_set_explosiveness_ratio(RID p_particles, float p_ratio) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	particles->explosiveness = p_ratio;
}
void RasterizerStorageRD::particles_set_randomness_ratio(RID p_particles, float p_ratio) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	particles->randomness = p_ratio;
}

void RasterizerStorageRD::particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	particles->custom_aabb = p_aabb;
	particles->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::particles_set_speed_scale(RID p_particles, float p_scale) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->speed_scale = p_scale;
}
void RasterizerStorageRD::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->use_local_coords = p_enable;
}

void RasterizerStorageRD::particles_set_fixed_fps(RID p_particles, int p_fps) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->fixed_fps = p_fps;
}

void RasterizerStorageRD::particles_set_fractional_delta(RID p_particles, bool p_enable) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->fractional_delta = p_enable;
}

void RasterizerStorageRD::particles_set_collision_base_size(RID p_particles, float p_size) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->collision_base_size = p_size;
}

void RasterizerStorageRD::particles_set_process_material(RID p_particles, RID p_material) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->process_material = p_material;
}

void RasterizerStorageRD::particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->draw_order = p_order;
}

void RasterizerStorageRD::particles_set_draw_passes(RID p_particles, int p_passes) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->draw_passes.resize(p_passes);
}

void RasterizerStorageRD::particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_INDEX(p_pass, particles->draw_passes.size());
	particles->draw_passes.write[p_pass] = p_mesh;
}

void RasterizerStorageRD::particles_restart(RID p_particles) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->restart_request = true;
}

void RasterizerStorageRD::_particles_allocate_emission_buffer(Particles *particles) {
	ERR_FAIL_COND(particles->emission_buffer != nullptr);

	particles->emission_buffer_data.resize(sizeof(ParticleEmissionBuffer::Data) * particles->amount + sizeof(uint32_t) * 4);
	zeromem(particles->emission_buffer_data.ptrw(), particles->emission_buffer_data.size());
	particles->emission_buffer = (ParticleEmissionBuffer *)particles->emission_buffer_data.ptrw();
	particles->emission_buffer->particle_max = particles->amount;

	particles->emission_storage_buffer = RD::get_singleton()->storage_buffer_create(particles->emission_buffer_data.size(), particles->emission_buffer_data);

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		//will need to be re-created
		RD::get_singleton()->free(particles->particles_material_uniform_set);
		particles->particles_material_uniform_set = RID();
	}
}

void RasterizerStorageRD::particles_set_subemitter(RID p_particles, RID p_subemitter_particles) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);
	ERR_FAIL_COND(p_particles == p_subemitter_particles);

	particles->sub_emitter = p_subemitter_particles;

	if (RD::get_singleton()->uniform_set_is_valid(particles->particles_material_uniform_set)) {
		RD::get_singleton()->free(particles->particles_material_uniform_set);
		particles->particles_material_uniform_set = RID(); //clear and force to re create sub emitting
	}
}

void RasterizerStorageRD::particles_emit(RID p_particles, const Transform &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) {
	Particles *particles = particles_owner.getornull(p_particles);
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

void RasterizerStorageRD::particles_request_process(RID p_particles) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	if (!particles->dirty) {
		particles->dirty = true;
		particles->update_list = particle_update_list;
		particle_update_list = particles;
	}
}

AABB RasterizerStorageRD::particles_get_current_aabb(RID p_particles) {
	const Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());

	Vector<ParticleData> data;
	data.resize(particles->amount);

	Vector<uint8_t> buffer = RD::get_singleton()->buffer_get_data(particles->particle_buffer);

	Transform inv = particles->emission_transform.affine_inverse();

	AABB aabb;
	if (buffer.size()) {
		bool first = true;
		const ParticleData *particle_data = (const ParticleData *)data.ptr();
		for (int i = 0; i < particles->amount; i++) {
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

AABB RasterizerStorageRD::particles_get_aabb(RID p_particles) const {
	const Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND_V(!particles, AABB());

	return particles->custom_aabb;
}

void RasterizerStorageRD::particles_set_emission_transform(RID p_particles, const Transform &p_transform) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->emission_transform = p_transform;
}

int RasterizerStorageRD::particles_get_draw_passes(RID p_particles) const {
	const Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND_V(!particles, 0);

	return particles->draw_passes.size();
}

RID RasterizerStorageRD::particles_get_draw_pass_mesh(RID p_particles, int p_pass) const {
	const Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND_V(!particles, RID());
	ERR_FAIL_INDEX_V(p_pass, particles->draw_passes.size(), RID());

	return particles->draw_passes[p_pass];
}

void RasterizerStorageRD::particles_add_collision(RID p_particles, RasterizerScene::InstanceBase *p_instance) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	ERR_FAIL_COND(p_instance->base_type != RS::INSTANCE_PARTICLES_COLLISION);

	particles->collisions.insert(p_instance);
}

void RasterizerStorageRD::particles_remove_collision(RID p_particles, RasterizerScene::InstanceBase *p_instance) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	particles->collisions.erase(p_instance);
}

void RasterizerStorageRD::_particles_process(Particles *p_particles, float p_delta) {
	if (p_particles->particles_material_uniform_set.is_null() || !RD::get_singleton()->uniform_set_is_valid(p_particles->particles_material_uniform_set)) {
		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.ids.push_back(p_particles->frame_params_buffer);
			uniforms.push_back(u);
		}
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 1;
			u.ids.push_back(p_particles->particle_buffer);
			uniforms.push_back(u);
		}

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
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
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 3;
			Particles *sub_emitter = particles_owner.getornull(p_particles->sub_emitter);
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

	float new_phase = Math::fmod((float)p_particles->phase + (p_delta / p_particles->lifetime) * p_particles->speed_scale, (float)1.0);

	ParticlesFrameParams &frame_params = p_particles->frame_params;

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

	frame_params.time = RasterizerRD::singleton->get_total_time();
	frame_params.delta = p_delta * p_particles->speed_scale;
	frame_params.random_seed = p_particles->random_seed;
	frame_params.explosiveness = p_particles->explosiveness;
	frame_params.randomness = p_particles->randomness;

	if (p_particles->use_local_coords) {
		store_transform(Transform(), frame_params.emission_transform);
	} else {
		store_transform(p_particles->emission_transform, frame_params.emission_transform);
	}

	frame_params.cycle = p_particles->cycle_number;

	{ //collision and attractors

		frame_params.collider_count = 0;
		frame_params.attractor_count = 0;
		frame_params.particle_size = p_particles->collision_base_size;

		RID collision_3d_textures[ParticlesFrameParams::MAX_3D_TEXTURES];
		RID collision_heightmap_texture;

		Transform to_particles;
		if (p_particles->use_local_coords) {
			to_particles = p_particles->emission_transform.affine_inverse();
		}
		uint32_t collision_3d_textures_used = 0;
		for (const Set<RasterizerScene::InstanceBase *>::Element *E = p_particles->collisions.front(); E; E = E->next()) {
			ParticlesCollision *pc = particles_collision_owner.getornull(E->get()->base);
			Transform to_collider = E->get()->transform;
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
				u.type = RD::UNIFORM_TYPE_TEXTURE;
				u.binding = 0;
				for (uint32_t i = 0; i < ParticlesFrameParams::MAX_3D_TEXTURES; i++) {
					RID rd_tex;
					if (i < collision_3d_textures_used) {
						Texture *t = texture_owner.getornull(collision_3d_textures[i]);
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
				u.type = RD::UNIFORM_TYPE_TEXTURE;
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

	push_constant.clear = p_particles->clear;
	push_constant.total_particles = p_particles->amount;
	push_constant.lifetime = p_particles->lifetime;
	push_constant.trail_size = 1;
	push_constant.use_fractional_delta = p_particles->fractional_delta;
	push_constant.sub_emitter_mode = !p_particles->emitting && p_particles->emission_buffer && (p_particles->emission_buffer->particle_count > 0 || p_particles->force_sub_emit);

	p_particles->force_sub_emit = false; //reset

	Particles *sub_emitter = particles_owner.getornull(p_particles->sub_emitter);

	if (sub_emitter && sub_emitter->emission_storage_buffer.is_valid()) {
		//	print_line("updating subemitter buffer");
		int32_t zero[4] = { 0, sub_emitter->amount, 0, 0 };
		RD::get_singleton()->buffer_update(sub_emitter->emission_storage_buffer, 0, sizeof(uint32_t) * 4, zero, true);
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
		RD::get_singleton()->buffer_update(p_particles->emission_storage_buffer, 0, sizeof(uint32_t) * 4 + sizeof(ParticleEmissionBuffer::Data) * p_particles->emission_buffer->particle_count, p_particles->emission_buffer, true);
		p_particles->emission_buffer->particle_count = 0;
	}

	p_particles->clear = false;

	RD::get_singleton()->buffer_update(p_particles->frame_params_buffer, 0, sizeof(ParticlesFrameParams), &frame_params, true);

	ParticlesMaterialData *m = (ParticlesMaterialData *)material_get_data(p_particles->process_material, SHADER_TYPE_PARTICLES);
	if (!m) {
		m = (ParticlesMaterialData *)material_get_data(particles_shader.default_material, SHADER_TYPE_PARTICLES);
	}

	ERR_FAIL_COND(!m);

	//todo should maybe compute all particle systems together?
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, m->shader_data->pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles_shader.base_uniform_set, 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_particles->particles_material_uniform_set, 1);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_particles->collision_textures_uniform_set, 2);

	if (m->uniform_set.is_valid()) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, m->uniform_set, 3);
	}

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ParticlesShader::PushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_particles->amount, 1, 1, 64, 1, 1);

	RD::get_singleton()->compute_list_end();
}

void RasterizerStorageRD::particles_set_view_axis(RID p_particles, const Vector3 &p_axis) {
	Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND(!particles);

	if (particles->draw_order != RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH) {
		return; //uninteresting for other modes
	}

	//copy to sort buffer
	if (particles->particles_sort_buffer == RID()) {
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
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 0;
				u.ids.push_back(particles->particles_sort_buffer);
				uniforms.push_back(u);
			}

			particles->particles_sort_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, ParticlesShader::COPY_MODE_FILL_SORT_BUFFER), 1);
		}
	}

	Vector3 axis = -p_axis; // cameras look to z negative

	if (particles->use_local_coords) {
		axis = particles->emission_transform.basis.xform_inv(axis).normalized();
	}

	ParticlesShader::CopyPushConstant copy_push_constant;
	copy_push_constant.total_particles = particles->amount;
	copy_push_constant.sort_direction[0] = axis.x;
	copy_push_constant.sort_direction[1] = axis.y;
	copy_push_constant.sort_direction[2] = axis.z;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[ParticlesShader::COPY_MODE_FILL_SORT_BUFFER]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_sort_uniform_set, 1);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, particles->amount, 1, 1, 64, 1, 1);

	RD::get_singleton()->compute_list_end();

	effects.sort_buffer(particles->particles_sort_uniform_set, particles->amount);

	compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[ParticlesShader::COPY_MODE_FILL_INSTANCES_WITH_SORT_BUFFER]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_sort_uniform_set, 1);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, particles->amount, 1, 1, 64, 1, 1);

	RD::get_singleton()->compute_list_end();
}

void RasterizerStorageRD::update_particles() {
	while (particle_update_list) {
		//use transform feedback to process particles

		Particles *particles = particle_update_list;

		//take and remove
		particle_update_list = particles->update_list;
		particles->update_list = nullptr;
		particles->dirty = false;

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
			particles->inactive_time += particles->speed_scale * RasterizerRD::singleton->get_frame_delta_time();
			if (particles->inactive_time > particles->lifetime * 1.2) {
				particles->inactive = true;
				continue;
			}
		}

		bool zero_time_scale = Engine::get_singleton()->get_time_scale() <= 0.0;

		if (particles->clear && particles->pre_process_time > 0.0) {
			float frame_time;
			if (particles->fixed_fps > 0)
				frame_time = 1.0 / particles->fixed_fps;
			else
				frame_time = 1.0 / 30.0;

			float todo = particles->pre_process_time;

			while (todo >= 0) {
				_particles_process(particles, frame_time);
				todo -= frame_time;
			}
		}

		if (particles->fixed_fps > 0) {
			float frame_time;
			float decr;
			if (zero_time_scale) {
				frame_time = 0.0;
				decr = 1.0 / particles->fixed_fps;
			} else {
				frame_time = 1.0 / particles->fixed_fps;
				decr = frame_time;
			}
			float delta = RasterizerRD::singleton->get_frame_delta_time();
			if (delta > 0.1) { //avoid recursive stalls if fps goes below 10
				delta = 0.1;
			} else if (delta <= 0.0) { //unlikely but..
				delta = 0.001;
			}
			float todo = particles->frame_remainder + delta;

			while (todo >= frame_time) {
				_particles_process(particles, frame_time);
				todo -= decr;
			}

			particles->frame_remainder = todo;

		} else {
			if (zero_time_scale)
				_particles_process(particles, 0.0);
			else
				_particles_process(particles, RasterizerRD::singleton->get_frame_delta_time());
		}

		//copy particles to instance buffer

		if (particles->draw_order != RS::PARTICLES_DRAW_ORDER_VIEW_DEPTH) {
			ParticlesShader::CopyPushConstant copy_push_constant;
			copy_push_constant.total_particles = particles->amount;

			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, particles_shader.copy_pipelines[ParticlesShader::COPY_MODE_FILL_INSTANCES]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, particles->particles_copy_uniform_set, 0);
			RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy_push_constant, sizeof(ParticlesShader::CopyPushConstant));

			RD::get_singleton()->compute_list_dispatch_threads(compute_list, particles->amount, 1, 1, 64, 1, 1);

			RD::get_singleton()->compute_list_end();
		}

		particles->instance_dependency.instance_notify_changed(true, false); //make sure shadows are updated
	}
}

bool RasterizerStorageRD::particles_is_inactive(RID p_particles) const {
	const Particles *particles = particles_owner.getornull(p_particles);
	ERR_FAIL_COND_V(!particles, false);
	return !particles->emitting && particles->inactive;
}

/* SKY SHADER */

void RasterizerStorageRD::ParticlesShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();

	if (code == String()) {
		return; //just invalid, but no error
	}

	ShaderCompilerRD::GeneratedCode gen_code;
	ShaderCompilerRD::IdentifierActions actions;

	/*
	uses_time = false;

	actions.render_mode_flags["use_half_res_pass"] = &uses_half_res;
	actions.render_mode_flags["use_quarter_res_pass"] = &uses_quarter_res;

	actions.usage_flag_pointers["TIME"] = &uses_time;
*/

	actions.uniforms = &uniforms;

	Error err = base_singleton->particles_shader.compiler.compile(RS::SHADER_PARTICLES, code, &actions, path, gen_code);

	ERR_FAIL_COND(err != OK);

	if (version.is_null()) {
		version = base_singleton->particles_shader.shader.version_create();
	}

	base_singleton->particles_shader.shader.version_set_compute_code(version, gen_code.uniforms, gen_code.compute_global, gen_code.compute, gen_code.defines);
	ERR_FAIL_COND(!base_singleton->particles_shader.shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//update pipelines

	pipeline = RD::get_singleton()->compute_pipeline_create(base_singleton->particles_shader.shader.version_get_shader(version, 0));

	valid = true;
}

void RasterizerStorageRD::ParticlesShaderData::set_default_texture_param(const StringName &p_name, RID p_texture) {
	if (!p_texture.is_valid()) {
		default_texture_params.erase(p_name);
	} else {
		default_texture_params[p_name] = p_texture;
	}
}

void RasterizerStorageRD::ParticlesShaderData::get_param_list(List<PropertyInfo> *p_param_list) const {
	Map<int, StringName> order;

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = uniforms.front(); E; E = E->next()) {
		if (E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_GLOBAL || E->get().scope == ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		if (E->get().texture_order >= 0) {
			order[E->get().texture_order + 100000] = E->key();
		} else {
			order[E->get().order] = E->key();
		}
	}

	for (Map<int, StringName>::Element *E = order.front(); E; E = E->next()) {
		PropertyInfo pi = ShaderLanguage::uniform_to_property_info(uniforms[E->get()]);
		pi.name = E->get();
		p_param_list->push_back(pi);
	}
}

void RasterizerStorageRD::ParticlesShaderData::get_instance_param_list(List<RasterizerStorage::InstanceShaderParam> *p_param_list) const {
	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = uniforms.front(); E; E = E->next()) {
		if (E->get().scope != ShaderLanguage::ShaderNode::Uniform::SCOPE_INSTANCE) {
			continue;
		}

		RasterizerStorage::InstanceShaderParam p;
		p.info = ShaderLanguage::uniform_to_property_info(E->get());
		p.info.name = E->key(); //supply name
		p.index = E->get().instance_index;
		p.default_value = ShaderLanguage::constant_value_to_variant(E->get().default_value, E->get().type, E->get().hint);
		p_param_list->push_back(p);
	}
}

bool RasterizerStorageRD::ParticlesShaderData::is_param_texture(const StringName &p_param) const {
	if (!uniforms.has(p_param)) {
		return false;
	}

	return uniforms[p_param].texture_order >= 0;
}

bool RasterizerStorageRD::ParticlesShaderData::is_animated() const {
	return false;
}

bool RasterizerStorageRD::ParticlesShaderData::casts_shadows() const {
	return false;
}

Variant RasterizerStorageRD::ParticlesShaderData::get_default_parameter(const StringName &p_parameter) const {
	if (uniforms.has(p_parameter)) {
		ShaderLanguage::ShaderNode::Uniform uniform = uniforms[p_parameter];
		Vector<ShaderLanguage::ConstantNode::Value> default_value = uniform.default_value;
		return ShaderLanguage::constant_value_to_variant(default_value, uniform.type, uniform.hint);
	}
	return Variant();
}

RasterizerStorageRD::ParticlesShaderData::ParticlesShaderData() {
	valid = false;
}

RasterizerStorageRD::ParticlesShaderData::~ParticlesShaderData() {
	//pipeline variants will clear themselves if shader is gone
	if (version.is_valid()) {
		base_singleton->particles_shader.shader.version_free(version);
	}
}

RasterizerStorageRD::ShaderData *RasterizerStorageRD::_create_particles_shader_func() {
	ParticlesShaderData *shader_data = memnew(ParticlesShaderData);
	return shader_data;
}

void RasterizerStorageRD::ParticlesMaterialData::update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty) {
	uniform_set_updated = true;

	if ((uint32_t)ubo_data.size() != shader_data->ubo_size) {
		p_uniform_dirty = true;
		if (uniform_buffer.is_valid()) {
			RD::get_singleton()->free(uniform_buffer);
			uniform_buffer = RID();
		}

		ubo_data.resize(shader_data->ubo_size);
		if (ubo_data.size()) {
			uniform_buffer = RD::get_singleton()->uniform_buffer_create(ubo_data.size());
			memset(ubo_data.ptrw(), 0, ubo_data.size()); //clear
		}

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	//check whether buffer changed
	if (p_uniform_dirty && ubo_data.size()) {
		update_uniform_buffer(shader_data->uniforms, shader_data->ubo_offsets.ptr(), p_parameters, ubo_data.ptrw(), ubo_data.size(), false);
		RD::get_singleton()->buffer_update(uniform_buffer, 0, ubo_data.size(), ubo_data.ptrw());
	}

	uint32_t tex_uniform_count = shader_data->texture_uniforms.size();

	if ((uint32_t)texture_cache.size() != tex_uniform_count) {
		texture_cache.resize(tex_uniform_count);
		p_textures_dirty = true;

		//clear previous uniform set
		if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			RD::get_singleton()->free(uniform_set);
			uniform_set = RID();
		}
	}

	if (p_textures_dirty && tex_uniform_count) {
		update_textures(p_parameters, shader_data->default_texture_params, shader_data->texture_uniforms, texture_cache.ptrw(), true);
	}

	if (shader_data->ubo_size == 0 && shader_data->texture_uniforms.size() == 0) {
		// This material does not require an uniform set, so don't create it.
		return;
	}

	if (!p_textures_dirty && uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		//no reason to update uniform set, only UBO (or nothing) was needed to update
		return;
	}

	Vector<RD::Uniform> uniforms;

	{
		if (shader_data->ubo_size) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
			u.binding = 0;
			u.ids.push_back(uniform_buffer);
			uniforms.push_back(u);
		}

		const RID *textures = texture_cache.ptrw();
		for (uint32_t i = 0; i < tex_uniform_count; i++) {
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_TEXTURE;
			u.binding = 1 + i;
			u.ids.push_back(textures[i]);
			uniforms.push_back(u);
		}
	}

	uniform_set = RD::get_singleton()->uniform_set_create(uniforms, base_singleton->particles_shader.shader.version_get_shader(shader_data->version, 0), 3);
}

RasterizerStorageRD::ParticlesMaterialData::~ParticlesMaterialData() {
	if (uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
		RD::get_singleton()->free(uniform_set);
	}

	if (uniform_buffer.is_valid()) {
		RD::get_singleton()->free(uniform_buffer);
	}
}

RasterizerStorageRD::MaterialData *RasterizerStorageRD::_create_particles_material_func(ParticlesShaderData *p_shader) {
	ParticlesMaterialData *material_data = memnew(ParticlesMaterialData);
	material_data->shader_data = p_shader;
	material_data->last_frame = false;
	//update will happen later anyway so do nothing.
	return material_data;
}
////////

/* PARTICLES COLLISION API */

RID RasterizerStorageRD::particles_collision_create() {
	return particles_collision_owner.make_rid(ParticlesCollision());
}

RID RasterizerStorageRD::particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
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
		tf.type = RD::TEXTURE_TYPE_2D;
		tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

		particles_collision->heightfield_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

		Vector<RID> fb_tex;
		fb_tex.push_back(particles_collision->heightfield_texture);
		particles_collision->heightfield_fb = RD::get_singleton()->framebuffer_create(fb_tex);
		particles_collision->heightfield_fb_size = size;
	}

	return particles_collision->heightfield_fb;
}

void RasterizerStorageRD::particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	if (p_type == particles_collision->type) {
		return;
	}

	if (particles_collision->heightfield_texture.is_valid()) {
		RD::get_singleton()->free(particles_collision->heightfield_texture);
		particles_collision->heightfield_texture = RID();
	}
	particles_collision->type = p_type;
	particles_collision->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	particles_collision->cull_mask = p_cull_mask;
}

void RasterizerStorageRD::particles_collision_set_sphere_radius(RID p_particles_collision, float p_radius) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->radius = p_radius;
	particles_collision->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->extents = p_extents;
	particles_collision->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::particles_collision_set_attractor_strength(RID p_particles_collision, float p_strength) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_strength = p_strength;
}

void RasterizerStorageRD::particles_collision_set_attractor_directionality(RID p_particles_collision, float p_directionality) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_directionality = p_directionality;
}

void RasterizerStorageRD::particles_collision_set_attractor_attenuation(RID p_particles_collision, float p_curve) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->attractor_attenuation = p_curve;
}

void RasterizerStorageRD::particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	particles_collision->field_texture = p_texture;
}

void RasterizerStorageRD::particles_collision_height_field_update(RID p_particles_collision) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);
	particles_collision->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND(!particles_collision);

	if (particles_collision->heightfield_resolution == p_resolution) {
		return;
	}

	particles_collision->heightfield_resolution = p_resolution;

	if (particles_collision->heightfield_texture.is_valid()) {
		RD::get_singleton()->free(particles_collision->heightfield_texture);
		particles_collision->heightfield_texture = RID();
	}
}

AABB RasterizerStorageRD::particles_collision_get_aabb(RID p_particles_collision) const {
	ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
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

Vector3 RasterizerStorageRD::particles_collision_get_extents(RID p_particles_collision) const {
	const ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, Vector3());
	return particles_collision->extents;
}

bool RasterizerStorageRD::particles_collision_is_heightfield(RID p_particles_collision) const {
	const ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_particles_collision);
	ERR_FAIL_COND_V(!particles_collision, false);
	return particles_collision->type == RS::PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE;
}

/* SKELETON API */

RID RasterizerStorageRD::skeleton_create() {
	return skeleton_owner.make_rid(Skeleton());
}

void RasterizerStorageRD::_skeleton_make_dirty(Skeleton *skeleton) {
	if (!skeleton->dirty) {
		skeleton->dirty = true;
		skeleton->dirty_list = skeleton_dirty_list;
		skeleton_dirty_list = skeleton;
	}
}

void RasterizerStorageRD::skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton) {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);
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
	}

	if (skeleton->size) {
		skeleton->data.resize(skeleton->size * (skeleton->use_2d ? 8 : 12));
		skeleton->buffer = RD::get_singleton()->storage_buffer_create(skeleton->data.size() * sizeof(float));
		zeromem(skeleton->data.ptrw(), skeleton->data.size() * sizeof(float));

		_skeleton_make_dirty(skeleton);
	}
}

int RasterizerStorageRD::skeleton_get_bone_count(RID p_skeleton) const {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);
	ERR_FAIL_COND_V(!skeleton, 0);

	return skeleton->size;
}

void RasterizerStorageRD::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);

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

Transform RasterizerStorageRD::skeleton_bone_get_transform(RID p_skeleton, int p_bone) const {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);

	ERR_FAIL_COND_V(!skeleton, Transform());
	ERR_FAIL_INDEX_V(p_bone, skeleton->size, Transform());
	ERR_FAIL_COND_V(skeleton->use_2d, Transform());

	const float *dataptr = skeleton->data.ptr() + p_bone * 12;

	Transform t;

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

void RasterizerStorageRD::skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);

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

Transform2D RasterizerStorageRD::skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);

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

void RasterizerStorageRD::skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);

	ERR_FAIL_COND(!skeleton->use_2d);

	skeleton->base_transform_2d = p_base_transform;
}

void RasterizerStorageRD::_update_dirty_skeletons() {
	while (skeleton_dirty_list) {
		Skeleton *skeleton = skeleton_dirty_list;

		if (skeleton->size) {
			RD::get_singleton()->buffer_update(skeleton->buffer, 0, skeleton->data.size() * sizeof(float), skeleton->data.ptr(), false);
		}

		skeleton_dirty_list = skeleton->dirty_list;

		skeleton->instance_dependency.instance_notify_changed(true, false);

		skeleton->dirty = false;
		skeleton->dirty_list = nullptr;
	}

	skeleton_dirty_list = nullptr;
}

/* LIGHT */

RID RasterizerStorageRD::light_create(RS::LightType p_type) {
	Light light;
	light.type = p_type;

	light.param[RS::LIGHT_PARAM_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_INDIRECT_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_SPECULAR] = 0.5;
	light.param[RS::LIGHT_PARAM_RANGE] = 1.0;
	light.param[RS::LIGHT_PARAM_SIZE] = 0.0;
	light.param[RS::LIGHT_PARAM_SPOT_ANGLE] = 45;
	light.param[RS::LIGHT_PARAM_SHADOW_MAX_DISTANCE] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET] = 0.1;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET] = 0.3;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET] = 0.6;
	light.param[RS::LIGHT_PARAM_SHADOW_FADE_START] = 0.8;
	light.param[RS::LIGHT_PARAM_SHADOW_BIAS] = 0.02;
	light.param[RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE] = 20.0;
	light.param[RS::LIGHT_PARAM_TRANSMITTANCE_BIAS] = 0.05;
	light.param[RS::LIGHT_PARAM_SHADOW_VOLUMETRIC_FOG_FADE] = 1.0;

	return light_owner.make_rid(light);
}

void RasterizerStorageRD::light_set_color(RID p_light, const Color &p_color) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->color = p_color;
}

void RasterizerStorageRD::light_set_param(RID p_light, RS::LightParam p_param, float p_value) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_param, RS::LIGHT_PARAM_MAX);

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
			light->instance_dependency.instance_notify_changed(true, false);
		} break;
		default: {
		}
	}

	light->param[p_param] = p_value;
}

void RasterizerStorageRD::light_set_shadow(RID p_light, bool p_enabled) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);
	light->shadow = p_enabled;

	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::light_set_shadow_color(RID p_light, const Color &p_color) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);
	light->shadow_color = p_color;
}

void RasterizerStorageRD::light_set_projector(RID p_light, RID p_texture) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	if (light->projector == p_texture) {
		return;
	}

	if (light->type != RS::LIGHT_DIRECTIONAL && light->projector.is_valid()) {
		texture_remove_from_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
	}

	light->projector = p_texture;

	if (light->type != RS::LIGHT_DIRECTIONAL && light->projector.is_valid()) {
		texture_add_to_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
	}
}

void RasterizerStorageRD::light_set_negative(RID p_light, bool p_enable) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->negative = p_enable;
}

void RasterizerStorageRD::light_set_cull_mask(RID p_light, uint32_t p_mask) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->cull_mask = p_mask;

	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->reverse_cull = p_enabled;

	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->bake_mode = p_bake_mode;

	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->max_sdfgi_cascade = p_cascade;

	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->omni_shadow_mode = p_mode;

	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

RS::LightOmniShadowMode RasterizerStorageRD::light_omni_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_OMNI_SHADOW_CUBE);

	return light->omni_shadow_mode;
}

void RasterizerStorageRD::light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->directional_shadow_mode = p_mode;
	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::light_directional_set_blend_splits(RID p_light, bool p_enable) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->directional_blend_splits = p_enable;
	light->version++;
	light->instance_dependency.instance_notify_changed(true, false);
}

bool RasterizerStorageRD::light_directional_get_blend_splits(RID p_light) const {
	const Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, false);

	return light->directional_blend_splits;
}

RS::LightDirectionalShadowMode RasterizerStorageRD::light_directional_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL);

	return light->directional_shadow_mode;
}

void RasterizerStorageRD::light_directional_set_shadow_depth_range_mode(RID p_light, RS::LightDirectionalShadowDepthRangeMode p_range_mode) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->directional_range_mode = p_range_mode;
}

RS::LightDirectionalShadowDepthRangeMode RasterizerStorageRD::light_directional_get_shadow_depth_range_mode(RID p_light) const {
	const Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL_SHADOW_DEPTH_RANGE_STABLE);

	return light->directional_range_mode;
}

uint32_t RasterizerStorageRD::light_get_max_sdfgi_cascade(RID p_light) {
	const Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->max_sdfgi_cascade;
}

RS::LightBakeMode RasterizerStorageRD::light_get_bake_mode(RID p_light) {
	const Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_BAKE_DISABLED);

	return light->bake_mode;
}

uint64_t RasterizerStorageRD::light_get_version(RID p_light) const {
	const Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->version;
}

AABB RasterizerStorageRD::light_get_aabb(RID p_light) const {
	const Light *light = light_owner.getornull(p_light);
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

RID RasterizerStorageRD::reflection_probe_create() {
	return reflection_probe_owner.make_rid(ReflectionProbe());
}

void RasterizerStorageRD::reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->update_mode = p_mode;
	reflection_probe->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->intensity = p_intensity;
}

void RasterizerStorageRD::reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_mode = p_mode;
}

void RasterizerStorageRD::reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color = p_color;
}

void RasterizerStorageRD::reflection_probe_set_ambient_energy(RID p_probe, float p_energy) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color_energy = p_energy;
}

void RasterizerStorageRD::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->max_distance = p_distance;

	reflection_probe->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	if (reflection_probe->extents == p_extents) {
		return;
	}
	reflection_probe->extents = p_extents;
	reflection_probe->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->origin_offset = p_offset;
	reflection_probe->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->interior = p_enable;
	reflection_probe->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->box_projection = p_enable;
}

void RasterizerStorageRD::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->enable_shadows = p_enable;
	reflection_probe->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->cull_mask = p_layers;
	reflection_probe->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::reflection_probe_set_resolution(RID p_probe, int p_resolution) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);
	ERR_FAIL_COND(p_resolution < 32);

	reflection_probe->resolution = p_resolution;
}

AABB RasterizerStorageRD::reflection_probe_get_aabb(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, AABB());

	AABB aabb;
	aabb.position = -reflection_probe->extents;
	aabb.size = reflection_probe->extents * 2.0;

	return aabb;
}

RS::ReflectionProbeUpdateMode RasterizerStorageRD::reflection_probe_get_update_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_UPDATE_ALWAYS);

	return reflection_probe->update_mode;
}

uint32_t RasterizerStorageRD::reflection_probe_get_cull_mask(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->cull_mask;
}

Vector3 RasterizerStorageRD::reflection_probe_get_extents(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->extents;
}

Vector3 RasterizerStorageRD::reflection_probe_get_origin_offset(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->origin_offset;
}

bool RasterizerStorageRD::reflection_probe_renders_shadows(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->enable_shadows;
}

float RasterizerStorageRD::reflection_probe_get_origin_max_distance(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->max_distance;
}

int RasterizerStorageRD::reflection_probe_get_resolution(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->resolution;
}

float RasterizerStorageRD::reflection_probe_get_intensity(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->intensity;
}

bool RasterizerStorageRD::reflection_probe_is_interior(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->interior;
}

bool RasterizerStorageRD::reflection_probe_is_box_projection(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->box_projection;
}

RS::ReflectionProbeAmbientMode RasterizerStorageRD::reflection_probe_get_ambient_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_AMBIENT_DISABLED);
	return reflection_probe->ambient_mode;
}

Color RasterizerStorageRD::reflection_probe_get_ambient_color(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Color());

	return reflection_probe->ambient_color;
}
float RasterizerStorageRD::reflection_probe_get_ambient_color_energy(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->ambient_color_energy;
}

RID RasterizerStorageRD::decal_create() {
	return decal_owner.make_rid(Decal());
}

void RasterizerStorageRD::decal_set_extents(RID p_decal, const Vector3 &p_extents) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->extents = p_extents;
	decal->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) {
	Decal *decal = decal_owner.getornull(p_decal);
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

	decal->instance_dependency.instance_notify_changed(false, true);
}

void RasterizerStorageRD::decal_set_emission_energy(RID p_decal, float p_energy) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->emission_energy = p_energy;
}

void RasterizerStorageRD::decal_set_albedo_mix(RID p_decal, float p_mix) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->albedo_mix = p_mix;
}

void RasterizerStorageRD::decal_set_modulate(RID p_decal, const Color &p_modulate) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->modulate = p_modulate;
}

void RasterizerStorageRD::decal_set_cull_mask(RID p_decal, uint32_t p_layers) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->cull_mask = p_layers;
	decal->instance_dependency.instance_notify_changed(true, false);
}

void RasterizerStorageRD::decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->distance_fade = p_enabled;
	decal->distance_fade_begin = p_begin;
	decal->distance_fade_length = p_length;
}

void RasterizerStorageRD::decal_set_fade(RID p_decal, float p_above, float p_below) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->upper_fade = p_above;
	decal->lower_fade = p_below;
}

void RasterizerStorageRD::decal_set_normal_fade(RID p_decal, float p_fade) {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND(!decal);
	decal->normal_fade = p_fade;
}

AABB RasterizerStorageRD::decal_get_aabb(RID p_decal) const {
	Decal *decal = decal_owner.getornull(p_decal);
	ERR_FAIL_COND_V(!decal, AABB());

	return AABB(-decal->extents, decal->extents * 2.0);
}

RID RasterizerStorageRD::gi_probe_create() {
	return gi_probe_owner.make_rid(GIProbe());
}

void RasterizerStorageRD::gi_probe_allocate(RID p_gi_probe, const Transform &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	if (gi_probe->octree_buffer.is_valid()) {
		RD::get_singleton()->free(gi_probe->octree_buffer);
		RD::get_singleton()->free(gi_probe->data_buffer);
		if (gi_probe->sdf_texture.is_valid()) {
			RD::get_singleton()->free(gi_probe->sdf_texture);
		}

		gi_probe->sdf_texture = RID();
		gi_probe->octree_buffer = RID();
		gi_probe->data_buffer = RID();
		gi_probe->octree_buffer_size = 0;
		gi_probe->data_buffer_size = 0;
		gi_probe->cell_count = 0;
	}

	gi_probe->to_cell_xform = p_to_cell_xform;
	gi_probe->bounds = p_aabb;
	gi_probe->octree_size = p_octree_size;
	gi_probe->level_counts = p_level_counts;

	if (p_octree_cells.size()) {
		ERR_FAIL_COND(p_octree_cells.size() % 32 != 0); //cells size must be a multiple of 32

		uint32_t cell_count = p_octree_cells.size() / 32;

		ERR_FAIL_COND(p_data_cells.size() != (int)cell_count * 16); //see that data size matches

		gi_probe->cell_count = cell_count;
		gi_probe->octree_buffer = RD::get_singleton()->storage_buffer_create(p_octree_cells.size(), p_octree_cells);
		gi_probe->octree_buffer_size = p_octree_cells.size();
		gi_probe->data_buffer = RD::get_singleton()->storage_buffer_create(p_data_cells.size(), p_data_cells);
		gi_probe->data_buffer_size = p_data_cells.size();

		if (p_distance_field.size()) {
			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8_UNORM;
			tf.width = gi_probe->octree_size.x;
			tf.height = gi_probe->octree_size.y;
			tf.depth = gi_probe->octree_size.z;
			tf.type = RD::TEXTURE_TYPE_3D;
			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
			Vector<Vector<uint8_t>> s;
			s.push_back(p_distance_field);
			gi_probe->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView(), s);
		}
#if 0
			{
				RD::TextureFormat tf;
				tf.format = RD::DATA_FORMAT_R8_UNORM;
				tf.width = gi_probe->octree_size.x;
				tf.height = gi_probe->octree_size.y;
				tf.depth = gi_probe->octree_size.z;
				tf.type = RD::TEXTURE_TYPE_3D;
				tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UNORM);
				tf.shareable_formats.push_back(RD::DATA_FORMAT_R8_UINT);
				gi_probe->sdf_texture = RD::get_singleton()->texture_create(tf, RD::TextureView());
			}
			RID shared_tex;
			{

				RD::TextureView tv;
				tv.format_override = RD::DATA_FORMAT_R8_UINT;
				shared_tex = RD::get_singleton()->texture_create_shared(tv, gi_probe->sdf_texture);
			}
			//update SDF texture
			Vector<RD::Uniform> uniforms;
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 1;
				u.ids.push_back(gi_probe->octree_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
				u.binding = 2;
				u.ids.push_back(gi_probe->data_buffer);
				uniforms.push_back(u);
			}
			{
				RD::Uniform u;
				u.type = RD::UNIFORM_TYPE_IMAGE;
				u.binding = 3;
				u.ids.push_back(shared_tex);
				uniforms.push_back(u);
			}

			RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, giprobe_sdf_shader_version_shader, 0);

			{
				uint32_t push_constant[4] = { 0, 0, 0, 0 };

				for (int i = 0; i < gi_probe->level_counts.size() - 1; i++) {
					push_constant[0] += gi_probe->level_counts[i];
				}
				push_constant[1] = push_constant[0] + gi_probe->level_counts[gi_probe->level_counts.size() - 1];

				print_line("offset: " + itos(push_constant[0]));
				print_line("size: " + itos(push_constant[1]));
				//create SDF
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, giprobe_sdf_shader_pipeline);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, push_constant, sizeof(uint32_t) * 4);
				RD::get_singleton()->compute_list_dispatch(compute_list, gi_probe->octree_size.x / 4, gi_probe->octree_size.y / 4, gi_probe->octree_size.z / 4);
				RD::get_singleton()->compute_list_end();
			}

			RD::get_singleton()->free(uniform_set);
			RD::get_singleton()->free(shared_tex);
		}
#endif
	}

	gi_probe->version++;
	gi_probe->data_version++;

	gi_probe->instance_dependency.instance_notify_changed(true, false);
}

AABB RasterizerStorageRD::gi_probe_get_bounds(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, AABB());

	return gi_probe->bounds;
}

Vector3i RasterizerStorageRD::gi_probe_get_octree_size(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, Vector3i());
	return gi_probe->octree_size;
}

Vector<uint8_t> RasterizerStorageRD::gi_probe_get_octree_cells(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, Vector<uint8_t>());

	if (gi_probe->octree_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(gi_probe->octree_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RasterizerStorageRD::gi_probe_get_data_cells(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, Vector<uint8_t>());

	if (gi_probe->data_buffer.is_valid()) {
		return RD::get_singleton()->buffer_get_data(gi_probe->data_buffer);
	}
	return Vector<uint8_t>();
}

Vector<uint8_t> RasterizerStorageRD::gi_probe_get_distance_field(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, Vector<uint8_t>());

	if (gi_probe->data_buffer.is_valid()) {
		return RD::get_singleton()->texture_get_data(gi_probe->sdf_texture, 0);
	}
	return Vector<uint8_t>();
}

Vector<int> RasterizerStorageRD::gi_probe_get_level_counts(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, Vector<int>());

	return gi_probe->level_counts;
}

Transform RasterizerStorageRD::gi_probe_get_to_cell_xform(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, Transform());

	return gi_probe->to_cell_xform;
}

void RasterizerStorageRD::gi_probe_set_dynamic_range(RID p_gi_probe, float p_range) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->dynamic_range = p_range;
	gi_probe->version++;
}

float RasterizerStorageRD::gi_probe_get_dynamic_range(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);

	return gi_probe->dynamic_range;
}

void RasterizerStorageRD::gi_probe_set_propagation(RID p_gi_probe, float p_range) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->propagation = p_range;
	gi_probe->version++;
}

float RasterizerStorageRD::gi_probe_get_propagation(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->propagation;
}

void RasterizerStorageRD::gi_probe_set_energy(RID p_gi_probe, float p_energy) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->energy = p_energy;
}

float RasterizerStorageRD::gi_probe_get_energy(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->energy;
}

void RasterizerStorageRD::gi_probe_set_ao(RID p_gi_probe, float p_ao) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->ao = p_ao;
}

float RasterizerStorageRD::gi_probe_get_ao(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->ao;
}

void RasterizerStorageRD::gi_probe_set_ao_size(RID p_gi_probe, float p_strength) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->ao_size = p_strength;
}

float RasterizerStorageRD::gi_probe_get_ao_size(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->ao_size;
}

void RasterizerStorageRD::gi_probe_set_bias(RID p_gi_probe, float p_bias) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->bias = p_bias;
}

float RasterizerStorageRD::gi_probe_get_bias(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->bias;
}

void RasterizerStorageRD::gi_probe_set_normal_bias(RID p_gi_probe, float p_normal_bias) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->normal_bias = p_normal_bias;
}

float RasterizerStorageRD::gi_probe_get_normal_bias(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->normal_bias;
}

void RasterizerStorageRD::gi_probe_set_anisotropy_strength(RID p_gi_probe, float p_strength) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->anisotropy_strength = p_strength;
}

float RasterizerStorageRD::gi_probe_get_anisotropy_strength(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->anisotropy_strength;
}

void RasterizerStorageRD::gi_probe_set_interior(RID p_gi_probe, bool p_enable) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->interior = p_enable;
}

void RasterizerStorageRD::gi_probe_set_use_two_bounces(RID p_gi_probe, bool p_enable) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND(!gi_probe);

	gi_probe->use_two_bounces = p_enable;
	gi_probe->version++;
}

bool RasterizerStorageRD::gi_probe_is_using_two_bounces(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, false);
	return gi_probe->use_two_bounces;
}

bool RasterizerStorageRD::gi_probe_is_interior(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->interior;
}

uint32_t RasterizerStorageRD::gi_probe_get_version(RID p_gi_probe) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->version;
}

uint32_t RasterizerStorageRD::gi_probe_get_data_version(RID p_gi_probe) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, 0);
	return gi_probe->data_version;
}

RID RasterizerStorageRD::gi_probe_get_octree_buffer(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, RID());
	return gi_probe->octree_buffer;
}

RID RasterizerStorageRD::gi_probe_get_data_buffer(RID p_gi_probe) const {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, RID());
	return gi_probe->data_buffer;
}

RID RasterizerStorageRD::gi_probe_get_sdf_texture(RID p_gi_probe) {
	GIProbe *gi_probe = gi_probe_owner.getornull(p_gi_probe);
	ERR_FAIL_COND_V(!gi_probe, RID());

	return gi_probe->sdf_texture;
}

/* LIGHTMAP API */

RID RasterizerStorageRD::lightmap_create() {
	return lightmap_owner.make_rid(Lightmap());
}

void RasterizerStorageRD::lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND(!lm);

	lightmap_array_version++;

	//erase lightmap users
	if (lm->light_texture.is_valid()) {
		Texture *t = texture_owner.getornull(lm->light_texture);
		if (t) {
			t->lightmap_users.erase(p_lightmap);
		}
	}

	Texture *t = texture_owner.getornull(p_light);
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

void RasterizerStorageRD::lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->bounds = p_bounds;
}

void RasterizerStorageRD::lightmap_set_probe_interior(RID p_lightmap, bool p_interior) {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->interior = p_interior;
}

void RasterizerStorageRD::lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
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

PackedVector3Array RasterizerStorageRD::lightmap_get_probe_capture_points(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedVector3Array());

	return lm->points;
}

PackedColorArray RasterizerStorageRD::lightmap_get_probe_capture_sh(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedColorArray());
	return lm->point_sh;
}

PackedInt32Array RasterizerStorageRD::lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->tetrahedra;
}

PackedInt32Array RasterizerStorageRD::lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->bsp_tree;
}

void RasterizerStorageRD::lightmap_set_probe_capture_update_speed(float p_speed) {
	lightmap_probe_capture_update_speed = p_speed;
}

void RasterizerStorageRD::lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) {
	Lightmap *lm = lightmap_owner.getornull(p_lightmap);
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

bool RasterizerStorageRD::lightmap_is_interior(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND_V(!lm, false);
	return lm->interior;
}

AABB RasterizerStorageRD::lightmap_get_aabb(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.getornull(p_lightmap);
	ERR_FAIL_COND_V(!lm, AABB());
	return lm->bounds;
}

/* RENDER TARGET API */

void RasterizerStorageRD::_clear_render_target(RenderTarget *rt) {
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

	rt->framebuffer = RID();
	rt->color = RID();
}

void RasterizerStorageRD::_update_render_target(RenderTarget *rt) {
	if (rt->texture.is_null()) {
		//create a placeholder until updated
		rt->texture = texture_2d_placeholder_create();
		Texture *tex = texture_owner.getornull(rt->texture);
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
		rd_format.array_layers = 1;
		rd_format.mipmaps = 1;
		rd_format.type = RD::TEXTURE_TYPE_2D;
		rd_format.samples = RD::TEXTURE_SAMPLES_1;
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;
		rd_format.shareable_formats.push_back(rt->color_format);
		rd_format.shareable_formats.push_back(rt->color_format_srgb);
	}

	rt->color = RD::get_singleton()->texture_create(rd_format, rd_view);
	ERR_FAIL_COND(rt->color.is_null());

	Vector<RID> fb_textures;
	fb_textures.push_back(rt->color);
	rt->framebuffer = RD::get_singleton()->framebuffer_create(fb_textures);
	if (rt->framebuffer.is_null()) {
		_clear_render_target(rt);
		ERR_FAIL_COND(rt->framebuffer.is_null());
	}

	{ //update texture

		Texture *tex = texture_owner.getornull(rt->texture);

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

void RasterizerStorageRD::_create_render_target_backbuffer(RenderTarget *rt) {
	ERR_FAIL_COND(rt->backbuffer.is_valid());

	uint32_t mipmaps_required = Image::get_image_required_mipmaps(rt->size.width, rt->size.height, Image::FORMAT_RGBA8);
	RD::TextureFormat tf;
	tf.format = rt->color_format;
	tf.width = rt->size.width;
	tf.height = rt->size.height;
	tf.type = RD::TEXTURE_TYPE_2D;
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

RID RasterizerStorageRD::render_target_create() {
	RenderTarget render_target;

	render_target.was_used = false;
	render_target.clear_requested = false;

	for (int i = 0; i < RENDER_TARGET_FLAG_MAX; i++) {
		render_target.flags[i] = false;
	}
	_update_render_target(&render_target);
	return render_target_owner.make_rid(render_target);
}

void RasterizerStorageRD::render_target_set_position(RID p_render_target, int p_x, int p_y) {
	//unused for this render target
}

void RasterizerStorageRD::render_target_set_size(RID p_render_target, int p_width, int p_height) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->size.x = p_width;
	rt->size.y = p_height;
	_update_render_target(rt);
}

RID RasterizerStorageRD::render_target_get_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->texture;
}

void RasterizerStorageRD::render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {
}

void RasterizerStorageRD::render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->flags[p_flag] = p_value;
	_update_render_target(rt);
}

bool RasterizerStorageRD::render_target_was_used(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->was_used;
}

void RasterizerStorageRD::render_target_set_as_unused(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->was_used = false;
}

Size2 RasterizerStorageRD::render_target_get_size(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, Size2());

	return rt->size;
}

RID RasterizerStorageRD::render_target_get_rd_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->framebuffer;
}

RID RasterizerStorageRD::render_target_get_rd_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	return rt->color;
}

RID RasterizerStorageRD::render_target_get_rd_backbuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer;
}

RID RasterizerStorageRD::render_target_get_rd_backbuffer_framebuffer(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	return rt->backbuffer_fb;
}

void RasterizerStorageRD::render_target_request_clear(RID p_render_target, const Color &p_clear_color) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = true;
	rt->clear_color = p_clear_color;
}

bool RasterizerStorageRD::render_target_is_clear_requested(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->clear_requested;
}

Color RasterizerStorageRD::render_target_get_clear_request_color(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, Color());
	return rt->clear_color;
}

void RasterizerStorageRD::render_target_disable_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = false;
}

void RasterizerStorageRD::render_target_do_clear_request(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
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

void RasterizerStorageRD::render_target_copy_to_back_buffer(RID p_render_target, const Rect2i &p_region, bool p_gen_mipmaps) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).clip(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//single texture copy for backbuffer
	//RD::get_singleton()->texture_copy(rt->color, rt->backbuffer_mipmap0, Vector3(region.position.x, region.position.y, 0), Vector3(region.position.x, region.position.y, 0), Vector3(region.size.x, region.size.y, 1), 0, 0, 0, 0, true);
	effects.copy_to_rect(rt->color, rt->backbuffer_mipmap0, region, false, false, false, true, true);

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
		effects.gaussian_blur(prev_texture, mm.mipmap, mm.mipmap_copy, region, true);
		prev_texture = mm.mipmap;
	}
}

void RasterizerStorageRD::render_target_clear_back_buffer(RID p_render_target, const Rect2i &p_region, const Color &p_color) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).clip(p_region);
		if (region.size == Size2i()) {
			return; //nothing to do
		}
	}

	//single texture copy for backbuffer
	effects.set_color(rt->backbuffer_mipmap0, p_color, region, true);
}

void RasterizerStorageRD::render_target_gen_back_buffer_mipmaps(RID p_render_target, const Rect2i &p_region) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region;
	if (p_region == Rect2i()) {
		region.size = rt->size;
	} else {
		region = Rect2i(Size2i(), rt->size).clip(p_region);
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
		effects.gaussian_blur(prev_texture, mm.mipmap, mm.mipmap_copy, region, true);
		prev_texture = mm.mipmap;
	}
}

RID RasterizerStorageRD::render_target_get_framebuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->framebuffer_uniform_set;
}
RID RasterizerStorageRD::render_target_get_backbuffer_uniform_set(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());
	return rt->backbuffer_uniform_set;
}

void RasterizerStorageRD::render_target_set_framebuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->framebuffer_uniform_set = p_uniform_set;
}
void RasterizerStorageRD::render_target_set_backbuffer_uniform_set(RID p_render_target, RID p_uniform_set) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->backbuffer_uniform_set = p_uniform_set;
}

void RasterizerStorageRD::base_update_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) {
	if (mesh_owner.owns(p_base)) {
		Mesh *mesh = mesh_owner.getornull(p_base);
		p_instance->update_dependency(&mesh->instance_dependency);
	} else if (multimesh_owner.owns(p_base)) {
		MultiMesh *multimesh = multimesh_owner.getornull(p_base);
		p_instance->update_dependency(&multimesh->instance_dependency);
		if (multimesh->mesh.is_valid()) {
			base_update_dependency(multimesh->mesh, p_instance);
		}
	} else if (reflection_probe_owner.owns(p_base)) {
		ReflectionProbe *rp = reflection_probe_owner.getornull(p_base);
		p_instance->update_dependency(&rp->instance_dependency);
	} else if (decal_owner.owns(p_base)) {
		Decal *decal = decal_owner.getornull(p_base);
		p_instance->update_dependency(&decal->instance_dependency);
	} else if (gi_probe_owner.owns(p_base)) {
		GIProbe *gip = gi_probe_owner.getornull(p_base);
		p_instance->update_dependency(&gip->instance_dependency);
	} else if (lightmap_owner.owns(p_base)) {
		Lightmap *lm = lightmap_owner.getornull(p_base);
		p_instance->update_dependency(&lm->instance_dependency);
	} else if (light_owner.owns(p_base)) {
		Light *l = light_owner.getornull(p_base);
		p_instance->update_dependency(&l->instance_dependency);
	} else if (particles_owner.owns(p_base)) {
		Particles *p = particles_owner.getornull(p_base);
		p_instance->update_dependency(&p->instance_dependency);
	} else if (particles_collision_owner.owns(p_base)) {
		ParticlesCollision *pc = particles_collision_owner.getornull(p_base);
		p_instance->update_dependency(&pc->instance_dependency);
	}
}

void RasterizerStorageRD::skeleton_update_dependency(RID p_skeleton, RasterizerScene::InstanceBase *p_instance) {
	Skeleton *skeleton = skeleton_owner.getornull(p_skeleton);
	ERR_FAIL_COND(!skeleton);

	p_instance->update_dependency(&skeleton->instance_dependency);
}

RS::InstanceType RasterizerStorageRD::get_base_type(RID p_rid) const {
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
	if (gi_probe_owner.owns(p_rid)) {
		return RS::INSTANCE_GI_PROBE;
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

	return RS::INSTANCE_NONE;
}

void RasterizerStorageRD::texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
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

void RasterizerStorageRD::texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp) {
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

RID RasterizerStorageRD::decal_atlas_get_texture() const {
	return decal_atlas.texture;
}

RID RasterizerStorageRD::decal_atlas_get_texture_srgb() const {
	return decal_atlas.texture_srgb;
}

void RasterizerStorageRD::_update_decal_atlas() {
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

			Texture *src_tex = texture_owner.getornull(*K);

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
			zeromem(v_offsets, sizeof(int) * base_size);

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
	tformat.type = RD::TEXTURE_TYPE_2D;
	tformat.mipmaps = decal_atlas.mipmaps;
	tformat.shareable_formats.push_back(RD::DATA_FORMAT_R8G8B8A8_UNORM);
	tformat.shareable_formats.push_back(RD::DATA_FORMAT_R8G8B8A8_SRGB);

	decal_atlas.texture = RD::get_singleton()->texture_create(tformat, RD::TextureView());

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
					Texture *src_tex = texture_owner.getornull(*K);
					effects.copy_to_atlas_fb(src_tex->rd_texture, mm.fb, t->uv_rect, draw_list, false, t->panorama_to_dp_users > 0);
				}

				RD::get_singleton()->draw_list_end();

				prev_texture = mm.texture;
			} else {
				effects.copy_to_fb_rect(prev_texture, mm.fb, Rect2i(Point2i(), mm.size));
				prev_texture = mm.texture;
			}
		} else {
			RD::get_singleton()->texture_clear(mm.texture, clear_color, 0, 1, 0, 1, false);
		}
	}
}

int32_t RasterizerStorageRD::_global_variable_allocate(uint32_t p_elements) {
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

void RasterizerStorageRD::_global_variable_store_in_buffer(int32_t p_index, RS::GlobalVariableType p_type, const Variant &p_value) {
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
			Transform v = p_value;
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

void RasterizerStorageRD::_global_variable_mark_buffer_dirty(int32_t p_index, int32_t p_elements) {
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

void RasterizerStorageRD::global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) {
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

void RasterizerStorageRD::global_variable_remove(const StringName &p_name) {
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

Vector<StringName> RasterizerStorageRD::global_variable_get_list() const {
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

void RasterizerStorageRD::global_variable_set(const StringName &p_name, const Variant &p_value) {
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
				Material *material = material_owner.getornull(E->get());
				ERR_CONTINUE(!material);
				_material_queue_update(material, false, true);
			}
		}
	}
}

void RasterizerStorageRD::global_variable_set_override(const StringName &p_name, const Variant &p_value) {
	if (!global_variables.variables.has(p_name)) {
		return; //variable may not exist
	}
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
		//texture
		for (Set<RID>::Element *E = gv.texture_materials.front(); E; E = E->next()) {
			Material *material = material_owner.getornull(E->get());
			ERR_CONTINUE(!material);
			_material_queue_update(material, false, true);
		}
	}
}

Variant RasterizerStorageRD::global_variable_get(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(Variant(), "This function should never be used outside the editor, it can severely damage performance.");
	}

	if (!global_variables.variables.has(p_name)) {
		return Variant();
	}

	return global_variables.variables[p_name].value;
}

RS::GlobalVariableType RasterizerStorageRD::global_variable_get_type_internal(const StringName &p_name) const {
	if (!global_variables.variables.has(p_name)) {
		return RS::GLOBAL_VAR_TYPE_MAX;
	}

	return global_variables.variables[p_name].type;
}

RS::GlobalVariableType RasterizerStorageRD::global_variable_get_type(const StringName &p_name) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_FAIL_V_MSG(RS::GLOBAL_VAR_TYPE_MAX, "This function should never be used outside the editor, it can severely damage performance.");
	}

	return global_variable_get_type_internal(p_name);
}

void RasterizerStorageRD::global_variables_load_settings(bool p_load_textures) {
	List<PropertyInfo> settings;
	ProjectSettings::get_singleton()->get_property_list(&settings);

	for (List<PropertyInfo>::Element *E = settings.front(); E; E = E->next()) {
		if (E->get().name.begins_with("shader_globals/")) {
			StringName name = E->get().name.get_slice("/", 1);
			Dictionary d = ProjectSettings::get_singleton()->get(E->get().name);

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

void RasterizerStorageRD::global_variables_clear() {
	global_variables.variables.clear(); //not right but for now enough
}

RID RasterizerStorageRD::global_variables_get_storage_buffer() const {
	return global_variables.buffer;
}

int32_t RasterizerStorageRD::global_variables_instance_allocate(RID p_instance) {
	ERR_FAIL_COND_V(global_variables.instance_buffer_pos.has(p_instance), -1);
	int32_t pos = _global_variable_allocate(ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES);
	global_variables.instance_buffer_pos[p_instance] = pos; //save anyway
	ERR_FAIL_COND_V_MSG(pos < 0, -1, "Too many instances using shader instance variables. Increase buffer size in Project Settings.");
	global_variables.buffer_usage[pos].elements = ShaderLanguage::MAX_INSTANCE_UNIFORM_INDICES;
	return pos;
}

void RasterizerStorageRD::global_variables_instance_free(RID p_instance) {
	ERR_FAIL_COND(!global_variables.instance_buffer_pos.has(p_instance));
	int32_t pos = global_variables.instance_buffer_pos[p_instance];
	if (pos >= 0) {
		global_variables.buffer_usage[pos].elements = 0;
	}
	global_variables.instance_buffer_pos.erase(p_instance);
}

void RasterizerStorageRD::global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) {
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

	_fill_std140_variant_ubo_value(datatype, p_value, (uint8_t *)&global_variables.buffer_values[pos], true); //instances always use linear color in this renderer
	_global_variable_mark_buffer_dirty(pos, 1);
}

void RasterizerStorageRD::_update_global_variables() {
	if (global_variables.buffer_dirty_region_count > 0) {
		uint32_t total_regions = global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE;
		if (total_regions / global_variables.buffer_dirty_region_count <= 4) {
			// 25% of regions dirty, just update all buffer
			RD::get_singleton()->buffer_update(global_variables.buffer, 0, sizeof(GlobalVariables::Value) * global_variables.buffer_size, global_variables.buffer_values);
			zeromem(global_variables.buffer_dirty_regions, sizeof(bool) * total_regions);
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
		for (List<RID>::Element *E = global_variables.materials_using_buffer.front(); E; E = E->next()) {
			Material *material = material_owner.getornull(E->get());
			ERR_CONTINUE(!material); //wtf

			_material_queue_update(material, true, false);
		}

		global_variables.must_update_buffer_materials = false;
	}

	if (global_variables.must_update_texture_materials) {
		// only happens in the case of a buffer variable added or removed,
		// so not often.
		for (List<RID>::Element *E = global_variables.materials_using_texture.front(); E; E = E->next()) {
			Material *material = material_owner.getornull(E->get());
			ERR_CONTINUE(!material); //wtf

			_material_queue_update(material, false, true);
			print_line("update material texture?");
		}

		global_variables.must_update_texture_materials = false;
	}
}

void RasterizerStorageRD::update_dirty_resources() {
	_update_global_variables(); //must do before materials, so it can queue them for update
	_update_queued_materials();
	_update_dirty_multimeshes();
	_update_dirty_skeletons();
	_update_decal_atlas();
}

bool RasterizerStorageRD::has_os_feature(const String &p_feature) const {
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

bool RasterizerStorageRD::free(RID p_rid) {
	if (texture_owner.owns(p_rid)) {
		Texture *t = texture_owner.getornull(p_rid);

		ERR_FAIL_COND_V(t->is_render_target, false);

		if (RD::get_singleton()->texture_is_valid(t->rd_texture_srgb)) {
			//erase this first, as it's a dependency of the one below
			RD::get_singleton()->free(t->rd_texture_srgb);
		}
		if (RD::get_singleton()->texture_is_valid(t->rd_texture)) {
			RD::get_singleton()->free(t->rd_texture);
		}

		if (t->is_proxy && t->proxy_to.is_valid()) {
			Texture *proxy_to = texture_owner.getornull(t->proxy_to);
			if (proxy_to) {
				proxy_to->proxies.erase(p_rid);
			}
		}

		if (decal_atlas.textures.has(p_rid)) {
			decal_atlas.textures.erase(p_rid);
			//there is not much a point of making it dirty, just let it be.
		}

		for (int i = 0; i < t->proxies.size(); i++) {
			Texture *p = texture_owner.getornull(t->proxies[i]);
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
		CanvasTexture *ct = canvas_texture_owner.getornull(p_rid);
		memdelete(ct);
		canvas_texture_owner.free(p_rid);
	} else if (shader_owner.owns(p_rid)) {
		Shader *shader = shader_owner.getornull(p_rid);
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
		Material *material = material_owner.getornull(p_rid);
		if (material->update_requested) {
			_update_queued_materials();
		}
		material_set_shader(p_rid, RID()); //clean up shader
		material->instance_dependency.instance_notify_deleted(p_rid);
		material_owner.free(p_rid);
	} else if (mesh_owner.owns(p_rid)) {
		mesh_clear(p_rid);
		Mesh *mesh = mesh_owner.getornull(p_rid);
		mesh->instance_dependency.instance_notify_deleted(p_rid);
		mesh_owner.free(p_rid);
	} else if (multimesh_owner.owns(p_rid)) {
		_update_dirty_multimeshes();
		multimesh_allocate(p_rid, 0, RS::MULTIMESH_TRANSFORM_2D);
		MultiMesh *multimesh = multimesh_owner.getornull(p_rid);
		multimesh->instance_dependency.instance_notify_deleted(p_rid);
		multimesh_owner.free(p_rid);
	} else if (skeleton_owner.owns(p_rid)) {
		_update_dirty_skeletons();
		skeleton_allocate(p_rid, 0);
		Skeleton *skeleton = skeleton_owner.getornull(p_rid);
		skeleton->instance_dependency.instance_notify_deleted(p_rid);
		skeleton_owner.free(p_rid);
	} else if (reflection_probe_owner.owns(p_rid)) {
		ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_rid);
		reflection_probe->instance_dependency.instance_notify_deleted(p_rid);
		reflection_probe_owner.free(p_rid);
	} else if (decal_owner.owns(p_rid)) {
		Decal *decal = decal_owner.getornull(p_rid);
		for (int i = 0; i < RS::DECAL_TEXTURE_MAX; i++) {
			if (decal->textures[i].is_valid() && texture_owner.owns(decal->textures[i])) {
				texture_remove_from_decal_atlas(decal->textures[i]);
			}
		}
		decal->instance_dependency.instance_notify_deleted(p_rid);
		decal_owner.free(p_rid);
	} else if (gi_probe_owner.owns(p_rid)) {
		gi_probe_allocate(p_rid, Transform(), AABB(), Vector3i(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<uint8_t>(), Vector<int>()); //deallocate
		GIProbe *gi_probe = gi_probe_owner.getornull(p_rid);
		gi_probe->instance_dependency.instance_notify_deleted(p_rid);
		gi_probe_owner.free(p_rid);
	} else if (lightmap_owner.owns(p_rid)) {
		lightmap_set_textures(p_rid, RID(), false);
		Lightmap *lightmap = lightmap_owner.getornull(p_rid);
		lightmap->instance_dependency.instance_notify_deleted(p_rid);
		lightmap_owner.free(p_rid);

	} else if (light_owner.owns(p_rid)) {
		light_set_projector(p_rid, RID()); //clear projector
		// delete the texture
		Light *light = light_owner.getornull(p_rid);
		light->instance_dependency.instance_notify_deleted(p_rid);
		light_owner.free(p_rid);

	} else if (particles_owner.owns(p_rid)) {
		Particles *particles = particles_owner.getornull(p_rid);
		_particles_free_data(particles);
		particles->instance_dependency.instance_notify_deleted(p_rid);
		particles_owner.free(p_rid);
	} else if (particles_collision_owner.owns(p_rid)) {
		ParticlesCollision *particles_collision = particles_collision_owner.getornull(p_rid);

		if (particles_collision->heightfield_texture.is_valid()) {
			RD::get_singleton()->free(particles_collision->heightfield_texture);
		}
		particles_collision->instance_dependency.instance_notify_deleted(p_rid);
		particles_collision_owner.free(p_rid);
	} else if (render_target_owner.owns(p_rid)) {
		RenderTarget *rt = render_target_owner.getornull(p_rid);

		_clear_render_target(rt);

		if (rt->texture.is_valid()) {
			Texture *tex = texture_owner.getornull(rt->texture);
			tex->is_render_target = false;
			free(rt->texture);
		}

		render_target_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

RasterizerEffectsRD *RasterizerStorageRD::get_effects() {
	return &effects;
}

void RasterizerStorageRD::capture_timestamps_begin() {
	RD::get_singleton()->capture_timestamp("Frame Begin", false);
}

void RasterizerStorageRD::capture_timestamp(const String &p_name) {
	RD::get_singleton()->capture_timestamp(p_name, true);
}

uint32_t RasterizerStorageRD::get_captured_timestamps_count() const {
	return RD::get_singleton()->get_captured_timestamps_count();
}

uint64_t RasterizerStorageRD::get_captured_timestamps_frame() const {
	return RD::get_singleton()->get_captured_timestamps_frame();
}

uint64_t RasterizerStorageRD::get_captured_timestamp_gpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_gpu_time(p_index);
}

uint64_t RasterizerStorageRD::get_captured_timestamp_cpu_time(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_cpu_time(p_index);
}

String RasterizerStorageRD::get_captured_timestamp_name(uint32_t p_index) const {
	return RD::get_singleton()->get_captured_timestamp_name(p_index);
}

RasterizerStorageRD *RasterizerStorageRD::base_singleton = nullptr;

RasterizerStorageRD::RasterizerStorageRD() {
	base_singleton = this;

	for (int i = 0; i < SHADER_TYPE_MAX; i++) {
		shader_data_request_func[i] = nullptr;
	}

	static_assert(sizeof(GlobalVariables::Value) == 16);

	global_variables.buffer_size = GLOBAL_GET("rendering/high_end/global_shader_variables_buffer_size");
	global_variables.buffer_size = MAX(4096, global_variables.buffer_size);
	global_variables.buffer_values = memnew_arr(GlobalVariables::Value, global_variables.buffer_size);
	zeromem(global_variables.buffer_values, sizeof(GlobalVariables::Value) * global_variables.buffer_size);
	global_variables.buffer_usage = memnew_arr(GlobalVariables::ValueUsage, global_variables.buffer_size);
	global_variables.buffer_dirty_regions = memnew_arr(bool, global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE);
	zeromem(global_variables.buffer_dirty_regions, sizeof(bool) * global_variables.buffer_size / GlobalVariables::BUFFER_DIRTY_REGION_SIZE);
	global_variables.buffer = RD::get_singleton()->storage_buffer_create(sizeof(GlobalVariables::Value) * global_variables.buffer_size);

	material_update_list = nullptr;
	{ //create default textures

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.type = RD::TEXTURE_TYPE_2D;

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
		tformat.type = RD::TEXTURE_TYPE_CUBE_ARRAY;

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
		tformat.type = RD::TEXTURE_TYPE_CUBE;

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

	{ //create default 3D

		RD::TextureFormat tformat;
		tformat.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
		tformat.width = 4;
		tformat.height = 4;
		tformat.depth = 4;
		tformat.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
		tformat.type = RD::TEXTURE_TYPE_3D;

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
		tformat.type = RD::TEXTURE_TYPE_2D_ARRAY;

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
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_NEAREST;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/quality/texture_filters/anisotropic_filtering_level"));
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = 1 << int(GLOBAL_GET("rendering/quality/texture_filters/anisotropic_filtering_level"));

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

	{
		Vector<String> sdf_versions;
		sdf_versions.push_back(""); //one only
		giprobe_sdf_shader.initialize(sdf_versions);
		giprobe_sdf_shader_version = giprobe_sdf_shader.version_create();
		giprobe_sdf_shader.version_set_compute_code(giprobe_sdf_shader_version, "", "", "", Vector<String>());
		giprobe_sdf_shader_version_shader = giprobe_sdf_shader.version_get_shader(giprobe_sdf_shader_version, 0);
		giprobe_sdf_shader_pipeline = RD::get_singleton()->compute_pipeline_create(giprobe_sdf_shader_version_shader);
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

	lightmap_probe_capture_update_speed = GLOBAL_GET("rendering/lightmapper/probe_capture_update_speed");

	/* Particles */

	{
		// Initialize particles
		Vector<String> particles_modes;
		particles_modes.push_back("");
		particles_shader.shader.initialize(particles_modes, String());
	}
	shader_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_PARTICLES, _create_particles_shader_funcs);
	material_set_data_request_function(RasterizerStorageRD::SHADER_TYPE_PARTICLES, _create_particles_material_funcs);

	{
		ShaderCompilerRD::DefaultIdentifierActions actions;

		actions.renames["COLOR"] = "PARTICLE.color";
		actions.renames["VELOCITY"] = "PARTICLE.velocity";
		//actions.renames["MASS"] = "mass"; ?
		actions.renames["ACTIVE"] = "PARTICLE.is_active";
		actions.renames["RESTART"] = "restart";
		actions.renames["CUSTOM"] = "PARTICLE.custom";
		actions.renames["TRANSFORM"] = "PARTICLE.xform";
		actions.renames["TIME"] = "FRAME.time";
		actions.renames["LIFETIME"] = "params.lifetime";
		actions.renames["DELTA"] = "local_delta";
		actions.renames["NUMBER"] = "particle";
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
		actions.renames["emit_particle"] = "emit_particle";
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
		particles_shader.default_shader = shader_create();
		shader_set_code(particles_shader.default_shader, "shader_type particles; void compute() { COLOR = vec4(1.0); } \n");
		particles_shader.default_material = material_create();
		material_set_shader(particles_shader.default_material, particles_shader.default_shader);

		ParticlesMaterialData *md = (ParticlesMaterialData *)material_get_data(particles_shader.default_material, RasterizerStorageRD::SHADER_TYPE_PARTICLES);
		particles_shader.default_shader_rd = particles_shader.shader.version_get_shader(md->shader_data->version, 0);

		Vector<RD::Uniform> uniforms;

		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_SAMPLER;
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
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
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
		copy_modes.push_back("\n#define MODE_FILL_SORT_BUFFER\n#define USE_SORT_BUFFER\n");
		copy_modes.push_back("\n#define MODE_FILL_INSTANCES\n#define USE_SORT_BUFFER\n");

		particles_shader.copy_shader.initialize(copy_modes);

		particles_shader.copy_shader_version = particles_shader.copy_shader.version_create();

		for (int i = 0; i < ParticlesShader::COPY_MODE_MAX; i++) {
			particles_shader.copy_pipelines[i] = RD::get_singleton()->compute_pipeline_create(particles_shader.copy_shader.version_get_shader(particles_shader.copy_shader_version, i));
		}
	}
}

RasterizerStorageRD::~RasterizerStorageRD() {
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

	giprobe_sdf_shader.version_free(giprobe_sdf_shader_version);
	particles_shader.copy_shader.version_free(particles_shader.copy_shader_version);

	RenderingServer::get_singleton()->free(particles_shader.default_material);
	RenderingServer::get_singleton()->free(particles_shader.default_shader);

	RD::get_singleton()->free(default_rd_storage_buffer);

	if (decal_atlas.textures.size()) {
		ERR_PRINT("Decal Atlas: " + itos(decal_atlas.textures.size()) + " textures were not removed from the atlas.");
	}

	if (decal_atlas.texture.is_valid()) {
		RD::get_singleton()->free(decal_atlas.texture);
	}
}
