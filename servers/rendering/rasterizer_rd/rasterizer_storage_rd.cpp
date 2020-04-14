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
#include "core/project_settings.h"
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

			r_format.format = RD::DATA_FORMAT_R32G32B32A32_SFLOAT;
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

			r_format.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
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

	return RID();
}
RID RasterizerStorageRD::texture_3d_create(const Vector<Ref<Image>> &p_slices) {

	return RID();
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
void RasterizerStorageRD::texture_3d_update(RID p_texture, const Ref<Image> &p_image, int p_depth, int p_mipmap) {
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
RID RasterizerStorageRD::texture_2d_layered_placeholder_create() {

	return RID();
}
RID RasterizerStorageRD::texture_3d_placeholder_create() {

	return RID();
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

	return Ref<Image>();
}
Ref<Image> RasterizerStorageRD::texture_3d_slice_get(RID p_texture, int p_depth, int p_mipmap) const {

	return Ref<Image>();
}

void RasterizerStorageRD::texture_replace(RID p_texture, RID p_by_texture) {

	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->proxy_to.is_valid()); //cant replace proxy
	Texture *by_tex = texture_owner.getornull(p_by_texture);
	ERR_FAIL_COND(!by_tex);
	ERR_FAIL_COND(by_tex->proxy_to.is_valid()); //cant replace proxy

	if (tex == by_tex) {
		return;
	}

	if (tex->rd_texture_srgb.is_valid()) {
		RD::get_singleton()->free(tex->rd_texture_srgb);
	}
	RD::get_singleton()->free(tex->rd_texture);

	Vector<RID> proxies_to_update = tex->proxies;
	Vector<RID> proxies_to_redirect = by_tex->proxies;

	*tex = *by_tex;

	tex->proxies = proxies_to_update; //restore proxies, so they can be updated

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
	if (mode_string == "canvas_item")
		new_type = SHADER_TYPE_2D;
	else if (mode_string == "particles")
		new_type = SHADER_TYPE_PARTICLES;
	else if (mode_string == "spatial")
		new_type = SHADER_TYPE_3D;
	else if (mode_string == "sky")
		new_type = SHADER_TYPE_SKY;
	else
		new_type = SHADER_TYPE_MAX;

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
				material->data->set_next_pass(material->next_pass);
				material->data->set_render_priority(material->priority);
			}
			material->shader_type = new_type;
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
	material->uniform_dirty = p_uniform;
	material->texture_dirty = p_texture;
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
				if (i < s)
					gui[i] = r[i];
				else
					gui[i] = 0;
			}

		} break;
		case ShaderLanguage::TYPE_IVEC3: {

			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 3; i++) {
				if (i < s)
					gui[i] = r[i];
				else
					gui[i] = 0;
			}
		} break;
		case ShaderLanguage::TYPE_IVEC4: {

			Vector<int> iv = value;
			int s = iv.size();
			int32_t *gui = (int32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 4; i++) {
				if (i < s)
					gui[i] = r[i];
				else
					gui[i] = 0;
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
				if (i < s)
					gui[i] = r[i];
				else
					gui[i] = 0;
			}
		} break;
		case ShaderLanguage::TYPE_UVEC3: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 3; i++) {
				if (i < s)
					gui[i] = r[i];
				else
					gui[i] = 0;
			}

		} break;
		case ShaderLanguage::TYPE_UVEC4: {
			Vector<int> iv = value;
			int s = iv.size();
			uint32_t *gui = (uint32_t *)data;

			const int *r = iv.ptr();

			for (int i = 0; i < 4; i++) {
				if (i < s)
					gui[i] = r[i];
				else
					gui[i] = 0;
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

	for (Map<StringName, ShaderLanguage::ShaderNode::Uniform>::Element *E = p_uniforms.front(); E; E = E->next()) {

		if (E->get().order < 0)
			continue; // texture, does not go here

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
}

void RasterizerStorageRD::MaterialData::update_textures(const Map<StringName, Variant> &p_parameters, const Map<StringName, RID> &p_default_textures, const Vector<ShaderCompilerRD::GeneratedCode::Texture> &p_texture_uniforms, RID *p_textures, bool p_use_linear_color) {

	RasterizerStorageRD *singleton = (RasterizerStorageRD *)RasterizerStorage::base_singleton;
#ifdef TOOLS_ENABLED
	Texture *roughness_detect_texture = nullptr;
	RS::TextureDetectRoughnessChannel roughness_channel = RS::TEXTURE_DETECT_ROUGNHESS_R;
	Texture *normal_detect_texture = nullptr;
#endif

	for (int i = 0; i < p_texture_uniforms.size(); i++) {

		const StringName &uniform_name = p_texture_uniforms[i].name;

		RID texture;

		const Map<StringName, Variant>::Element *V = p_parameters.find(uniform_name);
		if (V) {
			texture = V->get();
		}

		if (!texture.is_valid()) {
			const Map<StringName, RID>::Element *W = p_default_textures.find(uniform_name);
			if (W) {

				texture = W->get();
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

					if (skbones[0].size == Vector3())
						continue; //bone is unused

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

					if (skbones[0].size == Vector3())
						continue; //bone is unused

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

	Vector<RD::VertexDescription> attributes;
	Vector<RID> buffers;

	uint32_t stride = 0;

	for (int i = 0; i < RS::ARRAY_WEIGHTS; i++) {

		RD::VertexDescription vd;
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
		multimesh->data_cache_dirty_regions[i] = 0;
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
		ret.resize(multimesh->instances);
		{
			float *w = multimesh->data_cache.ptrw();
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

/* SKELETON */

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

	if (skeleton->size == p_bones && skeleton->use_2d == p_2d_skeleton)
		return;

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

void RasterizerStorageRD::light_set_use_gi(RID p_light, bool p_enabled) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND(!light);

	light->use_gi = p_enabled;

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

bool RasterizerStorageRD::light_get_use_gi(RID p_light) {
	Light *light = light_owner.getornull(p_light);
	ERR_FAIL_COND_V(!light, false);

	return light->use_gi;
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

void RasterizerStorageRD::reflection_probe_set_interior_ambient(RID p_probe, const Color &p_ambient) {

	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->interior_ambient = p_ambient;
}

void RasterizerStorageRD::reflection_probe_set_interior_ambient_energy(RID p_probe, float p_energy) {

	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->interior_ambient_energy = p_energy;
}

void RasterizerStorageRD::reflection_probe_set_interior_ambient_probe_contribution(RID p_probe, float p_contrib) {

	ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->interior_ambient_probe_contrib = p_contrib;
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

Color RasterizerStorageRD::reflection_probe_get_interior_ambient(RID p_probe) const {

	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Color());

	return reflection_probe->interior_ambient;
}
float RasterizerStorageRD::reflection_probe_get_interior_ambient_energy(RID p_probe) const {

	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->interior_ambient_energy;
}
float RasterizerStorageRD::reflection_probe_get_interior_ambient_probe_contribution(RID p_probe) const {

	const ReflectionProbe *reflection_probe = reflection_probe_owner.getornull(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->interior_ambient_probe_contrib;
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

/* RENDER TARGET API */

void RasterizerStorageRD::_clear_render_target(RenderTarget *rt) {

	//free in reverse dependency order
	if (rt->framebuffer.is_valid()) {
		RD::get_singleton()->free(rt->framebuffer);
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
		if (rt->backbuffer_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rt->backbuffer_uniform_set)) {
			RD::get_singleton()->free(rt->backbuffer_uniform_set);
		}
		rt->backbuffer_uniform_set = RID();
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
	tf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	tf.mipmaps = mipmaps_required;

	rt->backbuffer = RD::get_singleton()->texture_create(tf, RD::TextureView());
	rt->backbuffer_mipmap0 = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), rt->backbuffer, 0, 0);

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

void RasterizerStorageRD::render_target_copy_to_back_buffer(RID p_render_target, const Rect2i &p_region) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	Rect2i region = p_region;
	if (region == Rect2i()) {
		region.size = rt->size;
	}

	//single texture copy for backbuffer
	RD::get_singleton()->texture_copy(rt->color, rt->backbuffer_mipmap0, Vector3(region.position.x, region.position.y, 0), Vector3(region.position.x, region.position.y, 0), Vector3(region.size.x, region.size.y, 1), 0, 0, 0, 0, true);
	//effects.copy(rt->color, rt->backbuffer_fb, blur_region);

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

RID RasterizerStorageRD::render_target_get_back_buffer_uniform_set(RID p_render_target, RID p_base_shader) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (!rt->backbuffer.is_valid()) {
		_create_render_target_backbuffer(rt);
	}

	if (rt->backbuffer_uniform_set.is_valid() && RD::get_singleton()->uniform_set_is_valid(rt->backbuffer_uniform_set)) {
		return rt->backbuffer_uniform_set; //if still valid, return/reuse it.
	}

	//create otherwise
	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.type = RD::UNIFORM_TYPE_TEXTURE;
	u.binding = 0;
	u.ids.push_back(rt->backbuffer);
	uniforms.push_back(u);

	rt->backbuffer_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, p_base_shader, 3);
	ERR_FAIL_COND_V(!rt->backbuffer_uniform_set.is_valid(), RID());

	return rt->backbuffer_uniform_set;
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
	} else if (light_owner.owns(p_base)) {
		Light *l = light_owner.getornull(p_base);
		p_instance->update_dependency(&l->instance_dependency);
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
	return decal_atlas.texture;
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
		const RID *K = NULL;

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
			//print_line("blitrect: " + t->uv_rect);
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

				const RID *K = NULL;
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

void RasterizerStorageRD::update_dirty_resources() {
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
		texture_owner.free(p_rid);

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

	} else if (light_owner.owns(p_rid)) {

		light_set_projector(p_rid, RID()); //clear projector
		// delete the texture
		Light *light = light_owner.getornull(p_rid);
		light->instance_dependency.instance_notify_deleted(p_rid);
		light_owner.free(p_rid);

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

RasterizerStorageRD::RasterizerStorageRD() {

	for (int i = 0; i < SHADER_TYPE_MAX; i++) {
		shader_data_request_func[i] = nullptr;
	}

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
					sampler_state.anisotropy_max = GLOBAL_GET("rendering/quality/texture_filters/max_anisotropy");
				} break;
				case RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC: {
					sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.mip_filter = RD::SAMPLER_FILTER_LINEAR;
					sampler_state.use_anisotropy = true;
					sampler_state.anisotropy_max = GLOBAL_GET("rendering/quality/texture_filters/max_anisotropy");

				} break;
				default: {
				}
			}
			switch (j) {
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED: {

					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_ENABLED: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_REPEAT;
				} break;
				case RS::CANVAS_ITEM_TEXTURE_REPEAT_MIRROR: {
					sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
					sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT;
				} break;
				default: {
				}
			}

			default_rd_samplers[i][j] = RD::get_singleton()->sampler_create(sampler_state);
		}
	}

	//default rd buffers
	{

		//vertex
		{

				Vector<uint8_t> buffer;

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
	Vector<uint8_t> buffer;
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
	Vector<uint8_t> buffer;
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
	Vector<uint8_t> buffer;
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
	Vector<uint8_t> buffer;
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
	Vector<uint8_t> buffer;
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
	Vector<uint8_t> buffer;
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
	Vector<uint8_t> buffer;
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
}

RasterizerStorageRD::~RasterizerStorageRD() {

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

	if (decal_atlas.textures.size()) {
		ERR_PRINT("Decal Atlas: " + itos(decal_atlas.textures.size()) + " textures were not removed from the atlas.");
	}

	if (decal_atlas.texture.is_valid()) {
		RD::get_singleton()->free(decal_atlas.texture);
	}
}
