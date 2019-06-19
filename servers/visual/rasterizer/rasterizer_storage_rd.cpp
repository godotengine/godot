#include "rasterizer_storage_rd.h"

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
		case Image::FORMAT_RGBA5551: {
			r_format.format = RD::DATA_FORMAT_A1R5G5B5_UNORM_PACK16;
#warning TODO needs something in Texture to convert to this format internally
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_R;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_G;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_B;
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
#warning TODO need to make a function in Image to swap bits for this
			r_format.swizzle_r = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_g = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_b = RD::TEXTURE_SWIZZLE_IDENTITY;
			r_format.swizzle_a = RD::TEXTURE_SWIZZLE_IDENTITY;
		} break;
		case Image::FORMAT_DXT1: {
			if (RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK, RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT)) {
				r_format.format = RD::DATA_FORMAT_BC1_RGB_UNORM_BLOCK;
				r_format.format_srgb = RD::DATA_FORMAT_BC1_RGBA_SRGB_BLOCK;
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
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT;
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
	PoolVector<uint8_t> data = image->get_data(); //use image data
	Vector<PoolVector<uint8_t> > data_slices;
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

#warning TODO this is temporary to get things to work
	texture.image_cache_2d = p_image;

#warning texture owner needs a spinlock to make this really callable from any thread
	return texture_owner.make_rid(texture);
}

RID RasterizerStorageRD::texture_2d_layered_create(const Vector<Ref<Image> > &p_layers, VS::TextureLayeredType p_layered_type) {

	return RID();
}
RID RasterizerStorageRD::texture_3d_create(const Vector<Ref<Image> > &p_slices) {

	return RID();
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

#warning TODO this is temporary to get things to work
	tex->image_cache_2d = p_image;

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

//these two APIs can be used together or in combination with the others.
RID RasterizerStorageRD::texture_2d_placeholder_create() {

	//this could be better optimized to reuse an existing image , done this way
	//for now to get it working
	Ref<Image> image;
	image.instance();
	image->create(4, 4, false, Image::FORMAT_RGBA8);
	image->lock();
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			image->set_pixel(i, j, Color(1, 0, 1, 1));
		}
	}
	image->unlock();

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

#warning TODO this is temporary to get things to work
	return tex->image_cache_2d;
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
	Texture *by_tex = texture_owner.getornull(p_by_texture);
	ERR_FAIL_COND(!by_tex);

	if (tex == by_tex) {
		return;
	}

	RD::get_singleton()->free(tex->rd_texture);
	if (tex->rd_texture_srgb.is_valid()) {
		RD::get_singleton()->free(tex->rd_texture_srgb);
	}

	*tex = *by_tex;

	texture_owner.free(p_by_texture);
}
void RasterizerStorageRD::texture_set_size_override(RID p_texture, int p_width, int p_height) {
	Texture *tex = texture_owner.getornull(p_texture);
	ERR_FAIL_COND(!tex);
	ERR_FAIL_COND(tex->type != Texture::TYPE_2D);
	tex->width_2d = p_width;
	tex->height_2d = p_width;
}

void RasterizerStorageRD::texture_set_path(RID p_texture, const String &p_path) {
}
String RasterizerStorageRD::texture_get_path(RID p_texture) const {
	return String();
}

void RasterizerStorageRD::texture_set_detect_3d_callback(RID p_texture, VS::TextureDetectCallback p_callback, void *p_userdata) {
}
void RasterizerStorageRD::texture_set_detect_normal_callback(RID p_texture, VS::TextureDetectCallback p_callback, void *p_userdata) {
}
void RasterizerStorageRD::texture_set_detect_roughness_callback(RID p_texture, VS::TextureDetectRoughnessCallback p_callback, void *p_userdata) {
}
void RasterizerStorageRD::texture_debug_usage(List<VS::TextureInfo> *r_info) {
}

void RasterizerStorageRD::texture_set_proxy(RID p_proxy, RID p_base) {
}
void RasterizerStorageRD::texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {
}

Size2 RasterizerStorageRD::texture_size_with_proxy(RID p_proxy) const {
	return Size2();
}

/* RENDER TARGET API */

void RasterizerStorageRD::_clear_render_target(RenderTarget *rt) {

	//free in reverse dependency order
	if (rt->framebuffer.is_valid()) {
		RD::get_singleton()->free(rt->framebuffer);
	}
	if (rt->color_srgb.is_valid()) {
		RD::get_singleton()->free(rt->color_srgb);
	}

	if (rt->color.is_valid()) {
		RD::get_singleton()->free(rt->color);
	}

	rt->framebuffer = RID();
	rt->color = RID();
	rt->color_srgb = RID();

	rt->dirty = true;
	rt->texture_dirty = true;
}

void RasterizerStorageRD::_update_render_target(RenderTarget *rt) {

	_clear_render_target(rt);

	if (rt->size.width == 0 || rt->size.height == 0) {
		rt->dirty = false;
		return;
	}
	//until we implement suport for HDR monitors (and render target is attached to screen), this is enough.
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
		rd_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
		rd_format.shareable_formats.push_back(rt->color_format);
		rd_format.shareable_formats.push_back(rt->color_format_srgb);
	}

	rt->color = RD::get_singleton()->texture_create(rd_format, rd_view);
	ERR_FAIL_COND(rt->color.is_null());
	if (rt->color_format_srgb != RD::DATA_FORMAT_MAX) {
		rd_view.format_override = rt->color_format_srgb;
		rt->color_srgb = RD::get_singleton()->texture_create_shared(rd_view, rt->color);
		if (rt->color_srgb.is_null()) {
			_clear_render_target(rt);
			ERR_FAIL_COND(rt->color_srgb.is_null());
		}
	}

	Vector<RID> fb_textures;
	fb_textures.push_back(rt->color);
	rt->framebuffer = RD::get_singleton()->framebuffer_create(fb_textures);
	if (rt->framebuffer.is_null()) {
		_clear_render_target(rt);
		ERR_FAIL_COND(rt->framebuffer.is_null());
	}

	rt->dirty = false;
}

RID RasterizerStorageRD::render_target_create() {
	RenderTarget render_target;
	render_target.dirty = true;
	render_target.texture_dirty = true;
	render_target.was_used = false;

	for (int i = 0; i < RENDER_TARGET_FLAG_MAX; i++) {
		render_target.flags[i] = false;
	}
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
	rt->dirty = true;
	rt->texture_dirty = true;
}

RID RasterizerStorageRD::render_target_get_texture(RID p_render_target) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (rt->texture_dirty) {
		if (rt->dirty) {
			_update_render_target(rt);
		}

		if (rt->texture.is_null()) {
			//kinda hacky, but not terrible
			rt->texture = texture_2d_placeholder_create();
		}

		Texture *tex = texture_owner.getornull(rt->texture);

		if (!tex->is_render_target) {
			//first allocation, fix it up
			RD::get_singleton()->free(tex->rd_texture);
			if (tex->rd_texture_srgb.is_null()) {
				RD::get_singleton()->free(tex->rd_texture_srgb);
			}
			tex->is_render_target = true;
		}

		if (!rt->color.is_null()) {
			//there is a color buffer, update to it
			tex->rd_texture = rt->color;
			tex->rd_texture_srgb = rt->color_srgb;
			tex->width = rt->size.width;
			tex->height = rt->size.height;
			tex->width_2d = rt->size.width;
			tex->height_2d = rt->size.height;
			tex->rd_format = rt->color_format;
			tex->rd_format_srgb = rt->color_format_srgb;
			tex->format = rt->image_format;
		}

		rt->texture_dirty = false;
	}

	return rt->texture;
}

void RasterizerStorageRD::render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {
}

void RasterizerStorageRD::render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {
	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->flags[p_flag] = p_value;
	rt->dirty = true;
	rt->texture_dirty = true;
}

bool RasterizerStorageRD::render_target_was_used(RID p_render_target) {

	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->was_used;
}

void RasterizerStorageRD::render_target_clear_used_flag(RID p_render_target) {

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

	if (rt->dirty) {
		_update_render_target(rt);
	}

	return rt->framebuffer;
}

bool RasterizerStorageRD::free(RID p_rid) {

	if (texture_owner.owns(p_rid)) {
		Texture *t = texture_owner.getornull(p_rid);

		ERR_FAIL_COND_V(t->is_render_target, false);

		RD::get_singleton()->free(t->rd_texture);
		if (t->rd_texture_srgb.is_valid()) {
			RD::get_singleton()->free(t->rd_texture_srgb);
		}
		texture_owner.free(p_rid);

	} else if (render_target_owner.owns(p_rid)) {
		RenderTarget *rt = render_target_owner.getornull(p_rid);

		_clear_render_target(rt);

		if (rt->texture.is_valid()) {
			//no memory to be freed from here
			texture_owner.free(rt->texture);
		}

		render_target_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}
