/*************************************************************************/
/*  image_compress_etcpak.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "image_compress_etcpak.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#include "thirdparty/etcpak/ProcessDxtc.hpp"
#include "thirdparty/etcpak/ProcessRGB.hpp"

EtcpakType _determine_etc_type(Image::UsedChannels p_channels) {
	switch (p_channels) {
		case Image::USED_CHANNELS_L:
			return EtcpakType::ETCPAK_TYPE_ETC1;
		case Image::USED_CHANNELS_LA:
			return EtcpakType::ETCPAK_TYPE_ETC2_ALPHA;
		case Image::USED_CHANNELS_R:
			return EtcpakType::ETCPAK_TYPE_ETC2;
		case Image::USED_CHANNELS_RG:
			return EtcpakType::ETCPAK_TYPE_ETC2_RA_AS_RG;
		case Image::USED_CHANNELS_RGB:
			return EtcpakType::ETCPAK_TYPE_ETC2;
		case Image::USED_CHANNELS_RGBA:
			return EtcpakType::ETCPAK_TYPE_ETC2_ALPHA;
		default:
			return EtcpakType::ETCPAK_TYPE_ETC2_ALPHA;
	}
}

EtcpakType _determine_dxt_type(Image::UsedChannels p_channels) {
	switch (p_channels) {
		case Image::USED_CHANNELS_L:
			return EtcpakType::ETCPAK_TYPE_DXT1;
		case Image::USED_CHANNELS_LA:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		case Image::USED_CHANNELS_R:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		case Image::USED_CHANNELS_RG:
			return EtcpakType::ETCPAK_TYPE_DXT5_RA_AS_RG;
		case Image::USED_CHANNELS_RGB:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		case Image::USED_CHANNELS_RGBA:
			return EtcpakType::ETCPAK_TYPE_DXT5;
		default:
			return EtcpakType::ETCPAK_TYPE_DXT5;
	}
}

void _compress_etc1(Image *r_img, float p_lossy_quality) {
	_compress_etcpak(EtcpakType::ETCPAK_TYPE_ETC1, r_img, p_lossy_quality);
}

void _compress_etc2(Image *r_img, float p_lossy_quality, Image::UsedChannels p_channels) {
	EtcpakType type = _determine_etc_type(p_channels);
	_compress_etcpak(type, r_img, p_lossy_quality);
}

void _compress_bc(Image *r_img, float p_lossy_quality, Image::UsedChannels p_channels) {
	EtcpakType type = _determine_dxt_type(p_channels);
	_compress_etcpak(type, r_img, p_lossy_quality);
}

void _compress_etcpak(EtcpakType p_compresstype, Image *r_img, float p_lossy_quality) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	// TODO: See how to handle lossy quality.

	Image::Format img_format = r_img->get_format();
	if (img_format >= Image::FORMAT_DXT1) {
		return; // Do not compress, already compressed.
	}
	if (img_format > Image::FORMAT_RGBA8) {
		// TODO: we should be able to handle FORMAT_RGBA4444 and FORMAT_RGBA5551 eventually
		return;
	}

	// Use RGBA8 to convert.
	if (img_format != Image::FORMAT_RGBA8) {
		r_img->convert(Image::FORMAT_RGBA8);
	}

	// Determine output format based on Etcpak type.
	Image::Format target_format = Image::FORMAT_RGBA8;
	if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC1) {
		target_format = Image::FORMAT_ETC;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2) {
		target_format = Image::FORMAT_ETC2_RGB8;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_RA_AS_RG) {
		target_format = Image::FORMAT_ETC2_RA_AS_RG;
		r_img->convert_rg_to_ra_rgba8();
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_ALPHA) {
		target_format = Image::FORMAT_ETC2_RGBA8;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT1) {
		target_format = Image::FORMAT_DXT1;
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5_RA_AS_RG) {
		target_format = Image::FORMAT_DXT5_RA_AS_RG;
		r_img->convert_rg_to_ra_rgba8();
	} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5) {
		target_format = Image::FORMAT_DXT5;
	} else {
		ERR_FAIL_MSG("Invalid or unsupported Etcpak compression format.");
	}

	// Compress image data and (if required) mipmaps.

	const bool mipmaps = r_img->has_mipmaps();
	const int width = r_img->get_width();
	const int height = r_img->get_height();
	const uint8_t *src_read = r_img->get_data().ptr();

	print_verbose(vformat("ETCPAK: Encoding image size %dx%d to format %s.", width, height, Image::get_format_name(target_format)));

	int dest_size = Image::get_image_data_size(width, height, target_format, mipmaps);
	Vector<uint8_t> dest_data;
	dest_data.resize(dest_size);
	uint8_t *dest_write = dest_data.ptrw();

	int mip_count = mipmaps ? Image::get_image_required_mipmaps(width, height, target_format) : 0;

	for (int i = 0; i < mip_count + 1; i++) {
		// Get write mip metrics for target image.
		int mip_w, mip_h;
		int mip_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, target_format, i, mip_w, mip_h);
		// Ensure that mip offset is a multiple of 8 (etcpak expects uint64_t pointer).
		ERR_FAIL_COND(mip_ofs % 8 != 0);
		uint64_t *dest_mip_write = (uint64_t *)&dest_write[mip_ofs];

		// Block size. Align stride to multiple of 4 (RGBA8).
		mip_w = (mip_w + 3) & ~3;
		mip_h = (mip_h + 3) & ~3;
		const uint32_t blocks = mip_w * mip_h / 16;

		// Get mip data from source image for reading.
		int src_mip_ofs = r_img->get_mipmap_offset(i);
		const uint32_t *src_mip_read = (const uint32_t *)&src_read[src_mip_ofs];

		if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC1) {
			CompressEtc1RgbDither(src_mip_read, dest_mip_write, blocks, mip_w);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2 || p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_RA_AS_RG) {
			CompressEtc2Rgb(src_mip_read, dest_mip_write, blocks, mip_w, true);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_ETC2_ALPHA) {
			CompressEtc2Rgba(src_mip_read, dest_mip_write, blocks, mip_w, true);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT1) {
			CompressDxt1Dither(src_mip_read, dest_mip_write, blocks, mip_w);
		} else if (p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5 || p_compresstype == EtcpakType::ETCPAK_TYPE_DXT5_RA_AS_RG) {
			CompressDxt5(src_mip_read, dest_mip_write, blocks, mip_w);
		} else {
			ERR_FAIL_MSG("Invalid or unsupported Etcpak compression format.");
		}
	}

	// Replace original image with compressed one.
	r_img->create(width, height, mipmaps, target_format, dest_data);

	print_verbose(vformat("ETCPAK encode took %s ms.", rtos(OS::get_singleton()->get_ticks_msec() - start_time)));
}
