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
	int width = r_img->get_width();
	int height = r_img->get_height();

	/*
	The first mipmap level of a compressed texture must be a multiple of 4. Quote from D3D11.3 spec:

	BC format surfaces are always multiples of full blocks, each block representing 4x4 pixels.
	For mipmaps, the top level map is required to be a multiple of 4 size in all dimensions.
	The sizes for the lower level maps are computed as they are for all mipmapped surfaces,
	and thus may not be a multiple of 4, for example a top level map of 20 results in a second level
	map size of 10. For these cases, there is a differing 'physical' size and a 'virtual' size.
	The virtual size is that computed for each mip level without adjustment, which is 10 for the example.
	The physical size is the virtual size rounded up to the next multiple of 4, which is 12 for the example,
	and this represents the actual memory size. The sampling hardware will apply texture address
	processing based on the virtual size (using, for example, border color if specified for accesses
	beyond 10), and thus for the example case will not access the 11th and 12th row of the resource.
	So for mipmap chains when an axis becomes < 4 in size, only texels 'a','b','e','f'
	are used for a 2x2 map, and texel 'a' is used for 1x1. Note that this is similar to, but distinct from,
	the surface pitch, which can encompass additional padding beyond the physical surface size.
	*/
	int next_width = width <= 2 ? width : (width + 3) & ~3;
	int next_height = height <= 2 ? height : (height + 3) & ~3;
	if (next_width != width || next_height != height) {
		r_img->resize(next_width, next_height, Image::INTERPOLATE_LANCZOS);
		width = r_img->get_width();
		height = r_img->get_height();
	}
	// ERR_FAIL_COND(width % 4 != 0 || height % 4 != 0); // FIXME: No longer guaranteed.
	// Multiple-of-4 should be guaranteed by above.
	// However, power-of-two 3d textures will create Nx2 and Nx1 mipmap levels,
	// which are individually compressed Image objects that violate the above rule.
	// Hence, we allow Nx1 and Nx2 images through without forcing to multiple-of-4.

	const uint8_t *src_read = r_img->get_data().ptr();

	print_verbose(vformat("ETCPAK: Encoding image size %dx%d to format %s.", width, height, Image::get_format_name(target_format)));

	int dest_size = Image::get_image_data_size(width, height, target_format, mipmaps);
	Vector<uint8_t> dest_data;
	dest_data.resize(dest_size);
	uint8_t *dest_write = dest_data.ptrw();

	int mip_count = mipmaps ? Image::get_image_required_mipmaps(width, height, target_format) : 0;
	Vector<uint32_t> padded_src;

	for (int i = 0; i < mip_count + 1; i++) {
		// Get write mip metrics for target image.
		int orig_mip_w, orig_mip_h;
		int mip_ofs = Image::get_image_mipmap_offset_and_dimensions(width, height, target_format, i, orig_mip_w, orig_mip_h);
		// Ensure that mip offset is a multiple of 8 (etcpak expects uint64_t pointer).
		ERR_FAIL_COND(mip_ofs % 8 != 0);
		uint64_t *dest_mip_write = (uint64_t *)&dest_write[mip_ofs];

		// Block size. Align stride to multiple of 4 (RGBA8).
		int mip_w = (orig_mip_w + 3) & ~3;
		int mip_h = (orig_mip_h + 3) & ~3;
		const uint32_t blocks = mip_w * mip_h / 16;

		// Get mip data from source image for reading.
		int src_mip_ofs = r_img->get_mipmap_offset(i);
		const uint32_t *src_mip_read = (const uint32_t *)&src_read[src_mip_ofs];

		// Pad textures to nearest block by smearing.
		if (mip_w != orig_mip_w || mip_h != orig_mip_h) {
			padded_src.resize(mip_w * mip_h);
			uint32_t *ptrw = padded_src.ptrw();
			int x = 0, y = 0;
			for (y = 0; y < orig_mip_h; y++) {
				for (x = 0; x < orig_mip_w; x++) {
					ptrw[mip_w * y + x] = src_mip_read[orig_mip_w * y + x];
				}
				// First, smear in x.
				for (; x < mip_w; x++) {
					ptrw[mip_w * y + x] = ptrw[mip_w * y + x - 1];
				}
			}
			// Then, smear in y.
			for (; y < mip_h; y++) {
				for (x = 0; x < mip_w; x++) {
					ptrw[mip_w * y + x] = ptrw[mip_w * y + x - mip_w];
				}
			}
			// Override the src_mip_read pointer to our temporary Vector.
			src_mip_read = padded_src.ptr();
		}
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
