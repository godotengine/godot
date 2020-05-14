/*************************************************************************/
/*  image_compress_squish.cpp                                            */
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

#include "image_compress_squish.h"

#include <squish.h>

void image_decompress_squish(Image *p_image) {
	int w = p_image->get_width();
	int h = p_image->get_height();

	Image::Format target_format = Image::FORMAT_RGBA8;
	Vector<uint8_t> data;
	int target_size = Image::get_image_data_size(w, h, target_format, p_image->has_mipmaps());
	int mm_count = p_image->get_mipmap_count();
	data.resize(target_size);

	const uint8_t *rb = p_image->get_data().ptr();
	uint8_t *wb = data.ptrw();

	int squish_flags = Image::FORMAT_MAX;
	if (p_image->get_format() == Image::FORMAT_DXT1) {
		squish_flags = squish::kDxt1;
	} else if (p_image->get_format() == Image::FORMAT_DXT3) {
		squish_flags = squish::kDxt3;
	} else if (p_image->get_format() == Image::FORMAT_DXT5 || p_image->get_format() == Image::FORMAT_DXT5_RA_AS_RG) {
		squish_flags = squish::kDxt5;
	} else if (p_image->get_format() == Image::FORMAT_RGTC_R) {
		squish_flags = squish::kBc4;
	} else if (p_image->get_format() == Image::FORMAT_RGTC_RG) {
		squish_flags = squish::kBc5;
	} else {
		ERR_FAIL_MSG("Squish: Can't decompress unknown format: " + itos(p_image->get_format()) + ".");
		return;
	}

	for (int i = 0; i <= mm_count; i++) {
		int src_ofs = 0, mipmap_size = 0, mipmap_w = 0, mipmap_h = 0;
		p_image->get_mipmap_offset_size_and_dimensions(i, src_ofs, mipmap_size, mipmap_w, mipmap_h);
		int dst_ofs = Image::get_image_mipmap_offset(p_image->get_width(), p_image->get_height(), target_format, i);
		squish::DecompressImage(&wb[dst_ofs], w, h, &rb[src_ofs], squish_flags);
		w >>= 1;
		h >>= 1;
	}

	p_image->create(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format, data);

	if (p_image->get_format() == Image::FORMAT_DXT5_RA_AS_RG) {
		p_image->convert_ra_rgba8_to_rg();
	}
}

void image_compress_squish(Image *p_image, float p_lossy_quality, Image::UsedChannels p_channels) {
	if (p_image->get_format() >= Image::FORMAT_DXT1) {
		return; //do not compress, already compressed
	}

	int w = p_image->get_width();
	int h = p_image->get_height();

	if (p_image->get_format() <= Image::FORMAT_RGBA8) {
		int squish_comp = squish::kColourRangeFit;

		if (p_lossy_quality > 0.85) {
			squish_comp = squish::kColourIterativeClusterFit;
		} else if (p_lossy_quality > 0.75) {
			squish_comp = squish::kColourClusterFit;
		}

		Image::Format target_format = Image::FORMAT_RGBA8;

		p_image->convert(Image::FORMAT_RGBA8); //still uses RGBA to convert

		switch (p_channels) {
			case Image::USED_CHANNELS_L: {
				target_format = Image::FORMAT_DXT1;
				squish_comp |= squish::kDxt1;
			} break;
			case Image::USED_CHANNELS_LA: {
				target_format = Image::FORMAT_DXT5;
				squish_comp |= squish::kDxt5;
			} break;
			case Image::USED_CHANNELS_R: {
				target_format = Image::FORMAT_RGTC_R;
				squish_comp |= squish::kBc4;
			} break;
			case Image::USED_CHANNELS_RG: {
				target_format = Image::FORMAT_RGTC_RG;
				squish_comp |= squish::kBc5;
			} break;
			case Image::USED_CHANNELS_RGB: {
				target_format = Image::FORMAT_DXT1;
				squish_comp |= squish::kDxt1;
			} break;
			case Image::USED_CHANNELS_RGBA: {
				//TODO, should convert both, then measure which one does a better job
				target_format = Image::FORMAT_DXT5;
				squish_comp |= squish::kDxt5;

			} break;
			default: {
				ERR_PRINT("Unknown image format, defaulting to RGBA8");
				break;
			}
		}

		Vector<uint8_t> data;
		int target_size = Image::get_image_data_size(w, h, target_format, p_image->has_mipmaps());
		int mm_count = p_image->has_mipmaps() ? Image::get_image_required_mipmaps(w, h, target_format) : 0;
		data.resize(target_size);
		int shift = Image::get_format_pixel_rshift(target_format);

		const uint8_t *rb = p_image->get_data().ptr();
		uint8_t *wb = data.ptrw();

		int dst_ofs = 0;

		for (int i = 0; i <= mm_count; i++) {
			int bw = w % 4 != 0 ? w + (4 - w % 4) : w;
			int bh = h % 4 != 0 ? h + (4 - h % 4) : h;

			int src_ofs = p_image->get_mipmap_offset(i);
			squish::CompressImage(&rb[src_ofs], w, h, &wb[dst_ofs], squish_comp);
			dst_ofs += (MAX(4, bw) * MAX(4, bh)) >> shift;
			w = MAX(w / 2, 1);
			h = MAX(h / 2, 1);
		}

		p_image->create(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format, data);
	}
}
