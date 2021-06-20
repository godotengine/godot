/*************************************************************************/
/*  image_compress_bc7e.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
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

#ifndef IMAGE_COMPRESS_BC7E_H
#define IMAGE_COMPRESS_BC7E_H

#include "bc7enc_rdo.h"

#include "core/error/error_macros.h"
#include "thirdparty/bc7e/rdo_bc_encoder.h"

#include "core/io/image.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/os/threaded_array_processor.h"
#include "core/string/print_string.h"
#include <stdint.h>

void image_compress_bc7e(Image *p_image, float p_lossy_quality, Image::UsedChannels p_channels) {
	Image::Format input_format = p_image->get_format();
	if (p_image->is_compressed()) {
		return; //do not compress, already compressed
	}
	if (input_format != Image::FORMAT_RGB8 && input_format != Image::FORMAT_RGBA8) {
		return;
	}
	Image::Format target_format = Image::FORMAT_BPTC_RGBA;

	rdo_bc::rdo_bc_encoder encoder;
	int uber_level = 4;
	if (Math::is_equal_approx(p_lossy_quality, 1.0f)) {
		uber_level = 4;
	} else if (p_lossy_quality > 0.85) {
		uber_level = 4;
	} else if (p_lossy_quality > 0.75) {
		uber_level = 3;
	} else if (p_lossy_quality > 0.55) {
		uber_level = 3;
	} else if (p_lossy_quality > 0.35) {
		uber_level = 3;
	} else if (p_lossy_quality > 0.15) {
		uber_level = 2;
	}

	rdo_bc::rdo_bc_params pack_params;
	pack_params.m_rdo_lambda = 0.0f;
	pack_params.m_rdo_multithreading = false;
	switch (uber_level) {
		case 0:
			pack_params.m_bc7_uber_level = 0;
			break;
		case 1:
			pack_params.m_bc7_uber_level = 1;
			break;
		case 2:
			pack_params.m_bc7_uber_level = 2;
			break;
		case 3:
			pack_params.m_bc7_uber_level = 3;
			break;
		case 4:
			pack_params.m_bc7_uber_level = 4;
			break;
		case 5:
			pack_params.m_bc7_uber_level = 5;
			break;
		case 6:
		default:
			pack_params.m_bc7_uber_level = 3;
			break;
	}
	Ref<Image> new_img;
	new_img.instantiate();
	new_img->create(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format);

	Vector<uint8_t> data = new_img->get_data();

	const bool mipmaps = new_img->has_mipmaps();
	const int width = new_img->get_width();
	const int height = new_img->get_height();

	uint8_t *wr = data.ptrw();

	int mip_count = mipmaps ? Image::get_image_required_mipmaps(width, height, target_format) : 0;
	for (int i = 0; i < mip_count + 1; i++) {
		int ofs = 0, size = 0;
		int orig_mip_w = 0, orig_mip_h = 0;
		new_img->get_mipmap_offset_size_and_dimensions(i, ofs, size, orig_mip_w, orig_mip_h);


		// Block size. Align stride to multiple of 4 (RGBA8).
		int mip_w = (orig_mip_w + 3) & ~3;
		int mip_h = (orig_mip_h + 3) & ~3;
		
		Ref<Image> mip_image = p_image->duplicate();
		mip_image->resize(mip_w, mip_h);

		// Get mip data from source image for reading.
		int src_mip_ofs = p_image->get_mipmap_offset(i);
		const uint8_t *src_read = p_image->get_data().ptr();
		const uint32_t *src_mip_read = (const uint32_t *)&src_read[src_mip_ofs];

		int dest_size = Image::get_image_data_size(width, height, target_format, mipmaps);
		Vector<uint8_t> dest_data;
		dest_data.resize(dest_size);
		uint8_t *dest_write = dest_data.ptrw();

		int mip_count = mipmaps ? Image::get_image_required_mipmaps(width, height, target_format) : 0;
		utils::image_u8 mip_source_image(mip_w, mip_h);
		PackedByteArray image_data = p_image->get_data();

		// Pad textures to nearest block by smearing.
		if (mip_w != orig_mip_w || mip_h != orig_mip_h) {
			int x = 0, y = 0;
			for (y = 0; y < orig_mip_h; y++) {
				for (x = 0; x < orig_mip_w; x++) {
					Color c = p_image->get_pixel(x - 1, mip_w * y);
					mip_image->set_pixel(x, mip_w * y, c);
				}
			}
			// Then, smear in y.
			for (; y < mip_h; y++) {
				for (x = 0; x < mip_w; x++) {
					Color c = p_image->get_pixel(x - mip_w, mip_w * y);
					mip_image->set_pixel(x, mip_w * y, c);
				}
			}
		}
		for (int32_t y = 0; y < mip_h; y++) {
			for (int32_t x = 0; x < mip_w; x++) {
				Color c = mip_image->get_pixel(x, y);
				uint32_t color = src_mip_read[mip_w * y + x];
				uint8_t r = (color & 0x000000FF);
				uint8_t g = (color & 0x0000FF00) >> 8;
				uint8_t b = (color & 0x00FF0000) >> 16;
				uint8_t a = (color & 0xFF000000) >> 24;
				mip_source_image(x, y).set(r, g, b, a);
			}
		}

		ERR_FAIL_COND_MSG(!encoder.init(mip_source_image, pack_params), "bc7enc_rdo did not begin.");
		ERR_FAIL_COND_MSG(!encoder.encode(), "bc7enc_rdo could not encode.");
		Vector<uint8_t> packed_image;
		packed_image.resize(encoder.get_total_blocks_size_in_bytes());

		int target_size = packed_image.size();
		ERR_FAIL_COND(target_size != size);
		memcpy(&wr[ofs], encoder.get_blocks(), size);
	}

	Ref<Image> ref_image = p_image->duplicate();
	p_image->create(new_img->get_width(), new_img->get_height(), new_img->has_mipmaps(), new_img->get_format(), data);
}

#endif // IMAGE_COMPRESS_BC7E_H
