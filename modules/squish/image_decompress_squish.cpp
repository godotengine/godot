/*************************************************************************/
/*  image_decompress_squish.cpp                                          */
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

#include "image_decompress_squish.h"

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
