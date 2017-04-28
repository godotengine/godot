/*************************************************************************/
/*  image_compress_squish.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "print_string.h"

#include <squish.h>

void image_compress_squish(Image *p_image) {

	int w = p_image->get_width();
	int h = p_image->get_height();

	if (!p_image->has_mipmaps()) {
		ERR_FAIL_COND(!w || w % 4 != 0);
		ERR_FAIL_COND(!h || h % 4 != 0);
	} else {
		ERR_FAIL_COND(!w || w != nearest_power_of_2(w));
		ERR_FAIL_COND(!h || h != nearest_power_of_2(h));
	};

	if (p_image->get_format() >= Image::FORMAT_DXT1)
		return; //do not compress, already compressed

	int shift = 0;
	int squish_comp = squish::kColourRangeFit;
	Image::Format target_format;

	if (p_image->get_format() == Image::FORMAT_LA8) {
		//compressed normalmap
		target_format = Image::FORMAT_DXT5;
		squish_comp |= squish::kDxt5;
	} else if (p_image->detect_alpha() != Image::ALPHA_NONE) {

		target_format = Image::FORMAT_DXT3;
		squish_comp |= squish::kDxt3;
	} else {
		target_format = Image::FORMAT_DXT1;
		shift = 1;
		squish_comp |= squish::kDxt1;
	}

	p_image->convert(Image::FORMAT_RGBA8); //always expects rgba

	PoolVector<uint8_t> data;
	int target_size = Image::get_image_data_size(w, h, target_format, p_image->has_mipmaps() ? -1 : 0);
	int mm_count = p_image->has_mipmaps() ? Image::get_image_required_mipmaps(w, h, target_format) : 0;
	data.resize(target_size);

	PoolVector<uint8_t>::Read rb = p_image->get_data().read();
	PoolVector<uint8_t>::Write wb = data.write();

	int dst_ofs = 0;

	for (int i = 0; i <= mm_count; i++) {

		int src_ofs = p_image->get_mipmap_offset(i);
		squish::CompressImage(&rb[src_ofs], w, h, &wb[dst_ofs], squish_comp);
		dst_ofs += (MAX(4, w) * MAX(4, h)) >> shift;
		w >>= 1;
		h >>= 1;
	}

	rb = PoolVector<uint8_t>::Read();
	wb = PoolVector<uint8_t>::Write();

	p_image->create(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format, data);
}
