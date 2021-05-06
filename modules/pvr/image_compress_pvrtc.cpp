/*************************************************************************/
/*  image_compress_pvrtc.cpp                                             */
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

#include "image_compress_pvrtc.h"

#include "core/io/image.h"
#include "core/object/reference.h"

#include <PvrTcEncoder.h>
#include <RgbaBitmap.h>

static void _compress_pvrtc1_4bpp(Image *p_img) {
	Ref<Image> img = p_img->duplicate();

	bool make_mipmaps = false;
	if (!img->is_size_po2() || img->get_width() != img->get_height()) {
		make_mipmaps = img->has_mipmaps();
		img->resize_to_po2(true);
	}
	img->convert(Image::FORMAT_RGBA8);
	if (!img->has_mipmaps() && make_mipmaps) {
		img->generate_mipmaps();
	}

	bool use_alpha = img->detect_alpha();

	Ref<Image> new_img;
	new_img.instance();
	new_img->create(img->get_width(), img->get_height(), img->has_mipmaps(), use_alpha ? Image::FORMAT_PVRTC1_4A : Image::FORMAT_PVRTC1_4);

	Vector<uint8_t> data = new_img->get_data();
	{
		uint8_t *wr = data.ptrw();
		const uint8_t *r = img->get_data().ptr();

		for (int i = 0; i <= new_img->get_mipmap_count(); i++) {
			int ofs, size, w, h;
			img->get_mipmap_offset_size_and_dimensions(i, ofs, size, w, h);
			Javelin::RgbaBitmap bm(w, h);
			void *dst = (void *)bm.GetData();
			memcpy(dst, &r[ofs], size);
			Javelin::ColorRgba<unsigned char> *dp = bm.GetData();
			for (int j = 0; j < size / 4; j++) {
				// Red and blue colors are swapped.
				SWAP(dp[j].r, dp[j].b);
			}
			new_img->get_mipmap_offset_size_and_dimensions(i, ofs, size, w, h);
			Javelin::PvrTcEncoder::EncodeRgba4Bpp(&wr[ofs], bm);
		}
	}

	p_img->create(new_img->get_width(), new_img->get_height(), new_img->has_mipmaps(), new_img->get_format(), data);
}

void _register_pvrtc_compress_func() {
	Image::_image_compress_pvrtc1_4bpp_func = _compress_pvrtc1_4bpp;
}
