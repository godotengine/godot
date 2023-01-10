/**************************************************************************/
/*  image_compress_pvrtc.cpp                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "image_compress_pvrtc.h"

#include "core/image.h"
#include "core/reference.h"

#include <PvrTcEncoder.h>
#include <RgbaBitmap.h>

static void _compress_pvrtc4(Image *p_img) {
	Ref<Image> img = p_img->duplicate();

	bool make_mipmaps = false;
	if (!img->is_size_po2() || img->get_width() != img->get_height()) {
		make_mipmaps = img->has_mipmaps();
		img->resize_to_po2(true);
		// Resizing can fail for some formats
		if (!img->is_size_po2() || img->get_width() != img->get_height()) {
			ERR_FAIL_MSG("Failed to resize the image for compression.");
		}
	}
	img->convert(Image::FORMAT_RGBA8);
	if (!img->has_mipmaps() && make_mipmaps) {
		img->generate_mipmaps();
	}

	bool use_alpha = img->detect_alpha();

	Ref<Image> new_img;
	new_img.instance();
	new_img->create(img->get_width(), img->get_height(), img->has_mipmaps(), use_alpha ? Image::FORMAT_PVRTC4A : Image::FORMAT_PVRTC4);

	PoolVector<uint8_t> data = new_img->get_data();
	{
		PoolVector<uint8_t>::Write wr = data.write();
		PoolVector<uint8_t>::Read r = img->get_data().read();

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
	// FIXME: We claim to support PVRTC2 but use the same method as for PVRTC4.
	Image::_image_compress_pvrtc2_func = _compress_pvrtc4;
	Image::_image_compress_pvrtc4_func = _compress_pvrtc4;
}
