/**************************************************************************/
/*  image_loader_png.cpp                                                  */
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

#include "image_loader_png.h"

#include "core/os/os.h"
#include "core/print_string.h"
#include "drivers/png/png_driver_common.h"

#include <string.h>

Error ImageLoaderPNG::load_image(Ref<Image> p_image, FileAccess *f, bool p_force_linear, float p_scale) {
	const uint64_t buffer_size = f->get_len();
	PoolVector<uint8_t> file_buffer;
	Error err = file_buffer.resize(buffer_size);
	if (err) {
		f->close();
		return err;
	}
	{
		PoolVector<uint8_t>::Write writer = file_buffer.write();
		f->get_buffer(writer.ptr(), buffer_size);
		f->close();
	}
	PoolVector<uint8_t>::Read reader = file_buffer.read();
	return PNGDriverCommon::png_to_image(reader.ptr(), buffer_size, p_force_linear, p_image);
}

void ImageLoaderPNG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("png");
}

Ref<Image> ImageLoaderPNG::load_mem_png(const uint8_t *p_png, int p_size) {
	Ref<Image> img;
	img.instance();

	// the value of p_force_linear does not matter since it only applies to 16 bit
	Error err = PNGDriverCommon::png_to_image(p_png, p_size, false, img);
	ERR_FAIL_COND_V(err, Ref<Image>());

	return img;
}

Ref<Image> ImageLoaderPNG::lossless_unpack_png(const PoolVector<uint8_t> &p_data) {
	const int len = p_data.size();
	ERR_FAIL_COND_V(len < 4, Ref<Image>());
	PoolVector<uint8_t>::Read r = p_data.read();
	ERR_FAIL_COND_V(r[0] != 'P' || r[1] != 'N' || r[2] != 'G' || r[3] != ' ', Ref<Image>());
	return load_mem_png(&r[4], len - 4);
}

PoolVector<uint8_t> ImageLoaderPNG::lossless_pack_png(const Ref<Image> &p_image) {
	PoolVector<uint8_t> out_buffer;

	// add Godot's own "PNG " prefix
	if (out_buffer.resize(4) != OK) {
		ERR_FAIL_V(PoolVector<uint8_t>());
	}

	// scope for writer lifetime
	{
		// must be closed before call to image_to_png
		PoolVector<uint8_t>::Write writer = out_buffer.write();
		memcpy(writer.ptr(), "PNG ", 4);
	}

	Error err = PNGDriverCommon::image_to_png(p_image, out_buffer);
	if (err) {
		ERR_FAIL_V(PoolVector<uint8_t>());
	}

	return out_buffer;
}

ImageLoaderPNG::ImageLoaderPNG() {
	Image::_png_mem_loader_func = load_mem_png;
	Image::png_unpacker = lossless_unpack_png;
	Image::png_packer = lossless_pack_png;
}
