/*************************************************************************/
/*  image_loader_jpegd.cpp                                               */
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

#include "image_loader_jpegd.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

#include <jpgd.h>
#include <string.h>

Error jpeg_load_image_from_buffer(Image *p_image, const uint8_t *p_buffer, int p_buffer_len) {
	jpgd::jpeg_decoder_mem_stream mem_stream(p_buffer, p_buffer_len);

	jpgd::jpeg_decoder decoder(&mem_stream);

	if (decoder.get_error_code() != jpgd::JPGD_SUCCESS) {
		return ERR_CANT_OPEN;
	}

	const int image_width = decoder.get_width();
	const int image_height = decoder.get_height();
	const int comps = decoder.get_num_components();
	if (comps != 1 && comps != 3) {
		return ERR_FILE_CORRUPT;
	}

	if (decoder.begin_decoding() != jpgd::JPGD_SUCCESS) {
		return ERR_FILE_CORRUPT;
	}

	const int dst_bpl = image_width * comps;

	Vector<uint8_t> data;

	data.resize(dst_bpl * image_height);

	uint8_t *dw = data.ptrw();

	jpgd::uint8 *pImage_data = (jpgd::uint8 *)dw;

	for (int y = 0; y < image_height; y++) {
		const jpgd::uint8 *pScan_line;
		jpgd::uint scan_line_len;
		if (decoder.decode((const void **)&pScan_line, &scan_line_len) != jpgd::JPGD_SUCCESS) {
			return ERR_FILE_CORRUPT;
		}

		jpgd::uint8 *pDst = pImage_data + y * dst_bpl;

		if (comps == 1) {
			memcpy(pDst, pScan_line, dst_bpl);
		} else {
			// For images with more than 1 channel pScan_line will always point to a buffer
			// containing 32-bit RGBA pixels. Alpha is always 255 and we ignore it.
			for (int x = 0; x < image_width; x++) {
				pDst[0] = pScan_line[x * 4 + 0];
				pDst[1] = pScan_line[x * 4 + 1];
				pDst[2] = pScan_line[x * 4 + 2];
				pDst += 3;
			}
		}
	}

	//all good

	Image::Format fmt;
	if (comps == 1) {
		fmt = Image::FORMAT_L8;
	} else {
		fmt = Image::FORMAT_RGB8;
	}

	p_image->create(image_width, image_height, false, fmt, data);

	return OK;
}

Error ImageLoaderJPG::load_image(Ref<Image> p_image, FileAccess *f, bool p_force_linear, float p_scale) {
	Vector<uint8_t> src_image;
	uint64_t src_image_len = f->get_length();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	uint8_t *w = src_image.ptrw();

	f->get_buffer(&w[0], src_image_len);

	f->close();

	Error err = jpeg_load_image_from_buffer(p_image.ptr(), w, src_image_len);

	return err;
}

void ImageLoaderJPG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("jpg");
	p_extensions->push_back("jpeg");
}

static Ref<Image> _jpegd_mem_loader_func(const uint8_t *p_png, int p_size) {
	Ref<Image> img;
	img.instance();
	Error err = jpeg_load_image_from_buffer(img.ptr(), p_png, p_size);
	ERR_FAIL_COND_V(err, Ref<Image>());
	return img;
}

ImageLoaderJPG::ImageLoaderJPG() {
	Image::_jpg_mem_loader_func = _jpegd_mem_loader_func;
}
