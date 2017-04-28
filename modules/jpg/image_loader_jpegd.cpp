/*************************************************************************/
/*  image_loader_jpegd.cpp                                               */
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
#include "image_loader_jpegd.h"

#include "os/os.h"
#include "print_string.h"

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
	int comps = decoder.get_num_components();
	if (comps == 3)
		comps = 4; //weird

	if (decoder.begin_decoding() != jpgd::JPGD_SUCCESS)
		return ERR_FILE_CORRUPT;

	const int dst_bpl = image_width * comps;

	PoolVector<uint8_t> data;

	data.resize(dst_bpl * image_height);

	PoolVector<uint8_t>::Write dw = data.write();

	jpgd::uint8 *pImage_data = (jpgd::uint8 *)dw.ptr();

	for (int y = 0; y < image_height; y++) {
		const jpgd::uint8 *pScan_line;
		jpgd::uint scan_line_len;
		if (decoder.decode((const void **)&pScan_line, &scan_line_len) != jpgd::JPGD_SUCCESS) {
			return ERR_FILE_CORRUPT;
		}

		jpgd::uint8 *pDst = pImage_data + y * dst_bpl;
		memcpy(pDst, pScan_line, dst_bpl);
	}

	//all good

	Image::Format fmt;
	if (comps == 1)
		fmt = Image::FORMAT_L8;
	else
		fmt = Image::FORMAT_RGBA8;

	dw = PoolVector<uint8_t>::Write();
	p_image->create(image_width, image_height, 0, fmt, data);

	return OK;
}

Error ImageLoaderJPG::load_image(Image *p_image, FileAccess *f) {

	PoolVector<uint8_t> src_image;
	int src_image_len = f->get_len();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	PoolVector<uint8_t>::Write w = src_image.write();

	f->get_buffer(&w[0], src_image_len);

	f->close();

	Error err = jpeg_load_image_from_buffer(p_image, w.ptr(), src_image_len);

	w = PoolVector<uint8_t>::Write();

	return err;
}

void ImageLoaderJPG::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("jpg");
	p_extensions->push_back("jpeg");
}

static Image _jpegd_mem_loader_func(const uint8_t *p_png, int p_size) {

	Image img;
	Error err = jpeg_load_image_from_buffer(&img, p_png, p_size);
	if (err)
		ERR_PRINT("Couldn't initialize ImageLoaderJPG with the given resource.");

	return img;
}

ImageLoaderJPG::ImageLoaderJPG() {

	Image::_jpg_mem_loader_func = _jpegd_mem_loader_func;
}
