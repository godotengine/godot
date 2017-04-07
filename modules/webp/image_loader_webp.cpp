/*************************************************************************/
/*  image_loader_webp.cpp                                                */
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
#include "image_loader_webp.h"

#include "io/marshalls.h"
#include "os/os.h"
#include "print_string.h"

#include <stdlib.h>
#include <webp/decode.h>
#include <webp/encode.h>

static PoolVector<uint8_t> _webp_lossy_pack(const Image &p_image, float p_quality) {

	ERR_FAIL_COND_V(p_image.empty(), PoolVector<uint8_t>());

	Image img = p_image;
	if (img.detect_alpha())
		img.convert(Image::FORMAT_RGBA8);
	else
		img.convert(Image::FORMAT_RGB8);

	Size2 s(img.get_width(), img.get_height());
	PoolVector<uint8_t> data = img.get_data();
	PoolVector<uint8_t>::Read r = data.read();

	uint8_t *dst_buff = NULL;
	size_t dst_size = 0;
	if (img.get_format() == Image::FORMAT_RGB8) {

		dst_size = WebPEncodeRGB(r.ptr(), s.width, s.height, 3 * s.width, CLAMP(p_quality * 100.0, 0, 100.0), &dst_buff);
	} else {
		dst_size = WebPEncodeRGBA(r.ptr(), s.width, s.height, 4 * s.width, CLAMP(p_quality * 100.0, 0, 100.0), &dst_buff);
	}

	ERR_FAIL_COND_V(dst_size == 0, PoolVector<uint8_t>());
	PoolVector<uint8_t> dst;
	dst.resize(4 + dst_size);
	PoolVector<uint8_t>::Write w = dst.write();
	w[0] = 'W';
	w[1] = 'E';
	w[2] = 'B';
	w[3] = 'P';
	copymem(&w[4], dst_buff, dst_size);
	free(dst_buff);
	w = PoolVector<uint8_t>::Write();
	return dst;
}

static Image _webp_lossy_unpack(const PoolVector<uint8_t> &p_buffer) {

	int size = p_buffer.size() - 4;
	ERR_FAIL_COND_V(size <= 0, Image());
	PoolVector<uint8_t>::Read r = p_buffer.read();

	ERR_FAIL_COND_V(r[0] != 'W' || r[1] != 'E' || r[2] != 'B' || r[3] != 'P', Image());
	WebPBitstreamFeatures features;
	if (WebPGetFeatures(&r[4], size, &features) != VP8_STATUS_OK) {
		ERR_EXPLAIN("Error unpacking WEBP image:");
		ERR_FAIL_V(Image());
	}

	//print_line("width: "+itos(features.width));
	//print_line("height: "+itos(features.height));
	//print_line("alpha: "+itos(features.has_alpha));

	PoolVector<uint8_t> dst_image;
	int datasize = features.width * features.height * (features.has_alpha ? 4 : 3);
	dst_image.resize(datasize);

	PoolVector<uint8_t>::Write dst_w = dst_image.write();

	bool errdec = false;
	if (features.has_alpha) {
		errdec = WebPDecodeRGBAInto(&r[4], size, dst_w.ptr(), datasize, 4 * features.width) == NULL;
	} else {
		errdec = WebPDecodeRGBInto(&r[4], size, dst_w.ptr(), datasize, 3 * features.width) == NULL;
	}

	//ERR_EXPLAIN("Error decoding webp! - "+p_file);
	ERR_FAIL_COND_V(errdec, Image());

	dst_w = PoolVector<uint8_t>::Write();

	return Image(features.width, features.height, 0, features.has_alpha ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8, dst_image);
}

Error ImageLoaderWEBP::load_image(Image *p_image, FileAccess *f) {

	uint32_t size = f->get_len();
	PoolVector<uint8_t> src_image;
	src_image.resize(size);

	WebPBitstreamFeatures features;

	PoolVector<uint8_t>::Write src_w = src_image.write();
	f->get_buffer(src_w.ptr(), size);
	ERR_FAIL_COND_V(f->eof_reached(), ERR_FILE_EOF);

	if (WebPGetFeatures(src_w.ptr(), size, &features) != VP8_STATUS_OK) {
		f->close();
		//ERR_EXPLAIN("Error decoding WEBP image: "+p_file);
		ERR_FAIL_V(ERR_FILE_CORRUPT);
	}

	print_line("width: " + itos(features.width));
	print_line("height: " + itos(features.height));
	print_line("alpha: " + itos(features.has_alpha));

	src_w = PoolVector<uint8_t>::Write();

	PoolVector<uint8_t> dst_image;
	int datasize = features.width * features.height * (features.has_alpha ? 4 : 3);
	dst_image.resize(datasize);

	PoolVector<uint8_t>::Read src_r = src_image.read();
	PoolVector<uint8_t>::Write dst_w = dst_image.write();

	bool errdec = false;
	if (features.has_alpha) {
		errdec = WebPDecodeRGBAInto(src_r.ptr(), size, dst_w.ptr(), datasize, 4 * features.width) == NULL;
	} else {
		errdec = WebPDecodeRGBInto(src_r.ptr(), size, dst_w.ptr(), datasize, 3 * features.width) == NULL;
	}

	//ERR_EXPLAIN("Error decoding webp! - "+p_file);
	ERR_FAIL_COND_V(errdec, ERR_FILE_CORRUPT);

	src_r = PoolVector<uint8_t>::Read();
	dst_w = PoolVector<uint8_t>::Write();

	*p_image = Image(features.width, features.height, 0, features.has_alpha ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8, dst_image);

	return OK;
}

void ImageLoaderWEBP::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("webp");
}

ImageLoaderWEBP::ImageLoaderWEBP() {

	Image::lossy_packer = _webp_lossy_pack;
	Image::lossy_unpacker = _webp_lossy_unpack;
}
