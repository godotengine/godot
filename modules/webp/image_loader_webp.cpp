/*************************************************************************/
/*  image_loader_webp.cpp                                                */
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

#include "image_loader_webp.h"

#include "core/io/marshalls.h"
#include "core/os/os.h"
#include "core/print_string.h"
#include "core/project_settings.h"

#include <stdlib.h>
#include <webp/decode.h>
#include <webp/encode.h>

static PoolVector<uint8_t> _webp_lossy_pack(const Ref<Image> &p_image, float p_quality) {
	ERR_FAIL_COND_V(p_image.is_null() || p_image->empty(), PoolVector<uint8_t>());

	Ref<Image> img = p_image->duplicate();
	if (img->detect_alpha()) {
		img->convert(Image::FORMAT_RGBA8);
	} else {
		img->convert(Image::FORMAT_RGB8);
	}

	Size2 s(img->get_width(), img->get_height());
	PoolVector<uint8_t> data = img->get_data();
	PoolVector<uint8_t>::Read r = data.read();

	uint8_t *dst_buff = nullptr;
	size_t dst_size = 0;
	if (img->get_format() == Image::FORMAT_RGB8) {
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
	memcpy(&w[4], dst_buff, dst_size);
	WebPFree(dst_buff);
	w.release();
	return dst;
}

static PoolVector<uint8_t> _webp_lossless_pack(const Ref<Image> &p_image) {
	ERR_FAIL_COND_V(p_image.is_null() || p_image->empty(), PoolVector<uint8_t>());

	int compression_level = ProjectSettings::get_singleton()->get("rendering/misc/lossless_compression/webp_compression_level");
	compression_level = CLAMP(compression_level, 0, 9);

	Ref<Image> img = p_image->duplicate();
	if (img->detect_alpha()) {
		img->convert(Image::FORMAT_RGBA8);
	} else {
		img->convert(Image::FORMAT_RGB8);
	}

	Size2 s(img->get_width(), img->get_height());
	PoolVector<uint8_t> data = img->get_data();
	PoolVector<uint8_t>::Read r = data.read();

	// we need to use the more complex API in order to access the 'exact' flag...

	WebPConfig config;
	WebPPicture pic;
	if (!WebPConfigInit(&config) || !WebPConfigLosslessPreset(&config, compression_level) || !WebPPictureInit(&pic)) {
		ERR_FAIL_V(PoolVector<uint8_t>());
	}

	WebPMemoryWriter wrt;
	config.exact = 1;
	pic.use_argb = 1;
	pic.width = s.width;
	pic.height = s.height;
	pic.writer = WebPMemoryWrite;
	pic.custom_ptr = &wrt;
	WebPMemoryWriterInit(&wrt);

	bool success_import = false;
	if (img->get_format() == Image::FORMAT_RGB8) {
		success_import = WebPPictureImportRGB(&pic, r.ptr(), 3 * s.width);
	} else {
		success_import = WebPPictureImportRGBA(&pic, r.ptr(), 4 * s.width);
	}
	bool success_encode = false;
	if (success_import) {
		success_encode = WebPEncode(&config, &pic);
	}
	WebPPictureFree(&pic);

	if (!success_encode) {
		WebPMemoryWriterClear(&wrt);
		ERR_FAIL_V_MSG(PoolVector<uint8_t>(), "WebP packing failed.");
	}

	// copy from wrt
	PoolVector<uint8_t> dst;
	dst.resize(4 + wrt.size);
	PoolVector<uint8_t>::Write w = dst.write();
	w[0] = 'W';
	w[1] = 'E';
	w[2] = 'B';
	w[3] = 'P';
	memcpy(&w[4], wrt.mem, wrt.size);
	w.release();
	WebPMemoryWriterClear(&wrt);

	return dst;
}

static Ref<Image> _webp_lossy_unpack(const PoolVector<uint8_t> &p_buffer) {
	int size = p_buffer.size() - 4;
	ERR_FAIL_COND_V(size <= 0, Ref<Image>());
	PoolVector<uint8_t>::Read r = p_buffer.read();

	ERR_FAIL_COND_V(r[0] != 'W' || r[1] != 'E' || r[2] != 'B' || r[3] != 'P', Ref<Image>());
	WebPBitstreamFeatures features;
	if (WebPGetFeatures(&r[4], size, &features) != VP8_STATUS_OK) {
		ERR_FAIL_V_MSG(Ref<Image>(), "Error unpacking WEBP image.");
	}

	/*
	print_line("width: "+itos(features.width));
	print_line("height: "+itos(features.height));
	print_line("alpha: "+itos(features.has_alpha));
	*/

	PoolVector<uint8_t> dst_image;
	int datasize = features.width * features.height * (features.has_alpha ? 4 : 3);
	dst_image.resize(datasize);

	PoolVector<uint8_t>::Write dst_w = dst_image.write();

	bool errdec = false;
	if (features.has_alpha) {
		errdec = WebPDecodeRGBAInto(&r[4], size, dst_w.ptr(), datasize, 4 * features.width) == nullptr;
	} else {
		errdec = WebPDecodeRGBInto(&r[4], size, dst_w.ptr(), datasize, 3 * features.width) == nullptr;
	}

	ERR_FAIL_COND_V_MSG(errdec, Ref<Image>(), "Failed decoding WebP image.");

	dst_w.release();

	Ref<Image> img = memnew(Image(features.width, features.height, 0, features.has_alpha ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8, dst_image));
	return img;
}

Error webp_load_image_from_buffer(Image *p_image, const uint8_t *p_buffer, int p_buffer_len) {
	ERR_FAIL_NULL_V(p_image, ERR_INVALID_PARAMETER);

	WebPBitstreamFeatures features;
	if (WebPGetFeatures(p_buffer, p_buffer_len, &features) != VP8_STATUS_OK) {
		ERR_FAIL_V(ERR_FILE_CORRUPT);
	}

	PoolVector<uint8_t> dst_image;
	int datasize = features.width * features.height * (features.has_alpha ? 4 : 3);
	dst_image.resize(datasize);
	PoolVector<uint8_t>::Write dst_w = dst_image.write();

	bool errdec = false;
	if (features.has_alpha) {
		errdec = WebPDecodeRGBAInto(p_buffer, p_buffer_len, dst_w.ptr(), datasize, 4 * features.width) == nullptr;
	} else {
		errdec = WebPDecodeRGBInto(p_buffer, p_buffer_len, dst_w.ptr(), datasize, 3 * features.width) == nullptr;
	}
	dst_w.release();

	ERR_FAIL_COND_V_MSG(errdec, ERR_FILE_CORRUPT, "Failed decoding WebP image.");

	p_image->create(features.width, features.height, false, features.has_alpha ? Image::FORMAT_RGBA8 : Image::FORMAT_RGB8, dst_image);

	return OK;
}

static Ref<Image> _webp_mem_loader_func(const uint8_t *p_png, int p_size) {
	Ref<Image> img;
	img.instance();
	Error err = webp_load_image_from_buffer(img.ptr(), p_png, p_size);
	ERR_FAIL_COND_V(err, Ref<Image>());
	return img;
}

Error ImageLoaderWEBP::load_image(Ref<Image> p_image, FileAccess *f, bool p_force_linear, float p_scale) {
	PoolVector<uint8_t> src_image;
	uint64_t src_image_len = f->get_len();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	PoolVector<uint8_t>::Write w = src_image.write();

	f->get_buffer(&w[0], src_image_len);

	f->close();

	Error err = webp_load_image_from_buffer(p_image.ptr(), w.ptr(), src_image_len);

	return err;
}

void ImageLoaderWEBP::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("webp");
}

ImageLoaderWEBP::ImageLoaderWEBP() {
	Image::_webp_mem_loader_func = _webp_mem_loader_func;
	Image::webp_lossy_packer = _webp_lossy_pack;
	Image::webp_lossless_packer = _webp_lossless_pack;
	Image::webp_unpacker = _webp_lossy_unpack;
}
