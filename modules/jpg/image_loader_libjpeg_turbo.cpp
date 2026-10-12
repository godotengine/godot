/**************************************************************************/
/*  image_loader_libjpeg_turbo.cpp                                        */
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

#include "image_loader_libjpeg_turbo.h"

#include <turbojpeg.h>

Error jpeg_turbo_load_image_from_buffer(Image *p_image, const uint8_t *p_buffer, int p_buffer_len) {
	tjhandle tj_instance = tj3Init(TJINIT_DECOMPRESS);
	if (tj_instance == nullptr) {
		return FAILED;
	}

	if (tj3DecompressHeader(tj_instance, p_buffer, p_buffer_len) < 0) {
		tj3Destroy(tj_instance);
		return ERR_FILE_CORRUPT;
	}

	const int width = tj3Get(tj_instance, TJPARAM_JPEGWIDTH);
	const int height = tj3Get(tj_instance, TJPARAM_JPEGHEIGHT);
	const TJCS colorspace = (TJCS)tj3Get(tj_instance, TJPARAM_COLORSPACE);

	if (tj3Get(tj_instance, TJPARAM_PRECISION) > 8) {
		// TODO: Proceed anyway and convert to rgb8?
		ERR_PRINT("JPEGTurbo: JPEG files with precision > 8 are not currently supported.");
		tj3Destroy(tj_instance);
		return ERR_UNAVAILABLE;
	}

	TJPF tj_pixel_format;
	Image::Format gd_pixel_format;
	if (colorspace == TJCS_GRAY) {
		tj_pixel_format = TJPF_GRAY;
		gd_pixel_format = Image::FORMAT_L8;
	} else if (colorspace == TJCS_CMYK || colorspace == TJCS_YCCK) {
		ERR_PRINT("JPEGTurbo: JPEG files with CMYK or YCCK colorspaces are not currently supported.");
		tj3Destroy(tj_instance);
		return ERR_UNAVAILABLE;
	} else {
		// Other color spaces should be RGB.
		tj_pixel_format = TJPF_RGB;
		gd_pixel_format = Image::FORMAT_RGB8;
	}

	PoolVector<uint8_t> data;
	data.resize(width * height * tjPixelSize[tj_pixel_format]);
	PoolVector<uint8_t>::Write dw = data.write();

	if (tj3Decompress8(tj_instance, p_buffer, p_buffer_len, dw.ptr(), 0, tj_pixel_format) < 0) {
		if (tj3GetErrorCode(tj_instance) == TJERR_FATAL) {
			tj3Destroy(tj_instance);
			return ERR_FILE_CORRUPT;
		} else {
			WARN_PRINT(String::utf8(tj3GetErrorStr(tj_instance)));
		}
	}

	tj3Destroy(tj_instance);
	p_image->create(width, height, false, gd_pixel_format, data);
	return OK;
}

Error ImageLoaderLibJPEGTurbo::load_image(Ref<Image> p_image, FileAccess *f, bool p_force_linear, float p_scale) {	
	PoolVector<uint8_t> src_image;
	uint64_t src_image_len = f->get_len();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	PoolVector<uint8_t>::Write w = src_image.write();

	f->get_buffer(&w[0], src_image_len);

	f->close();	
	

	Error err = jpeg_turbo_load_image_from_buffer(p_image.ptr(), w.ptr(), src_image_len);

	return err;
}

void ImageLoaderLibJPEGTurbo::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("jpg");
	p_extensions->push_back("jpeg");
}

static Ref<Image> _jpeg_turbo_mem_loader_func(const uint8_t *p_png, int p_size) {
	Ref<Image> img;
	img.instance();
	Error err = jpeg_turbo_load_image_from_buffer(img.ptr(), p_png, p_size);
	ERR_FAIL_COND_V(err, Ref<Image>());
	return img;
}

ImageLoaderLibJPEGTurbo::ImageLoaderLibJPEGTurbo() {
	Image::_jpg_mem_loader_func = _jpeg_turbo_mem_loader_func;
}
