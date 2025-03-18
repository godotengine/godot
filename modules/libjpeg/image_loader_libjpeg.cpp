/**************************************************************************/
/*  image_loader_libjpeg.cpp                                              */
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

#include "image_loader_libjpeg.h"

#include <jpeglib.h>

#include <string.h>

METHODDEF(noreturn_t)
error_dontexit(j_common_ptr cinfo) {
	(*cinfo->err->output_message)(cinfo);
	jpeg_destroy(cinfo);
}

Error libjpeg_load_image_from_buffer(Image *p_image, const uint8_t *p_buffer, int p_buffer_len) {
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);
	jerr.error_exit = error_dontexit;
	jpeg_create_decompress(&cinfo);
	jpeg_mem_src(&cinfo, p_buffer, p_buffer_len);

	if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
		return FAILED;
	}

	jpeg_start_decompress(&cinfo);

	const unsigned int output_width = cinfo.output_width;
	const unsigned int output_height = cinfo.output_height;
	const unsigned int row_stride = output_width * cinfo.output_components;

	Vector<uint8_t> data;
	data.resize(row_stride * output_height);
	uint8_t *dw = data.ptrw();

	while (cinfo.output_scanline < output_height) {
		unsigned char *buffer_array[1];
		buffer_array[0] = dw + (cinfo.output_scanline) * row_stride;
		jpeg_read_scanlines(&cinfo, buffer_array, 1);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	Image::Format fmt;
	switch (cinfo.out_color_space) {
		case JCS_GRAYSCALE:
			fmt = Image::FORMAT_L8;
			break;

		case JCS_RGB:
			fmt = Image::FORMAT_RGB8;
			break;

		default:
			// TODO assert or warn?
			fmt = Image::FORMAT_RGB8;
	}

	p_image->set_data(output_width, output_height, false, fmt, data);

	return OK;
}

Error ImageLoaderLibJPEG::load_image(Ref<Image> p_image, Ref<FileAccess> f, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	Vector<uint8_t> src_image;
	uint64_t src_image_len = f->get_length();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	uint8_t *w = src_image.ptrw();

	f->get_buffer(&w[0], src_image_len);

	Error err = libjpeg_load_image_from_buffer(p_image.ptr(), w, src_image_len);

	return err;
}

void ImageLoaderLibJPEG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("jpg");
	p_extensions->push_back("jpeg");
}

static Ref<Image> _libjpegd_mem_loader_func(const uint8_t *p_png, int p_size) {
	Ref<Image> img;
	img.instantiate();
	Error err = libjpeg_load_image_from_buffer(img.ptr(), p_png, p_size);
	ERR_FAIL_COND_V(err, Ref<Image>());
	return img;
}

static Vector<uint8_t> _libjpeg_buffer_save_func(const Ref<Image> &p_img, float p_quality) {
	// TODO implementation
	return Vector<uint8_t>();
}

static Error _libjpeg_save_func(const String &p_path, const Ref<Image> &p_img, float p_quality) {
	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V_MSG(err, err, vformat("Can't save JPG at path: '%s'.", p_path));
	// TODO implementation
	return FAILED;
}

ImageLoaderLibJPEG::ImageLoaderLibJPEG() {
	Image::_jpg_mem_loader_func = _libjpegd_mem_loader_func;
	Image::save_jpg_func = _libjpeg_save_func;
	Image::save_jpg_buffer_func = _libjpeg_buffer_save_func;
}
