/**************************************************************************/
/*  image_loader_qoi.cpp                                                  */
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

#include "image_loader_qoi.h"

#include <string.h>

#include "core/io/file_access_memory.h"

Error ImageLoaderqoi::load_image(Ref<Image> p_image, Ref<FileAccess> f, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	Error err = ERR_INVALID_DATA;
	Vector<uint8_t> data;
	f->big_endian = true;

	//qoi data
	qoi_header_s qoi_header;
	qoi_rgba_t index[64];
	qoi_rgba_t px;
	int px_len, px_pos;
	int run = 0;
	int write_px_pos = 0;

	qoi_header.magic = f->get_32();

	//Every qoi file must start with "qoif".
	if (qoi_header.magic != QOI_MAGIC) {
		ERR_FAIL_COND_V_MSG(qoi_header.magic != QOI_MAGIC, ERR_BUG, "QOI images must start with 'qoif'.");
		return err;
	}

	qoi_header.width = f->get_32();
	qoi_header.height = f->get_32();
	qoi_header.channels = f->get_8();
	qoi_header.colorspace = f->get_8();

	if (qoi_header.width == 0 || qoi_header.height == 0 || qoi_header.channels < 3 ||
			qoi_header.channels > 4 || qoi_header.colorspace > 1 ||
			qoi_header.height >= QOI_PIXELS_MAX / qoi_header.width) {
		ERR_FAIL_COND_V_MSG(qoi_header.width == 0 || qoi_header.height == 0 || qoi_header.channels < 3 ||
						qoi_header.channels > 4 || qoi_header.colorspace > 1 ||
						qoi_header.height >= QOI_PIXELS_MAX / qoi_header.width,
				ERR_BUG, "Couldn't parse the qoi image data.");
		return err;
	}

	px_len = qoi_header.width * qoi_header.height * qoi_header.channels;
	px.rgba.r = 0;
	px.rgba.g = 0;
	px.rgba.b = 0;
	px.rgba.a = 255;

	data.resize(qoi_header.width * qoi_header.height * 4);
	uint8_t *data_w = data.ptrw();

	for (px_pos = 0; px_pos < px_len; px_pos += qoi_header.channels) {
		write_px_pos += 4;
		if (run > 0) {
			run--;
		} else {
			int b1 = f->get_8();
			if (b1 == QOI_OP_RGB) {
				px.rgba.r = f->get_8();
				px.rgba.g = f->get_8();
				px.rgba.b = f->get_8();
			} else if (b1 == QOI_OP_RGBA) {
				px.rgba.r = f->get_8();
				px.rgba.g = f->get_8();
				px.rgba.b = f->get_8();
				px.rgba.a = f->get_8();
			} else if ((b1 & QOI_MASK_2) == QOI_OP_INDEX) {
				px = index[b1];
			} else if ((b1 & QOI_MASK_2) == QOI_OP_DIFF) {
				px.rgba.r += ((b1 >> 4) & 0x03) - 2;
				px.rgba.g += ((b1 >> 2) & 0x03) - 2;
				px.rgba.b += (b1 & 0x03) - 2;
			} else if ((b1 & QOI_MASK_2) == QOI_OP_LUMA) {
				int b2 = f->get_8();
				int vg = (b1 & 0x3f) - 32;
				px.rgba.r += vg - 8 + ((b2 >> 4) & 0x0f);
				px.rgba.g += vg;
				px.rgba.b += vg - 8 + (b2 & 0x0f);
			} else if ((b1 & QOI_MASK_2) == QOI_OP_RUN) {
				run = (b1 & 0x3f);
			}
			index[QOI_COLOR_HASH(px) % 64] = px;
		}

		data_w[write_px_pos + 0] = px.rgba.r;
		data_w[write_px_pos + 1] = px.rgba.g;
		data_w[write_px_pos + 2] = px.rgba.b;

		if (qoi_header.channels == 4) {
			data_w[write_px_pos + 3] = px.rgba.a;
		} else {
			data_w[write_px_pos + 3] = 255;
		}
	}
	p_image->set_data(qoi_header.width, qoi_header.height, false, Image::FORMAT_RGBA8, data);
	err = OK;
	return err;
}

void ImageLoaderqoi::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("qoi");
}

static Ref<Image> _qoi_mem_loader_func(const uint8_t *p_qoi, int p_size) {
	Ref<FileAccessMemory> memfile;
	memfile.instantiate();
	Error open_memfile_error = memfile->open_custom(p_qoi, p_size);
	ERR_FAIL_COND_V_MSG(open_memfile_error, Ref<Image>(), "Could not create memfile for qoi image buffer.");

	Ref<Image> img;
	img.instantiate();
	Error load_error = ImageLoaderqoi().load_image(img, memfile, false, 1.0f);
	ERR_FAIL_COND_V_MSG(load_error, Ref<Image>(), "Failed to load qoi image.");
	return img;
}

ImageLoaderqoi::ImageLoaderqoi() {
	Image::_qoi_mem_loader_func = _qoi_mem_loader_func;
}
