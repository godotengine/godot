/**************************************************************************/
/*  image_loader_qoi.h                                                    */
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

#ifndef IMAGE_LOADER_QOI_H
#define IMAGE_LOADER_QOI_H

#include "core/io/image_loader.h"

class ImageLoaderqoi : public ImageFormatLoader {
protected:
	static const uint32_t QOI_MAGIC = 0x716f6966; // qoif in hex

	static const unsigned int QOI_PIXELS_MAX = 400000000;

	struct qoi_header_s {
		uint32_t magic;
		uint32_t width;
		uint32_t height;
		uint8_t channels;
		uint8_t colorspace;
	};

	typedef union {
		struct {
			unsigned char r, g, b, a;
		} rgba;
		unsigned int v;
	} qoi_rgba_t;

	static const uint8_t QOI_OP_INDEX = 0x00; /* 00xxxxxx */
	static const uint8_t QOI_OP_DIFF = 0x40; /* 01xxxxxx */
	static const uint8_t QOI_OP_LUMA = 0x80; /* 10xxxxxx */
	static const uint8_t QOI_OP_RUN = 0xc0; /* 11xxxxxx */
	static const uint8_t QOI_OP_RGB = 0xfe; /* 11111110 */
	static const uint8_t QOI_OP_RGBA = 0xff; /* 11111111 */
	static const uint8_t QOI_MASK_2 = 0xc0; /* 11000000 */

	int QOI_COLOR_HASH(qoi_rgba_t c) {
		return c.rgba.r * 3 + c.rgba.g * 5 + c.rgba.b * 7 + c.rgba.a * 11;
	}

public:
	virtual Error load_image(Ref<Image> p_image, Ref<FileAccess> f, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	ImageLoaderqoi();
};

#endif // IMAGE_LOADER_QOI_H
