/*************************************************************************/
/*  image_loader_lunasvg.cpp                                             */
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

#include "image_loader_lunasvg.h"

void ImageLoaderLunaSVG::create_image_from_string(Ref<Image> p_image, String p_string, float p_scale, bool upsample, bool p_convert_color) {
	using namespace lunasvg;
	Vector<uint8_t> data = p_string.to_utf8_buffer();
	std::unique_ptr<Document> document =
			Document::loadFromData((const char *)data.ptr(), data.size());
	std::uint32_t width = document->width(), height = document->height();
	width *= p_scale;
	height *= p_scale;
	std::uint32_t bgColor = 0x00000000;
	Bitmap bitmap = document->renderToBitmap(width, height, bgColor);
	ERR_FAIL_COND(!bitmap.valid());
	size_t size = width * height * 4; // RGBA8
	Vector<uint8_t> result;
	result.resize(size);
	memcpy(result.ptrw(), bitmap.data(), size);
	p_image->create(width, height, false, Image::FORMAT_RGBA8, result);
}

void ImageLoaderLunaSVG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("svg");
}

Error ImageLoaderLunaSVG::load_image(Ref<Image> p_image, FileAccess *p_fileaccess,
		bool p_force_linear, float p_scale) {
	String svg = p_fileaccess->get_as_utf8_string();
	create_image_from_string(p_image, svg, p_scale, false, false);
	ERR_FAIL_COND_V(p_image->is_empty(), FAILED);
	if (p_force_linear) {
		p_image->srgb_to_linear();
	}
	return OK;
}
