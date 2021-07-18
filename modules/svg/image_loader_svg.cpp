/*************************************************************************/
/*  image_loader_svg.cpp                                                 */
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

#include "image_loader_svg.h"
#include "core/error/error_macros.h"
#include "core/math/vector2.h"
#include "core/templates/local_vector.h"
#include <stdint.h>
#include <thorvg.h>
#include <memory>

void ImageLoaderSVG::set_convert_colors(Dictionary *p_replace_color) {
	if (p_replace_color) {
		Dictionary replace_color = *p_replace_color;
		for (int i = 0; i < replace_color.keys().size(); i++) {
			Variant o_c = replace_color.keys()[i];
			Variant n_c = replace_color[replace_color.keys()[i]];
			if (o_c.get_type() == Variant::COLOR && n_c.get_type() == Variant::COLOR) {
				Color old_color = o_c;
				Color new_color = n_c;
				replace_colors.old_colors.push_back(old_color.to_abgr32());
				replace_colors.new_colors.push_back(new_color.to_abgr32());
			}
		}
	} else {
		replace_colors.old_colors.clear();
		replace_colors.new_colors.clear();
	}
	// Restore dark and light color replacement
}

void ImageLoaderSVG::create_image_from_string(Ref<Image> p_image, String p_string, float p_scale, bool upsample, bool p_convert_color) {
	Vector<uint8_t> data = p_string.to_utf8_buffer();

	uint32_t bgColor = 0xffffffff;

	std::unique_ptr<tvg::Picture> picture = tvg::Picture::gen();

	float fw, fh;
	if (picture->load((const char *)data.ptr(), data.size()) != tvg::Result::Success) {
		return;
	}
	picture->viewbox(nullptr, nullptr, &fw, &fh);
	ERR_FAIL_COND(Math::is_zero_approx(p_scale));
	uint32_t width = MIN(fw * p_scale, 16 * 1024);
	uint32_t height = MIN(fh * p_scale, 16 * 1024);
	picture->size(width, height);
	std::unique_ptr<tvg::SwCanvas> swCanvas = tvg::SwCanvas::gen();
	uint32_t *buffer = (uint32_t *)malloc(sizeof(uint32_t) * width * height);
	tvg::Result res = swCanvas->target(buffer, width, width, height, tvg::SwCanvas::ARGB8888);
	if (res != tvg::Result::Success) {
		return;
	}

	if (bgColor != 0xffffffff) {
		uint8_t bgColorR = (uint8_t)((bgColor & 0xff0000) >> 16);
		uint8_t bgColorG = (uint8_t)((bgColor & 0x00ff00) >> 8);
		uint8_t bgColorB = (uint8_t)((bgColor & 0x0000ff));

		std::unique_ptr<tvg::Shape> shape = tvg::Shape::gen();
		shape->appendRect(0, 0, width, height, 0, 0); //x, y, w, h, rx, ry
		shape->fill(bgColorR, bgColorG, bgColorB, 255); //r, g, b, a

		if (swCanvas->push(move(shape)) != tvg::Result::Success) {
			return;
		}
	}

	swCanvas->push(move(picture));

	if (swCanvas->draw() == tvg::Result::Success) {
		swCanvas->sync();
	} else {
		return;
	}

	LocalVector<uint8_t> image;
	image.resize(width * height * 4);
	for (unsigned y = 0; y < height; y++) {
		for (unsigned x = 0; x < width; x++) {
			uint32_t n = buffer[y * width + x];
			image[4 * width * y + 4 * x + 0] = (n >> 16) & 0xff;
			image[4 * width * y + 4 * x + 1] = (n >> 8) & 0xff;
			image[4 * width * y + 4 * x + 2] = n & 0xff;
			image[4 * width * y + 4 * x + 3] = (n >> 24) & 0xff;
		}
	}

	free(buffer);
	p_image->create(width, height, false, Image::FORMAT_RGBA8, image);
}

void ImageLoaderSVG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("svg");
}

Error ImageLoaderSVG::load_image(Ref<Image> p_image, FileAccess *p_fileaccess,
		bool p_force_linear, float p_scale) {
	String svg = p_fileaccess->get_as_utf8_string();
	create_image_from_string(p_image, svg, p_scale, false, false);
	ERR_FAIL_COND_V(p_image->is_empty(), FAILED);
	if (p_force_linear) {
		p_image->srgb_to_linear();
	}
	return OK;
}

ImageLoaderSVG::ReplaceColors ImageLoaderSVG::replace_colors;
