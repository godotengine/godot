/*************************************************************************/
/*  image_loader_svg.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/os/memory.h"
#include "core/variant/variant.h"

#include <thorvg.h>

void ImageLoaderSVG::_replace_color_property(const String &p_prefix, String &r_string) {
	// Replace colors in the SVG based on what is configured in `replace_colors`.
	// Used to change the colors of editor icons based on the used theme.
	// The strings being replaced are typically of the form:
	//   fill="#5abbef"
	// But can also be 3-letter codes, include alpha, be "none" or a named color
	// string ("blue"). So we convert to Godot Color to compare with `replace_colors`.

	const int prefix_len = p_prefix.length();
	int pos = r_string.find(p_prefix);
	while (pos != -1) {
		pos += prefix_len; // Skip prefix.
		int end_pos = r_string.find("\"", pos);
		ERR_FAIL_COND_MSG(end_pos == -1, vformat("Malformed SVG string after property \"%s\".", p_prefix));
		const String color_code = r_string.substr(pos, end_pos - pos);
		if (color_code != "none" && !color_code.begins_with("url(")) {
			const Color color = Color(color_code); // Handles both HTML codes and named colors.
			if (replace_colors.has(color)) {
				r_string = r_string.left(pos) + "#" + replace_colors[color].operator Color().to_html(false) + r_string.substr(end_pos);
			}
		}
		// Search for other occurrences.
		pos = r_string.find(p_prefix, pos);
	}
}

void ImageLoaderSVG::create_image_from_string(Ref<Image> p_image, String p_string, float p_scale, bool p_upsample, bool p_convert_color) {
	ERR_FAIL_COND(Math::is_zero_approx(p_scale));

	if (p_convert_color) {
		_replace_color_property("stop-color=\"", p_string);
		_replace_color_property("fill=\"", p_string);
		_replace_color_property("stroke=\"", p_string);
	}

	std::unique_ptr<tvg::Picture> picture = tvg::Picture::gen();
	PackedByteArray bytes = p_string.to_utf8_buffer();

	tvg::Result result = picture->load((const char *)bytes.ptr(), bytes.size(), "svg", true);
	if (result != tvg::Result::Success) {
		return;
	}
	float fw, fh;
	picture->viewbox(nullptr, nullptr, &fw, &fh);

	uint32_t width = MIN(fw * p_scale, 16 * 1024);
	uint32_t height = MIN(fh * p_scale, 16 * 1024);
	picture->size(width, height);

	std::unique_ptr<tvg::SwCanvas> sw_canvas = tvg::SwCanvas::gen();
	// Note: memalloc here, be sure to memfree before any return.
	uint32_t *buffer = (uint32_t *)memalloc(sizeof(uint32_t) * width * height);

	tvg::Result res = sw_canvas->target(buffer, width, width, height, tvg::SwCanvas::ARGB8888_STRAIGHT);
	if (res != tvg::Result::Success) {
		memfree(buffer);
		ERR_FAIL_MSG("ImageLoaderSVG can't create image.");
	}

	res = sw_canvas->push(move(picture));
	if (res != tvg::Result::Success) {
		memfree(buffer);
		ERR_FAIL_MSG("ImageLoaderSVG can't create image.");
	}

	res = sw_canvas->draw();
	if (res != tvg::Result::Success) {
		memfree(buffer);
		ERR_FAIL_MSG("ImageLoaderSVG can't create image.");
	}

	res = sw_canvas->sync();
	if (res != tvg::Result::Success) {
		memfree(buffer);
		ERR_FAIL_MSG("ImageLoaderSVG can't create image.");
	}

	Vector<uint8_t> image;
	image.resize(width * height * sizeof(uint32_t));

	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			uint32_t n = buffer[y * width + x];
			const size_t offset = sizeof(uint32_t) * width * y + sizeof(uint32_t) * x;
			image.write[offset + 0] = (n >> 16) & 0xff;
			image.write[offset + 1] = (n >> 8) & 0xff;
			image.write[offset + 2] = n & 0xff;
			image.write[offset + 3] = (n >> 24) & 0xff;
		}
	}

	res = sw_canvas->clear(true);
	memfree(buffer);

	p_image->create(width, height, false, Image::FORMAT_RGBA8, image);
}

void ImageLoaderSVG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("svg");
}

Error ImageLoaderSVG::load_image(Ref<Image> p_image, FileAccess *p_fileaccess, bool p_force_linear, float p_scale) {
	String svg = p_fileaccess->get_as_utf8_string();
	create_image_from_string(p_image, svg, p_scale, false, false);
	ERR_FAIL_COND_V(p_image->is_empty(), FAILED);
	if (p_force_linear) {
		p_image->srgb_to_linear();
	}
	return OK;
}
