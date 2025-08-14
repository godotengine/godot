/**************************************************************************/
/*  image_loader_svg.cpp                                                  */
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

#include "image_loader_svg.h"

#include "core/os/memory.h"
#include "core/variant/variant.h"

#include <lunasvg.h>

#include <iostream>

HashMap<Color, Color> ImageLoaderSVG::forced_color_map = HashMap<Color, Color>();

void ImageLoaderSVG::set_forced_color_map(const HashMap<Color, Color> &p_color_map) {
	forced_color_map = p_color_map;
}

void ImageLoaderSVG::_replace_color_property(const HashMap<Color, Color> &p_color_map, const String &p_prefix, String &r_string) {
	// Replace colors in the SVG based on what is passed in `p_color_map`.
	// Used to change the colors of editor icons based on the used theme.
	// The strings being replaced are typically of the form:
	//   fill="#5abbef"
	// But can also be 3-letter codes, include alpha, be "none" or a named color
	// string ("blue"). So we convert to Godot Color to compare with `p_color_map`.

	const int prefix_len = p_prefix.length();
	int pos = r_string.find(p_prefix);
	while (pos != -1) {
		pos += prefix_len; // Skip prefix.
		int end_pos = r_string.find_char('"', pos);
		ERR_FAIL_COND_MSG(end_pos == -1, vformat("Malformed SVG string after property \"%s\".", p_prefix));
		const String color_code = r_string.substr(pos, end_pos - pos);
		if (color_code != "none" && !color_code.begins_with("url(")) {
			const Color color = Color(color_code); // Handles both HTML codes and named colors.
			if (p_color_map.has(color)) {
				r_string = r_string.left(pos) + "#" + p_color_map[color].to_html(false) + r_string.substr(end_pos);
			}
		}
		// Search for other occurrences.
		pos = r_string.find(p_prefix, pos);
	}
}

Ref<Image> ImageLoaderSVG::load_mem_svg(const uint8_t *p_svg, int p_size, float p_scale) {
	Ref<Image> img;
	img.instantiate();

	Error err = create_image_from_utf8_buffer(img, p_svg, p_size, p_scale, false);
	ERR_FAIL_COND_V_MSG(err != OK, Ref<Image>(), vformat("ImageLoaderSVG: Failed to create SVG from buffer, error code %d.", err));

	return img;
}

Error ImageLoaderSVG::create_image_from_utf8_buffer(Ref<Image> p_image, const uint8_t *p_buffer, int p_buffer_size, float p_scale, bool p_upsample) {
	ERR_FAIL_COND_V_MSG(Math::is_zero_approx(p_scale), ERR_INVALID_PARAMETER, "ImageLoaderSVG: Can't load SVG with a scale of 0.");

	auto document = lunasvg::Document::loadFromData((const char *)p_buffer, p_buffer_size);
	if (document == nullptr) {
		return ERR_INVALID_DATA;
	}
	uint32_t width = document->width(), height = document->height();
	// check the invalid svg file
	if(width ==0 || height ==0) {
		return ERR_INVALID_DATA;
	}
	width *= p_scale;
	height *= p_scale;

	auto bitmap = document->renderToBitmap(width, height, 0x00000000);

	Vector<uint8_t> result;
	result.resize(width * height * 4);

	uint32_t *buffer = (uint32_t *)bitmap.data();

	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			uint32_t n = buffer[y * width + x];
			const size_t offset = sizeof(uint32_t) * width * y + sizeof(uint32_t) * x;
			result.write[offset + 0] = (n >> 16) & 0xff;
			result.write[offset + 1] = (n >> 8) & 0xff;
			result.write[offset + 2] = n & 0xff;
			result.write[offset + 3] = (n >> 24) & 0xff;
		}
	}

	p_image->set_data(width, height, false, Image::FORMAT_RGBA8, result);

	return OK;
}

Error ImageLoaderSVG::create_image_from_utf8_buffer(Ref<Image> p_image, const PackedByteArray &p_buffer, float p_scale, bool p_upsample) {
	return create_image_from_utf8_buffer(p_image, p_buffer.ptr(), p_buffer.size(), p_scale, p_upsample);
}

Error ImageLoaderSVG::create_image_from_string(Ref<Image> p_image, String p_string, float p_scale, bool p_upsample, const HashMap<Color, Color> &p_color_map) {
	if (p_color_map.size()) {
		_replace_color_property(p_color_map, "stop-color=\"", p_string);
		_replace_color_property(p_color_map, "fill=\"", p_string);
		_replace_color_property(p_color_map, "stroke=\"", p_string);
	}

	PackedByteArray bytes = p_string.to_utf8_buffer();
	
	return create_image_from_utf8_buffer(p_image, bytes, p_scale, p_upsample);
}

void ImageLoaderSVG::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("svg");
}

Error ImageLoaderSVG::load_image(Ref<Image> p_image, Ref<FileAccess> p_fileaccess, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	const uint64_t len = p_fileaccess->get_length() - p_fileaccess->get_position();
	Vector<uint8_t> buffer;
	buffer.resize(len);
	p_fileaccess->get_buffer(buffer.ptrw(), buffer.size());

	String svg;
	Error err = svg.parse_utf8((const char *)buffer.ptr(), buffer.size());
	if (err != OK) {
		return err;
	}

	if (p_flags & FLAG_CONVERT_COLORS) {
		err = create_image_from_string(p_image, svg, p_scale, false, forced_color_map);
	} else {
		err = create_image_from_string(p_image, svg, p_scale, false, HashMap<Color, Color>());
	}

	if (err != OK) {
		return err;
	} else if (p_image->is_empty()) {
		return ERR_INVALID_DATA;
	}

	if (p_flags & FLAG_FORCE_LINEAR) {
		p_image->srgb_to_linear();
	}
	return OK;
}

ImageLoaderSVG::ImageLoaderSVG() {
	Image::_svg_scalable_mem_loader_func = load_mem_svg;
}
