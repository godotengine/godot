/*************************************************************************/
/*  noise.cpp                                                            */
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

#include "noise.h"

Ref<Image> Noise::get_seamless_image(int p_width, int p_height, bool p_invert, bool p_in_3d_space, real_t p_blend_skirt) const {
	int skirt_width = p_width * p_blend_skirt;
	int skirt_height = p_height * p_blend_skirt;
	int src_width = p_width + skirt_width;
	int src_height = p_height + skirt_height;

	Ref<Image> src = get_image(src_width, src_height, p_invert, p_in_3d_space);
	bool grayscale = (src->get_format() == Image::FORMAT_L8);
	if (grayscale) {
		return _generate_seamless_image<uint8_t>(src, p_width, p_height, p_invert, p_blend_skirt);
	} else {
		return _generate_seamless_image<uint32_t>(src, p_width, p_height, p_invert, p_blend_skirt);
	}
}

// Template specialization for faster grayscale blending.
template <>
uint8_t Noise::_alpha_blend<uint8_t>(uint8_t p_bg, uint8_t p_fg, int p_alpha) const {
	uint16_t alpha = p_alpha + 1;
	uint16_t inv_alpha = 256 - p_alpha;

	return (uint8_t)((alpha * p_fg + inv_alpha * p_bg) >> 8);
}

Ref<Image> Noise::get_image(int p_width, int p_height, bool p_invert, bool p_in_3d_space) const {
	Vector<uint8_t> data;
	data.resize(p_width * p_height);

	uint8_t *wd8 = data.ptrw();

	// Get all values and identify min/max values.
	Vector<real_t> values;
	values.resize(p_width * p_height);
	real_t min_val = 1000;
	real_t max_val = -1000;

	for (int y = 0, i = 0; y < p_height; y++) {
		for (int x = 0; x < p_width; x++, i++) {
			values.set(i, p_in_3d_space ? get_noise_3d(x, y, 0.0) : get_noise_2d(x, y));
			if (values[i] > max_val) {
				max_val = values[i];
			}
			if (values[i] < min_val) {
				min_val = values[i];
			}
		}
	}

	// Normalize values and write to texture.
	uint8_t value;
	for (int i = 0, x = 0; i < p_height; i++) {
		for (int j = 0; j < p_width; j++, x++) {
			if (max_val == min_val) {
				value = 0;
			} else {
				value = uint8_t(CLAMP((values[x] - min_val) / (max_val - min_val) * 255.f, 0, 255));
			}
			if (p_invert) {
				value = 255 - value;
			}

			wd8[x] = value;
		}
	}

	return memnew(Image(p_width, p_height, false, Image::FORMAT_L8, data));
}

void Noise::_bind_methods() {
	// Noise functions.
	ClassDB::bind_method(D_METHOD("get_noise_1d", "x"), &Noise::get_noise_1d);
	ClassDB::bind_method(D_METHOD("get_noise_2d", "x", "y"), &Noise::get_noise_2d);
	ClassDB::bind_method(D_METHOD("get_noise_2dv", "v"), &Noise::get_noise_2dv);
	ClassDB::bind_method(D_METHOD("get_noise_3d", "x", "y", "z"), &Noise::get_noise_3d);
	ClassDB::bind_method(D_METHOD("get_noise_3dv", "v"), &Noise::get_noise_3dv);

	// Textures.
	ClassDB::bind_method(D_METHOD("get_image", "width", "height", "invert", "in_3d_space"), &Noise::get_image, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_seamless_image", "width", "height", "invert", "in_3d_space", "skirt"), &Noise::get_seamless_image, DEFVAL(false), DEFVAL(false), DEFVAL(0.1));
}
