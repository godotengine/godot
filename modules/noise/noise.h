/*************************************************************************/
/*  noise.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef NOISE_H
#define NOISE_H

#include "core/image.h"

class Noise : public Resource {
	GDCLASS(Noise, Resource);

public:
	// Virtual destructor so we can delete any Noise derived object when referenced as a Noise*.
	virtual ~Noise() {}

	virtual Ref<Image> get_image(int p_width, int p_height, bool p_invert = false) = 0;
	virtual Ref<Image> get_seamless_image(int p_width, int p_height, bool p_invert = false);

private:
	// Helper struct for get_seamless_image(). See comments in .cpp for usage.
	struct img_buff {
		uint32_t *img;
		int width; // Array dimensions & default modulo for image
		int height;
		int offset_x; // Offset index location on image (wrapped by specified modulo)
		int offset_y;
		int alt_width; // Alternate module for image
		int alt_height;

		enum ALT_MODULO {
			DEFAULT = 0,
			ALT_X,
			ALT_Y,
			ALT_XY
		};

		// Multi-dimensional array indexer (e.g. img[x][y]) that supports multiple modulos
		uint32_t &operator()(int x, int y, ALT_MODULO mode = DEFAULT) {
			switch (mode) {
				case ALT_XY:
					return img[(x + offset_x) % alt_width + ((y + offset_y) % alt_height) * width];
				case ALT_X:
					return img[(x + offset_x) % alt_width + ((y + offset_y) % height) * width];
				case ALT_Y:
					return img[(x + offset_x) % width + ((y + offset_y) % alt_height) * width];
				default:
					return img[(x + offset_x) % width + ((y + offset_y) % height) * width];
			}
		}
	};

	union l2c {
		uint32_t l;
		uint8_t c[4];
		struct {
			uint8_t r;
			uint8_t g;
			uint8_t b;
			uint8_t a;
		};
	};

	uint32_t alpha_blend(uint32_t p_bg, uint32_t p_fg, int p_alpha = -1);
};

#endif // NOISE_H
