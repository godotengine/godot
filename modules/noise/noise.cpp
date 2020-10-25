/*************************************************************************/
/*  noise.cpp                                                            */
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

#include "noise.h"

Ref<Image> Noise::get_seamless_image(int p_width, int p_height, bool p_invert) {
	/*
	To make a seamless image, we swap the quadrants so the edges are perfect matches.
	We initially get a 10% larger image so we have an overlap we can use to blend over the seams.

	Noise::img_buff::operator() acts as a multi-dimensional array indexer.
	It does the array math, translates between the flipped and non-flipped quadrants, and manages offsets and modulos.

	Here is how the larger source image and final output image map to each other:

	Output size = p_width*p_height	Source w/ extra 10% skirt `s` size = src_width*src_height
	Q1   Q2							Q4	Q3 s1
	Q3   Q4							Q2	Q1 s2
									s5	s4 s3

	All of the loops use output coordinates, so Output:Q1 == Source:Q1
	Ex: Output(half_width, half_height) [the midpoint, corner of Q1/Q4] =>
	on Source it's translated to corner of Q1/s3 unless the ALT_XY modulo moves it to Q4
	*/

	int skirt_width = p_width * .1;
	int skirt_height = p_height * .1;
	int src_width = p_width + skirt_width;
	int src_height = p_height + skirt_height;
	int half_width = p_width * .5;
	int half_height = p_height * .5;
	int skirt_edge_x = half_width + skirt_width;
	int skirt_edge_y = half_height + skirt_height;

	Ref<Image> src = get_image(src_width, src_height, p_invert);

	// `rd_src` has half_width/height offset to flip the quadrants, and an alternate shorter modulo.
	img_buff rd_src = {
		(uint32_t *)src->get_data().ptr(), // pointer to image array
		src_width, src_height, // array dimensions
		half_width, half_height, // offset
		p_width, p_height // alternate modulo
	};

	Vector<uint8_t> dest;
	dest.resize(p_width * p_height * 4);

	// `wr` is setup for straight x/y coordinate array access.
	img_buff wr = {
		(uint32_t *)dest.ptrw(),
		p_width, p_height,
		0, 0, 0, 0 // offset & alt modulo
	};
	// `rd_dest` is a readable pointer to `wr`, i.e. what has already been written to the output buffer.
	img_buff rd_dest = {
		(uint32_t *)dest.ptr(),
		p_width, p_height,
		0, 0, 0, 0 // offset & alt modulo
	};

	// Swap the quadrants to make edges seamless
	for (int y = 0; y < p_height; y++) {
		for (int x = 0; x < p_width; x++) {
			// rd_src has a half offset and the shorter modulo ignores the skirt.
			// It reads and writes in Q1-4 order (see map above), skipping the skirt
			wr(x, y) = rd_src(x, y, img_buff::ALT_XY);
		}
	}

	// Blend the vertical skirt over the middle seam
	for (int x = half_width; x < skirt_edge_x; x++) {
		int alpha = 255 * (1 - Math::smoothstep(.1f, .9f, float(x - half_width) / float(skirt_width)));
		for (int y = 0; y < p_height; y++) {
			// Skip the center square
			if (y == half_height) {
				y = skirt_edge_y - 1;
			} else {
				// Starts reading at s2, ALT_Y skips s3, and continues with s1
				wr(x, y) = alpha_blend(rd_dest(x, y), rd_src(x, y, img_buff::ALT_Y), alpha);
			}
		}
	}

	// Blend the horizontal skirt over the middle seam
	for (int y = half_height; y < skirt_edge_y; y++) {
		int alpha = 255 * (1 - Math::smoothstep(.1f, .9f, float(y - half_height) / float(skirt_height)));
		for (int x = 0; x < p_width; x++) {
			// Skip the center square
			if (x == half_width) {
				x = skirt_edge_x - 1;
			} else {
				// Starts reading at s4, skips s3, continues with s5
				wr(x, y) = alpha_blend(rd_dest(x, y), rd_src(x, y, img_buff::ALT_X), alpha);
			}
		}
	}

	// Fill in the center square. Wr starts at the top left of Q4, which is the equivalent of the top left of s3, unless a modulo is used
	for (int y = half_height; y < skirt_edge_y; y++) {
		for (int x = half_width; x < skirt_edge_x; x++) {
			int xpos = 255 * (1 - Math::smoothstep(.1f, .9f, float(x - half_width) / float(skirt_width)));
			int ypos = 255 * (1 - Math::smoothstep(.1f, .9f, float(y - half_height) / float(skirt_height)));

			// Blend s3(Q1) onto s5(Q2) for the top half
			uint32_t top_blend = alpha_blend(rd_src(x, y, img_buff::ALT_X), rd_src(x, y, img_buff::DEFAULT), xpos);
			// Blend s1(Q3) onto Q4 for the bottom half
			uint32_t bottom_blend = alpha_blend(rd_src(x, y, img_buff::ALT_XY), rd_src(x, y, img_buff::ALT_Y), xpos);
			// Blend the top half onto the bottom half
			wr(x, y) = alpha_blend(bottom_blend, top_blend, ypos);
		}
	}

	src.unref();
	Ref<Image> image = memnew(Image(p_width, p_height, false, Image::FORMAT_RGBA8, dest));
	return image;
}

uint32_t Noise::alpha_blend(uint32_t p_bg, uint32_t p_fg, int p_alpha) {
	l2c fg, bg, out;

	fg.l = p_fg;
	bg.l = p_bg;

	uint16_t alpha;
	uint16_t inv_alpha;

	// If no alpha argument specified, use the alpha channel in the color
	if (p_alpha == -1) {
		alpha = fg.c[3] + 1;
		inv_alpha = 256 - fg.c[3];
	} else {
		alpha = p_alpha + 1;
		inv_alpha = 256 - p_alpha;
	}

	out.c[0] = (uint8_t)((alpha * fg.c[0] + inv_alpha * bg.c[0]) >> 8);
	out.c[1] = (uint8_t)((alpha * fg.c[1] + inv_alpha * bg.c[1]) >> 8);
	out.c[2] = (uint8_t)((alpha * fg.c[2] + inv_alpha * bg.c[2]) >> 8);
	out.c[3] = 0xFF;

	return out.l;
}
