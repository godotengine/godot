/**************************************************************************/
/*  rgb2yuv.h                                                             */
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

#pragma once

#include "core/typedefs.h"

// For reference, see:
// - https://stackoverflow.com/a/9467305
// - https://en.wikipedia.org/wiki/YCbCr#Approximate_8-bit_matrices_for_BT.601

#define WRITE_Y                                           \
	{                                                     \
		r = rgb[pixel_size * i];                          \
		g = rgb[pixel_size * i + 1];                      \
		b = rgb[pixel_size * i + 2];                      \
		y[i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16; \
	}

#define WRITE_YUV                                              \
	{                                                          \
		WRITE_Y                                                \
		u[uvpos] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128; \
		v[uvpos] = ((112 * r + -94 * g + -18 * b) >> 8) + 128; \
		uvpos++;                                               \
	}

static void _rgb2yuv420(uint8_t *y, uint8_t *u, uint8_t *v, uint8_t *rgb, size_t width, size_t height, size_t pixel_size) {
	size_t uvpos = 0;
	size_t i = 0;
	uint8_t r = 0;
	uint8_t g = 0;
	uint8_t b = 0;
	size_t line = 0;
	size_t x = 0;
	for (line = 0; line < height - 1; line += 2) {
		for (x = 0; x < width - 1; x += 2) {
			WRITE_YUV
			WRITE_Y
		}
		if (x == width - 1) {
			WRITE_YUV
		}
		for (x = 0; x < width; x += 1) {
			WRITE_Y
		}
	}
	if (line == height - 1) {
		for (x = 0; x < width - 1; x += 2) {
			WRITE_YUV
			WRITE_Y
		}
		if (x == width - 1) {
			WRITE_YUV
		}
	}
}

static void rgb2yuv420(uint8_t *y, uint8_t *u, uint8_t *v, uint8_t *rgb, size_t width, size_t height) {
	_rgb2yuv420(y, u, v, rgb, width, height, 3);
}

static void rgba2yuv420(uint8_t *y, uint8_t *u, uint8_t *v, uint8_t *rgba, size_t width, size_t height) {
	_rgb2yuv420(y, u, v, rgba, width, height, 4);
}
