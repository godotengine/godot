#ifndef RGB2YUV_H
#define RGB2YUV_H

#include "core/typedefs.h"

static void rgb2yuv420(uint8_t *y, uint8_t *u, uint8_t *v, uint8_t *rgb, size_t width, size_t height) {
	size_t upos = 0;
	size_t vpos = 0;
	size_t i = 0;

	for (size_t line = 0; line < height; ++line) {
		if (!(line % 2)) {
			for (size_t x = 0; x < width; x += 2) {
				uint8_t r = rgb[3 * i];
				uint8_t g = rgb[3 * i + 1];
				uint8_t b = rgb[3 * i + 2];

				y[i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

				u[upos++] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
				v[vpos++] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;

				r = rgb[3 * i];
				g = rgb[3 * i + 1];
				b = rgb[3 * i + 2];

				y[i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
			}
		} else {
			for (size_t x = 0; x < width; x += 1) {
				uint8_t r = rgb[3 * i];
				uint8_t g = rgb[3 * i + 1];
				uint8_t b = rgb[3 * i + 2];

				y[i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
			}
		}
	}
}

#endif // RGB2YUV_H
