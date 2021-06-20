/*************************************************************************/
/*  bc7enc_rdo.h                                                         */
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

// Copyright (C) 2018-2020 Binomial LLC, All rights reserved. Apache 2.0 license - see LICENSE.

#ifndef BC7E_H
#define BC7E_H

#include "thirdparty/bc7e/bc7decomp.h"

#include "core/variant/variant.h"

struct color_quad_u8 {
	uint8_t m_c[4];

	inline color_quad_u8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
		set(r, g, b, a);
	}

	inline color_quad_u8(uint8_t y = 0, uint8_t a = 255) {
		set(y, a);
	}

	inline color_quad_u8 &set(uint8_t y, uint8_t a = 255) {
		m_c[0] = y;
		m_c[1] = y;
		m_c[2] = y;
		m_c[3] = a;
		return *this;
	}

	inline color_quad_u8 &set(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
		m_c[0] = r;
		m_c[1] = g;
		m_c[2] = b;
		m_c[3] = a;
		return *this;
	}

	inline uint8_t operator[](int32_t i) const {
		ERR_FAIL_INDEX_V(i, 4, 0);
		return m_c[i];
	}

	inline int get_luma() const { return (13938U * m_c[0] + 46869U * m_c[1] + 4729U * m_c[2] + 32768U) >> 16U; } // REC709 weightings
};

typedef Vector<color_quad_u8> color_quad_u8_vec;
class image_u8 {
public:
	image_u8() :
			m_width(0), m_height(0) {
	}

	image_u8(uint32_t width, uint32_t height) :
			m_width(width), m_height(height) {
		m_pixels.resize(width * height);
	}

	inline const color_quad_u8_vec &get_pixels() const { return m_pixels; }
	inline color_quad_u8_vec &get_pixels() { return m_pixels; }

	inline uint32_t width() const { return m_width; }
	inline uint32_t height() const { return m_height; }
	inline uint32_t total_pixels() const { return m_width * m_height; }

	inline color_quad_u8 &operator()(uint32_t x, uint32_t y) {
		return m_pixels.write[x + m_width * y];
	}
	inline const color_quad_u8 &operator()(uint32_t x, uint32_t y) const {
		return m_pixels[x + m_width * y];
	}

	image_u8 &clear() {
		m_width = m_height = 0;
		m_pixels.clear();
		return *this;
	}

	image_u8 &init(uint32_t width, uint32_t height) {
		clear();

		m_width = width;
		m_height = height;
		m_pixels.resize(width * height);
		return *this;
	}

	image_u8 &set_all(const color_quad_u8 &p) {
		for (int32_t i = 0; i < m_pixels.size(); i++) {
			m_pixels.write[i] = p;
		}
		return *this;
	}

	image_u8 &crop(uint32_t new_width, uint32_t new_height) {
		if ((m_width == new_width) && (m_height == new_height)) {
			return *this;
		}

		image_u8 new_image(new_width, new_height);

		const uint32_t w = MIN(m_width, new_width);
		const uint32_t h = MIN(m_height, new_height);

		for (uint32_t y = 0; y < h; y++) {
			for (uint32_t x = 0; x < w; x++) {
				new_image(x, y) = (*this)(x, y);
			}
		}

		return swap(new_image);
	}

	image_u8 &swap(image_u8 &other) {
		SWAP(m_width, other.m_width);
		SWAP(m_height, other.m_height);
		SWAP(m_pixels, other.m_pixels);
		return *this;
	}

	inline void get_block(uint32_t bx, uint32_t by, uint32_t width, uint32_t height, color_quad_u8 *pPixels) {
		ERR_FAIL_COND((bx * width + width) > m_width);
		ERR_FAIL_COND((by * height + height) > m_height);

		for (uint32_t y = 0; y < height; y++) {
			memcpy(pPixels + y * width, &(*this)(bx * width, by * height + y), width * sizeof(color_quad_u8));
		}
	}

	inline void set_block(uint32_t bx, uint32_t by, uint32_t width, uint32_t height, const color_quad_u8 *pPixels) {
		ERR_FAIL_COND((bx * width + width) > m_width);
		ERR_FAIL_COND((by * height + height) > m_height);

		for (uint32_t y = 0; y < height; y++) {
			memcpy(&(*this)(bx * width, by * height + y), pPixels + y * width, width * sizeof(color_quad_u8));
		}
	}

	image_u8 &swizzle(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
		for (uint32_t y = 0; y < m_height; y++) {
			for (uint32_t x = 0; x < m_width; x++) {
				color_quad_u8 tmp((*this)(x, y));
				(*this)(x, y).set(tmp[r], tmp[g], tmp[b], tmp[a]);
			}
		}

		return *this;
	}

private:
	Vector<color_quad_u8> m_pixels;
	uint32_t m_width, m_height;
};

template <typename T>
inline T clamp(T v, T l, T h) {
	if (v < l) {
		v = l;
	} else if (v > h) {
		v = h;
	}
	return v;
}
inline int iabs(int i) {
	if (i < 0) {
		i = -i;
	}
	return i;
}
#endif
