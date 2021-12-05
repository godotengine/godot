/*************************************************************************/
/*  audio_frame.h                                                        */
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

#ifndef AUDIO_FRAME_H
#define AUDIO_FRAME_H

#include "core/math/vector2.h"
#include "core/typedefs.h"

static inline float undenormalise(volatile float f) {
	union {
		uint32_t i;
		float f;
	} v;

	v.f = f;

	// original: return (v.i & 0x7f800000) == 0 ? 0.0f : f;
	// version from Tim Blechmann:
	return (v.i & 0x7f800000) < 0x08000000 ? 0.0f : f;
}

static const float AUDIO_PEAK_OFFSET = 0.0000000001f;
static const float AUDIO_MIN_PEAK_DB = -200.0f; // linear2db(AUDIO_PEAK_OFFSET)

struct AudioFrame {
	//left and right samples
	float l, r;

	_ALWAYS_INLINE_ const float &operator[](int idx) const { return idx == 0 ? l : r; }
	_ALWAYS_INLINE_ float &operator[](int idx) { return idx == 0 ? l : r; }

	_ALWAYS_INLINE_ AudioFrame operator+(const AudioFrame &p_frame) const { return AudioFrame(l + p_frame.l, r + p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator-(const AudioFrame &p_frame) const { return AudioFrame(l - p_frame.l, r - p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator*(const AudioFrame &p_frame) const { return AudioFrame(l * p_frame.l, r * p_frame.r); }
	_ALWAYS_INLINE_ AudioFrame operator/(const AudioFrame &p_frame) const { return AudioFrame(l / p_frame.l, r / p_frame.r); }

	_ALWAYS_INLINE_ AudioFrame operator+(float p_sample) const { return AudioFrame(l + p_sample, r + p_sample); }
	_ALWAYS_INLINE_ AudioFrame operator-(float p_sample) const { return AudioFrame(l - p_sample, r - p_sample); }
	_ALWAYS_INLINE_ AudioFrame operator*(float p_sample) const { return AudioFrame(l * p_sample, r * p_sample); }
	_ALWAYS_INLINE_ AudioFrame operator/(float p_sample) const { return AudioFrame(l / p_sample, r / p_sample); }

	_ALWAYS_INLINE_ void operator+=(const AudioFrame &p_frame) {
		l += p_frame.l;
		r += p_frame.r;
	}
	_ALWAYS_INLINE_ void operator-=(const AudioFrame &p_frame) {
		l -= p_frame.l;
		r -= p_frame.r;
	}
	_ALWAYS_INLINE_ void operator*=(const AudioFrame &p_frame) {
		l *= p_frame.l;
		r *= p_frame.r;
	}
	_ALWAYS_INLINE_ void operator/=(const AudioFrame &p_frame) {
		l /= p_frame.l;
		r /= p_frame.r;
	}

	_ALWAYS_INLINE_ void operator+=(float p_sample) {
		l += p_sample;
		r += p_sample;
	}
	_ALWAYS_INLINE_ void operator-=(float p_sample) {
		l -= p_sample;
		r -= p_sample;
	}
	_ALWAYS_INLINE_ void operator*=(float p_sample) {
		l *= p_sample;
		r *= p_sample;
	}
	_ALWAYS_INLINE_ void operator/=(float p_sample) {
		l /= p_sample;
		r /= p_sample;
	}

	_ALWAYS_INLINE_ void undenormalise() {
		l = ::undenormalise(l);
		r = ::undenormalise(r);
	}

	_FORCE_INLINE_ AudioFrame lerp(const AudioFrame &p_b, float p_t) const {
		AudioFrame res = *this;

		res.l += (p_t * (p_b.l - l));
		res.r += (p_t * (p_b.r - r));

		return res;
	}

	_ALWAYS_INLINE_ AudioFrame(float p_l, float p_r) {
		l = p_l;
		r = p_r;
	}
	_ALWAYS_INLINE_ AudioFrame(const AudioFrame &p_frame) {
		l = p_frame.l;
		r = p_frame.r;
	}

	_ALWAYS_INLINE_ void operator=(const AudioFrame &p_frame) {
		l = p_frame.l;
		r = p_frame.r;
	}

	_ALWAYS_INLINE_ operator Vector2() const {
		return Vector2(l, r);
	}

	_ALWAYS_INLINE_ AudioFrame(const Vector2 &p_v2) {
		l = p_v2.x;
		r = p_v2.y;
	}
	_ALWAYS_INLINE_ AudioFrame() {}
};

#endif // AUDIO_FRAME_H
