/**************************************************************************/
/*  easing_equations.h                                                    */
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

#include "core/math/math_funcs.h"

/*
 * Derived from Robert Penner's easing equations: http://robertpenner.com/easing/
 *
 * Copyright (c) 2001 Robert Penner
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

namespace Linear {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	return c * t / d + b;
}
}; // namespace Linear

namespace Sine {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	return -c * std::cos(t / d * (Math::PI / 2)) + c + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	return c * std::sin(t / d * (Math::PI / 2)) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	return -c / 2 * (std::cos(Math::PI * t / d) - 1) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Sine

namespace Quint {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	return c * std::pow(t / d, 5) + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	return c * (std::pow(t / d - 1, 5) + 1) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	t = t / d * 2;

	if (t < 1) {
		return c / 2 * std::pow(t, 5) + b;
	}
	return c / 2 * (std::pow(t - 2, 5) + 2) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Quint

namespace Quart {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	return c * std::pow(t / d, 4) + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	return -c * (std::pow(t / d - 1, 4) - 1) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	t = t / d * 2;

	if (t < 1) {
		return c / 2 * std::pow(t, 4) + b;
	}
	return -c / 2 * (std::pow(t - 2, 4) - 2) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Quart

namespace Quad {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	return c * std::pow(t / d, 2) + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	t /= d;
	return -c * t * (t - 2) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	t = t / d * 2;

	if (t < 1) {
		return c / 2 * std::pow(t, 2) + b;
	}
	return -c / 2 * ((t - 1) * (t - 3) - 1) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Quad

namespace Expo {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	if (t == 0) {
		return b;
	}
	return c * std::pow(2, 10 * (t / d - 1)) + b - c * 0.001;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	if (t == d) {
		return b + c;
	}
	return c * 1.001 * (-std::pow(2, -10 * t / d) + 1) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	if (t == 0) {
		return b;
	}

	if (t == d) {
		return b + c;
	}

	t = t / d * 2;

	if (t < 1) {
		return c / 2 * std::pow(2, 10 * (t - 1)) + b - c * 0.0005;
	}
	return c / 2 * 1.0005 * (-std::pow(2, -10 * (t - 1)) + 2) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Expo

namespace Elastic {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	if (t == 0) {
		return b;
	}

	t /= d;
	if (t == 1) {
		return b + c;
	}

	t -= 1;
	float p = d * 0.3f;
	float a = c * std::pow(2, 10 * t);
	float s = p / 4;

	return -(a * std::sin((t * d - s) * (2 * Math::PI) / p)) + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	if (t == 0) {
		return b;
	}

	t /= d;
	if (t == 1) {
		return b + c;
	}

	float p = d * 0.3f;
	float s = p / 4;

	return (c * std::pow(2, -10 * t) * std::sin((t * d - s) * (2 * Math::PI) / p) + c + b);
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	if (t == 0) {
		return b;
	}

	if ((t /= d / 2) == 2) {
		return b + c;
	}

	float p = d * (0.3f * 1.5f);
	float a = c;
	float s = p / 4;

	if (t < 1) {
		t -= 1;
		a *= std::pow(2, 10 * t);
		return -0.5f * (a * std::sin((t * d - s) * (2 * Math::PI) / p)) + b;
	}

	t -= 1;
	a *= std::pow(2, -10 * t);
	return a * std::sin((t * d - s) * (2 * Math::PI) / p) * 0.5f + c + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Elastic

namespace Cubic {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	t /= d;
	return c * t * t * t + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	t = t / d - 1;
	return c * (t * t * t + 1) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	t /= d / 2;
	if (t < 1) {
		return c / 2 * t * t * t + b;
	}

	t -= 2;
	return c / 2 * (t * t * t + 2) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Cubic

namespace Circ {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	t /= d;
	return -c * (std::sqrt(1 - t * t) - 1) + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	t = t / d - 1;
	return c * std::sqrt(1 - t * t) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	t /= d / 2;
	if (t < 1) {
		return -c / 2 * (std::sqrt(1 - t * t) - 1) + b;
	}

	t -= 2;
	return c / 2 * (std::sqrt(1 - t * t) + 1) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Circ

namespace Bounce {
inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	t /= d;

	if (t < (1 / 2.75f)) {
		return c * (7.5625f * t * t) + b;
	}

	if (t < (2 / 2.75f)) {
		t -= 1.5f / 2.75f;
		return c * (7.5625f * t * t + 0.75f) + b;
	}

	if (t < (2.5 / 2.75)) {
		t -= 2.25f / 2.75f;
		return c * (7.5625f * t * t + 0.9375f) + b;
	}

	t -= 2.625f / 2.75f;
	return c * (7.5625f * t * t + 0.984375f) + b;
}

inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	return c - out(d - t, 0, c, d) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return in(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return out(t * 2 - d, b + h, h, d);
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Bounce

namespace Back {
inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	float s = 1.70158f;
	t /= d;

	return c * t * t * ((s + 1) * t - s) + b;
}

inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	float s = 1.70158f;
	t = t / d - 1;

	return c * (t * t * ((s + 1) * t + s) + 1) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	float s = 1.70158f * 1.525f;
	t /= d / 2;

	if (t < 1) {
		return c / 2 * (t * t * ((s + 1) * t - s)) + b;
	}

	t -= 2;
	return c / 2 * (t * t * ((s + 1) * t + s) + 2) + b;
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Back

namespace Spring {
inline real_t out(real_t t, real_t b, real_t c, real_t d) {
	t /= d;
	real_t s = 1.0 - t;
	t = (std::sin(t * Math::PI * (0.2 + 2.5 * t * t * t)) * std::pow(s, 2.2) + t) * (1.0 + (1.2 * s));
	return c * t + b;
}

inline real_t in(real_t t, real_t b, real_t c, real_t d) {
	return c - out(d - t, 0, c, d) + b;
}

inline real_t in_out(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return in(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return out(t * 2 - d, b + h, h, d);
}

inline real_t out_in(real_t t, real_t b, real_t c, real_t d) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d);
}
}; // namespace Spring
