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

#ifndef EASING_EQUATIONS_H
#define EASING_EQUATIONS_H

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

#define EASING_FUNC(m_type) static real_t m_type(real_t t, real_t b, real_t c, real_t d, real_t p1, real_t p2)

namespace linear {
EASING_FUNC(in) {
	return c * t / d + b;
}
}; // namespace linear

namespace sine {
EASING_FUNC(in) {
	return -c * cos(t / d * (Math_PI / 2)) + c + b;
}

EASING_FUNC(out) {
	return c * sin(t / d * (Math_PI / 2)) + b;
}

EASING_FUNC(in_out) {
	return -c / 2 * (cos(Math_PI * t / d) - 1) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace sine

namespace quint {
EASING_FUNC(in) {
	return c * pow(t / d, 5) + b;
}

EASING_FUNC(out) {
	return c * (pow(t / d - 1, 5) + 1) + b;
}

EASING_FUNC(in_out) {
	t = t / d * 2;

	if (t < 1) {
		return c / 2 * pow(t, 5) + b;
	}
	return c / 2 * (pow(t - 2, 5) + 2) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace quint

namespace quart {
EASING_FUNC(in) {
	return c * pow(t / d, 4) + b;
}

EASING_FUNC(out) {
	return -c * (pow(t / d - 1, 4) - 1) + b;
}

EASING_FUNC(in_out) {
	t = t / d * 2;

	if (t < 1) {
		return c / 2 * pow(t, 4) + b;
	}
	return -c / 2 * (pow(t - 2, 4) - 2) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace quart

namespace quad {
EASING_FUNC(in) {
	return c * pow(t / d, 2) + b;
}

EASING_FUNC(out) {
	t /= d;
	return -c * t * (t - 2) + b;
}

EASING_FUNC(in_out) {
	t = t / d * 2;

	if (t < 1) {
		return c / 2 * pow(t, 2) + b;
	}
	return -c / 2 * ((t - 1) * (t - 3) - 1) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace quad

namespace expo {
EASING_FUNC(in) {
	if (t == 0) {
		return b;
	}
	return c * pow(2, 10 * (t / d - 1)) + b - c * 0.001;
}

EASING_FUNC(out) {
	if (t == d) {
		return b + c;
	}
	return c * 1.001 * (-pow(2, -10 * t / d) + 1) + b;
}

EASING_FUNC(in_out) {
	if (t == 0) {
		return b;
	}

	if (t == d) {
		return b + c;
	}

	t = t / d * 2;

	if (t < 1) {
		return c / 2 * pow(2, 10 * (t - 1)) + b - c * 0.0005;
	}
	return c / 2 * 1.0005 * (-pow(2, -10 * (t - 1)) + 2) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace expo

namespace elastic {
EASING_FUNC(in) {
	if (t == 0) {
		return b;
	}

	t /= d;
	if (t == 1) {
		return b + c;
	}

	t -= 1;
	float p = d * p1;
	float a = c * pow(2, p2 * t);
	float s = p / 4;

	return -(a * sin((t * d - s) * (2 * Math_PI) / p)) + b;
}

EASING_FUNC(out) {
	if (t == 0) {
		return b;
	}

	t /= d;
	if (t == 1) {
		return b + c;
	}

	float p = d * p1;
	float s = p / 4;

	return (c * pow(2, -10 * t) * sin((t * d - s) * (2 * Math_PI) / p) + c + b);
}

EASING_FUNC(in_out) {
	if (t == 0) {
		return b;
	}

	if ((t /= d / 2) == 2) {
		return b + c;
	}

	float p = d * (p1 * 1.5f);
	float a = c;
	float s = p / 4;

	if (t < 1) {
		t -= 1;
		a *= pow(2, p2 * t);
		return -0.5f * (a * sin((t * d - s) * (2 * Math_PI) / p)) + b;
	}

	t -= 1;
	a *= pow(2, -p2 * t);
	return a * sin((t * d - s) * (2 * Math_PI) / p) * 0.5f + c + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace elastic

namespace cubic {
EASING_FUNC(in) {
	t /= d;
	return c * t * t * t + b;
}

EASING_FUNC(out) {
	t = t / d - 1;
	return c * (t * t * t + 1) + b;
}

EASING_FUNC(in_out) {
	t /= d / 2;
	if (t < 1) {
		return c / 2 * t * t * t + b;
	}

	t -= 2;
	return c / 2 * (t * t * t + 2) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace cubic

namespace circ {
EASING_FUNC(in) {
	t /= d;
	return -c * (sqrt(1 - t * t) - 1) + b;
}

EASING_FUNC(out) {
	t = t / d - 1;
	return c * sqrt(1 - t * t) + b;
}

EASING_FUNC(in_out) {
	t /= d / 2;
	if (t < 1) {
		return -c / 2 * (sqrt(1 - t * t) - 1) + b;
	}

	t -= 2;
	return c / 2 * (sqrt(1 - t * t) + 1) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace circ

namespace bounce {
EASING_FUNC(out) {
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

EASING_FUNC(in) {
	return c - out(d - t, 0, c, d, p1, p2) + b;
}

EASING_FUNC(in_out) {
	if (t < d / 2) {
		return in(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return out(t * 2 - d, b + h, h, d, p1, p2);
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace bounce

namespace back {
EASING_FUNC(in) {
	const real_t s = p1;
	t /= d;

	return c * t * t * ((s + 1) * t - s) + b;
}

EASING_FUNC(out) {
	const real_t s = p1;
	t = t / d - 1;

	return c * (t * t * ((s + 1) * t + s) + 1) + b;
}

EASING_FUNC(in_out) {
	const real_t s = p1 * 1.525f;
	t /= d / 2;

	if (t < 1) {
		return c / 2 * (t * t * ((s + 1) * t - s)) + b;
	}

	t -= 2;
	return c / 2 * (t * t * ((s + 1) * t + s) + 2) + b;
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace back

namespace spring {
EASING_FUNC(out) {
	t /= d;
	real_t s = 1.0 - t;
	t = (sin(t * Math_PI * (0.2 + 2.5 * t * t * t)) * pow(s, 2.2) + t) * (1.0 + (1.2 * s));
	return c * t + b;
}

EASING_FUNC(in) {
	return c - out(d - t, 0, c, d, p1, p2) + b;
}

EASING_FUNC(in_out) {
	if (t < d / 2) {
		return in(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return out(t * 2 - d, b + h, h, d, p1, p2);
}

EASING_FUNC(out_in) {
	if (t < d / 2) {
		return out(t * 2, b, c / 2, d, p1, p2);
	}
	real_t h = c / 2;
	return in(t * 2 - d, b + h, h, d, p1, p2);
}
}; // namespace spring

#undef EASING_FUNC

#endif // EASING_EQUATIONS_H
