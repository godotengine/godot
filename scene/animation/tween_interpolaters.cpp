/*************************************************************************/
/*  tween_interpolaters.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "tween.h"

const real_t pi = 3.1415926535898;

///////////////////////////////////////////////////////////////////////////
// linear
///////////////////////////////////////////////////////////////////////////
namespace linear {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * time / duration + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * time / duration + beginning;
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * time / duration + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * time / duration + beginning;
}
}; // namespace linear
///////////////////////////////////////////////////////////////////////////
// sine
///////////////////////////////////////////////////////////////////////////
namespace sine {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return -change * cos(time / duration * (pi / 2)) + change + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * sin(time / duration * (pi / 2)) + beginning;
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	return -change / 2 * (cos(pi * time / duration) - 1) + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace sine
///////////////////////////////////////////////////////////////////////////
// quint
///////////////////////////////////////////////////////////////////////////
namespace quint {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * pow(time / duration, 5) + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * (pow(time / duration - 1, 5) + 1) + beginning;
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	time = time / duration * 2;
	if (time < 1) return change / 2 * pow(time, 5) + beginning;
	return change / 2 * (pow(time - 2, 5) + 2) + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace quint
///////////////////////////////////////////////////////////////////////////
// quart
///////////////////////////////////////////////////////////////////////////
namespace quart {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * pow(time / duration, 4) + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	return -change * (pow(time / duration - 1, 4) - 1) + beginning;
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	time = time / duration * 2;
	if (time < 1) return change / 2 * pow(time, 4) + beginning;
	return -change / 2 * (pow(time - 2, 4) - 2) + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace quart
///////////////////////////////////////////////////////////////////////////
// quad
///////////////////////////////////////////////////////////////////////////
namespace quad {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * pow(time / duration, 2) + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	time = time / duration;
	return -change * time * (time - 2) + beginning;
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	time = time / duration * 2;
	if (time < 1) return change / 2 * pow(time, 2) + beginning;
	return -change / 2 * ((time - 1) * (time - 3) - 1) + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace quad
///////////////////////////////////////////////////////////////////////////
// expo
///////////////////////////////////////////////////////////////////////////
namespace expo {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	if (time == 0) return beginning;
	return change * pow(2, 10 * (time / duration - 1)) + beginning - change * 0.001;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	if (time == duration) return beginning + change;
	return change * 1.001 * (-pow(2, -10 * time / duration) + 1) + beginning;
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	if (time == 0) return beginning;
	if (time == duration) return beginning + change;
	time = time / duration * 2;
	if (time < 1) return change / 2 * pow(2, 10 * (time - 1)) + beginning - change * 0.0005;
	return change / 2 * 1.0005 * (-pow(2, -10 * (time - 1)) + 2) + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace expo
///////////////////////////////////////////////////////////////////////////
// elastic
///////////////////////////////////////////////////////////////////////////
namespace elastic {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	if (time == 0) return beginning;
	if ((time /= duration) == 1) return beginning + change;
	float p = duration * 0.3f;
	float a = change;
	float s = p / 4;
	float postFix = a * pow(2, 10 * (time -= 1)); // this is a fix, again, with post-increment operators
	return -(postFix * sin((time * duration - s) * (2 * pi) / p)) + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	if (time == 0) return beginning;
	if ((time /= duration) == 1) return beginning + change;
	float p = duration * 0.3f;
	float a = change;
	float s = p / 4;
	return (a * pow(2, -10 * time) * sin((time * duration - s) * (2 * pi) / p) + change + beginning);
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	if (time == 0) return beginning;
	if ((time /= duration / 2) == 2) return beginning + change;
	float p = duration * (0.3f * 1.5f);
	float a = change;
	float s = p / 4;

	if (time < 1) {
		float postFix = a * pow(2, 10 * (time -= 1)); // postIncrement is evil
		return -0.5f * (postFix * sin((time * duration - s) * (2 * pi) / p)) + beginning;
	}
	float postFix = a * pow(2, -10 * (time -= 1)); // postIncrement is evil
	return postFix * sin((time * duration - s) * (2 * pi) / p) * 0.5f + change + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace elastic
///////////////////////////////////////////////////////////////////////////
// cubic
///////////////////////////////////////////////////////////////////////////
namespace cubic {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * (time /= duration) * time * time + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	time = time / duration - 1;
	return change * (time * time * time + 1) + beginning;
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	if ((time /= duration / 2) < 1) return change / 2 * time * time * time + beginning;
	return change / 2 * ((time -= 2) * time * time + 2) + beginning;
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace cubic
///////////////////////////////////////////////////////////////////////////
// circ
///////////////////////////////////////////////////////////////////////////
namespace circ {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return -change * (sqrt(1 - (time /= duration) * time) - 1) + beginning; // TODO: ehrich: operation with time is undefined
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	return change * sqrt(1 - (time = time / duration - 1) * time) + beginning; // TODO: ehrich: operation with time is undefined
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	if ((time /= duration / 2) < 1) return -change / 2 * (sqrt(1 - time * time) - 1) + beginning;
	return change / 2 * (sqrt(1 - time * (time -= 2)) + 1) + beginning; // TODO: ehrich: operation with time is undefined
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace circ
///////////////////////////////////////////////////////////////////////////
// bounce
///////////////////////////////////////////////////////////////////////////
namespace bounce {
static real_t out(real_t time, real_t beginning, real_t change, real_t duration);

static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	return change - out(duration - time, 0, change, duration) + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	if ((time /= duration) < (1 / 2.75f)) {
		return change * (7.5625f * time * time) + beginning;
	} else if (time < (2 / 2.75f)) {
		float postFix = time -= (1.5f / 2.75f);
		return change * (7.5625f * (postFix)*time + .75f) + beginning;
	} else if (time < (2.5 / 2.75)) {
		float postFix = time -= (2.25f / 2.75f);
		return change * (7.5625f * (postFix)*time + .9375f) + beginning;
	} else {
		float postFix = time -= (2.625f / 2.75f);
		return change * (7.5625f * (postFix)*time + .984375f) + beginning;
	}
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? in(time * 2, beginning, change / 2, duration) : out((time * 2) - duration, beginning + change / 2, change / 2, duration);
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace bounce
///////////////////////////////////////////////////////////////////////////
// back
///////////////////////////////////////////////////////////////////////////
namespace back {
static real_t in(real_t time, real_t beginning, real_t change, real_t duration) {
	float s = 1.70158f;
	float postFix = time /= duration;
	return change * (postFix)*time * ((s + 1) * time - s) + beginning;
}

static real_t out(real_t time, real_t beginning, real_t change, real_t duration) {
	float s = 1.70158f;
	return change * ((time = time / duration - 1) * time * ((s + 1) * time + s) + 1) + beginning; // TODO: ehrich: operation with time is undefined
}

static real_t in_out(real_t time, real_t beginning, real_t change, real_t duration) {
	float s = 1.70158f;
	if ((time /= duration / 2) < 1) return change / 2 * (time * time * (((s *= (1.525f)) + 1) * time - s)) + beginning; // TODO: ehrich: operation with s is undefined
	float postFix = time -= 2;
	return change / 2 * ((postFix)*time * (((s *= (1.525f)) + 1) * time + s) + 2) + beginning; // TODO: ehrich: operation with s is undefined
}

static real_t out_in(real_t time, real_t beginning, real_t change, real_t duration) {
	return (time < duration / 2) ? out(time * 2, beginning, change / 2, duration) : in((time * 2) - duration, beginning + change / 2, change / 2, duration);
}
}; // namespace back

Tween::interpolater Tween::interpolaters[Tween::TRANS_COUNT][Tween::EASE_COUNT] = {
	{ &linear::in, &linear::out, &linear::in_out, &linear::out_in },
	{ &sine::in, &sine::out, &sine::in_out, &sine::out_in },
	{ &quint::in, &quint::out, &quint::in_out, &quint::out_in },
	{ &quart::in, &quart::out, &quart::in_out, &quart::out_in },
	{ &quad::in, &quad::out, &quad::in_out, &quad::out_in },
	{ &expo::in, &expo::out, &expo::in_out, &expo::out_in },
	{ &elastic::in, &elastic::out, &elastic::in_out, &elastic::out_in },
	{ &cubic::in, &cubic::out, &cubic::in_out, &cubic::out_in },
	{ &circ::in, &circ::out, &circ::in_out, &circ::out_in },
	{ &bounce::in, &bounce::out, &bounce::in_out, &bounce::out_in },
	{ &back::in, &back::out, &back::in_out, &back::out_in },
};

real_t Tween::_run_equation(TransitionType p_trans_type, EaseType p_ease_type, real_t time, real_t beginning, real_t change, real_t duration) {
	interpolater cb = interpolaters[p_trans_type][p_ease_type];
	ERR_FAIL_COND_V(cb == NULL, beginning);
	return cb(time, beginning, change, duration);
}
