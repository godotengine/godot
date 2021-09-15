/*************************************************************************/
/*  interpolator.cpp                                                     */
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

#include "interpolator.h"

#include "core/math/transform.h"

void Interpolator::interpolate_transform(const Transform &p_prev, const Transform &p_curr, Transform &r_result, real_t p_fraction, Method p_method) {
	switch (p_method) {
		default: {
			interpolate_transform_linear(p_prev, p_curr, r_result, p_fraction);
		} break;
		case INTERP_SLERP: {
			r_result.origin = p_prev.origin + ((p_curr.origin - p_prev.origin) * p_fraction);
			r_result.basis = p_prev.basis.slerp(p_curr.basis, p_fraction);
		} break;
	}
}

void Interpolator::interpolate_transform_linear(const Transform &p_prev, const Transform &p_curr, Transform &r_result, real_t p_fraction) {
	// interpolate translate
	r_result.origin = p_prev.origin + ((p_curr.origin - p_prev.origin) * p_fraction);

	// interpolate basis
	for (int n = 0; n < 3; n++) {
		r_result.basis.elements[n] = p_prev.basis.elements[n].linear_interpolate(p_curr.basis.elements[n], p_fraction);
	}
}

real_t Interpolator::checksum_transform(const Transform &p_transform) {
	// just a really basic checksum, this can probably be improved
	real_t sum = vec3_sum(p_transform.origin);
	sum -= vec3_sum(p_transform.basis.elements[0]);
	sum += vec3_sum(p_transform.basis.elements[1]);
	sum -= vec3_sum(p_transform.basis.elements[2]);
	return sum;
}

bool Interpolator::should_slerp(const Basis &p_a, const Basis &p_b) {
	// the two basis should be suitable, and also if they are close enough,
	// no need for a slerp anyway.
	bool slerp = false;
	if (p_a.is_rotation()) {
		Quat from(p_a);

		if (from.is_normalized() && p_b.is_rotation()) {
			Quat to(p_b);
			if (to.is_normalized()) {
				// are they close together?
				// calc cosine
				real_t cosom = Math::abs(from.dot(to));
				if ((1.0 - cosom) > CMP_EPSILON) {
					slerp = true;
				}
			}
		}
	}

	return slerp;
}
