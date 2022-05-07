/*************************************************************************/
/*  transform_interpolator.cpp                                           */
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

#include "transform_interpolator.h"

void TransformInterpolator::interpolate_transform(const Transform &p_prev, const Transform &p_curr, Transform &r_result, real_t p_fraction) {
	r_result.origin = p_prev.origin + ((p_curr.origin - p_prev.origin) * p_fraction);
	interpolate_basis(p_prev.basis, p_curr.basis, r_result.basis, p_fraction);
}

void TransformInterpolator::interpolate_basis(const Basis &p_prev, const Basis &p_curr, Basis &r_result, real_t p_fraction) {
	Method method = find_method(p_prev, p_curr);
	interpolate_basis_via_method(p_prev, p_curr, r_result, p_fraction, method);
}

void TransformInterpolator::interpolate_transform_via_method(const Transform &p_prev, const Transform &p_curr, Transform &r_result, real_t p_fraction, Method p_method) {
	r_result.origin = p_prev.origin + ((p_curr.origin - p_prev.origin) * p_fraction);
	interpolate_basis_via_method(p_prev.basis, p_curr.basis, r_result.basis, p_fraction, p_method);
}

void TransformInterpolator::interpolate_basis_via_method(const Basis &p_prev, const Basis &p_curr, Basis &r_result, real_t p_fraction, Method p_method) {
	switch (p_method) {
		default: {
			interpolate_basis_linear(p_prev, p_curr, r_result, p_fraction);
		} break;
		case INTERP_SLERP: {
			r_result = _basis_slerp_unchecked(p_prev, p_curr, p_fraction);
		} break;
		case INTERP_SCALED_SLERP: {
			interpolate_basis_scaled_slerp(p_prev, p_curr, r_result, p_fraction);
		} break;
	}
}

Quat TransformInterpolator::_basis_to_quat_unchecked(const Basis &p_basis) {
	Basis m = p_basis;
	real_t trace = m.elements[0][0] + m.elements[1][1] + m.elements[2][2];
	real_t temp[4];

	if (trace > 0) {
		real_t s = Math::sqrt(trace + 1.0f);
		temp[3] = (s * 0.5f);
		s = 0.5f / s;

		temp[0] = ((m.elements[2][1] - m.elements[1][2]) * s);
		temp[1] = ((m.elements[0][2] - m.elements[2][0]) * s);
		temp[2] = ((m.elements[1][0] - m.elements[0][1]) * s);
	} else {
		int i = m.elements[0][0] < m.elements[1][1]
				? (m.elements[1][1] < m.elements[2][2] ? 2 : 1)
				: (m.elements[0][0] < m.elements[2][2] ? 2 : 0);
		int j = (i + 1) % 3;
		int k = (i + 2) % 3;

		real_t s = Math::sqrt(m.elements[i][i] - m.elements[j][j] - m.elements[k][k] + 1.0f);
		temp[i] = s * 0.5f;
		s = 0.5f / s;

		temp[3] = (m.elements[k][j] - m.elements[j][k]) * s;
		temp[j] = (m.elements[j][i] + m.elements[i][j]) * s;
		temp[k] = (m.elements[k][i] + m.elements[i][k]) * s;
	}

	return Quat(temp[0], temp[1], temp[2], temp[3]);
}

Quat TransformInterpolator::_quat_slerp_unchecked(const Quat &p_from, const Quat &p_to, real_t p_fraction) {
	Quat to1;
	real_t omega, cosom, sinom, scale0, scale1;

	// calc cosine
	cosom = p_from.dot(p_to);

	// adjust signs (if necessary)
	if (cosom < 0.0f) {
		cosom = -cosom;
		to1.x = -p_to.x;
		to1.y = -p_to.y;
		to1.z = -p_to.z;
		to1.w = -p_to.w;
	} else {
		to1.x = p_to.x;
		to1.y = p_to.y;
		to1.z = p_to.z;
		to1.w = p_to.w;
	}

	// calculate coefficients

	// This check could possibly be removed as we dealt with this
	// case in the find_method() function, but is left for safety, it probably
	// isn't a bottleneck.
	if ((1.0f - cosom) > (real_t)CMP_EPSILON) {
		// standard case (slerp)
		omega = Math::acos(cosom);
		sinom = Math::sin(omega);
		scale0 = Math::sin((1.0f - p_fraction) * omega) / sinom;
		scale1 = Math::sin(p_fraction * omega) / sinom;
	} else {
		// "from" and "to" quaternions are very close
		//  ... so we can do a linear interpolation
		scale0 = 1.0f - p_fraction;
		scale1 = p_fraction;
	}
	// calculate final values
	return Quat(
			scale0 * p_from.x + scale1 * to1.x,
			scale0 * p_from.y + scale1 * to1.y,
			scale0 * p_from.z + scale1 * to1.z,
			scale0 * p_from.w + scale1 * to1.w);
}

Basis TransformInterpolator::_basis_slerp_unchecked(Basis p_from, Basis p_to, real_t p_fraction) {
	Quat from = _basis_to_quat_unchecked(p_from);
	Quat to = _basis_to_quat_unchecked(p_to);

	Basis b(_quat_slerp_unchecked(from, to, p_fraction));
	return b;
}

void TransformInterpolator::interpolate_basis_scaled_slerp(Basis p_prev, Basis p_curr, Basis &r_result, real_t p_fraction) {
	// normalize both and find lengths
	Vector3 lengths_prev = _basis_orthonormalize(p_prev);
	Vector3 lengths_curr = _basis_orthonormalize(p_curr);

	r_result = _basis_slerp_unchecked(p_prev, p_curr, p_fraction);

	// now the result is unit length basis, we need to scale
	Vector3 lengths_lerped = lengths_prev + ((lengths_curr - lengths_prev) * p_fraction);

	// keep a note that the column / row order of the basis is weird,
	// so keep an eye for bugs with this.
	r_result[0] *= lengths_lerped;
	r_result[1] *= lengths_lerped;
	r_result[2] *= lengths_lerped;
}

void TransformInterpolator::interpolate_basis_linear(const Basis &p_prev, const Basis &p_curr, Basis &r_result, real_t p_fraction) {
	// interpolate basis
	r_result = p_prev.lerp(p_curr, p_fraction);

	// It turns out we need to guard against zero scale basis.
	// This is kind of silly, as we should probably fix the bugs elsewhere in Godot that can't deal with
	// zero scale, but until that time...
	for (int n = 0; n < 3; n++) {
		Vector3 &axis = r_result[n];

		// not ok, this could cause errors due to bugs elsewhere,
		// so we will bodge set this to a small value
		const real_t smallest = 0.0001f;
		const real_t smallest_squared = smallest * smallest;
		if (axis.length_squared() < smallest_squared) {
			// setting a different component to the smallest
			// helps prevent the situation where all the axes are pointing in the same direction,
			// which could be a problem for e.g. cross products..
			axis[n] = smallest;
		}
	}
}

real_t TransformInterpolator::checksum_transform(const Transform &p_transform) {
	// just a really basic checksum, this can probably be improved
	real_t sum = vec3_sum(p_transform.origin);
	sum -= vec3_sum(p_transform.basis.elements[0]);
	sum += vec3_sum(p_transform.basis.elements[1]);
	sum -= vec3_sum(p_transform.basis.elements[2]);
	return sum;
}

// return length
real_t TransformInterpolator::_vec3_normalize(Vector3 &p_vec) {
	real_t lengthsq = p_vec.length_squared();
	if (lengthsq == 0.0f) {
		p_vec.x = p_vec.y = p_vec.z = 0.0f;
		return 0.0f;
	}
	real_t length = Math::sqrt(lengthsq);
	p_vec.x /= length;
	p_vec.y /= length;
	p_vec.z /= length;
	return length;
}

// returns lengths
Vector3 TransformInterpolator::_basis_orthonormalize(Basis &r_basis) {
	// Gram-Schmidt Process

	Vector3 x = r_basis.get_axis(0);
	Vector3 y = r_basis.get_axis(1);
	Vector3 z = r_basis.get_axis(2);

	Vector3 lengths;

	lengths.x = _vec3_normalize(x);
	y = (y - x * (x.dot(y)));
	lengths.y = _vec3_normalize(y);
	z = (z - x * (x.dot(z)) - y * (y.dot(z)));
	lengths.z = _vec3_normalize(z);

	r_basis.set_axis(0, x);
	r_basis.set_axis(1, y);
	r_basis.set_axis(2, z);

	return lengths;
}

TransformInterpolator::Method TransformInterpolator::_test_basis(Basis p_basis, bool r_needed_normalize, Quat &r_quat) {
	// axis lengths
	Vector3 al = Vector3(p_basis.get_axis(0).length_squared(),
			p_basis.get_axis(1).length_squared(),
			p_basis.get_axis(2).length_squared());

	// non unit scale?
	if (r_needed_normalize || !al.is_equal_approx(Vector3(1.0, 1.0, 1.0), (real_t)0.001f)) {
		// If the basis is not normalized (at least approximately), it will fail the checks needed for slerp.
		// So we try to detect a scaled (but not sheared) basis, which we *can* slerp by normalizing first,
		// and lerping the scales separately.

		// if any of the axes are really small, it is unlikely to be a valid rotation, or is scaled too small to deal with float error
		const real_t sl_epsilon = 0.00001f;
		if ((al.x < sl_epsilon) ||
				(al.y < sl_epsilon) ||
				(al.z < sl_epsilon)) {
			return INTERP_LERP;
		}

		// normalize the basis
		Basis norm_basis = p_basis;

		al.x = Math::sqrt(al.x);
		al.y = Math::sqrt(al.y);
		al.z = Math::sqrt(al.z);

		norm_basis.set_axis(0, norm_basis.get_axis(0) / al.x);
		norm_basis.set_axis(1, norm_basis.get_axis(1) / al.y);
		norm_basis.set_axis(2, norm_basis.get_axis(2) / al.z);

		// This doesn't appear necessary, as the later checks will catch it
		// if (!_basis_is_orthogonal_any_scale(norm_basis)) {
		// return INTERP_LERP;
		// }

		p_basis = norm_basis;

		// Orthonormalize not necessary as normal normalization(!) works if the
		// axes are orthonormal.
		// p_basis.orthonormalize();

		// if we needed to normalize one of the two basis, we will need to normalize both,
		// regardless of whether the 2nd needs it, just to make sure it takes the path to return
		// INTERP_SCALED_LERP on the 2nd call of _test_basis.
		r_needed_normalize = true;
	}

	// Apply less stringent tests than the built in slerp, the standard Godot slerp
	// is too susceptible to float error to be useful
	real_t det = p_basis.determinant();
	if (!Math::is_equal_approx(det, 1, (real_t)0.01f)) {
		return INTERP_LERP;
	}

	if (!_basis_is_orthogonal(p_basis)) {
		return INTERP_LERP;
	}

	// This could possibly be less stringent too, check this.
	r_quat = _basis_to_quat_unchecked(p_basis);
	if (!r_quat.is_normalized()) {
		return INTERP_LERP;
	}

	return r_needed_normalize ? INTERP_SCALED_SLERP : INTERP_SLERP;
}

// This check doesn't seem to be needed but is preserved in case of bugs.
bool TransformInterpolator::_basis_is_orthogonal_any_scale(const Basis &p_basis) {
	Vector3 cross = p_basis.get_axis(0).cross(p_basis.get_axis(1));
	real_t l = _vec3_normalize(cross);
	// too small numbers, revert to lerp
	if (l < 0.001f) {
		return false;
	}

	const real_t epsilon = 0.9995f;

	real_t dot = cross.dot(p_basis.get_axis(2));
	if (dot < epsilon) {
		return false;
	}

	cross = p_basis.get_axis(1).cross(p_basis.get_axis(2));
	l = _vec3_normalize(cross);
	// too small numbers, revert to lerp
	if (l < 0.001f) {
		return false;
	}

	dot = cross.dot(p_basis.get_axis(0));
	if (dot < epsilon) {
		return false;
	}

	return true;
}

bool TransformInterpolator::_basis_is_orthogonal(const Basis &p_basis, real_t p_epsilon) {
	Basis identity;
	Basis m = p_basis * p_basis.transposed();

	// Less stringent tests than the standard Godot slerp
	if (!m[0].is_equal_approx(identity[0], p_epsilon) || !m[1].is_equal_approx(identity[1], p_epsilon) || !m[2].is_equal_approx(identity[2], p_epsilon)) {
		return false;
	}
	return true;
}

TransformInterpolator::Method TransformInterpolator::find_method(const Basis &p_a, const Basis &p_b) {
	bool needed_normalize = false;

	Quat q0;
	Method method = _test_basis(p_a, needed_normalize, q0);
	if (method == INTERP_LERP) {
		return method;
	}

	Quat q1;
	method = _test_basis(p_b, needed_normalize, q1);
	if (method == INTERP_LERP) {
		return method;
	}

	// Are they close together?
	// Apply the same test that will revert to lerp as
	// is present in the slerp routine.
	// Calc cosine
	real_t cosom = Math::abs(q0.dot(q1));
	if ((1.0f - cosom) <= (real_t)CMP_EPSILON) {
		return INTERP_LERP;
	}

	return method;
}
