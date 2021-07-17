/*************************************************************************/
/*  transform.h                                                          */
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

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/math/plane.h"
#include "core/pool_vector.h"

class Transform {
public:
	Basis basis;
	Vector3 origin;

	void invert();
	Transform inverse() const;

	void affine_invert();
	Transform affine_inverse() const;

	Transform rotated(const Vector3 &p_axis, real_t p_phi) const;

	void rotate(const Vector3 &p_axis, real_t p_phi);
	void rotate_basis(const Vector3 &p_axis, real_t p_phi);

	void set_look_at(const Vector3 &p_eye, const Vector3 &p_target, const Vector3 &p_up);
	Transform looking_at(const Vector3 &p_target, const Vector3 &p_up) const;

	void scale(const Vector3 &p_scale);
	Transform scaled(const Vector3 &p_scale) const;
	void scale_basis(const Vector3 &p_scale);
	void translate(real_t p_tx, real_t p_ty, real_t p_tz);
	void translate(const Vector3 &p_translation);
	Transform translated(const Vector3 &p_translation) const;

	const Basis &get_basis() const { return basis; }
	void set_basis(const Basis &p_basis) { basis = p_basis; }

	const Vector3 &get_origin() const { return origin; }
	void set_origin(const Vector3 &p_origin) { origin = p_origin; }

	void orthonormalize();
	Transform orthonormalized() const;
	bool is_equal_approx(const Transform &p_transform) const;

	bool operator==(const Transform &p_transform) const;
	bool operator!=(const Transform &p_transform) const;

	// Safe, but possibly slow versions of xforms.
	// These work with non-uniform scale.
	// There is a comment to give a rough idea of speed and when to use alternative approaches.
	// If a function is slow, and you are xforming multiple items, it is far better to use the 'fast' approach.
	// The fast approach is to precompute the affine_inverse, then xform by that, or for Planes, where there are
	// 'fast' functions defined below, use those.
	// N.B. You should use the fast function for multiple items even for Plane forward xform, because it requires
	// an affine_inverse!!
	_FORCE_INLINE_ Vector3 xform(const Vector3 &p_vector) const; // fast
	_FORCE_INLINE_ Vector3 xform_inv(const Vector3 &p_vector) const; // slow

	_FORCE_INLINE_ Plane xform(const Plane &p_plane) const; // slow
	_FORCE_INLINE_ Plane xform_inv(const Plane &p_plane) const; // slow

	_FORCE_INLINE_ AABB xform(const AABB &p_aabb) const; // fast
	_FORCE_INLINE_ AABB xform_inv(const AABB &p_aabb) const; // fast-ish

	_FORCE_INLINE_ PoolVector<Vector3> xform(const PoolVector<Vector3> &p_array) const; // fast
	_FORCE_INLINE_ PoolVector<Vector3> xform_inv(const PoolVector<Vector3> &p_array) const; // fast
	//////////////////////////////////////
	// Fast versions, these operate with precomputed helper values which greatly speeds up use
	// with multiple items.
	// Use the affine inverse for the precomputed values in most cases (this works with scaling).
	// Note there are no fast versions for non-planes. In those cases simply precompute the affine_inverse,
	// and xform by that.
	_FORCE_INLINE_ Plane xform_fast(const Plane &p_plane, const Basis &p_basis_inverse_transpose) const;
	_FORCE_INLINE_ Plane xform_inv_fast(const Plane &p_plane, const Transform &p_inverse, const Basis &p_basis_transpose) const;

	// Uniform scale versions of functions.
	// These are faster but will return incorrect results with non-uniform scales.
	// They do not use affine_inverse. In the vast majority of cases you should not use these.
	_FORCE_INLINE_ Vector3 xform_uinv(const Vector3 &p_vector) const;
	_FORCE_INLINE_ AABB xform_uinv(const AABB &p_aabb) const;
	_FORCE_INLINE_ Plane xform_u(const Plane &p_plane) const;
	_FORCE_INLINE_ Plane xform_uinv(const Plane &p_plane) const;
	//////////////////////////////////////

	void operator*=(const Transform &p_transform);
	Transform operator*(const Transform &p_transform) const;

	Transform interpolate_with(const Transform &p_transform, real_t p_c) const;

	_FORCE_INLINE_ Transform inverse_xform(const Transform &t) const {
		Vector3 v = t.origin - origin;
		return Transform(basis.transpose_xform(t.basis),
				basis.xform(v));
	}

	void set(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz, real_t tx, real_t ty, real_t tz) {
		basis.set(xx, xy, xz, yx, yy, yz, zx, zy, zz);
		origin.x = tx;
		origin.y = ty;
		origin.z = tz;
	}

	operator String() const;

	Transform(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz, real_t ox, real_t oy, real_t oz);
	Transform(const Basis &p_basis, const Vector3 &p_origin = Vector3());
	Transform() {}
};

_FORCE_INLINE_ Vector3 Transform::xform(const Vector3 &p_vector) const {
	return Vector3(
			basis[0].dot(p_vector) + origin.x,
			basis[1].dot(p_vector) + origin.y,
			basis[2].dot(p_vector) + origin.z);
}

_FORCE_INLINE_ Vector3 Transform::xform_inv(const Vector3 &p_vector) const {
	return affine_inverse().xform(p_vector);
}

_FORCE_INLINE_ Vector3 Transform::xform_uinv(const Vector3 &p_vector) const {
	Vector3 v = p_vector - origin;

	return Vector3(
			(basis.elements[0][0] * v.x) + (basis.elements[1][0] * v.y) + (basis.elements[2][0] * v.z),
			(basis.elements[0][1] * v.x) + (basis.elements[1][1] * v.y) + (basis.elements[2][1] * v.z),
			(basis.elements[0][2] * v.x) + (basis.elements[1][2] * v.y) + (basis.elements[2][2] * v.z));
}

_FORCE_INLINE_ Plane Transform::xform_fast(const Plane &p_plane, const Basis &p_basis_inverse_transpose) const {
	// transform a single point on the plane
	Vector3 point = p_plane.normal * p_plane.d;
	point = xform(point);

	// use inverse transpose for correct normals with non-uniform scaling
	Vector3 normal = p_basis_inverse_transpose.xform(p_plane.normal);
	normal.normalize();

	real_t d = normal.dot(point);
	return Plane(normal, d);
}

_FORCE_INLINE_ Plane Transform::xform_inv_fast(const Plane &p_plane, const Transform &p_inverse, const Basis &p_basis_transpose) const {
	// transform a single point on the plane
	Vector3 point = p_plane.normal * p_plane.d;
	point = p_inverse.xform(point);

	// Note that instead of precalculating the transpose, an alternative
	// would be to use the transpose for the basis transform (i.e. Basis.xform_uinv).
	// However that would be less SIMD friendly (requiring a swizzle).
	// So the cost is one extra precalced value in the calling code.
	// This is probably worth it, as this could be used in bottleneck areas. And
	// where it is not a bottleneck, the non-fast method is fine.

	// use transpose for correct normals with non-uniform scaling
	Vector3 normal = p_basis_transpose.xform(p_plane.normal);
	normal.normalize();

	real_t d = normal.dot(point);
	return Plane(normal, d);
}

// Neither the plane regular xform or xform_inv are particularly efficient,
// as they do a basis inverse. For xforming a large number
// of planes it is better to pre-calculate the inverse transpose basis once
// and reuse it for each plane, by using the 'fast' version of the functions.
_FORCE_INLINE_ Plane Transform::xform(const Plane &p_plane) const {
	Basis b = basis.inverse();
	b.transpose();
	return xform_fast(p_plane, b);
}

_FORCE_INLINE_ Plane Transform::xform_inv(const Plane &p_plane) const {
	Transform inv = affine_inverse();
	Basis basis_transpose = basis.transposed();
	return xform_inv_fast(p_plane, inv, basis_transpose);
}

_FORCE_INLINE_ Plane Transform::xform_u(const Plane &p_plane) const {
	Vector3 point = p_plane.normal * p_plane.d;
	Vector3 point_dir = point + p_plane.normal;
	point = xform(point);
	point_dir = xform(point_dir);

	Vector3 normal = point_dir - point;
	normal.normalize();
	real_t d = normal.dot(point);

	return Plane(normal, d);
}

_FORCE_INLINE_ Plane Transform::xform_uinv(const Plane &p_plane) const {
	Vector3 point = p_plane.normal * p_plane.d;
	Vector3 point_dir = point + p_plane.normal;
	point = xform_uinv(point);
	point_dir = xform_uinv(point_dir);

	Vector3 normal = point_dir - point;
	normal.normalize();
	real_t d = normal.dot(point);

	return Plane(normal, d);
}

_FORCE_INLINE_ AABB Transform::xform(const AABB &p_aabb) const {
	/* http://dev.theomader.com/transform-bounding-boxes/ */
	Vector3 min = p_aabb.position;
	Vector3 max = p_aabb.position + p_aabb.size;
	Vector3 tmin, tmax;
	for (int i = 0; i < 3; i++) {
		tmin[i] = tmax[i] = origin[i];
		for (int j = 0; j < 3; j++) {
			real_t e = basis[i][j] * min[j];
			real_t f = basis[i][j] * max[j];
			if (e < f) {
				tmin[i] += e;
				tmax[i] += f;
			} else {
				tmin[i] += f;
				tmax[i] += e;
			}
		}
	}
	AABB r_aabb;
	r_aabb.position = tmin;
	r_aabb.size = tmax - tmin;
	return r_aabb;
}

_FORCE_INLINE_ AABB Transform::xform_inv(const AABB &p_aabb) const {
	return affine_inverse().xform(p_aabb);
}

// This may only be marginally faster than the safe (regular) version,
// as in the safe version, the cost of the inverse is amortized over 8 xforms.
_FORCE_INLINE_ AABB Transform::xform_uinv(const AABB &p_aabb) const {
	/* define vertices */
	Vector3 vertices[8] = {
		Vector3(p_aabb.position.x + p_aabb.size.x, p_aabb.position.y + p_aabb.size.y, p_aabb.position.z + p_aabb.size.z),
		Vector3(p_aabb.position.x + p_aabb.size.x, p_aabb.position.y + p_aabb.size.y, p_aabb.position.z),
		Vector3(p_aabb.position.x + p_aabb.size.x, p_aabb.position.y, p_aabb.position.z + p_aabb.size.z),
		Vector3(p_aabb.position.x + p_aabb.size.x, p_aabb.position.y, p_aabb.position.z),
		Vector3(p_aabb.position.x, p_aabb.position.y + p_aabb.size.y, p_aabb.position.z + p_aabb.size.z),
		Vector3(p_aabb.position.x, p_aabb.position.y + p_aabb.size.y, p_aabb.position.z),
		Vector3(p_aabb.position.x, p_aabb.position.y, p_aabb.position.z + p_aabb.size.z),
		Vector3(p_aabb.position.x, p_aabb.position.y, p_aabb.position.z)
	};

	AABB ret;

	ret.position = xform_uinv(vertices[0]);

	for (int i = 1; i < 8; i++) {
		ret.expand_to(xform_uinv(vertices[i]));
	}

	return ret;
}

PoolVector<Vector3> Transform::xform(const PoolVector<Vector3> &p_array) const {
	PoolVector<Vector3> array;
	array.resize(p_array.size());

	PoolVector<Vector3>::Read r = p_array.read();
	PoolVector<Vector3>::Write w = array.write();

	for (int i = 0; i < p_array.size(); ++i) {
		w[i] = xform(r[i]);
	}
	return array;
}

PoolVector<Vector3> Transform::xform_inv(const PoolVector<Vector3> &p_array) const {
	// Precomputing the inverse makes this ranged function
	// super cheap (almost the same cost as xform) except when there
	// are a small number of elements
	return affine_inverse().xform(p_array);
}

#endif // TRANSFORM_H
