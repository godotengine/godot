/*************************************************************************/
/*  transform.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "matrix3.h"
#include "plane.h"
#include "rect3.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
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

	bool operator==(const Transform &p_transform) const;
	bool operator!=(const Transform &p_transform) const;

	_FORCE_INLINE_ Vector3 xform(const Vector3 &p_vector) const;
	_FORCE_INLINE_ Vector3 xform_inv(const Vector3 &p_vector) const;

	_FORCE_INLINE_ Plane xform(const Plane &p_plane) const;
	_FORCE_INLINE_ Plane xform_inv(const Plane &p_plane) const;

	_FORCE_INLINE_ Rect3 xform(const Rect3 &p_aabb) const;
	_FORCE_INLINE_ Rect3 xform_inv(const Rect3 &p_aabb) const;

	void operator*=(const Transform &p_transform);
	Transform operator*(const Transform &p_transform) const;

	Transform interpolate_with(const Transform &p_transform, real_t p_c) const;

	_FORCE_INLINE_ Transform inverse_xform(const Transform &t) const {

		Vector3 v = t.origin - origin;
		return Transform(basis.transpose_xform(t.basis),
				basis.xform(v));
	}

	void set(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz, real_t tx, real_t ty, real_t tz) {

		basis.elements[0][0] = xx;
		basis.elements[0][1] = xy;
		basis.elements[0][2] = xz;
		basis.elements[1][0] = yx;
		basis.elements[1][1] = yy;
		basis.elements[1][2] = yz;
		basis.elements[2][0] = zx;
		basis.elements[2][1] = zy;
		basis.elements[2][2] = zz;
		origin.x = tx;
		origin.y = ty;
		origin.z = tz;
	}

	operator String() const;

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

	Vector3 v = p_vector - origin;

	return Vector3(
			(basis.elements[0][0] * v.x) + (basis.elements[1][0] * v.y) + (basis.elements[2][0] * v.z),
			(basis.elements[0][1] * v.x) + (basis.elements[1][1] * v.y) + (basis.elements[2][1] * v.z),
			(basis.elements[0][2] * v.x) + (basis.elements[1][2] * v.y) + (basis.elements[2][2] * v.z));
}

_FORCE_INLINE_ Plane Transform::xform(const Plane &p_plane) const {

	Vector3 point = p_plane.normal * p_plane.d;
	Vector3 point_dir = point + p_plane.normal;
	point = xform(point);
	point_dir = xform(point_dir);

	Vector3 normal = point_dir - point;
	normal.normalize();
	real_t d = normal.dot(point);

	return Plane(normal, d);
}
_FORCE_INLINE_ Plane Transform::xform_inv(const Plane &p_plane) const {

	Vector3 point = p_plane.normal * p_plane.d;
	Vector3 point_dir = point + p_plane.normal;
	xform_inv(point);
	xform_inv(point_dir);

	Vector3 normal = point_dir - point;
	normal.normalize();
	real_t d = normal.dot(point);

	return Plane(normal, d);
}

_FORCE_INLINE_ Rect3 Transform::xform(const Rect3 &p_aabb) const {
/* define vertices */
#if 1
	Vector3 x = basis.get_axis(0) * p_aabb.size.x;
	Vector3 y = basis.get_axis(1) * p_aabb.size.y;
	Vector3 z = basis.get_axis(2) * p_aabb.size.z;
	Vector3 pos = xform(p_aabb.pos);
	//could be even further optimized
	Rect3 new_aabb;
	new_aabb.pos = pos;
	new_aabb.expand_to(pos + x);
	new_aabb.expand_to(pos + y);
	new_aabb.expand_to(pos + z);
	new_aabb.expand_to(pos + x + y);
	new_aabb.expand_to(pos + x + z);
	new_aabb.expand_to(pos + y + z);
	new_aabb.expand_to(pos + x + y + z);
	return new_aabb;
#else

	Vector3 vertices[8] = {
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z),
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y, p_aabb.pos.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y, p_aabb.pos.z)
	};

	AABB ret;

	ret.pos = xform(vertices[0]);

	for (int i = 1; i < 8; i++) {

		ret.expand_to(xform(vertices[i]));
	}

	return ret;
#endif
}
_FORCE_INLINE_ Rect3 Transform::xform_inv(const Rect3 &p_aabb) const {

	/* define vertices */
	Vector3 vertices[8] = {
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z),
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x + p_aabb.size.x, p_aabb.pos.y, p_aabb.pos.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y + p_aabb.size.y, p_aabb.pos.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y, p_aabb.pos.z + p_aabb.size.z),
		Vector3(p_aabb.pos.x, p_aabb.pos.y, p_aabb.pos.z)
	};

	Rect3 ret;

	ret.pos = xform_inv(vertices[0]);

	for (int i = 1; i < 8; i++) {

		ret.expand_to(xform_inv(vertices[i]));
	}

	return ret;
}

#endif
