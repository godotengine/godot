/**************************************************************************/
/*  frustum.cpp                                                           */
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

#include "frustum.h"

#include "core/math/projection.h"
#include "core/math/rect2.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"

void Frustum::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov) {
	if (p_flip_fov) {
		p_fovy_degrees = Projection::get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	// Inspired by https://iquilezles.org/articles/frustum/
	real_t an = Math::deg_to_rad(p_fovy_degrees / 2);
	real_t si = Math::sin(an);
	real_t co = Math::cos(an);
	real_t mag = Math::sqrt(co * co + si * si * p_aspect * p_aspect);

	planes[Projection::PLANE_NEAR] = Plane(Vector3(0, 0, 1), -p_z_near);
	planes[Projection::PLANE_FAR] = Plane(Vector3(0, 0, -1), p_z_far);
	planes[Projection::PLANE_LEFT] = Plane(Vector3(-co, 0, si * p_aspect) / mag, 0);
	planes[Projection::PLANE_TOP] = Plane(Vector3(0, co, si), 0);
	planes[Projection::PLANE_RIGHT] = Plane(Vector3(co, 0, si * p_aspect) / mag, 0);
	planes[Projection::PLANE_BOTTOM] = Plane(Vector3(0, -co, si), 0);
}

void Frustum::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist) {
	if (p_flip_fov) {
		p_fovy_degrees = Projection::get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	real_t left, right, model_translation, y_max, x_max, frustum_shift;

	y_max = p_z_near * Math::tan(Math::deg_to_rad(p_fovy_degrees / 2.0));
	x_max = y_max * p_aspect;
	frustum_shift = (p_intraocular_dist / 2.0) * p_z_near / p_convergence_dist;

	switch (p_eye) {
		case 1: { // Left eye.
			left = -x_max + frustum_shift;
			right = x_max + frustum_shift;
			model_translation = p_intraocular_dist / 2.0;
		} break;
		case 2: { // Right eye.
			left = -x_max - frustum_shift;
			right = x_max - frustum_shift;
			model_translation = -p_intraocular_dist / 2.0;
		} break;
		default: { // Mono, should give the same result as set_perspective(p_fovy_degrees,p_aspect,p_z_near,p_z_far,p_flip_fov).
			left = -x_max;
			right = x_max;
			model_translation = 0.0;
		} break;
	}

	set_frustum(left, right, -y_max, y_max, p_z_near, p_z_far);

	// Translate frustum by (model_translation, 0.0, 0.0).
	planes[Projection::PLANE_LEFT].d = -model_translation * Math::cos(Math::atan2(left, p_z_near));
	planes[Projection::PLANE_RIGHT].d = model_translation * Math::cos(Math::atan2(right, p_z_near));
}

void Frustum::set_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far) {
	// We first calculate our base frustum on our values without taking our lens magnification into account.
	real_t f1 = (p_intraocular_dist * 0.5) / p_display_to_lens;
	real_t f2 = ((p_display_width - p_intraocular_dist) * 0.5) / p_display_to_lens;
	real_t f3 = (p_display_width / 4.0) / p_display_to_lens;

	// Now we apply our oversample factor to increase our FOV. how much we oversample is always a balance we strike between performance and how much
	// we're willing to sacrifice in FOV.
	real_t add = ((f1 + f2) * (p_oversample - 1.0)) / 2.0;
	f1 += add;
	f2 += add;
	f3 *= p_oversample;

	// Always apply KEEP_WIDTH aspect ratio.
	f3 /= p_aspect;

	switch (p_eye) {
		case 1: { // Left eye.
			set_frustum(-f2 * p_z_near, f1 * p_z_near, -f3 * p_z_near, f3 * p_z_near, p_z_near, p_z_far);
		} break;
		case 2: { // Right eye.
			set_frustum(-f1 * p_z_near, f2 * p_z_near, -f3 * p_z_near, f3 * p_z_near, p_z_near, p_z_far);
		} break;
		default: { // Mono, does not apply here.
		} break;
	}
}

void Frustum::set_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar) {
	planes[Projection::PLANE_NEAR] = Plane(Vector3(0, 0, 1), -p_znear);
	planes[Projection::PLANE_FAR] = Plane(Vector3(0, 0, -1), p_zfar);
	planes[Projection::PLANE_LEFT] = Plane(Vector3(-1, 0, 0), -p_left);
	planes[Projection::PLANE_TOP] = Plane(Vector3(0, 1, 0), p_top);
	planes[Projection::PLANE_RIGHT] = Plane(Vector3(1, 0, 0), p_right);
	planes[Projection::PLANE_BOTTOM] = Plane(Vector3(0, -1, 0), -p_bottom);
}

void Frustum::set_orthogonal(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov) {
	if (!p_flip_fov) {
		p_size *= p_aspect;
	}

	set_orthogonal(-p_size / 2, +p_size / 2, -p_size / p_aspect / 2, +p_size / p_aspect / 2, p_znear, p_zfar);
}

void Frustum::set_frustum(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov) {
	if (!p_flip_fov) {
		p_size *= p_aspect;
	}

	set_frustum(-p_size / 2 + p_offset.x, +p_size / 2 + p_offset.x, -p_size / p_aspect / 2 + p_offset.y, +p_size / p_aspect / 2 + p_offset.y, p_near, p_far);
}

void Frustum::set_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far) {
	ERR_FAIL_COND(p_right <= p_left);
	ERR_FAIL_COND(p_top <= p_bottom);
	ERR_FAIL_COND(p_far <= p_near);

	real_t left = Math::atan2(p_left, p_near);
	real_t top = Math::atan2(p_top, p_near);
	real_t right = Math::atan2(p_right, p_near);
	real_t bottom = Math::atan2(p_bottom, p_near);

	planes[Projection::PLANE_NEAR] = Plane(Vector3(0, 0, 1), -p_near);
	planes[Projection::PLANE_FAR] = Plane(Vector3(0, 0, -1), p_far);
	planes[Projection::PLANE_LEFT] = Plane(Vector3(-Math::cos(left), 0, -Math::sin(left)), 0);
	planes[Projection::PLANE_TOP] = Plane(Vector3(0, Math::cos(top), Math::sin(top)), 0);
	planes[Projection::PLANE_RIGHT] = Plane(Vector3(Math::cos(right), 0, Math::sin(right)), 0);
	planes[Projection::PLANE_BOTTOM] = Plane(Vector3(0, -Math::cos(bottom), -Math::sin(bottom)), 0);
}

Vector<Plane> Frustum::get_projection_planes(const Transform3D &p_transform) const {
	Vector<Plane> result;
	result.resize(6);
	Basis tr_inverse = p_transform.basis.inverse().transposed();
	for (int i = 0; i < 6; i++) {
		result.write[i] = p_transform.xform_fast(planes[i], tr_inverse);
	}
	return result;
}

Vector2 Frustum::get_viewport_half_extents() const {
	// NOTE: This assumes a symmetrical frustum, i.e. that :
	// - the frustum is a projection across z-axis
	// - the projection plane is rectangular
	// - there is no offset / skew
	Vector3 res;
	planes[Projection::PLANE_NEAR].normalized().intersect_3(planes[Projection::PLANE_RIGHT].normalized(), planes[Projection::PLANE_TOP].normalized(), &res);
	return Vector2(res.x, res.y);
}

Rect2 Frustum::get_viewport_rect() const {
	// NOTE: This assumes a rectangular projection plane, i.e. that :
	// - the matrix is a projection across z-axis
	// - the projection plane is rectangular
	Vector3 bottom_left;
	Vector3 top_right;
	planes[Projection::PLANE_NEAR].normalized().intersect_3(planes[Projection::PLANE_LEFT].normalized(), planes[Projection::PLANE_BOTTOM].normalized(), &bottom_left);
	planes[Projection::PLANE_NEAR].normalized().intersect_3(planes[Projection::PLANE_RIGHT].normalized(), planes[Projection::PLANE_TOP].normalized(), &top_right);
	return Rect2(Point2{ bottom_left.x, bottom_left.y }, Size2{ top_right.x - bottom_left.x, top_right.y - bottom_left.y });
}

Vector2 Frustum::get_far_plane_half_extents() const {
	// NOTE: This assumes a symmetrical frustum, i.e. that :
	// - the frustum is a projection across z-axis
	// - the projection plane is rectangular
	// - there is no offset / skew
	Vector3 res;
	planes[Projection::PLANE_FAR].normalized().intersect_3(planes[Projection::PLANE_RIGHT].normalized(), planes[Projection::PLANE_TOP].normalized(), &res);
	return Vector2(res.x, res.y);
}

Rect2 Frustum::get_far_plane_rect() const {
	// NOTE: This assumes a rectangular projection plane, i.e. that :
	// - the matrix is a projection across z-axis
	// - the projection plane is rectangular
	Vector3 bottom_left;
	Vector3 top_right;
	planes[Projection::PLANE_FAR].normalized().intersect_3(planes[Projection::PLANE_LEFT].normalized(), planes[Projection::PLANE_BOTTOM].normalized(), &bottom_left);
	planes[Projection::PLANE_FAR].normalized().intersect_3(planes[Projection::PLANE_RIGHT].normalized(), planes[Projection::PLANE_TOP].normalized(), &top_right);
	return Rect2(Point2{ bottom_left.x, bottom_left.y }, Size2{ top_right.x - bottom_left.x, top_right.y - bottom_left.y });
}

bool Frustum::get_endpoints(const Transform3D &p_transform, Vector3 *p_8points) const {
	const Projection::Planes intersections[8][3] = {
		{ Projection::PLANE_FAR, Projection::PLANE_LEFT, Projection::PLANE_TOP },
		{ Projection::PLANE_FAR, Projection::PLANE_LEFT, Projection::PLANE_BOTTOM },
		{ Projection::PLANE_FAR, Projection::PLANE_RIGHT, Projection::PLANE_TOP },
		{ Projection::PLANE_FAR, Projection::PLANE_RIGHT, Projection::PLANE_BOTTOM },
		{ Projection::PLANE_NEAR, Projection::PLANE_LEFT, Projection::PLANE_TOP },
		{ Projection::PLANE_NEAR, Projection::PLANE_LEFT, Projection::PLANE_BOTTOM },
		{ Projection::PLANE_NEAR, Projection::PLANE_RIGHT, Projection::PLANE_TOP },
		{ Projection::PLANE_NEAR, Projection::PLANE_RIGHT, Projection::PLANE_BOTTOM },
	};

	for (int i = 0; i < 8; i++) {
		Vector3 point;
		Plane a = planes[intersections[i][0]];
		Plane b = planes[intersections[i][1]];
		Plane c = planes[intersections[i][2]];
		bool res = a.intersect_3(b, c, &point);
		ERR_FAIL_COND_V(!res, false);
		p_8points[i] = p_transform.xform(point);
	}

	return true;
}

real_t Frustum::get_z_near() const {
	// NOTE: This assumes z-facing near and far planes, i.e. that :
	// - the frustum is a projection across z-axis
	// - near and far planes are z-facing
	return -planes[Projection::PLANE_NEAR].normalized().d;
}

real_t Frustum::get_z_far() const {
	// NOTE: This assumes z-facing near and far planes, i.e. that :
	// - the frustum is a projection across z-axis
	// - near and far planes are z-facing
	return planes[Projection::PLANE_FAR].normalized().d;
}

Frustum::Frustum(const Vector<Plane> &p_planes) {
	for (int i = 0; i < p_planes.size(); i++) {
		planes[i] = p_planes[i];
	}
}
