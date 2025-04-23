/**************************************************************************/
/*  projection.h                                                          */
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

#include "core/math/vector4.h"

template <typename T>
class Vector;

struct AABB;
struct Plane;
struct Rect2;
struct Transform3D;
struct Vector2;
struct Vector3;

struct [[nodiscard]] Projection {
	enum Planes {
		PLANE_NEAR,
		PLANE_FAR,
		PLANE_LEFT,
		PLANE_TOP,
		PLANE_RIGHT,
		PLANE_BOTTOM
	};

	Vector4 columns[4] = {
		{ 1, 0, 0, 0 },
		{ 0, 1, 0, 0 },
		{ 0, 0, 1, 0 },
		{ 0, 0, 0, 1 },
	};

	_FORCE_INLINE_ constexpr const Vector4 &operator[](int p_axis) const {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return columns[p_axis];
	}

	_FORCE_INLINE_ constexpr Vector4 &operator[](int p_axis) {
		DEV_ASSERT((unsigned int)p_axis < 4);
		return columns[p_axis];
	}

	_FORCE_INLINE_ real_t determinant() const;
	_FORCE_INLINE_ void set_identity();
	_FORCE_INLINE_ void set_zero();
	_FORCE_INLINE_ void set_light_bias();
	_FORCE_INLINE_ void set_depth_correction(bool p_flip_y = true, bool p_reverse_z = true, bool p_remap_z = true);

	void set_light_atlas_rect(const Rect2 &p_rect);
	_FORCE_INLINE_ void set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov = false);
	_FORCE_INLINE_ void set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist);
	_FORCE_INLINE_ void set_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far);
	_FORCE_INLINE_ void set_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar);
	_FORCE_INLINE_ void set_orthogonal(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov = false);
	_FORCE_INLINE_ void set_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far);
	void set_frustum(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov = false);
	_FORCE_INLINE_ void adjust_perspective_znear(real_t p_new_znear);

	static _FORCE_INLINE_ Projection create_depth_correction(bool p_flip_y);
	static _FORCE_INLINE_ Projection create_light_atlas_rect(const Rect2 &p_rect);
	static _FORCE_INLINE_ Projection create_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov = false);
	static _FORCE_INLINE_ Projection create_perspective_hmd(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist);
	static _FORCE_INLINE_ Projection create_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far);
	static _FORCE_INLINE_ Projection create_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar);
	static _FORCE_INLINE_ Projection create_orthogonal_aspect(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov = false);
	static _FORCE_INLINE_ Projection create_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far);
	static Projection create_frustum_aspect(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov = false);
	static _FORCE_INLINE_ Projection create_fit_aabb(const AABB &p_aabb);
	_FORCE_INLINE_ Projection perspective_znear_adjusted(real_t p_new_znear) const;
	Plane get_projection_plane(Planes p_plane) const;
	_FORCE_INLINE_ Projection flipped_y() const;
	_FORCE_INLINE_ Projection jitter_offseted(const Vector2 &p_offset) const;

	static _FORCE_INLINE_ real_t get_fovy(real_t p_fovx, real_t p_aspect) {
		return Math::rad_to_deg(Math::atan(p_aspect * Math::tan(Math::deg_to_rad(p_fovx) * 0.5)) * 2.0);
	}

	_FORCE_INLINE_ real_t get_z_far() const;
	_FORCE_INLINE_ real_t get_z_near() const;
	_FORCE_INLINE_ real_t get_aspect() const;
	_FORCE_INLINE_ real_t get_fov() const;
	_FORCE_INLINE_ bool is_orthogonal() const;

	Vector<Plane> get_projection_planes(const Transform3D &p_transform) const;

	bool get_endpoints(const Transform3D &p_transform, Vector3 *p_8points) const;
	Vector2 get_viewport_half_extents() const;
	Vector2 get_far_plane_half_extents() const;

	_FORCE_INLINE_ void invert();
	_FORCE_INLINE_ Projection inverse() const;

	_FORCE_INLINE_ constexpr Projection operator*(const Projection &p_matrix) const;

	Plane xform4(const Plane &p_vec4) const;
	Vector3 xform(const Vector3 &p_vec3) const;

	_FORCE_INLINE_ Vector4 xform(const Vector4 &p_vec4) const;
	_FORCE_INLINE_ Vector4 xform_inv(const Vector4 &p_vec4) const;

	operator String() const;

	void scale_translate_to_fit(const AABB &p_aabb);
	void add_jitter_offset(const Vector2 &p_offset);
	void make_scale(const Vector3 &p_scale);
	_FORCE_INLINE_ int get_pixels_per_meter(int p_for_pixel_width) const;
	operator Transform3D() const;

	_FORCE_INLINE_ void flip_y();

	_FORCE_INLINE_ bool is_same(const Projection &p_cam) const;

	_FORCE_INLINE_ constexpr bool operator==(const Projection &p_cam) const {
		for (uint32_t i = 0; i < 4; i++) {
			for (uint32_t j = 0; j < 4; j++) {
				if (columns[i][j] != p_cam.columns[i][j]) {
					return false;
				}
			}
		}
		return true;
	}

	_FORCE_INLINE_ constexpr bool operator!=(const Projection &p_cam) const {
		return !(*this == p_cam);
	}

	_FORCE_INLINE_ real_t get_lod_multiplier() const;

	_FORCE_INLINE_ constexpr Projection() = default;
	_FORCE_INLINE_ constexpr Projection(const Vector4 &p_x, const Vector4 &p_y, const Vector4 &p_z, const Vector4 &p_w) :
			columns{ p_x, p_y, p_z, p_w } {}
	_FORCE_INLINE_ constexpr Projection(real_t p_xx, real_t p_xy, real_t p_xz, real_t p_xw, real_t p_yx, real_t p_yy, real_t p_yz, real_t p_yw, real_t p_zx, real_t p_zy, real_t p_zz, real_t p_zw, real_t p_wx, real_t p_wy, real_t p_wz, real_t p_ww) :
			columns{
				{ p_xx, p_xy, p_xz, p_xw },
				{ p_yx, p_yy, p_yz, p_yw },
				{ p_zx, p_zy, p_zz, p_zw },
				{ p_wx, p_wy, p_wz, p_ww },
			} {}
	Projection(const Transform3D &p_transform);
};

constexpr Projection Projection::operator*(const Projection &p_matrix) const {
	Projection new_matrix;

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			real_t ab = 0;
			for (int k = 0; k < 4; k++) {
				ab += columns[k][i] * p_matrix.columns[j][k];
			}
			new_matrix.columns[j][i] = ab;
		}
	}

	return new_matrix;
}

real_t Projection::determinant() const {
	return columns[0][3] * columns[1][2] * columns[2][1] * columns[3][0] - columns[0][2] * columns[1][3] * columns[2][1] * columns[3][0] -
			columns[0][3] * columns[1][1] * columns[2][2] * columns[3][0] + columns[0][1] * columns[1][3] * columns[2][2] * columns[3][0] +
			columns[0][2] * columns[1][1] * columns[2][3] * columns[3][0] - columns[0][1] * columns[1][2] * columns[2][3] * columns[3][0] -
			columns[0][3] * columns[1][2] * columns[2][0] * columns[3][1] + columns[0][2] * columns[1][3] * columns[2][0] * columns[3][1] +
			columns[0][3] * columns[1][0] * columns[2][2] * columns[3][1] - columns[0][0] * columns[1][3] * columns[2][2] * columns[3][1] -
			columns[0][2] * columns[1][0] * columns[2][3] * columns[3][1] + columns[0][0] * columns[1][2] * columns[2][3] * columns[3][1] +
			columns[0][3] * columns[1][1] * columns[2][0] * columns[3][2] - columns[0][1] * columns[1][3] * columns[2][0] * columns[3][2] -
			columns[0][3] * columns[1][0] * columns[2][1] * columns[3][2] + columns[0][0] * columns[1][3] * columns[2][1] * columns[3][2] +
			columns[0][1] * columns[1][0] * columns[2][3] * columns[3][2] - columns[0][0] * columns[1][1] * columns[2][3] * columns[3][2] -
			columns[0][2] * columns[1][1] * columns[2][0] * columns[3][3] + columns[0][1] * columns[1][2] * columns[2][0] * columns[3][3] +
			columns[0][2] * columns[1][0] * columns[2][1] * columns[3][3] - columns[0][0] * columns[1][2] * columns[2][1] * columns[3][3] -
			columns[0][1] * columns[1][0] * columns[2][2] * columns[3][3] + columns[0][0] * columns[1][1] * columns[2][2] * columns[3][3];
}

void Projection::set_identity() {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			columns[i][j] = (i == j) ? 1 : 0;
		}
	}
}

void Projection::set_zero() {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			columns[i][j] = 0;
		}
	}
}

Vector4 Projection::xform(const Vector4 &p_vec4) const {
	return Vector4(
			columns[0][0] * p_vec4.x + columns[1][0] * p_vec4.y + columns[2][0] * p_vec4.z + columns[3][0] * p_vec4.w,
			columns[0][1] * p_vec4.x + columns[1][1] * p_vec4.y + columns[2][1] * p_vec4.z + columns[3][1] * p_vec4.w,
			columns[0][2] * p_vec4.x + columns[1][2] * p_vec4.y + columns[2][2] * p_vec4.z + columns[3][2] * p_vec4.w,
			columns[0][3] * p_vec4.x + columns[1][3] * p_vec4.y + columns[2][3] * p_vec4.z + columns[3][3] * p_vec4.w);
}
Vector4 Projection::xform_inv(const Vector4 &p_vec4) const {
	return Vector4(
			columns[0][0] * p_vec4.x + columns[0][1] * p_vec4.y + columns[0][2] * p_vec4.z + columns[0][3] * p_vec4.w,
			columns[1][0] * p_vec4.x + columns[1][1] * p_vec4.y + columns[1][2] * p_vec4.z + columns[1][3] * p_vec4.w,
			columns[2][0] * p_vec4.x + columns[2][1] * p_vec4.y + columns[2][2] * p_vec4.z + columns[2][3] * p_vec4.w,
			columns[3][0] * p_vec4.x + columns[3][1] * p_vec4.y + columns[3][2] * p_vec4.z + columns[3][3] * p_vec4.w);
}

void Projection::adjust_perspective_znear(real_t p_new_znear) {
	real_t zfar = get_z_far();
	real_t znear = p_new_znear;

	real_t deltaZ = zfar - znear;
	columns[2][2] = -(zfar + znear) / deltaZ;
	columns[3][2] = -2 * znear * zfar / deltaZ;
}

Projection Projection::create_depth_correction(bool p_flip_y) {
	Projection proj;
	proj.set_depth_correction(p_flip_y);
	return proj;
}

Projection Projection::create_light_atlas_rect(const Rect2 &p_rect) {
	Projection proj;
	proj.set_light_atlas_rect(p_rect);
	return proj;
}

Projection Projection::create_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov) {
	Projection proj;
	proj.set_perspective(p_fovy_degrees, p_aspect, p_z_near, p_z_far, p_flip_fov);
	return proj;
}

Projection Projection::create_perspective_hmd(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist) {
	Projection proj;
	proj.set_perspective(p_fovy_degrees, p_aspect, p_z_near, p_z_far, p_flip_fov, p_eye, p_intraocular_dist, p_convergence_dist);
	return proj;
}

Projection Projection::create_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far) {
	Projection proj;
	proj.set_for_hmd(p_eye, p_aspect, p_intraocular_dist, p_display_width, p_display_to_lens, p_oversample, p_z_near, p_z_far);
	return proj;
}

Projection Projection::create_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar) {
	Projection proj;
	proj.set_orthogonal(p_left, p_right, p_bottom, p_top, p_znear, p_zfar);
	return proj;
}

Projection Projection::create_orthogonal_aspect(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov) {
	Projection proj;
	proj.set_orthogonal(p_size, p_aspect, p_znear, p_zfar, p_flip_fov);
	return proj;
}

Projection Projection::create_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far) {
	Projection proj;
	proj.set_frustum(p_left, p_right, p_bottom, p_top, p_near, p_far);
	return proj;
}

Projection Projection::create_fit_aabb(const AABB &p_aabb) {
	Projection proj;
	proj.scale_translate_to_fit(p_aabb);
	return proj;
}

Projection Projection::perspective_znear_adjusted(real_t p_new_znear) const {
	Projection proj = *this;
	proj.adjust_perspective_znear(p_new_znear);
	return proj;
}

Projection Projection::flipped_y() const {
	Projection proj = *this;
	proj.flip_y();
	return proj;
}

Projection Projection ::jitter_offseted(const Vector2 &p_offset) const {
	Projection proj = *this;
	proj.add_jitter_offset(p_offset);
	return proj;
}

void Projection::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov) {
	if (p_flip_fov) {
		p_fovy_degrees = get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	real_t sine, cotangent, deltaZ;
	real_t radians = Math::deg_to_rad(p_fovy_degrees / 2.0);

	deltaZ = p_z_far - p_z_near;
	sine = Math::sin(radians);

	if ((deltaZ == 0) || (sine == 0) || (p_aspect == 0)) {
		return;
	}
	cotangent = Math::cos(radians) / sine;

	set_identity();

	columns[0][0] = cotangent / p_aspect;
	columns[1][1] = cotangent;
	columns[2][2] = -(p_z_far + p_z_near) / deltaZ;
	columns[2][3] = -1;
	columns[3][2] = -2 * p_z_near * p_z_far / deltaZ;
	columns[3][3] = 0;
}

void Projection::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist) {
	if (p_flip_fov) {
		p_fovy_degrees = get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	real_t left, right, modeltranslation, ymax, xmax, frustumshift;

	ymax = p_z_near * tan(Math::deg_to_rad(p_fovy_degrees / 2.0));
	xmax = ymax * p_aspect;
	frustumshift = (p_intraocular_dist / 2.0) * p_z_near / p_convergence_dist;

	switch (p_eye) {
		case 1: { // left eye
			left = -xmax + frustumshift;
			right = xmax + frustumshift;
			modeltranslation = p_intraocular_dist / 2.0;
		} break;
		case 2: { // right eye
			left = -xmax - frustumshift;
			right = xmax - frustumshift;
			modeltranslation = -p_intraocular_dist / 2.0;
		} break;
		default: { // mono, should give the same result as set_perspective(p_fovy_degrees,p_aspect,p_z_near,p_z_far,p_flip_fov)
			left = -xmax;
			right = xmax;
			modeltranslation = 0.0;
		} break;
	}

	set_frustum(left, right, -ymax, ymax, p_z_near, p_z_far);

	// translate matrix by (modeltranslation, 0.0, 0.0)
	Projection cm;
	cm.set_identity();
	cm.columns[3][0] = modeltranslation;
	*this = *this * cm;
}

void Projection::set_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far) {
	// we first calculate our base frustum on our values without taking our lens magnification into account.
	real_t f1 = (p_intraocular_dist * 0.5) / p_display_to_lens;
	real_t f2 = ((p_display_width - p_intraocular_dist) * 0.5) / p_display_to_lens;
	real_t f3 = (p_display_width / 4.0) / p_display_to_lens;

	// now we apply our oversample factor to increase our FOV. how much we oversample is always a balance we strike between performance and how much
	// we're willing to sacrifice in FOV.
	real_t add = ((f1 + f2) * (p_oversample - 1.0)) / 2.0;
	f1 += add;
	f2 += add;
	f3 *= p_oversample;

	// always apply KEEP_WIDTH aspect ratio
	f3 /= p_aspect;

	switch (p_eye) {
		case 1: { // left eye
			set_frustum(-f2 * p_z_near, f1 * p_z_near, -f3 * p_z_near, f3 * p_z_near, p_z_near, p_z_far);
		} break;
		case 2: { // right eye
			set_frustum(-f1 * p_z_near, f2 * p_z_near, -f3 * p_z_near, f3 * p_z_near, p_z_near, p_z_far);
		} break;
		default: { // mono, does not apply here!
		} break;
	}
}

void Projection::set_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar) {
	set_identity();

	columns[0][0] = 2.0 / (p_right - p_left);
	columns[3][0] = -((p_right + p_left) / (p_right - p_left));
	columns[1][1] = 2.0 / (p_top - p_bottom);
	columns[3][1] = -((p_top + p_bottom) / (p_top - p_bottom));
	columns[2][2] = -2.0 / (p_zfar - p_znear);
	columns[3][2] = -((p_zfar + p_znear) / (p_zfar - p_znear));
	columns[3][3] = 1.0;
}

void Projection::set_orthogonal(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov) {
	if (!p_flip_fov) {
		p_size *= p_aspect;
	}

	set_orthogonal(-p_size / 2, +p_size / 2, -p_size / p_aspect / 2, +p_size / p_aspect / 2, p_znear, p_zfar);
}

void Projection::set_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far) {
	ERR_FAIL_COND(p_right <= p_left);
	ERR_FAIL_COND(p_top <= p_bottom);
	ERR_FAIL_COND(p_far <= p_near);

	real_t *te = &columns[0][0];
	real_t x = 2 * p_near / (p_right - p_left);
	real_t y = 2 * p_near / (p_top - p_bottom);

	real_t a = (p_right + p_left) / (p_right - p_left);
	real_t b = (p_top + p_bottom) / (p_top - p_bottom);
	real_t c = -(p_far + p_near) / (p_far - p_near);
	real_t d = -2 * p_far * p_near / (p_far - p_near);

	te[0] = x;
	te[1] = 0;
	te[2] = 0;
	te[3] = 0;
	te[4] = 0;
	te[5] = y;
	te[6] = 0;
	te[7] = 0;
	te[8] = a;
	te[9] = b;
	te[10] = c;
	te[11] = -1;
	te[12] = 0;
	te[13] = 0;
	te[14] = d;
	te[15] = 0;
}

real_t Projection::get_z_far() const {
	// NOTE: This assumes z-facing near and far planes, i.e. that :
	// - the matrix is a projection across z-axis (i.e. is invertible and columns[0][1], [0][3], [1][0] and [1][3] == 0)
	// - near and far planes are z-facing (i.e. columns[0][2] and [1][2] == 0)
	return (columns[3][3] - columns[3][2]) / (columns[2][3] - columns[2][2]);
}

real_t Projection::get_z_near() const {
	// NOTE: This assumes z-facing near and far planes, i.e. that :
	// - the matrix is a projection across z-axis (i.e. is invertible and columns[0][1], [0][3], [1][0] and [1][3] == 0)
	// - near and far planes are z-facing (i.e. columns[0][2] and [1][2] == 0)
	return (columns[3][3] + columns[3][2]) / (columns[2][3] + columns[2][2]);
}

Projection Projection::inverse() const {
	Projection cm = *this;
	cm.invert();
	return cm;
}

void Projection::invert() {
	// Adapted from Mesa's `src/util/u_math.c` `util_invert_mat4x4`.
	// MIT licensed. Copyright 2008 VMware, Inc. Authored by Jacques Leroy.
	Projection temp;
	real_t *out = (real_t *)temp.columns;
	real_t *m = (real_t *)columns;

	real_t wtmp[4][8];
	real_t m0, m1, m2, m3, s;
	real_t *r0, *r1, *r2, *r3;

#define MAT(m, r, c) (m)[(c) * 4 + (r)]

	r0 = wtmp[0];
	r1 = wtmp[1];
	r2 = wtmp[2];
	r3 = wtmp[3];

	r0[0] = MAT(m, 0, 0);
	r0[1] = MAT(m, 0, 1);
	r0[2] = MAT(m, 0, 2);
	r0[3] = MAT(m, 0, 3);
	r0[4] = 1.0;
	r0[5] = 0.0;
	r0[6] = 0.0;
	r0[7] = 0.0;

	r1[0] = MAT(m, 1, 0);
	r1[1] = MAT(m, 1, 1);
	r1[2] = MAT(m, 1, 2);
	r1[3] = MAT(m, 1, 3);
	r1[5] = 1.0;
	r1[4] = 0.0;
	r1[6] = 0.0;
	r1[7] = 0.0;

	r2[0] = MAT(m, 2, 0);
	r2[1] = MAT(m, 2, 1);
	r2[2] = MAT(m, 2, 2);
	r2[3] = MAT(m, 2, 3);
	r2[6] = 1.0;
	r2[4] = 0.0;
	r2[5] = 0.0;
	r2[7] = 0.0;

	r3[0] = MAT(m, 3, 0);
	r3[1] = MAT(m, 3, 1);
	r3[2] = MAT(m, 3, 2);
	r3[3] = MAT(m, 3, 3);

	r3[7] = 1.0;
	r3[4] = 0.0;
	r3[5] = 0.0;
	r3[6] = 0.0;

	/* choose pivot - or die */
	if (Math::abs(r3[0]) > Math::abs(r2[0])) {
		SWAP(r3, r2);
	}
	if (Math::abs(r2[0]) > Math::abs(r1[0])) {
		SWAP(r2, r1);
	}
	if (Math::abs(r1[0]) > Math::abs(r0[0])) {
		SWAP(r1, r0);
	}
	ERR_FAIL_COND(0.0 == r0[0]);

	/* eliminate first variable     */
	m1 = r1[0] / r0[0];
	m2 = r2[0] / r0[0];
	m3 = r3[0] / r0[0];
	s = r0[1];
	r1[1] -= m1 * s;
	r2[1] -= m2 * s;
	r3[1] -= m3 * s;
	s = r0[2];
	r1[2] -= m1 * s;
	r2[2] -= m2 * s;
	r3[2] -= m3 * s;
	s = r0[3];
	r1[3] -= m1 * s;
	r2[3] -= m2 * s;
	r3[3] -= m3 * s;
	s = r0[4];
	if (s != 0.0) {
		r1[4] -= m1 * s;
		r2[4] -= m2 * s;
		r3[4] -= m3 * s;
	}
	s = r0[5];
	if (s != 0.0) {
		r1[5] -= m1 * s;
		r2[5] -= m2 * s;
		r3[5] -= m3 * s;
	}
	s = r0[6];
	if (s != 0.0) {
		r1[6] -= m1 * s;
		r2[6] -= m2 * s;
		r3[6] -= m3 * s;
	}
	s = r0[7];
	if (s != 0.0) {
		r1[7] -= m1 * s;
		r2[7] -= m2 * s;
		r3[7] -= m3 * s;
	}

	/* choose pivot - or die */
	if (Math::abs(r3[1]) > Math::abs(r2[1])) {
		SWAP(r3, r2);
	}
	if (Math::abs(r2[1]) > Math::abs(r1[1])) {
		SWAP(r2, r1);
	}
	ERR_FAIL_COND(0.0 == r1[1]);

	/* eliminate second variable */
	m2 = r2[1] / r1[1];
	m3 = r3[1] / r1[1];
	r2[2] -= m2 * r1[2];
	r3[2] -= m3 * r1[2];
	r2[3] -= m2 * r1[3];
	r3[3] -= m3 * r1[3];
	s = r1[4];
	if (0.0 != s) {
		r2[4] -= m2 * s;
		r3[4] -= m3 * s;
	}
	s = r1[5];
	if (0.0 != s) {
		r2[5] -= m2 * s;
		r3[5] -= m3 * s;
	}
	s = r1[6];
	if (0.0 != s) {
		r2[6] -= m2 * s;
		r3[6] -= m3 * s;
	}
	s = r1[7];
	if (0.0 != s) {
		r2[7] -= m2 * s;
		r3[7] -= m3 * s;
	}

	/* choose pivot - or die */
	if (Math::abs(r3[2]) > Math::abs(r2[2])) {
		SWAP(r3, r2);
	}
	ERR_FAIL_COND(0.0 == r2[2]);

	/* eliminate third variable */
	m3 = r3[2] / r2[2];
	r3[3] -= m3 * r2[3];
	r3[4] -= m3 * r2[4];
	r3[5] -= m3 * r2[5];
	r3[6] -= m3 * r2[6];
	r3[7] -= m3 * r2[7];

	/* last check */
	ERR_FAIL_COND(0.0 == r3[3]);

	s = 1.0 / r3[3]; /* now back substitute row 3 */
	r3[4] *= s;
	r3[5] *= s;
	r3[6] *= s;
	r3[7] *= s;

	m2 = r2[3]; /* now back substitute row 2 */
	s = 1.0 / r2[2];
	r2[4] = s * (r2[4] - r3[4] * m2);
	r2[5] = s * (r2[5] - r3[5] * m2);
	r2[6] = s * (r2[6] - r3[6] * m2);
	r2[7] = s * (r2[7] - r3[7] * m2);
	m1 = r1[3];
	r1[4] -= r3[4] * m1;
	r1[5] -= r3[5] * m1;
	r1[6] -= r3[6] * m1;
	r1[7] -= r3[7] * m1;
	m0 = r0[3];
	r0[4] -= r3[4] * m0;
	r0[5] -= r3[5] * m0;
	r0[6] -= r3[6] * m0;
	r0[7] -= r3[7] * m0;

	m1 = r1[2]; /* now back substitute row 1 */
	s = 1.0 / r1[1];
	r1[4] = s * (r1[4] - r2[4] * m1);
	r1[5] = s * (r1[5] - r2[5] * m1),
	r1[6] = s * (r1[6] - r2[6] * m1);
	r1[7] = s * (r1[7] - r2[7] * m1);
	m0 = r0[2];
	r0[4] -= r2[4] * m0;
	r0[5] -= r2[5] * m0;
	r0[6] -= r2[6] * m0;
	r0[7] -= r2[7] * m0;

	m0 = r0[1]; /* now back substitute row 0 */
	s = 1.0 / r0[0];
	r0[4] = s * (r0[4] - r1[4] * m0);
	r0[5] = s * (r0[5] - r1[5] * m0),
	r0[6] = s * (r0[6] - r1[6] * m0);
	r0[7] = s * (r0[7] - r1[7] * m0);

	MAT(out, 0, 0) = r0[4];
	MAT(out, 0, 1) = r0[5];
	MAT(out, 0, 2) = r0[6];
	MAT(out, 0, 3) = r0[7];
	MAT(out, 1, 0) = r1[4];
	MAT(out, 1, 1) = r1[5];
	MAT(out, 1, 2) = r1[6];
	MAT(out, 1, 3) = r1[7];
	MAT(out, 2, 0) = r2[4];
	MAT(out, 2, 1) = r2[5];
	MAT(out, 2, 2) = r2[6];
	MAT(out, 2, 3) = r2[7];
	MAT(out, 3, 0) = r3[4];
	MAT(out, 3, 1) = r3[5];
	MAT(out, 3, 2) = r3[6];
	MAT(out, 3, 3) = r3[7];

#undef MAT

	*this = temp;
}

void Projection::flip_y() {
	for (int i = 0; i < 4; i++) {
		columns[1][i] = -columns[1][i];
	}
}

bool Projection::is_same(const Projection &p_cam) const {
	return columns[0].is_same(p_cam.columns[0]) && columns[1].is_same(p_cam.columns[1]) && columns[2].is_same(p_cam.columns[2]) && columns[3].is_same(p_cam.columns[3]);
}

void Projection::set_depth_correction(bool p_flip_y, bool p_reverse_z, bool p_remap_z) {
	// p_remap_z is used to convert from OpenGL-style clip space (-1 - 1) to Vulkan style (0 - 1).
	real_t *m = &columns[0][0];

	m[0] = 1;
	m[1] = 0.0;
	m[2] = 0.0;
	m[3] = 0.0;
	m[4] = 0.0;
	m[5] = p_flip_y ? -1 : 1;
	m[6] = 0.0;
	m[7] = 0.0;
	m[8] = 0.0;
	m[9] = 0.0;
	m[10] = p_remap_z ? (p_reverse_z ? -0.5 : 0.5) : (p_reverse_z ? -1.0 : 1.0);
	m[11] = 0.0;
	m[12] = 0.0;
	m[13] = 0.0;
	m[14] = p_remap_z ? 0.5 : 0.0;
	m[15] = 1.0;
}

void Projection::set_light_bias() {
	real_t *m = &columns[0][0];

	m[0] = 0.5;
	m[1] = 0.0;
	m[2] = 0.0;
	m[3] = 0.0;
	m[4] = 0.0;
	m[5] = 0.5;
	m[6] = 0.0;
	m[7] = 0.0;
	m[8] = 0.0;
	m[9] = 0.0;
	m[10] = 0.5;
	m[11] = 0.0;
	m[12] = 0.5;
	m[13] = 0.5;
	m[14] = 0.5;
	m[15] = 1.0;
}

real_t Projection::get_aspect() const {
	// NOTE: This assumes a rectangular projection plane, i.e. that :
	// - the matrix is a projection across z-axis (i.e. is invertible and columns[0][1], [0][3], [1][0] and [1][3] == 0)
	// - the projection plane is rectangular (i.e. columns[0][2] and [1][2] == 0 if columns[2][3] != 0)
	return columns[1][1] / columns[0][0];
}

int Projection::get_pixels_per_meter(int p_for_pixel_width) const {
	// NOTE: This assumes a rectangular projection plane, i.e. that :
	// - the matrix is a projection across z-axis (i.e. is invertible and columns[0][1], [0][3], [1][0] and [1][3] == 0)
	// - the projection plane is rectangular (i.e. columns[0][2] and [1][2] == 0 if columns[2][3] != 0)
	real_t width = 2 * (-get_z_near() * columns[2][3] + columns[3][3]) / columns[0][0];
	return p_for_pixel_width / width; // Note : return type should be real_t (kept as int for compatibility for now).
}

bool Projection::is_orthogonal() const {
	// NOTE: This assumes that the matrix is a projection across z-axis
	// i.e. is invertible and columns[0][1], [0][3], [1][0] and [1][3] == 0
	return columns[2][3] == 0.0;
}

real_t Projection::get_fov() const {
	// NOTE: This assumes a rectangular projection plane, i.e. that :
	// - the matrix is a projection across z-axis (i.e. is invertible and columns[0][1], [0][3], [1][0] and [1][3] == 0)
	// - the projection plane is rectangular (i.e. columns[0][2] and [1][2] == 0 if columns[2][3] != 0)
	if (columns[2][0] == 0) {
		return Math::rad_to_deg(2 * Math::atan2(1, columns[0][0]));
	} else {
		// The frustum is asymmetrical so we need to calculate the left and right angles separately.
		real_t right = Math::atan2(columns[2][0] + 1, columns[0][0]);
		real_t left = Math::atan2(columns[2][0] - 1, columns[0][0]);
		return Math::rad_to_deg(right - left);
	}
}

real_t Projection::get_lod_multiplier() const {
	// NOTE: This assumes a rectangular projection plane, i.e. that :
	// - the matrix is a projection across z-axis (i.e. is invertible and columns[0][1], [0][3], [1][0] and [1][3] == 0)
	// - the projection plane is rectangular (i.e. columns[0][2] and [1][2] == 0 if columns[2][3] != 0)
	return 2 / columns[0][0];
}
