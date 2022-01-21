/*************************************************************************/
/*  camera_matrix.cpp                                                    */
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

#include "camera_matrix.h"

#include "core/math/math_funcs.h"
#include "core/string/print_string.h"

float CameraMatrix::determinant() const {
	return matrix[0][3] * matrix[1][2] * matrix[2][1] * matrix[3][0] - matrix[0][2] * matrix[1][3] * matrix[2][1] * matrix[3][0] -
			matrix[0][3] * matrix[1][1] * matrix[2][2] * matrix[3][0] + matrix[0][1] * matrix[1][3] * matrix[2][2] * matrix[3][0] +
			matrix[0][2] * matrix[1][1] * matrix[2][3] * matrix[3][0] - matrix[0][1] * matrix[1][2] * matrix[2][3] * matrix[3][0] -
			matrix[0][3] * matrix[1][2] * matrix[2][0] * matrix[3][1] + matrix[0][2] * matrix[1][3] * matrix[2][0] * matrix[3][1] +
			matrix[0][3] * matrix[1][0] * matrix[2][2] * matrix[3][1] - matrix[0][0] * matrix[1][3] * matrix[2][2] * matrix[3][1] -
			matrix[0][2] * matrix[1][0] * matrix[2][3] * matrix[3][1] + matrix[0][0] * matrix[1][2] * matrix[2][3] * matrix[3][1] +
			matrix[0][3] * matrix[1][1] * matrix[2][0] * matrix[3][2] - matrix[0][1] * matrix[1][3] * matrix[2][0] * matrix[3][2] -
			matrix[0][3] * matrix[1][0] * matrix[2][1] * matrix[3][2] + matrix[0][0] * matrix[1][3] * matrix[2][1] * matrix[3][2] +
			matrix[0][1] * matrix[1][0] * matrix[2][3] * matrix[3][2] - matrix[0][0] * matrix[1][1] * matrix[2][3] * matrix[3][2] -
			matrix[0][2] * matrix[1][1] * matrix[2][0] * matrix[3][3] + matrix[0][1] * matrix[1][2] * matrix[2][0] * matrix[3][3] +
			matrix[0][2] * matrix[1][0] * matrix[2][1] * matrix[3][3] - matrix[0][0] * matrix[1][2] * matrix[2][1] * matrix[3][3] -
			matrix[0][1] * matrix[1][0] * matrix[2][2] * matrix[3][3] + matrix[0][0] * matrix[1][1] * matrix[2][2] * matrix[3][3];
}

void CameraMatrix::set_identity() {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			matrix[i][j] = (i == j) ? 1 : 0;
		}
	}
}

void CameraMatrix::set_zero() {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			matrix[i][j] = 0;
		}
	}
}

Plane CameraMatrix::xform4(const Plane &p_vec4) const {
	Plane ret;

	ret.normal.x = matrix[0][0] * p_vec4.normal.x + matrix[1][0] * p_vec4.normal.y + matrix[2][0] * p_vec4.normal.z + matrix[3][0] * p_vec4.d;
	ret.normal.y = matrix[0][1] * p_vec4.normal.x + matrix[1][1] * p_vec4.normal.y + matrix[2][1] * p_vec4.normal.z + matrix[3][1] * p_vec4.d;
	ret.normal.z = matrix[0][2] * p_vec4.normal.x + matrix[1][2] * p_vec4.normal.y + matrix[2][2] * p_vec4.normal.z + matrix[3][2] * p_vec4.d;
	ret.d = matrix[0][3] * p_vec4.normal.x + matrix[1][3] * p_vec4.normal.y + matrix[2][3] * p_vec4.normal.z + matrix[3][3] * p_vec4.d;
	return ret;
}

void CameraMatrix::adjust_perspective_znear(real_t p_new_znear) {
	real_t zfar = get_z_far();
	real_t znear = p_new_znear;

	real_t deltaZ = zfar - znear;
	matrix[2][2] = -(zfar + znear) / deltaZ;
	matrix[3][2] = -2 * znear * zfar / deltaZ;
}

void CameraMatrix::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov) {
	if (p_flip_fov) {
		p_fovy_degrees = get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	real_t sine, cotangent, deltaZ;
	real_t radians = Math::deg2rad(p_fovy_degrees / 2.0);

	deltaZ = p_z_far - p_z_near;
	sine = Math::sin(radians);

	if ((deltaZ == 0) || (sine == 0) || (p_aspect == 0)) {
		return;
	}
	cotangent = Math::cos(radians) / sine;

	set_identity();

	matrix[0][0] = cotangent / p_aspect;
	matrix[1][1] = cotangent;
	matrix[2][2] = -(p_z_far + p_z_near) / deltaZ;
	matrix[2][3] = -1;
	matrix[3][2] = -2 * p_z_near * p_z_far / deltaZ;
	matrix[3][3] = 0;
}

void CameraMatrix::set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist) {
	if (p_flip_fov) {
		p_fovy_degrees = get_fovy(p_fovy_degrees, 1.0 / p_aspect);
	}

	real_t left, right, modeltranslation, ymax, xmax, frustumshift;

	ymax = p_z_near * tan(Math::deg2rad(p_fovy_degrees / 2.0));
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
	CameraMatrix cm;
	cm.set_identity();
	cm.matrix[3][0] = modeltranslation;
	*this = *this * cm;
}

void CameraMatrix::set_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far) {
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

void CameraMatrix::set_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar) {
	set_identity();

	matrix[0][0] = 2.0 / (p_right - p_left);
	matrix[3][0] = -((p_right + p_left) / (p_right - p_left));
	matrix[1][1] = 2.0 / (p_top - p_bottom);
	matrix[3][1] = -((p_top + p_bottom) / (p_top - p_bottom));
	matrix[2][2] = -2.0 / (p_zfar - p_znear);
	matrix[3][2] = -((p_zfar + p_znear) / (p_zfar - p_znear));
	matrix[3][3] = 1.0;
}

void CameraMatrix::set_orthogonal(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov) {
	if (!p_flip_fov) {
		p_size *= p_aspect;
	}

	set_orthogonal(-p_size / 2, +p_size / 2, -p_size / p_aspect / 2, +p_size / p_aspect / 2, p_znear, p_zfar);
}

void CameraMatrix::set_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far) {
	ERR_FAIL_COND(p_right <= p_left);
	ERR_FAIL_COND(p_top <= p_bottom);
	ERR_FAIL_COND(p_far <= p_near);

	real_t *te = &matrix[0][0];
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

void CameraMatrix::set_frustum(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov) {
	if (!p_flip_fov) {
		p_size *= p_aspect;
	}

	set_frustum(-p_size / 2 + p_offset.x, +p_size / 2 + p_offset.x, -p_size / p_aspect / 2 + p_offset.y, +p_size / p_aspect / 2 + p_offset.y, p_near, p_far);
}

real_t CameraMatrix::get_z_far() const {
	const real_t *matrix = (const real_t *)this->matrix;
	Plane new_plane = Plane(matrix[3] - matrix[2],
			matrix[7] - matrix[6],
			matrix[11] - matrix[10],
			matrix[15] - matrix[14]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	return new_plane.d;
}

real_t CameraMatrix::get_z_near() const {
	const real_t *matrix = (const real_t *)this->matrix;
	Plane new_plane = Plane(matrix[3] + matrix[2],
			matrix[7] + matrix[6],
			matrix[11] + matrix[10],
			-matrix[15] - matrix[14]);

	new_plane.normalize();
	return new_plane.d;
}

Vector2 CameraMatrix::get_viewport_half_extents() const {
	const real_t *matrix = (const real_t *)this->matrix;
	///////--- Near Plane ---///////
	Plane near_plane = Plane(matrix[3] + matrix[2],
			matrix[7] + matrix[6],
			matrix[11] + matrix[10],
			-matrix[15] - matrix[14]);
	near_plane.normalize();

	///////--- Right Plane ---///////
	Plane right_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			-matrix[15] + matrix[12]);
	right_plane.normalize();

	Plane top_plane = Plane(matrix[3] - matrix[1],
			matrix[7] - matrix[5],
			matrix[11] - matrix[9],
			-matrix[15] + matrix[13]);
	top_plane.normalize();

	Vector3 res;
	near_plane.intersect_3(right_plane, top_plane, &res);

	return Vector2(res.x, res.y);
}

Vector2 CameraMatrix::get_far_plane_half_extents() const {
	const real_t *matrix = (const real_t *)this->matrix;
	///////--- Far Plane ---///////
	Plane far_plane = Plane(matrix[3] - matrix[2],
			matrix[7] - matrix[6],
			matrix[11] - matrix[10],
			-matrix[15] + matrix[14]);
	far_plane.normalize();

	///////--- Right Plane ---///////
	Plane right_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			-matrix[15] + matrix[12]);
	right_plane.normalize();

	Plane top_plane = Plane(matrix[3] - matrix[1],
			matrix[7] - matrix[5],
			matrix[11] - matrix[9],
			-matrix[15] + matrix[13]);
	top_plane.normalize();

	Vector3 res;
	far_plane.intersect_3(right_plane, top_plane, &res);

	return Vector2(res.x, res.y);
}

bool CameraMatrix::get_endpoints(const Transform3D &p_transform, Vector3 *p_8points) const {
	Vector<Plane> planes = get_projection_planes(Transform3D());
	const Planes intersections[8][3] = {
		{ PLANE_FAR, PLANE_LEFT, PLANE_TOP },
		{ PLANE_FAR, PLANE_LEFT, PLANE_BOTTOM },
		{ PLANE_FAR, PLANE_RIGHT, PLANE_TOP },
		{ PLANE_FAR, PLANE_RIGHT, PLANE_BOTTOM },
		{ PLANE_NEAR, PLANE_LEFT, PLANE_TOP },
		{ PLANE_NEAR, PLANE_LEFT, PLANE_BOTTOM },
		{ PLANE_NEAR, PLANE_RIGHT, PLANE_TOP },
		{ PLANE_NEAR, PLANE_RIGHT, PLANE_BOTTOM },
	};

	for (int i = 0; i < 8; i++) {
		Vector3 point;
		bool res = planes[intersections[i][0]].intersect_3(planes[intersections[i][1]], planes[intersections[i][2]], &point);
		ERR_FAIL_COND_V(!res, false);
		p_8points[i] = p_transform.xform(point);
	}

	return true;
}

Vector<Plane> CameraMatrix::get_projection_planes(const Transform3D &p_transform) const {
	/** Fast Plane Extraction from combined modelview/projection matrices.
	 * References:
	 * https://web.archive.org/web/20011221205252/https://www.markmorley.com/opengl/frustumculling.html
	 * https://web.archive.org/web/20061020020112/https://www2.ravensoft.com/users/ggribb/plane%20extraction.pdf
	 */

	Vector<Plane> planes;
	planes.resize(6);

	const real_t *matrix = (const real_t *)this->matrix;

	Plane new_plane;

	///////--- Near Plane ---///////
	new_plane = Plane(matrix[3] + matrix[2],
			matrix[7] + matrix[6],
			matrix[11] + matrix[10],
			matrix[15] + matrix[14]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[0] = p_transform.xform(new_plane);

	///////--- Far Plane ---///////
	new_plane = Plane(matrix[3] - matrix[2],
			matrix[7] - matrix[6],
			matrix[11] - matrix[10],
			matrix[15] - matrix[14]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[1] = p_transform.xform(new_plane);

	///////--- Left Plane ---///////
	new_plane = Plane(matrix[3] + matrix[0],
			matrix[7] + matrix[4],
			matrix[11] + matrix[8],
			matrix[15] + matrix[12]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[2] = p_transform.xform(new_plane);

	///////--- Top Plane ---///////
	new_plane = Plane(matrix[3] - matrix[1],
			matrix[7] - matrix[5],
			matrix[11] - matrix[9],
			matrix[15] - matrix[13]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[3] = p_transform.xform(new_plane);

	///////--- Right Plane ---///////
	new_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			matrix[15] - matrix[12]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[4] = p_transform.xform(new_plane);

	///////--- Bottom Plane ---///////
	new_plane = Plane(matrix[3] + matrix[1],
			matrix[7] + matrix[5],
			matrix[11] + matrix[9],
			matrix[15] + matrix[13]);

	new_plane.normal = -new_plane.normal;
	new_plane.normalize();

	planes.write[5] = p_transform.xform(new_plane);

	return planes;
}

CameraMatrix CameraMatrix::inverse() const {
	CameraMatrix cm = *this;
	cm.invert();
	return cm;
}

void CameraMatrix::invert() {
	int i, j, k;
	int pvt_i[4], pvt_j[4]; /* Locations of pivot matrix */
	real_t pvt_val; /* Value of current pivot element */
	real_t hold; /* Temporary storage */
	real_t determinat; /* Determinant */

	determinat = 1.0;
	for (k = 0; k < 4; k++) {
		/** Locate k'th pivot element **/
		pvt_val = matrix[k][k]; /** Initialize for search **/
		pvt_i[k] = k;
		pvt_j[k] = k;
		for (i = k; i < 4; i++) {
			for (j = k; j < 4; j++) {
				if (Math::absd(matrix[i][j]) > Math::absd(pvt_val)) {
					pvt_i[k] = i;
					pvt_j[k] = j;
					pvt_val = matrix[i][j];
				}
			}
		}

		/** Product of pivots, gives determinant when finished **/
		determinat *= pvt_val;
		if (Math::absd(determinat) < 1e-7) {
			return; //(false);  /** Matrix is singular (zero determinant). **/
		}

		/** "Interchange" rows (with sign change stuff) **/
		i = pvt_i[k];
		if (i != k) { /** If rows are different **/
			for (j = 0; j < 4; j++) {
				hold = -matrix[k][j];
				matrix[k][j] = matrix[i][j];
				matrix[i][j] = hold;
			}
		}

		/** "Interchange" columns **/
		j = pvt_j[k];
		if (j != k) { /** If columns are different **/
			for (i = 0; i < 4; i++) {
				hold = -matrix[i][k];
				matrix[i][k] = matrix[i][j];
				matrix[i][j] = hold;
			}
		}

		/** Divide column by minus pivot value **/
		for (i = 0; i < 4; i++) {
			if (i != k) {
				matrix[i][k] /= (-pvt_val);
			}
		}

		/** Reduce the matrix **/
		for (i = 0; i < 4; i++) {
			hold = matrix[i][k];
			for (j = 0; j < 4; j++) {
				if (i != k && j != k) {
					matrix[i][j] += hold * matrix[k][j];
				}
			}
		}

		/** Divide row by pivot **/
		for (j = 0; j < 4; j++) {
			if (j != k) {
				matrix[k][j] /= pvt_val;
			}
		}

		/** Replace pivot by reciprocal (at last we can touch it). **/
		matrix[k][k] = 1.0 / pvt_val;
	}

	/* That was most of the work, one final pass of row/column interchange */
	/* to finish */
	for (k = 4 - 2; k >= 0; k--) { /* Don't need to work with 1 by 1 corner*/
		i = pvt_j[k]; /* Rows to swap correspond to pivot COLUMN */
		if (i != k) { /* If rows are different */
			for (j = 0; j < 4; j++) {
				hold = matrix[k][j];
				matrix[k][j] = -matrix[i][j];
				matrix[i][j] = hold;
			}
		}

		j = pvt_i[k]; /* Columns to swap correspond to pivot ROW */
		if (j != k) { /* If columns are different */
			for (i = 0; i < 4; i++) {
				hold = matrix[i][k];
				matrix[i][k] = -matrix[i][j];
				matrix[i][j] = hold;
			}
		}
	}
}

void CameraMatrix::flip_y() {
	for (int i = 0; i < 4; i++) {
		matrix[1][i] = -matrix[1][i];
	}
}

CameraMatrix::CameraMatrix() {
	set_identity();
}

CameraMatrix CameraMatrix::operator*(const CameraMatrix &p_matrix) const {
	CameraMatrix new_matrix;

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			real_t ab = 0;
			for (int k = 0; k < 4; k++) {
				ab += matrix[k][i] * p_matrix.matrix[j][k];
			}
			new_matrix.matrix[j][i] = ab;
		}
	}

	return new_matrix;
}

void CameraMatrix::set_depth_correction(bool p_flip_y) {
	real_t *m = &matrix[0][0];

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
	m[10] = 0.5;
	m[11] = 0.0;
	m[12] = 0.0;
	m[13] = 0.0;
	m[14] = 0.5;
	m[15] = 1.0;
}

void CameraMatrix::set_light_bias() {
	real_t *m = &matrix[0][0];

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

void CameraMatrix::set_light_atlas_rect(const Rect2 &p_rect) {
	real_t *m = &matrix[0][0];

	m[0] = p_rect.size.width;
	m[1] = 0.0;
	m[2] = 0.0;
	m[3] = 0.0;
	m[4] = 0.0;
	m[5] = p_rect.size.height;
	m[6] = 0.0;
	m[7] = 0.0;
	m[8] = 0.0;
	m[9] = 0.0;
	m[10] = 1.0;
	m[11] = 0.0;
	m[12] = p_rect.position.x;
	m[13] = p_rect.position.y;
	m[14] = 0.0;
	m[15] = 1.0;
}

CameraMatrix::operator String() const {
	String str;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			str += String((j > 0) ? ", " : "\n") + rtos(matrix[i][j]);
		}
	}

	return str;
}

real_t CameraMatrix::get_aspect() const {
	Vector2 vp_he = get_viewport_half_extents();
	return vp_he.x / vp_he.y;
}

int CameraMatrix::get_pixels_per_meter(int p_for_pixel_width) const {
	Vector3 result = xform(Vector3(1, 0, -1));

	return int((result.x * 0.5 + 0.5) * p_for_pixel_width);
}

bool CameraMatrix::is_orthogonal() const {
	return matrix[3][3] == 1.0;
}

real_t CameraMatrix::get_fov() const {
	const real_t *matrix = (const real_t *)this->matrix;

	Plane right_plane = Plane(matrix[3] - matrix[0],
			matrix[7] - matrix[4],
			matrix[11] - matrix[8],
			-matrix[15] + matrix[12]);
	right_plane.normalize();

	if ((matrix[8] == 0) && (matrix[9] == 0)) {
		return Math::rad2deg(Math::acos(Math::abs(right_plane.normal.x))) * 2.0;
	} else {
		// our frustum is asymmetrical need to calculate the left planes angle separately..
		Plane left_plane = Plane(matrix[3] + matrix[0],
				matrix[7] + matrix[4],
				matrix[11] + matrix[8],
				matrix[15] + matrix[12]);
		left_plane.normalize();

		return Math::rad2deg(Math::acos(Math::abs(left_plane.normal.x))) + Math::rad2deg(Math::acos(Math::abs(right_plane.normal.x)));
	}
}

float CameraMatrix::get_lod_multiplier() const {
	if (is_orthogonal()) {
		return get_viewport_half_extents().x;
	} else {
		float zn = get_z_near();
		float width = get_viewport_half_extents().x * 2.0;
		return 1.0 / (zn / width);
	}

	//usage is lod_size / (lod_distance * multiplier) < threshold
}
void CameraMatrix::make_scale(const Vector3 &p_scale) {
	set_identity();
	matrix[0][0] = p_scale.x;
	matrix[1][1] = p_scale.y;
	matrix[2][2] = p_scale.z;
}

void CameraMatrix::scale_translate_to_fit(const AABB &p_aabb) {
	Vector3 min = p_aabb.position;
	Vector3 max = p_aabb.position + p_aabb.size;

	matrix[0][0] = 2 / (max.x - min.x);
	matrix[1][0] = 0;
	matrix[2][0] = 0;
	matrix[3][0] = -(max.x + min.x) / (max.x - min.x);

	matrix[0][1] = 0;
	matrix[1][1] = 2 / (max.y - min.y);
	matrix[2][1] = 0;
	matrix[3][1] = -(max.y + min.y) / (max.y - min.y);

	matrix[0][2] = 0;
	matrix[1][2] = 0;
	matrix[2][2] = 2 / (max.z - min.z);
	matrix[3][2] = -(max.z + min.z) / (max.z - min.z);

	matrix[0][3] = 0;
	matrix[1][3] = 0;
	matrix[2][3] = 0;
	matrix[3][3] = 1;
}

CameraMatrix::operator Transform3D() const {
	Transform3D tr;
	const real_t *m = &matrix[0][0];

	tr.basis.elements[0][0] = m[0];
	tr.basis.elements[1][0] = m[1];
	tr.basis.elements[2][0] = m[2];

	tr.basis.elements[0][1] = m[4];
	tr.basis.elements[1][1] = m[5];
	tr.basis.elements[2][1] = m[6];

	tr.basis.elements[0][2] = m[8];
	tr.basis.elements[1][2] = m[9];
	tr.basis.elements[2][2] = m[10];

	tr.origin.x = m[12];
	tr.origin.y = m[13];
	tr.origin.z = m[14];

	return tr;
}

CameraMatrix::CameraMatrix(const Transform3D &p_transform) {
	const Transform3D &tr = p_transform;
	real_t *m = &matrix[0][0];

	m[0] = tr.basis.elements[0][0];
	m[1] = tr.basis.elements[1][0];
	m[2] = tr.basis.elements[2][0];
	m[3] = 0.0;
	m[4] = tr.basis.elements[0][1];
	m[5] = tr.basis.elements[1][1];
	m[6] = tr.basis.elements[2][1];
	m[7] = 0.0;
	m[8] = tr.basis.elements[0][2];
	m[9] = tr.basis.elements[1][2];
	m[10] = tr.basis.elements[2][2];
	m[11] = 0.0;
	m[12] = tr.origin.x;
	m[13] = tr.origin.y;
	m[14] = tr.origin.z;
	m[15] = 1.0;
}

CameraMatrix::~CameraMatrix() {
}
