/*************************************************************************/
/*  frustum.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "frustum.h"
#include "math_funcs.h"

void Frustum::set_frustum(float p_left, float p_right ,float p_top, float p_bottom) {
	left = p_left;
	right = p_right;
	top = p_top;
	bottom = p_bottom;
};

void Frustum::set_frustum(float p_fov_degrees) {
	right = tan(p_fov_degrees * Math_PI / 360.0);
	left = -right;
	top = right;
	bottom = left;
};

void Frustum::set_frustum(float p_fov_degrees, int p_eye, float p_intraocular_dist, float p_convergency_dist) {
	right = tan(p_fov_degrees * Math_PI / 360.0);
	left = -right;
	top = right;
	bottom = left;

	float shift = p_intraocular_dist / (2.0 * p_convergency_dist);
	if (p_eye == Frustum::EYE_LEFT) {
		left += shift;
		right += shift;
	} else {
		left -= shift;
		right -= shift;
	};
};

CameraMatrix Frustum::make_camera_matrix(float p_aspect, bool p_vaspect, float p_znear, float p_zfar) const {
		// Slightly modified version of ComposeProjection found in: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetProjectionRaw
		// We are expecting a left/right/top/bottom frustum not adjusted to the near place but unified.

		CameraMatrix M;
		float *m = &M.matrix[0][0];
		float l = left;
		float r = right;
		float t = top;
		float b = bottom;

		// apply aspect ratio of our viewport
		if (p_vaspect) {
			t /= p_aspect;
			b /= p_aspect;
		} else {
			l *= p_aspect;
			r *= p_aspect;
		}

		float idx = 1.0f / (r - l);
		float idy = 1.0f / (t - b);
		float idz = 1.0f / (p_zfar - p_znear);
		float sx = r + l;
		float sy = b + t;

		m[0] = 2.0f*idx;  m[4] = 0.0f;      m[8]  = sx*idx;       m[12] = 0.0f;
		m[1] = 0.0f;      m[5] = 2.0f*idy;  m[9]  = sy*idy;       m[13] = 0.0f;
		m[2] = 0.0f;      m[6] = 0.0f;      m[10] = -p_zfar*idz;  m[14] = -p_zfar*p_znear*idz;
		m[3] = 0.0f;      m[7] = 0.0f;      m[11] = -1.0f;        m[15] = 0.0f;

		return M;
};

Frustum::Frustum() {
	set_frustum(-0.5f, 0.5f, 0.5f, -0.5f);
};

Frustum::Frustum(float p_left, float p_right ,float p_top, float p_bottom) {
	set_frustum(p_left, p_right, p_top, p_bottom);
};

Frustum::~Frustum() {
};
