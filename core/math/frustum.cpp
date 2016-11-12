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

void Frustum::set_frustum(real_t p_left, real_t p_right, real_t p_top, real_t p_bottom) {
	left = p_left;
	right = p_right;
	top = p_top;
	bottom = p_bottom;
};

void Frustum::set_frustum(real_t p_fov_degrees) {
	right = tan(p_fov_degrees * Math_PI / 360.0);
	left = -right;
	top = right;
	bottom = left;
};

void Frustum::set_frustum(real_t p_fov_degrees, int p_eye, real_t p_intraocular_dist, real_t p_convergency_dist) {
	right = tan(p_fov_degrees * Math_PI / 360.0);
	left = -right;
	top = right;
	bottom = left;

	real_t shift = p_intraocular_dist / (2.0 * p_convergency_dist);
	if (p_eye == Frustum::EYE_LEFT) {
		left += shift;
		right += shift;
	} else {
		left -= shift;
		right -= shift;
	};
};

void Frustum::set_frustum_for_hmd(int p_eye, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample) {
	/*
		We calculate our frustum initialy based initialy without taking the magnifying properties of the lens into account limiting our initial FOV
		to the physical size of our device.

		The magnification of our lens is determined by the k1..kn constants that will be used to distort our rendered image but we can't use those
		directly to figure out by how much we need to increase our FOV. The calculation that does this assumes that a distance of 1.0 is equivilent
		to our unmagnified FOV. If we oversample our render target by a factor of two, we'll be adjusting our coordinates accordingly by multiplying
		our distance by the oversample, applying our magnification, and then divided the result. 

		By how much we oversample is always a tradeoff between performance and how much of the screen we want to use, especially with lenses like
		we find in headsets such as the Vive and Rift which often oversize by 2.0 while mobile VR often uses much lower magnification due to
		performance limitations of the phones.
	*/

	// create our factors based on our physical screen dimentions
	real_t f1 = (p_intraocular_dist / 2.0) / (2.0 * p_display_to_lens);
	real_t f2 = ((p_display_width - p_intraocular_dist) / 2.0) / (2.0 * p_display_to_lens);
	real_t f3 = (p_display_width / 4.0) / (2.0 * p_display_to_lens);

	// apply our oversample to increase the FOV
	f3 *= p_oversample;
	real_t add_width = ((f1 + f2) * (p_oversample - 1.0)) / 2.0;
	f1 += add_width;
	f2 += add_width;

	// and set our frustum for the correct eye
	left = p_eye == Frustum::EYE_LEFT ? -f2 : -f1;
	right = p_eye == Frustum::EYE_LEFT ? f1 : f2;
	top = f3;
	bottom = -f3;
};

CameraMatrix Frustum::make_camera_matrix(real_t p_aspect, bool p_vaspect, real_t p_znear, real_t p_zfar) const {
	// Slightly modified version of ComposeProjection found in: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetProjectionRaw
	// We are expecting a left/right/top/bottom frustum not adjusted to the near place but unified.

	CameraMatrix M;
	real_t *m = &M.matrix[0][0];
	real_t l = left;
	real_t r = right;
	real_t t = top;
	real_t b = bottom;

	// apply aspect ratio of our viewport
	if (p_vaspect) {
		t /= p_aspect;
		b /= p_aspect;
	} else {
		l *= p_aspect;
		r *= p_aspect;
	}

	real_t idx = 1.0f / (r - l);
	real_t idy = 1.0f / (t - b);
	real_t idz = 1.0f / (p_zfar - p_znear);
	real_t sx = r + l;
	real_t sy = t + b;

	m[0] = 2.0f * idx;
	m[4] = 0.0f;
	m[8] = sx * idx;
	m[12] = 0.0f;
	m[1] = 0.0f;
	m[5] = 2.0f * idy;
	m[9] = sy * idy;
	m[13] = 0.0f;
	m[2] = 0.0f;
	m[6] = 0.0f;
	m[10] = -p_zfar * idz;
	m[14] = -p_zfar * p_znear * idz;
	m[3] = 0.0f;
	m[7] = 0.0f;
	m[11] = -1.0f;
	m[15] = 0.0f;

	return M;
};

Frustum::Frustum() {
	set_frustum(-0.5f, 0.5f, 0.5f, -0.5f);
};

Frustum::Frustum(real_t p_left, real_t p_right, real_t p_top, real_t p_bottom) {
	set_frustum(p_left, p_right, p_top, p_bottom);
};

Frustum::~Frustum(){};
