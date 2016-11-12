/*************************************************************************/
/*  frustum.h                                                            */
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
#ifndef FRUSTUM_H
#define FRUSTUM_H

#include "camera_matrix.h"
/**
	@author Bastiaan Olij <mux213@gmail.com>
*/

struct Frustum {
	enum Eyes {
		EYE_LEFT,
		EYE_RIGHT
	};

	real_t left, right, top, bottom;

	void set_frustum(real_t p_left, real_t p_right, real_t p_top, real_t p_bottom);
	void set_frustum(real_t p_fov_degrees);
	void set_frustum(real_t p_fov_degrees, int p_eye, real_t p_intraocular_dist, real_t p_convergency_dist);
	void set_frustum_for_hmd(int p_eye, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample = 1.0);

	CameraMatrix make_camera_matrix(real_t p_aspect, bool p_vaspect, real_t p_znear, real_t p_zfar) const;

	Frustum();
	Frustum(real_t p_left, real_t p_right, real_t p_top, real_t p_bottom);
	~Frustum();
};

#endif
