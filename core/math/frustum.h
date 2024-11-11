/**************************************************************************/
/*  frustum.h                                                             */
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

#include "core/math/plane.h"

struct Transform3D;
struct Rect2;

struct Frustum {
	/*
	Frustum stands for the 6 planes that define a frustum in 3D space.
	It is an alternative way to struct Projection to represent a projection matrix.

	Unlike Projection though, it does not suffer from hard numerical precision
	issues when retrieving near/far planes and distances, as well as boundary points.

	It must be preferred over Projection when the frustum is to be used for culling
	or other geometric operations, in contrast to (un)projecting points. This ensures
	a reliable behavior with near and far distances across the whole range of floating
	point values.

	Planes are stored in the same order defined in struct Projection :

		enum Projection::Planes {
			PLANE_NEAR,
			PLANE_FAR,
			PLANE_LEFT,
			PLANE_TOP,
			PLANE_RIGHT,
			PLANE_BOTTOM
		};
	*/

	Plane planes[6];

	_FORCE_INLINE_ Plane &operator[](int p_index) { return planes[p_index]; }
	_FORCE_INLINE_ const Plane &operator[](int p_index) const { return planes[p_index]; }

	void set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov = false);
	void set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist);
	void set_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far);
	void set_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar);
	void set_orthogonal(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov = false);
	void set_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far);
	void set_frustum(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov = false);

	Vector<Plane> get_projection_planes(const Transform3D &p_transform) const;

	bool get_endpoints(const Transform3D &p_transform, Vector3 *p_8points) const;
	Vector2 get_viewport_half_extents() const;
	Rect2 get_viewport_rect() const;
	Vector2 get_far_plane_half_extents() const;
	Rect2 get_far_plane_rect() const;
	real_t get_z_near() const;
	real_t get_z_far() const;

	Frustum() = default;
	Frustum(const Vector<Plane> &p_planes);
	Frustum(const Frustum &p_frustum) = default;
	Frustum &operator=(const Frustum &p_frustum) = default;
};
