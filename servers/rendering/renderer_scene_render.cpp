/*************************************************************************/
/*  renderer_scene_render.cpp                                            */
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

#include "renderer_scene_render.h"

void RendererSceneRender::CameraData::set_camera(const Transform3D p_transform, const CameraMatrix p_projection, bool p_is_ortogonal, bool p_vaspect) {
	view_count = 1;
	is_ortogonal = p_is_ortogonal;
	vaspect = p_vaspect;

	main_transform = p_transform;
	main_projection = p_projection;

	view_offset[0] = Transform3D();
	view_projection[0] = p_projection;
}

void RendererSceneRender::CameraData::set_multiview_camera(uint32_t p_view_count, const Transform3D *p_transforms, const CameraMatrix *p_projections, bool p_is_ortogonal, bool p_vaspect) {
	ERR_FAIL_COND_MSG(p_view_count != 2, "Incorrect view count for stereoscopic view");

	view_count = p_view_count;
	is_ortogonal = p_is_ortogonal;
	vaspect = p_vaspect;
	Vector<Plane> planes[2];

	/////////////////////////////////////////////////////////////////////////////
	// Figure out our center transform

	// 1. obtain our planes
	for (uint32_t v = 0; v < view_count; v++) {
		planes[v] = p_projections[v].get_projection_planes(p_transforms[v]);
	}

	// 2. average and normalize plane normals to obtain z vector, cross them to obtain y vector, and from there the x vector for combined camera basis.
	Vector3 n0 = planes[0][CameraMatrix::PLANE_LEFT].normal;
	Vector3 n1 = planes[1][CameraMatrix::PLANE_RIGHT].normal;
	Vector3 z = (n0 + n1).normalized();
	Vector3 y = n0.cross(n1).normalized();
	Vector3 x = y.cross(z).normalized();
	y = z.cross(x).normalized();
	main_transform.basis.set(x, y, z);

	// 3. create a horizon plane with one of the eyes and the up vector as normal.
	Plane horizon(y, p_transforms[0].origin);

	// 4. Intersect horizon, left and right to obtain the combined camera origin.
	ERR_FAIL_COND_MSG(
			!horizon.intersect_3(planes[0][CameraMatrix::PLANE_LEFT], planes[1][CameraMatrix::PLANE_RIGHT], &main_transform.origin), "Can't determine camera origin");

	// handy to have the inverse of the transform we just build
	Transform3D main_transform_inv = main_transform.inverse();

	// 5. figure out far plane, this could use some improvement, we may have our far plane too close like this, not sure if this matters
	Vector3 far_center = (planes[0][CameraMatrix::PLANE_FAR].center() + planes[1][CameraMatrix::PLANE_FAR].center()) * 0.5;
	Plane far(-z, far_center);

	/////////////////////////////////////////////////////////////////////////////
	// Figure out our top/bottom planes

	// 6. Intersect far and left planes with top planes from both eyes, save the point with highest y as top_left.
	Vector3 top_left, other;
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[0][CameraMatrix::PLANE_LEFT], planes[0][CameraMatrix::PLANE_TOP], &top_left), "Can't determine left camera far/left/top vector");
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[1][CameraMatrix::PLANE_LEFT], planes[1][CameraMatrix::PLANE_TOP], &other), "Can't determine right camera far/left/top vector");
	if (y.dot(top_left) < y.dot(other)) {
		top_left = other;
	}

	// 7. Intersect far and left planes with bottom planes from both eyes, save the point with lowest y as bottom_left.
	Vector3 bottom_left;
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[0][CameraMatrix::PLANE_LEFT], planes[0][CameraMatrix::PLANE_BOTTOM], &bottom_left), "Can't determine left camera far/left/bottom vector");
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[1][CameraMatrix::PLANE_LEFT], planes[1][CameraMatrix::PLANE_BOTTOM], &other), "Can't determine right camera far/left/bottom vector");
	if (y.dot(other) < y.dot(bottom_left)) {
		bottom_left = other;
	}

	// 8. Intersect far and right planes with top planes from both eyes, save the point with highest y as top_right.
	Vector3 top_right;
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[0][CameraMatrix::PLANE_RIGHT], planes[0][CameraMatrix::PLANE_TOP], &top_right), "Can't determine left camera far/right/top vector");
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[1][CameraMatrix::PLANE_RIGHT], planes[1][CameraMatrix::PLANE_TOP], &other), "Can't determine right camera far/right/top vector");
	if (y.dot(top_right) < y.dot(other)) {
		top_right = other;
	}

	//  9. Intersect far and right planes with bottom planes from both eyes, save the point with lowest y as bottom_right.
	Vector3 bottom_right;
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[0][CameraMatrix::PLANE_RIGHT], planes[0][CameraMatrix::PLANE_BOTTOM], &bottom_right), "Can't determine left camera far/right/bottom vector");
	ERR_FAIL_COND_MSG(
			!far.intersect_3(planes[1][CameraMatrix::PLANE_RIGHT], planes[1][CameraMatrix::PLANE_BOTTOM], &other), "Can't determine right camera far/right/bottom vector");
	if (y.dot(other) < y.dot(bottom_right)) {
		bottom_right = other;
	}

	// 10. Create top plane with these points: camera origin, top_left, top_right
	Plane top(main_transform.origin, top_left, top_right);

	// 11. Create bottom plane with these points: camera origin, bottom_left, bottom_right
	Plane bottom(main_transform.origin, bottom_left, bottom_right);

	/////////////////////////////////////////////////////////////////////////////
	// Figure out our near plane points

	// 12. Create a near plane using -camera z and the eye further along in that axis.
	Plane near;
	Vector3 neg_z = -z;
	if (neg_z.dot(p_transforms[1].origin) < neg_z.dot(p_transforms[0].origin)) {
		near = Plane(neg_z, p_transforms[0].origin);
	} else {
		near = Plane(neg_z, p_transforms[1].origin);
	}

	// 13. Intersect near plane with bottm/left planes, to obtain min_vec then top/right to obtain max_vec
	Vector3 min_vec;
	ERR_FAIL_COND_MSG(
			!near.intersect_3(bottom, planes[0][CameraMatrix::PLANE_LEFT], &min_vec), "Can't determine left camera near/left/bottom vector");
	ERR_FAIL_COND_MSG(
			!near.intersect_3(bottom, planes[1][CameraMatrix::PLANE_LEFT], &other), "Can't determine right camera near/left/bottom vector");
	if (x.dot(other) < x.dot(min_vec)) {
		min_vec = other;
	}

	Vector3 max_vec;
	ERR_FAIL_COND_MSG(
			!near.intersect_3(top, planes[0][CameraMatrix::PLANE_RIGHT], &max_vec), "Can't determine left camera near/right/top vector");
	ERR_FAIL_COND_MSG(
			!near.intersect_3(top, planes[1][CameraMatrix::PLANE_RIGHT], &other), "Can't determine right camera near/right/top vector");
	if (x.dot(max_vec) < x.dot(other)) {
		max_vec = other;
	}

	// 14. transform these points by the inverse camera to obtain local_min_vec and local_max_vec
	Vector3 local_min_vec = main_transform_inv.xform(min_vec);
	Vector3 local_max_vec = main_transform_inv.xform(max_vec);

	// 15. get x and y from these to obtain left, top, right bottom for the frustum. Get the distance from near plane to camera origin to obtain near, and the distance from the far plane to the camer origin to obtain far.
	float z_near = -near.distance_to(main_transform.origin);
	float z_far = -far.distance_to(main_transform.origin);

	// 16. Use this to build the combined camera matrix.
	main_projection.set_frustum(local_min_vec.x, local_max_vec.x, local_min_vec.y, local_max_vec.y, z_near, z_far);

	/////////////////////////////////////////////////////////////////////////////
	// 3. Copy our view data
	for (uint32_t v = 0; v < view_count; v++) {
		view_offset[v] = main_transform_inv * p_transforms[v];
		view_projection[v] = p_projections[v] * CameraMatrix(view_offset[v].inverse());
	}
}
