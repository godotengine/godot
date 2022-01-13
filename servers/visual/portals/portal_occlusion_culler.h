/*************************************************************************/
/*  portal_occlusion_culler.h                                            */
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

#ifndef PORTAL_OCCLUSION_CULLER_H
#define PORTAL_OCCLUSION_CULLER_H

class PortalRenderer;
#include "portal_types.h"

class PortalOcclusionCuller {
	enum {
		MAX_SPHERES = 64,
	};

public:
	PortalOcclusionCuller();
	void prepare(PortalRenderer &p_portal_renderer, const VSRoom &p_room, const Vector3 &pt_camera, const LocalVector<Plane> &p_planes, const Plane *p_near_plane) {
		if (p_near_plane) {
			static LocalVector<Plane> local_planes;
			int size_wanted = p_planes.size() + 1;

			if ((int)local_planes.size() != size_wanted) {
				local_planes.resize(size_wanted);
			}
			for (int n = 0; n < (int)p_planes.size(); n++) {
				local_planes[n] = p_planes[n];
			}
			local_planes[size_wanted - 1] = *p_near_plane;

			prepare_generic(p_portal_renderer, p_room._occluder_pool_ids, pt_camera, local_planes);
		} else {
			prepare_generic(p_portal_renderer, p_room._occluder_pool_ids, pt_camera, p_planes);
		}
	}

	void prepare_generic(PortalRenderer &p_portal_renderer, const LocalVector<uint32_t, uint32_t> &p_occluder_pool_ids, const Vector3 &pt_camera, const LocalVector<Plane> &p_planes);
	bool cull_aabb(const AABB &p_aabb) const {
		if (!_num_spheres) {
			return false;
		}

		return cull_sphere(p_aabb.get_center(), p_aabb.size.length() * 0.5);
	}
	bool cull_sphere(const Vector3 &p_occludee_center, real_t p_occludee_radius, int p_ignore_sphere = -1) const;

private:
	// if a sphere is entirely in front of any of the culling planes, it can't be seen so returns false
	bool is_sphere_culled(const Vector3 &p_pos, real_t p_radius, const LocalVector<Plane> &p_planes) const {
		for (unsigned int p = 0; p < p_planes.size(); p++) {
			real_t dist = p_planes[p].distance_to(p_pos);
			if (dist > p_radius) {
				return true;
			}
		}

		return false;
	}

	bool is_aabb_culled(const AABB &p_aabb, const LocalVector<Plane> &p_planes) const {
		const Vector3 &size = p_aabb.size;
		Vector3 half_extents = size * 0.5;
		Vector3 ofs = p_aabb.position + half_extents;

		for (unsigned int i = 0; i < p_planes.size(); i++) {
			const Plane &p = p_planes[i];
			Vector3 point(
					(p.normal.x > 0) ? -half_extents.x : half_extents.x,
					(p.normal.y > 0) ? -half_extents.y : half_extents.y,
					(p.normal.z > 0) ? -half_extents.z : half_extents.z);
			point += ofs;
			if (p.is_point_over(point)) {
				return true;
			}
		}

		return false;
	}

	// only a number of the spheres in the scene will be chosen to be
	// active based on their distance to the camera, screen space etc.
	Occlusion::Sphere _spheres[MAX_SPHERES];
	real_t _sphere_distances[MAX_SPHERES];
	real_t _sphere_closest_dist = 0.0;
	int _num_spheres = 0;
	int _max_spheres = 8;

	Vector3 _pt_camera;
};

#endif // PORTAL_OCCLUSION_CULLER_H
