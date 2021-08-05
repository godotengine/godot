/*************************************************************************/
/*  portal_occlusion_culler.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
		prepare_generic(p_portal_renderer, p_room._occluder_pool_ids, pt_camera, p_planes, p_near_plane);
	}

	void prepare_generic(PortalRenderer &p_portal_renderer, const LocalVector<uint32_t, uint32_t> &p_occluder_pool_ids, const Vector3 &pt_camera, const LocalVector<Plane> &p_planes, const Plane *p_near_plane);
	bool cull_aabb(const AABB &p_aabb) const {
		if (!_num_spheres) {
			return false;
		}

		return cull_sphere(p_aabb.get_center(), p_aabb.size.length() * 0.5);
	}
	bool cull_sphere(const Vector3 &p_occludee_center, real_t p_occludee_radius) const;

private:
	// if a sphere is entirely in front of any of the culling planes, it can't be seen so returns false
	bool is_sphere_culled(const Vector3 &p_pos, real_t p_radius, const LocalVector<Plane> &p_planes, const Plane *p_near_plane) const {
		if (p_near_plane) {
			real_t dist = p_near_plane->distance_to(p_pos);
			if (dist > p_radius) {
				return true;
			}
		}

		for (unsigned int p = 0; p < p_planes.size(); p++) {
			real_t dist = p_planes[p].distance_to(p_pos);
			if (dist > p_radius) {
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
