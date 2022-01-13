/*************************************************************************/
/*  portal_occlusion_culler.cpp                                          */
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

#include "portal_occlusion_culler.h"

#include "core/project_settings.h"
#include "portal_renderer.h"

void PortalOcclusionCuller::prepare_generic(PortalRenderer &p_portal_renderer, const LocalVector<uint32_t, uint32_t> &p_occluder_pool_ids, const Vector3 &pt_camera, const LocalVector<Plane> &p_planes) {
	_num_spheres = 0;
	_pt_camera = pt_camera;

	real_t goodness_of_fit[MAX_SPHERES];
	for (int n = 0; n < _max_spheres; n++) {
		goodness_of_fit[n] = 0.0;
	}
	real_t weakest_fit = FLT_MAX;
	int weakest_sphere = 0;
	_sphere_closest_dist = FLT_MAX;

	// TODO : occlusion cull spheres AGAINST themselves.
	// i.e. a sphere that is occluded by another occluder is no
	// use as an occluder...

	// find sphere occluders
	for (unsigned int o = 0; o < p_occluder_pool_ids.size(); o++) {
		int id = p_occluder_pool_ids[o];
		VSOccluder &occ = p_portal_renderer.get_pool_occluder(id);

		// is it active?
		// in the case of rooms, they will always be active, as inactive
		// are removed from rooms. But for whole scene mode, some may be inactive.
		if (!occ.active) {
			continue;
		}

		if (occ.type == VSOccluder::OT_SPHERE) {
			// make sure world space spheres are up to date
			p_portal_renderer.occluder_ensure_up_to_date_sphere(occ);

			// cull entire AABB
			if (is_aabb_culled(occ.aabb, p_planes)) {
				continue;
			}

			// multiple spheres
			for (int n = 0; n < occ.list_ids.size(); n++) {
				const Occlusion::Sphere &occluder_sphere = p_portal_renderer.get_pool_occluder_sphere(occ.list_ids[n]).world;

				// is the occluder sphere culled?
				if (is_sphere_culled(occluder_sphere.pos, occluder_sphere.radius, p_planes)) {
					continue;
				}

				real_t dist = (occluder_sphere.pos - pt_camera).length();

				// calculate the goodness of fit .. smaller distance better, and larger radius
				// calculate adjusted radius at 100.0
				real_t fit = 100 / MAX(dist, 0.01);
				fit *= occluder_sphere.radius;

				// until we reach the max, just keep recording, and keep track
				// of the worst fit
				if (_num_spheres < _max_spheres) {
					_spheres[_num_spheres] = occluder_sphere;
					_sphere_distances[_num_spheres] = dist;
					goodness_of_fit[_num_spheres] = fit;

					if (fit < weakest_fit) {
						weakest_fit = fit;
						weakest_sphere = _num_spheres;
					}

					// keep a record of the closest sphere for quick rejects
					if (dist < _sphere_closest_dist) {
						_sphere_closest_dist = dist;
					}

					_num_spheres++;
				} else {
					// must beat the weakest
					if (fit > weakest_fit) {
						_spheres[weakest_sphere] = occluder_sphere;
						_sphere_distances[weakest_sphere] = dist;
						goodness_of_fit[weakest_sphere] = fit;

						// keep a record of the closest sphere for quick rejects
						if (dist < _sphere_closest_dist) {
							_sphere_closest_dist = dist;
						}

						// the weakest may have changed (this could be done more efficiently)
						weakest_fit = FLT_MAX;
						for (int s = 0; s < _max_spheres; s++) {
							if (goodness_of_fit[s] < weakest_fit) {
								weakest_fit = goodness_of_fit[s];
								weakest_sphere = s;
							}
						}
					}
				}
			}
		} // sphere
	} // for o

	// force the sphere closest distance to above zero to prevent
	// divide by zero in the quick reject
	_sphere_closest_dist = MAX(_sphere_closest_dist, 0.001);

	// sphere self occlusion.
	// we could avoid testing the closest sphere, but the complexity isn't worth any speed benefit
	for (int n = 0; n < _num_spheres; n++) {
		const Occlusion::Sphere &sphere = _spheres[n];

		// is it occluded by another sphere?
		if (cull_sphere(sphere.pos, sphere.radius, n)) {
			// yes, unordered remove
			_num_spheres--;
			_spheres[n] = _spheres[_num_spheres];
			_sphere_distances[n] = _sphere_distances[_num_spheres];

			// repeat this n
			n--;
		}
	}
}

bool PortalOcclusionCuller::cull_sphere(const Vector3 &p_occludee_center, real_t p_occludee_radius, int p_ignore_sphere) const {
	if (!_num_spheres) {
		return false;
	}

	// ray from origin to the occludee
	Vector3 ray_dir = p_occludee_center - _pt_camera;
	real_t dist_to_occludee_raw = ray_dir.length();

	// account for occludee radius
	real_t dist_to_occludee = dist_to_occludee_raw - p_occludee_radius;

	// prevent divide by zero, and the occludee cannot be occluded if we are WITHIN
	// its bounding sphere... so no need to check
	if (dist_to_occludee < _sphere_closest_dist) {
		return false;
	}

	// normalize ray
	// hopefully by this point, dist_to_occludee_raw cannot possibly be zero due to above check
	ray_dir *= 1.0 / dist_to_occludee_raw;

	// this can probably be done cheaper with dot products but the math might be a bit fiddly to get right
	for (int s = 0; s < _num_spheres; s++) {
		//  first get the sphere distance
		real_t occluder_dist_to_cam = _sphere_distances[s];
		if (dist_to_occludee < occluder_dist_to_cam) {
			// can't occlude
			continue;
		}

		// the perspective adjusted occludee radius
		real_t adjusted_occludee_radius = p_occludee_radius * (occluder_dist_to_cam / dist_to_occludee);

		const Occlusion::Sphere &occluder_sphere = _spheres[s];
		real_t occluder_radius = occluder_sphere.radius - adjusted_occludee_radius;

		if (occluder_radius > 0.0) {
			occluder_radius = occluder_radius * occluder_radius;

			// distance to hit
			real_t dist;

			if (occluder_sphere.intersect_ray(_pt_camera, ray_dir, dist, occluder_radius)) {
				if ((dist < dist_to_occludee) && (s != p_ignore_sphere)) {
					// occluded
					return true;
				}
			}
		} // expanded occluder radius is more than 0
	}

	return false;
}

PortalOcclusionCuller::PortalOcclusionCuller() {
	_max_spheres = GLOBAL_GET("rendering/misc/occlusion_culling/max_active_spheres");
}
