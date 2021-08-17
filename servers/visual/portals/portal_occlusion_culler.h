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
#include "core/math/camera_matrix.h"
#include "core/math/geometry.h"
#include "portal_types.h"

class PortalOcclusionCuller {
	enum {
		MAX_SPHERES = 64,
		MAX_POLYS = 64,
	};

	class Clipper {
	public:
		real_t clip_and_find_poly_area(bool p_debug, const Plane *p_verts, int p_num_verts);

	private:
		enum Boundary {
			B_LEFT,
			B_RIGHT,
			B_TOP,
			B_BOTTOM,
			B_NEAR,
			B_FAR,
		};

		//bool clip(Boundary p_boundary);
		bool is_inside(const Plane &p_pt, Boundary p_boundary);
		Plane intersect(const Plane &p_a, const Plane &p_b, Boundary p_boundary);
		void debug_print_points(String p_string);

		Plane interpolate(const Plane &p_a, const Plane &p_b, real_t p_t) const;
		bool clip_to_plane(real_t a, real_t b, real_t c, real_t d);

		LocalVectori<Plane> _pts_in;
		LocalVectori<Plane> _pts_out;

		// after perspective divide
		LocalVectori<Vector3> _pts_final;

		template <typename T>
		int sgn(T val) {
			return (T(0) < val) - (val < T(0));
		}
	};

public:
	PortalOcclusionCuller();

	void prepare_camera(const CameraMatrix &p_cam_matrix, const Vector3 &p_cam_dir) {
		_matrix_camera = p_cam_matrix;
		_pt_cam_dir = p_cam_dir;
	}

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
		if (!_occluders_present) {
			return false;
		}
		if (cull_aabb_to_polys(p_aabb)) {
			return true;
		}

		return cull_sphere(p_aabb.get_center(), p_aabb.size.length() * 0.5, false);
	}

	bool cull_sphere(const Vector3 &p_occludee_center, real_t p_occludee_radius, bool p_cull_to_polys = true) const;

	Geometry::MeshData debug_get_current_polys() const;

	static bool _redraw_gizmo;

private:
	bool cull_sphere_to_spheres(const Vector3 &p_occludee_center, real_t p_occludee_radius, const Vector3 &p_ray_dir, real_t p_dist_to_occludee) const;
	bool cull_sphere_to_polys(const Vector3 &p_occludee_center, real_t p_occludee_radius) const;
	bool cull_aabb_to_polys(const AABB &p_aabb) const;

	// experimental
	bool cull_aabb_to_polys_ex(int p_ignore_poly_id, const AABB &p_aabb, const Plane *p_poly_split_plane, int p_depth = 0) const;
	bool _is_poly_of_interest_to_split_plane(const Plane *p_poly_split_plane, int p_poly_id) const;

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

	bool calculate_poly_goodness_of_fit(bool debug, const VSOccluder_Mesh &p_opoly, real_t &r_fit);
	void whittle_polys();

	bool is_vso_poly_culled(const VSOccluder_Mesh &p_opoly, const LocalVector<Plane> &p_planes) const {
		return is_poly_culled(p_opoly.poly_world, p_planes);
	}

	// If all the points of the poly are beyond one of the planes (e.g. frustum), it is completely culled.
	bool is_poly_culled(const Occlusion::Poly &p_opoly, const LocalVector<Plane> &p_planes) const {
		for (unsigned int p = 0; p < p_planes.size(); p++) {
			const Plane &plane = p_planes[p];

			int points_outside = 0;
			for (int n = 0; n < p_opoly.num_verts; n++) {
				const Vector3 &pt = p_opoly.verts[n];
				if (!plane.is_point_over(pt)) {
					break;
				} else {
					points_outside++;
				}
			}

			if (points_outside == p_opoly.num_verts) {
				return true;
			}
		}
		return false;
	}

	// All the points of the poly must be within ALL the planes to return true.
	bool is_poly_inside_occlusion_volume(const Occlusion::Poly &p_opoly, const LocalVector<Plane> &p_planes) const {
		for (unsigned int p = 0; p < p_planes.size(); p++) {
			const Plane &plane = p_planes[p];

			for (int n = 0; n < p_opoly.num_verts; n++) {
				const Vector3 &pt = p_opoly.verts[n];
				if (plane.is_point_over(pt)) {
					return false;
				}
			}
		}
		return true;
	}

	void log(String p_string, int p_depth = 0) const;

	// only a number of the spheres in the scene will be chosen to be
	// active based on their distance to the camera, screen space etc.
	Occlusion::Sphere _spheres[MAX_SPHERES];
	real_t _sphere_distances[MAX_SPHERES];
	real_t _sphere_closest_dist = 0.0;
	int _num_spheres = 0;
	int _max_spheres = 8;

	struct SortPoly {
		enum SortPolyFlags {
			SPF_FACES_CAMERA = 1,
			SPF_DONE = 2,
			SPF_TESTED_AS_OCCLUDER = 4,
		};

		Occlusion::Poly poly;
		uint32_t flags;
		uint32_t poly_source_id;
		real_t goodness_of_fit;
	};

	SortPoly _polys[MAX_POLYS];
	int _num_polys = 0;
	int _max_polys = 8;

#ifdef TOOLS_ENABLED
	uint32_t _poly_checksum = 0;
#endif

	Vector3 _pt_camera;
	Vector3 _pt_cam_dir;

	CameraMatrix _matrix_camera;

	Clipper _clipper;

	bool _occluders_present = false;

	static bool _debug_log;
};

#endif // PORTAL_OCCLUSION_CULLER_H
