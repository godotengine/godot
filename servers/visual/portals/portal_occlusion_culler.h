/**************************************************************************/
/*  portal_occlusion_culler.h                                             */
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
		real_t clip_and_find_poly_area(const Plane *p_verts, int p_num_verts);

	private:
		enum Boundary {
			B_LEFT,
			B_RIGHT,
			B_TOP,
			B_BOTTOM,
			B_NEAR,
			B_FAR,
		};

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

		return cull_sphere(p_aabb.get_center(), p_aabb.size.length() * 0.5, -1, false);
	}

	bool cull_sphere(const Vector3 &p_occludee_center, real_t p_occludee_radius, int p_ignore_sphere = -1, bool p_cull_to_polys = true) const;

	Geometry::MeshData debug_get_current_polys() const;

	static bool _redraw_gizmo;

private:
	bool cull_sphere_to_spheres(const Vector3 &p_occludee_center, real_t p_occludee_radius, const Vector3 &p_ray_dir, real_t p_dist_to_occludee, int p_ignore_sphere) const;
	bool cull_sphere_to_polys(const Vector3 &p_occludee_center, real_t p_occludee_radius) const;
	bool cull_aabb_to_polys(const AABB &p_aabb) const;

	// experimental
	bool cull_aabb_to_polys_ex(const AABB &p_aabb) const;
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

	bool calculate_poly_goodness_of_fit(const VSOccluder_Poly &p_opoly, real_t &r_fit);
	void whittle_polys();
	void precalc_poly_edge_planes(const Vector3 &p_pt_camera);

	// If all the points of the poly are beyond one of the planes (e.g. frustum), it is completely culled.
	bool is_poly_culled(const Occlusion::PolyPlane &p_opoly, const LocalVector<Plane> &p_planes) const {
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
	struct PlaneSet;
	bool is_poly_inside_occlusion_volume(const Occlusion::Poly &p_test_poly, const Plane &p_occluder_plane, const PlaneSet &p_planeset) const {
		// first test against the occluder poly plane
		for (int n = 0; n < p_test_poly.num_verts; n++) {
			const Vector3 &pt = p_test_poly.verts[n];
			if (p_occluder_plane.is_point_over(pt)) {
				return false;
			}
		}

		for (int p = 0; p < p_planeset.num_planes; p++) {
			const Plane &plane = p_planeset.planes[p];

			for (int n = 0; n < p_test_poly.num_verts; n++) {
				const Vector3 &pt = p_test_poly.verts[n];
				if (plane.is_point_over(pt)) {
					return false;
				}
			}
		}
		return true;
	}

	bool is_poly_touching_hole(const Occlusion::Poly &p_opoly, const PlaneSet &p_planeset) const {
		if (!p_opoly.num_verts) {
			// should not happen?
			return false;
		}
		// find aabb
		AABB bb;
		bb.position = p_opoly.verts[0];
		for (int n = 1; n < p_opoly.num_verts; n++) {
			bb.expand_to(p_opoly.verts[n]);
		}

		// if the AABB is totally outside any edge, it is safe for a hit
		real_t omin, omax;

		for (int e = 0; e < p_planeset.num_planes; e++) {
			// edge plane to camera
			const Plane &plane = p_planeset.planes[e];
			bb.project_range_in_plane(plane, omin, omax);

			// if inside the hole, no longer a hit on this poly
			if (omin > 0.0) {
				return false;
			}
		} // for e

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
			SPF_HAS_HOLES = 8,
		};

		Occlusion::PolyPlane poly;
		uint32_t flags;
#ifdef TOOLS_ENABLED
		uint32_t poly_source_id;
#endif
		uint32_t mesh_source_id;
		real_t goodness_of_fit;
	};

	struct PlaneSet {
		void flip() {
			for (int n = 0; n < num_planes; n++) {
				planes[n] = -planes[n];
			}
		}
		// pre-calculated edge planes to the camera
		int num_planes = 0;
		Plane planes[PortalDefines::OCCLUSION_POLY_MAX_VERTS];
	};

	struct PreCalcedPoly {
		void flip() {
			edge_planes.flip();
			for (int n = 0; n < num_holes; n++) {
				hole_edge_planes[n].flip();
			}
		}
		int num_holes = 0;
		PlaneSet edge_planes;
		PlaneSet hole_edge_planes[PortalDefines::OCCLUSION_POLY_MAX_HOLES];
		Occlusion::Poly hole_polys[PortalDefines::OCCLUSION_POLY_MAX_HOLES];
	};

	SortPoly _polys[MAX_POLYS];
	PreCalcedPoly _precalced_poly[MAX_POLYS];
	int _num_polys = 0;
	int _max_polys = 8;

#ifdef TOOLS_ENABLED
	uint32_t _poly_checksum = 0;
#endif

	Vector3 _pt_camera;
	Vector3 _pt_cam_dir;

	CameraMatrix _matrix_camera;
	PortalRenderer *_portal_renderer = nullptr;

	Clipper _clipper;

	bool _occluders_present = false;

	static bool _debug_log;
};

#endif // PORTAL_OCCLUSION_CULLER_H
