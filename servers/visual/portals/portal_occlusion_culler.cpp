/*************************************************************************/
/*  portal_occlusion_culler.cpp                                          */
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

#include "portal_occlusion_culler.h"

#include "core/engine.h"
#include "core/math/aabb.h"
#include "core/project_settings.h"
#include "portal_renderer.h"

#define _log(a, b) ;
//#define _log_prepare(a) log(a, 0)
#define _log_prepare(a) ;

bool PortalOcclusionCuller::_debug_log = true;
bool PortalOcclusionCuller::_redraw_gizmo = false;

void PortalOcclusionCuller::Clipper::debug_print_points(String p_string) {
	print_line(p_string);
	for (int n = 0; n < _pts_in.size(); n++) {
		print_line("\t" + itos(n) + " : " + String(Variant(_pts_in[n])));
	}
}

Plane PortalOcclusionCuller::Clipper::interpolate(const Plane &p_a, const Plane &p_b, real_t p_t) const {
	Vector3 diff = p_b.normal - p_a.normal;
	real_t d = p_b.d - p_a.d;

	diff *= p_t;
	d *= p_t;

	return Plane(p_a.normal + diff, p_a.d + d);
}

real_t PortalOcclusionCuller::Clipper::clip_and_find_poly_area(bool p_debug, const Plane *p_verts, int p_num_verts) {
	_pts_in.clear();
	_pts_out.clear();

	// seed
	for (int n = 0; n < p_num_verts; n++) {
		_pts_in.push_back(p_verts[n]);
	}

	if (p_debug) {
		print_line("clip_and_find_poly_area");
	}

	if (p_debug) {
		debug_print_points("BEFORE");
	}

	if (!clip_to_plane(-1, 0, 0, 1)) {
		return 0.0;
	}
	if (p_debug) {
		debug_print_points("LEFT");
	}
	if (!clip_to_plane(1, 0, 0, 1)) {
		return 0.0;
	}
	if (p_debug) {
		debug_print_points("RIGHT");
	}
	if (!clip_to_plane(0, -1, 0, 1)) {
		return 0.0;
	}
	if (p_debug) {
		debug_print_points("TOP");
	}
	if (!clip_to_plane(0, 1, 0, 1)) {
		return 0.0;
	}
	if (p_debug) {
		debug_print_points("BOTTOM");
	}
	if (!clip_to_plane(0, 0, -1, 1)) {
		return 0.0;
	}
	if (p_debug) {
		debug_print_points("NEAR");
	}
	if (!clip_to_plane(0, 0, 1, 1)) {
		return 0.0;
	}
	if (p_debug) {
		debug_print_points("FAR");
	}

	// perspective divide
	_pts_final.resize(_pts_in.size());
	for (int n = 0; n < _pts_in.size(); n++) {
		_pts_final[n] = _pts_in[n].normal / _pts_in[n].d;
	}

	real_t area = Geometry::find_polygon_area(&_pts_final[0], _pts_final.size());
	if (p_debug) {
		print_line("area : " + rtos(area));
	}
	return area;
}

bool PortalOcclusionCuller::Clipper::is_inside(const Plane &p_pt, Boundary p_boundary) {
	real_t w = p_pt.d;

	switch (p_boundary) {
		case B_LEFT: {
			return p_pt.normal.x > -w;
		} break;
		case B_RIGHT: {
			return p_pt.normal.x < w;
		} break;
		case B_TOP: {
			return p_pt.normal.y < w;
		} break;
		case B_BOTTOM: {
			return p_pt.normal.y > -w;
		} break;
		case B_NEAR: {
			return p_pt.normal.z < w;
		} break;
		case B_FAR: {
			return p_pt.normal.z > -w;
		} break;
		default:
			break;
	}

	return false;
}

// a is out, b is in
Plane PortalOcclusionCuller::Clipper::intersect(const Plane &p_a, const Plane &p_b, Boundary p_boundary) {
	Plane diff_plane(p_b.normal - p_a.normal, p_b.d - p_a.d);
	const Vector3 &diff = diff_plane.normal;

	real_t t = 0.0;
	const real_t epsilon = 0.001;

	// prevent divide by zero
	switch (p_boundary) {
		case B_LEFT: {
			if (diff.x > epsilon) {
				t = (-1.0 - p_a.normal.x) / diff.x;
			}
		} break;
		case B_RIGHT: {
			if (-diff.x > epsilon) {
				t = (p_a.normal.x - 1.0) / -diff.x;
			}
		} break;
		case B_TOP: {
			if (-diff.y > epsilon) {
				t = (p_a.normal.y - 1.0) / -diff.y;
			}
		} break;
		case B_BOTTOM: {
			if (diff.y > epsilon) {
				t = (-1.0 - p_a.normal.y) / diff.y;
			}
		} break;
		case B_NEAR: {
			if (-diff.z > epsilon) {
				t = (p_a.normal.z - 1.0) / -diff.z;
			}
		} break;
		case B_FAR: {
			if (diff.z > epsilon) {
				t = (-1.0 - p_a.normal.z) / diff.z;
			}
		} break;
		default:
			break;
	}

	diff_plane.normal *= t;
	diff_plane.d *= t;
	return Plane(p_a.normal + diff_plane.normal, p_a.d + diff_plane.d);
}

// Clip the poly to the plane given by the formula a * x + b * y + c * z + d * w.
bool PortalOcclusionCuller::Clipper::clip_to_plane(real_t a, real_t b, real_t c, real_t d) {
	_pts_out.clear();

	// repeat the first
	_pts_in.push_back(_pts_in[0]);

	Plane vPrev = _pts_in[0];
	real_t dpPrev = a * vPrev.normal.x + b * vPrev.normal.y + c * vPrev.normal.z + d * vPrev.d;

	for (int i = 1; i < _pts_in.size(); ++i) {
		Plane v = _pts_in[i];
		real_t dp = a * v.normal.x + b * v.normal.y + c * v.normal.z + d * v.d;

		if (dpPrev >= 0) {
			_pts_out.push_back(vPrev);
		}

		if (sgn(dp) != sgn(dpPrev)) {
			real_t t = dp < 0 ? dpPrev / (dpPrev - dp) : -dpPrev / (dp - dpPrev);

			Plane vOut = interpolate(vPrev, v, t);
			_pts_out.push_back(vOut);
		}

		vPrev = v;
		dpPrev = dp;
	}

	// start again from the output points next time
	_pts_in = _pts_out;

	return _pts_in.size() > 2;
}

Geometry::MeshData PortalOcclusionCuller::debug_get_current_polys() const {
	Geometry::MeshData md;

	for (int n = 0; n < _num_polys; n++) {
		const Occlusion::Poly &p = _polys[n].poly;

		int first_index = md.vertices.size();

		Vector3 normal_push = p.plane.normal * 0.001;

		// copy verts
		for (int c = 0; c < p.num_verts; c++) {
			md.vertices.push_back(p.verts[c] + normal_push);
		}

		// indices
		Geometry::MeshData::Face face;

		// triangle fan
		face.indices.resize(p.num_verts);

		for (int c = 0; c < p.num_verts; c++) {
			face.indices.set(c, first_index + c);
		}

		md.faces.push_back(face);
	}

	return md;
}

void PortalOcclusionCuller::prepare_generic(PortalRenderer &p_portal_renderer, const LocalVector<uint32_t, uint32_t> &p_occluder_pool_ids, const Vector3 &pt_camera, const LocalVector<Plane> &p_planes) {
	// bodge to keep settings up to date
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && ((Engine::get_singleton()->get_frames_drawn() % 16) == 0)) {
		_max_polys = GLOBAL_GET("rendering/misc/occlusion_culling/max_active_polys");
	}
#endif

	_pt_camera = pt_camera;

	// spheres
	_num_spheres = 0;
	real_t goodness_of_fit_sphere[MAX_SPHERES];
	for (int n = 0; n < _max_spheres; n++) {
		goodness_of_fit_sphere[n] = 0.0;
	}
	real_t weakest_fit_sphere = FLT_MAX;
	int weakest_sphere = 0;
	_sphere_closest_dist = FLT_MAX;

	// polys
	_num_polys = 0;
	for (int n = 0; n < _max_polys; n++) {
		_polys[n].goodness_of_fit = 0.0;
	}
	real_t weakest_fit_poly = FLT_MAX;
	int weakest_poly_id = 0;

	// find occluders
	for (unsigned int o = 0; o < p_occluder_pool_ids.size(); o++) {
		int id = p_occluder_pool_ids[o];
		VSOccluder &occ = p_portal_renderer.get_pool_occluder(id);

		// is it active?
		// in the case of rooms, they will always be active, as inactive
		// are removed from rooms. But for whole scene mode, some may be inactive.
		if (!occ.active) {
			continue;
		}

		// TODO : occlusion cull spheres AGAINST themselves.
		// i.e. a sphere that is occluded by another occluder is no
		// use as an occluder...
		if (occ.type == VSOccluder::OT_SPHERE) {
			// make sure world space spheres are up to date
			p_portal_renderer.occluder_ensure_up_to_date_sphere(occ);

			// multiple spheres
			for (int n = 0; n < occ.list_ids.size(); n++) {
				const Occlusion::Sphere &occluder_sphere = p_portal_renderer.get_pool_occluder_sphere(occ.list_ids[n]).world;

				// is the occluder sphere culled?
				if (is_sphere_culled(occluder_sphere.pos, occluder_sphere.radius, p_planes)) {
					continue;
				}

				real_t dist = (occluder_sphere.pos - pt_camera).length();

				// keep a record of the closest sphere for quick rejects
				if (dist < _sphere_closest_dist) {
					_sphere_closest_dist = dist;
				}

				// calculate the goodness of fit .. smaller distance better, and larger radius
				// calculate adjusted radius at 100.0
				real_t fit = 100 / MAX(dist, 0.01);
				fit *= occluder_sphere.radius;

				// until we reach the max, just keep recording, and keep track
				// of the worst fit
				if (_num_spheres < _max_spheres) {
					_spheres[_num_spheres] = occluder_sphere;
					_sphere_distances[_num_spheres] = dist;
					goodness_of_fit_sphere[_num_spheres] = fit;

					if (fit < weakest_fit_sphere) {
						weakest_fit_sphere = fit;
						weakest_sphere = _num_spheres;
					}

					_num_spheres++;
				} else {
					// must beat the weakest
					if (fit > weakest_fit_sphere) {
						_spheres[weakest_sphere] = occluder_sphere;
						_sphere_distances[weakest_sphere] = dist;
						goodness_of_fit_sphere[weakest_sphere] = fit;

						// the weakest may have changed (this could be done more efficiently)
						weakest_fit_sphere = FLT_MAX;
						for (int s = 0; s < _max_spheres; s++) {
							if (goodness_of_fit_sphere[s] < weakest_fit_sphere) {
								weakest_fit_sphere = goodness_of_fit_sphere[s];
								weakest_sphere = s;
							}
						}
					}
				}
			}
		} // sphere

		if (occ.type == VSOccluder::OT_MESH) {
			// make sure world space spheres are up to date
			p_portal_renderer.occluder_ensure_up_to_date_polys(occ);

			// multiple polys
			for (int n = 0; n < occ.list_ids.size(); n++) {
				const VSOccluder_Mesh &opoly = p_portal_renderer.get_pool_occluder_mesh(occ.list_ids[n]);
				const Occlusion::Poly &poly = opoly.poly_world;

				// backface cull
				bool faces_camera = poly.plane.is_point_over(pt_camera);

				if (!faces_camera) {
					continue;
				}

				// try culling by center
				//				Vector3 offset = opoly.center - pt_camera;
				//				if (_pt_cam_dir.dot(offset) < 0.0)
				//					continue;

				// cull behind camera (this is rough, will cull some unnecessarily
				// that are parallel, but these polys won't be much use for occlusion anyway)
				//				real_t dot = _pt_cam_dir.dot(poly.plane.normal);
				//				if (!faces_camera) {
				//					dot = -dot;
				//				}
				//				if (dot >= 0.0) {
				//					continue;
				//				}

				//				if (n != 158)
				//					continue;

				bool debug = false;
				//				if ((Engine::get_singleton()->get_frames_drawn() % 8) == 0) {
				//					if (n == 158) {
				//						//debug = true;
				//					}
				//				}

				real_t fit;
				if (!calculate_poly_goodness_of_fit(debug, opoly, fit)) {
					continue;
				}

				if (_num_polys < _max_polys) {
					SortPoly &dest = _polys[_num_polys];
					dest.poly = poly;
					dest.flags = faces_camera ? SortPoly::SPF_FACES_CAMERA : 0;
					dest.poly_source_id = n;
					dest.goodness_of_fit = fit;

					if (fit < weakest_fit_poly) {
						weakest_fit_poly = fit;
						weakest_poly_id = _num_polys;
					}

					_num_polys++;
				} else {
					// must beat the weakest
					if (fit > weakest_fit_poly) {
						SortPoly &dest = _polys[weakest_poly_id];
						dest.poly = poly;
						//dest.faces_camera = faces_camera;
						dest.flags = faces_camera ? SortPoly::SPF_FACES_CAMERA : 0;
						dest.poly_source_id = n;
						dest.goodness_of_fit = fit;

						// the weakest may have changed (this could be done more efficiently)
						weakest_fit_poly = FLT_MAX;
						for (int p = 0; p < _max_polys; p++) {
							real_t goodness_of_fit = _polys[p].goodness_of_fit;

							if (goodness_of_fit < weakest_fit_poly) {
								weakest_fit_poly = goodness_of_fit;
								weakest_poly_id = p;
							}
						}
					}

				} // polys full up, replace
			}
		}
	} // for o

	// flip polys so always facing camera

	for (int n = 0; n < _num_polys; n++) {
		if (!(_polys[n].flags & SortPoly::SPF_FACES_CAMERA)) {
			_polys[n].poly.flip();
		}
	}

	// call polys against each other.
	whittle_polys();

	// checksum is used only in the editor, to decide
	// whether to redraw the gizmo of active polys
#ifdef TOOLS_ENABLED
	uint32_t last_checksum = _poly_checksum;
	_poly_checksum = 0;
	for (int n = 0; n < _num_polys; n++) {
		_poly_checksum += _polys[n].poly_source_id;
		//_log_prepare("prepfinal : " + itos(_polys[n].poly_source_id) + " fit : " + rtos(_polys[n].goodness_of_fit));
	}
	if (_poly_checksum != last_checksum) {
		_redraw_gizmo = true;
	}
#endif

	//	if ((Engine::get_singleton()->get_frames_drawn() % 8) == 0) {
	//		for (int n = 0; n < _num_polys; n++) {
	//			print_line(itos(n) + "\tfit : " + rtos(goodness_of_fit_poly[n]) + ", source : " + itos(_poly_source_id[n]) + ", facescam : " + String(Variant(_poly_faces_camera[n])) + ", plane : " + String(Variant(_polys[n].plane)));
	//		}
	//	}

	// force the sphere closest distance to above zero to prevent
	// divide by zero in the quick reject
	_sphere_closest_dist = MAX(_sphere_closest_dist, 0.001);

	// record whether to do any occlusion culling at all..
	_occluders_present = _num_spheres || _num_polys;
}

void PortalOcclusionCuller::whittle_polys() {
//#define GODOT_OCCLUSION_FLASH_POLYS
#ifdef GODOT_OCCLUSION_FLASH_POLYS
	if (((Engine::get_singleton()->get_frames_drawn() / 4) % 2) == 0) {
		return;
	}
#endif

	LocalVector<Plane> planes;

	bool repeat = true;

	while (repeat) {
		repeat = false;
		// Check for complete occlusion of polys by a closer poly.
		// Such polys can be completely removed from checks.
		for (int n = 0; n < _num_polys; n++) {
			// ensure we test each occluder once and only once
			// (as this routine will repeat each time an occluded poly is found)
			SortPoly &sort_poly = _polys[n];
			if (!(sort_poly.flags & SortPoly::SPF_TESTED_AS_OCCLUDER)) {
				sort_poly.flags |= SortPoly::SPF_TESTED_AS_OCCLUDER;
			} else {
				continue;
			}

			const Occlusion::Poly &poly = _polys[n].poly;
			planes.clear();

			// the goodness of fit is the screen space area at the moment,
			// so we can use it as a quick reject .. polys behind occluders will always
			// be smaller area than the occluder.
			real_t occluder_area = _polys[n].goodness_of_fit;

			// add a plane for the occluder poly itself, anything occluded must be BEHIND it
			planes.push_back(poly.plane);

			// construct planes from camera to this occluder
			for (int e = 0; e < poly.num_verts; e++) {
				// point a and b of the edge
				const Vector3 &pt_a = poly.verts[e];
				const Vector3 &pt_b = poly.verts[(e + 1) % poly.num_verts];

				// edge plane to camera
				Plane plane = Plane(_pt_camera, pt_a, pt_b);
				planes.push_back(plane);

			} // for e through edges

			// check each other poly as an occludee
			for (int t = 0; t < _num_polys; t++) {
				if (n == t) {
					continue;
				}

				// quick reject based on screen space area.
				// if the area of the test poly is larger, it can't be completely behind
				// the occluder.
				if (_polys[t].goodness_of_fit > occluder_area) {
					continue;
				}

				const Occlusion::Poly &test_poly = _polys[t].poly;

				if (is_poly_inside_occlusion_volume(test_poly, planes)) {
					// yes .. we can remove this poly .. but do not muck up the iteration of the list
					//print_line("poly is occluded " + itos(t));

					// this condition should never happen, we should never be checking occludee against itself
					DEV_ASSERT(_polys[t].poly_source_id != _polys[n].poly_source_id);

					// unordered remove
					_polys[t] = _polys[_num_polys - 1];
					_num_polys--;

					// no NOT repeat the test poly if it was copied from n, i.e. the occludee would
					// be the same as the occluder
					if (_num_polys != n) {
						// repeat this test poly as it will be the next
						t--;
					}

					// If we end up removing a poly BEFORE n, the replacement poly (from the unordered remove)
					// will never get tested as an occluder. So we have to account for this by rerunning the routine.
					repeat = true;
				}
			} // for t through occludees

		} // for n through occluders

	} // while repeat

	// order polys by distance to camera / area? NYI
}

bool PortalOcclusionCuller::calculate_poly_goodness_of_fit(bool debug, const VSOccluder_Mesh &p_opoly, real_t &r_fit) {
	// transform each of the poly points, find the area in screen space

	// The points must be homogeneous coordinates, i.e. BEFORE
	// the perspective divide, in clip space. They will have the perspective
	// divide applied after clipping, to calculate the area.
	// We therefore store them as planes to store the w coordinate as d.
	Plane xpoints[Occlusion::Poly::MAX_POLY_VERTS];
	int num_verts = p_opoly.poly_world.num_verts;

	//	real_t z_min = 1.0;
	//	real_t z_max = 0.0;

	for (int n = 0; n < num_verts; n++) {
		// source and dest in homogeneous coords
		Plane source(p_opoly.poly_world.verts[n], 1.0);
		Plane &dest = xpoints[n];

		dest = _matrix_camera.xform4(source);

		if (debug) {
			//			print_line("\t" + itos(n) + " : "	+ String(Variant(sep_src)) + " xform: " + String(Variant(sep_xform)) + " proj: " + String(Variant(sep_proj)) + " proj_plane: " + String(Variant(sep_proj_plane)));
			//_log_prepare(itos(n) + " : " + String(Variant(dest)));
			//print_line(itos(n) + " : " + String(Variant(dest)));
		}

		/*
		if (dest.z > 1.0) {
			off_near++;
			z_max = 1.0;
		} else {
			z_min = MIN(z_min, dest.z);
			z_max = MAX(z_max, dest.z);
		}
*/
		// make 2d
		//dest.z = 0.0;
	}

	//	if (off_near == num_verts)
	//		return false;

	// find screen space area
	real_t area = _clipper.clip_and_find_poly_area(debug, xpoints, num_verts);
	if (area <= 0.0) {
		return false;
	}

	if (debug) {
		_log_prepare("Area : " + rtos(area));
	}
	//	if (area < 0.01) {
	//		return false;
	//	}
	r_fit = area;

	/*
	// apply an effect for bias to closer polys
	real_t z_average = (z_min + z_max) * 0.5;
	z_average = 1.0 - z_average;
	//	z_average -= 0.98;
	z_average *= 50.0;

	z_average *= z_average;
	//z_average *= z_average;

	z_average = MIN(1.0, z_average);
	z_average = MAX(0.01, z_average);

	r_fit *= z_average;
*/
	return true;
}

bool PortalOcclusionCuller::_is_poly_of_interest_to_split_plane(const Plane *p_poly_split_plane, int p_poly_id) const {
	const Occlusion::Poly &poly = _polys[p_poly_id].poly;

	int over = 0;
	int under = 0;

	// we need an epsilon because adjacent polys that just
	// join with a wall may have small floating point error ahead
	// of the splitting plane.
	const real_t epsilon = 0.005;

	for (int n = 0; n < poly.num_verts; n++) {
		// point a and b of the edge
		const Vector3 &pt = poly.verts[n];

		real_t dist = p_poly_split_plane->distance_to(pt);
		if (dist > epsilon) {
			over++;
		} else {
			under++;
		}
	}

	// return whether straddles the plane
	return over && under;
}

bool PortalOcclusionCuller::cull_aabb_to_polys_ex(int p_ignore_poly_id, const AABB &p_aabb, const Plane *p_poly_split_plane, int p_depth) const {
	_debug_log = false;
	if ((Engine::get_singleton()->get_frames_drawn() % 8) == 0) {
		_debug_log = true;
	}

	_log("\n", 0);
	_log("* cull_aabb_to_polys_ex " + String(Variant(p_aabb)) + " ignore id " + itos(p_ignore_poly_id), p_depth);

	Plane plane;

	for (int n = 0; n < _num_polys; n++) {
		if (n == p_ignore_poly_id) {
			continue;
		}

		//		if (_polys[n].done) {
		//			continue;
		//		}

		_log("\tchecking poly " + itos(n), p_depth);

		const Occlusion::Poly &poly = _polys[n].poly;

		// we allow exactly one crossing
		// (in order to allow an aabb to be shared between 2 polys)
		int crossed_edge_p1 = 0;

		// test against each edge of the poly, and expand the edge
		bool hit = true;

		// occludee must be on opposite side to camera
		real_t omin, omax;
		p_aabb.project_range_in_plane(poly.plane, omin, omax);

		if (omax > -0.2) {
			_log("\t\tAABB is in front of occluder, ignoring", p_depth);
			continue;
		}

		// if we have a split plane, this poly to be tested must straddle it to be of interest
		if (p_poly_split_plane && !_is_poly_of_interest_to_split_plane(p_poly_split_plane, n)) {
			continue;
		}

		for (int e = 0; e < poly.num_verts; e++) {
			// point a and b of the edge
			const Vector3 &pt_a = poly.verts[e];
			const Vector3 &pt_b = poly.verts[(e + 1) % poly.num_verts];

			// edge plane to camera
			plane = Plane(_pt_camera, pt_a, pt_b);
			p_aabb.project_range_in_plane(plane, omin, omax);

			if (omax > 0.0) {
				_log("\tover edge " + itos(e), p_depth);
				// is this edge allowed by a recursion?
				if (p_poly_split_plane) {
					real_t over_a = p_poly_split_plane->distance_to(pt_a);
					real_t over_b = p_poly_split_plane->distance_to(pt_b);
					const real_t edge_epsilon = 0.01;

					if ((over_a < edge_epsilon) && (over_b < edge_epsilon)) {
						//if (!p_poly_split_plane->is_point_over(pt_a) && !p_poly_split_plane->is_point_over(pt_b)) {
						// discount this edge
						_log("\t\tsplit plane - edge was over, ALLOWED", p_depth);
					} else {
						_log("\t\tsplit plane - edge not over, DISALLOWED", p_depth);
						hit = false;
						break;
					}
				} else {
					if (!crossed_edge_p1 && (omin < 0.0)) {
						_log("\t\tcrossing edge " + itos(e), p_depth);
						crossed_edge_p1 = e + 1;
					} else {
						_log("\t\tcrossing second edge or outside NO HIT", p_depth);
						hit = false;
						break;
					}
				} // can't discount this edge crossing
			}
		}

		// hit?

		if (hit) {
			if (!crossed_edge_p1) {
				_log("\t\tHIT", p_depth);
				return true;
			} else {
				// we crossed an edge
				int e = crossed_edge_p1 - 1;
				plane = Plane(_pt_camera, poly.verts[e], poly.verts[(e + 1) % poly.num_verts]);

				// call recursively, but with an edge to check
				if (cull_aabb_to_polys_ex(n, p_aabb, &plane, p_depth + 1)) {
					_log("\tRecursive cull_aabb_to_polys_ex returned HIT", p_depth);
					return true;
				}
			}
		}
	}

	_log("\tno hit", p_depth);
	return false;
}

bool PortalOcclusionCuller::cull_aabb_to_polys(const AABB &p_aabb) const {
	if (!_num_polys) {
		return false;
	}

	return cull_aabb_to_polys_ex(-1, p_aabb, nullptr);

	Plane plane;

	for (int n = 0; n < _num_polys; n++) {
		const Occlusion::Poly &poly = _polys[n].poly;

		// test against each edge of the poly, and expand the edge
		bool hit = true;

		// occludee must be on opposite side to camera
		real_t omin, omax;
		p_aabb.project_range_in_plane(poly.plane, omin, omax);

		if (omax > -0.2) {
			continue;
		}

		for (int e = 0; e < poly.num_verts; e++) {
			plane = Plane(_pt_camera, poly.verts[e], poly.verts[(e + 1) % poly.num_verts]);
			p_aabb.project_range_in_plane(plane, omin, omax);

			if (omax > 0.0) {
				hit = false;
				break;
			}
		}

		// hit?
		if (hit) {
			return true;
		}
	}

	return false;
}

bool PortalOcclusionCuller::cull_sphere_to_polys(const Vector3 &p_occludee_center, real_t p_occludee_radius) const {
	if (!_num_polys) {
		return false;
	}

	Plane plane;

	for (int n = 0; n < _num_polys; n++) {
		const Occlusion::Poly &poly = _polys[n].poly;

		// test against each edge of the poly, and expand the edge
		bool hit = true;

		// occludee must be on opposite side to camera
		real_t dist = poly.plane.distance_to(p_occludee_center);

		if (dist > -p_occludee_radius) {
			continue;
		}

		for (int e = 0; e < poly.num_verts; e++) {
			plane = Plane(_pt_camera, poly.verts[e], poly.verts[(e + 1) % poly.num_verts]);

			// de-expand
			plane.d -= p_occludee_radius;

			if (plane.is_point_over(p_occludee_center)) {
				hit = false;
				break;
			}
		}

		// hit?
		if (hit) {
			return true;
		}
	}

	return false;
}

bool PortalOcclusionCuller::cull_sphere_to_spheres(const Vector3 &p_occludee_center, real_t p_occludee_radius, const Vector3 &p_ray_dir, real_t p_dist_to_occludee) const {
	// maybe not required
	if (!_num_spheres) {
		return false;
	}

	// prevent divide by zero, and the occludee cannot be occluded if we are WITHIN
	// its bounding sphere... so no need to check
	if (p_dist_to_occludee < _sphere_closest_dist) {
		return false;
	}

	// this can probably be done cheaper with dot products but the math might be a bit fiddly to get right
	for (int s = 0; s < _num_spheres; s++) {
		//  first get the sphere distance
		real_t occluder_dist_to_cam = _sphere_distances[s];
		if (p_dist_to_occludee < occluder_dist_to_cam) {
			// can't occlude
			continue;
		}

		// the perspective adjusted occludee radius
		real_t adjusted_occludee_radius = p_occludee_radius * (occluder_dist_to_cam / p_dist_to_occludee);

		const Occlusion::Sphere &occluder_sphere = _spheres[s];
		real_t occluder_radius = occluder_sphere.radius - adjusted_occludee_radius;

		if (occluder_radius > 0.0) {
			occluder_radius = occluder_radius * occluder_radius;

			// distance to hit
			real_t dist;

			if (occluder_sphere.intersect_ray(_pt_camera, p_ray_dir, dist, occluder_radius)) {
				if (dist < p_dist_to_occludee) {
					// occluded
					return true;
				}
			}
		} // expanded occluder radius is more than 0
	}

	return false;
}

bool PortalOcclusionCuller::cull_sphere(const Vector3 &p_occludee_center, real_t p_occludee_radius, bool p_cull_to_polys) const {
	if (!_occluders_present) {
		return false;
	}

	// ray from origin to the occludee
	Vector3 ray_dir = p_occludee_center - _pt_camera;
	real_t dist_to_occludee_raw = ray_dir.length();

	// account for occludee radius
	real_t dist_to_occludee = dist_to_occludee_raw - p_occludee_radius;

	// ignore occlusion for closeup, and avoid divide by zero
	if (dist_to_occludee_raw < 0.1) {
		return false;
	}

	// normalize ray
	// hopefully by this point, dist_to_occludee_raw cannot possibly be zero due to above check
	ray_dir *= 1.0 / dist_to_occludee_raw;

	if (cull_sphere_to_spheres(p_occludee_center, p_occludee_radius, ray_dir, dist_to_occludee)) {
		return true;
	}

	if (p_cull_to_polys && cull_sphere_to_polys(p_occludee_center, p_occludee_radius)) {
		return true;
	}

	return false;
}

PortalOcclusionCuller::PortalOcclusionCuller() {
	_max_spheres = GLOBAL_GET("rendering/misc/occlusion_culling/max_active_spheres");
	_max_polys = GLOBAL_GET("rendering/misc/occlusion_culling/max_active_polys");
}

void PortalOcclusionCuller::log(String p_string, int p_depth) const {
	if (_debug_log) {
		for (int n = 0; n < p_depth; n++) {
			p_string = "\t\t\t" + p_string;
		}
		print_line(p_string);
	}
}

#undef _log
#undef _log_prepare
