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

#include "core/engine.h"
#include "core/math/aabb.h"
#include "core/project_settings.h"
#include "portal_renderer.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual/visual_server_scene.h"

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

real_t PortalOcclusionCuller::Clipper::clip_and_find_poly_area(const Plane *p_verts, int p_num_verts) {
	_pts_in.clear();
	_pts_out.clear();

	// seed
	for (int n = 0; n < p_num_verts; n++) {
		_pts_in.push_back(p_verts[n]);
	}

	if (!clip_to_plane(-1, 0, 0, 1)) {
		return 0.0;
	}
	if (!clip_to_plane(1, 0, 0, 1)) {
		return 0.0;
	}
	if (!clip_to_plane(0, -1, 0, 1)) {
		return 0.0;
	}
	if (!clip_to_plane(0, 1, 0, 1)) {
		return 0.0;
	}
	if (!clip_to_plane(0, 0, -1, 1)) {
		return 0.0;
	}
	if (!clip_to_plane(0, 0, 1, 1)) {
		return 0.0;
	}

	// perspective divide
	_pts_final.resize(_pts_in.size());
	for (int n = 0; n < _pts_in.size(); n++) {
		_pts_final[n] = _pts_in[n].normal / _pts_in[n].d;
	}

	return Geometry::find_polygon_area(&_pts_final[0], _pts_final.size());
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
	const real_t epsilon = 0.001f;

	// prevent divide by zero
	switch (p_boundary) {
		case B_LEFT: {
			if (diff.x > epsilon) {
				t = (-1.0f - p_a.normal.x) / diff.x;
			}
		} break;
		case B_RIGHT: {
			if (-diff.x > epsilon) {
				t = (p_a.normal.x - 1.0f) / -diff.x;
			}
		} break;
		case B_TOP: {
			if (-diff.y > epsilon) {
				t = (p_a.normal.y - 1.0f) / -diff.y;
			}
		} break;
		case B_BOTTOM: {
			if (diff.y > epsilon) {
				t = (-1.0f - p_a.normal.y) / diff.y;
			}
		} break;
		case B_NEAR: {
			if (-diff.z > epsilon) {
				t = (p_a.normal.z - 1.0f) / -diff.z;
			}
		} break;
		case B_FAR: {
			if (diff.z > epsilon) {
				t = (-1.0f - p_a.normal.z) / diff.z;
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
		const Occlusion::PolyPlane &p = _polys[n].poly;

		int first_index = md.vertices.size();

		Vector3 normal_push = p.plane.normal * 0.001f;

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
	_portal_renderer = &p_portal_renderer;

	// Bodge to keep settings up to date, until the project settings PR is merged
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && ((Engine::get_singleton()->get_frames_drawn() % 16) == 0)) {
		_max_polys = GLOBAL_GET("rendering/misc/occlusion_culling/max_active_polygons");
	}
#endif
	_num_spheres = 0;

	_pt_camera = pt_camera;

	// spheres
	_num_spheres = 0;
	real_t goodness_of_fit_sphere[MAX_SPHERES];
	for (int n = 0; n < _max_spheres; n++) {
		goodness_of_fit_sphere[n] = 0.0f;
	}
	real_t weakest_fit_sphere = FLT_MAX;
	int weakest_sphere = 0;
	_sphere_closest_dist = FLT_MAX;

	// polys
	_num_polys = 0;
	for (int n = 0; n < _max_polys; n++) {
		_polys[n].goodness_of_fit = 0.0f;
	}
	real_t weakest_fit_poly = FLT_MAX;
	int weakest_poly_id = 0;

#ifdef TOOLS_ENABLED
	uint32_t polycount = 0;
#endif

	const PortalResources &resources = VSG::scene->get_portal_resources();

	// find occluders
	for (unsigned int o = 0; o < p_occluder_pool_ids.size(); o++) {
		int id = p_occluder_pool_ids[o];
		VSOccluder_Instance &occ = p_portal_renderer.get_pool_occluder_instance(id);

		// is it active?
		// in the case of rooms, they will always be active, as inactive
		// are removed from rooms. But for whole scene mode, some may be inactive.
		if (!occ.active) {
			continue;
		}

		// TODO : occlusion cull spheres AGAINST themselves.
		// i.e. a sphere that is occluded by another occluder is no
		// use as an occluder...
		if (occ.type == VSOccluder_Instance::OT_SPHERE) {
			// make sure world space spheres are up to date
			p_portal_renderer.occluder_ensure_up_to_date_sphere(resources, occ);

			// cull entire AABB
			if (is_aabb_culled(occ.aabb, p_planes)) {
				continue;
			}

			// multiple spheres
			for (int n = 0; n < occ.list_ids.size(); n++) {
				const Occlusion::Sphere &occluder_sphere = p_portal_renderer.get_pool_occluder_world_sphere(occ.list_ids[n]);

				// is the occluder sphere culled?
				if (is_sphere_culled(occluder_sphere.pos, occluder_sphere.radius, p_planes)) {
					continue;
				}

				real_t dist = (occluder_sphere.pos - pt_camera).length();

				// calculate the goodness of fit .. smaller distance better, and larger radius
				// calculate adjusted radius at 100.0
				real_t fit = 100 / MAX(dist, 0.01f);
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

					// keep a record of the closest sphere for quick rejects
					if (dist < _sphere_closest_dist) {
						_sphere_closest_dist = dist;
					}

					_num_spheres++;
				} else {
					// must beat the weakest
					if (fit > weakest_fit_sphere) {
						_spheres[weakest_sphere] = occluder_sphere;
						_sphere_distances[weakest_sphere] = dist;
						goodness_of_fit_sphere[weakest_sphere] = fit;

						// keep a record of the closest sphere for quick rejects
						if (dist < _sphere_closest_dist) {
							_sphere_closest_dist = dist;
						}

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

		if (occ.type == VSOccluder_Instance::OT_MESH) {
			// make sure world space spheres are up to date
			p_portal_renderer.occluder_ensure_up_to_date_polys(resources, occ);

			// multiple polys
			for (int n = 0; n < occ.list_ids.size(); n++) {
				const VSOccluder_Poly &opoly = p_portal_renderer.get_pool_occluder_world_poly(occ.list_ids[n]);
				const Occlusion::PolyPlane &poly = opoly.poly;

				// backface cull
				bool faces_camera = poly.plane.is_point_over(pt_camera);

				if (!faces_camera && !opoly.two_way) {
					continue;
				}

				real_t fit;
				if (!calculate_poly_goodness_of_fit(opoly, fit)) {
					continue;
				}

				if (_num_polys < _max_polys) {
					SortPoly &dest = _polys[_num_polys];
					dest.poly = poly;
					dest.flags = faces_camera ? SortPoly::SPF_FACES_CAMERA : 0;
					if (opoly.num_holes) {
						dest.flags |= SortPoly::SPF_HAS_HOLES;
					}
#ifdef TOOLS_ENABLED
					dest.poly_source_id = polycount++;
#endif
					dest.mesh_source_id = occ.list_ids[n];
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
						if (opoly.num_holes) {
							dest.flags |= SortPoly::SPF_HAS_HOLES;
						}
#ifdef TOOLS_ENABLED
						dest.poly_source_id = polycount++;
#endif
						dest.mesh_source_id = occ.list_ids[n];
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

	precalc_poly_edge_planes(pt_camera);

	// flip polys so always facing camera
	for (int n = 0; n < _num_polys; n++) {
		if (!(_polys[n].flags & SortPoly::SPF_FACES_CAMERA)) {
			_polys[n].poly.flip();

			// must flip holes and planes too
			_precalced_poly[n].flip();
		}
	}

	// cull polys against each other.
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

	// record whether to do any occlusion culling at all..
	_occluders_present = _num_spheres || _num_polys;
}

void PortalOcclusionCuller::precalc_poly_edge_planes(const Vector3 &p_pt_camera) {
	for (int n = 0; n < _num_polys; n++) {
		const SortPoly &sortpoly = _polys[n];
		const Occlusion::PolyPlane &spoly = sortpoly.poly;

		PreCalcedPoly &dpoly = _precalced_poly[n];
		dpoly.edge_planes.num_planes = spoly.num_verts;

		for (int e = 0; e < spoly.num_verts; e++) {
			// point a and b of the edge
			const Vector3 &pt_a = spoly.verts[e];
			const Vector3 &pt_b = spoly.verts[(e + 1) % spoly.num_verts];

			// edge plane to camera
			dpoly.edge_planes.planes[e] = Plane(p_pt_camera, pt_a, pt_b);
		}

		dpoly.num_holes = 0;

		// holes
		if (sortpoly.flags & SortPoly::SPF_HAS_HOLES) {
			// get the mesh poly and the holes
			const VSOccluder_Poly &mesh = _portal_renderer->get_pool_occluder_world_poly(sortpoly.mesh_source_id);

			dpoly.num_holes = mesh.num_holes;

			for (int h = 0; h < mesh.num_holes; h++) {
				uint32_t hid = mesh.hole_pool_ids[h];
				const VSOccluder_Hole &hole = _portal_renderer->get_pool_occluder_world_hole(hid);

				// copy the verts to the precalced poly,
				// we will need these later for whittling polys.
				// We could alternatively link back to the original verts, but that gets messy.
				dpoly.hole_polys[h] = hole;

				int hole_num_verts = hole.num_verts;
				const Vector3 *hverts = hole.verts;

				// number of planes equals number of verts forming edges
				dpoly.hole_edge_planes[h].num_planes = hole_num_verts;

				for (int e = 0; e < hole_num_verts; e++) {
					const Vector3 &pt_a = hverts[e];
					const Vector3 &pt_b = hverts[(e + 1) % hole_num_verts];

					dpoly.hole_edge_planes[h].planes[e] = Plane(p_pt_camera, pt_a, pt_b);
				} // for e

			} // for h
		} // if has holes
	}
}

void PortalOcclusionCuller::whittle_polys() {
//#define GODOT_OCCLUSION_FLASH_POLYS
#ifdef GODOT_OCCLUSION_FLASH_POLYS
	if (((Engine::get_singleton()->get_frames_drawn() / 4) % 2) == 0) {
		return;
	}
#endif

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

			const Occlusion::PolyPlane &poly = _polys[n].poly;
			const Plane &occluder_plane = poly.plane;
			const PreCalcedPoly &pcp = _precalced_poly[n];

			// the goodness of fit is the screen space area at the moment,
			// so we can use it as a quick reject .. polys behind occluders will always
			// be smaller area than the occluder.
			real_t occluder_area = _polys[n].goodness_of_fit;

			// check each other poly as an occludee
			for (int t = 0; t < _num_polys; t++) {
				if (n == t) {
					continue;
				}

				// quick reject based on screen space area.
				// if the area of the test poly is larger, it can't be completely behind
				// the occluder.
				bool quick_reject_entire_occludee = _polys[t].goodness_of_fit > occluder_area;

				const Occlusion::PolyPlane &test_poly = _polys[t].poly;
				PreCalcedPoly &pcp_test = _precalced_poly[t];

				// We have two considerations:
				// (1) Entire poly is occluded
				// (2) If not (1), then maybe a hole is occluded

				bool completely_reject = false;

				if (!quick_reject_entire_occludee && is_poly_inside_occlusion_volume(test_poly, occluder_plane, pcp.edge_planes)) {
					completely_reject = true;

					// we must also test against all holes if some are present
					for (int h = 0; h < pcp.num_holes; h++) {
						if (is_poly_touching_hole(test_poly, pcp.hole_edge_planes[h])) {
							completely_reject = false;
							break;
						}
					}

					if (completely_reject) {
						// yes .. we can remove this poly .. but do not muck up the iteration of the list
						//print_line("poly is occluded " + itos(t));

#ifdef TOOLS_ENABLED
						// this condition should never happen, we should never be checking occludee against itself
						DEV_ASSERT(_polys[t].poly_source_id != _polys[n].poly_source_id);
#endif

						// unordered remove
						_polys[t] = _polys[_num_polys - 1];
						_precalced_poly[t] = _precalced_poly[_num_polys - 1];
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
					} // allow due to holes
				} // if poly inside occlusion volume

				// if we did not completely reject, there could be holes that could be rejected
				if (!completely_reject) {
					if (pcp_test.num_holes) {
						for (int h = 0; h < pcp_test.num_holes; h++) {
							const Occlusion::Poly &hole_poly = pcp_test.hole_polys[h];

							// is the hole within the occluder?
							if (is_poly_inside_occlusion_volume(hole_poly, occluder_plane, pcp.edge_planes)) {
								// if the hole touching a hole in the occluder? if so we can't eliminate it
								bool allow = true;

								for (int oh = 0; oh < pcp.num_holes; oh++) {
									if (is_poly_touching_hole(hole_poly, pcp.hole_edge_planes[oh])) {
										allow = false;
										break;
									}
								}

								if (allow) {
									// Unordered remove the hole. No need to repeat the whole while loop I don't think?
									// As this just makes it more efficient at runtime, it doesn't make the further whittling more accurate.
									pcp_test.num_holes--;
									pcp_test.hole_edge_planes[h] = pcp_test.hole_edge_planes[pcp_test.num_holes];
									pcp_test.hole_polys[h] = pcp_test.hole_polys[pcp_test.num_holes];

									h--; // repeat this as the unordered remove has placed a new member into h slot
								} // allow

							} // hole is within
						}
					} // has holes
				} // did not completely reject

			} // for t through occludees

		} // for n through occluders

	} // while repeat

	// order polys by distance to camera / area? NYI
}

bool PortalOcclusionCuller::calculate_poly_goodness_of_fit(const VSOccluder_Poly &p_opoly, real_t &r_fit) {
	// transform each of the poly points, find the area in screen space

	// The points must be homogeneous coordinates, i.e. BEFORE
	// the perspective divide, in clip space. They will have the perspective
	// divide applied after clipping, to calculate the area.
	// We therefore store them as planes to store the w coordinate as d.
	Plane xpoints[Occlusion::PolyPlane::MAX_POLY_VERTS];
	int num_verts = p_opoly.poly.num_verts;

	for (int n = 0; n < num_verts; n++) {
		// source and dest in homogeneous coords
		Plane source(p_opoly.poly.verts[n], 1.0f);
		Plane &dest = xpoints[n];

		dest = _matrix_camera.xform4(source);
	}

	// find screen space area
	real_t area = _clipper.clip_and_find_poly_area(xpoints, num_verts);
	if (area <= 0.0f) {
		return false;
	}

	r_fit = area;

	return true;
}

bool PortalOcclusionCuller::_is_poly_of_interest_to_split_plane(const Plane *p_poly_split_plane, int p_poly_id) const {
	const Occlusion::PolyPlane &poly = _polys[p_poly_id].poly;

	int over = 0;
	int under = 0;

	// we need an epsilon because adjacent polys that just
	// join with a wall may have small floating point error ahead
	// of the splitting plane.
	const real_t epsilon = 0.005f;

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

bool PortalOcclusionCuller::cull_aabb_to_polys_ex(const AABB &p_aabb) const {
	_log("\n", 0);
	_log("* cull_aabb_to_polys_ex " + String(Variant(p_aabb)), 0);

	Plane plane;

	for (int n = 0; n < _num_polys; n++) {
		_log("\tchecking poly " + itos(n), 0);

		const SortPoly &sortpoly = _polys[n];
		const Occlusion::PolyPlane &poly = sortpoly.poly;

		// occludee must be on opposite side to camera
		real_t omin, omax;
		p_aabb.project_range_in_plane(poly.plane, omin, omax);

		if (omax > -0.2f) {
			_log("\t\tAABB is in front of occluder, ignoring", 0);
			continue;
		}

		// test against each edge of the poly, and expand the edge
		bool hit = true;

		const PreCalcedPoly &pcp = _precalced_poly[n];

		for (int e = 0; e < pcp.edge_planes.num_planes; e++) {
			// edge plane to camera
			plane = pcp.edge_planes.planes[e];
			p_aabb.project_range_in_plane(plane, omin, omax);

			if (omax > 0.0f) {
				hit = false;
				break;
			}
		}

		// if it hit, check against holes
		if (hit && pcp.num_holes) {
			for (int h = 0; h < pcp.num_holes; h++) {
				const PlaneSet &hole = pcp.hole_edge_planes[h];

				// if the AABB is totally outside any edge, it is safe for a hit
				bool safe = false;
				for (int e = 0; e < hole.num_planes; e++) {
					// edge plane to camera
					plane = hole.planes[e];
					p_aabb.project_range_in_plane(plane, omin, omax);

					// if inside the hole, no longer a hit on this poly
					if (omin > 0.0f) {
						safe = true;
						break;
					}
				} // for e

				if (!safe) {
					hit = false;
				}

				if (!hit) {
					break;
				}
			} // for h
		} // if has holes

		// hit?

		if (hit) {
			return true;
		}
	}

	_log("\tno hit", 0);
	return false;
}

bool PortalOcclusionCuller::cull_aabb_to_polys(const AABB &p_aabb) const {
	if (!_num_polys) {
		return false;
	}

	return cull_aabb_to_polys_ex(p_aabb);
}

bool PortalOcclusionCuller::cull_sphere_to_polys(const Vector3 &p_occludee_center, real_t p_occludee_radius) const {
	if (!_num_polys) {
		return false;
	}

	Plane plane;

	for (int n = 0; n < _num_polys; n++) {
		const Occlusion::PolyPlane &poly = _polys[n].poly;

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

bool PortalOcclusionCuller::cull_sphere_to_spheres(const Vector3 &p_occludee_center, real_t p_occludee_radius, const Vector3 &p_ray_dir, real_t p_dist_to_occludee, int p_ignore_sphere) const {
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
				if ((dist < p_dist_to_occludee) && (s != p_ignore_sphere)) {
					// occluded
					return true;
				}
			}
		} // expanded occluder radius is more than 0
	}

	return false;
}

bool PortalOcclusionCuller::cull_sphere(const Vector3 &p_occludee_center, real_t p_occludee_radius, int p_ignore_sphere, bool p_cull_to_polys) const {
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

	if (cull_sphere_to_spheres(p_occludee_center, p_occludee_radius, ray_dir, dist_to_occludee, p_ignore_sphere)) {
		return true;
	}

	if (p_cull_to_polys && cull_sphere_to_polys(p_occludee_center, p_occludee_radius)) {
		return true;
	}

	return false;
}

PortalOcclusionCuller::PortalOcclusionCuller() {
	_max_spheres = GLOBAL_GET("rendering/misc/occlusion_culling/max_active_spheres");
	_max_polys = GLOBAL_GET("rendering/misc/occlusion_culling/max_active_polygons");
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
