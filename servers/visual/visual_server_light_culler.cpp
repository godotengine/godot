/*************************************************************************/
/*  visual_server_light_culler.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "visual_server_light_culler.h"
#include "core/math/camera_matrix.h"
#include "core/math/plane.h"
#include "scene/3d/camera.h"
#include "visual_server_globals.h"
#include "visual_server_scene.h"

#ifdef LIGHT_CULLER_DEBUG_LOGGING
const char *VisualServerLightCuller::string_planes[] = {
	"NEAR",
	"FAR",
	"LEFT",
	"TOP",
	"RIGHT",
	"BOTTOM",
};
const char *VisualServerLightCuller::string_points[] = {
	"FAR_LEFT_TOP",
	"FAR_LEFT_BOTTOM",
	"FAR_RIGHT_TOP",
	"FAR_RIGHT_BOTTOM",
	"NEAR_LEFT_TOP",
	"NEAR_LEFT_BOTTOM",
	"NEAR_RIGHT_TOP",
	"NEAR_RIGHT_BOTTOM",
};
#endif

bool VisualServerLightCuller::prepare_light(const VisualServerScene::Instance &p_instance) {
	if (!is_active())
		return true;

	LightSource lsource;
	switch (VSG::storage->light_get_type(p_instance.base)) {
		case VS::LIGHT_SPOT:
			lsource.etype = LightSource::ST_SPOTLIGHT;
			lsource.angle = VSG::storage->light_get_param(p_instance.base, VS::LIGHT_PARAM_SPOT_ANGLE);
			lsource.range = VSG::storage->light_get_param(p_instance.base, VS::LIGHT_PARAM_RANGE);
			break;
		case VS::LIGHT_OMNI:
			lsource.etype = LightSource::ST_OMNI;
			lsource.range = VSG::storage->light_get_param(p_instance.base, VS::LIGHT_PARAM_RANGE);
			break;
		case VS::LIGHT_DIRECTIONAL:
			lsource.etype = LightSource::ST_DIRECTIONAL;
			// could deal with a max directional shadow range here? NYI
			// LIGHT_PARAM_SHADOW_MAX_DISTANCE
			break;
	}

	lsource.pos = p_instance.transform.origin;
	lsource.dir = -p_instance.transform.basis.get_axis(2);
	lsource.dir.normalize();

	return _add_light_camera_planes(lsource);
}

#ifdef LIGHT_CULLER_DEBUG_LOGGING
String VisualServerLightCuller::plane_bitfield_to_string(unsigned int BF) {
	String sz;

	for (int n = 0; n < 6; n++) {
		unsigned int bit = 1 << n;
		if (BF & bit) {
			sz += String(string_planes[n]) + ", ";
		}
	}

	return sz;
}
#endif

int VisualServerLightCuller::cull(int count, VisualServerScene::Instance **ppInstances) {
	if (!is_active())
		return count;

	// if the light is out of range, no need to check anything, just return 0 casters.
	// ideally an out of range light should not even be drawn AT ALL (no shadow map, no PCF etc)
	if (out_of_range) {
		return 0;
	}

	int new_count = count;

	// go through all the casters in the list (the list will hopefully shrink as we go)
	for (int n = 0; n < new_count; n++) {
		// world space aabb
		const AABB &bb = ppInstances[n]->transformed_aabb;

#ifdef LIGHT_CULLER_DEBUG_LOGGING
		if (is_logging()) {
			print_line("bb : " + String(bb));
		}
#endif

		float r_min, r_max;
		bool bShow = true;

		for (int p = 0; p < num_cull_planes; p++) {
			// as we only need r_min, could this be optimized?
			bb.project_range_in_plane(cull_planes[p], r_min, r_max);

#ifdef LIGHT_CULLER_DEBUG_LOGGING
			if (is_logging()) {
				print_line("\tplane " + itos(p) + " : " + String(cull_planes[p]) + " r_min " + String(Variant(r_min)) + " r_max " + String(Variant(r_max)));
			}
#endif

			if (r_min > 0.0f) {
				bShow = false;
				break;
			}
		}

		// remove
		if (!bShow) {
			// quick unsorted remove - swap last element and reduce count
			ppInstances[n] = ppInstances[new_count - 1];
			new_count--;

			// repeat this element next iteration of the loop as it has been removed and replaced by the last
			n--;
		}
	}

#ifdef LIGHT_CULLER_DEBUG_LOGGING
	int removed = count - new_count;
	if (removed) {
		if (((debug_count) % 60) == 0)
			print_line("[" + itos(debug_count) + "] linear cull before " + itos(count) + " after " + itos(new_count));
	}
#endif

	return new_count;
}

void VisualServerLightCuller::add_cull_plane(const Plane &p) {
	ERR_FAIL_COND(num_cull_planes >= MAX_CULL_PLANES);
	cull_planes[num_cull_planes++] = p;
}

// directional lights are different to points, as the origin is infinitely in the distance, so the plane third
// points are derived differently
bool VisualServerLightCuller::add_light_camera_planes_directional(const LightSource &lsource) {
	uint32_t lookup = 0;

	// directional light, we will use dot against the light direction to determine back facing planes
	for (int n = 0; n < 6; n++) {
		float dot = frustum_planes[n].normal.dot(lsource.dir);
		if (dot > 0.0f) {
			lookup |= 1 << n;

			// add backfacing camera frustum planes
			add_cull_plane(frustum_planes[n]);
		}
	}

	ERR_FAIL_COND_V(lookup >= LUT_SIZE, true);

	// deal with special case... if the light is INSIDE the view frustum (i.e. all planes face away)
	// then we will add the camera frustum planes to clip the light volume .. there is no need to
	// render shadow casters outside the frustum as shadows can never re-enter the frustum.

	if (lookup == 63) // should never happen with directional light?? this may be able to be removed
	{
		num_cull_planes = 0;
		for (int n = 0; n < frustum_planes.size(); n++) {
			//planes.push_back(frustum_planes[n]);
			add_cull_plane(frustum_planes[n]);
		}

		return true;
	}

	// each edge forms a plane
	uint8_t *entry = &LUT_entries[lookup][0];
	int nEdges = LUT_entry_sizes[lookup] - 1;

	for (int e = 0; e < nEdges; e++) {
		int i0 = entry[e];
		int i1 = entry[e + 1];
		const Vector3 &pt0 = frustum_points[i0];
		const Vector3 &pt1 = frustum_points[i1];

		// create a third point from the light direction
		Vector3 pt2 = pt0 - lsource.dir;

		// create plane from 3 points
		Plane p(pt0, pt1, pt2);
		add_cull_plane(p);
	}

	// last to 0 edge
	if (nEdges) {
		int i0 = entry[nEdges]; // last
		int i1 = entry[0]; // first

		const Vector3 &pt0 = frustum_points[i0];
		const Vector3 &pt1 = frustum_points[i1];

		// create a third point from the light direction
		Vector3 pt2 = pt0 - lsource.dir;

		// create plane from 3 points
		Plane p(pt0, pt1, pt2);
		add_cull_plane(p);
	}

#ifdef LIGHT_CULLER_DEBUG_LOGGING
	if (is_logging()) {
		print_line("lcam.pos is " + String(lsource.pos));
	}
#endif

	return true;
}

bool VisualServerLightCuller::_add_light_camera_planes(const LightSource &lsource) {
	if (!is_active())
		return true;

	// we should have called prepare_camera before this
	ERR_FAIL_COND_V(frustum_planes.size() != 6, true);

	// start with 0 cull planes
	num_cull_planes = 0;
	out_of_range = false;

	// doesn't account for directional lights yet! only points
	switch (lsource.etype) {
		case LightSource::ST_SPOTLIGHT:
		case LightSource::ST_OMNI:
			break;
		case LightSource::ST_DIRECTIONAL:
			return add_light_camera_planes_directional(lsource);
			break;
		default:
			return false; // not yet supported
			break;
	}

	uint32_t lookup = 0;

	// find which of the camera planes are facing away from the light

	// POINT LIGHT (spotlight, omni)
	// BRAINWAVE!! Instead of using dot product to compare light direction to plane, we can simply
	// find out which side of the plane the camera is on!! By definition this marks the point at which the plane
	// becomes invisible. This works for portals too!
	for (int n = 0; n < 6; n++) {
		float dist = frustum_planes[n].distance_to(lsource.pos);
		if (dist < 0.0f) {
			lookup |= 1 << n;

			// add backfacing camera frustum planes
			add_cull_plane(frustum_planes[n]);
		} else {
			// is the light out of range?
			if (dist >= lsource.range) {
				// if the light is out of range, no need to do anything else, everything will be culled
				out_of_range = true;
				return false;
			}
		}
	}

	// the lookup should be within the LUT, logic should prevent this
	ERR_FAIL_COND_V(lookup >= LUT_SIZE, true);

	// deal with special case... if the light is INSIDE the view frustum (i.e. all planes face away)
	// then we will add the camera frustum planes to clip the light volume .. there is no need to
	// render shadow casters outside the frustum as shadows can never re-enter the frustum.
	if (lookup == 63) {
		num_cull_planes = 0;
		for (int n = 0; n < frustum_planes.size(); n++) {
			add_cull_plane(frustum_planes[n]);
		}

		return true;
	}

	// each edge forms a plane
	uint8_t *entry = &LUT_entries[lookup][0];
	int nEdges = LUT_entry_sizes[lookup] - 1;

	for (int e = 0; e < nEdges; e++) {
		int i0 = entry[e];
		int i1 = entry[e + 1];
		const Vector3 &pt0 = frustum_points[i0];
		const Vector3 &pt1 = frustum_points[i1];

		// create plane from 3 points
		Plane p(pt0, pt1, lsource.pos);
		add_cull_plane(p);
	}

	// last to 0 edge
	if (nEdges) {
		int i0 = entry[nEdges]; // last
		int i1 = entry[0]; // first

		const Vector3 &pt0 = frustum_points[i0];
		const Vector3 &pt1 = frustum_points[i1];

		// create plane from 3 points
		Plane p(pt0, pt1, lsource.pos);
		add_cull_plane(p);
	}

#ifdef LIGHT_CULLER_DEBUG_LOGGING
	if (is_logging()) {
		print_line("lsource.pos is " + String(lsource.pos));
	}
#endif

	return true;
}

bool VisualServerLightCuller::prepare_camera(const Transform &p_cam_transform, const CameraMatrix &p_cam_matrix) {
	debug_count++;

	// for debug flash off and on
#ifdef LIGHT_CULLER_DEBUG_FLASH
	if (!Engine::get_singleton()->is_editor_hint()) {
		int dc = debug_count / LIGHT_CULLER_DEBUG_FLASH_FREQUENCY;
		bool bnew_active;
		if ((dc % 2) == 0)
			bnew_active = true;
		else
			bnew_active = false;

		if (bnew_active != bactive) {
			bactive = bnew_active;
			print_line("switching light culler " + String(Variant(bactive)));
		}
	}
#endif

	if (!is_active())
		return false;

	// get the camera frustum planes in world space
	frustum_planes = p_cam_matrix.get_projection_planes(p_cam_transform);

	num_cull_planes = 0;

#ifdef LIGHT_CULLER_DEBUG_LOGGING
	if (is_logging()) {
		for (int p = 0; p < 6; p++) {
			print_line("plane " + itos(p) + " : " + String(frustum_planes[p]));
		}
	}
#endif

	// we want to calculate the frustum corners in a specific order
	const CameraMatrix::Planes intersections[8][3] = {
		{ CameraMatrix::PLANE_FAR, CameraMatrix::PLANE_LEFT, CameraMatrix::PLANE_TOP },
		{ CameraMatrix::PLANE_FAR, CameraMatrix::PLANE_LEFT, CameraMatrix::PLANE_BOTTOM },
		{ CameraMatrix::PLANE_FAR, CameraMatrix::PLANE_RIGHT, CameraMatrix::PLANE_TOP },
		{ CameraMatrix::PLANE_FAR, CameraMatrix::PLANE_RIGHT, CameraMatrix::PLANE_BOTTOM },
		{ CameraMatrix::PLANE_NEAR, CameraMatrix::PLANE_LEFT, CameraMatrix::PLANE_TOP },
		{ CameraMatrix::PLANE_NEAR, CameraMatrix::PLANE_LEFT, CameraMatrix::PLANE_BOTTOM },
		{ CameraMatrix::PLANE_NEAR, CameraMatrix::PLANE_RIGHT, CameraMatrix::PLANE_TOP },
		{ CameraMatrix::PLANE_NEAR, CameraMatrix::PLANE_RIGHT, CameraMatrix::PLANE_BOTTOM },
	};

	for (int i = 0; i < 8; i++) {

		// 3 plane intersection, gives us a point
		bool res = frustum_planes[intersections[i][0]].intersect_3(frustum_planes[intersections[i][1]], frustum_planes[intersections[i][2]], &frustum_points[i]);

		// what happens with a zero frustum? NYI - deal with this
		ERR_FAIL_COND_V(!res, false);

#ifdef LIGHT_CULLER_DEBUG_LOGGING
		if (is_logging())
			print_line("point " + itos(i) + " -> " + String(frustum_points[i]));
#endif
	}

	return true;
}

VisualServerLightCuller::VisualServerLightCuller() {
	// used to determine which frame to give debug output
	debug_count = -1;

	// b active is switching on and off the light culler
	bactive = Engine::get_singleton()->is_editor_hint() == false;
}

/* clang-format off */
uint8_t VisualServerLightCuller::LUT_entry_sizes[LUT_SIZE] = {0, 4, 4, 0, 4, 6, 6, 8, 4, 6, 6, 8, 6, 6, 6, 6, 4, 6, 6, 8, 0, 8, 8, 0, 6, 6, 6, 6, 8, 6, 6, 4, 4, 6, 6, 8, 6, 6, 6, 6, 0, 8, 8, 0, 8, 6, 6, 4, 6, 6, 6, 6, 8, 6, 6, 4, 8, 6, 6, 4, 0, 4, 4, 0, };

// the lookup table used to determine which edges form the silhouette of the camera frustum,
// depending on the viewing angle (defined by which camera planes are backward facing)
uint8_t VisualServerLightCuller::LUT_entries[LUT_SIZE][8] = {
{0, 0, 0, 0, 0, 0, 0, },
{7, 6, 4, 5, 0, 0, 0, },
{1, 0, 2, 3, 0, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, },
{1, 5, 4, 0, 0, 0, 0, },
{1, 5, 7, 6, 4, 0, 0, },
{4, 0, 2, 3, 1, 5, 0, },
{5, 7, 6, 4, 0, 2, 3, },
{0, 4, 6, 2, 0, 0, 0, },
{0, 4, 5, 7, 6, 2, 0, },
{6, 2, 3, 1, 0, 4, 0, },
{2, 3, 1, 0, 4, 5, 7, },
{0, 1, 5, 4, 6, 2, 0, },
{0, 1, 5, 7, 6, 2, 0, },
{6, 2, 3, 1, 5, 4, 0, },
{2, 3, 1, 5, 7, 6, 0, },
{2, 6, 7, 3, 0, 0, 0, },
{2, 6, 4, 5, 7, 3, 0, },
{7, 3, 1, 0, 2, 6, 0, },
{3, 1, 0, 2, 6, 4, 5, },
{0, 0, 0, 0, 0, 0, 0, },
{2, 6, 4, 0, 1, 5, 7, },
{7, 3, 1, 5, 4, 0, 2, },
{0, 0, 0, 0, 0, 0, 0, },
{2, 0, 4, 6, 7, 3, 0, },
{2, 0, 4, 5, 7, 3, 0, },
{7, 3, 1, 0, 4, 6, 0, },
{3, 1, 0, 4, 5, 7, 0, },
{2, 0, 1, 5, 4, 6, 7, },
{2, 0, 1, 5, 7, 3, 0, },
{7, 3, 1, 5, 4, 6, 0, },
{3, 1, 5, 7, 0, 0, 0, },
{3, 7, 5, 1, 0, 0, 0, },
{3, 7, 6, 4, 5, 1, 0, },
{5, 1, 0, 2, 3, 7, 0, },
{7, 6, 4, 5, 1, 0, 2, },
{3, 7, 5, 4, 0, 1, 0, },
{3, 7, 6, 4, 0, 1, 0, },
{5, 4, 0, 2, 3, 7, 0, },
{7, 6, 4, 0, 2, 3, 0, },
{0, 0, 0, 0, 0, 0, 0, },
{3, 7, 6, 2, 0, 4, 5, },
{5, 1, 0, 4, 6, 2, 3, },
{0, 0, 0, 0, 0, 0, 0, },
{3, 7, 5, 4, 6, 2, 0, },
{3, 7, 6, 2, 0, 1, 0, },
{5, 4, 6, 2, 3, 7, 0, },
{7, 6, 2, 3, 0, 0, 0, },
{3, 2, 6, 7, 5, 1, 0, },
{3, 2, 6, 4, 5, 1, 0, },
{5, 1, 0, 2, 6, 7, 0, },
{1, 0, 2, 6, 4, 5, 0, },
{3, 2, 6, 7, 5, 4, 0, },
{3, 2, 6, 4, 0, 1, 0, },
{5, 4, 0, 2, 6, 7, 0, },
{6, 4, 0, 2, 0, 0, 0, },
{3, 2, 0, 4, 6, 7, 5, },
{3, 2, 0, 4, 5, 1, 0, },
{5, 1, 0, 4, 6, 7, 0, },
{1, 0, 4, 5, 0, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, },
{3, 2, 0, 1, 0, 0, 0, },
{5, 4, 6, 7, 0, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, },
};

/* clang-format on */
