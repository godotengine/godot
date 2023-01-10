/**************************************************************************/
/*  portal_types.cpp                                                      */
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

#include "portal_types.h"

VSPortal::ClipResult VSPortal::clip_with_plane(const Plane &p) const {
	int nOutside = 0;
	int nPoints = _pts_world.size();

	for (int n = 0; n < nPoints; n++) {
		real_t d = p.distance_to(_pts_world[n]);

		if (d >= 0.0) {
			nOutside++;
		}
	}

	if (nOutside == nPoints) {
		return CLIP_OUTSIDE;
	}

	if (nOutside == 0) {
		return CLIP_INSIDE;
	}

	return CLIP_PARTIAL;
}

void VSPortal::add_pvs_planes(const VSPortal &p_first, bool p_first_outgoing, LocalVector<Plane, int32_t> &r_planes, bool p_outgoing) const {
	int num_a = p_first._pts_world.size();
	int num_b = _pts_world.size();

	// get the world points of both in the correct order based on whether outgoing .. note this isn't very efficient...
	Vector3 *pts_a = (Vector3 *)alloca(num_a * sizeof(Vector3));
	Vector3 *pts_b = (Vector3 *)alloca(num_b * sizeof(Vector3));

	if (p_first_outgoing) {
		// straight copy
		for (int n = 0; n < num_a; n++) {
			pts_a[n] = p_first._pts_world[n];
		}
	} else {
		for (int n = 0; n < num_a; n++) {
			pts_a[n] = p_first._pts_world[num_a - 1 - n];
		}
	}

	if (p_outgoing) {
		// straight copy
		for (int n = 0; n < num_b; n++) {
			pts_b[n] = _pts_world[n];
		}
	} else {
		for (int n = 0; n < num_b; n++) {
			pts_b[n] = _pts_world[num_b - 1 - n];
		}
	}

	// go through and try every combination of points to form a clipping plane
	for (int pvA = 0; pvA < num_a; pvA++) {
		for (int pvB = 0; pvB < num_b; pvB++) {
			int pvC = (pvB + 1) % num_b;

			// three verts
			const Vector3 &va = pts_a[pvA];
			const Vector3 &vb = pts_b[pvB];
			const Vector3 &vc = pts_b[pvC];

			// create plane
			Plane plane = Plane(va, vc, vb);

			// already exists similar plane, so ignore
			if (_is_plane_duplicate(plane, r_planes)) {
				continue;
			}

			if (_test_pvs_plane(-plane, pts_a, num_a, pts_b, num_b)) {
				// add the plane
				r_planes.push_back(plane);
			}

		} // for pvB
	} // for pvA
}

// typically we will end up with a bunch of duplicate planes being trying to be added for a portal.
// we can remove any that are too similar
bool VSPortal::_is_plane_duplicate(const Plane &p_plane, const LocalVector<Plane, int32_t> &p_planes) const {
	const real_t epsilon_d = 0.001;
	const real_t epsilon_dot = 0.98;

	for (int n = 0; n < p_planes.size(); n++) {
		const Plane &p = p_planes[n];
		if (Math::absf(p_plane.d - p.d) > epsilon_d) {
			continue;
		}

		real_t dot = p_plane.normal.dot(p.normal);
		if (dot < epsilon_dot) {
			continue;
		}

		// match
		return true;
	}

	return false;
}

bool VSPortal::_pvs_is_outside_planes(const LocalVector<Plane, int32_t> &p_planes) const {
	// short version
	const Vector<Vector3> &pts = _pts_world;
	int nPoints = pts.size();

	const real_t epsilon = 0.1;

	for (int p = 0; p < p_planes.size(); p++) {
		for (int n = 0; n < nPoints; n++) {
			const Vector3 &pt = pts[n];
			real_t dist = p_planes[p].distance_to(pt);

			if (dist < -epsilon) {
				return false;
			}
		}
	}

	return true;
}

bool VSPortal::_test_pvs_plane(const Plane &p_plane, const Vector3 *pts_a, int num_a, const Vector3 *pts_b, int num_b) const {
	const real_t epsilon = 0.1;

	for (int n = 0; n < num_a; n++) {
		real_t dist = p_plane.distance_to(pts_a[n]);

		if (dist > epsilon) {
			return false;
		}
	}

	for (int n = 0; n < num_b; n++) {
		real_t dist = p_plane.distance_to(pts_b[n]);

		if (dist < -epsilon) {
			return false;
		}
	}

	return true;
}

// add clipping planes to the vector formed by each portal edge and the camera
void VSPortal::add_planes(const Vector3 &p_cam, LocalVector<Plane> &r_planes, bool p_outgoing) const {
	// short version
	const Vector<Vector3> &pts = _pts_world;

	int nPoints = pts.size();
	ERR_FAIL_COND(nPoints < 3);

	Plane p;

	int offset_a, offset_b;
	if (p_outgoing) {
		offset_a = 0;
		offset_b = -1;
	} else {
		offset_a = -1;
		offset_b = 0;
	}

	for (int n = 1; n < nPoints; n++) {
		p = Plane(p_cam, pts[n + offset_a], pts[n + offset_b]);

		// detect null plane
		// if (p.normal.length_squared() < 0.1)
		// {
		// print("NULL plane detected from points : ");
		// print(ptCam + pts[n] + pts[n-1]);
		// }
		r_planes.push_back(p);
		debug_check_plane_validity(p);
	}

	// first and last
	if (p_outgoing) {
		p = Plane(p_cam, pts[0], pts[nPoints - 1]);
	} else {
		p = Plane(p_cam, pts[nPoints - 1], pts[0]);
	}

	r_planes.push_back(p);
	debug_check_plane_validity(p);

	// debug
	// if (!manager.m_bDebugPlanes)
	// return;

	// for (int n=0; n<nPoints; n++)
	// {
	// manager.m_DebugPlanes.push_back(pts[n]);
	// }
}

void VSPortal::debug_check_plane_validity(const Plane &p) const {
	// DEV_ASSERT(p.distance_to(center) < 0.0);
}
