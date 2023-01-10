/**************************************************************************/
/*  portal_types.h                                                        */
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

#ifndef PORTAL_TYPES_H
#define PORTAL_TYPES_H

#include "core/local_vector.h"
#include "core/math/aabb.h"
#include "core/math/plane.h"
#include "core/math/quat.h"
#include "core/math/transform.h"
#include "core/math/vector3.h"
#include "core/object_id.h"
#include "core/rid.h"
#include "portal_defines.h"

// visual server scene instance.
// we can't have a pointer to nested class outside of visual server scene...
// so rather than use a void * straight we can use this for some semblance
// of safety
typedef void VSInstance;
typedef void VSGhost;

// the handles are just IDs, but nicer to be specific on what they are in the code.
// Also - the handles are plus one based (i.e. 0 is unset, 1 is id 0, 2 is id 1 etc.
typedef uint32_t PortalHandle;
typedef uint32_t RoomHandle;
typedef uint32_t RoomGroupHandle;
typedef uint32_t OcclusionHandle;
typedef uint32_t RGhostHandle;
typedef uint32_t OccluderInstanceHandle;
typedef uint32_t OccluderResourceHandle;

struct VSPortal {
	enum ClipResult {
		CLIP_OUTSIDE,
		CLIP_PARTIAL,
		CLIP_INSIDE,
	};

	// explicit create and destroy rather than constructors / destructors
	// because we are using a pool so objects may be reused
	void create() {
		_linkedroom_ID[0] = -1;
		_linkedroom_ID[1] = -1;
		_active = true;
		_plane = Plane();
		_portal_id = -1;
		_aabb = AABB();
	}

	void destroy() {
		_pts_world.reset();
	}

	void rooms_and_portals_clear() {
		_linkedroom_ID[0] = -1;
		_linkedroom_ID[1] = -1;
		_active = true;
		_plane = Plane();
		_aabb = AABB();
		_pts_world.reset();
	}

	VSPortal::ClipResult clip_with_plane(const Plane &p) const;
	void add_planes(const Vector3 &p_cam, LocalVector<Plane> &r_planes, bool p_outgoing) const;
	void debug_check_plane_validity(const Plane &p) const;

	void add_pvs_planes(const VSPortal &p_first, bool p_first_outgoing, LocalVector<Plane, int32_t> &r_planes, bool p_outgoing) const;
	bool _pvs_is_outside_planes(const LocalVector<Plane, int32_t> &p_planes) const;

private:
	bool _test_pvs_plane(const Plane &p_plane, const Vector3 *pts_a, int num_a, const Vector3 *pts_b, int num_b) const;
	bool _is_plane_duplicate(const Plane &p_plane, const LocalVector<Plane, int32_t> &p_planes) const;

public:
	// returns the room to if if crosses, or else returns -1
	int geometry_crosses_portal(int p_room_from, const AABB &p_aabb, const Vector<Vector3> &p_pts) const {
		// first aabb check
		if (!p_aabb.intersects(_aabb)) {
			return -1;
		}

		// disallow sprawling from outer to inner rooms.
		// This is a convenience feature that stops e.g. terrain sprawling into
		// a building. If you want geometry to feature in the inner room and the outer,
		// simply place it in the inner room.
		if (_internal && (_linkedroom_ID[0] != p_room_from)) {
			return -1;
		}

		// accurate check use portal triangles
		// NYI

		const real_t epsilon = _margin;

		if (p_room_from == _linkedroom_ID[0]) {
			// outward
			// how far do points project over the portal
			for (int n = 0; n < p_pts.size(); n++) {
				real_t dist = _plane.distance_to(p_pts[n]);
				if (dist > epsilon) {
					return _linkedroom_ID[1];
				}
			}
		} else {
			// inward
			DEV_ASSERT(p_room_from == _linkedroom_ID[1]);
			for (int n = 0; n < p_pts.size(); n++) {
				real_t dist = _plane.distance_to(p_pts[n]);
				if (dist < -epsilon) {
					return _linkedroom_ID[0];
				}
			}
		}

		// no points crossed the portal
		return -1;
	}

	// returns the room to if if crosses, or else returns -1
	int crosses_portal(int p_room_from, const AABB &p_aabb, bool p_disallow_crossing_internal = false, bool p_accurate_check = false) const {
		// first aabb check
		if (!p_aabb.intersects(_aabb)) {
			return -1;
		}

		// disallow sprawling from outer to inner rooms.
		// This is a convenience feature that stops e.g. terrain sprawling into
		// a building. If you want geometry to feature in the inner room and the outer,
		// simply place it in the inner room.
		if (p_disallow_crossing_internal && _internal && (_linkedroom_ID[0] != p_room_from)) {
			return -1;
		}

		// accurate check use portal triangles
		// NYI
		real_t r_min, r_max;
		p_aabb.project_range_in_plane(_plane, r_min, r_max);

		const real_t epsilon = _margin; //10.0;

		if (p_room_from == _linkedroom_ID[0]) {
			if (r_max > epsilon) {
				return _linkedroom_ID[1];
			} else {
				return -1;
			}
		}

		DEV_ASSERT(p_room_from == _linkedroom_ID[1]);
		if (r_min < -epsilon) {
			return _linkedroom_ID[0];
		}

		return -1;
	}

	// the portal needs a list of unique world points (in order, clockwise?)
	LocalVector<Vector3> _pts_world;

	// used in PVS calculation
	Vector3 _pt_center;

	// used for occlusion culling with occluders
	real_t _bounding_sphere_radius = 0.0;

	// portal plane
	Plane _plane;

	// aabb for quick bounds checks
	AABB _aabb;

	uint32_t _portal_id = -1;

	// in order to detect objects crossing portals,
	// an extension margin can be used to prevent objects
	// that *just* cross the portal extending into the next room
	real_t _margin = 1.0;

	// these are room IDs, or -1 if unset
	int _linkedroom_ID[2];

	// can be turned on and off by the user
	bool _active = true;

	// internal portals have slightly different behaviour
	bool _internal = false;

	VSPortal() {
		_linkedroom_ID[0] = -1;
		_linkedroom_ID[1] = -1;
	}
};

struct VSRoomGroup {
	void create() {
	}

	void destroy() {
		_room_ids.reset();
	}

	// used for calculating gameplay notifications
	uint32_t last_room_tick_hit = 0;

	ObjectID _godot_instance_ID = 0;

	LocalVector<uint32_t, int32_t> _room_ids;
};

struct VSRoom {
	// explicit create and destroy rather than constructors / destructors
	// because we are using a pool so objects may be reused
	void create() {
		_room_ID = -1;
		_aabb = AABB();
	}

	void destroy() {
		_static_ids.reset();
		_static_ghost_ids.reset();
		_planes.reset();
		_verts.reset();
		_portal_ids.reset();
		_roamer_pool_ids.reset();
		_rghost_pool_ids.reset();
		_roomgroup_ids.reset();
		_pvs_first = 0;
		_pvs_size = 0;
		_secondary_pvs_first = 0;
		_secondary_pvs_size = 0;
		_priority = 0;
		_contains_internal_rooms = false;
		last_room_tick_hit = 0;
	}

	void cleanup_after_conversion() {
		_verts.reset();
	}

	void rooms_and_portals_clear() {
		destroy();
		_aabb = AABB();
		// don't unset the room_ID here, because rooms may be accessed after this is called
	}

	// this isn't just useful for checking whether a point is within (i.e. returned value is 0 or less)
	// it is useful for finding the CLOSEST room to a point (by plane distance, doesn't take into account corners etc)
	real_t is_point_within(const Vector3 &p_pos) const {
		// inside by default
		real_t closest_dist = -FLT_MAX;

		for (int n = 0; n < _planes.size(); n++) {
			real_t dist = _planes[n].distance_to(p_pos);
			if (dist > closest_dist) {
				closest_dist = dist;
			}
		}

		return closest_dist;
	}

	// not super fast, but there shouldn't be that many roamers per room
	bool remove_roamer(uint32_t p_pool_id) {
		for (int n = 0; n < _roamer_pool_ids.size(); n++) {
			if (_roamer_pool_ids[n] == p_pool_id) {
				_roamer_pool_ids.remove_unordered(n);
				return true;
			}
		}
		return false;
	}

	bool remove_rghost(uint32_t p_pool_id) {
		for (int n = 0; n < _rghost_pool_ids.size(); n++) {
			if (_rghost_pool_ids[n] == p_pool_id) {
				_rghost_pool_ids.remove_unordered(n);
				return true;
			}
		}
		return false;
	}

	bool remove_occluder(uint32_t p_pool_id) {
		for (unsigned int n = 0; n < _occluder_pool_ids.size(); n++) {
			if (_occluder_pool_ids[n] == p_pool_id) {
				_occluder_pool_ids.remove_unordered(n);
				return true;
			}
		}
		return false;
	}

	void add_roamer(uint32_t p_pool_id) {
		_roamer_pool_ids.push_back(p_pool_id);
	}

	void add_rghost(uint32_t p_pool_id) {
		_rghost_pool_ids.push_back(p_pool_id);
	}

	void add_occluder(uint32_t p_pool_id) {
		_occluder_pool_ids.push_back(p_pool_id);
	}

	// keep a list of statics in the room .. statics may appear
	// in more than one room due to sprawling!
	LocalVector<uint32_t, int32_t> _static_ids;
	LocalVector<uint32_t, int32_t> _static_ghost_ids;

	// very rough
	AABB _aabb;

	int32_t _room_ID = -1;
	ObjectID _godot_instance_ID = 0;

	// rooms with a higher priority are internal rooms ..
	// rooms within a room. These will be chosen in preference
	// when finding the room within, when within more than one room.
	// Example, house in a terrain room.
	int32_t _priority = 0;

	bool _contains_internal_rooms = false;

	int32_t _pvs_first = 0;
	int32_t _secondary_pvs_first = 0;
	uint16_t _pvs_size = 0;
	uint16_t _secondary_pvs_size = 0;

	// used for calculating gameplay notifications
	uint32_t last_room_tick_hit = 0;

	// convex hull of the room, either determined by geometry or manual bound
	LocalVector<Plane, int32_t> _planes;

	// vertices of the corners of the hull, passed from the scene tree
	// (note these don't take account of any final portal planes adjusted by the portal renderer)
	LocalVector<Vector3, int32_t> _verts;

	// which portals are in the room (ingoing and outgoing)
	LocalVector<uint32_t, int32_t> _portal_ids;

	// roaming movers currently in the room
	LocalVector<uint32_t, int32_t> _roamer_pool_ids;
	LocalVector<uint32_t, int32_t> _rghost_pool_ids;

	// only using uint here for compatibility with TrackedPoolList,
	// as we will use either this or TrackedPoolList for occlusion testing
	LocalVector<uint32_t, uint32_t> _occluder_pool_ids;

	// keep track of which roomgroups the room is in, that
	// way we can switch on and off roomgroups as they enter / exit view
	LocalVector<uint32_t, int32_t> _roomgroup_ids;
};

// Possibly shared data, in local space
struct VSOccluder_Resource {
	void create() {
		type = OT_UNDEFINED;
		revision = 0;
		list_ids.clear();
	}

	// these should match the values in VisualServer::OccluderType
	enum Type : uint32_t {
		OT_UNDEFINED,
		OT_SPHERE,
		OT_MESH,
		OT_NUM_TYPES,
	} type;

	// If the revision of the instance and the resource don't match,
	// then the local versions have been updated and need transforming
	// to world space in the instance (i.e. it is dirty)
	uint32_t revision;

	// ids of multiple objects in the appropriate occluder pool:
	// local space for resources, and world space for occluder instances
	LocalVector<uint32_t, int32_t> list_ids;
};

struct VSOccluder_Instance : public VSOccluder_Resource {
	void create() {
		VSOccluder_Resource::create();
		room_id = -1;
		active = true;
		resource_pool_id = UINT32_MAX;
	}

	// Occluder instance can be bound to one resource (which will include data in local space)
	// This should be set back to NULL if the resource is deleted
	uint32_t resource_pool_id;

	// which is the primary room this group of occluders is in
	// (it may sprawl into multiple rooms)
	int32_t room_id;

	// location for finding the room
	Vector3 pt_center;

	// world space aabb, only updated when dirty
	AABB aabb;

	// global xform
	Transform xform;

	// controlled by the visible flag on the occluder
	bool active;
};

namespace Occlusion {
struct Sphere {
	Vector3 pos;
	real_t radius;

	void create() { radius = 0.0; }
	void from_plane(const Plane &p_plane) {
		pos = p_plane.normal;
		// Disallow negative radius. Even zero radius should not really be sent.
		radius = MAX(p_plane.d, 0.0);
	}

	bool intersect_ray(const Vector3 &p_ray_origin, const Vector3 &p_ray_dir, real_t &r_dist, real_t radius_squared) const {
		Vector3 offset = pos - p_ray_origin;
		real_t c2 = offset.length_squared();

		real_t v = offset.dot(p_ray_dir);
		real_t d = radius_squared - (c2 - (v * v));

		if (d < 0.0) {
			return false;
		}

		r_dist = (v - Math::sqrt(d));
		return true;
	}
};

struct Poly {
	static const int MAX_POLY_VERTS = PortalDefines::OCCLUSION_POLY_MAX_VERTS;
	void create() {
		num_verts = 0;
	}
	void flip() {
		for (int n = 0; n < num_verts / 2; n++) {
			SWAP(verts[n], verts[num_verts - n - 1]);
		}
	}

	int num_verts;
	Vector3 verts[MAX_POLY_VERTS];
};

struct PolyPlane : public Poly {
	void flip() {
		plane = -plane;
		Poly::flip();
	}
	Plane plane;
};

} // namespace Occlusion

struct VSOccluder_Sphere : public Occlusion::Sphere {
};

struct VSOccluder_Poly {
	static const int MAX_POLY_HOLES = PortalDefines::OCCLUSION_POLY_MAX_HOLES;
	void create() {
		poly.create();
		num_holes = 0;
		two_way = false;
		for (int n = 0; n < MAX_POLY_HOLES; n++) {
			hole_pool_ids[n] = UINT32_MAX;
		}
	}
	Occlusion::PolyPlane poly;
	bool two_way;

	int num_holes;
	uint32_t hole_pool_ids[MAX_POLY_HOLES];
};

struct VSOccluder_Hole : public Occlusion::Poly {
};

#endif // PORTAL_TYPES_H
