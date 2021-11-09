/*************************************************************************/
/*  room.cpp                                                             */
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

#include "room.h"

#include "portal.h"
#include "room_group.h"
#include "room_manager.h"
#include "servers/visual_server.h"

void Room::SimplifyInfo::set_simplify(real_t p_value, real_t p_room_size) {
	_plane_simplify = CLAMP(p_value, 0.0, 1.0);

	// just for reference in case we later want to use degrees...
	// _plane_simplify_dot = Math::cos(Math::deg2rad(_plane_simplify_degrees));

	// _plane_simplify_dot = _plane_simplify;
	// _plane_simplify_dot *= _plane_simplify_dot;
	// _plane_simplify_dot = 1.0 - _plane_simplify_dot;

	// distance based on size of room
	// _plane_simplify_dist = p_room_size * 0.1 * _plane_simplify;
	// _plane_simplify_dist = MAX(_plane_simplify_dist, 0.08);

	// test fix
	_plane_simplify_dot = 0.99;
	_plane_simplify_dist = 0.08;

	// print_verbose("plane simplify dot : " + String(Variant(_plane_simplify_dot)));
	// print_verbose("plane simplify dist : " + String(Variant(_plane_simplify_dist)));
}

bool Room::SimplifyInfo::add_plane_if_unique(LocalVector<Plane, int32_t> &r_planes, const Plane &p) const {
	for (int n = 0; n < r_planes.size(); n++) {
		const Plane &o = r_planes[n];

		// this is a fudge factor for how close planes can be to be considered the same ...
		// to prevent ridiculous amounts of planes
		const real_t d = _plane_simplify_dist; // 0.08f

		if (Math::abs(p.d - o.d) > d) {
			continue;
		}

		real_t dot = p.normal.dot(o.normal);
		if (dot < _plane_simplify_dot) // 0.98f
		{
			continue;
		}

		// match!
		return false;
	}

	r_planes.push_back(p);
	return true;
}

void Room::clear() {
	_room_ID = -1;
	_planes.clear();
	_preliminary_planes.clear();
	_roomgroups.clear();
	_portals.clear();
	_bound_mesh_data.edges.clear();
	_bound_mesh_data.faces.clear();
	_bound_mesh_data.vertices.clear();
	_aabb = AABB();
#ifdef TOOLS_ENABLED
	_gizmo_overlap_zones.clear();
#endif
}

Room::Room() {
	_room_rid = RID_PRIME(VisualServer::get_singleton()->room_create());
}

Room::~Room() {
	if (_room_rid != RID()) {
		VisualServer::get_singleton()->free(_room_rid);
	}
}

bool Room::contains_point(const Vector3 &p_pt) const {
	if (!_aabb.has_point(p_pt)) {
		return false;
	}

	for (int n = 0; n < _planes.size(); n++) {
		if (_planes[n].is_point_over(p_pt)) {
			return false;
		}
	}

	return true;
}

void Room::set_room_simplify(real_t p_value) {
	_simplify_info.set_simplify(p_value, _aabb.get_longest_axis_size());
}

void Room::set_use_default_simplify(bool p_use) {
	_use_default_simplify = p_use;
}

void Room::set_point(int p_idx, const Vector3 &p_point) {
	if (p_idx >= _bound_pts.size()) {
		return;
	}

	_bound_pts.set(p_idx, p_point);

#ifdef TOOLS_ENABLED
	_changed(true);
#endif
}

void Room::set_points(const PoolVector<Vector3> &p_points) {
	_bound_pts = p_points;

#ifdef TOOLS_ENABLED
	if (p_points.size()) {
		_changed(true);
	}
#endif
}

PoolVector<Vector3> Room::get_points() const {
	return _bound_pts;
}

PoolVector<Vector3> Room::generate_points() {
	PoolVector<Vector3> pts_returned;
#ifdef TOOLS_ENABLED
	// do a rooms convert to make sure the planes are up to date
	RoomManager *rm = RoomManager::active_room_manager;
	if (rm) {
		rm->rooms_convert();
	}

	if (!_planes.size()) {
		return pts_returned;
	}

	// scale an epsilon using 10.0 for a normal sized room
	real_t scaled_epsilon = _aabb.get_longest_axis_size() / 10.0;
	scaled_epsilon = MAX(scaled_epsilon * 0.01, 0.001);

	LocalVector<Vector3, int32_t> pts;
	pts = Geometry::compute_convex_mesh_points(&_planes[0], _planes.size(), scaled_epsilon);

	// eliminate duplicates
	for (int n = 0; n < pts.size(); n++) {
		const Vector3 &a = pts[n];

		for (int m = n + 1; m < pts.size(); m++) {
			const Vector3 &b = pts[m];
			if (a.is_equal_approx(b, scaled_epsilon)) {
				// remove b
				pts.remove_unordered(m);
				m--; // repeat m as the new m is the old last
			}
		}
	}

	// convert vector to poolvector
	pts_returned.resize(pts.size());

	Transform tr = get_global_transform();
	tr.affine_invert();

	for (int n = 0; n < pts.size(); n++) {
		// the points should be saved in LOCAL space,
		// so that if we move the room afterwards, the bound points
		// will also move in relation to the room.
		pts_returned.set(n, tr.xform(pts[n]));
	}

#endif
	return pts_returned;
}

String Room::get_configuration_warning() const {
	String warning = Spatial::get_configuration_warning();

	auto lambda = [](const Node *p_node) {
		return static_cast<bool>((Object::cast_to<Room>(p_node) || Object::cast_to<RoomManager>(p_node) || Object::cast_to<RoomGroup>(p_node)));
	};

	if (detect_nodes_using_lambda(this, lambda)) {
		if (detect_nodes_of_type<Room>(this)) {
			if (!warning.empty()) {
				warning += "\n\n";
			}
			warning += TTR("A Room cannot have another Room as a child or grandchild.");
		}

		if (detect_nodes_of_type<RoomManager>(this)) {
			if (!warning.empty()) {
				warning += "\n\n";
			}
			warning += TTR("The RoomManager should not be placed inside a Room.");
		}

		if (detect_nodes_of_type<RoomGroup>(this)) {
			if (!warning.empty()) {
				warning += "\n\n";
			}
			warning += TTR("A RoomGroup should not be placed inside a Room.");
		}
	}

	if (_planes.size() > 80) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("Room convex hull contains a large number of planes.\nConsider simplifying the room bound in order to increase performance.");
	}

	return warning;
}

// extra editor links to the room manager to allow unloading
// on change, or re-converting
void Room::_changed(bool p_regenerate_bounds) {
#ifdef TOOLS_ENABLED
	RoomManager *rm = RoomManager::active_room_manager;
	if (!rm) {
		return;
	}

	if (p_regenerate_bounds) {
		rm->_room_regenerate_bound(this);
	}
	rm->_rooms_changed("changed Room " + get_name());
#endif
}

void Room::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			ERR_FAIL_COND(get_world().is_null());
			VisualServer::get_singleton()->room_set_scenario(_room_rid, get_world()->get_scenario());
		} break;
		case NOTIFICATION_EXIT_WORLD: {
			VisualServer::get_singleton()->room_set_scenario(_room_rid, RID());
		} break;
	}
}

void Room::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_use_default_simplify", "p_use"), &Room::set_use_default_simplify);
	ClassDB::bind_method(D_METHOD("get_use_default_simplify"), &Room::get_use_default_simplify);

	ClassDB::bind_method(D_METHOD("set_room_simplify", "p_value"), &Room::set_room_simplify);
	ClassDB::bind_method(D_METHOD("get_room_simplify"), &Room::get_room_simplify);

	ClassDB::bind_method(D_METHOD("set_points", "points"), &Room::set_points);
	ClassDB::bind_method(D_METHOD("get_points"), &Room::get_points);

	ClassDB::bind_method(D_METHOD("set_point", "index", "position"), &Room::set_point);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_default_simplify"), "set_use_default_simplify", "get_use_default_simplify");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "room_simplify", PROPERTY_HINT_RANGE, "0.0,1.0,0.005"), "set_room_simplify", "get_room_simplify");

	ADD_GROUP("Bound", "");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR3_ARRAY, "points"), "set_points", "get_points");
}
