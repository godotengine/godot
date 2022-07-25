/*************************************************************************/
/*  portal.h                                                             */
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

#ifndef PORTAL_H
#define PORTAL_H

#include "core/local_vector.h"
#include "core/rid.h"
#include "spatial.h"

class RoomManager;
class MeshInstance;
class Room;

class Portal : public Spatial {
	GDCLASS(Portal, Spatial);

	RID _portal_rid;

	friend class RoomManager;
	friend class PortalGizmoPlugin;
	friend class PortalEditorPlugin;
	friend class PortalSpatialGizmo;

public:
	// ui interface .. will have no effect after room conversion
	void set_linked_room(const NodePath &link_path);
	NodePath get_linked_room() const;

	// open and close doors
	void set_portal_active(bool p_active);
	bool get_portal_active() const;

	// whether the portal can be seen through in both directions or not
	void set_two_way(bool p_two_way) {
		_settings_two_way = p_two_way;
		_changed();
	}
	bool is_two_way() const { return _settings_two_way; }

	// call during each conversion
	void clear();

	// whether to use the room manager default
	void set_use_default_margin(bool p_use);
	bool get_use_default_margin() const;

	// custom portal margin (per portal) .. only valid if use_default_margin is off
	void set_portal_margin(real_t p_margin);
	real_t get_portal_margin() const;

	// either the default margin or the custom portal margin, depending on the setting
	real_t get_active_portal_margin() const;

	// the raw points are used for the IDE Inspector, and also to allow the user
	// to edit the geometry of the portal at runtime (they can also just change the portal node transform)
	void set_points(const PoolVector<Vector2> &p_points);
	PoolVector<Vector2> get_points() const;

	// primarily for the gizmo
	void set_point(int p_idx, const Vector2 &p_point);

	String get_configuration_warning() const;

	Portal();
	~Portal();

	// whether the convention is that the normal of the portal points outward (false) or inward (true)
	// normally I'd recommend portal normal faces outward. But you may make a booboo, so this can work
	// with either convention.
	static bool _portal_plane_convention;

private:
	// updates world coords when the transform changes, and updates the visual server
	void portal_update();

	void set_linked_room_internal(const NodePath &link_path);
	bool try_set_unique_name(const String &p_name);
	bool is_portal_internal(int p_room_outer) const { return _internal && (_linkedroom_ID[0] != p_room_outer); }

	bool create_from_mesh_instance(const MeshInstance *p_mi);
	void flip();
	void _sanitize_points();
	void _update_aabb();
	static Vector3 _vec2to3(const Vector2 &p_pt) { return Vector3(p_pt.x, p_pt.y, 0.0); }
	void _sort_verts_clockwise(const Vector3 &p_portal_normal, Vector<Vector3> &r_verts);
	Plane _plane_from_points_newell(const Vector<Vector3> &p_pts);
	void resolve_links(const LocalVector<Room *, int32_t> &p_rooms, const RID &p_from_room_rid);
	void _changed();

	// nodepath to the room this outgoing portal leads to
	NodePath _settings_path_linkedroom;

	// portal can be turned on and off at runtime, for e.g.
	// opening and closing a door
	bool _settings_active;

	// user can choose not to include the portal in the convex hull of the room
	// during conversion
	bool _settings_include_in_bound;

	// portals can be seen through one way or two way
	bool _settings_two_way;

	// room from and to, ID in the room manager
	int _linkedroom_ID[2];

	// whether the portal is from a room within a room
	bool _internal;

	// normal determined by winding order
	Vector<Vector3> _pts_world;

	// points in local space of the plane,
	// not necessary in correct winding order
	// (as they can be edited by the user)
	// Note: these are saved by the IDE
	PoolVector<Vector2> _pts_local_raw;

	// sanitized
	Vector<Vector2> _pts_local;
	AABB _aabb_local;

	// center of the world points
	Vector3 _pt_center_world;

	// portal plane in world space, always pointing OUTWARD from the source room
	Plane _plane;

	// extension margin
	real_t _margin;
	bool _use_default_margin;

	// during conversion, we need to know
	// whether this portal is being imported from a mesh
	// and is using an explicitly named link room with prefix.
	// If this is not the case, and it is already a Godot Portal node,
	// we will either use the assigned nodepath, or autolink.
	bool _importing_portal = false;

	// for editing
#ifdef TOOLS_ENABLED
	ObjectID _room_manager_godot_ID;

	// warnings
	bool _warning_outside_room_aabb = false;
	bool _warning_facing_wrong_way = false;
	bool _warning_autolink_failed = false;
#endif

	// this is read from the gizmo
	static bool _settings_gizmo_show_margins;

public:
	// makes sure portals are not converted more than once per
	// call to rooms_convert
	int _conversion_tick = -1;

protected:
	static void _bind_methods();
	void _notification(int p_what);
};

#endif // PORTAL_H
