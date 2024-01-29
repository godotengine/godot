/**************************************************************************/
/*  room_manager.h                                                        */
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

#ifndef ROOM_MANAGER_H
#define ROOM_MANAGER_H

#include "core/local_vector.h"
#include "room.h"
#include "spatial.h"

class Portal;
class RoomGroup;
class MeshInstance;
class GeometryInstance;
class VisualInstance;

#define GODOT_PORTAL_WILDCARD ('*')

class RoomManager : public Spatial {
	GDCLASS(RoomManager, Spatial);

public:
	enum PVSMode {
		PVS_MODE_DISABLED,
		PVS_MODE_PARTIAL,
		PVS_MODE_FULL,
	};

	void set_roomlist_path(const NodePath &p_path);
	NodePath get_roomlist_path() const {
		return _settings_path_roomlist;
	}

	void set_preview_camera_path(const NodePath &p_path);

	NodePath get_preview_camera_path() const {
		return _settings_path_preview_camera;
	}

	void rooms_set_active(bool p_active);
	bool rooms_get_active() const;

	void set_show_margins(bool p_show);
	bool get_show_margins() const;

	void set_debug_sprawl(bool p_enable);
	bool get_debug_sprawl() const;

	void set_merge_meshes(bool p_enable);
	bool get_merge_meshes() const;

	void set_room_simplify(real_t p_value);
	real_t get_room_simplify() const;

	void set_default_portal_margin(real_t p_dist);
	real_t get_default_portal_margin() const;

	void set_overlap_warning_threshold(int p_value) { _overlap_warning_threshold = p_value; }
	int get_overlap_warning_threshold() const { return (int)_overlap_warning_threshold; }

	void set_portal_depth_limit(int p_limit);
	int get_portal_depth_limit() const { return _settings_portal_depth_limit; }

	void set_roaming_expansion_margin(real_t p_dist);
	real_t get_roaming_expansion_margin() const { return _settings_roaming_expansion_margin; }

	void set_pvs_mode(PVSMode p_mode);
	PVSMode get_pvs_mode() const;

	void set_pvs_filename(String p_filename);
	String get_pvs_filename() const;

	void set_use_secondary_pvs(bool p_enable) { _settings_use_secondary_pvs = p_enable; }
	bool get_use_secondary_pvs() const { return _settings_use_secondary_pvs; }

	void set_gameplay_monitor_enabled(bool p_enable) { _settings_gameplay_monitor_enabled = p_enable; }
	bool get_gameplay_monitor_enabled() const { return _settings_gameplay_monitor_enabled; }

	void rooms_convert();
	void rooms_clear();
	void rooms_flip_portals();

	String get_configuration_warning() const;

	// for internal use in the editor..
	// either we can clear the rooms and unload,
	// or reconvert.
	void _rooms_changed(String p_reason);

#ifdef TOOLS_ENABLED
	// for a preview, we allow the editor to change the bound
	bool _room_regenerate_bound(Room *p_room);
#endif

	RoomManager();
	~RoomManager();

	// an easy way of grabbing the active room manager for tools purposes
#ifdef TOOLS_ENABLED
	static RoomManager *active_room_manager;

	// static versions of functions for use from editor toolbars
	static void static_rooms_set_active(bool p_active);
	static bool static_rooms_get_active();
	static bool static_rooms_get_active_and_loaded();
	static void static_rooms_convert();
#endif

private:
	// funcs
	bool resolve_preview_camera_path();
	void _preview_camera_update();

	// conversion
	// FIRST PASS
	void _convert_rooms_recursive(Spatial *p_node, LocalVector<Portal *> &r_portals, LocalVector<RoomGroup *> &r_roomgroups, int p_roomgroup = -1);
	void _convert_room(Spatial *p_node, LocalVector<Portal *> &r_portals, const LocalVector<RoomGroup *> &p_roomgroups, int p_roomgroup);
	int _convert_roomgroup(Spatial *p_node, LocalVector<RoomGroup *> &r_roomgroups);

	void _find_portals_recursive(Spatial *p_node, Room *p_room, LocalVector<Portal *> &r_portals);
	void _convert_portal(Room *p_room, Spatial *p_node, LocalVector<Portal *> &portals);

	// SECOND PASS
	void _second_pass_portals(Spatial *p_roomlist, LocalVector<Portal *> &r_portals);
	void _second_pass_rooms(const LocalVector<RoomGroup *> &p_roomgroups, const LocalVector<Portal *> &p_portals);
	void _second_pass_room(Room *p_room, const LocalVector<RoomGroup *> &p_roomgroups, const LocalVector<Portal *> &p_portals);

	bool _convert_manual_bound(Room *p_room, Spatial *p_node, const LocalVector<Portal *> &p_portals);
	void _check_portal_for_warnings(Portal *p_portal, const AABB &p_room_aabb_without_portals);
	void _process_static(Room *p_room, Spatial *p_node, Vector<Vector3> &r_room_pts, bool p_add_to_portal_renderer, bool p_autoplaced);
	void _find_statics_recursive(Room *p_room, Spatial *p_node, Vector<Vector3> &r_room_pts, bool p_add_to_portal_renderer);
	bool _convert_room_hull_preliminary(Room *p_room, const Vector<Vector3> &p_room_pts, const LocalVector<Portal *> &p_portals);

	bool _bound_findpoints_mesh_instance(MeshInstance *p_mi, Vector<Vector3> &r_room_pts, AABB &r_aabb);
	bool _bound_findpoints_geom_instance(GeometryInstance *p_gi, Vector<Vector3> &r_room_pts, AABB &r_aabb);

	// THIRD PASS
	void _autolink_portals(Spatial *p_roomlist, LocalVector<Portal *> &r_portals);
	void _third_pass_rooms(const LocalVector<Portal *> &p_portals);

	bool _convert_room_hull_final(Room *p_room, const LocalVector<Portal *> &p_portals);
	void _build_simplified_bound(const Room *p_room, Geometry::MeshData &r_md, LocalVector<Plane, int32_t> &r_planes, int p_num_portal_planes);

	// AUTOPLACE - automatically place STATIC and DYNAMICs that are not within a room
	// into the most appropriate room, and sprawl
	void _autoplace_recursive(Spatial *p_node);
	bool _autoplace_object(VisualInstance *p_vi);

	// misc
	bool _add_plane_if_unique(const Room *p_room, LocalVector<Plane, int32_t> &r_planes, const Plane &p);
	void _update_portal_gizmos(Spatial *p_node);
	bool _check_roomlist_validity(Node *p_node);
	void _cleanup_after_conversion();
	Error _build_room_convex_hull(const Room *p_room, const Vector<Vector3> &p_points, Geometry::MeshData &r_mesh);
#ifdef TOOLS_ENABLED
	void _generate_room_overlap_zones();
#endif

	// merging
	void _merge_meshes_in_room(Room *p_room);
	void _list_mergeable_mesh_instances(Spatial *p_node, LocalVector<MeshInstance *, int32_t> &r_list);
	void _merge_log(String p_string) { debug_print_line(p_string); }
	bool _remove_redundant_dangling_nodes(Spatial *p_node);

	// helper funcs
	bool _name_ends_with(const Node *p_node, String p_postfix) const;
	template <class NODE_TYPE>
	NODE_TYPE *_resolve_path(NodePath p_path) const;
	template <class NODE_TYPE>
	bool _node_is_type(Node *p_node) const;
	template <class T>
	T *_change_node_type(Spatial *p_node, String p_prefix, bool p_delete = true);
	void _update_gizmos_recursive(Node *p_node);
	void _set_owner_recursive(Node *p_node, Node *p_owner);
	void _flip_portals_recursive(Spatial *p_node);
	Error _build_convex_hull(const Vector<Vector3> &p_points, Geometry::MeshData &r_mesh, real_t p_epsilon = 3.0 * UNIT_EPSILON);

	// output strings during conversion process
	void convert_log(String p_string, int p_priority = 0) { debug_print_line(p_string, 1); }

	// only prints when user has set 'debug' in the room manager inspector
	// also does not show in non editor builds
	void debug_print_line(String p_string, int p_priority = 0);
	void show_warning(const String &p_string, bool p_skippable = false, bool p_alert = true);

public:
	static String _find_name_before(Node *p_node, String p_postfix, bool p_allow_no_postfix = false);
	static real_t _get_default_portal_margin() { return _default_portal_margin; }

private:
	// accessible from UI
	NodePath _settings_path_roomlist;
	NodePath _settings_path_preview_camera;

	// resolved node
	Spatial *_roomlist = nullptr;
	bool _warning_misnamed_nodes_detected = false;
	bool _warning_portal_link_room_not_found = false;
	bool _warning_portal_autolink_failed = false;
	bool _warning_room_overlap_detected = false;

	// merge suitable meshes in rooms?
	bool _settings_merge_meshes = false;

	// remove redundant childless spatials after merging
	bool _settings_remove_danglers = true;

	bool _active = true;

	// portals, room hulls etc
	bool _show_debug = true;
	bool _debug_sprawl = false;

	// pvs
	PVSMode _pvs_mode = PVS_MODE_PARTIAL;
	String _pvs_filename;
	bool _settings_use_secondary_pvs = false;
	bool _settings_use_simple_pvs = false;
	bool _settings_log_pvs_generation = false;

	bool _settings_use_signals = true;
	bool _settings_gameplay_monitor_enabled = false;

	int _conversion_tick = 0;

	// just used during conversion, could be invalidated
	// later by user deleting rooms etc.
	LocalVector<Room *, int32_t> _rooms;

	// advanced params
	static real_t _default_portal_margin;
	real_t _overlap_warning_threshold = 1.0;
	Room::SimplifyInfo _room_simplify_info;
	int _settings_portal_depth_limit = 16;
	real_t _settings_roaming_expansion_margin = 1.0;

	// debug override camera
	ObjectID _godot_preview_camera_ID = -1;
	// local version of the godot camera frustum,
	// to prevent updating the visual server (and causing
	// a screen refresh) where not necessary.
	Vector3 _godot_camera_pos;
	Vector<Plane> _godot_camera_planes;

protected:
	static void _bind_methods();
	void _notification(int p_what);
	void _refresh_from_project_settings();
};

VARIANT_ENUM_CAST(RoomManager::PVSMode);

#endif // ROOM_MANAGER_H
