/*************************************************************************/
/*  room_manager.cpp                                                     */
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

#include "room_manager.h"

#include "core/bitfield_dynamic.h"
#include "core/engine.h"
#include "core/math/quick_hull.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "mesh_instance.h"
#include "multimesh_instance.h"
#include "portal.h"
#include "room_group.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/sprite_3d.h"
#include "visibility_notifier.h"

#ifdef TOOLS_ENABLED
#include "editor/plugins/spatial_editor_plugin.h"
#endif

#include "modules/modules_enabled.gen.h" // For csg.
#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif

// #define GODOT_PORTALS_USE_BULLET_CONVEX_HULL

#ifdef GODOT_PORTALS_USE_BULLET_CONVEX_HULL
#include "core/math/convex_hull.h"
#endif

// This needs to be static because it cannot easily be propagated to portals
// during load (as the RoomManager may be loaded before Portals enter the scene tree)
real_t RoomManager::_default_portal_margin = 1.0;

#ifdef TOOLS_ENABLED
RoomManager *RoomManager::active_room_manager = nullptr;

// static versions of functions for use from editor toolbars
void RoomManager::static_rooms_set_active(bool p_active) {
	if (active_room_manager) {
		active_room_manager->rooms_set_active(p_active);
		active_room_manager->property_list_changed_notify();
	}
}

bool RoomManager::static_rooms_get_active() {
	if (active_room_manager) {
		return active_room_manager->rooms_get_active();
	}

	return false;
}

bool RoomManager::static_rooms_get_active_and_loaded() {
	if (active_room_manager) {
		if (active_room_manager->rooms_get_active()) {
			Ref<World> world = active_room_manager->get_world();
			RID scenario = world->get_scenario();
			return active_room_manager->rooms_get_active() && VisualServer::get_singleton()->rooms_is_loaded(scenario);
		}
	}

	return false;
}

void RoomManager::static_rooms_convert() {
	if (active_room_manager) {
		return active_room_manager->rooms_convert();
	}
}
#endif

RoomManager::RoomManager() {
	// some high value, we want room manager to be processed after other
	// nodes because the camera should be moved first
	set_process_priority(10000);
}

RoomManager::~RoomManager() {
}

String RoomManager::get_configuration_warning() const {
	String warning = Spatial::get_configuration_warning();

	if (_settings_path_roomlist == NodePath()) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("The RoomList has not been assigned.");
	} else {
		Spatial *roomlist = _resolve_path<Spatial>(_settings_path_roomlist);
		if (!roomlist) {
			// possibly also check (roomlist->get_class_name() != StringName("Spatial"))
			if (!warning.empty()) {
				warning += "\n\n";
			}
			warning += TTR("The RoomList node should be a Spatial (or derived from Spatial).");
		}
	}

	if (_settings_portal_depth_limit == 0) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("Portal Depth Limit is set to Zero.\nOnly the Room that the Camera is in will render.");
	}

	if (Room::detect_nodes_of_type<RoomManager>(this)) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("There should only be one RoomManager in the SceneTree.");
	}

	return warning;
}

void RoomManager::_preview_camera_update() {
	Ref<World> world = get_world();
	RID scenario = world->get_scenario();

	if (_godot_preview_camera_ID != (ObjectID)-1) {
		Camera *cam = Object::cast_to<Camera>(ObjectDB::get_instance(_godot_preview_camera_ID));
		if (!cam) {
			_godot_preview_camera_ID = (ObjectID)-1;
		} else {
			// get camera position and direction
			Vector3 camera_pos = cam->get_global_transform().origin;
			Vector<Plane> planes = cam->get_frustum();

			// only update the visual server when there is a change.. as it will request a screen redraw
			// this is kinda silly, but the other way would be keeping track of the override camera in visual server
			// and tracking the camera deletes, which might be more error prone for a debug feature...
			bool changed = false;
			if (camera_pos != _godot_camera_pos) {
				changed = true;
			}
			// check planes
			if (!changed) {
				if (planes.size() != _godot_camera_planes.size()) {
					changed = true;
				}
			}

			if (!changed) {
				// num of planes must be identical
				for (int n = 0; n < planes.size(); n++) {
					if (planes[n] != _godot_camera_planes[n]) {
						changed = true;
						break;
					}
				}
			}

			if (changed) {
				_godot_camera_pos = camera_pos;
				_godot_camera_planes = planes;
				VisualServer::get_singleton()->rooms_override_camera(scenario, true, camera_pos, &planes);
			}
		}
	}
}

void RoomManager::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint()) {
				set_process_internal(_godot_preview_camera_ID != (ObjectID)-1);
#ifdef TOOLS_ENABLED
				// note this mechanism may fail to work correctly if the user creates two room managers,
				// but should not create major problems as it is just used to auto update when portals etc
				// are changed in the editor, and there is a check for nullptr.
				active_room_manager = this;
				SpatialEditor *spatial_editor = SpatialEditor::get_singleton();
				if (spatial_editor) {
					spatial_editor->update_portal_tools();
				}
#endif
			} else {
				if (_settings_gameplay_monitor_enabled) {
					set_process_internal(true);
				}
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
#ifdef TOOLS_ENABLED
			active_room_manager = nullptr;
			if (Engine::get_singleton()->is_editor_hint()) {
				SpatialEditor *spatial_editor = SpatialEditor::get_singleton();
				if (spatial_editor) {
					spatial_editor->update_portal_tools();
				}
			}
#endif
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			// can't call visual server if not inside world
			if (!is_inside_world()) {
				return;
			}

			if (Engine::get_singleton()->is_editor_hint()) {
				_preview_camera_update();
				return;
			}

			if (_settings_gameplay_monitor_enabled) {
				Ref<World> world = get_world();
				RID scenario = world->get_scenario();

				List<Camera *> cameras;
				world->get_camera_list(&cameras);

				Vector<Vector3> positions;

				for (int n = 0; n < cameras.size(); n++) {
					positions.push_back(cameras[n]->get_global_transform().origin);
				}

				VisualServer::get_singleton()->rooms_update_gameplay_monitor(scenario, positions);
			}

		} break;
	}
}

void RoomManager::_bind_methods() {
	BIND_ENUM_CONSTANT(PVS_MODE_DISABLED);
	BIND_ENUM_CONSTANT(PVS_MODE_PARTIAL);
	BIND_ENUM_CONSTANT(PVS_MODE_FULL);

	// main functions
	ClassDB::bind_method(D_METHOD("rooms_convert"), &RoomManager::rooms_convert);
	ClassDB::bind_method(D_METHOD("rooms_clear"), &RoomManager::rooms_clear);

	ClassDB::bind_method(D_METHOD("set_pvs_mode", "pvs_mode"), &RoomManager::set_pvs_mode);
	ClassDB::bind_method(D_METHOD("get_pvs_mode"), &RoomManager::get_pvs_mode);

	ClassDB::bind_method(D_METHOD("set_roomlist_path", "p_path"), &RoomManager::set_roomlist_path);
	ClassDB::bind_method(D_METHOD("get_roomlist_path"), &RoomManager::get_roomlist_path);

	// These are commented out for now, but available in case we want to cache PVS to disk, the functionality exists
	// ClassDB::bind_method(D_METHOD("set_pvs_filename", "pvs_filename"), &RoomManager::set_pvs_filename);
	// ClassDB::bind_method(D_METHOD("get_pvs_filename"), &RoomManager::get_pvs_filename);

	// just some macros to make setting inspector values easier
#define LPORTAL_STRINGIFY(x) #x
#define LPORTAL_TOSTRING(x) LPORTAL_STRINGIFY(x)

#define LIMPL_PROPERTY(P_TYPE, P_NAME, P_SET, P_GET)                                                        \
	ClassDB::bind_method(D_METHOD(LPORTAL_TOSTRING(P_SET), LPORTAL_TOSTRING(P_NAME)), &RoomManager::P_SET); \
	ClassDB::bind_method(D_METHOD(LPORTAL_TOSTRING(P_GET)), &RoomManager::P_GET);                           \
	ADD_PROPERTY(PropertyInfo(P_TYPE, LPORTAL_TOSTRING(P_NAME)), LPORTAL_TOSTRING(P_SET), LPORTAL_TOSTRING(P_GET));

#define LIMPL_PROPERTY_RANGE(P_TYPE, P_NAME, P_SET, P_GET, P_RANGE_STRING)                                  \
	ClassDB::bind_method(D_METHOD(LPORTAL_TOSTRING(P_SET), LPORTAL_TOSTRING(P_NAME)), &RoomManager::P_SET); \
	ClassDB::bind_method(D_METHOD(LPORTAL_TOSTRING(P_GET)), &RoomManager::P_GET);                           \
	ADD_PROPERTY(PropertyInfo(P_TYPE, LPORTAL_TOSTRING(P_NAME), PROPERTY_HINT_RANGE, P_RANGE_STRING), LPORTAL_TOSTRING(P_SET), LPORTAL_TOSTRING(P_GET));

	ADD_GROUP("Main", "");
	LIMPL_PROPERTY(Variant::BOOL, active, rooms_set_active, rooms_get_active);
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "roomlist", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Spatial"), "set_roomlist_path", "get_roomlist_path");

	ADD_GROUP("PVS", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "pvs_mode", PROPERTY_HINT_ENUM, "Disabled,Partial,Full"), "set_pvs_mode", "get_pvs_mode");
	// ADD_PROPERTY(PropertyInfo(Variant::STRING, "pvs_filename", PROPERTY_HINT_FILE, "*.pvs"), "set_pvs_filename", "get_pvs_filename");

	ADD_GROUP("Gameplay", "");
	LIMPL_PROPERTY(Variant::BOOL, gameplay_monitor, set_gameplay_monitor_enabled, get_gameplay_monitor_enabled);
	LIMPL_PROPERTY(Variant::BOOL, use_secondary_pvs, set_use_secondary_pvs, get_use_secondary_pvs);

	ADD_GROUP("Optimize", "");
	LIMPL_PROPERTY(Variant::BOOL, merge_meshes, set_merge_meshes, get_merge_meshes);

	ADD_GROUP("Debug", "");
	LIMPL_PROPERTY(Variant::BOOL, show_margins, set_show_margins, get_show_margins);
	LIMPL_PROPERTY(Variant::BOOL, debug_sprawl, set_debug_sprawl, get_debug_sprawl);
	LIMPL_PROPERTY_RANGE(Variant::INT, overlap_warning_threshold, set_overlap_warning_threshold, get_overlap_warning_threshold, "1,1000,1");
	LIMPL_PROPERTY(Variant::NODE_PATH, preview_camera, set_preview_camera_path, get_preview_camera_path);

	ADD_GROUP("Advanced", "");
	LIMPL_PROPERTY_RANGE(Variant::INT, portal_depth_limit, set_portal_depth_limit, get_portal_depth_limit, "0,255,1");
	LIMPL_PROPERTY_RANGE(Variant::REAL, room_simplify, set_room_simplify, get_room_simplify, "0.0,1.0,0.005");
	LIMPL_PROPERTY_RANGE(Variant::REAL, default_portal_margin, set_default_portal_margin, get_default_portal_margin, "0.0, 10.0, 0.01");
	LIMPL_PROPERTY_RANGE(Variant::REAL, roaming_expansion_margin, set_roaming_expansion_margin, get_roaming_expansion_margin, "0.0, 3.0, 0.01");

#undef LIMPL_PROPERTY
#undef LIMPL_PROPERTY_RANGE
#undef LPORTAL_STRINGIFY
#undef LPORTAL_TOSTRING
}

void RoomManager::_refresh_from_project_settings() {
	_settings_use_simple_pvs = GLOBAL_GET("rendering/portals/pvs/use_simple_pvs");
	_settings_log_pvs_generation = GLOBAL_GET("rendering/portals/pvs/pvs_logging");
	_settings_use_signals = GLOBAL_GET("rendering/portals/gameplay/use_signals");
	_settings_remove_danglers = GLOBAL_GET("rendering/portals/optimize/remove_danglers");
	_show_debug = GLOBAL_GET("rendering/portals/debug/logging");
	Portal::_portal_plane_convention = GLOBAL_GET("rendering/portals/advanced/flip_imported_portals");

	// force not to show logs when not in editor
	if (!Engine::get_singleton()->is_editor_hint()) {
		_show_debug = false;
		_settings_log_pvs_generation = false;
	}
}

void RoomManager::set_roomlist_path(const NodePath &p_path) {
	_settings_path_roomlist = p_path;
	update_configuration_warning();
}

void RoomManager::set_preview_camera_path(const NodePath &p_path) {
	_settings_path_preview_camera = p_path;

	resolve_preview_camera_path();

	bool camera_on = _godot_preview_camera_ID != (ObjectID)-1;

	// make sure the cached camera planes are invalid, this will
	// force an update to the visual server on the next internal_process
	_godot_camera_planes.clear();

	// if in the editor, turn processing on or off
	// according to whether the camera is overridden
	if (Engine::get_singleton()->is_editor_hint()) {
		if (is_inside_tree()) {
			set_process_internal(camera_on);
		}
	}

	// if we are turning camera override off, must inform visual server
	if (!camera_on && is_inside_world() && get_world().is_valid() && get_world()->get_scenario().is_valid()) {
		VisualServer::get_singleton()->rooms_override_camera(get_world()->get_scenario(), false, Vector3(), nullptr);
	}

	// we couldn't resolve the path, let's set it to null
	if (!camera_on) {
		_settings_path_preview_camera = NodePath();
	}
}

void RoomManager::set_room_simplify(real_t p_value) {
	_room_simplify_info.set_simplify(p_value);
}

real_t RoomManager::get_room_simplify() const {
	return _room_simplify_info._plane_simplify;
}

void RoomManager::set_portal_depth_limit(int p_limit) {
	_settings_portal_depth_limit = p_limit;

	if (is_inside_world() && get_world().is_valid()) {
		VisualServer::get_singleton()->rooms_set_params(get_world()->get_scenario(), p_limit, _settings_roaming_expansion_margin);
	}
}

void RoomManager::set_roaming_expansion_margin(real_t p_dist) {
	_settings_roaming_expansion_margin = p_dist;

	if (is_inside_world() && get_world().is_valid()) {
		VisualServer::get_singleton()->rooms_set_params(get_world()->get_scenario(), _settings_portal_depth_limit, _settings_roaming_expansion_margin);
	}
}

void RoomManager::set_default_portal_margin(real_t p_dist) {
	_default_portal_margin = p_dist;

	// send to portals
	Spatial *roomlist = _resolve_path<Spatial>(_settings_path_roomlist);
	if (!roomlist) {
		return;
	}

	_update_portal_gizmos(roomlist);
}

void RoomManager::_update_portal_gizmos(Spatial *p_node) {
	Portal *portal = Object::cast_to<Portal>(p_node);

	if (portal) {
		portal->update_gizmo();
	}

	// recurse
	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));

		if (child) {
			_update_portal_gizmos(child);
		}
	}
}

real_t RoomManager::get_default_portal_margin() const {
	return _default_portal_margin;
}

void RoomManager::set_show_margins(bool p_show) {
	Portal::_settings_gizmo_show_margins = p_show;

	Spatial *roomlist = _resolve_path<Spatial>(_settings_path_roomlist);
	if (!roomlist) {
		return;
	}

	_update_gizmos_recursive(roomlist);
}

bool RoomManager::get_show_margins() const {
	return Portal::_settings_gizmo_show_margins;
}

void RoomManager::set_debug_sprawl(bool p_enable) {
	if (is_inside_world() && get_world().is_valid()) {
		VisualServer::get_singleton()->rooms_set_debug_feature(get_world()->get_scenario(), VisualServer::ROOMS_DEBUG_SPRAWL, p_enable);
		_debug_sprawl = p_enable;
	}
}

bool RoomManager::get_debug_sprawl() const {
	return _debug_sprawl;
}

void RoomManager::set_merge_meshes(bool p_enable) {
	_settings_merge_meshes = p_enable;
}

bool RoomManager::get_merge_meshes() const {
	return _settings_merge_meshes;
}

void RoomManager::show_warning(const String &p_string, const String &p_extra_string, bool p_alert) {
	if (p_extra_string != "") {
		WARN_PRINT(p_string + " " + p_extra_string);
#ifdef TOOLS_ENABLED
		if (p_alert && Engine::get_singleton()->is_editor_hint()) {
			EditorNode::get_singleton()->show_warning(TTRGET(p_string) + "\n" + TTRGET(p_extra_string));
		}
#endif
	} else {
		WARN_PRINT(p_string);
		// OS::get_singleton()->alert(p_string, p_title);
#ifdef TOOLS_ENABLED
		if (p_alert && Engine::get_singleton()->is_editor_hint()) {
			EditorNode::get_singleton()->show_warning(TTRGET(p_string));
		}
#endif
	}
}

void RoomManager::debug_print_line(String p_string, int p_priority) {
	if (_show_debug) {
		if (!p_priority) {
			print_verbose(p_string);
		} else {
			print_line(p_string);
		}
	}
}

void RoomManager::rooms_set_active(bool p_active) {
	if (is_inside_world() && get_world().is_valid()) {
		VisualServer::get_singleton()->rooms_set_active(get_world()->get_scenario(), p_active);
		_active = p_active;

#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			SpatialEditor *spatial_editor = SpatialEditor::get_singleton();
			if (spatial_editor) {
				spatial_editor->update_portal_tools();
			}
		}
#endif
	}
}

bool RoomManager::rooms_get_active() const {
	return _active;
}

void RoomManager::set_pvs_mode(PVSMode p_mode) {
	_pvs_mode = p_mode;
}

RoomManager::PVSMode RoomManager::get_pvs_mode() const {
	return _pvs_mode;
}

void RoomManager::set_pvs_filename(String p_filename) {
	_pvs_filename = p_filename;
}

String RoomManager::get_pvs_filename() const {
	return _pvs_filename;
}

void RoomManager::_rooms_changed(String p_reason) {
	_rooms.clear();
	if (is_inside_world() && get_world().is_valid()) {
		VisualServer::get_singleton()->rooms_unload(get_world()->get_scenario(), p_reason);
	}
}

void RoomManager::rooms_clear() {
	_rooms.clear();
	if (is_inside_world() && get_world().is_valid()) {
		VisualServer::get_singleton()->rooms_and_portals_clear(get_world()->get_scenario());
	}
}

void RoomManager::rooms_flip_portals() {
	// this is a helper emergency function to deal with situations where the user has ended up with Portal nodes
	// pointing in the wrong direction (by doing initial conversion with flip_portal_meshes set incorrectly).
	_roomlist = _resolve_path<Spatial>(_settings_path_roomlist);
	if (!_roomlist) {
		WARN_PRINT("Cannot resolve nodepath");
		show_warning(TTR("RoomList path is invalid.\nPlease check the RoomList branch has been assigned in the RoomManager."));
		return;
	}

	_flip_portals_recursive(_roomlist);
	_rooms_changed("flipped Portals");
}

void RoomManager::rooms_convert() {
	// set all error conditions to false
	_warning_misnamed_nodes_detected = false;
	_warning_portal_link_room_not_found = false;
	_warning_portal_autolink_failed = false;
	_warning_room_overlap_detected = false;

	_refresh_from_project_settings();

	_roomlist = _resolve_path<Spatial>(_settings_path_roomlist);
	if (!_roomlist) {
		WARN_PRINT("Cannot resolve nodepath");
		show_warning(TTR("RoomList path is invalid.\nPlease check the RoomList branch has been assigned in the RoomManager."));
		return;
	}

	ERR_FAIL_COND(!is_inside_world() || !get_world().is_valid());

	// every time we run convert we increment this,
	// to prevent individual rooms / portals being converted
	// more than once in one run
	_conversion_tick++;

	rooms_clear();

	// first check that the roomlist is valid, and the user hasn't made
	// a silly scene tree
	if (!_check_roomlist_validity(_roomlist)) {
		return;
	}

	LocalVector<Portal *> portals;
	LocalVector<RoomGroup *> roomgroups;

	// find the rooms and portals
	_convert_rooms_recursive(_roomlist, portals, roomgroups);

	if (!_rooms.size()) {
		rooms_clear();
		show_warning(TTR("RoomList contains no Rooms, aborting."));
		return;
	}

	// add portal links
	_second_pass_portals(_roomlist, portals);

	// create the statics
	_second_pass_rooms(roomgroups, portals);

	// third pass

	// autolink portals that are not already manually linked
	// and finalize the portals
	_autolink_portals(_roomlist, portals);

	// Find the statics AGAIN and only this time add them to the PortalRenderer.
	// We need to do this twice because the room points determine the room bound...
	// but the bound is needed for autolinking,
	// and the autolinking needs to be done BEFORE adding to the PortalRenderer so that
	// the static objects will correctly sprawl. It is a chicken and egg situation.
	// Also finalize the room hulls.
	_third_pass_rooms(portals);

	// now we run autoplace to place any statics that have not been explicitly placed in rooms.
	// These will by definition not affect the room bounds, but is convenient for users to edit
	// levels in a more freeform manner
	_autoplace_recursive(_roomlist);

	bool generate_pvs = false;
	bool pvs_cull = false;
	switch (_pvs_mode) {
		default: {
		} break;
		case PVS_MODE_PARTIAL: {
			generate_pvs = true;
		} break;
		case PVS_MODE_FULL: {
			generate_pvs = true;
			pvs_cull = true;
		} break;
	}

	VisualServer::get_singleton()->rooms_finalize(get_world()->get_scenario(), generate_pvs, pvs_cull, _settings_use_secondary_pvs, _settings_use_signals, _pvs_filename, _settings_use_simple_pvs, _settings_log_pvs_generation);

	// refresh portal depth limit
	set_portal_depth_limit(get_portal_depth_limit());

#ifdef TOOLS_ENABLED
	_generate_room_overlap_zones();
#endif

	// just delete any intermediate data
	_cleanup_after_conversion();

	// display error dialogs
	if (_warning_misnamed_nodes_detected) {
		show_warning(TTR("Misnamed nodes detected, check output log for details. Aborting."));
		rooms_clear();
	}

	if (_warning_portal_link_room_not_found) {
		show_warning(TTR("Portal link room not found, check output log for details."));
	}

	if (_warning_portal_autolink_failed) {
		show_warning(TTR("Portal autolink failed, check output log for details.\nCheck the portal is facing outwards from the source room."));
	}

	if (_warning_room_overlap_detected) {
		show_warning(TTR("Room overlap detected, cameras may work incorrectly in overlapping area.\nCheck output log for details."));
	}
}

void RoomManager::_second_pass_room(Room *p_room, const LocalVector<RoomGroup *> &p_roomgroups, const LocalVector<Portal *> &p_portals) {
	if (_settings_merge_meshes) {
		_merge_meshes_in_room(p_room);
	}

	// find statics and manual bound
	bool manual_bound_found = false;

	// points making up the room geometry, in world space, to create the convex hull
	Vector<Vector3> room_pts;

	for (int n = 0; n < p_room->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_room->get_child(n));

		if (child) {
			if (_node_is_type<Portal>(child) || child->is_queued_for_deletion()) {
				// the adding of portal points is done after this stage, because
				// we need to take into account incoming as well as outgoing portals
			} else if (_name_ends_with(child, "-bound")) {
				manual_bound_found = _convert_manual_bound(p_room, child, p_portals);
			} else {
				// don't add the instances to the portal renderer on the first pass of _find_statics,
				// just find the geometry points in order to make sure the bound is correct.
				_find_statics_recursive(p_room, child, room_pts, false);
			}
		}
	}

	// Has the bound been specified using points in the room?
	// in that case, overwrite the room_pts
	if (p_room->_bound_pts.size() && p_room->is_inside_tree()) {
		Transform tr = p_room->get_global_transform();

		room_pts.clear();
		room_pts.resize(p_room->_bound_pts.size());
		for (int n = 0; n < room_pts.size(); n++) {
			room_pts.set(n, tr.xform(p_room->_bound_pts[n]));
		}

		// we override and manual bound with the room points
		manual_bound_found = false;
	}

	if (!manual_bound_found) {
		// rough aabb for checking portals for warning conditions
		AABB aabb;
		aabb.create_from_points(room_pts);

		for (int n = 0; n < p_room->_portals.size(); n++) {
			int portal_id = p_room->_portals[n];
			Portal *portal = p_portals[portal_id];

			// only checking portals out from source room
			if (portal->_linkedroom_ID[0] != p_room->_room_ID) {
				continue;
			}

			// don't add portals to the world bound that are internal to this room!
			if (portal->is_portal_internal(p_room->_room_ID)) {
				continue;
			}

			// check portal for suspect conditions, like a long way from the room AABB,
			// or possibly flipped the wrong way
			_check_portal_for_warnings(portal, aabb);
		}

		// create convex hull
		_convert_room_hull_preliminary(p_room, room_pts, p_portals);
	}

	// add the room to roomgroups
	for (int n = 0; n < p_room->_roomgroups.size(); n++) {
		int roomgroup_id = p_room->_roomgroups[n];
		p_roomgroups[roomgroup_id]->add_room(p_room);
	}
}

void RoomManager::_second_pass_rooms(const LocalVector<RoomGroup *> &p_roomgroups, const LocalVector<Portal *> &p_portals) {
	for (int n = 0; n < _rooms.size(); n++) {
		_second_pass_room(_rooms[n], p_roomgroups, p_portals);
	}
}

#ifdef TOOLS_ENABLED
void RoomManager::_generate_room_overlap_zones() {
	for (int n = 0; n < _rooms.size(); n++) {
		Room *room = _rooms[n];

		// no planes .. no overlap
		if (!room->_planes.size()) {
			continue;
		}

		for (int c = n + 1; c < _rooms.size(); c++) {
			if (c == n) {
				continue;
			}
			Room *other = _rooms[c];

			// do a quick reject AABB
			if (!room->_aabb.intersects(other->_aabb) || (!other->_planes.size())) {
				continue;
			}

			// if the room priorities are different (i.e. an internal room), they are allowed to overlap
			if (room->_room_priority != other->_room_priority) {
				continue;
			}

			// get all the planes of both rooms in a contiguous list
			LocalVector<Plane, int32_t> planes;
			planes.resize(room->_planes.size() + other->_planes.size());
			Plane *dest = planes.ptr();
			memcpy(dest, &room->_planes[0], room->_planes.size() * sizeof(Plane));
			dest += room->_planes.size();

			memcpy(dest, &other->_planes[0], other->_planes.size() * sizeof(Plane));

			Vector<Vector3> overlap_pts = Geometry::compute_convex_mesh_points(planes.ptr(), planes.size());

			if (overlap_pts.size() < 4) {
				continue;
			}

			// there is an overlap, create a mesh from the points
			Geometry::MeshData md;
			Error err = _build_convex_hull(overlap_pts, md);

			if (err != OK) {
				WARN_PRINT("QuickHull failed building room overlap hull");
				continue;
			}

			// only if the volume is more than some threshold
			real_t volume = Geometry::calculate_convex_hull_volume(md);
			if (volume > _overlap_warning_threshold) {
				WARN_PRINT("Room overlap of " + String(Variant(volume)) + " detected between " + room->get_name() + " and " + other->get_name());
				room->_gizmo_overlap_zones.push_back(md);
				_warning_room_overlap_detected = true;
			}
		}
	}
}
#endif

void RoomManager::_third_pass_rooms(const LocalVector<Portal *> &p_portals) {
	bool found_errors = false;

	for (int n = 0; n < _rooms.size(); n++) {
		Room *room = _rooms[n];

		// no need to do all these string operations if we are not debugging and don't need logs
		if (_show_debug) {
			String room_short_name = _find_name_before(room, "-room", true);
			convert_log("ROOM\t" + room_short_name);

			// log output the portals associated with this room
			for (int p = 0; p < room->_portals.size(); p++) {
				const Portal &portal = *p_portals[room->_portals[p]];

				bool portal_links_out = portal._linkedroom_ID[0] == room->_room_ID;

				int linked_room_id = (portal_links_out) ? portal._linkedroom_ID[1] : portal._linkedroom_ID[0];

				// this shouldn't be out of range, but just in case
				if ((linked_room_id >= 0) && (linked_room_id < _rooms.size())) {
					Room *linked_room = _rooms[linked_room_id];

					String portal_link_room_name = _find_name_before(linked_room, "-room", true);
					String in_or_out = (portal_links_out) ? "POUT" : "PIN ";

					// display the name of the room linked to
					convert_log("\t\t" + in_or_out + "\t" + portal_link_room_name);
				} else {
					WARN_PRINT_ONCE("linked_room_id is out of range");
				}
			}

		} // if _show_debug

		// do a second pass finding the statics, where they are
		// finally added to the rooms in the portal_renderer.
		Vector<Vector3> room_pts;

		// the true indicates we DO want to add to the portal renderer this second time
		// we call _find_statics_recursive
		_find_statics_recursive(room, room, room_pts, true);

		if (!_convert_room_hull_final(room, p_portals)) {
			found_errors = true;
		}
		room->update_gizmo();
		room->update_configuration_warning();
	}

	if (found_errors) {
		show_warning(TTR("Error calculating room bounds.\nEnsure all rooms contain geometry or manual bounds."));
	}
}

void RoomManager::_second_pass_portals(Spatial *p_roomlist, LocalVector<Portal *> &r_portals) {
	for (unsigned int n = 0; n < r_portals.size(); n++) {
		Portal *portal = r_portals[n];

		// we have a choice here.
		// If we are importing, we will try linking using the naming convention method.
		// We do this by setting the assigned nodepath if we find the link room, then
		// the resolving links is done in the usual manner from the nodepath.
		if (portal->_importing_portal) {
			String string_link_room_shortname = _find_name_before(portal, "-portal");
			String string_link_room = string_link_room_shortname + "-room";

			if (string_link_room_shortname != "") {
				// try the room name plus the postfix first, this will be the most common case during import
				Room *linked_room = Object::cast_to<Room>(p_roomlist->find_node(string_link_room, true, false));

				// try the short name as a last ditch attempt
				if (!linked_room) {
					linked_room = Object::cast_to<Room>(p_roomlist->find_node(string_link_room_shortname, true, false));
				}

				if (linked_room) {
					NodePath path = portal->get_path_to(linked_room);
					portal->set_linked_room_internal(path);
				} else {
					WARN_PRINT("Portal link room : " + string_link_room + " not found.");
					_warning_portal_link_room_not_found = true;
				}
			}
		}

		// get the room we are linking from
		int room_from_id = portal->_linkedroom_ID[0];
		if (room_from_id != -1) {
			Room *room_from = _rooms[room_from_id];
			portal->resolve_links(_rooms, room_from->_room_rid);

			// add the portal id to the room from and the room to.
			// These are used so we can later add the portal geometry to the room bounds.
			room_from->_portals.push_back(n);

			int room_to_id = portal->_linkedroom_ID[1];
			if (room_to_id != -1) {
				Room *room_to = _rooms[room_to_id];
				room_to->_portals.push_back(n);

				// make the portal internal if necessary
				portal->_internal = room_from->_room_priority > room_to->_room_priority;
			}
		}
	}
}

void RoomManager::_autolink_portals(Spatial *p_roomlist, LocalVector<Portal *> &r_portals) {
	for (unsigned int n = 0; n < r_portals.size(); n++) {
		Portal *portal = r_portals[n];

		// all portals should have a source room
		DEV_ASSERT(portal->_linkedroom_ID[0] != -1);
		const Room *source_room = _rooms[portal->_linkedroom_ID[0]];

		if (portal->_linkedroom_ID[1] != -1) {
			// already manually linked
			continue;
		}

		bool autolink_found = false;

		// try to autolink
		// try points iteratively out from the portal center and find the first that is in a room that isn't the source room
		for (int attempt = 0; attempt < 4; attempt++) {
			// found
			if (portal->_linkedroom_ID[1] != -1) {
				break;
			}

			// these numbers are arbitrary .. we could alternatively reuse the portal margins for this?
			real_t dist = 0.01;
			switch (attempt) {
				default: {
					dist = 0.01;
				} break;
				case 1: {
					dist = 0.1;
				} break;
				case 2: {
					dist = 1.0;
				} break;
				case 3: {
					dist = 2.0;
				} break;
			}

			Vector3 test_pos = portal->_pt_center_world + (dist * portal->_plane.normal);

			int best_priority = -1000;
			int best_room = -1;

			for (int r = 0; r < _rooms.size(); r++) {
				Room *room = _rooms[r];
				if (room->_room_ID == portal->_linkedroom_ID[0]) {
					// can't link back to the source room
					continue;
				}

				// first do a rough aabb check
				if (!room->_aabb.has_point(test_pos)) {
					continue;
				}

				bool outside = false;
				for (int p = 0; p < room->_preliminary_planes.size(); p++) {
					const Plane &plane = room->_preliminary_planes[p];
					if (plane.distance_to(test_pos) > 0.0) {
						outside = true;
						break;
					}
				} // for through planes

				if (!outside) {
					// we found a suitable room, but we want the highest priority in
					// case there are internal rooms...
					if (room->_room_priority > best_priority) {
						best_priority = room->_room_priority;
						best_room = r;
					}
				}

			} // for through rooms

			// found a suitable link room
			if (best_room != -1) {
				Room *room = _rooms[best_room];

				// great, we found a linked room!
				convert_log("\t\tAUTOLINK OK from " + source_room->get_name() + " to " + room->get_name(), 1);
				portal->_linkedroom_ID[1] = best_room;

				// add the portal to the portals list for the receiving room
				room->_portals.push_back(n);

				// send complete link to visual server so the portal will be active in the visual server room system
				VisualServer::get_singleton()->portal_link(portal->_portal_rid, source_room->_room_rid, room->_room_rid, portal->_settings_two_way);

				// make the portal internal if necessary
				// (this prevents the portal plane clipping the room bound)
				portal->_internal = source_room->_room_priority > room->_room_priority;

				autolink_found = true;
				break;
			}

		} // for attempt

		// error condition
		if (!autolink_found) {
			WARN_PRINT("Portal AUTOLINK failed for " + portal->get_name() + " from " + source_room->get_name());
			_warning_portal_autolink_failed = true;

#ifdef TOOLS_ENABLED
			portal->_warning_autolink_failed = true;
			portal->update_gizmo();
#endif
		}
	} // for portal
}

// to prevent users creating mistakes for themselves, we limit what can be put into the room list branch.
// returns invalid node, or NULL
bool RoomManager::_check_roomlist_validity(Node *p_node) {
	// restrictions lifted here, but we can add more if required
	return true;
}

void RoomManager::_convert_rooms_recursive(Spatial *p_node, LocalVector<Portal *> &r_portals, LocalVector<RoomGroup *> &r_roomgroups, int p_roomgroup) {
	// is this a room?
	if (_node_is_type<Room>(p_node) || _name_ends_with(p_node, "-room")) {
		_convert_room(p_node, r_portals, r_roomgroups, p_roomgroup);
	}

	// is this a roomgroup?
	if (_node_is_type<RoomGroup>(p_node) || _name_ends_with(p_node, "-roomgroup")) {
		p_roomgroup = _convert_roomgroup(p_node, r_roomgroups);
	}

	// recurse through children
	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));

		if (child) {
			_convert_rooms_recursive(child, r_portals, r_roomgroups, p_roomgroup);
		}
	}
}

int RoomManager::_convert_roomgroup(Spatial *p_node, LocalVector<RoomGroup *> &r_roomgroups) {
	String string_full_name = p_node->get_name();

	// is it already a roomgroup?
	RoomGroup *roomgroup = Object::cast_to<RoomGroup>(p_node);

	// if not already a RoomGroup, convert the node and move all children
	if (!roomgroup) {
		// create a RoomGroup
		roomgroup = _change_node_type<RoomGroup>(p_node, "G");
	} else {
		// already hit this tick?
		if (roomgroup->_conversion_tick == _conversion_tick) {
			return roomgroup->_roomgroup_ID;
		}
	}

	convert_log("convert_roomgroup : " + string_full_name, 1);

	// make sure the roomgroup is blank, especially if already created
	roomgroup->clear();

	// make sure the object ID is sent to the visual server
	VisualServer::get_singleton()->roomgroup_prepare(roomgroup->_room_group_rid, roomgroup->get_instance_id());

	// mark so as only to convert once
	roomgroup->_conversion_tick = _conversion_tick;

	roomgroup->_roomgroup_ID = r_roomgroups.size();

	r_roomgroups.push_back(roomgroup);

	return r_roomgroups.size() - 1;
}

void RoomManager::_convert_room(Spatial *p_node, LocalVector<Portal *> &r_portals, const LocalVector<RoomGroup *> &p_roomgroups, int p_roomgroup) {
	String string_full_name = p_node->get_name();

	// is it already an lroom?
	Room *room = Object::cast_to<Room>(p_node);

	// if not already a Room, convert the node and move all children
	if (!room) {
		// create a Room
		room = _change_node_type<Room>(p_node, "G");
	} else {
		// already hit this tick?
		if (room->_conversion_tick == _conversion_tick) {
			return;
		}
	}

	// make sure the room is blank, especially if already created
	room->clear();

	// mark so as only to convert once
	room->_conversion_tick = _conversion_tick;

	// set roomgroup
	if (p_roomgroup != -1) {
		room->_roomgroups.push_back(p_roomgroup);
		room->_room_priority = p_roomgroups[p_roomgroup]->_settings_priority;

		VisualServer::get_singleton()->room_prepare(room->_room_rid, room->_room_priority);
	}

	// add to the list of rooms
	room->_room_ID = _rooms.size();
	_rooms.push_back(room);

	_find_portals_recursive(room, room, r_portals);
}

void RoomManager::_find_portals_recursive(Spatial *p_node, Room *p_room, LocalVector<Portal *> &r_portals) {
	MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);
	if (_node_is_type<Portal>(p_node) || (mi && _name_ends_with(mi, "-portal"))) {
		_convert_portal(p_room, p_node, r_portals);
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));

		if (child) {
			_find_portals_recursive(child, p_room, r_portals);
		}
	}
}

void RoomManager::_check_portal_for_warnings(Portal *p_portal, const AABB &p_room_aabb_without_portals) {
#ifdef TOOLS_ENABLED
	AABB bb = p_room_aabb_without_portals;
	bb = bb.grow(bb.get_longest_axis_size() * 0.5);

	bool changed = false;

	// far outside the room?
	const Vector3 &pos = p_portal->get_global_transform().origin;

	bool old_outside = p_portal->_warning_outside_room_aabb;
	p_portal->_warning_outside_room_aabb = !bb.has_point(pos);

	if (p_portal->_warning_outside_room_aabb != old_outside) {
		changed = true;
	}

	if (p_portal->_warning_outside_room_aabb) {
		WARN_PRINT(String(p_portal->get_name()) + " possibly in the wrong room.");
	}

	// facing wrong way?
	Vector3 offset = pos - bb.get_center();
	real_t dot = offset.dot(p_portal->_plane.normal);

	bool old_facing = p_portal->_warning_facing_wrong_way;
	p_portal->_warning_facing_wrong_way = dot < 0.0;

	if (p_portal->_warning_facing_wrong_way != old_facing) {
		changed = true;
	}

	if (p_portal->_warning_facing_wrong_way) {
		WARN_PRINT(String(p_portal->get_name()) + " possibly facing the wrong way.");
	}

	// handled later
	p_portal->_warning_autolink_failed = false;

	if (changed) {
		p_portal->update_gizmo();
	}
#endif
}

bool RoomManager::_autoplace_object(VisualInstance *p_vi) {
	// note we could alternatively use the portal_renderer to do this more efficiently
	// (as it has a BSP) but at a cost of returning result from the visual server
	AABB bb = p_vi->get_transformed_aabb();
	Vector3 centre = bb.get_center();

	// in order to deal with internal rooms, we can't just stop at the first
	// room the point is within, as there could be later rooms with a higher priority
	int best_priority = -INT32_MAX;
	Room *best_room = nullptr;

	// if not set to zero (no preference) this can override a preference
	// for a certain RoomGroup priority to ensure the instance gets placed in the correct
	// RoomGroup (e.g. outside, for building exteriors)
	int preferred_priority = p_vi->get_portal_autoplace_priority();

	for (int n = 0; n < _rooms.size(); n++) {
		Room *room = _rooms[n];

		if (room->contains_point(centre)) {
			// the standard routine autoplaces in the highest priority room
			if (room->_room_priority > best_priority) {
				best_priority = room->_room_priority;
				best_room = room;
			}

			// if we override the preferred priority we always choose this
			if (preferred_priority && (room->_room_priority == preferred_priority)) {
				best_room = room;
				break;
			}
		}
	}

	if (best_room) {
		// just dummies, we won't use these this time
		Vector<Vector3> room_pts;

		// we can reuse this function
		_process_static(best_room, p_vi, room_pts, true);
		return true;
	}

	return false;
}

void RoomManager::_autoplace_recursive(Spatial *p_node) {
	if (p_node->is_queued_for_deletion()) {
		return;
	}

	// as soon as we hit a room, quit the recursion as the objects
	// will already have been added inside rooms
	if (Object::cast_to<Room>(p_node)) {
		return;
	}

	VisualInstance *vi = Object::cast_to<VisualInstance>(p_node);

	// we are only interested in VIs with static or dynamic mode
	if (vi) {
		switch (vi->get_portal_mode()) {
			default: {
			} break;
			case CullInstance::PORTAL_MODE_DYNAMIC:
			case CullInstance::PORTAL_MODE_STATIC: {
				_autoplace_object(vi);
			} break;
		}
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));

		if (child) {
			_autoplace_recursive(child);
		}
	}
}

void RoomManager::_process_static(Room *p_room, Spatial *p_node, Vector<Vector3> &r_room_pts, bool p_add_to_portal_renderer) {
	bool ignore = false;
	VisualInstance *vi = Object::cast_to<VisualInstance>(p_node);

	// we are only interested in VIs with static or dynamic mode
	if (vi) {
		switch (vi->get_portal_mode()) {
			default: {
				ignore = true;
			} break;
			case CullInstance::PORTAL_MODE_DYNAMIC:
			case CullInstance::PORTAL_MODE_STATIC:
				break;
		}
	}

	if (!ignore) {
		// We'll have a done flag. This isn't strictly speaking necessary
		// because the types should be mutually exclusive, but this would
		// break if something changes the inheritance hierarchy of the nodes
		// at a later date, so having a done flag makes it more robust.
		bool done = false;

		Light *light = Object::cast_to<Light>(p_node);

		if (!done && light) {
			done = true;

			// lights (don't affect bound, so aren't added in first pass)
			if (p_add_to_portal_renderer) {
				Vector<Vector3> dummy_pts;
				VisualServer::get_singleton()->room_add_instance(p_room->_room_rid, light->get_instance(), light->get_transformed_aabb(), dummy_pts);
				convert_log("\t\t\tLIGT\t" + light->get_name());
			}
		}

		GeometryInstance *gi = Object::cast_to<GeometryInstance>(p_node);

		if (!done && gi) {
			done = true;

			// MeshInstance is the most interesting type for portalling, so we handle this explicitly
			MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);
			if (mi) {
				if (p_add_to_portal_renderer) {
					convert_log("\t\t\tMESH\t" + mi->get_name());
				}

				Vector<Vector3> object_pts;
				AABB aabb;
				// get the object points and don't immediately add to the room
				// points, as we want to use these points for sprawling algorithm in
				// the visual server.
				if (_bound_findpoints_mesh_instance(mi, object_pts, aabb)) {
					// need to keep track of room bound
					// NOTE the is_visible check MAY cause problems if conversion run on nodes that
					// aren't properly in the tree. It can optionally be removed. Certainly calling is_visible_in_tree
					// DID cause problems.
					if (mi->get_include_in_bound() && mi->is_visible()) {
						r_room_pts.append_array(object_pts);
					}

					if (p_add_to_portal_renderer) {
						VisualServer::get_singleton()->room_add_instance(p_room->_room_rid, mi->get_instance(), mi->get_transformed_aabb(), object_pts);
					}
				} // if bound found points
			} else {
				// geometry instance but not a mesh instance ..
				if (p_add_to_portal_renderer) {
					convert_log("\t\t\tGEOM\t" + gi->get_name());
				}

				Vector<Vector3> object_pts;
				AABB aabb;

				// attempt to recognise this GeometryInstance and read back the geometry
				if (_bound_findpoints_geom_instance(gi, object_pts, aabb)) {
					// need to keep track of room bound
					// NOTE the is_visible check MAY cause problems if conversion run on nodes that
					// aren't properly in the tree. It can optionally be removed. Certainly calling is_visible_in_tree
					// DID cause problems.
					if (gi->get_include_in_bound() && gi->is_visible()) {
						r_room_pts.append_array(object_pts);
					}

					if (p_add_to_portal_renderer) {
						VisualServer::get_singleton()->room_add_instance(p_room->_room_rid, gi->get_instance(), gi->get_transformed_aabb(), object_pts);
					}
				} // if bound found points
			}
		} // if gi

		VisibilityNotifier *vn = Object::cast_to<VisibilityNotifier>(p_node);
		if (!done && vn && ((vn->get_portal_mode() == CullInstance::PORTAL_MODE_DYNAMIC) || (vn->get_portal_mode() == CullInstance::PORTAL_MODE_STATIC))) {
			done = true;

			if (p_add_to_portal_renderer) {
				AABB world_aabb = vn->get_global_transform().xform(vn->get_aabb());
				VisualServer::get_singleton()->room_add_ghost(p_room->_room_rid, vn->get_instance_id(), world_aabb);
				convert_log("\t\t\tVIS \t" + vn->get_name());
			}
		}

	} // if not ignore
}

void RoomManager::_find_statics_recursive(Room *p_room, Spatial *p_node, Vector<Vector3> &r_room_pts, bool p_add_to_portal_renderer) {
	// don't process portal MeshInstances that are being deleted
	// (and replaced by proper Portal nodes)
	if (p_node->is_queued_for_deletion()) {
		return;
	}

	_process_static(p_room, p_node, r_room_pts, p_add_to_portal_renderer);

	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));

		if (child) {
			_find_statics_recursive(p_room, child, r_room_pts, p_add_to_portal_renderer);
		}
	}
}

bool RoomManager::_convert_manual_bound(Room *p_room, Spatial *p_node, const LocalVector<Portal *> &p_portals) {
	MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);
	if (!mi) {
		return false;
	}

	Vector<Vector3> points;
	AABB aabb;
	if (!_bound_findpoints_mesh_instance(mi, points, aabb)) {
		return false;
	}

	mi->set_portal_mode(CullInstance::PORTAL_MODE_IGNORE);

	// hide bounds after conversion
	// set to portal mode ignore?
	mi->hide();

	return _convert_room_hull_preliminary(p_room, points, p_portals);
}

bool RoomManager::_convert_room_hull_preliminary(Room *p_room, const Vector<Vector3> &p_room_pts, const LocalVector<Portal *> &p_portals) {
	if (p_room_pts.size() <= 3) {
		return false;
	}

	Geometry::MeshData md;

	Error err = OK;

	// if there are too many room points, quickhull will fail or freeze etc, so we will revert
	// to a bounding rect and send an error message
	if (p_room_pts.size() > 100000) {
		WARN_PRINT(String(p_room->get_name()) + " contains too many vertices to find convex hull, use a manual bound instead.");

		AABB aabb;
		aabb.create_from_points(p_room_pts);

		LocalVector<Vector3> pts;
		Vector3 mins = aabb.position;
		Vector3 maxs = mins + aabb.size;

		pts.push_back(Vector3(mins.x, mins.y, mins.z));
		pts.push_back(Vector3(mins.x, maxs.y, mins.z));
		pts.push_back(Vector3(maxs.x, maxs.y, mins.z));
		pts.push_back(Vector3(maxs.x, mins.y, mins.z));
		pts.push_back(Vector3(mins.x, mins.y, maxs.z));
		pts.push_back(Vector3(mins.x, maxs.y, maxs.z));
		pts.push_back(Vector3(maxs.x, maxs.y, maxs.z));
		pts.push_back(Vector3(maxs.x, mins.y, maxs.z));

		err = _build_convex_hull(pts, md);
	} else {
		err = _build_room_convex_hull(p_room, p_room_pts, md);
	}

	if (err != OK) {
		return false;
	}

	// add any existing portals planes first, as these will trump any other existing planes further out
	for (int n = 0; n < p_room->_portals.size(); n++) {
		int portal_id = p_room->_portals[n];
		Portal *portal = p_portals[portal_id];

		// don't add portals to the hull that are internal to this room!
		if (portal->is_portal_internal(p_room->_room_ID)) {
			continue;
		}

		Plane plane = portal->_plane;

		// does it need to be reversed? (i.e. is the portal incoming rather than outgoing)
		if (portal->_linkedroom_ID[1] == p_room->_room_ID) {
			plane = -plane;
		}

		_add_plane_if_unique(p_room, p_room->_preliminary_planes, plane);
	}

	// add the planes from the geometry or manual bound
	for (int n = 0; n < md.faces.size(); n++) {
		const Plane &p = md.faces[n].plane;
		_add_plane_if_unique(p_room, p_room->_preliminary_planes, p);
	}

	// temporary copy of mesh data for the boundary points
	// to form a new hull in _convert_room_hull_final
	p_room->_bound_mesh_data = md;

	// aabb (should later include portals too, these are added in _convert_room_hull_final)
	p_room->_aabb.create_from_points(md.vertices);

	return true;
}

bool RoomManager::_convert_room_hull_final(Room *p_room, const LocalVector<Portal *> &p_portals) {
	Vector<Vector3> vertices_including_portals = p_room->_bound_mesh_data.vertices;

	// add the portals planes first, as these will trump any other existing planes further out
	int num_portals_added = 0;

	for (int n = 0; n < p_room->_portals.size(); n++) {
		int portal_id = p_room->_portals[n];
		Portal *portal = p_portals[portal_id];

		// don't add portals to the world bound that are internal to this room!
		if (portal->is_portal_internal(p_room->_room_ID)) {
			continue;
		}

		Plane plane = portal->_plane;

		// does it need to be reversed? (i.e. is the portal incoming rather than outgoing)
		if (portal->_linkedroom_ID[1] == p_room->_room_ID) {
			plane = -plane;
		}

		if (_add_plane_if_unique(p_room, p_room->_planes, plane)) {
			num_portals_added++;
		}

		// add any new portals to the aabb of the room
		for (int p = 0; p < portal->_pts_world.size(); p++) {
			const Vector3 &pt = portal->_pts_world[p];
			vertices_including_portals.push_back(pt);
			p_room->_aabb.expand_to(pt);
		}
	}

	// create new convex hull
	Geometry::MeshData md;
	Error err = _build_room_convex_hull(p_room, vertices_including_portals, md);

	if (err != OK) {
		return false;
	}

	// add the planes from the new hull
	for (int n = 0; n < md.faces.size(); n++) {
		const Plane &p = md.faces[n].plane;
		_add_plane_if_unique(p_room, p_room->_planes, p);
	}

	// recreate the points within the new simplified bound, and then recreate the convex hull
	// by running quickhull a second time... (this enables the gizmo to accurately show the simplified hull)
	int num_planes_before_simplification = p_room->_planes.size();
	Geometry::MeshData md_simplified;
	_build_simplified_bound(p_room, md_simplified, p_room->_planes, num_portals_added);

	if (num_planes_before_simplification != p_room->_planes.size()) {
		convert_log("\t\t\tcontained " + itos(num_planes_before_simplification) + " planes before simplification, " + itos(p_room->_planes.size()) + " planes after.");
	}

	// make a copy of the mesh data for debugging
	// note this could be avoided in release builds? NYI
	p_room->_bound_mesh_data = md_simplified;

	// send bound to visual server
	VisualServer::get_singleton()->room_set_bound(p_room->_room_rid, p_room->get_instance_id(), p_room->_planes, p_room->_aabb, md_simplified.vertices);

	return true;
}

#ifdef TOOLS_ENABLED
bool RoomManager::_room_regenerate_bound(Room *p_room) {
	// for a preview, we allow the editor to change the bound
	ERR_FAIL_COND_V(!p_room, false);

	if (!p_room->_bound_pts.size()) {
		return false;
	}

	// can't do yet if not in the tree
	if (!p_room->is_inside_tree()) {
		return false;
	}

	Transform tr = p_room->get_global_transform();

	Vector<Vector3> pts;
	pts.resize(p_room->_bound_pts.size());
	for (int n = 0; n < pts.size(); n++) {
		pts.set(n, tr.xform(p_room->_bound_pts[n]));
	}

	Geometry::MeshData md;
	Error err = _build_room_convex_hull(p_room, pts, md);

	if (err != OK) {
		return false;
	}

	p_room->_bound_mesh_data = md;
	p_room->update_gizmo();

	return true;
}
#endif

void RoomManager::_build_simplified_bound(const Room *p_room, Geometry::MeshData &r_md, LocalVector<Plane, int32_t> &r_planes, int p_num_portal_planes) {
	if (!r_planes.size()) {
		return;
	}

	Vector<Vector3> pts = Geometry::compute_convex_mesh_points(&r_planes[0], r_planes.size(), 0.001);
	Error err = _build_room_convex_hull(p_room, pts, r_md);

	if (err != OK) {
		WARN_PRINT("QuickHull failed building simplified bound");
		return;
	}

	// if the number of faces is less than the number of planes, we can use this simplified version to reduce the number of planes
	if (r_md.faces.size() < r_planes.size()) {
		// always include the portal planes
		r_planes.resize(p_num_portal_planes);

		for (int n = 0; n < r_md.faces.size(); n++) {
			_add_plane_if_unique(p_room, r_planes, r_md.faces[n].plane);
		}
	}
}

Error RoomManager::_build_room_convex_hull(const Room *p_room, const Vector<Vector3> &p_points, Geometry::MeshData &r_mesh) {
	// calculate an epsilon based on the simplify value, and use this to build the hull
	real_t s = 0.0;

	DEV_ASSERT(p_room);
	if (p_room->_use_default_simplify) {
		s = _room_simplify_info._plane_simplify;
	} else {
		s = p_room->_simplify_info._plane_simplify;
	}

	// value between  0.3 (accurate) and 10.0 (very rough)
	// * UNIT_EPSILON
	s *= s;
	s *= 40.0;
	s += 0.3; // minimum
	s *= UNIT_EPSILON;
	return _build_convex_hull(p_points, r_mesh, s);
}

bool RoomManager::_add_plane_if_unique(const Room *p_room, LocalVector<Plane, int32_t> &r_planes, const Plane &p) {
	DEV_ASSERT(p_room);
	if (p_room->_use_default_simplify) {
		return _room_simplify_info.add_plane_if_unique(r_planes, p);
	}

	return p_room->_simplify_info.add_plane_if_unique(r_planes, p);
}

void RoomManager::_convert_portal(Room *p_room, Spatial *p_node, LocalVector<Portal *> &portals) {
	Portal *portal = Object::cast_to<Portal>(p_node);

	bool importing = false;

	// if not a gportal already, convert the node type
	if (!portal) {
		importing = true;
		portal = _change_node_type<Portal>(p_node, "G", false);
		portal->create_from_mesh_instance(Object::cast_to<MeshInstance>(p_node));

		p_node->queue_delete();

	} else {
		// only allow converting once
		if (portal->_conversion_tick == _conversion_tick) {
			return;
		}
	}

	// make sure to start with fresh internal data each time (for linked rooms etc)
	portal->clear();

	// mark the portal if we are importing, because we will need to use the naming
	// prefix system to look for linked rooms in that case
	portal->_importing_portal = importing;

	// mark so as only to convert once
	portal->_conversion_tick = _conversion_tick;

	// link rooms
	portal->portal_update();

	// keep a list of portals for second pass
	portals.push_back(portal);

	// the portal  is linking from this first room it is added to
	portal->_linkedroom_ID[0] = p_room->_room_ID;
}

bool RoomManager::_bound_findpoints_geom_instance(GeometryInstance *p_gi, Vector<Vector3> &r_room_pts, AABB &r_aabb) {
	// max opposite extents .. note AABB storing size is rubbish in this aspect
	// it can fail once mesh min is larger than FLT_MAX / 2.
	r_aabb.position = Vector3(FLT_MAX / 2, FLT_MAX / 2, FLT_MAX / 2);
	r_aabb.size = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

#ifdef MODULE_CSG_ENABLED
	CSGShape *shape = Object::cast_to<CSGShape>(p_gi);
	if (shape) {
		// Shapes will not be up to date on the first frame due to a quirk
		// of CSG - it defers updates to the next frame. So we need to explicitly
		// force an update to make sure the CSG is correct on level load.
		shape->force_update_shape();

		Array arr = shape->get_meshes();
		if (!arr.size()) {
			return false;
		}

		Ref<ArrayMesh> arr_mesh = arr[1];
		if (!arr_mesh.is_valid()) {
			return false;
		}

		if (arr_mesh->get_surface_count() == 0) {
			return false;
		}

		// for converting meshes to world space
		Transform trans = p_gi->get_global_transform();

		for (int surf = 0; surf < arr_mesh->get_surface_count(); surf++) {
			Array arrays = arr_mesh->surface_get_arrays(surf);

			if (!arrays.size()) {
				continue;
			}

			PoolVector<Vector3> vertices = arrays[VS::ARRAY_VERTEX];

			// convert to world space
			for (int n = 0; n < vertices.size(); n++) {
				Vector3 pt_world = trans.xform(vertices[n]);
				r_room_pts.push_back(pt_world);

				// keep the bound up to date
				r_aabb.expand_to(pt_world);
			}

		} // for through the surfaces

		return true;
	} // if csg shape
#endif

	// multimesh
	MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(p_gi);
	if (mmi) {
		Ref<MultiMesh> rmm = mmi->get_multimesh();
		if (!rmm.is_valid()) {
			return false;
		}

		// first get the mesh verts in local space
		LocalVector<Vector3, int32_t> local_verts;
		Ref<Mesh> rmesh = rmm->get_mesh();

		if (rmesh->get_surface_count() == 0) {
			String string;
			string = "MultiMeshInstance '" + mmi->get_name() + "' has no surfaces, ignoring";
			WARN_PRINT(string);
			return false;
		}

		for (int surf = 0; surf < rmesh->get_surface_count(); surf++) {
			Array arrays = rmesh->surface_get_arrays(surf);

			if (!arrays.size()) {
				WARN_PRINT_ONCE("MultiMesh mesh surface with no mesh, ignoring");
				continue;
			}

			const PoolVector<Vector3> &vertices = arrays[VS::ARRAY_VERTEX];

			int count = local_verts.size();
			local_verts.resize(local_verts.size() + vertices.size());

			for (int n = 0; n < vertices.size(); n++) {
				local_verts[count++] = vertices[n];
			}
		}

		if (!local_verts.size()) {
			return false;
		}

		// now we have the local space verts, add a bunch for each instance, and find the AABB
		for (int i = 0; i < rmm->get_instance_count(); i++) {
			Transform trans = rmm->get_instance_transform(i);
			trans = mmi->get_global_transform() * trans;

			for (int n = 0; n < local_verts.size(); n++) {
				Vector3 pt_world = trans.xform(local_verts[n]);
				r_room_pts.push_back(pt_world);

				// keep the bound up to date
				r_aabb.expand_to(pt_world);
			}
		}
		return true;
	}

	// Sprite3D
	SpriteBase3D *sprite = Object::cast_to<SpriteBase3D>(p_gi);
	if (sprite) {
		Ref<TriangleMesh> tmesh = sprite->generate_triangle_mesh();
		PoolVector<Vector3> vertices = tmesh->get_vertices();

		// for converting meshes to world space
		Transform trans = p_gi->get_global_transform();

		// convert to world space
		for (int n = 0; n < vertices.size(); n++) {
			Vector3 pt_world = trans.xform(vertices[n]);
			r_room_pts.push_back(pt_world);

			// keep the bound up to date
			r_aabb.expand_to(pt_world);
		}

		return true;
	}

	return false;
}

bool RoomManager::_bound_findpoints_mesh_instance(MeshInstance *p_mi, Vector<Vector3> &r_room_pts, AABB &r_aabb) {
	// max opposite extents .. note AABB storing size is rubbish in this aspect
	// it can fail once mesh min is larger than FLT_MAX / 2.
	r_aabb.position = Vector3(FLT_MAX / 2, FLT_MAX / 2, FLT_MAX / 2);
	r_aabb.size = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	// some godot jiggery pokery to get the mesh verts in local space
	Ref<Mesh> rmesh = p_mi->get_mesh();

	ERR_FAIL_COND_V(!rmesh.is_valid(), false);

	if (rmesh->get_surface_count() == 0) {
		String string;
		string = "MeshInstance '" + p_mi->get_name() + "' has no surfaces, ignoring";
		WARN_PRINT(string);
		return false;
	}

	bool success = false;

	// for converting meshes to world space
	Transform trans = p_mi->get_global_transform();

	for (int surf = 0; surf < rmesh->get_surface_count(); surf++) {
		Array arrays = rmesh->surface_get_arrays(surf);

		// possible to have a meshinstance with no geometry .. don't want to crash
		if (!arrays.size()) {
			WARN_PRINT_ONCE("MeshInstance surface with no mesh, ignoring");
			continue;
		}

		success = true;

		PoolVector<Vector3> vertices = arrays[VS::ARRAY_VERTEX];

		// convert to world space
		for (int n = 0; n < vertices.size(); n++) {
			Vector3 ptWorld = trans.xform(vertices[n]);
			r_room_pts.push_back(ptWorld);

			// keep the bound up to date
			r_aabb.expand_to(ptWorld);
		}

	} // for through the surfaces

	return success;
}

void RoomManager::_cleanup_after_conversion() {
	for (int n = 0; n < _rooms.size(); n++) {
		Room *room = _rooms[n];
		room->_portals.reset();
		room->_preliminary_planes.reset();

		// outside the editor, there's no need to keep the data for the convex hull
		// drawing, as it is only used for gizmos.
		if (!Engine::get_singleton()->is_editor_hint()) {
			room->_bound_mesh_data = Geometry::MeshData();
		}
	}
}

bool RoomManager::resolve_preview_camera_path() {
	Camera *camera = _resolve_path<Camera>(_settings_path_preview_camera);

	if (camera) {
		_godot_preview_camera_ID = camera->get_instance_id();
		return true;
	}
	_godot_preview_camera_ID = -1;
	return false;
}

template <class NODE_TYPE>
NODE_TYPE *RoomManager::_resolve_path(NodePath p_path) const {
	if (has_node(p_path)) {
		NODE_TYPE *node = Object::cast_to<NODE_TYPE>(get_node(p_path));
		if (node) {
			return node;
		} else {
			WARN_PRINT("node is incorrect type");
		}
	}

	return nullptr;
}

template <class NODE_TYPE>
bool RoomManager::_node_is_type(Node *p_node) const {
	NODE_TYPE *node = Object::cast_to<NODE_TYPE>(p_node);
	return node != nullptr;
}

template <class T>
T *RoomManager::_change_node_type(Spatial *p_node, String p_prefix, bool p_delete) {
	String string_full_name = p_node->get_name();

	Node *parent = p_node->get_parent();
	if (!parent) {
		return nullptr;
	}

	// owner should normally be root
	Node *owner = p_node->get_owner();

	// change the name of the node to be deleted
	p_node->set_name(p_prefix + string_full_name);

	// create the new class T object
	T *pNew = memnew(T);
	pNew->set_name(string_full_name);

	// add the child at the same position as the old node
	// (this is more convenient for users)
	parent->add_child_below_node(p_node, pNew);

	// new lroom should have same transform
	pNew->set_transform(p_node->get_transform());

	// move each child
	while (p_node->get_child_count()) {
		Node *child = p_node->get_child(0);
		p_node->remove_child(child);

		// needs to set owner to appear in IDE
		pNew->add_child(child);
	}

	// needs to set owner to appear in IDE
	_set_owner_recursive(pNew, owner);

	// delete old node
	if (p_delete) {
		p_node->queue_delete();
	}

	return pNew;
}

void RoomManager::_update_gizmos_recursive(Node *p_node) {
	Portal *portal = Object::cast_to<Portal>(p_node);

	if (portal) {
		portal->update_gizmo();
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		_update_gizmos_recursive(p_node->get_child(n));
	}
}

Error RoomManager::_build_convex_hull(const Vector<Vector3> &p_points, Geometry::MeshData &r_mesh, real_t p_epsilon) {
#ifdef GODOT_PORTALS_USE_BULLET_CONVEX_HULL
	return ConvexHullComputer::convex_hull(p_points, r_mesh);
#if 0
	// test comparison of methods
	QuickHull::build(p_points, r_mesh, p_epsilon);

	int qh_faces = r_mesh.faces.size();
	int qh_verts = r_mesh.vertices.size();

	r_mesh.vertices.clear();
	r_mesh.faces.clear();
	r_mesh.edges.clear();
	Error err = ConvexHullComputer::convex_hull(p_points, r_mesh);

	int bh_faces = r_mesh.faces.size();
	int bh_verts = r_mesh.vertices.size();

	if (qh_faces != bh_faces) {
		print_line("qh_faces : " + itos(qh_faces) + ", bh_faces : " + itos(bh_faces));
	}
	if (qh_verts != bh_verts) {
		print_line("qh_verts : " + itos(qh_verts) + ", bh_verts : " + itos(bh_verts));
	}

	return err;
#endif

#else
	QuickHull::_flag_warnings = false;
	Error err = QuickHull::build(p_points, r_mesh, p_epsilon);
	QuickHull::_flag_warnings = true;
	return err;
#endif
}

void RoomManager::_flip_portals_recursive(Spatial *p_node) {
	Portal *portal = Object::cast_to<Portal>(p_node);

	if (portal) {
		portal->flip();
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));
		if (child) {
			_flip_portals_recursive(child);
		}
	}
}

void RoomManager::_set_owner_recursive(Node *p_node, Node *p_owner) {
	if (p_node != p_owner) {
		p_node->set_owner(p_owner);
	}
	for (int n = 0; n < p_node->get_child_count(); n++) {
		_set_owner_recursive(p_node->get_child(n), p_owner);
	}
}

bool RoomManager::_name_ends_with(const Node *p_node, String p_postfix) const {
	ERR_FAIL_NULL_V(p_node, false);
	String name = p_node->get_name();

	int pf_l = p_postfix.length();
	int l = name.length();

	if (pf_l > l) {
		return false;
	}

	// allow capitalization errors
	if (name.substr(l - pf_l, pf_l).to_lower() == p_postfix) {
		return true;
	}

	return false;
}

String RoomManager::_find_name_before(Node *p_node, String p_postfix, bool p_allow_no_postfix) {
	ERR_FAIL_NULL_V(p_node, String());
	String name = p_node->get_name();

	int pf_l = p_postfix.length();
	int l = name.length();

	if (pf_l > l) {
		if (!p_allow_no_postfix) {
			return String();
		}
	} else {
		if (name.substr(l - pf_l, pf_l) == p_postfix) {
			name = name.substr(0, l - pf_l);
		} else {
			if (!p_allow_no_postfix) {
				return String();
			}
		}
	}

	// because godot doesn't support multiple nodes with the same name, we will strip e.g. a number
	// after an * on the end of the name...
	// e.g. kitchen*2-portal
	for (int c = 0; c < name.length(); c++) {
		if (name[c] == GODOT_PORTAL_WILDCARD) {
			// remove everything after and including this character
			name = name.substr(0, c);
			break;
		}
	}

	return name;
}

void RoomManager::_merge_meshes_in_room(Room *p_room) {
	// only do in running game so as not to lose data
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	_merge_log("merging room " + p_room->get_name());

	// list of meshes suitable
	LocalVector<MeshInstance *, int32_t> source_meshes;
	_list_mergeable_mesh_instances(p_room, source_meshes);

	// none suitable
	if (!source_meshes.size()) {
		return;
	}

	_merge_log("\t" + itos(source_meshes.size()) + " source meshes");

	BitFieldDynamic bf;
	bf.create(source_meshes.size(), true);

	for (int n = 0; n < source_meshes.size(); n++) {
		LocalVector<MeshInstance *, int32_t> merge_list;

		// find similar meshes
		MeshInstance *a = source_meshes[n];
		merge_list.push_back(a);

		// may not be necessary
		bf.set_bit(n, true);

		for (int c = n + 1; c < source_meshes.size(); c++) {
			// if not merged already
			if (!bf.get_bit(c)) {
				MeshInstance *b = source_meshes[c];

				//				if (_are_meshes_mergeable(a, b)) {
				if (a->is_mergeable_with(*b)) {
					merge_list.push_back(b);
					bf.set_bit(c, true);
				}
			} // if not merged already
		} // for c through secondary mesh

		// only merge if more than 1
		if (merge_list.size() > 1) {
			// we can merge!
			// create a new holder mesh

			MeshInstance *merged = memnew(MeshInstance);
			merged->set_name("MergedMesh");

			_merge_log("\t\t" + merged->get_name());

			if (merged->create_by_merging(merge_list)) {
				// set all the source meshes to portal mode ignore so not shown
				for (int i = 0; i < merge_list.size(); i++) {
					merge_list[i]->set_portal_mode(CullInstance::PORTAL_MODE_IGNORE);
				}

				// and set the new merged mesh to static
				merged->set_portal_mode(CullInstance::PORTAL_MODE_STATIC);

				// attach to scene tree
				p_room->add_child(merged);
				merged->set_owner(p_room->get_owner());

				// compensate for room transform, as the verts are now in world space
				Transform tr = p_room->get_global_transform();
				tr.affine_invert();
				merged->set_transform(tr);

				// delete originals?
				// note this isn't perfect, it may still end up with dangling spatials, but they can be
				// deleted later.
				for (int i = 0; i < merge_list.size(); i++) {
					MeshInstance *mi = merge_list[i];
					if (!mi->get_child_count()) {
						mi->queue_delete();
					} else {
						Node *parent = mi->get_parent();
						if (parent) {
							// if there are children, we don't want to delete it, but we do want to
							// remove the mesh drawing, e.g. by replacing it with a spatial
							String name = mi->get_name();
							mi->set_name("DeleteMe"); // can be anything, just to avoid name conflict with replacement node
							Spatial *replacement = memnew(Spatial);
							replacement->set_name(name);

							parent->add_child(replacement);

							// make the transform and owner match
							replacement->set_owner(mi->get_owner());
							replacement->set_transform(mi->get_transform());

							// move all children from the mesh instance to the replacement
							while (mi->get_child_count()) {
								Node *child = mi->get_child(0);
								mi->remove_child(child);
								replacement->add_child(child);
							}

						} // if the mesh instance has a parent (should hopefully be always the case?)
					}
				}

			} else {
				// no success
				memdelete(merged);
			}
		}

	} // for n through primary mesh

	if (_settings_remove_danglers) {
		_remove_redundant_dangling_nodes(p_room);
	}
}

bool RoomManager::_remove_redundant_dangling_nodes(Spatial *p_node) {
	int non_queue_delete_children = 0;

	// do the children first
	for (int n = 0; n < p_node->get_child_count(); n++) {
		Node *node_child = p_node->get_child(n);

		Spatial *child = Object::cast_to<Spatial>(node_child);
		if (child) {
			_remove_redundant_dangling_nodes(child);
		}

		if (node_child && !node_child->is_queued_for_deletion()) {
			non_queue_delete_children++;
		}
	}

	if (!non_queue_delete_children) {
		// only remove true spatials, not derived classes
		if (p_node->get_class_name() == "Spatial") {
			p_node->queue_delete();
			return true;
		}
	}

	return false;
}

void RoomManager::_list_mergeable_mesh_instances(Spatial *p_node, LocalVector<MeshInstance *, int32_t> &r_list) {
	MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);

	if (mi) {
		// only interested in static portal mode meshes
		VisualInstance *vi = Object::cast_to<VisualInstance>(mi);

		// we are only interested in VIs with static or dynamic mode
		if (vi && vi->get_portal_mode() == CullInstance::PORTAL_MODE_STATIC) {
			// disallow for portals or bounds
			// mesh instance portals should be queued for deletion by this point, we don't want to merge portals!
			if (!_node_is_type<Portal>(mi) && !_name_ends_with(mi, "-bound") && !mi->is_queued_for_deletion()) {
				// only merge if visible
				if (mi->is_inside_tree() && mi->is_visible()) {
					r_list.push_back(mi);
				}
			}
		}
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		Spatial *child = Object::cast_to<Spatial>(p_node->get_child(n));
		if (child) {
			_list_mergeable_mesh_instances(child, r_list);
		}
	}
}
