/*************************************************************************/
/*  portal_renderer.cpp                                                  */
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

#include "portal_renderer.h"

#include "portal_pvs_builder.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual/visual_server_scene.h"

bool PortalRenderer::use_occlusion_culling = true;

OcclusionHandle PortalRenderer::instance_moving_create(VSInstance *p_instance, RID p_instance_rid, bool p_global, AABB p_aabb) {
	uint32_t pool_id = 0;
	Moving *moving = _moving_pool.request(pool_id);
	moving->global = p_global;
	moving->pool_id = pool_id;
	moving->instance = p_instance;
	moving->room_id = -1;

#ifdef PORTAL_RENDERER_STORE_MOVING_RIDS
	moving->instance_rid = p_instance_rid;
#endif

	// add to the appropriate list
	if (p_global) {
		moving->list_id = _moving_list_global.size();
		_moving_list_global.push_back(pool_id);
	} else {
		// do we need a roaming master list? not sure yet
		moving->list_id = _moving_list_roaming.size();
		_moving_list_roaming.push_back(pool_id);
	}

	OcclusionHandle handle = pool_id + 1;
	instance_moving_update(handle, p_aabb, true);
	return handle;
}

void PortalRenderer::instance_moving_update(OcclusionHandle p_handle, const AABB &p_aabb, bool p_force_reinsert) {
	p_handle--;
	Moving &moving = _moving_pool[p_handle];
	moving.exact_aabb = p_aabb;

	// globals (e.g. interface elements) need their aabb updated irrespective of whether the system is loaded
	if (!_loaded || moving.global) {
		return;
	}

	// we can ignore these, they are statics / dynamics, and don't need updating
	// .. these should have been filtered out before calling the visual server...
	DEV_CHECK_ONCE(!_occlusion_handle_is_in_room(p_handle));

	// quick reject for most roaming cases
	if (!p_force_reinsert && moving.expanded_aabb.encloses(p_aabb)) {
		return;
	}

	// using an expanded aabb allows us to make 'no op' moves
	// where the new aabb is within the expanded
	moving.expanded_aabb = p_aabb.grow(_roaming_expansion_margin);

	// if we got to here, it is roaming (moving between rooms)
	// remove from current rooms
	_moving_remove_from_rooms(p_handle);

	// add to new rooms
	Vector3 center = p_aabb.position + (p_aabb.size * 0.5);
	int new_room = find_room_within(center, moving.room_id);

	moving.room_id = new_room;
	if (new_room != -1) {
		_bitfield_rooms.blank();
		sprawl_roaming(p_handle, moving, new_room, true);
	}
}

void PortalRenderer::_rghost_remove_from_rooms(uint32_t p_pool_id) {
	RGhost &moving = _rghost_pool[p_pool_id];

	// if we have unloaded the rooms and we try this, it will crash
	if (_loaded) {
		for (int n = 0; n < moving._rooms.size(); n++) {
			VSRoom &room = get_room(moving._rooms[n]);
			room.remove_rghost(p_pool_id);
		}
	}

	// moving is now in no rooms
	moving._rooms.clear();
}

void PortalRenderer::_occluder_remove_from_rooms(uint32_t p_pool_id) {
	VSOccluder_Instance &occ = _occluder_instance_pool[p_pool_id];
	if (_loaded && (occ.room_id != -1)) {
		VSRoom &room = get_room(occ.room_id);
		bool res = room.remove_occluder(p_pool_id);
		if (!res) {
			WARN_PRINT_ONCE("OccluderInstance was not present in Room");
		}
	}
}

void PortalRenderer::_moving_remove_from_rooms(uint32_t p_moving_pool_id) {
	Moving &moving = _moving_pool[p_moving_pool_id];

	// if we have unloaded the rooms and we try this, it will crash
	if (_loaded) {
		for (int n = 0; n < moving._rooms.size(); n++) {
			VSRoom &room = get_room(moving._rooms[n]);
			room.remove_roamer(p_moving_pool_id);
		}
	}

	// moving is now in no rooms
	moving._rooms.clear();
}

void PortalRenderer::_debug_print_global_list() {
	_log("globals:");
	for (int n = 0; n < _moving_list_global.size(); n++) {
		uint32_t id = _moving_list_global[n];
		const Moving &moving = _moving_pool[id];
		_log("\t" + _addr_to_string(&moving));
	}
}

void PortalRenderer::_log(String p_string, int p_priority) {
	if (_show_debug) {
		// change this for more debug output ..
		// not selectable at runtime yet.
		if (p_priority >= 1) {
			print_line(p_string);
		} else {
			print_verbose(p_string);
		}
	}
}

void PortalRenderer::instance_moving_destroy(OcclusionHandle p_handle) {
	// deleting an instance that is assigned to a room (STATIC or DYNAMIC)
	// is special, it must set the PortalRenderer into unloaded state, because
	// there will now be a dangling reference to the instance that was destroyed.
	// The alternative is to remove the reference, but this is not currently supported
	// (it would mean rejigging rooms etc)
	if (_occlusion_handle_is_in_room(p_handle)) {
		_ensure_unloaded("deleting STATIC or DYNAMIC");
		return;
	}

	p_handle--;

	Moving *moving = &_moving_pool[p_handle];

	// if a roamer, remove from any current rooms
	if (!moving->global) {
		_moving_remove_from_rooms(p_handle);
	}

	// remove from list (and keep in sync)
	uint32_t list_id = moving->list_id;

	if (moving->global) {
		_moving_list_global.remove_unordered(list_id);

		// keep the replacement moving in sync with the correct list Id
		if (list_id < (uint32_t)_moving_list_global.size()) {
			uint32_t replacement_id = _moving_list_global[list_id];
			Moving &replacement = _moving_pool[replacement_id];
			replacement.list_id = list_id;
		}
	} else {
		_moving_list_roaming.remove_unordered(list_id);

		// keep the replacement moving in sync with the correct list Id
		if (list_id < (uint32_t)_moving_list_roaming.size()) {
			uint32_t replacement_id = _moving_list_roaming[list_id];
			Moving &replacement = _moving_pool[replacement_id];
			replacement.list_id = list_id;
		}
	}

	moving->destroy();

	// can now free the moving
	_moving_pool.free(p_handle);
}

PortalHandle PortalRenderer::portal_create() {
	uint32_t pool_id = 0;
	VSPortal *portal = _portal_pool.request(pool_id);

	// explicit constructor
	portal->create();
	portal->_portal_id = _portal_pool_ids.size();

	_portal_pool_ids.push_back(pool_id);

	// plus one based handles, 0 is unset
	pool_id++;
	return pool_id;
}

void PortalRenderer::portal_destroy(PortalHandle p_portal) {
	ERR_FAIL_COND(!p_portal);
	_ensure_unloaded("deleting Portal");

	// plus one based
	p_portal--;

	// remove from list of valid portals
	VSPortal &portal = _portal_pool[p_portal];
	int portal_id = portal._portal_id;

	// we need to replace the last element in the list
	_portal_pool_ids.remove_unordered(portal_id);

	// and reset the id of the portal that was the replacement
	if (portal_id < _portal_pool_ids.size()) {
		int replacement_pool_id = _portal_pool_ids[portal_id];
		VSPortal &replacement = _portal_pool[replacement_pool_id];
		replacement._portal_id = portal_id;
	}

	// explicitly run destructor
	_portal_pool[p_portal].destroy();

	// return to the pool
	_portal_pool.free(p_portal);
}

void PortalRenderer::portal_set_geometry(PortalHandle p_portal, const Vector<Vector3> &p_points, real_t p_margin) {
	ERR_FAIL_COND(!p_portal);
	p_portal--; // plus 1 based
	VSPortal &portal = _portal_pool[p_portal];

	portal._pts_world = p_points;
	portal._margin = p_margin;

	if (portal._pts_world.size() < 3) {
		WARN_PRINT("Portal must have at least 3 vertices");
		return;
	}

	// create plane from points
	// Allow averaging in case of wonky portals.

	// first calculate average normal
	Vector3 average_normal = Vector3(0, 0, 0);
	for (int t = 2; t < (int)portal._pts_world.size(); t++) {
		Plane p = Plane(portal._pts_world[0], portal._pts_world[t - 1], portal._pts_world[t]);
		average_normal += p.normal;
	}
	// average normal
	average_normal /= portal._pts_world.size() - 2;

	// detect user error
	ERR_FAIL_COND_MSG(average_normal.length() < 0.1, "Nonsense portal detected, normals should be consistent");
	if (average_normal.length() < 0.7) {
		WARN_PRINT("Wonky portal detected, you may see culling errors");
	}

	// calc average point
	Vector3 average_pt = Vector3(0, 0, 0);
	for (unsigned int n = 0; n < portal._pts_world.size(); n++) {
		average_pt += portal._pts_world[n];
	}
	average_pt /= portal._pts_world.size();

	// record the center for use in PVS
	portal._pt_center = average_pt;

	// calculate bounding sphere radius
	portal._bounding_sphere_radius = 0.0;
	for (unsigned int n = 0; n < portal._pts_world.size(); n++) {
		real_t sl = (portal._pts_world[n] - average_pt).length_squared();
		if (sl > portal._bounding_sphere_radius) {
			portal._bounding_sphere_radius = sl;
		}
	}
	portal._bounding_sphere_radius = Math::sqrt(portal._bounding_sphere_radius);

	// use the average point and normal to derive the plane
	portal._plane = Plane(average_pt, average_normal);

	// aabb
	AABB &bb = portal._aabb;
	bb.position = p_points[0];
	bb.size = Vector3(0, 0, 0);

	for (int n = 1; n < p_points.size(); n++) {
		bb.expand_to(p_points[n]);
	}
}

void PortalRenderer::portal_link(PortalHandle p_portal, RoomHandle p_room_from, RoomHandle p_room_to, bool p_two_way) {
	ERR_FAIL_COND(!p_portal);
	p_portal--; // plus 1 based
	VSPortal &portal = _portal_pool[p_portal];

	ERR_FAIL_COND(!p_room_from);
	p_room_from--;
	VSRoom &room_from = _room_pool[p_room_from];

	ERR_FAIL_COND(!p_room_to);
	p_room_to--;
	VSRoom &room_to = _room_pool[p_room_to];

	portal._linkedroom_ID[0] = room_from._room_ID;
	portal._linkedroom_ID[1] = room_to._room_ID;

	// is the portal internal? internal portals are treated differently
	portal._internal = room_from._priority > room_to._priority;

	// if it is internal, mark the outer room as containing an internal room.
	// this is used for rooms lookup.
	if (portal._internal) {
		room_to._contains_internal_rooms = true;
	}

	// _log("portal_link from room " + itos(room_from._room_ID) + " to room " + itos(room_to._room_ID));

	room_from._portal_ids.push_back(portal._portal_id);

	// one way portals simply aren't added to the destination room, so they don't get seen through
	if (p_two_way) {
		room_to._portal_ids.push_back(portal._portal_id);
	}
}

void PortalRenderer::portal_set_active(PortalHandle p_portal, bool p_active) {
	ERR_FAIL_COND(!p_portal);
	p_portal--; // plus 1 based
	VSPortal &portal = _portal_pool[p_portal];

	portal._active = p_active;
}

RoomGroupHandle PortalRenderer::roomgroup_create() {
	uint32_t pool_id = 0;
	VSRoomGroup *rg = _roomgroup_pool.request(pool_id);

	// explicit constructor
	rg->create();

	// plus one based handles, 0 is unset
	pool_id++;
	return pool_id;
}

void PortalRenderer::roomgroup_prepare(RoomGroupHandle p_roomgroup, ObjectID p_roomgroup_object_id) {
	// plus one based
	p_roomgroup--;
	VSRoomGroup &rg = _roomgroup_pool[p_roomgroup];
	rg._godot_instance_ID = p_roomgroup_object_id;
}

void PortalRenderer::roomgroup_destroy(RoomGroupHandle p_roomgroup) {
	ERR_FAIL_COND(!p_roomgroup);
	_ensure_unloaded("deleting RoomGroup");

	// plus one based
	p_roomgroup--;

	VSRoomGroup &rg = _roomgroup_pool[p_roomgroup];

	// explicitly run destructor
	rg.destroy();

	// return to the pool
	_roomgroup_pool.free(p_roomgroup);
}

void PortalRenderer::roomgroup_add_room(RoomGroupHandle p_roomgroup, RoomHandle p_room) {
	// plus one based
	p_roomgroup--;
	VSRoomGroup &rg = _roomgroup_pool[p_roomgroup];

	p_room--;

	// add to room group
	rg._room_ids.push_back(p_room);

	// add the room group to the room
	VSRoom &room = _room_pool[p_room];
	room._roomgroup_ids.push_back(p_roomgroup);
}

// Cull Instances
RGhostHandle PortalRenderer::rghost_create(ObjectID p_object_id, const AABB &p_aabb) {
	uint32_t pool_id = 0;
	RGhost *moving = _rghost_pool.request(pool_id);
	moving->pool_id = pool_id;
	moving->object_id = p_object_id;
	moving->room_id = -1;

	RGhostHandle handle = pool_id + 1;
	rghost_update(handle, p_aabb);
	return handle;
}

void PortalRenderer::rghost_update(RGhostHandle p_handle, const AABB &p_aabb, bool p_force_reinsert) {
	if (!_loaded) {
		return;
	}

	p_handle--;
	RGhost &moving = _rghost_pool[p_handle];
	moving.exact_aabb = p_aabb;

	// quick reject for most roaming cases
	if (!p_force_reinsert && moving.expanded_aabb.encloses(p_aabb)) {
		return;
	}

	// using an expanded aabb allows us to make 'no op' moves
	// where the new aabb is within the expanded
	moving.expanded_aabb = p_aabb.grow(_roaming_expansion_margin);

	// if we got to here, it is roaming (moving between rooms)
	// remove from current rooms
	_rghost_remove_from_rooms(p_handle);

	// add to new rooms
	Vector3 center = p_aabb.position + (p_aabb.size * 0.5);
	int new_room = find_room_within(center, moving.room_id);

	moving.room_id = new_room;
	if (new_room != -1) {
		_bitfield_rooms.blank();
		sprawl_roaming(p_handle, moving, new_room, false);
	}
}

void PortalRenderer::rghost_destroy(RGhostHandle p_handle) {
	p_handle--;

	RGhost *moving = &_rghost_pool[p_handle];

	// if a roamer, remove from any current rooms
	_rghost_remove_from_rooms(p_handle);

	moving->destroy();

	// can now free the moving
	_rghost_pool.free(p_handle);
}

OccluderInstanceHandle PortalRenderer::occluder_instance_create() {
	uint32_t pool_id = 0;
	VSOccluder_Instance *occ = _occluder_instance_pool.request(pool_id);
	occ->create();

	OccluderInstanceHandle handle = pool_id + 1;
	return handle;
}

void PortalRenderer::occluder_instance_link(OccluderInstanceHandle p_handle, OccluderResourceHandle p_resource_handle) {
	p_handle--;
	VSOccluder_Instance &occ = _occluder_instance_pool[p_handle];

	// Unlink with any already linked, and destroy world resources
	if (occ.resource_pool_id != UINT32_MAX) {
		// Watch for bugs in future with the room within, this is not changed here,
		// but could potentially be removed and re-added in future if we use sprawling.
		occluder_instance_destroy(p_handle + 1, false);
		occ.resource_pool_id = UINT32_MAX;
	}

	p_resource_handle--;
	VSOccluder_Resource &res = VSG::scene->get_portal_resources().get_pool_occluder_resource(p_resource_handle);

	occ.resource_pool_id = p_resource_handle;
	occ.type = res.type;
	occ.revision = 0;
}

void PortalRenderer::occluder_instance_set_active(OccluderInstanceHandle p_handle, bool p_active) {
	p_handle--;
	VSOccluder_Instance &occ = _occluder_instance_pool[p_handle];

	if (occ.active == p_active) {
		return;
	}
	occ.active = p_active;

	// this will take care of adding or removing from rooms
	occluder_refresh_room_within(p_handle);
}

void PortalRenderer::occluder_instance_set_transform(OccluderInstanceHandle p_handle, const Transform &p_xform) {
	p_handle--;
	VSOccluder_Instance &occ = _occluder_instance_pool[p_handle];
	occ.xform = p_xform;

	// mark as dirty as the world space spheres will be out of date
	occ.revision = 0;

	// The room within is based on the xform, rather than the AABB so this
	// should still work even though the world space transform is deferred.
	// N.B. Occluders are a single room based on the center of the Occluder transform,
	// this may need to be improved at a later date.
	occluder_refresh_room_within(p_handle);
}

void PortalRenderer::occluder_refresh_room_within(uint32_t p_occluder_pool_id) {
	VSOccluder_Instance &occ = _occluder_instance_pool[p_occluder_pool_id];

	// if we aren't loaded, the room within can't be valid
	if (!_loaded) {
		occ.room_id = -1;
		return;
	}

	// inactive?
	if (!occ.active) {
		// remove from any rooms present in
		if (occ.room_id != -1) {
			_occluder_remove_from_rooms(p_occluder_pool_id);
			occ.room_id = -1;
		}
		return;
	}

	// prevent checks with no significant changes
	Vector3 offset = occ.xform.origin - occ.pt_center;

	// could possibly make this epsilon editable?
	// is highly world size dependent.
	if ((offset.length_squared() < 0.01) && (occ.room_id != -1)) {
		return;
	}

	// standardize on the node origin for now
	occ.pt_center = occ.xform.origin;

	int new_room = find_room_within(occ.pt_center, occ.room_id);

	if (new_room != occ.room_id) {
		_occluder_remove_from_rooms(p_occluder_pool_id);
		occ.room_id = new_room;

		if (new_room != -1) {
			VSRoom &room = get_room(new_room);
			room.add_occluder(p_occluder_pool_id);
		}
	}
}

void PortalRenderer::occluder_instance_destroy(OccluderInstanceHandle p_handle, bool p_free) {
	p_handle--;

	if (p_free) {
		_occluder_remove_from_rooms(p_handle);
	}

	// depending on the occluder type, remove the spheres etc
	VSOccluder_Instance &occ = _occluder_instance_pool[p_handle];
	switch (occ.type) {
		case VSOccluder_Instance::OT_SPHERE: {
			// free any spheres owned by the occluder
			for (int n = 0; n < occ.list_ids.size(); n++) {
				uint32_t id = occ.list_ids[n];
				_occluder_world_sphere_pool.free(id);
			}
			occ.list_ids.clear();
		} break;
		case VSOccluder_Instance::OT_MESH: {
			// free any polys owned by the occluder
			for (int n = 0; n < occ.list_ids.size(); n++) {
				uint32_t id = occ.list_ids[n];
				VSOccluder_Poly &poly = _occluder_world_poly_pool[id];

				// free any holes owned by the poly
				for (int h = 0; h < poly.num_holes; h++) {
					_occluder_world_hole_pool.free(poly.hole_pool_ids[h]);
				}
				// blanks
				poly.create();
				_occluder_world_poly_pool.free(id);
			}
			occ.list_ids.clear();
		} break;
		default: {
		} break;
	}

	if (p_free) {
		_occluder_instance_pool.free(p_handle);
	}
}

// Rooms
RoomHandle PortalRenderer::room_create() {
	uint32_t pool_id = 0;
	VSRoom *room = _room_pool.request(pool_id);

	// explicit constructor
	room->create();

	// keep our own internal list of rooms
	room->_room_ID = _room_pool_ids.size();
	_room_pool_ids.push_back(pool_id);

	// plus one based handles, 0 is unset
	pool_id++;
	return pool_id;
}

void PortalRenderer::room_destroy(RoomHandle p_room) {
	ERR_FAIL_COND(!p_room);
	_ensure_unloaded("deleting Room");

	// plus one based
	p_room--;

	// remove from list of valid rooms
	VSRoom &room = _room_pool[p_room];
	int room_id = room._room_ID;

	// we need to replace the last element in the list
	_room_pool_ids.remove_unordered(room_id);

	// and reset the id of the portal that was the replacement
	if (room_id < _room_pool_ids.size()) {
		int replacement_pool_id = _room_pool_ids[room_id];
		VSRoom &replacement = _room_pool[replacement_pool_id];
		replacement._room_ID = room_id;
	}

	// explicitly run destructor
	_room_pool[p_room].destroy();

	// return to the pool
	_room_pool.free(p_room);
}

OcclusionHandle PortalRenderer::room_add_ghost(RoomHandle p_room, ObjectID p_object_id, const AABB &p_aabb) {
	ERR_FAIL_COND_V(!p_room, 0);
	p_room--; // plus one based

	VSStaticGhost ghost;
	ghost.object_id = p_object_id;
	_static_ghosts.push_back(ghost);

	// sprawl immediately
	// precreate a useful bitfield of rooms for use in sprawling
	if ((int)_bitfield_rooms.get_num_bits() != get_num_rooms()) {
		_bitfield_rooms.create(get_num_rooms());
	}

	// only can do if rooms exist
	if (get_num_rooms()) {
		// the last one was just added
		int ghost_id = _static_ghosts.size() - 1;

		// create a bitfield to indicate which rooms have been
		// visited already, to prevent visiting rooms multiple times
		_bitfield_rooms.blank();
		if (sprawl_static_ghost(ghost_id, p_aabb, p_room)) {
			_log("\t\tSPRAWLED");
		}
	}

	return OCCLUSION_HANDLE_ROOM_BIT;
}

OcclusionHandle PortalRenderer::room_add_instance(RoomHandle p_room, RID p_instance, const AABB &p_aabb, bool p_dynamic, const Vector<Vector3> &p_object_pts) {
	ERR_FAIL_COND_V(!p_room, 0);
	p_room--; // plus one based
	VSRoom &room = _room_pool[p_room];

	VSStatic stat;
	stat.instance = p_instance;
	stat.source_room_id = room._room_ID;
	stat.dynamic = p_dynamic;
	stat.aabb = p_aabb;
	_statics.push_back(stat);

	// sprawl immediately
	// precreate a useful bitfield of rooms for use in sprawling
	if ((int)_bitfield_rooms.get_num_bits() != get_num_rooms()) {
		_bitfield_rooms.create(get_num_rooms());
	}

	// only can do if rooms exist
	if (get_num_rooms()) {
		// the last one was just added
		int static_id = _statics.size() - 1;

		// pop last static
		const VSStatic &st = _statics[static_id];

		// create a bitfield to indicate which rooms have been
		// visited already, to prevent visiting rooms multiple times
		_bitfield_rooms.blank();

		if (p_object_pts.size()) {
			if (sprawl_static_geometry(static_id, st, st.source_room_id, p_object_pts)) {
				_log("\t\tSPRAWLED");
			}
		} else {
			if (sprawl_static(static_id, st, st.source_room_id)) {
				_log("\t\tSPRAWLED");
			}
		}
	}

	return OCCLUSION_HANDLE_ROOM_BIT;
}

void PortalRenderer::room_prepare(RoomHandle p_room, int32_t p_priority) {
	ERR_FAIL_COND(!p_room);
	p_room--; // plus one based
	VSRoom &room = _room_pool[p_room];
	room._priority = p_priority;
}

void PortalRenderer::room_set_bound(RoomHandle p_room, ObjectID p_room_object_id, const Vector<Plane> &p_convex, const AABB &p_aabb, const Vector<Vector3> &p_verts) {
	ERR_FAIL_COND(!p_room);
	p_room--; // plus one based
	VSRoom &room = _room_pool[p_room];

	room._planes = p_convex;
	room._verts = p_verts;
	room._aabb = p_aabb;
	room._godot_instance_ID = p_room_object_id;
}

void PortalRenderer::_add_portal_to_convex_hull(LocalVector<Plane, int32_t> &p_planes, const Plane &p) {
	for (int n = 0; n < p_planes.size(); n++) {
		Plane &o = p_planes[n];

		// this is a fudge factor for how close the portal can be to an existing plane
		// to be to be considered the same ...
		// to prevent needless extra checks.

		// the epsilons should probably be more exact here than for the convex hull simplification, as it is
		// fairly crucial that the portal planes are reasonably accurate for determining the hull.

		// and because the portal plane is more important, we will REPLACE the existing similar plane
		// with the portal plane.
		const real_t d = 0.03; // 0.08f

		if (Math::abs(p.d - o.d) > d) {
			continue;
		}

		real_t dot = p.normal.dot(o.normal);
		if (dot < 0.99) // 0.98f
		{
			continue;
		}

		// match!
		// replace the existing plane
		o = p;
		return;
	}

	// there is no existing plane that is similar, create a new one especially for the portal
	p_planes.push_back(p);
}

void PortalRenderer::_rooms_add_portals_to_convex_hulls() {
	for (int n = 0; n < get_num_rooms(); n++) {
		VSRoom &room = get_room(n);

		for (int p = 0; p < room._portal_ids.size(); p++) {
			const VSPortal &portal = get_portal(room._portal_ids[p]);

			// everything depends on whether the portal is incoming or outgoing.
			// if incoming we reverse the logic.
			int outgoing = 1;

			int room_a_id = portal._linkedroom_ID[0];
			if (room_a_id != n) {
				outgoing = 0;
				DEV_ASSERT(portal._linkedroom_ID[1] == n);
			}

			// do not add internal portals to the convex hull of outer rooms!
			if (!outgoing && portal._internal) {
				continue;
			}

			// add the portal plane
			Plane portal_plane = portal._plane;
			if (!outgoing) {
				portal_plane = -portal_plane;
			}

			// add if sufficiently different from existing convex hull planes
			_add_portal_to_convex_hull(room._planes, portal_plane);
		}
	}
}

void PortalRenderer::rooms_finalize(bool p_generate_pvs, bool p_cull_using_pvs, bool p_use_secondary_pvs, bool p_use_signals, String p_pvs_filename, bool p_use_simple_pvs, bool p_log_pvs_generation) {
	_gameplay_monitor.set_params(p_use_secondary_pvs, p_use_signals);

	// portals should also bound the rooms, the room geometry may extend past the portal
	_rooms_add_portals_to_convex_hulls();

	// the trace results can never have more hits than the number of static objects
	_trace_results.create(_statics.size());

	// precreate a useful bitfield of rooms for use in sprawling, if not created already
	// (may not be necessary but just in case, rooms with no statics etc)
	_bitfield_rooms.create(_room_pool_ids.size());

	// the rooms looksup is a pre-calced grid structure for faster lookup of the nearest room
	// from position
	_rooms_lookup_bsp.create(*this);

	// calculate PVS
	if (p_generate_pvs) {
		PVSBuilder pvs;
		pvs.calculate_pvs(*this, p_pvs_filename, _tracer.get_depth_limit(), p_use_simple_pvs, p_log_pvs_generation);
		_cull_using_pvs = p_cull_using_pvs; // hard code to on for test
	} else {
		_cull_using_pvs = false;
	}

	_loaded = true;

	// all the roaming objects need to be sprawled into the rooms
	// (they may have been created before the rooms)
	_load_finalize_roaming();

	// allow deleting any intermediate data
	for (int n = 0; n < get_num_rooms(); n++) {
		get_room(n).cleanup_after_conversion();
	}

	// this should probably have some thread protection, but I doubt it matters
	// as this will worst case give wrong result for a frame
	Engine::get_singleton()->set_portals_active(true);

	_log("Room conversion complete. " + itos(_room_pool_ids.size()) + " rooms, " + itos(_portal_pool_ids.size()) + " portals.", 1);
}

bool PortalRenderer::sprawl_static_geometry(int p_static_id, const VSStatic &p_static, int p_room_id, const Vector<Vector3> &p_object_pts) {
	// set, and if room already done, ignore
	if (!_bitfield_rooms.check_and_set(p_room_id))
		return false;

	VSRoom &room = get_room(p_room_id);
	room._static_ids.push_back(p_static_id);

	bool sprawled = false;

	// go through portals
	for (int p = 0; p < room._portal_ids.size(); p++) {
		const VSPortal &portal = get_portal(room._portal_ids[p]);

		int room_to_id = portal.geometry_crosses_portal(p_room_id, p_static.aabb, p_object_pts);

		if (room_to_id != -1) {
			// _log(String(Variant(p_static.aabb)) + " crosses portal");
			sprawl_static_geometry(p_static_id, p_static, room_to_id, p_object_pts);
			sprawled = true;
		}
	}

	return sprawled;
}

bool PortalRenderer::sprawl_static_ghost(int p_ghost_id, const AABB &p_aabb, int p_room_id) {
	// set, and if room already done, ignore
	if (!_bitfield_rooms.check_and_set(p_room_id)) {
		return false;
	}

	VSRoom &room = get_room(p_room_id);
	room._static_ghost_ids.push_back(p_ghost_id);

	bool sprawled = false;

	// go through portals
	for (int p = 0; p < room._portal_ids.size(); p++) {
		const VSPortal &portal = get_portal(room._portal_ids[p]);

		int room_to_id = portal.crosses_portal(p_room_id, p_aabb, true);

		if (room_to_id != -1) {
			// _log(String(Variant(p_aabb)) + " crosses portal");
			sprawl_static_ghost(p_ghost_id, p_aabb, room_to_id);
			sprawled = true;
		}
	}

	return sprawled;
}

bool PortalRenderer::sprawl_static(int p_static_id, const VSStatic &p_static, int p_room_id) {
	// set, and if room already done, ignore
	if (!_bitfield_rooms.check_and_set(p_room_id)) {
		return false;
	}

	VSRoom &room = get_room(p_room_id);
	room._static_ids.push_back(p_static_id);

	bool sprawled = false;

	// go through portals
	for (int p = 0; p < room._portal_ids.size(); p++) {
		const VSPortal &portal = get_portal(room._portal_ids[p]);

		int room_to_id = portal.crosses_portal(p_room_id, p_static.aabb, true);

		if (room_to_id != -1) {
			// _log(String(Variant(p_static.aabb)) + " crosses portal");
			sprawl_static(p_static_id, p_static, room_to_id);
			sprawled = true;
		}
	}

	return sprawled;
}

void PortalRenderer::_load_finalize_roaming() {
	for (int n = 0; n < _moving_list_roaming.size(); n++) {
		uint32_t pool_id = _moving_list_roaming[n];

		Moving &moving = _moving_pool[pool_id];
		const AABB &aabb = moving.exact_aabb;

		OcclusionHandle handle = pool_id + 1;
		instance_moving_update(handle, aabb, true);
	}

	for (unsigned int n = 0; n < _rghost_pool.active_size(); n++) {
		RGhost &moving = _rghost_pool.get_active(n);
		const AABB &aabb = moving.exact_aabb;

		rghost_update(_rghost_pool.get_active_id(n) + 1, aabb, true);
	}

	for (unsigned int n = 0; n < _occluder_instance_pool.active_size(); n++) {
		VSOccluder_Instance &occ = _occluder_instance_pool.get_active(n);
		int occluder_id = _occluder_instance_pool.get_active_id(n);

		// make sure occluder is in the correct room
		occ.room_id = find_room_within(occ.pt_center, -1);

		if (occ.room_id != -1) {
			VSRoom &room = get_room(occ.room_id);
			room.add_occluder(occluder_id);
		}
	}
}

void PortalRenderer::sprawl_roaming(uint32_t p_mover_pool_id, MovingBase &r_moving, int p_room_id, bool p_moving_or_ghost) {
	// set, and if room already done, ignore
	if (!_bitfield_rooms.check_and_set(p_room_id)) {
		return;
	}

	// add to the room
	VSRoom &room = get_room(p_room_id);

	if (p_moving_or_ghost) {
		room.add_roamer(p_mover_pool_id);
	} else {
		room.add_rghost(p_mover_pool_id);
	}

	// add the room	to the mover
	r_moving._rooms.push_back(p_room_id);

	// go through portals
	for (int p = 0; p < room._portal_ids.size(); p++) {
		const VSPortal &portal = get_portal(room._portal_ids[p]);

		int room_to_id = portal.crosses_portal(p_room_id, r_moving.expanded_aabb);

		if (room_to_id != -1) {
			// _log(String(Variant(p_static.aabb)) + " crosses portal");
			sprawl_roaming(p_mover_pool_id, r_moving, room_to_id, p_moving_or_ghost);
		}
	}
}

// This gets called when you delete an instance the the room system depends on
void PortalRenderer::_ensure_unloaded(String p_reason) {
	if (_loaded) {
		_loaded = false;
		_gameplay_monitor.unload(*this);

		String str;
		if (p_reason != String()) {
			str = "Portal system unloaded ( " + p_reason + " ).";
		} else {
			str = "Portal system unloaded.";
		}

		_log(str, 1);

		// this should probably have some thread protection, but I doubt it matters
		// as this will worst case give wrong result for a frame
		Engine::get_singleton()->set_portals_active(false);
	}
}

void PortalRenderer::rooms_and_portals_clear() {
	_loaded = false;

	// N.B. We want to make sure all the tick counters on movings rooms etc to zero,
	// so that on loading the next level gameplay entered signals etc will be
	// correctly sent and everything is fresh.
	// This is mostly done by the gameplay_monitor, but rooms_and_portals_clear()
	// will also clear tick counters where possible
	// (there is no TrackedList for the RoomGroup pool for example).
	// This could be made neater by moving everything to TrackedPooledLists, but this
	// may be overkill.
	_gameplay_monitor.unload(*this);

	_statics.clear();
	_static_ghosts.clear();

	// the rooms and portals should remove their id when they delete themselves
	// from the scene tree by calling room_destroy and portal_destroy ...
	// therefore there should be no need to clear these here
	// _room_pool_ids.clear();
	// _portal_pool_ids.clear();

	_rooms_lookup_bsp.clear();

	// clear the portals out of each existing room
	for (int n = 0; n < get_num_rooms(); n++) {
		VSRoom &room = get_room(n);
		room.rooms_and_portals_clear();
	}

	for (int n = 0; n < get_num_portals(); n++) {
		VSPortal &portal = get_portal(n);
		portal.rooms_and_portals_clear();
	}

	// when the rooms_and_portals_clear message is sent,
	// we want to remove all references to old rooms in the moving
	// objects, to prevent dangling references.
	for (int n = 0; n < get_num_moving_globals(); n++) {
		Moving &moving = get_pool_moving(_moving_list_global[n]);
		moving.rooms_and_portals_clear();
	}
	for (int n = 0; n < _moving_list_roaming.size(); n++) {
		Moving &moving = get_pool_moving(_moving_list_roaming[n]);
		moving.rooms_and_portals_clear();
	}

	for (unsigned int n = 0; n < _rghost_pool.active_size(); n++) {
		RGhost &moving = _rghost_pool.get_active(n);
		moving.rooms_and_portals_clear();
	}

	_pvs.clear();
}

void PortalRenderer::rooms_override_camera(bool p_override, const Vector3 &p_point, const Vector<Plane> *p_convex) {
	_override_camera = p_override;
	_override_camera_pos = p_point;
	if (p_convex) {
		_override_camera_planes = *p_convex;
	}
}

void PortalRenderer::rooms_update_gameplay_monitor(const Vector<Vector3> &p_camera_positions) {
	// is the pvs loaded?
	if (!_loaded || !_pvs.is_loaded()) {
		if (!_pvs.is_loaded()) {
			WARN_PRINT_ONCE("RoomManager PVS is required for this functionality");
		}
		return;
	}

	int *source_rooms = (int *)alloca(sizeof(int) * p_camera_positions.size());
	int num_source_rooms = 0;

	for (int n = 0; n < p_camera_positions.size(); n++) {
		int source_room_id = find_room_within(p_camera_positions[n]);
		if (source_room_id == -1) {
			continue;
		}

		source_rooms[num_source_rooms++] = source_room_id;
	}

	_gameplay_monitor.update_gameplay(*this, source_rooms, num_source_rooms);
}

int PortalRenderer::cull_convex_implementation(const Vector3 &p_point, const Vector3 &p_cam_dir, const CameraMatrix &p_cam_matrix, const Vector<Plane> &p_convex, VSInstance **p_result_array, int p_result_max, uint32_t p_mask, int32_t &r_previous_room_id_hint) {
	// start room
	int start_room_id = find_room_within(p_point, r_previous_room_id_hint);

	// return the previous room hint
	r_previous_room_id_hint = start_room_id;

	if (start_room_id == -1) {
		return -1;
	}

	// set up the occlusion culler once off .. this is a prepare before the prepare is done PER room
	_tracer.get_occlusion_culler().prepare_camera(p_cam_matrix, p_cam_dir);

	// planes must be in CameraMatrix order
	DEV_ASSERT(p_convex.size() == 6);

	LocalVector<Plane> planes;
	planes = p_convex;

	_trace_results.clear();

	if (!_debug_sprawl) {
		_tracer.trace(*this, p_point, planes, start_room_id, _trace_results); //, near_and_far_planes);
	} else {
		_tracer.trace_debug_sprawl(*this, p_point, start_room_id, _trace_results);
	}

	int num_results = _trace_results.visible_static_ids.size();
	int out_count = 0;

	for (int n = 0; n < num_results; n++) {
		uint32_t static_id = _trace_results.visible_static_ids[n];
		RID static_rid = _statics[static_id].instance;
		VSInstance *instance = VSG::scene->_instance_get_from_rid(static_rid);

		if (VSG::scene->_instance_cull_check(instance, p_mask)) {
			p_result_array[out_count++] = instance;
			if (out_count >= p_result_max) {
				break;
			}
		}
	}

	// results could be full up already
	if (out_count >= p_result_max) {
		return out_count;
	}

	// add the roaming results

	// cap to the maximum results
	int num_roam_hits = _trace_results.visible_roamer_pool_ids.size();

	// translate
	for (int n = 0; n < num_roam_hits; n++) {
		const Moving &moving = get_pool_moving(_trace_results.visible_roamer_pool_ids[n]);

		if (VSG::scene->_instance_cull_check(moving.instance, p_mask)) {
			p_result_array[out_count++] = moving.instance;
			if (out_count >= p_result_max) {
				break;
			}
		}
	}

	// results could be full up already
	if (out_count >= p_result_max) {
		return out_count;
	}

	out_count = _tracer.trace_globals(planes, p_result_array, out_count, p_result_max, p_mask, _override_camera);

	return out_count;
}

String PortalRenderer::_rid_to_string(RID p_rid) {
	return itos(p_rid.get_id());
}

String PortalRenderer::_addr_to_string(const void *p_addr) {
	return String::num_uint64((uint64_t)p_addr, 16);
}

PortalRenderer::PortalRenderer() {
	_show_debug = GLOBAL_GET("rendering/portals/debug/logging");
}
