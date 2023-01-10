/**************************************************************************/
/*  portal_renderer.h                                                     */
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

#ifndef PORTAL_RENDERER_H
#define PORTAL_RENDERER_H

#include "core/math/camera_matrix.h"
#include "core/math/geometry.h"
#include "core/math/plane.h"
#include "core/pooled_list.h"
#include "core/vector.h"
#include "portal_gameplay_monitor.h"
#include "portal_pvs.h"
#include "portal_resources.h"
#include "portal_rooms_bsp.h"
#include "portal_tracer.h"
#include "portal_types.h"

class Transform;

struct VSStatic {
	// the lifetime of statics is not strictly monitored like moving objects
	// therefore we store a RID which could return NULL if the object has been deleted
	RID instance;

	AABB aabb;

	// statics are placed in a room, but they can optionally sprawl to other rooms
	// if large (like lights)
	uint32_t source_room_id;

	// dynamics will request their AABB each frame
	// from the visual server in case they have moved.
	// But they will NOT update the rooms they are in...
	// so this works well for e.g. moving platforms, but not for objects
	// that will move between rooms.
	uint32_t dynamic;
};

// static / dynamic visibility notifiers.
// ghost objects are not culled, but are present in rooms
// and expect to receive gameplay notifications
struct VSStaticGhost {
	ObjectID object_id;

	uint32_t last_tick_hit = 0;
	uint32_t last_room_tick_hit = 0;
};

class PortalRenderer {
public:
	// use most significant bit to store whether an instance is being used in the room system
	// in which case, deleting such an instance should deactivate the portal system to prevent
	// crashes due to dangling references to instances.
	static const uint32_t OCCLUSION_HANDLE_ROOM_BIT = 1 << 31;
	static bool use_occlusion_culling;

	struct MovingBase {
		// when the rooms_and_portals_clear message is sent,
		// we want to remove all references to old rooms in the moving
		// objects, to prevent dangling references.
		void rooms_and_portals_clear() { destroy(); }
		void destroy() {
			_rooms.clear();
			room_id = -1;

			last_tick_hit = 0;
			last_gameplay_tick_hit = 0;
		}

		// the expanded aabb allows objects to move on most frames
		// without needing to determine a change of room
		AABB expanded_aabb;

		// exact aabb of the object should be used for culling
		AABB exact_aabb;

		// which is the primary room this moving object is in
		// (it may sprawl into multiple rooms)
		int32_t room_id;

		// id in the allocation pool
		uint32_t pool_id;

		uint32_t last_tick_hit = 0;
		uint32_t last_gameplay_tick_hit = 0;

		// room ids of rooms this moving object is sprawled into
		LocalVector<uint32_t, int32_t> _rooms;
	};

	struct Moving : public MovingBase {
		// either roaming or global
		bool global;

		// in _moving_lists .. not the same as pool ID (handle)
		uint32_t list_id;

		// a void pointer, but this is ultimately a pointer to a VisualServerScene::Instance
		// (can't have direct pointer because it is a nested class...)
		VSInstance *instance;

#ifdef PORTAL_RENDERER_STORE_MOVING_RIDS
		// primarily for testing
		RID instance_rid;
#endif
	};

	// So far the only roaming ghosts are VisibilityNotifiers.
	// this will always be roaming... statics and dynamics are handled separately,
	// and global ghosts do not get created.
	struct RGhost : public MovingBase {
		ObjectID object_id;
	};

	PortalHandle portal_create();
	void portal_destroy(PortalHandle p_portal);
	void portal_set_geometry(PortalHandle p_portal, const Vector<Vector3> &p_points, real_t p_margin);
	void portal_link(PortalHandle p_portal, RoomHandle p_room_from, RoomHandle p_room_to, bool p_two_way);
	void portal_set_active(PortalHandle p_portal, bool p_active);

	RoomGroupHandle roomgroup_create();
	void roomgroup_prepare(RoomGroupHandle p_roomgroup, ObjectID p_roomgroup_object_id);
	void roomgroup_destroy(RoomGroupHandle p_roomgroup);
	void roomgroup_add_room(RoomGroupHandle p_roomgroup, RoomHandle p_room);

	// Rooms
	RoomHandle room_create();
	void room_destroy(RoomHandle p_room);
	OcclusionHandle room_add_instance(RoomHandle p_room, RID p_instance, const AABB &p_aabb, bool p_dynamic, const Vector<Vector3> &p_object_pts);
	OcclusionHandle room_add_ghost(RoomHandle p_room, ObjectID p_object_id, const AABB &p_aabb);
	void room_set_bound(RoomHandle p_room, ObjectID p_room_object_id, const Vector<Plane> &p_convex, const AABB &p_aabb, const Vector<Vector3> &p_verts);
	void room_prepare(RoomHandle p_room, int32_t p_priority);
	void rooms_and_portals_clear();
	void rooms_finalize(bool p_generate_pvs, bool p_cull_using_pvs, bool p_use_secondary_pvs, bool p_use_signals, String p_pvs_filename, bool p_use_simple_pvs, bool p_log_pvs_generation);
	void rooms_override_camera(bool p_override, const Vector3 &p_point, const Vector<Plane> *p_convex);
	void rooms_set_active(bool p_active) { _active = p_active; }
	void rooms_set_params(int p_portal_depth_limit, real_t p_roaming_expansion_margin) {
		_tracer.set_depth_limit(p_portal_depth_limit);
		_roaming_expansion_margin = p_roaming_expansion_margin;
	}
	void rooms_set_cull_using_pvs(bool p_enable) { _cull_using_pvs = p_enable; }
	void rooms_update_gameplay_monitor(const Vector<Vector3> &p_camera_positions);

	// for use in the editor only, to allow a cheap way of turning off portals
	// if there has been a change, e.g. moving a room etc.
	void rooms_unload(String p_reason) { _ensure_unloaded(p_reason); }
	bool rooms_is_loaded() const { return _loaded; }

	// debugging
	void set_debug_sprawl(bool p_active) { _debug_sprawl = p_active; }

	// this section handles moving objects - roaming (change rooms) and globals (not in any room)
	OcclusionHandle instance_moving_create(VSInstance *p_instance, RID p_instance_rid, bool p_global, AABB p_aabb);
	void instance_moving_update(OcclusionHandle p_handle, const AABB &p_aabb, bool p_force_reinsert = false);
	void instance_moving_destroy(OcclusionHandle p_handle);

	// spatial derived roamers (non VisualInstances that still need to be portal culled, especially VisibilityNotifiers)
	RGhostHandle rghost_create(ObjectID p_object_id, const AABB &p_aabb);
	void rghost_update(RGhostHandle p_handle, const AABB &p_aabb, bool p_force_reinsert = false);
	void rghost_destroy(RGhostHandle p_handle);

	// occluders
	OccluderInstanceHandle occluder_instance_create();
	void occluder_instance_link(OccluderInstanceHandle p_handle, OccluderResourceHandle p_resource_handle);
	void occluder_instance_set_transform(OccluderInstanceHandle p_handle, const Transform &p_xform);
	void occluder_instance_set_active(OccluderInstanceHandle p_handle, bool p_active);
	void occluder_instance_destroy(OccluderInstanceHandle p_handle, bool p_free = true);

	// editor only .. slow
	Geometry::MeshData occlusion_debug_get_current_polys() const { return _tracer.get_occlusion_culler().debug_get_current_polys(); }

	// note that this relies on a 'frustum' type cull, from a point, and that the planes are specified as in
	// CameraMatrix, i.e.
	// order PLANE_NEAR,PLANE_FAR,PLANE_LEFT,PLANE_TOP,PLANE_RIGHT,PLANE_BOTTOM
	int cull_convex(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, const Vector<Plane> &p_convex, VSInstance **p_result_array, int p_result_max, uint32_t p_mask, int32_t &r_previous_room_id_hint) {
		// combined camera matrix
		CameraMatrix cm = CameraMatrix(p_cam_transform.affine_inverse());
		cm = p_cam_projection * cm;
		Vector3 point = p_cam_transform.origin;
		Vector3 cam_dir = -p_cam_transform.basis.get_axis(2).normalized();

		if (!_override_camera)
			return cull_convex_implementation(point, cam_dir, cm, p_convex, p_result_array, p_result_max, p_mask, r_previous_room_id_hint);

		// override camera matrix NYI
		return cull_convex_implementation(_override_camera_pos, cam_dir, cm, _override_camera_planes, p_result_array, p_result_max, p_mask, r_previous_room_id_hint);
	}

	int cull_convex_implementation(const Vector3 &p_point, const Vector3 &p_cam_dir, const CameraMatrix &p_cam_matrix, const Vector<Plane> &p_convex, VSInstance **p_result_array, int p_result_max, uint32_t p_mask, int32_t &r_previous_room_id_hint);

	bool occlusion_is_active() const { return _occluder_instance_pool.active_size() && use_occlusion_culling; }

	// special function for occlusion culling only that does not use portals / rooms,
	// but allows using occluders with the main scene
	int occlusion_cull(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, const Vector<Plane> &p_convex, VSInstance **p_result_array, int p_num_results) {
		// inactive?
		if (!_occluder_instance_pool.active_size() || !use_occlusion_culling) {
			return p_num_results;
		}

		// combined camera matrix
		CameraMatrix cm = CameraMatrix(p_cam_transform.affine_inverse());
		cm = p_cam_projection * cm;
		Vector3 point = p_cam_transform.origin;
		Vector3 cam_dir = -p_cam_transform.basis.get_axis(2).normalized();

		return _tracer.occlusion_cull(*this, point, cam_dir, cm, p_convex, p_result_array, p_num_results);
	}

	bool is_active() const { return _active && _loaded; }

	VSStatic &get_static(int p_id) { return _statics[p_id]; }
	const VSStatic &get_static(int p_id) const { return _statics[p_id]; }

	int32_t get_num_rooms() const { return _room_pool_ids.size(); }
	VSRoom &get_room(int p_id) { return _room_pool[_room_pool_ids[p_id]]; }
	const VSRoom &get_room(int p_id) const { return _room_pool[_room_pool_ids[p_id]]; }

	int32_t get_num_portals() const { return _portal_pool_ids.size(); }
	VSPortal &get_portal(int p_id) { return _portal_pool[_portal_pool_ids[p_id]]; }
	const VSPortal &get_portal(int p_id) const { return _portal_pool[_portal_pool_ids[p_id]]; }

	int32_t get_num_moving_globals() const { return _moving_list_global.size(); }
	const Moving &get_moving_global(uint32_t p_id) const { return _moving_pool[_moving_list_global[p_id]]; }

	Moving &get_pool_moving(uint32_t p_pool_id) { return _moving_pool[p_pool_id]; }
	const Moving &get_pool_moving(uint32_t p_pool_id) const { return _moving_pool[p_pool_id]; }

	RGhost &get_pool_rghost(uint32_t p_pool_id) { return _rghost_pool[p_pool_id]; }
	const RGhost &get_pool_rghost(uint32_t p_pool_id) const { return _rghost_pool[p_pool_id]; }

	VSStaticGhost &get_static_ghost(uint32_t p_id) { return _static_ghosts[p_id]; }

	VSRoomGroup &get_roomgroup(uint32_t p_pool_id) { return _roomgroup_pool[p_pool_id]; }

	PVS &get_pvs() { return _pvs; }
	const PVS &get_pvs() const { return _pvs; }

	bool get_cull_using_pvs() const { return _cull_using_pvs; }

	// occluders
	const LocalVector<uint32_t, uint32_t> &get_occluders_active_list() const { return _occluder_instance_pool.get_active_list(); }
	const VSOccluder_Instance &get_pool_occluder_instance(uint32_t p_pool_id) const { return _occluder_instance_pool[p_pool_id]; }
	VSOccluder_Instance &get_pool_occluder_instance(uint32_t p_pool_id) { return _occluder_instance_pool[p_pool_id]; }
	const VSOccluder_Sphere &get_pool_occluder_world_sphere(uint32_t p_pool_id) const { return _occluder_world_sphere_pool[p_pool_id]; }
	const VSOccluder_Poly &get_pool_occluder_world_poly(uint32_t p_pool_id) const { return _occluder_world_poly_pool[p_pool_id]; }
	const VSOccluder_Hole &get_pool_occluder_world_hole(uint32_t p_pool_id) const { return _occluder_world_hole_pool[p_pool_id]; }
	VSOccluder_Hole &get_pool_occluder_world_hole(uint32_t p_pool_id) { return _occluder_world_hole_pool[p_pool_id]; }

private:
	int find_room_within(const Vector3 &p_pos, int p_previous_room_id = -1) {
		return _rooms_lookup_bsp.find_room_within(*this, p_pos, p_previous_room_id);
	}

	bool sprawl_static(int p_static_id, const VSStatic &p_static, int p_room_id);
	bool sprawl_static_geometry(int p_static_id, const VSStatic &p_static, int p_room_id, const Vector<Vector3> &p_object_pts);
	bool sprawl_static_ghost(int p_ghost_id, const AABB &p_aabb, int p_room_id);

	void _load_finalize_roaming();
	void sprawl_roaming(uint32_t p_mover_pool_id, MovingBase &r_moving, int p_room_id, bool p_moving_or_ghost);
	void _moving_remove_from_rooms(uint32_t p_moving_pool_id);
	void _rghost_remove_from_rooms(uint32_t p_pool_id);
	void _occluder_remove_from_rooms(uint32_t p_pool_id);
	void _ensure_unloaded(String p_reason = String());
	void _rooms_add_portals_to_convex_hulls();
	void _add_portal_to_convex_hull(LocalVector<Plane, int32_t> &p_planes, const Plane &p);

	void _debug_print_global_list();
	bool _occlusion_handle_is_in_room(OcclusionHandle p_h) const {
		return p_h == OCCLUSION_HANDLE_ROOM_BIT;
	}

	void _log(String p_string, int p_priority = 0);

	// note this is vulnerable to crashes, we must monitor for deletion of rooms
	LocalVector<uint32_t, int32_t> _room_pool_ids;
	LocalVector<uint32_t, int32_t> _portal_pool_ids;

	LocalVector<VSStatic, int32_t> _statics;
	LocalVector<VSStaticGhost, int32_t> _static_ghosts;

	// all rooms and portals are allocated from pools.
	PooledList<VSPortal> _portal_pool;
	PooledList<VSRoom> _room_pool;
	PooledList<VSRoomGroup> _roomgroup_pool;

	// moving objects, global and roaming
	PooledList<Moving> _moving_pool;
	TrackedPooledList<RGhost> _rghost_pool;
	LocalVector<uint32_t, int32_t> _moving_list_global;
	LocalVector<uint32_t, int32_t> _moving_list_roaming;

	// occluders
	TrackedPooledList<VSOccluder_Instance> _occluder_instance_pool;
	TrackedPooledList<VSOccluder_Sphere, uint32_t, true> _occluder_world_sphere_pool;
	TrackedPooledList<VSOccluder_Poly, uint32_t, true> _occluder_world_poly_pool;
	TrackedPooledList<VSOccluder_Hole, uint32_t, true> _occluder_world_hole_pool;

	PVS _pvs;

	bool _active = true;
	bool _loaded = false;
	bool _debug_sprawl = false;
	bool _show_debug = true;

	// if the pvs is generated, we can either cull using dynamic portals or PVS
	bool _cull_using_pvs = false;

	PortalTracer _tracer;
	PortalTracer::TraceResult _trace_results;
	PortalRoomsBSP _rooms_lookup_bsp;
	PortalGameplayMonitor _gameplay_monitor;

	// when moving roaming objects, we expand their bound
	// to prevent too many updates.
	real_t _roaming_expansion_margin = 1.0;

	// a bitfield to indicate which rooms have been
	// visited already in sprawling, to prevent visiting rooms multiple times
	BitFieldDynamic _bitfield_rooms;

	bool _override_camera = false;
	Vector3 _override_camera_pos;
	LocalVector<Plane, int32_t> _override_camera_planes;

public:
	static String _rid_to_string(RID p_rid);
	static String _addr_to_string(const void *p_addr);

	void occluder_ensure_up_to_date_sphere(const PortalResources &p_resources, VSOccluder_Instance &r_occluder);
	void occluder_ensure_up_to_date_polys(const PortalResources &p_resources, VSOccluder_Instance &r_occluder);
	void occluder_refresh_room_within(uint32_t p_occluder_pool_id);

	PortalRenderer();
};

inline void PortalRenderer::occluder_ensure_up_to_date_sphere(const PortalResources &p_resources, VSOccluder_Instance &r_occluder) {
	// occluder is not bound to a resource, cannot be used
	if (r_occluder.resource_pool_id == UINT32_MAX) {
		return;
	}

	// get the resource
	const VSOccluder_Resource &res = p_resources.get_pool_occluder_resource(r_occluder.resource_pool_id);

	// dirty?
	if (r_occluder.revision == res.revision) {
		return;
	}
	r_occluder.revision = res.revision;

	// must be same type, if not an error has occurred
	ERR_FAIL_COND(res.type != r_occluder.type);

	// first make sure the instance has the correct number of world space spheres
	if (r_occluder.list_ids.size() != res.list_ids.size()) {
		// not the most efficient, but works...
		// remove existing
		for (int n = 0; n < r_occluder.list_ids.size(); n++) {
			uint32_t id = r_occluder.list_ids[n];
			_occluder_world_sphere_pool.free(id);
		}

		r_occluder.list_ids.clear();
		// create new
		for (int n = 0; n < res.list_ids.size(); n++) {
			uint32_t id;
			VSOccluder_Sphere *sphere = _occluder_world_sphere_pool.request(id);
			sphere->create();
			r_occluder.list_ids.push_back(id);
		}
	}

	const Transform &tr = r_occluder.xform;

	Vector3 scale3 = tr.basis.get_scale_abs();
	real_t scale = (scale3.x + scale3.y + scale3.z) / 3.0;

	// update the AABB
	Vector3 bb_min = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3 bb_max = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	// transform spheres
	for (int n = 0; n < r_occluder.list_ids.size(); n++) {
		uint32_t world_pool_id = r_occluder.list_ids[n];
		VSOccluder_Sphere &world_osphere = _occluder_world_sphere_pool[world_pool_id];
		const VSOccluder_Sphere &local_osphere = p_resources.get_pool_occluder_local_sphere(res.list_ids[n]);

		world_osphere.pos = tr.xform(local_osphere.pos);
		world_osphere.radius = local_osphere.radius * scale;

		Vector3 bradius = Vector3(world_osphere.radius, world_osphere.radius, world_osphere.radius);
		Vector3 bmin = world_osphere.pos - bradius;
		Vector3 bmax = world_osphere.pos + bradius;

		bb_min.x = MIN(bb_min.x, bmin.x);
		bb_min.y = MIN(bb_min.y, bmin.y);
		bb_min.z = MIN(bb_min.z, bmin.z);
		bb_max.x = MAX(bb_max.x, bmax.x);
		bb_max.y = MAX(bb_max.y, bmax.y);
		bb_max.z = MAX(bb_max.z, bmax.z);
	}

	r_occluder.aabb.position = bb_min;
	r_occluder.aabb.size = bb_max - bb_min;
}

inline void PortalRenderer::occluder_ensure_up_to_date_polys(const PortalResources &p_resources, VSOccluder_Instance &r_occluder) {
	// occluder is not bound to a resource, cannot be used
	if (r_occluder.resource_pool_id == UINT32_MAX) {
		return;
	}

	// get the resource
	const VSOccluder_Resource &res = p_resources.get_pool_occluder_resource(r_occluder.resource_pool_id);

	// dirty?
	if (r_occluder.revision == res.revision) {
		return;
	}
	r_occluder.revision = res.revision;

	// must be same type, if not an error has occurred
	ERR_FAIL_COND(res.type != r_occluder.type);

	// first make sure the instance has the correct number of world space spheres
	if (r_occluder.list_ids.size() != res.list_ids.size()) {
		// not the most efficient, but works...
		// remove existing
		for (int n = 0; n < r_occluder.list_ids.size(); n++) {
			uint32_t id = r_occluder.list_ids[n];
			_occluder_world_poly_pool.free(id);
		}

		r_occluder.list_ids.clear();
		// create new
		for (int n = 0; n < res.list_ids.size(); n++) {
			uint32_t id;
			VSOccluder_Poly *poly = _occluder_world_poly_pool.request(id);
			poly->create();
			r_occluder.list_ids.push_back(id);
		}
	}

	const Transform &tr = r_occluder.xform;

	for (int n = 0; n < r_occluder.list_ids.size(); n++) {
		uint32_t world_pool_id = r_occluder.list_ids[n];
		uint32_t local_pool_id = res.list_ids[n];

		VSOccluder_Poly &world_opoly = _occluder_world_poly_pool[world_pool_id];
		const VSOccluder_Poly &local_opoly = p_resources._occluder_local_poly_pool[local_pool_id];

		world_opoly.poly.num_verts = local_opoly.poly.num_verts;
		world_opoly.two_way = local_opoly.two_way;

		for (int i = 0; i < local_opoly.poly.num_verts; i++) {
			world_opoly.poly.verts[i] = tr.xform(local_opoly.poly.verts[i]);
		}

		world_opoly.poly.plane = tr.xform(local_opoly.poly.plane);

		// number of holes must be correct for each poly
		if (world_opoly.num_holes != local_opoly.num_holes) {
			// remove existing
			for (int h = 0; h < world_opoly.num_holes; h++) {
				uint32_t id = world_opoly.hole_pool_ids[h];
				_occluder_world_hole_pool.free(id);
				// not strictly necessary
				world_opoly.hole_pool_ids[h] = UINT32_MAX;
			}

			world_opoly.num_holes = local_opoly.num_holes;
			for (int h = 0; h < world_opoly.num_holes; h++) {
				uint32_t id;
				VSOccluder_Hole *hole = _occluder_world_hole_pool.request(id);
				hole->create();
				world_opoly.hole_pool_ids[h] = id;
			}
		}

		// holes
		for (int h = 0; h < world_opoly.num_holes; h++) {
			uint32_t world_hid = world_opoly.hole_pool_ids[h];
			uint32_t local_hid = local_opoly.hole_pool_ids[h];
			VSOccluder_Hole &world_hole = _occluder_world_hole_pool[world_hid];
			const VSOccluder_Hole &local_hole = p_resources._occluder_local_hole_pool[local_hid];

			world_hole.num_verts = local_hole.num_verts;

			for (int i = 0; i < world_hole.num_verts; i++) {
				world_hole.verts[i] = tr.xform(local_hole.verts[i]);
			}
		}
	}
}

#endif // PORTAL_RENDERER_H
