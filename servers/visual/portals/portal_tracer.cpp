/**************************************************************************/
/*  portal_tracer.cpp                                                     */
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

#include "portal_tracer.h"

#include "portal_renderer.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual/visual_server_scene.h"

PortalTracer::PlanesPool::PlanesPool() {
	reset();

	// preallocate the vectors to a reasonable size
	for (int n = 0; n < POOL_MAX; n++) {
		_planes[n].resize(32);
	}
}

void PortalTracer::PlanesPool::reset() {
	for (int n = 0; n < POOL_MAX; n++) {
		_freelist[n] = POOL_MAX - n - 1;
	}

	_num_free = POOL_MAX;
}

unsigned int PortalTracer::PlanesPool::request() {
	if (!_num_free) {
		return -1;
	}

	_num_free--;
	return _freelist[_num_free];
}

void PortalTracer::PlanesPool::free(unsigned int ui) {
	DEV_ASSERT(ui < POOL_MAX);
	DEV_ASSERT(_num_free < POOL_MAX);

	_freelist[_num_free] = ui;
	_num_free++;
}

void PortalTracer::trace_debug_sprawl(PortalRenderer &p_portal_renderer, const Vector3 &p_pos, int p_start_room_id, TraceResult &r_result) {
	_portal_renderer = &p_portal_renderer;
	_trace_start_point = p_pos;
	_result = &r_result;

	// all the statics should be not hit to start with
	_result->clear();

	// new test, new tick, to prevent hitting objects more than once
	// on a test.
	_tick++;

	// if the camera is not in a room do nothing
	if (p_start_room_id == -1) {
		return;
	}

	trace_debug_sprawl_recursive(0, p_start_room_id);
}

void PortalTracer::trace(PortalRenderer &p_portal_renderer, const Vector3 &p_pos, const LocalVector<Plane> &p_planes, int p_start_room_id, TraceResult &r_result) {
	// store local versions to prevent passing around recursive functions
	_portal_renderer = &p_portal_renderer;
	_trace_start_point = p_pos;
	_result = &r_result;

	// The near and far clipping planes needs special treatment. The problem is, if it is
	// say a metre from the camera, it will clip out a portal immediately in front of the camera.
	// as a result we want to use the near clipping plane for objects, but construct a fake
	// near plane at exactly the position of the camera, to clip out portals that are behind us.
	_near_and_far_planes[0] = p_planes[0];
	_near_and_far_planes[1] = p_planes[1];

	// all the statics should be not hit to start with
	_result->clear();

	// new test, new tick, to prevent hitting objects more than once
	// on a test.
	_tick++;

	// if the camera is not in a room do nothing
	// (this will return no hits, but is unlikely because the find_rooms lookup will return the nearest
	// room even if not inside)
	if (p_start_room_id == -1) {
		return;
	}

	// start off the trace with the planes from the camera
	LocalVector<Plane> cam_planes;
	cam_planes = p_planes;

	if (p_portal_renderer.get_cull_using_pvs()) {
		trace_pvs(p_start_room_id, cam_planes);
	} else {
		// alternative : instead of copying straight, we create the first (near) clipping
		// plane manually, at 0 distance from the camera. This ensures that portals will not be
		// missed, while still culling portals and objects behind us. If we use the actual near clipping plane
		// then a portal in front of the camera may not be seen through, giving glitches
		cam_planes[0] = Plane(p_pos, cam_planes[0].normal);

		TraceParams params;
		params.use_pvs = p_portal_renderer.get_pvs().is_loaded();
		params.start_room_id = p_start_room_id;

		// create bitfield
		if (params.use_pvs) {
			const PVS &pvs = _portal_renderer->get_pvs();
			if (!pvs.get_pvs_size()) {
				params.use_pvs = false;
			} else {
				// decompress a simple to read roomlist bitfield (could use bits maybe but bytes ok for now)
				params.decompressed_room_pvs = nullptr;
				params.decompressed_room_pvs = (uint8_t *)alloca(sizeof(uint8_t) * pvs.get_pvs_size());
				memset(params.decompressed_room_pvs, 0, sizeof(uint8_t) * pvs.get_pvs_size());
				const VSRoom &source_room = _portal_renderer->get_room(p_start_room_id);

				for (int n = 0; n < source_room._pvs_size; n++) {
					int room_id = pvs.get_pvs_room_id(source_room._pvs_first + n);
					params.decompressed_room_pvs[room_id] = 255;
				}
			}
		}

		trace_recursive(params, 0, p_start_room_id, cam_planes);
	}
}

void PortalTracer::cull_roamers(const VSRoom &p_room, const LocalVector<Plane> &p_planes) {
	int num_roamers = p_room._roamer_pool_ids.size();

	for (int n = 0; n < num_roamers; n++) {
		uint32_t pool_id = p_room._roamer_pool_ids[n];

		PortalRenderer::Moving &moving = _portal_renderer->get_pool_moving(pool_id);

		// done already?
		if (moving.last_tick_hit == _tick) {
			continue;
		}

		if (test_cull_inside(moving.exact_aabb, p_planes)) {
			if (!_occlusion_culler.cull_aabb(moving.exact_aabb)) {
				// mark as done (and on visible list)
				moving.last_tick_hit = _tick;

				_result->visible_roamer_pool_ids.push_back(pool_id);
			}
		}
	}
}

void PortalTracer::cull_statics_debug_sprawl(const VSRoom &p_room) {
	int num_statics = p_room._static_ids.size();

	for (int n = 0; n < num_statics; n++) {
		uint32_t static_id = p_room._static_ids[n];

		// VSStatic &stat = _portal_renderer->get_static(static_id);

		// deal with dynamic stats
		// if (stat.dynamic) {
		// VSG::scene->_instance_get_transformed_aabb(stat.instance, stat.aabb);
		// }

		// set the visible bit if not set
		if (!_result->bf_visible_statics.check_and_set(static_id)) {
			_result->visible_static_ids.push_back(static_id);
		}
	}
}

void PortalTracer::cull_statics(const VSRoom &p_room, const LocalVector<Plane> &p_planes) {
	int num_statics = p_room._static_ids.size();

	for (int n = 0; n < num_statics; n++) {
		uint32_t static_id = p_room._static_ids[n];

		VSStatic &stat = _portal_renderer->get_static(static_id);

		// deal with dynamic stats
		if (stat.dynamic) {
			VSG::scene->_instance_get_transformed_aabb(stat.instance, stat.aabb);
		}

		// estimate the radius .. for now
		const AABB &bb = stat.aabb;

		// print("\t\t\tculling object " + pObj->get_name());

		if (test_cull_inside(bb, p_planes)) {
			if (_occlusion_culler.cull_aabb(bb)) {
				continue;
			}

			// bypass the bitfield for now and just show / hide
			//stat.show(bShow);

			// set the visible bit if not set
			if (_result->bf_visible_statics.check_and_set(static_id)) {
				// if wasn't previously set, add to the visible list
				_result->visible_static_ids.push_back(static_id);
			}
		}

	} // for n through statics
}

int PortalTracer::trace_globals(const LocalVector<Plane> &p_planes, VSInstance **p_result_array, int first_result, int p_result_max, uint32_t p_mask, bool p_override_camera) {
	uint32_t num_globals = _portal_renderer->get_num_moving_globals();
	int current_result = first_result;

	if (!p_override_camera) {
		for (uint32_t n = 0; n < num_globals; n++) {
			const PortalRenderer::Moving &moving = _portal_renderer->get_moving_global(n);

#ifdef PORTAL_RENDERER_STORE_MOVING_RIDS
			// debug check the instance is valid
			void *vss_instance = VSG::scene->_instance_get_from_rid(moving.instance_rid);

			if (vss_instance) {
#endif
				if (test_cull_inside(moving.exact_aabb, p_planes, false)) {
					if (VSG::scene->_instance_cull_check(moving.instance, p_mask)) {
						p_result_array[current_result++] = moving.instance;

						// full up?
						if (current_result >= p_result_max) {
							return current_result;
						}
					}
				}

#ifdef PORTAL_RENDERER_STORE_MOVING_RIDS
			} else {
				WARN_PRINT("vss instance is null " + PortalRenderer::_addr_to_string(moving.instance));
			}
#endif
		}
	} // if not override camera
	else {
		// If we are overriding the camera there is a potential problem in the editor:
		// gizmos BEHIND the override camera will not be drawn.
		// As this should be editor only and performance is not critical, we will just disable
		// frustum culling for global objects when the camera is overridden.
		for (uint32_t n = 0; n < num_globals; n++) {
			const PortalRenderer::Moving &moving = _portal_renderer->get_moving_global(n);

			if (VSG::scene->_instance_cull_check(moving.instance, p_mask)) {
				p_result_array[current_result++] = moving.instance;

				// full up?
				if (current_result >= p_result_max) {
					return current_result;
				}
			}
		}
	} // if override camera

	return current_result;
}

void PortalTracer::trace_debug_sprawl_recursive(int p_depth, int p_room_id) {
	if (p_depth > 1) {
		return;
	}

	// prevent too much depth
	ERR_FAIL_COND_MSG(p_depth > 8, "Portal Depth Limit reached");

	// get the room
	const VSRoom &room = _portal_renderer->get_room(p_room_id);

	int num_portals = room._portal_ids.size();

	for (int p = 0; p < num_portals; p++) {
		const VSPortal &portal = _portal_renderer->get_portal(room._portal_ids[p]);

		if (!portal._active) {
			continue;
		}

		cull_statics_debug_sprawl(room);

		// everything depends on whether the portal is incoming or outgoing.
		int outgoing = 1;

		int room_a_id = portal._linkedroom_ID[0];
		if (room_a_id != p_room_id) {
			outgoing = 0;
			DEV_ASSERT(portal._linkedroom_ID[1] == p_room_id);
		}

		// trace through this portal to the next room
		int linked_room_id = portal._linkedroom_ID[outgoing];

		if (linked_room_id != -1) {
			trace_debug_sprawl_recursive(p_depth + 1, linked_room_id);
		} // if a linked room exists

	} // for p through portals
}

void PortalTracer::trace_pvs(int p_source_room_id, const LocalVector<Plane> &p_planes) {
	const PVS &pvs = _portal_renderer->get_pvs();
	const VSRoom &source_room = _portal_renderer->get_room(p_source_room_id);

	for (int r = 0; r < source_room._pvs_size; r++) {
		int room_id = pvs.get_pvs_room_id(source_room._pvs_first + r);

		// get the room
		const VSRoom &room = _portal_renderer->get_room(room_id);

		cull_statics(room, p_planes);
		cull_roamers(room, p_planes);
	}
}

void PortalTracer::trace_recursive(const TraceParams &p_params, int p_depth, int p_room_id, const LocalVector<Plane> &p_planes, int p_from_external_room_id) {
	// Prevent too much depth.
	if (p_depth > _depth_limit) {
		WARN_PRINT_ONCE("Portal Depth Limit reached (seeing through too many portals)");
		return;
	}

	// Get the room.
	const VSRoom &room = _portal_renderer->get_room(p_room_id);

	// Set up the occlusion culler as a one off.
	_occlusion_culler.prepare(*_portal_renderer, room, _trace_start_point, p_planes, &_near_and_far_planes[0]);

	cull_statics(room, p_planes);
	cull_roamers(room, p_planes);

	int num_portals = room._portal_ids.size();

	for (int p = 0; p < num_portals; p++) {
		const VSPortal &portal = _portal_renderer->get_portal(room._portal_ids[p]);

		// Portals can be switched on and off at runtime, like opening and closing a door.
		if (!portal._active) {
			continue;
		}

		// Everything depends on whether the portal is incoming or outgoing.
		// If incoming we reverse the logic.
		int outgoing = 1;

		int room_a_id = portal._linkedroom_ID[0];
		if (room_a_id != p_room_id) {
			outgoing = 0;
			DEV_ASSERT(portal._linkedroom_ID[1] == p_room_id);
		}

		// Trace through this portal to the next room.
		int linked_room_id = portal._linkedroom_ID[outgoing];

		// Cull by PVS.
		if (p_params.use_pvs && (!p_params.decompressed_room_pvs[linked_room_id])) {
			continue;
		}

		// Cull by portal angle to camera.

		// Much better way of culling portals by direction to camera...
		// instead of using dot product with a varying view direction, we simply find which side of the portal
		// plane the camera is on! If it is behind, the portal can be seen through, if in front, it can't.

		// There is one exception to this, for the first source room. If we are in front of any portal in the first
		// source room, we will render EVERYTHING through it into the next room. This can happen due
		// to precision errors, or inaccuracy in setting up the portal planes relative to the room bounds -
		// in which case we can end up IN FRONT of a portal in the same room.

		bool start_room_in_front_portal_exception = false;

		/////////////////////////////////////////////
		real_t dist_cam = portal._plane.distance_to(_trace_start_point);

		if (!outgoing) {
			dist_cam = -dist_cam;
		}

		// If the camera is IN FRONT of the portal plane...
		if (dist_cam >= 0.0) {
			if ((p_room_id != p_params.start_room_id) || portal._internal) {
				continue;
			} else {
				start_room_in_front_portal_exception = true;
			}
		}
		/////////////////////////////////////////////

		// While clipping to the planes we maintain a list of partial planes, so we can add them to the
		// recursive next iteration of planes to check.
		static LocalVector<int> partial_planes;
		partial_planes.clear();

		// Is it culled by the planes?
		VSPortal::ClipResult overall_res = VSPortal::ClipResult::CLIP_INSIDE;
		if (!start_room_in_front_portal_exception) {
			// For portals, we want to ignore the near clipping plane, as we might be right on the edge of a doorway
			// and still want to look through the portal.
			// So earlier we have set it that the first plane (ASSUMING that plane zero is the near clipping plane)
			// starts from the camera position, and NOT the actual near clipping plane.
			// If we need quite a distant near plane, we may need a different strategy.
			for (uint32_t l = 0; l < p_planes.size(); l++) {
				VSPortal::ClipResult res = portal.clip_with_plane(p_planes[l]);

				switch (res) {
					case VSPortal::ClipResult::CLIP_OUTSIDE: {
						overall_res = res;
					} break;
					case VSPortal::ClipResult::CLIP_PARTIAL: {
						// If the portal intersects one of the planes, we should take this plane into account
						// in the next call of this recursive trace, because it can be used to cull out more objects.
						overall_res = res;
						partial_planes.push_back(l);
					} break;
					default: // Suppress warning.
						break;
				}

				// If the portal was totally outside the 'frustum' then we can ignore it.
				if (overall_res == VSPortal::ClipResult::CLIP_OUTSIDE)
					break;
			}

			// This portal is culled.
			if (overall_res == VSPortal::ClipResult::CLIP_OUTSIDE) {
				continue;
			}
		}

		// Don't allow portals from internal to external room to be followed
		// if the external room has already been processed in this trace stack. This prevents
		// unneeded processing, and also prevents recursive feedback where you
		// see into internal room -> external room and back into the same internal room
		// via the same portal.
		if (portal._internal && (linked_room_id != -1)) {
			if (outgoing) {
				if (linked_room_id == p_from_external_room_id) {
					continue;
				}
			} else {
				// We are entering an internal portal from an external room.
				// set the external room id, so we can recognise this when we are
				// later exiting the internal rooms.
				// Note that as we can only store 1 previous external room, this system
				// won't work completely correctly when you have 2 levels of internal room
				// and you can see from roomgroup a -> b -> c. However this should just result
				// in a little slower culling for that particular view, and hopefully will not break
				// with recursive loop looking through the same portal multiple times. (don't think this
				// is possible in this scenario).
				p_from_external_room_id = p_room_id;
			}
		}

		// Occlusion culling of portals.
		if (_occlusion_culler.cull_sphere(portal._pt_center, portal._bounding_sphere_radius)) {
			continue;
		}

		// Hopefully the portal actually leads somewhere...
		if (linked_room_id != -1) {
			// We need some new planes.
			unsigned int pool_mem = _planes_pool.request();

			// If the planes pool is not empty, we got some planes, and can recurse.
			if (pool_mem != (unsigned int)-1) {
				// Get a new vector of planes from the pool.
				LocalVector<Plane> &new_planes = _planes_pool.get(pool_mem);

				// Makes sure there are none left over (as the pool may not clear them).
				new_planes.clear();

				if (!start_room_in_front_portal_exception) {
					// If portal is totally inside the planes, don't copy the old planes ..
					// i.e. we can now cull using the portal and forget about the rest of the frustum (yay).
					// Note that this loses the far clipping plane .. but that shouldn't be important usually?
					// (maybe we might need to account for this in future .. look for issues)
					if (overall_res != VSPortal::ClipResult::CLIP_INSIDE) {
						// If it WASN'T totally inside the existing frustum, we also need to add any existing planes
						// that cut the portal.
						for (uint32_t n = 0; n < partial_planes.size(); n++) {
							new_planes.push_back(p_planes[partial_planes[n]]);
						}
					}

					// We will always add the portals planes. This could probably be optimized, as some
					// portal planes may be culled out by partial planes... NYI
					portal.add_planes(_trace_start_point, new_planes, outgoing != 0);

					// Always add the far plane. It is likely the portal is inside the far plane,
					// but it is still needed in future for culling portals and objects.
					// Note that there is a small possibility of far plane being added twice here
					// in some situations, but I don't think it should be a problem.
					// The fake near plane BTW is almost never added (otherwise it would prematurely
					// break traversal through the portals), so near clipping must be done
					// explicitly on objects.
					new_planes.push_back(_near_and_far_planes[1]);
				} else {
					// start_room_in_front_portal_exception
					// Copy the existing planes and reuse when tracing into the next room.
					new_planes = p_planes;
				}

				// Go and do the whole lot again in the next room...
				trace_recursive(p_params, p_depth + 1, linked_room_id, new_planes, p_from_external_room_id);

				// We no longer need these planes, return them to the pool.
				_planes_pool.free(pool_mem);

			} // pool mem allocated
			else {
				// Planes pool is empty!
				// This will happen if the view goes through shedloads of portals.
				// The solution is either to increase the plane pool size, or not build levels
				// with views through multiple portals. Looking through multiple portals is likely to be
				// slow anyway because of the number of planes to test.
				WARN_PRINT_ONCE("planes pool is empty");
				// Note we also have a depth check at the top of this function. Which will probably get hit
				// before the pool gets empty.
			}

		} // if a linked room exists
	} // for p through portals
}

int PortalTracer::occlusion_cull(PortalRenderer &p_portal_renderer, const Vector3 &p_point, const Vector3 &p_cam_dir, const CameraMatrix &p_cam_matrix, const Vector<Plane> &p_convex, VSInstance **p_result_array, int p_num_results) {
	_occlusion_culler.prepare_camera(p_cam_matrix, p_cam_dir);

	// silly conversion of vector to local vector
	// can this be avoided? NYI
	// pretty cheap anyway as it will just copy 6 planes, max a few times per frame...
	static LocalVector<Plane> local_planes;
	if ((int)local_planes.size() != p_convex.size()) {
		local_planes.resize(p_convex.size());
	}
	for (int n = 0; n < p_convex.size(); n++) {
		local_planes[n] = p_convex[n];
	}

	_occlusion_culler.prepare_generic(p_portal_renderer, p_portal_renderer.get_occluders_active_list(), p_point, local_planes);

	// cull each instance
	int count = p_num_results;
	AABB bb;

	for (int n = 0; n < count; n++) {
		VSInstance *instance = p_result_array[n];

		// this will return false for GLOBAL instances, so we don't occlusion cull gizmos
		if (VSG::scene->_instance_get_transformed_aabb_for_occlusion(instance, bb)) {
			if (_occlusion_culler.cull_aabb(bb)) {
				// remove from list with unordered swap from the end of list
				p_result_array[n] = p_result_array[count - 1];
				count--;
				n--; // repeat this element, as it will have changed
			}
		}
	}

	return count;
}
