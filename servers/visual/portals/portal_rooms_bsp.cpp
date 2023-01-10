/**************************************************************************/
/*  portal_rooms_bsp.cpp                                                  */
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

#include "portal_rooms_bsp.h"

#include "core/math/geometry.h"
#include "core/math/plane.h"
#include "core/print_string.h"
#include "core/variant.h"
#include "portal_renderer.h"

// #define GODOT_VERBOSE_PORTAL_ROOMS_BSP

void PortalRoomsBSP::_log(String p_string) {
#ifdef GODOT_VERBOSE_PORTAL_ROOMS_BSP
	print_line(p_string);
#endif
}

// rooms which contain internal rooms cannot use the optimization where it terminates the search for
// room within if inside the previous room. We can't use just use the rooms already marked as internal due
// to a portal leading to them, because the internal room network may spread into another room (e.g. terrain)
// which has internal room exit portal. So we need to detect manually all cases of overlap of internal rooms,
// and set the flag.
void PortalRoomsBSP::detect_internal_room_containment(PortalRenderer &r_portal_renderer) {
	int num_rooms = r_portal_renderer.get_num_rooms();

	for (int n = 0; n < num_rooms; n++) {
		VSRoom &room = r_portal_renderer.get_room(n);
		if (room._contains_internal_rooms) {
			// already established it contains internal rooms, no need to test
			continue;
		}

		// safety
		if (!room._planes.size()) {
			continue;
		}

		for (int i = 0; i < num_rooms; i++) {
			// don't test against ourself
			if (n == i) {
				continue;
			}

			// only interested in rooms with a higher priority, these are potential internal rooms
			const VSRoom &other = r_portal_renderer.get_room(i);
			if (other._priority <= room._priority) {
				continue;
			}

			// quick aabb check first
			if (!room._aabb.intersects(other._aabb)) {
				continue;
			}

			// safety
			if (!other._planes.size()) {
				continue;
			}

			if (Geometry::convex_hull_intersects_convex_hull(&room._planes[0], room._planes.size(), &other._planes[0], other._planes.size())) {
				// it intersects an internal room
				room._contains_internal_rooms = true;
				break;
			}
		}
	}
}

int PortalRoomsBSP::find_room_within(const PortalRenderer &p_portal_renderer, const Vector3 &p_pos, int p_previous_room_id) const {
	real_t closest = FLT_MAX;
	int closest_room_id = -1;
	int closest_priority = -10000;

	// first try previous room
	if (p_previous_room_id != -1) {
		if (p_previous_room_id < p_portal_renderer.get_num_rooms()) {
			const VSRoom &prev_room = p_portal_renderer.get_room(p_previous_room_id);

			// we can only use this shortcut if the room doesn't include internal rooms.
			// otherwise the point may be inside more than one room, and we need to find the room of highest priority.
			if (!prev_room._contains_internal_rooms) {
				closest = prev_room.is_point_within(p_pos);
				closest_room_id = p_previous_room_id;

				if (closest < 0.0) {
					return p_previous_room_id;
				}
			} else {
				// don't mark it as checked later, as we haven't done it because it contains internal rooms
				p_previous_room_id = -1;
			}
		} else {
			// previous room was out of range (perhaps due to reconverting room system and the number of rooms decreasing)
			p_previous_room_id = -1;
		}
	}

	int num_bsp_rooms = 0;
	const int32_t *bsp_rooms = find_shortlist(p_pos, num_bsp_rooms);
	if (!num_bsp_rooms) {
		return -1;
	}

	// special case, only 1 room in the shortlist, no need to check further
	if (num_bsp_rooms == 1) {
		return bsp_rooms[0];
	}

	for (int n = 0; n < num_bsp_rooms; n++) {
		int room_id = bsp_rooms[n];

		// the previous room has already been done above, and will be in closest + closest_room_id
		if (room_id == p_previous_room_id) {
			continue;
		}

		const VSRoom &room = p_portal_renderer.get_room(room_id);
		real_t dist = room.is_point_within(p_pos);

		// if we are actually inside a room, unless we are dealing with internal rooms,
		// we can terminate early, no need to search more
		if (dist < 0.0) {
			if (!room._contains_internal_rooms) {
				// this will happen in most cases
				closest = dist;
				closest_room_id = room_id;
				break;
			} else {
				// if we are inside, and there are internal rooms involved we need to be a bit careful.
				// higher priority always wins (i.e. the internal room)
				// but with equal priority we just choose the regular best fit.
				if ((room._priority > closest_priority) || ((room._priority == closest_priority) && (dist < closest))) {
					closest = dist;
					closest_room_id = room_id;
					closest_priority = room._priority;
					continue;
				}
			}
		} else {
			// if we are outside we just pick the closest room, irrespective of priority
			if (dist < closest) {
				closest = dist;
				closest_room_id = room_id;
				// do NOT store the priority, we don't want an room that isn't a true hit
				// overriding a hit inside the room
			}
		}
	}

	return closest_room_id;
}

const int32_t *PortalRoomsBSP::find_shortlist(const Vector3 &p_pt, int &r_num_rooms) const {
	if (!_nodes.size()) {
		r_num_rooms = 0;
		return nullptr;
	}

	const Node *node = &_nodes[0];

	while (!node->leaf) {
		if (node->plane.is_point_over(p_pt)) {
			node = &_nodes[node->child[1]];
		} else {
			node = &_nodes[node->child[0]];
		}
	}

	r_num_rooms = node->num_ids;
	return &_room_ids[node->first_id];
}

void PortalRoomsBSP::create(PortalRenderer &r_portal_renderer) {
	clear();
	_portal_renderer = &r_portal_renderer;
	detect_internal_room_containment(r_portal_renderer);

	// noop
	int num_rooms = r_portal_renderer.get_num_rooms();

	if (!num_rooms) {
		return;
	}

	LocalVector<int32_t, int32_t> room_ids;
	room_ids.resize(num_rooms);
	for (int n = 0; n < num_rooms; n++) {
		room_ids[n] = n;
	}

	_nodes.push_back(Node());
	_nodes[0].clear();

	build(0, room_ids);

#ifdef GODOT_VERBOSE_PORTAL_ROOMS_BSP
	debug_print_tree();
#endif
	_log("PortalRoomsBSP " + itos(_nodes.size()) + " nodes.");
}

void PortalRoomsBSP::build(int p_start_node_id, LocalVector<int32_t, int32_t> p_orig_room_ids) {
	struct Element {
		void clear() { room_ids.clear(); }
		int node_id;
		LocalVector<int32_t, int32_t> room_ids;
	};

	Element first;
	first.node_id = p_start_node_id;
	first.room_ids = p_orig_room_ids;

	LocalVector<Element, int32_t> stack;
	stack.reserve(1024);
	stack.push_back(first);
	int stack_size = 1;

	while (stack_size) {
		stack_size--;
		Element curr = stack[stack_size];

		Node *node = &_nodes[curr.node_id];

		int best_fit = 0;
		int best_portal_id = -1;
		int best_room_a = -1;
		int best_room_b = -1;

		// find a splitting plane
		for (int n = 0; n < curr.room_ids.size(); n++) {
			// go through the portals in this room
			int rid = curr.room_ids[n];
			const VSRoom &room = _portal_renderer->get_room(rid);

			for (int p = 0; p < room._portal_ids.size(); p++) {
				int pid = room._portal_ids[p];
				// only outward portals
				const VSPortal &portal = _portal_renderer->get_portal(pid);
				if (portal._linkedroom_ID[1] == rid) {
					continue;
				}

				int fit = evaluate_portal(pid, curr.room_ids);
				if (fit > best_fit) {
					best_fit = fit;
					best_portal_id = pid;
				}
			}
		}

		bool split_found = false;
		Plane split_plane;

		// if a splitting portal was found, we are done
		if (best_portal_id != -1) {
			_log("found splitting portal : " + itos(best_portal_id));

			const VSPortal &portal = _portal_renderer->get_portal(best_portal_id);
			split_plane = portal._plane;
			split_found = true;
		} else {
			// let's try and find an arbitrary splitting plane
			for (int a = 0; a < curr.room_ids.size(); a++) {
				for (int b = a + 1; b < curr.room_ids.size(); b++) {
					Plane plane;

					// note the actual room ids are not the same as a and b!!
					int room_a_id = curr.room_ids[a];
					int room_b_id = curr.room_ids[b];

					int fit = evaluate_room_split_plane(room_a_id, room_b_id, curr.room_ids, plane);

					if (fit > best_fit) {
						best_fit = fit;

						// the room ids, NOT a and b
						best_room_a = room_a_id;
						best_room_b = room_b_id;
						split_plane = plane;
					}
				} // for b through rooms
			} // for a through rooms

			if (best_room_a != -1) {
				split_found = true;
				// print_line("found splitting plane between rooms : " + itos(best_room_a) + " and " + itos(best_room_b));
			}
		}

		// found either a portal plane or arbitrary
		if (split_found) {
			node->plane = split_plane;

			// add to stack
			stack_size += 2;
			if (stack_size > stack.size()) {
				stack.resize(stack_size);
			}
			stack[stack_size - 2].clear();
			stack[stack_size - 1].clear();

			LocalVector<int32_t, int32_t> &room_ids_back = stack[stack_size - 2].room_ids;
			LocalVector<int32_t, int32_t> &room_ids_front = stack[stack_size - 1].room_ids;

			if (best_portal_id != -1) {
				evaluate_portal(best_portal_id, curr.room_ids, &room_ids_back, &room_ids_front);
			} else {
				DEV_ASSERT(best_room_a != -1);
				evaluate_room_split_plane(best_room_a, best_room_b, curr.room_ids, split_plane, &room_ids_back, &room_ids_front);
			}

			DEV_ASSERT(room_ids_back.size() <= curr.room_ids.size());
			DEV_ASSERT(room_ids_front.size() <= curr.room_ids.size());

			_log("\tback contains : " + itos(room_ids_back.size()) + " rooms");
			_log("\tfront contains : " + itos(room_ids_front.size()) + " rooms");

			// create child nodes
			_nodes.push_back(Node());
			_nodes.push_back(Node());

			// need to reget the node pointer as we may have resized the vector
			node = &_nodes[curr.node_id];

			node->child[0] = _nodes.size() - 2;
			node->child[1] = _nodes.size() - 1;

			stack[stack_size - 2].node_id = node->child[0];
			stack[stack_size - 1].node_id = node->child[1];

		} else {
			// couldn't split any further, is leaf
			node->leaf = true;
			node->first_id = _room_ids.size();
			node->num_ids = curr.room_ids.size();

			_log("leaf contains : " + itos(curr.room_ids.size()) + " rooms");

			// add to the main list
			int start = _room_ids.size();
			_room_ids.resize(start + curr.room_ids.size());
			for (int n = 0; n < curr.room_ids.size(); n++) {
				_room_ids[start + n] = curr.room_ids[n];
			}
		}

	} // while stack not empty
}

void PortalRoomsBSP::debug_print_tree(int p_node_id, int p_depth) {
	String string = "";
	for (int n = 0; n < p_depth; n++) {
		string += "\t";
	}

	const Node &node = _nodes[p_node_id];
	if (node.leaf) {
		string += "L ";
		for (int n = 0; n < node.num_ids; n++) {
			int room_id = _room_ids[node.first_id + n];
			string += itos(room_id) + ", ";
		}
	} else {
		string += "N ";
	}

	print_line(string);

	// children
	if (!node.leaf) {
		debug_print_tree(node.child[0], p_depth + 1);
		debug_print_tree(node.child[1], p_depth + 1);
	}
}

bool PortalRoomsBSP::find_1d_split_point(real_t p_min_a, real_t p_max_a, real_t p_min_b, real_t p_max_b, real_t &r_split_point) const {
	if (p_max_a <= p_min_b) {
		r_split_point = p_max_a + ((p_min_b - p_max_a) * 0.5);
		return true;
	}
	if (p_max_b <= p_min_a) {
		r_split_point = p_max_b + ((p_min_a - p_max_b) * 0.5);
		return true;
	}

	return false;
}

bool PortalRoomsBSP::test_freeform_plane(const LocalVector<Vector3, int32_t> &p_verts_a, const LocalVector<Vector3, int32_t> &p_verts_b, const Plane &p_plane) const {
	// print_line("test_freeform_plane " + String(Variant(p_plane)));

	for (int n = 0; n < p_verts_a.size(); n++) {
		real_t dist = p_plane.distance_to(p_verts_a[n]);
		// print_line("\tdist_a " + String(Variant(dist)));
		if (dist > _plane_epsilon) {
			return false;
		}
	}

	for (int n = 0; n < p_verts_b.size(); n++) {
		real_t dist = p_plane.distance_to(p_verts_b[n]);
		// print_line("\tdist_b " + String(Variant(dist)));
		if (dist < -_plane_epsilon) {
			return false;
		}
	}

	return true;
}

// even if AABBs fail to have a splitting plane, there still may be another orientation that can split rooms (e.g. diagonal)
bool PortalRoomsBSP::calculate_freeform_splitting_plane(const VSRoom &p_room_a, const VSRoom &p_room_b, Plane &r_plane) const {
	const LocalVector<Vector3, int32_t> &verts_a = p_room_a._verts;
	const LocalVector<Vector3, int32_t> &verts_b = p_room_b._verts;

	// test from room a to room b
	for (int i = 0; i < verts_a.size(); i++) {
		const Vector3 &pt_a = verts_a[i];

		for (int j = 0; j < verts_b.size(); j++) {
			const Vector3 &pt_b = verts_b[j];

			for (int k = j + 1; k < verts_b.size(); k++) {
				const Vector3 &pt_c = verts_b[k];

				// make a plane
				r_plane = Plane(pt_a, pt_b, pt_c);

				// test the plane
				if (test_freeform_plane(verts_a, verts_b, r_plane)) {
					return true;
				}
			}
		}
	}

	// test from room b to room a
	for (int i = 0; i < verts_b.size(); i++) {
		const Vector3 &pt_a = verts_b[i];

		for (int j = 0; j < verts_a.size(); j++) {
			const Vector3 &pt_b = verts_a[j];

			for (int k = j + 1; k < verts_a.size(); k++) {
				const Vector3 &pt_c = verts_a[k];

				// make a plane
				r_plane = Plane(pt_a, pt_b, pt_c);

				// test the plane
				if (test_freeform_plane(verts_b, verts_a, r_plane)) {
					return true;
				}
			}
		}
	}

	return false;
}

bool PortalRoomsBSP::calculate_aabb_splitting_plane(const AABB &p_a, const AABB &p_b, Plane &r_plane) const {
	real_t split_point = 0.0;

	const Vector3 &min_a = p_a.position;
	const Vector3 &min_b = p_b.position;
	Vector3 max_a = min_a + p_a.size;
	Vector3 max_b = min_b + p_b.size;

	if (find_1d_split_point(min_a.x, max_a.x, min_b.x, max_b.x, split_point)) {
		r_plane = Plane(Vector3(1, 0, 0), split_point);
		return true;
	}
	if (find_1d_split_point(min_a.y, max_a.y, min_b.y, max_b.y, split_point)) {
		r_plane = Plane(Vector3(0, 1, 0), split_point);
		return true;
	}
	if (find_1d_split_point(min_a.z, max_a.z, min_b.z, max_b.z, split_point)) {
		r_plane = Plane(Vector3(0, 0, 1), split_point);
		return true;
	}

	return false;
}

int PortalRoomsBSP::evaluate_room_split_plane(int p_room_a_id, int p_room_b_id, const LocalVector<int32_t, int32_t> &p_room_ids, Plane &r_plane, LocalVector<int32_t, int32_t> *r_room_ids_back, LocalVector<int32_t, int32_t> *r_room_ids_front) {
	// try and create a splitting plane between room a and b, then evaluate it.
	const VSRoom &room_a = _portal_renderer->get_room(p_room_a_id);
	const VSRoom &room_b = _portal_renderer->get_room(p_room_b_id);

	// easiest case, if the rooms don't overlap AABB, we can create an axis aligned plane between them
	if (calculate_aabb_splitting_plane(room_a._aabb, room_b._aabb, r_plane)) {
		return evaluate_plane(nullptr, r_plane, p_room_ids, r_room_ids_back, r_room_ids_front);
	}

	if (calculate_freeform_splitting_plane(room_a, room_b, r_plane)) {
		return evaluate_plane(nullptr, r_plane, p_room_ids, r_room_ids_back, r_room_ids_front);
	}

	return 0;
}

int PortalRoomsBSP::evaluate_plane(const VSPortal *p_portal, const Plane &p_plane, const LocalVector<int32_t, int32_t> &p_room_ids, LocalVector<int32_t, int32_t> *r_room_ids_back, LocalVector<int32_t, int32_t> *r_room_ids_front) {
	int rooms_front = 0;
	int rooms_back = 0;

	if (r_room_ids_back) {
		DEV_ASSERT(!r_room_ids_back->size());
	}

	if (r_room_ids_front) {
		DEV_ASSERT(!r_room_ids_front->size());
	}

#define GODOT_BSP_PUSH_FRONT              \
	rooms_front++;                        \
	if (r_room_ids_front) {               \
		r_room_ids_front->push_back(rid); \
	}

#define GODOT_BSP_PUSH_BACK              \
	rooms_back++;                        \
	if (r_room_ids_back) {               \
		r_room_ids_back->push_back(rid); \
	}

	for (int n = 0; n < p_room_ids.size(); n++) {
		int rid = p_room_ids[n];
		const VSRoom &room = _portal_renderer->get_room(rid);

		// easy cases first
		real_t r_min, r_max;
		room._aabb.project_range_in_plane(p_plane, r_min, r_max);

		if ((r_min <= 0.0) && (r_max <= 0.0)) {
			GODOT_BSP_PUSH_BACK
			continue;
		}
		if ((r_min >= 0.0) && (r_max >= 0.0)) {
			GODOT_BSP_PUSH_FRONT
			continue;
		}

		// check if the room uses this portal
		// internal portals can link to a room that is both in front and behind,
		// so we can only deal with non internal portals here with this cheap test.
		if (p_portal && !p_portal->_internal) {
			if (p_portal->_linkedroom_ID[0] == rid) {
				GODOT_BSP_PUSH_BACK
				continue;
			}

			if (p_portal->_linkedroom_ID[1] == rid) {
				GODOT_BSP_PUSH_FRONT
				continue;
			}
		}

		// most expensive test, test the individual points of the room
		// This will catch some off axis rooms that aren't caught by the AABB alone
		int points_front = 0;
		int points_back = 0;

		for (int p = 0; p < room._verts.size(); p++) {
			const Vector3 &pt = room._verts[p];
			real_t dist = p_plane.distance_to(pt);

			// don't take account of points in the epsilon zone,
			// these are within the margin of error and could be in front OR behind the plane
			if (dist > _plane_epsilon) {
				points_front++;
				if (points_back) {
					break;
				}
			} else if (dist < -_plane_epsilon) {
				points_back++;
				if (points_front) {
					break;
				}
			}
		}

		// if all points are in front
		if (!points_back) {
			GODOT_BSP_PUSH_FRONT
			continue;
		}
		// if all points are behind
		if (!points_front) {
			GODOT_BSP_PUSH_BACK
			continue;
		}

		// if split, push to both children
		if (r_room_ids_front) {
			r_room_ids_front->push_back(rid);
		}
		if (r_room_ids_back) {
			r_room_ids_back->push_back(rid);
		}
	}

#undef GODOT_BSP_PUSH_BACK
#undef GODOT_BSP_PUSH_FRONT

	// we want the split that splits the most front and back rooms
	return rooms_front * rooms_back;
}

int PortalRoomsBSP::evaluate_portal(int p_portal_id, const LocalVector<int32_t, int32_t> &p_room_ids, LocalVector<int32_t, int32_t> *r_room_ids_back, LocalVector<int32_t, int32_t> *r_room_ids_front) {
	const VSPortal &portal = _portal_renderer->get_portal(p_portal_id);
	const Plane &plane = portal._plane;

	return evaluate_plane(&portal, plane, p_room_ids, r_room_ids_back, r_room_ids_front);
}
