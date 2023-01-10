/**************************************************************************/
/*  portal_rooms_bsp.h                                                    */
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

#ifndef PORTAL_ROOMS_BSP_H
#define PORTAL_ROOMS_BSP_H

#include "core/local_vector.h"
#include "core/math/aabb.h"
#include "core/math/plane.h"

class PortalRenderer;
struct VSPortal;
struct VSRoom;

class PortalRoomsBSP {
	struct Node {
		Node() { clear(); }
		void clear() {
			leaf = false;
			child[0] = -1;
			child[1] = -1;
		}
		bool leaf;
		union {
			int32_t child[2];
			struct {
				int32_t first_id;
				int32_t num_ids;
			};
		};
		Plane plane;
	};

	LocalVector<Node, int32_t> _nodes;
	LocalVector<int32_t, int32_t> _room_ids;

	PortalRenderer *_portal_renderer = nullptr;

	const real_t _plane_epsilon = 0.001;

public:
	// build the BSP on level start
	void create(PortalRenderer &r_portal_renderer);

	// clear data, and ready for a new level
	void clear() {
		_nodes.reset();
		_room_ids.reset();
	}

	// the main function, returns a shortlist of rooms that are possible for a test point
	const int32_t *find_shortlist(const Vector3 &p_pt, int &r_num_rooms) const;

	// This is a 'sticky' function, it prefers to stay in the previous room where possible.
	// This means there is a hysteresis for room choice that may occur if the user creates
	// overlapping rooms...
	int find_room_within(const PortalRenderer &p_portal_renderer, const Vector3 &p_pos, int p_previous_room_id) const;

private:
	void build(int p_start_node_id, LocalVector<int32_t, int32_t> p_orig_room_ids);
	void detect_internal_room_containment(PortalRenderer &r_portal_renderer);

	int evaluate_portal(int p_portal_id, const LocalVector<int32_t, int32_t> &p_room_ids, LocalVector<int32_t, int32_t> *r_room_ids_back = nullptr, LocalVector<int32_t, int32_t> *r_room_ids_front = nullptr);

	int evaluate_room_split_plane(int p_room_a_id, int p_room_b_id, const LocalVector<int32_t, int32_t> &p_room_ids, Plane &r_plane, LocalVector<int32_t, int32_t> *r_room_ids_back = nullptr, LocalVector<int32_t, int32_t> *r_room_ids_front = nullptr);

	int evaluate_plane(const VSPortal *p_portal, const Plane &p_plane, const LocalVector<int32_t, int32_t> &p_room_ids, LocalVector<int32_t, int32_t> *r_room_ids_back = nullptr, LocalVector<int32_t, int32_t> *r_room_ids_front = nullptr);

	bool calculate_aabb_splitting_plane(const AABB &p_a, const AABB &p_b, Plane &r_plane) const;
	bool calculate_freeform_splitting_plane(const VSRoom &p_room_a, const VSRoom &p_room_b, Plane &r_plane) const;
	bool find_1d_split_point(real_t p_min_a, real_t p_max_a, real_t p_min_b, real_t p_max_b, real_t &r_split_point) const;
	bool test_freeform_plane(const LocalVector<Vector3, int32_t> &p_verts_a, const LocalVector<Vector3, int32_t> &p_verts_b, const Plane &p_plane) const;

	void debug_print_tree(int p_node_id = 0, int p_depth = 0);

	void _log(String p_string);
};

#endif // PORTAL_ROOMS_BSP_H
