/**************************************************************************/
/*  nav_utils.h                                                           */
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

#ifndef NAV_UTILS_H
#define NAV_UTILS_H

#include "core/math/vector3.h"
#include "core/templates/hash_map.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/local_vector.h"

class NavBase;

namespace gd {
struct Polygon;

union PointKey {
	struct {
		int64_t x : 21;
		int64_t y : 22;
		int64_t z : 21;
	};

	uint64_t key = 0;
};

struct EdgeKey {
	PointKey a;
	PointKey b;

	static uint32_t hash(const EdgeKey &p_val) {
		return hash_one_uint64(p_val.a.key) ^ hash_one_uint64(p_val.b.key);
	}

	bool operator==(const EdgeKey &p_key) const {
		return (a.key == p_key.a.key) && (b.key == p_key.b.key);
	}

	EdgeKey(const PointKey &p_a = PointKey(), const PointKey &p_b = PointKey()) :
			a(p_a),
			b(p_b) {
		if (a.key > b.key) {
			SWAP(a, b);
		}
	}
};

struct Connection {
	/// Polygon that this connection leads to.
	Polygon *polygon = nullptr;

	/// Edge of the source polygon where this connection starts from.
	int edge = -1;

	/// Is this a connection to a polygon from a different region?
	bool is_external = false;

	/// Point on the edge where the gateway leading to the polygon starts.
	Vector3 pathway_start;

	/// Point on the edge where the gateway leading to the polygon ends.
	Vector3 pathway_end;
};

struct Polygon {
	/// Navigation region or link that contains this polygon.
	const NavBase *owner = nullptr;

	/// The vertices of this `Polygon`
	LocalVector<Vector3> vertices;

	/// Are the vertices clockwise?
	bool clockwise;

	/// The connection of this `Polygon`
	LocalVector<Connection> connections;

	/// The center of this `Polygon`
	Vector3 center;
};

struct Node {
	/// The id of this node.
	uint32_t id = 0;
	/// The polygon that this `Node` is on.
	const Polygon *polygon;
	/// The the position of this `Node`.
	Vector3 position;
	/// The best cost so far to reach this `Node` from the start `Node`.
	real_t cost = 0.0;
	/// The id of the previous `Node` in the path.
	int previous_node_id = -1;
	/// The connection that was used to reach this `Node`.
	Connection last_connection;

	Node() { polygon = nullptr; }

	Node(const Polygon *p_polygon) :
			polygon(p_polygon) {}

	bool operator==(const Node &other) const {
		return this->polygon == other.polygon;
	}

	bool operator!=(const Node &other) const {
		return !operator==(other);
	}
};

struct ClosestPointQueryResult {
	Vector3 point;
	Vector3 normal;
	RID owner;
};

} // namespace gd

#endif // NAV_UTILS_H
