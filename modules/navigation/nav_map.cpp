/**************************************************************************/
/*  nav_map.cpp                                                           */
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

#include "nav_map.h"

#include "nav_agent.h"
#include "nav_link.h"
#include "nav_obstacle.h"
#include "nav_region.h"

#include "core/config/project_settings.h"
#include "core/object/worker_thread_pool.h"

#include <Obstacle2d.h>

#define THREE_POINTS_CROSS_PRODUCT(m_a, m_b, m_c) (((m_c) - (m_a)).cross((m_b) - (m_a)))

// Helper macro
#define APPEND_METADATA(polygon)                                  \
	if (r_path_types) {                                           \
		r_path_types->push_back(polygon->owner->get_type());      \
	}                                                             \
	if (r_path_rids) {                                            \
		r_path_rids->push_back(polygon->owner->get_self());       \
	}                                                             \
	if (r_path_owners) {                                          \
		r_path_owners->push_back(polygon->owner->get_owner_id()); \
	}

void NavMap::set_up(Vector3 p_up) {
	if (up == p_up) {
		return;
	}
	up = p_up;
	polygons_dirty = true;
}

void NavMap::set_cell_size(real_t p_cell_size) {
	if (cell_size == p_cell_size) {
		return;
	}
	cell_size = p_cell_size;
	polygons_dirty = true;
}

void NavMap::set_cell_height(real_t p_cell_height) {
	if (cell_height == p_cell_height) {
		return;
	}
	cell_height = p_cell_height;
	polygons_dirty = true;
}

void NavMap::set_use_edge_connections(bool p_enabled) {
	if (use_edge_connections == p_enabled) {
		return;
	}
	use_edge_connections = p_enabled;
	connections_dirty = true;
}

void NavMap::set_edge_connection_margin(real_t p_edge_connection_margin) {
	if (edge_connection_margin == p_edge_connection_margin) {
		return;
	}
	edge_connection_margin = p_edge_connection_margin;
	connections_dirty = true;
}

void NavMap::set_link_connection_radius(real_t p_link_connection_radius) {
	if (link_connection_radius == p_link_connection_radius) {
		return;
	}
	link_connection_radius = p_link_connection_radius;
	connections_dirty = true;
}

gd::PointKey NavMap::get_point_key(const Vector3 &p_pos) const {
	const int x = static_cast<int>(Math::floor(p_pos.x / cell_size));
	const int y = static_cast<int>(Math::floor(p_pos.y / cell_height));
	const int z = static_cast<int>(Math::floor(p_pos.z / cell_size));

	gd::PointKey p;
	p.key = 0;
	p.x = x;
	p.y = y;
	p.z = z;
	return p;
}

Vector<Vector3> NavMap::get_path(Vector3 p_origin, Vector3 p_destination, bool p_optimize, uint32_t p_navigation_layers, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector<Vector3>(), "NavigationServer map query failed because it was made before first map synchronization.");
	// Clear metadata outputs.
	if (r_path_types) {
		r_path_types->clear();
	}
	if (r_path_rids) {
		r_path_rids->clear();
	}
	if (r_path_owners) {
		r_path_owners->clear();
	}

	// Find the start polygon and the end polygon on this map.
	const gd::Polygon *start_polygon = nullptr;
	const gd::Polygon *end_polygon = nullptr;
	Vector3 start_point;
	Vector3 end_point;
	real_t start_min_distance = FLT_MAX;
	real_t end_min_distance = FLT_MAX;

	for (const gd::Polygon &polygon : polygons) {
		// Only consider the polygon if it in a region with compatible layers.
		if ((p_navigation_layers & polygon.owner->get_navigation_layers()) == 0) {
			continue;
		}

		// For each face check the distance between the origin/destination
		for (size_t i = 1; i < polygon.vertices.size() - 1; i++) {
			const Face3 face(polygon.vertices[0], polygon.vertices[i], polygon.vertices[i + 1]);

			Vector3 point = face.get_closest_point_to(p_origin);
			real_t distance_to_point = point.distance_to(p_origin);
			if (distance_to_point < start_min_distance) {
				start_min_distance = distance_to_point;
				start_polygon = &polygon;
				start_point = point;
			}

			point = face.get_closest_point_to(p_destination);
			distance_to_point = point.distance_to(p_destination);
			if (distance_to_point < end_min_distance) {
				end_min_distance = distance_to_point;
				end_polygon = &polygon;
				end_point = point;
			}
		}
	}

	// Check for trivial case
	if (!start_polygon || !end_polygon) {
		return Vector<Vector3>();
	}
	if (start_polygon == end_polygon) {
		if (r_path_types) {
			r_path_types->resize(2);
			r_path_types->write[0] = start_polygon->owner->get_type();
			r_path_types->write[1] = end_polygon->owner->get_type();
		}

		if (r_path_rids) {
			r_path_rids->resize(2);
			(*r_path_rids)[0] = start_polygon->owner->get_self();
			(*r_path_rids)[1] = end_polygon->owner->get_self();
		}

		if (r_path_owners) {
			r_path_owners->resize(2);
			r_path_owners->write[0] = start_polygon->owner->get_owner_id();
			r_path_owners->write[1] = end_polygon->owner->get_owner_id();
		}

		Vector<Vector3> path;
		path.resize(2);
		path.write[0] = start_point;
		path.write[1] = end_point;
		return path;
	}

	// The following is an implementation of the A* algorithm:

	// List of all nodes reached by the algorithm,
	// this includes the visited nodes and the nodes in the frontier.
	LocalVector<gd::Node> discovered_nodes;
	discovered_nodes.reserve(polygons.size() * 0.75);

	// List of ids of the nodes in the frontier,
	// these are the candidate nodes to visit next.
	List<uint32_t> frontier_node_ids;

	// Create and add the start node to the discovered and frontier lists.
	// And select it for the first iteration.
	gd::Node start_node = gd::Node(start_polygon);
	start_node.id = 0;
	start_node.position = start_point;
	start_node.last_connection.pathway_start = start_point;
	start_node.last_connection.pathway_end = start_point;

	discovered_nodes.push_back(start_node);
	frontier_node_ids.push_back(0);
	int node_id = 0;

	// The algorithm will try to find a path to the end point
	// but if this point is not reachable, it will start again searching
	// for a path to the closest polygon to the end point that's reachable.
	// The following variables are used for this purpose.
	bool is_initial_end_point_reachable = true;
	const gd::Polygon *reachable_polygon_closest_to_end_point = nullptr;
	real_t min_distance_to_end_point = FLT_MAX;

	bool found_route = false;

	while (true) {
		// Select the node to visit and remove it from the frontier list.
		const gd::Node &node = discovered_nodes[node_id];
		frontier_node_ids.erase(node_id);

		// Takes the current node neighbors (iterating over its connections) and set/update its travel cost.
		for (const gd::Connection &connection : discovered_nodes[node_id].polygon->connections) {
			// Compute the neighbor polygon, position and cost from the connection
			gd::Polygon *neighbor_polygon = connection.polygon;

			// Only consider the connection if the region of the neighbor polygon has compatible layers.
			if ((p_navigation_layers & neighbor_polygon->owner->get_navigation_layers()) == 0) {
				continue;
			}

			// Set the costs to travel to, and enter the neighbor polygon.
			real_t travel_cost = node.polygon->owner->get_travel_cost();
			real_t enter_cost = connection.is_external ? neighbor_polygon->owner->get_enter_cost() : 0.0;

			// Compute the position and cost of the neighbor node.
			Vector3 pathway[2] = { connection.pathway_start, connection.pathway_end };
			Vector3 neighbor_node_position = Geometry3D::get_closest_point_to_segment(node.position, pathway);
			real_t neighbor_node_cost = node.cost + (node.position.distance_to(neighbor_node_position) * travel_cost) + enter_cost;

			// Check if the neighbor polygon has already been discovered.
			int64_t discovered_node_index = discovered_nodes.find(gd::Node(neighbor_polygon));
			if (discovered_node_index != -1) {
				// Check if we can reduce the travel cost.
				gd::Node &discovered_node = discovered_nodes[discovered_node_index];
				if (neighbor_node_cost < discovered_node.cost) {
					discovered_node.previous_node_id = node_id;
					discovered_node.last_connection = connection;
					discovered_node.position = neighbor_node_position;
					discovered_node.cost = neighbor_node_cost;
				}
			} else {
				// Add the neighbor node to the list of discovered nodes.
				gd::Node neighbor_node = gd::Node(neighbor_polygon);
				neighbor_node.id = discovered_nodes.size();
				neighbor_node.previous_node_id = node_id;
				neighbor_node.last_connection = connection;
				neighbor_node.position = neighbor_node_position;
				neighbor_node.cost = neighbor_node_cost;
				discovered_nodes.push_back(neighbor_node);

				// Add the neighbor polygon to the frontier.
				frontier_node_ids.push_back(neighbor_node.id);
			}
		}

		// When the frontier is empty it means the end point is not reachable
		if (frontier_node_ids.size() == 0) {
			// Thus use the further reachable polygon

			// We shouldn't get to this line twice, the first time indicates the original end point is not reachable.
			// but the second time indicates the new reachable end point is not reachable either.
			ERR_BREAK_MSG(is_initial_end_point_reachable == false, "Unable to find path to a reachable end point.");

			is_initial_end_point_reachable = false;
			if (reachable_polygon_closest_to_end_point == nullptr) {
				// The path is not found and there is not a way out.
				break;
			}

			// Set as end point to the reachable polygon closest to end point.
			end_polygon = reachable_polygon_closest_to_end_point;
			end_min_distance = FLT_MAX;
			for (size_t i = 1; i < end_polygon->vertices.size() - 1; i++) {
				Face3 face(end_polygon->vertices[0], end_polygon->vertices[i], end_polygon->vertices[i + 1]);
				Vector3 closest_point = face.get_closest_point_to(p_destination);
				real_t distance = closest_point.distance_to(p_destination);
				if (distance < end_min_distance) {
					end_point = closest_point;
					end_min_distance = distance;
				}
			}

			// Search all faces of start polygon as well.
			bool are_start_and_end_polygon_the_same = false;
			for (size_t i = 1; i < start_polygon->vertices.size() - 1; i++) {
				Face3 face(start_polygon->vertices[0], start_polygon->vertices[i], start_polygon->vertices[i + 1]);
				Vector3 closest_point = face.get_closest_point_to(p_destination);
				real_t distance_to_destination = closest_point.distance_to(p_destination);
				if (distance_to_destination < end_min_distance) {
					end_point = closest_point;
					end_min_distance = distance_to_destination;
					are_start_and_end_polygon_the_same = true;
				}
			}
			if (are_start_and_end_polygon_the_same) {
				// No point to run PostProcessing when start and end polygons are the same.
				if (r_path_types) {
					r_path_types->resize(2);
					r_path_types->write[0] = start_polygon->owner->get_type();
					r_path_types->write[1] = start_polygon->owner->get_type();
				}
				if (r_path_rids) {
					r_path_rids->resize(2);
					(*r_path_rids)[0] = start_polygon->owner->get_self();
					(*r_path_rids)[1] = start_polygon->owner->get_self();
				}
				if (r_path_owners) {
					r_path_owners->resize(2);
					r_path_owners->write[0] = start_polygon->owner->get_owner_id();
					r_path_owners->write[1] = start_polygon->owner->get_owner_id();
				}
				Vector<Vector3> path;
				path.resize(2);
				path.write[0] = start_point;
				path.write[1] = end_point;
				return path;
			}

			// Reset open and navigation_polys
			gd::Node np = discovered_nodes[0];
			discovered_nodes.clear();
			discovered_nodes.push_back(np);
			frontier_node_ids.clear();
			frontier_node_ids.push_back(0);
			node_id = 0;

			reachable_polygon_closest_to_end_point = nullptr;

			continue;
		}

		// Find the polygon with the minimum cost from the list of polygons to visit.
		node_id = -1;
		real_t least_cost = FLT_MAX;
		for (List<uint32_t>::Element *element = frontier_node_ids.front(); element != nullptr; element = element->next()) {
			gd::Node *frontier_node = &discovered_nodes[element->get()];
			real_t cost = frontier_node->cost;
			cost += (frontier_node->position.distance_to(end_point) * frontier_node->polygon->owner->get_travel_cost());
			if (cost < least_cost) {
				node_id = frontier_node->id;
				least_cost = cost;
			}
		}
		ERR_BREAK(node_id == -1);

		// Stores the further reachable end polygon, in case our end point is not reachable.
		if (is_initial_end_point_reachable) {
			real_t distance_to_end_point = discovered_nodes[node_id].position.distance_to(p_destination) * discovered_nodes[node_id].polygon->owner->get_travel_cost();
			if (min_distance_to_end_point > distance_to_end_point) {
				min_distance_to_end_point = distance_to_end_point;
				reachable_polygon_closest_to_end_point = discovered_nodes[node_id].polygon;
			}
		}

		// Check if we reached the end
		if (discovered_nodes[node_id].polygon == end_polygon) {
			found_route = true;
			break;
		}
	}

	// We did not find a route but we have both a start polygon and an end polygon at this point.
	// Usually this happens because there was not a single external or internal connected edge, e.g. our start polygon is an isolated, single convex polygon.
	if (!found_route) {
		end_min_distance = FLT_MAX;
		// Search all faces of the start polygon for the closest point to our target position.
		for (size_t i = 1; i < start_polygon->vertices.size() - 1; i++) {
			Face3 face(start_polygon->vertices[0], start_polygon->vertices[i], start_polygon->vertices[i + 1]);
			Vector3 closest_point = face.get_closest_point_to(p_destination);
			real_t distance = closest_point.distance_to(p_destination);
			if (distance < end_min_distance) {
				end_point = closest_point;
				end_min_distance = distance;
			}
		}

		if (r_path_types) {
			r_path_types->resize(2);
			r_path_types->write[0] = start_polygon->owner->get_type();
			r_path_types->write[1] = start_polygon->owner->get_type();
		}

		if (r_path_rids) {
			r_path_rids->resize(2);
			(*r_path_rids)[0] = start_polygon->owner->get_self();
			(*r_path_rids)[1] = start_polygon->owner->get_self();
		}

		if (r_path_owners) {
			r_path_owners->resize(2);
			r_path_owners->write[0] = start_polygon->owner->get_owner_id();
			r_path_owners->write[1] = start_polygon->owner->get_owner_id();
		}

		Vector<Vector3> path;
		path.resize(2);
		path.write[0] = start_point;
		path.write[1] = end_point;
		return path;
	}

	Vector<Vector3> path;
	// Optimize the path.
	if (p_optimize) {
		// The following is an implementation of the Simple Stupid Funnel Algorithm (that's the actual name):

		// Set the apex polygon/point to the end point
		gd::Node *apex_node = &discovered_nodes[node_id];

		gd::Connection last_connection = apex_node->last_connection;
		Vector3 back_pathway[2] = { last_connection.pathway_start, last_connection.pathway_end };
		const Vector3 back_edge_closest_point = Geometry3D::get_closest_point_to_segment(end_point, back_pathway);
		if (end_point.is_equal_approx(back_edge_closest_point)) {
			// The end point is basically on top of the last crossed edge, funneling around the corners would at best do nothing.
			// At worst it would add an unwanted path point before the last point due to precision issues so skip to the next polygon.
			if (apex_node->previous_node_id != -1) {
				apex_node = &discovered_nodes[apex_node->previous_node_id];
			}
		}

		Vector3 apex_point = end_point;

		gd::Node *left_node = apex_node;
		Vector3 left_portal = apex_point;
		gd::Node *right_node = apex_node;
		Vector3 right_portal = apex_point;

		gd::Node *node = apex_node;

		path.push_back(end_point);
		APPEND_METADATA(end_polygon);

		while (node) {
			// Set left and right points of the pathway between polygons.
			Vector3 left = node->last_connection.pathway_start;
			Vector3 right = node->last_connection.pathway_end;
			if (THREE_POINTS_CROSS_PRODUCT(apex_point, left, right).dot(up) < 0) {
				SWAP(left, right);
			}

			bool skip = false;
			if (THREE_POINTS_CROSS_PRODUCT(apex_point, left_portal, left).dot(up) >= 0) {
				//process
				if (left_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, left, right_portal).dot(up) > 0) {
					left_node = node;
					left_portal = left;
				} else {
					clip_path(discovered_nodes, path, apex_node, right_portal, right_node, r_path_types, r_path_rids, r_path_owners);

					apex_point = right_portal;
					node = right_node;
					left_node = node;
					apex_node = node;
					left_portal = apex_point;
					right_portal = apex_point;

					path.push_back(apex_point);
					APPEND_METADATA(apex_node->polygon);
					skip = true;
				}
			}

			if (!skip && THREE_POINTS_CROSS_PRODUCT(apex_point, right_portal, right).dot(up) <= 0) {
				//process
				if (right_portal == apex_point || THREE_POINTS_CROSS_PRODUCT(apex_point, right, left_portal).dot(up) < 0) {
					right_node = node;
					right_portal = right;
				} else {
					clip_path(discovered_nodes, path, apex_node, left_portal, left_node, r_path_types, r_path_rids, r_path_owners);

					apex_point = left_portal;
					node = left_node;
					right_node = node;
					apex_node = node;
					right_portal = apex_point;
					left_portal = apex_point;

					path.push_back(apex_point);
					APPEND_METADATA(apex_node->polygon);
				}
			}

			// Go to the previous polygon.
			if (node->previous_node_id != -1) {
				node = &discovered_nodes[node->previous_node_id];
			} else {
				// The end
				node = nullptr;
			}
		}

		// If the last point is not the begin point, add it to the list.
		if (path[path.size() - 1] != start_point) {
			path.push_back(start_point);
			APPEND_METADATA(start_polygon);
		}

		path.reverse();
		if (r_path_types) {
			r_path_types->reverse();
		}
		if (r_path_rids) {
			r_path_rids->reverse();
		}
		if (r_path_owners) {
			r_path_owners->reverse();
		}

	} else {
		path.push_back(end_point);
		APPEND_METADATA(end_polygon);

		// Add mid points
		int previous_node_id = node_id;
		while (previous_node_id != -1 && discovered_nodes[previous_node_id].previous_node_id != -1) {
			if (discovered_nodes[previous_node_id].last_connection.edge != -1) {
				int prev = discovered_nodes[previous_node_id].last_connection.edge;
				int prev_n = (discovered_nodes[previous_node_id].last_connection.edge + 1) % discovered_nodes[previous_node_id].polygon->vertices.size();
				Vector3 point = (discovered_nodes[previous_node_id].polygon->vertices[prev] + discovered_nodes[previous_node_id].polygon->vertices[prev_n]) * 0.5;

				path.push_back(point);
				APPEND_METADATA(discovered_nodes[previous_node_id].polygon);
			} else {
				path.push_back(discovered_nodes[previous_node_id].position);
				APPEND_METADATA(discovered_nodes[previous_node_id].polygon);
			}

			previous_node_id = discovered_nodes[previous_node_id].previous_node_id;
		}

		path.push_back(start_point);
		APPEND_METADATA(start_polygon);

		path.reverse();
		if (r_path_types) {
			r_path_types->reverse();
		}
		if (r_path_rids) {
			r_path_rids->reverse();
		}
		if (r_path_owners) {
			r_path_owners->reverse();
		}
	}

	// Ensure post conditions (path arrays MUST match in size).
	CRASH_COND(r_path_types && path.size() != r_path_types->size());
	CRASH_COND(r_path_rids && path.size() != r_path_rids->size());
	CRASH_COND(r_path_owners && path.size() != r_path_owners->size());

	return path;
}

Vector3 NavMap::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, const bool p_use_collision) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector3(), "NavigationServer map query failed because it was made before first map synchronization.");
	bool use_collision = p_use_collision;
	Vector3 closest_point;
	real_t min_distance = FLT_MAX;

	for (const gd::Polygon &p : polygons) {
		// For each face check the distance to the segment
		for (size_t i = 1; i < p.vertices.size() - 1; i += 1) {
			const Face3 face(p.vertices[0], p.vertices[i], p.vertices[i + 1]);
			Vector3 intersection;
			if (face.intersects_segment(p_from, p_to, &intersection)) {
				const real_t distance = min_distance = p_from.distance_to(intersection);
				if (use_collision == false) {
					closest_point = intersection;
					use_collision = true;
					min_distance = distance;
				} else if (min_distance > distance) {
					closest_point = intersection;
					min_distance = distance;
				}
			}
		}

		if (use_collision == false) {
			for (size_t i = 0; i < p.vertices.size(); i += 1) {
				Vector3 a, b;

				Geometry3D::get_closest_points_between_segments(
						p_from,
						p_to,
						p.vertices[i],
						p.vertices[(i + 1) % p.vertices.size()],
						a,
						b);

				const real_t d = a.distance_to(b);
				if (d < min_distance) {
					min_distance = d;
					closest_point = b;
				}
			}
		}
	}

	return closest_point;
}

Vector3 NavMap::get_closest_point(const Vector3 &p_point) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector3(), "NavigationServer map query failed because it was made before first map synchronization.");
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.point;
}

Vector3 NavMap::get_closest_point_normal(const Vector3 &p_point) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, Vector3(), "NavigationServer map query failed because it was made before first map synchronization.");
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.normal;
}

RID NavMap::get_closest_point_owner(const Vector3 &p_point) const {
	ERR_FAIL_COND_V_MSG(map_update_id == 0, RID(), "NavigationServer map query failed because it was made before first map synchronization.");
	gd::ClosestPointQueryResult cp = get_closest_point_info(p_point);
	return cp.owner;
}

gd::ClosestPointQueryResult NavMap::get_closest_point_info(const Vector3 &p_point) const {
	gd::ClosestPointQueryResult result;
	real_t min_distance = FLT_MAX;

	for (const gd::Polygon &p : polygons) {
		// For each face check the distance to the point
		for (size_t i = 1; i < p.vertices.size() - 1; i += 1) {
			const Face3 face(p.vertices[0], p.vertices[i], p.vertices[i + 1]);
			const Vector3 closest_point = face.get_closest_point_to(p_point);
			const real_t distance = closest_point.distance_squared_to(p_point);
			if (distance < min_distance) {
				result.point = closest_point;
				result.normal = face.get_plane().normal;
				result.owner = p.owner->get_self();
				min_distance = distance;
			}
		}
	}

	return result;
}

void NavMap::add_region(NavRegion *p_region) {
	regions.push_back(p_region);
	connections_dirty = true;
}

void NavMap::remove_region(NavRegion *p_region) {
	int64_t region_index = regions.find(p_region);
	if (region_index >= 0) {
		regions.remove_at_unordered(region_index);
		connections_dirty = true;
	}
}

void NavMap::add_link(NavLink *p_link) {
	links.push_back(p_link);
	connections_dirty = true;
}

void NavMap::remove_link(NavLink *p_link) {
	int64_t link_index = links.find(p_link);
	if (link_index >= 0) {
		links.remove_at_unordered(link_index);
		connections_dirty = true;
	}
}

bool NavMap::has_agent(NavAgent *agent) const {
	return (agents.find(agent) >= 0);
}

void NavMap::add_agent(NavAgent *agent) {
	if (!has_agent(agent)) {
		agents.push_back(agent);
		agents_dirty = true;
	}
}

void NavMap::remove_agent(NavAgent *agent) {
	remove_agent_as_controlled(agent);
	int64_t agent_index = agents.find(agent);
	if (agent_index >= 0) {
		agents.remove_at_unordered(agent_index);
		agents_dirty = true;
	}
}

bool NavMap::has_obstacle(NavObstacle *obstacle) const {
	return (obstacles.find(obstacle) >= 0);
}

void NavMap::add_obstacle(NavObstacle *obstacle) {
	if (obstacle->get_paused()) {
		// No point in adding a paused obstacle, it will add itself when unpaused again.
		return;
	}

	if (!has_obstacle(obstacle)) {
		obstacles.push_back(obstacle);
		obstacles_dirty = true;
	}
}

void NavMap::remove_obstacle(NavObstacle *obstacle) {
	int64_t obstacle_index = obstacles.find(obstacle);
	if (obstacle_index >= 0) {
		obstacles.remove_at_unordered(obstacle_index);
		obstacles_dirty = true;
	}
}

void NavMap::set_agent_as_controlled(NavAgent *agent) {
	remove_agent_as_controlled(agent);

	if (agent->get_paused()) {
		// No point in adding a paused agent, it will add itself when unpaused again.
		return;
	}

	if (agent->get_use_3d_avoidance()) {
		int64_t agent_3d_index = active_3d_avoidance_agents.find(agent);
		if (agent_3d_index < 0) {
			active_3d_avoidance_agents.push_back(agent);
			agents_dirty = true;
		}
	} else {
		int64_t agent_2d_index = active_2d_avoidance_agents.find(agent);
		if (agent_2d_index < 0) {
			active_2d_avoidance_agents.push_back(agent);
			agents_dirty = true;
		}
	}
}

void NavMap::remove_agent_as_controlled(NavAgent *agent) {
	int64_t agent_3d_index = active_3d_avoidance_agents.find(agent);
	if (agent_3d_index >= 0) {
		active_3d_avoidance_agents.remove_at_unordered(agent_3d_index);
		agents_dirty = true;
	}
	int64_t agent_2d_index = active_2d_avoidance_agents.find(agent);
	if (agent_2d_index >= 0) {
		active_2d_avoidance_agents.remove_at_unordered(agent_2d_index);
		agents_dirty = true;
	}
}

void NavMap::sync() {
	// Check if we need to update the navigation graph.
	if (polygons_dirty) {
		for (NavRegion *region : regions) {
			region->scratch_polygons();
		}
		connections_dirty = true;
	}

	for (NavRegion *region : regions) {
		if (region->sync()) {
			connections_dirty = true;
		}
	}

	for (NavLink *link : links) {
		if (link->check_dirty()) {
			connections_dirty = true;
		}
	}

	if (connections_dirty) {
		_update_connections();
	}

	// Do we have modified obstacle positions?
	for (NavObstacle *obstacle : obstacles) {
		if (obstacle->check_dirty()) {
			obstacles_dirty = true;
		}
	}
	// Do we have modified agent arrays?
	for (NavAgent *agent : agents) {
		if (agent->check_dirty()) {
			agents_dirty = true;
		}
	}

	// Update avoidance worlds.
	if (obstacles_dirty || agents_dirty) {
		_update_rvo_simulation();
	}

	polygons_dirty = false;
	connections_dirty = false;
	obstacles_dirty = false;
	agents_dirty = false;

	// Performance Monitor.
	pm_region_count = regions.size();
	pm_agent_count = agents.size();
	pm_link_count = links.size();
}

void NavMap::_update_connections() {
	// Remove regions connections.
	for (NavRegion *region : regions) {
		region->get_connections().clear();
	}

	// Resize the polygon count.
	int count = 0;
	for (const NavRegion *region : regions) {
		if (region->get_enabled()) {
			count += region->get_polygons().size();
		}
	}
	polygons.resize(count);

	// Copy all region polygons in the map.
	pm_polygon_count = 0;
	for (const NavRegion *region : regions) {
		if (region->get_enabled()) {
			const LocalVector<gd::Polygon> &polygons_source = region->get_polygons();
			for (uint32_t i = 0; i < polygons_source.size(); i++) {
				const gd::Polygon &polygon = polygons_source[i];
				polygons[pm_polygon_count++] = polygon;
			}
		}
	}

	// Group all edges per key to detect polygons that share edges.
	pm_edge_count = 0;
	HashMap<gd::EdgeKey, Vector<gd::Connection>, gd::EdgeKey> edges;

	for (gd::Polygon &polygon : polygons) {
		for (uint32_t i = 0; i < polygon.vertices.size(); i++) {
			Vector3 vertex = polygon.vertices[i];
			Vector3 next_vertex = polygon.vertices[(i + 1) % polygon.vertices.size()];
			gd::EdgeKey edge_key(get_point_key(vertex), get_point_key(next_vertex));

			HashMap<gd::EdgeKey, Vector<gd::Connection>, gd::EdgeKey>::Iterator connection = edges.find(edge_key);
			if (!connection) {
				edges[edge_key] = Vector<gd::Connection>();
				pm_edge_count += 1;
			}
			if (edges[edge_key].size() <= 1) {
				// Add the polygon/edge tuple to this key.
				gd::Connection new_connection;
				new_connection.polygon = &polygon;
				new_connection.edge = i;
				new_connection.pathway_start = vertex;
				new_connection.pathway_end = next_vertex;
				edges[edge_key].push_back(new_connection);
			} else {
				// The edge is already connected with another edge, skip.
				ERR_PRINT_ONCE("Navigation map synchronization error. Attempted to merge a navigation mesh polygon edge with another already-merged edge. This is usually caused by crossing edges, overlapping polygons, or a mismatch of the NavigationMesh / NavigationPolygon baked 'cell_size' and navigation map 'cell_size'.");
			}
		}
	}

	// Connect polygons with shared edges and list remaining free connections from outside edges.
	Vector<gd::Connection> free_connections;
	pm_edge_merge_count = 0;
	for (KeyValue<gd::EdgeKey, Vector<gd::Connection>> &edge : edges) {
		if (edge.value.size() == 2) {
			// Connect edge that are shared in different polygons.
			gd::Connection &connection_1 = edge.value.write[0];
			gd::Connection &connection_2 = edge.value.write[1];
			connection_1.polygon->connections.push_back(gd::Connection(connection_2));
			connection_2.polygon->connections.push_back(gd::Connection(connection_1));
			// Note: The pathway_start/end are full for those connection and do not need to be modified.
			pm_edge_merge_count += 1;
		} else {
			CRASH_COND_MSG(edge.value.size() != 1, vformat("Number of connection != 1. Found: %d", edge.value.size()));
			if (use_edge_connections && edge.value[0].polygon->owner->get_use_edge_connections()) {
				free_connections.push_back(edge.value[0]);
			}
		}
	}

	// Find the compatible near edges to create connections across regions.
	pm_edge_free_count = free_connections.size();
	pm_edge_connection_count = 0;
	for (int i = 0; i < free_connections.size(); i++) {
		const gd::Connection &connection = free_connections[i];
		Vector3 edge_point_1 = connection.pathway_start;
		Vector3 edge_point_2 = connection.pathway_end;

		for (int j = 0; j < free_connections.size(); j++) {
			const gd::Connection &other_connection = free_connections[j];
			if (i == j || connection.polygon->owner == other_connection.polygon->owner) {
				continue;
			}

			Vector3 other_edge_point_1 = other_connection.pathway_start;
			Vector3 other_edge_point_2 = other_connection.pathway_end;

			// Compute the projection of the opposite edge on the current one
			Vector3 edge_vector = edge_point_2 - edge_point_1;
			real_t projected_point_1_ratio = edge_vector.dot(other_edge_point_1 - edge_point_1) / (edge_vector.length_squared());
			real_t projected_point_2_ratio = edge_vector.dot(other_edge_point_2 - edge_point_1) / (edge_vector.length_squared());
			if ((projected_point_1_ratio < 0.0 && projected_point_2_ratio < 0.0) || (projected_point_1_ratio > 1.0 && projected_point_2_ratio > 1.0)) {
				continue;
			}

			// Check if the two edges are close to each other enough and compute a pathway between the two regions.
			Vector3 edge_pathway_start = edge_vector * CLAMP(projected_point_1_ratio, 0.0, 1.0) + edge_point_1;
			Vector3 other_edge_pathway_start;
			if (projected_point_1_ratio >= 0.0 && projected_point_1_ratio <= 1.0) {
				other_edge_pathway_start = other_edge_point_1;
			} else {
				other_edge_pathway_start = other_edge_point_1.lerp(other_edge_point_2, (1.0 - projected_point_1_ratio) / (projected_point_2_ratio - projected_point_1_ratio));
			}
			if (other_edge_pathway_start.distance_to(edge_pathway_start) > edge_connection_margin) {
				continue;
			}

			Vector3 edge_pathway_end = edge_vector * CLAMP(projected_point_2_ratio, 0.0, 1.0) + edge_point_1;
			Vector3 other_edge_pathway_end;
			if (projected_point_2_ratio >= 0.0 && projected_point_2_ratio <= 1.0) {
				other_edge_pathway_end = other_edge_point_2;
			} else {
				other_edge_pathway_end = other_edge_point_1.lerp(other_edge_point_2, (0.0 - projected_point_1_ratio) / (projected_point_2_ratio - projected_point_1_ratio));
			}
			if (other_edge_pathway_end.distance_to(edge_pathway_end) > edge_connection_margin) {
				continue;
			}

			// The edges can be connected.
			gd::Connection external_connection = other_connection;
			external_connection.pathway_start = (edge_pathway_start + other_edge_pathway_start) / 2.0;
			external_connection.pathway_end = (edge_pathway_end + other_edge_pathway_end) / 2.0;
			external_connection.is_external = true;
			connection.polygon->connections.push_back(gd::Connection(external_connection));

			// Add the connection to the region_connection map.
			((NavRegion *)connection.polygon->owner)->get_connections().push_back(external_connection);
			pm_edge_connection_count += 1;
		}
	}

	// Create link connections.
	uint32_t link_polygon_id = 0;
	link_polygons.resize(links.size());

	for (NavLink *link : links) {
		if (!link->get_enabled()) {
			continue;
		}
		const Vector3 link_start_point = link->get_start_position();
		const Vector3 link_end_point = link->get_end_position();

		// Find the closest polygons within the search radius of the start and end points.
		gd::Polygon *link_start_polygon = nullptr;
		Vector3 link_start_vertex;
		real_t link_start_min_distance = link_connection_radius;

		gd::Polygon *link_end_polygon = nullptr;
		Vector3 link_end_vertex;
		real_t link_end_min_distance = link_connection_radius;

		for (uint32_t i = 0; i < polygons.size(); i++) {
			gd::Polygon &polygon = polygons[i];
			for (uint32_t j = 1; j < polygon.vertices.size() - 1; j += 1) {
				const Face3 face(polygon.vertices[0], polygon.vertices[j], polygon.vertices[j + 1]);

				Vector3 vertex = face.get_closest_point_to(link_start_point);
				real_t distance = vertex.distance_to(link_start_point);
				if (distance < link_start_min_distance) {
					link_start_min_distance = distance;
					link_start_vertex = vertex;
					link_start_polygon = &polygon;
				}

				vertex = face.get_closest_point_to(link_end_point);
				distance = vertex.distance_to(link_end_point);
				if (distance < link_end_min_distance) {
					link_end_min_distance = distance;
					link_end_vertex = vertex;
					link_end_polygon = &polygon;
				}
			}
		}

		// If we have both a start and end polygons, then create a synthetic polygon to route through.
		if (link_start_polygon && link_end_polygon) {
			gd::Polygon &link_polygon = link_polygons[link_polygon_id++];
			link_polygon.owner = link;

			link_polygon.vertices.clear();
			link_polygon.vertices.reserve(4);

			// Build a set of vertices that create a thin polygon going from the start to the end point.
			link_polygon.vertices.push_back(link_start_point);
			link_polygon.vertices.push_back(link_start_point);
			link_polygon.vertices.push_back(link_end_point);
			link_polygon.vertices.push_back(link_end_point);

			Vector3 center;
			for (int i = 0; i < 4; ++i) {
				center += link_polygon.vertices[i];
			}
			link_polygon.center = center / real_t(link_polygon.vertices.size());
			link_polygon.clockwise = true;

			// Setup connections to go forward in the link.
			gd::Connection entry_connection;
			entry_connection.polygon = &link_polygon;
			entry_connection.edge = -1;
			entry_connection.pathway_start = link_polygon.vertices[0];
			entry_connection.pathway_end = link_polygon.vertices[1];
			entry_connection.is_external = true;
			link_start_polygon->connections.push_back(entry_connection);

			gd::Connection exit_connection;
			exit_connection.polygon = link_end_polygon;
			exit_connection.edge = -1;
			exit_connection.pathway_start = link_polygon.vertices[2];
			exit_connection.pathway_end = link_polygon.vertices[3];
			exit_connection.is_external = true;
			link_polygon.connections.push_back(exit_connection);

			// If the link is bi-directional, create connections from the end to the start.
			if (link->is_bidirectional()) {
				entry_connection.polygon = &link_polygon;
				entry_connection.edge = -1;
				entry_connection.pathway_start = link_polygon.vertices[2];
				entry_connection.pathway_end = link_polygon.vertices[3];
				entry_connection.is_external = true;
				link_end_polygon->connections.push_back(entry_connection);

				exit_connection.polygon = link_start_polygon;
				exit_connection.edge = -1;
				exit_connection.pathway_start = link_polygon.vertices[0];
				exit_connection.pathway_end = link_polygon.vertices[1];
				exit_connection.is_external = true;
				link_polygon.connections.push_back(exit_connection);
			}
		}
	}

	// Update the update ID.
	// Some code treats 0 as a failure case, so we avoid returning 0.
	map_update_id = map_update_id % 9999999 + 1;
}

void NavMap::_update_rvo_obstacles_tree_2d() {
	int obstacle_vertex_count = 0;
	for (NavObstacle *obstacle : obstacles) {
		obstacle_vertex_count += obstacle->get_vertices().size();
	}

	// Cleaning old obstacles.
	for (size_t i = 0; i < rvo_simulation_2d.obstacles_.size(); ++i) {
		delete rvo_simulation_2d.obstacles_[i];
	}
	rvo_simulation_2d.obstacles_.clear();

	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree
	std::vector<RVO2D::Obstacle2D *> &raw_obstacles = rvo_simulation_2d.obstacles_;
	raw_obstacles.reserve(obstacle_vertex_count);

	// The following block is modified copy from RVO2D::AddObstacle()
	// Obstacles are linked and depend on all other obstacles.
	for (NavObstacle *obstacle : obstacles) {
		const Vector3 &_obstacle_position = obstacle->get_position();
		const Vector<Vector3> &_obstacle_vertices = obstacle->get_vertices();

		if (_obstacle_vertices.size() < 2) {
			continue;
		}

		std::vector<RVO2D::Vector2> rvo_2d_vertices;
		rvo_2d_vertices.reserve(_obstacle_vertices.size());

		uint32_t _obstacle_avoidance_layers = obstacle->get_avoidance_layers();
		real_t _obstacle_height = obstacle->get_height();

		for (const Vector3 &_obstacle_vertex : _obstacle_vertices) {
#ifdef TOOLS_ENABLED
			if (_obstacle_vertex.y != 0) {
				WARN_PRINT_ONCE("Y coordinates of static obstacle vertices are ignored. Please use obstacle position Y to change elevation of obstacle.");
			}
#endif
			rvo_2d_vertices.push_back(RVO2D::Vector2(_obstacle_vertex.x + _obstacle_position.x, _obstacle_vertex.z + _obstacle_position.z));
		}

		const size_t obstacleNo = raw_obstacles.size();

		for (size_t i = 0; i < rvo_2d_vertices.size(); i++) {
			RVO2D::Obstacle2D *rvo_2d_obstacle = new RVO2D::Obstacle2D();
			rvo_2d_obstacle->point_ = rvo_2d_vertices[i];
			rvo_2d_obstacle->height_ = _obstacle_height;
			rvo_2d_obstacle->elevation_ = _obstacle_position.y;

			rvo_2d_obstacle->avoidance_layers_ = _obstacle_avoidance_layers;

			if (i != 0) {
				rvo_2d_obstacle->prevObstacle_ = raw_obstacles.back();
				rvo_2d_obstacle->prevObstacle_->nextObstacle_ = rvo_2d_obstacle;
			}

			if (i == rvo_2d_vertices.size() - 1) {
				rvo_2d_obstacle->nextObstacle_ = raw_obstacles[obstacleNo];
				rvo_2d_obstacle->nextObstacle_->prevObstacle_ = rvo_2d_obstacle;
			}

			rvo_2d_obstacle->unitDir_ = normalize(rvo_2d_vertices[(i == rvo_2d_vertices.size() - 1 ? 0 : i + 1)] - rvo_2d_vertices[i]);

			if (rvo_2d_vertices.size() == 2) {
				rvo_2d_obstacle->isConvex_ = true;
			} else {
				rvo_2d_obstacle->isConvex_ = (leftOf(rvo_2d_vertices[(i == 0 ? rvo_2d_vertices.size() - 1 : i - 1)], rvo_2d_vertices[i], rvo_2d_vertices[(i == rvo_2d_vertices.size() - 1 ? 0 : i + 1)]) >= 0.0f);
			}

			rvo_2d_obstacle->id_ = raw_obstacles.size();

			raw_obstacles.push_back(rvo_2d_obstacle);
		}
	}

	rvo_simulation_2d.kdTree_->buildObstacleTree(raw_obstacles);
}

void NavMap::_update_rvo_agents_tree_2d() {
	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree.
	std::vector<RVO2D::Agent2D *> raw_agents;
	raw_agents.reserve(active_2d_avoidance_agents.size());
	for (NavAgent *agent : active_2d_avoidance_agents) {
		raw_agents.push_back(agent->get_rvo_agent_2d());
	}
	rvo_simulation_2d.kdTree_->buildAgentTree(raw_agents);
}

void NavMap::_update_rvo_agents_tree_3d() {
	// Cannot use LocalVector here as RVO library expects std::vector to build KdTree.
	std::vector<RVO3D::Agent3D *> raw_agents;
	raw_agents.reserve(active_3d_avoidance_agents.size());
	for (NavAgent *agent : active_3d_avoidance_agents) {
		raw_agents.push_back(agent->get_rvo_agent_3d());
	}
	rvo_simulation_3d.kdTree_->buildAgentTree(raw_agents);
}

void NavMap::_update_rvo_simulation() {
	if (obstacles_dirty) {
		_update_rvo_obstacles_tree_2d();
	}
	if (agents_dirty) {
		_update_rvo_agents_tree_2d();
		_update_rvo_agents_tree_3d();
	}
}

void NavMap::compute_single_avoidance_step_2d(uint32_t index, NavAgent **agent) {
	(*(agent + index))->get_rvo_agent_2d()->computeNeighbors(&rvo_simulation_2d);
	(*(agent + index))->get_rvo_agent_2d()->computeNewVelocity(&rvo_simulation_2d);
	(*(agent + index))->get_rvo_agent_2d()->update(&rvo_simulation_2d);
	(*(agent + index))->update();
}

void NavMap::compute_single_avoidance_step_3d(uint32_t index, NavAgent **agent) {
	(*(agent + index))->get_rvo_agent_3d()->computeNeighbors(&rvo_simulation_3d);
	(*(agent + index))->get_rvo_agent_3d()->computeNewVelocity(&rvo_simulation_3d);
	(*(agent + index))->get_rvo_agent_3d()->update(&rvo_simulation_3d);
	(*(agent + index))->update();
}

void NavMap::step(real_t p_deltatime) {
	deltatime = p_deltatime;

	rvo_simulation_2d.setTimeStep(float(deltatime));
	rvo_simulation_3d.setTimeStep(float(deltatime));

	if (active_2d_avoidance_agents.size() > 0) {
		if (use_threads && avoidance_use_multiple_threads) {
			WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &NavMap::compute_single_avoidance_step_2d, active_2d_avoidance_agents.ptr(), active_2d_avoidance_agents.size(), -1, true, SNAME("RVOAvoidanceAgents2D"));
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
		} else {
			for (NavAgent *agent : active_2d_avoidance_agents) {
				agent->get_rvo_agent_2d()->computeNeighbors(&rvo_simulation_2d);
				agent->get_rvo_agent_2d()->computeNewVelocity(&rvo_simulation_2d);
				agent->get_rvo_agent_2d()->update(&rvo_simulation_2d);
				agent->update();
			}
		}
	}

	if (active_3d_avoidance_agents.size() > 0) {
		if (use_threads && avoidance_use_multiple_threads) {
			WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &NavMap::compute_single_avoidance_step_3d, active_3d_avoidance_agents.ptr(), active_3d_avoidance_agents.size(), -1, true, SNAME("RVOAvoidanceAgents3D"));
			WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
		} else {
			for (NavAgent *agent : active_3d_avoidance_agents) {
				agent->get_rvo_agent_3d()->computeNeighbors(&rvo_simulation_3d);
				agent->get_rvo_agent_3d()->computeNewVelocity(&rvo_simulation_3d);
				agent->get_rvo_agent_3d()->update(&rvo_simulation_3d);
				agent->update();
			}
		}
	}
}

void NavMap::dispatch_callbacks() {
	for (NavAgent *agent : active_2d_avoidance_agents) {
		agent->dispatch_avoidance_callback();
	}

	for (NavAgent *agent : active_3d_avoidance_agents) {
		agent->dispatch_avoidance_callback();
	}
}

void NavMap::clip_path(const LocalVector<gd::Node> &p_nodes, Vector<Vector3> &path, const gd::Node *p_from_node, const Vector3 &p_to_point, const gd::Node *p_to_node, Vector<int32_t> *r_path_types, TypedArray<RID> *r_path_rids, Vector<int64_t> *r_path_owners) const {
	Vector3 from = path[path.size() - 1];

	if (from.is_equal_approx(p_to_point)) {
		return;
	}
	Plane cut_plane;
	cut_plane.normal = (from - p_to_point).cross(up);
	if (cut_plane.normal == Vector3()) {
		return;
	}
	cut_plane.normal.normalize();
	cut_plane.d = cut_plane.normal.dot(from);

	while (p_from_node != p_to_node) {
		Vector3 pathway_start = p_from_node->last_connection.pathway_start;
		Vector3 pathway_end = p_from_node->last_connection.pathway_end;

		ERR_FAIL_COND(p_from_node->previous_node_id == -1);
		p_from_node = &p_nodes[p_from_node->previous_node_id];

		if (!pathway_start.is_equal_approx(pathway_end)) {
			Vector3 inters;
			if (cut_plane.intersects_segment(pathway_start, pathway_end, &inters)) {
				if (!inters.is_equal_approx(p_to_point) && !inters.is_equal_approx(path[path.size() - 1])) {
					path.push_back(inters);
					APPEND_METADATA(p_from_node->polygon);
				}
			}
		}
	}
}

NavMap::NavMap() {
	avoidance_use_multiple_threads = GLOBAL_GET("navigation/avoidance/thread_model/avoidance_use_multiple_threads");
	avoidance_use_high_priority_threads = GLOBAL_GET("navigation/avoidance/thread_model/avoidance_use_high_priority_threads");
}

NavMap::~NavMap() {
}
