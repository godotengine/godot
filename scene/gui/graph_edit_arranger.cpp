/**************************************************************************/
/*  graph_edit_arranger.cpp                                               */
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

#include "graph_edit_arranger.h"

#include "scene/gui/graph_edit.h"

// How this arranger works:
// 1. Data is taken from the GraphEdit and formed into a useful tree-like datastructure for arranging.
// 2. The graph is split into connected chunks, and an ideal start node is found in each one.
// 3. From the start node, every node in each chunk chooses a particular node to be arranged by.
//    Nodes attempt to place themselves the most connections away from the start node as possible,
//    with the least number of 'turns' as possible (turns being a change in the direction of connection
//    being traversed)
// 4. Nodes are arranged based on the node they chose to be arranged by. To avoid overlaps, nodes are
//	  'bumped' out of each-other's way.
// 5. A final pass is taken that bumps nodes off of connections, which can seriously impact readability
//    otherwise.

constexpr Vector2 node_padding = Vector2(200, 100);

enum Direction {
	DIRECTION_LEFT,
	DIRECTION_RIGHT,
};

struct GraphNodeData;
struct GraphConnection {
	GraphNodeData *node;
	int index;

	int find_sorted_index(const Vector<GraphConnection> &p_connections) {
		int connection_count = p_connections.size();
		int dest_index = connection_count;
		for (int i = 0; i < connection_count; i++) {
			const GraphConnection &connection = p_connections[i];
			if (connection.index > index) {
				dest_index = i;
				break;
			}
		}

		return dest_index;
	}
};

static void _bump_rect(const Vector<GraphNodeData *> &p_nodes, const GraphNodeData *p_bumper, const Rect2 &p_check_rect, const Vector2 &p_offset, bool p_bump_siblings);
struct GraphNodeData {
	GraphNode *node;
	GraphNodeData *queued_by;
	// The direction this node is relative to the node it was queued by.
	Direction queued_direction;
	Rect2 rect;
	int turn_distance;
	int distance;
	bool scanned;
	bool arranged;
	bool bumping;
	Vector<GraphConnection> left_connections;
	Vector<GraphConnection> right_connections;

	int connection_count() const {
		return left_connections.size() + right_connections.size();
	}

	// Finds if the given node is connected to the callee. Useful to avoid scanning through loops forever.
	bool is_connected_recursive(GraphNodeData *p_check, Direction p_scan_direction, HashSet<GraphNodeData *> &p_avoid_loop_set) {
		if (p_avoid_loop_set.has(this)) {
			return false;
		} else if (this == p_check) {
			return true;
		}

		p_avoid_loop_set.insert(this);

		Vector<GraphConnection> to_check = p_scan_direction == DIRECTION_RIGHT ? right_connections : left_connections;
		for (const GraphConnection &connection : to_check) {
			if (connection.node->is_connected_recursive(p_check, p_scan_direction, p_avoid_loop_set)) {
				return true;
			}
		}

		return false;
	}

	GraphNodeData *get_overlapping(const Rect2 &p_check_rect, const Vector<GraphNodeData *> &p_nodes) {
		for (GraphNodeData *overlapping : p_nodes) {
			if (overlapping != this && overlapping->rect.intersects(p_check_rect)) {
				return overlapping;
			}
		}

		return nullptr;
	}

	// Returns false if testing was inconclusive.
	bool should_be_above(const GraphNodeData *p_node, bool *r_should_be_above) {
		const GraphNodeData *previous_this_ancestor = this;
		const GraphNodeData *this_ancestor = queued_by;

		while (this_ancestor != nullptr) {
			const GraphNodeData *previous_check_ancestor = p_node;
			const GraphNodeData *check_ancestor = p_node->queued_by;
			while (check_ancestor != nullptr) {
				// In this case, the node we've found must be the nearest common ancestor.
				if (this_ancestor == check_ancestor) {
					// There's no easy way to compare them if they were arranged in different directions here.
					if (previous_this_ancestor->queued_direction != previous_check_ancestor->queued_direction) {
						return false;
					}

					const Vector<GraphConnection> siblings = previous_this_ancestor->queued_direction == DIRECTION_LEFT ? this_ancestor->left_connections : this_ancestor->right_connections;
					int this_port = -1;
					int check_port = -1;
					// Connections are organized from highest to lowest, so the first one found must be
					// organized above.
					for (GraphConnection connection : siblings) {
						if (connection.node == previous_this_ancestor) {
							this_port = connection.index;
						} else if (connection.node == previous_check_ancestor) {
							check_port = connection.index;
						}
					}

					if (this_port == check_port || this_port == -1 || check_port == -1) {
						return false;
					}

					*r_should_be_above = this_port < check_port;
					return true;
				}

				previous_check_ancestor = check_ancestor;
				check_ancestor = check_ancestor->queued_by;
			}

			previous_this_ancestor = this_ancestor;
			this_ancestor = this_ancestor->queued_by;
		}

		// This should never be able to happen, as all nodes should share at least one common ancestor.
		return false;
	}

	// Moves the node to the given y position, and bumps all contacted nodes recursively.
	void bump(const Vector<GraphNodeData *> &p_nodes, const Vector2 &p_new_position, bool p_bump_siblings) {
		if (!bumping) {
			bumping = true;
		} else {
			return;
		}

		Vector2 offset = p_new_position - rect.position;
		Rect2 check_rect = rect;

		bool bump_siblings = p_bump_siblings;
		// If p_bump_siblings is true, then all siblings of this node should be bumped along with this
		// one. This fixes a lot of strange overlapping behavior.
		if (queued_by != nullptr && p_bump_siblings) {
			const Vector<GraphConnection> siblings = queued_direction == DIRECTION_LEFT ? queued_by->left_connections : queued_by->right_connections;
			for (GraphConnection connection : siblings) {
				if (connection.node != this && (connection.node->queued_by == queued_by || connection.node == queued_by->queued_by) && connection.node->bumping) {
					bump_siblings = false;
				}
			}

			if (bump_siblings) {
				for (GraphConnection connection : siblings) {
					if (connection.node == this || (connection.node->queued_by != queued_by && connection.node != queued_by->queued_by)) {
						continue;
					}

					check_rect = check_rect.merge(connection.node->rect);
					connection.node->bumping = true;
				}
			}
		}

		bool offset_x = true;
		bool offset_y = true;

		// When siblings are bumped, we need to increase the offset to account for siblings that
		// may be further within the rect we're being bumped by. The outer and inner corners below
		// refer to the corner of the current rect that siblings all take up and the corner of
		// the rect that they should take up to avoid any overlapping respectively.

		Vector2 outer_corner;
		Vector2 inner_corner;
		if (offset.x > 0.1) {
			outer_corner.x = check_rect.position.x;
			inner_corner.x = rect.position.x;
		} else if (offset.x < -0.1) {
			outer_corner.x = check_rect.position.x + check_rect.size.x;
			inner_corner.x = rect.position.x + rect.size.x;
		} else {
			offset_x = false;
		}

		if (offset.y > 0.1) {
			outer_corner.y = check_rect.position.y;
			inner_corner.y = rect.position.y;
		} else if (offset.y < -0.1) {
			outer_corner.y = check_rect.position.y + check_rect.size.y;
			inner_corner.y = rect.position.y + rect.size.y;
		} else {
			offset_y = false;
		}

		if (offset_x) {
			offset.x += inner_corner.x - outer_corner.x;
		}
		if (offset_y) {
			offset.y += inner_corner.y - outer_corner.y;
		}

		check_rect = check_rect.merge(Rect2(check_rect.position + offset, check_rect.size));
		_bump_rect(p_nodes, this, check_rect, offset, p_bump_siblings);

		rect.position += offset;

		if (queued_by != nullptr && bump_siblings) {
			const Vector<GraphConnection> siblings = queued_direction == DIRECTION_LEFT ? queued_by->left_connections : queued_by->right_connections;
			for (GraphConnection connection : siblings) {
				if (connection.node == this || (connection.node->queued_by != queued_by && connection.node != queued_by->queued_by)) {
					continue;
				}

				connection.node->bumping = false;
				connection.node->rect.position += offset;
			}
		}
		bumping = false;
	}

	void get_connected_recursive(Vector<GraphNodeData *> &p_dest) {
		if (scanned) {
			return;
		}
		scanned = true;
		p_dest.push_back(this);
		for (const GraphConnection &connection : left_connections) {
			connection.node->get_connected_recursive(p_dest);
		}
		for (const GraphConnection &connection : right_connections) {
			connection.node->get_connected_recursive(p_dest);
		}
	}

	void determine_max_depth_recursive(Direction p_direction) {
		for (int i = 0; i < 2; i++) {
			Direction search_direction = i == 0 ? DIRECTION_LEFT : DIRECTION_RIGHT;
			Vector<GraphConnection> connections = search_direction == DIRECTION_LEFT ? left_connections : right_connections;

			for (GraphConnection connection : connections) {
				if (connection.node->queued_by != this) {
					continue;
				}

				connection.node->determine_max_depth_recursive(search_direction);

				if (search_direction == p_direction && connection.node->distance >= distance) {
					distance = connection.node->distance + 1;
				}
			}
		}
	}
};

static void _bump_rect(const Vector<GraphNodeData *> &p_nodes, const GraphNodeData *p_bumper, const Rect2 &p_check_rect, const Vector2 &p_offset, bool p_bump_siblings) {
	for (GraphNodeData *node : p_nodes) {
		if (!node->arranged || node->bumping || !node->rect.intersects(p_check_rect)) {
			continue;
		}

		Rect2 overlap = node->rect.intersection(p_check_rect);
		int axis = overlap.size.x < overlap.size.y ? 0 : 1;
		Vector2 bump_amount;
		int alt_axis = (axis + 1) % 2;
		bump_amount[alt_axis] = node->rect.position[alt_axis];

		Direction bump_direction;

		if (Math::abs(p_offset[axis]) < 0.1f) {
			bool should_be_above;
			if (axis == 1 && node->should_be_above(p_bumper, &should_be_above)) {
				bump_direction = should_be_above ? DIRECTION_LEFT : DIRECTION_RIGHT;
			} else {
				int cutoff = p_check_rect.position[axis] + p_check_rect.size[axis] / 2 - node->rect.size[axis] / 2;
				bump_direction = node->rect.position[axis] > cutoff ? DIRECTION_RIGHT : DIRECTION_LEFT;
			}
		} else {
			bump_direction = p_offset[axis] > 0 ? DIRECTION_RIGHT : DIRECTION_LEFT;
		}

		if (bump_direction == DIRECTION_RIGHT) {
			bump_amount[axis] = p_check_rect.position[axis] + p_check_rect.size[axis] + 1.0f;
			node->bump(p_nodes, bump_amount, p_bump_siblings);
		} else {
			bump_amount[axis] = p_check_rect.position[axis] - node->rect.size[axis] - 1.0f;
			node->bump(p_nodes, bump_amount, p_bump_siblings);
		}
	}
}

// Finds the corresponding y coordinate of an x coordinate along a line segment. Returns false if the x
// coordinate is not on the line segment.
static bool _get_line_point_y(const Vector2 &p_start, const Vector2 &p_end, float p_x, float *r_y) {
	float x_distance = p_start.x - p_end.x;
	float x_offset = p_start.x - p_x;
	float fraction = x_offset / x_distance;
	if (fraction < 0.0 || fraction > 1.0) {
		return false;
	}
	*r_y = p_start.y * (1.0f - fraction) + p_end.y * fraction;
	return true;
}

static bool _rect_has_y(const Rect2 &p_rect, float p_y) {
	return p_y > p_rect.position.y && p_y < p_rect.position.y + p_rect.size.y;
}

static void _bump_line(const Vector<GraphNodeData *> &p_nodes, const Vector2 &p_start, const Vector2 &p_end) {
	for (GraphNodeData *node : p_nodes) {
		float y;
		bool contacted = false;
		float furthest_contact = 0;

		Rect2 node_rect = node->rect;
		node_rect.size -= Vector2(node_padding.x, 0);
		node_rect.position += Vector2(node_padding.x / 2, 0);

		if (_get_line_point_y(p_start, p_end, node_rect.position.x + node_rect.size.x, &y) && _rect_has_y(node_rect, y)) {
			float ratio = (y - node_rect.position.y) / node_rect.size.y - 0.5f;
			furthest_contact = ratio;
			contacted = true;
		}
		if (_get_line_point_y(p_start, p_end, node_rect.position.x, &y) && _rect_has_y(node_rect, y)) {
			float ratio = (y - node_rect.position.y) / node_rect.size.y - 0.5f;
			if (contacted == false || Math::abs(ratio) < Math::abs(furthest_contact)) {
				furthest_contact = ratio;
			}

			contacted = true;
		}

		// We actually don't want to bump siblings here, as it prevents many cases from being improved.
		const float padding = 50.0f;
		if (contacted && furthest_contact < 0) {
			node->bump(p_nodes, node->rect.position + Vector2(0, padding + (furthest_contact + 0.5f) * node_rect.size.y), false);
		} else if (contacted) {
			node->bump(p_nodes, node->rect.position - Vector2(0, padding + (0.5f - furthest_contact) * node_rect.size.y), false);
		}
	}
}

void GraphEditArranger::arrange_nodes() {
	ERR_FAIL_NULL(graph_edit);

	if (!arranging_graph) {
		arranging_graph = true;
	} else {
		return;
	}

	Vector<GraphNodeData> nodes;
	int child_count = graph_edit->get_child_count();

	bool arrange_all = true;
	for (int i = 0; i < child_count; i++) {
		GraphNode *graph_element = Object::cast_to<GraphNode>(graph_edit->get_child(i));
		if (graph_element && graph_element->is_selected()) {
			arrange_all = false;
		}
	}

	for (int i = 0; i < child_count; i++) {
		GraphNode *graph_element = Object::cast_to<GraphNode>(graph_edit->get_child(i));
		if (!graph_element || (!graph_element->is_selected() && !arrange_all)) {
			continue;
		}

		GraphNodeData data;
		data.node = graph_element;
		data.scanned = false;
		data.arranged = false;
		data.bumping = false;
		data.queued_by = nullptr;
		data.queued_direction = DIRECTION_LEFT;
		data.distance = 0;
		data.turn_distance = -1;
		// The position set here should never get used in the ideal case, but if any bugs occur it's best to leave
		// things where they are.
		data.rect = Rect2(data.node->get_position_offset() - node_padding / 2, data.node->get_size() + node_padding);

		nodes.push_back(data);
	}

	if (nodes.is_empty()) {
		arranging_graph = false;
		return;
	}

	GraphNodeData *writeable_nodes = nodes.ptrw();

	// Converting connections into a much more useful data format for arranging.
	const Vector<Ref<GraphEdit::Connection>> connection_list = graph_edit->get_connections();
	for (const Ref<GraphEdit::Connection> &connection : connection_list) {
		int from_port = 0;
		int from = -1;
		int to_port = 0;
		int to = -1;

		int node_count = nodes.size();
		for (int j = 0; j < node_count; j++) {
			const GraphNodeData *node = &nodes[j];
			if (from == -1 && node->node->get_name() == connection->from_node) {
				from_port = connection->from_port;
				from = j;
			} else if (to == -1 && node->node->get_name() == connection->to_node) {
				to_port = connection->to_port;
				to = j;
			}

			if (from != -1 && to != -1) {
				break;
			}
		}

		// In this case, the connection connects to nodes that are not being arranged.
		if (from == -1 || to == -1) {
			continue;
		}

		// Sorting connections as they are inserted in order to avoid crossings later.
		GraphConnection from_connection = { &writeable_nodes[to], from_port };
		writeable_nodes[from].right_connections.insert(from_connection.find_sorted_index(writeable_nodes[from].right_connections), from_connection);
		GraphConnection to_connection = { &writeable_nodes[from], to_port };
		writeable_nodes[to].left_connections.insert(to_connection.find_sorted_index(writeable_nodes[to].left_connections), to_connection);
	}

	// Keeping this set out here to avoid reallocating it all the time.
	HashSet<GraphNodeData *> avoid_loop_set;

	// Breaking up nodes into connected groups and figuring out the order in which they should be sorted.
	int node_count = nodes.size();
	for (int j = 0; j < node_count; j++) {
		GraphNodeData *arranging = writeable_nodes + j;
		if (arranging->scanned) {
			continue;
		}

		// I would keep these vectors outside the for loop and clear them every iteration to avoid
		// extra allocations, but Vector::clear() frees the vector's memory anyways.

		Vector<GraphNodeData *> queued_nodes;

		// We keep track of all possible 'queuers' so that when a turn happens and their true depths can be calculated,
		// each queued node can choose the right node to be queued by.
		struct QueuedNode {
			GraphNodeData *node;
			Vector<GraphNodeData *> queued_by;
		};

		Vector<QueuedNode> turn_queued_nodes;
		Vector<GraphNodeData *> node_group;

		// This has the side effect of marking all found nodes as scanned.
		arranging->get_connected_recursive(node_group);

		// Finding the nodes with the fewest connections, as those are almost certainly the best points
		// to start arranging from.
		GraphNodeData *start_node = nullptr;
		int least_dependencies = -1;

		for (GraphNodeData *node : node_group) {
			int dependencies = node->connection_count();

			if (least_dependencies == -1 || least_dependencies > dependencies) {
				least_dependencies = dependencies;
				start_node = node;
			}
		}

		// The direction we start scanning from doesn't affect anything.
		Direction scan_direction = DIRECTION_RIGHT;

		// If the start node moves, then the entire graph will move every time it is arranged! This accounts for that by
		// shifting everything back into place.
		Vector2 start_node_start_position = start_node->rect.position;

		start_node->turn_distance = 0;
		queued_nodes.push_back(start_node);

		// Figures out the order in which nodes should be sorted.
		while (true) {
			if (queued_nodes.is_empty()) {
				if (turn_queued_nodes.is_empty()) {
					break;
				}

				// The direction passed here doesn't really matter.
				start_node->determine_max_depth_recursive(DIRECTION_RIGHT);
				scan_direction = scan_direction == DIRECTION_RIGHT ? DIRECTION_LEFT : DIRECTION_RIGHT;

				// When a turn occurs, turn queued nodes need to determine the best (most distant)
				// node to be arranged by, which can only be found at this point by determine_max_depth_recursive,
				// hence the need to store all possible queuers.
				int turn_queue_count = turn_queued_nodes.size();
				for (int i = 0; i < turn_queue_count; i++) {
					const QueuedNode *queued = &turn_queued_nodes[i];
					bool successful = false;
					for (GraphNodeData *node : queued->queued_by) {
						if (node->distance >= queued->node->distance) {
							queued->node->distance = node->distance + 1;
							queued->node->queued_by = node;
							queued->node->queued_direction = scan_direction;
							successful = true;
						}
					}
					if (successful) {
						queued_nodes.push_back(queued->node);
					}
				}

				turn_queued_nodes.clear();
				continue;
			}

			GraphNodeData *to_arrange = queued_nodes[queued_nodes.size() - 1];
			queued_nodes.resize(queued_nodes.size() - 1);
			int distance = to_arrange->distance + 1;

			// Using a for loop here instead of a function, as I have absolutely no idea what I would
			// name it otherwise.
			for (int i = 0; i < 2; i++) {
				Direction direction = i == 0 ? DIRECTION_LEFT : DIRECTION_RIGHT;
				Vector<GraphConnection> side_connections = direction == DIRECTION_LEFT ? to_arrange->left_connections : to_arrange->right_connections;
				for (GraphConnection connection : side_connections) {
					GraphNodeData *node = connection.node;
					if (scan_direction != direction && (node->turn_distance == -1 || node->turn_distance > to_arrange->turn_distance)) {
						int turn_queue_count = turn_queued_nodes.size();
						QueuedNode *queued = nullptr;
						// If a QueuedNode already exists, we should find and add to that rather than
						// creating a new one.
						for (int k = 0; k < turn_queue_count; k++) {
							if (turn_queued_nodes[k].node == node) {
								queued = &turn_queued_nodes.ptrw()[k];
							}
						}
						if (queued == nullptr) {
							QueuedNode new_queued;
							new_queued.node = node;

							turn_queued_nodes.push_back(new_queued);
							queued = &turn_queued_nodes.ptrw()[turn_queue_count];
						}

						queued->queued_by.push_back(to_arrange);

						node->distance = distance - 1;
						node->turn_distance = to_arrange->turn_distance + 1;
						node->queued_by = to_arrange;
					} else if (scan_direction == direction && (node->turn_distance == -1 || node->turn_distance > to_arrange->turn_distance || (node->distance < distance && node->turn_distance == to_arrange->turn_distance)) && !node->is_connected_recursive(to_arrange, scan_direction, avoid_loop_set)) {
						queued_nodes.push_back(node);

						// I mistakenly placed this here, which should break things, but it turns out that circular connection arrangement
						// actually breaks if I move it to the 'correct' place outside of this if statement.
						avoid_loop_set.clear();

						// If this is not removed here, there can be some weird overwriting issues.
						int turn_queue_count = turn_queued_nodes.size();
						for (int k = 0; k < turn_queue_count; k++) {
							if (turn_queued_nodes[k].node == node) {
								turn_queued_nodes.set(k, turn_queued_nodes[turn_queue_count - 1]);
								turn_queued_nodes.remove_at(turn_queue_count - 1);
								k -= 1;
								turn_queue_count -= 1;
							}
						}
						node->distance = distance;
						node->turn_distance = to_arrange->turn_distance;
						node->queued_by = to_arrange;
						node->queued_direction = scan_direction;
					}
				}
			}
		}

		queued_nodes.push_back(start_node);
		start_node->arranged = true;

		// Arrange nodes based on the order we just determined.
		while (queued_nodes.size() != 0) {
			int controlling_node_index = -1;
			GraphNodeData *controlling_node = nullptr;

			// Finding the lowest distance node to organize next, as this tends to yield the best results.
			int queued_nodes_count = queued_nodes.size();
			for (int i = 0; i < queued_nodes_count; i++) {
				GraphNodeData *node = queued_nodes[i];
				if (controlling_node == nullptr || node->distance < controlling_node->distance) {
					controlling_node = node;
					controlling_node_index = i;
				}
			}

			controlling_node->bumping = true;

			queued_nodes.set(controlling_node_index, queued_nodes[queued_nodes.size() - 1]);
			queued_nodes.resize(queued_nodes.size() - 1);

			Vector<GraphNodeData *> node_connections;

			for (int i = 0; i < 2; i++) {
				Direction side = i == 0 ? DIRECTION_LEFT : DIRECTION_RIGHT;
				Vector<GraphConnection> side_connections = side == DIRECTION_LEFT ? controlling_node->left_connections : controlling_node->right_connections;
				int side_connections_count = side_connections.size();

				// This rect represents the total area that all the nodes on this side will take up when organized.
				Rect2 side_rect;
				for (int k = 0; k < side_connections_count; k++) {
					GraphConnection connection = side_connections[k];
					if (connection.node->bumping) {
						continue;
					}

					bool should_organize = false;
					if (connection.node->queued_by == controlling_node) {
						queued_nodes.push_back(connection.node);
						should_organize = true;
					}

					// This handles the case where a single node queues a node with lots of connections back
					// in the direction of the first node, so that the single node is organized correctly
					// with the rest of the nodes.
					if (connection.node == controlling_node->queued_by) {
						should_organize = true;
					}

					if (should_organize) {
						node_connections.push_back(connection.node);
						// These nodes really shouldn't be bumped during this process, so we're disabling them here.
						connection.node->bumping = true;
						side_rect.size = Vector2(MAX(side_rect.size.x, connection.node->rect.size.x), connection.node->rect.size.y + side_rect.size.y);
					}
				}

				// If the only connection is the node that organized this one, then it we'd be doing more harm than good by trying to
				// organize it again.
				if (node_connections.size() != 0 && (node_connections.size() != 1 || node_connections[0] != controlling_node->queued_by)) {
					if (side == DIRECTION_LEFT) {
						if (node_connections.size() == 1) {
							side_rect.position = controlling_node->rect.position - Vector2(side_rect.size.x + 1, 0);
						} else {
							side_rect.position = controlling_node->rect.position - Vector2(side_rect.size.x + 1, (side_rect.size.y - controlling_node->rect.size.y) / 2);
						}
					} else {
						if (node_connections.size() == 1) {
							side_rect.position = controlling_node->rect.position + Vector2(controlling_node->rect.size.x + 1, 0);
						} else {
							side_rect.position = controlling_node->rect.position + Vector2(controlling_node->rect.size.x + 1, -(side_rect.size.y - controlling_node->rect.size.y) / 2);
						}
					}

					// Moving any existing nodes out of the way.
					// The 'offset' is being passed based on the direction, in order to prevent existing nodes from bumping back into
					// the controlling node, which is almost certainly undesirable.
					_bump_rect(node_group, controlling_node, side_rect, Vector2(side == DIRECTION_LEFT ? -1 : 1, 0), true);

					int side_rect_offset = 0;
					int connection_count = node_connections.size();
					for (int k = 0; k < connection_count; k++) {
						GraphNodeData *node = node_connections[k];
						// This ensures that nodes can be bumped after being organized.
						node->arranged = true;
						node->rect.position = side_rect.position + Vector2(0, side_rect_offset);
						side_rect_offset += node->rect.size.y + 1;
					}
				}

				for (GraphConnection connection : side_connections) {
					connection.node->bumping = false;
				}

				node_connections.clear();
			}

			controlling_node->bumping = false;
		}

		// Performing three separate passes of line bumping, as bumping one line can cause other lines to overlap!
		for (int i = 0; i < 3; i++) {
			for (GraphNodeData *node : node_group) {
				for (GraphConnection connection : node->right_connections) {
					// Padding is added to avoid connections bumping their own nodes!
					float padding = 5.0f;

					// Currently assuming that all connections come from a point in the middle of each node,
					// as it is surprisingly difficult to calculate their actual positions, and this tends
					// to work well anyways.
					Vector2 start = node->rect.position + Vector2(node->rect.size.x - node_padding.x / 2 + padding, node->rect.size.y / 2);
					Vector2 end = connection.node->rect.position + Vector2(-padding + node_padding.x / 2, connection.node->rect.size.y / 2);

					// Looping / switchback connections should be ignored for somewhat obvious reasons.
					if (end.x < start.x) {
						continue;
					}

					_bump_line(node_group, start, end);
				}
			}
		}

		// Moving all nodes so that the start_node is in its original position, which prevents
		// the graph from sliding when arranged multiple times.
		Vector2 start_node_offset = start_node->rect.position - start_node_start_position;
		for (GraphNodeData *node : node_group) {
			node->rect.position -= start_node_offset;
		}
	}

	// Finally moving the nodes themselves to their final positions.
	graph_edit->emit_signal(SNAME("begin_node_move"));
	for (const GraphNodeData &node : nodes) {
		GraphNode *graph_node = node.node;
		graph_node->set_drag(true);
		Vector2 pos = node.rect.position + node_padding / 2;

		if (graph_edit->is_snapping_enabled()) {
			float snapping_distance = graph_edit->get_snapping_distance();
			pos = pos.snappedf(snapping_distance);
		}
		graph_node->set_position_offset(pos);
		graph_node->set_drag(false);
	}
	graph_edit->emit_signal(SNAME("end_node_move"));
	arranging_graph = false;
}
