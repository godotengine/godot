/**************************************************************************/
/*  graph_node_indexed.cpp                                                */
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

#include "graph_node_indexed.h"

#include "scene/gui/box_container.h"
#include "scene/gui/graph_connection.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/graph_port.h"
#include "scene/gui/label.h"
#include "scene/theme/theme_db.h"
/*
void GraphNodeIndexed::_notification(int p_what) {
	return;
	switch (p_what) {
		case NOTIFICATION_CHILD_ORDER_CHANGED: {
			slots.clear();
		} break;
	}
}*/

void GraphNodeIndexed::add_child_notify(Node *p_child) {
	GraphNode::add_child_notify(p_child);

	if (p_child->is_internal()) {
		return;
	}

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	int index = p_child->get_index(false);
	create_slot_and_ports(index, true);
	_slot_node_map_cache[p_child->get_name()] = index;
}

void GraphNodeIndexed::move_child_notify(Node *p_child) {
	GraphNode::move_child_notify(p_child);

	if (p_child->is_internal()) {
		return;
	}

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	StringName node_name = p_child->get_name();
	int new_index = p_child->get_index(false);
	int old_index = _slot_node_map_cache[node_name];
	_slot_node_map_cache[node_name] = new_index;

	Slot swap_buffer = slots[new_index];
	slots.set(new_index, slots[old_index]);
	slots.set(old_index, swap_buffer);
}

void GraphNodeIndexed::remove_child_notify(Node *p_child) {
	GraphNode::remove_child_notify(p_child);

	if (p_child->is_internal()) {
		return;
	}

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	int index = p_child->get_index(false);
	ERR_FAIL_INDEX(index, slots.size());

	Slot slot = slots[index];
	slots.remove_at(index);
	_slot_node_map_cache.erase(p_child->get_name());
}

void GraphNodeIndexed::create_slot(int p_slot_index, Ref<GraphPort> p_left_port, Ref<GraphPort> p_right_port, bool draw_stylebox) {
	Slot new_slot = { p_left_port, p_right_port, true };
	slots.insert(p_slot_index, new_slot);
}

void GraphNodeIndexed::create_slot_and_ports(int p_slot_index, bool draw_stylebox) {
	return create_slot(p_slot_index, memnew(GraphPort(this)), memnew(GraphPort(this)), draw_stylebox);
}

TypedArray<Array> GraphNodeIndexed::get_slots() {
	TypedArray<Array> ret;
	for (GraphNodeIndexed::Slot slot : slots) {
		Array s;
		s.push_back(slot.left_port);
		s.push_back(slot.right_port);
		s.push_back(slot.draw_stylebox);
		ret.push_back(s);
	}
	return ret;
}

Vector<GraphNodeIndexed::Slot> GraphNodeIndexed::_get_slots() {
	return slots;
}

GraphNodeIndexed::Slot GraphNodeIndexed::_get_slot(int p_slot_index) {
	return slots[p_slot_index];
}

void GraphNodeIndexed::_set_slot(int p_slot_index, const Ref<GraphPort> p_left_port, const Ref<GraphPort> p_right_port, bool draw_stylebox) {
	slots.set(p_slot_index, Slot({ p_left_port, p_right_port, draw_stylebox }));
}

void GraphNodeIndexed::set_slot(int p_slot_index, const Ref<GraphPort> p_left_port, const Ref<GraphPort> p_right_port, bool draw_stylebox) {
	_set_slot(p_slot_index, p_left_port, p_right_port, draw_stylebox);
}

int GraphNodeIndexed::slot_index_of_port(const Ref<GraphPort> p_port) {
	return port_to_slot_index(index_of_port(p_port));
}

void GraphNodeIndexed::set_input(int p_slot_index, bool p_enabled, bool p_exclusive, int p_type, Color p_color, Ref<Texture2D> p_icon) {
	Ref<GraphPort> port = get_input_port(p_slot_index);
	ERR_FAIL_COND(port.is_null());
	port->set(p_enabled, p_exclusive, p_type, p_color, GraphPort::PortDirection::INPUT, p_icon);
}

void GraphNodeIndexed::set_output(int p_slot_index, bool p_enabled, bool p_exclusive, int p_type, Color p_color, Ref<Texture2D> p_icon) {
	Ref<GraphPort> port = get_output_port(p_slot_index);
	ERR_FAIL_COND(port.is_null());
	port->set(p_enabled, p_exclusive, p_type, p_color, GraphPort::PortDirection::OUTPUT, p_icon);
}

Ref<GraphPort> GraphNodeIndexed::get_input_port(int p_slot_index) {
	int port_index = slot_to_port_index(p_slot_index, true);
	return get_port(port_index);
}

Ref<GraphPort> GraphNodeIndexed::get_output_port(int p_slot_index) {
	int port_index = slot_to_port_index(p_slot_index, false);
	return get_port(port_index);
}

void GraphNodeIndexed::set_input_port(int p_slot_index, const Ref<GraphPort> p_port) {
	Slot old_slot = slots[p_slot_index];
	slots.set(p_slot_index, Slot(p_port, old_slot.right_port, old_slot.draw_stylebox));
}

void GraphNodeIndexed::set_output_port(int p_slot_index, const Ref<GraphPort> p_port) {
	Slot old_slot = slots[p_slot_index];
	slots.set(p_slot_index, Slot(old_slot.left_port, p_port, old_slot.draw_stylebox));
}

int GraphNodeIndexed::port_to_slot_index(int p_port_index) {
	int idx = 0;
	int slot_idx = 0;
	for (Slot slot : slots) {
		if (slot.left_port.is_valid() && slot.left_port->get_enabled()) {
			if (idx == p_port_index) {
				return slot_idx;
			}
			idx++;
		}
		if (slot.right_port.is_valid() && slot.right_port->get_enabled()) {
			if (idx == p_port_index) {
				return slot_idx;
			}
			idx++;
		}
		slot_idx++;
	}
	return -1;
}

int GraphNodeIndexed::split_port_to_slot_index(int p_port_index, bool p_input) {
	int idx = 0;
	int slot_idx = 0;
	for (Slot slot : slots) {
		if (p_input && slot.left_port.is_valid() && slot.left_port->get_enabled()) {
			if (idx == p_port_index) {
				return slot_idx;
			}
			idx++;
		}
		if (!p_input && slot.right_port.is_valid() && slot.right_port->get_enabled()) {
			if (idx == p_port_index) {
				return slot_idx;
			}
			idx++;
		}
		slot_idx++;
	}
	return -1;
}

int GraphNodeIndexed::slot_to_port_index(int p_slot_index, bool p_input) {
	int idx = 0;
	int slot_idx = 0;
	for (Slot slot : slots) {
		if (slot_idx == p_slot_index) {
			if (p_input) {
				ERR_FAIL_COND_V(slot.left_port.is_null(), -1);
			} else {
				ERR_FAIL_COND_V(slot.right_port.is_null(), -1);
				if (slot.left_port.is_valid() && slot.left_port->get_enabled()) {
					idx++;
				}
			}
			return idx;
		}
		if (slot.left_port.is_valid() && slot.left_port->get_enabled()) {
			idx++;
		}
		if (slot.right_port.is_valid() && slot.right_port->get_enabled()) {
			idx++;
		}
		slot_idx++;
	}
	return -1;
}

int GraphNodeIndexed::slot_to_split_port_index(int p_slot_index, bool p_input) {
	int idx = 0;
	int slot_idx = 0;
	for (Slot slot : slots) {
		if (slot_idx == p_slot_index) {
			if (p_input) {
				ERR_FAIL_COND_V(slot.left_port.is_null(), -1);
			} else {
				ERR_FAIL_COND_V(slot.right_port.is_null(), -1);
			}
			return idx;
		}
		if (p_input && slot.left_port.is_valid() && slot.left_port->get_enabled()) {
			idx++;
		}
		if (!p_input && slot.right_port.is_valid() && slot.right_port->get_enabled()) {
			idx++;
		}
		slot_idx++;
	}
	return -1;
}

bool GraphNodeIndexed::get_slot_draw_stylebox(int p_slot_index) {
	ERR_FAIL_INDEX_V(p_slot_index, slots.size(), false);
	return slots[p_slot_index].draw_stylebox;
}

void GraphNodeIndexed::set_slot_draw_stylebox(int p_slot_index, bool p_draw_stylebox) {
	ERR_FAIL_INDEX(p_slot_index, slots.size());
	Slot old_slot = slots[p_slot_index];
	slots.set(p_slot_index, Slot(old_slot.left_port, old_slot.right_port, p_draw_stylebox));
}

GraphNodeIndexed::GraphNodeIndexed() {
	GraphNode::GraphNode();
}

GraphNodeIndexed::Slot::Slot() {
	left_port = Ref<GraphPort>(nullptr);
	right_port = Ref<GraphPort>(nullptr);
}
