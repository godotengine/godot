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
#include <scene/resources/style_box_flat.h>

void GraphNodeIndexed::add_child_notify(Node *p_child) {
	GraphNode::add_child_notify(p_child);

	if (p_child->is_internal() || !Object::cast_to<Control>(p_child) || p_child == port_container) {
		return;
	}

	StringName child_name = p_child->get_name();
	int slot_index = slot_index_of_node(p_child);

	// Child already exists in slot node cache - ignore and move on.
	if (_slot_node_map_cache.has(child_name)) {
		return;
	}

	if (!is_ready() && slots.size() > slot_index) {
		// Not ready yet! This should only happen when the node is instantiated along with children.
		// Properties are assigned before children are created, and a slot already exists for this child, so don't create a new one.
		// This keeps ports from being overridden/recreated on scene instantiation
		_slot_node_map_cache[child_name] = slot_index;
		return;
	}

	create_slot_and_ports(slot_index, true, child_name);
}

void GraphNodeIndexed::move_child_notify(Node *p_child) {
	GraphNode::move_child_notify(p_child);

	if (p_child->is_internal() || !is_ready() || !Object::cast_to<Control>(p_child) || p_child == port_container) {
		return;
	}

	StringName child_name = p_child->get_name();
	int new_index = slot_index_of_node(p_child);
	int old_index = _slot_node_map_cache[child_name];

	move_slot_with_ports(old_index, new_index);
}

void GraphNodeIndexed::remove_child_notify(Node *p_child) {
	GraphNode::remove_child_notify(p_child);

	if (p_child->is_internal() || !is_ready() || p_child == port_container || !Object::cast_to<Control>(p_child) || !_slot_node_map_cache.has(p_child->get_name()) || port_container->get_parent() != this) {
		return;
	}

	int index = _slot_node_map_cache[p_child->get_name()];
	ERR_FAIL_INDEX(index, slots.size());

	remove_slot_and_ports(index);
}

void GraphNodeIndexed::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			// Used for layout calculations.
			Ref<StyleBox> sb_panel = theme_cache.panel;

			Ref<StyleBox> sb_slot = theme_cache.slot;
			Ref<StyleBox> sb_slot_selected = theme_cache.slot_selected;

			//int port_h_offset = theme_cache.port_h_offset;

			int width = get_size().width - sb_panel->get_minimum_size().x;

			for (int i = 0; i < get_child_count(false); i++) {
				Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::VISIBLE_IN_TREE);
				if (!child || !child->is_visible_in_tree() || child == port_container || !_slot_node_map_cache.has(child->get_name())) {
					continue;
				}
				int slot_index = _slot_node_map_cache[child->get_name()];
				const Slot slot = slots[slot_index];

				// TODO: keyboard navigation override for slot selection
				/* if (slot_index == selected_slot) {
					Size2i port_sz = theme_cache.port->get_size();
					draw_style_box(sb_slot_selected, Rect2i(port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
					draw_style_box(sb_slot_selected, Rect2i(get_size().x - port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
				}*/

				if (slot.draw_stylebox) {
					Control *child = Object::cast_to<Control>(get_child(slot_index, false));
					if (!child || !child->is_visible_in_tree()) {
						continue;
					}
					Rect2 child_rect = child->get_rect();
					child_rect.position.x = sb_panel->get_margin(SIDE_LEFT);
					child_rect.size.width = width;
					draw_style_box(sb_slot, child_rect);
				}
			}
		} break;
	}
}

void GraphNodeIndexed::create_slot_and_ports(int p_slot_index, bool p_draw_stylebox, StringName p_slot_node_name) {
	GraphPort *p_left = memnew(GraphPort(false, true, 0, GraphPort::PortDirection::INPUT));
	GraphPort *p_right = memnew(GraphPort(false, false, 0, GraphPort::PortDirection::OUTPUT));
	p_left->set_name(vformat("InputPort%s", String(p_slot_node_name)));
	p_right->set_name(vformat("OutputPort%s", String(p_slot_node_name)));

	p_right->add_theme_constant_override("hotzone_offset_h", -p_right->get_theme_constant("hotzone_offset_h"));

	_insert_slot(p_slot_index, Slot(p_draw_stylebox));
	insert_port(slot_to_port_index(p_slot_index, true), p_left);
	insert_port(slot_to_port_index(p_slot_index, false), p_right);

	emit_signal("slot_added", p_slot_index);
}

void GraphNodeIndexed::remove_slot_and_ports(int p_slot_index) {
	Slot s = slots[p_slot_index];
	_remove_slot(p_slot_index);
	int input_port_idx = slot_to_port_index(p_slot_index, true);
	int output_port_idx = slot_to_port_index(p_slot_index, false);
	if (output_port_idx < ports.size()) {
		GraphPort *in_port = ports[input_port_idx];
		GraphPort *out_port = ports[output_port_idx];
		if (in_port && in_port->get_parent() == port_container) {
			port_container->remove_child(in_port);
			in_port->queue_free();
		}
		if (out_port && out_port->get_parent() == port_container) {
			port_container->remove_child(out_port);
			out_port->queue_free();
		}
		remove_port(output_port_idx);
		remove_port(input_port_idx);
	}
	for (KeyValue<StringName, int> &kv_pair : _slot_node_map_cache) {
		if (kv_pair.value > p_slot_index) {
			kv_pair.value--;
		}
	}

	emit_signal("slot_removed", p_slot_index);
}

void GraphNodeIndexed::move_slot_with_ports(int p_old_slot_index, int p_new_slot_index) {
	Slot swap_buffer = slots[p_old_slot_index];
	GraphPort *input_port_swap_buffer = get_input_port_by_slot(p_old_slot_index);
	GraphPort *output_port_swap_buffer = get_output_port_by_slot(p_old_slot_index);
	if (p_new_slot_index < p_old_slot_index) {
		for (int i = p_old_slot_index; i > p_new_slot_index; i--) {
			copy_slot_with_ports(i - 1, i);
		}
	} else {
		for (int i = p_old_slot_index; i < p_new_slot_index; i++) {
			copy_slot_with_ports(i + 1, i);
		}
	}
	set_slot_with_ports(p_new_slot_index, swap_buffer, input_port_swap_buffer, output_port_swap_buffer);

	emit_signal("slot_moved", p_old_slot_index, p_new_slot_index);
}

void GraphNodeIndexed::set_slot_with_ports(int p_slot_index, Slot p_slot, GraphPort *p_input_port, GraphPort *p_output_port) {
	_set_slot(p_slot_index, p_slot);
	set_port(slot_to_port_index(p_slot_index, true), p_input_port);
	set_port(slot_to_port_index(p_slot_index, false), p_output_port);
}

void GraphNodeIndexed::copy_slot_with_ports(int p_old_slot_index, int p_new_slot_index) {
	set_slot_with_ports(p_new_slot_index, slots[p_old_slot_index], get_input_port_by_slot(p_old_slot_index), get_output_port_by_slot(p_old_slot_index));
}

void GraphNodeIndexed::_set_slots(const Vector<Slot> &p_slots) {
	_remove_all_slots();
	int i = 0;
	for (const Slot &slot : p_slots) {
		_insert_slot(i, slot);
		i++;
	}
}

void GraphNodeIndexed::set_slots(const TypedArray<Array> &p_slots) {
	_remove_all_slots();
	int i = 0;
	for (Array p_slot : p_slots) {
		bool draw_sb = p_slot[0];
		GraphNodeIndexed::Slot slot = Slot(draw_sb);
		_insert_slot(i, slot);
		i++;
	}
}

const Vector<GraphNodeIndexed::Slot> &GraphNodeIndexed::_get_slots() {
	return slots;
}

TypedArray<Array> GraphNodeIndexed::get_slots() {
	TypedArray<Array> ret;
	for (GraphNodeIndexed::Slot slot : slots) {
		Array s;
		s.push_back(slot.draw_stylebox);
		ret.push_back(s);
	}
	return ret;
}

GraphNodeIndexed::Slot GraphNodeIndexed::_get_slot(int p_slot_index) {
	return slots[p_slot_index];
}

Array GraphNodeIndexed::get_slot(int p_slot_index) {
	Slot slot = _get_slot(p_slot_index);
	return Array({ slot.draw_stylebox });
}

void GraphNodeIndexed::_remove_all_slots() {
	if (slots.is_empty()) {
		return;
	}
	for (int i = slots.size() - 1; i >= 0; i--) {
		_remove_slot(i);
	}
}

void GraphNodeIndexed::_remove_slot(int p_slot_index) {
	slots.remove_at(p_slot_index);

	for (KeyValue<StringName, int> &kv_pair : _slot_node_map_cache) {
		if (kv_pair.value == p_slot_index) {
			_slot_node_map_cache.erase(kv_pair.key);
		}
	}
}

void GraphNodeIndexed::_set_slot_node_cache(const TypedDictionary<StringName, int> &p_slot_node_map_cache) {
	_slot_node_map_cache.clear();
	for (const KeyValue<Variant, Variant> &kv : p_slot_node_map_cache) {
		_slot_node_map_cache[kv.key] = kv.value;
	}
}

TypedDictionary<StringName, int> GraphNodeIndexed::_get_slot_node_cache() {
	TypedDictionary<StringName, int> ret;
	for (const KeyValue<StringName, int> &kv : _slot_node_map_cache) {
		ret[kv.key] = kv.value;
	}
	return ret;
}

void GraphNodeIndexed::_insert_slot(int p_slot_index, const Slot p_slot) {
	slots.insert(p_slot_index, p_slot);

	if (p_slot_index < get_child_count(false)) {
		for (KeyValue<StringName, int> &kv_pair : _slot_node_map_cache) {
			if (p_slot_index <= kv_pair.value) {
				kv_pair.value++;
			}
		}
		int child_index = slot_to_child_index(p_slot_index);
		if (child_index < get_child_count(false)) {
			Node *slot_node = get_child(child_index, false);
			if (slot_node) {
				_slot_node_map_cache[slot_node->get_name()] = p_slot_index;
			}
		}
	}
}

void GraphNodeIndexed::_set_slot(int p_slot_index, const Slot p_slot) {
	slots.set(p_slot_index, p_slot);

	if (p_slot_index < get_child_count(false)) {
		_slot_node_map_cache[get_child(slot_to_child_index(p_slot_index), false)->get_name()] = p_slot_index;
	}

	notify_property_list_changed();
}

void GraphNodeIndexed::_resort() {
	GraphNode::_resort();
	emit_signal(SNAME("slot_sizes_changed"), this);
}

void GraphNodeIndexed::_update_port_positions() {
	// Grab theme references for layout
	Ref<StyleBoxFlat> sb_panel = theme_cache.panel;
	Ref<StyleBox> sb_titlebar = theme_cache.titlebar;
	int separation = theme_cache.separation;

	// This helps to immediately achieve the initial y "original point" of the slots, which the sum of the titlebar height and the top margin of the panel.
	int vertical_ofs = titlebar_hbox->get_size().height + sb_titlebar->get_minimum_size().height + sb_panel->get_margin(SIDE_TOP);

	// Node x and port x positions are uniform for all ports, so find them now
	int node_width = get_size().width;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::VISIBLE_IN_TREE);
		if (!child || child == port_container) {
			continue;
		}

		Size2i size = child->get_size();
		int port_y = vertical_ofs + size.height * 0.5;

		if (_slot_node_map_cache.has(child->get_name())) {
			int slot_index = _slot_node_map_cache[child->get_name()];
			const Slot &slot = slots[slot_index];
			GraphPort *left_port = get_input_port_by_slot(slot_index);
			GraphPort *right_port = get_output_port_by_slot(slot_index);

			if (left_port) {
				left_port->set_position(Point2(-theme_cache.port_h_offset, port_y));
			}
			if (right_port) {
				right_port->set_position(Point2(node_width + theme_cache.port_h_offset, port_y));
			}
		}

		vertical_ofs += size.height + separation;
	}

	GraphNode::_update_port_positions();
}

int GraphNodeIndexed::slot_index_of_node(Node *p_node) {
	int node_idx = p_node->get_index(false);
	return child_to_slot_index(node_idx);
}

// helpers to account for port container when indexing children for slots
int GraphNodeIndexed::child_to_slot_index(int idx) {
	return port_container_idx < idx ? idx - 1 : idx;
}
int GraphNodeIndexed::slot_to_child_index(int idx) {
	return port_container_idx <= idx ? idx + 1 : idx;
}

int GraphNodeIndexed::slot_index_of_port(GraphPort *p_port) {
	return floor(index_of_port(p_port) / 2);
}

int GraphNodeIndexed::index_of_input_port(GraphPort *p_port, bool p_include_disabled) {
	return filtered_index_of_port(p_port, p_include_disabled);
}

int GraphNodeIndexed::index_of_output_port(GraphPort *p_port, bool p_include_disabled) {
	return filtered_index_of_port(p_port, p_include_disabled);
}

void GraphNodeIndexed::set_slot_properties(int p_slot_index, bool p_input_enabled, int p_input_type, bool p_output_enabled, int p_output_type) {
	set_input_port_properties(p_slot_index, p_input_enabled, p_input_type);
	set_output_port_properties(p_slot_index, p_output_enabled, p_output_type);
}

void GraphNodeIndexed::set_input_port_properties(int p_slot_index, bool p_enabled, int p_type) {
	GraphPort *port = get_input_port_by_slot(p_slot_index);
	ERR_FAIL_NULL(port);
	port->set_properties(p_enabled, true, p_type, GraphPort::PortDirection::INPUT);
}

void GraphNodeIndexed::set_output_port_properties(int p_slot_index, bool p_enabled, int p_type) {
	GraphPort *port = get_output_port_by_slot(p_slot_index);
	ERR_FAIL_NULL(port);
	port->set_properties(p_enabled, false, p_type, GraphPort::PortDirection::OUTPUT);
}

GraphPort *GraphNodeIndexed::get_input_port_by_slot(int p_slot_index) {
	int port_index = slot_to_port_index(p_slot_index, true);
	return get_port(port_index);
}

GraphPort *GraphNodeIndexed::get_output_port_by_slot(int p_slot_index) {
	int port_index = slot_to_port_index(p_slot_index, false);
	return get_port(port_index);
}

TypedArray<GraphPort> GraphNodeIndexed::get_input_ports(bool p_include_disabled) {
	return get_filtered_ports(GraphPort::PortDirection::INPUT, p_include_disabled);
}

TypedArray<GraphPort> GraphNodeIndexed::get_output_ports(bool p_include_disabled) {
	return get_filtered_ports(GraphPort::PortDirection::OUTPUT, p_include_disabled);
}

int GraphNodeIndexed::get_input_port_count() {
	return get_filtered_port_count(GraphPort::PortDirection::INPUT);
}

int GraphNodeIndexed::get_output_port_count() {
	return get_filtered_port_count(GraphPort::PortDirection::OUTPUT);
}

int GraphNodeIndexed::port_to_slot_index(int p_port_index, bool p_include_disabled) {
	if (p_include_disabled) {
		return floor(p_port_index / 2);
	}
	int idx = 0;
	int slot_idx = 0;
	for (const Slot &slot : slots) {
		int input_port_idx = slot_idx * 2;
		int output_port_idx = input_port_idx + 1;
		if (input_port_idx >= ports.size()) {
			break;
		}
		if (ports[input_port_idx] && ports[input_port_idx]->is_enabled()) {
			if (idx == p_port_index) {
				return slot_idx;
			}
			idx++;
		}
		if (ports[output_port_idx] && ports[output_port_idx]->is_enabled()) {
			if (idx == p_port_index) {
				return slot_idx;
			}
			idx++;
		}
		slot_idx++;
	}
	return -1;
}

int GraphNodeIndexed::slot_to_port_index(int p_slot_index, bool p_input, bool p_include_disabled) {
	int idx = p_slot_index * 2 + (p_input ? 0 : 1);
	if (p_include_disabled) {
		return idx;
	} else {
		GraphPort *port = Object::cast_to<GraphPort>(ports[idx]);
		return port->get_port_index(false);
	}
}

int GraphNodeIndexed::slot_to_input_port_index(int p_slot_index, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_slot_index * 2, ports.size(), -1);
	GraphPort *port = Object::cast_to<GraphPort>(ports[p_slot_index * 2]);
	ERR_FAIL_NULL_V(port, -1);
	return port->get_filtered_port_index(p_include_disabled);
}

int GraphNodeIndexed::slot_to_output_port_index(int p_slot_index, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_slot_index * 2 + 1, ports.size(), -1);
	GraphPort *port = Object::cast_to<GraphPort>(ports[p_slot_index * 2 + 1]);
	ERR_FAIL_NULL_V(port, -1);
	return port->get_filtered_port_index(p_include_disabled);
}

int GraphNodeIndexed::input_port_to_slot_index(int p_port_index, bool p_include_disabled) {
	GraphPort *port = get_filtered_port(p_port_index, GraphPort::PortDirection::INPUT, p_include_disabled);
	return slot_index_of_port(port);
}

int GraphNodeIndexed::output_port_to_slot_index(int p_port_index, bool p_include_disabled) {
	GraphPort *port = get_filtered_port(p_port_index, GraphPort::PortDirection::OUTPUT, p_include_disabled);
	return slot_index_of_port(port);
}

TypedArray<Ref<GraphConnection>> GraphNodeIndexed::get_input_connections() {
	return get_filtered_connections(GraphPort::INPUT);
}

TypedArray<Ref<GraphConnection>> GraphNodeIndexed::get_output_connections() {
	return get_filtered_connections(GraphPort::OUTPUT);
}

bool GraphNodeIndexed::get_slot_draw_stylebox(int p_slot_index) {
	ERR_FAIL_INDEX_V(p_slot_index, slots.size(), false);
	return slots[p_slot_index].draw_stylebox;
}

void GraphNodeIndexed::set_slot_draw_stylebox(int p_slot_index, bool p_draw_stylebox) {
	ERR_FAIL_INDEX(p_slot_index, slots.size());
	slots.set(p_slot_index, Slot(p_draw_stylebox));
	notify_property_list_changed();
}

void GraphNodeIndexed::set_slot_focus_mode(Control::FocusMode p_focus_mode) {
	if (slot_focus_mode == p_focus_mode) {
		return;
	}
	ERR_FAIL_COND((int)p_focus_mode < 1 || (int)p_focus_mode > 3);

	slot_focus_mode = p_focus_mode;
	if (slot_focus_mode == Control::FOCUS_CLICK && selected_slot > -1) {
		selected_slot = -1;
		queue_redraw();
	}
}

Control::FocusMode GraphNodeIndexed::get_slot_focus_mode() const {
	return slot_focus_mode;
}

Size2 GraphNodeIndexed::get_minimum_size() const {
	Ref<StyleBox> sb_panel = theme_cache.panel;
	Ref<StyleBox> sb_titlebar = theme_cache.titlebar;
	Ref<StyleBox> sb_slot = theme_cache.slot;

	int separation = theme_cache.separation;
	Size2 minsize = titlebar_hbox->get_minimum_size() + sb_titlebar->get_minimum_size();

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false));
		if (!child) {
			continue;
		}

		Size2i size = child->get_combined_minimum_size();
		size.width += sb_panel->get_minimum_size().width;

		if (i < slots.size()) {
			size += slots[i].draw_stylebox ? sb_slot->get_minimum_size() : Size2();
		}

		minsize.height += size.height;
		minsize.width = MAX(minsize.width, size.width);

		if (i > 0) {
			minsize.height += separation;
		}
	}

	minsize.height += sb_panel->get_minimum_size().height;

	return minsize;
}

void GraphNodeIndexed::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_slots", "slots"), &GraphNodeIndexed::set_slots);
	ClassDB::bind_method(D_METHOD("get_slots"), &GraphNodeIndexed::get_slots);

	ClassDB::bind_method(D_METHOD("_set_slot_node_cache", "_slot_node_map_cache"), &GraphNodeIndexed::_set_slot_node_cache);
	ClassDB::bind_method(D_METHOD("_get_slot_node_cache"), &GraphNodeIndexed::_get_slot_node_cache);

	ClassDB::bind_method(D_METHOD("set_slot_properties", "slot_index", "input_enabled", "input_type", "output_enabled", "output_type"), &GraphNodeIndexed::set_slot_properties);
	ClassDB::bind_method(D_METHOD("set_input_port_properties", "slot_index", "enabled", "type"), &GraphNodeIndexed::set_input_port_properties);
	ClassDB::bind_method(D_METHOD("set_output_port_properties", "slot_index", "enabled", "type"), &GraphNodeIndexed::set_output_port_properties);

	ClassDB::bind_method(D_METHOD("get_input_port_by_slot", "slot_index"), &GraphNodeIndexed::get_input_port_by_slot);
	ClassDB::bind_method(D_METHOD("get_output_port_by_slot", "slot_index"), &GraphNodeIndexed::get_output_port_by_slot);

	ClassDB::bind_method(D_METHOD("get_input_ports", "include_disabled"), &GraphNodeIndexed::get_input_ports, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_output_ports", "include_disabled"), &GraphNodeIndexed::get_output_ports, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("get_input_port_count"), &GraphNodeIndexed::get_input_port_count);
	ClassDB::bind_method(D_METHOD("get_output_port_count"), &GraphNodeIndexed::get_output_port_count);

	ClassDB::bind_method(D_METHOD("slot_index_of_node", "node"), &GraphNodeIndexed::slot_index_of_node);
	ClassDB::bind_method(D_METHOD("slot_index_of_port", "port"), &GraphNodeIndexed::slot_index_of_port);
	ClassDB::bind_method(D_METHOD("index_of_input_port", "port", "include_disabled"), &GraphNodeIndexed::index_of_input_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("index_of_output_port", "port", "include_disabled"), &GraphNodeIndexed::index_of_output_port, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::port_to_slot_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("slot_to_port_index", "slot_index", "is_input_port", "include_disabled"), &GraphNodeIndexed::slot_to_port_index, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("child_to_slot_index", "child_index"), &GraphNodeIndexed::child_to_slot_index);
	ClassDB::bind_method(D_METHOD("slot_to_child_index", "slot_index"), &GraphNodeIndexed::slot_to_child_index);

	ClassDB::bind_method(D_METHOD("slot_to_input_port_index", "slot_index", "include_disabled"), &GraphNodeIndexed::slot_to_input_port_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("slot_to_output_port_index", "slot_index", "include_disabled"), &GraphNodeIndexed::slot_to_output_port_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("input_port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::input_port_to_slot_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("output_port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::output_port_to_slot_index, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("set_slot_focus_mode", "focus_mode"), &GraphNodeIndexed::set_slot_focus_mode);
	ClassDB::bind_method(D_METHOD("get_slot_focus_mode"), &GraphNodeIndexed::get_slot_focus_mode);

	ClassDB::bind_method(D_METHOD("get_input_connections"), &GraphNodeIndexed::get_input_connections);
	ClassDB::bind_method(D_METHOD("get_output_connections"), &GraphNodeIndexed::get_output_connections);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "slots", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_slots", "get_slots");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_slot_node_map_cache", PROPERTY_HINT_DICTIONARY_TYPE, "StringName:int", PROPERTY_USAGE_STORAGE), "_set_slot_node_cache", "_get_slot_node_cache");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "slot_focus_mode", PROPERTY_HINT_ENUM, "Click:1,All:2,Accessibility:3"), "set_slot_focus_mode", "get_slot_focus_mode");

	ADD_SIGNAL(MethodInfo("slot_added", PropertyInfo(Variant::INT, "slot_index")));
	ADD_SIGNAL(MethodInfo("slot_removed", PropertyInfo(Variant::INT, "slot_index")));
	ADD_SIGNAL(MethodInfo("slot_moved", PropertyInfo(Variant::INT, "old_slot_index")), PropertyInfo(Variant::INT, "new_slot_index"));
	ADD_SIGNAL(MethodInfo("slot_sizes_changed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_NODE_TYPE, "GraphNodeIndexed")));

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, panel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, panel_selected);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, panel_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, titlebar);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, titlebar_selected);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, port_selected);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphNodeIndexed, separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphNodeIndexed, port_h_offset);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphNodeIndexed, port);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphNodeIndexed, resizer);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphNodeIndexed, resizer_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, slot);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNodeIndexed, slot_selected);

	ADD_CLASS_DEPENDENCY("BoxContainer");
	ADD_CLASS_DEPENDENCY("GraphConnection");
	ADD_CLASS_DEPENDENCY("GraphNode");
	ADD_CLASS_DEPENDENCY("GraphPort");
	ADD_CLASS_DEPENDENCY("Label");
}

void GraphNodeIndexed::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "ports") {
		p_property.usage ^= PROPERTY_USAGE_READ_ONLY;
	}
}

GraphNodeIndexed::GraphNodeIndexed() {
}

GraphNodeIndexed::Slot::Slot() {
}
