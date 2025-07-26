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

	Control *control = cast_to<Control>(p_child);
	if (p_child->is_internal() || !control) {
		return;
	}
	if (p_child == port_container) {
		port_container_idx = p_child->get_index(false);
		return;
	}
	if (p_child->get_name() == port_container_name) {
		// Grab existing PortContainer here when it's loaded in
		set_port_container((Container *)p_child);
		port_container_idx = p_child->get_index(false);
	}
	// Check this after port container checks because other nodes might be tagged with this
	if (p_child->has_meta(ignore_node_meta_tag)) {
		return;
	}

	if (!control->is_connected("resized", callable_mp((GraphNode *)this, &GraphNode::_deferred_resort))) {
		control->connect("resized", callable_mp((GraphNode *)this, &GraphNode::_deferred_resort));
	}
	StringName child_name = p_child->get_name();

	// Child already exists in slot node cache - ignore and move on.
	if (_node_to_slot_cache.has(child_name)) {
		// edge case: node name not assigned, overwrite it
		int slot_idx = _node_to_slot_cache[child_name];
		if (slots[slot_idx].node_name.is_empty()) {
			slots.set(slot_idx, Slot(slots[slot_idx].draw_stylebox, child_name));
		}
		return;
	}

	int slot_index = slot_index_of_node(p_child);
	ERR_FAIL_COND_MSG(slot_index < 0, "Added child to GraphNodeIndexed, but couldn't find it in the node's children?? Failed to assign slot index, aborting!");

	if (!is_ready() && slots.size() > slot_index) {
		// Not ready yet! This should only happen when the node is instantiated along with children.
		// Properties are assigned before children are created, and a slot already exists for this child, so don't create a new one.
		// This keeps ports from being overridden/recreated on scene instantiation
		_node_to_slot_cache[child_name] = slot_index;
		return;
	}

	create_slot_and_ports(slot_index, true, child_name);
}

void GraphNodeIndexed::move_child_notify(Node *p_child) {
	GraphNode::move_child_notify(p_child);

	if (p_child->is_internal() || !is_ready() || !cast_to<Control>(p_child)) {
		return;
	}
	if (p_child == port_container) {
		port_container_idx = p_child->get_index(false);
		return;
	}
	// Check this after port container checks because other nodes might be tagged with this
	if (p_child->has_meta(ignore_node_meta_tag)) {
		return;
	}

	StringName child_name = p_child->get_name();
	ERR_FAIL_COND_MSG(!_node_to_slot_cache.has(child_name), "Moved child of GraphNodeIndexed that was never assigned a slot index?? Failed to swap slots, aborting!");
	int old_index = _node_to_slot_cache[child_name];
	int new_index = slot_index_of_node(p_child);

	move_slot_with_ports(old_index, new_index);
}

void GraphNodeIndexed::remove_child_notify(Node *p_child) {
	GraphNode::remove_child_notify(p_child);

	Control *control = cast_to<Control>(p_child);
	if (p_child->is_internal() || !is_ready() || !control || p_child == port_container || p_child->has_meta(ignore_node_meta_tag) || !_node_to_slot_cache.has(p_child->get_name()) || (port_container && port_container->get_parent() != this)) {
		return;
	}

	int index = _node_to_slot_cache[p_child->get_name()];
	ERR_FAIL_INDEX(index, slots.size());

	remove_slot_and_ports(index);

	if (control->is_connected("resized", callable_mp((GraphNode *)this, &GraphNode::_deferred_resort))) {
		control->disconnect("resized", callable_mp((GraphNode *)this, &GraphNode::_deferred_resort));
	}
}

void GraphNodeIndexed::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_RESIZED: {
			if (port_container) {
				port_container->set_rect(Rect2(0, 0, get_size().width, get_size().height));
			}
		} break;
		case NOTIFICATION_DRAW: {
			// Used for layout calculations.
			Ref<StyleBox> sb_panel = theme_cache.panel;

			Ref<StyleBox> sb_slot = theme_cache.slot;
			Ref<StyleBox> sb_slot_selected = theme_cache.slot_selected;

			//int port_h_offset = theme_cache.port_h_offset;

			int width = get_size().width - sb_panel->get_minimum_size().x;

			for (int i = 0; i < get_child_count(false); i++) {
				Control *_child = as_sortable_control(get_child(i, false), SortableVisibilityMode::VISIBLE_IN_TREE);
				if (!_child || _child->has_meta(ignore_node_meta_tag) || !_node_to_slot_cache.has(_child->get_name())) {
					continue;
				}
				int slot_index = _node_to_slot_cache[_child->get_name()];
				const Slot slot = slots[slot_index];

				// TODO: keyboard navigation override for slot selection
				/* if (slot_index == selected_slot) {
					Size2i port_sz = theme_cache.port->get_size();
					draw_style_box(sb_slot_selected, Rect2i(port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
					draw_style_box(sb_slot_selected, Rect2i(get_size().x - port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
				}*/

				if (slot.draw_stylebox) {
					Rect2 child_rect = _child->get_rect();
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

	_insert_slot(p_slot_index, Slot(p_draw_stylebox, p_slot_node_name));
	insert_port(slot_to_port_index(p_slot_index, true), p_left);
	insert_port(slot_to_port_index(p_slot_index, false), p_right);

	emit_signal("slot_added", p_slot_index);
}

void GraphNodeIndexed::remove_slot_and_ports(int p_slot_index) {
	_remove_slot(p_slot_index);
	int input_port_idx = slot_to_port_index(p_slot_index, true);
	int output_port_idx = slot_to_port_index(p_slot_index, false);
	if (output_port_idx < ports.size()) {
		remove_port(output_port_idx);
		remove_port(input_port_idx);
	}
	for (KeyValue<StringName, int> &kv_pair : _node_to_slot_cache) {
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
	set_ports_at_slot(p_slot_index, p_input_port, p_output_port);
}

void GraphNodeIndexed::set_ports_at_slot(int p_slot_index, GraphPort *p_input_port, GraphPort *p_output_port) {
	set_input_port_at_slot(p_slot_index, p_input_port);
	set_output_port_at_slot(p_slot_index, p_output_port);
}

GraphPort *GraphNodeIndexed::set_output_port_at_slot(int p_slot_index, GraphPort *p_port) {
	return set_port(slot_to_port_index(p_slot_index, false), p_port);
}

GraphPort *GraphNodeIndexed::set_input_port_at_slot(int p_slot_index, GraphPort *p_port) {
	return set_port(slot_to_port_index(p_slot_index, true), p_port);
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
		StringName n_name = StringName();
		if (p_slot.size() > 1) {
			n_name = p_slot[1];
		}
		GraphNodeIndexed::Slot slot = Slot(draw_sb, n_name);
		_insert_slot(i, slot);
		i++;
	}
}

GraphPort *GraphNodeIndexed::set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled) {
	GraphPort *old_port = GraphNode::_set_port(p_port_index, p_port, p_include_disabled);

	if (old_port == p_port) {
		return old_port;
	}
	if (!old_port || !p_port || old_port->get_parent() != p_port->get_parent()) {
		if (old_port && old_port->get_parent()) {
			//WARN_PRINT("hello i am removing on _set_port");
			old_port->get_parent()->remove_child(old_port);
		}
		if (p_port && (!port_container || p_port->get_parent() != port_container)) {
			//WARN_PRINT("hello i am adding on _set_port");
			ensure_port_container();
			port_container->add_child(p_port, true, Node::INTERNAL_MODE_DISABLED);
			if (get_owner()) {
				p_port->set_owner(get_owner());
			} else {
				p_port->set_owner(this);
			}
		}
	}

	_port_modified();
	return old_port;
}

void GraphNodeIndexed::add_port(GraphPort *p_port) {
	GraphNode::_add_port(p_port);

	if (p_port && (!port_container || p_port->get_parent() != port_container)) {
		//WARN_PRINT("hello i am working on _add_port");
		ensure_port_container();
		port_container->add_child(p_port, true, Node::INTERNAL_MODE_DISABLED);
		if (get_owner()) {
			p_port->set_owner(get_owner());
		} else {
			p_port->set_owner(this);
		}
	}

	_port_modified();
}

void GraphNodeIndexed::insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled) {
	GraphNode::_insert_port(p_port_index, p_port, p_include_disabled);

	if (p_port && (!port_container || p_port->get_parent() != port_container)) {
		//WARN_PRINT("hello i am working on _insert_port");
		ensure_port_container();
		port_container->add_child(p_port, true, Node::INTERNAL_MODE_DISABLED);
		if (get_owner()) {
			p_port->set_owner(get_owner());
		} else {
			p_port->set_owner(this);
		}
	}

	_port_modified();
}

GraphPort *GraphNodeIndexed::remove_port(int p_port_index, bool p_include_disabled) {
	GraphPort *ret = GraphNode::_remove_port(p_port_index, p_include_disabled);

	if (ret && ret->get_parent()) {
		//WARN_PRINT("uhhh i am _remove_port and this shit's cursed");
		ret->get_parent()->remove_child(ret);
		if (free_ports_on_slot_removed) {
			ret->queue_free();
		}
	}

	_port_modified();
	return ret;
}

const Vector<GraphNodeIndexed::Slot> &GraphNodeIndexed::_get_slots() {
	return slots;
}

TypedArray<Array> GraphNodeIndexed::get_slots() const {
	TypedArray<Array> ret;
	for (GraphNodeIndexed::Slot slot : slots) {
		Array s;
		s.push_back(slot.draw_stylebox);
		s.push_back(slot.node_name);
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

	for (KeyValue<StringName, int> &kv_pair : _node_to_slot_cache) {
		if (kv_pair.value == p_slot_index) {
			_node_to_slot_cache.erase(kv_pair.key);
		}
	}
}

void GraphNodeIndexed::_set_slot_node_cache(const TypedDictionary<StringName, int> &p_slot_node_map_cache) {
	_node_to_slot_cache.clear();
	for (const KeyValue<Variant, Variant> &kv : p_slot_node_map_cache) {
		_node_to_slot_cache[kv.key] = kv.value;
	}
}

TypedDictionary<StringName, int> GraphNodeIndexed::_get_slot_node_cache() {
	TypedDictionary<StringName, int> ret;
	for (const KeyValue<StringName, int> &kv : _node_to_slot_cache) {
		ret[kv.key] = kv.value;
	}
	return ret;
}

void GraphNodeIndexed::_insert_slot(int p_slot_index, const Slot p_slot) {
	slots.insert(p_slot_index, p_slot);
	if (p_slot_index < get_child_count(false)) {
		for (KeyValue<StringName, int> &kv_pair : _node_to_slot_cache) {
			if (p_slot_index <= kv_pair.value) {
				kv_pair.value++;
			}
		}
	}
	_node_to_slot_cache[p_slot.node_name] = p_slot_index;
}

void GraphNodeIndexed::_set_slot(int p_slot_index, const Slot p_slot) {
	slots.set(p_slot_index, p_slot);
	_node_to_slot_cache[p_slot.node_name] = p_slot_index;
}

void GraphNodeIndexed::_resort() {
	GraphNode::_resort();

	// Special case: stretch port container to fill node
	if (port_container) {
		port_container->set_rect(Rect2(0, 0, get_size().width, get_size().height));
	}

	emit_signal(SNAME("slot_sizes_changed"), this);
}

void GraphNodeIndexed::_update_port_positions() {
	// Grab theme references for layout
	Ref<StyleBoxFlat> sb_panel = theme_cache.panel;
	Ref<StyleBox> sb_titlebar = theme_cache.titlebar;
	int separation = theme_cache.separation;

	// This helps to immediately achieve the initial y "original point" of the slots, which the sum of the titlebar height and the top margin of the panel.
	int vertical_ofs = titlebar_hbox->get_size().height + (sb_titlebar.is_valid() ? sb_titlebar->get_minimum_size().height : 0) + (sb_panel.is_valid() ? sb_panel->get_margin(SIDE_TOP) : 0);

	// Node x and port x positions are uniform for all ports, so find them now
	int node_width = get_size().width;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::VISIBLE_IN_TREE);
		if (!child || child->has_meta(ignore_node_meta_tag)) {
			continue;
		}

		Size2i size = child->get_size();
		int port_y = vertical_ofs + size.height * 0.5;

		if (_node_to_slot_cache.has(child->get_name())) {
			int slot_index = _node_to_slot_cache[child->get_name()];
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

// Notably, this method does not query or update _node_to_slot_cache
int GraphNodeIndexed::slot_index_of_node(const Node *p_node) const {
	ERR_FAIL_NULL_V(p_node, -1);
	ERR_FAIL_COND_V(!is_ancestor_of(p_node), -1);

	// Count control nodes above this node in the hierarchy to get the slot index
	int slot_counter = 0;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = cast_to<Control>(get_child(i, false));
		if (!child || child->has_meta(ignore_node_meta_tag)) {
			continue;
		}
		if (child == p_node) {
			return slot_counter;
		}
		slot_counter++;
	}
	return -1;
}

// helpers to account for port container when indexing children for slots
int GraphNodeIndexed::child_to_slot_index(int idx, bool p_include_internal) const {
	ERR_FAIL_INDEX_V(idx, get_child_count(p_include_internal), -1);
	return slot_index_of_node(get_child(idx, p_include_internal));
}
int GraphNodeIndexed::slot_to_child_index(int idx, bool p_include_internal) const {
	Node *child = get_child_by_slot_index(idx);
	ERR_FAIL_NULL_V(child, -1);
	return child->get_index(p_include_internal);
}

Node *GraphNodeIndexed::get_child_by_slot_index(int p_slot_index) const {
	ERR_FAIL_INDEX_V(p_slot_index, slots.size(), nullptr);
	return get_node_or_null(NodePath(slots[p_slot_index].node_name));
}

Node *GraphNodeIndexed::get_child_by_port(const GraphPort *p_port) const {
	return get_child_by_slot_index(slot_index_of_port(p_port));
}

int GraphNodeIndexed::slot_index_of_port(const GraphPort *p_port) const {
	return floor(index_of_port(p_port) / 2);
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

GraphPort *GraphNodeIndexed::get_input_port_by_slot(int p_slot_index) const {
	int port_index = slot_to_port_index(p_slot_index, true);
	return get_port(port_index);
}

GraphPort *GraphNodeIndexed::get_output_port_by_slot(int p_slot_index) const {
	int port_index = slot_to_port_index(p_slot_index, false);
	return get_port(port_index);
}

GraphPort *GraphNodeIndexed::get_input_port_by_node(const Node *p_node) const {
	ERR_FAIL_NULL_V(p_node, nullptr);
	int slot_idx = slot_index_of_node(p_node);
	if (slot_idx < 0) {
		return nullptr;
	}
	return get_input_port_by_slot(slot_idx);
}

GraphPort *GraphNodeIndexed::get_output_port_by_node(const Node *p_node) const {
	ERR_FAIL_NULL_V(p_node, nullptr);
	int slot_idx = slot_index_of_node(p_node);
	if (slot_idx < 0) {
		return nullptr;
	}
	return get_output_port_by_slot(slot_idx);
}

int GraphNodeIndexed::port_to_slot_index(int p_port_index, bool p_include_disabled) const {
	ERR_FAIL_INDEX_V(p_port_index, ports.size(), -1);
	ERR_FAIL_INDEX_V(p_port_index, slots.size() * 2, -1);
	if (p_include_disabled) {
		return floor(p_port_index / 2);
	}
	int idx = 0;
	for (int i = 0; i < slots.size(); i++) {
		int input_port_idx = i * 2;
		int output_port_idx = input_port_idx + 1;
		if (ports[input_port_idx] && ports[input_port_idx]->is_enabled()) {
			if (idx == p_port_index) {
				return i;
			}
			idx++;
		}
		if (ports[output_port_idx] && ports[output_port_idx]->is_enabled()) {
			if (idx == p_port_index) {
				return i;
			}
			idx++;
		}
	}
	return -1;
}

int GraphNodeIndexed::slot_to_port_index(int p_slot_index, bool p_input, bool p_include_disabled) const {
	int idx = p_slot_index * 2 + (p_input ? 0 : 1);
	if (p_include_disabled) {
		return idx;
	} else {
		ERR_FAIL_INDEX_V(idx, ports.size(), -1);
		GraphPort *port = Object::cast_to<GraphPort>(ports[idx]);
		return port->get_port_index(false);
	}
}

int GraphNodeIndexed::slot_to_input_port_index(int p_slot_index, bool p_include_disabled) const {
	ERR_FAIL_INDEX_V(p_slot_index * 2, ports.size(), -1);
	GraphPort *port = Object::cast_to<GraphPort>(ports[p_slot_index * 2]);
	ERR_FAIL_NULL_V(port, -1);
	return port->get_filtered_port_index(p_include_disabled);
}

int GraphNodeIndexed::slot_to_output_port_index(int p_slot_index, bool p_include_disabled) const {
	ERR_FAIL_INDEX_V(p_slot_index * 2 + 1, ports.size(), -1);
	GraphPort *port = Object::cast_to<GraphPort>(ports[p_slot_index * 2 + 1]);
	ERR_FAIL_NULL_V(port, -1);
	return port->get_filtered_port_index(p_include_disabled);
}

int GraphNodeIndexed::input_port_to_slot_index(int p_port_index, bool p_include_disabled) const {
	GraphPort *port = get_filtered_port(p_port_index, GraphPort::PortDirection::INPUT, p_include_disabled);
	return slot_index_of_port(port);
}

int GraphNodeIndexed::output_port_to_slot_index(int p_port_index, bool p_include_disabled) const {
	GraphPort *port = get_filtered_port(p_port_index, GraphPort::PortDirection::OUTPUT, p_include_disabled);
	return slot_index_of_port(port);
}

bool GraphNodeIndexed::get_slot_draw_stylebox(int p_slot_index) const {
	ERR_FAIL_INDEX_V(p_slot_index, slots.size(), false);
	return slots[p_slot_index].draw_stylebox;
}

void GraphNodeIndexed::set_slot_draw_stylebox(int p_slot_index, bool p_draw_stylebox) {
	ERR_FAIL_INDEX(p_slot_index, slots.size());
	Slot old_slot = slots[p_slot_index];
	slots.set(p_slot_index, Slot(p_draw_stylebox, old_slot.node_name));
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

void GraphNodeIndexed::set_port_container(Control *p_container) {
	ERR_FAIL_NULL(p_container);
	if (port_container == p_container) {
		return;
	}
	port_container = p_container;
	port_container->set_meta(ignore_node_meta_tag, true);
}

Control *GraphNodeIndexed::get_port_container() const {
	return port_container;
}

void GraphNodeIndexed::ensure_port_container() {
	if (!port_container) {
		port_container = memnew(Control);
		port_container->set_name(port_container_name);
		port_container->set_focus_mode(Control::FOCUS_NONE);
		port_container->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
		port_container->set_h_size_flags(SIZE_EXPAND_FILL);
		port_container->set_anchors_preset(Control::PRESET_TOP_WIDE);
		port_container->set_meta(ignore_node_meta_tag, true);
		add_child(port_container, true, Node::INTERNAL_MODE_DISABLED);
		move_child(port_container, 0);
		if (get_owner()) {
			port_container->set_owner(get_owner());
		} else {
			port_container->set_owner(this);
		}
		port_container->set_rect(Rect2(0, 0, get_size().width, get_size().height));
	}
}

void GraphNodeIndexed::set_free_ports_on_slot_removed(bool p_free_ports) {
	free_ports_on_slot_removed = p_free_ports;
}

bool GraphNodeIndexed::get_free_ports_on_slot_removed() const {
	return free_ports_on_slot_removed;
}

void GraphNodeIndexed::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_slots", "slots"), &GraphNodeIndexed::set_slots);
	ClassDB::bind_method(D_METHOD("get_slots"), &GraphNodeIndexed::get_slots);

	ClassDB::bind_method(D_METHOD("_set_slot_node_cache", "_node_to_slot_cache"), &GraphNodeIndexed::_set_slot_node_cache);
	ClassDB::bind_method(D_METHOD("_get_slot_node_cache"), &GraphNodeIndexed::_get_slot_node_cache);

	ClassDB::bind_method(D_METHOD("set_slot_properties", "slot_index", "input_enabled", "input_type", "output_enabled", "output_type"), &GraphNodeIndexed::set_slot_properties);
	ClassDB::bind_method(D_METHOD("set_input_port_properties", "slot_index", "enabled", "type"), &GraphNodeIndexed::set_input_port_properties);
	ClassDB::bind_method(D_METHOD("set_output_port_properties", "slot_index", "enabled", "type"), &GraphNodeIndexed::set_output_port_properties);

	ClassDB::bind_method(D_METHOD("get_input_port_by_slot", "slot_index"), &GraphNodeIndexed::get_input_port_by_slot);
	ClassDB::bind_method(D_METHOD("get_output_port_by_slot", "slot_index"), &GraphNodeIndexed::get_output_port_by_slot);
	ClassDB::bind_method(D_METHOD("get_input_port_by_node", "node"), &GraphNodeIndexed::get_input_port_by_node);
	ClassDB::bind_method(D_METHOD("get_output_port_by_node", "node"), &GraphNodeIndexed::get_output_port_by_node);

	ClassDB::bind_method(D_METHOD("slot_index_of_node", "node"), &GraphNodeIndexed::slot_index_of_node);
	ClassDB::bind_method(D_METHOD("slot_index_of_port", "port"), &GraphNodeIndexed::slot_index_of_port);

	ClassDB::bind_method(D_METHOD("port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::port_to_slot_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("slot_to_port_index", "slot_index", "is_input_port", "include_disabled"), &GraphNodeIndexed::slot_to_port_index, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("child_to_slot_index", "child_index", "include_internal"), &GraphNodeIndexed::child_to_slot_index, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("slot_to_child_index", "slot_index", "include_internal"), &GraphNodeIndexed::slot_to_child_index, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("slot_to_input_port_index", "slot_index", "include_disabled"), &GraphNodeIndexed::slot_to_input_port_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("slot_to_output_port_index", "slot_index", "include_disabled"), &GraphNodeIndexed::slot_to_output_port_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("input_port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::input_port_to_slot_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("output_port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::output_port_to_slot_index, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("set_slot_focus_mode", "focus_mode"), &GraphNodeIndexed::set_slot_focus_mode);
	ClassDB::bind_method(D_METHOD("get_slot_focus_mode"), &GraphNodeIndexed::get_slot_focus_mode);

	ClassDB::bind_method(D_METHOD("get_child_by_slot_index", "slot_index"), &GraphNodeIndexed::get_child_by_slot_index);
	ClassDB::bind_method(D_METHOD("get_child_by_port", "port"), &GraphNodeIndexed::get_child_by_port);

	ClassDB::bind_method(D_METHOD("set_port_container", "port_container"), &GraphNodeIndexed::set_port_container);

	ClassDB::bind_method(D_METHOD("set_ports_at_slot", "slot_index", "input_port", "output_port"), &GraphNodeIndexed::set_ports_at_slot);
	ClassDB::bind_method(D_METHOD("set_input_port_at_slot", "slot_index", "port"), &GraphNodeIndexed::set_input_port_at_slot);
	ClassDB::bind_method(D_METHOD("set_output_port_at_slot", "slot_index", "port"), &GraphNodeIndexed::set_output_port_at_slot);
	ClassDB::bind_method(D_METHOD("get_port_container"), &GraphNodeIndexed::get_port_container);

	ClassDB::bind_method(D_METHOD("set_free_ports_on_slot_removed", "free_ports"), &GraphNodeIndexed::set_free_ports_on_slot_removed);
	ClassDB::bind_method(D_METHOD("get_free_ports_on_slot_removed"), &GraphNodeIndexed::get_free_ports_on_slot_removed);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "slots", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_slots", "get_slots");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "slot_focus_mode", PROPERTY_HINT_ENUM, "Click:1,All:2,Accessibility:3"), "set_slot_focus_mode", "get_slot_focus_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "free_ports_on_slot_removed"), "set_free_ports_on_slot_removed", "get_free_ports_on_slot_removed");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_node_to_slot_cache", PROPERTY_HINT_DICTIONARY_TYPE, "StringName:int", PROPERTY_USAGE_STORAGE), "_set_slot_node_cache", "_get_slot_node_cache");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "port_container", PROPERTY_HINT_NODE_TYPE, "Container", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL), "set_port_container", "get_port_container");

	ADD_SIGNAL(MethodInfo("slot_added", PropertyInfo(Variant::INT, "slot_index")));
	ADD_SIGNAL(MethodInfo("slot_removed", PropertyInfo(Variant::INT, "slot_index")));
	ADD_SIGNAL(MethodInfo("slot_moved", PropertyInfo(Variant::INT, "old_slot_index"), PropertyInfo(Variant::INT, "new_slot_index")));
	ADD_SIGNAL(MethodInfo("slot_sizes_changed", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "GraphNodeIndexed")));

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
	callable_mp(this, &GraphNodeIndexed::ensure_port_container).call_deferred();
}

GraphNodeIndexed::Slot::Slot() {
}
