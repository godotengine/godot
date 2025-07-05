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

void GraphNodeIndexed::add_child_notify(Node *p_child) {
	GraphNode::add_child_notify(p_child);

	if (p_child->is_internal() || p_child == port_container) {
		return;
	}

	int index = p_child->get_index(false);
	if (!is_ready()) {
		_slot_node_map_cache[p_child->get_name()] = index;
		return;
	}

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	create_slot_and_ports(index, true);
}

void GraphNodeIndexed::move_child_notify(Node *p_child) {
	GraphNode::move_child_notify(p_child);

	if (p_child->is_internal() || !is_ready() || p_child == port_container) {
		return;
	}

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	StringName node_name = p_child->get_name();
	int new_index = p_child->get_index(false);
	int old_index = _slot_node_map_cache[node_name];

	if (old_index < new_index) {
		Slot swap_buffer = slots[old_index];
		for (int i = old_index; i < new_index; i++) {
			_set_slot(i, slots[i + 1]);
		}
		_set_slot(new_index, swap_buffer);
	} else {
		Slot swap_buffer = slots[new_index];
		for (int i = new_index; i < old_index; i++) {
			_set_slot(i, slots[i - 1]);
		}
		_set_slot(old_index, swap_buffer);
	}
}

void GraphNodeIndexed::remove_child_notify(Node *p_child) {
	GraphNode::remove_child_notify(p_child);

	if (p_child->is_internal() || !is_ready() || p_child == port_container) {
		return;
	}

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	int index = p_child->get_index(false);
	ERR_FAIL_INDEX(index, slots.size());

	_remove_slot(index);
}

void GraphNodeIndexed::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			if (theme_cache.slot.is_valid()) {
				Size2 min_slot_s = theme_cache.slot->get_minimum_size();
				if (theme_cache.slot_selected.is_valid()) {
					min_slot_s = min_slot_s.max(theme_cache.slot_selected->get_minimum_size());
				}
				port_container->set_custom_minimum_size(min_slot_s);
			}
		} break;
		case NOTIFICATION_DRAW: {
			// Used for layout calculations.
			Ref<StyleBox> sb_panel = theme_cache.panel;

			Ref<StyleBox> sb_slot = theme_cache.slot;
			Ref<StyleBox> sb_slot_selected = theme_cache.slot_selected;

			//int port_h_offset = theme_cache.port_h_offset;

			int width = get_size().width - sb_panel->get_minimum_size().x;

			if (get_child_count(false) > 0) {
				int slot_index = 0;
				for (const Slot &slot : slots) {
					// TODO: keyboard navigation override for slot selection
					/* if (slot_index == selected_slot) {
						Size2i port_sz = theme_cache.port->get_size();
						draw_style_box(sb_slot_selected, Rect2i(port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
						draw_style_box(sb_slot_selected, Rect2i(get_size().x - port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
					}*/

					// Draw slot stylebox.
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

					slot_index++;
				}
			}

			if (resizable) {
				draw_texture(theme_cache.resizer, get_size() - theme_cache.resizer->get_size(), theme_cache.resizer_color);
			}
		} break;
	}
}

void GraphNodeIndexed::create_slot(int p_slot_index, GraphPort *p_left_port, GraphPort *p_right_port, bool draw_stylebox) {
	Slot new_slot = { p_left_port, p_right_port, true };
	p_right_port->add_theme_constant_override("hotzone_offset_h", -p_right_port->get_theme_constant("hotzone_offset_h"));
	_insert_slot(p_slot_index, new_slot);
}

void GraphNodeIndexed::create_slot_and_ports(int p_slot_index, bool draw_stylebox) {
	GraphPort *p_left = memnew(GraphPort(false, true, 0, GraphPort::PortDirection::INPUT));
	GraphPort *p_right = memnew(GraphPort(false, false, 0, GraphPort::PortDirection::OUTPUT));
	port_container->add_child(p_left);
	port_container->add_child(p_right);
	return create_slot(p_slot_index, p_left, p_right, draw_stylebox);
}

void GraphNodeIndexed::_set_slots(const Vector<Slot> &p_slots) {
	_remove_all_slots();
	int i = 0;
	for (const Slot &slot : p_slots) {
		_insert_slot(i, slot, true);
		i++;
	}
}

void GraphNodeIndexed::set_slots(TypedArray<Array> p_slots) {
	_remove_all_slots();
	int i = 0;
	for (Array p_slot : p_slots) {
		GraphPort *first_port = Object::cast_to<GraphPort>(p_slot[0]);
		GraphPort *second_port = Object::cast_to<GraphPort>(p_slot[0]);
		bool draw_sb = p_slot[2];
		GraphNodeIndexed::Slot slot = Slot(first_port, second_port, draw_sb);
		_insert_slot(i, slot, false);
		i++;
	}
	notify_property_list_changed();
}

const Vector<GraphNodeIndexed::Slot> &GraphNodeIndexed::_get_slots() {
	return slots;
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

GraphNodeIndexed::Slot GraphNodeIndexed::_get_slot(int p_slot_index) {
	return slots[p_slot_index];
}

Array GraphNodeIndexed::get_slot(int p_slot_index) {
	Slot slot = _get_slot(p_slot_index);
	return Array({ slot.left_port, slot.right_port, slot.draw_stylebox });
}

void GraphNodeIndexed::_remove_all_slots(bool p_with_ports) {
	if (slots.is_empty()) {
		return;
	}
	for (int i = slots.size() - 1; i >= 0; i--) {
		_remove_slot(i, p_with_ports);
	}
}

void GraphNodeIndexed::_remove_slot(int p_slot_index, bool p_with_ports) {
	if (p_with_ports) {
		int p_right_port_index = slot_to_port_index(p_slot_index, false);
		remove_port(p_right_port_index);
		int p_left_port_index = slot_to_port_index(p_slot_index, true);
		remove_port(p_left_port_index);
	}

	slots.remove_at(p_slot_index);

	for (const KeyValue<StringName, int> &kv_pair : _slot_node_map_cache) {
		if (kv_pair.value == p_slot_index) {
			_slot_node_map_cache.erase(kv_pair.key);
			break;
		}
	}

	notify_property_list_changed();
}

void GraphNodeIndexed::_insert_slot(int p_slot_index, const Slot p_slot, bool p_with_ports) {
	slots.insert(p_slot_index, p_slot);

	if (p_with_ports) {
		int p_left_port_index = slot_to_port_index(p_slot_index, true);
		insert_port(p_left_port_index, p_slot.left_port);
		int p_right_port_index = slot_to_port_index(p_slot_index, false);
		insert_port(p_right_port_index, p_slot.right_port);
	}

	if (p_slot_index < get_child_count(false)) {
		_slot_node_map_cache[get_child(p_slot_index, false)->get_name()] = p_slot_index;
	}

	notify_property_list_changed();
}

void GraphNodeIndexed::_set_slot(int p_slot_index, const Slot p_slot, bool p_with_ports) {
	slots.set(p_slot_index, p_slot);

	if (p_with_ports) {
		int p_left_port_index = slot_to_port_index(p_slot_index, true);
		set_port(p_left_port_index, p_slot.left_port);
		int p_right_port_index = slot_to_port_index(p_slot_index, false);
		set_port(p_right_port_index, p_slot.right_port);
	}

	if (p_slot_index < get_child_count(false)) {
		_slot_node_map_cache[get_child(p_slot_index, false)->get_name()] = p_slot_index;
	}

	notify_property_list_changed();
}

void GraphNodeIndexed::set_slot(int p_slot_index, GraphPort *p_left_port, GraphPort *p_right_port, bool draw_stylebox) {
	_set_slot(p_slot_index, Slot({ p_left_port, p_right_port, draw_stylebox }));
}

void GraphNodeIndexed::_resort() {
	GraphNode::_resort();
	emit_signal(SNAME("slot_sizes_changed"), this);
}

// I have absolutely no idea why so many things in this function need to be divided in half to align ports properly, but here we are
void GraphNodeIndexed::_port_pos_update() {
	int edgeofs = theme_cache.port_h_offset;
	int separation = theme_cache.separation;

	// This helps to immediately achieve the initial y "original point" of the slots, which the sum of the titlebar height and the top margin of the panel.
	int vertical_ofs = titlebar_hbox->get_size().height / 2 + theme_cache.titlebar->get_minimum_size().height / 2 + theme_cache.panel->get_margin(SIDE_TOP);

	Size2 node_size = get_size() / 2;
	Point2 port_container_size = port_container->get_size();

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::IGNORE);
		if (!child) {
			continue;
		}

		Size2i size = child->get_size() / 2;
		int port_y = vertical_ofs + size.height / 2;

		if (_slot_node_map_cache.has(child->get_name())) {
			int slot_index = _slot_node_map_cache[child->get_name()];
			const Slot &slot = slots[slot_index];
			if (slot.left_port) {
				slot.left_port->set_position(Point2(edgeofs, port_y));
			}
			if (slot.right_port) {
				slot.right_port->set_position(Point2(node_size.width + port_container_size.width - edgeofs, port_y));
			}
		}

		vertical_ofs += size.height + separation;
	}

	GraphNode::_port_pos_update();
}

int GraphNodeIndexed::slot_index_of_port(GraphPort *p_port) {
	int idx = 0;
	for (const Slot &slot : slots) {
		if (slot.left_port == p_port || slot.right_port == p_port) {
			return idx;
		}
		idx++;
	}
	return -1;
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
	GraphPort *port = get_input_port(p_slot_index);
	ERR_FAIL_NULL(port);
	port->set_properties(p_enabled, true, p_type, GraphPort::PortDirection::INPUT);
}

void GraphNodeIndexed::set_output_port_properties(int p_slot_index, bool p_enabled, int p_type) {
	GraphPort *port = get_output_port(p_slot_index);
	ERR_FAIL_NULL(port);
	port->set_properties(p_enabled, false, p_type, GraphPort::PortDirection::OUTPUT);
}

GraphPort *GraphNodeIndexed::get_input_port(int p_slot_index) {
	int port_index = slot_to_port_index(p_slot_index, true);
	return get_port(port_index);
}

GraphPort *GraphNodeIndexed::get_output_port(int p_slot_index) {
	int port_index = slot_to_port_index(p_slot_index, false);
	return get_port(port_index);
}

void GraphNodeIndexed::set_input_port(int p_slot_index, GraphPort *p_port) {
	Slot old_slot = slots[p_slot_index];
	slots.set(p_slot_index, Slot(p_port, old_slot.right_port, old_slot.draw_stylebox));
	set_port(slot_to_port_index(p_slot_index, true), p_port);
	notify_property_list_changed();
}

void GraphNodeIndexed::set_output_port(int p_slot_index, GraphPort *p_port) {
	Slot old_slot = slots[p_slot_index];
	slots.set(p_slot_index, Slot(old_slot.left_port, p_port, old_slot.draw_stylebox));
	set_port(slot_to_port_index(p_slot_index, false), p_port);
	notify_property_list_changed();
}

int GraphNodeIndexed::get_input_port_count() {
	return get_filtered_port_count(GraphPort::PortDirection::INPUT);
}

int GraphNodeIndexed::get_output_port_count() {
	return get_filtered_port_count(GraphPort::PortDirection::OUTPUT);
}

int GraphNodeIndexed::port_to_slot_index(int p_port_index, bool p_include_disabled) {
	int idx = 0;
	int slot_idx = 0;
	for (const Slot &slot : slots) {
		if (slot.left_port && (p_include_disabled || slot.left_port->get_enabled())) {
			if (idx == p_port_index) {
				return slot_idx;
			}
			idx++;
		}
		if (slot.right_port && (p_include_disabled || slot.right_port->get_enabled())) {
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
		return port->get_index(false);
	}
}

int GraphNodeIndexed::slot_to_input_port_index(int p_slot_index, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_slot_index * 2, ports.size(), -1);
	GraphPort *port = Object::cast_to<GraphPort>(ports[p_slot_index * 2]);
	ERR_FAIL_NULL_V(port, -1);
	return port->get_filtered_index(p_include_disabled);
}

int GraphNodeIndexed::slot_to_output_port_index(int p_slot_index, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_slot_index * 2 + 1, ports.size(), -1);
	GraphPort *port = Object::cast_to<GraphPort>(ports[p_slot_index * 2 + 1]);
	ERR_FAIL_NULL_V(port, -1);
	return port->get_filtered_index(p_include_disabled);
}

int GraphNodeIndexed::input_port_to_slot_index(int p_port_index, bool p_include_disabled) {
	GraphPort *port = get_filtered_port(p_port_index, GraphPort::PortDirection::INPUT, p_include_disabled);
	return slot_index_of_port(port);
}

int GraphNodeIndexed::output_port_to_slot_index(int p_port_index, bool p_include_disabled) {
	GraphPort *port = get_filtered_port(p_port_index, GraphPort::PortDirection::OUTPUT, p_include_disabled);
	return slot_index_of_port(port);
}

bool GraphNodeIndexed::get_slot_draw_stylebox(int p_slot_index) {
	ERR_FAIL_INDEX_V(p_slot_index, slots.size(), false);
	return slots[p_slot_index].draw_stylebox;
}

void GraphNodeIndexed::set_slot_draw_stylebox(int p_slot_index, bool p_draw_stylebox) {
	ERR_FAIL_INDEX(p_slot_index, slots.size());
	Slot old_slot = slots[p_slot_index];
	// set slot directly since ports aren't changed
	slots.set(p_slot_index, Slot(old_slot.left_port, old_slot.right_port, p_draw_stylebox));
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

	ClassDB::bind_method(D_METHOD("set_slot_properties", "slot_index", "input_enabled", "input_type", "output_enabled", "output_type"), &GraphNodeIndexed::set_slot_properties);
	ClassDB::bind_method(D_METHOD("set_input_port_properties", "slot_index", "enabled", "type"), &GraphNodeIndexed::set_input_port_properties);
	ClassDB::bind_method(D_METHOD("set_output_port_properties", "slot_index", "enabled", "type"), &GraphNodeIndexed::set_output_port_properties);

	ClassDB::bind_method(D_METHOD("set_input_port", "port"), &GraphNodeIndexed::set_input_port);
	ClassDB::bind_method(D_METHOD("set_output_port", "port"), &GraphNodeIndexed::set_output_port);
	ClassDB::bind_method(D_METHOD("get_input_port"), &GraphNodeIndexed::get_input_port);
	ClassDB::bind_method(D_METHOD("get_output_port"), &GraphNodeIndexed::get_output_port);

	ClassDB::bind_method(D_METHOD("get_input_port_count"), &GraphNodeIndexed::get_input_port_count);
	ClassDB::bind_method(D_METHOD("get_output_port_count"), &GraphNodeIndexed::get_output_port_count);

	ClassDB::bind_method(D_METHOD("slot_index_of_port", "port"), &GraphNodeIndexed::slot_index_of_port);
	ClassDB::bind_method(D_METHOD("index_of_input_port", "port", "include_disabled"), &GraphNodeIndexed::index_of_input_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("index_of_output_port", "port", "include_disabled"), &GraphNodeIndexed::index_of_output_port, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::port_to_slot_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("slot_to_port_index", "slot_index", "is_input_port", "include_disabled"), &GraphNodeIndexed::slot_to_port_index, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("slot_to_input_port_index", "slot_index", "include_disabled"), &GraphNodeIndexed::slot_to_input_port_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("slot_to_output_port_index", "slot_index", "include_disabled"), &GraphNodeIndexed::slot_to_output_port_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("input_port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::input_port_to_slot_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("output_port_to_slot_index", "port_index", "include_disabled"), &GraphNodeIndexed::output_port_to_slot_index, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("set_slot_focus_mode", "focus_mode"), &GraphNodeIndexed::set_slot_focus_mode);
	ClassDB::bind_method(D_METHOD("get_slot_focus_mode"), &GraphNodeIndexed::get_slot_focus_mode);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "slots", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_slots", "get_slots");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "slot_focus_mode", PROPERTY_HINT_ENUM, "Click:1,All:2,Accessibility:3"), "set_slot_focus_mode", "get_slot_focus_mode");

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
}

GraphNodeIndexed::GraphNodeIndexed() {
	set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	if (theme_cache.slot.is_valid()) {
		Size2 min_slot_s = theme_cache.slot->get_minimum_size();
		if (theme_cache.slot_selected.is_valid()) {
			min_slot_s = min_slot_s.max(theme_cache.slot_selected->get_minimum_size());
		}
		port_container->set_custom_minimum_size(min_slot_s);
	}
}

GraphNodeIndexed::Slot::Slot() {
	left_port = nullptr;
	right_port = nullptr;
}
