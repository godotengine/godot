/**************************************************************************/
/*  graph_node.cpp                                                        */
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

#include "graph_node.h"

#include "scene/gui/box_container.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/graph_port.h"
#include "scene/gui/label.h"
#include "scene/theme/theme_db.h"

void GraphNode::_set_ports(const TypedArray<Ref<GraphPort>> &p_ports) {
	remove_all_ports();

	for (Ref<GraphPort> port : p_ports) {
		add_port(port);
	}
}

const TypedArray<Ref<GraphPort>> &GraphNode::_get_ports() {
	return ports;
}
/*
bool GraphNode::_set(const StringName &p_name, const Variant &p_value) {
	//if (property_helper.property_set_value(p_name, p_value)) {
	//	return true;
	//}

	String str = p_name;

	if (str == "title") {
		title = p_value;
		return true;
	} else if (str == "ignore_invalid_connection_type") {
		ignore_invalid_connection_type = p_value;
		return true;
	}

	int idx = str.get_slicec('/', 1).to_int();
	if (ports.size() <= idx) {
		return false;
	}

	set_port(idx, p_value);

	return true;
}

bool GraphNode::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;

	if (str == "title") {
		r_ret = title;
		return true;
	} else if (str == "ignore_invalid_connection_type") {
		r_ret = ignore_invalid_connection_type;
		return true;
	}

	if (!str.begins_with("port/")) {
		return false;
	}

	int idx = str.get_slicec('/', 1).to_int();
	if (ports.size() <= idx) {
		return false;
	}

	r_ret = ports[idx];
	return true;
}

void GraphNode::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::STRING, "title"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "ignore_invalid_connection_type"));
	int idx = 0;
	for (Ref<GraphPort> port : ports) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "port/" + itos(idx), PROPERTY_HINT_RESOURCE_TYPE, "GraphPort"));
		idx++;
	}
}*/

void GraphNode::_resort() {
	Size2 new_size = get_size();
	Ref<StyleBox> sb_panel = theme_cache.panel;
	Ref<StyleBox> sb_titlebar = theme_cache.titlebar;

	// Resort titlebar first.
	Size2 titlebar_size = Size2(new_size.width, titlebar_hbox->get_size().height);
	titlebar_size -= sb_titlebar->get_minimum_size();
	Rect2 titlebar_rect = Rect2(sb_titlebar->get_offset(), titlebar_size);
	fit_child_in_rect(titlebar_hbox, titlebar_rect);

	// After resort, the children of the titlebar container may have changed their height (e.g. Label autowrap).
	Size2i titlebar_min_size = titlebar_hbox->get_combined_minimum_size();

	// First pass, determine minimum size AND amount of stretchable elements.
	int separation = theme_cache.separation;

	int children_count = 0;
	int stretch_min = 0;
	int available_stretch_space = 0;
	float stretch_ratio_total = 0;
	HashMap<Control *, _MinSizeCache> min_size_cache;

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false));
		if (!child) {
			continue;
		}

		Size2i size = child->get_combined_minimum_size();

		stretch_min += size.height;

		_MinSizeCache msc;
		msc.min_size = size.height;
		msc.will_stretch = child->get_v_size_flags().has_flag(SIZE_EXPAND);
		msc.final_size = msc.min_size;
		min_size_cache[child] = msc;

		if (msc.will_stretch) {
			available_stretch_space += msc.min_size;
			stretch_ratio_total += child->get_stretch_ratio();
		}

		children_count++;
	}
	if (selected_port >= enabled_port_count) {
		selected_port = -1;
	}

	if (children_count == 0) {
		return;
	}

	int stretch_max = new_size.height - (children_count - 1) * separation;
	int stretch_diff = stretch_max - stretch_min;

	// Avoid negative stretch space.
	stretch_diff = MAX(stretch_diff, 0);

	available_stretch_space += stretch_diff - sb_panel->get_margin(SIDE_BOTTOM) - sb_panel->get_margin(SIDE_TOP) - titlebar_min_size.height - sb_titlebar->get_minimum_size().height;

	// Second pass, discard elements that can't be stretched, this will run while stretchable elements exist.

	while (stretch_ratio_total > 0) {
		// First of all, don't even be here if no stretchable objects exist.
		bool refit_successful = true;

		for (int i = 0; i < get_child_count(false); i++) {
			Control *child = as_sortable_control(get_child(i, false));
			if (!child) {
				continue;
			}

			ERR_FAIL_COND(!min_size_cache.has(child));
			_MinSizeCache &msc = min_size_cache[child];

			if (msc.will_stretch) {
				int final_pixel_size = available_stretch_space * child->get_stretch_ratio() / stretch_ratio_total;
				if (final_pixel_size < msc.min_size) {
					// If the available stretching area is too small for a Control,
					// then remove it from stretching area.
					msc.will_stretch = false;
					stretch_ratio_total -= child->get_stretch_ratio();
					refit_successful = false;
					available_stretch_space -= msc.min_size;
					msc.final_size = msc.min_size;
					break;
				} else {
					msc.final_size = final_pixel_size;
				}
			}
		}

		if (refit_successful) {
			break;
		}
	}

	// Final pass, draw and stretch elements.

	int ofs_y = sb_panel->get_margin(SIDE_TOP) + titlebar_min_size.height + sb_titlebar->get_minimum_size().height;
	int width = new_size.width - sb_panel->get_minimum_size().width;
	int valid_children_idx = 0;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false));
		if (!child) {
			continue;
		}

		_MinSizeCache &msc = min_size_cache[child];

		if (valid_children_idx > 0) {
			ofs_y += separation;
		}

		int from_y_pos = ofs_y;
		int to_y_pos = ofs_y + msc.final_size;

		// Adjust so the last valid child always fits perfect, compensating for numerical imprecision.
		if (msc.will_stretch && valid_children_idx == children_count - 1) {
			to_y_pos = new_size.height - sb_panel->get_margin(SIDE_BOTTOM);
		}

		int height = to_y_pos - from_y_pos;
		float margin = sb_panel->get_margin(SIDE_LEFT);
		float final_width = width;
		Rect2 rect(margin, from_y_pos, final_width, height);
		fit_child_in_rect(child, rect);

		ofs_y = to_y_pos;
		valid_children_idx++;
	}

	queue_accessibility_update();
	queue_redraw();
	port_pos_dirty = true;
}

void GraphNode::draw_port(const Ref<GraphPort> p_port) {
	if (GDVIRTUAL_CALL(_draw_port, p_port)) {
		return;
	}

	Ref<Texture2D> port_icon = p_port->icon;

	Point2 icon_offset;
	if (port_icon.is_null()) {
		port_icon = theme_cache.port;
	}

	icon_offset = -port_icon->get_size() * 0.5;
	port_icon->draw(get_canvas_item(), p_port->position + icon_offset, p_port->color);
}

void GraphNode::_accessibility_action_port(const Variant &p_data) {
	CustomAccessibilityAction action = (CustomAccessibilityAction)p_data.operator int();
	if (!ports.has(selected_port)) {
		return;
	}
	Ref<GraphPort> port = get_port(selected_port, false);
	ERR_FAIL_COND(port.is_null());
	ERR_FAIL_COND(!port->get_enabled());
	GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
	ERR_FAIL_NULL(graph);
	switch (action) {
		case ACTION_CONNECT:
			if (graph->is_keyboard_connecting()) {
				graph->end_connecting(port, true);
			} else {
				graph->start_connecting(port, true);
			}
			queue_accessibility_update();
			queue_redraw();
			break;
		case ACTION_FOLLOW:
			GraphNode *target = graph->get_connection_target(port);
			if (target) {
				target->grab_focus();
			}
			break;
	}
}

void GraphNode::gui_input(const Ref<InputEvent> &p_event) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	if (p_event->is_pressed() && enabled_port_count > 0) {
		if (p_event->is_action("ui_up", true)) {
			selected_port--;
			if (selected_port < 0) {
				selected_port = -1;
			} else {
				accept_event();
			}
		} else if (p_event->is_action("ui_down", true)) {
			selected_port++;
			if (selected_port >= enabled_port_count) {
				selected_port = -1;
			} else {
				accept_event();
			}
		} else if (p_event->is_action("ui_cancel", true)) {
			GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
			if (graph && graph->is_keyboard_connecting()) {
				graph->force_connection_drag_end();
				accept_event();
			}
		} else if (p_event->is_action("ui_graph_delete", true)) {
			GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
			if (graph && graph->is_keyboard_connecting()) {
				graph->end_connecting(nullptr, false);
				accept_event();
			}
		} else if (p_event->is_action("ui_graph_follow_left", true) || p_event->is_action("ui_graph_follow_right", true)) {
			if (ports.has(selected_port)) {
				const Ref<GraphPort> port = get_port(selected_port, false);
				if (port.is_valid() && port->get_enabled()) {
					GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
					if (graph) {
						GraphNode *target = graph->get_connection_target(port);
						if (target) {
							target->grab_focus();
							accept_event();
						}
					}
				}
			}
		} else if (p_event->is_action("ui_left", true) || p_event->is_action("ui_right", true)) {
			if (ports.has(selected_port)) {
				const Ref<GraphPort> port = get_port(selected_port, false);
				if (port.is_valid() && port->get_enabled()) {
					GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
					if (graph) {
						if (graph->is_keyboard_connecting()) {
							graph->end_connecting(port, true);
						} else {
							graph->start_connecting(port, true);
						}
						accept_event();
					}
				}
			}
		} else if (p_event->is_action("ui_accept", true)) {
			if (ports.has(selected_port)) {
				Control *accessible_node = get_accessibility_node_by_port(selected_port);
				if (accessible_node) {
					selected_port = -1;
					accessible_node->grab_focus();
				}
				accept_event();
			}
		}
		queue_accessibility_update();
		queue_redraw();
	}
}

Control *GraphNode::get_accessibility_node_by_port(int p_port_idx) {
	return this;
}

void GraphNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			String name = get_accessibility_name();
			if (name.is_empty()) {
				name = get_name();
			}
			name = vformat(ETR("graph node %s (%s)"), name, get_title());

			if (ports.has(selected_port)) {
				GraphEdit *graph = Object::cast_to<GraphEdit>(get_parent());
				Dictionary type_info;
				if (graph) {
					type_info = graph->get_type_names();
				}
				const Ref<GraphPort> port = get_port(selected_port, false);
				if (port.is_valid()) {
					name += ", " + vformat(ETR("port %d of %d"), selected_port + 1, enabled_port_count);
					if (port->get_enabled()) {
						if (type_info.has(port->get_type())) {
							name += "," + vformat(ETR("type: %s"), type_info[port->get_type()]);
						} else {
							name += "," + vformat(ETR("type: %d"), port->get_type());
						}
						if (graph) {
							String cd = graph->get_connections_description(port);
							if (cd.is_empty()) {
								name += " " + ETR("no connections");
							} else {
								name += " " + cd;
							}
						}
					}
				}
				if (graph && graph->is_keyboard_connecting()) {
					name += ", " + ETR("currently selecting target port");
				}
			} else {
				name += ", " + vformat(ETR("has %d ports"), enabled_port_count);
			}
			DisplayServer::get_singleton()->accessibility_update_set_role(ae, DisplayServer::AccessibilityRole::ROLE_LIST);
			DisplayServer::get_singleton()->accessibility_update_set_name(ae, name);
			DisplayServer::get_singleton()->accessibility_update_add_custom_action(ae, CustomAccessibilityAction::ACTION_CONNECT, ETR("Edit Port Connection"));
			DisplayServer::get_singleton()->accessibility_update_add_custom_action(ae, CustomAccessibilityAction::ACTION_FOLLOW, ETR("Follow Port Connection"));
			DisplayServer::get_singleton()->accessibility_update_add_action(ae, DisplayServer::AccessibilityAction::ACTION_CUSTOM, callable_mp(this, &GraphNode::_accessibility_action_port));
		} break;
		case NOTIFICATION_FOCUS_EXIT: {
			selected_port = -1;
			queue_redraw();
		} break;
		case NOTIFICATION_DRAW: {
			// Used for layout calculations.
			Ref<StyleBox> sb_panel = theme_cache.panel;
			Ref<StyleBox> sb_titlebar = theme_cache.titlebar;
			// Used for drawing.
			Ref<StyleBox> sb_to_draw_panel = selected ? theme_cache.panel_selected : theme_cache.panel;
			Ref<StyleBox> sb_to_draw_titlebar = selected ? theme_cache.titlebar_selected : theme_cache.titlebar;

			Ref<StyleBox> sb_port_selected = theme_cache.port_selected;

			int port_h_offset = theme_cache.port_h_offset;

			Rect2 titlebar_rect(Point2(), titlebar_hbox->get_size() + sb_titlebar->get_minimum_size());
			Size2 body_size = get_size();
			titlebar_rect.size.width = body_size.width;
			body_size.height -= titlebar_rect.size.height;
			Rect2 body_rect(0, titlebar_rect.size.height, body_size.width, body_size.height);

			// Draw body (slots area) stylebox.
			draw_style_box(sb_to_draw_panel, body_rect);

			if (has_focus()) {
				draw_style_box(theme_cache.panel_focus, body_rect);
			}

			// Draw title bar stylebox above.
			draw_style_box(sb_to_draw_titlebar, titlebar_rect);

			int width = get_size().width - sb_panel->get_minimum_size().x;

			// Take ports into account
			if (port_pos_dirty) {
				_port_pos_update();
			}
			for (Ref<GraphPort> port : ports) {
				if (port.is_valid() && port->get_enabled()) {
					draw_port(port);
				}
			}
			if (selected_port >= 0) {
				Ref<GraphPort> port = get_port(selected_port, false);
				if (port.is_valid()) {
					Size2i port_sz = theme_cache.port->get_size();
					Vector2 port_pos = port->get_position();
					draw_style_box(sb_port_selected, Rect2i(port_pos.x + port_h_offset - port_sz.x, port_pos.y + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
					draw_style_box(sb_port_selected, Rect2i(port_pos.x + get_size().x - port_h_offset - port_sz.x, port_pos.y + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
				}
			}

			if (resizable) {
				draw_texture(theme_cache.resizer, get_size() - theme_cache.resizer->get_size(), theme_cache.resizer_color);
			}
		} break;
	}
}

void GraphNode::add_port(const Ref<GraphPort> p_port) {
	ports.append(p_port);

	if (p_port.is_valid()) {
		p_port->graph_node = this;
		p_port->connect("modified", callable_mp(this, &GraphNode::_port_modified));
	}

	_port_modified();

	emit_signal(SNAME("port_added"), p_port);
}

void GraphNode::insert_port(int p_port_index, const Ref<GraphPort> p_port, bool p_include_disabled) {
	ERR_FAIL_INDEX(p_port_index, get_port_count(p_include_disabled) + 1);

	int idx = p_port_index;
	if (!p_include_disabled) {
		idx = enabled_index_to_port_index(p_port_index);
		ERR_FAIL_COND_MSG(idx == -1, "port index out of bounds - fewer than p_port_index ports are enabled.");
	}

	ports.insert(idx, p_port);

	if (p_port.is_valid()) {
		p_port->graph_node = this;
		p_port->connect("modified", callable_mp(this, &GraphNode::_port_modified));
	}

	_port_modified();

	emit_signal(SNAME("port_added"), p_port);
}

void GraphNode::set_port(int p_port_index, const Ref<GraphPort> p_port, bool p_include_disabled) {
	ERR_FAIL_INDEX(p_port_index, get_port_count(p_include_disabled));

	int idx = p_port_index;
	if (!p_include_disabled) {
		idx = enabled_index_to_port_index(p_port_index);
		ERR_FAIL_COND_MSG(idx == -1, "port index out of bounds - fewer than p_port_index ports are enabled.");
	}

	Ref<GraphPort> old_port = ports[idx];
	ports.set(idx, p_port);

	if (old_port.is_valid()) {
		old_port->disconnect("modified", callable_mp(this, &GraphNode::_port_modified));
	}
	if (p_port.is_valid()) {
		p_port->graph_node = this;
		p_port->connect("modified", callable_mp(this, &GraphNode::_port_modified));
	}

	_port_modified();

	emit_signal(SNAME("port_replaced"), old_port, p_port);
}

void GraphNode::remove_port(int p_port_index, bool p_include_disabled) {
	ERR_FAIL_INDEX(p_port_index, get_port_count(p_include_disabled));

	int idx = p_port_index;
	if (!p_include_disabled) {
		idx = enabled_index_to_port_index(p_port_index);
		ERR_FAIL_COND_MSG(idx == -1, "port index out of bounds - fewer than p_port_index ports are enabled.");
	}

	Ref<GraphPort> old_port = ports[idx];
	ports.remove_at(idx);

	if (old_port.is_valid()) {
		old_port->disconnect("modified", callable_mp(this, &GraphNode::_port_modified));
	}

	_port_modified();

	emit_signal(SNAME("port_removed"), old_port);
}

void GraphNode::remove_all_ports() {
	if (ports.is_empty()) {
		return;
	}
	TypedArray<Ref<GraphPort>> old_ports = ports;
	for (int i = port_count - 1; i >= 0; i--) {
		remove_port(i);
	}

	emit_signal(SNAME("ports_cleared"), old_ports);
}

void GraphNode::_port_modified() {
	queue_accessibility_update();
	queue_redraw();
	_port_rebuild_cache();
	port_pos_dirty = true;
	notify_property_list_changed();
}

void GraphNode::set_ignore_invalid_connection_type(bool p_ignore) {
	ignore_invalid_connection_type = p_ignore;
}

bool GraphNode::is_ignoring_valid_connection_type() const {
	return ignore_invalid_connection_type;
}

Ref<GraphPort> GraphNode::get_port(int p_port_idx, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_port_idx, get_port_count(p_include_disabled), Ref<GraphPort>(nullptr));
	int idx = p_port_idx;
	if (!p_include_disabled) {
		int idx = enabled_index_to_port_index(p_port_idx);
		ERR_FAIL_COND_V_MSG(idx == -1, Ref<GraphPort>(nullptr), "port index out of bounds - fewer than p_port_index ports are enabled.");
	}
	return ports[idx];
}

Ref<GraphPort> GraphNode::get_filtered_port(int p_port_idx, GraphPort::PortDirection p_direction, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_port_idx, get_filtered_port_count(p_direction, p_include_disabled), Ref<GraphPort>(nullptr));
	int filtered_idx = 0;
	for (Ref<GraphPort> port : ports) {
		if (port.is_null() || (!port->get_enabled() && !p_include_disabled)) {
			continue;
		}
		if (filtered_idx == p_port_idx) {
			return port;
		}
		filtered_idx++;
	}
	ERR_FAIL_V_MSG(Ref<GraphPort>(nullptr), "filtered port index out of bounds - fewer than p_port_index ports with the given direction are enabled.");
}

void GraphNode::set_ports(Array p_ports) {
	_set_ports(p_ports);
}

Array GraphNode::get_ports() {
	return _get_ports();
}

int GraphNode::index_of_port(const Ref<GraphPort> p_port, bool p_include_disabled) {
	int idx = ports.find(p_port);
	if (p_include_disabled || idx < 0) {
		return idx;
	} else {
		PortCache _cache = port_cache[idx];
		return _cache.enabled_index;
	}
}

int GraphNode::filtered_index_of_port(const Ref<GraphPort> p_port, bool p_include_disabled) {
	int idx = index_of_port(p_port, p_include_disabled);
	PortCache _cache = port_cache[idx];
	return p_include_disabled ? _cache.filtered_index : _cache.filtered_enabled_index;
}

int GraphNode::get_port_count(bool p_include_disabled) {
	return p_include_disabled ? port_count : enabled_port_count;
}

int GraphNode::get_filtered_port_count(GraphPort::PortDirection p_filter_direction, bool p_include_disabled) {
	return p_include_disabled ? directed_port_count[p_filter_direction] : directed_enabled_port_count[p_filter_direction];
}

int GraphNode::enabled_index_to_port_index(int p_enabled_port_index) {
	int idx = 0;
	int enabled_idx = 0;
	for (Ref<GraphPort> port : ports) {
		if (port.is_null() || !port->get_enabled()) {
			idx++;
			continue;
		}
		if (enabled_idx == p_enabled_port_index) {
			return idx;
		}
		idx++;
		enabled_idx++;
	}
	return -1;
}

int GraphNode::port_index_to_enabled_index(int p_port_index) {
	if (p_port_index < 0 || p_port_index >= ports.size()) {
		return -1;
	}
	return port_cache[p_port_index].enabled_index;
}

Size2 GraphNode::get_minimum_size() const {
	Ref<StyleBox> sb_panel = theme_cache.panel;
	Ref<StyleBox> sb_titlebar = theme_cache.titlebar;

	int separation = theme_cache.separation;
	Size2 minsize = titlebar_hbox->get_minimum_size() + sb_titlebar->get_minimum_size();

	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false));
		if (!child) {
			continue;
		}

		Size2i size = child->get_combined_minimum_size();
		size.width += sb_panel->get_minimum_size().width;

		minsize.height += size.height;
		minsize.width = MAX(minsize.width, size.width);

		if (i > 0) {
			minsize.height += separation;
		}
	}

	minsize.height += sb_panel->get_minimum_size().height;

	return minsize;
}

void GraphNode::_port_rebuild_cache() {
	port_cache.clear();

	port_count = 0;
	enabled_port_count = 0;
	directed_port_count = { 0, 0, 0 };
	directed_enabled_port_count = { 0, 0, 0 };

	for (Ref<GraphPort> port : ports) {
		if (port.is_null()) {
			continue;
		}
		GraphPort::PortDirection _dir = port->get_direction();

		int dir_port_count = directed_port_count[_dir];
		int dir_enabled_port_count = directed_enabled_port_count[_dir];
		port_cache.push_back(PortCache(port, enabled_port_count, dir_port_count, dir_enabled_port_count));

		port_count++;
		directed_port_count.set(_dir, dir_port_count + 1);

		if (!port->get_enabled()) {
			continue;
		}

		enabled_port_count++;
		directed_enabled_port_count.set(_dir, dir_enabled_port_count + 1);
	}

	if (selected_port >= enabled_port_count) {
		selected_port = -1;
	}
}

void GraphNode::_port_pos_update() {
	int idx = 0;
	for (PortCache &p_cache : port_cache) {
		p_cache.pos = get_port(idx)->get_position();
		idx++;
	}
	port_pos_dirty = false;
}

Vector2 GraphNode::update_port_position(int p_port_idx, bool p_include_disabled) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, get_port_count(p_include_disabled), Vector2());
	return get_port(p_port_idx, p_include_disabled)->position;
}

String GraphNode::get_accessibility_container_name(const Node *p_node) const {
	int idx = 0;
	for (int i = 0; i < get_child_count(false); i++) {
		Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::IGNORE);
		if (!child) {
			continue;
		}
		if (child == p_node) {
			String name = get_accessibility_name();
			if (name.is_empty()) {
				name = get_name();
			}
			return vformat(ETR(", in slot %d of graph node %s (%s)"), idx + 1, name, get_title());
		}
		idx++;
	}
	return String();
}

void GraphNode::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	if (title_label) {
		title_label->set_text(title);
	}
	update_minimum_size();
}

String GraphNode::get_title() const {
	return title;
}

HBoxContainer *GraphNode::get_titlebar_hbox() {
	return titlebar_hbox;
}

Control::CursorShape GraphNode::get_cursor_shape(const Point2 &p_pos) const {
	if (resizable) {
		if (resizing || (p_pos.x > get_size().x - theme_cache.resizer->get_width() && p_pos.y > get_size().y - theme_cache.resizer->get_height())) {
			return CURSOR_FDIAGSIZE;
		}
	}

	return Control::get_cursor_shape(p_pos);
}

Vector<int> GraphNode::get_allowed_size_flags_horizontal() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

Vector<int> GraphNode::get_allowed_size_flags_vertical() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_EXPAND);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

void GraphNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &GraphNode::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &GraphNode::get_title);

	ClassDB::bind_method(D_METHOD("get_titlebar_hbox"), &GraphNode::get_titlebar_hbox);

	ClassDB::bind_method(D_METHOD("set_ports", "ports"), &GraphNode::_set_ports);
	ClassDB::bind_method(D_METHOD("get_ports"), &GraphNode::_get_ports);
	ClassDB::bind_method(D_METHOD("remove_all_ports"), &GraphNode::remove_all_ports);

	ClassDB::bind_method(D_METHOD("set_port", "port_index", "port", "include_disabled"), &GraphNode::set_port, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_port", "port_index", "include_disabled"), &GraphNode::get_port, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_port", "port", "include_disabled"), &GraphNode::add_port, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("insert_port", "port_index", "port", "include_disabled"), &GraphNode::insert_port, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_port", "port_index", "include_disabled"), &GraphNode::remove_port, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_port_count", "include_disabled"), &GraphNode::get_port_count, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_filtered_port_count", "filter_direction"), &GraphNode::get_filtered_port_count);
	ClassDB::bind_method(D_METHOD("index_of_port", "port", "include_disabled"), &GraphNode::index_of_port, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("enabled_index_to_port_index", "enabled_port_index"), &GraphNode::enabled_index_to_port_index);
	ClassDB::bind_method(D_METHOD("port_index_to_enabled_index", "port_index"), &GraphNode::port_index_to_enabled_index);

	ClassDB::bind_method(D_METHOD("set_ignore_invalid_connection_type", "ignore"), &GraphNode::set_ignore_invalid_connection_type);
	ClassDB::bind_method(D_METHOD("is_ignoring_valid_connection_type"), &GraphNode::is_ignoring_valid_connection_type);

	GDVIRTUAL_BIND(_draw_port, "port");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_invalid_connection_type"), "set_ignore_invalid_connection_type", "is_ignoring_valid_connection_type");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "ports", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("GraphPort")), "set_ports", "get_ports");

	/* GraphPort *default_port = memnew(GraphPort);
	base_property_helper.set_prefix("ports_");
	base_property_helper.set_array_length_getter(&GraphNode::get_port_count);
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort"), default_port, &GraphNode::set_port, &GraphNode::get_port);
	PropertyListHelper::register_base_helper(&base_property_helper);
	memdelete(default_port);*/

	ADD_SIGNAL(MethodInfo("port_added", PropertyInfo(Variant::OBJECT, "port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort")));
	ADD_SIGNAL(MethodInfo("port_removed", PropertyInfo(Variant::OBJECT, "port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort")));
	ADD_SIGNAL(MethodInfo("port_replaced", PropertyInfo(Variant::OBJECT, "old_port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort"), PropertyInfo(Variant::OBJECT, "new_port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort")));
	ADD_SIGNAL(MethodInfo("ports_cleared", PropertyInfo(Variant::ARRAY, "old_ports", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("GraphPort"))));

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, panel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, panel_selected);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, panel_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, titlebar);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, titlebar_selected);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, port_selected);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphNode, separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphNode, port_h_offset);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphNode, port);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphNode, resizer);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphNode, resizer_color);
}

GraphNode::GraphNode() {
	titlebar_hbox = memnew(HBoxContainer);
	titlebar_hbox->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(titlebar_hbox, false, INTERNAL_MODE_FRONT);

	title_label = memnew(Label);
	title_label->set_theme_type_variation("GraphNodeTitleLabel");
	title_label->set_h_size_flags(SIZE_EXPAND_FILL);
	title_label->set_focus_mode(Control::FOCUS_NONE);
	titlebar_hbox->add_child(title_label);

	set_mouse_filter(MOUSE_FILTER_STOP);
	set_focus_mode(FOCUS_ACCESSIBILITY);

	//property_helper.setup_for_instance(base_property_helper, this);
}
