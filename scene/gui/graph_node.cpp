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
#include "scene/gui/graph_connection.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/label.h"
#include "scene/theme/theme_db.h"

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
		if (!child || child->has_meta(ignore_node_meta_tag)) {
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
			if (!child || child->has_meta(ignore_node_meta_tag)) {
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
		if (!child || child->has_meta(ignore_node_meta_tag)) {
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

void GraphNode::_accessibility_action_port(const Variant &p_data) {
	CustomAccessibilityAction action = (CustomAccessibilityAction)p_data.operator int();
	if (!selected_port) {
		return;
	}
	ERR_FAIL_COND(!selected_port->is_enabled());
	GraphEdit *graph = cast_to<GraphEdit>(get_parent());
	ERR_FAIL_NULL(graph);
	switch (action) {
		case ACTION_CONNECT:
			if (graph->is_keyboard_connecting()) {
				graph->end_connecting(selected_port, true);
			} else {
				graph->start_connecting(selected_port, true);
			}
			queue_accessibility_update();
			queue_redraw();
			break;
		case ACTION_FOLLOW:
			GraphNode *target = graph->get_connection_target(selected_port);
			if (target) {
				target->grab_focus();
			}
			break;
	}
}

void GraphNode::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (port_pos_dirty) {
		_update_port_positions();
	}

	if (p_event->is_pressed() && enabled_port_count > 0) {
		bool ac_enabled = get_tree() && get_tree()->is_accessibility_enabled();
		if (selected_port && ((ac_enabled && selected_port->get_focus_mode() == Control::FOCUS_ACCESSIBILITY) || selected_port->get_focus_mode() == Control::FOCUS_ALL)) {
			if (p_event->is_action("ui_up", true)) {
				selected_port = get_previous_matching_port(selected_port);
				if (selected_port) {
					accept_event();
				}
			} else if (p_event->is_action("ui_down", true)) {
				selected_port = get_next_matching_port(selected_port);
				if (selected_port) {
					accept_event();
				}
			}
		}
		if (p_event->is_action("ui_cancel", true)) {
			GraphEdit *graph = cast_to<GraphEdit>(get_parent());
			if (graph && graph->is_keyboard_connecting()) {
				graph->force_connection_drag_end();
				accept_event();
			}
		} else if (p_event->is_action("ui_graph_delete", true)) {
			GraphEdit *graph = cast_to<GraphEdit>(get_parent());
			if (graph && graph->is_keyboard_connecting()) {
				graph->end_connecting(nullptr, false);
				accept_event();
			}
		} else if (p_event->is_action("ui_graph_follow_left", true) || p_event->is_action("ui_graph_follow_right", true)) {
			if (selected_port && selected_port->is_enabled()) {
				GraphEdit *graph = cast_to<GraphEdit>(get_parent());
				if (graph) {
					GraphNode *target = graph->get_connection_target(selected_port);
					if (target) {
						target->grab_focus();
						accept_event();
					}
				}
			}
		} else if (p_event->is_action("ui_left", true) || p_event->is_action("ui_right", true)) {
			if (selected_port && selected_port->is_enabled()) {
				GraphEdit *graph = cast_to<GraphEdit>(get_parent());
				if (graph) {
					if (graph->is_keyboard_connecting()) {
						graph->end_connecting(selected_port, true);
					} else {
						graph->start_connecting(selected_port, true);
					}
					accept_event();
				}
			}
		} else if (p_event->is_action("ui_accept", true)) {
			if (selected_port) {
				if (selected_port->graph_node) {
					selected_port->graph_node->grab_focus();
					selected_port = nullptr;
				}
				accept_event();
			}
		}
		queue_accessibility_update();
		queue_redraw();
	}

	GraphElement::gui_input(p_event);
}

const Control *GraphNode::get_accessibility_node_by_port(int p_port_idx) const {
	return this;
}

void GraphNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			graph_edit = cast_to<GraphEdit>(get_parent());
		} break;
		case NOTIFICATION_EXIT_TREE: {
			graph_edit = nullptr;
		} break;
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			String name = get_accessibility_name();
			if (name.is_empty()) {
				name = get_name();
			}
			name = vformat(ETR("graph node %s (%s)"), name, get_title());

			if (selected_port) {
				GraphEdit *graph = cast_to<GraphEdit>(get_parent());
				Dictionary type_info;
				if (graph) {
					type_info = graph->get_type_names();
				}
				name += ", " + vformat(ETR("port %d of %d"), selected_port->get_port_index() + 1, enabled_port_count);
				if (selected_port->is_enabled()) {
					if (type_info.has(selected_port->get_port_type())) {
						name += "," + vformat(ETR("type: %s"), type_info[selected_port->get_port_type()]);
					} else {
						name += "," + vformat(ETR("type: %d"), selected_port->get_port_type());
					}
					if (graph) {
						String cd = graph->get_connections_description(selected_port);
						if (cd.is_empty()) {
							name += " " + ETR("no connections");
						} else {
							name += " " + cd;
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
			selected_port = nullptr;
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

			// Draw title bar stylebox above.
			draw_style_box(sb_to_draw_titlebar, titlebar_rect);

			//int width = get_size().width - sb_panel->get_minimum_size().x;

			// Take ports into account
			if (selected_port) {
				Size2i port_sz = selected_port->get_size();
				Vector2 port_pos = selected_port->get_position();
				draw_style_box(sb_port_selected, Rect2i(port_pos.x + port_h_offset - port_sz.x, port_pos.y + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
				draw_style_box(sb_port_selected, Rect2i(port_pos.x + get_size().x - port_h_offset - port_sz.x, port_pos.y + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
			}

			if (resizable) {
				draw_texture(theme_cache.resizer, get_size() - theme_cache.resizer->get_size(), theme_cache.resizer_color);
			}
		} break;
	}
}

void GraphNode::_set_ports(const Vector<GraphPort *> &p_ports) {
	_remove_all_ports();
	for (GraphPort *port : p_ports) {
		_add_port(port);
	}
}

const Vector<GraphPort *> &GraphNode::_get_ports() {
	return ports;
}

void GraphNode::_add_port(GraphPort *p_port) {
	ports.push_back(p_port);
	if (p_port) {
		p_port->graph_node = this;
		if (!p_port->is_connected("modified", modified_callable)) {
			p_port->connect("modified", modified_callable);
		}
		if (!p_port->is_connected("connected", connected_callable)) {
			p_port->connect("connected", connected_callable);
		}
		if (!p_port->is_connected("disconnected", disconnected_callable)) {
			p_port->connect("disconnected", disconnected_callable);
		}
	}
}

void GraphNode::_insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled) {
	ERR_FAIL_INDEX(p_port_index, get_port_count(p_include_disabled) + 1);

	int idx = p_port_index;
	if (!p_include_disabled) {
		idx = enabled_index_to_port_index(p_port_index);
		ERR_FAIL_COND_MSG(idx == -1, "port index out of bounds - fewer than p_port_index ports are enabled.");
	}

	ports.insert(idx, p_port);

	if (p_port) {
		p_port->graph_node = this;
		if (!p_port->is_connected("modified", modified_callable)) {
			p_port->connect("modified", modified_callable);
		}
		if (!p_port->is_connected("connected", connected_callable)) {
			p_port->connect("connected", connected_callable);
		}
		if (!p_port->is_connected("disconnected", disconnected_callable)) {
			p_port->connect("disconnected", disconnected_callable);
		}
	}
}

GraphPort *GraphNode::_remove_port(int p_port_index, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_port_index, get_port_count(p_include_disabled), nullptr);

	int idx = p_port_index;
	if (!p_include_disabled) {
		idx = enabled_index_to_port_index(p_port_index);
		ERR_FAIL_COND_V_MSG(idx == -1, nullptr, "port index out of bounds - fewer than p_port_index ports are enabled.");
	}

	GraphPort *old_port = ports[idx];
	ports.remove_at(idx);

	if (old_port) {
		if (old_port->is_connected("modified", modified_callable)) {
			old_port->disconnect("modified", modified_callable);
		}
		if (old_port->is_connected("connected", connected_callable)) {
			old_port->disconnect("connected", connected_callable);
		}
		if (old_port->is_connected("disconnected", disconnected_callable)) {
			old_port->disconnect("disconnected", disconnected_callable);
		}
	}

	return old_port;
}

GraphPort *GraphNode::_set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled) {
	ERR_FAIL_INDEX_V(p_port_index, get_port_count(p_include_disabled), nullptr);

	int idx = p_port_index;
	if (!p_include_disabled) {
		idx = enabled_index_to_port_index(p_port_index);
		ERR_FAIL_COND_V_MSG(idx == -1, nullptr, "port index out of bounds - fewer than p_port_index ports are enabled.");
	}

	GraphPort *old_port = ports[idx];
	if (old_port == p_port) {
		return p_port;
	}
	ports.set(idx, p_port);

	if (old_port) {
		if (old_port->is_connected("modified", modified_callable)) {
			old_port->disconnect("modified", modified_callable);
		}
		if (old_port->is_connected("connected", connected_callable)) {
			old_port->disconnect("connected", connected_callable);
		}
		if (old_port->is_connected("disconnected", disconnected_callable)) {
			old_port->disconnect("disconnected", disconnected_callable);
		}
	}
	if (p_port) {
		p_port->graph_node = this;
		if (!p_port->is_connected("modified", modified_callable)) {
			p_port->connect("modified", modified_callable);
		}
		if (!p_port->is_connected("connected", connected_callable)) {
			p_port->connect("connected", connected_callable);
		}
		if (!p_port->is_connected("disconnected", disconnected_callable)) {
			p_port->connect("disconnected", disconnected_callable);
		}
	}

	return old_port;
}

void GraphNode::_remove_all_ports() {
	if (ports.is_empty()) {
		return;
	}
	for (int i = port_count - 1; i >= 0; i--) {
		_remove_port(i);
	}
}

void GraphNode::_port_modified() {
	queue_accessibility_update();
	queue_redraw();
	_port_rebuild_cache();
	port_pos_dirty = true;
	callable_mp(this, &GraphNode::_queue_update_port_positions).call_deferred();
	notify_property_list_changed();
	emit_signal(SNAME("ports_updated"), this);
}

void GraphNode::add_port(GraphPort *p_port) {
	_add_port(p_port);
	_port_modified();
}

void GraphNode::insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled) {
	_insert_port(p_port_index, p_port, p_include_disabled);
	_port_modified();
}

GraphPort *GraphNode::set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled) {
	GraphPort *ret = _set_port(p_port_index, p_port, p_include_disabled);
	if (ret != p_port) {
		_port_modified();
	}
	return ret;
}

GraphPort *GraphNode::remove_port(int p_port_index, bool p_include_disabled) {
	GraphPort *ret = _remove_port(p_port_index, p_include_disabled);
	_port_modified();
	return ret;
}

void GraphNode::remove_all_ports() {
	_remove_all_ports();
	_port_modified();
}

void GraphNode::set_ignore_invalid_connection_type(bool p_ignore) {
	ignore_invalid_connection_type = p_ignore;
}

bool GraphNode::is_ignoring_valid_connection_type() const {
	return ignore_invalid_connection_type;
}

GraphPort *GraphNode::get_port(int p_port_idx, bool p_include_disabled) const {
	ERR_FAIL_INDEX_V(p_port_idx, get_port_count(p_include_disabled), nullptr);
	int idx = p_port_idx;
	if (!p_include_disabled) {
		idx = enabled_index_to_port_index(p_port_idx);
		ERR_FAIL_COND_V_MSG(idx == -1, nullptr, "port index out of bounds - fewer than p_port_index ports are enabled.");
	}
	return ports[idx];
}

GraphPort *GraphNode::get_filtered_port(int p_port_idx, GraphPort::PortDirection p_direction, bool p_include_disabled) const {
	ERR_FAIL_INDEX_V(p_port_idx, get_filtered_port_count(p_direction, p_include_disabled), nullptr);
	int filtered_idx = 0;
	for (GraphPort *port : ports) {
		if (!port || (!port->is_enabled() && !p_include_disabled) || port->direction != p_direction) {
			continue;
		}
		if (filtered_idx == p_port_idx) {
			return port;
		}
		filtered_idx++;
	}
	ERR_FAIL_V_MSG(nullptr, "filtered port index out of bounds - fewer than p_port_index ports with the given direction are enabled.");
}

GraphPort *GraphNode::get_input_port(int p_port_index, bool p_include_disabled) const {
	return get_filtered_port(p_port_index, GraphPort::PortDirection::INPUT, p_include_disabled);
}

GraphPort *GraphNode::get_output_port(int p_port_index, bool p_include_disabled) const {
	return get_filtered_port(p_port_index, GraphPort::PortDirection::OUTPUT, p_include_disabled);
}

GraphPort *GraphNode::get_next_matching_port(const GraphPort *p_port, bool p_include_disabled) const {
	ERR_FAIL_NULL_V(p_port, nullptr);
	int filtered_selected_port_idx = filtered_index_of_port(p_port, false);
	if (filtered_selected_port_idx + 1 >= get_filtered_port_count(p_port->direction, false)) {
		return nullptr;
	}
	return get_filtered_port(filtered_selected_port_idx + 1, p_port->direction, false);
}

GraphPort *GraphNode::get_previous_matching_port(const GraphPort *p_port, bool p_include_disabled) const {
	ERR_FAIL_NULL_V(p_port, nullptr);
	int filtered_selected_port_idx = filtered_index_of_port(p_port, false);
	if (filtered_selected_port_idx - 1 < 0) {
		return nullptr;
	}
	return get_filtered_port(filtered_selected_port_idx - 1, p_port->direction, false);
}

void GraphNode::set_ports(const TypedArray<GraphPort> &p_ports) {
	_remove_all_ports();
	for (Variant p : p_ports) {
		GraphPort *port = cast_to<GraphPort>(p);
		_add_port(port);
	}
	_port_modified();
}

TypedArray<GraphPort> GraphNode::get_ports() const {
	TypedArray<GraphPort> _ports;
	for (GraphPort *port : ports) {
		_ports.append(port);
	}
	return _ports;
}

TypedArray<GraphPort> GraphNode::get_filtered_ports(GraphPort::PortDirection p_direction, bool p_include_disabled) const {
	TypedArray<GraphPort> _ports;
	for (GraphPort *port : ports) {
		if (!port || port->direction != p_direction) {
			continue;
		}
		_ports.append(port);
	}
	return _ports;
}

TypedArray<GraphPort> GraphNode::get_input_ports(bool p_include_disabled) const {
	return get_filtered_ports(GraphPort::PortDirection::INPUT, p_include_disabled);
}

TypedArray<GraphPort> GraphNode::get_output_ports(bool p_include_disabled) const {
	return get_filtered_ports(GraphPort::PortDirection::OUTPUT, p_include_disabled);
}

int GraphNode::index_of_port(const GraphPort *p_port, bool p_include_disabled) const {
	ERR_FAIL_NULL_V(p_port, -1);
	return p_port->get_port_index(p_include_disabled);
}

int GraphNode::filtered_index_of_port(const GraphPort *p_port, bool p_include_disabled) const {
	ERR_FAIL_NULL_V(p_port, -1);
	return p_port->get_filtered_port_index(p_include_disabled);
}

int GraphNode::get_port_count(bool p_include_disabled) const {
	return p_include_disabled ? port_count : enabled_port_count;
}

int GraphNode::get_filtered_port_count(GraphPort::PortDirection p_filter_direction, bool p_include_disabled) const {
	return p_include_disabled ? directed_port_count[p_filter_direction] : directed_enabled_port_count[p_filter_direction];
}

int GraphNode::get_input_port_count(bool p_include_disabled) const {
	return get_filtered_port_count(GraphPort::PortDirection::INPUT, p_include_disabled);
}

int GraphNode::get_output_port_count(bool p_include_disabled) const {
	return get_filtered_port_count(GraphPort::PortDirection::OUTPUT, p_include_disabled);
}

int GraphNode::enabled_index_to_port_index(int p_enabled_port_index) const {
	ERR_FAIL_INDEX_V(p_enabled_port_index, enabled_port_count, -1);
	int idx = 0;
	for (GraphPort *port : ports) {
		if (!port || !port->is_enabled()) {
			idx++;
			continue;
		}
		if (port->_enabled_index == p_enabled_port_index) {
			return idx;
		}
		idx++;
	}
	return -1;
}

int GraphNode::port_index_to_enabled_index(int p_port_index) const {
	ERR_FAIL_INDEX_V(p_port_index, ports.size(), -1);
	GraphPort *port = get_port(p_port_index, true);
	ERR_FAIL_NULL_V(port, -1);
	return port->_enabled_index;
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
	port_count = 0;
	enabled_port_count = 0;
	directed_port_count = { 0, 0, 0 };
	directed_enabled_port_count = { 0, 0, 0 };

	for (GraphPort *port : ports) {
		if (!port) {
			continue;
		}
		GraphPort::PortDirection _dir = port->get_direction();

		int dir_port_count = directed_port_count[_dir];
		int dir_enabled_port_count = directed_enabled_port_count[_dir];

		port->_index = port_count;
		port->_enabled_index = enabled_port_count;
		port->_filtered_index = dir_port_count;
		port->_filtered_enabled_index = dir_enabled_port_count;

		port_count++;
		directed_port_count.set(_dir, dir_port_count + 1);

		if (!port->is_enabled()) {
			continue;
		}

		enabled_port_count++;
		directed_enabled_port_count.set(_dir, dir_enabled_port_count + 1);
	}
}

void GraphNode::_update_port_positions() {
	updating_port_pos = false;
	port_pos_dirty = false;
}

void GraphNode::_queue_update_port_positions() {
	ERR_THREAD_GUARD

	if (updating_port_pos) {
		return;
	}

	updating_port_pos = true;

	callable_mp(this, &GraphNode::_update_port_positions).call_deferred();
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

HBoxContainer *GraphNode::get_titlebar_hbox() const {
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

void GraphNode::add_connection(Ref<GraphConnection> p_connection) {
	ERR_FAIL_NULL(graph_edit);
	ERR_FAIL_COND_MSG(p_connection->get_first_node() != this && p_connection->get_second_node() != this, "Failed to add GraphConnection to GraphNode: neither connection port is part of the GraphNode!");
	graph_edit->add_connection(p_connection);
}

void GraphNode::remove_connection(Ref<GraphConnection> p_connection) {
	ERR_FAIL_NULL(graph_edit);
	ERR_FAIL_COND_MSG(p_connection->get_first_node() != this && p_connection->get_second_node() != this, "Failed to remove GraphConnection from GraphNode: neither connection port is part of the GraphNode!");
	graph_edit->remove_connection(p_connection);
}

bool GraphNode::is_connected_to(GraphNode *p_node) const {
	ERR_FAIL_NULL_V(graph_edit, false);
	return graph_edit->are_nodes_connected(this, p_node);
}

bool GraphNode::has_connection() const {
	ERR_FAIL_NULL_V(graph_edit, false);
	return graph_edit->is_node_connected(this);
}

TypedArray<Ref<GraphConnection>> GraphNode::get_connections() const {
	ERR_FAIL_NULL_V(graph_edit, TypedArray<Ref<GraphConnection>>());
	return graph_edit->get_connections_by_node(this);
}

TypedArray<Ref<GraphConnection>> GraphNode::get_filtered_connections(GraphPort::PortDirection p_filter_direction) const {
	ERR_FAIL_NULL_V(graph_edit, TypedArray<Ref<GraphConnection>>());
	return graph_edit->get_filtered_connections_by_node(this, p_filter_direction);
}

TypedArray<GraphNode> GraphNode::get_connected_nodes() const {
	const TypedArray<Ref<GraphConnection>> conns = get_connections();
	TypedArray<GraphNode> ret;
	for (const Ref<GraphConnection> conn : conns) {
		if (conn.is_null()) {
			continue;
		}
		ret.push_back(conn->get_other_node(this));
	}
	return ret;
}

TypedArray<GraphNode> GraphNode::get_filtered_connected_nodes(GraphPort::PortDirection p_filter_direction) const {
	const TypedArray<Ref<GraphConnection>> conns = get_filtered_connections(p_filter_direction);
	TypedArray<GraphNode> ret;
	for (const Ref<GraphConnection> conn : conns) {
		if (conn.is_null()) {
			continue;
		}
		ret.push_back(conn->get_other_node(this));
	}
	return ret;
}

TypedArray<GraphNode> GraphNode::get_input_connected_nodes() const {
	return get_filtered_connected_nodes(GraphPort::PortDirection::INPUT);
}

TypedArray<GraphNode> GraphNode::get_output_connected_nodes() const {
	return get_filtered_connected_nodes(GraphPort::PortDirection::OUTPUT);
}

TypedArray<Ref<GraphConnection>> GraphNode::get_input_connections() const {
	return get_filtered_connections(GraphPort::INPUT);
}

TypedArray<Ref<GraphConnection>> GraphNode::get_output_connections() const {
	return get_filtered_connections(GraphPort::OUTPUT);
}

void GraphNode::set_connections(const TypedArray<Ref<GraphConnection>> &p_connections) {
	ERR_FAIL_NULL(graph_edit);
	graph_edit->set_node_connections(this, p_connections);
}

void GraphNode::clear_connections() {
	ERR_FAIL_NULL(graph_edit);
	graph_edit->clear_node_connections(this);
}

void GraphNode::clear_filtered_connections(GraphPort::PortDirection p_filter_direction) {
	ERR_FAIL_NULL(graph_edit);
	for (GraphPort *port : ports) {
		if (!port || port->direction != p_filter_direction) {
			continue;
		}
		graph_edit->clear_port_connections(port);
	}
}

void GraphNode::clear_input_connections() {
	clear_filtered_connections(GraphPort::INPUT);
}

void GraphNode::clear_output_connections() {
	clear_filtered_connections(GraphPort::OUTPUT);
}

void GraphNode::_on_connected(const Ref<GraphConnection> p_conn) {
	emit_signal(SNAME("connected"), p_conn);
}
void GraphNode::_on_disconnected(const Ref<GraphConnection> p_conn) {
	emit_signal(SNAME("disconnected"), p_conn);
}

void GraphNode::_on_replacing_by(Node *new_node) {
	for (GraphPort *port : ports) {
		if (!port) {
			continue;
		}
		if (port->is_connected("modified", modified_callable)) {
			port->disconnect("modified", modified_callable);
		}
		if (port->is_connected("connected", connected_callable)) {
			port->disconnect("connected", connected_callable);
		}
		if (port->is_connected("disconnected", disconnected_callable)) {
			port->disconnect("disconnected", disconnected_callable);
		}
	}
}

void GraphNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &GraphNode::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &GraphNode::get_title);

	ClassDB::bind_method(D_METHOD("get_titlebar_hbox"), &GraphNode::get_titlebar_hbox);

	ClassDB::bind_method(D_METHOD("set_ports", "ports"), &GraphNode::set_ports);
	ClassDB::bind_method(D_METHOD("get_ports"), &GraphNode::get_ports);
	ClassDB::bind_method(D_METHOD("get_filtered_ports", "filter_direction", "include_disabled"), &GraphNode::get_filtered_ports, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_input_ports", "include_disabled"), &GraphNodeIndexed::get_input_ports, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_output_ports", "include_disabled"), &GraphNodeIndexed::get_output_ports, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("remove_all_ports"), &GraphNode::remove_all_ports);

	ClassDB::bind_method(D_METHOD("set_port", "port_index", "port", "include_disabled"), &GraphNode::set_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_port", "port_index", "include_disabled"), &GraphNode::get_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_filtered_port", "port_index", "filter_direction", "include_disabled"), &GraphNode::get_filtered_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_input_port", "port_index", "include_disabled"), &GraphNode::get_input_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_output_port", "port_index", "include_disabled"), &GraphNode::get_output_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_next_matching_port", "port", "include_disabled"), &GraphNode::get_next_matching_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_previous_matching_port", "port", "include_disabled"), &GraphNode::get_previous_matching_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("add_port", "port"), &GraphNode::add_port);
	ClassDB::bind_method(D_METHOD("insert_port", "port_index", "port", "include_disabled"), &GraphNode::insert_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("remove_port", "port_index", "include_disabled"), &GraphNode::remove_port, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("get_port_count", "include_disabled"), &GraphNode::get_port_count, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_filtered_port_count", "filter_direction", "include_disabled"), &GraphNode::get_filtered_port_count, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_input_port_count", "include_disabled"), &GraphNode::get_input_port_count, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_output_port_count", "include_disabled"), &GraphNode::get_output_port_count, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("index_of_port", "port", "include_disabled"), &GraphNode::index_of_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("filtered_index_of_port", "port", "include_disabled"), &GraphNode::filtered_index_of_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("enabled_index_to_port_index", "enabled_port_index"), &GraphNode::enabled_index_to_port_index);
	ClassDB::bind_method(D_METHOD("port_index_to_enabled_index", "port_index"), &GraphNode::port_index_to_enabled_index);

	ClassDB::bind_method(D_METHOD("get_connections"), &GraphNode::get_connections);
	ClassDB::bind_method(D_METHOD("get_filtered_connections", "filter_direction"), &GraphNode::get_filtered_connections);
	ClassDB::bind_method(D_METHOD("get_input_connections"), &GraphNode::get_input_connections);
	ClassDB::bind_method(D_METHOD("get_output_connections"), &GraphNode::get_output_connections);
	ClassDB::bind_method(D_METHOD("set_connections", "connections"), &GraphNode::set_connections);
	ClassDB::bind_method(D_METHOD("clear_connections"), &GraphNode::clear_connections);
	ClassDB::bind_method(D_METHOD("clear_filtered_connections", "filter_direction"), &GraphNode::clear_filtered_connections);
	ClassDB::bind_method(D_METHOD("clear_input_connections"), &GraphNode::clear_input_connections);
	ClassDB::bind_method(D_METHOD("clear_output_connections"), &GraphNode::clear_output_connections);

	ClassDB::bind_method(D_METHOD("add_connection", "connection"), &GraphNode::add_connection);
	ClassDB::bind_method(D_METHOD("remove_connection", "connection"), &GraphNode::remove_connection);
	ClassDB::bind_method(D_METHOD("has_connection"), &GraphNode::has_connection);
	ClassDB::bind_method(D_METHOD("is_connected_to", "node"), &GraphNode::is_connected_to);

	ClassDB::bind_method(D_METHOD("get_connected_nodes"), &GraphNode::get_connected_nodes);
	ClassDB::bind_method(D_METHOD("get_filtered_connected_nodes", "filter_direction"), &GraphNode::get_filtered_connected_nodes);
	ClassDB::bind_method(D_METHOD("get_input_connected_nodes"), &GraphNode::get_input_connected_nodes);
	ClassDB::bind_method(D_METHOD("get_output_connected_nodes"), &GraphNode::get_output_connected_nodes);

	ClassDB::bind_method(D_METHOD("set_ignore_invalid_connection_type", "ignore"), &GraphNode::set_ignore_invalid_connection_type);
	ClassDB::bind_method(D_METHOD("is_ignoring_valid_connection_type"), &GraphNode::is_ignoring_valid_connection_type);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_invalid_connection_type"), "set_ignore_invalid_connection_type", "is_ignoring_valid_connection_type");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "ports", PROPERTY_HINT_TYPE_STRING, MAKE_NODE_TYPE_HINT("GraphPort")), "set_ports", "get_ports");

	ADD_SIGNAL(MethodInfo("connected", PropertyInfo(Variant::OBJECT, "connection", PROPERTY_HINT_RESOURCE_TYPE, "GraphConnection")));
	ADD_SIGNAL(MethodInfo("disconnected", PropertyInfo(Variant::OBJECT, "connection", PROPERTY_HINT_RESOURCE_TYPE, "GraphConnection")));
	ADD_SIGNAL(MethodInfo("ports_updated", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "GraphNode")));

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, panel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, panel_selected);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, panel_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, titlebar);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphNode, titlebar_selected);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphNode, separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphNode, port_h_offset);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphNode, resizer);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphNode, resizer_color);
}

GraphNode::GraphNode() {
	modified_callable = callable_mp(this, &GraphNode::_port_modified);
	connected_callable = callable_mp(this, &GraphNode::_on_connected);
	disconnected_callable = callable_mp(this, &GraphNode::_on_disconnected);

	titlebar_hbox = memnew(HBoxContainer);
	titlebar_hbox->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(titlebar_hbox, false, INTERNAL_MODE_FRONT);

	title_label = memnew(Label);
	title_label->set_theme_type_variation("GraphNodeTitleLabel");
	title_label->set_h_size_flags(SIZE_EXPAND_FILL);
	titlebar_hbox->add_child(title_label);

	connect("replacing_by", callable_mp(this, &GraphNodeIndexed::_on_replacing_by));

	set_mouse_filter(MOUSE_FILTER_STOP);
	set_focus_mode(FOCUS_ACCESSIBILITY);
}
