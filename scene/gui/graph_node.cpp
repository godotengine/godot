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
	if (selected_port >= get_port_count()) {
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
	// TODO: Move to GraphNodeIndexed
	//slot_y_cache.clear();
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

		// TODO: Move to GraphNodeIndexed
		//slot_y_cache.push_back(child->get_rect().position.y + child->get_rect().size.height * 0.5);

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
	Ref<GraphPort> port = get_port(selected_port);
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

	int port_count = get_port_count();
	if (p_event->is_pressed() && port_count > 0) {
		if (p_event->is_action("ui_up", true)) {
			selected_port--;
			if (selected_port < 0) {
				selected_port = -1;
			} else {
				accept_event();
			}
		} else if (p_event->is_action("ui_down", true)) {
			selected_port++;
			if (selected_port >= port_count) {
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
				const Ref<GraphPort> port = get_port(selected_port);
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
				const Ref<GraphPort> port = get_port(selected_port);
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
				const Ref<GraphPort> port = get_port(selected_port);
				if (port.is_valid()) {
					name += ", " + vformat(ETR("port %d of %d"), selected_port + 1, get_port_count());
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
				name += ", " + vformat(ETR("has %d ports"), get_port_count());
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
			for (Ref<GraphPort> port : ports) {
				if (port.is_valid()) {
					draw_port(port);
				}
			}
			if (selected_port >= 0) {
				Ref<GraphPort> port = get_port(selected_port);
				if (port.is_valid()) {
					Size2i port_sz = theme_cache.port->get_size();
					Vector2 port_pos = port->get_position();
					draw_style_box(sb_port_selected, Rect2i(port_pos.x + port_h_offset - port_sz.x, port_pos.y + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
					draw_style_box(sb_port_selected, Rect2i(port_pos.x + get_size().x - port_h_offset - port_sz.x, port_pos.y + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
				}
			}
			// TODO: Move to GraphNodeIndexed
			/*
			// Take the HboxContainer child into account.
			if (get_child_count(false) > 0) {
				int slot_index = 0;
				for (const KeyValue<int, Slot> &E : slot_table) {
					if (E.key < 0 || E.key >= slot_y_cache.size()) {
						continue;
					}
					if (!slot_table.has(E.key)) {
						continue;
					}
					const Slot &slot = slot_table[E.key];

					// Left port.
					if (slot.enable_left) {
						draw_port(slot_index, Point2i(port_h_offset, slot_y_cache[E.key]), true, slot.color_left);
					}

					// Right port.
					if (slot.enable_right) {
						draw_port(slot_index, Point2i(get_size().x - port_h_offset, slot_y_cache[E.key]), false, slot.color_right);
					}

					if (slot_index == selected_slot) {
						Size2i port_sz = theme_cache.port->get_size();
						draw_style_box(sb_slot_selected, Rect2i(port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
						draw_style_box(sb_slot_selected, Rect2i(get_size().x - port_h_offset - port_sz.x, slot_y_cache[E.key] + sb_panel->get_margin(SIDE_TOP) - port_sz.y, port_sz.x * 2, port_sz.y * 2));
					}

					// Draw slot stylebox.
					if (slot.draw_stylebox) {
						Control *child = Object::cast_to<Control>(get_child(E.key, false));
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
			*/

			if (resizable) {
				draw_texture(theme_cache.resizer, get_size() - theme_cache.resizer->get_size(), theme_cache.resizer_color);
			}
		} break;
	}
}

void GraphNode::add_port(const Ref<GraphPort> p_port) {
	ports.append(p_port);

	queue_accessibility_update();
	queue_redraw();
	port_pos_dirty = true;
	notify_property_list_changed();

	emit_signal(SNAME("port_added"), p_port);
}

void GraphNode::set_port(int p_port_index, const Ref<GraphPort> p_port) {
	ERR_FAIL_INDEX(p_port_index, ports.size());

	Ref<GraphPort> old_port = ports[p_port_index];
	ports.set(p_port_index, p_port);

	queue_accessibility_update();
	queue_redraw();
	port_pos_dirty = true;
	notify_property_list_changed();

	emit_signal(SNAME("port_replaced"), old_port, p_port);
}

void GraphNode::remove_port(int p_port_index) {
	ERR_FAIL_INDEX(p_port_index, ports.size());

	Ref<GraphPort> old_port = ports[p_port_index];
	ports.remove_at(p_port_index);

	queue_accessibility_update();
	queue_redraw();
	port_pos_dirty = true;
	notify_property_list_changed();

	emit_signal(SNAME("port_removed"), old_port);
}

void GraphNode::remove_all_ports() {
	ports.clear();

	queue_accessibility_update();
	queue_redraw();
	port_pos_dirty = true;
	notify_property_list_changed();
}

void GraphNode::set_ignore_invalid_connection_type(bool p_ignore) {
	ignore_invalid_connection_type = p_ignore;
}

bool GraphNode::is_ignoring_valid_connection_type() const {
	return ignore_invalid_connection_type;
}

Ref<GraphPort> GraphNode::get_port(int p_port_idx) {
	ERR_FAIL_INDEX_V(p_port_idx, ports.size(), Ref<GraphPort>(nullptr));
	return ports[p_port_idx];
}

void GraphNode::set_ports(Array p_ports) {
	_set_ports(p_ports);
}

Array GraphNode::get_ports() {
	return _get_ports();
}

int GraphNode::index_of_port(const Ref<GraphPort> p_port) {
	return ports.find(p_port);
}

int GraphNode::get_port_count() {
	return ports.size();
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
		// TODO: Move to GraphNodeIndexed
		/*if (slot_table.has(i)) {
			size += slot_table[i].draw_stylebox ? sb_slot->get_minimum_size() : Size2();
		}*/

		minsize.height += size.height;
		minsize.width = MAX(minsize.width, size.width);

		if (i > 0) {
			minsize.height += separation;
		}
	}

	minsize.height += sb_panel->get_minimum_size().height;

	return minsize;
}

void GraphNode::_port_pos_update() {
	int edgeofs = theme_cache.port_h_offset;

	port_cache.clear();

	for (Ref<GraphPort> port : ports) {
		if (port.is_null()) {
			continue;
		}
		Vector2 pos = port->get_position();

		PortCache cached_port;
		cached_port.pos = Point2i(edgeofs + pos.x, pos.y);
		cached_port.type = port->get_type();
		cached_port.color = port->color;
		port_cache.push_back(cached_port);
	}
	if (selected_port >= get_port_count()) {
		selected_port = -1;
	}

	port_pos_dirty = false;
	// TODO: Move some of this to GraphNodeIndexed
	/*

		int slot_index = 0;

		for (int i = 0; i < get_child_count(false); i++) {
			Control *child = as_sortable_control(get_child(i, false), SortableVisibilityMode::IGNORE);
			if (!child) {
				continue;
			}

			Size2i size = child->get_rect().size;
			Point2 pos = child->get_position();

			if (slot_table.has(slot_index)) {
				if (slot_table[slot_index].enable_left) {
					PortCache port_cache;
					port_cache.pos = Point2i(edgeofs, pos.y + size.height / 2);
					port_cache.type = slot_table[slot_index].type_left;
					port_cache.color = slot_table[slot_index].color_left;
					port_cache.slot_index = slot_index;
					left_port_cache.push_back(port_cache);
				}
				if (slot_table[slot_index].enable_right) {
					PortCache port_cache;
					port_cache.pos = Point2i(get_size().width - edgeofs, pos.y + size.height / 2);
					port_cache.type = slot_table[slot_index].type_right;
					port_cache.color = slot_table[slot_index].color_right;
					port_cache.slot_index = slot_index;
					right_port_cache.push_back(port_cache);
				}
			}

			slot_index++;
		}
	*/
}

Vector2 GraphNode::update_port_position(int p_port_idx) {
	if (port_pos_dirty) {
		_port_pos_update();
	}

	ERR_FAIL_INDEX_V(p_port_idx, ports.size(), Vector2());
	Vector2 pos = get_port(p_port_idx)->position;
	return pos;
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

	ClassDB::bind_method(D_METHOD("set_port", "port_index", "port"), &GraphNode::set_port, DEFVAL(Ref<Texture2D>()), DEFVAL(Ref<Texture2D>()));
	ClassDB::bind_method(D_METHOD("set_ports", "ports"), &GraphNode::_set_ports);
	ClassDB::bind_method(D_METHOD("get_ports"), &GraphNode::_get_ports);
	ClassDB::bind_method(D_METHOD("remove_port", "port_index"), &GraphNode::remove_port);
	ClassDB::bind_method(D_METHOD("remove_all_ports"), &GraphNode::remove_all_ports);

	ClassDB::bind_method(D_METHOD("set_ignore_invalid_connection_type", "ignore"), &GraphNode::set_ignore_invalid_connection_type);
	ClassDB::bind_method(D_METHOD("is_ignoring_valid_connection_type"), &GraphNode::is_ignoring_valid_connection_type);

	GDVIRTUAL_BIND(_draw_port, "port")

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_invalid_connection_type"), "set_ignore_invalid_connection_type", "is_ignoring_valid_connection_type");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "ports", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("GraphPort")), "set_ports", "get_ports");

	//GraphPort default_port;
	//base_property_helper.set_prefix("ports_");
	//base_property_helper.set_array_length_getter(&GraphNode::get_port_count);
	//base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort"), default_port, &GraphNode::set_port, &GraphNode::get_port);
	//PropertyListHelper::register_base_helper(&base_property_helper);

	ADD_SIGNAL(MethodInfo("port_added", PropertyInfo(Variant::OBJECT, "port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort")));
	ADD_SIGNAL(MethodInfo("port_removed", PropertyInfo(Variant::OBJECT, "port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort")));
	ADD_SIGNAL(MethodInfo("port_replaced", PropertyInfo(Variant::OBJECT, "old_port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort"), PropertyInfo(Variant::OBJECT, "new_port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort")));

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
