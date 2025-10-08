/**************************************************************************/
/*  graph_port.cpp                                                        */
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

#include "graph_port.h"

#include "scene/gui/graph_connection.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/graph_node.h"
#include "scene/theme/theme_db.h"

void GraphPort::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Node *parent = get_parent();
			graph_edit = nullptr;
			graph_node = nullptr;
			while (parent) {
				if (!graph_node) {
					GraphNode *parent_node = cast_to<GraphNode>(parent);
					if (parent_node) {
						graph_node = parent_node;
					}
				}
				GraphEdit *parent_graph = cast_to<GraphEdit>(parent);
				if (parent_graph) {
					graph_edit = parent_graph;
					break;
				}
				parent = parent->get_parent();
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (graph_edit) {
				graph_edit->clear_port_connections(this);
			}
			graph_edit = nullptr;
			graph_node = nullptr;
		} break;
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			String name = get_accessibility_name();
			if (name.is_empty()) {
				name = get_name();
			}

			DisplayServer::get_singleton()->accessibility_update_set_role(ae, DisplayServer::AccessibilityRole::ROLE_UNKNOWN);
			DisplayServer::get_singleton()->accessibility_update_set_name(ae, name);
			DisplayServer::get_singleton()->accessibility_update_add_custom_action(ae, CustomAccessibilityAction::ACTION_CONNECT, ETR("Edit Port Connection"));
			DisplayServer::get_singleton()->accessibility_update_add_custom_action(ae, CustomAccessibilityAction::ACTION_FOLLOW, ETR("Follow Port Connection"));
			DisplayServer::get_singleton()->accessibility_update_add_action(ae, DisplayServer::AccessibilityAction::ACTION_CUSTOM, callable_mp(this, &GraphPort::_accessibility_action));
		} break;
		case NOTIFICATION_DRAW: {
			_draw();
		} break;
	}
}

void GraphPort::_accessibility_action(const Variant &p_data) {
	CustomAccessibilityAction action = (CustomAccessibilityAction)p_data.operator int();
	if (!graph_edit) {
		return;
	}
	ERR_FAIL_COND(!enabled);
	switch (action) {
		case ACTION_CONNECT:
			if (graph_edit->is_keyboard_connecting()) {
				graph_edit->end_connecting(this, true);
			} else {
				graph_edit->start_connecting(this, true);
			}
			queue_accessibility_update();
			queue_redraw();
			break;
		case ACTION_FOLLOW:
			GraphPort *target = graph_edit->get_connection_target(this, nav_conn_index);
			if (target) {
				target->grab_focus();
			}
			nav_conn_index++;
			if (nav_conn_index >= get_connection_count()) {
				nav_conn_index = 0;
			}
			break;
	}
}

void GraphPort::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (p_event->is_pressed() && is_enabled()) {
		if (graph_node) {
			GraphPort *nav_port = nullptr;
			if (p_event->is_action("ui_up", true)) {
				nav_port = graph_node->get_port_navigation(SIDE_TOP, this);
				accept_event();
			} else if (p_event->is_action("ui_down", true)) {
				nav_port = graph_node->get_port_navigation(SIDE_BOTTOM, this);
				accept_event();
			} else if (p_event->is_action("ui_left", true)) {
				nav_port = graph_node->get_port_navigation(SIDE_LEFT, this);
				accept_event();
			} else if (p_event->is_action("ui_right", true)) {
				nav_port = graph_node->get_port_navigation(SIDE_RIGHT, this);
				accept_event();
			}
			if (nav_port) {
				nav_port->grab_focus();
			}
		}
		if (p_event->is_action("ui_graph_follow_left", true) || p_event->is_action("ui_graph_follow_right", true)) {
			if (graph_edit && has_connection()) {
				GraphPort *target = graph_edit->get_connection_target(this, nav_conn_index);
				if (target) {
					target->grab_focus();
				}
				accept_event();
				nav_conn_index++;
				if (nav_conn_index >= get_connection_count()) {
					nav_conn_index = 0;
				}
			}
		} else if (p_event->is_action("ui_accept", true)) {
			if (graph_edit) {
				if (graph_edit->is_keyboard_connecting()) {
					graph_edit->end_connecting(this, true);
				} else {
					graph_edit->start_connecting(this, true);
				}
				accept_event();
			}
		}
	}
}

void GraphPort::set_properties(bool p_enabled, bool p_exclusive, int p_type, PortDirection p_direction) {
	exclusive = p_exclusive;
	set_port_type(p_type);
	set_direction(p_direction);
	set_enabled(p_enabled);
}

void GraphPort::enable() {
	enabled = true;
	set_focus_mode(FOCUS_ALL);
	set_mouse_filter(MOUSE_FILTER_STOP);

	queue_redraw();

	_on_enabled();
	_on_modified();
}

void GraphPort::disable() {
	enabled = false;
	set_focus_mode(FOCUS_NONE);
	set_mouse_filter(MOUSE_FILTER_IGNORE);

	if (graph_edit) {
		switch (on_disabled_behaviour) {
			case GraphPort::DisconnectBehaviour::MOVE_TO_PREVIOUS_PORT_OR_DISCONNECT: {
				if (!graph_node) {
					WARN_PRINT("Port is not assigned to a graph node, should not use DisconnectBehaviour::MOVE_TO_PREVIOUS_PORT_OR_DISCONNECT");
					break;
				}
				GraphPort *prev_port = graph_node->get_previous_matching_port(this, false);
				if (prev_port) {
					graph_edit->move_connections(this, prev_port);
				}
			} break;
			case GraphPort::DisconnectBehaviour::MOVE_TO_NEXT_PORT_OR_DISCONNECT: {
				if (!graph_node) {
					WARN_PRINT("Port is not assigned to a graph node, should not use DisconnectBehaviour::MOVE_TO_NEXT_PORT_OR_DISCONNECT");
					break;
				}
				GraphPort *next_port = graph_node->get_next_matching_port(this, false);
				if (next_port) {
					graph_edit->move_connections(this, next_port);
				}
			} break;
			case GraphPort::DisconnectBehaviour::DISCONNECT_ALL:
			default:
				break;
		}
		graph_edit->clear_port_connections(this);
	}

	queue_redraw();

	_on_disabled();
	_on_modified();
}

void GraphPort::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	if (p_enabled) {
		enable();
	} else {
		disable();
	}
}

bool GraphPort::is_enabled() const {
	return enabled;
}

int GraphPort::get_port_type() const {
	return port_type;
}

void GraphPort::set_port_type(int p_type) {
	port_type = p_type;

	queue_redraw();

	_on_changed_type(p_type);
	_on_modified();
}

Color GraphPort::get_color() const {
	Color base_col = selected ? theme_cache.selected_color : theme_cache.color;
	if (!graph_edit) {
		return base_col;
	}
	const TypedDictionary<int, Color> &graph_colors = graph_edit->get_type_colors();
	return (port_type > 0 && port_type < graph_colors.size()) ? Color(graph_colors[port_type]) : base_col;
}

Color GraphPort::get_rim_color() {
	return selected ? theme_cache.selected_rim_color : theme_cache.rim_color;
}

Ref<Texture2D> GraphPort::get_icon() const {
	return theme_cache.icon;
}

bool GraphPort::get_exclusive() const {
	return exclusive;
}

void GraphPort::set_exclusive(bool p_exclusive) {
	exclusive = p_exclusive;

	queue_redraw();

	_on_modified();
}

bool GraphPort::is_implying_direction() const {
	return imply_direction;
}

void GraphPort::set_imply_direction(bool p_imply_direction) {
	if (imply_direction == p_imply_direction) {
		return;
	}

	imply_direction = p_imply_direction;

	direction = UNDIRECTED;
	if (graph_edit && has_connection()) {
		for (const Ref<GraphConnection> conn : get_connections()) {
			if (_try_imply_direction(conn)) {
				break;
			}
		}
	}

	notify_property_list_changed();
	_on_modified();
}

GraphPort::PortDirection GraphPort::get_direction() const {
	return direction;
}

void GraphPort::set_direction(const GraphPort::PortDirection p_direction) {
	direction = p_direction;

	queue_redraw();

	_on_changed_direction(p_direction);
	_on_modified();
}

GraphPort::DisconnectBehaviour GraphPort::get_disabled_behaviour() const {
	return on_disabled_behaviour;
}

void GraphPort::set_disabled_behaviour(const GraphPort::DisconnectBehaviour p_disconnect_behaviour) {
	on_disabled_behaviour = p_disconnect_behaviour;
	notify_property_list_changed();
	_on_modified();
}

GraphNode *GraphPort::get_graph_node() const {
	return graph_node;
}

Rect2 GraphPort::get_hotzone() const {
	Vector2 pos = get_position() + get_connection_point();

	Ref<Texture2D> icon = theme_cache.icon;
	Vector2 icon_size = Vector2(0.0, 0.0);
	if (icon.is_valid()) {
		icon_size = Vector2(icon->get_width(), icon->get_height());
	}

	Vector2 theme_extent;
	switch (direction) {
		case GraphPort::PortDirection::INPUT:
			theme_extent.x = theme_cache.hotzone_extent_h_input;
			theme_extent.y = theme_cache.hotzone_extent_v_input;
			break;
		case GraphPort::PortDirection::OUTPUT:
			theme_extent.x = theme_cache.hotzone_extent_h_output;
			theme_extent.y = theme_cache.hotzone_extent_v_output;
			break;
		case GraphPort::PortDirection::UNDIRECTED:
		default:
			theme_extent.x = theme_cache.hotzone_extent_h_undirected;
			theme_extent.y = theme_cache.hotzone_extent_v_undirected;
			break;
	}
	Vector2 size = icon_size.max(theme_extent);

	Vector2 offset = Vector2(theme_cache.hotzone_offset_h, theme_cache.hotzone_offset_v);

	return Rect2(pos.x + offset.x - size.x / 2, pos.y + offset.y - size.y / 2, size.x, size.y);
}

int GraphPort::get_port_index(bool p_include_disabled) const {
	return p_include_disabled ? _index : _enabled_index;
}

int GraphPort::get_filtered_port_index(bool p_include_disabled) const {
	return p_include_disabled ? _filtered_index : _filtered_enabled_index;
}

void GraphPort::add_connection(Ref<GraphConnection> p_connection) {
	ERR_FAIL_NULL(graph_edit);
	ERR_FAIL_COND_MSG(p_connection->first_port != this && p_connection->second_port != this, "Failed to add GraphConnection to GraphNode: neither connection port is part of the GraphNode!");
	graph_edit->add_connection(p_connection);
}

void GraphPort::remove_connection(Ref<GraphConnection> p_connection) {
	ERR_FAIL_NULL(graph_edit);
	ERR_FAIL_COND_MSG(p_connection->first_port != this && p_connection->second_port != this, "Failed to add GraphConnection to GraphNode: neither connection port is part of the GraphNode!");
	graph_edit->remove_connection(p_connection);
}

void GraphPort::connect_to_port(GraphPort *p_port, bool p_clear_if_invalid) {
	ERR_FAIL_NULL(graph_edit);
	graph_edit->connect_nodes(this, p_port, p_clear_if_invalid);
}

bool GraphPort::has_connection() const {
	ERR_FAIL_NULL_V(graph_edit, false);
	return graph_edit->is_port_connected(this);
}

TypedArray<Ref<GraphConnection>> GraphPort::get_connections() const {
	ERR_FAIL_NULL_V(graph_edit, TypedArray<Ref<GraphConnection>>());
	return graph_edit->get_connections_by_port(this);
}

void GraphPort::set_connections(const TypedArray<Ref<GraphConnection>> &p_connections) {
	ERR_FAIL_NULL(graph_edit);
	graph_edit->set_port_connections(this, p_connections);
}

void GraphPort::clear_connections() {
	ERR_FAIL_NULL(graph_edit);
	graph_edit->clear_port_connections(this);
}

int GraphPort::get_connection_count() const {
	ERR_FAIL_NULL_V(graph_edit, 0);
	return graph_edit->get_connection_count_by_port(this);
}

Ref<GraphConnection> GraphPort::get_first_connection() const {
	ERR_FAIL_NULL_V(graph_edit, Ref<GraphConnection>(nullptr));
	return graph_edit->get_first_connection_by_port(this);
}

bool GraphPort::is_connected_to(const GraphPort *p_port) const {
	ERR_FAIL_NULL_V(graph_edit, false);
	return graph_edit->are_ports_connected(this, p_port);
}

TypedArray<GraphPort> GraphPort::get_connected_ports() const {
	ERR_FAIL_NULL_V(graph_edit, TypedArray<GraphPort>());
	return graph_edit->get_connected_ports(this);
}

GraphPort *GraphPort::get_first_connected_port() const {
	ERR_FAIL_NULL_V(graph_edit, nullptr);
	return graph_edit->get_first_connected_port(this);
}

GraphNode *GraphPort::get_first_connected_node() const {
	ERR_FAIL_NULL_V(graph_edit, nullptr);
	return graph_edit->get_first_connected_node(this);
}

Vector2 GraphPort::get_connection_point() const {
	return theme_cache.icon.is_valid() ? theme_cache.icon->get_size() / 2 : Vector2(0, 0);
}

int GraphPort::get_connection_angle() const {
	switch (direction) {
		case GraphPort::INPUT:
			return theme_cache.connection_angle_input;
		case GraphPort::OUTPUT:
			return theme_cache.connection_angle_output;
		case GraphPort::UNDIRECTED:
		default:
			return theme_cache.connection_angle_undirected;
	}
}

Size2 GraphPort::get_minimum_size() const {
	return theme_cache.icon.is_valid() ? theme_cache.icon->get_size() : Size2(0, 0);
}

void GraphPort::_draw() {
	if (!enabled) {
		return;
	}

	Ref<Texture2D> port_icon = theme_cache.icon;
	if (port_icon.is_null()) {
		return;
	}

	Vector2 pos = graph_node ? get_position_offset() : Vector2(0, 0);

	Size2 port_icon_size = port_icon->get_size();

	// Draw "shadow"/outline in the connection rim color.
	Color rim_color = get_rim_color();
	int s = theme_cache.rim_size;
	if (rim_color.a > 0 && s > 0) {
		draw_texture_rect(port_icon, Rect2(pos - Size2(s, s), port_icon_size + Size2(s * 2, s * 2)), false, rim_color);
	}

	// Focus box
	if (has_focus()) {
		const RID ci = get_canvas_item();
		const Size2 size = get_size();

		Ref<StyleBox> panel_focus = theme_cache.panel_focus;
		if (panel_focus.is_valid()) {
			panel_focus->draw(ci, Rect2i(pos, size));
		}
	}

	port_icon->draw(get_canvas_item(), pos, get_color());
}

void GraphPort::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "direction") {
		p_property.usage = imply_direction ? PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_READ_ONLY : PROPERTY_USAGE_DEFAULT;
	}
}

void GraphPort::_on_enabled() {
	emit_signal(SNAME("on_enabled"), this);
}
void GraphPort::_on_disabled() {
	emit_signal(SNAME("on_disabled"), this);
}
void GraphPort::_on_changed_direction(const PortDirection p_direction) {
	emit_signal(SNAME("changed_direction"), p_direction);
}
void GraphPort::_on_changed_type(const int p_type) {
	emit_signal(SNAME("changed_type"), p_type);
}
void GraphPort::_on_modified() {
	emit_signal(SNAME("modified"));
}
void GraphPort::_on_connected(const Ref<GraphConnection> p_conn) {
	_try_imply_direction(p_conn);
	emit_signal(SNAME("connected"), p_conn);
}
void GraphPort::_on_disconnected(const Ref<GraphConnection> p_conn) {
	if (imply_direction && graph_edit && !has_connection()) {
		direction = GraphPort::PortDirection::UNDIRECTED;
	}
	emit_signal(SNAME("disconnected"), p_conn);
}

void GraphPort::_bind_methods() {
	ClassDB::bind_method(D_METHOD("enable"), &GraphPort::enable);
	ClassDB::bind_method(D_METHOD("disable"), &GraphPort::disable);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &GraphPort::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &GraphPort::is_enabled);

	ClassDB::bind_method(D_METHOD("set_port_type", "type"), &GraphPort::set_port_type);
	ClassDB::bind_method(D_METHOD("get_port_type"), &GraphPort::get_port_type);

	ClassDB::bind_method(D_METHOD("set_exclusive", "exclusive"), &GraphPort::set_exclusive);
	ClassDB::bind_method(D_METHOD("get_exclusive"), &GraphPort::get_exclusive);

	ClassDB::bind_method(D_METHOD("get_color"), &GraphPort::get_color);
	ClassDB::bind_method(D_METHOD("get_icon"), &GraphPort::get_icon);
	ClassDB::bind_method(D_METHOD("get_graph_node"), &GraphPort::get_graph_node);

	ClassDB::bind_method(D_METHOD("get_connection_point"), &GraphPort::get_connection_point);
	ClassDB::bind_method(D_METHOD("get_connection_angle"), &GraphPort::get_connection_angle);

	ClassDB::bind_method(D_METHOD("set_direction", "direction"), &GraphPort::set_direction);
	ClassDB::bind_method(D_METHOD("get_direction"), &GraphPort::get_direction);
	ClassDB::bind_method(D_METHOD("set_imply_direction", "imply"), &GraphPort::set_imply_direction);
	ClassDB::bind_method(D_METHOD("is_implying_direction"), &GraphPort::is_implying_direction);

	ClassDB::bind_method(D_METHOD("set_disabled_behaviour", "on_disabled_behaviour"), &GraphPort::set_disabled_behaviour);
	ClassDB::bind_method(D_METHOD("get_disabled_behaviour"), &GraphPort::get_disabled_behaviour);

	ClassDB::bind_method(D_METHOD("get_connections"), &GraphPort::get_connections);
	ClassDB::bind_method(D_METHOD("set_connections", "connections"), &GraphPort::set_connections);
	ClassDB::bind_method(D_METHOD("clear_connections"), &GraphPort::clear_connections);
	ClassDB::bind_method(D_METHOD("get_connection_count"), &GraphPort::get_connection_count);

	ClassDB::bind_method(D_METHOD("add_connection", "connection"), &GraphPort::add_connection);
	ClassDB::bind_method(D_METHOD("connect_to_port", "port", "clear_if_invalid"), &GraphPort::connect_to_port, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("remove_connection", "connection"), &GraphPort::remove_connection);
	ClassDB::bind_method(D_METHOD("has_connection"), &GraphPort::has_connection);
	ClassDB::bind_method(D_METHOD("get_first_connection"), &GraphPort::get_first_connection);
	ClassDB::bind_method(D_METHOD("is_connected_to", "port"), &GraphPort::is_connected_to);

	ClassDB::bind_method(D_METHOD("get_connected_ports"), &GraphPort::get_connected_ports);
	ClassDB::bind_method(D_METHOD("get_first_connected_port"), &GraphPort::get_first_connected_port);
	ClassDB::bind_method(D_METHOD("get_first_connected_node"), &GraphPort::get_first_connected_node);

	ClassDB::bind_method(D_METHOD("get_port_index", "include_disabled"), &GraphPort::get_port_index, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_filtered_port_index", "include_disabled"), &GraphPort::get_filtered_port_index, DEFVAL(true));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "port_type"), "set_port_type", "get_port_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Input,Output,Undirected", PROPERTY_USAGE_DEFAULT), "set_direction", "get_direction");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclusive"), "set_exclusive", "get_exclusive");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "imply_direction"), "set_imply_direction", "is_implying_direction");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "on_disabled_behaviour", PROPERTY_HINT_ENUM, "Disconnect all,Move to previous port or disconnect,Move to next port or disconnect"), "set_disabled_behaviour", "get_disabled_behaviour");

	ADD_SIGNAL(MethodInfo("on_enabled"));
	ADD_SIGNAL(MethodInfo("on_disabled"));
	ADD_SIGNAL(MethodInfo("changed_direction", PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Input,Output,Undirected")));
	ADD_SIGNAL(MethodInfo("changed_type", PropertyInfo(Variant::INT, "type")));
	ADD_SIGNAL(MethodInfo("modified"));
	ADD_SIGNAL(MethodInfo("connected", PropertyInfo(Variant::OBJECT, "connection", PROPERTY_HINT_RESOURCE_TYPE, "GraphConnection")));
	ADD_SIGNAL(MethodInfo("disconnected", PropertyInfo(Variant::OBJECT, "connection", PROPERTY_HINT_RESOURCE_TYPE, "GraphConnection")));

	BIND_ENUM_CONSTANT(INPUT);
	BIND_ENUM_CONSTANT(OUTPUT);
	BIND_ENUM_CONSTANT(UNDIRECTED);

	BIND_ENUM_CONSTANT(DISCONNECT_ALL);
	BIND_ENUM_CONSTANT(MOVE_TO_PREVIOUS_PORT_OR_DISCONNECT);
	BIND_ENUM_CONSTANT(MOVE_TO_NEXT_PORT_OR_DISCONNECT);

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphPort, panel_focus);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, GraphPort, icon);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, rim_size);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphPort, color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphPort, selected_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphPort, rim_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, GraphPort, selected_rim_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_extent_h_input);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_extent_v_input);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_extent_h_output);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_extent_v_output);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_extent_h_undirected);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_extent_v_undirected);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_offset_h);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, hotzone_offset_v);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, connection_angle_input);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, connection_angle_output);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, GraphPort, connection_angle_undirected);
}

bool GraphPort::_try_imply_direction(const Ref<GraphConnection> p_conn) {
	if (p_conn.is_null() || !imply_direction || direction != GraphPort::PortDirection::UNDIRECTED) {
		return false;
	}
	GraphPort *other_port = p_conn->get_other_port(this);
	if (!other_port) {
		return false;
	}
	if (other_port->direction != GraphPort::PortDirection::UNDIRECTED) {
		set_direction(other_port->direction == GraphPort::PortDirection::INPUT ? GraphPort::PortDirection::OUTPUT : GraphPort::PortDirection::INPUT);
	} else {
		set_direction(this == p_conn->first_port ? GraphPort::PortDirection::OUTPUT : GraphPort::PortDirection::INPUT);
	}
	other_port->_try_imply_direction(p_conn);
	return true;
}

GraphPort::GraphPort() {
	set_focus_mode(FOCUS_NONE);
	set_mouse_filter(MOUSE_FILTER_IGNORE);
}

GraphPort::GraphPort(bool p_enabled, bool p_exclusive, int p_type, PortDirection p_direction) {
	enabled = p_enabled;
	exclusive = p_exclusive;
	port_type = p_type;
	direction = p_direction;

	set_focus_mode(p_enabled ? FOCUS_ALL : FOCUS_NONE);
	set_mouse_filter(p_enabled ? MOUSE_FILTER_STOP : MOUSE_FILTER_IGNORE);
}
