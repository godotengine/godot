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
		case NOTIFICATION_DRAW: {
			_draw();
		} break;
	}
}

void GraphPort::set_properties(bool p_enabled, bool p_exclusive, int p_type, PortDirection p_direction) {
	exclusive = p_exclusive;
	set_type(p_type);
	set_direction(p_direction);
	set_enabled(p_enabled);
}

void GraphPort::enable() {
	enabled = true;

	queue_redraw();
	notify_property_list_changed();

	_on_enabled();
	_on_modified();
}

void GraphPort::disable() {
	enabled = false;

	ERR_FAIL_NULL(graph_edit);
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

	queue_redraw();
	notify_property_list_changed();

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

int GraphPort::get_type() const {
	return type;
}

void GraphPort::set_type(int p_type) {
	type = p_type;

	queue_redraw();
	notify_property_list_changed();

	_on_changed_type(p_type);
	_on_modified();
}

Color GraphPort::get_color() const {
	Color base_col = selected ? theme_cache.selected_color : theme_cache.color;
	if (!graph_edit) {
		return base_col;
	}
	const TypedArray<Color> &graph_colors = graph_edit->get_type_colors();
	return (type > 0 && type < graph_colors.size()) ? Color(graph_colors[type]) : base_col;
}

Color GraphPort::get_rim_color() {
	return selected ? theme_cache.selected_rim_color : theme_cache.rim_color;
}

bool GraphPort::get_exclusive() const {
	return exclusive;
}

void GraphPort::set_exclusive(bool p_exclusive) {
	exclusive = p_exclusive;

	queue_redraw();
	notify_property_list_changed();

	_on_modified();
}

GraphPort::PortDirection GraphPort::get_direction() const {
	return direction;
}

void GraphPort::set_direction(GraphPort::PortDirection p_direction) {
	direction = p_direction;

	queue_redraw();
	notify_property_list_changed();

	_on_changed_direction(p_direction);
	_on_modified();
}

GraphPort::DisconnectBehaviour GraphPort::get_disabled_behaviour() const {
	return on_disabled_behaviour;
}

void GraphPort::set_disabled_behaviour(GraphPort::DisconnectBehaviour p_disconnect_behaviour) {
	on_disabled_behaviour = p_disconnect_behaviour;
	notify_property_list_changed();
	_on_modified();
}

GraphNode *GraphPort::get_graph_node() const {
	return graph_node;
}

Rect2 GraphPort::get_hotzone() const {
	Vector2 pos = get_position();

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
	return Rect2(pos.x + theme_cache.hotzone_offset_h - size.x / 2, pos.y + theme_cache.hotzone_offset_v - size.y / 2, size.x, size.y);
}

int GraphPort::get_port_index(bool p_include_disabled) const {
	return p_include_disabled ? _index : _enabled_index;
}

int GraphPort::get_filtered_port_index(bool p_include_disabled) const {
	return p_include_disabled ? _filtered_index : _filtered_enabled_index;
}

void GraphPort::disconnect_all() {
	ERR_FAIL_NULL(graph_edit);
	graph_edit->clear_port_connections(this);
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
	const Ref<GraphConnection> first_conn = get_first_connection();
	if (first_conn.is_null()) {
		return nullptr;
	}
	return first_conn->get_other_port(this);
}

GraphNode *GraphPort::get_first_connected_node() const {
	GraphPort *other_port = get_first_connected_port();
	if (!other_port) {
		return nullptr;
	}
	return other_port->graph_node;
}

void GraphPort::_draw() {
	if (!enabled) {
		return;
	}

	Ref<Texture2D> port_icon = theme_cache.icon;
	if (port_icon.is_null()) {
		return;
	}

	Size2 port_icon_size = port_icon->get_size();
	Point2 icon_offset = -port_icon_size * 0.5;

	// Draw "shadow"/outline in the connection rim color.
	Color rim_color = get_rim_color();
	int s = theme_cache.rim_size;
	if (rim_color.a > 0 && s > 0) {
		draw_texture_rect(port_icon, Rect2(get_position_offset() + icon_offset - Size2(s, s), port_icon_size + Size2(s * 2, s * 2)), false, rim_color);
	}
	port_icon->draw(get_canvas_item(), get_position_offset() + icon_offset, get_color());
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
	emit_signal(SNAME("connected"), p_conn);
}
void GraphPort::_on_disconnected(const Ref<GraphConnection> p_conn) {
	emit_signal(SNAME("disconnected"), p_conn);
}

void GraphPort::_bind_methods() {
	ClassDB::bind_method(D_METHOD("enable"), &GraphPort::enable);
	ClassDB::bind_method(D_METHOD("disable"), &GraphPort::disable);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &GraphPort::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &GraphPort::is_enabled);

	ClassDB::bind_method(D_METHOD("set_type", "type"), &GraphPort::set_type);
	ClassDB::bind_method(D_METHOD("get_type"), &GraphPort::get_type);

	ClassDB::bind_method(D_METHOD("set_exclusive", "exclusive"), &GraphPort::set_exclusive);
	ClassDB::bind_method(D_METHOD("get_exclusive"), &GraphPort::get_exclusive);

	ClassDB::bind_method(D_METHOD("get_color"), &GraphPort::get_color);
	ClassDB::bind_method(D_METHOD("get_graph_node"), &GraphPort::get_graph_node);

	ClassDB::bind_method(D_METHOD("set_direction", "direction"), &GraphPort::set_direction);
	ClassDB::bind_method(D_METHOD("get_direction"), &GraphPort::get_direction);

	ClassDB::bind_method(D_METHOD("set_disabled_behaviour", "on_disabled_behaviour"), &GraphPort::set_disabled_behaviour);
	ClassDB::bind_method(D_METHOD("get_disabled_behaviour"), &GraphPort::get_disabled_behaviour);

	ClassDB::bind_method(D_METHOD("get_connections"), &GraphPort::get_connections);
	ClassDB::bind_method(D_METHOD("disconnect_all"), &GraphPort::disconnect_all);
	ClassDB::bind_method(D_METHOD("set_connections", "connections"), &GraphPort::set_connections);
	ClassDB::bind_method(D_METHOD("clear_connections"), &GraphPort::clear_connections);

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
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Input,Output,Undirected"), "set_direction", "get_direction");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclusive"), "set_exclusive", "get_exclusive");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "on_disabled_behaviour", PROPERTY_HINT_ENUM, "Disconnect all,Move to previous port or disconnect,Move to next port or disconnect"), "set_disabled_behaviour", "get_disabled_behaviour");

	ADD_SIGNAL(MethodInfo("on_enabled"));
	ADD_SIGNAL(MethodInfo("on_disabled"));
	ADD_SIGNAL(MethodInfo("changed_direction", PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Input,Output,Undirected")));
	ADD_SIGNAL(MethodInfo("changed_type", PropertyInfo(Variant::INT, "type")));
	ADD_SIGNAL(MethodInfo("modified"));
	ADD_SIGNAL(MethodInfo("connected", PropertyInfo(Variant::OBJECT, "connection", PROPERTY_HINT_RESOURCE_TYPE, "GraphConnection")));
	ADD_SIGNAL(MethodInfo("disconnected", PropertyInfo(Variant::OBJECT, "connection", PROPERTY_HINT_RESOURCE_TYPE, "GraphConnection")));

	BIND_ENUM_CONSTANT(INPUT)
	BIND_ENUM_CONSTANT(OUTPUT)
	BIND_ENUM_CONSTANT(UNDIRECTED)

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphPort, panel);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, GraphPort, panel_selected);
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
}

GraphPort::GraphPort() {
}

GraphPort::GraphPort(bool p_enabled, bool p_exclusive, int p_type, PortDirection p_direction) {
	enabled = p_enabled;
	exclusive = p_exclusive;
	type = p_type;
	direction = p_direction;
}
