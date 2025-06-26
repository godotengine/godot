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

bool GraphPort::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "enabled") {
		enabled = p_value;
	} else if (p_name == "type") {
		type = p_value;
	} else if (p_name == "icon") {
		icon = p_value;
	} else if (p_name == "color") {
		color = p_value;
	} else if (p_name == "direction") {
		direction = p_value;
	} else {
		return false;
	}

	return true;
}

bool GraphPort::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "enabled") {
		r_ret = enabled;
	} else if (p_name == "type") {
		r_ret = type;
	} else if (p_name == "icon") {
		r_ret = icon;
	} else if (p_name == "color") {
		r_ret = color;
	} else if (p_name == "direction") {
		r_ret = direction;
	} else {
		return false;
	}

	return true;
}

void GraphPort::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "enabled"));
	p_list->push_back(PropertyInfo(Variant::INT, "type"));
	p_list->push_back(PropertyInfo(Variant::COLOR, "color"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
}

void GraphPort::_mark_all_connections_dirty() {
	for (Ref<GraphConnection> conn : get_connections()) {
		conn->_cache.dirty = true;
	}
}

void GraphPort::set(bool p_enabled, bool p_exclusive, int p_type, Color p_color, PortDirection p_direction, Ref<Texture2D> p_icon = Ref<Texture2D>(nullptr)) {
	exclusive = p_exclusive;
	type = p_type;
	color = p_color;
	icon = p_icon;
	set_direction(p_direction);
	set_enabled(p_enabled);
}

int GraphPort::get_index() {
	return graph_node->index_of_port(this);
}

void GraphPort::enable() {
	enabled = true;
	_enabled();
}

void GraphPort::disable() {
	GraphEdit *graph = Object::cast_to<GraphEdit>(graph_node->get_parent());
	ERR_FAIL_NULL(graph);
	switch (on_disabled_behaviour) {
		case DisconnectBehaviour::MOVE_TO_PREVIOUS_PORT_OR_DISCONNECT: {
			int prev_port_idx = get_index() - 1;
			if (prev_port_idx >= 0) {
				Ref<GraphPort> prev_port = graph_node->get_port(prev_port_idx);
				if (prev_port.is_valid()) {
					graph->move_connections(this, prev_port);
					break;
				}
			}
			disconnect_all();
		} break;
		case DisconnectBehaviour::MOVE_TO_NEXT_PORT_OR_DISCONNECT: {
			int next_port_idx = get_index() + 1;
			if (next_port_idx < graph_node->get_port_count()) {
				Ref<GraphPort> next_port = graph_node->get_port(next_port_idx);
				if (next_port.is_valid()) {
					graph->move_connections(this, next_port);
					break;
				}
			}
			disconnect_all();
		} break;
		case DisconnectBehaviour::DISCONNECT_ALL:
		default: {
			disconnect_all();
		} break;
	}
	enabled = false;
	_disabled();
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

bool GraphPort::get_enabled() {
	return enabled;
}

void GraphPort::show() {
	set_visible(true);
}

void GraphPort::hide() {
	set_visible(false);
}

bool GraphPort::get_visible() {
	return visible;
}

void GraphPort::set_visible(bool p_visible) {
	if (visible == p_visible) {
		return;
	}
	visible = p_visible;
	_mark_all_connections_dirty();
}

int GraphPort::get_type() {
	return type;
}

void GraphPort::set_type(int p_type) {
	type = p_type;
}

GraphPort::PortDirection GraphPort::get_direction() {
	return direction;
}

void GraphPort::set_direction(PortDirection p_direction) {
	direction = p_direction;
	_changed_direction(p_direction);
}

GraphNode *GraphPort::get_graph_node() {
	return graph_node;
}

bool GraphPort::is_connected(const Ref<GraphPort> other) {
	GraphEdit *graph = Object::cast_to<GraphEdit>(graph_node->get_parent());
	if (!graph) {
		return false;
	}
	for (Ref<GraphConnection> connection : graph->get_connections_by_port(this)) {
		if (connection->get_other(this) == other) {
			return true;
		}
	}
	return false;
}

void GraphPort::disconnect_all() {
	GraphEdit *graph = Object::cast_to<GraphEdit>(graph_node->get_parent());
	if (graph) {
		graph->disconnect_all_by_port(this);
	}
}

Vector2 GraphPort::get_position() {
	return position;
}

TypedArray<Ref<GraphConnection>> GraphPort::get_connections() {
	GraphEdit *graph = Object::cast_to<GraphEdit>(graph_node->get_parent());
	if (!graph) {
		return;
	}
	return graph->get_connections_by_port(this);
}

void GraphPort::_enabled() {
	emit_signal(SNAME("enabled"));
}

void GraphPort::_disabled() {
	emit_signal(SNAME("disabled"));
}

void GraphPort::_connected(const Ref<GraphConnection> p_connection) {
	emit_signal(SNAME("connected"), p_connection);
}

void GraphPort::_disconnected(const Ref<GraphConnection> p_connection) {
	emit_signal(SNAME("disconnected"), p_connection);
}

void GraphPort::_changed_direction(const PortDirection p_direction) {
	emit_signal(SNAME("changed_direction"), p_direction);
}

void GraphPort::_bind_methods() {
	ADD_SIGNAL(MethodInfo("_enabled"));
	ADD_SIGNAL(MethodInfo("_disabled"));
	ADD_SIGNAL(MethodInfo("_connected"), PropertyInfo(Ref<GraphConnection>, "connection"));
	ADD_SIGNAL(MethodInfo("_disconnected"), PropertyInfo(Ref<GraphConnection>, "connection"));
	ADD_SIGNAL(MethodInfo("_changed_direction", PropertyInfo(Variant::INT, "direction")));
}

GraphPort::GraphPort(GraphNode *p_graph_node) {
	graph_node = p_graph_node;
	set(false, false, 0, Color(1, 1, 1, 1), GraphPort::PortDirection::UNDIRECTED);
}

GraphPort::GraphPort(GraphNode *p_graph_node, bool p_enabled, bool p_exclusive, int p_type, Color p_color, PortDirection p_direction, Ref<Texture2D> p_icon) {
	graph_node = p_graph_node;
	set(p_enabled, p_exclusive, p_type, p_color, p_direction, p_icon);
}
