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
/*
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
	} else if (p_name == "exclusive") {
		exclusive = p_value;
	} else if (p_name == "icon") {
		icon = p_value;
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
	} else if (p_name == "exclusive") {
		r_ret = exclusive;
	} else if (p_name == "icon") {
		r_ret = icon;
	} else {
		return false;
	}

	return true;
}

void GraphPort::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::BOOL, "enabled"));
	p_list->push_back(PropertyInfo(Variant::INT, "type"));
	p_list->push_back(PropertyInfo(Variant::COLOR, "color"));
	p_list->push_back(PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Input,Output,Undirected"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "exclusive"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_STORE_IF_NULL));
	//p_list->push_back(PropertyInfo(Variant::INT, "on_disabled_behaviour", PROPERTY_HINT_ENUM, "Disconnect all,Move to previous port or disconnect,Move to next port or disconnect"));
}*/

void GraphPort::set_properties(bool p_enabled, bool p_exclusive, int p_type, Color p_color, PortDirection p_direction, Ref<Texture2D> p_icon) {
	exclusive = p_exclusive;
	color = p_color;
	icon = p_icon;
	set_type(p_type);
	set_direction(p_direction);
	set_enabled(p_enabled);
	notify_property_list_changed();
}

void GraphPort::enable() {
	enabled = true;
	notify_property_list_changed();
	_enabled();
	_modified();
}

void GraphPort::disable() {
	enabled = false;
	notify_property_list_changed();
	_disabled();
	_modified();
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

int GraphPort::get_type() {
	return type;
}

void GraphPort::set_type(int p_type) {
	type = p_type;
	notify_property_list_changed();
	_changed_type(p_type);
	_modified();
}

Color GraphPort::get_color() {
	return color;
}

void GraphPort::set_color(Color p_color) {
	color = p_color;
	notify_property_list_changed();
	_modified();
}

bool GraphPort::get_exclusive() {
	return exclusive;
}

void GraphPort::set_exclusive(bool p_exclusive) {
	exclusive = p_exclusive;
	notify_property_list_changed();
	_modified();
}

Ref<Texture2D> GraphPort::get_icon() {
	return icon;
}

void GraphPort::set_icon(Ref<Texture2D> p_icon) {
	icon = p_icon;
	notify_property_list_changed();
	_modified();
}

GraphPort::PortDirection GraphPort::get_direction() const {
	return direction;
}

void GraphPort::set_direction(GraphPort::PortDirection p_direction) {
	direction = p_direction;
	notify_property_list_changed();
	_changed_direction(p_direction);
	_modified();
}

void GraphPort::set_position(const Vector2 p_position) {
	position = p_position;
	_modified();
}

Vector2 GraphPort::get_position() {
	return position;
}

GraphNode *GraphPort::get_graph_node() {
	return graph_node;
}

void GraphPort::_enabled() {
	emit_signal(SNAME("enabled"), this);
}

void GraphPort::_disabled() {
	emit_signal(SNAME("disabled"), this);
}

void GraphPort::_changed_direction(const PortDirection p_direction) {
	emit_signal(SNAME("changed_direction"), p_direction);
}
void GraphPort::_changed_type(const int p_type) {
	emit_signal(SNAME("changed_type"), p_type);
}
void GraphPort::_modified() {
	emit_signal(SNAME("modified"));
}

void GraphPort::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_position", "position"), &GraphPort::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &GraphPort::get_position);

	ClassDB::bind_method(D_METHOD("enable"), &GraphPort::enable);
	ClassDB::bind_method(D_METHOD("disable"), &GraphPort::disable);
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &GraphPort::set_enabled);
	ClassDB::bind_method(D_METHOD("get_enabled"), &GraphPort::get_enabled);

	ClassDB::bind_method(D_METHOD("set_type", "type"), &GraphPort::set_type);
	ClassDB::bind_method(D_METHOD("get_type"), &GraphPort::get_type);

	ClassDB::bind_method(D_METHOD("set_icon", "icon"), &GraphPort::set_icon);
	ClassDB::bind_method(D_METHOD("get_icon"), &GraphPort::get_icon);

	ClassDB::bind_method(D_METHOD("set_exclusive", "exclusive"), &GraphPort::set_exclusive);
	ClassDB::bind_method(D_METHOD("get_exclusive"), &GraphPort::get_exclusive);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &GraphPort::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &GraphPort::get_color);

	ClassDB::bind_method(D_METHOD("set_direction", "direction"), &GraphPort::set_direction);
	ClassDB::bind_method(D_METHOD("get_direction"), &GraphPort::get_direction);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "get_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_icon", "get_icon");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Input,Output,Undirected"), "set_direction", "get_direction");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclusive"), "set_exclusive", "get_exclusive");
	//ADD_PROPERTY(PropertyInfo(Variant::INT, "on_disabled_behaviour", PROPERTY_HINT_ENUM, "Disconnect all,Move to previous port or disconnect,Move to next port or disconnect"), "", "");

	ADD_SIGNAL(MethodInfo("enabled"));
	ADD_SIGNAL(MethodInfo("disabled"));
	ADD_SIGNAL(MethodInfo("changed_direction", PropertyInfo(Variant::INT, "direction", PROPERTY_HINT_ENUM, "Input,Output,Undirected")));
	ADD_SIGNAL(MethodInfo("changed_type", PropertyInfo(Variant::INT, "type")));
	ADD_SIGNAL(MethodInfo("modified"));
}

GraphPort::GraphPort() {
	icon = Ref<Texture2D>(nullptr);
}

GraphPort::GraphPort(GraphNode *p_graph_node) {
	graph_node = p_graph_node;
	icon = Ref<Texture2D>(nullptr);
}

GraphPort::GraphPort(GraphNode *p_graph_node, bool p_enabled, bool p_exclusive, int p_type, Color p_color, PortDirection p_direction, Ref<Texture2D> p_icon) {
	graph_node = p_graph_node;
	enabled = p_enabled;
	exclusive = p_exclusive;
	type = p_type;
	color = p_color;
	direction = p_direction;
	icon = p_icon;
}
