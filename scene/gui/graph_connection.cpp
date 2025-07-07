/**************************************************************************/
/*  graph_connection.cpp                                                  */
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

#include "graph_connection.h"

#include "scene/gui/graph_node.h"

GraphPort *GraphConnection::get_other_port(GraphPort *p_port) {
	if (p_port == first_port) {
		return second_port;
	} else if (p_port == second_port) {
		return first_port;
	} else {
		ERR_FAIL_V_MSG(nullptr, "Connection does not contain port");
	}
}

GraphNode *GraphConnection::get_other_node(GraphNode *p_node) {
	if (p_node == get_first_node()) {
		return get_first_node();
	} else if (p_node == get_second_node()) {
		return get_second_node();
	} else {
		ERR_FAIL_V_MSG(nullptr, "Connection does not connect to");
	}
}

// This legacy method is exclusively used by visual shaders, which use legacy port indices and expect GraphNodeIndexed's behavior
Pair<Pair<String, int>, Pair<String, int>> GraphConnection::_to_legacy_data() {
	ERR_FAIL_NULL_V(first_port->graph_node, Pair(Pair(String(""), -1), Pair(String(""), -1)));
	ERR_FAIL_NULL_V(second_port->graph_node, Pair(Pair(String(""), -1), Pair(String(""), -1)));
	return Pair(Pair(String(first_port->graph_node->get_name()), first_port->get_filtered_port_index(false)), Pair(String(second_port->graph_node->get_name()), second_port->get_filtered_port_index(false)));
}

bool GraphConnection::matches_legacy_data(String p_first_node, int p_first_port, String p_second_node, int p_second_port) {
	ERR_FAIL_NULL_V(first_port, false);
	ERR_FAIL_NULL_V(second_port, false);
	ERR_FAIL_NULL_V(first_port->graph_node, false);
	ERR_FAIL_NULL_V(second_port->graph_node, false);
	return first_port->get_filtered_port_index(false) == p_first_port &&
			second_port->get_filtered_port_index(false) == p_second_port &&
			String(first_port->graph_node->get_name()) == p_first_node &&
			String(second_port->graph_node->get_name()) == p_second_node;
}

void GraphConnection::set_first_port(GraphPort *p_port) {
	first_port = p_port;
}

GraphPort *GraphConnection::get_first_port() {
	return first_port;
}

void GraphConnection::set_second_port(GraphPort *p_port) {
	second_port = p_port;
}

GraphPort *GraphConnection::get_second_port() {
	return second_port;
}

GraphNode *GraphConnection::get_first_node() {
	ERR_FAIL_NULL_V(first_port, nullptr);
	return first_port->graph_node;
}

GraphNode *GraphConnection::get_second_node() {
	ERR_FAIL_NULL_V(second_port, nullptr);
	return second_port->graph_node;
}

void GraphConnection::set_clear_if_invalid(bool p_clear_if_invalid) {
	clear_if_invalid = p_clear_if_invalid;
}

bool GraphConnection::get_clear_if_invalid() {
	return clear_if_invalid;
}

void GraphConnection::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_first_port", "port"), &GraphConnection::set_first_port);
	ClassDB::bind_method(D_METHOD("get_first_port"), &GraphConnection::get_first_port);
	ClassDB::bind_method(D_METHOD("set_second_port", "port"), &GraphConnection::set_second_port);
	ClassDB::bind_method(D_METHOD("get_second_port"), &GraphConnection::get_second_port);
	ClassDB::bind_method(D_METHOD("get_other_port"), &GraphConnection::get_other_port);

	ClassDB::bind_method(D_METHOD("get_first_node"), &GraphConnection::get_first_node);
	ClassDB::bind_method(D_METHOD("get_second_node"), &GraphConnection::get_second_node);
	ClassDB::bind_method(D_METHOD("get_other_node"), &GraphConnection::get_other_node);

	ClassDB::bind_method(D_METHOD("set_clear_if_invalid", "clear_if_invalid"), &GraphConnection::set_clear_if_invalid);
	ClassDB::bind_method(D_METHOD("get_clear_if_invalid"), &GraphConnection::get_clear_if_invalid);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "first_port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort", PROPERTY_USAGE_DEFAULT), "set_first_port", "get_first_port");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "second_port", PROPERTY_HINT_RESOURCE_TYPE, "GraphPort", PROPERTY_USAGE_DEFAULT), "set_second_port", "get_second_port");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clear_if_invalid"), "set_clear_if_invalid", "get_clear_if_invalid");
}

GraphConnection::GraphConnection() {
}

GraphConnection::GraphConnection(GraphPort *p_first_port, GraphPort *p_second_port, bool p_clear_if_invalid) {
	first_port = p_first_port;
	second_port = p_second_port;
	clear_if_invalid = p_clear_if_invalid;
}
