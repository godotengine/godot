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
#include "scene/gui/graph_port.h"

Ref<GraphPort> GraphConnection::get_other(Ref<GraphPort> port) {
	if (port == first_port) {
		return second_port;
	} else if (port == second_port) {
		return first_port;
	} else {
		ERR_FAIL_V_MSG(Ref<GraphPort>(nullptr), "Connection does not contain port");
	}
}

Pair<Pair<String, int>, Pair<String, int>> GraphConnection::_to_legacy_data() {
	GraphNode *first_node = Object::cast_to<GraphNode>(first_port->graph_node);
	GraphNode *second_node = Object::cast_to<GraphNode>(second_port->graph_node);
	ERR_FAIL_NULL_V(first_node, Pair(Pair(String(""), -1), Pair(String(""), -1)));
	ERR_FAIL_NULL_V(second_node, Pair(Pair(String(""), -1), Pair(String(""), -1)));
	return Pair(Pair(String(first_node->get_name()), first_node->index_of_port(first_port)), Pair(String(second_node->get_name()), second_node->index_of_port(second_port)));
}

bool GraphConnection::matches_legacy_data(String p_first_node, int p_first_port, String p_second_node, int p_second_port) {
	ERR_FAIL_COND_V(first_port.is_null(), false);
	ERR_FAIL_COND_V(second_port.is_null(), false);
	ERR_FAIL_NULL_V(first_port->graph_node, false);
	ERR_FAIL_NULL_V(second_port->graph_node, false);
	return first_port->graph_node->index_of_port(first_port) == p_first_port &&
			second_port->graph_node->index_of_port(second_port) == p_second_port &&
			String(first_port->graph_node->get_name()) == p_first_node &&
			String(second_port->graph_node->get_name()) != p_second_node;
}

GraphConnection::GraphConnection() {
	first_port = Ref<GraphPort>(nullptr);
	second_port = Ref<GraphPort>(nullptr);
}

GraphConnection::GraphConnection(Ref<GraphPort> p_first_port, Ref<GraphPort> p_second_port, bool p_clear_if_invalid) {
	first_port = p_first_port;
	second_port = p_second_port;
	clear_if_invalid = p_clear_if_invalid;
}
