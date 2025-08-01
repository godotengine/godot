/**************************************************************************/
/*  graph_edit.compat.inc                                                 */
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

#ifndef DISABLE_DEPRECATED

bool GraphEdit::_is_arrange_nodes_button_hidden_bind_compat_81582() const {
	return !is_showing_arrange_button();
}

void GraphEdit::_set_arrange_nodes_button_hidden_bind_compat_81582(bool p_enable) {
	set_show_arrange_button(!p_enable);
}

PackedVector2Array GraphEdit::_get_connection_line_bind_compat_86158(const Vector2 &p_from, const Vector2 &p_to) {
	return get_connection_line(p_from, p_to);
}

Ref<GraphConnection> GraphEdit::_connect_node_bind_compat_97449(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port) {
	return connect_nodes_indexed_legacy(p_from, p_from_port, p_to, p_to_port);
}

bool GraphEdit::_is_in_input_hotzone_bind_compat_108099(Object *in_node, int in_port, Vector2 mouse_position) {
	GraphNode *gn = cast_to<GraphNode>(in_node);
	ERR_FAIL_NULL_V(gn, false);
	GraphPort *gp = gn->get_input_port(in_port, false);
	ERR_FAIL_NULL_V(gp, false);
	return is_in_port_hotzone(gp, mouse_position);
}

bool GraphEdit::_is_in_output_hotzone_bind_compat_108099(Object *in_node, int in_port, Vector2 mouse_position) {
	GraphNode *gn = cast_to<GraphNode>(in_node);
	ERR_FAIL_NULL_V(gn, false);
	GraphPort *gp = gn->get_output_port(in_port, false);
	ERR_FAIL_NULL_V(gp, false);
	return is_in_port_hotzone(gp, mouse_position);
}

void GraphEdit::_add_valid_left_disconnect_type_bind_compat_108099(int type) {
	add_valid_input_disconnect_type(type);
}

void GraphEdit::_add_valid_right_disconnect_type_bind_compat_108099(int type) {
	add_valid_output_disconnect_type(type);
}

Error GraphEdit::_connect_node_bind_compat_108099(StringName from_node, int from_port, StringName to_node, int to_port, bool keep_alive) {
	Ref<GraphConnection> conn = connect_nodes_indexed_legacy(from_node, from_port, to_node, to_port, keep_alive);
	if (conn.is_valid()) {
		return OK;
	} else {
		return FAILED;
	}
}

void GraphEdit::_disconnect_node_bind_compat_108099(StringName from_node, int from_port, StringName to_node, int to_port) {
	disconnect_nodes_indexed_legacy(from_node, from_port, to_node, to_port);
}

Array GraphEdit::_get_connection_list_bind_compat_108099() {
	Array ret;
	for (const Ref<GraphConnection> conn : graph_connections) {
		if (conn.is_null()) {
			continue;
		}
		Pair<Pair<String, int>, Pair<String, int>> legacy_conn = conn->_to_legacy_data();
		Dictionary conn_dict = {};
		conn_dict["from_node"] = legacy_conn.first.first;
		conn_dict["from_port"] = legacy_conn.first.second;
		conn_dict["to_node"] = legacy_conn.second.first;
		conn_dict["to_port"] = legacy_conn.second.second;
		conn_dict["keep_alive"] = !conn->clear_if_invalid;
		ret.push_back(conn_dict);
	}
	return ret;
}

bool GraphEdit::_is_right_disconnects_enabled_bind_compat_108099() {
	return input_disconnects;
}

void GraphEdit::_remove_valid_left_disconnect_type_bind_compat_108099(int type) {
	remove_valid_input_disconnect_type(type);
}

void GraphEdit::_remove_valid_right_disconnect_type_bind_compat_108099(int type) {
	remove_valid_output_disconnect_type(type);
}

void GraphEdit::_set_right_disconnects_bind_compat_108099(bool value) {
	input_disconnects = value;
}

void GraphEdit::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("is_arrange_nodes_button_hidden"), &GraphEdit::_is_arrange_nodes_button_hidden_bind_compat_81582);
	ClassDB::bind_compatibility_method(D_METHOD("set_arrange_nodes_button_hidden", "enable"), &GraphEdit::_set_arrange_nodes_button_hidden_bind_compat_81582);
	ClassDB::bind_compatibility_method(D_METHOD("get_connection_line", "from_node", "to_node"), &GraphEdit::_get_connection_line_bind_compat_86158);
	ClassDB::bind_compatibility_method(D_METHOD("connect_node", "from_node", "from_port", "to_node", "to_port"), &GraphEdit::_connect_node_bind_compat_97449);
	ClassDB::bind_compatibility_method(D_METHOD("disconnect_node", "from_node", "from_port", "to_node", "to_port"), &GraphEdit::_disconnect_node_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("_is_in_input_hotzone", "in_node", "in_port", "mouse_position"), &GraphEdit::_is_in_input_hotzone_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("_is_in_output_hotzone", "in_node", "in_port", "mouse_position"), &GraphEdit::_is_in_output_hotzone_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("add_valid_left_disconnect_type", "type"), &GraphEdit::_add_valid_left_disconnect_type_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("add_valid_right_disconnect_type", "type"), &GraphEdit::_add_valid_right_disconnect_type_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("get_connection_list"), &GraphEdit::_get_connection_list_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("is_right_disconnects_enabled"), &GraphEdit::_is_right_disconnects_enabled_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("set_right_disconnects", "value"), &GraphEdit::_set_right_disconnects_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("remove_valid_left_disconnect_type", "type"), &GraphEdit::_remove_valid_left_disconnect_type_bind_compat_108099);
	ClassDB::bind_compatibility_method(D_METHOD("remove_valid_right_disconnect_type", "type"), &GraphEdit::_remove_valid_right_disconnect_type_bind_compat_108099);
}

#endif
