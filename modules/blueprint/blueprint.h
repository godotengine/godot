/**************************************************************************/
/*  blueprint.h                                                           */
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

#pragma once

#include "core/io/resource.h"

// A single input or output port of a blueprint node type.
struct BlueprintPort {
	String name;
	bool exec = false; // true = execution (white) port, false = data (cyan) port.
	String param_key; // Non-empty = the editor shows a LineEdit bound to params[param_key] on this row.
};

// Static definition of a blueprint node type (title, ports, default params).
struct BlueprintNodeDef {
	String type;
	String title;
	String category; // "event", "action", "flow" or "data" — drives editor grouping and colors.
	bool is_event = false;
	Vector<BlueprintPort> inputs;
	Vector<BlueprintPort> outputs;
	// Portless settings edited on the node itself (e.g. a NodePath or method name),
	// stored in params like port defaults. Rendered above the port rows in the editor.
	Vector<String> config_params;
	Dictionary default_params;
};

const Vector<BlueprintNodeDef> &blueprint_get_node_defs();
const BlueprintNodeDef *blueprint_get_node_def(const String &p_type);

class Blueprint : public Resource {
	GDCLASS(Blueprint, Resource);

	// Each node is a Dictionary: { id: int, type: String, position: Vector2, params: Dictionary }.
	Array nodes;
	// Each connection is a Dictionary: { from_node: int, from_port: int, to_node: int, to_port: int }.
	// Port indices are indices into the node definition's inputs/outputs arrays.
	Array connections;
	int next_id = 1;

protected:
	static void _bind_methods();

public:
	void set_nodes(const Array &p_nodes);
	Array get_nodes() const;
	void set_connections(const Array &p_connections);
	Array get_connections() const;
	void set_next_id(int p_next_id);
	int get_next_id() const;

	int add_node(const String &p_type, const Vector2 &p_position);
	void remove_node(int p_id);
	Dictionary get_node_data(int p_id) const;
	void set_node_position(int p_id, const Vector2 &p_position);
	void set_node_param(int p_id, const String &p_key, const Variant &p_value);

	void add_connection(int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void remove_connection(int p_from_node, int p_from_port, int p_to_node, int p_to_port);

	static PackedStringArray get_node_type_list();
};
