/**************************************************************************/
/*  graph_port.h                                                          */
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
#include "core/variant/typed_array.h"
#include "scene/resources/texture.h"

class GraphPort : public Resource {
	GDCLASS(GraphPort, Resource);

	friend class GraphEdit;
	friend class GraphNode;
	friend class GraphConnection;

public:
	enum PortDirection {
		INPUT,
		OUTPUT,
		UNDIRECTED
	};

	enum DisconnectBehaviour {
		DISCONNECT_ALL,
		MOVE_TO_PREVIOUS_PORT_OR_DISCONNECT,
		MOVE_TO_NEXT_PORT_OR_DISCONNECT
	};

protected:
	bool enabled;
	bool exclusive;
	int type = 0;
	Color color = Color(1, 1, 1, 1);
	Ref<Texture2D> icon;
	PortDirection direction = PortDirection::UNDIRECTED;
	DisconnectBehaviour on_disabled_behaviour = DisconnectBehaviour::DISCONNECT_ALL;

	GraphNode *graph_node;
	Vector2 position = Vector2(0.0, 0.0);

	static void _bind_methods();

	void _enabled();
	void _disabled();

	void _changed_direction(const PortDirection p_direction);
	void _changed_type(const int p_type);

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void set(bool p_enabled, bool p_exclusive, int p_type, Color p_color, PortDirection p_direction, Ref<Texture2D> p_icon = Ref<Texture2D>(nullptr));

	void enable();
	void disable();
	void set_enabled(bool p_enabled);
	bool get_enabled();

	int get_type();
	void set_type(int p_type);

	Color get_color();
	void set_color(Color p_color);

	bool get_exclusive();
	void set_exclusive(bool p_exclusive);

	Ref<Texture2D> get_icon();
	void set_icon(Ref<Texture2D> p_icon);

	PortDirection get_direction() const;
	void set_direction(const PortDirection p_direction);

	Vector2 get_position();
	GraphNode *get_graph_node();

	GraphPort();
	GraphPort(GraphNode *p_graph_node);
	GraphPort(GraphNode *p_graph_node, bool p_enabled, bool p_exclusive, int p_type, Color p_color, PortDirection p_direction, Ref<Texture2D> p_icon);
};

VARIANT_ENUM_CAST(GraphPort::PortDirection);
VARIANT_ENUM_CAST(GraphPort::DisconnectBehaviour);
