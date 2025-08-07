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

#include "core/variant/typed_array.h"
#include "scene/gui/graph_element.h"
#include "scene/resources/texture.h"

class GraphEdit;
class GraphNode;
class GraphConnection;

class GraphPort : public GraphElement {
	GDCLASS(GraphPort, GraphElement);

	friend GraphEdit;
	friend GraphNode;
	friend GraphConnection;

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

	enum CustomAccessibilityAction {
		ACTION_CONNECT,
		ACTION_FOLLOW,
	};

protected:
	struct ThemeCache {
		Ref<StyleBox> panel_focus;

		Ref<Texture2D> icon;

		float rim_size = 2;

		Color color = Color(1, 1, 1, 1);
		Color selected_color = Color(1, 1, 1, 1);
		Color rim_color = Color(0, 0, 0, 0);
		Color selected_rim_color = Color(0, 0, 0, 0);

		float hotzone_extent_h_input;
		float hotzone_extent_v_input;
		float hotzone_extent_h_output;
		float hotzone_extent_v_output;
		float hotzone_extent_h_undirected;
		float hotzone_extent_v_undirected;
		float hotzone_offset_h = 0.9;
		float hotzone_offset_v = 0.5;
	} theme_cache;

	bool enabled = false;
	bool exclusive = false;
	int port_type = 0;
	PortDirection direction = PortDirection::UNDIRECTED;
	DisconnectBehaviour on_disabled_behaviour = DisconnectBehaviour::DISCONNECT_ALL;

	GraphEdit *graph_edit = nullptr;
	GraphNode *graph_node = nullptr;

	int _index = -1;
	int _enabled_index = -1;
	int _filtered_index = -1;
	int _filtered_enabled_index = -1;

	static void _bind_methods();

	void _on_enabled();
	void _on_disabled();
	void _on_changed_direction(const PortDirection p_direction);
	void _on_changed_type(const int p_type);
	void _on_modified();
	void _on_connected(const Ref<GraphConnection> p_conn);
	void _on_disconnected(const Ref<GraphConnection> p_conn);

	void _notification(int p_what);
	virtual void _draw();

	void _accessibility_action(const Variant &p_data);

	virtual void gui_input(const Ref<InputEvent> &p_event) override;

public:
	void set_properties(bool p_enabled, bool p_exclusive, int p_type, PortDirection p_direction);

	void enable();
	void disable();
	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	int get_port_type() const;
	void set_port_type(int p_type);

	Color get_color() const;
	Color get_rim_color();

	bool get_exclusive() const;
	void set_exclusive(bool p_exclusive);

	PortDirection get_direction() const;
	void set_direction(const PortDirection p_direction);

	DisconnectBehaviour get_disabled_behaviour() const;
	void set_disabled_behaviour(const DisconnectBehaviour p_disconnect_behaviour);

	GraphNode *get_graph_node() const;

	Rect2 get_hotzone() const;

	int get_port_index(bool p_include_disabled = true) const;
	int get_filtered_port_index(bool p_include_disabled = true) const;

	TypedArray<Ref<GraphConnection>> get_connections() const;
	void set_connections(const TypedArray<Ref<GraphConnection>> &p_connections);
	void clear_connections();

	void add_connection(Ref<GraphConnection> p_connection);
	void remove_connection(Ref<GraphConnection> p_connection);
	void connect_to_port(GraphPort *p_port, bool p_clear_if_invalid = true);
	bool has_connection() const;
	Ref<GraphConnection> get_first_connection() const;
	bool is_connected_to(const GraphPort *p_port) const;

	TypedArray<GraphPort> get_connected_ports() const;
	GraphPort *get_first_connected_port() const;
	GraphNode *get_first_connected_node() const;

	GraphPort();
	GraphPort(bool p_enabled, bool p_exclusive, int p_type, PortDirection p_direction);
};

VARIANT_ENUM_CAST(GraphPort::PortDirection);
VARIANT_ENUM_CAST(GraphPort::DisconnectBehaviour);
