/**************************************************************************/
/*  graph_node.h                                                          */
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

#include "core/templates/vector.h"
#include "core/variant/typed_array.h"
#include "scene/gui/graph_port.h"

class HBoxContainer;
class GraphConnection;
class GraphEdit;

class GraphNode : public GraphElement {
	GDCLASS(GraphNode, GraphElement);

	friend GraphEdit;
	friend class GraphNodeIndexed;

protected:
	struct _MinSizeCache {
		int min_size = 0;
		bool will_stretch = false;
		int final_size = 0;
	};

	GraphEdit *graph_edit = nullptr;

	HBoxContainer *titlebar_hbox = nullptr;
	Label *title_label = nullptr;
	String title;
	bool title_hidden = false;

	Vector<GraphPort *> ports;
	int port_count = 0;
	int enabled_port_count = 0;
	PackedByteArray directed_port_count = { 0, 0, 0 };
	PackedByteArray directed_enabled_port_count = { 0, 0, 0 };

	bool port_pos_dirty = true;
	bool updating_port_pos = false;

	bool ignore_invalid_connection_type = false;
	const StringName ignore_node_meta_tag = StringName("GraphNodeIgnored");

	struct ThemeCache {
		Ref<StyleBox> panel;
		Ref<StyleBox> panel_selected;
		Ref<StyleBox> panel_focus;
		Ref<StyleBox> titlebar;
		Ref<StyleBox> titlebar_selected;

		int separation = 0;
		int port_h_offset = 0;

		Ref<Texture2D> resizer;
		Color resizer_color;
	} theme_cache;

	void _notification(int p_what);
	static void _bind_methods();
	virtual void _resort() override;
	void _deferred_resort();

	void _set_ports(const Vector<GraphPort *> &p_ports);
	const Vector<GraphPort *> &_get_ports();

	void _add_port(GraphPort *p_port);
	void _insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	GraphPort *_remove_port(int p_port_index, bool p_include_disabled = true);
	GraphPort *_set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	void _remove_all_ports();

	virtual void _update_port_positions();
	void _queue_update_port_positions();
	virtual void _port_rebuild_cache();
	void _port_modified();

	virtual void _on_replacing_by(Node *new_node);

	Callable modified_callable;
	Callable connected_callable;
	Callable disconnected_callable;

public:
	virtual String get_accessibility_container_name(const Node *p_node) const override;
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

	void set_title(const String &p_title);
	String get_title() const;
	void set_hide_title(bool p_hide);
	bool is_title_hidden() const;

	HBoxContainer *get_titlebar_hbox() const;
	void add_node_to_titlebar(Control *p_node);
	void remove_node_from_titlebar(Control *p_node);
	void clear_titlebar_nodes();

	void set_ports(const TypedArray<GraphPort> &p_ports);
	TypedArray<GraphPort> get_ports() const;
	TypedArray<GraphPort> get_filtered_ports(GraphPort::PortDirection p_direction, bool p_include_disabled = true) const;
	TypedArray<GraphPort> get_input_ports(bool p_include_disabled = true) const;
	TypedArray<GraphPort> get_output_ports(bool p_include_disabled = true) const;
	void remove_all_ports();

	virtual GraphPort *set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	GraphPort *get_port(int p_port_index, bool p_include_disabled = true) const;
	GraphPort *get_filtered_port(int p_port_index, GraphPort::PortDirection p_direction, bool p_include_disabled = true) const;
	GraphPort *get_input_port(int p_port_index, bool p_include_disabled = true) const;
	GraphPort *get_output_port(int p_port_index, bool p_include_disabled = true) const;
	GraphPort *get_next_port(const GraphPort *p_port, bool p_include_disabled = true) const;
	GraphPort *get_next_matching_port(const GraphPort *p_port, bool p_include_disabled = true) const;
	GraphPort *get_previous_port(const GraphPort *p_port, bool p_include_disabled = true) const;
	GraphPort *get_previous_matching_port(const GraphPort *p_port, bool p_include_disabled = true) const;
	virtual void add_port(GraphPort *port);
	virtual void insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	virtual GraphPort *remove_port(int p_port_index, bool p_include_disabled = true);

	int get_port_count(bool p_include_disabled = true) const;
	int get_filtered_port_count(GraphPort::PortDirection p_filter_direction, bool p_include_disabled = true) const;
	int get_input_port_count(bool p_include_disabled = true) const;
	int get_output_port_count(bool p_include_disabled = true) const;

	int index_of_port(const GraphPort *p_port, bool p_include_disabled = true) const;
	int filtered_index_of_port(const GraphPort *p_port, bool p_include_disabled = true) const;

	int enabled_index_to_port_index(int p_port_index) const;
	int port_index_to_enabled_index(int p_port_index) const;

	virtual GraphPort *get_port_navigation(Side side, const GraphPort *p_port) const;

	void set_ignore_invalid_connection_type(bool p_ignore);
	bool is_ignoring_valid_connection_type() const;

	TypedArray<Ref<GraphConnection>> get_connections() const;
	TypedArray<Ref<GraphConnection>> get_filtered_connections(GraphPort::PortDirection p_filter_direction) const;
	TypedArray<Ref<GraphConnection>> get_input_connections() const;
	TypedArray<Ref<GraphConnection>> get_output_connections() const;
	void set_connections(const TypedArray<Ref<GraphConnection>> &p_connections);
	void add_connections(const TypedArray<Ref<GraphConnection>> &p_connections);
	void clear_connections();

	void add_connection(Ref<GraphConnection> p_connection);
	void remove_connection(Ref<GraphConnection> p_connection);
	bool is_connected_to(GraphNode *p_node) const;
	bool has_connection() const;

	TypedArray<GraphNode> get_connected_nodes() const;
	TypedArray<GraphNode> get_filtered_connected_nodes(GraphPort::PortDirection p_filter_direction) const;
	TypedArray<GraphNode> get_input_connected_nodes() const;
	TypedArray<GraphNode> get_output_connected_nodes() const;

	void clear_filtered_connections(GraphPort::PortDirection p_filter_direction);
	void clear_input_connections();
	void clear_output_connections();

	void _on_connected(const Ref<GraphConnection> p_conn);
	void _on_disconnected(const Ref<GraphConnection> p_conn);

	virtual Size2 get_minimum_size() const override;

	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	GraphNode();
};
