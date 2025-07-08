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

protected:
	struct _MinSizeCache {
		int min_size = 0;
		bool will_stretch = false;
		int final_size = 0;
	};

	enum CustomAccessibilityAction {
		ACTION_CONNECT,
		ACTION_FOLLOW,
	};
	void _accessibility_action_port(const Variant &p_data);

	HBoxContainer *titlebar_hbox = nullptr;
	Label *title_label = nullptr;

	String title;

	Vector<GraphPort *> ports;
	Container *port_container;

	int port_count = 0;
	int enabled_port_count = 0;
	PackedByteArray directed_port_count = { 0, 0, 0 };
	PackedByteArray directed_enabled_port_count = { 0, 0, 0 };
	GraphPort *selected_port = nullptr;

	Control::FocusMode port_focus_mode = Control::FOCUS_ACCESSIBILITY;
	bool port_pos_dirty = true;
	bool updating_port_pos = false;

	bool ignore_invalid_connection_type = false;

	struct ThemeCache {
		Ref<StyleBox> panel;
		Ref<StyleBox> panel_selected;
		Ref<StyleBox> panel_focus;
		Ref<StyleBox> titlebar;
		Ref<StyleBox> titlebar_selected;
		Ref<StyleBox> port_selected;

		int separation = 0;
		int port_h_offset = 0;

		Ref<Texture2D> resizer;
		Color resizer_color;
	} theme_cache;

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _resort() override;

	void _set_ports(const Vector<GraphPort *> &p_ports);
	const Vector<GraphPort *> &_get_ports();

	void _add_port(GraphPort *p_port);
	void _insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	void _remove_port(int p_port_index, bool p_include_disabled = true);
	void _set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	void _remove_all_ports();

	virtual void _update_port_positions();
	void _queue_update_port_positions();
	virtual void _port_rebuild_cache();
	void _port_modified();

public:
	virtual String get_accessibility_container_name(const Node *p_node) const override;
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	Control *get_accessibility_node_by_port(int p_port_index);

	void set_title(const String &p_title);
	String get_title() const;

	HBoxContainer *get_titlebar_hbox();

	void set_ports(TypedArray<GraphPort> p_ports);
	TypedArray<GraphPort> get_ports();
	TypedArray<GraphPort> get_filtered_ports(GraphPort::PortDirection p_direction, bool p_include_disabled = true);
	void remove_all_ports();
	void set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	GraphPort *get_port(int p_port_index, bool p_include_disabled = true);
	GraphPort *get_filtered_port(int p_port_index, GraphPort::PortDirection p_direction, bool p_include_disabled = true);
	GraphPort *get_next_matching_port(GraphPort *p_port, bool p_include_disabled = true);
	GraphPort *get_previous_matching_port(GraphPort *p_port, bool p_include_disabled = true);
	void add_port(GraphPort *port);
	void insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true);
	void remove_port(int p_port_index, bool p_include_disabled = true);

	int get_port_count(bool p_include_disabled = true);
	int get_filtered_port_count(GraphPort::PortDirection p_filter_direction, bool p_include_disabled = true);
	int index_of_port(GraphPort *p_port, bool p_include_disabled = true);
	int filtered_index_of_port(GraphPort *p_port, bool p_include_disabled = true);
	int enabled_index_to_port_index(int p_port_index);
	int port_index_to_enabled_index(int p_port_index);

	void set_ignore_invalid_connection_type(bool p_ignore);
	bool is_ignoring_valid_connection_type() const;

	void set_port_focus_mode(Control::FocusMode p_focus_mode);
	Control::FocusMode get_port_focus_mode() const;

	bool has_connection();
	TypedArray<Ref<GraphConnection>> get_connections();
	TypedArray<Ref<GraphConnection>> get_filtered_connections(GraphPort::PortDirection p_filter_direction);
	TypedArray<GraphNode> get_connected_nodes();
	TypedArray<GraphNode> get_filtered_connected_nodes(GraphPort::PortDirection p_filter_direction);

	void _on_connected(const Ref<GraphConnection> p_conn);
	void _on_disconnected(const Ref<GraphConnection> p_conn);

	virtual Size2 get_minimum_size() const override;

	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	GraphNode();
};
