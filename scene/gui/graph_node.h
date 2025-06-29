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

#include "scene/gui/graph_element.h"
#include "scene/gui/graph_port.h"
//#include "scene/property_list_helper.h"

class HBoxContainer;
class GraphConnection;

class GraphNode : public GraphElement {
	GDCLASS(GraphNode, GraphElement);

	friend class GraphEdit;

protected:
	struct PortCache {
		Vector2 pos = Vector2(0.0, 0.0);
		int type = 0;
		Color color = Color(1, 1, 1, 1);

		PortCache() {}
		PortCache(const Vector2 p_pos, int p_type, const Color p_color) :
				pos(p_pos), type(p_type), color(p_color) {}
		PortCache(const Ref<GraphPort> p_port) {
			pos = p_port->position;
			type = p_port->type;
			color = p_port->color;
		}
	};

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

	//static inline PropertyListHelper base_property_helper;
	//PropertyListHelper property_helper;

	HBoxContainer *titlebar_hbox = nullptr;
	Label *title_label = nullptr;

	String title;

	TypedArray<Ref<GraphPort>> ports;
	Vector<PortCache> port_cache;

	int selected_port = -1;

	bool port_pos_dirty = true;

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

		Ref<Texture2D> port;
		Ref<Texture2D> resizer;
		Color resizer_color;
	} theme_cache;

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _resort() override;

	virtual void draw_port(const Ref<GraphPort> p_port);
	GDVIRTUAL1(_draw_port, const Ref<GraphPort>);

	void _set_ports(const TypedArray<Ref<GraphPort>> &p_ports);
	const TypedArray<Ref<GraphPort>> &_get_ports();

	virtual void _port_pos_update();
	void _port_modified();
	//bool _set(const StringName &p_name, const Variant &p_value);
	//bool _get(const StringName &p_name, Variant &r_ret) const;
	//void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual String get_accessibility_container_name(const Node *p_node) const override;
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	Control *get_accessibility_node_by_port(int p_port_idx);

	void set_title(const String &p_title);
	String get_title() const;

	HBoxContainer *get_titlebar_hbox();

	Ref<GraphPort> get_port(int p_port_idx);
	void set_ports(Array p_ports);
	Array get_ports();
	int index_of_port(const Ref<GraphPort> p_port);
	void add_port(const Ref<GraphPort> port);
	void insert_port(int p_port_index, const Ref<GraphPort> p_port);
	void set_port(int p_port_index, const Ref<GraphPort> p_port);
	void remove_port(int p_port_index);
	void remove_all_ports();
	int get_port_count();
	Vector2 update_port_position(int p_port_idx);

	void set_ignore_invalid_connection_type(bool p_ignore);
	bool is_ignoring_valid_connection_type() const;

	virtual Size2 get_minimum_size() const override;

	virtual CursorShape get_cursor_shape(const Point2 &p_pos = Point2i()) const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	GraphNode();
};
