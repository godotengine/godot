/**************************************************************************/
/*  graph_node_indexed.h                                                  */
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

#include "core/variant/typed_dictionary.h"
#include "scene/gui/graph_node.h"

class GraphNodeIndexed : public GraphNode {
	GDCLASS(GraphNodeIndexed, GraphNode);

protected:
	struct Slot {
		bool draw_stylebox = true;
		StringName node_name;

		Slot();
		Slot(bool p_draw_sb, StringName p_node_name) :
				draw_stylebox(p_draw_sb), node_name(p_node_name) {}
	};
	Vector<Slot> slots;
	HashMap<StringName, int> _node_to_slot_cache;
	Vector<float> _slot_y_cache;

	int selected_slot = -1;
	Control::FocusMode slot_focus_mode = Control::FOCUS_ACCESSIBILITY;

	Control *port_container = nullptr;
	StringName port_container_name = StringName("PortContainer");
	int port_container_idx = 0;

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

		Ref<StyleBox> slot;
		Ref<StyleBox> slot_selected;
	} theme_cache;

	void _set_slots(const Vector<Slot> &p_slots);
	const Vector<Slot> &_get_slots();
	void _set_slot(int p_slot_index, const Slot p_slot);
	Slot _get_slot(int p_slot_index);
	void _insert_slot(int p_slot_index, const Slot p_slot);
	void _remove_all_slots();
	void _remove_slot(int p_slot_index);

	void _set_slot_node_cache(const TypedDictionary<StringName, int> &p_slot_node_map_cache);
	TypedDictionary<StringName, int> _get_slot_node_cache();

	virtual void _resort() override;
	virtual void _update_port_positions() override;

	virtual void add_child_notify(Node *p_child) override;
	virtual void move_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;

	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

	virtual void create_slot_and_ports(int p_slot_index, bool p_draw_stylebox, StringName p_slot_node_name);
	void remove_slot_and_ports(int p_slot_index);
	void move_slot_with_ports(int p_old_slot_index, int p_new_slot_index);
	void set_slot_with_ports(int p_slot_index, Slot p_slot, GraphPort *p_input_port, GraphPort *p_output_port);
	void copy_slot_with_ports(int p_old_slot_index, int p_new_slot_index);

	void set_slots(const TypedArray<Array> &p_slots);

	virtual void add_port(GraphPort *p_port) override;
	virtual void insert_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true) override;
	virtual GraphPort *remove_port(int p_port_index, bool p_include_disabled = true) override;

	virtual GraphPort *set_port(int p_port_index, GraphPort *p_port, bool p_include_disabled = true) override;

public:
	void _notification(int p_what);

	TypedArray<Array> get_slots() const;

	Array get_slot(int p_slot_index);

	void set_slot_properties(int p_slot_index, bool p_input_enabled, int p_input_type, bool p_output_enabled, int p_output_type);
	void set_input_port_properties(int p_slot_index, bool p_enabled, int p_type);
	void set_output_port_properties(int p_slot_index, bool p_enabled, int p_type);

	void set_ports_at_slot(int p_slot_index, GraphPort *p_input_port, GraphPort *p_output_port);
	GraphPort *set_input_port_at_slot(int p_slot_index, GraphPort *p_port);
	GraphPort *set_output_port_at_slot(int p_slot_index, GraphPort *p_port);

	GraphPort *get_input_port_by_slot(int p_slot_index) const;
	GraphPort *get_output_port_by_slot(int p_slot_index) const;
	GraphPort *get_input_port_by_node(const Node *p_node) const;
	GraphPort *get_output_port_by_node(const Node *p_node) const;

	int slot_index_of_node(const Node *p_node) const;
	int slot_index_of_port(const GraphPort *p_port) const;

	int port_to_slot_index(int p_port_index, bool p_include_disabled = true) const;
	int slot_to_port_index(int p_slot_index, bool p_input, bool p_include_disabled = true) const;

	int slot_to_input_port_index(int p_slot_index, bool p_include_disabled = true) const;
	int slot_to_output_port_index(int p_slot_index, bool p_include_disabled = true) const;
	int input_port_to_slot_index(int p_port_index, bool p_include_disabled = true) const;
	int output_port_to_slot_index(int p_port_index, bool p_include_disabled = true) const;

	int child_to_slot_index(int idx, bool p_include_internal = false) const;
	int slot_to_child_index(int idx, bool p_include_internal = false) const;

	Node *get_child_by_slot_index(int p_slot_index) const;
	Node *get_child_by_port(const GraphPort *p_port) const;

	bool get_slot_draw_stylebox(int p_slot_index) const;
	void set_slot_draw_stylebox(int p_slot_index, bool p_draw_stylebox);

	void set_slot_focus_mode(Control::FocusMode p_focus_mode);
	Control::FocusMode get_slot_focus_mode() const;

	virtual Size2 get_minimum_size() const override;

	void set_port_container(Control *p_container);
	Control *get_port_container() const;
	void ensure_port_container();

	GraphNodeIndexed();
};
