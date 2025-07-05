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
		GraphPort *left_port;
		GraphPort *right_port;
		bool draw_stylebox = true;

		Slot();
		Slot(GraphPort *lp, GraphPort *rp, bool draw_sb) :
				left_port(lp), right_port(rp), draw_stylebox(draw_sb) {}
	};
	Vector<Slot> slots;
	HashMap<StringName, int> _slot_node_map_cache;
	Vector<float> _slot_y_cache;

	int selected_slot = -1;
	Control::FocusMode slot_focus_mode = Control::FOCUS_ACCESSIBILITY;

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
	void _set_slot(int p_slot_index, const Slot p_slot, bool p_with_ports = true);
	Slot _get_slot(int p_slot_index);
	void _insert_slot(int p_slot_index, const Slot p_slot, bool p_with_ports = true);
	void _remove_all_slots(bool p_with_ports = true);
	void _remove_slot(int p_slot_index, bool p_with_ports = true);

	virtual void _resort() override;
	virtual void _port_pos_update() override;

	virtual void add_child_notify(Node *p_child) override;
	virtual void move_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;

	static void _bind_methods();

public:
	void _notification(int p_what);

	void create_slot(int p_slot_index, GraphPort *p_left_port, GraphPort *p_right_port, bool draw_stylebox);
	virtual void create_slot_and_ports(int p_slot_index, bool draw_stylebox);

	void set_slots(TypedArray<Array> p_slots);
	TypedArray<Array> get_slots();

	void set_slot(int p_slot_index, GraphPort *p_left_port, GraphPort *p_right_port, bool draw_stylebox);
	Array get_slot(int p_slot_index);

	void set_slot_properties(int p_slot_index, bool p_input_enabled, bool p_input_type, bool p_output_enabled, bool p_output_type);
	void set_input_port_properties(int p_slot_index, bool p_enabled, int p_type);
	void set_output_port_properties(int p_slot_index, bool p_enabled, int p_type);

	void set_input_port(int p_slot_index, GraphPort *p_port);
	void set_output_port(int p_slot_index, GraphPort *p_port);
	GraphPort *get_input_port(int p_slot_index);
	GraphPort *get_output_port(int p_slot_index);

	int get_input_port_count();
	int get_output_port_count();

	int slot_index_of_port(GraphPort *p_port);
	int index_of_input_port(GraphPort *p_port, bool p_include_disabled = true);
	int index_of_output_port(GraphPort *p_port, bool p_include_disabled = true);

	int port_to_slot_index(int p_port_index, bool p_include_disabled = true);
	int slot_to_port_index(int p_slot_index, bool p_input, bool p_include_disabled = true);

	int slot_to_input_port_index(int p_slot_index, bool p_include_disabled = true);
	int slot_to_output_port_index(int p_slot_index, bool p_include_disabled = true);
	int input_port_to_slot_index(int p_port_index, bool p_include_disabled = true);
	int output_port_to_slot_index(int p_port_index, bool p_include_disabled = true);

	bool get_slot_draw_stylebox(int p_slot_index);
	void set_slot_draw_stylebox(int p_slot_index, bool p_draw_stylebox);

	void set_slot_focus_mode(Control::FocusMode p_focus_mode);
	Control::FocusMode get_slot_focus_mode() const;

	virtual Size2 get_minimum_size() const override;

	GraphNodeIndexed();
};
