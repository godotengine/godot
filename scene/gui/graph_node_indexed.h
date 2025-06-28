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

#include "scene/gui/graph_node.h"

#include "core/variant/typed_dictionary.h"

class GraphNodeIndexed : public GraphNode {
	GDCLASS(GraphNodeIndexed, GraphNode);

protected:
	struct Slot {
		Ref<GraphPort> left_port;
		Ref<GraphPort> right_port;
		bool draw_stylebox = true;

		Slot();
		Slot(Ref<GraphPort> lp, Ref<GraphPort> rp, bool draw_sb) :
				left_port(lp), right_port(rp), draw_stylebox(draw_sb) {}
	};
	Vector<Slot> slots;
	HashMap<StringName, int> _slot_node_map_cache;

	struct ThemeCacheIndexed : ThemeCache {
		Ref<StyleBox> theme_cache_slot;
		Ref<StyleBox> theme_cache_slot_selected;
	};

	Vector<Slot> _get_slots();
	Slot _get_slot(int p_slot_index);
	void _set_slot(int p_slot_index, const Slot p_slot);
	void _remove_slot(int p_slot_index);

	virtual void add_child_notify(Node *p_child) override;
	virtual void move_child_notify(Node *p_child) override;
	virtual void remove_child_notify(Node *p_child) override;

public:
	//void _notification(int p_what);

	void create_slot(int p_slot_index, Ref<GraphPort> p_left_port, Ref<GraphPort> p_right_port, bool draw_stylebox);
	void create_slot_and_ports(int p_slot_index, bool draw_stylebox);
	TypedArray<Array> get_slots();
	Array get_slot(int p_slot_index);
	void set_slot(int p_slot_index, const Ref<GraphPort> p_left_port, const Ref<GraphPort> p_right_port, bool draw_stylebox);
	int slot_index_of_port(const Ref<GraphPort> p_port);

	void set_input(int p_slot_index, bool p_enabled, bool p_exclusive, int p_type, Color p_color, Ref<Texture2D> p_icon = Ref<Texture2D>(nullptr));
	void set_output(int p_slot_index, bool p_enabled, bool p_exclusive, int p_type, Color p_color, Ref<Texture2D> p_icon = Ref<Texture2D>(nullptr));
	Ref<GraphPort> get_input_port(int p_slot_index);
	Ref<GraphPort> get_output_port(int p_slot_index);
	void set_input_port(int p_slot_index, const Ref<GraphPort> p_port);
	void set_output_port(int p_slot_index, const Ref<GraphPort> p_port);

	int port_to_slot_index(int p_port_index);
	int split_port_to_slot_index(int p_port_index, bool p_input);
	int slot_to_port_index(int p_slot_index, bool p_input);
	int slot_to_split_port_index(int p_port_index, bool p_input);

	bool get_slot_draw_stylebox(int p_slot_index);
	void set_slot_draw_stylebox(int p_slot_index, bool p_draw_stylebox);

	GraphNodeIndexed();
};
