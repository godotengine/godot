/*************************************************************************/
/*  graph_node.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "scene/gui/base_graph_node.h"
#include "scene/resources/text_line.h"

class GraphNode : public BaseGraphNode {
	GDCLASS(GraphNode, BaseGraphNode);

public:
	enum Overlay {
		OVERLAY_DISABLED,
		OVERLAY_BREAKPOINT,
		OVERLAY_POSITION
	};

private:
	struct Slot {
		bool enable_left = false;
		int type_left = 0;
		Color color_left = Color(1, 1, 1, 1);
		Ref<Texture2D> custom_slot_left;

		bool enable_right = false;
		int type_right = 0;
		Color color_right = Color(1, 1, 1, 1);
		Ref<Texture2D> custom_slot_right;

		bool draw_stylebox = true;
	};

	struct PortCache {
		Color color;
		Vector2 pos;
		int type = 0;
		int height;
	};

	Vector<PortCache> input_port_cache;
	Vector<PortCache> output_port_cache;

	HashMap<int, Slot> slot_table;

	bool port_pos_dirty = true;

	Overlay overlay = OVERLAY_DISABLED;

	void _port_pos_update();
	void _resort();
	void _shape_title();

protected:
	virtual void gui_input(const Ref<InputEvent> &p_ev) override;
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void set_slot(int p_idx, bool p_enable_left, int p_type_left, const Color &p_color_left, bool p_enable_right, int p_type_right, const Color &p_color_right, const Ref<Texture2D> &p_custom_left = Ref<Texture2D>(), const Ref<Texture2D> &p_custom_right = Ref<Texture2D>(), bool p_draw_stylebox = true);
	void clear_slot(int p_idx);
	void clear_all_slots();

	bool is_slot_enabled_left(int p_idx) const;
	void set_slot_enabled_left(int p_idx, bool p_enable_left);

	void set_slot_type_left(int p_idx, int p_type_left);
	int get_slot_type_left(int p_idx) const;

	void set_slot_color_left(int p_idx, const Color &p_color_left);
	Color get_slot_color_left(int p_idx) const;

	bool is_slot_enabled_right(int p_idx) const;
	void set_slot_enabled_right(int p_idx, bool p_enable_right);

	void set_slot_type_right(int p_idx, int p_type_right);
	int get_slot_type_right(int p_idx) const;

	void set_slot_color_right(int p_idx, const Color &p_color_right);
	Color get_slot_color_right(int p_idx) const;

	bool is_slot_draw_stylebox(int p_idx) const;
	void set_slot_draw_stylebox(int p_idx, bool p_enable);

	int get_port_input_count();
	int get_port_input_height(int p_idx);
	Vector2 get_port_input_position(int p_idx);
	int get_port_input_type(int p_idx);
	Color get_port_input_color(int p_idx);

	int get_port_output_count();
	int get_port_output_height(int p_idx);
	Vector2 get_port_output_position(int p_idx);
	int get_port_output_type(int p_idx);
	Color get_port_output_color(int p_idx);

	void set_overlay(Overlay p_overlay);
	Overlay get_overlay() const;

	virtual Size2 get_minimum_size() const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	GraphNode();
};

VARIANT_ENUM_CAST(GraphNode::Overlay)

#endif // GRAPH_NODE_H
