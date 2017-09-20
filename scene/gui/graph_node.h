/*************************************************************************/
/*  graph_node.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "scene/gui/container.h"

class GraphNode : public Container {

	GDCLASS(GraphNode, Container);

public:
	enum Overlay {
		OVERLAY_DISABLED,
		OVERLAY_BREAKPOINT,
		OVERLAY_POSITION
	};

private:
	struct Slot {
		bool enable_left;
		int type_left;
		Color color_left;
		bool enable_right;
		int type_right;
		Color color_right;
		Ref<Texture> custom_slot_left;
		Ref<Texture> custom_slot_right;

		Slot() {
			enable_left = false;
			type_left = 0;
			color_left = Color(1, 1, 1, 1);
			enable_right = false;
			type_right = 0;
			color_right = Color(1, 1, 1, 1);
		}
	};

	String title;
	bool show_close;
	Vector2 offset;
	bool comment;
	bool resizable;

	bool resizing;
	Vector2 resizing_from;
	Vector2 resizing_from_size;

	Rect2 close_rect;

	Vector<int> cache_y;

	struct ConnCache {
		Vector2 pos;
		int type;
		Color color;
	};

	Vector<ConnCache> conn_input_cache;
	Vector<ConnCache> conn_output_cache;

	Map<int, Slot> slot_info;

	bool connpos_dirty;

	void _connpos_update();
	void _resort();

	Vector2 drag_from;
	bool selected;

	Overlay overlay;

protected:
	void _gui_input(const Ref<InputEvent> &p_ev);
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	bool has_point(const Point2 &p_point) const;

	void set_slot(int p_idx, bool p_enable_left, int p_type_left, const Color &p_color_left, bool p_enable_right, int p_type_right, const Color &p_color_right, const Ref<Texture> &p_custom_left = Ref<Texture>(), const Ref<Texture> &p_custom_right = Ref<Texture>());
	void clear_slot(int p_idx);
	void clear_all_slots();
	bool is_slot_enabled_left(int p_idx) const;
	int get_slot_type_left(int p_idx) const;
	Color get_slot_color_left(int p_idx) const;
	bool is_slot_enabled_right(int p_idx) const;
	int get_slot_type_right(int p_idx) const;
	Color get_slot_color_right(int p_idx) const;

	void set_title(const String &p_title);
	String get_title() const;

	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;

	void set_selected(bool p_selected);
	bool is_selected();

	void set_drag(bool p_drag);
	Vector2 get_drag_from();

	void set_show_close_button(bool p_enable);
	bool is_close_button_visible() const;

	int get_connection_input_count();
	int get_connection_output_count();
	Vector2 get_connection_input_position(int p_idx);
	int get_connection_input_type(int p_idx);
	Color get_connection_input_color(int p_idx);
	Vector2 get_connection_output_position(int p_idx);
	int get_connection_output_type(int p_idx);
	Color get_connection_output_color(int p_idx);

	void set_overlay(Overlay p_overlay);
	Overlay get_overlay() const;

	void set_comment(bool p_enable);
	bool is_comment() const;

	void set_resizable(bool p_enable);
	bool is_resizable() const;

	virtual Size2 get_minimum_size() const;

	bool is_resizing() const { return resizing; }

	GraphNode();
};

VARIANT_ENUM_CAST(GraphNode::Overlay)

#endif // GRAPH_NODE_H
