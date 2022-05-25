/*************************************************************************/
/*  base_graph_node.h                                                    */
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

#ifndef BASE_GRAPH_NODE_H
#define BASE_GRAPH_NODE_H

#include "scene/gui/button.h"
#include "scene/gui/container.h"
#include "scene/resources/text_line.h"

class BaseGraphNode : public Container {
	GDCLASS(BaseGraphNode, Container);

protected:
	String title;
	Ref<TextLine> title_buf;

	Control *titlebar_control = nullptr;
	Button *close_button = nullptr;
	bool show_close = false;
	Rect2 close_rect;

	Dictionary opentype_features;
	String language;
	TextDirection text_direction = TEXT_DIRECTION_AUTO;

	bool selected = false;
	bool resizable = false;
	bool resizing = false;

	Vector2 drag_from;
	Vector2 resizing_from;
	Vector2 resizing_from_size;

	Vector2 position_offset;

	Vector<int> cache_y;

	void _close_requested();
	void _resort_titlebar();
	void _shape_title();

#ifdef TOOLS_ENABLED
	void _edit_set_position(const Point2 &p_position) override;
	void _validate_property(PropertyInfo &property) const override;
#endif

protected:
	virtual void gui_input(const Ref<InputEvent> &p_ev) override;
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void set_title(const String &p_title);
	String get_title() const;

	void set_text_direction(TextDirection p_text_direction);
	TextDirection get_text_direction() const;

	void set_opentype_feature(const String &p_name, int p_value);
	int get_opentype_feature(const String &p_name) const;
	void clear_opentype_features();

	void set_language(const String &p_language);
	String get_language() const;

	void set_position_offset(const Vector2 &p_offset);
	Vector2 get_position_offset() const;

	void set_selected(bool p_selected);
	bool is_selected();

	void set_drag(bool p_drag);
	Vector2 get_drag_from();

	void set_resizable(bool p_enable);
	bool is_resizable() const;

	void set_titlebar_control(Control *p_enable);
	Control *get_titlebar_control() const;

	void set_show_close_button(bool p_enable);
	bool is_close_button_visible() const;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	virtual Size2 get_minimum_size() const override;

	bool is_resizing() const {
		return resizing;
	}

	BaseGraphNode();
};

#endif // BASE_GRAPH_NODE_H
