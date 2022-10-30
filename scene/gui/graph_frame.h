/**************************************************************************/
/*  graph_frame.h                                                         */
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

#ifndef GRAPH_FRAME_H
#define GRAPH_FRAME_H

#include "scene/gui/graph_control.h"

class HBoxContainer;

class GraphFrame : public GraphControl {
	GDCLASS(GraphFrame, GraphControl);

private:
	String title;

	HBoxContainer *titlebar_hbox;
	Label *title_label;

	bool title_centered = true;

	int drag_margin = 12;

	bool tint_color_enabled = false;
	Color tint_color = Color(0.4, 0.8, 0.4);

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _resort() override;

public:
	void set_title(const String &p_title);
	String get_title() const;

	void set_title_centered(bool p_centered);
	bool is_title_centered() const;

	HBoxContainer *get_titlebar_hbox();

	void set_drag_margin(int p_margin);
	int get_drag_margin() const;

	void set_tint_color_enabled(bool p_enable);
	bool is_tint_color_enabled() const;

	void set_tint_color(const Color &p_tint_color);
	Color get_tint_color() const;

	virtual bool has_point(const Point2 &p_point) const override;
	virtual Size2 get_minimum_size() const override;

	GraphFrame();
};

#endif // GRAPH_FRAME_H
