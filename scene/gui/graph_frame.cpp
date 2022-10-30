/**************************************************************************/
/*  graph_frame.cpp                                                       */
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

#include "graph_frame.h"

#include "core/string/translation.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"

struct _MinSizeCache {
	int min_size;
	bool will_stretch;
	int final_size;
};

void GraphFrame::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			// Used for layout calculations.
			Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));
			Ref<StyleBox> sb_titlebar = get_theme_stylebox(SNAME("titlebar"));

			// Used for drawing.
			Ref<StyleBox> sb_to_draw_frame = get_theme_stylebox(selected ? SNAME("frame_selected") : SNAME("frame"));
			Ref<StyleBox> sb_to_draw_titlebar = get_theme_stylebox(selected ? SNAME("titlebar_selected") : SNAME("titlebar"));
			Ref<StyleBoxFlat> sb_frame_flat = sb_to_draw_frame;
			Ref<StyleBoxTexture> sb_frame_texture = sb_to_draw_frame;

			Rect2 titlebar_rect(Point2(), titlebar_hbox->get_size() + sb_titlebar->get_minimum_size());
			Size2 body_size = get_size();
			body_size.y -= titlebar_rect.size.height;
			Rect2 body_rect(0, titlebar_rect.size.height, body_size.width, body_size.height);

			// Draw body stylebox.
			//TODO: [Optimization] These StyleBoxes could be cached eventually.
			if (tint_color_enabled) {
				if (sb_frame_flat.is_valid()) {
					sb_frame_flat = sb_frame_flat->duplicate();
					sb_frame_flat->set_bg_color(tint_color);
					sb_frame_flat->set_border_color(tint_color.lightened(0.3));
					draw_style_box(sb_frame_flat, body_rect);
				} else if (sb_frame_texture.is_valid()) {
					sb_frame_texture = sb_frame_flat->duplicate();
					sb_frame_texture->set_modulate(tint_color);
					draw_style_box(sb_frame_texture, body_rect);
				}
			} else {
				draw_style_box(sb_frame_flat, body_rect);
			}

			// Draw title bar stylebox above.
			draw_style_box(sb_to_draw_titlebar, titlebar_rect);

			Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));
			Color resizer_color = get_theme_color(SNAME("resizer_color"));
			if (resizable) {
				draw_texture(resizer, get_size() - resizer->get_size(), resizer_color);
			}
		} break;
	}
}

void GraphFrame::_resort() {
	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));
	Ref<StyleBox> sb_titlebar = get_theme_stylebox(SNAME("titlebar"));

	// Resort titlebar first.
	Size2 titlebar_size = Size2(get_size().width, titlebar_hbox->get_size().height);
	titlebar_size -= sb_titlebar->get_minimum_size();
	Rect2 titlebar_rect = Rect2(sb_titlebar->get_offset(), titlebar_size);

	fit_child_in_rect(titlebar_hbox, titlebar_rect);

	// After resort the children of the titlebar container may have changed their height (e.g. Label autowrap).
	Size2i titlebar_min_size = titlebar_hbox->get_combined_minimum_size();

	Size2 size = get_size() - sb_frame->get_minimum_size() - Size2(0, titlebar_min_size.height + sb_titlebar->get_minimum_size().height);
	Point2 offset = Point2(sb_frame->get_margin(SIDE_LEFT), sb_frame->get_margin(SIDE_TOP) + titlebar_min_size.height + sb_titlebar->get_minimum_size().height);

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c || !c->is_visible_in_tree()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		fit_child_in_rect(c, Rect2(offset, size));
	}
}

void GraphFrame::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &GraphFrame::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &GraphFrame::get_title);

	ClassDB::bind_method(D_METHOD("get_titlebar_hbox"), &GraphFrame::get_titlebar_hbox);

	ClassDB::bind_method(D_METHOD("set_title_centered", "centered"), &GraphFrame::set_title_centered);
	ClassDB::bind_method(D_METHOD("is_title_centered"), &GraphFrame::is_title_centered);

	ClassDB::bind_method(D_METHOD("set_drag_margin", "drag_margin"), &GraphFrame::set_drag_margin);
	ClassDB::bind_method(D_METHOD("get_drag_margin"), &GraphFrame::get_drag_margin);

	ClassDB::bind_method(D_METHOD("set_tint_color_enabled", "p_enable"), &GraphFrame::set_tint_color_enabled);
	ClassDB::bind_method(D_METHOD("is_tint_color_enabled"), &GraphFrame::is_tint_color_enabled);

	ClassDB::bind_method(D_METHOD("set_tint_color", "p_color"), &GraphFrame::set_tint_color);
	ClassDB::bind_method(D_METHOD("get_tint_color"), &GraphFrame::get_tint_color);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "title_centered"), "set_title_centered", "is_title_centered");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "drag_margin", PROPERTY_HINT_RANGE, "0,128,1"), "set_drag_margin", "get_drag_margin");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "tint_color_enabled"), "set_tint_color_enabled", "is_tint_color_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "tint_color"), "set_tint_color", "get_tint_color");
}

void GraphFrame::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	if (title_label) {
		title_label->set_text(title);
	}
	update_minimum_size();
}

String GraphFrame::get_title() const {
	return title;
}

void GraphFrame::set_title_centered(bool p_centered) {
	if (title_centered == p_centered) {
		return;
	}
	title_centered = p_centered;
	if (title_label != nullptr) {
		title_label->set_horizontal_alignment(title_centered ? HORIZONTAL_ALIGNMENT_CENTER : HORIZONTAL_ALIGNMENT_LEFT);
	}
}

bool GraphFrame::is_title_centered() const {
	return title_centered;
}

HBoxContainer *GraphFrame::get_titlebar_hbox() {
	return titlebar_hbox;
}

void GraphFrame::set_drag_margin(int p_margin) {
	drag_margin = p_margin;
}

int GraphFrame::get_drag_margin() const {
	return drag_margin;
}

void GraphFrame::set_tint_color_enabled(bool p_enable) {
	tint_color_enabled = p_enable;
	queue_redraw();
}

bool GraphFrame::is_tint_color_enabled() const {
	return tint_color_enabled;
}

void GraphFrame::set_tint_color(const Color &p_color) {
	tint_color = p_color;
	queue_redraw();
}

Color GraphFrame::get_tint_color() const {
	return tint_color;
}

bool GraphFrame::has_point(const Point2 &p_point) const {
	Ref<StyleBox> frame = get_theme_stylebox(SNAME("frame"));
	Ref<StyleBox> titlebar = get_theme_stylebox(SNAME("titlebar"));
	Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));

	if (Rect2(get_size() - resizer->get_size(), resizer->get_size()).has_point(p_point)) {
		return true;
	}

	// For grabbing on the titlebar.
	int titlebar_height = titlebar_hbox->get_size().height + titlebar->get_minimum_size().height;
	if (Rect2(0, 0, get_size().width, titlebar_height).has_point(p_point)) {
		return true;
	}

	// Allow grabbing on all sides of the frame.
	Rect2 frame_rect = Rect2(0, 0, get_size().width, get_size().height);
	Rect2 no_drag_rect = frame_rect.grow(-drag_margin);

	if (frame_rect.has_point(p_point) && !no_drag_rect.has_point(p_point)) {
		return true;
	}

	return false;
}

Size2 GraphFrame::get_minimum_size() const {
	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));
	Ref<StyleBox> sb_titlebar = get_theme_stylebox(SNAME("titlebar"));

	Size2 minsize = titlebar_hbox->get_minimum_size() + sb_titlebar->get_minimum_size();

	for (int i = 0; i < get_child_count(false); i++) {
		Control *c = Object::cast_to<Control>(get_child(i, false));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		Size2i size = c->get_combined_minimum_size();
		size.width += sb_frame->get_minimum_size().width;

		minsize.x = MAX(minsize.x, size.x);
		minsize.y += MAX(minsize.y, size.y);
	}

	minsize.height += sb_frame->get_minimum_size().height;

	return minsize;
}

GraphFrame::GraphFrame() {
	titlebar_hbox = memnew(HBoxContainer);
	titlebar_hbox->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(titlebar_hbox, false, INTERNAL_MODE_FRONT);

	title_label = memnew(Label);
	title_label->set_theme_type_variation("GraphFrameTitleLabel");
	title_label->set_h_size_flags(SIZE_EXPAND_FILL);
	title_label->set_horizontal_alignment(title_centered ? HORIZONTAL_ALIGNMENT_CENTER : HORIZONTAL_ALIGNMENT_LEFT);
	titlebar_hbox->add_child(title_label);

	set_mouse_filter(MOUSE_FILTER_STOP);
}
