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

struct _MinSizeCache {
	int min_size;
	bool will_stretch;
	int final_size;
};

void GraphFrame::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			Ref<StyleBox> sb_frame = get_theme_stylebox(selected ? SNAME("selected_frame") : SNAME("frame"));
			Ref<StyleBoxFlat> sb_frame_flat = sb_frame;
			Ref<StyleBoxTexture> sb_frame_texture = sb_frame;

			//TODO: [Optimization] These StyleBoxes could be cached eventually.
			if (tint_color_enabled) {
				if (sb_frame_flat.is_valid()) {
					sb_frame_flat = sb_frame_flat->duplicate();
					sb_frame_flat->set_bg_color(tint_color);
					sb_frame_flat->set_border_color(tint_color.darkened(0.2));
					draw_style_box(sb_frame_flat, Rect2(Point2(), get_size()));
				} else if (sb_frame_texture.is_valid()) {
					sb_frame_texture = sb_frame_flat->duplicate();
					sb_frame_texture->set_modulate(tint_color);
					draw_style_box(sb_frame_texture, Rect2(Point2(), get_size()));
				}
			} else {
				draw_style_box(sb_frame_flat, Rect2(Point2(), get_size()));
			}

			Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));
			Color resizer_color = get_theme_color(SNAME("resizer_color"));
			int title_offset = get_theme_constant(SNAME("title_v_offset"));
			int title_h_offset = get_theme_constant(SNAME("title_h_offset"));
			Color title_color = get_theme_color(SNAME("title_color"));

			title_buf->draw(get_canvas_item(), Point2(sb_frame->get_margin(SIDE_LEFT) + title_h_offset, -title_buf->get_size().y + title_offset), title_color);

			if (resizable) {
				draw_texture(resizer, get_size() - resizer->get_size(), resizer_color);
			}
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			_resort();
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_shape_title();

			update_minimum_size();
			queue_redraw();
		} break;
	}
}

void GraphFrame::_resort() {
	Size2 size = get_size();
	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));

	Point2 ofs;
	if (sb_frame.is_valid()) {
		size -= sb_frame->get_minimum_size();
		ofs += sb_frame->get_offset();
	}

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || !c->is_visible_in_tree()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}

		fit_child_in_rect(c, Rect2(ofs, size));
	}
}

void GraphFrame::_shape_title() {
	Ref<Font> font = get_theme_font(SNAME("title_font"));
	int font_size = get_theme_font_size(SNAME("title_font_size"));

	title_buf->clear();
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		title_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		title_buf->set_direction((TextServer::Direction)text_direction);
	}
	title_buf->add_string(title, font, font_size, (!language.is_empty()) ? language : TranslationServer::get_singleton()->get_tool_locale());
}

void GraphFrame::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &GraphFrame::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &GraphFrame::get_title);

	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &GraphFrame::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &GraphFrame::get_text_direction);

	ClassDB::bind_method(D_METHOD("set_language", "language"), &GraphFrame::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &GraphFrame::get_language);

	ClassDB::bind_method(D_METHOD("set_tint_color_enabled", "p_enable"), &GraphFrame::set_tint_color_enabled);
	ClassDB::bind_method(D_METHOD("is_tint_color_enabled"), &GraphFrame::is_tint_color_enabled);

	ClassDB::bind_method(D_METHOD("set_tint_color", "p_color"), &GraphFrame::set_tint_color);
	ClassDB::bind_method(D_METHOD("get_tint_color"), &GraphFrame::get_tint_color);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "tint_color_enabled"), "set_tint_color_enabled", "is_tint_color_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "tint_color"), "set_tint_color", "get_tint_color");

	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");
	ADD_GROUP("", "");
}

void GraphFrame::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	_shape_title();

	queue_redraw();
	update_minimum_size();
}

String GraphFrame::get_title() const {
	return title;
}

void GraphFrame::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		_shape_title();
		queue_redraw();
	}
}

Control::TextDirection GraphFrame::get_text_direction() const {
	return text_direction;
}

void GraphFrame::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape_title();
		queue_redraw();
	}
}

String GraphFrame::get_language() const {
	return language;
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
	Ref<Texture2D> resizer = get_theme_icon(SNAME("resizer"));

	if (Rect2(get_size() - resizer->get_size(), resizer->get_size()).has_point(p_point)) {
		return true;
	}

	if (Rect2(0, 0, get_size().width, frame->get_margin(SIDE_TOP)).has_point(p_point)) {
		return true;
	}

	return false;
}

Size2 GraphFrame::get_minimum_size() const {
	Ref<StyleBox> sb_frame = get_theme_stylebox(SNAME("frame"));

	Size2 minsize;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
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
	title_buf.instantiate();
}
