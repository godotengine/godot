/*************************************************************************/
/*  output_strings.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "output_strings.h"

void OutputStrings::update_scrollbars() {

	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	v_scroll->set_anchor(MARGIN_LEFT, ANCHOR_END);
	v_scroll->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	v_scroll->set_anchor(MARGIN_BOTTOM, ANCHOR_END);

	v_scroll->set_begin(Point2(vmin.width, 0));
	v_scroll->set_end(Point2(0, 0));

	h_scroll->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	h_scroll->set_anchor(MARGIN_TOP, ANCHOR_END);
	h_scroll->set_anchor(MARGIN_BOTTOM, ANCHOR_END);

	h_scroll->set_begin(Point2(0, hmin.y));
	h_scroll->set_end(Point2(vmin.x, 0));

	margin.y = hmin.y;
	margin.x = vmin.x;

	Ref<StyleBox> tree_st = get_stylebox("bg", "Tree");
	int page = ((size_height - (int)margin.y - tree_st->get_margin(MARGIN_TOP)) / font_height);
	v_scroll->set_page(page);
}

void OutputStrings::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_DRAW: {

			if (following) {

				updating = true;
				v_scroll->set_value(v_scroll->get_max() - v_scroll->get_page());
				updating = false;
			}

			RID ci = get_canvas_item();
			Size2 size = get_size();

			Ref<Font> font = get_font("font", "Tree");
			Ref<StyleBox> tree_st = get_stylebox("bg", "Tree");
			tree_st->draw(ci, Rect2(Point2(), size));
			Color color = get_color("font_color", "Tree");
			Ref<Texture> icon_error = get_icon("Error", "EditorIcons");
			Ref<Texture> icon_warning = get_icon("Warning", "EditorIcons");

			//int lines = (size_height-(int)margin.y) / font_height;
			Point2 ofs = tree_st->get_offset();

			LineMap::Element *E = line_map.find(v_scroll->get_value());
			float h_ofs = (int)h_scroll->get_value();
			Point2 icon_ofs = Point2(0, (font_height - (int)icon_error->get_height()) / 2);

			while (E && ofs.y < (size_height - (int)margin.y)) {

				String str = E->get().text;
				Point2 line_ofs = ofs;

				switch (E->get().type) {

					case LINE_WARNING: {
						icon_warning->draw(ci, line_ofs + icon_ofs);

					} break;
					case LINE_ERROR: {
						icon_error->draw(ci, line_ofs + icon_ofs);
					} break;
					case LINE_LINK: {

					} break;
					default: {}
				}

				line_ofs.y += font->get_ascent();
				line_ofs.x += icon_error->get_width() + 4;

				for (int i = 0; i < str.length(); i++) {
					if (line_ofs.x - h_ofs < 0) {
						line_ofs.x += font->get_char_size(str[i], str[i + 1]).width;
					} else if (line_ofs.x - h_ofs > size.width - margin.width) {
						break;
					} else {
						line_ofs.x += font->draw_char(ci, Point2(line_ofs.x - h_ofs, line_ofs.y), str[i], str[i + 1], color);
					}
				}

				ofs.y += font_height;
				E = E->next();
			}

		} break;

		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_RESIZED: {

			font_height = get_font("font", "Tree")->get_height();
			size_height = get_size().height;
			update_scrollbars();
		} break;
	}
}

void OutputStrings::_hscroll_changed(float p_value) {

	if (updating)
		return;

	update();
}
void OutputStrings::_vscroll_changed(float p_value) {

	if (updating)
		return;
	//user changed scroll
	following = (p_value + v_scroll->get_page()) >= v_scroll->get_max();
	update();
}

void OutputStrings::add_line(const String &p_text, const Variant &p_meta, const LineType p_type) {

	Vector<String> strings = p_text.split("\n");

	for (int i = 0; i < strings.size(); i++) {

		if (strings[i].length() == 0)
			continue;

		int last = line_map.empty() ? 0 : (line_map.back()->key() + 1);

		Line l;
		l.text = strings[i];
		l.meta = p_meta;
		l.type = p_type;
		line_map.insert(last, l);

		updating = true;
		v_scroll->set_max(last + 1);
		v_scroll->set_min(line_map.front()->key());
		updating = false;
	}

	while (line_map.size() > line_max_count) {

		line_map.erase(line_map.front());
	}

	update();
}

void OutputStrings::_bind_methods() {

	ClassDB::bind_method("_vscroll_changed", &OutputStrings::_vscroll_changed);
	ClassDB::bind_method("_hscroll_changed", &OutputStrings::_hscroll_changed);
}

OutputStrings::OutputStrings() {

	following = true;
	updating = false;
	line_max_count = 4096;
	h_scroll = memnew(HScrollBar);
	v_scroll = memnew(VScrollBar);
	add_child(h_scroll);
	add_child(v_scroll);
	size_height = 1;
	font_height = 1;
	update_scrollbars();
	h_scroll->connect("value_changed", this, "_hscroll_changed");
	v_scroll->connect("value_changed", this, "_vscroll_changed");
}
