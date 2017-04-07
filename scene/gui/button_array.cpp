/*************************************************************************/
/*  button_array.cpp                                                     */
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
#include "button_array.h"

bool ButtonArray::_set(const StringName &p_name, const Variant &p_value) {

	String n = String(p_name);
	if (n.begins_with("button/")) {

		String what = n.get_slicec('/', 1);
		if (what == "count") {
			int new_size = p_value;
			if (new_size > 0 && buttons.size() == 0) {
				selected = 0;
			}

			if (new_size < buttons.size()) {
				if (selected >= new_size)
					selected = new_size - 1;
			}
			buttons.resize(new_size);
			_change_notify();
			minimum_size_changed();
		} else if (what == "align") {
			set_align(Align(p_value.operator int()));
		} else if (what == "selected") {
			set_selected(p_value);
		} else if (what == "min_button_size") {
			min_button_size = p_value;
		} else {
			int idx = what.to_int();
			ERR_FAIL_INDEX_V(idx, buttons.size(), false);
			String f = n.get_slicec('/', 2);
			if (f == "text") {
				buttons[idx].text = p_value;
				buttons[idx].xl_text = XL_MESSAGE(p_value);
			} else if (f == "tooltip")
				buttons[idx].tooltip = p_value;
			else if (f == "icon")
				buttons[idx].icon = p_value;
			else
				return false;
		}

		update();
		return true;
	}

	return false;
}

bool ButtonArray::_get(const StringName &p_name, Variant &r_ret) const {

	String n = String(p_name);
	if (n.begins_with("button/")) {

		String what = n.get_slicec('/', 1);
		if (what == "count") {
			r_ret = buttons.size();
		} else if (what == "align") {
			r_ret = get_align();
		} else if (what == "selected") {
			r_ret = get_selected();
		} else if (what == "min_button_size") {
			r_ret = min_button_size;
		} else {
			int idx = what.to_int();
			ERR_FAIL_INDEX_V(idx, buttons.size(), false);
			String f = n.get_slicec('/', 2);
			if (f == "text")
				r_ret = buttons[idx].text;
			else if (f == "tooltip")
				r_ret = buttons[idx].tooltip;
			else if (f == "icon")
				r_ret = buttons[idx].icon;
			else
				return false;
		}

		return true;
	}

	return false;
}
void ButtonArray::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::INT, "button/count", PROPERTY_HINT_RANGE, "0,512,1"));
	p_list->push_back(PropertyInfo(Variant::INT, "button/min_button_size", PROPERTY_HINT_RANGE, "0,1024,1"));
	p_list->push_back(PropertyInfo(Variant::INT, "button/align", PROPERTY_HINT_ENUM, "Begin,Center,End,Fill,Expand"));
	for (int i = 0; i < buttons.size(); i++) {
		String base = "button/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, base + "text"));
		p_list->push_back(PropertyInfo(Variant::STRING, base + "tooltip"));
		p_list->push_back(PropertyInfo(Variant::OBJECT, base + "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture"));
	}
	if (buttons.size() > 0) {
		p_list->push_back(PropertyInfo(Variant::INT, "button/selected", PROPERTY_HINT_RANGE, "0," + itos(buttons.size() - 1) + ",1"));
	}
}

Size2 ButtonArray::get_minimum_size() const {

	Ref<StyleBox> style_normal = get_stylebox("normal");
	Ref<StyleBox> style_selected = get_stylebox("selected");
	Ref<Font> font_normal = get_font("font");
	Ref<Font> font_selected = get_font("font_selected");
	int icon_sep = get_constant("icon_separator");
	int button_sep = get_constant("button_separator");

	Size2 minsize;

	for (int i = 0; i < buttons.size(); i++) {

		Ref<StyleBox> sb = i == selected ? style_selected : style_normal;
		Ref<Font> f = i == selected ? font_selected : font_normal;

		Size2 ms;
		ms = f->get_string_size(buttons[i].xl_text);
		if (buttons[i].icon.is_valid()) {

			Size2 bs = buttons[i].icon->get_size();
			ms.height = MAX(ms.height, bs.height);
			ms.width += bs.width + icon_sep;
		}

		ms += sb->get_minimum_size();

		buttons[i]._ms_cache = ms[orientation];

		minsize[orientation] += ms[orientation];
		if (i > 0)
			minsize[orientation] += button_sep;
		minsize[!orientation] = MAX(minsize[!orientation], ms[!orientation]);
	}

	return minsize;
}

void ButtonArray::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_MOUSE_EXIT: {
			hover = -1;
			update();
		} break;
		case NOTIFICATION_READY: {
			MethodInfo mi;
			mi.name = "mouse_sub_enter";

			add_user_signal(mi);

		} break;
		case NOTIFICATION_DRAW: {

			Size2 size = get_size();
			Size2 minsize = get_combined_minimum_size();
			Ref<StyleBox> style_normal = get_stylebox("normal");
			Ref<StyleBox> style_selected = get_stylebox("selected");
			Ref<StyleBox> style_focus = get_stylebox("focus");
			Ref<StyleBox> style_hover = get_stylebox("hover");
			Ref<Font> font_normal = get_font("font");
			Ref<Font> font_selected = get_font("font_selected");
			int icon_sep = get_constant("icon_separator");
			int button_sep = get_constant("button_separator");
			Color color_normal = get_color("font_color");
			Color color_selected = get_color("font_color_selected");

			int sep = button_sep;
			int ofs = 0;
			int expand = 0;

			switch (align) {
				case ALIGN_BEGIN: {

					ofs = 0;
				} break;
				case ALIGN_CENTER: {

					ofs = Math::floor((size[orientation] - minsize[orientation]) / 2);
				} break;
				case ALIGN_END: {

					ofs = Math::floor((size[orientation] - minsize[orientation]));
				} break;
				case ALIGN_FILL: {

					if (buttons.size() > 1)
						sep += Math::floor((size[orientation] - minsize[orientation]) / (buttons.size() - 1.0));
					ofs = 0;
				} break;
				case ALIGN_EXPAND_FILL: {

					ofs = 0;
					expand = size[orientation] - minsize[orientation];
				} break;
			}

			int op_size = orientation == VERTICAL ? size.width : size.height;

			for (int i = 0; i < buttons.size(); i++) {

				int ms = buttons[i]._ms_cache;
				int s = ms;
				if (expand > 0) {
					s += expand / buttons.size();
				}
				if (min_button_size != -1 && s < min_button_size) {
					s = min_button_size;
				}

				Rect2 r;
				r.pos[orientation] = ofs;
				r.pos[!orientation] = 0;
				r.size[orientation] = s;
				r.size[!orientation] = op_size;

				Ref<Font> f;
				Color c;
				Point2 sbsize;
				Point2 sbofs;
				if (i == selected) {
					draw_style_box(style_selected, r);
					sbsize = style_selected->get_minimum_size();
					sbofs = style_selected->get_offset();
					f = font_selected;
					c = color_selected;
					if (has_focus())
						draw_style_box(style_focus, r);
				} else {
					if (hover == i)
						draw_style_box(style_hover, r);
					else if (!flat)
						draw_style_box(style_normal, r);
					sbsize = style_normal->get_minimum_size();
					sbofs = style_normal->get_offset();
					f = font_normal;
					c = color_normal;
				}

				Size2 ssize = f->get_string_size(buttons[i].xl_text);
				if (buttons[i].icon.is_valid()) {

					ssize.x += buttons[i].icon->get_width();
				}
				Point2 text_ofs = ((r.size - ssize - sbsize) / 2.0 + Point2(0, f->get_ascent())).floor() + sbofs;
				if (buttons[i].icon.is_valid()) {

					draw_texture(buttons[i].icon, r.pos + Point2(text_ofs.x, Math::floor((r.size.height - buttons[i].icon->get_height()) / 2.0)));
					text_ofs.x += buttons[i].icon->get_width() + icon_sep;
				}
				draw_string(f, text_ofs + r.pos, buttons[i].xl_text, c);
				buttons[i]._pos_cache = ofs;
				buttons[i]._size_cache = s;

				ofs += s;
				ofs += sep;
			}

		} break;
	}
}

void ButtonArray::_gui_input(const InputEvent &p_event) {

	if (
			((orientation == HORIZONTAL && p_event.is_action("ui_left")) ||
					(orientation == VERTICAL && p_event.is_action("ui_up"))) &&
			p_event.is_pressed() && selected > 0) {
		set_selected(selected - 1);
		accept_event();
		emit_signal("button_selected", selected);
		return;
	}

	if (
			((orientation == HORIZONTAL && p_event.is_action("ui_right")) ||
					(orientation == VERTICAL && p_event.is_action("ui_down"))) &&
			p_event.is_pressed() && selected < (buttons.size() - 1)) {
		set_selected(selected + 1);
		accept_event();
		emit_signal("button_selected", selected);
		return;
	}

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.pressed && p_event.mouse_button.button_index == BUTTON_LEFT) {

		int ofs = orientation == HORIZONTAL ? p_event.mouse_button.x : p_event.mouse_button.y;

		for (int i = 0; i < buttons.size(); i++) {

			if (ofs >= buttons[i]._pos_cache && ofs < buttons[i]._pos_cache + buttons[i]._size_cache) {

				set_selected(i);
				emit_signal("button_selected", i);
				return;
			}
		}
	}

	if (p_event.type == InputEvent::MOUSE_MOTION) {

		int ofs = orientation == HORIZONTAL ? p_event.mouse_motion.x : p_event.mouse_motion.y;
		int new_hover = -1;
		for (int i = 0; i < buttons.size(); i++) {

			if (ofs >= buttons[i]._pos_cache && ofs < buttons[i]._pos_cache + buttons[i]._size_cache) {

				new_hover = i;
				break;
			}
		}

		if (new_hover != hover) {
			hover = new_hover;
			emit_signal("mouse_sub_enter");
			update();
		}
	}
}

String ButtonArray::get_tooltip(const Point2 &p_pos) const {

	int ofs = orientation == HORIZONTAL ? p_pos.x : p_pos.y;
	for (int i = 0; i < buttons.size(); i++) {

		if (ofs >= buttons[i]._pos_cache && ofs < buttons[i]._pos_cache + buttons[i]._size_cache)
			return buttons[i].tooltip;
	}
	return Control::get_tooltip(p_pos);
}

void ButtonArray::set_align(Align p_align) {

	align = p_align;
	update();
}

ButtonArray::Align ButtonArray::get_align() const {

	return align;
}

void ButtonArray::set_flat(bool p_flat) {

	flat = p_flat;
	update();
}

bool ButtonArray::is_flat() const {

	return flat;
}

void ButtonArray::add_button(const String &p_text, const String &p_tooltip) {

	Button button;
	button.text = p_text;
	button.xl_text = XL_MESSAGE(p_text);
	button.tooltip = p_tooltip;
	buttons.push_back(button);
	update();

	if (selected == -1)
		selected = 0;

	minimum_size_changed();
}

void ButtonArray::add_icon_button(const Ref<Texture> &p_icon, const String &p_text, const String &p_tooltip) {

	Button button;
	button.text = p_text;
	button.xl_text = XL_MESSAGE(p_text);
	button.icon = p_icon;
	button.tooltip = p_tooltip;
	buttons.push_back(button);
	if (selected == -1)
		selected = 0;

	update();
}

void ButtonArray::set_button_text(int p_button, const String &p_text) {

	ERR_FAIL_INDEX(p_button, buttons.size());
	buttons[p_button].text = p_text;
	buttons[p_button].xl_text = XL_MESSAGE(p_text);
	update();
	minimum_size_changed();
}

void ButtonArray::set_button_tooltip(int p_button, const String &p_text) {

	ERR_FAIL_INDEX(p_button, buttons.size());
	buttons[p_button].tooltip = p_text;
}

void ButtonArray::set_button_icon(int p_button, const Ref<Texture> &p_icon) {

	ERR_FAIL_INDEX(p_button, buttons.size());
	buttons[p_button].icon = p_icon;
	update();
	minimum_size_changed();
}

String ButtonArray::get_button_text(int p_button) const {

	ERR_FAIL_INDEX_V(p_button, buttons.size(), "");
	return buttons[p_button].text;
}

String ButtonArray::get_button_tooltip(int p_button) const {

	ERR_FAIL_INDEX_V(p_button, buttons.size(), "");
	return buttons[p_button].tooltip;
}

Ref<Texture> ButtonArray::get_button_icon(int p_button) const {

	ERR_FAIL_INDEX_V(p_button, buttons.size(), Ref<Texture>());
	return buttons[p_button].icon;
}

int ButtonArray::get_selected() const {

	return selected;
}

int ButtonArray::get_hovered() const {

	return hover;
}

void ButtonArray::set_selected(int p_selected) {

	ERR_FAIL_INDEX(p_selected, buttons.size());
	selected = p_selected;
	update();
}

void ButtonArray::erase_button(int p_button) {

	ERR_FAIL_INDEX(p_button, buttons.size());
	buttons.remove(p_button);
	if (p_button >= selected)
		selected--;
	if (selected < 0)
		selected = 0;
	if (selected >= buttons.size())
		selected = buttons.size() - 1;

	update();
}

void ButtonArray::clear() {

	buttons.clear();
	selected = -1;
	update();
}

int ButtonArray::get_button_count() const {

	return buttons.size();
}

void ButtonArray::get_translatable_strings(List<String> *p_strings) const {

	for (int i = 0; i < buttons.size(); i++) {
		p_strings->push_back(buttons[i].text);
		p_strings->push_back(buttons[i].tooltip);
	}
}

void ButtonArray::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_button", "text", "tooltip"), &ButtonArray::add_button, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_icon_button", "icon:Texture", "text", "tooltip"), &ButtonArray::add_icon_button, DEFVAL(""), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("set_button_text", "button_idx", "text"), &ButtonArray::set_button_text);
	ClassDB::bind_method(D_METHOD("set_button_tooltip", "button_idx", "text"), &ButtonArray::set_button_tooltip);
	ClassDB::bind_method(D_METHOD("set_button_icon", "button_idx", "icon:Texture"), &ButtonArray::set_button_icon);
	ClassDB::bind_method(D_METHOD("get_button_text", "button_idx"), &ButtonArray::get_button_text);
	ClassDB::bind_method(D_METHOD("get_button_tooltip", "button_idx"), &ButtonArray::get_button_tooltip);
	ClassDB::bind_method(D_METHOD("get_button_icon:Texture", "button_idx"), &ButtonArray::get_button_icon);
	ClassDB::bind_method(D_METHOD("get_button_count"), &ButtonArray::get_button_count);
	ClassDB::bind_method(D_METHOD("set_flat", "enabled"), &ButtonArray::set_flat);
	ClassDB::bind_method(D_METHOD("is_flat"), &ButtonArray::is_flat);
	ClassDB::bind_method(D_METHOD("get_selected"), &ButtonArray::get_selected);
	ClassDB::bind_method(D_METHOD("get_hovered"), &ButtonArray::get_hovered);
	ClassDB::bind_method(D_METHOD("set_selected", "button_idx"), &ButtonArray::set_selected);
	ClassDB::bind_method(D_METHOD("erase_button", "button_idx"), &ButtonArray::erase_button);
	ClassDB::bind_method(D_METHOD("clear"), &ButtonArray::clear);

	ClassDB::bind_method(D_METHOD("_gui_input"), &ButtonArray::_gui_input);

	BIND_CONSTANT(ALIGN_BEGIN);
	BIND_CONSTANT(ALIGN_CENTER);
	BIND_CONSTANT(ALIGN_END);
	BIND_CONSTANT(ALIGN_FILL);
	BIND_CONSTANT(ALIGN_EXPAND_FILL);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");

	ADD_SIGNAL(MethodInfo("button_selected", PropertyInfo(Variant::INT, "button_idx")));
}

ButtonArray::ButtonArray(Orientation p_orientation) {

	orientation = p_orientation;
	selected = -1;
	set_focus_mode(FOCUS_ALL);
	hover = -1;
	flat = false;
	min_button_size = -1;
}
