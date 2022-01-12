/*************************************************************************/
/*  link_button.cpp                                                      */
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

#include "link_button.h"
#include "core/string/translation.h"

void LinkButton::_shape() {
	Ref<Font> font = get_theme_font(SNAME("font"));
	int font_size = get_theme_font_size(SNAME("font_size"));

	text_buf->clear();
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		text_buf->set_direction((TextServer::Direction)text_direction);
	}
	TS->shaped_text_set_bidi_override(text_buf->get_rid(), structured_text_parser(st_parser, st_args, xl_text));
	text_buf->add_string(xl_text, font, font_size, opentype_features, (!language.is_empty()) ? language : TranslationServer::get_singleton()->get_tool_locale());
}

void LinkButton::set_text(const String &p_text) {
	if (text == p_text) {
		return;
	}
	text = p_text;
	xl_text = atr(text);
	_shape();
	update_minimum_size();
	update();
}

String LinkButton::get_text() const {
	return text;
}

void LinkButton::set_structured_text_bidi_override(Control::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		_shape();
		update();
	}
}

Control::StructuredTextParser LinkButton::get_structured_text_bidi_override() const {
	return st_parser;
}

void LinkButton::set_structured_text_bidi_override_options(Array p_args) {
	st_args = p_args;
	_shape();
	update();
}

Array LinkButton::get_structured_text_bidi_override_options() const {
	return st_args;
}

void LinkButton::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		_shape();
		update();
	}
}

Control::TextDirection LinkButton::get_text_direction() const {
	return text_direction;
}

void LinkButton::clear_opentype_features() {
	opentype_features.clear();
	_shape();
	update();
}

void LinkButton::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		_shape();
		update();
	}
}

int LinkButton::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void LinkButton::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape();
		update();
	}
}

String LinkButton::get_language() const {
	return language;
}

void LinkButton::set_underline_mode(UnderlineMode p_underline_mode) {
	underline_mode = p_underline_mode;
	update();
}

LinkButton::UnderlineMode LinkButton::get_underline_mode() const {
	return underline_mode;
}

Size2 LinkButton::get_minimum_size() const {
	return text_buf->get_size();
}

void LinkButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			xl_text = atr(text);
			_shape();

			update_minimum_size();
			update();
		} break;
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			update();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_shape();
			update_minimum_size();
			update();
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2 size = get_size();
			Color color;
			bool do_underline = false;

			switch (get_draw_mode()) {
				case DRAW_NORMAL: {
					if (has_focus()) {
						color = get_theme_color(SNAME("font_focus_color"));
					} else {
						color = get_theme_color(SNAME("font_color"));
					}

					do_underline = underline_mode == UNDERLINE_MODE_ALWAYS;
				} break;
				case DRAW_HOVER_PRESSED:
				case DRAW_PRESSED: {
					if (has_theme_color(SNAME("font_pressed_color"))) {
						color = get_theme_color(SNAME("font_pressed_color"));
					} else {
						color = get_theme_color(SNAME("font_color"));
					}

					do_underline = underline_mode != UNDERLINE_MODE_NEVER;

				} break;
				case DRAW_HOVER: {
					color = get_theme_color(SNAME("font_hover_color"));
					do_underline = underline_mode != UNDERLINE_MODE_NEVER;

				} break;
				case DRAW_DISABLED: {
					color = get_theme_color(SNAME("font_disabled_color"));
					do_underline = underline_mode == UNDERLINE_MODE_ALWAYS;

				} break;
			}

			if (has_focus()) {
				Ref<StyleBox> style = get_theme_stylebox(SNAME("focus"));
				style->draw(ci, Rect2(Point2(), size));
			}

			int width = text_buf->get_line_width();

			Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
			int outline_size = get_theme_constant(SNAME("outline_size"));
			if (is_layout_rtl()) {
				if (outline_size > 0 && font_outline_color.a > 0) {
					text_buf->draw_outline(get_canvas_item(), Vector2(size.width - width, 0), outline_size, font_outline_color);
				}
				text_buf->draw(get_canvas_item(), Vector2(size.width - width, 0), color);
			} else {
				if (outline_size > 0 && font_outline_color.a > 0) {
					text_buf->draw_outline(get_canvas_item(), Vector2(0, 0), outline_size, font_outline_color);
				}
				text_buf->draw(get_canvas_item(), Vector2(0, 0), color);
			}

			if (do_underline) {
				int underline_spacing = get_theme_constant(SNAME("underline_spacing")) + text_buf->get_line_underline_position();
				int y = text_buf->get_line_ascent() + underline_spacing;

				if (is_layout_rtl()) {
					draw_line(Vector2(size.width - width, y), Vector2(size.width, y), color, text_buf->get_line_underline_thickness());
				} else {
					draw_line(Vector2(0, y), Vector2(width, y), color, text_buf->get_line_underline_thickness());
				}
			}

		} break;
	}
}

bool LinkButton::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		int value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				_shape();
				update();
			}
		} else {
			if (!opentype_features.has(tag) || (int)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				_shape();
				update();
			}
		}
		notify_property_list_changed();
		return true;
	}

	return false;
}

bool LinkButton::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		if (opentype_features.has(tag)) {
			r_ret = opentype_features[tag];
			return true;
		} else {
			r_ret = -1;
			return true;
		}
	}
	return false;
}

void LinkButton::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::INT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

void LinkButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &LinkButton::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &LinkButton::get_text);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &LinkButton::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &LinkButton::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &LinkButton::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &LinkButton::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &LinkButton::clear_opentype_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &LinkButton::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &LinkButton::get_language);
	ClassDB::bind_method(D_METHOD("set_underline_mode", "underline_mode"), &LinkButton::set_underline_mode);
	ClassDB::bind_method(D_METHOD("get_underline_mode"), &LinkButton::get_underline_mode);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &LinkButton::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &LinkButton::get_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &LinkButton::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &LinkButton::get_structured_text_bidi_override_options);

	BIND_ENUM_CONSTANT(UNDERLINE_MODE_ALWAYS);
	BIND_ENUM_CONSTANT(UNDERLINE_MODE_ON_HOVER);
	BIND_ENUM_CONSTANT(UNDERLINE_MODE_NEVER);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "underline", PROPERTY_HINT_ENUM, "Always,On Hover,Never"), "set_underline_mode", "get_underline_mode");
	ADD_GROUP("Structured Text", "structured_text_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");
}

LinkButton::LinkButton() {
	text_buf.instantiate();
	set_focus_mode(FOCUS_NONE);
	set_default_cursor_shape(CURSOR_POINTING_HAND);
}
