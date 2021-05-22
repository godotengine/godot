/*************************************************************************/
/*  button.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "button.h"

#include "core/string/translation.h"
#include "servers/rendering_server.h"

Size2 Button::get_minimum_size() const {
	Size2 minsize = text_buf->get_size();
	if (clip_text) {
		minsize.width = 0;
	}

	if (!expand_icon) {
		Ref<Texture2D> _icon;
		if (icon.is_null() && has_theme_icon("icon")) {
			_icon = Control::get_theme_icon("icon");
		} else {
			_icon = icon;
		}

		if (!_icon.is_null()) {
			minsize.height = MAX(minsize.height, _icon->get_height());
			minsize.width += _icon->get_width();
			if (xl_text != "") {
				minsize.width += get_theme_constant("hseparation");
			}
		}
	}

	Ref<Font> font = get_theme_font("font");
	float font_height = font->get_height(get_theme_font_size("font_size"));

	minsize.height = MAX(font_height, minsize.height);

	return get_theme_stylebox("normal")->get_minimum_size() + minsize;
}

void Button::_set_internal_margin(Side p_side, float p_value) {
	_internal_margin[p_side] = p_value;
}

void Button::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			update();
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED: {
			xl_text = tr(text);
			_shape();

			minimum_size_changed();
			update();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_shape();

			minimum_size_changed();
			update();
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2 size = get_size();
			Color color;
			Color color_icon(1, 1, 1, 1);

			Ref<StyleBox> style = get_theme_stylebox("normal");
			bool rtl = is_layout_rtl();

			switch (get_draw_mode()) {
				case DRAW_NORMAL: {
					if (rtl && has_theme_stylebox("normal_mirrored")) {
						style = get_theme_stylebox("normal_mirrored");
					} else {
						style = get_theme_stylebox("normal");
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = get_theme_color("font_color");
					if (has_theme_color("icon_normal_color")) {
						color_icon = get_theme_color("icon_normal_color");
					}
				} break;
				case DRAW_HOVER_PRESSED: {
					if (has_theme_stylebox("hover_pressed") && has_theme_stylebox_override("hover_pressed")) {
						if (rtl && has_theme_stylebox("hover_pressed_mirrored")) {
							style = get_theme_stylebox("hover_pressed_mirrored");
						} else {
							style = get_theme_stylebox("hover_pressed");
						}

						if (!flat) {
							style->draw(ci, Rect2(Point2(0, 0), size));
						}
						if (has_theme_color("font_hover_pressed_color")) {
							color = get_theme_color("font_hover_pressed_color");
						} else {
							color = get_theme_color("font_color");
						}
						if (has_theme_color("icon_hover_pressed_color")) {
							color_icon = get_theme_color("icon_hover_pressed_color");
						}

						break;
					}
					[[fallthrough]];
				}
				case DRAW_PRESSED: {
					if (rtl && has_theme_stylebox("pressed_mirrored")) {
						style = get_theme_stylebox("pressed_mirrored");
					} else {
						style = get_theme_stylebox("pressed");
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					if (has_theme_color("font_pressed_color")) {
						color = get_theme_color("font_pressed_color");
					} else {
						color = get_theme_color("font_color");
					}
					if (has_theme_color("icon_pressed_color")) {
						color_icon = get_theme_color("icon_pressed_color");
					}

				} break;
				case DRAW_HOVER: {
					if (rtl && has_theme_stylebox("hover_mirrored")) {
						style = get_theme_stylebox("hover_mirrored");
					} else {
						style = get_theme_stylebox("hover");
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = get_theme_color("font_hover_color");
					if (has_theme_color("icon_hover_color")) {
						color_icon = get_theme_color("icon_hover_color");
					}

				} break;
				case DRAW_DISABLED: {
					if (rtl && has_theme_stylebox("disabled_mirrored")) {
						style = get_theme_stylebox("disabled_mirrored");
					} else {
						style = get_theme_stylebox("disabled");
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = get_theme_color("font_disabled_color");
					if (has_theme_color("icon_disabled_color")) {
						color_icon = get_theme_color("icon_disabled_color");
					}

				} break;
			}

			if (has_focus()) {
				Ref<StyleBox> style2 = get_theme_stylebox("focus");
				style2->draw(ci, Rect2(Point2(), size));
			}

			Ref<Texture2D> _icon;
			if (icon.is_null() && has_theme_icon("icon")) {
				_icon = Control::get_theme_icon("icon");
			} else {
				_icon = icon;
			}

			Rect2 icon_region = Rect2();
			if (!_icon.is_null()) {
				int valign = size.height - style->get_minimum_size().y;
				if (is_disabled()) {
					color_icon.a = 0.4;
				}

				float icon_ofs_region = 0.0;
				if (rtl) {
					if (_internal_margin[SIDE_RIGHT] > 0) {
						icon_ofs_region = _internal_margin[SIDE_RIGHT] + get_theme_constant("hseparation");
					}
				} else {
					if (_internal_margin[SIDE_LEFT] > 0) {
						icon_ofs_region = _internal_margin[SIDE_LEFT] + get_theme_constant("hseparation");
					}
				}

				if (expand_icon) {
					Size2 _size = get_size() - style->get_offset() * 2;
					_size.width -= get_theme_constant("hseparation") + icon_ofs_region;
					if (!clip_text) {
						_size.width -= text_buf->get_size().width;
					}
					float icon_width = _icon->get_width() * _size.height / _icon->get_height();
					float icon_height = _size.height;

					if (icon_width > _size.width) {
						icon_width = _size.width;
						icon_height = _icon->get_height() * icon_width / _icon->get_width();
					}

					if (rtl) {
						icon_region = Rect2(Point2(size.width - (icon_ofs_region + icon_width + style->get_margin(SIDE_RIGHT)), style->get_margin(SIDE_TOP) + (_size.height - icon_height) / 2), Size2(icon_width, icon_height));
					} else {
						icon_region = Rect2(style->get_offset() + Point2(icon_ofs_region, (_size.height - icon_height) / 2), Size2(icon_width, icon_height));
					}
				} else {
					if (rtl) {
						icon_region = Rect2(Point2(size.width - (icon_ofs_region + _icon->get_size().width + style->get_margin(SIDE_RIGHT)), style->get_margin(SIDE_TOP) + Math::floor((valign - _icon->get_height()) / 2.0)), _icon->get_size());
					} else {
						icon_region = Rect2(style->get_offset() + Point2(icon_ofs_region, Math::floor((valign - _icon->get_height()) / 2.0)), _icon->get_size());
					}
				}
			}

			Point2 icon_ofs = !_icon.is_null() ? Point2(icon_region.size.width + get_theme_constant("hseparation"), 0) : Point2();
			int text_clip = size.width - style->get_minimum_size().width - icon_ofs.width;
			text_buf->set_width(clip_text ? text_clip : -1);

			int text_width = clip_text ? MIN(text_clip, text_buf->get_size().x) : text_buf->get_size().x;

			if (_internal_margin[SIDE_LEFT] > 0) {
				text_clip -= _internal_margin[SIDE_LEFT] + get_theme_constant("hseparation");
			}
			if (_internal_margin[SIDE_RIGHT] > 0) {
				text_clip -= _internal_margin[SIDE_RIGHT] + get_theme_constant("hseparation");
			}

			Point2 text_ofs = (size - style->get_minimum_size() - icon_ofs - text_buf->get_size() - Point2(_internal_margin[SIDE_RIGHT] - _internal_margin[SIDE_LEFT], 0)) / 2.0;

			switch (align) {
				case ALIGN_LEFT: {
					if (rtl) {
						if (_internal_margin[SIDE_RIGHT] > 0) {
							text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width - _internal_margin[SIDE_RIGHT] - get_theme_constant("hseparation");
						} else {
							text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width;
						}
					} else {
						if (_internal_margin[SIDE_LEFT] > 0) {
							text_ofs.x = style->get_margin(SIDE_LEFT) + icon_ofs.x + _internal_margin[SIDE_LEFT] + get_theme_constant("hseparation");
						} else {
							text_ofs.x = style->get_margin(SIDE_LEFT) + icon_ofs.x;
						}
					}
					text_ofs.y += style->get_offset().y;
				} break;
				case ALIGN_CENTER: {
					if (text_ofs.x < 0) {
						text_ofs.x = 0;
					}
					text_ofs += icon_ofs;
					text_ofs += style->get_offset();
				} break;
				case ALIGN_RIGHT: {
					if (rtl) {
						if (_internal_margin[SIDE_LEFT] > 0) {
							text_ofs.x = style->get_margin(SIDE_LEFT) + icon_ofs.x + _internal_margin[SIDE_LEFT] + get_theme_constant("hseparation");
						} else {
							text_ofs.x = style->get_margin(SIDE_LEFT) + icon_ofs.x;
						}
					} else {
						if (_internal_margin[SIDE_RIGHT] > 0) {
							text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width - _internal_margin[SIDE_RIGHT] - get_theme_constant("hseparation");
						} else {
							text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width;
						}
					}
					text_ofs.y += style->get_offset().y;
				} break;
			}

			if (rtl) {
				text_ofs.x -= icon_ofs.x;
			}

			Color font_outline_color = get_theme_color("font_outline_color");
			int outline_size = get_theme_constant("outline_size");
			if (outline_size > 0 && font_outline_color.a > 0) {
				text_buf->draw_outline(ci, text_ofs, outline_size, font_outline_color);
			}

			text_buf->draw(ci, text_ofs, color);

			if (!_icon.is_null() && icon_region.size.width > 0) {
				draw_texture_rect_region(_icon, icon_region, Rect2(Point2(), _icon->get_size()), color_icon);
			}
		} break;
	}
}

void Button::_shape() {
	Ref<Font> font = get_theme_font("font");
	int font_size = get_theme_font_size("font_size");

	text_buf->clear();
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		text_buf->set_direction((TextServer::Direction)text_direction);
	}
	text_buf->add_string(xl_text, font, font_size, opentype_features, (language != "") ? language : TranslationServer::get_singleton()->get_tool_locale());
}

void Button::set_text(const String &p_text) {
	if (text != p_text) {
		text = p_text;
		xl_text = tr(text);
		_shape();

		update();
		minimum_size_changed();
	}
}

String Button::get_text() const {
	return text;
}

void Button::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		_shape();
		update();
	}
}

Control::TextDirection Button::get_text_direction() const {
	return text_direction;
}

void Button::clear_opentype_features() {
	opentype_features.clear();
	_shape();
	update();
}

void Button::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		_shape();
		update();
	}
}

int Button::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void Button::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape();
		update();
	}
}

String Button::get_language() const {
	return language;
}

void Button::set_icon(const Ref<Texture2D> &p_icon) {
	if (icon != p_icon) {
		icon = p_icon;
		update();
		minimum_size_changed();
	}
}

Ref<Texture2D> Button::get_icon() const {
	return icon;
}

void Button::set_expand_icon(bool p_expand_icon) {
	if (expand_icon != p_expand_icon) {
		expand_icon = p_expand_icon;
		update();
		minimum_size_changed();
	}
}

bool Button::is_expand_icon() const {
	return expand_icon;
}

void Button::set_flat(bool p_flat) {
	if (flat != p_flat) {
		flat = p_flat;
		update();
	}
}

bool Button::is_flat() const {
	return flat;
}

void Button::set_clip_text(bool p_clip_text) {
	if (clip_text != p_clip_text) {
		clip_text = p_clip_text;
		update();
		minimum_size_changed();
	}
}

bool Button::get_clip_text() const {
	return clip_text;
}

void Button::set_text_align(TextAlign p_align) {
	if (align != p_align) {
		align = p_align;
		update();
	}
}

Button::TextAlign Button::get_text_align() const {
	return align;
}

bool Button::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		double value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				_shape();
				update();
			}
		} else {
			if ((double)opentype_features[tag] != value) {
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

bool Button::_get(const StringName &p_name, Variant &r_ret) const {
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

void Button::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::FLOAT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

void Button::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &Button::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Button::get_text);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Button::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Button::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &Button::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &Button::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &Button::clear_opentype_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &Button::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &Button::get_language);
	ClassDB::bind_method(D_METHOD("set_button_icon", "texture"), &Button::set_icon);
	ClassDB::bind_method(D_METHOD("get_button_icon"), &Button::get_icon);
	ClassDB::bind_method(D_METHOD("set_expand_icon"), &Button::set_expand_icon);
	ClassDB::bind_method(D_METHOD("is_expand_icon"), &Button::is_expand_icon);
	ClassDB::bind_method(D_METHOD("set_flat", "enabled"), &Button::set_flat);
	ClassDB::bind_method(D_METHOD("set_clip_text", "enabled"), &Button::set_clip_text);
	ClassDB::bind_method(D_METHOD("get_clip_text"), &Button::get_clip_text);
	ClassDB::bind_method(D_METHOD("set_text_align", "align"), &Button::set_text_align);
	ClassDB::bind_method(D_METHOD("get_text_align"), &Button::get_text_align);
	ClassDB::bind_method(D_METHOD("is_flat"), &Button::is_flat);

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_button_icon", "get_button_icon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "get_clip_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_text_align", "get_text_align");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expand_icon"), "set_expand_icon", "is_expand_icon");
}

Button::Button(const String &p_text) {
	text_buf.instance();
	text_buf->set_flags(TextServer::BREAK_MANDATORY);

	set_mouse_filter(MOUSE_FILTER_STOP);
	set_text(p_text);
}

Button::~Button() {
}
