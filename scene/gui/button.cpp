/*************************************************************************/
/*  button.cpp                                                           */
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
		if (icon.is_null() && has_theme_icon(SNAME("icon"))) {
			_icon = Control::get_theme_icon(SNAME("icon"));
		} else {
			_icon = icon;
		}

		if (!_icon.is_null()) {
			minsize.height = MAX(minsize.height, _icon->get_height());

			if (icon_alignment != HORIZONTAL_ALIGNMENT_CENTER) {
				minsize.width += _icon->get_width();
				if (!xl_text.is_empty()) {
					minsize.width += get_theme_constant(SNAME("hseparation"));
				}
			} else {
				minsize.width = MAX(minsize.width, _icon->get_width());
			}
		}
	}

	Ref<Font> font = get_theme_font(SNAME("font"));
	float font_height = font->get_height(get_theme_font_size(SNAME("font_size")));

	minsize.height = MAX(font_height, minsize.height);

	return get_theme_stylebox(SNAME("normal"))->get_minimum_size() + minsize;
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
			xl_text = atr(text);
			_shape();

			update_minimum_size();
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
			Color color_icon(1, 1, 1, 1);

			Ref<StyleBox> style = get_theme_stylebox(SNAME("normal"));
			bool rtl = is_layout_rtl();

			switch (get_draw_mode()) {
				case DRAW_NORMAL: {
					if (rtl && has_theme_stylebox(SNAME("normal_mirrored"))) {
						style = get_theme_stylebox(SNAME("normal_mirrored"));
					} else {
						style = get_theme_stylebox(SNAME("normal"));
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}

					// Focus colors only take precedence over normal state.
					if (has_focus()) {
						color = get_theme_color(SNAME("font_focus_color"));
						if (has_theme_color(SNAME("icon_focus_color"))) {
							color_icon = get_theme_color(SNAME("icon_focus_color"));
						}
					} else {
						color = get_theme_color(SNAME("font_color"));
						if (has_theme_color(SNAME("icon_normal_color"))) {
							color_icon = get_theme_color(SNAME("icon_normal_color"));
						}
					}
				} break;
				case DRAW_HOVER_PRESSED: {
					if (has_theme_stylebox(SNAME("hover_pressed")) && has_theme_stylebox_override("hover_pressed")) {
						if (rtl && has_theme_stylebox(SNAME("hover_pressed_mirrored"))) {
							style = get_theme_stylebox(SNAME("hover_pressed_mirrored"));
						} else {
							style = get_theme_stylebox(SNAME("hover_pressed"));
						}

						if (!flat) {
							style->draw(ci, Rect2(Point2(0, 0), size));
						}
						if (has_theme_color(SNAME("font_hover_pressed_color"))) {
							color = get_theme_color(SNAME("font_hover_pressed_color"));
						} else {
							color = get_theme_color(SNAME("font_color"));
						}
						if (has_theme_color(SNAME("icon_hover_pressed_color"))) {
							color_icon = get_theme_color(SNAME("icon_hover_pressed_color"));
						}

						break;
					}
					[[fallthrough]];
				}
				case DRAW_PRESSED: {
					if (rtl && has_theme_stylebox(SNAME("pressed_mirrored"))) {
						style = get_theme_stylebox(SNAME("pressed_mirrored"));
					} else {
						style = get_theme_stylebox(SNAME("pressed"));
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					if (has_theme_color(SNAME("font_pressed_color"))) {
						color = get_theme_color(SNAME("font_pressed_color"));
					} else {
						color = get_theme_color(SNAME("font_color"));
					}
					if (has_theme_color(SNAME("icon_pressed_color"))) {
						color_icon = get_theme_color(SNAME("icon_pressed_color"));
					}

				} break;
				case DRAW_HOVER: {
					if (rtl && has_theme_stylebox(SNAME("hover_mirrored"))) {
						style = get_theme_stylebox(SNAME("hover_mirrored"));
					} else {
						style = get_theme_stylebox(SNAME("hover"));
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = get_theme_color(SNAME("font_hover_color"));
					if (has_theme_color(SNAME("icon_hover_color"))) {
						color_icon = get_theme_color(SNAME("icon_hover_color"));
					}

				} break;
				case DRAW_DISABLED: {
					if (rtl && has_theme_stylebox(SNAME("disabled_mirrored"))) {
						style = get_theme_stylebox(SNAME("disabled_mirrored"));
					} else {
						style = get_theme_stylebox(SNAME("disabled"));
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = get_theme_color(SNAME("font_disabled_color"));
					if (has_theme_color(SNAME("icon_disabled_color"))) {
						color_icon = get_theme_color(SNAME("icon_disabled_color"));
					}

				} break;
			}

			if (has_focus()) {
				Ref<StyleBox> style2 = get_theme_stylebox(SNAME("focus"));
				style2->draw(ci, Rect2(Point2(), size));
			}

			Ref<Texture2D> _icon;
			if (icon.is_null() && has_theme_icon(SNAME("icon"))) {
				_icon = Control::get_theme_icon(SNAME("icon"));
			} else {
				_icon = icon;
			}

			Rect2 icon_region = Rect2();
			HorizontalAlignment icon_align_rtl_checked = icon_alignment;
			HorizontalAlignment align_rtl_checked = alignment;
			// Swap icon and text alignment sides if right-to-left layout is set.
			if (rtl) {
				if (icon_alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
					icon_align_rtl_checked = HORIZONTAL_ALIGNMENT_LEFT;
				} else if (icon_alignment == HORIZONTAL_ALIGNMENT_LEFT) {
					icon_align_rtl_checked = HORIZONTAL_ALIGNMENT_RIGHT;
				}
				if (alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
					align_rtl_checked = HORIZONTAL_ALIGNMENT_LEFT;
				} else if (alignment == HORIZONTAL_ALIGNMENT_LEFT) {
					align_rtl_checked = HORIZONTAL_ALIGNMENT_RIGHT;
				}
			}
			if (!_icon.is_null()) {
				int valign = size.height - style->get_minimum_size().y;
				if (is_disabled()) {
					color_icon.a = 0.4;
				}

				float icon_ofs_region = 0.0;
				Point2 style_offset;
				Size2 icon_size = _icon->get_size();
				if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_LEFT) {
					style_offset.x = style->get_margin(SIDE_LEFT);
					if (_internal_margin[SIDE_LEFT] > 0) {
						icon_ofs_region = _internal_margin[SIDE_LEFT] + get_theme_constant(SNAME("hseparation"));
					}
				} else if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER) {
					style_offset.x = 0.0;
				} else if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_RIGHT) {
					style_offset.x = -style->get_margin(SIDE_RIGHT);
					if (_internal_margin[SIDE_RIGHT] > 0) {
						icon_ofs_region = -_internal_margin[SIDE_RIGHT] - get_theme_constant(SNAME("hseparation"));
					}
				}
				style_offset.y = style->get_margin(SIDE_TOP);

				if (expand_icon) {
					Size2 _size = get_size() - style->get_offset() * 2;
					_size.width -= get_theme_constant(SNAME("hseparation")) + icon_ofs_region;
					if (!clip_text && icon_align_rtl_checked != HORIZONTAL_ALIGNMENT_CENTER) {
						_size.width -= text_buf->get_size().width;
					}
					float icon_width = _icon->get_width() * _size.height / _icon->get_height();
					float icon_height = _size.height;

					if (icon_width > _size.width) {
						icon_width = _size.width;
						icon_height = _icon->get_height() * icon_width / _icon->get_width();
					}

					icon_size = Size2(icon_width, icon_height);
				}

				if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_LEFT) {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region, Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				} else if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER) {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region + Math::floor((size.x - icon_size.x) * 0.5), Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				} else {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region + size.x - icon_size.x, Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				}

				if (icon_region.size.width > 0) {
					draw_texture_rect_region(_icon, icon_region, Rect2(Point2(), _icon->get_size()), color_icon);
				}
			}

			Point2 icon_ofs = !_icon.is_null() ? Point2(icon_region.size.width + get_theme_constant(SNAME("hseparation")), 0) : Point2();
			if (align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER && icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER) {
				icon_ofs.x = 0.0;
			}
			int text_clip = size.width - style->get_minimum_size().width - icon_ofs.width;
			text_buf->set_width(clip_text ? text_clip : -1);

			int text_width = clip_text ? MIN(text_clip, text_buf->get_size().x) : text_buf->get_size().x;

			if (_internal_margin[SIDE_LEFT] > 0) {
				text_clip -= _internal_margin[SIDE_LEFT] + get_theme_constant(SNAME("hseparation"));
			}
			if (_internal_margin[SIDE_RIGHT] > 0) {
				text_clip -= _internal_margin[SIDE_RIGHT] + get_theme_constant(SNAME("hseparation"));
			}

			Point2 text_ofs = (size - style->get_minimum_size() - icon_ofs - text_buf->get_size() - Point2(_internal_margin[SIDE_RIGHT] - _internal_margin[SIDE_LEFT], 0)) / 2.0;

			switch (align_rtl_checked) {
				case HORIZONTAL_ALIGNMENT_FILL:
				case HORIZONTAL_ALIGNMENT_LEFT: {
					if (icon_align_rtl_checked != HORIZONTAL_ALIGNMENT_LEFT) {
						icon_ofs.x = 0.0;
					}
					if (_internal_margin[SIDE_LEFT] > 0) {
						text_ofs.x = style->get_margin(SIDE_LEFT) + icon_ofs.x + _internal_margin[SIDE_LEFT] + get_theme_constant(SNAME("hseparation"));
					} else {
						text_ofs.x = style->get_margin(SIDE_LEFT) + icon_ofs.x;
					}
					text_ofs.y += style->get_offset().y;
				} break;
				case HORIZONTAL_ALIGNMENT_CENTER: {
					if (text_ofs.x < 0) {
						text_ofs.x = 0;
					}
					if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_LEFT) {
						text_ofs += icon_ofs;
					}
					text_ofs += style->get_offset();
				} break;
				case HORIZONTAL_ALIGNMENT_RIGHT: {
					if (_internal_margin[SIDE_RIGHT] > 0) {
						text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width - _internal_margin[SIDE_RIGHT] - get_theme_constant(SNAME("hseparation"));
					} else {
						text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width;
					}
					text_ofs.y += style->get_offset().y;
					if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_RIGHT) {
						text_ofs.x -= icon_ofs.x;
					}
				} break;
			}

			Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
			int outline_size = get_theme_constant(SNAME("outline_size"));
			if (outline_size > 0 && font_outline_color.a > 0) {
				text_buf->draw_outline(ci, text_ofs, outline_size, font_outline_color);
			}

			text_buf->draw(ci, text_ofs, color);
		} break;
	}
}

void Button::_shape() {
	Ref<Font> font = get_theme_font(SNAME("font"));
	int font_size = get_theme_font_size(SNAME("font_size"));

	text_buf->clear();
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		text_buf->set_direction((TextServer::Direction)text_direction);
	}
	text_buf->add_string(xl_text, font, font_size, opentype_features, (!language.is_empty()) ? language : TranslationServer::get_singleton()->get_tool_locale());
}

void Button::set_text(const String &p_text) {
	if (text != p_text) {
		text = p_text;
		xl_text = atr(text);
		_shape();

		update();
		update_minimum_size();
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
		update_minimum_size();
	}
}

Ref<Texture2D> Button::get_icon() const {
	return icon;
}

void Button::set_expand_icon(bool p_enabled) {
	if (expand_icon != p_enabled) {
		expand_icon = p_enabled;
		update();
		update_minimum_size();
	}
}

bool Button::is_expand_icon() const {
	return expand_icon;
}

void Button::set_flat(bool p_enabled) {
	if (flat != p_enabled) {
		flat = p_enabled;
		update();
	}
}

bool Button::is_flat() const {
	return flat;
}

void Button::set_clip_text(bool p_enabled) {
	if (clip_text != p_enabled) {
		clip_text = p_enabled;
		update();
		update_minimum_size();
	}
}

bool Button::get_clip_text() const {
	return clip_text;
}

void Button::set_text_alignment(HorizontalAlignment p_alignment) {
	if (alignment != p_alignment) {
		alignment = p_alignment;
		update();
	}
}

HorizontalAlignment Button::get_text_alignment() const {
	return alignment;
}

void Button::set_icon_alignment(HorizontalAlignment p_alignment) {
	icon_alignment = p_alignment;
	update_minimum_size();
	update();
}

HorizontalAlignment Button::get_icon_alignment() const {
	return icon_alignment;
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
	ClassDB::bind_method(D_METHOD("set_flat", "enabled"), &Button::set_flat);
	ClassDB::bind_method(D_METHOD("is_flat"), &Button::is_flat);
	ClassDB::bind_method(D_METHOD("set_clip_text", "enabled"), &Button::set_clip_text);
	ClassDB::bind_method(D_METHOD("get_clip_text"), &Button::get_clip_text);
	ClassDB::bind_method(D_METHOD("set_text_alignment", "alignment"), &Button::set_text_alignment);
	ClassDB::bind_method(D_METHOD("get_text_alignment"), &Button::get_text_alignment);
	ClassDB::bind_method(D_METHOD("set_icon_alignment", "icon_alignment"), &Button::set_icon_alignment);
	ClassDB::bind_method(D_METHOD("get_icon_alignment"), &Button::get_icon_alignment);
	ClassDB::bind_method(D_METHOD("set_expand_icon", "enabled"), &Button::set_expand_icon);
	ClassDB::bind_method(D_METHOD("is_expand_icon"), &Button::is_expand_icon);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_button_icon", "get_button_icon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "get_clip_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_text_alignment", "get_text_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "icon_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_icon_alignment", "get_icon_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expand_icon"), "set_expand_icon", "is_expand_icon");
}

Button::Button(const String &p_text) {
	text_buf.instantiate();
	text_buf->set_flags(TextServer::BREAK_MANDATORY);

	set_mouse_filter(MOUSE_FILTER_STOP);
	set_text(p_text);
}

Button::~Button() {
}
