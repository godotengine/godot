/**************************************************************************/
/*  button.cpp                                                            */
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

#include "button.h"

#include "core/translation.h"
#include "scene/scene_string_names.h"
#include "servers/visual_server.h"

Size2 Button::get_minimum_size() const {
	Size2 minsize = get_font("font")->total_size_of_lines(xl_text.split("\n"));
	if (clip_text) {
		minsize.width = 0;
	}

	if (!expand_icon) {
		Ref<Texture> _icon;
		if (icon.is_null() && has_icon("icon")) {
			_icon = Control::get_icon("icon");
		} else {
			_icon = icon;
		}

		if (!_icon.is_null()) {
			minsize.height = MAX(minsize.height, _icon->get_height());

			if (icon_align != ALIGN_CENTER) {
				minsize.width += _icon->get_width();
				if (xl_text != "") {
					minsize.width += get_constant("hseparation");
				}
			} else {
				minsize.width = MAX(minsize.width, _icon->get_width());
			}
		}
	}

	return get_stylebox("normal")->get_minimum_size() + minsize;
}

void Button::_set_internal_margin(Margin p_margin, float p_value) {
	_internal_margin[p_margin] = p_value;
}

void Button::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			xl_text = tr(text);
			minimum_size_changed();
			update();
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2 size = get_size();
			Color color;
			Color color_icon(1, 1, 1, 1);

			Ref<StyleBox> style = get_stylebox("normal");

			Vector<String> lines = xl_text.split("\n");

			switch (get_draw_mode()) {
				case DRAW_NORMAL: {
					style = get_stylebox("normal");
					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}

					// Focus colors only take precedence over normal state.
					if (has_focus()) {
						color = get_color("font_color_focus");
						if (has_color("icon_color_focus")) {
							color_icon = get_color("icon_color_focus");
						}
					} else {
						color = get_color("font_color");
						if (has_color("icon_color_normal")) {
							color_icon = get_color("icon_color_normal");
						}
					}
				} break;
				case DRAW_HOVER_PRESSED: {
					if (has_stylebox("hover_pressed")) {
						style = get_stylebox("hover_pressed");
						if (!flat) {
							style->draw(ci, Rect2(Point2(0, 0), size));
						}
						if (has_color("font_color_hover_pressed")) {
							color = get_color("font_color_hover_pressed");
						} else {
							color = get_color("font_color");
						}
						if (has_color("icon_color_hover_pressed")) {
							color_icon = get_color("icon_color_hover_pressed");
						}

						break;
					}
					FALLTHROUGH;
				}
				case DRAW_PRESSED: {
					style = get_stylebox("pressed");
					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					if (has_color("font_color_pressed")) {
						color = get_color("font_color_pressed");
					} else {
						color = get_color("font_color");
					}
					if (has_color("icon_color_pressed")) {
						color_icon = get_color("icon_color_pressed");
					}

				} break;
				case DRAW_HOVER: {
					style = get_stylebox("hover");
					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = get_color("font_color_hover");
					if (has_color("icon_color_hover")) {
						color_icon = get_color("icon_color_hover");
					}

				} break;
				case DRAW_DISABLED: {
					style = get_stylebox("disabled");
					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = get_color("font_color_disabled");
					if (has_color("icon_color_disabled")) {
						color_icon = get_color("icon_color_disabled");
					} else {
						color_icon.a = 0.4;
					}
				} break;
			}

			if (has_focus()) {
				Ref<StyleBox> style2 = get_stylebox("focus");
				style2->draw(ci, Rect2(Point2(), size));
			}

			Ref<Font> font = get_font("font");
			Ref<Texture> _icon;
			if (icon.is_null() && has_icon("icon")) {
				_icon = Control::get_icon("icon");
			} else {
				_icon = icon;
			}

			Rect2 icon_region = Rect2();
			if (!_icon.is_null()) {
				int valign = size.height - style->get_minimum_size().y;

				float icon_ofs_region = 0;
				Point2 style_offset;
				Size2 icon_size = _icon->get_size();
				if (icon_align == ALIGN_LEFT) {
					style_offset.x = style->get_margin(MARGIN_LEFT);
					if (_internal_margin[MARGIN_LEFT] > 0) {
						icon_ofs_region = _internal_margin[MARGIN_LEFT] + get_constant("hseparation");
					}
				} else if (icon_align == ALIGN_CENTER) {
					style_offset.x = 0;
				} else if (icon_align == ALIGN_RIGHT) {
					style_offset.x = -style->get_margin(MARGIN_RIGHT);
					if (_internal_margin[MARGIN_RIGHT] > 0) {
						icon_ofs_region = -_internal_margin[MARGIN_RIGHT] - get_constant("hseparation");
					}
				}
				style_offset.y = style->get_margin(MARGIN_TOP);

				if (expand_icon) {
					Size2 _size = get_size() - style->get_offset() * 2;
					int icon_text_separation = text.empty() ? 0 : get_constant("h_separation");
					_size.width -= icon_text_separation + icon_ofs_region;
					if (!clip_text && icon_align != ALIGN_CENTER) {
						_size.width -= get_font("font")->total_size_of_lines(lines).width;
					}
					float icon_width = _icon->get_width() * _size.height / _icon->get_height();
					float icon_height = _size.height;

					if (icon_width > _size.width) {
						icon_width = _size.width;
						icon_height = _icon->get_height() * icon_width / _icon->get_width();
					}

					icon_size = Size2(icon_width, icon_height);
				}

				if (icon_align == ALIGN_LEFT) {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region, Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				} else if (icon_align == ALIGN_CENTER) {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region + Math::floor((size.x - icon_size.x) * 0.5), Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				} else {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region + size.x - icon_size.x, Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				}

				if (icon_region.size.width > 0) {
					draw_texture_rect_region(_icon, icon_region, Rect2(Point2(), _icon->get_size()), color_icon);
				}
			}

			Point2 icon_ofs = !_icon.is_null() ? Point2(icon_region.size.width + get_constant("hseparation"), 0) : Point2();
			if (align == ALIGN_CENTER && icon_align == ALIGN_CENTER) {
				icon_ofs.x = 0;
			}
			int text_clip = size.width - style->get_minimum_size().width - icon_ofs.width;
			if (_internal_margin[MARGIN_LEFT] > 0) {
				text_clip -= _internal_margin[MARGIN_LEFT] + get_constant("hseparation");
			}
			if (_internal_margin[MARGIN_RIGHT] > 0) {
				text_clip -= _internal_margin[MARGIN_RIGHT] + get_constant("hseparation");
			}

			int num_lines = lines.size();
			float line_height = font->get_height();

			for (int i = 0; i < num_lines; i++) {
				String line_text = lines[i];
				Point2 text_ofs = (size - style->get_minimum_size() - icon_ofs - font->get_string_size(line_text) - Point2(_internal_margin[MARGIN_RIGHT] - _internal_margin[MARGIN_LEFT], 0)) / 2.0;
				switch (align) {
					case ALIGN_LEFT: {
						if (icon_align != ALIGN_LEFT) {
							icon_ofs.x = 0;
						}
						if (_internal_margin[MARGIN_LEFT] > 0) {
							text_ofs.x = style->get_margin(MARGIN_LEFT) + icon_ofs.x + _internal_margin[MARGIN_LEFT] + get_constant("hseparation");
						} else {
							text_ofs.x = style->get_margin(MARGIN_LEFT) + icon_ofs.x;
						}
						text_ofs.y += style->get_offset().y;
					} break;
					case ALIGN_CENTER: {
						if (text_ofs.x < 0) {
							text_ofs.x = 0;
						}
						if (icon_align == ALIGN_LEFT) {
							text_ofs += icon_ofs;
						}
						text_ofs += style->get_offset();
					} break;
					case ALIGN_RIGHT: {
						int text_width = font->get_string_size(line_text).x;
						if (_internal_margin[MARGIN_RIGHT] > 0) {
							text_ofs.x = size.x - style->get_margin(MARGIN_RIGHT) - text_width - _internal_margin[MARGIN_RIGHT] - get_constant("hseparation");
						} else {
							text_ofs.x = size.x - style->get_margin(MARGIN_RIGHT) - text_width;
						}
						text_ofs.y += style->get_offset().y;
						if (icon_align == ALIGN_RIGHT) {
							text_ofs.x -= icon_ofs.x;
						}
					} break;
				}

				text_ofs.y += font->get_ascent();
				text_ofs.y += line_height * (((float)i) - (((float)(num_lines - 1)) / 2.0));
				select_font(font);
				font->draw(ci, text_ofs.floor(), line_text, color, clip_text ? text_clip : -1);
			}
		} break;
	}
}

void Button::set_text(const String &p_text) {
	if (text == p_text) {
		return;
	}
	text = p_text;
	xl_text = tr(p_text);
	update();
	_change_notify("text");
	minimum_size_changed();
}
String Button::get_text() const {
	return text;
}

void Button::set_icon(const Ref<Texture> &p_icon) {
	if (icon == p_icon) {
		return;
	}
	if (icon.is_valid()) {
		icon->disconnect(SceneStringNames::get_singleton()->changed, this, "_texture_changed");
	}
	icon = p_icon;
	if (icon.is_valid()) {
		icon->connect(SceneStringNames::get_singleton()->changed, this, "_texture_changed");
	}
	update();
	_change_notify("icon");
	minimum_size_changed();
}

Ref<Texture> Button::get_icon() const {
	return icon;
}

void Button::_texture_changed() {
	update();
	minimum_size_changed();
}

void Button::set_expand_icon(bool p_expand_icon) {
	expand_icon = p_expand_icon;
	update();
	minimum_size_changed();
}

bool Button::is_expand_icon() const {
	return expand_icon;
}

void Button::set_flat(bool p_flat) {
	flat = p_flat;
	update();
	_change_notify("flat");
}

bool Button::is_flat() const {
	return flat;
}

void Button::set_clip_text(bool p_clip_text) {
	clip_text = p_clip_text;
	update();
	minimum_size_changed();
}

bool Button::get_clip_text() const {
	return clip_text;
}

void Button::set_text_align(TextAlign p_align) {
	align = p_align;
	update();
}

Button::TextAlign Button::get_text_align() const {
	return align;
}

void Button::set_icon_align(TextAlign p_align) {
	icon_align = p_align;
	minimum_size_changed();
	update();
}

Button::TextAlign Button::get_icon_align() const {
	return icon_align;
}

void Button::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &Button::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Button::get_text);
	ClassDB::bind_method(D_METHOD("set_button_icon", "texture"), &Button::set_icon);
	ClassDB::bind_method(D_METHOD("get_button_icon"), &Button::get_icon);
	ClassDB::bind_method(D_METHOD("set_flat", "enabled"), &Button::set_flat);
	ClassDB::bind_method(D_METHOD("is_flat"), &Button::is_flat);
	ClassDB::bind_method(D_METHOD("set_clip_text", "enabled"), &Button::set_clip_text);
	ClassDB::bind_method(D_METHOD("get_clip_text"), &Button::get_clip_text);
	ClassDB::bind_method(D_METHOD("set_text_align", "align"), &Button::set_text_align);
	ClassDB::bind_method(D_METHOD("get_text_align"), &Button::get_text_align);
	ClassDB::bind_method(D_METHOD("set_icon_align", "icon_align"), &Button::set_icon_align);
	ClassDB::bind_method(D_METHOD("get_icon_align"), &Button::get_icon_align);
	ClassDB::bind_method(D_METHOD("set_expand_icon", "enabled"), &Button::set_expand_icon);
	ClassDB::bind_method(D_METHOD("is_expand_icon"), &Button::is_expand_icon);

	ClassDB::bind_method(D_METHOD("_texture_changed"), &Button::_texture_changed);

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_button_icon", "get_button_icon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "get_clip_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_text_align", "get_text_align");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "icon_align", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_icon_align", "get_icon_align");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expand_icon"), "set_expand_icon", "is_expand_icon");
}

Button::Button(const String &p_text) {
	flat = false;
	clip_text = false;
	expand_icon = false;
	set_mouse_filter(MOUSE_FILTER_STOP);
	set_text(p_text);
	align = ALIGN_CENTER;
	icon_align = ALIGN_LEFT;

	for (int i = 0; i < 4; i++) {
		_internal_margin[i] = 0;
	}
}

Button::~Button() {
}
