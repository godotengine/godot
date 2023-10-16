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

#include "core/string/translation.h"
#include "scene/theme/theme_db.h"
#include "servers/rendering_server.h"

Size2 Button::get_minimum_size() const {
	Ref<Texture2D> _icon = icon;
	if (_icon.is_null() && has_theme_icon(SNAME("icon"))) {
		_icon = theme_cache.icon;
	}

	return get_minimum_size_for_text_and_icon("", _icon);
}

void Button::_set_internal_margin(Side p_side, float p_value) {
	_internal_margin[p_side] = p_value;
}

void Button::_queue_update_size_cache() {
}

void Button::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			queue_redraw();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			xl_text = atr(text);
			_shape();

			update_minimum_size();
			queue_redraw();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_shape();

			update_minimum_size();
			queue_redraw();
		} break;

		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2 size = get_size();
			Color color;
			Color color_icon(1, 1, 1, 1);

			Ref<StyleBox> style = theme_cache.normal;
			bool rtl = is_layout_rtl();
			const bool is_clipped = clip_text || overrun_behavior != TextServer::OVERRUN_NO_TRIMMING;

			switch (get_draw_mode()) {
				case DRAW_NORMAL: {
					if (rtl && has_theme_stylebox(SNAME("normal_mirrored"))) {
						style = theme_cache.normal_mirrored;
					} else {
						style = theme_cache.normal;
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}

					// Focus colors only take precedence over normal state.
					if (has_focus()) {
						color = theme_cache.font_focus_color;
						if (has_theme_color(SNAME("icon_focus_color"))) {
							color_icon = theme_cache.icon_focus_color;
						}
					} else {
						color = theme_cache.font_color;
						if (has_theme_color(SNAME("icon_normal_color"))) {
							color_icon = theme_cache.icon_normal_color;
						}
					}
				} break;
				case DRAW_HOVER_PRESSED: {
					// Edge case for CheckButton and CheckBox.
					if (has_theme_stylebox("hover_pressed")) {
						if (rtl && has_theme_stylebox(SNAME("hover_pressed_mirrored"))) {
							style = theme_cache.hover_pressed_mirrored;
						} else {
							style = theme_cache.hover_pressed;
						}

						if (!flat) {
							style->draw(ci, Rect2(Point2(0, 0), size));
						}
						if (has_theme_color(SNAME("font_hover_pressed_color"))) {
							color = theme_cache.font_hover_pressed_color;
						}
						if (has_theme_color(SNAME("icon_hover_pressed_color"))) {
							color_icon = theme_cache.icon_hover_pressed_color;
						}

						break;
					}
					[[fallthrough]];
				}
				case DRAW_PRESSED: {
					if (rtl && has_theme_stylebox(SNAME("pressed_mirrored"))) {
						style = theme_cache.pressed_mirrored;
					} else {
						style = theme_cache.pressed;
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					if (has_theme_color(SNAME("font_pressed_color"))) {
						color = theme_cache.font_pressed_color;
					} else {
						color = theme_cache.font_color;
					}
					if (has_theme_color(SNAME("icon_pressed_color"))) {
						color_icon = theme_cache.icon_pressed_color;
					}

				} break;
				case DRAW_HOVER: {
					if (rtl && has_theme_stylebox(SNAME("hover_mirrored"))) {
						style = theme_cache.hover_mirrored;
					} else {
						style = theme_cache.hover;
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = theme_cache.font_hover_color;
					if (has_theme_color(SNAME("icon_hover_color"))) {
						color_icon = theme_cache.icon_hover_color;
					}

				} break;
				case DRAW_DISABLED: {
					if (rtl && has_theme_stylebox(SNAME("disabled_mirrored"))) {
						style = theme_cache.disabled_mirrored;
					} else {
						style = theme_cache.disabled;
					}

					if (!flat) {
						style->draw(ci, Rect2(Point2(0, 0), size));
					}
					color = theme_cache.font_disabled_color;
					if (has_theme_color(SNAME("icon_disabled_color"))) {
						color_icon = theme_cache.icon_disabled_color;
					} else {
						color_icon.a = 0.4;
					}

				} break;
			}

			if (has_focus()) {
				Ref<StyleBox> style2 = theme_cache.focus;
				style2->draw(ci, Rect2(Point2(), size));
			}

			Ref<Texture2D> _icon;
			if (icon.is_null() && has_theme_icon(SNAME("icon"))) {
				_icon = theme_cache.icon;
			} else {
				_icon = icon;
			}

			Rect2 icon_region;
			HorizontalAlignment icon_align_rtl_checked = horizontal_icon_alignment;
			HorizontalAlignment align_rtl_checked = alignment;
			// Swap icon and text alignment sides if right-to-left layout is set.
			if (rtl) {
				if (horizontal_icon_alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
					icon_align_rtl_checked = HORIZONTAL_ALIGNMENT_LEFT;
				} else if (horizontal_icon_alignment == HORIZONTAL_ALIGNMENT_LEFT) {
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

				int voffset = 0;
				Size2 icon_size = _icon->get_size();

				// Fix vertical size.
				if (vertical_icon_alignment != VERTICAL_ALIGNMENT_CENTER) {
					valign -= text_buf->get_size().height;
				}

				float icon_ofs_region = 0.0;
				Point2 style_offset;
				if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_LEFT) {
					style_offset.x = style->get_margin(SIDE_LEFT);
					if (_internal_margin[SIDE_LEFT] > 0) {
						icon_ofs_region = _internal_margin[SIDE_LEFT] + theme_cache.h_separation;
					}
				} else if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER) {
					style_offset.x = 0.0;
				} else if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_RIGHT) {
					style_offset.x = -style->get_margin(SIDE_RIGHT);
					if (_internal_margin[SIDE_RIGHT] > 0) {
						icon_ofs_region = -_internal_margin[SIDE_RIGHT] - theme_cache.h_separation;
					}
				}
				style_offset.y = style->get_margin(SIDE_TOP);

				if (expand_icon) {
					Size2 _size = get_size() - style->get_offset() * 2;
					int icon_text_separation = text.is_empty() ? 0 : theme_cache.h_separation;
					_size.width -= icon_text_separation + icon_ofs_region;
					if (!is_clipped && icon_align_rtl_checked != HORIZONTAL_ALIGNMENT_CENTER) {
						_size.width -= text_buf->get_size().width;
					}
					if (vertical_icon_alignment != VERTICAL_ALIGNMENT_CENTER) {
						_size.height -= text_buf->get_size().height;
					}
					float icon_width = _icon->get_width() * _size.height / _icon->get_height();
					float icon_height = _size.height;

					if (icon_width > _size.width) {
						icon_width = _size.width;
						icon_height = _icon->get_height() * icon_width / _icon->get_width();
					}

					icon_size = Size2(icon_width, icon_height);
				}
				icon_size = _fit_icon_size(icon_size);

				if (vertical_icon_alignment == VERTICAL_ALIGNMENT_TOP) {
					voffset = -(valign - icon_size.y) / 2;
				}
				if (vertical_icon_alignment == VERTICAL_ALIGNMENT_BOTTOM) {
					voffset = (valign - icon_size.y) / 2 + text_buf->get_size().y;
				}

				if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_LEFT) {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region, voffset + Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				} else if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER) {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region + Math::floor((size.x - icon_size.x) * 0.5), voffset + Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				} else {
					icon_region = Rect2(style_offset + Point2(icon_ofs_region + size.x - icon_size.x, voffset + Math::floor((valign - icon_size.y) * 0.5)), icon_size);
				}

				if (icon_region.size.width > 0) {
					Rect2 icon_region_rounded = Rect2(icon_region.position.round(), icon_region.size.round());
					draw_texture_rect(_icon, icon_region_rounded, false, color_icon);
				}
			}

			Point2 icon_ofs = !_icon.is_null() ? Point2(icon_region.size.width + theme_cache.h_separation, 0) : Point2();
			if (align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER && icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_CENTER) {
				icon_ofs.x = 0.0;
			}

			int text_clip = size.width - style->get_minimum_size().width - icon_ofs.width;
			if (_internal_margin[SIDE_LEFT] > 0) {
				text_clip -= _internal_margin[SIDE_LEFT] + theme_cache.h_separation;
			}
			if (_internal_margin[SIDE_RIGHT] > 0) {
				text_clip -= _internal_margin[SIDE_RIGHT] + theme_cache.h_separation;
			}

			text_buf->set_width(is_clipped ? text_clip : -1);

			int text_width = MAX(1, is_clipped ? MIN(text_clip, text_buf->get_size().x) : text_buf->get_size().x);

			Point2 text_ofs = (size - style->get_minimum_size() - icon_ofs - text_buf->get_size() - Point2(_internal_margin[SIDE_RIGHT] - _internal_margin[SIDE_LEFT], 0)) / 2.0;

			if (vertical_icon_alignment == VERTICAL_ALIGNMENT_TOP) {
				text_ofs.y += icon_region.size.height / 2;
			}
			if (vertical_icon_alignment == VERTICAL_ALIGNMENT_BOTTOM) {
				text_ofs.y -= icon_region.size.height / 2;
			}

			text_buf->set_alignment(align_rtl_checked);
			text_buf->set_width(text_width);
			switch (align_rtl_checked) {
				case HORIZONTAL_ALIGNMENT_FILL:
				case HORIZONTAL_ALIGNMENT_LEFT: {
					if (icon_align_rtl_checked != HORIZONTAL_ALIGNMENT_LEFT) {
						icon_ofs.x = 0.0;
					}
					if (_internal_margin[SIDE_LEFT] > 0) {
						text_ofs.x = style->get_margin(SIDE_LEFT) + icon_ofs.x + _internal_margin[SIDE_LEFT] + theme_cache.h_separation;
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
						text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width - _internal_margin[SIDE_RIGHT] - theme_cache.h_separation;
					} else {
						text_ofs.x = size.x - style->get_margin(SIDE_RIGHT) - text_width;
					}
					text_ofs.y += style->get_offset().y;
					if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_RIGHT) {
						text_ofs.x -= icon_ofs.x;
					}
				} break;
			}

			Color font_outline_color = theme_cache.font_outline_color;
			int outline_size = theme_cache.outline_size;
			if (outline_size > 0 && font_outline_color.a > 0) {
				text_buf->draw_outline(ci, text_ofs, outline_size, font_outline_color);
			}
			text_buf->draw(ci, text_ofs, color);
		} break;
	}
}

Size2 Button::_fit_icon_size(const Size2 &p_size) const {
	int max_width = theme_cache.icon_max_width;
	Size2 icon_size = p_size;

	if (max_width > 0 && icon_size.width > max_width) {
		icon_size.height = icon_size.height * max_width / icon_size.width;
		icon_size.width = max_width;
	}

	return icon_size;
}

Size2 Button::get_minimum_size_for_text_and_icon(const String &p_text, Ref<Texture2D> p_icon) const {
	Ref<TextParagraph> paragraph;
	if (p_text.is_empty()) {
		paragraph = text_buf;
	} else {
		paragraph.instantiate();
		const_cast<Button *>(this)->_shape(paragraph, p_text);
	}

	Size2 minsize = paragraph->get_size();
	if (clip_text || overrun_behavior != TextServer::OVERRUN_NO_TRIMMING) {
		minsize.width = 0;
	}

	if (!expand_icon && p_icon.is_valid()) {
		Size2 icon_size = _fit_icon_size(p_icon->get_size());
		if (vertical_icon_alignment == VERTICAL_ALIGNMENT_CENTER) {
			minsize.height = MAX(minsize.height, icon_size.height);
		} else {
			minsize.height += icon_size.height;
		}

		if (horizontal_icon_alignment != HORIZONTAL_ALIGNMENT_CENTER) {
			minsize.width += icon_size.width;
			if (!xl_text.is_empty() || !p_text.is_empty()) {
				minsize.width += MAX(0, theme_cache.h_separation);
			}
		} else {
			minsize.width = MAX(minsize.width, icon_size.width);
		}
	}

	if (!xl_text.is_empty() || !p_text.is_empty()) {
		Ref<Font> font = theme_cache.font;
		float font_height = font->get_height(theme_cache.font_size);
		if (vertical_icon_alignment == VERTICAL_ALIGNMENT_CENTER) {
			minsize.height = MAX(font_height, minsize.height);
		} else {
			minsize.height += font_height;
		}
	}

	return theme_cache.normal->get_minimum_size() + minsize;
}

void Button::_shape(Ref<TextParagraph> p_paragraph, String p_text) {
	if (p_paragraph.is_null()) {
		p_paragraph = text_buf;
	}

	if (p_text.is_empty()) {
		p_text = xl_text;
	}

	p_paragraph->clear();

	Ref<Font> font = theme_cache.font;
	int font_size = theme_cache.font_size;
	if (font.is_null() || font_size == 0) {
		// Can't shape without a valid font and a non-zero size.
		return;
	}

	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		p_paragraph->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		p_paragraph->set_direction((TextServer::Direction)text_direction);
	}
	p_paragraph->add_string(p_text, font, font_size, language);
	p_paragraph->set_text_overrun_behavior(overrun_behavior);
}

void Button::set_text_overrun_behavior(TextServer::OverrunBehavior p_behavior) {
	if (overrun_behavior != p_behavior) {
		overrun_behavior = p_behavior;
		_shape();

		queue_redraw();
		update_minimum_size();
	}
}

TextServer::OverrunBehavior Button::get_text_overrun_behavior() const {
	return overrun_behavior;
}

void Button::set_text(const String &p_text) {
	if (text != p_text) {
		text = p_text;
		xl_text = atr(text);
		_shape();

		queue_redraw();
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
		queue_redraw();
	}
}

Control::TextDirection Button::get_text_direction() const {
	return text_direction;
}

void Button::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape();
		queue_redraw();
	}
}

String Button::get_language() const {
	return language;
}

void Button::set_icon(const Ref<Texture2D> &p_icon) {
	if (icon == p_icon) {
		return;
	}

	if (icon.is_valid()) {
		icon->disconnect_changed(callable_mp(this, &Button::_texture_changed));
	}

	icon = p_icon;

	if (icon.is_valid()) {
		icon->connect_changed(callable_mp(this, &Button::_texture_changed));
	}

	queue_redraw();
	update_minimum_size();
}

void Button::_texture_changed() {
	queue_redraw();
	update_minimum_size();
}

Ref<Texture2D> Button::get_icon() const {
	return icon;
}

void Button::set_expand_icon(bool p_enabled) {
	if (expand_icon != p_enabled) {
		expand_icon = p_enabled;
		_queue_update_size_cache();
		queue_redraw();
		update_minimum_size();
	}
}

bool Button::is_expand_icon() const {
	return expand_icon;
}

void Button::set_flat(bool p_enabled) {
	if (flat != p_enabled) {
		flat = p_enabled;
		queue_redraw();
	}
}

bool Button::is_flat() const {
	return flat;
}

void Button::set_clip_text(bool p_enabled) {
	if (clip_text != p_enabled) {
		clip_text = p_enabled;
		queue_redraw();
		update_minimum_size();
	}
}

bool Button::get_clip_text() const {
	return clip_text;
}

void Button::set_text_alignment(HorizontalAlignment p_alignment) {
	if (alignment != p_alignment) {
		alignment = p_alignment;
		queue_redraw();
	}
}

HorizontalAlignment Button::get_text_alignment() const {
	return alignment;
}

void Button::set_icon_alignment(HorizontalAlignment p_alignment) {
	horizontal_icon_alignment = p_alignment;
	update_minimum_size();
	queue_redraw();
}

void Button::set_vertical_icon_alignment(VerticalAlignment p_alignment) {
	vertical_icon_alignment = p_alignment;
	update_minimum_size();
	queue_redraw();
}

HorizontalAlignment Button::get_icon_alignment() const {
	return horizontal_icon_alignment;
}

VerticalAlignment Button::get_vertical_icon_alignment() const {
	return vertical_icon_alignment;
}

void Button::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_text", "text"), &Button::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &Button::get_text);
	ClassDB::bind_method(D_METHOD("set_text_overrun_behavior", "overrun_behavior"), &Button::set_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("get_text_overrun_behavior"), &Button::get_text_overrun_behavior);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Button::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Button::get_text_direction);
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
	ClassDB::bind_method(D_METHOD("set_vertical_icon_alignment", "vertical_icon_alignment"), &Button::set_vertical_icon_alignment);
	ClassDB::bind_method(D_METHOD("get_vertical_icon_alignment"), &Button::get_vertical_icon_alignment);
	ClassDB::bind_method(D_METHOD("set_expand_icon", "enabled"), &Button::set_expand_icon);
	ClassDB::bind_method(D_METHOD("is_expand_icon"), &Button::is_expand_icon);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_button_icon", "get_button_icon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");

	ADD_GROUP("Text Behavior", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_text_alignment", "get_text_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis,Word Ellipsis"), "set_text_overrun_behavior", "get_text_overrun_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_text"), "set_clip_text", "get_clip_text");

	ADD_GROUP("Icon Behavior", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "icon_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_icon_alignment", "get_icon_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vertical_icon_alignment", PROPERTY_HINT_ENUM, "Top,Center,Bottom"), "set_vertical_icon_alignment", "get_vertical_icon_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expand_icon"), "set_expand_icon", "is_expand_icon");

	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, normal);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, normal_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, pressed_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, hover);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, hover_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, hover_pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, hover_pressed_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, disabled);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, disabled_mirrored);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, Button, focus);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_focus_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_hover_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_hover_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_disabled_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, Button, font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, Button, font_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, Button, outline_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, font_outline_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, icon_normal_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, icon_focus_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, icon_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, icon_hover_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, icon_hover_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, Button, icon_disabled_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, Button, icon);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, Button, h_separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, Button, icon_max_width);
}

Button::Button(const String &p_text) {
	text_buf.instantiate();
	text_buf->set_break_flags(TextServer::BREAK_MANDATORY | TextServer::BREAK_TRIM_EDGE_SPACES);
	set_mouse_filter(MOUSE_FILTER_STOP);

	set_text(p_text);
}

Button::~Button() {
}
