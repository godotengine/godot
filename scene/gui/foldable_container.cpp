/**************************************************************************/
/*  foldable_container.cpp                                                */
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

#include "foldable_container.h"

#include "scene/theme/theme_db.h"

Size2 FoldableContainer::get_minimum_size() const {
	Size2 title_ms = _get_title_panel_min_size();

	if (!expanded) {
		return title_ms;
	}
	Size2 ms;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || !c->is_visible()) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}
		ms = ms.max(c->get_combined_minimum_size());
	}
	if (theme_cache.panel_style.is_valid()) {
		ms += theme_cache.panel_style->get_minimum_size();
	}
	return Size2(MAX(ms.width, title_ms.width), ms.height + title_ms.height);
}

Vector<int> FoldableContainer::get_allowed_size_flags_horizontal() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

Vector<int> FoldableContainer::get_allowed_size_flags_vertical() const {
	Vector<int> flags;
	flags.append(SIZE_FILL);
	flags.append(SIZE_SHRINK_BEGIN);
	flags.append(SIZE_SHRINK_CENTER);
	flags.append(SIZE_SHRINK_END);
	return flags;
}

void FoldableContainer::set_expanded(bool p_expanded) {
	if (expanded == p_expanded) {
		return;
	}
	expanded = p_expanded;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c || c->is_set_as_top_level()) {
			continue;
		}
		c->set_visible(expanded);
	}
	update_minimum_size();
	queue_redraw();
	emit_signal(SNAME("folding_changed"), !expanded);
}

bool FoldableContainer::is_expanded() const {
	return expanded;
}

void FoldableContainer::set_title(const String &p_title) {
	if (title == p_title) {
		return;
	}
	title = p_title;
	_shape();
	queue_redraw();
	update_minimum_size();
}

String FoldableContainer::get_title() const {
	return title;
}

void FoldableContainer::set_title_alignment(HorizontalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 3);
	HorizontalAlignment alignment = p_alignment;

	if (is_layout_rtl()) {
		if (p_alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
			alignment = HORIZONTAL_ALIGNMENT_LEFT;
		} else if (p_alignment == HORIZONTAL_ALIGNMENT_LEFT) {
			alignment = HORIZONTAL_ALIGNMENT_RIGHT;
		}
	}
	if (text_buf->get_horizontal_alignment() == alignment) {
		return;
	}
	if (title_alignment != p_alignment) {
		title_alignment = p_alignment;
	}
	text_buf->set_horizontal_alignment(alignment);
	queue_redraw();
}

HorizontalAlignment FoldableContainer::get_title_alignment() const {
	return title_alignment;
}

void FoldableContainer::set_language(const String &p_language) {
	if (language == p_language) {
		return;
	}
	language = p_language;
	_shape();
	queue_redraw();
	update_minimum_size();
}

String FoldableContainer::get_language() const {
	return language;
}

void FoldableContainer::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction == p_text_direction) {
		return;
	}
	text_direction = p_text_direction;

	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		text_buf->set_direction((TextServer::Direction)text_direction);
	}
	queue_redraw();
	update_minimum_size();
}

Control::TextDirection FoldableContainer::get_text_direction() const {
	return text_direction;
}

void FoldableContainer::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		if (m->get_position().y <= title_panel_height) {
			if (!is_hovering) {
				is_hovering = true;
				queue_redraw();
			}
		} else if (is_hovering) {
			is_hovering = false;
			queue_redraw();
		}
	}

	if (has_focus() && p_event->is_action_pressed("ui_accept", false, true)) {
		set_expanded(!expanded);
		accept_event();
		return;
	}

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (b->is_pressed() && b->get_button_index() == MouseButton::LEFT && b->get_position().y <= title_panel_height) {
			set_expanded(!expanded);
			accept_event();
			return;
		}
	}
}

void FoldableContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			title_panel_height = _get_title_panel_min_size().height;
			Ref<Texture2D> icon = _get_title_icon();
			Ref<StyleBox> title_style = _get_title_style();
			Point2 title_pos;
			int title_width = 0;

			if (title_style.is_valid()) {
				title_style->draw(ci, Rect2(Point2(), Size2(get_size().x, title_panel_height)));
				title_pos += title_style->get_offset();
				title_width += get_size().width - title_style->get_minimum_size().width;
			}
			if (icon.is_valid()) {
				int h_separation = MAX(theme_cache.h_separation, 0);
				if (!is_layout_rtl()) {
					title_pos.x += icon->get_width() + h_separation;
				}
				title_width -= icon->get_width() + h_separation;
				Point2 icon_pos;

				if (title_style.is_valid()) {
					icon_pos = title_style->get_offset();
				}
				if (theme_cache.title_font.is_valid() && title.length() > 0) {
					if (text_buf->get_size().height <= icon->get_height()) {
						title_pos.y += (icon->get_height() - text_buf->get_line_ascent()) / 2;
					} else {
						icon_pos.y += (text_buf->get_size().height - icon->get_height()) / 2;
					}
				}
				if (is_layout_rtl()) {
					icon_pos.x = get_size().x - icon_pos.x - icon->get_size().width;
				}
				icon->draw(ci, icon_pos);
			}

			if (theme_cache.title_font.is_valid()) {
				Color font_color = expanded ? theme_cache.title_font_color : theme_cache.title_collapsed_font_color;
				if (is_hovering) {
					font_color = theme_cache.title_hover_font_color;
				}
				text_buf->set_width(title_width);

				if (theme_cache.title_font_outline_size > 0 && theme_cache.title_font_outline_color.a > 0) {
					text_buf->draw_outline(ci, title_pos, theme_cache.title_font_outline_size, theme_cache.title_font_outline_color);
				}
				text_buf->draw(ci, title_pos, font_color);
			}

			if (expanded && theme_cache.panel_style.is_valid()) {
				theme_cache.panel_style->draw(ci, Rect2(0, title_panel_height, get_size().width, get_size().height - title_panel_height));
			}

			if (has_focus() && theme_cache.focus_style.is_valid()) {
				theme_cache.focus_style->draw(ci, Rect2(Point2(), Size2(get_size().width, expanded ? get_size().height : title_panel_height)));
			}
		} break;

		case NOTIFICATION_SORT_CHILDREN: {
			title_panel_height = _get_title_panel_min_size().height;
			Size2 size = get_size();
			size.height -= title_panel_height;

			Point2 ofs;
			ofs.y += title_panel_height;

			if (theme_cache.panel_style.is_valid()) {
				size -= theme_cache.panel_style->get_minimum_size();
				ofs += theme_cache.panel_style->get_offset();
			}

			for (int i = 0; i < get_child_count(); i++) {
				Control *c = Object::cast_to<Control>(get_child(i));
				if (!c || !c->is_visible_in_tree()) {
					continue;
				}
				if (c->is_set_as_top_level()) {
					continue;
				}
				c->set_visible(expanded);
				fit_child_in_rect(c, Rect2(ofs, size));
			}
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			if (is_hovering) {
				is_hovering = false;
				queue_redraw();
			}
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			set_title_alignment(title_alignment);
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_shape();
			update_minimum_size();
			queue_redraw();
		} break;
	}
}

Size2 FoldableContainer::_get_title_panel_min_size() const {
	Size2 s;
	Ref<StyleBox> title_style = expanded ? theme_cache.title_style : theme_cache.title_collapsed_style;
	Ref<Texture2D> icon = _get_title_icon();

	if (title_style.is_valid()) {
		s += title_style->get_minimum_size();
	}
	if (theme_cache.title_font.is_valid()) {
		if (title.length() > 0) {
			s.width += icon.is_valid() ? MAX(0, theme_cache.h_separation) : 0;
			s += text_buf->get_size();
		}
	}
	if (icon.is_valid()) {
		s.width += icon->get_width();
		if (title_style.is_valid()) {
			s.height = MAX(s.height, title_style->get_minimum_size().height + icon->get_height());
		} else {
			s.height = MAX(s.height, icon->get_height());
		}
	}
	return s;
}

Ref<StyleBox> FoldableContainer::_get_title_style() const {
	if (is_hovering) {
		return expanded ? theme_cache.title_hover_style : theme_cache.title_collapsed_hover_style;
	}
	return expanded ? theme_cache.title_style : theme_cache.title_collapsed_style;
}

Ref<Texture2D> FoldableContainer::_get_title_icon() const {
	if (expanded) {
		return theme_cache.arrow;
	} else if (is_layout_rtl()) {
		return theme_cache.arrow_collapsed_mirrored;
	}
	return theme_cache.arrow_collapsed;
}

void FoldableContainer::_shape() {
	text_buf->clear();

	if (!theme_cache.title_font.is_valid()) {
		return;
	}
	text_buf->add_string(atr(title), theme_cache.title_font, theme_cache.title_font_size, language);
}

void FoldableContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_expanded", "expanded"), &FoldableContainer::set_expanded);
	ClassDB::bind_method(D_METHOD("is_expanded"), &FoldableContainer::is_expanded);
	ClassDB::bind_method(D_METHOD("set_title", "title"), &FoldableContainer::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &FoldableContainer::get_title);
	ClassDB::bind_method(D_METHOD("set_title_alignment", "alignment"), &FoldableContainer::set_title_alignment);
	ClassDB::bind_method(D_METHOD("get_title_alignment"), &FoldableContainer::get_title_alignment);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &FoldableContainer::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &FoldableContainer::get_language);
	ClassDB::bind_method(D_METHOD("set_text_direction", "text_direction"), &FoldableContainer::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &FoldableContainer::get_text_direction);

	ADD_SIGNAL(MethodInfo("folding_changed", PropertyInfo(Variant::BOOL, "is_folded")));

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expanded"), "set_expanded", "is_expanded");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "title_alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_title_alignment", "get_title_alignment");

	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID), "set_language", "get_language");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_style, "title_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_hover_style, "title_hover_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_collapsed_style, "title_collapsed_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, title_collapsed_hover_style, "title_collapsed_hover_panel");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, focus_style, "focus");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, FoldableContainer, panel_style, "panel");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_FONT, FoldableContainer, title_font, "font");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_FONT_SIZE, FoldableContainer, title_font_size, "font_size");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_CONSTANT, FoldableContainer, title_font_outline_size, "outline_size");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_font_color, "font_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_hover_font_color, "hover_font_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_collapsed_font_color, "collapsed_font_color");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_COLOR, FoldableContainer, title_font_outline_color, "font_outline_color");

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FoldableContainer, arrow);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FoldableContainer, arrow_collapsed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, FoldableContainer, arrow_collapsed_mirrored);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, FoldableContainer, h_separation);
}

FoldableContainer::FoldableContainer(const String &p_title) {
	text_buf.instantiate();
	set_title(p_title);
	set_focus_mode(FOCUS_ALL);
	set_mouse_filter(MOUSE_FILTER_STOP);
}
