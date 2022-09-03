/*************************************************************************/
/*  check_box.cpp                                                        */
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

#include "check_box.h"

#include "servers/rendering_server.h"

Size2 CheckBox::get_icon_size() const {
	Size2 tex_size = Size2(0, 0);
	if (!theme_cache.checked.is_null()) {
		tex_size = Size2(theme_cache.checked->get_width(), theme_cache.checked->get_height());
	}
	if (!theme_cache.unchecked.is_null()) {
		tex_size = Size2(MAX(tex_size.width, theme_cache.unchecked->get_width()), MAX(tex_size.height, theme_cache.unchecked->get_height()));
	}
	if (!theme_cache.radio_checked.is_null()) {
		tex_size = Size2(MAX(tex_size.width, theme_cache.radio_checked->get_width()), MAX(tex_size.height, theme_cache.radio_checked->get_height()));
	}
	if (!theme_cache.radio_unchecked.is_null()) {
		tex_size = Size2(MAX(tex_size.width, theme_cache.radio_unchecked->get_width()), MAX(tex_size.height, theme_cache.radio_unchecked->get_height()));
	}
	if (!theme_cache.checked_disabled.is_null()) {
		tex_size = Size2(MAX(tex_size.width, theme_cache.checked_disabled->get_width()), MAX(tex_size.height, theme_cache.checked_disabled->get_height()));
	}
	if (!theme_cache.unchecked_disabled.is_null()) {
		tex_size = Size2(MAX(tex_size.width, theme_cache.unchecked_disabled->get_width()), MAX(tex_size.height, theme_cache.unchecked_disabled->get_height()));
	}
	if (!theme_cache.radio_checked_disabled.is_null()) {
		tex_size = Size2(MAX(tex_size.width, theme_cache.radio_checked_disabled->get_width()), MAX(tex_size.height, theme_cache.radio_checked_disabled->get_height()));
	}
	if (!theme_cache.radio_unchecked_disabled.is_null()) {
		tex_size = Size2(MAX(tex_size.width, theme_cache.radio_unchecked_disabled->get_width()), MAX(tex_size.height, theme_cache.radio_unchecked_disabled->get_height()));
	}
	return tex_size;
}

Size2 CheckBox::get_minimum_size() const {
	Size2 minsize = Button::get_minimum_size();
	Size2 tex_size = get_icon_size();
	minsize.width += tex_size.width;
	if (get_text().length() > 0) {
		minsize.width += MAX(0, theme_cache.h_separation);
	}
	minsize.height = MAX(minsize.height, tex_size.height + theme_cache.normal_style->get_margin(SIDE_TOP) + theme_cache.normal_style->get_margin(SIDE_BOTTOM));

	return minsize;
}

void CheckBox::_update_theme_item_cache() {
	Button::_update_theme_item_cache();

	theme_cache.h_separation = get_theme_constant(SNAME("h_separation"));
	theme_cache.check_v_adjust = get_theme_constant(SNAME("check_v_adjust"));
	theme_cache.normal_style = get_theme_stylebox(SNAME("normal"));

	theme_cache.checked = get_theme_icon(SNAME("checked"));
	theme_cache.unchecked = get_theme_icon(SNAME("unchecked"));
	theme_cache.radio_checked = get_theme_icon(SNAME("radio_checked"));
	theme_cache.radio_unchecked = get_theme_icon(SNAME("radio_unchecked"));
	theme_cache.checked_disabled = get_theme_icon(SNAME("checked_disabled"));
	theme_cache.unchecked_disabled = get_theme_icon(SNAME("unchecked_disabled"));
	theme_cache.radio_checked_disabled = get_theme_icon(SNAME("radio_checked_disabled"));
	theme_cache.radio_unchecked_disabled = get_theme_icon(SNAME("radio_unchecked_disabled"));
}

void CheckBox::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (is_layout_rtl()) {
				_set_internal_margin(SIDE_LEFT, 0.f);
				_set_internal_margin(SIDE_RIGHT, get_icon_size().width);
			} else {
				_set_internal_margin(SIDE_LEFT, get_icon_size().width);
				_set_internal_margin(SIDE_RIGHT, 0.f);
			}
		} break;

		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();

			Ref<Texture2D> on_tex;
			Ref<Texture2D> off_tex;

			if (is_radio()) {
				if (is_disabled()) {
					on_tex = theme_cache.radio_checked_disabled;
					off_tex = theme_cache.radio_unchecked_disabled;
				} else {
					on_tex = theme_cache.radio_checked;
					off_tex = theme_cache.radio_unchecked;
				}
			} else {
				if (is_disabled()) {
					on_tex = theme_cache.checked_disabled;
					off_tex = theme_cache.unchecked_disabled;
				} else {
					on_tex = theme_cache.checked;
					off_tex = theme_cache.unchecked;
				}
			}

			Vector2 ofs;
			if (is_layout_rtl()) {
				ofs.x = get_size().x - theme_cache.normal_style->get_margin(SIDE_RIGHT) - get_icon_size().width;
			} else {
				ofs.x = theme_cache.normal_style->get_margin(SIDE_LEFT);
			}
			ofs.y = int((get_size().height - get_icon_size().height) / 2) + theme_cache.check_v_adjust;

			if (is_pressed()) {
				on_tex->draw(ci, ofs);
			} else {
				off_tex->draw(ci, ofs);
			}
		} break;
	}
}

bool CheckBox::is_radio() {
	return get_button_group().is_valid();
}

CheckBox::CheckBox(const String &p_text) :
		Button(p_text) {
	set_toggle_mode(true);

	set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);

	if (is_layout_rtl()) {
		_set_internal_margin(SIDE_RIGHT, get_icon_size().width);
	} else {
		_set_internal_margin(SIDE_LEFT, get_icon_size().width);
	}
}

CheckBox::~CheckBox() {
}
