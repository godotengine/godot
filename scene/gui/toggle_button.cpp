/**************************************************************************/
/*  check_box.cpp                                                         */
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

#include "toggle_button.h"

#include "scene/theme/theme_db.h"
#include "servers/display/accessibility_server.h"

Size2 ToggleButton::get_icon_size() const {
	Size2 tex_size = Size2(0, 0);
	if (theme_cache.checked.is_valid()) {
		tex_size = theme_cache.checked->get_size();
	}
	if (theme_cache.checked_hover.is_valid()) {
		tex_size = tex_size.max(theme_cache.checked_hover->get_size());
	}
	if (theme_cache.checked_pressed.is_valid()) {
		tex_size = tex_size.max(theme_cache.checked_pressed->get_size());
	}
	if (theme_cache.checked_hover_pressed.is_valid()) {
		tex_size = tex_size.max(theme_cache.checked_hover_pressed->get_size());
	}
	if (theme_cache.checked_disabled.is_valid()) {
		tex_size = tex_size.max(theme_cache.checked_disabled->get_size());
	}
	if (theme_cache.unchecked.is_valid()) {
		tex_size = tex_size.max(theme_cache.unchecked->get_size());
	}
	if (theme_cache.unchecked_hover.is_valid()) {
		tex_size = tex_size.max(theme_cache.unchecked_hover->get_size());
	}
	if (theme_cache.unchecked_pressed.is_valid()) {
		tex_size = tex_size.max(theme_cache.unchecked_pressed->get_size());
	}
	if (theme_cache.unchecked_hover_pressed.is_valid()) {
		tex_size = tex_size.max(theme_cache.unchecked_hover_pressed->get_size());
	}
	if (theme_cache.unchecked_disabled.is_valid()) {
		tex_size = tex_size.max(theme_cache.unchecked_disabled->get_size());
	}
	return _fit_icon_size(tex_size);
}

Size2 ToggleButton::get_minimum_size() const {
	Size2 minsize = Button::get_minimum_size();
	const Size2 tex_size = get_icon_size();
	if (tex_size.width > 0 || tex_size.height > 0) {
		const Size2 padding = _get_largest_stylebox_size();
		Size2 content_size = minsize - padding;
		if (content_size.width > 0 && tex_size.width > 0) {
			content_size.width += MAX(0, theme_cache.h_separation);
		}
		content_size.width += tex_size.width;
		content_size.height = MAX(content_size.height, tex_size.height);

		minsize = content_size + padding;
	}

	return minsize;
}

std::tuple<Ref<Texture2D>, Ref<Texture2D>> ToggleButton::_get_current_icon() const {
	Ref<Texture2D> icon = theme_cache.checked;
	if (!is_pressed()) {
		icon = theme_cache.unchecked;
	}
	
	Ref<Texture2D> icon_focus = theme_cache.checked_focus;
	if (!is_pressed()) {
		icon_focus = theme_cache.unchecked_focus;
	}

	#define APPLY_ICONS(state_name) \
		if (has_theme_icon(SNAME(#state_name))) icon = theme_cache.state_name; \
		if (has_theme_icon(SNAME(#state_name "_focus"))) icon_focus = theme_cache.state_name##_focus; \

	switch (get_draw_mode()) {
		case DRAW_NORMAL:
			if (is_pressed()) {
				if (is_pressing()) {
					APPLY_ICONS(checked_pressed)
				} else {
					APPLY_ICONS(checked)
				}
			} else {
				if (is_pressing()) {
					APPLY_ICONS(unchecked_pressed)
				} else {
					APPLY_ICONS(unchecked)
				}
			}
			break;
		case DRAW_HOVER:
			if (is_pressed()) {
				APPLY_ICONS(checked_hover)
			} else {
				APPLY_ICONS(unchecked_hover)
			}
			break;
		case DRAW_PRESSED:
			if (is_pressed()) {
				if (is_pressing()) {
					APPLY_ICONS(checked_pressed)
				} else {
					APPLY_ICONS(checked)
				}
			} else {
				if (is_pressing()) {
					APPLY_ICONS(unchecked_pressed)
				} else {
					APPLY_ICONS(unchecked)
				}
			}
			break;
		case DRAW_HOVER_PRESSED:
			if (is_pressed()) {
				if (is_pressing()) {
					APPLY_ICONS(checked_hover_pressed)
				} else {
					APPLY_ICONS(checked_hover)
				}
			} else {
				if (is_pressing()) {
					APPLY_ICONS(unchecked_hover_pressed)
				} else {
					APPLY_ICONS(unchecked_hover)
				}
			}
			break;
		case DRAW_DISABLED:
			if (is_pressed()) {
				APPLY_ICONS(checked_disabled)
			} else {
				APPLY_ICONS(unchecked_disabled)
			}
			break;
	}

#undef APPLY_ICONS

	return std::make_tuple(icon, icon_focus);
}

void ToggleButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			if (is_radio()) {
				AccessibilityServer::get_singleton()->update_set_role(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_RADIO_BUTTON);
			} else {
				AccessibilityServer::get_singleton()->update_set_role(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_CHECK_BOX);
			}
		} break;

		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (theme_cache.button_on_right || right_aligned) {
				if (is_layout_rtl()) {
					_set_internal_margin(SIDE_LEFT, get_icon_size().width);
					_set_internal_margin(SIDE_RIGHT, 0.f);
				} else {
					_set_internal_margin(SIDE_LEFT, 0.f);
					_set_internal_margin(SIDE_RIGHT, get_icon_size().width);
				}
			} else {
				if (is_layout_rtl()) {
					_set_internal_margin(SIDE_LEFT, 0.f);
					_set_internal_margin(SIDE_RIGHT, get_icon_size().width);
				} else {
					_set_internal_margin(SIDE_LEFT, get_icon_size().width);
					_set_internal_margin(SIDE_RIGHT, 0.f);
				}
			}
		} break;

		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();

			std::tuple<Ref<Texture2D>, Ref<Texture2D>> icons = _get_current_icon();
			Ref<Texture2D> icon = std::get<0>(icons);
			Ref<Texture2D> icon_focus = std::get<1>(icons);

			Vector2 ofs;
			if (theme_cache.button_on_right || right_aligned) {
				if (is_layout_rtl()) {
					ofs.x = theme_cache.normal_style->get_margin(SIDE_LEFT);
				} else {
					ofs.x = get_size().width - (get_icon_size().width + theme_cache.normal_style->get_margin(SIDE_RIGHT));
				}
				ofs.y = (get_size().height - get_icon_size().height) / 2 + theme_cache.check_v_offset;
			} else {
				if (is_layout_rtl()) {
					ofs.x = get_size().x - theme_cache.normal_style->get_margin(SIDE_RIGHT) - get_icon_size().width;
				} else {
					ofs.x = theme_cache.normal_style->get_margin(SIDE_LEFT);
				}
				ofs.y = int((get_size().height - get_icon_size().height) / 2) + theme_cache.check_v_offset;
			}

			Color color = theme_cache.checkbox_checked_color;
			if (!is_pressed()) {
				color = theme_cache.checkbox_unchecked_color;
			}

			icon->draw_rect(ci, Rect2(ofs, _fit_icon_size(icon->get_size())), false, color);
			if (has_focus(true) && icon_focus.is_valid() && !icon_focus.is_null()) {
				icon_focus->draw_rect(ci, Rect2(ofs, _fit_icon_size(icon_focus->get_size())), false, color);
			}
		} break;
	}
}

bool ToggleButton::is_radio() const {
	return get_button_group().is_valid();
}

void ToggleButton::_bind_methods() {
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ToggleButton, h_separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ToggleButton, check_v_offset);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, ToggleButton, button_on_right);
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, ToggleButton, normal_style, "normal");

	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_hover);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_hover_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_pressed_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_hover_pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_hover_pressed_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_disabled);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, checked_disabled_focus);
	
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_hover);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_hover_pressed);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_hover_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_pressed_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_hover_pressed_focus);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_disabled);
	BIND_THEME_ITEM(Theme::DATA_TYPE_ICON, ToggleButton, unchecked_disabled_focus);
	
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, ToggleButton, checkbox_checked_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, ToggleButton, checkbox_unchecked_color);
}

ToggleButton::ToggleButton(const String &p_text) :
		Button(p_text) {
	set_toggle_mode(true);

	set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);

	if (is_layout_rtl()) {
		_set_internal_margin(SIDE_RIGHT, get_icon_size().width);
	} else {
		_set_internal_margin(SIDE_LEFT, get_icon_size().width);
	}
}

