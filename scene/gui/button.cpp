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

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/resources/style_box.h"
#include "scene/theme/theme_db.h"
#include "servers/display/accessibility_server.h"

Size2 Button::_get_minimum_size(bool p_desired) const {
	Ref<Texture2D> _icon = icon;
	if (_icon.is_null() && has_theme_icon(SNAME("icon"))) {
		_icon = theme_cache.icon;
	}

	Size2 text_min_size = text.is_empty() ? Size2() : (p_desired ? text_label->get_desired_size() : text_label->get_minimum_size());
	Size2 icon_min_size = (!expand_icon && _icon.is_valid()) ? _fit_icon_size(_icon->get_size()) : Size2();

	Size2 minsize;

	switch (horizontal_icon_alignment) {
		case HORIZONTAL_ALIGNMENT_CENTER: {
			minsize.width = MAX(text_min_size.width, icon_min_size.width);
		} break;
		case HORIZONTAL_ALIGNMENT_LEFT:
		case HORIZONTAL_ALIGNMENT_RIGHT: {
			minsize.width = text_min_size.width + icon_min_size.width + theme_cache.h_separation;
		} break;
		default: {
			// Button doesn't support horizontal fill alignment for icon, so do nothing here.
		} break;
	}

	switch (vertical_icon_alignment) {
		case VERTICAL_ALIGNMENT_CENTER: {
			minsize.height = MAX(text_min_size.height, icon_min_size.height);
		} break;
		case VERTICAL_ALIGNMENT_TOP:
		case VERTICAL_ALIGNMENT_BOTTOM: {
			minsize.height = text_min_size.height + icon_min_size.height;
		} break;
		default: {
			// Button doesn't support vertical fill alignment for icon, so do nothing here.
		} break;
	}

	return (theme_cache.align_to_largest_stylebox ? _get_largest_stylebox_size() : _get_current_stylebox()->get_minimum_size()) + minsize;
}

Size2 Button::get_minimum_size() const {
	return _get_minimum_size();
}

Size2 Button::get_desired_size() const {
	return _get_minimum_size(true);
}

void Button::_maximum_size_changed() {
	Size2 ms = get_combined_maximum_size();
	ms -= theme_cache.align_to_largest_stylebox ? _get_largest_stylebox_size() : _get_current_stylebox()->get_minimum_size();

	Ref<Texture2D> _icon = icon;
	if (_icon.is_null() && has_theme_icon(SNAME("icon"))) {
		_icon = theme_cache.icon;
	}
	Size2 icon_min_size = (!expand_icon && _icon.is_valid()) ? _fit_icon_size(_icon->get_size()) : Size2();

	if (horizontal_icon_alignment != HORIZONTAL_ALIGNMENT_CENTER) {
		ms.width -= icon_min_size.width;
		ms.width -= icon_min_size.width > 0 ? theme_cache.h_separation : 0;
	}
	if (vertical_icon_alignment != VERTICAL_ALIGNMENT_CENTER) {
		ms.height -= icon_min_size.height;
	}

	text_label->set_parent_maximum_size_cache(ms);
	update_desired_size();
}

void Button::_set_internal_margin(Side p_side, float p_value) {
	_internal_margin[p_side] = p_value;
}

void Button::_queue_update_size_cache() {
}

String Button::_get_translated_text(const String &p_text) const {
	return atr(p_text);
}

void Button::_update_theme_item_cache() {
	Control::_update_theme_item_cache();

	theme_cache.max_style_size = Vector2();
	theme_cache.style_margin_left = 0;
	theme_cache.style_margin_right = 0;
	theme_cache.style_margin_top = 0;
	theme_cache.style_margin_bottom = 0;

	const bool rtl = is_layout_rtl();
	if (rtl && has_theme_stylebox(SNAME("normal_mirrored"))) {
		_update_style_margins(theme_cache.normal_mirrored);
	} else {
		_update_style_margins(theme_cache.normal);
	}
	if (has_theme_stylebox("hover_pressed")) {
		if (rtl && has_theme_stylebox(SNAME("hover_pressed_mirrored"))) {
			_update_style_margins(theme_cache.hover_pressed_mirrored);
		} else {
			_update_style_margins(theme_cache.hover_pressed);
		}
	}
	if (rtl && has_theme_stylebox(SNAME("pressed_mirrored"))) {
		_update_style_margins(theme_cache.pressed_mirrored);
	} else {
		_update_style_margins(theme_cache.pressed);
	}
	if (rtl && has_theme_stylebox(SNAME("hover_mirrored"))) {
		_update_style_margins(theme_cache.hover_mirrored);
	} else {
		_update_style_margins(theme_cache.hover);
	}
	if (rtl && has_theme_stylebox(SNAME("disabled_mirrored"))) {
		_update_style_margins(theme_cache.disabled_mirrored);
	} else {
		_update_style_margins(theme_cache.disabled);
	}
	theme_cache.max_style_size = theme_cache.max_style_size.max(Vector2(theme_cache.style_margin_left + theme_cache.style_margin_right, theme_cache.style_margin_top + theme_cache.style_margin_bottom));

	// Propagate font theme items to the internal Label.
	if (text_label) {
		text_label->add_theme_font_override(SceneStringName(font), theme_cache.font);
		text_label->add_theme_font_size_override(SceneStringName(font_size), theme_cache.font_size);
		text_label->add_theme_constant_override(SNAME("outline_size"), theme_cache.outline_size);
		text_label->add_theme_color_override(SNAME("font_outline_color"), theme_cache.font_outline_color);
		text_label->add_theme_constant_override(SNAME("line_spacing"), theme_cache.line_spacing);
	}
}

Size2 Button::_get_largest_stylebox_size() const {
	return theme_cache.max_style_size;
}

Ref<StyleBox> Button::_get_current_stylebox() const {
	Ref<StyleBox> stylebox = theme_cache.normal;
	const bool rtl = is_layout_rtl();

	switch (get_draw_mode()) {
		case DRAW_NORMAL: {
			if (rtl && has_theme_stylebox(SNAME("normal_mirrored"))) {
				stylebox = theme_cache.normal_mirrored;
			} else {
				stylebox = theme_cache.normal;
			}
		} break;

		case DRAW_HOVER_PRESSED: {
			// Edge case for CheckButton and CheckBox.
			if (has_theme_stylebox("hover_pressed")) {
				if (rtl && has_theme_stylebox(SNAME("hover_pressed_mirrored"))) {
					stylebox = theme_cache.hover_pressed_mirrored;
				} else {
					stylebox = theme_cache.hover_pressed;
				}
				break;
			}
		}
			[[fallthrough]];
		case DRAW_PRESSED: {
			if (rtl && has_theme_stylebox(SNAME("pressed_mirrored"))) {
				stylebox = theme_cache.pressed_mirrored;
			} else {
				stylebox = theme_cache.pressed;
			}
		} break;

		case DRAW_HOVER: {
			if (rtl && has_theme_stylebox(SNAME("hover_mirrored"))) {
				stylebox = theme_cache.hover_mirrored;
			} else {
				stylebox = theme_cache.hover;
			}
		} break;

		case DRAW_DISABLED: {
			if (rtl && has_theme_stylebox(SNAME("disabled_mirrored"))) {
				stylebox = theme_cache.disabled_mirrored;
			} else {
				stylebox = theme_cache.disabled;
			}
		} break;
	}

	return stylebox;
}

String Button::_get_accessibility_name() const {
	const String &ac_name = Control::_get_accessibility_name();
	const String &label_text = text_label->get_text();
	if (!label_text.is_empty() && ac_name.is_empty()) {
		return label_text;
	} else if (!label_text.is_empty() && !ac_name.is_empty() && ac_name != label_text) {
		return ac_name + ": " + label_text;
	} else if (label_text.is_empty() && ac_name.is_empty() && !get_tooltip_text().is_empty()) {
		return get_tooltip_text(); // Fall back to tooltip.
	} else {
		return ac_name;
	}
}

void Button::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			AcceptDialog *dlg = Object::cast_to<AcceptDialog>(get_parent());
			if (dlg && dlg->get_ok_button() == this) {
				AccessibilityServer::get_singleton()->update_set_role(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_DEFAULT_BUTTON);
			}
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			queue_redraw();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			text_label->set_text(_get_translated_text(text));

			update_desired_size();
			update_minimum_size();
			update_configuration_warnings();
			queue_accessibility_update();
			queue_redraw();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			update_desired_size();
			update_minimum_size();
			queue_redraw();
		} break;

		case NOTIFICATION_RESIZED: {
			queue_redraw();
		} break;

		case NOTIFICATION_DRAW: {
			const RID ci = get_canvas_item();
			const Size2 size = get_size();

			Ref<StyleBox> style = _get_current_stylebox();
			// Draws the stylebox in the current state.
			if (!flat) {
				style->draw(ci, Rect2(Point2(), size));
			}

			if (has_focus(true)) {
				theme_cache.focus->draw(ci, Rect2(Point2(), size));
			}

			Ref<Texture2D> _icon = icon;
			if (_icon.is_null() && has_theme_icon(SNAME("icon"))) {
				_icon = theme_cache.icon;
			}

			const String &label_text = text_label->get_text();

			if (label_text.is_empty() && _icon.is_null()) {
				text_label->set_visible(false);
				break;
			}

			const float style_margin_left = (theme_cache.align_to_largest_stylebox) ? theme_cache.style_margin_left : style->get_margin(SIDE_LEFT);
			const float style_margin_right = (theme_cache.align_to_largest_stylebox) ? theme_cache.style_margin_right : style->get_margin(SIDE_RIGHT);
			const float style_margin_top = (theme_cache.align_to_largest_stylebox) ? theme_cache.style_margin_top : style->get_margin(SIDE_TOP);
			const float style_margin_bottom = (theme_cache.align_to_largest_stylebox) ? theme_cache.style_margin_bottom : style->get_margin(SIDE_BOTTOM);

			Size2 drawable_size_remained = size;

			{ // The size after the stylebox is stripped.
				drawable_size_remained.width -= style_margin_left + style_margin_right;
				drawable_size_remained.height -= style_margin_top + style_margin_bottom;
			}

			const int h_separation = MAX(0, theme_cache.h_separation);

			float left_internal_margin_with_h_separation = _internal_margin[SIDE_LEFT];
			float right_internal_margin_with_h_separation = _internal_margin[SIDE_RIGHT];
			{ // The width reserved for internal element in derived classes (and h_separation if needed).

				if (_internal_margin[SIDE_LEFT] > 0.0f) {
					left_internal_margin_with_h_separation += h_separation;
				}

				if (_internal_margin[SIDE_RIGHT] > 0.0f) {
					right_internal_margin_with_h_separation += h_separation;
				}

				drawable_size_remained.width -= left_internal_margin_with_h_separation + right_internal_margin_with_h_separation; // The size after the internal element is stripped.
			}

			HorizontalAlignment icon_align_rtl_checked = horizontal_icon_alignment;
			// Swap icon alignment sides if right-to-left layout is set.
			if (is_layout_rtl()) {
				if (horizontal_icon_alignment == HORIZONTAL_ALIGNMENT_RIGHT) {
					icon_align_rtl_checked = HORIZONTAL_ALIGNMENT_LEFT;
				} else if (horizontal_icon_alignment == HORIZONTAL_ALIGNMENT_LEFT) {
					icon_align_rtl_checked = HORIZONTAL_ALIGNMENT_RIGHT;
				}
			}

			Color font_color;
			Color icon_modulate_color(1, 1, 1, 1);
			// Get the font color and icon modulate color in the current state.
			switch (get_draw_mode()) {
				case DRAW_NORMAL: {
					// Focus colors only take precedence over normal state.
					if (has_focus(true)) {
						font_color = theme_cache.font_focus_color;
						if (has_theme_color(SNAME("icon_focus_color"))) {
							icon_modulate_color = theme_cache.icon_focus_color;
						}
					} else {
						font_color = theme_cache.font_color;
						if (has_theme_color(SNAME("icon_normal_color"))) {
							icon_modulate_color = theme_cache.icon_normal_color;
						}
					}
				} break;
				case DRAW_HOVER_PRESSED: {
					font_color = theme_cache.font_hover_pressed_color;
					if (has_theme_color(SNAME("icon_hover_pressed_color"))) {
						icon_modulate_color = theme_cache.icon_hover_pressed_color;
					}

				} break;
				case DRAW_PRESSED: {
					if (has_theme_color(SNAME("font_pressed_color"))) {
						font_color = theme_cache.font_pressed_color;
					} else {
						font_color = theme_cache.font_color;
					}
					if (has_theme_color(SNAME("icon_pressed_color"))) {
						icon_modulate_color = theme_cache.icon_pressed_color;
					}

				} break;
				case DRAW_HOVER: {
					font_color = theme_cache.font_hover_color;
					if (has_theme_color(SNAME("icon_hover_color"))) {
						icon_modulate_color = theme_cache.icon_hover_color;
					}

				} break;
				case DRAW_DISABLED: {
					font_color = theme_cache.font_disabled_color;
					if (has_theme_color(SNAME("icon_disabled_color"))) {
						icon_modulate_color = theme_cache.icon_disabled_color;
					} else {
						icon_modulate_color.a = 0.4;
					}

				} break;
			}

			const bool is_clipped = text_label->is_clipping_text() || text_label->get_text_overrun_behavior() != TextServer::OVERRUN_NO_TRIMMING || text_label->get_autowrap_mode() != TextServer::AUTOWRAP_OFF;
			const Size2 custom_element_size = drawable_size_remained;

			// Draw the icon.
			if (_icon.is_valid()) {
				Size2 icon_size;

				{ // Calculate the drawing size of the icon.
					icon_size = _icon->get_size();

					if (expand_icon) {
						const Size2 text_min_size = text_label->get_minimum_size();
						Size2 _size = custom_element_size;
						if (!is_clipped && icon_align_rtl_checked != HORIZONTAL_ALIGNMENT_CENTER && text_min_size.width > 1.0f) {
							// If there is not enough space for icon and h_separation, h_separation will occupy the space first,
							// so the icon's width may be negative. Keep it negative to make it easier to calculate the space
							// reserved for text later.
							_size.width -= text_min_size.width + h_separation;
						}
						if (vertical_icon_alignment != VERTICAL_ALIGNMENT_CENTER && !label_text.is_empty()) {
							_size.height -= text_min_size.height;
						}

						float icon_width = icon_size.width * _size.height / icon_size.height;
						float icon_height = _size.height;

						if (icon_width > _size.width) {
							icon_width = _size.width;
							icon_height = icon_size.height * icon_width / icon_size.width;
						}

						icon_size = Size2(icon_width, icon_height);
					}
					icon_size = _fit_icon_size(icon_size);
					icon_size = icon_size.round();
				}

				if (icon_size.width > 0.0f) {
					// Calculate the drawing position of the icon.
					Point2 icon_ofs;

					switch (icon_align_rtl_checked) {
						case HORIZONTAL_ALIGNMENT_CENTER: {
							icon_ofs.x = (custom_element_size.width - icon_size.width) / 2.0f;
						}
							[[fallthrough]];
						case HORIZONTAL_ALIGNMENT_FILL:
						case HORIZONTAL_ALIGNMENT_LEFT: {
							icon_ofs.x += style_margin_left;
							icon_ofs.x += left_internal_margin_with_h_separation;
						} break;

						case HORIZONTAL_ALIGNMENT_RIGHT: {
							icon_ofs.x = size.x - style_margin_right;
							icon_ofs.x -= right_internal_margin_with_h_separation;
							icon_ofs.x -= icon_size.width;
						} break;
					}

					switch (vertical_icon_alignment) {
						case VERTICAL_ALIGNMENT_CENTER: {
							icon_ofs.y = (custom_element_size.height - icon_size.height) / 2.0f;
						}
							[[fallthrough]];
						case VERTICAL_ALIGNMENT_FILL:
						case VERTICAL_ALIGNMENT_TOP: {
							icon_ofs.y += style_margin_top;
						} break;

						case VERTICAL_ALIGNMENT_BOTTOM: {
							icon_ofs.y = size.y - style_margin_bottom - icon_size.height;
						} break;
					}
					icon_ofs = icon_ofs.floor();

					Rect2 icon_region = Rect2(icon_ofs, icon_size);
					draw_texture_rect(_icon, icon_region, false, icon_modulate_color);
				}

				if (!label_text.is_empty()) {
					// Update the size after the icon is stripped. Stripping only when the icon alignments are not center.
					if (icon_align_rtl_checked != HORIZONTAL_ALIGNMENT_CENTER) {
						// Subtract the space's width occupied by icon and h_separation together.
						drawable_size_remained.width -= icon_size.width + h_separation;
					}

					if (vertical_icon_alignment != VERTICAL_ALIGNMENT_CENTER) {
						drawable_size_remained.height -= icon_size.height;
					}
				}
			}

			// Position the internal Label for text rendering.
			if (!label_text.is_empty()) {
				text_label->set_visible(true);

				Point2 label_pos;
				label_pos.x = style_margin_left + left_internal_margin_with_h_separation;
				label_pos.y = style_margin_top;

				if (icon_align_rtl_checked == HORIZONTAL_ALIGNMENT_LEFT && _icon.is_valid()) {
					label_pos.x += custom_element_size.width - drawable_size_remained.width;
				}
				if (vertical_icon_alignment == VERTICAL_ALIGNMENT_TOP && _icon.is_valid()) {
					label_pos.y += custom_element_size.height - drawable_size_remained.height;
				}

				Size2 label_size;
				label_size.width = MAX(1.0f, drawable_size_remained.width);
				label_size.height = MAX(1.0f, drawable_size_remained.height);

				if (text_label->get_position() != label_pos) {
					text_label->set_position(label_pos);
				}
				if (text_label->get_size() != label_size) {
					text_label->set_size(label_size);
				}

				// Update font color based on current draw state.
				text_label->add_theme_color_override(SceneStringName(font_color), font_color);
			} else {
				text_label->set_visible(false);
			}
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

Label *Button::_get_text_label() const {
	return text_label;
}

void Button::set_text_overrun_behavior(TextServer::OverrunBehavior p_behavior) {
	if (text_label->get_text_overrun_behavior() != p_behavior) {
		bool need_update_cache = text_label->get_text_overrun_behavior() == TextServer::OVERRUN_NO_TRIMMING || p_behavior == TextServer::OVERRUN_NO_TRIMMING;
		text_label->set_text_overrun_behavior(p_behavior);

		if (need_update_cache) {
			_queue_update_size_cache();
		}
		queue_redraw();
		update_desired_size();
		update_minimum_size();
	}
}

TextServer::OverrunBehavior Button::get_text_overrun_behavior() const {
	return text_label->get_text_overrun_behavior();
}

void Button::set_text(const String &p_text) {
	const String translated_text = _get_translated_text(p_text);
	if (text == p_text && text_label->get_text() == translated_text) {
		return;
	}
	text = p_text;
	text_label->set_text(translated_text);

	update_configuration_warnings();
	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

String Button::get_text() const {
	return text;
}

void Button::set_autowrap_mode(TextServer::AutowrapMode p_mode) {
	if (text_label->get_autowrap_mode() != p_mode) {
		text_label->set_autowrap_mode(p_mode);
		queue_redraw();
		update_desired_size();
		update_minimum_size();
	}
}

TextServer::AutowrapMode Button::get_autowrap_mode() const {
	return text_label->get_autowrap_mode();
}

void Button::set_autowrap_trim_flags(BitField<TextServer::LineBreakFlag> p_flags) {
	if (text_label->get_autowrap_trim_flags() != (p_flags & TextServer::BREAK_TRIM_MASK)) {
		text_label->set_autowrap_trim_flags(p_flags);
		queue_redraw();
		update_desired_size();
		update_minimum_size();
	}
}

BitField<TextServer::LineBreakFlag> Button::get_autowrap_trim_flags() const {
	return text_label->get_autowrap_trim_flags();
}

void Button::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_label->get_text_direction() != p_text_direction) {
		text_label->set_text_direction(p_text_direction);
		queue_accessibility_update();
		queue_redraw();
	}
}

Control::TextDirection Button::get_text_direction() const {
	return text_label->get_text_direction();
}

void Button::set_language(const String &p_language) {
	if (text_label->get_language() != p_language) {
		text_label->set_language(p_language);
		queue_accessibility_update();
		queue_redraw();
	}
}

String Button::get_language() const {
	return text_label->get_language();
}

void Button::set_button_icon(const Ref<Texture2D> &p_icon) {
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
	update_desired_size();
	update_minimum_size();
}

void Button::_texture_changed() {
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

void Button::_update_style_margins(const Ref<StyleBox> &p_stylebox) {
	theme_cache.max_style_size = theme_cache.max_style_size.max(p_stylebox->get_minimum_size());
	theme_cache.style_margin_left = MAX(theme_cache.style_margin_left, p_stylebox->get_margin(SIDE_LEFT));
	theme_cache.style_margin_right = MAX(theme_cache.style_margin_right, p_stylebox->get_margin(SIDE_RIGHT));
	theme_cache.style_margin_top = MAX(theme_cache.style_margin_top, p_stylebox->get_margin(SIDE_TOP));
	theme_cache.style_margin_bottom = MAX(theme_cache.style_margin_bottom, p_stylebox->get_margin(SIDE_BOTTOM));
}

Ref<Texture2D> Button::get_button_icon() const {
	return icon;
}

void Button::set_expand_icon(bool p_enabled) {
	if (expand_icon != p_enabled) {
		expand_icon = p_enabled;
		_queue_update_size_cache();
		queue_redraw();
		update_desired_size();
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
	if (text_label->is_clipping_text() != p_enabled) {
		text_label->set_clip_text(p_enabled);

		_queue_update_size_cache();
		queue_redraw();
		update_desired_size();
		update_minimum_size();
	}
}

bool Button::get_clip_text() const {
	return text_label->is_clipping_text();
}

void Button::set_text_alignment(HorizontalAlignment p_alignment) {
	if (text_label->get_horizontal_alignment() != p_alignment) {
		text_label->set_horizontal_alignment(p_alignment);
		queue_accessibility_update();
		queue_redraw();
	}
}

HorizontalAlignment Button::get_text_alignment() const {
	return text_label->get_horizontal_alignment();
}

void Button::set_icon_alignment(HorizontalAlignment p_alignment) {
	if (horizontal_icon_alignment == p_alignment) {
		return;
	}

	horizontal_icon_alignment = p_alignment;
	update_desired_size();
	update_minimum_size();
	queue_redraw();
}

void Button::set_vertical_icon_alignment(VerticalAlignment p_alignment) {
	if (vertical_icon_alignment == p_alignment) {
		return;
	}
	bool need_update_cache = vertical_icon_alignment == VERTICAL_ALIGNMENT_CENTER || p_alignment == VERTICAL_ALIGNMENT_CENTER;
	vertical_icon_alignment = p_alignment;

	if (need_update_cache) {
		_queue_update_size_cache();
	}
	update_desired_size();
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
	ClassDB::bind_method(D_METHOD("set_autowrap_mode", "autowrap_mode"), &Button::set_autowrap_mode);
	ClassDB::bind_method(D_METHOD("get_autowrap_mode"), &Button::get_autowrap_mode);
	ClassDB::bind_method(D_METHOD("set_autowrap_trim_flags", "autowrap_trim_flags"), &Button::set_autowrap_trim_flags);
	ClassDB::bind_method(D_METHOD("get_autowrap_trim_flags"), &Button::get_autowrap_trim_flags);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &Button::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &Button::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &Button::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &Button::get_language);
	ClassDB::bind_method(D_METHOD("set_button_icon", "texture"), &Button::set_button_icon);
	ClassDB::bind_method(D_METHOD("get_button_icon"), &Button::get_button_icon);
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
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, Texture2D::get_class_static()), "set_button_icon", "get_button_icon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");

	ADD_GROUP("Text Behavior", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Left,Center,Right"), "set_text_alignment", "get_text_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_overrun_behavior", PROPERTY_HINT_ENUM, "Trim Nothing,Trim Characters,Trim Words,Ellipsis (6+ Characters),Word Ellipsis (6+ Characters),Ellipsis (Always),Word Ellipsis (Always)"), "set_text_overrun_behavior", "get_text_overrun_behavior");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "autowrap_mode", PROPERTY_HINT_ENUM, "Off,Arbitrary,Word,Word (Smart)"), "set_autowrap_mode", "get_autowrap_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "autowrap_trim_flags", PROPERTY_HINT_FLAGS, vformat("Trim Spaces After Break:%d,Trim Spaces Before Break:%d", TextServer::BREAK_TRIM_START_EDGE_SPACES, TextServer::BREAK_TRIM_END_EDGE_SPACES)), "set_autowrap_trim_flags", "get_autowrap_trim_flags");
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

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, Button, align_to_largest_stylebox);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, Button, line_spacing);
}

Button::Button(const String &p_text) {
	connect(SceneStringName(maximum_size_changed), callable_mp(this, &Button::_maximum_size_changed));

	set_mouse_filter(MOUSE_FILTER_STOP);

	text_label = memnew(Label);
	text_label->set_mouse_filter(MOUSE_FILTER_IGNORE);
	text_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	text_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	text_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	text_label->add_theme_style_override(CoreStringName(normal), memnew(StyleBoxEmpty));
	add_child(text_label, false, INTERNAL_MODE_FRONT);
	set_autowrap_trim_flags(TextServer::BREAK_MANDATORY | TextServer::BREAK_TRIM_END_EDGE_SPACES);
	set_text(p_text);
}

Button::~Button() {
}
