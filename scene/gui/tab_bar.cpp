/**************************************************************************/
/*  tab_bar.cpp                                                           */
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

#include "tab_bar.h"

#include "core/input/input.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/timer.h"
#include "scene/main/viewport.h"
#include "scene/theme/theme_db.h"
#include "servers/display/accessibility_server.h"

#include <cfloat> // FLT_MAX

static inline Color _select_color(const Color &p_override_color, const Color &p_default_color) {
	return p_override_color.a > 0 ? p_override_color : p_default_color;
}

Size2 TabBar::get_minimum_size() const {
	Size2 ms;

	if (tabs.is_empty()) {
		return ms;
	}

	if (!theme_cache.tab_unselected_style.is_valid() || !theme_cache.tab_hovered_style.is_valid() || !theme_cache.tab_selected_style.is_valid() || !theme_cache.tab_disabled_style.is_valid() || !theme_cache.button_hl_style.is_valid()) {
		return ms;
	}

	int primary_sum = 0;
	int cross_max = 0;
	int primary_max = 0;
	int visible_tabs_count = 0;

	for (int i = 0; i < tabs.size(); i++) {
		if (tabs[i].hidden) {
			continue;
		}
		visible_tabs_count++;

		const TabMetrics metrics = _get_tab_metrics(i, true);
		int tab_w = metrics.row_width;
		if (max_width > 0 && tab_w > max_width) {
			const int size_textless = tab_w - tabs[i].size_text;
			tab_w = MAX(size_textless, max_width);
		}
		const int tab_h = metrics.row_height;
		const int tab_primary = vertical ? tab_h : tab_w;
		const int tab_cross = vertical ? tab_w : tab_h;

		primary_sum += tab_primary;
		primary_max = MAX(primary_max, tab_primary);
		cross_max = MAX(cross_max, tab_cross);

		if (i < tabs.size() - 1) {
			primary_sum += theme_cache.tab_separation;
		}
	}

	if (tab_sizing == TAB_SIZING_UNIFORM && visible_tabs_count > 0) {
		int total_separation = (visible_tabs_count - 1) * theme_cache.tab_separation;
		if (vertical) { /* VERTICAL */
			ms.height = (primary_max * visible_tabs_count) + total_separation;
		} else { /* HORIZONTAL */
			ms.width = (primary_max * visible_tabs_count) + total_separation;
		}
	}

	if (clip_tabs) {
		const Size2 desired_ms = get_desired_size();
		const Size2 custom_max = get_custom_maximum_size();
		const int desired_primary = vertical ? desired_ms.height : desired_ms.width;
		const int custom_max_primary = vertical ? custom_max.height : custom_max.width;
		if (custom_max_primary >= 0 && custom_max_primary >= desired_primary) {
			return desired_ms;
		}

		const int buttons_primary = (get_tab_count() > 1 && theme_cache.decrement_icon.is_valid() && theme_cache.increment_icon.is_valid()) ? (vertical ? MAX(theme_cache.decrement_icon->get_height(), theme_cache.increment_icon->get_height()) : (theme_cache.decrement_icon->get_width() + theme_cache.increment_icon->get_width())) : 0;
		const int popup_width = vertical ? _get_vertical_popup_button_min_size(this).width : 0;
		int buttons_cross = 0;
		if (vertical && get_tab_count() > 1 && theme_cache.decrement_vertical_icon.is_valid() && theme_cache.increment_vertical_icon.is_valid()) { /* VERTICAL */
			buttons_cross = theme_cache.decrement_vertical_icon->get_width() + theme_cache.increment_vertical_icon->get_width();
		}

		Size2 clipped_ms;
		if (vertical) { /* VERTICAL */
			clipped_ms.width = MAX(cross_max, popup_width + buttons_cross);
			clipped_ms.height = primary_max + buttons_primary;
		} else { /* HORIZONTAL */
			clipped_ms.width = primary_max + buttons_primary;
			clipped_ms.height = cross_max;
		}

		return clipped_ms;
	}

	if (vertical) { /* VERTICAL */
		ms.width = MAX(cross_max, (int)_get_vertical_popup_button_min_size(this).width);
		ms.height = primary_sum + (buttons_visible ? _get_reserved_vertical_buttons_row_height() : 0);
	} else { /* HORIZONTAL */
		ms.width = primary_sum;
		ms.height = cross_max;
	}

	return ms;
}

void TabBar::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		Point2 pos = mm->get_position();

		if (buttons_visible) {
			Rect2 dec_rect;
			Rect2 inc_rect;
			_get_scroll_button_rects(dec_rect, inc_rect);
			int hover_arrow = -1;
			if (dec_rect.has_point(pos)) {
				hover_arrow = vertical || !is_layout_rtl() ? 0 : 1;
			} else if (inc_rect.has_point(pos)) {
				hover_arrow = vertical || !is_layout_rtl() ? 1 : 0;
			}
			if (highlight_arrow != hover_arrow) {
				highlight_arrow = hover_arrow;
				queue_redraw();
			}
		}

		if (get_viewport()->gui_is_dragging()) {
			Variant drag_data = get_viewport()->gui_get_drag_data();
			if (can_drop_data(pos, drag_data) || _handle_can_drop_data("tab_container_tab", pos, drag_data)) {
				dragging_valid_tab = true;
				queue_redraw();
			}
		}

		if (!tabs.is_empty()) {
			_update_hover();
		}

		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->is_pressed() && !mb->is_command_or_control_pressed()) {
			if (vertical) { /* VERTICAL */
				if (mb->get_button_index() == MouseButton::WHEEL_UP) {
					if (scrolling_enabled && buttons_visible && offset > 0) {
						offset--;
						_update_cache();
						queue_redraw();
					}
				} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN) {
					if (scrolling_enabled && buttons_visible && missing_right && offset < tabs.size()) {
						offset++;
						_update_cache();
						queue_redraw();
					}
				}
			} else { /* HORIZONTAL */
				const bool rtl = is_layout_rtl();
				if (mb->get_button_index() == MouseButton::WHEEL_UP || mb->get_button_index() == (rtl ? MouseButton::WHEEL_RIGHT : MouseButton::WHEEL_LEFT)) {
					if (scrolling_enabled && buttons_visible && offset > 0) {
						offset--;
						_update_cache();
						queue_redraw();
					}
				} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN || mb->get_button_index() == (rtl ? MouseButton::WHEEL_LEFT : MouseButton::WHEEL_RIGHT)) {
					if (scrolling_enabled && buttons_visible && missing_right && offset < tabs.size()) {
						offset++;
						_update_cache();
						queue_redraw();
					}
				}
			}
		}

		if (hover != -1 && mb->get_button_index() == MouseButton::LEFT) {
			accept_event();
		}

		if (rb_pressing && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			if (rb_hover != -1) {
				emit_signal(SNAME("tab_button_pressed"), rb_hover);
			}

			rb_pressing = false;
			queue_redraw();
		}

		if (cb_pressing && !mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
			if (cb_hover != -1) {
				emit_signal(SNAME("tab_close_pressed"), cb_hover);
			}

			cb_pressing = false;
			queue_redraw();
		}

		if (close_with_middle_mouse && mb->is_pressed() && mb->get_button_index() == MouseButton::MIDDLE) {
			if (hover != -1) {
				emit_signal(SNAME("tab_close_pressed"), hover);
			}
		}

		if (mb->is_pressed() != switch_on_release) {
			Point2 pos = mb->get_position();
			bool selecting = mb->get_button_index() == MouseButton::LEFT || (select_with_rmb && mb->get_button_index() == MouseButton::RIGHT);

			if (buttons_visible && selecting) {
				Rect2 dec_rect;
				Rect2 inc_rect;
				_get_scroll_button_rects(dec_rect, inc_rect);
				bool dec_hit = dec_rect.has_point(pos);
				bool inc_hit = inc_rect.has_point(pos);
				const bool rtl = is_layout_rtl();
				if (dec_hit) {
					if (vertical || !rtl) {
						if (offset > 0) {
							offset--;
							_update_cache();
							queue_redraw();
						}
					} else if (missing_right) {
						offset++;
						_update_cache();
						queue_redraw();
					}
					return;
				}
				if (inc_hit) {
					if (vertical || !rtl) {
						if (missing_right) {
							offset++;
							_update_cache();
							queue_redraw();
						}
					} else if (offset > 0) {
						offset--;
						_update_cache();
						queue_redraw();
					}
					return;
				}
			}

			if (tabs.is_empty()) {
				// Return early if there are no actual tabs to handle input for.
				return;
			}

			int found = get_tab_idx_at_point(pos);
			if (found != -1) {
				// Clicking right button icon.
				if (tabs[found].rb_rect.has_point(pos)) {
					if (selecting) {
						rb_pressing = true;
						_update_hover();
						queue_redraw();
					}
					return;
				}

				// Clicking close button.
				if (tabs[found].cb_rect.has_point(pos) && (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && found == current))) {
					if (selecting) {
						cb_pressing = true;
						_update_hover();
						queue_redraw();
					}
					return;
				}

				// Selecting a tab.
				if (selecting && !tabs[found].disabled) {
					if (deselect_enabled && get_current_tab() == found) {
						set_current_tab(-1);
					} else {
						set_current_tab(found);
					}

					emit_signal(SNAME("tab_clicked"), found);
				}

				// Right mouse button clicked on a tab.
				if (mb->get_button_index() == MouseButton::RIGHT) {
					emit_signal(SNAME("tab_rmb_clicked"), found);
				}
			}
		}
	}

	if (p_event->is_pressed()) {
		Input *input = Input::get_singleton();
		Ref<InputEventJoypadMotion> joypadmotion_event = p_event;
		Ref<InputEventJoypadButton> joypadbutton_event = p_event;
		bool is_joypad_event = (joypadmotion_event.is_valid() || joypadbutton_event.is_valid());
		if (p_event->is_action("ui_right", true)) {
			grab_focus(); // Ensure focus is visible.
			if (is_joypad_event) {
				if (!input->is_action_just_pressed_by_event("ui_right", p_event, true)) {
					return;
				}
				set_process_internal(true);
			}
			if (is_layout_rtl() ? select_previous_available() : select_next_available()) {
				accept_event();
			}
		} else if (p_event->is_action("ui_left", true)) {
			grab_focus();
			if (is_joypad_event) {
				if (!input->is_action_just_pressed_by_event("ui_left", p_event, true)) {
					return;
				}
				set_process_internal(true);
			}
			if (is_layout_rtl() ? select_next_available() : select_previous_available()) {
				accept_event();
			}
		}
	}
}

String TabBar::get_tooltip(const Point2 &p_pos) const {
	int tab_idx = get_tab_idx_at_point(p_pos);
	if (tab_idx < 0) {
		return Control::get_tooltip(p_pos);
	}

	if (tabs[tab_idx].tooltip.is_empty() && tabs[tab_idx].truncated) {
		return tabs[tab_idx].text;
	}

	return tabs[tab_idx].tooltip;
}

void TabBar::_shape(int p_tab) {
	tabs.write[p_tab].text_buf->clear();
	tabs.write[p_tab].text_buf->set_width(-1);
	if (tabs[p_tab].text_direction == Control::TEXT_DIRECTION_INHERITED) {
		tabs.write[p_tab].text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		tabs.write[p_tab].text_buf->set_direction((TextServer::Direction)tabs[p_tab].text_direction);
	}

	const String &lang = tabs[p_tab].language.is_empty() ? _get_locale() : tabs[p_tab].language;
	tabs.write[p_tab].text_buf->add_string(atr(tabs[p_tab].text), theme_cache.font, theme_cache.font_size, lang);
}

RID TabBar::get_tab_accessibility_element(int p_tab) const {
	RID ae = get_accessibility_element();
	ERR_FAIL_COND_V(ae.is_null(), RID());

	const Tab &item = tabs[p_tab];
	if (item.accessibility_item_element.is_null()) {
		item.accessibility_item_element = AccessibilityServer::get_singleton()->create_sub_element(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_TAB);
		item.accessibility_item_dirty = true;
	}
	return item.accessibility_item_element;
}

RID TabBar::get_focused_accessibility_element() const {
	if (current == -1) {
		return get_accessibility_element();
	} else {
		const Tab &item = tabs[current];
		return item.accessibility_item_element;
	}
}

void TabBar::_accessibility_action_scroll_into_view(const Variant &p_data, int p_index) {
	ensure_tab_visible(p_index);
}

void TabBar::_accessibility_action_focus(const Variant &p_data, int p_index) {
	set_current_tab(p_index);
}

void TabBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (scroll_to_selected) {
				ensure_tab_visible(current);
			}
			// Set initialized even if no tabs were set.
			initialized = true;
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			Input *input = Input::get_singleton();

			if (input->is_action_just_released("ui_left") || input->is_action_just_released("ui_right")) {
				gamepad_event_delay_ms = DEFAULT_GAMEPAD_EVENT_DELAY_MS;
				set_process_internal(false);
				return;
			}

			gamepad_event_delay_ms -= get_process_delta_time();
			if (gamepad_event_delay_ms <= 0) {
				gamepad_event_delay_ms = GAMEPAD_EVENT_REPEAT_RATE_MS + gamepad_event_delay_ms;
				if (input->is_action_pressed("ui_right")) {
					is_layout_rtl() ? select_previous_available() : select_next_available();
				}

				if (input->is_action_pressed("ui_left")) {
					is_layout_rtl() ? select_next_available() : select_previous_available();
				}
			}
		} break;

		case NOTIFICATION_EXIT_TREE:
		case NOTIFICATION_ACCESSIBILITY_INVALIDATE: {
			for (int i = 0; i < tabs.size(); i++) {
				tabs.write[i].accessibility_item_element = RID();
			}
		} break;

		case NOTIFICATION_ACCESSIBILITY_UPDATE: {
			RID ae = get_accessibility_element();
			ERR_FAIL_COND(ae.is_null());

			AccessibilityServer::get_singleton()->update_set_role(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_TAB_BAR);
			AccessibilityServer::get_singleton()->update_set_list_item_count(ae, tabs.size());

			for (int i = 0; i < tabs.size(); i++) {
				const Tab &item = tabs[i];

				if (item.accessibility_item_element.is_null()) {
					item.accessibility_item_element = AccessibilityServer::get_singleton()->create_sub_element(ae, AccessibilityServerEnums::AccessibilityRole::ROLE_TAB);
					item.accessibility_item_dirty = true;
				}

				if (item.accessibility_item_dirty) {
					AccessibilityServer::get_singleton()->update_add_action(item.accessibility_item_element, AccessibilityServerEnums::AccessibilityAction::ACTION_SCROLL_INTO_VIEW, callable_mp(this, &TabBar::_accessibility_action_scroll_into_view).bind(i));
					AccessibilityServer::get_singleton()->update_add_action(item.accessibility_item_element, AccessibilityServerEnums::AccessibilityAction::ACTION_FOCUS, callable_mp(this, &TabBar::_accessibility_action_focus).bind(i));

					AccessibilityServer::get_singleton()->update_set_list_item_index(item.accessibility_item_element, i);
					AccessibilityServer::get_singleton()->update_set_name(item.accessibility_item_element, atr(item.text));
					AccessibilityServer::get_singleton()->update_set_list_item_selected(item.accessibility_item_element, i == current);
					AccessibilityServer::get_singleton()->update_set_flag(item.accessibility_item_element, AccessibilityServerEnums::AccessibilityFlags::FLAG_DISABLED, item.disabled);
					AccessibilityServer::get_singleton()->update_set_flag(item.accessibility_item_element, AccessibilityServerEnums::AccessibilityFlags::FLAG_HIDDEN, item.hidden);
					AccessibilityServer::get_singleton()->update_set_tooltip(item.accessibility_item_element, item.tooltip);

					const Rect2 content_rect = _get_tabs_content_rect();
					if (vertical) { /* VERTICAL */
						AccessibilityServer::get_singleton()->update_set_bounds(item.accessibility_item_element, Rect2(Point2(content_rect.position.x, item.ofs_cache), Size2(content_rect.size.x, item.size_cache)));
					} else { /* HORIZONTAL */
						AccessibilityServer::get_singleton()->update_set_bounds(item.accessibility_item_element, Rect2(Point2(item.ofs_cache, content_rect.position.y), Size2(item.size_cache, content_rect.size.y)));
					}

					item.accessibility_item_dirty = false;
				}
			}
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			queue_redraw();
		} break;

		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			for (int i = 0; i < tabs.size(); ++i) {
				_shape(i);
			}

			queue_accessibility_update();
			queue_redraw();

			int ofs_old = offset;
			int max_old = max_drawn_tab;

			_update_cache();
			_ensure_no_over_offset();

			if (scroll_to_selected && (offset != ofs_old || max_drawn_tab != max_old)) {
				ensure_tab_visible(current);
			}

			update_desired_size();
			update_minimum_size();
		} break;

		case NOTIFICATION_RESIZED: {
			int ofs_old = offset;
			int max_old = max_drawn_tab;

			_update_cache();
			_ensure_no_over_offset();

			if (scroll_to_selected && (offset != ofs_old || max_drawn_tab != max_old)) {
				ensure_tab_visible(current);
			}
			update_minimum_size();
		} break;

		case NOTIFICATION_DRAG_BEGIN: {
			if (drag_to_rearrange_enabled) {
				Variant drag_data = get_viewport()->gui_get_drag_data();
				if (can_drop_data(Point2(), drag_data) || _handle_can_drop_data("tab_container_tab", Point2(), drag_data)) {
					dragging_valid_tab = true;
					queue_redraw();
				}
			}
		} break;

		case NOTIFICATION_DRAG_END: {
			if (dragging_valid_tab) {
				dragging_valid_tab = false;
				queue_redraw();
			}
			[[fallthrough]];
		}

		case NOTIFICATION_MOUSE_EXIT: {
			if (!hover_switch_delay->is_stopped()) {
				hover_switch_delay->stop();
			}
		} break;

		case NOTIFICATION_DRAW: {
			bool rtl = is_layout_rtl();
			Vector2 size = get_size();

			Ref<Texture2D> v_dec_icon;
			Ref<Texture2D> v_inc_icon;
			_get_scroll_button_icons(v_dec_icon, v_inc_icon);
			Ref<Texture2D> v_dec_hl_icon = theme_cache.decrement_vertical_hl_icon.is_valid() ? theme_cache.decrement_vertical_hl_icon : v_dec_icon;
			Ref<Texture2D> v_inc_hl_icon = theme_cache.increment_vertical_hl_icon.is_valid() ? theme_cache.increment_vertical_hl_icon : v_inc_icon;
			Ref<Texture2D> v_drop_mark_icon = theme_cache.vertical_drop_mark_icon.is_valid() ? theme_cache.vertical_drop_mark_icon : theme_cache.drop_mark_icon;
			auto get_tab_draw_pos = [&](int p_tab) -> float {
				return vertical ? tabs[p_tab].ofs_cache : (rtl ? (size.width - tabs[p_tab].ofs_cache - tabs[p_tab].size_cache) : tabs[p_tab].ofs_cache);
			};

			if (tabs.is_empty()) {
				// Draw the drop indicator where the first tab would be if there are no tabs.
				if (dragging_valid_tab) {
					if (vertical) { /* VERTICAL */
						int y = 0;
						v_drop_mark_icon->draw(get_canvas_item(), Point2((size.width - v_drop_mark_icon->get_width()) / 2, y - (v_drop_mark_icon->get_height() / 2)), theme_cache.drop_mark_color);
					} else { /* HORIZONTAL */
						int x = rtl ? size.x : 0;
						theme_cache.drop_mark_icon->draw(get_canvas_item(), Point2(x - (theme_cache.drop_mark_icon->get_width() / 2), (size.height - theme_cache.drop_mark_icon->get_height()) / 2), theme_cache.drop_mark_color);
					}
				}

				return;
			}

			// Draw unselected tabs in the back.
			for (int i = offset; i <= max_drawn_tab; i++) {
				if (tabs[i].hidden) {
					continue;
				}

				if (i != current) {
					Ref<StyleBox> sb;
					Color fnt_col;
					Color icn_col;

					if (tabs[i].disabled) {
						sb = theme_cache.tab_disabled_style;
						fnt_col = _select_color(tabs[i].font_color_overrides[DrawMode::DRAW_DISABLED], theme_cache.font_disabled_color);
						icn_col = theme_cache.icon_disabled_color;
					} else if (i == hover) {
						sb = theme_cache.tab_hovered_style;
						fnt_col = _select_color(tabs[i].font_color_overrides[DrawMode::DRAW_HOVER], theme_cache.font_hovered_color);
						icn_col = theme_cache.icon_hovered_color;
					} else {
						sb = theme_cache.tab_unselected_style;
						fnt_col = _select_color(tabs[i].font_color_overrides[DrawMode::DRAW_NORMAL], theme_cache.font_unselected_color);
						icn_col = theme_cache.icon_unselected_color;
					}

					_draw_tab(sb, fnt_col, icn_col, i, get_tab_draw_pos(i), false);
				}
			}

			// Draw selected tab in the front, but only if it's visible.
			if (current >= offset && current <= max_drawn_tab && !tabs[current].hidden) {
				Ref<StyleBox> sb = tabs[current].disabled ? theme_cache.tab_disabled_style : theme_cache.tab_selected_style;
				Color col = _select_color(tabs[current].font_color_overrides[DrawMode::DRAW_PRESSED], theme_cache.font_selected_color);

				_draw_tab(sb, col, theme_cache.icon_selected_color, current, get_tab_draw_pos(current), has_focus(true));
			}

			if (buttons_visible) {
				Rect2 dec_rect;
				Rect2 inc_rect;
				_get_scroll_button_rects(dec_rect, inc_rect);
				if (vertical) { /* VERTICAL */
					Texture2D *dec_icon = (highlight_arrow == 0 ? theme_cache.decrement_vertical_hl_icon : v_dec_icon).ptr();
					Texture2D *inc_icon = (highlight_arrow == 1 ? theme_cache.increment_vertical_hl_icon : v_inc_icon).ptr();

					float dec_opacity = offset > 0 ? 1.0f : 0.5f;
					float inc_opacity = missing_right ? 1.0f : 0.5f;

					draw_texture(dec_icon, dec_rect.position, Color(1, 1, 1, dec_opacity));
					draw_texture(inc_icon, inc_rect.position, Color(1, 1, 1, inc_opacity));
				} else { /* HORIZONTAL */
					const bool dec_enabled = rtl ? missing_right : offset > 0;
					const bool inc_enabled = rtl ? offset > 0 : missing_right;
					const bool dec_highlighted = highlight_arrow == (rtl ? 1 : 0);
					const bool inc_highlighted = highlight_arrow == (rtl ? 0 : 1);

					Ref<Texture2D> dec_draw = dec_highlighted ? theme_cache.decrement_hl_icon : theme_cache.decrement_icon;
					Ref<Texture2D> inc_draw = inc_highlighted ? theme_cache.increment_hl_icon : theme_cache.increment_icon;
					Point2 dec_pos(dec_rect.position.x, dec_rect.position.y + (dec_rect.size.y - dec_draw->get_height()) * 0.5f);
					Point2 inc_pos(inc_rect.position.x, inc_rect.position.y + (inc_rect.size.y - inc_draw->get_height()) * 0.5f);

					draw_texture(dec_draw, dec_pos, Color(1, 1, 1, dec_enabled ? 1.0f : 0.5f));
					draw_texture(inc_draw, inc_pos, Color(1, 1, 1, inc_enabled ? 1.0f : 0.5f));
				}
			}

			if (dragging_valid_tab) {
				_draw_tab_drop(get_canvas_item());
			}
		} break;
	}
}

void TabBar::_draw_tab_drop(RID p_canvas_item) {
	Vector2 size = get_size();
	bool rtl = is_layout_rtl();

	int closest_tab = get_closest_tab_idx_to_point(get_local_mouse_position());
	if (closest_tab != -1) {
		Rect2 tab_rect = get_tab_rect(closest_tab);
		const Point2 mouse_pos = get_local_mouse_position();

		if (vertical) { /* VERTICAL */
			int y = tab_rect.position.y;

			// Only add the tab_separation if closest tab is not on the edge.
			bool not_topmost_tab = -1 != get_previous_available(closest_tab);
			bool not_bottommost_tab = -1 != get_next_available(closest_tab);

			// Calculate midpoint between tabs.
			if (_is_point_primary_after_mid(mouse_pos, tab_rect)) {
				y += tab_rect.size.y;
				if (not_bottommost_tab) {
					y += Math::ceil(0.5f * theme_cache.tab_separation);
				}
			} else if (not_topmost_tab) {
				y -= Math::floor(0.5f * theme_cache.tab_separation);
			}

			theme_cache.vertical_drop_mark_icon->draw(p_canvas_item, Point2((size.width - theme_cache.vertical_drop_mark_icon->get_width()) / 2, y - theme_cache.vertical_drop_mark_icon->get_height() / 2), theme_cache.drop_mark_color);
		} else { /* HORIZONTAL */
			int x = tab_rect.position.x;

			// Only add the tab_separation if closest tab is not on the edge.
			bool not_leftmost_tab = -1 != (rtl ? get_next_available(closest_tab) : get_previous_available(closest_tab));
			bool not_rightmost_tab = -1 != (rtl ? get_previous_available(closest_tab) : get_next_available(closest_tab));

			// Calculate midpoint between tabs.
			if (_is_point_primary_after_mid(mouse_pos, tab_rect)) {
				x += tab_rect.size.x;
				if (not_rightmost_tab) {
					x += Math::ceil(0.5f * theme_cache.tab_separation);
				}
			} else if (not_leftmost_tab) {
				x -= Math::floor(0.5f * theme_cache.tab_separation);
			}

			theme_cache.drop_mark_icon->draw(p_canvas_item, Point2(x - theme_cache.drop_mark_icon->get_width() / 2, (size.height - theme_cache.drop_mark_icon->get_height()) / 2), theme_cache.drop_mark_color);
		}
	} else {
		const Point2 mouse_pos = get_local_mouse_position();
		if (vertical) { /* VERTICAL */
			int y = get_tab_rect(0).position.y;
			if (_is_point_before_first_tab(mouse_pos)) {
				// Above first tab
				theme_cache.vertical_drop_mark_icon->draw(p_canvas_item, Point2((size.width - theme_cache.vertical_drop_mark_icon->get_width()) / 2, y - theme_cache.vertical_drop_mark_icon->get_height() / 2), theme_cache.drop_mark_color);
			} else {
				// Below last tab
				Rect2 tab_rect = get_tab_rect(get_tab_count() - 1);
				y = tab_rect.position.y + tab_rect.size.y;
				theme_cache.vertical_drop_mark_icon->draw(p_canvas_item, Point2((size.width - theme_cache.vertical_drop_mark_icon->get_width()) / 2, y - theme_cache.vertical_drop_mark_icon->get_height() / 2), theme_cache.drop_mark_color);
			}
		} else { /* HORIZONTAL */
			int x;
			if (_is_point_before_first_tab(mouse_pos)) {
				x = get_tab_rect(0).position.x;
				if (rtl) {
					x += get_tab_rect(0).size.width;
				}
			} else {
				Rect2 tab_rect = get_tab_rect(get_tab_count() - 1);
				x = tab_rect.position.x;
				if (!rtl) {
					x += tab_rect.size.width;
				}
			}

			theme_cache.drop_mark_icon->draw(p_canvas_item, Point2(x - theme_cache.drop_mark_icon->get_width() / 2, (size.height - theme_cache.drop_mark_icon->get_height()) / 2), theme_cache.drop_mark_color);
		}
	}
}

void TabBar::_draw_tab(Ref<StyleBox> &p_tab_style, const Color &p_font_color, const Color &p_icon_color, int p_index, float p_x, bool p_focus) {
	RID ci = get_canvas_item();
	bool rtl = is_layout_rtl();

	Rect2 sb_rect = get_tab_rect(p_index);
	if (!vertical) {
		sb_rect = Rect2(p_x, 0, tabs[p_index].size_cache, get_size().height);
	}
	// Orient the tab style so its highlight edge faces the configured side.
	// TOP/BOTTOM mirror vertically (legacy v-flip behavior for bottom tabs),
	// LEFT/RIGHT rotate the style 90 degrees so a top-edge highlight faces sideways.
	const Rect2 style_rect = p_tab_style->get_draw_rect(sb_rect);
	switch (tab_style_side) {
		case TAB_STYLE_SIDE_BOTTOM: {
			// Mirror around the style rect's own vertical center, so this
			// works for both full-height horizontal tabs and stacked
			// vertical tabs (flipping around y=0 would displace the latter).
			draw_set_transform(Point2(0.0, style_rect.position.y + style_rect.get_end().y), 0.0, Size2(1.0, -1.0));
			p_tab_style->draw(ci, sb_rect);
			draw_set_transform(Point2(), 0.0, Size2(1.0, 1.0));
		} break;
		case TAB_STYLE_SIDE_LEFT: {
			draw_set_transform(Point2(style_rect.position.x, style_rect.position.y + style_rect.size.y), -Math::PI / 2.0, Size2(1.0, 1.0));
			p_tab_style->draw(ci, Rect2(Point2(), Size2(style_rect.size.y, style_rect.size.x)));
			draw_set_transform(Point2(), 0.0, Size2(1.0, 1.0));
		} break;
		case TAB_STYLE_SIDE_RIGHT: {
			draw_set_transform(Point2(style_rect.position.x + style_rect.size.x, style_rect.position.y), Math::PI / 2.0, Size2(1.0, 1.0));
			p_tab_style->draw(ci, Rect2(Point2(), Size2(style_rect.size.y, style_rect.size.x)));
			draw_set_transform(Point2(), 0.0, Size2(1.0, 1.0));
		} break;
		default: {
			p_tab_style->draw(ci, sb_rect);
		} break;
	}

	if (p_focus) {
		Ref<StyleBox> focus_style = theme_cache.tab_focus_style;
		focus_style->draw(ci, sb_rect);
	}

	Size2i sb_ms = p_tab_style->get_minimum_size();
	if (tab_text_rotation != TAB_TEXT_ROTATION_NONE) {
		// Stacked layout: icon, rotated text, right button and close button
		// are arranged vertically, centered horizontally.
		const int center_x = sb_rect.position.x + p_tab_style->get_margin(SIDE_LEFT) + (sb_rect.size.x - sb_ms.x) / 2;

		Ref<Texture2D> v_icon = tabs[p_index].icon;
		Size2 v_icon_size;
		if (v_icon.is_valid()) {
			v_icon_size = _get_tab_icon_size(p_index);
		}
		int v_text_len = 0;
		int v_text_h = 0;
		if (!tabs[p_index].text.is_empty()) {
			v_text_len = Math::ceil(tabs[p_index].text_buf->get_size().x);
			v_text_h = Math::ceil(tabs[p_index].text_buf->get_size().y);
		}
		Size2 v_rb_size;
		bool v_has_rb = tabs[p_index].right_button.is_valid();
		if (v_has_rb) {
			Ref<StyleBox> v_rb_style = theme_cache.button_hl_style;
			v_rb_size = v_rb_style->get_minimum_size() + tabs[p_index].right_button->get_size();
		}
		Size2 v_cb_size;
		bool v_has_cb = cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && p_index == current);
		if (v_has_cb) {
			Ref<StyleBox> v_cb_style = theme_cache.button_hl_style;
			v_cb_size = v_cb_style->get_minimum_size() + theme_cache.close_icon->get_size();
		}

		int v_total_h = 0;
		int v_items = 0;
		if (v_icon.is_valid()) {
			v_total_h += v_icon_size.height;
			v_items++;
		}
		if (v_text_len > 0) {
			v_total_h += v_text_len;
			v_items++;
		}
		if (v_has_rb) {
			v_total_h += v_rb_size.height;
			v_items++;
		}
		if (v_has_cb) {
			v_total_h += v_cb_size.height;
			v_items++;
		}
		if (v_items > 1) {
			v_total_h += (v_items - 1) * theme_cache.h_separation;
		}

		int v_draw_y = sb_rect.position.y + p_tab_style->get_margin(SIDE_TOP) + (sb_rect.size.y - sb_ms.y - v_total_h) / 2;

		if (v_icon.is_valid()) {
			Point2 v_icon_pos = Point2(center_x - v_icon_size.width / 2, v_draw_y);
			v_icon->draw_rect(ci, Rect2(v_icon_pos, v_icon_size), false, p_icon_color);
			v_draw_y += v_icon_size.height + theme_cache.h_separation;
		}

		if (v_text_len > 0) {
			// Rotate 90 degrees so the text reads top-to-bottom (clockwise)
			// or bottom-to-top (counter-clockwise).
			Point2 v_origin;
			float v_angle;
			if (tab_text_rotation == TAB_TEXT_ROTATION_COUNTER_CLOCKWISE) {
				v_origin = Point2(center_x - v_text_h / 2.0, v_draw_y + v_text_len);
				v_angle = -Math::PI / 2.0;
			} else {
				v_origin = Point2(center_x + v_text_h / 2.0, v_draw_y);
				v_angle = Math::PI / 2.0;
			}
			draw_set_transform(v_origin, v_angle, Size2(1.0, 1.0));
			if (theme_cache.outline_size > 0 && theme_cache.font_outline_color.a > 0) {
				tabs[p_index].text_buf->draw_outline(ci, Point2(), theme_cache.outline_size, theme_cache.font_outline_color);
			}
			tabs[p_index].text_buf->draw(ci, Point2(), p_font_color);
			draw_set_transform(Point2(), 0.0, Size2(1.0, 1.0));
			v_draw_y += v_text_len + theme_cache.h_separation;
		}

		if (v_has_rb) {
			Ref<StyleBox> v_rb_style = theme_cache.button_hl_style;
			Ref<Texture2D> v_rb = tabs[p_index].right_button;
			Rect2 v_rb_rect;
			v_rb_rect.size = v_rb_size;
			v_rb_rect.position = Point2(center_x - v_rb_size.width / 2, v_draw_y);
			tabs.write[p_index].rb_rect = v_rb_rect;
			if (rb_hover == p_index) {
				if (rb_pressing) {
					theme_cache.button_pressed_style->draw(ci, v_rb_rect);
				} else {
					v_rb_style->draw(ci, v_rb_rect);
				}
			}
			v_rb->draw(ci, Point2i(v_rb_rect.position.x + v_rb_style->get_margin(SIDE_LEFT), v_rb_rect.position.y + v_rb_style->get_margin(SIDE_TOP)));
			v_draw_y += v_rb_size.height + theme_cache.h_separation;
		} else {
			tabs.write[p_index].rb_rect = Rect2();
		}

		if (v_has_cb) {
			Ref<StyleBox> v_cb_style = theme_cache.button_hl_style;
			Ref<Texture2D> v_cb = theme_cache.close_icon;
			Rect2 v_cb_rect;
			v_cb_rect.size = v_cb_size;
			v_cb_rect.position = Point2(center_x - v_cb_size.width / 2, v_draw_y);
			tabs.write[p_index].cb_rect = v_cb_rect;
			if (!tabs[p_index].disabled && cb_hover == p_index) {
				if (cb_pressing) {
					theme_cache.button_pressed_style->draw(ci, v_cb_rect);
				} else {
					v_cb_style->draw(ci, v_cb_rect);
				}
			}
			v_cb->draw(ci, Point2i(v_cb_rect.position.x + v_cb_style->get_margin(SIDE_LEFT), v_cb_rect.position.y + v_cb_style->get_margin(SIDE_TOP)));
		} else {
			tabs.write[p_index].cb_rect = Rect2();
		}
		return;
	}

	const int content_h = sb_rect.size.y - sb_ms.y;
	const int center_y = sb_rect.position.y + p_tab_style->get_margin(SIDE_TOP) + content_h / 2;

	int draw_x;
	const int inner_content_x = sb_rect.position.x + p_tab_style->get_margin(SIDE_LEFT);
	const int inner_content_w = sb_rect.size.x - sb_ms.x;

	int total_content_width = 0;
	int icon_w = 0;
	int text_w = 0;
	int rb_w = 0;
	int cb_w = 0;

	Ref<Texture2D> icon = tabs[p_index].icon;
	if (icon.is_valid()) {
		const Size2 icon_size = _get_tab_icon_size(p_index);
		icon_w = icon_size.width + theme_cache.h_separation;
		total_content_width += icon_w;
	}

	if (!tabs[p_index].text.is_empty()) {
		const int drawn_text_width = Math::ceil(tabs[p_index].text_buf->get_size().x);
		text_w = drawn_text_width + theme_cache.h_separation;
		total_content_width += text_w;
	}

	if (tabs[p_index].right_button.is_valid()) {
		rb_w = theme_cache.button_hl_style->get_minimum_size().width + tabs[p_index].right_button->get_width() + theme_cache.h_separation;
		total_content_width += rb_w;
	}

	if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && p_index == current)) {
		cb_w = theme_cache.button_hl_style->get_minimum_size().width + theme_cache.close_icon->get_width() + theme_cache.h_separation;
		total_content_width += cb_w;
	}

	if (total_content_width > 0) {
		total_content_width -= theme_cache.h_separation;
	}

	int start_x = inner_content_x + (inner_content_w - total_content_width) / 2;
	if (rtl) {
		start_x = inner_content_x + inner_content_w - (start_x - inner_content_x) - total_content_width;
	}
	draw_x = rtl ? (start_x + total_content_width) : start_x;

	if (icon.is_valid()) {
		const Size2 icon_size = _get_tab_icon_size(p_index);
		const int icon_y = center_y - icon_size.height / 2;
		const Point2 icon_pos = Point2i(rtl ? draw_x - icon_size.width : draw_x, icon_y);
		icon->draw_rect(ci, Rect2(icon_pos, icon_size), false, p_icon_color);
		draw_x = rtl ? (draw_x - icon_size.width - theme_cache.h_separation) : (draw_x + icon_size.width + theme_cache.h_separation);
	}

	if (!tabs[p_index].text.is_empty()) {
		const int drawn_text_width = Math::ceil(tabs[p_index].text_buf->get_size().x);
		const int text_y = center_y - tabs[p_index].text_buf->get_size().y / 2;
		Point2i text_pos = Point2i(rtl ? draw_x - drawn_text_width : draw_x, text_y);

		if (theme_cache.outline_size > 0 && theme_cache.font_outline_color.a > 0) {
			tabs[p_index].text_buf->draw_outline(ci, text_pos, theme_cache.outline_size, theme_cache.font_outline_color);
		}
		tabs[p_index].text_buf->draw(ci, text_pos, p_font_color);

		draw_x = rtl ? (draw_x - drawn_text_width - theme_cache.h_separation) : (draw_x + drawn_text_width + theme_cache.h_separation);
	}

	if (tabs[p_index].right_button.is_valid()) {
		Ref<StyleBox> style = theme_cache.button_hl_style;
		Ref<Texture2D> rb = tabs[p_index].right_button;

		Rect2 rb_rect;
		rb_rect.size = style->get_minimum_size() + rb->get_size();
		rb_rect.position.x = rtl ? draw_x - rb_rect.size.width : draw_x;
		rb_rect.position.y = center_y - rb_rect.size.height / 2;

		tabs.write[p_index].rb_rect = rb_rect;

		if (rb_hover == p_index) {
			if (rb_pressing) {
				theme_cache.button_pressed_style->draw(ci, rb_rect);
			} else {
				style->draw(ci, rb_rect);
			}
		}

		rb->draw(ci, Point2i(rb_rect.position.x + style->get_margin(SIDE_LEFT), rb_rect.position.y + style->get_margin(SIDE_TOP)));
		draw_x = rtl ? rb_rect.position.x : (rb_rect.position.x + rb_rect.size.width);
	} else {
		tabs.write[p_index].rb_rect = Rect2();
	}

	if (cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && p_index == current)) {
		Ref<StyleBox> style = theme_cache.button_hl_style;
		Ref<Texture2D> cb = theme_cache.close_icon;

		Rect2 cb_rect;
		cb_rect.size = style->get_minimum_size() + cb->get_size();
		cb_rect.position.x = rtl ? draw_x - cb_rect.size.width : draw_x;
		cb_rect.position.y = center_y - cb_rect.size.height / 2;

		tabs.write[p_index].cb_rect = cb_rect;

		if (!tabs[p_index].disabled && cb_hover == p_index) {
			if (cb_pressing) {
				theme_cache.button_pressed_style->draw(ci, cb_rect);
			} else {
				style->draw(ci, cb_rect);
			}
		}

		cb->draw(ci, Point2i(cb_rect.position.x + style->get_margin(SIDE_LEFT), cb_rect.position.y + style->get_margin(SIDE_TOP)));
	} else {
		tabs.write[p_index].cb_rect = Rect2();
	}
}

void TabBar::set_tab_count(int p_count) {
	if (p_count == tabs.size()) {
		return;
	}

	ERR_FAIL_COND(p_count < 0);

	if (tabs.size() > p_count) {
		for (int i = p_count; i < tabs.size(); i++) {
			if (tabs[i].accessibility_item_element.is_valid()) {
				AccessibilityServer::get_singleton()->free_element(tabs.write[i].accessibility_item_element);
				tabs.write[i].accessibility_item_element = RID();
			}
		}
	}
	tabs.resize(p_count);

	if (p_count == 0) {
		offset = 0;
		max_drawn_tab = 0;
		current = -1;
		previous = -1;
	} else {
		offset = MIN(offset, p_count - 1);
		max_drawn_tab = MIN(max_drawn_tab, p_count - 1);
		current = MIN(current, p_count - 1);
		// Fix range if unable to deselect.
		if (current == -1 && !_can_deselect()) {
			current = 0;
		}

		_update_cache();
		_ensure_no_over_offset();
		if (scroll_to_selected) {
			ensure_tab_visible(current);
		}
	}

	if (!initialized) {
		initialized = true;
		if (queued_current != CURRENT_TAB_UNINITIALIZED && queued_current != current) {
			set_current_tab(queued_current);
		}
	}

	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();
	notify_property_list_changed();
}

int TabBar::get_tab_count() const {
	return tabs.size();
}

void TabBar::set_current_tab(int p_current) {
	if (p_current == -1) {
		// An index of -1 is only valid if deselecting is enabled or there are no valid tabs.
		ERR_FAIL_COND_MSG(!_can_deselect(), "Cannot deselect tabs, deselection is not enabled.");
	} else {
		if (!initialized && p_current >= get_tab_count()) {
			queued_current = p_current;
			return;
		}
		ERR_FAIL_INDEX(p_current, get_tab_count());
	}

	previous = current;
	current = p_current;

	if (current == previous) {
		emit_signal(SNAME("tab_selected"), current);
		return;
	}

	emit_signal(SNAME("tab_selected"), current);

	_update_cache();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_accessibility_update();
	queue_redraw();

	emit_signal(SNAME("tab_changed"), p_current);
}

int TabBar::get_current_tab() const {
	return current;
}

int TabBar::get_previous_tab() const {
	return previous;
}

int TabBar::get_hovered_tab() const {
	return hover;
}

int TabBar::get_previous_available(int p_idx) const {
	ERR_FAIL_COND_V(p_idx < -1 || p_idx > get_tab_count(), -1);
	const int idx = p_idx == -1 ? get_current_tab() : p_idx;
	const int offset_end = idx + 1;
	for (int i = 1; i < offset_end; i++) {
		int target_tab = idx - i;
		if (target_tab < 0) {
			target_tab += get_tab_count();
		}
		if (!is_tab_disabled(target_tab) && !is_tab_hidden(target_tab)) {
			return target_tab;
		}
	}
	return -1;
}

int TabBar::get_next_available(int p_idx) const {
	ERR_FAIL_COND_V(p_idx < -1 || p_idx > get_tab_count(), -1);
	const int idx = p_idx == -1 ? get_current_tab() : p_idx;
	const int offset_end = get_tab_count() - idx;
	for (int i = 1; i < offset_end; i++) {
		int target_tab = (idx + i) % get_tab_count();
		if (!is_tab_disabled(target_tab) && !is_tab_hidden(target_tab)) {
			return target_tab;
		}
	}
	return -1;
}

bool TabBar::select_previous_available() {
	const int previous_available = get_previous_available();
	if (previous_available != -1) {
		set_current_tab(previous_available);
	}
	return previous_available != -1;
}

bool TabBar::select_next_available() {
	const int next_available = get_next_available();
	if (next_available != -1) {
		set_current_tab(next_available);
	}
	return next_available != -1;
}

void TabBar::set_tab_offset(int p_offset) {
	ERR_FAIL_INDEX(p_offset, tabs.size());
	offset = p_offset;
	_update_cache();
	queue_accessibility_update();
	queue_redraw();
}

int TabBar::get_tab_offset() const {
	return offset;
}

bool TabBar::get_offset_buttons_visible() const {
	return buttons_visible;
}

void TabBar::set_tab_title(int p_tab, const String &p_title) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	if (tabs[p_tab].text == p_title) {
		return;
	}

	tabs.write[p_tab].text = p_title;

	_shape(p_tab);
	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

String TabBar::get_tab_title(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), "");
	return tabs[p_tab].text;
}

void TabBar::set_tab_tooltip(int p_tab, const String &p_tooltip) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs.write[p_tab].tooltip = p_tooltip;
	queue_accessibility_update();
}

String TabBar::get_tab_tooltip(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), "");
	return tabs[p_tab].tooltip;
}

void TabBar::set_tab_text_direction(int p_tab, Control::TextDirection p_text_direction) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);

	if (tabs[p_tab].text_direction != p_text_direction) {
		tabs.write[p_tab].text_direction = p_text_direction;

		_shape(p_tab);
		queue_accessibility_update();
		queue_redraw();
	}
}

Control::TextDirection TabBar::get_tab_text_direction(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Control::TEXT_DIRECTION_INHERITED);
	return tabs[p_tab].text_direction;
}

void TabBar::set_tab_language(int p_tab, const String &p_language) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	if (tabs[p_tab].language != p_language) {
		tabs.write[p_tab].language = p_language;

		_shape(p_tab);
		_update_cache();
		_ensure_no_over_offset();
		if (scroll_to_selected) {
			ensure_tab_visible(current);
		}
		queue_accessibility_update();
		queue_redraw();
		update_desired_size();
		update_minimum_size();
	}
}

String TabBar::get_tab_language(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), "");
	return tabs[p_tab].language;
}

void TabBar::set_tab_icon(int p_tab, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	if (tabs[p_tab].icon == p_icon) {
		return;
	}

	tabs.write[p_tab].icon = p_icon;

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

Ref<Texture2D> TabBar::get_tab_icon(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Ref<Texture2D>());
	return tabs[p_tab].icon;
}

void TabBar::set_tab_icon_max_width(int p_tab, int p_width) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	if (tabs[p_tab].icon_max_width == p_width) {
		return;
	}

	tabs.write[p_tab].icon_max_width = p_width;

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

int TabBar::get_tab_icon_max_width(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), 0);
	return tabs[p_tab].icon_max_width;
}

void TabBar::set_font_color_override_all(int p_tab, const Color &p_color) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	Tab &tab = tabs.write[p_tab];
	for (int i = 0; i < DrawMode::DRAW_MAX; i++) {
		tab.font_color_overrides[i] = p_color;
	}

	queue_redraw();
}

void TabBar::set_font_color_override(int p_tab, DrawMode p_draw_mode, const Color &p_color) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	ERR_FAIL_INDEX(p_draw_mode, DrawMode::DRAW_MAX);

	if (tabs[p_tab].font_color_overrides[p_draw_mode] == p_color) {
		return;
	}

	tabs.write[p_tab].font_color_overrides[p_draw_mode] = p_color;

	queue_redraw();
}

Color TabBar::get_font_color_override(int p_tab, DrawMode p_draw_mode) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Color());
	ERR_FAIL_INDEX_V(p_draw_mode, DrawMode::DRAW_MAX, Color());

	return tabs[p_tab].font_color_overrides[p_draw_mode];
}

void TabBar::set_tab_disabled(int p_tab, bool p_disabled) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	if (tabs[p_tab].disabled == p_disabled) {
		return;
	}

	tabs.write[p_tab].disabled = p_disabled;

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

bool TabBar::is_tab_disabled(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), false);
	return tabs[p_tab].disabled;
}

void TabBar::set_tab_hidden(int p_tab, bool p_hidden) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	if (tabs[p_tab].hidden == p_hidden) {
		return;
	}

	tabs.write[p_tab].hidden = p_hidden;

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

bool TabBar::is_tab_hidden(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), false);
	return tabs[p_tab].hidden;
}

void TabBar::set_tab_metadata(int p_tab, const Variant &p_metadata) {
	ERR_FAIL_INDEX(p_tab, tabs.size());
	tabs.write[p_tab].metadata = p_metadata;
}

Variant TabBar::get_tab_metadata(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Variant());
	return tabs[p_tab].metadata;
}

void TabBar::set_tab_button_icon(int p_tab, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_INDEX(p_tab, tabs.size());

	if (tabs[p_tab].right_button == p_icon) {
		return;
	}

	tabs.write[p_tab].right_button = p_icon;

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

Ref<Texture2D> TabBar::get_tab_button_icon(int p_tab) const {
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Ref<Texture2D>());
	return tabs[p_tab].right_button;
}

void TabBar::_update_hover() {
	if (!is_inside_tree()) {
		return;
	}

	ERR_FAIL_COND(tabs.is_empty());

	const Point2 &pos = get_local_mouse_position();
	// Test hovering to display right or close button.
	int hover_now = -1;
	int hover_buttons = -1;
	for (int i = offset; i <= max_drawn_tab; i++) {
		if (tabs[i].hidden) {
			continue;
		}

		Rect2 rect = get_tab_rect(i);
		if (rect.has_point(pos)) {
			hover_now = i;
		}

		if (tabs[i].rb_rect.has_point(pos)) {
			rb_hover = i;
			cb_hover = -1;
			hover_buttons = i;
		} else if (!tabs[i].disabled && tabs[i].cb_rect.has_point(pos)) {
			cb_hover = i;
			rb_hover = -1;
			hover_buttons = i;
		}

		if (hover_buttons != -1) {
			queue_redraw();
			break;
		}
	}

	if (hover != hover_now) {
		hover = hover_now;

		if (hover != -1) {
			emit_signal(SNAME("tab_hovered"), hover);
		}

		_update_cache();
		queue_redraw();
	}

	if (hover_buttons == -1) { // No hover.
		int rb_hover_old = rb_hover;
		int cb_hover_old = cb_hover;

		rb_hover = hover_buttons;
		cb_hover = hover_buttons;

		if (rb_hover != rb_hover_old || cb_hover != cb_hover_old) {
			queue_redraw();
		}
	}
}

void TabBar::_update_cache(bool p_update_hover) {
	if (tabs.is_empty()) {
		buttons_visible = false;
		return;
	}

	if (!theme_cache.tab_unselected_style.is_valid() || !theme_cache.tab_hovered_style.is_valid() || !theme_cache.tab_selected_style.is_valid() || !theme_cache.tab_disabled_style.is_valid() || !theme_cache.button_hl_style.is_valid() || !theme_cache.decrement_icon.is_valid() || !theme_cache.increment_icon.is_valid()) {
		buttons_visible = false;
		return;
	}

	Size2 combined_max = get_combined_maximum_size();
	const bool previous_buttons_visible = buttons_visible;

	int combined_max_width = INT_MAX;
	if (combined_max.width >= 0) {
		combined_max_width = int(combined_max.width);
		if (!vertical && clip_tabs && previous_buttons_visible) {
			combined_max_width -= theme_cache.increment_icon->get_width() + theme_cache.decrement_icon->get_width();
		}
	}
	int effective_max_width = max_width > 0 ? MIN(max_width, combined_max_width) : combined_max_width;

	int limit, limit_minus_buttons;
	if (vertical) { /* VERTICAL */
		int button_height = MAX(theme_cache.decrement_vertical_icon->get_height(), theme_cache.increment_vertical_icon->get_height());
		int control_height = int(get_size().height);
		int combined_max_height = combined_max.height >= 0 ? int(combined_max.height) : control_height;
		limit = MIN(control_height, combined_max_height);
		limit_minus_buttons = limit - button_height;
	} else { /* HORIZONTAL */
		limit = combined_max.width > 0 ? MIN(combined_max.width, int(get_size().width)) : int(get_size().width);
		limit_minus_buttons = limit - theme_cache.increment_icon->get_width() - theme_cache.decrement_icon->get_width();
	}
	int w = 0; // For horizontal: width, for vertical: height

	// Calculate sizing information for the chosen tab sizing mode.
	int visible_count = 0;
	int total_base_width = 0;
	int max_base_width = 0;

	for (int i = 0; i < tabs.size(); i++) {
		// Update text buffer so that the sizing calculations use the correct text size.
		tabs.write[i].text_buf->set_width(-1);
		tabs.write[i].size_text = Math::ceil(tabs[i].text_buf->get_size().x);

		int tab_width = get_tab_width(i);
		tabs.write[i].size_cache = tab_width;

		if (tabs[i].hidden) {
			continue;
		}

		total_base_width += tab_width;
		max_base_width = MAX(max_base_width, tab_width);
		visible_count++;
	}

	bool can_expand = false;
	int expand_remainder = 0;
	float justify_ratio = 1.0;
	int uniform_width = 0;

	if (visible_count > 0 && tab_sizing != TAB_SIZING_FIT_CONTENT) {
		int total_separation = MAX(0, visible_count - 1) * theme_cache.tab_separation;
		int available_space = limit - total_separation;

		if (tab_sizing == TAB_SIZING_UNIFORM) {
			// All tabs take the width of the largest tab.
			uniform_width = max_base_width;
		} else if (tab_sizing == TAB_SIZING_EXPAND) {
			// Distribute space equally among all tabs.
			int width_per_tab = available_space / visible_count;
			// Only expand if the resulting width fits the largest tab, otherwise fall back to fit content.
			if (width_per_tab >= max_base_width) {
				can_expand = true;
				uniform_width = width_per_tab;
				expand_remainder = available_space % visible_count;
			}
		} else if (tab_sizing == TAB_SIZING_JUSTIFY) {
			// Scale tabs proportionally to fill the space, if we have space to fill.
			if (total_base_width <= available_space) {
				can_expand = true;
				justify_ratio = (float)available_space / (float)total_base_width;

				// Recalculate the remainder that results from integer rounding.
				int simulated_total = 0;
				for (int i = 0; i < tabs.size(); i++) {
					if (tabs[i].hidden) {
						continue;
					}
					simulated_total += (int)(tabs[i].size_cache * justify_ratio);
				}
				expand_remainder = available_space - simulated_total;
			}
		}
	}

	max_drawn_tab = tabs.size() - 1;

	int visible_index = 0;

	for (int i = 0; i < tabs.size(); i++) {
		int natural_width = tabs[i].size_cache;
		int final_width = natural_width;

		// Apply sizing.
		if (!tabs[i].hidden) {
			if (tab_sizing == TAB_SIZING_UNIFORM) {
				final_width = uniform_width;
			} else if (can_expand) {
				if (tab_sizing == TAB_SIZING_EXPAND) {
					final_width = uniform_width;
				} else if (tab_sizing == TAB_SIZING_JUSTIFY) {
					final_width = (int)(final_width * justify_ratio);
				}

				// Distribute remaining pixels.
				if (visible_index < expand_remainder) {
					final_width++;
				}
				visible_index++;
			}
		}

		tabs.write[i].size_cache = final_width;
		tabs.write[i].accessibility_item_dirty = true;

		tabs.write[i].truncated = effective_max_width > 0 && effective_max_width < INT_MAX && tabs[i].size_cache > effective_max_width;
		if (tabs[i].truncated) {
			int size_textless = natural_width;
			if (!tabs[i].text.is_empty()) {
				size_textless -= tabs[i].size_text;
			}
			int mw = MAX(size_textless, effective_max_width);

			tabs.write[i].size_text = MAX(mw - size_textless, 1);
			tabs.write[i].text_buf->set_width(tabs[i].size_text);
			tabs.write[i].size_cache = size_textless + tabs[i].size_text;
		}
	}

	auto layout_tabs = [&](int p_limit, int p_limit_minus_buttons) {
		int local_w = 0;
		max_drawn_tab = tabs.size() - 1;

		for (int i = 0; i < tabs.size(); i++) {
			if (i < offset || i > max_drawn_tab) {
				tabs.write[i].ofs_cache = 0;
				continue;
			}

			tabs.write[i].ofs_cache = local_w;

			if (tabs[i].hidden) {
				continue;
			}

			local_w += tabs[i].size_cache;

			// Check if all tabs would fit inside the area.
			if (clip_tabs && i > offset && (local_w > p_limit || (offset > 0 && local_w > p_limit_minus_buttons))) {
				tabs.write[i].ofs_cache = 0;

				local_w -= tabs[i].size_cache;
				local_w -= theme_cache.tab_separation;

				max_drawn_tab = i - 1;

				while (local_w > p_limit_minus_buttons && max_drawn_tab > offset) {
					tabs.write[max_drawn_tab].ofs_cache = 0;

					if (!tabs[max_drawn_tab].hidden) {
						local_w -= tabs[max_drawn_tab].size_cache;
						local_w -= theme_cache.tab_separation;
					}

					max_drawn_tab--;
				}
			} else if (i < tabs.size() - 1) {
				// Only add the tab separation if this isn't the last tab drawn.
				local_w += theme_cache.tab_separation;
			}
		}

		return local_w;
	};

	if (vertical && clip_tabs && offset == 0) { /* VERTICAL */
		int base_vertical_limit_minus_popup = limit;
		int base_vertical_limit_minus_buttons = limit_minus_buttons;

		// First, check whether all tabs fit without reserving the arrows row.
		w = layout_tabs(base_vertical_limit_minus_popup, base_vertical_limit_minus_buttons);
		if (max_drawn_tab < tabs.size() - 1) {
			// If clipping is still required, reserve row space and recompute.
			w = layout_tabs(base_vertical_limit_minus_buttons, base_vertical_limit_minus_buttons);
			limit = base_vertical_limit_minus_buttons;
			limit_minus_buttons = base_vertical_limit_minus_buttons;
		} else {
			limit = base_vertical_limit_minus_popup;
			limit_minus_buttons = base_vertical_limit_minus_buttons;
		}
	} else { /* HORIZONTAL */
		w = layout_tabs(limit, limit_minus_buttons);
	}

	missing_right = max_drawn_tab < tabs.size() - 1;
	buttons_visible = clip_tabs && (offset > 0 || missing_right);

	// Truncation width can depend on whether the scroll strip is reserved.
	// If visibility changed this frame, run once more with the updated state.
	if (clip_tabs && previous_buttons_visible != buttons_visible) {
		_update_cache(p_update_hover);
		return;
	}

	if (tab_alignment == ALIGNMENT_BEGIN) {
		if (p_update_hover) {
			_update_hover();
		}
		return;
	}

	if (tab_alignment == ALIGNMENT_CENTER) {
		w = ((buttons_visible ? limit_minus_buttons : limit) - w) / 2;
	} else if (tab_alignment == ALIGNMENT_END) {
		w = (buttons_visible ? limit_minus_buttons : limit) - w;
	}

	for (int i = offset; i <= max_drawn_tab; i++) {
		if (!tabs[i].hidden) {
			tabs.write[i].ofs_cache = w;

			w += tabs[i].size_cache;
			w += theme_cache.tab_separation;
		}
	}

	if (p_update_hover) {
		_update_hover();
	}
}

Size2 TabBar::get_desired_size() const {
	if (!clip_tabs || tabs.is_empty()) {
		return Size2();
	}
	Size2 combined_max = get_combined_maximum_size();
	if (vertical) { /* VERTICAL */
		if (combined_max.height < 0) {
			return Size2();
		}
	} else if (combined_max.width < 0) { /* HORIZONTAL */
		return Size2();
	}

	int limit = vertical ? int(combined_max.height) : int(combined_max.width);

	int primary_sum = 0;
	int cross_max = 0;

	for (int i = 0; i < tabs.size(); i++) {
		if (tabs[i].hidden) {
			continue;
		}

		const TabMetrics metrics = _get_tab_metrics(i, true);
		int tab_w = metrics.row_width;
		if (max_width > 0 && tab_w > max_width) {
			const int size_textless = tab_w - tabs[i].size_text;
			tab_w = MAX(size_textless, max_width);
		}
		const int tab_h = metrics.row_height;

		if (vertical) { /* VERTICAL */
			primary_sum += tab_h;
			cross_max = MAX(cross_max, tab_w);
		} else { /* HORIZONTAL */
			primary_sum += tab_w;
			cross_max = MAX(cross_max, tab_h);
		}

		if (i < tabs.size() - 1) {
			primary_sum += theme_cache.tab_separation;
		}
	}

	if (vertical) { /* VERTICAL */
		return Size2(MAX(cross_max, (int)_get_vertical_popup_button_min_size(this).width), MIN(primary_sum, limit));
	}

	return Size2(MIN(primary_sum, limit), cross_max);
}

void TabBar::_hover_switch_timeout() {
	set_current_tab(hover);
}

void TabBar::_on_mouse_exited() {
	rb_hover = -1;
	cb_hover = -1;
	hover = -1;
	highlight_arrow = -1;
	dragging_valid_tab = false;

	_update_cache(false);
	queue_redraw();
}

void TabBar::_on_maximum_size_changed() {
	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_redraw();
	update_minimum_size();
}

void TabBar::add_tab(const String &p_str, const Ref<Texture2D> &p_icon) {
	Tab t;
	t.text = p_str;
	t.text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	t.icon = p_icon;
	tabs.push_back(t);

	_shape(tabs.size() - 1);
	_update_cache();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();

	if (!deselect_enabled && tabs.size() == 1) {
		if (is_inside_tree()) {
			set_current_tab(0);
		} else {
			current = 0;
			previous = -1;
		}
	}
}

void TabBar::clear_tabs() {
	if (tabs.is_empty()) {
		return;
	}

	for (int i = 0; i < tabs.size(); i++) {
		if (tabs[i].accessibility_item_element.is_valid()) {
			AccessibilityServer::get_singleton()->free_element(tabs.write[i].accessibility_item_element);
			tabs.write[i].accessibility_item_element = RID();
		}
	}
	tabs.clear();
	offset = 0;
	max_drawn_tab = 0;
	current = -1;
	previous = -1;

	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();
	notify_property_list_changed();
}

void TabBar::remove_tab(int p_idx) {
	ERR_FAIL_INDEX(p_idx, tabs.size());

	if (tabs[p_idx].accessibility_item_element.is_valid()) {
		AccessibilityServer::get_singleton()->free_element(tabs.write[p_idx].accessibility_item_element);
		tabs.write[p_idx].accessibility_item_element = RID();
	}
	tabs.remove_at(p_idx);

	bool is_tab_changing = current == p_idx;

	if (current >= p_idx && current > 0) {
		current--;
	}
	if (previous >= p_idx && previous > 0) {
		previous--;
	}

	if (tabs.is_empty()) {
		offset = 0;
		max_drawn_tab = 0;
		current = -1;
		previous = -1;
	} else {
		if (current != -1) {
			// Try to change to a valid tab if possible (without firing the `tab_selected` signal).
			for (int i = current; i < tabs.size(); i++) {
				if (!is_tab_disabled(i) && !is_tab_hidden(i)) {
					current = i;
					break;
				}
			}
			// If nothing, try backwards.
			if (is_tab_disabled(current) || is_tab_hidden(current)) {
				for (int i = current - 1; i >= 0; i--) {
					if (!is_tab_disabled(i) && !is_tab_hidden(i)) {
						current = i;
						break;
					}
				}
			}
			// If still no valid tab, deselect.
			if (is_tab_disabled(current) || is_tab_hidden(current)) {
				current = -1;
			}
		}
		offset = MIN(offset, tabs.size() - 1);
		max_drawn_tab = MIN(max_drawn_tab, tabs.size() - 1);

		_update_cache();
		_ensure_no_over_offset();
		if (scroll_to_selected) {
			ensure_tab_visible(current);
		}
	}

	queue_accessibility_update();
	queue_redraw();
	update_desired_size();
	update_minimum_size();
	notify_property_list_changed();

	if (is_tab_changing && is_inside_tree()) {
		emit_signal(SNAME("tab_changed"), current);
	}
}

Variant TabBar::get_drag_data(const Point2 &p_point) {
	Variant drag_data = Control::get_drag_data(p_point);
	if (drag_data != Variant()) {
		return drag_data;
	}

	if (drag_to_rearrange_enabled) {
		return _handle_get_drag_data("tab_bar_tab", p_point);
	}
	return Variant();
}

bool TabBar::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	if (switch_on_drag_hover) {
		_handle_switch_on_hover(p_data);
	}

	bool drop_override = Control::can_drop_data(p_point, p_data);
	if (drop_override) {
		return drop_override;
	}

	if (drag_to_rearrange_enabled) {
		return _handle_can_drop_data("tab_bar_tab", p_point, p_data);
	}
	return false;
}

void TabBar::drop_data(const Point2 &p_point, const Variant &p_data) {
	Control::drop_data(p_point, p_data);

	if (drag_to_rearrange_enabled) {
		_handle_drop_data("tab_bar_tab", p_point, p_data, callable_mp(this, &TabBar::move_tab), callable_mp(this, &TabBar::_move_tab_from));
	}
}

Variant TabBar::_handle_get_drag_data(const String &p_type, const Point2 &p_point) {
	int tab_over = (p_point == Vector2(Math::INF, Math::INF)) ? current : get_tab_idx_at_point(p_point);
	if (tab_over < 0) {
		return Variant();
	}

	HBoxContainer *drag_preview = memnew(HBoxContainer);

	if (tabs[tab_over].icon.is_valid()) {
		const Size2 icon_size = _get_tab_icon_size(tab_over);

		TextureRect *tf = memnew(TextureRect);
		tf->set_texture(tabs[tab_over].icon);
		tf->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
		tf->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
		tf->set_custom_minimum_size(icon_size);

		drag_preview->add_child(tf);
	}

	Label *label = memnew(Label(get_tab_title(tab_over)));
	label->set_auto_translate_mode(get_auto_translate_mode()); // Reflect how the title is displayed.
	drag_preview->add_child(label);

	set_drag_preview(drag_preview);

	Dictionary drag_data;
	drag_data["type"] = "tab";
	drag_data["tab_type"] = p_type;
	drag_data["tab_index"] = tab_over;
	drag_data["from_path"] = get_path();

	return drag_data;
}

bool TabBar::_handle_can_drop_data(const String &p_type, const Point2 &p_point, const Variant &p_data) const {
	Dictionary d = p_data;
	if (d.get("type", "").operator String() != "tab") {
		return false;
	}

	const String tab_type = d.get("tab_type", "");
	if (tab_type == p_type) {
		NodePath from_path = d["from_path"];
		NodePath to_path = get_path();
		if (from_path == to_path) {
			return true;
		} else if (get_tabs_rearrange_group() != -1) {
			// Drag and drop between other TabBars.
			Node *from_node = get_node(from_path);
			TabBar *from_tabs = Object::cast_to<TabBar>(from_node);
			if (from_tabs && from_tabs->get_tabs_rearrange_group() == get_tabs_rearrange_group()) {
				return true;
			}
		}
	}
	return false;
}

void TabBar::_handle_drop_data(const String &p_type, const Point2 &p_point, const Variant &p_data, const Callable &p_move_tab_callback, const Callable &p_move_tab_from_other_callback) {
	Dictionary d = p_data;
	if (d.get("type", "").operator String() != "tab") {
		return;
	}

	const String tab_type = d.get("tab_type", "");
	if (tab_type == p_type) {
		int tab_from_id = d["tab_index"];
		int hover_now = (p_point == Vector2(Math::INF, Math::INF)) ? current : get_closest_tab_idx_to_point(p_point);
		NodePath from_path = d["from_path"];
		NodePath to_path = get_path();

		if (from_path == to_path) {
			if (tab_from_id == hover_now) {
				return;
			}

			// Drop the new tab to the left or right (or above/below for vertical) depending on where the target tab is being hovered.
			if (hover_now != -1) {
				Rect2 tab_rect = get_tab_rect(hover_now);
				const bool drop_before = _is_point_primary_before_or_at_mid(p_point, tab_rect);
				if (drop_before) {
					if (hover_now > tab_from_id) {
						hover_now -= 1;
					}
				} else if (tab_from_id > hover_now) {
					hover_now += 1;
				}
			} else {
				hover_now = _is_point_before_first_tab(p_point) ? 0 : get_tab_count() - 1;
			}

			p_move_tab_callback.call(tab_from_id, hover_now);
			if (!is_tab_disabled(hover_now)) {
				emit_signal(SNAME("active_tab_rearranged"), hover_now);
				set_current_tab(hover_now);
			}
		} else if (get_tabs_rearrange_group() != -1) {
			// Drag and drop between Tabs.

			Node *from_node = get_node(from_path);
			TabBar *from_tabs = Object::cast_to<TabBar>(from_node);

			if (from_tabs && from_tabs->get_tabs_rearrange_group() == get_tabs_rearrange_group()) {
				if (tab_from_id >= from_tabs->get_tab_count()) {
					return;
				}

				// Drop the new tab to the left or right (or above/below for vertical) depending on where the target tab is being hovered.
				if (hover_now != -1) {
					Rect2 tab_rect = get_tab_rect(hover_now);
					const bool drop_after = _is_point_primary_after_mid(p_point, tab_rect);
					if (drop_after) {
						hover_now += 1;
					}
				} else {
					hover_now = tabs.is_empty() || _is_point_before_first_tab(p_point) ? 0 : get_tab_count();
				}

				p_move_tab_from_other_callback.call(from_tabs, tab_from_id, hover_now);
			}
		}
	}
}

void TabBar::_handle_switch_on_hover(const Variant &p_data) const {
	Dictionary d = p_data;
	if (d.get("type", "").operator String() == "tab") {
		// Dragging a tab shouldn't switch on hover.
		return;
	}

	if (hover > -1 && hover != current) {
		if (hover_switch_delay->is_stopped()) {
			const_cast<TabBar *>(this)->hover_switch_delay->start(theme_cache.hover_switch_wait_msec * 0.001);
		}
	} else if (!hover_switch_delay->is_stopped()) {
		hover_switch_delay->stop();
	}
}

void TabBar::_move_tab_from(TabBar *p_from_tabbar, int p_from_index, int p_to_index) {
	Tab moving_tab = p_from_tabbar->tabs[p_from_index];
	moving_tab.accessibility_item_element = RID();
	moving_tab.accessibility_item_dirty = true;
	p_from_tabbar->remove_tab(p_from_index);
	tabs.insert(p_to_index, moving_tab);

	if (tabs.size() > 1) {
		if (current >= p_to_index) {
			current++;
		}
		if (previous >= p_to_index) {
			previous++;
		}
	}

	if (!is_tab_disabled(p_to_index)) {
		set_current_tab(p_to_index);
	} else {
		_update_cache();
		queue_redraw();
	}

	queue_accessibility_update();
	update_desired_size();
	update_minimum_size();
}

int TabBar::get_tab_idx_at_point(const Point2 &p_point) const {
	if (tabs.is_empty()) {
		return -1;
	}

	int hover_now = -1;

	for (int i = offset; i <= max_drawn_tab; i++) {
		if (!tabs[i].hidden) {
			Rect2 rect = get_tab_rect(i);
			if (rect.has_point(p_point)) {
				hover_now = i;
			}
		}
	}

	return hover_now;
}

int TabBar::get_closest_tab_idx_to_point(const Point2 &p_point) const {
	if (tabs.is_empty()) {
		return -1;
	}

	int closest_tab = get_tab_idx_at_point(p_point);
	float closest_distance = FLT_MAX;

	// Search along the tab layout axis.
	if (closest_tab == -1) {
		for (int i = offset; i <= max_drawn_tab; i++) {
			if (!tabs[i].hidden) {
				const Rect2 tab_rect = get_tab_rect(i);
				const float point_primary = vertical ? p_point.y : p_point.x;
				const float rect_center = vertical ? tab_rect.get_center().y : tab_rect.get_center().x;
				float distance = Math::abs(rect_center - point_primary);
				if (distance < closest_distance) {
					closest_distance = distance;
					closest_tab = i;
				}
			}
		}
	}

	return closest_tab;
}

void TabBar::set_tab_alignment(AlignmentMode p_alignment) {
	ERR_FAIL_INDEX(p_alignment, ALIGNMENT_MAX);

	if (tab_alignment == p_alignment) {
		return;
	}

	tab_alignment = p_alignment;

	_update_cache();
	queue_redraw();
}

TabBar::AlignmentMode TabBar::get_tab_alignment() const {
	return tab_alignment;
}

void TabBar::set_tab_sizing(SizingMode p_sizing) {
	ERR_FAIL_INDEX(p_sizing, TAB_SIZING_MAX);

	if (tab_sizing == p_sizing) {
		return;
	}

	tab_sizing = p_sizing;

	_update_cache();
	queue_redraw();
	update_minimum_size();
}

TabBar::SizingMode TabBar::get_tab_sizing() const {
	return tab_sizing;
}

void TabBar::set_clip_tabs(bool p_clip_tabs) {
	if (clip_tabs == p_clip_tabs) {
		return;
	}
	clip_tabs = p_clip_tabs;

	if (!clip_tabs) {
		offset = 0;
		max_drawn_tab = 0;
	}

	_update_cache();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

bool TabBar::get_clip_tabs() const {
	return clip_tabs;
}

void TabBar::set_tab_style_side(TabStyleSide p_side) {
	ERR_FAIL_INDEX(p_side, TAB_STYLE_SIDE_MAX);
	if (tab_style_side == p_side) {
		return;
	}
	tab_style_side = p_side;
	queue_redraw();
}

TabBar::TabStyleSide TabBar::get_tab_style_side() const {
	return tab_style_side;
}

#ifndef DISABLE_DEPRECATED
void TabBar::set_tab_style_v_flip(bool p_tab_style_v_flip) {
	set_tab_style_side(p_tab_style_v_flip ? TAB_STYLE_SIDE_BOTTOM : TAB_STYLE_SIDE_TOP);
}
#endif

void TabBar::set_vertical(bool p_vertical) {
	if (vertical == p_vertical) {
		return;
	}
	vertical = p_vertical;
	_update_cache();
	queue_redraw();
	update_minimum_size();
}

bool TabBar::is_vertical() const {
	return vertical;
}

void TabBar::set_tab_text_rotation(TabTextRotation p_rotation) {
	ERR_FAIL_INDEX(p_rotation, TAB_TEXT_ROTATION_MAX);
	if (tab_text_rotation == p_rotation) {
		return;
	}
	tab_text_rotation = p_rotation;
	_update_cache();
	queue_redraw();
	update_minimum_size();
}

TabBar::TabTextRotation TabBar::get_tab_text_rotation() const {
	return tab_text_rotation;
}

void TabBar::move_tab(int p_from, int p_to) {
	if (p_from == p_to) {
		return;
	}

	ERR_FAIL_INDEX(p_from, tabs.size());
	ERR_FAIL_INDEX(p_to, tabs.size());

	Tab tab_from = tabs[p_from];
	tab_from.accessibility_item_dirty = true;

	tabs.remove_at(p_from);
	tabs.insert(p_to, tab_from);

	if (current == p_from) {
		current = p_to;
	} else if (current > p_from && current <= p_to) {
		current--;
	} else if (current < p_from && current >= p_to) {
		current++;
	}

	if (previous == p_from) {
		previous = p_to;
	} else if (previous > p_from && previous <= p_to) {
		previous--;
	} else if (previous < p_from && previous >= p_to) {
		previous++;
	}

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_accessibility_update();
	queue_redraw();
	notify_property_list_changed();
}

void TabBar::_get_scroll_button_icons(Ref<Texture2D> &r_dec_icon, Ref<Texture2D> &r_inc_icon) const {
	if (vertical) { /* VERTICAL */
		r_dec_icon = theme_cache.decrement_vertical_icon;
		r_inc_icon = theme_cache.increment_vertical_icon;
	} else { /* HORIZONTAL */
		r_dec_icon = theme_cache.decrement_icon;
		r_inc_icon = theme_cache.increment_icon;
	}
}

Size2 TabBar::_get_vertical_popup_button_min_size(const TabBar *p_tab_bar) const {
	// The vertical popup/menu button (owned by TabContainer) is reparented to be
	// a direct child of the TabBar. Only direct children may contribute here:
	// looking at siblings/parents would pick up unrelated controls and shift
	// the scroll arrows' x position.
	for (int i = 0; i < p_tab_bar->get_child_count(); i++) {
		Control *popup_button = Object::cast_to<Control>(p_tab_bar->get_child(i));
		if (popup_button && popup_button->is_visible()) {
			return popup_button->get_combined_minimum_size();
		}
	}

	return Size2();
}

int TabBar::_get_reserved_vertical_buttons_row_height(bool p_assume_buttons_visible) const {
	if (!vertical) {
		return 0;
	}

	if (!clip_tabs) {
		return _get_vertical_popup_button_min_size(this).height;
	}

	// Only reserve a dedicated bottom row when the scroll buttons are actually visible (or
	// when we are computing layout assuming they will be visible). When tabs fit and the
	// scroll buttons are hidden, any optional popup/menu button is positioned inline after
	// the last tab and should not consume extra empty space at the bottom.
	if (!(p_assume_buttons_visible || buttons_visible)) {
		return 0;
	}

	int row_height = _get_vertical_popup_button_min_size(this).height;
	Ref<Texture2D> dec_icon;
	Ref<Texture2D> inc_icon;
	_get_scroll_button_icons(dec_icon, inc_icon);
	if (dec_icon.is_valid() && inc_icon.is_valid()) {
		row_height = MAX(row_height, (int)MAX(dec_icon->get_height(), inc_icon->get_height()));
	}

	return row_height;
}

int TabBar::get_vertical_buttons_row_top() const {
	if (!vertical || !buttons_visible) {
		return 0;
	}

	for (int i = max_drawn_tab; i >= offset; i--) {
		if (tabs[i].hidden) {
			continue;
		}

		return tabs[i].ofs_cache + tabs[i].size_cache;
	}

	return 0;
}

int TabBar::_get_primary_limit_minus_buttons(int p_primary_limit) const {
	if (vertical) { /* VERTICAL */
		return MAX(0, p_primary_limit - _get_reserved_vertical_buttons_row_height());
	}

	Ref<Texture2D> dec_icon;
	Ref<Texture2D> inc_icon;
	_get_scroll_button_icons(dec_icon, inc_icon);
	if (!dec_icon.is_valid() || !inc_icon.is_valid()) {
		return p_primary_limit;
	}

	return p_primary_limit - dec_icon->get_width() - inc_icon->get_width();
}

Rect2 TabBar::_get_tabs_content_rect(bool p_assume_buttons_visible) const {
	if (!vertical) {
		return Rect2(Point2(), get_size());
	}

	const int buttons_row = _get_reserved_vertical_buttons_row_height(p_assume_buttons_visible);
	const int content_height = MAX(0, (int)get_size().height - buttons_row);
	return Rect2(0, 0, get_size().width, content_height);
}

void TabBar::_get_scroll_button_rects(Rect2 &r_dec_rect, Rect2 &r_inc_rect) const {
	r_dec_rect = Rect2();
	r_inc_rect = Rect2();

	Ref<Texture2D> dec_icon;
	Ref<Texture2D> inc_icon;
	_get_scroll_button_icons(dec_icon, inc_icon);
	if (!dec_icon.is_valid() || !inc_icon.is_valid()) {
		return;
	}

	if (vertical) { /* VERTICAL */
		const Size2 dec_size = dec_icon->get_size();
		const Size2 inc_size = inc_icon->get_size();
		const float row_height = _get_reserved_vertical_buttons_row_height();
		const float row_y = get_vertical_buttons_row_top();
		const float popup_width = _get_vertical_popup_button_min_size(this).width;
		const float buttons_row_width = popup_width + dec_size.width + inc_size.width;
		const float buttons_row_x = MAX(0.0f, (get_size().width - buttons_row_width) * 0.5f) + popup_width;

		r_dec_rect = Rect2(Point2(buttons_row_x, row_y + (row_height - dec_size.height) * 0.5f), dec_size);
		r_inc_rect = Rect2(Point2(buttons_row_x + dec_size.width, row_y + (row_height - inc_size.height) * 0.5f), inc_size);
		return;
	}

	const bool rtl = is_layout_rtl();
	const int buttons_width = dec_icon->get_width() + inc_icon->get_width();
	int buttons_x = rtl ? 0 : _get_primary_limit_minus_buttons(get_size().width);

	if (clip_tabs && buttons_visible) {
		bool found_visible_tab = false;
		int tabs_left = 0;
		int tabs_right = 0;

		for (int i = offset; i <= max_drawn_tab && i < tabs.size(); i++) {
			if (tabs[i].hidden) {
				continue;
			}

			const int tab_x = rtl ? (get_size().width - tabs[i].ofs_cache - tabs[i].size_cache) : tabs[i].ofs_cache;
			const int tab_end = tab_x + tabs[i].size_cache;

			if (!found_visible_tab) {
				tabs_left = tab_x;
				tabs_right = tab_end;
				found_visible_tab = true;
			} else {
				tabs_left = MIN(tabs_left, tab_x);
				tabs_right = MAX(tabs_right, tab_end);
			}
		}

		if (found_visible_tab) {
			const int max_buttons_x = MAX((int)get_size().width - buttons_width, 0);
			buttons_x = rtl ? MAX(0, tabs_left - buttons_width) : MIN(max_buttons_x, MAX(0, tabs_right));
		}
	}

	r_dec_rect = Rect2(Point2(buttons_x, 0), Size2(dec_icon->get_width(), get_size().height));
	r_inc_rect = Rect2(Point2(buttons_x + dec_icon->get_width(), 0), Size2(inc_icon->get_width(), get_size().height));
}

bool TabBar::_is_point_primary_before_or_at_mid(const Point2 &p_point, const Rect2 &p_rect) const {
	if (vertical) { /* VERTICAL */
		return p_point.y <= p_rect.position.y + p_rect.size.height / 2;
	}
	return is_layout_rtl() != (p_point.x <= p_rect.position.x + p_rect.size.width / 2);
}

bool TabBar::_is_point_primary_after_mid(const Point2 &p_point, const Rect2 &p_rect) const {
	if (vertical) { /* VERTICAL */
		return p_point.y > p_rect.position.y + p_rect.size.height / 2;
	}
	return is_layout_rtl() != (p_point.x > p_rect.position.x + p_rect.size.width / 2);
}

bool TabBar::_is_point_before_first_tab(const Point2 &p_point) const {
	if (tabs.is_empty()) {
		return true;
	}

	const Rect2 first_tab_rect = get_tab_rect(0);
	if (vertical) { /* VERTICAL */
		return p_point.y < first_tab_rect.position.y;
	}
	return is_layout_rtl() != (p_point.x < first_tab_rect.position.x);
}

TabBar::TabMetrics TabBar::_get_tab_metrics(int p_idx, bool p_for_minimum_size) const {
	ERR_FAIL_INDEX_V(p_idx, tabs.size(), TabMetrics());

	TabMetrics metrics;

	Ref<StyleBox> style;
	if (tabs[p_idx].disabled) {
		style = theme_cache.tab_disabled_style;
	} else if (current == p_idx) {
		style = theme_cache.tab_selected_style;
	} else if (!p_for_minimum_size && theme_cache.tab_hovered_style->get_minimum_size().width > theme_cache.tab_unselected_style->get_minimum_size().width) {
		style = theme_cache.tab_hovered_style;
	} else {
		style = theme_cache.tab_unselected_style;
	}

	int row_width = style->get_minimum_size().width;
	const int row_width_base = row_width;

	const bool close_visible = cb_displaypolicy == CLOSE_BUTTON_SHOW_ALWAYS || (cb_displaypolicy == CLOSE_BUTTON_SHOW_ACTIVE_ONLY && p_idx == current);
	if (tab_text_rotation != TAB_TEXT_ROTATION_NONE) {
		// Stacked layout: items are arranged vertically, text is rotated 90 degrees.
		// Width is the widest item, height is the sum of all items.
		int content_w = 0;
		int content_h = 0;
		int visible_items = 0;

		if (tabs[p_idx].icon.is_valid()) {
			const Size2 icon_size = _get_tab_icon_size(p_idx);
			content_w = MAX(content_w, int(icon_size.width));
			content_h += int(icon_size.height);
			visible_items++;
		}

		if (!tabs[p_idx].text.is_empty()) {
			// Rotated 90 degrees: text height becomes width, text width becomes height.
			content_w = MAX(content_w, int(tabs[p_idx].text_buf->get_size().y));
			content_h += tabs[p_idx].size_text;
			visible_items++;
		}

		if (tabs[p_idx].right_button.is_valid()) {
			Ref<Texture2D> rb = tabs[p_idx].right_button;
			int rb_w = theme_cache.button_hl_style->get_minimum_size().width + rb->get_width();
			int rb_h = theme_cache.button_hl_style->get_minimum_size().height + rb->get_height();
			content_w = MAX(content_w, rb_w);
			content_h += rb_h;
			visible_items++;
		}

		if (close_visible) {
			int cb_w = theme_cache.button_hl_style->get_minimum_size().width + theme_cache.close_icon->get_width();
			int cb_h = theme_cache.button_hl_style->get_minimum_size().height + theme_cache.close_icon->get_height();
			content_w = MAX(content_w, cb_w);
			content_h += cb_h;
			visible_items++;
		}

		if (visible_items > 1) {
			content_h += (visible_items - 1) * theme_cache.h_separation;
		}

		row_width += content_w;

		const int y_margin = MAX(MAX(MAX(theme_cache.tab_unselected_style->get_minimum_size().height, theme_cache.tab_hovered_style->get_minimum_size().height), theme_cache.tab_selected_style->get_minimum_size().height), theme_cache.tab_disabled_style->get_minimum_size().height);
		int row_height = y_margin + content_h;

		metrics.row_width = row_width;
		metrics.row_height = row_height;
		metrics.layout_size = vertical ? row_height : row_width;
		return metrics;
	}

	if (tabs[p_idx].icon.is_valid()) {
		const Size2 icon_size = _get_tab_icon_size(p_idx);
		row_width += icon_size.width + theme_cache.h_separation;
	}

	if (!tabs[p_idx].text.is_empty()) {
		row_width += tabs[p_idx].size_text + theme_cache.h_separation;
	}

	if (tabs[p_idx].right_button.is_valid()) {
		Ref<Texture2D> rb = tabs[p_idx].right_button;
		if (close_visible) {
			row_width += theme_cache.button_hl_style->get_minimum_size().width + rb->get_width();
		} else {
			row_width += theme_cache.button_hl_style->get_margin(SIDE_LEFT) + rb->get_width() + theme_cache.h_separation;
		}
	}

	if (close_visible) {
		row_width += theme_cache.button_hl_style->get_margin(SIDE_LEFT) + theme_cache.close_icon->get_width() + theme_cache.h_separation;
	}

	if (row_width - row_width_base > style->get_minimum_size().width) {
		row_width -= theme_cache.h_separation;
	}

	const int y_margin = MAX(MAX(MAX(theme_cache.tab_unselected_style->get_minimum_size().height, theme_cache.tab_hovered_style->get_minimum_size().height), theme_cache.tab_selected_style->get_minimum_size().height), theme_cache.tab_disabled_style->get_minimum_size().height);
	int row_height = y_margin;

	if (tabs[p_idx].icon.is_valid()) {
		const Size2 icon_size = _get_tab_icon_size(p_idx);
		row_height = MAX(row_height, icon_size.height + y_margin);
	}

	if (!tabs[p_idx].text.is_empty()) {
		row_height = MAX(row_height, int(tabs[p_idx].text_buf->get_size().y) + y_margin);
	}

	if (tabs[p_idx].right_button.is_valid()) {
		row_height = MAX(row_height, tabs[p_idx].right_button->get_height() + y_margin);
	}

	if (close_visible) {
		row_height = MAX(row_height, theme_cache.close_icon->get_height() + y_margin);
	}

	metrics.row_width = row_width;
	metrics.row_height = row_height;
	metrics.layout_size = vertical ? row_height : row_width;
	return metrics;
}
int TabBar::get_tab_width(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, tabs.size(), 0);
	return _get_tab_metrics(p_idx, false).layout_size;
}

Size2 TabBar::_get_tab_icon_size(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, tabs.size(), Size2());
	const TabBar::Tab &tab = tabs[p_index];
	Size2 icon_size;
	if (tab.icon.is_valid()) {
		icon_size = tab.icon->get_size();
	} else {
		icon_size = Size2();
	}

	int icon_max_width = 0;
	if (theme_cache.icon_max_width > 0) {
		icon_max_width = theme_cache.icon_max_width;
	}
	if (tab.icon_max_width > 0 && (icon_max_width == 0 || tab.icon_max_width < icon_max_width)) {
		icon_max_width = tab.icon_max_width;
	}

	if (icon_max_width > 0 && icon_size.width > icon_max_width) {
		icon_size.height = icon_size.height * icon_max_width / icon_size.width;
		icon_size.width = icon_max_width;
	}

	return icon_size;
}

void TabBar::_ensure_no_over_offset() {
	if (!is_inside_tree() || !buttons_visible) {
		return;
	}

	int limit_with_buttons;
	int limit_with_no_button;
	const int primary_size = vertical ? get_size().height : get_size().width;
	limit_with_buttons = _get_primary_limit_minus_buttons(primary_size);
	limit_with_no_button = primary_size;
	int offset_with_buttons = offset;
	int offset_with_no_button = offset;

	int total_w = tabs[max_drawn_tab].ofs_cache + tabs[max_drawn_tab].size_cache - tabs[offset].ofs_cache;
	for (int i = offset - 1; i >= 0; i--) {
		if (!tabs[i].hidden) {
			total_w += tabs[i].size_cache;
		}

		if (total_w < limit_with_buttons) {
			offset_with_buttons--;
			offset_with_no_button--;
		} else if (total_w < limit_with_no_button) {
			offset_with_no_button--;
		} else {
			break;
		}
	}

	int new_offset = (offset_with_no_button == 0) ? 0 : offset_with_buttons;

	if (new_offset != offset) {
		offset = new_offset;
		_update_cache();
		queue_redraw();
	}
}

bool TabBar::_can_deselect() const {
	if (deselect_enabled) {
		return true;
	}
	// All tabs must be disabled or hidden.
	for (const Tab &tab : tabs) {
		if (!tab.disabled && !tab.hidden) {
			return false;
		}
	}
	return true;
}

void TabBar::ensure_tab_visible(int p_idx) {
	if (p_idx == -1 || !is_inside_tree() || !buttons_visible) {
		return;
	}
	ERR_FAIL_INDEX(p_idx, tabs.size());

	if (tabs[p_idx].hidden || (p_idx >= offset && p_idx <= max_drawn_tab)) {
		return;
	}

	if (p_idx < offset) {
		offset = p_idx;
		_update_cache();
		queue_redraw();

		return;
	}

	int limit_minus_buttons;
	const int primary_size = vertical ? get_size().height : get_size().width;
	limit_minus_buttons = _get_primary_limit_minus_buttons(primary_size);

	int total_w = tabs[max_drawn_tab].ofs_cache - tabs[offset].ofs_cache;
	for (int i = max_drawn_tab; i <= p_idx; i++) {
		if (tabs[i].hidden) {
			continue;
		}

		total_w += tabs[i].size_cache;
	}

	int prev_offset = offset;

	for (int i = offset; i < p_idx; i++) {
		if (tabs[i].hidden) {
			continue;
		}

		if (total_w > limit_minus_buttons) {
			total_w -= tabs[i].size_cache;
			offset++;
		} else {
			break;
		}
	}

	if (prev_offset != offset) {
		_update_cache();
		queue_redraw();
	}
}

Rect2 TabBar::get_tab_rect(int p_tab) const {
	if (p_tab < 0) {
		p_tab += tabs.size();
	}
	ERR_FAIL_INDEX_V(p_tab, tabs.size(), Rect2());

	if (vertical) { /* VERTICAL */
		const Rect2 content_rect = _get_tabs_content_rect();
		return Rect2(content_rect.position.x, tabs[p_tab].ofs_cache, content_rect.size.x, tabs[p_tab].size_cache);
	} else { /* HORIZONTAL */
		if (is_layout_rtl()) {
			return Rect2(get_size().width - tabs[p_tab].ofs_cache - tabs[p_tab].size_cache, 0, tabs[p_tab].size_cache, get_size().height);
		} else {
			return Rect2(tabs[p_tab].ofs_cache, 0, tabs[p_tab].size_cache, get_size().height);
		}
	}
}

void TabBar::set_close_with_middle_mouse(bool p_scroll_close) {
	close_with_middle_mouse = p_scroll_close;
}

bool TabBar::get_close_with_middle_mouse() const {
	return close_with_middle_mouse;
}

void TabBar::set_tab_close_display_policy(CloseButtonDisplayPolicy p_policy) {
	ERR_FAIL_INDEX(p_policy, CLOSE_BUTTON_MAX);

	if (cb_displaypolicy == p_policy) {
		return;
	}

	cb_displaypolicy = p_policy;

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

TabBar::CloseButtonDisplayPolicy TabBar::get_tab_close_display_policy() const {
	return cb_displaypolicy;
}

void TabBar::set_max_tab_width(int p_width) {
	ERR_FAIL_COND(p_width < 0);

	if (max_width == p_width) {
		return;
	}

	max_width = p_width;

	_update_cache();
	_ensure_no_over_offset();
	if (scroll_to_selected) {
		ensure_tab_visible(current);
	}
	queue_redraw();
	update_desired_size();
	update_minimum_size();
}

int TabBar::get_max_tab_width() const {
	return max_width;
}

void TabBar::set_scrolling_enabled(bool p_enabled) {
	scrolling_enabled = p_enabled;
}

bool TabBar::get_scrolling_enabled() const {
	return scrolling_enabled;
}

void TabBar::set_drag_to_rearrange_enabled(bool p_enabled) {
	drag_to_rearrange_enabled = p_enabled;
}

bool TabBar::get_drag_to_rearrange_enabled() const {
	return drag_to_rearrange_enabled;
}

void TabBar::set_tabs_rearrange_group(int p_group_id) {
	tabs_rearrange_group = p_group_id;
}

int TabBar::get_tabs_rearrange_group() const {
	return tabs_rearrange_group;
}

void TabBar::set_scroll_to_selected(bool p_enabled) {
	scroll_to_selected = p_enabled;
	if (p_enabled) {
		ensure_tab_visible(current);
	}
}

bool TabBar::get_scroll_to_selected() const {
	return scroll_to_selected;
}

void TabBar::set_switch_on_drag_hover(bool p_enabled) {
	switch_on_drag_hover = p_enabled;
}

bool TabBar::get_switch_on_drag_hover() const {
	return switch_on_drag_hover;
}

void TabBar::set_select_with_rmb(bool p_enabled) {
	select_with_rmb = p_enabled;
}

bool TabBar::get_select_with_rmb() const {
	return select_with_rmb;
}

void TabBar::set_deselect_enabled(bool p_enabled) {
	if (deselect_enabled == p_enabled) {
		return;
	}
	deselect_enabled = p_enabled;
	if (!deselect_enabled && current == -1 && !tabs.is_empty()) {
		select_next_available();
	}
}

bool TabBar::get_deselect_enabled() const {
	return deselect_enabled;
}

void TabBar::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tab_count", "count"), &TabBar::set_tab_count);
	ClassDB::bind_method(D_METHOD("get_tab_count"), &TabBar::get_tab_count);
	ClassDB::bind_method(D_METHOD("set_current_tab", "tab_idx"), &TabBar::set_current_tab);
	ClassDB::bind_method(D_METHOD("get_current_tab"), &TabBar::get_current_tab);
	ClassDB::bind_method(D_METHOD("get_previous_tab"), &TabBar::get_previous_tab);
	ClassDB::bind_method(D_METHOD("select_previous_available"), &TabBar::select_previous_available);
	ClassDB::bind_method(D_METHOD("select_next_available"), &TabBar::select_next_available);
	ClassDB::bind_method(D_METHOD("set_tab_title", "tab_idx", "title"), &TabBar::set_tab_title);
	ClassDB::bind_method(D_METHOD("get_tab_title", "tab_idx"), &TabBar::get_tab_title);
	ClassDB::bind_method(D_METHOD("set_tab_tooltip", "tab_idx", "tooltip"), &TabBar::set_tab_tooltip);
	ClassDB::bind_method(D_METHOD("get_tab_tooltip", "tab_idx"), &TabBar::get_tab_tooltip);
	ClassDB::bind_method(D_METHOD("set_tab_text_direction", "tab_idx", "direction"), &TabBar::set_tab_text_direction);
	ClassDB::bind_method(D_METHOD("get_tab_text_direction", "tab_idx"), &TabBar::get_tab_text_direction);
	ClassDB::bind_method(D_METHOD("set_tab_language", "tab_idx", "language"), &TabBar::set_tab_language);
	ClassDB::bind_method(D_METHOD("get_tab_language", "tab_idx"), &TabBar::get_tab_language);
	ClassDB::bind_method(D_METHOD("set_tab_icon", "tab_idx", "icon"), &TabBar::set_tab_icon);
	ClassDB::bind_method(D_METHOD("get_tab_icon", "tab_idx"), &TabBar::get_tab_icon);
	ClassDB::bind_method(D_METHOD("set_tab_icon_max_width", "tab_idx", "width"), &TabBar::set_tab_icon_max_width);
	ClassDB::bind_method(D_METHOD("get_tab_icon_max_width", "tab_idx"), &TabBar::get_tab_icon_max_width);
	ClassDB::bind_method(D_METHOD("set_tab_button_icon", "tab_idx", "icon"), &TabBar::set_tab_button_icon);
	ClassDB::bind_method(D_METHOD("get_tab_button_icon", "tab_idx"), &TabBar::get_tab_button_icon);
	ClassDB::bind_method(D_METHOD("set_tab_disabled", "tab_idx", "disabled"), &TabBar::set_tab_disabled);
	ClassDB::bind_method(D_METHOD("is_tab_disabled", "tab_idx"), &TabBar::is_tab_disabled);
	ClassDB::bind_method(D_METHOD("set_tab_hidden", "tab_idx", "hidden"), &TabBar::set_tab_hidden);
	ClassDB::bind_method(D_METHOD("is_tab_hidden", "tab_idx"), &TabBar::is_tab_hidden);
	ClassDB::bind_method(D_METHOD("set_tab_metadata", "tab_idx", "metadata"), &TabBar::set_tab_metadata);
	ClassDB::bind_method(D_METHOD("get_tab_metadata", "tab_idx"), &TabBar::get_tab_metadata);
	ClassDB::bind_method(D_METHOD("remove_tab", "tab_idx"), &TabBar::remove_tab);
	ClassDB::bind_method(D_METHOD("add_tab", "title", "icon"), &TabBar::add_tab, DEFVAL(""), DEFVAL(Ref<Texture2D>()));
	ClassDB::bind_method(D_METHOD("get_tab_idx_at_point", "point"), &TabBar::get_tab_idx_at_point);
	ClassDB::bind_method(D_METHOD("set_tab_alignment", "alignment"), &TabBar::set_tab_alignment);
	ClassDB::bind_method(D_METHOD("get_tab_alignment"), &TabBar::get_tab_alignment);
	ClassDB::bind_method(D_METHOD("set_tab_sizing", "tab_sizing"), &TabBar::set_tab_sizing);
	ClassDB::bind_method(D_METHOD("get_tab_sizing"), &TabBar::get_tab_sizing);
	ClassDB::bind_method(D_METHOD("set_clip_tabs", "clip_tabs"), &TabBar::set_clip_tabs);
	ClassDB::bind_method(D_METHOD("get_clip_tabs"), &TabBar::get_clip_tabs);
#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("set_tab_style_v_flip", "tab_style_v_flip"), &TabBar::set_tab_style_v_flip);
#endif
	ClassDB::bind_method(D_METHOD("set_tab_style_side", "side"), &TabBar::set_tab_style_side);
	ClassDB::bind_method(D_METHOD("get_tab_style_side"), &TabBar::get_tab_style_side);
	ClassDB::bind_method(D_METHOD("set_vertical", "vertical"), &TabBar::set_vertical);
	ClassDB::bind_method(D_METHOD("is_vertical"), &TabBar::is_vertical);
	ClassDB::bind_method(D_METHOD("set_tab_text_rotation", "rotation"), &TabBar::set_tab_text_rotation);
	ClassDB::bind_method(D_METHOD("get_tab_text_rotation"), &TabBar::get_tab_text_rotation);
	ClassDB::bind_method(D_METHOD("get_tab_offset"), &TabBar::get_tab_offset);
	ClassDB::bind_method(D_METHOD("get_offset_buttons_visible"), &TabBar::get_offset_buttons_visible);
	ClassDB::bind_method(D_METHOD("ensure_tab_visible", "idx"), &TabBar::ensure_tab_visible);
	ClassDB::bind_method(D_METHOD("get_tab_rect", "tab_idx"), &TabBar::get_tab_rect);
	ClassDB::bind_method(D_METHOD("move_tab", "from", "to"), &TabBar::move_tab);
	ClassDB::bind_method(D_METHOD("set_close_with_middle_mouse", "enabled"), &TabBar::set_close_with_middle_mouse);
	ClassDB::bind_method(D_METHOD("get_close_with_middle_mouse"), &TabBar::get_close_with_middle_mouse);
	ClassDB::bind_method(D_METHOD("set_tab_close_display_policy", "policy"), &TabBar::set_tab_close_display_policy);
	ClassDB::bind_method(D_METHOD("get_tab_close_display_policy"), &TabBar::get_tab_close_display_policy);
	ClassDB::bind_method(D_METHOD("set_max_tab_width", "width"), &TabBar::set_max_tab_width);
	ClassDB::bind_method(D_METHOD("get_max_tab_width"), &TabBar::get_max_tab_width);
	ClassDB::bind_method(D_METHOD("set_scrolling_enabled", "enabled"), &TabBar::set_scrolling_enabled);
	ClassDB::bind_method(D_METHOD("get_scrolling_enabled"), &TabBar::get_scrolling_enabled);
	ClassDB::bind_method(D_METHOD("set_drag_to_rearrange_enabled", "enabled"), &TabBar::set_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("get_drag_to_rearrange_enabled"), &TabBar::get_drag_to_rearrange_enabled);
	ClassDB::bind_method(D_METHOD("set_switch_on_drag_hover", "enabled"), &TabBar::set_switch_on_drag_hover);
	ClassDB::bind_method(D_METHOD("get_switch_on_drag_hover"), &TabBar::get_switch_on_drag_hover);
	ClassDB::bind_method(D_METHOD("set_tabs_rearrange_group", "group_id"), &TabBar::set_tabs_rearrange_group);
	ClassDB::bind_method(D_METHOD("get_tabs_rearrange_group"), &TabBar::get_tabs_rearrange_group);
	ClassDB::bind_method(D_METHOD("set_scroll_to_selected", "enabled"), &TabBar::set_scroll_to_selected);
	ClassDB::bind_method(D_METHOD("get_scroll_to_selected"), &TabBar::get_scroll_to_selected);
	ClassDB::bind_method(D_METHOD("set_select_with_rmb", "enabled"), &TabBar::set_select_with_rmb);
	ClassDB::bind_method(D_METHOD("get_select_with_rmb"), &TabBar::get_select_with_rmb);
	ClassDB::bind_method(D_METHOD("set_deselect_enabled", "enabled"), &TabBar::set_deselect_enabled);
	ClassDB::bind_method(D_METHOD("get_deselect_enabled"), &TabBar::get_deselect_enabled);
	ClassDB::bind_method(D_METHOD("clear_tabs"), &TabBar::clear_tabs);

	ADD_SIGNAL(MethodInfo("tab_selected", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_changed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_clicked", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_rmb_clicked", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_close_pressed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_button_pressed", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("tab_hovered", PropertyInfo(Variant::INT, "tab")));
	ADD_SIGNAL(MethodInfo("active_tab_rearranged", PropertyInfo(Variant::INT, "idx_to")));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_tab", PROPERTY_HINT_RANGE, "-1,4096,1"), "set_current_tab", "get_current_tab");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_alignment", PROPERTY_HINT_ENUM, "Begin,Center,End"), "set_tab_alignment", "get_tab_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_sizing", PROPERTY_HINT_ENUM, "Fit Content,Uniform,Justify,Expand"), "set_tab_sizing", "get_tab_sizing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertical"), "set_vertical", "is_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_text_rotation", PROPERTY_HINT_ENUM, "None,Clockwise,Counter-Clockwise"), "set_tab_text_rotation", "get_tab_text_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_style_side", PROPERTY_HINT_ENUM, "Top,Bottom,Left,Right"), "set_tab_style_side", "get_tab_style_side");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_tabs"), "set_clip_tabs", "get_clip_tabs");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "close_with_middle_mouse"), "set_close_with_middle_mouse", "get_close_with_middle_mouse");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tab_close_display_policy", PROPERTY_HINT_ENUM, "Show Never,Show Active Only,Show Always"), "set_tab_close_display_policy", "get_tab_close_display_policy");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tab_width", PROPERTY_HINT_RANGE, "0,99999,1,suffix:px"), "set_max_tab_width", "get_max_tab_width");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scrolling_enabled"), "set_scrolling_enabled", "get_scrolling_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_to_rearrange_enabled"), "set_drag_to_rearrange_enabled", "get_drag_to_rearrange_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "switch_on_drag_hover"), "set_switch_on_drag_hover", "get_switch_on_drag_hover");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tabs_rearrange_group"), "set_tabs_rearrange_group", "get_tabs_rearrange_group");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "scroll_to_selected"), "set_scroll_to_selected", "get_scroll_to_selected");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "select_with_rmb"), "set_select_with_rmb", "get_select_with_rmb");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deselect_enabled"), "set_deselect_enabled", "get_deselect_enabled");

	ADD_ARRAY_COUNT("Tabs", "tab_count", "set_tab_count", "get_tab_count", "tab_");

	BIND_ENUM_CONSTANT(ALIGNMENT_BEGIN);
	BIND_ENUM_CONSTANT(ALIGNMENT_CENTER);
	BIND_ENUM_CONSTANT(ALIGNMENT_END);
	BIND_ENUM_CONSTANT(ALIGNMENT_MAX);
#ifndef DISABLE_DEPRECATED
	BIND_ENUM_CONSTANT(ALIGNMENT_LEFT);
	BIND_ENUM_CONSTANT(ALIGNMENT_RIGHT);
#endif

	BIND_ENUM_CONSTANT(TAB_SIZING_FIT_CONTENT);
	BIND_ENUM_CONSTANT(TAB_SIZING_UNIFORM);
	BIND_ENUM_CONSTANT(TAB_SIZING_JUSTIFY);
	BIND_ENUM_CONSTANT(TAB_SIZING_EXPAND);
	BIND_ENUM_CONSTANT(TAB_SIZING_MAX);

	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_NEVER);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_ACTIVE_ONLY);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_SHOW_ALWAYS);
	BIND_ENUM_CONSTANT(CLOSE_BUTTON_MAX);

	BIND_ENUM_CONSTANT(TAB_TEXT_ROTATION_NONE);
	BIND_ENUM_CONSTANT(TAB_TEXT_ROTATION_CLOCKWISE);
	BIND_ENUM_CONSTANT(TAB_TEXT_ROTATION_COUNTER_CLOCKWISE);
	BIND_ENUM_CONSTANT(TAB_TEXT_ROTATION_MAX);

	BIND_ENUM_CONSTANT(TAB_STYLE_SIDE_TOP);
	BIND_ENUM_CONSTANT(TAB_STYLE_SIDE_BOTTOM);
	BIND_ENUM_CONSTANT(TAB_STYLE_SIDE_LEFT);
	BIND_ENUM_CONSTANT(TAB_STYLE_SIDE_RIGHT);
	BIND_ENUM_CONSTANT(TAB_STYLE_SIDE_MAX);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, TabBar, h_separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, TabBar, tab_separation);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, TabBar, icon_max_width);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, TabBar, hover_switch_wait_msec);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, TabBar, tab_unselected_style, "tab_unselected");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, TabBar, tab_hovered_style, "tab_hovered");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, TabBar, tab_selected_style, "tab_selected");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, TabBar, tab_disabled_style, "tab_disabled");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, TabBar, tab_focus_style, "tab_focus");

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, increment_icon, "increment");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, increment_hl_icon, "increment_highlight");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, decrement_icon, "decrement");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, decrement_hl_icon, "decrement_highlight");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, increment_vertical_icon, "increment_vertical");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, increment_vertical_hl_icon, "increment_vertical_highlight");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, decrement_vertical_icon, "decrement_vertical");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, decrement_vertical_hl_icon, "decrement_vertical_highlight");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, drop_mark_icon, "drop_mark");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, vertical_drop_mark_icon, "vertical_drop_mark");
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, drop_mark_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, font_selected_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, font_hovered_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, font_unselected_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, font_disabled_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, font_outline_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, icon_selected_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, icon_hovered_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, icon_unselected_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, TabBar, icon_disabled_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, TabBar, font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, TabBar, font_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, TabBar, outline_size);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, TabBar, close_icon, "close");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, TabBar, button_pressed_style, "button_pressed");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_STYLEBOX, TabBar, button_hl_style, "button_highlight");

	Tab defaults(true);

	base_property_helper.set_prefix("tab_");
	base_property_helper.set_array_length_getter(&TabBar::get_tab_count);
	base_property_helper.register_property(PropertyInfo(Variant::STRING, "title"), defaults.text, &TabBar::set_tab_title, &TabBar::get_tab_title);
	base_property_helper.register_property(PropertyInfo(Variant::STRING, "tooltip"), defaults.tooltip, &TabBar::set_tab_tooltip, &TabBar::get_tab_tooltip);
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, Texture2D::get_class_static()), defaults.icon, &TabBar::set_tab_icon, &TabBar::get_tab_icon);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "disabled"), defaults.disabled, &TabBar::set_tab_disabled, &TabBar::is_tab_disabled);
	PropertyListHelper::register_base_helper(get_class_static(), &base_property_helper);
}

TabBar::TabBar() {
	set_focus_mode(FOCUS_ACCESSIBILITY);
	set_size(Size2(get_size().width, get_minimum_size().height));
	set_focus_mode(FOCUS_ALL);
	connect(SceneStringName(mouse_exited), callable_mp(this, &TabBar::_on_mouse_exited));
	connect(SceneStringName(maximum_size_changed), callable_mp(this, &TabBar::_on_maximum_size_changed));

	hover_switch_delay = memnew(Timer);
	hover_switch_delay->connect("timeout", callable_mp(this, &TabBar::_hover_switch_timeout));
	hover_switch_delay->set_one_shot(true);
	add_child(hover_switch_delay, false, INTERNAL_MODE_FRONT);

	property_helper.setup_for_instance(base_property_helper, this);
}
