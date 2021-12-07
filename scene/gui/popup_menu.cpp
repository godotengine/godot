/*************************************************************************/
/*  popup_menu.cpp                                                       */
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

#include "popup_menu.h"

#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"

String PopupMenu::_get_accel_text(const Item &p_item) const {
	if (p_item.shortcut.is_valid()) {
		return p_item.shortcut->get_as_text();
	} else if (p_item.accel != Key::NONE) {
		return keycode_get_string(p_item.accel);
	}
	return String();
}

Size2 PopupMenu::_get_contents_minimum_size() const {
	int vseparation = get_theme_constant(SNAME("vseparation"));
	int hseparation = get_theme_constant(SNAME("hseparation"));

	Size2 minsize = get_theme_stylebox(SNAME("panel"))->get_minimum_size(); // Accounts for margin in the margin container
	minsize.x += scroll_container->get_v_scrollbar()->get_size().width * 2; // Adds a buffer so that the scrollbar does not render over the top of content

	float max_w = 0.0;
	float icon_w = 0.0;
	int check_w = MAX(get_theme_icon(SNAME("checked"))->get_width(), get_theme_icon(SNAME("radio_checked"))->get_width()) + hseparation;
	int accel_max_w = 0;
	bool has_check = false;

	for (int i = 0; i < items.size(); i++) {
		Size2 size;

		Size2 icon_size = items[i].get_icon_size();
		size.height = _get_item_height(i);
		icon_w = MAX(icon_size.width, icon_w);

		size.width += items[i].h_ofs;

		if (items[i].checkable_type) {
			has_check = true;
		}

		size.width += items[i].text_buf->get_size().x;
		size.height += vseparation;

		if (items[i].accel != Key::NONE || (items[i].shortcut.is_valid() && items[i].shortcut->has_valid_event())) {
			int accel_w = hseparation * 2;
			accel_w += items[i].accel_text_buf->get_size().x;
			accel_max_w = MAX(accel_w, accel_max_w);
		}

		if (items[i].submenu != "") {
			size.width += get_theme_icon(SNAME("submenu"))->get_width();
		}

		max_w = MAX(max_w, size.width);

		minsize.height += size.height;
	}

	int item_side_padding = get_theme_constant(SNAME("item_start_padding")) + get_theme_constant(SNAME("item_end_padding"));
	minsize.width += max_w + icon_w + accel_max_w + item_side_padding;

	if (has_check) {
		minsize.width += check_w;
	}

	if (is_inside_tree()) {
		int height_limit = get_usable_parent_rect().size.height;
		if (minsize.height > height_limit) {
			minsize.height = height_limit;
		}
	}

	return minsize;
}

int PopupMenu::_get_item_height(int p_item) const {
	ERR_FAIL_INDEX_V(p_item, items.size(), 0);
	ERR_FAIL_COND_V(p_item < 0, 0);

	int icon_height = items[p_item].get_icon_size().height;
	if (items[p_item].checkable_type) {
		icon_height = MAX(icon_height, MAX(get_theme_icon(SNAME("checked"))->get_height(), get_theme_icon(SNAME("radio_checked"))->get_height()));
	}

	int text_height = items[p_item].text_buf->get_size().height;
	if (text_height == 0 && !items[p_item].separator) {
		text_height = get_theme_font(SNAME("font"))->get_height(get_theme_font_size(SNAME("font_size")));
	}

	int separator_height = 0;
	if (items[p_item].separator) {
		separator_height = MAX(get_theme_stylebox(SNAME("separator"))->get_minimum_size().height, MAX(get_theme_stylebox(SNAME("labeled_separator_left"))->get_minimum_size().height, get_theme_stylebox(SNAME("labeled_separator_right"))->get_minimum_size().height));
	}

	return MAX(separator_height, MAX(text_height, icon_height));
}

int PopupMenu::_get_items_total_height() const {
	int vsep = get_theme_constant(SNAME("vseparation"));

	// Get total height of all items by taking max of icon height and font height
	int items_total_height = 0;
	for (int i = 0; i < items.size(); i++) {
		items_total_height += _get_item_height(i) + vsep;
	}

	// Subtract a separator which is not needed for the last item.
	return items_total_height - vsep;
}

void PopupMenu::_scroll_to_item(int p_item) {
	ERR_FAIL_INDEX(p_item, items.size());
	ERR_FAIL_COND(p_item < 0);

	// Scroll item into view (upwards)
	if (items[p_item]._ofs_cache < -control->get_position().y) {
		int amnt_over = items[p_item]._ofs_cache + control->get_position().y;
		scroll_container->set_v_scroll(scroll_container->get_v_scroll() + amnt_over);
	}

	// Scroll item into view (downwards)
	if (items[p_item]._ofs_cache + items[p_item]._height_cache > -control->get_position().y + scroll_container->get_size().height) {
		int amnt_over = items[p_item]._ofs_cache + items[p_item]._height_cache + control->get_position().y - scroll_container->get_size().height;
		scroll_container->set_v_scroll(scroll_container->get_v_scroll() + amnt_over);
	}
}

int PopupMenu::_get_mouse_over(const Point2 &p_over) const {
	if (p_over.x < 0 || p_over.x >= get_size().width) {
		return -1;
	}

	Ref<StyleBox> style = get_theme_stylebox(SNAME("panel")); // Accounts for margin in the margin container

	int vseparation = get_theme_constant(SNAME("vseparation"));

	Point2 ofs = style->get_offset() + Point2(0, vseparation / 2);

	if (ofs.y > p_over.y) {
		return -1;
	}

	for (int i = 0; i < items.size(); i++) {
		ofs.y += i > 0 ? vseparation : (float)vseparation / 2;

		ofs.y += _get_item_height(i);

		if (p_over.y - control->get_position().y < ofs.y) {
			return i;
		}
	}

	return -1;
}

void PopupMenu::_activate_submenu(int p_over) {
	Node *n = get_node(items[p_over].submenu);
	ERR_FAIL_COND_MSG(!n, "Item subnode does not exist: " + items[p_over].submenu + ".");
	Popup *submenu_popup = Object::cast_to<Popup>(n);
	ERR_FAIL_COND_MSG(!submenu_popup, "Item subnode is not a Popup: " + items[p_over].submenu + ".");
	if (submenu_popup->is_visible()) {
		return; //already visible!
	}

	Ref<StyleBox> style = get_theme_stylebox(SNAME("panel"));
	int vsep = get_theme_constant(SNAME("vseparation"));

	Point2 this_pos = get_position();
	Rect2 this_rect(this_pos, get_size());

	float scroll_offset = control->get_position().y;

	Point2 submenu_pos;
	Size2 submenu_size = submenu_popup->get_size();
	if (control->is_layout_rtl()) {
		submenu_pos = this_pos + Point2(-submenu_size.width, items[p_over]._ofs_cache + scroll_offset);
	} else {
		submenu_pos = this_pos + Point2(this_rect.size.width, items[p_over]._ofs_cache + scroll_offset);
	}

	// Fix pos if going outside parent rect
	if (submenu_pos.x < get_parent_rect().position.x) {
		submenu_pos.x = this_pos.x + submenu_size.width;
	}

	if (submenu_pos.x + submenu_size.width > get_parent_rect().position.x + get_parent_rect().size.width) {
		submenu_pos.x = this_pos.x - submenu_size.width;
	}

	submenu_popup->set_close_on_parent_focus(false);
	submenu_popup->set_position(submenu_pos);
	submenu_popup->set_as_minsize(); // Shrink the popup size to its contents.
	submenu_popup->popup();

	// Set autohide areas
	PopupMenu *submenu_pum = Object::cast_to<PopupMenu>(submenu_popup);
	if (submenu_pum) {
		submenu_pum->take_mouse_focus();
		// Make the position of the parent popup relative to submenu popup
		this_rect.position = this_rect.position - submenu_pum->get_position();

		// Autohide area above the submenu item
		submenu_pum->clear_autohide_areas();
		submenu_pum->add_autohide_area(Rect2(this_rect.position.x, this_rect.position.y, this_rect.size.x, items[p_over]._ofs_cache + scroll_offset + style->get_offset().height - vsep / 2));

		// If there is an area below the submenu item, add an autohide area there.
		if (items[p_over]._ofs_cache + items[p_over]._height_cache + scroll_offset <= control->get_size().height) {
			int from = items[p_over]._ofs_cache + items[p_over]._height_cache + scroll_offset + vsep / 2 + style->get_offset().height;
			submenu_pum->add_autohide_area(Rect2(this_rect.position.x, this_rect.position.y + from, this_rect.size.x, this_rect.size.y - from));
		}
	}
}

void PopupMenu::_submenu_timeout() {
	if (mouse_over == submenu_over) {
		_activate_submenu(mouse_over);
	}

	submenu_over = -1;
}

void PopupMenu::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (p_event->is_action("ui_down") && p_event->is_pressed()) {
		int search_from = mouse_over + 1;
		if (search_from >= items.size()) {
			search_from = 0;
		}

		bool match_found = false;
		for (int i = search_from; i < items.size(); i++) {
			if (!items[i].separator && !items[i].disabled) {
				mouse_over = i;
				emit_signal(SNAME("id_focused"), i);
				_scroll_to_item(i);
				control->update();
				set_input_as_handled();
				match_found = true;
				break;
			}
		}

		if (!match_found) {
			// If the last item is not selectable, try re-searching from the start.
			for (int i = 0; i < search_from; i++) {
				if (!items[i].separator && !items[i].disabled) {
					mouse_over = i;
					emit_signal(SNAME("id_focused"), i);
					_scroll_to_item(i);
					control->update();
					set_input_as_handled();
					break;
				}
			}
		}
	} else if (p_event->is_action("ui_up") && p_event->is_pressed()) {
		int search_from = mouse_over - 1;
		if (search_from < 0) {
			search_from = items.size() - 1;
		}

		bool match_found = false;
		for (int i = search_from; i >= 0; i--) {
			if (!items[i].separator && !items[i].disabled) {
				mouse_over = i;
				emit_signal(SNAME("id_focused"), i);
				_scroll_to_item(i);
				control->update();
				set_input_as_handled();
				match_found = true;
				break;
			}
		}

		if (!match_found) {
			// If the first item is not selectable, try re-searching from the end.
			for (int i = items.size() - 1; i >= search_from; i--) {
				if (!items[i].separator && !items[i].disabled) {
					mouse_over = i;
					emit_signal(SNAME("id_focused"), i);
					_scroll_to_item(i);
					control->update();
					set_input_as_handled();
					break;
				}
			}
		}
	} else if (p_event->is_action("ui_left") && p_event->is_pressed()) {
		Node *n = get_parent();
		if (n && Object::cast_to<PopupMenu>(n)) {
			hide();
			set_input_as_handled();
		}
	} else if (p_event->is_action("ui_right") && p_event->is_pressed()) {
		if (mouse_over >= 0 && mouse_over < items.size() && !items[mouse_over].separator && items[mouse_over].submenu != "" && submenu_over != mouse_over) {
			_activate_submenu(mouse_over);
			set_input_as_handled();
		}
	} else if (p_event->is_action("ui_accept") && p_event->is_pressed()) {
		if (mouse_over >= 0 && mouse_over < items.size() && !items[mouse_over].separator) {
			if (items[mouse_over].submenu != "" && submenu_over != mouse_over) {
				_activate_submenu(mouse_over);
			} else {
				activate_item(mouse_over);
			}
			set_input_as_handled();
		}
	}

	// Make an area which does not include v scrollbar, so that items are not activated when dragging scrollbar.
	Rect2 item_clickable_area = scroll_container->get_rect();
	if (scroll_container->get_v_scrollbar()->is_visible_in_tree()) {
		if (is_layout_rtl()) {
			item_clickable_area.position.x += scroll_container->get_v_scrollbar()->get_size().width;
		} else {
			item_clickable_area.size.width -= scroll_container->get_v_scrollbar()->get_size().width;
		}
	}

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (!item_clickable_area.has_point(b->get_position())) {
			return;
		}

		MouseButton button_idx = b->get_button_index();
		if (!b->is_pressed()) {
			// Activate the item on release of either the left mouse button or
			// any mouse button held down when the popup was opened.
			// This allows for opening the popup and triggering an action in a single mouse click.
			if (button_idx == MouseButton::LEFT || (initial_button_mask & mouse_button_to_mask(button_idx)) != MouseButton::NONE) {
				bool was_during_grabbed_click = during_grabbed_click;
				during_grabbed_click = false;
				initial_button_mask = MouseButton::NONE;

				// Disable clicks under a time threshold to avoid selection right when opening the popup.
				uint64_t now = OS::get_singleton()->get_ticks_msec();
				uint64_t diff = now - popup_time_msec;
				if (diff < 100) {
					return;
				}

				int over = _get_mouse_over(b->get_position());
				if (over < 0) {
					if (!was_during_grabbed_click) {
						hide();
					}
					return;
				}

				if (items[over].separator || items[over].disabled) {
					return;
				}

				if (items[over].submenu != "") {
					_activate_submenu(over);
					return;
				}
				activate_item(over);
			}
		}
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		for (const Rect2 &E : autohide_areas) {
			if (!Rect2(Point2(), get_size()).has_point(m->get_position()) && E.has_point(m->get_position())) {
				_close_pressed();
				return;
			}
		}

		if (!item_clickable_area.has_point(m->get_position())) {
			return;
		}

		int over = _get_mouse_over(m->get_position());
		int id = (over < 0 || items[over].separator || items[over].disabled) ? -1 : (items[over].id >= 0 ? items[over].id : over);

		if (id < 0) {
			mouse_over = -1;
			control->update();
			return;
		}

		if (items[over].submenu != "" && submenu_over != over) {
			submenu_over = over;
			submenu_timer->start();
		}

		if (over != mouse_over) {
			mouse_over = over;
			control->update();
		}
	}

	Ref<InputEventKey> k = p_event;

	if (allow_search && k.is_valid() && k->get_unicode() && k->is_pressed()) {
		uint64_t now = OS::get_singleton()->get_ticks_msec();
		uint64_t diff = now - search_time_msec;
		uint64_t max_interval = uint64_t(GLOBAL_DEF("gui/timers/incremental_search_max_interval_msec", 2000));
		search_time_msec = now;

		if (diff > max_interval) {
			search_string = "";
		}

		if (String::chr(k->get_unicode()) != search_string) {
			search_string += String::chr(k->get_unicode());
		}

		for (int i = mouse_over + 1; i <= items.size(); i++) {
			if (i == items.size()) {
				if (mouse_over <= 0) {
					break;
				} else {
					i = 0;
				}
			}

			if (i == mouse_over) {
				break;
			}

			if (items[i].text.findn(search_string) == 0) {
				mouse_over = i;
				emit_signal(SNAME("id_focused"), i);
				_scroll_to_item(i);
				control->update();
				set_input_as_handled();
				break;
			}
		}
	}
}

void PopupMenu::_draw_items() {
	control->set_custom_minimum_size(Size2(0, _get_items_total_height()));
	RID ci = control->get_canvas_item();

	Size2 margin_size;
	margin_size.width = margin_container->get_theme_constant(SNAME("margin_right")) + margin_container->get_theme_constant(SNAME("margin_left"));
	margin_size.height = margin_container->get_theme_constant(SNAME("margin_top")) + margin_container->get_theme_constant(SNAME("margin_bottom"));

	// Space between the item content and the sides of popup menu.
	int item_start_padding = get_theme_constant(SNAME("item_start_padding"));
	int item_end_padding = get_theme_constant(SNAME("item_end_padding"));

	bool rtl = control->is_layout_rtl();
	Ref<StyleBox> style = get_theme_stylebox(SNAME("panel"));
	Ref<StyleBox> hover = get_theme_stylebox(SNAME("hover"));
	// In Item::checkable_type enum order (less the non-checkable member)
	Ref<Texture2D> check[] = { get_theme_icon(SNAME("checked")), get_theme_icon(SNAME("radio_checked")) };
	Ref<Texture2D> uncheck[] = { get_theme_icon(SNAME("unchecked")), get_theme_icon(SNAME("radio_unchecked")) };
	Ref<Texture2D> submenu;
	if (rtl) {
		submenu = get_theme_icon(SNAME("submenu_mirrored"));
	} else {
		submenu = get_theme_icon(SNAME("submenu"));
	}

	Ref<StyleBox> separator = get_theme_stylebox(SNAME("separator"));
	Ref<StyleBox> labeled_separator_left = get_theme_stylebox(SNAME("labeled_separator_left"));
	Ref<StyleBox> labeled_separator_right = get_theme_stylebox(SNAME("labeled_separator_right"));

	int vseparation = get_theme_constant(SNAME("vseparation"));
	int hseparation = get_theme_constant(SNAME("hseparation"));
	Color font_color = get_theme_color(SNAME("font_color"));
	Color font_disabled_color = get_theme_color(SNAME("font_disabled_color"));
	Color font_accelerator_color = get_theme_color(SNAME("font_accelerator_color"));
	Color font_hover_color = get_theme_color(SNAME("font_hover_color"));
	Color font_separator_color = get_theme_color(SNAME("font_separator_color"));

	float scroll_width = scroll_container->get_v_scrollbar()->is_visible_in_tree() ? scroll_container->get_v_scrollbar()->get_size().width : 0;
	float display_width = control->get_size().width - scroll_width;

	// Find the widest icon and whether any items have a checkbox, and store the offsets for each.
	float icon_ofs = 0.0;
	bool has_check = false;
	for (int i = 0; i < items.size(); i++) {
		icon_ofs = MAX(items[i].get_icon_size().width, icon_ofs);

		if (items[i].checkable_type) {
			has_check = true;
		}
	}
	if (icon_ofs > 0.0) {
		icon_ofs += hseparation;
	}

	float check_ofs = 0.0;
	if (has_check) {
		check_ofs = MAX(get_theme_icon(SNAME("checked"))->get_width(), get_theme_icon(SNAME("radio_checked"))->get_width()) + hseparation;
	}

	Point2 ofs = Point2();

	// Loop through all items and draw each.
	for (int i = 0; i < items.size(); i++) {
		// For the first item only add half a separation. For all other items, add a whole separation to the offset.
		ofs.y += i > 0 ? vseparation : (float)vseparation / 2;

		_shape_item(i);

		Point2 item_ofs = ofs;
		Size2 icon_size = items[i].get_icon_size();
		float h = _get_item_height(i);

		if (i == mouse_over) {
			if (rtl) {
				hover->draw(ci, Rect2(item_ofs + Point2(scroll_width, -vseparation / 2), Size2(display_width, h + vseparation)));
			} else {
				hover->draw(ci, Rect2(item_ofs + Point2(0, -vseparation / 2), Size2(display_width, h + vseparation)));
			}
		}

		String text = items[i].xl_text;

		// Separator
		item_ofs.x += items[i].h_ofs;
		if (items[i].separator) {
			int sep_h = separator->get_center_size().height + separator->get_minimum_size().height;
			int sep_ofs = Math::floor((h - sep_h) / 2.0);
			if (text != String()) {
				int text_size = items[i].text_buf->get_size().width;
				int text_center = display_width / 2;
				int text_left = text_center - text_size / 2;
				int text_right = text_center + text_size / 2;
				if (text_left > item_ofs.x) {
					labeled_separator_left->draw(ci, Rect2(item_ofs + Point2(0, sep_ofs), Size2(MAX(0, text_left - item_ofs.x), sep_h)));
				}
				if (text_right < display_width) {
					labeled_separator_right->draw(ci, Rect2(Point2(text_right, item_ofs.y + sep_ofs), Size2(MAX(0, display_width - text_right), sep_h)));
				}
			} else {
				separator->draw(ci, Rect2(item_ofs + Point2(0, sep_ofs), Size2(display_width, sep_h)));
			}
		}

		Color icon_color(1, 1, 1, items[i].disabled ? 0.5 : 1);

		// For non-separator items, add some padding for the content.
		item_ofs.x += item_start_padding;

		// Checkboxes
		if (items[i].checkable_type) {
			Texture2D *icon = (items[i].checked ? check[items[i].checkable_type - 1] : uncheck[items[i].checkable_type - 1]).ptr();
			if (rtl) {
				icon->draw(ci, Size2(control->get_size().width - item_ofs.x - icon->get_width(), item_ofs.y) + Point2(0, Math::floor((h - icon->get_height()) / 2.0)), icon_color);
			} else {
				icon->draw(ci, item_ofs + Point2(0, Math::floor((h - icon->get_height()) / 2.0)), icon_color);
			}
		}

		// Icon
		if (!items[i].icon.is_null()) {
			if (rtl) {
				items[i].icon->draw(ci, Size2(control->get_size().width - item_ofs.x - check_ofs - icon_size.width, item_ofs.y) + Point2(0, Math::floor((h - icon_size.height) / 2.0)), icon_color);
			} else {
				items[i].icon->draw(ci, item_ofs + Size2(check_ofs, 0) + Point2(0, Math::floor((h - icon_size.height) / 2.0)), icon_color);
			}
		}

		// Submenu arrow on right hand side
		if (items[i].submenu != "") {
			if (rtl) {
				submenu->draw(ci, Point2(scroll_width + style->get_margin(SIDE_LEFT) + item_end_padding, item_ofs.y + Math::floor(h - submenu->get_height()) / 2), icon_color);
			} else {
				submenu->draw(ci, Point2(display_width - style->get_margin(SIDE_RIGHT) - submenu->get_width() - item_end_padding, item_ofs.y + Math::floor(h - submenu->get_height()) / 2), icon_color);
			}
		}

		// Text
		Color font_outline_color = get_theme_color(SNAME("font_outline_color"));
		int outline_size = get_theme_constant(SNAME("outline_size"));
		if (items[i].separator) {
			if (text != String()) {
				int center = (display_width - items[i].text_buf->get_size().width) / 2;
				Vector2 text_pos = Point2(center, item_ofs.y + Math::floor((h - items[i].text_buf->get_size().y) / 2.0));
				if (outline_size > 0 && font_outline_color.a > 0) {
					items[i].text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
				}
				items[i].text_buf->draw(ci, text_pos, font_separator_color);
			}
		} else {
			item_ofs.x += icon_ofs + check_ofs;
			if (rtl) {
				Vector2 text_pos = Size2(control->get_size().width - items[i].text_buf->get_size().width - item_ofs.x, item_ofs.y) + Point2(0, Math::floor((h - items[i].text_buf->get_size().y) / 2.0));
				if (outline_size > 0 && font_outline_color.a > 0) {
					items[i].text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
				}
				items[i].text_buf->draw(ci, text_pos, items[i].disabled ? font_disabled_color : (i == mouse_over ? font_hover_color : font_color));
			} else {
				Vector2 text_pos = item_ofs + Point2(0, Math::floor((h - items[i].text_buf->get_size().y) / 2.0));
				if (outline_size > 0 && font_outline_color.a > 0) {
					items[i].text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
				}
				items[i].text_buf->draw(ci, text_pos, items[i].disabled ? font_disabled_color : (i == mouse_over ? font_hover_color : font_color));
			}
		}

		// Accelerator / Shortcut
		if (items[i].accel != Key::NONE || (items[i].shortcut.is_valid() && items[i].shortcut->has_valid_event())) {
			if (rtl) {
				item_ofs.x = scroll_width + style->get_margin(SIDE_LEFT) + item_end_padding;
			} else {
				item_ofs.x = display_width - style->get_margin(SIDE_RIGHT) - items[i].accel_text_buf->get_size().x - item_end_padding;
			}
			Vector2 text_pos = item_ofs + Point2(0, Math::floor((h - items[i].text_buf->get_size().y) / 2.0));
			if (outline_size > 0 && font_outline_color.a > 0) {
				items[i].accel_text_buf->draw_outline(ci, text_pos, outline_size, font_outline_color);
			}
			items[i].accel_text_buf->draw(ci, text_pos, i == mouse_over ? font_hover_color : font_accelerator_color);
		}

		// Cache the item vertical offset from the first item and the height
		items.write[i]._ofs_cache = ofs.y;
		items.write[i]._height_cache = h;

		ofs.y += h;
	}
}

void PopupMenu::_draw_background() {
	Ref<StyleBox> style = get_theme_stylebox(SNAME("panel"));
	RID ci2 = margin_container->get_canvas_item();
	style->draw(ci2, Rect2(Point2(), margin_container->get_size()));
}

void PopupMenu::_minimum_lifetime_timeout() {
	close_allowed = true;
	// If the mouse still isn't in this popup after timer expires, close.
	if (!get_visible_rect().has_point(get_mouse_position())) {
		_close_pressed();
	}
}

void PopupMenu::_close_pressed() {
	// Only apply minimum lifetime to submenus.
	PopupMenu *parent_pum = Object::cast_to<PopupMenu>(get_parent());
	if (!parent_pum) {
		Popup::_close_pressed();
		return;
	}

	// If the timer has expired, close. If timer is still running, do nothing.
	if (close_allowed) {
		close_allowed = false;
		Popup::_close_pressed();
	} else if (minimum_lifetime_timer->is_stopped()) {
		minimum_lifetime_timer->start();
	}
}

void PopupMenu::_shape_item(int p_item) {
	if (items.write[p_item].dirty) {
		items.write[p_item].text_buf->clear();

		Ref<Font> font = get_theme_font(SNAME("font"));
		int font_size = get_theme_font_size(SNAME("font_size"));

		if (items[p_item].text_direction == Control::TEXT_DIRECTION_INHERITED) {
			items.write[p_item].text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
		} else {
			items.write[p_item].text_buf->set_direction((TextServer::Direction)items[p_item].text_direction);
		}
		items.write[p_item].text_buf->add_string(items.write[p_item].xl_text, font, font_size, items[p_item].opentype_features, (items[p_item].language != "") ? items[p_item].language : TranslationServer::get_singleton()->get_tool_locale());

		items.write[p_item].accel_text_buf->clear();
		items.write[p_item].accel_text_buf->set_direction(is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
		items.write[p_item].accel_text_buf->add_string(_get_accel_text(items.write[p_item]), font, font_size, Dictionary(), TranslationServer::get_singleton()->get_tool_locale());
		items.write[p_item].dirty = false;
	}
}

void PopupMenu::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			PopupMenu *pm = Object::cast_to<PopupMenu>(get_parent());
			if (pm) {
				// Inherit submenu's popup delay time from parent menu
				float pm_delay = pm->get_submenu_popup_delay();
				set_submenu_popup_delay(pm_delay);
			}
		} break;
		case NOTIFICATION_THEME_CHANGED:
		case Control::NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED: {
			for (int i = 0; i < items.size(); i++) {
				items.write[i].xl_text = atr(items[i].text);
				items.write[i].dirty = true;
				_shape_item(i);
			}

			child_controls_changed();
			control->update();
		} break;
		case NOTIFICATION_WM_MOUSE_ENTER: {
			grab_focus();
		} break;
		case NOTIFICATION_WM_MOUSE_EXIT: {
			if (mouse_over >= 0 && (items[mouse_over].submenu == "" || submenu_over != -1)) {
				mouse_over = -1;
				control->update();
			}
		} break;
		case NOTIFICATION_POST_POPUP: {
			initial_button_mask = Input::get_singleton()->get_mouse_button_mask();
			during_grabbed_click = (bool)initial_button_mask;
		} break;
		case NOTIFICATION_WM_SIZE_CHANGED: {
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			//only used when using operating system windows
			if (!is_embedded() && autohide_areas.size()) {
				Point2 mouse_pos = DisplayServer::get_singleton()->mouse_get_position();
				mouse_pos -= get_position();

				for (const Rect2 &E : autohide_areas) {
					if (!Rect2(Point2(), get_size()).has_point(mouse_pos) && E.has_point(mouse_pos)) {
						_close_pressed();
						return;
					}
				}
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				if (mouse_over >= 0) {
					mouse_over = -1;
					control->update();
				}

				for (int i = 0; i < items.size(); i++) {
					if (items[i].submenu == "") {
						continue;
					}

					Node *n = get_node(items[i].submenu);
					if (!n) {
						continue;
					}

					PopupMenu *pm = Object::cast_to<PopupMenu>(n);
					if (!pm || !pm->is_visible()) {
						continue;
					}

					pm->hide();
				}

				set_process_internal(false);
			} else {
				if (!is_embedded()) {
					set_process_internal(true);
				}

				// Set margin on the margin container
				Ref<StyleBox> panel_style = get_theme_stylebox(SNAME("panel"));
				margin_container->add_theme_constant_override("margin_top", panel_style->get_margin(Side::SIDE_TOP));
				margin_container->add_theme_constant_override("margin_bottom", panel_style->get_margin(Side::SIDE_BOTTOM));
				margin_container->add_theme_constant_override("margin_left", panel_style->get_margin(Side::SIDE_LEFT));
				margin_container->add_theme_constant_override("margin_right", panel_style->get_margin(Side::SIDE_RIGHT));
			}
		} break;
	}
}

/* Methods to add items with or without icon, checkbox, shortcut.
 * Be sure to keep them in sync when adding new properties in the Item struct.
 */

#define ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel) \
	item.text = p_label;                              \
	item.xl_text = atr(p_label);                      \
	item.id = p_id == -1 ? items.size() : p_id;       \
	item.accel = p_accel;

void PopupMenu::add_item(const String &p_label, int p_id, Key p_accel) {
	Item item;
	ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel);
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
	notify_property_list_changed();
}

void PopupMenu::add_icon_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id, Key p_accel) {
	Item item;
	ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel);
	item.icon = p_icon;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
	notify_property_list_changed();
}

void PopupMenu::add_check_item(const String &p_label, int p_id, Key p_accel) {
	Item item;
	ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel);
	item.checkable_type = Item::CHECKABLE_TYPE_CHECK_BOX;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_icon_check_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id, Key p_accel) {
	Item item;
	ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel);
	item.icon = p_icon;
	item.checkable_type = Item::CHECKABLE_TYPE_CHECK_BOX;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_radio_check_item(const String &p_label, int p_id, Key p_accel) {
	Item item;
	ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel);
	item.checkable_type = Item::CHECKABLE_TYPE_RADIO_BUTTON;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_icon_radio_check_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id, Key p_accel) {
	Item item;
	ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel);
	item.icon = p_icon;
	item.checkable_type = Item::CHECKABLE_TYPE_RADIO_BUTTON;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_multistate_item(const String &p_label, int p_max_states, int p_default_state, int p_id, Key p_accel) {
	Item item;
	ITEM_SETUP_WITH_ACCEL(p_label, p_id, p_accel);
	item.max_states = p_max_states;
	item.state = p_default_state;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

#define ITEM_SETUP_WITH_SHORTCUT(p_shortcut, p_id, p_global)                           \
	ERR_FAIL_COND_MSG(p_shortcut.is_null(), "Cannot add item with invalid Shortcut."); \
	_ref_shortcut(p_shortcut);                                                         \
	item.text = p_shortcut->get_name();                                                \
	item.xl_text = atr(item.text);                                                     \
	item.id = p_id == -1 ? items.size() : p_id;                                        \
	item.shortcut = p_shortcut;                                                        \
	item.shortcut_is_global = p_global;

void PopupMenu::add_shortcut(const Ref<Shortcut> &p_shortcut, int p_id, bool p_global) {
	Item item;
	ITEM_SETUP_WITH_SHORTCUT(p_shortcut, p_id, p_global);
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_icon_shortcut(const Ref<Texture2D> &p_icon, const Ref<Shortcut> &p_shortcut, int p_id, bool p_global) {
	Item item;
	ITEM_SETUP_WITH_SHORTCUT(p_shortcut, p_id, p_global);
	item.icon = p_icon;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_check_shortcut(const Ref<Shortcut> &p_shortcut, int p_id, bool p_global) {
	Item item;
	ITEM_SETUP_WITH_SHORTCUT(p_shortcut, p_id, p_global);
	item.checkable_type = Item::CHECKABLE_TYPE_CHECK_BOX;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_icon_check_shortcut(const Ref<Texture2D> &p_icon, const Ref<Shortcut> &p_shortcut, int p_id, bool p_global) {
	Item item;
	ITEM_SETUP_WITH_SHORTCUT(p_shortcut, p_id, p_global);
	item.icon = p_icon;
	item.checkable_type = Item::CHECKABLE_TYPE_CHECK_BOX;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_radio_check_shortcut(const Ref<Shortcut> &p_shortcut, int p_id, bool p_global) {
	Item item;
	ITEM_SETUP_WITH_SHORTCUT(p_shortcut, p_id, p_global);
	item.checkable_type = Item::CHECKABLE_TYPE_RADIO_BUTTON;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_icon_radio_check_shortcut(const Ref<Texture2D> &p_icon, const Ref<Shortcut> &p_shortcut, int p_id, bool p_global) {
	Item item;
	ITEM_SETUP_WITH_SHORTCUT(p_shortcut, p_id, p_global);
	item.icon = p_icon;
	item.checkable_type = Item::CHECKABLE_TYPE_RADIO_BUTTON;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_submenu_item(const String &p_label, const String &p_submenu, int p_id) {
	Item item;
	item.text = p_label;
	item.xl_text = atr(p_label);
	item.id = p_id == -1 ? items.size() : p_id;
	item.submenu = p_submenu;
	items.push_back(item);
	_shape_item(items.size() - 1);
	control->update();
	child_controls_changed();
}

#undef ITEM_SETUP_WITH_ACCEL
#undef ITEM_SETUP_WITH_SHORTCUT

/* Methods to modify existing items. */

void PopupMenu::set_item_text(int p_idx, const String &p_text) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].text = p_text;
	items.write[p_idx].xl_text = atr(p_text);
	_shape_item(p_idx);

	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_text_direction(int p_item, Control::TextDirection p_text_direction) {
	ERR_FAIL_INDEX(p_item, items.size());
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (items[p_item].text_direction != p_text_direction) {
		items.write[p_item].text_direction = p_text_direction;
		items.write[p_item].dirty = true;
		control->update();
	}
}

void PopupMenu::clear_item_opentype_features(int p_item) {
	ERR_FAIL_INDEX(p_item, items.size());
	items.write[p_item].opentype_features.clear();
	items.write[p_item].dirty = true;
	control->update();
}

void PopupMenu::set_item_opentype_feature(int p_item, const String &p_name, int p_value) {
	ERR_FAIL_INDEX(p_item, items.size());
	int32_t tag = TS->name_to_tag(p_name);
	if (!items[p_item].opentype_features.has(tag) || (int)items[p_item].opentype_features[tag] != p_value) {
		items.write[p_item].opentype_features[tag] = p_value;
		items.write[p_item].dirty = true;
		control->update();
	}
}

void PopupMenu::set_item_language(int p_item, const String &p_language) {
	ERR_FAIL_INDEX(p_item, items.size());
	if (items[p_item].language != p_language) {
		items.write[p_item].language = p_language;
		items.write[p_item].dirty = true;
		control->update();
	}
}

void PopupMenu::set_item_icon(int p_idx, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].icon = p_icon;

	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_checked(int p_idx, bool p_checked) {
	ERR_FAIL_INDEX(p_idx, items.size());

	items.write[p_idx].checked = p_checked;

	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_id(int p_idx, int p_id) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].id = p_id;

	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_accelerator(int p_idx, Key p_accel) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].accel = p_accel;
	items.write[p_idx].dirty = true;

	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_metadata(int p_idx, const Variant &p_meta) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].metadata = p_meta;
	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_disabled(int p_idx, bool p_disabled) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].disabled = p_disabled;
	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_submenu(int p_idx, const String &p_submenu) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].submenu = p_submenu;
	control->update();
	child_controls_changed();
}

void PopupMenu::toggle_item_checked(int p_idx) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].checked = !items[p_idx].checked;
	control->update();
	child_controls_changed();
}

String PopupMenu::get_item_text(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), "");
	return items[p_idx].text;
}

Control::TextDirection PopupMenu::get_item_text_direction(int p_item) const {
	ERR_FAIL_INDEX_V(p_item, items.size(), Control::TEXT_DIRECTION_INHERITED);
	return items[p_item].text_direction;
}

int PopupMenu::get_item_opentype_feature(int p_item, const String &p_name) const {
	ERR_FAIL_INDEX_V(p_item, items.size(), -1);
	int32_t tag = TS->name_to_tag(p_name);
	if (!items[p_item].opentype_features.has(tag)) {
		return -1;
	}
	return items[p_item].opentype_features[tag];
}

String PopupMenu::get_item_language(int p_item) const {
	ERR_FAIL_INDEX_V(p_item, items.size(), "");
	return items[p_item].language;
}

int PopupMenu::get_item_idx_from_text(const String &text) const {
	for (int idx = 0; idx < items.size(); idx++) {
		if (items[idx].text == text) {
			return idx;
		}
	}

	return -1;
}

Ref<Texture2D> PopupMenu::get_item_icon(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Ref<Texture2D>());
	return items[p_idx].icon;
}

Key PopupMenu::get_item_accelerator(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Key::NONE);
	return items[p_idx].accel;
}

Variant PopupMenu::get_item_metadata(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Variant());
	return items[p_idx].metadata;
}

bool PopupMenu::is_item_disabled(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].disabled;
}

bool PopupMenu::is_item_checked(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].checked;
}

int PopupMenu::get_item_id(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), 0);
	return items[p_idx].id;
}

int PopupMenu::get_item_index(int p_id) const {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].id == p_id) {
			return i;
		}
	}

	return -1;
}

String PopupMenu::get_item_submenu(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), "");
	return items[p_idx].submenu;
}

String PopupMenu::get_item_tooltip(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), "");
	return items[p_idx].tooltip;
}

Ref<Shortcut> PopupMenu::get_item_shortcut(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), Ref<Shortcut>());
	return items[p_idx].shortcut;
}

int PopupMenu::get_item_state(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), -1);
	return items[p_idx].state;
}

void PopupMenu::set_item_as_separator(int p_idx, bool p_separator) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].separator = p_separator;
	control->update();
}

bool PopupMenu::is_item_separator(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].separator;
}

void PopupMenu::set_item_as_checkable(int p_idx, bool p_checkable) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].checkable_type = p_checkable ? Item::CHECKABLE_TYPE_CHECK_BOX : Item::CHECKABLE_TYPE_NONE;
	control->update();
}

void PopupMenu::set_item_as_radio_checkable(int p_idx, bool p_radio_checkable) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].checkable_type = p_radio_checkable ? Item::CHECKABLE_TYPE_RADIO_BUTTON : Item::CHECKABLE_TYPE_NONE;
	control->update();
}

void PopupMenu::set_item_tooltip(int p_idx, const String &p_tooltip) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].tooltip = p_tooltip;
	control->update();
}

void PopupMenu::set_item_shortcut(int p_idx, const Ref<Shortcut> &p_shortcut, bool p_global) {
	ERR_FAIL_INDEX(p_idx, items.size());
	if (items[p_idx].shortcut.is_valid()) {
		_unref_shortcut(items[p_idx].shortcut);
	}
	items.write[p_idx].shortcut = p_shortcut;
	items.write[p_idx].shortcut_is_global = p_global;
	items.write[p_idx].dirty = true;

	if (items[p_idx].shortcut.is_valid()) {
		_ref_shortcut(items[p_idx].shortcut);
	}

	control->update();
}

void PopupMenu::set_item_h_offset(int p_idx, int p_offset) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].h_ofs = p_offset;
	control->update();
	child_controls_changed();
}

void PopupMenu::set_item_multistate(int p_idx, int p_state) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].state = p_state;
	control->update();
}

void PopupMenu::set_item_shortcut_disabled(int p_idx, bool p_disabled) {
	ERR_FAIL_INDEX(p_idx, items.size());
	items.write[p_idx].shortcut_is_disabled = p_disabled;
	control->update();
}

void PopupMenu::toggle_item_multistate(int p_idx) {
	ERR_FAIL_INDEX(p_idx, items.size());
	if (0 >= items[p_idx].max_states) {
		return;
	}

	++items.write[p_idx].state;
	if (items.write[p_idx].max_states <= items[p_idx].state) {
		items.write[p_idx].state = 0;
	}

	control->update();
}

bool PopupMenu::is_item_checkable(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].checkable_type;
}

bool PopupMenu::is_item_radio_checkable(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].checkable_type == Item::CHECKABLE_TYPE_RADIO_BUTTON;
}

bool PopupMenu::is_item_shortcut_disabled(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, items.size(), false);
	return items[p_idx].shortcut_is_disabled;
}

int PopupMenu::get_current_index() const {
	return mouse_over;
}

void PopupMenu::set_item_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	items.resize(p_count);
	control->update();
	child_controls_changed();
	notify_property_list_changed();
}

int PopupMenu::get_item_count() const {
	return items.size();
}

bool PopupMenu::activate_item_by_event(const Ref<InputEvent> &p_event, bool p_for_global_only) {
	Key code = Key::NONE;
	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		code = k->get_keycode();
		if (code == Key::NONE) {
			code = (Key)k->get_unicode();
		}
		if (k->is_ctrl_pressed()) {
			code |= KeyModifierMask::CTRL;
		}
		if (k->is_alt_pressed()) {
			code |= KeyModifierMask::ALT;
		}
		if (k->is_meta_pressed()) {
			code |= KeyModifierMask::META;
		}
		if (k->is_shift_pressed()) {
			code |= KeyModifierMask::SHIFT;
		}
	}

	for (int i = 0; i < items.size(); i++) {
		if (is_item_disabled(i) || items[i].shortcut_is_disabled) {
			continue;
		}

		if (items[i].shortcut.is_valid() && items[i].shortcut->matches_event(p_event) && (items[i].shortcut_is_global || !p_for_global_only)) {
			activate_item(i);
			return true;
		}

		if (code != Key::NONE && items[i].accel == code) {
			activate_item(i);
			return true;
		}

		if (items[i].submenu != "") {
			Node *n = get_node(items[i].submenu);
			if (!n) {
				continue;
			}

			PopupMenu *pm = Object::cast_to<PopupMenu>(n);
			if (!pm) {
				continue;
			}

			if (pm->activate_item_by_event(p_event, p_for_global_only)) {
				return true;
			}
		}
	}
	return false;
}

void PopupMenu::activate_item(int p_item) {
	ERR_FAIL_INDEX(p_item, items.size());
	ERR_FAIL_COND(items[p_item].separator);
	int id = items[p_item].id >= 0 ? items[p_item].id : p_item;

	//hide all parent PopupMenus
	Node *next = get_parent();
	PopupMenu *pop = Object::cast_to<PopupMenu>(next);
	while (pop) {
		// We close all parents that are chained together,
		// with hide_on_item_selection enabled

		if (items[p_item].checkable_type) {
			if (!hide_on_checkable_item_selection || !pop->is_hide_on_checkable_item_selection()) {
				break;
			}
		} else if (0 < items[p_item].max_states) {
			if (!hide_on_multistate_item_selection || !pop->is_hide_on_multistate_item_selection()) {
				break;
			}
		} else if (!hide_on_item_selection || !pop->is_hide_on_item_selection()) {
			break;
		}

		pop->hide();
		next = next->get_parent();
		pop = Object::cast_to<PopupMenu>(next);
	}

	// Hides popup by default; unless otherwise specified
	// by using set_hide_on_item_selection and set_hide_on_checkable_item_selection

	bool need_hide = true;

	if (items[p_item].checkable_type) {
		if (!hide_on_checkable_item_selection) {
			need_hide = false;
		}
	} else if (0 < items[p_item].max_states) {
		if (!hide_on_multistate_item_selection) {
			need_hide = false;
		}
	} else if (!hide_on_item_selection) {
		need_hide = false;
	}

	emit_signal(SNAME("id_pressed"), id);
	emit_signal(SNAME("index_pressed"), p_item);

	if (need_hide) {
		hide();
	}
}

void PopupMenu::remove_item(int p_idx) {
	ERR_FAIL_INDEX(p_idx, items.size());

	if (items[p_idx].shortcut.is_valid()) {
		_unref_shortcut(items[p_idx].shortcut);
	}

	items.remove_at(p_idx);
	control->update();
	child_controls_changed();
}

void PopupMenu::add_separator(const String &p_text, int p_id) {
	Item sep;
	sep.separator = true;
	sep.id = p_id;
	if (p_text != String()) {
		sep.text = p_text;
		sep.xl_text = atr(p_text);
	}
	items.push_back(sep);
	control->update();
}

void PopupMenu::clear() {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].shortcut.is_valid()) {
			_unref_shortcut(items[i].shortcut);
		}
	}
	items.clear();
	mouse_over = -1;
	control->update();
	child_controls_changed();
	notify_property_list_changed();
}

void PopupMenu::_ref_shortcut(Ref<Shortcut> p_sc) {
	if (!shortcut_refcount.has(p_sc)) {
		shortcut_refcount[p_sc] = 1;
		p_sc->connect("changed", callable_mp((CanvasItem *)this, &CanvasItem::update));
	} else {
		shortcut_refcount[p_sc] += 1;
	}
}

void PopupMenu::_unref_shortcut(Ref<Shortcut> p_sc) {
	ERR_FAIL_COND(!shortcut_refcount.has(p_sc));
	shortcut_refcount[p_sc]--;
	if (shortcut_refcount[p_sc] == 0) {
		p_sc->disconnect("changed", callable_mp((CanvasItem *)this, &CanvasItem::update));
		shortcut_refcount.erase(p_sc);
	}
}

// Hide on item selection determines whether or not the popup will close after item selection
void PopupMenu::set_hide_on_item_selection(bool p_enabled) {
	hide_on_item_selection = p_enabled;
}

bool PopupMenu::is_hide_on_item_selection() const {
	return hide_on_item_selection;
}

void PopupMenu::set_hide_on_checkable_item_selection(bool p_enabled) {
	hide_on_checkable_item_selection = p_enabled;
}

bool PopupMenu::is_hide_on_checkable_item_selection() const {
	return hide_on_checkable_item_selection;
}

void PopupMenu::set_hide_on_multistate_item_selection(bool p_enabled) {
	hide_on_multistate_item_selection = p_enabled;
}

bool PopupMenu::is_hide_on_multistate_item_selection() const {
	return hide_on_multistate_item_selection;
}

void PopupMenu::set_submenu_popup_delay(float p_time) {
	if (p_time <= 0) {
		p_time = 0.01;
	}

	submenu_timer->set_wait_time(p_time);
}

float PopupMenu::get_submenu_popup_delay() const {
	return submenu_timer->get_wait_time();
}

void PopupMenu::set_allow_search(bool p_allow) {
	allow_search = p_allow;
}

bool PopupMenu::get_allow_search() const {
	return allow_search;
}

String PopupMenu::get_tooltip(const Point2 &p_pos) const {
	int over = _get_mouse_over(p_pos);
	if (over < 0 || over >= items.size()) {
		return "";
	}
	return items[over].tooltip;
}

void PopupMenu::set_parent_rect(const Rect2 &p_rect) {
	parent_rect = p_rect;
}

void PopupMenu::get_translatable_strings(List<String> *p_strings) const {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].xl_text != "") {
			p_strings->push_back(items[i].xl_text);
		}
	}
}

void PopupMenu::add_autohide_area(const Rect2 &p_area) {
	autohide_areas.push_back(p_area);
}

void PopupMenu::clear_autohide_areas() {
	autohide_areas.clear();
}

void PopupMenu::take_mouse_focus() {
	ERR_FAIL_COND(!is_inside_tree());

	if (get_parent()) {
		get_parent()->get_viewport()->pass_mouse_focus_to(this, control);
	}
}

bool PopupMenu::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() >= 2 && components[0].begins_with("item_") && components[0].trim_prefix("item_").is_valid_int()) {
		int item_index = components[0].trim_prefix("item_").to_int();
		String property = components[1];
		if (property == "text") {
			set_item_text(item_index, p_value);
			return true;
		} else if (property == "icon") {
			set_item_icon(item_index, p_value);
			return true;
		} else if (property == "checkable") {
			bool radio_checkable = (int)p_value == Item::CHECKABLE_TYPE_RADIO_BUTTON;
			if (radio_checkable) {
				set_item_as_radio_checkable(item_index, true);
			} else {
				bool checkable = p_value;
				set_item_as_checkable(item_index, checkable);
			}
			return true;
		} else if (property == "checked") {
			set_item_checked(item_index, p_value);
			return true;
		} else if (property == "id") {
			set_item_id(item_index, p_value);
			return true;
		} else if (components[1] == "disabled") {
			set_item_disabled(item_index, p_value);
			return true;
		} else if (property == "separator") {
			set_item_as_separator(item_index, p_value);
			return true;
		}
	}
#ifndef DISABLE_DEPRECATED
	// Compatibility.
	if (p_name == "items") {
		Array arr = p_value;
		ERR_FAIL_COND_V(arr.size() % 10, false);
		clear();

		for (int i = 0; i < arr.size(); i += 10) {
			String text = arr[i + 0];
			Ref<Texture2D> icon = arr[i + 1];
			// For compatibility, use false/true for no/checkbox and integers for other values
			bool checkable = arr[i + 2];
			bool radio_checkable = (int)arr[i + 2] == Item::CHECKABLE_TYPE_RADIO_BUTTON;
			bool checked = arr[i + 3];
			bool disabled = arr[i + 4];

			int id = arr[i + 5];
			int accel = arr[i + 6];
			Variant meta = arr[i + 7];
			String subm = arr[i + 8];
			bool sep = arr[i + 9];

			int idx = get_item_count();
			add_item(text, id);
			set_item_icon(idx, icon);
			if (checkable) {
				if (radio_checkable) {
					set_item_as_radio_checkable(idx, true);
				} else {
					set_item_as_checkable(idx, true);
				}
			}
			set_item_checked(idx, checked);
			set_item_disabled(idx, disabled);
			set_item_id(idx, id);
			set_item_metadata(idx, meta);
			set_item_as_separator(idx, sep);
			set_item_accelerator(idx, (Key)accel);
			set_item_submenu(idx, subm);
		}
	}
#endif
	return false;
}

bool PopupMenu::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() >= 2 && components[0].begins_with("item_") && components[0].trim_prefix("item_").is_valid_int()) {
		int item_index = components[0].trim_prefix("item_").to_int();
		String property = components[1];
		if (property == "text") {
			r_ret = get_item_text(item_index);
			return true;
		} else if (property == "icon") {
			r_ret = get_item_icon(item_index);
			return true;
		} else if (property == "checkable") {
			r_ret = this->items[item_index].checkable_type;
			return true;
		} else if (property == "checked") {
			r_ret = is_item_checked(item_index);
			return true;
		} else if (property == "id") {
			r_ret = get_item_id(item_index);
			return true;
		} else if (components[1] == "disabled") {
			r_ret = is_item_disabled(item_index);
			return true;
		} else if (property == "separator") {
			r_ret = is_item_separator(item_index);
			return true;
		}
	}
	return false;
}

void PopupMenu::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < items.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, vformat("item_%d/text", i)));

		PropertyInfo pi = PropertyInfo(Variant::OBJECT, vformat("item_%d/icon", i), PROPERTY_HINT_RESOURCE_TYPE, "Texture2D");
		pi.usage &= ~(get_item_icon(i).is_null() ? PROPERTY_USAGE_STORAGE : 0);
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::INT, vformat("item_%d/checkable", i), PROPERTY_HINT_ENUM, "No,As checkbox,As radio button");
		pi.usage &= ~(!is_item_checkable(i) ? PROPERTY_USAGE_STORAGE : 0);
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::BOOL, vformat("item_%d/checked", i));
		pi.usage &= ~(!is_item_checked(i) ? PROPERTY_USAGE_STORAGE : 0);
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::INT, vformat("item_%d/id", i), PROPERTY_HINT_RANGE, "1,10,1,or_greater");
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::BOOL, vformat("item_%d/disabled", i));
		pi.usage &= ~(!is_item_disabled(i) ? PROPERTY_USAGE_STORAGE : 0);
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::BOOL, vformat("item_%d/separator", i));
		pi.usage &= ~(!is_item_separator(i) ? PROPERTY_USAGE_STORAGE : 0);
		p_list->push_back(pi);
	}
}

void PopupMenu::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_item", "label", "id", "accel"), &PopupMenu::add_item, DEFVAL(-1), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("add_icon_item", "texture", "label", "id", "accel"), &PopupMenu::add_icon_item, DEFVAL(-1), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("add_check_item", "label", "id", "accel"), &PopupMenu::add_check_item, DEFVAL(-1), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("add_icon_check_item", "texture", "label", "id", "accel"), &PopupMenu::add_icon_check_item, DEFVAL(-1), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("add_radio_check_item", "label", "id", "accel"), &PopupMenu::add_radio_check_item, DEFVAL(-1), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("add_icon_radio_check_item", "texture", "label", "id", "accel"), &PopupMenu::add_icon_radio_check_item, DEFVAL(-1), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("add_multistate_item", "label", "max_states", "default_state", "id", "accel"), &PopupMenu::add_multistate_item, DEFVAL(0), DEFVAL(-1), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("add_shortcut", "shortcut", "id", "global"), &PopupMenu::add_shortcut, DEFVAL(-1), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_icon_shortcut", "texture", "shortcut", "id", "global"), &PopupMenu::add_icon_shortcut, DEFVAL(-1), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_check_shortcut", "shortcut", "id", "global"), &PopupMenu::add_check_shortcut, DEFVAL(-1), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_icon_check_shortcut", "texture", "shortcut", "id", "global"), &PopupMenu::add_icon_check_shortcut, DEFVAL(-1), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_radio_check_shortcut", "shortcut", "id", "global"), &PopupMenu::add_radio_check_shortcut, DEFVAL(-1), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_icon_radio_check_shortcut", "texture", "shortcut", "id", "global"), &PopupMenu::add_icon_radio_check_shortcut, DEFVAL(-1), DEFVAL(false));

	ClassDB::bind_method(D_METHOD("add_submenu_item", "label", "submenu", "id"), &PopupMenu::add_submenu_item, DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("set_item_text", "idx", "text"), &PopupMenu::set_item_text);
	ClassDB::bind_method(D_METHOD("set_item_text_direction", "idx", "direction"), &PopupMenu::set_item_text_direction);
	ClassDB::bind_method(D_METHOD("set_item_opentype_feature", "idx", "tag", "value"), &PopupMenu::set_item_opentype_feature);
	ClassDB::bind_method(D_METHOD("set_item_language", "idx", "language"), &PopupMenu::set_item_language);
	ClassDB::bind_method(D_METHOD("set_item_icon", "idx", "icon"), &PopupMenu::set_item_icon);
	ClassDB::bind_method(D_METHOD("set_item_checked", "idx", "checked"), &PopupMenu::set_item_checked);
	ClassDB::bind_method(D_METHOD("set_item_id", "idx", "id"), &PopupMenu::set_item_id);
	ClassDB::bind_method(D_METHOD("set_item_accelerator", "idx", "accel"), &PopupMenu::set_item_accelerator);
	ClassDB::bind_method(D_METHOD("set_item_metadata", "idx", "metadata"), &PopupMenu::set_item_metadata);
	ClassDB::bind_method(D_METHOD("set_item_disabled", "idx", "disabled"), &PopupMenu::set_item_disabled);
	ClassDB::bind_method(D_METHOD("set_item_submenu", "idx", "submenu"), &PopupMenu::set_item_submenu);
	ClassDB::bind_method(D_METHOD("set_item_as_separator", "idx", "enable"), &PopupMenu::set_item_as_separator);
	ClassDB::bind_method(D_METHOD("set_item_as_checkable", "idx", "enable"), &PopupMenu::set_item_as_checkable);
	ClassDB::bind_method(D_METHOD("set_item_as_radio_checkable", "idx", "enable"), &PopupMenu::set_item_as_radio_checkable);
	ClassDB::bind_method(D_METHOD("set_item_tooltip", "idx", "tooltip"), &PopupMenu::set_item_tooltip);
	ClassDB::bind_method(D_METHOD("set_item_shortcut", "idx", "shortcut", "global"), &PopupMenu::set_item_shortcut, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_item_multistate", "idx", "state"), &PopupMenu::set_item_multistate);
	ClassDB::bind_method(D_METHOD("set_item_shortcut_disabled", "idx", "disabled"), &PopupMenu::set_item_shortcut_disabled);

	ClassDB::bind_method(D_METHOD("toggle_item_checked", "idx"), &PopupMenu::toggle_item_checked);
	ClassDB::bind_method(D_METHOD("toggle_item_multistate", "idx"), &PopupMenu::toggle_item_multistate);

	ClassDB::bind_method(D_METHOD("get_item_text", "idx"), &PopupMenu::get_item_text);
	ClassDB::bind_method(D_METHOD("get_item_text_direction", "idx"), &PopupMenu::get_item_text_direction);
	ClassDB::bind_method(D_METHOD("get_item_opentype_feature", "idx", "tag"), &PopupMenu::get_item_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_item_opentype_features", "idx"), &PopupMenu::clear_item_opentype_features);
	ClassDB::bind_method(D_METHOD("get_item_language", "idx"), &PopupMenu::get_item_language);
	ClassDB::bind_method(D_METHOD("get_item_icon", "idx"), &PopupMenu::get_item_icon);
	ClassDB::bind_method(D_METHOD("is_item_checked", "idx"), &PopupMenu::is_item_checked);
	ClassDB::bind_method(D_METHOD("get_item_id", "idx"), &PopupMenu::get_item_id);
	ClassDB::bind_method(D_METHOD("get_item_index", "id"), &PopupMenu::get_item_index);
	ClassDB::bind_method(D_METHOD("get_item_accelerator", "idx"), &PopupMenu::get_item_accelerator);
	ClassDB::bind_method(D_METHOD("get_item_metadata", "idx"), &PopupMenu::get_item_metadata);
	ClassDB::bind_method(D_METHOD("is_item_disabled", "idx"), &PopupMenu::is_item_disabled);
	ClassDB::bind_method(D_METHOD("get_item_submenu", "idx"), &PopupMenu::get_item_submenu);
	ClassDB::bind_method(D_METHOD("is_item_separator", "idx"), &PopupMenu::is_item_separator);
	ClassDB::bind_method(D_METHOD("is_item_checkable", "idx"), &PopupMenu::is_item_checkable);
	ClassDB::bind_method(D_METHOD("is_item_radio_checkable", "idx"), &PopupMenu::is_item_radio_checkable);
	ClassDB::bind_method(D_METHOD("is_item_shortcut_disabled", "idx"), &PopupMenu::is_item_shortcut_disabled);
	ClassDB::bind_method(D_METHOD("get_item_tooltip", "idx"), &PopupMenu::get_item_tooltip);
	ClassDB::bind_method(D_METHOD("get_item_shortcut", "idx"), &PopupMenu::get_item_shortcut);

	ClassDB::bind_method(D_METHOD("get_current_index"), &PopupMenu::get_current_index);
	ClassDB::bind_method(D_METHOD("set_item_count", "count"), &PopupMenu::set_item_count);
	ClassDB::bind_method(D_METHOD("get_item_count"), &PopupMenu::get_item_count);

	ClassDB::bind_method(D_METHOD("remove_item", "idx"), &PopupMenu::remove_item);

	ClassDB::bind_method(D_METHOD("add_separator", "label", "id"), &PopupMenu::add_separator, DEFVAL(String()), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("clear"), &PopupMenu::clear);

	ClassDB::bind_method(D_METHOD("set_hide_on_item_selection", "enable"), &PopupMenu::set_hide_on_item_selection);
	ClassDB::bind_method(D_METHOD("is_hide_on_item_selection"), &PopupMenu::is_hide_on_item_selection);

	ClassDB::bind_method(D_METHOD("set_hide_on_checkable_item_selection", "enable"), &PopupMenu::set_hide_on_checkable_item_selection);
	ClassDB::bind_method(D_METHOD("is_hide_on_checkable_item_selection"), &PopupMenu::is_hide_on_checkable_item_selection);

	ClassDB::bind_method(D_METHOD("set_hide_on_state_item_selection", "enable"), &PopupMenu::set_hide_on_multistate_item_selection);
	ClassDB::bind_method(D_METHOD("is_hide_on_state_item_selection"), &PopupMenu::is_hide_on_multistate_item_selection);

	ClassDB::bind_method(D_METHOD("set_submenu_popup_delay", "seconds"), &PopupMenu::set_submenu_popup_delay);
	ClassDB::bind_method(D_METHOD("get_submenu_popup_delay"), &PopupMenu::get_submenu_popup_delay);

	ClassDB::bind_method(D_METHOD("set_allow_search", "allow"), &PopupMenu::set_allow_search);
	ClassDB::bind_method(D_METHOD("get_allow_search"), &PopupMenu::get_allow_search);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_on_item_selection"), "set_hide_on_item_selection", "is_hide_on_item_selection");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_on_checkable_item_selection"), "set_hide_on_checkable_item_selection", "is_hide_on_checkable_item_selection");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hide_on_state_item_selection"), "set_hide_on_state_item_selection", "is_hide_on_state_item_selection");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "submenu_popup_delay"), "set_submenu_popup_delay", "get_submenu_popup_delay");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_search"), "set_allow_search", "get_allow_search");

	ADD_ARRAY_COUNT("Items", "item_count", "set_item_count", "get_item_count", "item_");

	ADD_SIGNAL(MethodInfo("id_pressed", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("id_focused", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("index_pressed", PropertyInfo(Variant::INT, "index")));
}

void PopupMenu::popup(const Rect2 &p_bounds) {
	moved = Vector2();
	popup_time_msec = OS::get_singleton()->get_ticks_msec();
	Popup::popup(p_bounds);
}

PopupMenu::PopupMenu() {
	// Margin Container
	margin_container = memnew(MarginContainer);
	margin_container->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	add_child(margin_container, false, INTERNAL_MODE_FRONT);
	margin_container->connect("draw", callable_mp(this, &PopupMenu::_draw_background));

	// Scroll Container
	scroll_container = memnew(ScrollContainer);
	scroll_container->set_clip_contents(true);
	margin_container->add_child(scroll_container);

	// The control which will display the items
	control = memnew(Control);
	control->set_clip_contents(false);
	control->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	control->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	control->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	scroll_container->add_child(control, false, INTERNAL_MODE_FRONT);
	control->connect("draw", callable_mp(this, &PopupMenu::_draw_items));

	connect("window_input", callable_mp(this, &PopupMenu::gui_input));

	submenu_timer = memnew(Timer);
	submenu_timer->set_wait_time(0.3);
	submenu_timer->set_one_shot(true);
	submenu_timer->connect("timeout", callable_mp(this, &PopupMenu::_submenu_timeout));
	add_child(submenu_timer, false, INTERNAL_MODE_FRONT);

	minimum_lifetime_timer = memnew(Timer);
	minimum_lifetime_timer->set_wait_time(0.3);
	minimum_lifetime_timer->set_one_shot(true);
	minimum_lifetime_timer->connect("timeout", callable_mp(this, &PopupMenu::_minimum_lifetime_timeout));
	add_child(minimum_lifetime_timer, false, INTERNAL_MODE_FRONT);
}

PopupMenu::~PopupMenu() {
}
