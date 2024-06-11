/**************************************************************************/
/*  option_button.cpp                                                     */
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

#include "option_button.h"

#include "core/os/keyboard.h"
#include "core/string/print_string.h"
#include "scene/theme/theme_db.h"

static const int NONE_SELECTED = -1;
static const bool LEFT = false;
static const bool RIGHT = true;

void OptionButton::shortcut_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (disable_shortcuts) {
		return;
	}

	if (p_event->is_pressed() && !p_event->is_echo() && !is_disabled() && is_visible_in_tree() && popup->activate_item_by_event(p_event, false)) {
		accept_event();
		return;
	}

	Button::shortcut_input(p_event);
}

Size2 OptionButton::get_minimum_size() const {
	Size2 minsize;
	if (fit_to_longest_item) {
		minsize = _cached_size;
	} else {
		minsize = Button::get_minimum_size();
	}

	if (has_theme_icon(SNAME("arrow"))) {
		const Size2 padding = _get_largest_stylebox_size();
		const Size2 arrow_size = theme_cache.arrow_icon->get_size();

		Size2 content_size = minsize - padding;
		content_size.width += arrow_size.width + MAX(0, theme_cache.h_separation);
		content_size.height = MAX(content_size.height, arrow_size.height);

		minsize = content_size + padding;
	}

	return minsize;
}

void OptionButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			_refresh_size_cache();
			if (type != DROPDOWN) {
				Ref<Texture2D> left_arrow_icon = _get_arrow_icon(LEFT);
				Ref<Texture2D> right_arrow_icon = _get_arrow_icon(RIGHT);
				if (left_arrow_icon.is_valid()) {
					_set_internal_margin(SIDE_LEFT, left_arrow_icon->get_width() + theme_cache.arrow_margin);
				}
				if (right_arrow_icon.is_valid()) {
					_set_internal_margin(SIDE_RIGHT, right_arrow_icon->get_width() + theme_cache.arrow_margin);
				}
			} else if (has_theme_icon(SNAME("arrow"))) {
				if (is_layout_rtl()) {
					_set_internal_margin(SIDE_LEFT, theme_cache.arrow_icon->get_width());
				} else {
					_set_internal_margin(SIDE_RIGHT, theme_cache.arrow_icon->get_width());
				}
			}
		} break;

		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			if (is_disabled()) {
				_set_arrow_disabled(LEFT, true);
				_set_arrow_disabled(RIGHT, true);
			}
			if (type != DROPDOWN) {
				Ref<Texture2D> left_arrow_icon = _get_arrow_icon(LEFT);
				Ref<Texture2D> right_arrow_icon = _get_arrow_icon(RIGHT);
				if (left_arrow_icon.is_valid()) {
					Color clr = Color(1, 1, 1);
					if (theme_cache.modulate_arrow) {
						switch (left_arrow_mode) {
							case DRAW_PRESSED:
								clr = theme_cache.font_pressed_color;
								break;
							case DRAW_HOVER:
								clr = theme_cache.font_hover_color;
								break;
							case DRAW_HOVER_PRESSED:
								clr = theme_cache.font_hover_pressed_color;
								break;
							case DRAW_DISABLED:
								clr = theme_cache.font_disabled_color;
								break;
							default:
								clr = theme_cache.font_color;
						}
					}
					Size2 size = get_size();
					Point2 ofs = Point2(theme_cache.arrow_margin,
							int(Math::abs((size.height - left_arrow_icon->get_height()) / 2)));
					left_arrow_icon->draw(ci, ofs, clr);
				}
				if (right_arrow_icon.is_valid()) {
					Color clr = Color(1, 1, 1);
					if (theme_cache.modulate_arrow) {
						switch (right_arrow_mode) {
							case DRAW_PRESSED:
								clr = theme_cache.font_pressed_color;
								break;
							case DRAW_HOVER:
								clr = theme_cache.font_hover_color;
								break;
							case DRAW_HOVER_PRESSED:
								clr = theme_cache.font_hover_pressed_color;
								break;
							case DRAW_DISABLED:
								clr = theme_cache.font_disabled_color;
								break;
							default:
								clr = theme_cache.font_color;
						}
					}
					Size2 size = get_size();
					Point2 ofs = Point2(size.x - right_arrow_icon->get_width() - theme_cache.arrow_margin,
							int(Math::abs((size.height - right_arrow_icon->get_height()) / 2)));
					right_arrow_icon->draw(ci, ofs, clr);
				}
			} else {
				if (!has_theme_icon(SNAME("arrow"))) {
					return;
				}

				Color clr = Color(1, 1, 1);
				if (theme_cache.modulate_arrow) {
					switch (get_draw_mode()) {
						case DRAW_PRESSED:
							clr = theme_cache.font_pressed_color;
							break;
						case DRAW_HOVER:
							clr = theme_cache.font_hover_color;
							break;
						case DRAW_HOVER_PRESSED:
							clr = theme_cache.font_hover_pressed_color;
							break;
						case DRAW_DISABLED:
							clr = theme_cache.font_disabled_color;
							break;
						default:
							if (has_focus()) {
								clr = theme_cache.font_focus_color;
							} else {
								clr = theme_cache.font_color;
							}
					}
				}

				Size2 size = get_size();

				Point2 ofs;
				if (is_layout_rtl()) {
					ofs = Point2(theme_cache.arrow_margin, int(Math::abs((size.height - theme_cache.arrow_icon->get_height()) / 2)));
				} else {
					ofs = Point2(size.width - theme_cache.arrow_icon->get_width() - theme_cache.arrow_margin, int(Math::abs((size.height - theme_cache.arrow_icon->get_height()) / 2)));
				}
				theme_cache.arrow_icon->draw(ci, ofs, clr);
			}
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			_set_arrow_hovered(LEFT, false);
			_set_arrow_hovered(RIGHT, false);
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED: {
			popup->set_layout_direction((Window::LayoutDirection)get_layout_direction());
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			if (type != DROPDOWN) {
				Ref<Texture2D> left_arrow_icon = _get_arrow_icon(LEFT);
				Ref<Texture2D> right_arrow_icon = _get_arrow_icon(RIGHT);
				if (left_arrow_icon.is_valid()) {
					_set_internal_margin(SIDE_LEFT, left_arrow_icon->get_width() + theme_cache.arrow_margin);
				} else {
					_set_internal_margin(SIDE_LEFT, 0.f);
				}
				if (right_arrow_icon.is_valid()) {
					_set_internal_margin(SIDE_RIGHT, right_arrow_icon->get_width() + theme_cache.arrow_margin);
				} else {
					_set_internal_margin(SIDE_RIGHT, 0.f);
				}
			} else {
				if (has_theme_icon(SNAME("arrow"))) {
					if (is_layout_rtl()) {
						_set_internal_margin(SIDE_LEFT, theme_cache.arrow_icon->get_width());
						_set_internal_margin(SIDE_RIGHT, 0.f);
					} else {
						_set_internal_margin(SIDE_LEFT, 0.f);
						_set_internal_margin(SIDE_RIGHT, theme_cache.arrow_icon->get_width());
					}
				} else {
					_set_internal_margin(SIDE_LEFT, 0.f);
					_set_internal_margin(SIDE_RIGHT, 0.f);
				}
			}
			_refresh_size_cache();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible_in_tree()) {
				popup->hide();
			}
		} break;
	}
}

bool OptionButton::_set(const StringName &p_name, const Variant &p_value) {
	int index;
	const String sname = p_name;

	if (property_helper.is_property_valid(sname, &index)) {
		bool valid;
		popup->set(sname.trim_prefix("popup/"), p_value, &valid);

		if (index == current) {
			// Force refreshing currently displayed item.
			current = NONE_SELECTED;
			_select(index, false);
		}

		const String property = sname.get_slice("/", 2);
		if (property == "text" || property == "icon") {
			_queue_update_size_cache();
		}

		return valid;
	}
	return false;
}

void OptionButton::_focused(int p_which) {
	emit_signal(SNAME("item_focused"), p_which);
}

void OptionButton::_selected(int p_which) {
	_select(p_which, true);
}

void OptionButton::pressed() {
	if (type == CAROUSEL) {
		if (current_mouse_button == MouseButton::RIGHT) {
			_select_previous(true);
		} else {
			_select_next(true);
		}
	} else {
		if (popup->is_visible()) {
			popup->hide();
			return;
		}

		show_popup();
	}
}

void OptionButton::add_icon_item(const Ref<Texture2D> &p_icon, const String &p_label, int p_id) {
	bool first_selectable = !has_selectable_items();
	popup->add_icon_radio_check_item(p_icon, p_label, p_id);
	if (first_selectable) {
		select(get_item_count() - 1);
	}
	_queue_update_size_cache();
}

void OptionButton::add_item(const String &p_label, int p_id) {
	bool first_selectable = !has_selectable_items();
	popup->add_radio_check_item(p_label, p_id);
	if (first_selectable) {
		select(get_item_count() - 1);
	}
	_queue_update_size_cache();
}

void OptionButton::set_item_text(int p_idx, const String &p_text) {
	popup->set_item_text(p_idx, p_text);

	if (current == p_idx) {
		set_text(p_text);
	}
	_queue_update_size_cache();
}

void OptionButton::set_item_icon(int p_idx, const Ref<Texture2D> &p_icon) {
	popup->set_item_icon(p_idx, p_icon);

	if (current == p_idx) {
		set_icon(p_icon);
	}
	_queue_update_size_cache();
}

void OptionButton::set_item_id(int p_idx, int p_id) {
	popup->set_item_id(p_idx, p_id);
}

void OptionButton::set_item_metadata(int p_idx, const Variant &p_metadata) {
	popup->set_item_metadata(p_idx, p_metadata);
}

void OptionButton::set_item_tooltip(int p_idx, const String &p_tooltip) {
	popup->set_item_tooltip(p_idx, p_tooltip);
}

void OptionButton::set_item_disabled(int p_idx, bool p_disabled) {
	popup->set_item_disabled(p_idx, p_disabled);
}

String OptionButton::get_item_text(int p_idx) const {
	return popup->get_item_text(p_idx);
}

Ref<Texture2D> OptionButton::get_item_icon(int p_idx) const {
	return popup->get_item_icon(p_idx);
}

int OptionButton::get_item_id(int p_idx) const {
	if (p_idx == NONE_SELECTED) {
		return NONE_SELECTED;
	}

	return popup->get_item_id(p_idx);
}

int OptionButton::get_item_index(int p_id) const {
	return popup->get_item_index(p_id);
}

Variant OptionButton::get_item_metadata(int p_idx) const {
	return popup->get_item_metadata(p_idx);
}

String OptionButton::get_item_tooltip(int p_idx) const {
	return popup->get_item_tooltip(p_idx);
}

bool OptionButton::is_item_disabled(int p_idx) const {
	return popup->is_item_disabled(p_idx);
}

bool OptionButton::is_item_separator(int p_idx) const {
	return popup->is_item_separator(p_idx);
}
void OptionButton::set_item_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);

	int count_old = get_item_count();
	if (p_count == count_old) {
		return;
	}

	popup->set_item_count(p_count);

	if (p_count > count_old) {
		for (int i = count_old; i < p_count; i++) {
			popup->set_item_as_radio_checkable(i, true);
		}
	}

	if (!initialized) {
		if (queued_current != current) {
			current = queued_current;
		}
		initialized = true;
	}

	_refresh_size_cache();
	notify_property_list_changed();
}

bool OptionButton::has_selectable_items() const {
	for (int i = 0; i < get_item_count(); i++) {
		if (!is_item_disabled(i) && !is_item_separator(i)) {
			return true;
		}
	}
	return false;
}
int OptionButton::get_selectable_item(bool p_from_last) const {
	if (!p_from_last) {
		for (int i = 0; i < get_item_count(); i++) {
			if (!is_item_disabled(i) && !is_item_separator(i)) {
				return i;
			}
		}
	} else {
		for (int i = get_item_count() - 1; i >= 0; i--) {
			if (!is_item_disabled(i) && !is_item_separator(i)) {
				return i;
			}
		}
	}
	return -1;
}

int OptionButton::get_item_count() const {
	return popup->get_item_count();
}

void OptionButton::set_fit_to_longest_item(bool p_fit) {
	if (p_fit == fit_to_longest_item) {
		return;
	}
	fit_to_longest_item = p_fit;

	_refresh_size_cache();
}

bool OptionButton::is_fit_to_longest_item() const {
	return fit_to_longest_item;
}

void OptionButton::set_allow_reselect(bool p_allow) {
	allow_reselect = p_allow;
}

bool OptionButton::get_allow_reselect() const {
	return allow_reselect;
}

void OptionButton::add_separator(const String &p_text) {
	popup->add_separator(p_text);
}

void OptionButton::clear() {
	popup->clear();
	set_text("");
	current = NONE_SELECTED;
	_refresh_size_cache();
}

void OptionButton::_select(int p_which, bool p_emit) {
	if (p_which == current && !allow_reselect) {
		return;
	}

	if (p_which == NONE_SELECTED) {
		for (int i = 0; i < popup->get_item_count(); i++) {
			popup->set_item_checked(i, false);
		}

		current = NONE_SELECTED;
		set_text("");
		set_icon(nullptr);
	} else {
		ERR_FAIL_INDEX(p_which, popup->get_item_count());

		for (int i = 0; i < popup->get_item_count(); i++) {
			popup->set_item_checked(i, i == p_which);
		}

		current = p_which;
		set_text(popup->get_item_text(current));
		set_icon(popup->get_item_icon(current));
	}

	if (is_layout_rtl()) {
		_set_arrow_disabled(LEFT, !_has_next_selectable_item());
		_set_arrow_disabled(RIGHT, !_has_previous_selectable_item());
	} else {
		_set_arrow_disabled(LEFT, !_has_previous_selectable_item());
		_set_arrow_disabled(RIGHT, !_has_next_selectable_item());
	}

	if (is_inside_tree() && p_emit) {
		emit_signal(SNAME("item_selected"), current);
	}
}

void OptionButton::_select_int(int p_which) {
	if (p_which < NONE_SELECTED) {
		return;
	}
	if (p_which >= popup->get_item_count()) {
		if (!initialized) {
			queued_current = p_which;
		}
		return;
	}
	_select(p_which, false);
}

void OptionButton::_refresh_size_cache() {
	cache_refresh_pending = false;

	if (fit_to_longest_item) {
		_cached_size = theme_cache.normal->get_minimum_size();
		for (int i = 0; i < get_item_count(); i++) {
			_cached_size = _cached_size.max(get_minimum_size_for_text_and_icon(popup->get_item_xl_text(i), get_item_icon(i)));
		}
	}
	update_minimum_size();
}

void OptionButton::_queue_update_size_cache() {
	if (cache_refresh_pending) {
		return;
	}
	cache_refresh_pending = true;

	callable_mp(this, &OptionButton::_refresh_size_cache).call_deferred();
}

void OptionButton::select(int p_idx) {
	_select(p_idx, false);
}

int OptionButton::get_selected() const {
	return current;
}

int OptionButton::get_selected_id() const {
	return get_item_id(current);
}

Variant OptionButton::get_selected_metadata() const {
	int idx = get_selected();
	if (idx < 0) {
		return Variant();
	}
	return get_item_metadata(current);
}

void OptionButton::remove_item(int p_idx) {
	popup->remove_item(p_idx);
	if (current == p_idx) {
		_select(NONE_SELECTED);
	}
	_queue_update_size_cache();
}

PopupMenu *OptionButton::get_popup() const {
	return popup;
}

void OptionButton::show_popup() {
	if (!get_viewport()) {
		return;
	}

	Rect2 rect = get_screen_rect();
	rect.position.y += rect.size.height;
	rect.size.height = 0;
	popup->set_position(rect.position);
	popup->set_size(rect.size);

	// If not triggered by the mouse, start the popup with the checked item (or the first enabled one) focused.
	if (current != NONE_SELECTED && !popup->is_item_disabled(current)) {
		if (!_was_pressed_by_mouse()) {
			popup->set_focused_item(current);
		} else {
			popup->scroll_to_item(current);
		}
	} else {
		for (int i = 0; i < popup->get_item_count(); i++) {
			if (!popup->is_item_disabled(i)) {
				if (!_was_pressed_by_mouse()) {
					popup->set_focused_item(i);
				} else {
					popup->scroll_to_item(i);
				}

				break;
			}
		}
	}

	popup->popup();
}

void OptionButton::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "text" || p_property.name == "icon") {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
	if (p_property.name == "carousel_wraparound" && type == DROPDOWN) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void OptionButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_item", "label", "id"), &OptionButton::add_item, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("add_icon_item", "texture", "label", "id"), &OptionButton::add_icon_item, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("set_item_text", "idx", "text"), &OptionButton::set_item_text);
	ClassDB::bind_method(D_METHOD("set_item_icon", "idx", "texture"), &OptionButton::set_item_icon);
	ClassDB::bind_method(D_METHOD("set_item_disabled", "idx", "disabled"), &OptionButton::set_item_disabled);
	ClassDB::bind_method(D_METHOD("set_item_id", "idx", "id"), &OptionButton::set_item_id);
	ClassDB::bind_method(D_METHOD("set_item_metadata", "idx", "metadata"), &OptionButton::set_item_metadata);
	ClassDB::bind_method(D_METHOD("set_item_tooltip", "idx", "tooltip"), &OptionButton::set_item_tooltip);
	ClassDB::bind_method(D_METHOD("get_item_text", "idx"), &OptionButton::get_item_text);
	ClassDB::bind_method(D_METHOD("get_item_icon", "idx"), &OptionButton::get_item_icon);
	ClassDB::bind_method(D_METHOD("get_item_id", "idx"), &OptionButton::get_item_id);
	ClassDB::bind_method(D_METHOD("get_item_index", "id"), &OptionButton::get_item_index);
	ClassDB::bind_method(D_METHOD("get_item_metadata", "idx"), &OptionButton::get_item_metadata);
	ClassDB::bind_method(D_METHOD("get_item_tooltip", "idx"), &OptionButton::get_item_tooltip);
	ClassDB::bind_method(D_METHOD("is_item_disabled", "idx"), &OptionButton::is_item_disabled);
	ClassDB::bind_method(D_METHOD("is_item_separator", "idx"), &OptionButton::is_item_separator);
	ClassDB::bind_method(D_METHOD("add_separator", "text"), &OptionButton::add_separator, DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("clear"), &OptionButton::clear);
	ClassDB::bind_method(D_METHOD("select", "idx"), &OptionButton::select);
	ClassDB::bind_method(D_METHOD("get_selected"), &OptionButton::get_selected);
	ClassDB::bind_method(D_METHOD("get_selected_id"), &OptionButton::get_selected_id);
	ClassDB::bind_method(D_METHOD("get_selected_metadata"), &OptionButton::get_selected_metadata);
	ClassDB::bind_method(D_METHOD("remove_item", "idx"), &OptionButton::remove_item);
	ClassDB::bind_method(D_METHOD("_select_int", "idx"), &OptionButton::_select_int);

	ClassDB::bind_method(D_METHOD("get_popup"), &OptionButton::get_popup);
	ClassDB::bind_method(D_METHOD("show_popup"), &OptionButton::show_popup);

	ClassDB::bind_method(D_METHOD("set_item_count", "count"), &OptionButton::set_item_count);
	ClassDB::bind_method(D_METHOD("get_item_count"), &OptionButton::get_item_count);
	ClassDB::bind_method(D_METHOD("has_selectable_items"), &OptionButton::has_selectable_items);
	ClassDB::bind_method(D_METHOD("get_selectable_item", "from_last"), &OptionButton::get_selectable_item, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_fit_to_longest_item", "fit"), &OptionButton::set_fit_to_longest_item);
	ClassDB::bind_method(D_METHOD("is_fit_to_longest_item"), &OptionButton::is_fit_to_longest_item);
	ClassDB::bind_method(D_METHOD("set_allow_reselect", "allow"), &OptionButton::set_allow_reselect);
	ClassDB::bind_method(D_METHOD("get_allow_reselect"), &OptionButton::get_allow_reselect);
	ClassDB::bind_method(D_METHOD("set_disable_shortcuts", "disabled"), &OptionButton::set_disable_shortcuts);
	ClassDB::bind_method(D_METHOD("set_carousel_wraparound", "carousel_wraparound"), &OptionButton::set_carousel_wraparound);
	ClassDB::bind_method(D_METHOD("is_carousel_wraparound"), &OptionButton::is_carousel_wraparound);
	ClassDB::bind_method(D_METHOD("set_type", "type"), &OptionButton::set_type);
	ClassDB::bind_method(D_METHOD("get_type"), &OptionButton::get_type);

	BIND_ENUM_CONSTANT(DROPDOWN);
	BIND_ENUM_CONSTANT(HYBRID);
	BIND_ENUM_CONSTANT(CAROUSEL);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "selected"), "_select_int", "get_selected");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fit_to_longest_item"), "set_fit_to_longest_item", "is_fit_to_longest_item");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "allow_reselect"), "set_allow_reselect", "get_allow_reselect");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "carousel_wraparound"), "set_carousel_wraparound", "is_carousel_wraparound");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_ENUM, "Dropdown,Hybrid,Carousel"), "set_type", "get_type");
	ADD_ARRAY_COUNT("Items", "item_count", "set_item_count", "get_item_count", "popup/item_");

	ADD_SIGNAL(MethodInfo("item_selected", PropertyInfo(Variant::INT, "index")));
	ADD_SIGNAL(MethodInfo("item_focused", PropertyInfo(Variant::INT, "index")));

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, OptionButton, normal);

	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OptionButton, font_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OptionButton, font_focus_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OptionButton, font_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OptionButton, font_hover_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OptionButton, font_hover_pressed_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, OptionButton, font_disabled_color);

	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, OptionButton, h_separation);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, arrow_icon, "arrow");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, left_arrow_normal_icon, "left_arrow_normal");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, left_arrow_hover_icon, "left_arrow_hover");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, left_arrow_hover_pressed_icon, "left_arrow_hover_pressed");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, left_arrow_pressed_icon, "left_arrow_pressed");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, left_arrow_disabled_icon, "left_arrow_disabled");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, right_arrow_normal_icon, "right_arrow_normal");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, right_arrow_hover_icon, "right_arrow_hover");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, right_arrow_hover_pressed_icon, "right_arrow_hover_pressed");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, right_arrow_pressed_icon, "right_arrow_pressed");
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, OptionButton, right_arrow_disabled_icon, "right_arrow_disabled");
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, OptionButton, arrow_margin);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, OptionButton, modulate_arrow);

	PopupMenu::Item defaults(true);

	base_property_helper.set_prefix("popup/item_");
	base_property_helper.set_array_length_getter(&OptionButton::get_item_count);
	base_property_helper.register_property(PropertyInfo(Variant::STRING, "text"), defaults.text, &OptionButton::_dummy_setter, &OptionButton::get_item_text);
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), defaults.icon, &OptionButton::_dummy_setter, &OptionButton::get_item_icon);
	base_property_helper.register_property(PropertyInfo(Variant::INT, "id", PROPERTY_HINT_RANGE, "0,10,1,or_greater"), defaults.id, &OptionButton::_dummy_setter, &OptionButton::get_item_id);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "disabled"), defaults.disabled, &OptionButton::_dummy_setter, &OptionButton::is_item_disabled);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "separator"), defaults.separator, &OptionButton::_dummy_setter, &OptionButton::is_item_separator);
}

void OptionButton::set_disable_shortcuts(bool p_disabled) {
	disable_shortcuts = p_disabled;
}

void OptionButton::set_type(OptionButton::Type p_type) {
	bool initial = type;
	type = p_type;
	set_toggle_mode(type != CAROUSEL);
	_select(current);
	if (initial != p_type) {
		notification(NOTIFICATION_THEME_CHANGED);
		queue_redraw();
	}
	if (type != DROPDOWN && get_selected() == NONE_SELECTED && get_item_count() > 0) {
		_select_next(true);
	}
}

OptionButton::Type OptionButton::get_type() {
	return type;
}

void OptionButton::set_carousel_wraparound(bool p_carousel_wraparound) {
	carousel_wraparound = p_carousel_wraparound;
	_select(current);
}

bool OptionButton::is_carousel_wraparound() {
	return carousel_wraparound;
}

OptionButton::OptionButton(const String &p_text) :
		Button(p_text) {
	set_toggle_mode(true);
	set_process_shortcut_input(true);
	set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	set_action_mode(ACTION_MODE_BUTTON_PRESS);

	popup = memnew(PopupMenu);
	popup->hide();
	add_child(popup, false, INTERNAL_MODE_FRONT);
	popup->connect("index_pressed", callable_mp(this, &OptionButton::_selected));
	popup->connect("id_focused", callable_mp(this, &OptionButton::_focused));
	popup->connect("popup_hide", callable_mp((BaseButton *)this, &BaseButton::set_pressed).bind(false));

	property_helper.setup_for_instance(base_property_helper, this);
}

// assumes p_pos is already inside the control
bool OptionButton::_is_over_arrow(bool p_arrow, Vector2 p_pos) {
	return p_arrow ? (p_pos.x > get_size().x - _get_arrow_icon(p_arrow)->get_size().x - theme_cache.arrow_margin) : (p_pos.x < _get_arrow_icon(p_arrow)->get_size().x + theme_cache.arrow_margin);
}

void OptionButton::_set_arrow_mode(bool p_arrow, DrawMode p_mode) {
	DrawMode initial = p_arrow ? right_arrow_mode : left_arrow_mode;
	if (p_arrow) {
		right_arrow_mode = p_mode;
	} else {
		left_arrow_mode = p_mode;
	}
	if (initial != p_mode) {
		queue_redraw();
	}
}

bool OptionButton::_is_arrow_hovered(bool p_arrow) {
	DrawMode mode = p_arrow ? right_arrow_mode : left_arrow_mode;
	return mode == DRAW_HOVER || mode == DRAW_HOVER_PRESSED;
}

bool OptionButton::_is_arrow_pressed(bool p_arrow) {
	DrawMode mode = p_arrow ? right_arrow_mode : left_arrow_mode;
	return mode == DRAW_PRESSED || mode == DRAW_HOVER_PRESSED;
}

bool OptionButton::_is_arrow_disabled(bool p_arrow) {
	DrawMode mode = p_arrow ? right_arrow_mode : left_arrow_mode;
	return mode == DRAW_DISABLED;
}

void OptionButton::_set_arrow_pressed(bool p_arrow, bool p_pressed) {
	if ((p_arrow ? right_arrow_mode : left_arrow_mode) == DRAW_DISABLED) {
		return;
	}
	DrawMode hover = p_pressed ? DRAW_HOVER_PRESSED : DRAW_HOVER;
	DrawMode no_hover = p_pressed ? DRAW_PRESSED : DRAW_NORMAL;
	_set_arrow_mode(p_arrow, _is_arrow_hovered(p_arrow) ? hover : no_hover);
}

void OptionButton::_set_arrow_hovered(bool p_arrow, bool p_hovered) {
	if ((p_arrow ? right_arrow_mode : left_arrow_mode) == DRAW_DISABLED) {
		return;
	}
	DrawMode press = p_hovered ? DRAW_HOVER_PRESSED : DRAW_PRESSED;
	DrawMode no_press = p_hovered ? DRAW_HOVER : DRAW_NORMAL;
	_set_arrow_mode(p_arrow, _is_arrow_pressed(p_arrow) ? press : no_press);
}

void OptionButton::_set_arrow_disabled(bool p_arrow, bool p_disabled) {
	DrawMode initial = p_arrow ? right_arrow_mode : left_arrow_mode;
	DrawMode not_disabled = DRAW_NORMAL;
	if (initial != DRAW_DISABLED) {
		not_disabled = initial;
	}
	_set_arrow_mode(p_arrow, p_disabled ? DRAW_DISABLED : not_disabled);
}

Ref<Texture2D> OptionButton::_get_arrow_icon(bool p_arrow) {
	if (p_arrow) {
		switch (right_arrow_mode) {
			default:
			case DRAW_NORMAL:
				return theme_cache.right_arrow_normal_icon;
			case DRAW_HOVER:
				return theme_cache.right_arrow_hover_icon;
			case DRAW_HOVER_PRESSED:
				return theme_cache.right_arrow_hover_pressed_icon;
			case DRAW_PRESSED:
				return theme_cache.right_arrow_pressed_icon;
			case DRAW_DISABLED:
				return theme_cache.right_arrow_disabled_icon;
		}
	} else {
		switch (left_arrow_mode) {
			default:
			case DRAW_NORMAL:
				return theme_cache.left_arrow_normal_icon;
			case DRAW_HOVER:
				return theme_cache.left_arrow_hover_icon;
			case DRAW_HOVER_PRESSED:
				return theme_cache.left_arrow_hover_pressed_icon;
			case DRAW_PRESSED:
				return theme_cache.left_arrow_pressed_icon;
			case DRAW_DISABLED:
				return theme_cache.left_arrow_disabled_icon;
		}
	}
}

void OptionButton::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (type == DROPDOWN) {
		Button::gui_input(p_event);
		return;
	}

	if (is_disabled()) {
		Button::gui_input(p_event);
		return;
	}

	Ref<InputEventMouseMotion> mouse_motion = p_event;
	if (mouse_motion.is_valid()) {
		Vector2 pos = mouse_motion->get_position();
		_set_arrow_hovered(LEFT, _is_over_arrow(LEFT, pos));
		_set_arrow_hovered(RIGHT, _is_over_arrow(RIGHT, pos));
		return;
	}

	bool pressed_left = p_event->is_action("ui_left") && !p_event->is_echo();
	bool pressed_right = p_event->is_action("ui_right") && !p_event->is_echo();

	if (pressed_left || pressed_right) {
		bool pressed = p_event->is_pressed();

		_set_arrow_pressed(LEFT, pressed_left && pressed);
		_set_arrow_pressed(RIGHT, pressed_right && pressed);

		if (pressed) {
			// we only want to register the press when we release
			if (get_action_mode() == ACTION_MODE_BUTTON_RELEASE) {
				// required so that focus can leave if an arrow is disabled
				if (!((pressed_left && _is_arrow_disabled(LEFT)) || (pressed_right && _is_arrow_disabled(RIGHT)))) {
					accept_event();
				}
				return;
			}
		} else {
			if (get_action_mode() == ACTION_MODE_BUTTON_PRESS) {
				accept_event();
				return;
			}
		}
	}

	Ref<InputEventMouseButton> mouse_button = p_event;
	bool button_masked = mouse_button.is_valid() &&
			((int)mouse_button_to_mask(mouse_button->get_button_index()) & (int)get_button_mask()) != 0;

	if (button_masked) {
		current_mouse_button = mouse_button->get_button_index();
		bool pressed = mouse_button->is_pressed();
		if (_is_over_arrow(LEFT, mouse_button->get_position())) {
			pressed_left = true;
		}
		if (_is_over_arrow(RIGHT, mouse_button->get_position())) {
			pressed_right = true;
		}
		_set_arrow_pressed(LEFT, pressed_left && pressed);
		_set_arrow_pressed(RIGHT, pressed_right && pressed);
		if (get_action_mode() == ACTION_MODE_BUTTON_PRESS) {
			pressed = !pressed;
		}
		if (pressed && (pressed_left || pressed_right)) {
			// wait for release
			accept_event();
			return;
		}
	}

	if (pressed_left) {
		accept_event();
		if (!_is_arrow_disabled(LEFT)) {
			_on_left_pressed();
		}
	} else if (pressed_right) {
		accept_event();
		if (!_is_arrow_disabled(RIGHT)) {
			_on_right_pressed();
		}
	} else {
		Button::gui_input(p_event);
	}
}

void OptionButton::_on_left_pressed() {
	if (is_layout_rtl()) {
		_select_next(true);
	} else {
		_select_previous(true);
	}
}

void OptionButton::_on_right_pressed() {
	if (is_layout_rtl()) {
		_select_previous(true);
	} else {
		_select_next(true);
	}
}

void OptionButton::_select_previous(bool p_emit) {
	int count = get_item_count();
	if (carousel_wraparound) {
		if (!has_selectable_items()) {
			return;
		}
		int start_idx = current == NONE_SELECTED ? 0 : current;
		for (int i = (start_idx - 1 + count) % count; i != start_idx; i = (i - 1 + count) % count) {
			if (!popup->is_item_disabled(i) && !popup->is_item_separator(i)) {
				_select(i, p_emit);
				return;
			}
		}
	} else {
		for (int i = current - 1; i >= 0; i--) {
			if (!popup->is_item_disabled(i) && !popup->is_item_separator(i)) {
				_select(i, p_emit);
				return;
			}
		}
	}
}

void OptionButton::_select_next(bool p_emit) {
	int count = get_item_count();
	if (carousel_wraparound) {
		if (!has_selectable_items()) {
			return;
		}
		int start_idx = current;
		for (int i = (start_idx + 1) % count; i != start_idx; i = (i + 1) % count) {
			if (!popup->is_item_disabled(i) && !popup->is_item_separator(i)) {
				_select(i, p_emit);
				return;
			}
		}
	} else {
		for (int i = current + 1; i < count; i++) {
			if (!popup->is_item_disabled(i) && !popup->is_item_separator(i)) {
				_select(i, p_emit);
				return;
			}
		}
	}
}

bool OptionButton::_has_next_selectable_item() {
	if (carousel_wraparound) {
		return has_selectable_items();
	} else {
		for (int i = current + 1; i < get_item_count(); i++) {
			if (!popup->is_item_disabled(i) && !popup->is_item_separator(i)) {
				return true;
			}
		}
		return false;
	}
}

bool OptionButton::_has_previous_selectable_item() {
	if (carousel_wraparound) {
		return has_selectable_items();
	} else {
		for (int i = current - 1; i >= 0; i--) {
			if (!popup->is_item_disabled(i) && !popup->is_item_separator(i)) {
				return true;
			}
		}
		return false;
	}
}

OptionButton::~OptionButton() {
}
