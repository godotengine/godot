/*************************************************************************/
/*  carousel_button.cpp                                                  */
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

#include "carousel_button.h"

void CarouselButton::add_item(const String &p_item_name, int p_idx) {
	if (p_idx < 0) {
		p_idx += items.size();
	}
	ERR_FAIL_INDEX_MSG(p_idx, items.size() + 1, "Index out of range.");

	CarouselButtonItem new_item;
	new_item.text = p_item_name;
	items.insert(p_idx, new_item);
	if (p_idx < selected) {
		selected += 1;
	}
	update_selected();
}

void CarouselButton::remove_item(int p_idx) {
	if (p_idx < 0) {
		p_idx += items.size();
	}
	ERR_FAIL_INDEX_MSG(p_idx, items.size(), "Index out of range.");

	items.remove_at(p_idx);
	if (p_idx < selected) {
		selected -= 1;
	}
	update_selected();
}

void CarouselButton::set_item_text(int p_idx, const String &p_text) {
	if (p_idx < 0) {
		p_idx += items.size();
	}
	ERR_FAIL_INDEX_MSG(p_idx, items.size(), "Index out of range.");

	CarouselButtonItem item = items.get(p_idx);
	item.text = p_text;
	items.set(p_idx, item);
	update_selected();
	if (fit_to_longest_item) {
		update_minimum_size();
	}
}

String CarouselButton::get_item_text(int p_idx) const {
	if (p_idx < 0) {
		p_idx += items.size();
	}
	ERR_FAIL_INDEX_V_MSG(p_idx, items.size(), "", "Index out of range.");
	return items[p_idx].text;
}

void CarouselButton::set_item_icon(int p_idx, const Ref<Texture2D> &p_icon) {
	if (p_idx < 0) {
		p_idx += items.size();
	}
	ERR_FAIL_INDEX_MSG(p_idx, items.size(), "Index out of range.");
	CarouselButtonItem item = items.get(p_idx);
	item.icon = p_icon;
	items.set(p_idx, item);
	update_selected();
	if (fit_to_longest_item) {
		update_minimum_size();
	}
}

Ref<Texture2D> CarouselButton::get_item_icon(int p_idx) const {
	if (p_idx < 0) {
		p_idx += items.size();
	}
	ERR_FAIL_INDEX_V_MSG(p_idx, items.size(), nullptr, "Index out of range.");
	return items[p_idx].icon;
}

bool CarouselButton::update_selected() {
	if (items.size() == 0) {
		set_text("");
		set_icon(nullptr);
		return false;
	}
	if (wraparound) {
		selected = selected % items.size();
		if (selected < 0) {
			selected += items.size();
		}
	} else {
		selected = CLAMP(selected, 0, items.size() - 1);
		// disable the arrows if we are at an edge
		set_arrow_disabled(false, selected == 0);
		set_arrow_disabled(true, selected == items.size() - 1);
	}
	set_text(items[selected].text);
	set_icon(items[selected].icon);
	return true;
}

void CarouselButton::pressed() {
	if (current_mouse_button == MouseButton::RIGHT) {
		select(selected - 1);
	} else {
		select(selected + 1);
	}
}

void CarouselButton::select(int p_idx) {
	int initial = selected;
	selected = p_idx;
	if (update_selected()) {
		if (is_inside_tree()) {
			emit_signal(SNAME("item_selected"), selected);
		}
	} else {
		selected = initial;
	}
}

int CarouselButton::get_selected() {
	return selected;
}

void CarouselButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_item", "name", "index"), &CarouselButton::add_item, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_item", "index"), &CarouselButton::remove_item, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("select", "index"), &CarouselButton::select);
	ClassDB::bind_method(D_METHOD("get_selected"), &CarouselButton::get_selected);
	ClassDB::bind_method(D_METHOD("set_wraparound", "wraparound"), &CarouselButton::set_wraparound);
	ClassDB::bind_method(D_METHOD("get_wraparound"), &CarouselButton::get_wraparound);
	ClassDB::bind_method(D_METHOD("set_fit_to_longest_item", "fit_to_longest_item"), &CarouselButton::set_fit_to_longest_item);
	ClassDB::bind_method(D_METHOD("get_fit_to_longest_item"), &CarouselButton::get_fit_to_longest_item);

	ClassDB::bind_method(D_METHOD("set_item_count", "count"), &CarouselButton::set_item_count);
	ClassDB::bind_method(D_METHOD("get_item_count"), &CarouselButton::get_item_count);

	ClassDB::bind_method(D_METHOD("set_item_text", "index", "text"), &CarouselButton::set_item_text);
	ClassDB::bind_method(D_METHOD("get_item_text", "index"), &CarouselButton::get_item_text);

	ClassDB::bind_method(D_METHOD("set_item_icon", "index", "icon"), &CarouselButton::set_item_icon);
	ClassDB::bind_method(D_METHOD("get_item_icon", "index"), &CarouselButton::get_item_icon);

	ADD_ARRAY_COUNT("Items", "item_count", "set_item_count", "get_item_count", "item_");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "selected"), "select", "get_selected");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "wraparound"), "set_wraparound", "get_wraparound");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fit_to_longest_item"), "set_fit_to_longest_item", "get_fit_to_longest_item");

	ADD_SIGNAL(MethodInfo("item_selected", PropertyInfo(Variant::INT, "index")));
}

void CarouselButton::_validate_property(PropertyInfo &property) const {
	if (property.name == "text" || property.name == "icon" || property.name == "toggle_mode" || property.name == "button_pressed") {
		property.usage = PROPERTY_USAGE_NONE;
	}
}

void CarouselButton::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < items.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, vformat("item_%d/text", i)));

		PropertyInfo pi = PropertyInfo(Variant::OBJECT, vformat("item_%d/icon", i), PROPERTY_HINT_RESOURCE_TYPE, "Texture2D");
		p_list->push_back(pi);
	}
}

bool CarouselButton::_set(const StringName &p_name, const Variant &p_value) {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() >= 2 && components[0].begins_with("item_")) {
		String property = components[1];
		if (property != "text" && property != "icon") {
			return false;
		}
		int idx = components[0].trim_prefix("item_").to_int();
		if (property == "text") {
			if (p_value.get_type() == Variant::STRING) {
				set_item_text(idx, p_value);
				return true;
			} else {
				return false;
			}
		}
		if (property == "icon") {
			if (p_value.get_type() == Variant::OBJECT && (p_value.is_null() || (*p_value).is_class("Texture2D"))) {
				set_item_icon(idx, p_value);
				return true;
			} else {
				return false;
			}
		}
	}
	return false;
}

bool CarouselButton::_get(const StringName &p_name, Variant &r_ret) const {
	Vector<String> components = String(p_name).split("/", true, 2);
	if (components.size() >= 2 && components[0].begins_with("item_")) {
		String property = components[1];
		if (property != "text" && property != "icon") {
			return false;
		}
		int idx = components[0].trim_prefix("item_").to_int();
		if (property == "text") {
			r_ret = get_item_text(idx);
			return true;
		}
		if (property == "icon") {
			r_ret = get_item_icon(idx);
			return true;
		}
	}
	return false;
}

int CarouselButton::get_item_count() {
	return items.size();
}

void CarouselButton::set_item_count(int p_count) {
	while (p_count > items.size()) {
		CarouselButtonItem new_item;
		new_item.text = "";
		items.append(new_item);
	}
	while (p_count < items.size()) {
		items.remove_at(items.size() - 1);
	}
	notify_property_list_changed();
}

void CarouselButton::_on_left_pressed() {
	select(selected - 1);
}

void CarouselButton::_on_right_pressed() {
	select(selected + 1);
}

void CarouselButton::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (is_disabled()) {
		set_arrow_disabled(true, true);
		set_arrow_disabled(false, true);
		Button::gui_input(p_event);
		return;
	}

	Ref<InputEventMouseMotion> mouse_motion = p_event;
	if (mouse_motion.is_valid()) {
		Vector2 pos = mouse_motion->get_position();

		set_arrow_hovered(true, is_over_arrow(true, pos));
		set_arrow_hovered(false, is_over_arrow(false, pos));

		return;
	}

	bool pressed_left = p_event->is_action("ui_left") && !p_event->is_echo();
	bool pressed_right = p_event->is_action("ui_right") && !p_event->is_echo();

	if (pressed_left || pressed_right) {
		bool pressed = p_event->is_pressed();

		set_arrow_pressed(false, pressed_left && pressed);
		set_arrow_pressed(true, pressed_right && pressed);

		if (pressed) {
			// we only want to register the press when we release
			if (get_action_mode() == ACTION_MODE_BUTTON_RELEASE) {
				// required so that focus can leave if an arrow is disabled
				if (!((pressed_left && arrow_disabled(false)) || (pressed_right && arrow_disabled(true)))) {
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
	bool button_masked = mouse_button.is_valid() && (mouse_button_to_mask(mouse_button->get_button_index()) & get_button_mask()) != MouseButton::NONE;

	if (button_masked) {
		current_mouse_button = mouse_button->get_button_index();
		bool pressed = mouse_button->is_pressed();
		if (is_over_arrow(false, mouse_button->get_position())) {
			pressed_left = true;
		}
		if (is_over_arrow(true, mouse_button->get_position())) {
			pressed_right = true;
		}
		set_arrow_pressed(false, pressed_left && pressed);
		set_arrow_pressed(true, pressed_right && pressed);
		if (get_action_mode() == ACTION_MODE_BUTTON_PRESS) {
			pressed = !pressed;
		}
		if (pressed && (pressed_left || pressed_right)) {
			// wait for release
			accept_event();
			return;
		}
	}

	if (pressed_left && !arrow_disabled(false)) {
		accept_event();
		_on_left_pressed();
	} else if (pressed_right && !arrow_disabled(true)) {
		accept_event();
		_on_right_pressed();
	} else {
		Button::gui_input(p_event);
	}
}

void CarouselButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_MOUSE_EXIT: {
			set_arrow_hovered(true, false);
			set_arrow_hovered(false, false);
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			// left arrow
			if (has_theme_icon(SNAME("arrow_left_normal"))) {
				Ref<Texture2D> arrow = get_arrow_texture(false);
				Size2 size = get_size();
				Point2 ofs = Point2(0.0, int(Math::abs((size.height - arrow->get_height()) / 2)));
				Color clr = get_arrow_modulate(false);
				arrow->draw(ci, ofs, clr);
			}
			// right arrow
			if (has_theme_icon(SNAME("arrow_right_normal"))) {
				Ref<Texture2D> arrow = get_arrow_texture(true);
				Size2 size = get_size();
				Point2 ofs = Point2(size.width - arrow->get_width(), int(Math::abs((size.height - arrow->get_height()) / 2)));
				Color clr = get_arrow_modulate(true);
				arrow->draw(ci, ofs, clr);
			}
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			update_internal_margin();
		} break;
	}
}

Size2 CarouselButton::get_right_arrow_size() const {
	Size2 arrow_right_size = Size2(0.0, 0.0);
	if (has_theme_icon(SNAME("arrow_right_normal"))) {
		arrow_right_size = Control::get_theme_icon(SNAME("arrow_right_normal"))->get_size();
	}
	return arrow_right_size;
}

Size2 CarouselButton::get_left_arrow_size() const {
	Size2 arrow_left_size = Size2(0.0, 0.0);
	if (has_theme_icon(SNAME("arrow_left_normal"))) {
		arrow_left_size = Control::get_theme_icon(SNAME("arrow_left_normal"))->get_size();
	}
	return arrow_left_size;
}

Ref<Texture2D> CarouselButton::get_arrow_texture(bool p_arrow) {
	if (p_arrow) {
		// right arrow
		switch (right_draw_mode) {
			case DRAW_NORMAL:
				return Control::get_theme_icon(SNAME("arrow_right_normal"));
			case DRAW_HOVER:
				return Control::get_theme_icon(SNAME("arrow_right_hover"));
			case DRAW_HOVER_PRESSED:
			case DRAW_PRESSED:
				return Control::get_theme_icon(SNAME("arrow_right_pressed"));
			default:
			case DRAW_DISABLED:
				return Control::get_theme_icon(SNAME("arrow_right_disabled"));
		}
	} else {
		// left arrow
		switch (left_draw_mode) {
			case DRAW_NORMAL:
				return Control::get_theme_icon(SNAME("arrow_left_normal"));
			case DRAW_HOVER:
				return Control::get_theme_icon(SNAME("arrow_left_hover"));
			case DRAW_HOVER_PRESSED:
			case DRAW_PRESSED:
				return Control::get_theme_icon(SNAME("arrow_left_pressed"));
			default:
			case DRAW_DISABLED:
				return Control::get_theme_icon(SNAME("arrow_left_disabled"));
		}
	}
}

Color CarouselButton::get_arrow_modulate(bool p_arrow) const {
	switch (p_arrow ? right_draw_mode : left_draw_mode) {
		case DRAW_NORMAL:
			return Control::get_theme_color(SNAME("arrow_normal_modulate"));
		case DRAW_HOVER:
			return Control::get_theme_color(SNAME("arrow_hover_modulate"));
		case DRAW_HOVER_PRESSED:
		case DRAW_PRESSED:
			return Control::get_theme_color(SNAME("arrow_pressed_modulate"));
		default:
		case DRAW_DISABLED:
			return Control::get_theme_color(SNAME("arrow_disabled_modulate"));
	}
}

// assumes the pos is already inside the control
bool CarouselButton::is_over_arrow(bool p_arrow, Vector2 p_pos) {
	if (p_arrow) {
		// right arrow
		return p_pos.x > get_size().x - get_right_arrow_size().x;
	} else {
		// left arrow
		return p_pos.x < get_left_arrow_size().x;
	}
}

bool CarouselButton::arrow_hovered(bool p_arrow) {
	DrawMode draw_mode = p_arrow ? right_draw_mode : left_draw_mode;
	return draw_mode == DRAW_HOVER || draw_mode == DRAW_HOVER_PRESSED;
}

bool CarouselButton::arrow_pressed(bool p_arrow) {
	DrawMode draw_mode = p_arrow ? right_draw_mode : left_draw_mode;
	return draw_mode == DRAW_PRESSED || draw_mode == DRAW_HOVER_PRESSED;
}

bool CarouselButton::arrow_disabled(bool p_arrow) {
	DrawMode draw_mode = p_arrow ? right_draw_mode : left_draw_mode;
	return draw_mode == DRAW_DISABLED;
}

void CarouselButton::set_arrow_pressed(bool p_arrow, bool p_pressed) {
	DrawMode initial = p_arrow ? right_draw_mode : left_draw_mode;
	if (initial == DRAW_DISABLED) {
		return;
	}
	DrawMode hover = p_pressed ? DRAW_HOVER_PRESSED : DRAW_HOVER;
	DrawMode no_hover = p_pressed ? DRAW_PRESSED : DRAW_NORMAL;
	if (p_arrow) {
		right_draw_mode = arrow_hovered(p_arrow) ? hover : no_hover;
	} else {
		left_draw_mode = arrow_hovered(p_arrow) ? hover : no_hover;
	}
	if (initial != (p_arrow ? right_draw_mode : left_draw_mode)) {
		update();
	}
}

void CarouselButton::set_arrow_hovered(bool p_arrow, bool p_hovered) {
	DrawMode initial = p_arrow ? right_draw_mode : left_draw_mode;
	if (initial == DRAW_DISABLED) {
		return;
	}
	DrawMode press = p_hovered ? DRAW_HOVER_PRESSED : DRAW_PRESSED;
	DrawMode no_press = p_hovered ? DRAW_HOVER : DRAW_NORMAL;
	if (p_arrow) {
		right_draw_mode = arrow_pressed(p_arrow) ? press : no_press;
	} else {
		left_draw_mode = arrow_pressed(p_arrow) ? press : no_press;
	}
	if (initial != (p_arrow ? right_draw_mode : left_draw_mode)) {
		update();
	}
}

void CarouselButton::set_arrow_disabled(bool p_arrow, bool p_disabled) {
	DrawMode initial = p_arrow ? right_draw_mode : left_draw_mode;
	DrawMode not_disabled = DRAW_NORMAL;
	if (initial != DRAW_DISABLED) {
		not_disabled = initial;
	}
	if (p_arrow) {
		right_draw_mode = p_disabled ? DRAW_DISABLED : not_disabled;
	} else {
		left_draw_mode = p_disabled ? DRAW_DISABLED : not_disabled;
	}
	if (initial != (p_arrow ? right_draw_mode : left_draw_mode)) {
		update();
	}
}

void CarouselButton::set_wraparound(bool p_wraparound) {
	wraparound = p_wraparound;
	update_selected();
}

bool CarouselButton::get_wraparound() {
	return wraparound;
}

void CarouselButton::set_fit_to_longest_item(bool p_fit_to_longest_item) {
	fit_to_longest_item = p_fit_to_longest_item;
	update_minimum_size();
}

bool CarouselButton::get_fit_to_longest_item() {
	return fit_to_longest_item;
}

Size2 CarouselButton::get_largest_size() const {
	Size2 largest = Size2(0.0, 0.0);
	for (int i = 0; i < items.size(); i++) {
		CarouselButtonItem item = items[i];
		Size2 s = get_minimum_size_for_text_and_icon(item.text, item.icon);
		if (s.x > largest.x) {
			largest.x = s.x;
		}
		if (s.y > largest.y) {
			largest.y = s.y;
		}
	}
	return largest;
}

Size2 CarouselButton::get_minimum_size() const {
	Size2 minsize;
	minsize = Button::get_minimum_size();

	const Size2 padding = get_theme_stylebox(SNAME("normal"))->get_minimum_size();

	Size2 arrow_left_size = get_left_arrow_size();
	Size2 arrow_right_size = get_right_arrow_size();

	Size2 content_size = minsize - padding;
	if (fit_to_longest_item) {
		content_size = get_largest_size();
	}
	content_size.width += arrow_left_size.width + arrow_right_size.width + get_theme_constant(SNAME("h_separation"));
	content_size.height = MAX(MAX(content_size.height, arrow_left_size.height), arrow_right_size.height);

	minsize = content_size + padding;

	return minsize;
}

void CarouselButton::update_internal_margin() {
	if (has_theme_icon(SNAME("arrow_left_normal"))) {
		_set_internal_margin(SIDE_LEFT, Control::get_theme_icon(SNAME("arrow_left_normal"))->get_width());
	}
	if (has_theme_icon(SNAME("arrow_right_normal"))) {
		_set_internal_margin(SIDE_RIGHT, Control::get_theme_icon(SNAME("arrow_right_normal"))->get_width());
	}
}

CarouselButton::CarouselButton() {
	set_button_mask(MouseButton::LEFT | MouseButton::RIGHT);
	update_internal_margin();
}
