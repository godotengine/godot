/*************************************************************************/
/*  popup.cpp                                                            */
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

#include "popup.h"

#include "core/engine.h"
#include "core/os/keyboard.h"

void Popup::_gui_input(Ref<InputEvent> p_event) {
}

void Popup::_notification(int p_what) {
	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		if (popped_up && !is_visible_in_tree()) {
			popped_up = false;
			notification(NOTIFICATION_POPUP_HIDE);
			emit_signal("popup_hide");
		}

		update_configuration_warning();
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (popped_up) {
			popped_up = false;
			notification(NOTIFICATION_POPUP_HIDE);
			emit_signal("popup_hide");
		}
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
//small helper to make editing of these easier in editor
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
			//edited on editor
			set_as_toplevel(false);
		} else
#endif
				if (is_visible()) {
			hide();
		}
	}
}

void Popup::_fix_size() {
	Point2 pos = get_global_position();
	Size2 size = get_size() * get_scale();
	Point2 window_size = get_viewport_rect().size - get_viewport_transform().get_origin();

	if (pos.x + size.width > window_size.width) {
		pos.x = window_size.width - size.width;
	}
	if (pos.x < 0) {
		pos.x = 0;
	}

	if (pos.y + size.height > window_size.height) {
		pos.y = window_size.height - size.height;
	}
	if (pos.y < 0) {
		pos.y = 0;
	}
	if (pos != get_position()) {
		set_global_position(pos);
	}
}

void Popup::set_as_minsize() {
	Size2 total_minsize;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}
		if (!c->is_visible()) {
			continue;
		}

		Size2 minsize = c->get_combined_minimum_size();

		for (int j = 0; j < 2; j++) {
			Margin m_beg = Margin(0 + j);
			Margin m_end = Margin(2 + j);

			float margin_begin = c->get_margin(m_beg);
			float margin_end = c->get_margin(m_end);
			float anchor_begin = c->get_anchor(m_beg);
			float anchor_end = c->get_anchor(m_end);

			minsize[j] += margin_begin * (ANCHOR_END - anchor_begin) + margin_end * anchor_end;
		}

		total_minsize.width = MAX(total_minsize.width, minsize.width);
		total_minsize.height = MAX(total_minsize.height, minsize.height);
	}

	set_size(total_minsize);
}

void Popup::popup_centered_clamped(const Size2 &p_size, float p_fallback_ratio) {
	Size2 popup_size = p_size;
	Size2 window_size = get_viewport_rect().size;

	// clamp popup size in each dimension if window size is too small (using fallback ratio)
	popup_size.x = MIN(window_size.x * p_fallback_ratio, popup_size.x);
	popup_size.y = MIN(window_size.y * p_fallback_ratio, popup_size.y);

	popup_centered(popup_size);
}

void Popup::popup_centered_minsize(const Size2 &p_minsize) {
	set_custom_minimum_size(p_minsize);
	_fix_size();
	popup_centered();
}

void Popup::popup_centered(const Size2 &p_size) {
	Rect2 rect;
	Size2 window_size = get_viewport_rect().size;
	rect.size = p_size == Size2() ? get_size() : p_size;
	rect.position = ((window_size - rect.size * get_scale()) / 2.0).floor();

	_popup(rect, true);
}

void Popup::popup_centered_ratio(float p_screen_ratio) {
	Rect2 rect;
	Size2 window_size = get_viewport_rect().size;
	rect.size = (window_size * p_screen_ratio).floor();
	rect.position = ((window_size - rect.size * get_scale()) / 2.0).floor();

	_popup(rect, true);
}

void Popup::popup(const Rect2 &p_bounds) {
	_popup(p_bounds);
}

void Popup::_popup(const Rect2 &p_bounds, const bool p_centered) {
	emit_signal("about_to_show");
	show_modal(exclusive);

	// Fit the popup into the optionally provided bounds.
	if (!p_bounds.has_no_area()) {
		set_size(p_bounds.size);

		// check if p_bounds.size was using an outdated cached values
		if (p_centered && p_bounds.size != get_size()) {
			set_position(p_bounds.position - ((get_size() - p_bounds.size) / 2.0).floor());
		} else {
			set_position(p_bounds.position);
		}
	}
	_fix_size();

	Control *focusable = find_next_valid_focus();

	if (focusable) {
		focusable->grab_focus();
	}

	_post_popup();
	notification(NOTIFICATION_POST_POPUP);
	popped_up = true;
}

void Popup::set_exclusive(bool p_exclusive) {
	exclusive = p_exclusive;
	if (popped_up) {
		set_modal_exclusive(exclusive);
	}
}

bool Popup::is_exclusive() const {
	return exclusive;
}

void Popup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_as_minsize"), &Popup::set_as_minsize);
	ClassDB::bind_method(D_METHOD("popup_centered", "size"), &Popup::popup_centered, DEFVAL(Size2()));
	ClassDB::bind_method(D_METHOD("popup_centered_ratio", "ratio"), &Popup::popup_centered_ratio, DEFVAL(0.75));
	ClassDB::bind_method(D_METHOD("popup_centered_minsize", "minsize"), &Popup::popup_centered_minsize, DEFVAL(Size2()));
	ClassDB::bind_method(D_METHOD("popup_centered_clamped", "size", "fallback_ratio"), &Popup::popup_centered_clamped, DEFVAL(Size2()), DEFVAL(0.75));
	ClassDB::bind_method(D_METHOD("popup", "bounds"), &Popup::popup, DEFVAL(Rect2()));
	ClassDB::bind_method(D_METHOD("set_exclusive", "enable"), &Popup::set_exclusive);
	ClassDB::bind_method(D_METHOD("is_exclusive"), &Popup::is_exclusive);
	ADD_SIGNAL(MethodInfo("about_to_show"));
	ADD_SIGNAL(MethodInfo("popup_hide"));
	ADD_GROUP("Popup", "popup_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "popup_exclusive"), "set_exclusive", "is_exclusive");

	BIND_CONSTANT(NOTIFICATION_POST_POPUP);
	BIND_CONSTANT(NOTIFICATION_POPUP_HIDE);
}

Popup::Popup() {
	set_as_toplevel(true);
	exclusive = false;
	popped_up = false;
	hide();
}

String Popup::get_configuration_warning() const {
	String warning = Control::get_configuration_warning();
	if (is_visible_in_tree()) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("Popups will hide by default unless you call popup() or any of the popup*() functions. Making them visible for editing is fine, but they will hide upon running.");
	}

	return warning;
}

Popup::~Popup() {
}

Size2 PopupPanel::get_minimum_size() const {
	Ref<StyleBox> p = get_stylebox("panel");

	Size2 ms;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}

		if (c->is_set_as_toplevel()) {
			continue;
		}

		Size2 cms = c->get_combined_minimum_size();
		ms.x = MAX(cms.x, ms.x);
		ms.y = MAX(cms.y, ms.y);
	}

	return ms + p->get_minimum_size();
}

void PopupPanel::_update_child_rects() {
	Ref<StyleBox> p = get_stylebox("panel");

	Vector2 cpos(p->get_offset());
	Vector2 csize(get_size() - p->get_minimum_size());

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}

		if (c->is_set_as_toplevel()) {
			continue;
		}

		c->set_position(cpos);
		c->set_size(csize);
	}
}

void PopupPanel::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		get_stylebox("panel")->draw(get_canvas_item(), Rect2(Point2(), get_size()));
	} else if (p_what == NOTIFICATION_READY) {
		_update_child_rects();
	} else if (p_what == NOTIFICATION_RESIZED) {
		_update_child_rects();
	}
}

PopupPanel::PopupPanel() {
}
