/*************************************************************************/
/*  popup.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "os/keyboard.h"

void Popup::_gui_input(InputEvent p_event) {
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

	if (p_what == NOTIFICATION_ENTER_TREE) {
//small helper to make editing of these easier in editor
#ifdef TOOLS_ENABLED
		if (get_tree()->is_editor_hint() && get_tree()->get_edited_scene_root() && get_tree()->get_edited_scene_root()->is_a_parent_of(this)) {
			set_as_toplevel(false);
		}
#endif
	}
}

void Popup::_fix_size() {

#if 0
	Point2 pos = get_pos();
	Size2 size = get_size();
	Point2 window_size = window==this ? get_parent_area_size()  :window->get_size();
#else

	Point2 pos = get_global_pos();
	Size2 size = get_size();
	Point2 window_size = get_viewport_rect().size;

#endif
	if (pos.x + size.width > window_size.width)
		pos.x = window_size.width - size.width;
	if (pos.x < 0)
		pos.x = 0;

	if (pos.y + size.height > window_size.height)
		pos.y = window_size.height - size.height;
	if (pos.y < 0)
		pos.y = 0;
#if 0
	if (pos!=get_pos())
		set_pos(pos);
#else
	if (pos != get_pos())
		set_global_pos(pos);

#endif
}

void Popup::set_as_minsize() {

	Size2 total_minsize;

	for (int i = 0; i < get_child_count(); i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (!c->is_visible())
			continue;

		Size2 minsize = c->get_combined_minimum_size();

		for (int j = 0; j < 2; j++) {

			Margin m_beg = Margin(0 + j);
			Margin m_end = Margin(2 + j);

			float margin_begin = c->get_margin(m_beg);
			float margin_end = c->get_margin(m_end);
			AnchorType anchor_begin = c->get_anchor(m_beg);
			AnchorType anchor_end = c->get_anchor(m_end);

			if (anchor_begin == ANCHOR_BEGIN)
				minsize[j] += margin_begin;
			if (anchor_end == ANCHOR_END)
				minsize[j] += margin_end;
		}

		total_minsize.width = MAX(total_minsize.width, minsize.width);
		total_minsize.height = MAX(total_minsize.height, minsize.height);
	}

	set_size(total_minsize);
}

void Popup::popup_centered_minsize(const Size2 &p_minsize) {

	Size2 total_minsize = p_minsize;

	for (int i = 0; i < get_child_count(); i++) {

		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;
		if (!c->is_visible())
			continue;

		Size2 minsize = c->get_combined_minimum_size();

		for (int j = 0; j < 2; j++) {

			Margin m_beg = Margin(0 + j);
			Margin m_end = Margin(2 + j);

			float margin_begin = c->get_margin(m_beg);
			float margin_end = c->get_margin(m_end);
			AnchorType anchor_begin = c->get_anchor(m_beg);
			AnchorType anchor_end = c->get_anchor(m_end);

			if (anchor_begin == ANCHOR_BEGIN)
				minsize[j] += margin_begin;
			if (anchor_end == ANCHOR_END)
				minsize[j] += margin_end;
		}

		total_minsize.width = MAX(total_minsize.width, minsize.width);
		total_minsize.height = MAX(total_minsize.height, minsize.height);
	}

	popup_centered(total_minsize);
	popped_up = true;
}

void Popup::popup_centered(const Size2 &p_size) {

	Point2 window_size = get_viewport_rect().size;

	emit_signal("about_to_show");
	Rect2 rect;
	rect.size = p_size == Size2() ? get_size() : p_size;

	rect.pos = ((window_size - rect.size) / 2.0).floor();
	set_pos(rect.pos);
	set_size(rect.size);

	show_modal(exclusive);
	_fix_size();

	Control *focusable = find_next_valid_focus();
	if (focusable)
		focusable->grab_focus();

	_post_popup();
	notification(NOTIFICATION_POST_POPUP);
	popped_up = true;
}

void Popup::popup_centered_ratio(float p_screen_ratio) {

	emit_signal("about_to_show");

	Rect2 rect;
	Point2 window_size = get_viewport_rect().size;
	rect.size = (window_size * p_screen_ratio).floor();
	rect.pos = ((window_size - rect.size) / 2.0).floor();
	set_pos(rect.pos);
	set_size(rect.size);

	show_modal(exclusive);
	_fix_size();

	Control *focusable = find_next_valid_focus();
	if (focusable)
		focusable->grab_focus();

	_post_popup();
	notification(NOTIFICATION_POST_POPUP);
	popped_up = true;
}

void Popup::popup(const Rect2 &bounds) {

	emit_signal("about_to_show");
	show_modal(exclusive);

	// Fit the popup into the optionally provided bounds.
	if (!bounds.has_no_area()) {
		set_pos(bounds.pos);
		set_size(bounds.size);
	}
	_fix_size();

	Control *focusable = find_next_valid_focus();

	if (focusable)
		focusable->grab_focus();

	_post_popup();
	notification(NOTIFICATION_POST_POPUP);
	popped_up = true;
}

void Popup::set_exclusive(bool p_exclusive) {

	exclusive = p_exclusive;
}

bool Popup::is_exclusive() const {

	return exclusive;
}

void Popup::_bind_methods() {

	ClassDB::bind_method(D_METHOD("popup_centered", "size"), &Popup::popup_centered, DEFVAL(Size2()));
	ClassDB::bind_method(D_METHOD("popup_centered_ratio", "ratio"), &Popup::popup_centered_ratio, DEFVAL(0.75));
	ClassDB::bind_method(D_METHOD("popup_centered_minsize", "minsize"), &Popup::popup_centered_minsize, DEFVAL(Size2()));
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

	if (is_visible_in_tree()) {
		return TTR("Popups will hide by default unless you call popup() or any of the popup*() functions. Making them visible for editing is fine though, but they will hide upon running.");
	}

	return String();
}

Popup::~Popup() {
}

void PopupPanel::set_child_rect(Control *p_child) {
	ERR_FAIL_NULL(p_child);

	Ref<StyleBox> p = get_stylebox("panel");
	p_child->set_area_as_parent_rect();
	for (int i = 0; i < 4; i++) {
		p_child->set_margin(Margin(i), p->get_margin(Margin(i)));
	}
}

void PopupPanel::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		get_stylebox("panel")->draw(get_canvas_item(), Rect2(Point2(), get_size()));
	}
}

PopupPanel::PopupPanel() {
}
