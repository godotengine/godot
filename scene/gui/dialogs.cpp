/*************************************************************************/
/*  dialogs.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "dialogs.h"

#include "core/os/keyboard.h"
#include "core/print_string.h"
#include "core/translation.h"
#include "line_edit.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/main/window.h" // Only used to check for more modals when dimming the editor.
#endif

// AcceptDialog

void AcceptDialog::_input_from_window(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key = p_event;
	if (key.is_valid() && key->is_pressed() && key->get_keycode() == KEY_ESCAPE) {
		_cancel_pressed();
	}
}

void AcceptDialog::_parent_focused() {
	if (!is_exclusive()) {
		_cancel_pressed();
	}
}

void AcceptDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				get_ok()->grab_focus();
				_update_child_rects();
				parent_visible = get_parent_visible_window();
				if (parent_visible) {
					parent_visible->connect("focus_entered", callable_mp(this, &AcceptDialog::_parent_focused));
				}
			} else {
				if (parent_visible) {
					parent_visible->disconnect("focus_entered", callable_mp(this, &AcceptDialog::_parent_focused));
					parent_visible = nullptr;
				}
			}

		} break;

		case NOTIFICATION_THEME_CHANGED: {
			bg->add_theme_style_override("panel", bg->get_theme_stylebox("panel", "AcceptDialog"));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (parent_visible) {
				parent_visible->disconnect("focus_entered", callable_mp(this, &AcceptDialog::_parent_focused));
				parent_visible = nullptr;
			}
		} break;
		case NOTIFICATION_READY:
		case NOTIFICATION_WM_SIZE_CHANGED: {
			if (is_visible()) {
				_update_child_rects();
			}
		} break;
		case NOTIFICATION_WM_CLOSE_REQUEST: {
			_cancel_pressed();
		} break;
	}
}

void AcceptDialog::_text_entered(const String &p_text) {
	_ok_pressed();
}

void AcceptDialog::_ok_pressed() {
	if (hide_on_ok) {
		set_visible(false);
	}
	ok_pressed();
	emit_signal("confirmed");
}

void AcceptDialog::_cancel_pressed() {
	Window *parent_window = parent_visible;
	if (parent_visible) {
		parent_visible->disconnect("focus_entered", callable_mp(this, &AcceptDialog::_parent_focused));
		parent_visible = nullptr;
	}

	call_deferred("hide");

	emit_signal("cancelled");

	cancel_pressed();

	if (parent_window) {
		//parent_window->grab_focus();
	}
}

String AcceptDialog::get_text() const {
	return label->get_text();
}

void AcceptDialog::set_text(String p_text) {
	label->set_text(p_text);
	child_controls_changed();
	if (is_visible()) {
		_update_child_rects();
	}
}

void AcceptDialog::set_hide_on_ok(bool p_hide) {
	hide_on_ok = p_hide;
}

bool AcceptDialog::get_hide_on_ok() const {
	return hide_on_ok;
}

void AcceptDialog::set_autowrap(bool p_autowrap) {
	label->set_autowrap(p_autowrap);
}

bool AcceptDialog::has_autowrap() {
	return label->has_autowrap();
}

void AcceptDialog::register_text_enter(Node *p_line_edit) {
	ERR_FAIL_NULL(p_line_edit);
	LineEdit *line_edit = Object::cast_to<LineEdit>(p_line_edit);
	if (line_edit) {
		line_edit->connect("text_entered", callable_mp(this, &AcceptDialog::_text_entered));
	}
}

void AcceptDialog::_update_child_rects() {
	Size2 label_size = label->get_minimum_size();
	if (label->get_text().empty()) {
		label_size.height = 0;
	}
	int margin = hbc->get_theme_constant("margin", "Dialogs");
	Size2 size = get_size();
	Size2 hminsize = hbc->get_combined_minimum_size();

	Vector2 cpos(margin, margin + label_size.height);
	Vector2 csize(size.x - margin * 2, size.y - margin * 3 - hminsize.y - label_size.height);

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}

		if (c == hbc || c == label || c == bg || c->is_set_as_top_level()) {
			continue;
		}

		c->set_position(cpos);
		c->set_size(csize);
	}

	cpos.y += csize.y + margin;
	csize.y = hminsize.y;

	hbc->set_position(cpos);
	hbc->set_size(csize);

	bg->set_position(Point2());
	bg->set_size(size);
}

Size2 AcceptDialog::_get_contents_minimum_size() const {
	int margin = hbc->get_theme_constant("margin", "Dialogs");
	Size2 minsize = label->get_combined_minimum_size();

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}

		if (c == hbc || c == label || c->is_set_as_top_level()) {
			continue;
		}

		Size2 cminsize = c->get_combined_minimum_size();
		minsize.x = MAX(cminsize.x, minsize.x);
		minsize.y = MAX(cminsize.y, minsize.y);
	}

	Size2 hminsize = hbc->get_combined_minimum_size();
	minsize.x = MAX(hminsize.x, minsize.x);
	minsize.y += hminsize.y;
	minsize.x += margin * 2;
	minsize.y += margin * 3; //one as separation between hbc and child

	Size2 wmsize = get_min_size();
	minsize.x = MAX(wmsize.x, minsize.x);
	return minsize;
}

void AcceptDialog::_custom_action(const String &p_action) {
	emit_signal("custom_action", p_action);
	custom_action(p_action);
}

Button *AcceptDialog::add_button(const String &p_text, bool p_right, const String &p_action) {
	Button *button = memnew(Button);
	button->set_text(p_text);
	if (p_right) {
		hbc->add_child(button);
		hbc->add_spacer();
	} else {
		hbc->add_child(button);
		hbc->move_child(button, 0);
		hbc->add_spacer(true);
	}

	if (p_action != "") {
		button->connect("pressed", callable_mp(this, &AcceptDialog::_custom_action), varray(p_action));
	}

	return button;
}

Button *AcceptDialog::add_cancel(const String &p_cancel) {
	String c = p_cancel;
	if (p_cancel == "") {
		c = RTR("Cancel");
	}
	Button *b = swap_cancel_ok ? add_button(c, true) : add_button(c);
	b->connect("pressed", callable_mp(this, &AcceptDialog::_cancel_pressed));
	return b;
}

void AcceptDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_ok"), &AcceptDialog::get_ok);
	ClassDB::bind_method(D_METHOD("get_label"), &AcceptDialog::get_label);
	ClassDB::bind_method(D_METHOD("set_hide_on_ok", "enabled"), &AcceptDialog::set_hide_on_ok);
	ClassDB::bind_method(D_METHOD("get_hide_on_ok"), &AcceptDialog::get_hide_on_ok);
	ClassDB::bind_method(D_METHOD("add_button", "text", "right", "action"), &AcceptDialog::add_button, DEFVAL(false), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_cancel", "name"), &AcceptDialog::add_cancel);
	ClassDB::bind_method(D_METHOD("register_text_enter", "line_edit"), &AcceptDialog::register_text_enter);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &AcceptDialog::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &AcceptDialog::get_text);
	ClassDB::bind_method(D_METHOD("set_autowrap", "autowrap"), &AcceptDialog::set_autowrap);
	ClassDB::bind_method(D_METHOD("has_autowrap"), &AcceptDialog::has_autowrap);

	ADD_SIGNAL(MethodInfo("confirmed"));
	ADD_SIGNAL(MethodInfo("cancelled"));
	ADD_SIGNAL(MethodInfo("custom_action", PropertyInfo(Variant::STRING_NAME, "action")));

	ADD_GROUP("Dialog", "dialog");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "dialog_text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dialog_hide_on_ok"), "set_hide_on_ok", "get_hide_on_ok");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dialog_autowrap"), "set_autowrap", "has_autowrap");
}

bool AcceptDialog::swap_cancel_ok = false;
void AcceptDialog::set_swap_cancel_ok(bool p_swap) {
	swap_cancel_ok = p_swap;
}

AcceptDialog::AcceptDialog() {
	parent_visible = nullptr;

	set_wrap_controls(true);
	set_visible(false);
	set_transient(true);
	set_exclusive(true);
	set_clamp_to_embedder(true);

	bg = memnew(Panel);
	add_child(bg);

	hbc = memnew(HBoxContainer);

	int margin = hbc->get_theme_constant("margin", "Dialogs");
	int button_margin = hbc->get_theme_constant("button_margin", "Dialogs");

	label = memnew(Label);
	label->set_anchor(MARGIN_RIGHT, Control::ANCHOR_END);
	label->set_anchor(MARGIN_BOTTOM, Control::ANCHOR_END);
	label->set_begin(Point2(margin, margin));
	label->set_end(Point2(-margin, -button_margin - 10));
	add_child(label);

	add_child(hbc);

	hbc->add_spacer();
	ok = memnew(Button);
	ok->set_text(RTR("OK"));
	hbc->add_child(ok);
	hbc->add_spacer();

	ok->connect("pressed", callable_mp(this, &AcceptDialog::_ok_pressed));

	hide_on_ok = true;
	set_title(RTR("Alert!"));

	connect("window_input", callable_mp(this, &AcceptDialog::_input_from_window));
}

AcceptDialog::~AcceptDialog() {
}

// ConfirmationDialog

void ConfirmationDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_cancel"), &ConfirmationDialog::get_cancel);
}

Button *ConfirmationDialog::get_cancel() {
	return cancel;
}

ConfirmationDialog::ConfirmationDialog() {
	set_title(RTR("Please Confirm..."));
#ifdef TOOLS_ENABLED
	set_min_size(Size2(200, 70) * EDSCALE);
#endif
	cancel = add_cancel();
}
