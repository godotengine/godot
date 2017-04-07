/*************************************************************************/
/*  dialogs.cpp                                                          */
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
#include "dialogs.h"
#include "line_edit.h"
#include "print_string.h"
#include "translation.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

void WindowDialog::_post_popup() {

	drag_type = DRAG_NONE; // just in case
}

void WindowDialog::_fix_size() {

	// Perhaps this should be called when the viewport resizes as well or windows go out of bounds...

	// Ensure the whole window is visible.
	Point2i pos = get_global_pos();
	Size2i size = get_size();
	Size2i viewport_size = get_viewport_rect().size;

	// Windows require additional padding to keep the window chrome visible.
	Ref<StyleBox> panel = get_stylebox("panel", "WindowDialog");
	float top = panel->get_margin(MARGIN_TOP);
	float left = panel->get_margin(MARGIN_LEFT);
	float bottom = panel->get_margin(MARGIN_BOTTOM);
	float right = panel->get_margin(MARGIN_RIGHT);

	pos.x = MAX(left, MIN(pos.x, viewport_size.x - size.x - right));
	pos.y = MAX(top, MIN(pos.y, viewport_size.y - size.y - bottom));
	set_global_pos(pos);

	// Also resize the window to fit if a resize should be possible at all.
	if (resizable) {
		size.x = MIN(size.x, viewport_size.x - left - right);
		size.y = MIN(size.y, viewport_size.y - top - bottom);
		set_size(size);
	}
}

bool WindowDialog::has_point(const Point2 &p_point) const {

	Rect2 r(Point2(), get_size());

	// Enlarge upwards for title bar.
	int titlebar_height = get_constant("titlebar_height", "WindowDialog");
	r.pos.y -= titlebar_height;
	r.size.y += titlebar_height;

	// Inflate by the resizable border thickness.
	if (resizable) {
		int scaleborder_size = get_constant("scaleborder_size", "WindowDialog");
		r.pos.x -= scaleborder_size;
		r.size.width += scaleborder_size * 2;
		r.pos.y -= scaleborder_size;
		r.size.height += scaleborder_size * 2;
	}

	return r.has_point(p_point);
}

void WindowDialog::_gui_input(const InputEvent &p_event) {

	if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.button_index == BUTTON_LEFT) {

		if (p_event.mouse_button.pressed) {
			// Begin a possible dragging operation.
			drag_type = _drag_hit_test(Point2(p_event.mouse_button.x, p_event.mouse_button.y));
			if (drag_type != DRAG_NONE)
				drag_offset = get_global_mouse_pos() - get_pos();
			drag_offset_far = get_pos() + get_size() - get_global_mouse_pos();
		} else if (drag_type != DRAG_NONE && !p_event.mouse_button.pressed) {
			// End a dragging operation.
			drag_type = DRAG_NONE;
		}
	}

	if (p_event.type == InputEvent::MOUSE_MOTION) {

		if (drag_type == DRAG_NONE) {
			// Update the cursor while moving along the borders.
			CursorShape cursor = CURSOR_ARROW;
			if (resizable) {
				int preview_drag_type = _drag_hit_test(Point2(p_event.mouse_button.x, p_event.mouse_button.y));
				switch (preview_drag_type) {
					case DRAG_RESIZE_TOP:
					case DRAG_RESIZE_BOTTOM:
						cursor = CURSOR_VSIZE;
						break;
					case DRAG_RESIZE_LEFT:
					case DRAG_RESIZE_RIGHT:
						cursor = CURSOR_HSIZE;
						break;
					case DRAG_RESIZE_TOP + DRAG_RESIZE_LEFT:
					case DRAG_RESIZE_BOTTOM + DRAG_RESIZE_RIGHT:
						cursor = CURSOR_FDIAGSIZE;
						break;
					case DRAG_RESIZE_TOP + DRAG_RESIZE_RIGHT:
					case DRAG_RESIZE_BOTTOM + DRAG_RESIZE_LEFT:
						cursor = CURSOR_BDIAGSIZE;
						break;
				}
			}
			if (get_cursor_shape() != cursor)
				set_default_cursor_shape(cursor);
		} else {
			// Update while in a dragging operation.
			Point2 global_pos = get_global_mouse_pos();
			global_pos.y = MAX(global_pos.y, 0); // Ensure title bar stays visible.

			Rect2 rect = get_rect();
			Size2 min_size = get_minimum_size();

			if (drag_type == DRAG_MOVE) {
				rect.pos = global_pos - drag_offset;
			} else {
				if (drag_type & DRAG_RESIZE_TOP) {
					int bottom = rect.pos.y + rect.size.height;
					int max_y = bottom - min_size.height;
					rect.pos.y = MIN(global_pos.y - drag_offset.y, max_y);
					rect.size.height = bottom - rect.pos.y;
				} else if (drag_type & DRAG_RESIZE_BOTTOM) {
					rect.size.height = global_pos.y - rect.pos.y + drag_offset_far.y;
				}
				if (drag_type & DRAG_RESIZE_LEFT) {
					int right = rect.pos.x + rect.size.width;
					int max_x = right - min_size.width;
					rect.pos.x = MIN(global_pos.x - drag_offset.x, max_x);
					rect.size.width = right - rect.pos.x;
				} else if (drag_type & DRAG_RESIZE_RIGHT) {
					rect.size.width = global_pos.x - rect.pos.x + drag_offset_far.x;
				}
			}

			set_size(rect.size);
			set_pos(rect.pos);
		}
	}
}

void WindowDialog::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_DRAW: {

			RID canvas = get_canvas_item();
			Size2 size = get_size();

			Ref<StyleBox> panel = get_stylebox("panel", "WindowDialog");
			int margin_left = static_cast<int>(panel->get_margin(MARGIN_LEFT));
			int margin_top = static_cast<int>(panel->get_margin(MARGIN_TOP));
			int margin_right = static_cast<int>(panel->get_margin(MARGIN_RIGHT));
			int margin_bottom = static_cast<int>(panel->get_margin(MARGIN_BOTTOM));

			Rect2 rect;
			rect.pos.x = -margin_left;
			rect.pos.y = -margin_top;
			rect.size.width = size.width + margin_left + margin_right;
			rect.size.height = size.height + margin_top + margin_bottom;

			panel->draw(canvas, rect);

			int title_height = get_constant("title_height", "WindowDialog");
			Color title_color = get_color("title_color", "WindowDialog");
			Ref<Font> font = get_font("title_font", "WindowDialog");
			int ofs = (size.width - font->get_string_size(title).width) / 2;
			draw_string(font, Point2(ofs, -title_height + font->get_ascent()), title, title_color, size.width - panel->get_minimum_size().width);

		} break;

		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			close_button->set_normal_texture(get_icon("close", "WindowDialog"));
			close_button->set_pressed_texture(get_icon("close", "WindowDialog"));
			close_button->set_hover_texture(get_icon("close_hilite", "WindowDialog"));
			close_button->set_anchor(MARGIN_LEFT, ANCHOR_END);
			close_button->set_begin(Point2(get_constant("close_h_ofs", "WindowDialog"), -get_constant("close_v_ofs", "WindowDialog")));
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			// Reset the mouse cursor when leaving the resizable window border.
			if (resizable && !drag_type) {
				if (get_default_cursor_shape() != CURSOR_ARROW)
					set_default_cursor_shape(CURSOR_ARROW);
			}
		} break;
#ifdef TOOLS_ENABLED
		case NOTIFICATION_POST_POPUP: {
			if (get_tree() && get_tree()->is_editor_hint() && EditorNode::get_singleton())
				EditorNode::get_singleton()->dim_editor(true);
		} break;
		case NOTIFICATION_POPUP_HIDE: {
			if (get_tree() && get_tree()->is_editor_hint() && EditorNode::get_singleton())
				EditorNode::get_singleton()->dim_editor(false);
		} break;
#endif
	}
}

void WindowDialog::_closed() {

	_close_pressed();
	hide();
}

int WindowDialog::_drag_hit_test(const Point2 &pos) const {
	int drag_type = DRAG_NONE;

	if (resizable) {
		int titlebar_height = get_constant("titlebar_height", "WindowDialog");
		int scaleborder_size = get_constant("scaleborder_size", "WindowDialog");

		Rect2 rect = get_rect();

		if (pos.y < (-titlebar_height + scaleborder_size))
			drag_type = DRAG_RESIZE_TOP;
		else if (pos.y >= (rect.size.height - scaleborder_size))
			drag_type = DRAG_RESIZE_BOTTOM;
		if (pos.x < scaleborder_size)
			drag_type |= DRAG_RESIZE_LEFT;
		else if (pos.x >= (rect.size.width - scaleborder_size))
			drag_type |= DRAG_RESIZE_RIGHT;
	}

	if (drag_type == DRAG_NONE && pos.y < 0)
		drag_type = DRAG_MOVE;

	return drag_type;
}

void WindowDialog::set_title(const String &p_title) {

	title = XL_MESSAGE(p_title);
	update();
}
String WindowDialog::get_title() const {

	return title;
}

void WindowDialog::set_resizable(bool p_resizable) {
	resizable = p_resizable;
}
bool WindowDialog::get_resizable() const {
	return resizable;
}

Size2 WindowDialog::get_minimum_size() const {

	Ref<Font> font = get_font("title_font", "WindowDialog");
	int msx = close_button->get_combined_minimum_size().x;
	msx += font->get_string_size(title).x;

	return Size2(msx, 1);
}

TextureButton *WindowDialog::get_close_button() {

	return close_button;
}

void WindowDialog::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &WindowDialog::_gui_input);
	ClassDB::bind_method(D_METHOD("set_title", "title"), &WindowDialog::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &WindowDialog::get_title);
	ClassDB::bind_method(D_METHOD("set_resizable", "resizable"), &WindowDialog::set_resizable);
	ClassDB::bind_method(D_METHOD("get_resizable"), &WindowDialog::get_resizable);
	ClassDB::bind_method(D_METHOD("_closed"), &WindowDialog::_closed);
	ClassDB::bind_method(D_METHOD("get_close_button:TextureButton"), &WindowDialog::get_close_button);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "window_title", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT_INTL), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resizable", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT_INTL), "set_resizable", "get_resizable");
}

WindowDialog::WindowDialog() {

	//title="Hello!";
	drag_type = DRAG_NONE;
	resizable = false;
	close_button = memnew(TextureButton);
	add_child(close_button);
	close_button->connect("pressed", this, "_closed");
}

WindowDialog::~WindowDialog() {
}

void PopupDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		get_stylebox("panel", "PopupMenu")->draw(ci, Rect2(Point2(), get_size()));
	}
}

PopupDialog::PopupDialog() {
}

PopupDialog::~PopupDialog() {
}

// AcceptDialog

void AcceptDialog::_post_popup() {

	WindowDialog::_post_popup();
	get_ok()->grab_focus();
}

void AcceptDialog::_notification(int p_what) {

	if (p_what == NOTIFICATION_MODAL_CLOSE) {

		cancel_pressed();
	} else if (p_what == NOTIFICATION_RESIZED) {

		_update_child_rects();
	}
}

void AcceptDialog::_builtin_text_entered(const String &p_text) {

	_ok_pressed();
}

void AcceptDialog::_ok_pressed() {

	if (hide_on_ok)
		hide();
	ok_pressed();
	emit_signal("confirmed");
}
void AcceptDialog::_close_pressed() {

	cancel_pressed();
}

String AcceptDialog::get_text() const {

	return label->get_text();
}
void AcceptDialog::set_text(String p_text) {

	label->set_text(p_text);
	minimum_size_changed();
	_update_child_rects();
}

void AcceptDialog::set_hide_on_ok(bool p_hide) {

	hide_on_ok = p_hide;
}

bool AcceptDialog::get_hide_on_ok() const {

	return hide_on_ok;
}

void AcceptDialog::register_text_enter(Node *p_line_edit) {

	ERR_FAIL_NULL(p_line_edit);
	p_line_edit->connect("text_entered", this, "_builtin_text_entered");
}

void AcceptDialog::_update_child_rects() {

	Size2 label_size = label->get_minimum_size();
	if (label->get_text().empty()) {
		label_size.height = 0;
	}
	int margin = get_constant("margin", "Dialogs");
	Size2 size = get_size();
	Size2 hminsize = hbc->get_combined_minimum_size();

	Vector2 cpos(margin, margin + label_size.height);
	Vector2 csize(size.x - margin * 2, size.y - margin * 3 - hminsize.y - label_size.height);

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;

		if (c == hbc || c == label || c == get_close_button() || c->is_set_as_toplevel())
			continue;

		c->set_pos(cpos);
		c->set_size(csize);
	}

	cpos.y += csize.y + margin;
	csize.y = hminsize.y;

	hbc->set_pos(cpos);
	hbc->set_size(csize);
}

Size2 AcceptDialog::get_minimum_size() const {

	int margin = get_constant("margin", "Dialogs");
	Size2 minsize = label->get_combined_minimum_size();

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = get_child(i)->cast_to<Control>();
		if (!c)
			continue;

		if (c == hbc || c == label || c == const_cast<AcceptDialog *>(this)->get_close_button() || c->is_set_as_toplevel())
			continue;

		Size2 cminsize = c->get_combined_minimum_size();
		minsize.x = MAX(cminsize.x, minsize.x);
		minsize.y = MAX(cminsize.y, minsize.y);
	}

	Size2 hminsize = hbc->get_combined_minimum_size();
	minsize.x = MAX(hminsize.x, minsize.x);
	minsize.y += hminsize.y;
	minsize.x += margin * 2;
	minsize.y += margin * 3; //one as separation between hbc and child

	Size2 wmsize = WindowDialog::get_minimum_size();
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
		button->connect("pressed", this, "_custom_action", varray(p_action));
	}

	return button;
}

Button *AcceptDialog::add_cancel(const String &p_cancel) {

	String c = p_cancel;
	if (p_cancel == "")
		c = RTR("Cancel");
	Button *b = swap_ok_cancel ? add_button(c, true) : add_button(c);
	b->connect("pressed", this, "_closed");
	return b;
}

void AcceptDialog::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_ok"), &AcceptDialog::_ok_pressed);
	ClassDB::bind_method(D_METHOD("get_ok"), &AcceptDialog::get_ok);
	ClassDB::bind_method(D_METHOD("get_label"), &AcceptDialog::get_label);
	ClassDB::bind_method(D_METHOD("set_hide_on_ok", "enabled"), &AcceptDialog::set_hide_on_ok);
	ClassDB::bind_method(D_METHOD("get_hide_on_ok"), &AcceptDialog::get_hide_on_ok);
	ClassDB::bind_method(D_METHOD("add_button:Button", "text", "right", "action"), &AcceptDialog::add_button, DEFVAL(false), DEFVAL(""));
	ClassDB::bind_method(D_METHOD("add_cancel:Button", "name"), &AcceptDialog::add_cancel);
	ClassDB::bind_method(D_METHOD("_builtin_text_entered"), &AcceptDialog::_builtin_text_entered);
	ClassDB::bind_method(D_METHOD("register_text_enter:LineEdit", "line_edit"), &AcceptDialog::register_text_enter);
	ClassDB::bind_method(D_METHOD("_custom_action"), &AcceptDialog::_custom_action);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &AcceptDialog::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &AcceptDialog::get_text);

	ADD_SIGNAL(MethodInfo("confirmed"));
	ADD_SIGNAL(MethodInfo("custom_action", PropertyInfo(Variant::STRING, "action")));

	ADD_GROUP("Dialog", "dialog");
	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "dialog_text", PROPERTY_HINT_MULTILINE_TEXT, "", PROPERTY_USAGE_DEFAULT_INTL), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "dialog_hide_on_ok"), "set_hide_on_ok", "get_hide_on_ok");
}

bool AcceptDialog::swap_ok_cancel = false;
void AcceptDialog::set_swap_ok_cancel(bool p_swap) {

	swap_ok_cancel = p_swap;
}

AcceptDialog::AcceptDialog() {

	int margin = get_constant("margin", "Dialogs");
	int button_margin = get_constant("button_margin", "Dialogs");

	label = memnew(Label);
	label->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	label->set_anchor(MARGIN_BOTTOM, ANCHOR_END);
	label->set_begin(Point2(margin, margin));
	label->set_end(Point2(margin, button_margin + 10));
	//label->set_autowrap(true);
	add_child(label);

	hbc = memnew(HBoxContainer);
	add_child(hbc);

	hbc->add_spacer();
	ok = memnew(Button);
	ok->set_text(RTR("OK"));
	hbc->add_child(ok);
	hbc->add_spacer();

	ok->connect("pressed", this, "_ok");
	set_as_toplevel(true);

	hide_on_ok = true;
	set_title(RTR("Alert!"));
}

AcceptDialog::~AcceptDialog() {
}

void ConfirmationDialog::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_cancel:Button"), &ConfirmationDialog::get_cancel);
}

Button *ConfirmationDialog::get_cancel() {

	return cancel;
}

ConfirmationDialog::ConfirmationDialog() {

	set_title(RTR("Please Confirm..."));
	cancel = add_cancel();
}
