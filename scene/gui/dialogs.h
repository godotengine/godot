/**************************************************************************/
/*  dialogs.h                                                             */
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

#ifndef DIALOGS_H
#define DIALOGS_H

#include "box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel.h"
#include "scene/gui/popup.h"
#include "scene/gui/texture_button.h"

class WindowDialog : public Popup {
	GDCLASS(WindowDialog, Popup);

	enum DRAG_TYPE {
		DRAG_NONE = 0,
		DRAG_MOVE = 1,
		DRAG_RESIZE_TOP = 1 << 1,
		DRAG_RESIZE_RIGHT = 1 << 2,
		DRAG_RESIZE_BOTTOM = 1 << 3,
		DRAG_RESIZE_LEFT = 1 << 4
	};

	TextureButton *close_button;
	String title;
	String xl_title;
	int drag_type;
	Point2 drag_offset;
	Point2 drag_offset_far;
	bool resizable;

#ifdef TOOLS_ENABLED
	bool was_editor_dimmed;
#endif

	void _gui_input(const Ref<InputEvent> &p_event);
	void _closed();
	int _drag_hit_test(const Point2 &pos) const;

protected:
	virtual void _post_popup();
	virtual void _fix_size();
	virtual void _close_pressed() {}
	virtual bool has_point(const Point2 &p_point) const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	TextureButton *get_close_button();

	void set_title(const String &p_title);
	String get_title() const;
	void set_resizable(bool p_resizable);
	bool get_resizable() const;

	Size2 get_minimum_size() const;

	WindowDialog();
	~WindowDialog();
};

class PopupDialog : public Popup {
	GDCLASS(PopupDialog, Popup);

protected:
	void _notification(int p_what);

public:
	PopupDialog();
	~PopupDialog();
};

class LineEdit;

class AcceptDialog : public WindowDialog {
	GDCLASS(AcceptDialog, WindowDialog);

	HBoxContainer *hbc;
	Label *label;
	Button *ok;
	bool hide_on_ok;

	void _custom_action(const String &p_action);
	void _ok_pressed();
	void _close_pressed();
	void _builtin_text_entered(const String &p_text);
	void _update_child_rects();

	static bool swap_ok_cancel;

protected:
	virtual void _post_popup();
	void _notification(int p_what);
	static void _bind_methods();
	virtual void ok_pressed() {}
	virtual void cancel_pressed() {}
	virtual void custom_action(const String &) {}

public:
	Size2 get_minimum_size() const;

	Label *get_label() { return label; }
	static void set_swap_ok_cancel(bool p_swap);

	void register_text_enter(Node *p_line_edit);

	Button *get_ok() { return ok; }
	Button *add_button(const String &p_text, bool p_right = false, const String &p_action = "");
	Button *add_cancel(const String &p_cancel = "");
	void remove_button(Control *p_button);

	void set_hide_on_ok(bool p_hide);
	bool get_hide_on_ok() const;

	void set_text(String p_text);
	String get_text() const;

	void set_autowrap(bool p_autowrap);
	bool has_autowrap();

	AcceptDialog();
	~AcceptDialog();
};

class ConfirmationDialog : public AcceptDialog {
	GDCLASS(ConfirmationDialog, AcceptDialog);
	Button *cancel;

protected:
	static void _bind_methods();

public:
	Button *get_cancel();
	ConfirmationDialog();
};

#endif // DIALOGS_H
