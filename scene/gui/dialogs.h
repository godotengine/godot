/*************************************************************************/
/*  dialogs.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef DIALOGS_H
#define DIALOGS_H

#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/texture_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/popup.h"
#include "box_container.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class WindowDialog : public Popup {

	GDCLASS(WindowDialog,Popup);

	TextureButton *close_button;
	String title;
	bool dragging;

	void _input_event(const InputEvent& p_event);
	void _closed();
protected:
	virtual void _post_popup();

	virtual void _close_pressed()  {}
	virtual bool has_point(const Point2& p_point) const;
	void _notification(int p_what);
	static void _bind_methods();
public:

	TextureButton *get_close_button();

	void set_title(const String& p_title);
	String get_title() const;

	Size2 get_minimum_size() const;

	WindowDialog();
	~WindowDialog();

};

class PopupDialog : public Popup {

	GDCLASS(PopupDialog,Popup);

protected:
	void _notification(int p_what);
public:

	PopupDialog();
	~PopupDialog();

};


class LineEdit;

class AcceptDialog : public WindowDialog {

	GDCLASS(AcceptDialog,WindowDialog);

	Control *child;
	HBoxContainer *hbc;
	Label *label;
	Button *ok;
//	Button *cancel; no more cancel (there is X on tht titlebar)
	bool hide_on_ok;


	void _custom_action(const String& p_action);
	void _ok_pressed();
	void _close_pressed();
	void _builtin_text_entered(const String& p_text);
	void _update_child_rect();

	static bool swap_ok_cancel;


	virtual void remove_child_notify(Node *p_child);


protected:

	virtual void _post_popup();
	void _notification(int p_what);
	static void _bind_methods();
	virtual void ok_pressed() {}
	virtual void cancel_pressed() {}
	virtual void custom_action(const String&) {}
public:

	Size2 get_minimum_size() const;

	Label *get_label() { return label; }
	static void set_swap_ok_cancel(bool p_swap);


	void register_text_enter(Node *p_line_edit);

	Button *get_ok() { return ok; }
	Button* add_button(const String& p_text,bool p_right=false,const String& p_action="");
	Button* add_cancel(const String &p_cancel="");


	void set_hide_on_ok(bool p_hide);
	bool get_hide_on_ok() const;

	void set_text(String p_text);
	String get_text() const;

	void set_child_rect(Control *p_child);

	AcceptDialog();
	~AcceptDialog();

};


class ConfirmationDialog : public AcceptDialog {

	GDCLASS(ConfirmationDialog,AcceptDialog);
	Button *cancel;
protected:
	static void _bind_methods();
public:

	Button *get_cancel();
	ConfirmationDialog();

};

#endif
